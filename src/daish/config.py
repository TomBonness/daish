"""Configuration model and (de)serialization.

A configuration is ``{"version":1,"model_roots":[...],"profiles":[...]}``.
Profiles are the single source of truth for launch recipes. Path-valued fields
are stored absolute; ``base_url`` is a loopback URL with no trailing ``/v1``.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKENDS = ("ninfer", "sglang", "vllm", "llamacpp", "openai")
CONTROLS = ("transient", "user_unit", "observe")

_UNIT_RE = re.compile(r"^[A-Za-z0-9._:@+-]+\.service$")
_LOOPBACK_RE = re.compile(r"^https?://(127\.0\.0\.1|localhost|0\.0\.0\.0|::1)(:\d+)?/?$")


def xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))


def xdg_state_home() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state"))


def xdg_runtime_dir() -> Path | None:
    v = os.environ.get("XDG_RUNTIME_DIR")
    return Path(v) if v else None


def default_config_path() -> Path:
    return xdg_config_home() / "daish" / "config.json"


def default_history_path() -> Path:
    return xdg_state_home() / "daish" / "history.sqlite3"


def default_lock_path() -> Path | None:
    rt = xdg_runtime_dir()
    return (rt / "daish" / "control.lock") if rt else None


def ninfer_log_dir() -> Path:
    return xdg_state_home() / "daish" / "ninfer"


@dataclass
class ModelProfile:
    id: str
    label: str
    backend: str
    control: str
    argv: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    unit: str | None = None
    base_url: str | None = None
    model_id: str | None = None
    model_paths: list[str] = field(default_factory=list)
    draft_paths: list[str] = field(default_factory=list)
    startup_timeout_seconds: float = 900.0
    ninfer_jsonl: str | None = None
    memory_max: str | None = None
    memory_swap_max: str | None = None

    def port(self) -> int | None:
        if not self.base_url:
            return None
        try:
            return int(self.base_url.rsplit(":", 1)[1].rstrip("/"))
        except (IndexError, ValueError):
            return None

    def host(self) -> str | None:
        if not self.base_url:
            return None
        host = self.base_url.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        return host


@dataclass
class AppConfig:
    version: int = 1
    model_roots: list[str] = field(default_factory=list)
    profiles: list[ModelProfile] = field(default_factory=list)

    def by_id(self) -> dict[str, ModelProfile]:
        return {p.id: p for p in self.profiles}


class ConfigError(Exception):
    """Raised for a structurally invalid configuration file."""

    def __init__(self, message: str, path: Path | None = None):
        super().__init__(message)
        self.message = message
        self.path = path


def _abs(value: str | None) -> str | None:
    if value is None:
        return None
    v = os.path.expanduser(value)
    if not v:
        return v
    return str(Path(v).expanduser().resolve())


def _profile_from_dict(d: dict[str, Any]) -> ModelProfile:
    argv = d.get("argv")
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise ConfigError(f"profile {d.get('id')!r}: argv must be a list of strings")
    env = d.get("env") or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ConfigError(f"profile {d.get('id')!r}: env must be an object of strings")

    def pathlist(key: str) -> list[str]:
        v = d.get(key)
        if v is None:
            return []
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ConfigError(f"profile {d.get('id')!r}: {key} must be a list of strings")
        return [os.path.expanduser(x) for x in v]

    base = d.get("base_url")
    if base is not None and not _LOOPBACK_RE.match(base):
        raise ConfigError(f"profile {d.get('id')!r}: base_url must be a loopback URL: {base!r}")

    unit = d.get("unit")
    if unit is not None and not _UNIT_RE.match(unit):
        raise ConfigError(f"profile {d.get('id')!r}: unit must be a .service name: {unit!r}")

    port = None
    if base is not None:
        m = re.search(r":(\d+)/?$", base)
        if m:
            port = int(m.group(1))

    return ModelProfile(
        id=str(d["id"]),
        label=str(d.get("label", d["id"])),
        backend=str(d.get("backend", "openai")),
        control=str(d.get("control", "observe")),
        argv=[_abs(a) or a for a in argv],
        cwd=_abs(d.get("cwd")),
        env={k: (os.path.expanduser(v) if os.sep in v or v.startswith("~") else v) for k, v in env.items()},
        unit=unit,
        base_url=base.rstrip("/") if base else None,
        model_id=d.get("model_id"),
        model_paths=pathlist("model_paths"),
        draft_paths=pathlist("draft_paths"),
        startup_timeout_seconds=float(d.get("startup_timeout_seconds", 900)),
        ninfer_jsonl=_abs(d.get("ninfer_jsonl")),
        memory_max=d.get("memory_max"),
        memory_swap_max=d.get("memory_swap_max"),
    )


def validate_profiles(profiles: list[ModelProfile]) -> list[str]:
    """Return a list of per-profile problem strings (empty == all valid).

    Problems are aligned with ``profiles`` by index. Valid profiles continue
    working; only the returned index is disabled in the UI.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_units: set[str] = set()
    for p in profiles:
        if p.backend not in BACKENDS:
            problems.append(f"unknown backend {p.backend!r}")
            continue
        if p.control not in CONTROLS:
            problems.append(f"unknown control {p.control!r}")
            continue
        if p.id in seen_ids:
            problems.append(f"duplicate id {p.id!r}")
            continue
        seen_ids.add(p.id)
        if p.control in ("transient", "user_unit"):
            if not p.unit:
                problems.append("missing unit name")
                continue
            if p.unit in seen_units:
                problems.append(f"duplicate unit {p.unit!r}")
                continue
            seen_units.add(p.unit)
        if p.control == "transient" and not p.argv:
            problems.append("transient profile needs a nonempty argv")
            continue
        if p.base_url:
            port = p.port()
            if port is not None and not (1 <= port <= 65535):
                problems.append(f"port out of range: {port}")
        problems.append("")
    return problems


def load_config(path: Path) -> AppConfig:
    """Load and validate a config file. Raises ConfigError on invalid JSON/schema."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ConfigError("expected version:1 object", path)
    roots = data.get("model_roots") or []
    if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
        raise ConfigError("model_roots must be a list of strings", path)
    prows = data.get("profiles") or []
    if not isinstance(prows, list):
        raise ConfigError("profiles must be a list", path)
    profiles = [_profile_from_dict(d) for d in prows]
    return AppConfig(version=1, model_roots=[os.path.expanduser(r) for r in roots], profiles=profiles)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    out: dict[str, Any] = {"version": 1, "model_roots": config.model_roots, "profiles": []}
    for p in config.profiles:
        out["profiles"].append(
            {
                "id": p.id,
                "label": p.label,
                "backend": p.backend,
                "control": p.control,
                "unit": p.unit,
                "argv": p.argv,
                "cwd": p.cwd,
                "env": p.env,
                "base_url": p.base_url,
                "model_id": p.model_id,
                "model_paths": p.model_paths,
                "draft_paths": p.draft_paths,
                "startup_timeout_seconds": p.startup_timeout_seconds,
                "ninfer_jsonl": p.ninfer_jsonl,
                "memory_max": p.memory_max,
                "memory_swap_max": p.memory_swap_max,
            }
        )
    return out


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def save_config(path: Path, config: AppConfig) -> None:
    _atomic_write(path, json.dumps(config_to_dict(config), indent=2) + "\n")
