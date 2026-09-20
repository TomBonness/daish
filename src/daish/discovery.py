"""Discovery: ship the curated local recipes and scan the model inventory.

``local-profiles.json`` is a curated import, **not** a shell-script
interpreter. Paths/environment containing the ``{root}`` token are materialized
relative to the selected local root. Unknown artifacts are surfaced, never
invented.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .config import ModelProfile

_ROOT_TOKEN = "{root}"


def _materialize(value: str, root: str) -> str:
    return value.replace(_ROOT_TOKEN, root)


def _materialize_profile(raw: dict, root: str) -> ModelProfile:
    root = str(Path(root).expanduser().resolve())
    env = raw.get("env") or {}
    env = {k: _materialize(str(v), root) for k, v in env.items()}
    # Resolve the current SGLang flashinfer workspace override once at import.
    if "SGLANG_FLASHINFER_WORKSPACE_SIZE" in env and os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE"):
        env["SGLANG_FLASHINFER_WORKSPACE_SIZE"] = os.environ["SGLANG_FLASHINFER_WORKSPACE_SIZE"]
    argv = [_materialize(a, root) for a in raw.get("argv", [])]
    cwd = raw.get("cwd")
    model_paths = [_materialize(p, root) for p in raw.get("model_paths", [])]
    draft_paths = [_materialize(p, root) for p in raw.get("draft_paths", [])]
    nj = raw.get("ninfer_jsonl")
    nj = _materialize(nj, root) if nj else None
    from .config import ModelProfile as _P
    return _P(
        id=raw["id"],
        label=raw.get("label", raw["id"]),
        backend=raw.get("backend", "openai"),
        control=raw.get("control", "observe"),
        argv=argv,
        cwd=cwd and _materialize(cwd, root),
        env=env,
        unit=raw.get("unit"),
        base_url=raw.get("base_url"),
        model_id=raw.get("model_id"),
        model_paths=model_paths,
        draft_paths=draft_paths,
        startup_timeout_seconds=float(raw.get("startup_timeout_seconds", 900)),
        ninfer_jsonl=nj,
        memory_max=raw.get("memory_max"),
        memory_swap_max=raw.get("memory_swap_max"),
    )


@dataclass
class ProfileCandidate:
    profile: ModelProfile
    source: str                      # origin (shipped recipe / user config)
    status: str = ""                  # "" ok, MISSING FILE, ...
    missing: list[str] = field(default_factory=list)
    unit_active: bool = False
    conflict: str | None = None


def _shipped_data() -> dict:
    ref = resources.files("daish.data").joinpath("local-profiles.json")
    return json.loads(ref.read_text(encoding="utf-8"))


def discover_profiles(local_root: Path) -> list[ProfileCandidate]:
    """Materialize the curated recipes against ``local_root``.

    Returns candidates with artifact presence checks; no services are created.
    """
    local_root = Path(local_root).expanduser()
    data = _shipped_data()
    out: list[ProfileCandidate] = []
    for raw in data.get("profiles", []):
        prof = _materialize_profile(raw, str(local_root))
        missing = _missing_artifacts(prof)
        status = "MISSING FILE" if missing else ""
        out.append(ProfileCandidate(profile=prof, source="local-recipes",
                                    status=status, missing=missing))
    return out


def _missing_artifacts(prof: ModelProfile) -> list[str]:
    missing: list[str] = []
    if prof.control != "observe" and prof.argv:
        exe = prof.argv[0]
        if exe and not Path(exe).exists():
            missing.append(exe)
    for p in list(prof.model_paths) + list(prof.draft_paths):
        if p and not Path(p).exists():
            missing.append(p)
    return missing


# --- Artifact inventory ----------------------------------------------------
@dataclass
class ModelArtifact:
    path: str
    kind: str                 # ninfer | gguf | hf | draft | unknown
    label: str
    bytes: int
    draft: bool = False
    related_profiles: list[str] = field(default_factory=list)


def _config_architectures(config_path: Path) -> list[str]:
    try:
        raw = config_path.read_text(encoding="utf-8", errors="replace")[:65536]
        data = json.loads(raw)
        arch = data.get("architectures")
        return [str(a) for a in arch] if isinstance(arch, list) else []
    except Exception:
        return []


def _weight_bytes(entries: list[Path], seen: set[tuple[int, int]]) -> int:
    total = 0
    for e in entries:
        try:
            st = e.stat()
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
        except OSError:
            continue
    return total


def scan_inventory(model_roots: list[Path], profiles: list[ModelProfile]) -> list[ModelArtifact]:
    """Scan configured roots with directory metadata + bounded config reads.

    No weight loading, checksumming, or downloading. Inode identity avoids
    counting hard-linked weights twice.
    """
    draft_archs = {"DFlash2DraftModel", "Qwen3DSparkModel"}
    related: dict[str, set[str]] = {}
    for p in profiles:
        for art in list(p.model_paths) + list(p.draft_paths):
            rp = str(Path(art).expanduser().resolve())
            related.setdefault(rp, set()).add(p.id)
            pp = str(Path(rp).parent)
            if pp != rp:
                related.setdefault(pp, set()).add(p.id)
    out: list[ModelArtifact] = []
    seen: set[tuple[int, int]] = set()
    for root in model_roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            try:
                if entry.name.startswith("."):
                    continue
            except Exception:
                continue
            if entry.is_file():
                suf = entry.suffix.lower()
                if suf == ".ninfer":
                    kind, label, draft = "ninfer", "NInfer artifact", False
                elif suf == ".gguf":
                    kind, label, draft = "gguf", "GGUF", False
                else:
                    continue
                out.append(_artifact(entry, kind, label, draft, seen, related))
            elif entry.is_dir():
                cfg = entry / "config.json"
                has_cfg = cfg.is_file()
                ninfer_files = [w for w in entry.iterdir() if w.suffix == ".ninfer"]
                weights = [w for w in entry.iterdir() if w.suffix in (".safetensors", ".bin", ".gguf")]
                if ninfer_files:
                    out.append(_artifact(entry, "ninfer", "NInfer artifact bundle", False, seen, related))
                elif has_cfg and weights:
                    archs = _config_architectures(cfg)
                    if set(archs) & draft_archs:
                        kind, label, draft = "draft", f"draft ({archs[0]})", True
                    else:
                        kind, label, draft = "hf", "HuggingFace weights", False
                    out.append(_artifact(entry, kind, label, draft, seen, related))
                else:
                    out.append(_artifact(entry, "unknown", "unknown", False, seen, related))
    return out


def _artifact(entry: Path, kind: str, label: str, draft: bool,
              seen: set[tuple[int, int]], related: dict[str, set[str]]) -> ModelArtifact:
    if entry.is_file():
        try:
            st = entry.stat()
            key = (st.st_dev, st.st_ino)
            nbytes = 0 if key in seen else st.st_size
            seen.add(key)
        except OSError:
            nbytes = 0
    else:
        weights = [w for w in entry.iterdir() if w.is_file() and w.suffix in
                   (".safetensors", ".bin", ".gguf", ".ninfer")]
        nbytes = _weight_bytes(weights, seen)
    return ModelArtifact(path=str(entry), kind=kind, label=label, bytes=nbytes,
                         draft=draft, related_profiles=sorted(related.get(str(entry.resolve()), set())))
