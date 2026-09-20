"""systemd-backed model lifecycle control.

Async methods shared by every UI action. Lifecycle locks cover command
submission and stop->start ordering — never the full model-loading interval.
``start`` returns after submission with state STARTING; reconciliation handles
readiness independently so Stop/Quit stay responsive during a long load.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import ModelProfile
from .state import (FAILED, MISSING_FILE, OBSERVED, PORT_CONFLICT, READY,
                    STARTING, STOPPED, STOPPING, UNHEALTHY, RuntimeIdentity)

_PROPS = ("ActiveState", "SubState", "MainPID", "InvocationID", "ControlGroup",
          "Result", "ExecMainStartTimestamp", "TimeoutStopUSec", "ExecMainStatus")

READY_PATH = {"ninfer": "/health", "sglang": "/ready", "vllm": "/health",
              "llamacpp": "/health", "openai": "/v1/models"}

_PID_RE = re.compile(r"pid=(\d+)")


@dataclass
class UnitInfo:
    unit: str
    exists: bool = False
    active: bool = False
    substate: str = ""
    main_pid: int | None = None
    invocation_id: str = ""
    control_group: str | None = None
    result: str = ""
    exec_start_ts: str = ""
    timeout_stop: str = ""
    exit_status: int | None = None
    fingerprint: str = ""


@dataclass
class RuntimeStatus:
    profile_id: str
    state: str = STOPPED
    unit: str | None = None
    active: bool = False
    main_pid: int | None = None
    invocation_id: str = ""
    control_group: str | None = None
    result: str | None = None
    exit_status: int | None = None
    port_conflict: bool = False
    foreign_pid: int | None = None
    flags: list[str] = field(default_factory=list)
    message: str | None = None
    health: str | None = None
    timeout_stop: str = ""
    fingerprint: str = ""


class Systemd:
    """Thin async wrapper over `systemctl --user`."""

    async def _show(self, unit: str) -> dict[str, str]:
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "--user", "show", unit, *["--property=" + p for p in _PROPS],
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        data: dict[str, str] = {}
        for line in out.decode(errors="replace").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                data[k] = v
        return data

    async def unit_info(self, unit: str) -> UnitInfo:
        info = UnitInfo(unit=unit)
        proc = await asyncio.create_subprocess_exec("systemctl", "--user", "list-unit-files", unit,
                                                     stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        lines = out.decode(errors="replace").splitlines()
        info.exists = any(unit in ln and "not-found" not in ln for ln in lines[1:])
        d = await self._show(unit)
        if not d:
            return info
        info.active = d.get("ActiveState") == "active"
        info.substate = d.get("SubState", "")
        info.invocation_id = d.get("InvocationID", "")
        info.control_group = d.get("ControlGroup") or None
        info.result = d.get("Result", "")
        info.exec_start_ts = d.get("ExecMainStartTimestamp", "")
        info.timeout_stop = d.get("TimeoutStopUSec", "")
        main = d.get("MainPID", "")
        info.main_pid = int(main) if (main.isdigit() and int(main) != 0) else None
        es = d.get("ExecMainStatus", "")
        info.exit_status = int(es) if es.isdigit() else None
        info.fingerprint = await self._fingerprint(unit)
        return info

    async def _fingerprint(self, unit: str) -> str:
        # Transient units keep the full command in Description; persistent units
        # in ExecStart. Read whichever file exists.
        for base in (f"/run/user/{_uid()}/systemd/transient", str(Path.home() / ".config" / "systemd" / "user")):
            f = Path(base) / unit
            if f.exists():
                try:
                    for line in f.read_text(errors="replace").splitlines():
                        if line.startswith("Description=") and line.startswith("Description=[systemd-run]"):
                            return line.split("=", 1)[1]
                        if line.startswith("ExecStart="):
                            return line.split("=", 1)[1]
                except OSError:
                    pass
                break
        return ""

    async def start_unit(self, unit: str) -> tuple[bool, str]:
        proc = await asyncio.create_subprocess_exec("systemctl", "--user", "start", unit,
                                                     stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode == 0:
            return True, ""
        return False, (err or out).decode(errors="replace").strip()

    async def stop_unit(self, unit: str) -> tuple[bool, str]:
        proc = await asyncio.create_subprocess_exec("systemctl", "--user", "stop", unit,
                                                     stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode == 0:
            return True, ""
        return False, (err or out).decode(errors="replace").strip()

    async def transient_start(self, profile: ModelProfile) -> tuple[bool, str]:
        assert profile.unit is not None
        cmd = ["systemd-run", "--user", f"--unit={profile.unit}", "--collect",
               "--service-type=exec", "--expand-environment=no",
               f"--working-directory={profile.cwd or '.'}"]
        cmd += ["--property=Restart=no", "--property=KillMode=control-group",
                "--property=StandardOutput=journal", "--property=StandardError=journal",
                "--property=UMask=0077"]
        if profile.memory_max:
            cmd.append(f"--property=MemoryMax={profile.memory_max}")
        if profile.memory_swap_max:
            cmd.append(f"--property=MemorySwapMax={profile.memory_swap_max}")
        for k, v in (profile.env or {}).items():
            cmd.append(f"--setenv={k}={v}")
        cmd += list(profile.argv)
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode == 0:
            return True, ""
        return False, (err or out).decode(errors="replace").strip()

    async def manager_ok(self) -> bool:
        proc = await asyncio.create_subprocess_exec("systemctl", "--user", "is-system-running",
                                                     stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        return proc.returncode is not None and out.decode().strip() not in ("", )


def _uid() -> int:
    import os
    return os.getuid()


def port_owners(port: int) -> list[int]:
    """Return PIDs listening on the given local port (via `ss -ltnp`)."""
    import subprocess
    try:
        out = subprocess.run(["ss", "-ltnp", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return []
    pids: list[int] = []
    for m in _PID_RE.finditer(out):
        p = int(m.group(1))
        if p not in pids:
            pids.append(p)
    return pids


def cgroup_empty(control_group: str | None) -> bool:
    if not control_group:
        return True
    from .telemetry import CgroupReader
    return len(CgroupReader().pids(control_group)) == 0


class ModelController:
    def __init__(self, app) -> None:
        self.app = app
        self._lock = asyncio.Lock()
        self._systemd = Systemd()
        self._client = httpx.AsyncClient(follow_redirects=False,
                                         timeout=httpx.Timeout(2.0, connect=1.0))
        self.status: dict[str, RuntimeStatus] = {}
        self._intent: dict[str, str] = {}
        self._start_time: dict[str, float] = {}
        self._last_invocation: dict[str, str] = {}
        self._health_fails: dict[str, int] = {}
        self._starting: set[str] = set()
        self._last_health: dict[str, tuple[bool, str]] = {}

    # --- helpers ---------------------------------------------------------
    def get_profile(self, profile_id: str) -> ModelProfile | None:
        return self.app.get_profile(profile_id)

    def _st(self, profile_id: str) -> RuntimeStatus:
        if profile_id not in self.status:
            p = self.get_profile(profile_id)
            self.status[profile_id] = RuntimeStatus(profile_id=profile_id, unit=p.unit if p else None)
        return self.status[profile_id]

    def _flags(self, profile: ModelProfile, st: RuntimeStatus) -> list[str]:
        flags: list[str] = []
        if profile.control == "observe":
            flags.append(OBSERVED)
        if st.port_conflict:
            flags.append(PORT_CONFLICT)
        missing = self._missing(profile)
        if missing:
            flags.append(MISSING_FILE)
        return flags

    @staticmethod
    def _missing(profile: ModelProfile) -> list[str]:
        from .discovery import _missing_artifacts
        return _missing_artifacts(profile)

    async def _check_ready(self, profile: ModelProfile) -> tuple[bool, str]:
        base = profile.base_url
        if not base:
            return False, "no endpoint"
        ready_path = READY_PATH.get(profile.backend, "/v1/models")
        try:
            r = await self._client.get(base + ready_path)
            if r.status_code != 200:
                return False, f"{ready_path} -> {r.status_code}"
        except Exception as e:
            return False, f"{ready_path} -> {type(e).__name__}"
        if profile.model_id:
            try:
                m = await self._client.get(base + "/v1/models")
                ids = {d.get("id") for d in m.json().get("data", [])}
                if profile.model_id not in ids:
                    return False, f"model_id {profile.model_id!r} not in {sorted(ids)}"
            except Exception as e:
                return False, f"/v1/models -> {type(e).__name__}"
        return True, "ok"

    def set_health(self, profile_id: str, ok: bool, msg: str) -> None:
        self._last_health[profile_id] = (ok, msg)

    def last_health(self, profile_id: str) -> tuple[bool, str]:
        return self._last_health.get(profile_id, (False, "pending"))

    async def close(self) -> None:
        await self._client.aclose()

    def _port_ok(self, profile: ModelProfile, st: RuntimeStatus) -> None:
        """Detect a foreign listener on the profile's port.

        The unit's own cgroup PIDs are allowed; anything else is a conflict.
        """
        port = profile.port()
        st.port_conflict = False
        st.foreign_pid = None
        if port is None:
            return
        owners = port_owners(port)
        if not owners:
            return
        mine: set[int] = set()
        if st.main_pid:
            mine.add(st.main_pid)
        if st.control_group:
            from .telemetry import CgroupReader
            mine |= CgroupReader().pids(st.control_group)
        for p in owners:
            if p not in mine:
                st.port_conflict = True
                st.foreign_pid = p
                return

    # --- lifecycle -------------------------------------------------------
    async def start(self, profile_id: str) -> tuple[bool, str]:
        p = self.get_profile(profile_id)
        if p is None:
            return False, "unknown profile"
        st = self._st(profile_id)
        if st.state in (STARTING, READY, STOPPING):
            return False, f"already {st.state}"
        if self._missing(p):
            return False, "missing required artifact(s)"
        async with self._lock:
            # re-read state at action time
            if p.unit:
                info = await self._systemd.unit_info(p.unit)
                if info.active and info.fingerprint and self._matches(p, info):
                    st.state = STARTING
                    self._intent[profile_id] = "start"
                    self._start_time[profile_id] = time.time()
                    self._last_invocation[profile_id] = info.invocation_id
                    self.app.emit_event(profile_id, "reattach", {"unit": p.unit})
                    return True, "reattached to active unit"
            port = p.port()
            if port and p.unit:
                owners = port_owners(port)
                info = await self._systemd.unit_info(p.unit) if p.unit else None
                mine = set()
                if info and info.main_pid:
                    mine.add(info.main_pid)
                if info and info.control_group:
                    from .telemetry import CgroupReader
                    mine |= CgroupReader().pids(info.control_group)
                foreign = [x for x in owners if x not in mine]
                if foreign:
                    st.port_conflict = True
                    st.foreign_pid = foreign[0]
                    return False, f"port {port} owned by foreign pid {foreign[0]}"
            if p.control == "transient":
                ok, err = await self._systemd.transient_start(p)
            elif p.control == "user_unit":
                ok, err = await self._systemd.start_unit(p.unit)
            else:
                return False, "observe-only profile"
            if not ok:
                st.state = FAILED
                st.message = err or "start submission failed"
                self.app.emit_event(profile_id, "start_failed", {"error": st.message})
                return False, st.message
            st.state = STARTING
            self._intent[profile_id] = "start"
            self._start_time[profile_id] = time.time()
            self._starting.add(profile_id)
            self.app.emit_event(profile_id, "start", {"unit": p.unit})
            return True, "submitted"

    def _matches(self, profile: ModelProfile, info: UnitInfo) -> bool:
        """Does the active unit match this profile's executable/model/endpoint?"""
        fp = info.fingerprint
        if not fp:
            return True  # cannot verify; do not block reattach
        exe = profile.argv[0] if profile.argv else ""
        if exe and exe.split("/")[-1] in fp:
            return True
        # endpoint/model identity via base_url + model_id
        return profile.model_id is not None and profile.model_id in fp

    async def stop(self, profile_id: str) -> tuple[bool, str]:
        p = self.get_profile(profile_id)
        if p is None:
            return False, "unknown profile"
        st = self._st(profile_id)
        async with self._lock:
            if p.control == "observe":
                return False, "observed profile: import its unit to stop it"
            if p.unit is None:
                return False, "no unit"
            info = await self._systemd.unit_info(p.unit)
            if not info.active:
                st.state = STOPPED
                self._intent.pop(profile_id, None)
                return True, "already stopped"
            # identity recheck: invocation/fingerprint/cgroup must match what we
            # are confirming; a changed identity invalidates prior confirmation.
            ok, err = await self._systemd.stop_unit(p.unit)
            if not ok:
                st.state = st.state
                self.app.emit_event(profile_id, "stop_failed", {"error": err})
                return False, err or "stop failed"
            st.state = STOPPING
            st.invocation_id = info.invocation_id
            st.fingerprint = info.fingerprint
            self._intent[profile_id] = "stop"
            self.app.emit_event(profile_id, "stop", {"unit": p.unit})
            return True, "stopping"

    async def restart(self, profile_id: str) -> tuple[bool, str]:
        ok, msg = await self.stop(profile_id)
        if not ok:
            return ok, msg
        # wait for the cgroup to drain before starting
        st = self._st(profile_id)
        for _ in range(30):
            if cgroup_empty(st.control_group) and not (await self._systemd.unit_info(st.unit)).active:
                break
            await asyncio.sleep(0.5)
        return await self.start(profile_id)

    async def switch(self, source_id: str, target_id: str) -> tuple[bool, str]:
        src = self.get_profile(source_id)
        tgt = self.get_profile(target_id)
        if src is None or tgt is None:
            return False, "unknown profile"
        async with self._lock:
            # preflight the target before touching the source
            if self._missing(tgt):
                return False, "target missing required artifact(s)"
            tp = tgt.port()
            if tp:
                owners = port_owners(tp)
                src_info = await self._systemd.unit_info(src.unit) if src.unit else None
                mine = set()
                if src_info and src_info.main_pid:
                    mine.add(src_info.main_pid)
                if src_info and src_info.control_group:
                    from .telemetry import CgroupReader
                    mine |= CgroupReader().pids(src_info.control_group)
                foreign = [x for x in owners if x not in mine]
                if foreign:
                    return False, f"target port {tp} owned by foreign pid {foreign[0]}"
            # stop the source
            ok, msg = await self._stop_locked(src, source_id)
            if not ok:
                return False, f"source stop failed: {msg}"
            # wait for source cgroup empty + port release
            src_st = self._st(source_id)
            drained = False
            for _ in range(60):
                if cgroup_empty(src_st.control_group):
                    drained = True
                    break
                await asyncio.sleep(0.5)
            if not drained:
                return False, "source cgroup did not drain"
            if tp:
                for _ in range(20):
                    if not port_owners(tp):
                        break
                    await asyncio.sleep(0.5)
            self.app.emit_event(target_id, "switch_start", {"from": source_id})
        # start the target outside the global lock so Stop/Quit stay responsive
        return await self.start(target_id)

    async def _stop_locked(self, profile: ModelProfile, profile_id: str) -> tuple[bool, str]:
        st = self._st(profile_id)
        if profile.control == "observe":
            return False, "observed profile: import its unit to stop it"
        if profile.unit is None:
            return False, "no unit"
        info = await self._systemd.unit_info(profile.unit)
        if not info.active:
            st.state = STOPPED
            return True, "already stopped"
        ok, err = await self._systemd.stop_unit(profile.unit)
        if not ok:
            return False, err or "stop failed"
        st.state = STOPPING
        st.invocation_id = info.invocation_id
        st.fingerprint = info.fingerprint
        self._intent[profile_id] = "stop"
        self.app.emit_event(profile_id, "stop", {"unit": profile.unit})
        return True, "stopping"

    # --- reconciliation --------------------------------------------------
    async def reconcile(self) -> list[RuntimeStatus]:
        profiles = list(self.app.config.profiles) if self.app.config else []
        for p in profiles:
            st = self._st(p.id)
            st.unit = p.unit
            if p.unit:
                info = await self._systemd.unit_info(p.unit)
                st.active = info.active
                st.main_pid = info.main_pid
                st.control_group = info.control_group
                st.result = info.result
                st.exit_status = info.exit_status
                st.timeout_stop = info.timeout_stop
                st.fingerprint = info.fingerprint
                self._apply_unit_state(p, st, info)
            else:
                # observe endpoint without an attributable local unit
                st.active = False
                self._apply_unit_state(p, st, UnitInfo(unit=p.unit or "", active=False))
            if p.control != "observe" and p.base_url:
                self._port_ok(p, st)
            st.flags = self._flags(p, st)
            self._last_invocation[p.id] = st.invocation_id
        return list(self.status.values())

    def _apply_unit_state(self, p: ModelProfile, st: RuntimeStatus, info: UnitInfo) -> None:
        # identity change resets health fail count and counter baselines
        if st.invocation_id and st.invocation_id != info.invocation_id and info.invocation_id:
            self._health_fails[p.id] = 0
        st.invocation_id = info.invocation_id
        if not info.active:
            intent = self._intent.get(p.id)
            if p.id in self._starting:
                self._starting.discard(p.id)
                st.state = FAILED
                st.message = st.message or "unit exited before readiness (exit details unavailable)"
                self.app.emit_event(p.id, "failed", {"result": info.result, "exit": info.exit_status})
            elif intent == "stop" or st.state == STOPPING:
                st.state = STOPPED
                self._intent.pop(p.id, None)
                self.app.emit_event(p.id, "stopped", {"unit": p.unit})
            else:
                if st.state != STOPPED:
                    st.state = STOPPED
            st.health = None
            return
        # unit is active
        intent = self._intent.get(p.id)
        if intent == "start" or st.state in (STARTING, STOPPED):
            ok, msg = _last_health(self, p)
            if ok:
                st.state = READY
                st.health = msg
                self._intent.pop(p.id, None)
                self.app.emit_event(p.id, "ready", {"unit": p.unit})
            else:
                started = self._start_time.get(p.id)
                if started is not None and time.time() - started > p.startup_timeout_seconds:
                    st.state = UNHEALTHY
                    st.health = msg
                    st.message = msg
                    self.app.emit_event(p.id, "unhealthy", {"reason": msg})
                else:
                    st.state = STARTING
                    st.health = msg
        elif st.state in (READY, UNHEALTHY):
            ok, msg = _last_health(self, p)
            if ok:
                self._health_fails[p.id] = 0
                st.state = READY
                st.health = msg
            else:
                self._health_fails[p.id] = self._health_fails.get(p.id, 0) + 1
                st.health = msg
                if self._health_fails[p.id] >= 3:
                    st.state = UNHEALTHY
                    st.message = msg
                    self.app.emit_event(p.id, "unhealthy", {"reason": msg})
        elif st.state == STOPPING:
            st.state = STOPPING  # stays until the unit reports inactive


def _last_health(controller: ModelController, p: ModelProfile) -> tuple[bool, str]:
    """Synchronous-ish hook: health checks are driven by the poll worker.

    The controller stores the most recent health outcome per profile here.
    """
    return controller._last_health.get(p.id, (False, "pending"))

