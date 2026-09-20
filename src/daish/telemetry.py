"""Hardware + per-model resource telemetry.

One bounded sampler at 1 Hz. NVML is initialized once and shut down on exit.
``Metric.value is None`` means unknown/not-supported — never zero.
"""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import psutil

from .state import (CpuSnapshot, DiskSnapshot, GpuProcess, GpuSnapshot,
                    HardwareSnapshot, Metric, ModelResources, RamSnapshot)

try:
    import pynvml
    _HAVE_NVML = True
except Exception:  # pragma: no cover - driver absent
    pynvml = None
    _HAVE_NVML = False

# Throttle-reason bit flags (NVML).
_THROTTLE_BITS = {
    0: "HW Slowdown", 1: "HW Thermal Slowdown", 2: "SW Power Cap",
    3: "SW Thermal Slowdown", 4: "SW Power Brake", 5: "HW Brake",
    6: "HW Thermal Shutdown", 7: "SW Thermal Shutdown", 8: "HW Power Brake",
}


def _m(value, source: str, reason: str | None = None) -> Metric:
    return Metric(value=value, source=source, observed_at=time.time(), reason=reason)


def _try(fn, source: str):
    """Call a pynvml getter; return (value, reason). None => not supported."""
    try:
        return fn(), None
    except Exception as e:  # NotSupported / permission
        name = type(e).__name__.replace("NVMLError_", "")
        return None, name


class CgroupReader:
    """Read v2 cgroup memory/cpu files for a systemd ControlGroup path."""

    def __init__(self) -> None:
        self._base = Path("/sys/fs/cgroup")

    def _dir(self, control_group: str) -> Path | None:
        if not control_group:
            return None
        p = self._base / control_group.strip("/")
        return p if p.is_dir() else None

    def _read_int(self, path: Path) -> float | None:
        try:
            txt = path.read_text().strip()
            if txt in ("max", ""):
                return None
            return float(txt)
        except Exception:
            return None

    @staticmethod
    def _events(path: Path) -> dict[str, int]:
        out: dict[str, int] = {}
        try:
            for line in path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    out[parts[0]] = int(parts[1])
        except Exception:
            pass
        return out

    def pids(self, control_group: str) -> set[int]:
        d = self._dir(control_group)
        pids: set[int] = set()
        if d is None:
            return pids
        stack = [d]
        while stack:
            cur = stack.pop()
            try:
                for line in (cur / "cgroup.procs").read_text().split():
                    if line.strip().isdigit():
                        pids.add(int(line))
                if (cur / "cgroup.controllers").exists():
                    for child in cur.iterdir():
                        if child.is_dir() and (child / "cgroup.procs").exists():
                            stack.append(child)
            except Exception:
                continue
        return pids

    def resources(self, control_group: str, gpu_vram_mib: float | None) -> ModelResources:
        d = self._dir(control_group)
        res = ModelResources(cgroup_path=control_group)
        if d is None:
            return res
        res.memory_current.value = self._read_int(d / "memory.current")
        res.memory_current.source = "cgroupv2"
        mm = self._read_int(d / "memory.max")
        res.memory_max.value = None if mm is None else mm
        res.memory_max.source = "cgroupv2"
        res.swap_current.value = self._read_int(d / "memory.swap.current")
        res.swap_current.source = "cgroupv2"
        ev = self._events(d / "memory.events")
        res.oom_kill.value = float(ev.get("oom_kill", 0))
        res.oom_group_kill.value = float(ev.get("oom_group_kill", 0))
        res.gpu_vram_mib = _m(gpu_vram_mib, "nvml", "unavailable" if gpu_vram_mib is None else None)
        return res


class HardwareSampler:
    """Sample GPU/CPU/RAM/disk at ~1 Hz. Call sample() from one thread only."""

    def __init__(self, model_roots: list[str] | None = None,
                 control_groups: dict[str, str] | None = None) -> None:
        self.model_roots = model_roots or []
        self.control_groups = control_groups or {}
        self._nvml_up = False
        self._devs: list[Any] = []
        self._gpu_static: dict[int, dict[str, Any]] = {}
        self._cpu_last: float | None = None
        self._cpu_primed = False
        self._per_core_primed = False
        self._proc_cache: dict[int, psutil.Process] = {}
        # series history: 900 one-second samples per displayed series
        self.history: dict[str, deque] = {}

    # --- lifecycle -------------------------------------------------------
    def init(self) -> None:
        if not _HAVE_NVML:
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._devs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._nvml_up = True
            for d in self._devs:
                idx = pynvml.nvmlDeviceGetIndex(d)
                self._gpu_static[idx] = {
                    "name": _s(pynvml.nvmlDeviceGetName(d)),
                    "uuid": _s(pynvml.nvmlDeviceGetUUID(d)),
                }
            # prime CPU counters (first real read next tick)
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None, percpu=True)
            self._cpu_primed = True
            self._per_core_primed = True
        except Exception:
            self._nvml_up = False

    def shutdown(self) -> None:
        if self._nvml_up and _HAVE_NVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_up = False

    # --- sampling --------------------------------------------------------
    def sample(self) -> HardwareSnapshot:
        snap = HardwareSnapshot(ts=time.time(), nvml_available=self._nvml_up,
                                nvml_reason=None if self._nvml_up else ("nvml" if _HAVE_NVML else "no-nvidia-ml-py"))
        snap.gpus = [self._gpu(d) for d in self._devs] if self._nvml_up else []
        snap.cpu = self._cpu()
        snap.ram = self._ram()
        snap.disks = self._disks()
        self._record(snap)
        return snap

    def _gpu(self, dev) -> GpuSnapshot:
        idx = pynvml.nvmlDeviceGetIndex(dev)
        st = self._gpu_static.get(idx, {})
        g = GpuSnapshot(index=idx, name=st.get("name", "GPU%d" % idx), uuid=st.get("uuid", ""))
        u, r = _try(lambda: pynvml.nvmlDeviceGetUtilizationRates(dev), "nvml")
        if u is not None:
            g.util_pct = _m(float(u.gpu), "nvml")
        else:
            g.util_pct = _m(None, "nvml", r)
        m, r = _try(lambda: pynvml.nvmlDeviceGetMemoryInfo(dev), "nvml")
        if m is not None:
            g.vram_used_mib = _m(m.used / (1024 * 1024), "nvml")
            g.vram_total_mib = _m(m.total / (1024 * 1024), "nvml")
            g.vram_free_mib = _m(m.free / (1024 * 1024), "nvml")
        else:
            g.vram_used_mib = _m(None, "nvml", r)
        t, r = _try(lambda: pynvml.nvmlDeviceGetTemperature(dev, pynvml.NVML_TEMPERATURE_GPU), "nvml")
        g.temp_c = _m(float(t) if t is not None else None, "nvml", r if t is None else None)
        p, r = _try(lambda: pynvml.nvmlDeviceGetPowerUsage(dev), "nvml")
        g.power_w = _m(p / 1000.0 if p is not None else None, "nvml", r if p is None else None)
        pl, r = _try(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(dev), "nvml")
        g.power_limit_w = _m(pl / 1000.0 if pl is not None else None, "nvml", r if pl is None else None)
        f, r = _try(lambda: pynvml.nvmlDeviceGetFanSpeed(dev), "nvml")
        g.fan_pct = _m(float(f) if f is not None else None, "nvml", r if f is None else None)
        gc, r = _try(lambda: pynvml.nvmlDeviceGetClockInfo(dev, pynvml.NVML_CLOCK_GRAPHICS), "nvml")
        g.graphics_clock_mhz = _m(gc / 1000.0 if gc is not None else None, "nvml", r if gc is None else None)
        mc, r = _try(lambda: pynvml.nvmlDeviceGetClockInfo(dev, pynvml.NVML_CLOCK_MEMORY), "nvml")
        g.memory_clock_mhz = _m(mc / 1000.0 if mc is not None else None, "nvml", r if mc is None else None)
        try:
            flags = pynvml.nvmlDeviceGetThrottleReasons(dev)
            g.throttle_reasons = [_THROTTLE_BITS.get(i, f"bit{i}") for i in range(16) if flags & (1 << i)]
        except Exception:
            g.throttle_reasons = []
        g.processes = self._gpu_procs(dev, g)
        return g

    def _gpu_procs(self, dev, g: GpuSnapshot) -> list[GpuProcess]:
        if not self._nvml_up:
            return []
        merged: dict[tuple[int, float], GpuProcess] = {}
        try:
            utils = pynvml.nvmlDeviceGetProcessUtilization(dev)
        except Exception:
            utils = []
        util_map: dict[int, tuple[float | None, float | None, float | None]] = {}
        for up in utils:
            util_map[up.pid] = (
                None if up.smUtil is None else float(up.smUtil),
                None if up.memoryUtil is None else float(up.memoryUtil),
                getattr(up, "processCreateTimestamp", None),
            )
        for kind in ("compute", "graphics"):
            try:
                infos = (pynvml.nvmlDeviceGetComputeRunningProcesses(dev)
                         if kind == "compute" else
                         pynvml.nvmlDeviceGetGraphicsRunningProcesses(dev))
            except Exception:
                continue
            for info in infos:
                ct = None
                if info.pid in util_map:
                    smu, memu, cts = util_map[info.pid]
                    ct = cts
                key = (info.pid, ct if ct else 0.0)
                if key in merged:
                    continue
                smu, memu = (util_map.get(info.pid, (None, None, None))[:2])
                cmd = _cmd(info.pid)
                merged[key] = GpuProcess(
                    pid=info.pid, command=cmd,
                    vram_mib=(info.usedGpuMemory or 0) / (1024 * 1024),
                    util_pct=smu, create_time=ct)
        procs = list(merged.values())
        g.model_pids = set()
        for cg in self.control_groups.values():
            g.model_pids |= CgroupReader().pids(cg)
        return procs

    def _cpu(self) -> CpuSnapshot:
        c = CpuSnapshot()
        c.core_count = psutil.cpu_count(logical=False) or 0
        c.threads = psutil.cpu_count(logical=True) or 0
        total = psutil.cpu_percent(interval=None)
        per = psutil.cpu_percent(interval=None, percpu=True)
        if self._cpu_primed:
            c.total_pct = _m(float(total), "psutil")
        else:
            c.total_pct = _m(None, "psutil", "priming")
            self._cpu_primed = True
        if self._per_core_primed:
            c.per_core_pct = [_m(float(x), "psutil") for x in per]
        else:
            c.per_core_pct = [_m(None, "psutil", "priming") for x in per]
            self._per_core_primed = True
        try:
            freq = psutil.cpu_freq()
            c.frequency_mhz = _m(freq.current if freq else None, "psutil",
                                 None if freq else "unsupported")
        except Exception:
            c.frequency_mhz = _m(None, "psutil", "unsupported")
        try:
            l1, l5, l15 = psutil.getloadavg()
            c.load1 = _m(float(l1), "psutil")
            c.load5 = _m(float(l5), "psutil")
            c.load15 = _m(float(l15), "psutil")
        except Exception:
            pass
        try:
            sensors = psutil.sensors_temperatures()
            if "k10temp" in sensors:
                for s in sensors["k10temp"]:
                    if s.label in ("Tctl", "TctlTmp"):
                        c.temperatures["CPU (k10temp/Tctl)"] = _m(float(s.current), "psutil")
            for name, lst in sensors.items():
                for s in lst:
                    c.temperatures.setdefault(s.label, _m(float(s.current), "psutil", name))
        except Exception:
            pass
        return c

    def _ram(self) -> RamSnapshot:
        r = RamSnapshot()
        vm = psutil.virtual_memory()
        gi = 1024 ** 3
        r.total_gib = _m(vm.total / gi, "meminfo")
        r.available_gib = _m(vm.available / gi, "meminfo")
        r.free_gib = _m(vm.free / gi, "meminfo")
        try:
            sw = psutil.swap_memory()
            r.swap_total_gib = _m(sw.total / gi, "meminfo")
            r.swap_free_gib = _m(sw.free / gi, "meminfo")
        except Exception:
            pass
        return r

    def _disks(self) -> list[DiskSnapshot]:
        out: list[DiskSnapshot] = []
        seen: set[str] = set()
        gi = 1024 ** 3
        candidates = list(self.model_roots)
        try:
            candidates.append(Path.home())
        except Exception:
            pass
        for root in candidates:
            try:
                real = str(Path(root).expanduser().resolve())
            except Exception:
                continue
            if real in seen:
                continue
            seen.add(real)
            try:
                u = psutil.disk_usage(real)
                out.append(DiskSnapshot(mount=str(root), used_gib=_m(u.used / gi, "psutil"),
                                        free_gib=_m(u.free / gi, "psutil"),
                                        total_gib=_m(u.total / gi, "psutil")))
            except Exception:
                continue
        return out

    def _record(self, snap: HardwareSnapshot) -> None:
        def put(key: str, v: float | None):
            self.history.setdefault(key, deque(maxlen=900)).append(v)
        for g in snap.gpus:
            put(f"gpu{g.index}.util", g.util_pct.value)
            put(f"gpu{g.index}.vram", g.vram_used_mib.value)
            put(f"gpu{g.index}.temp", g.temp_c.value)
            put(f"gpu{g.index}.power", g.power_w.value)
        put("cpu.total", snap.cpu.total_pct.value)
        put("ram.used", _gib_used(snap.ram))

    def model_resources(self, control_group: str, gpu_vram_mib: float | None) -> ModelResources:
        return CgroupReader().resources(control_group, gpu_vram_mib)

    def model_cpu_total(self, pids: set[int]) -> float | None:
        if not pids:
            return None
        total = 0.0
        ok = False
        for pid in pids:
            p = self._proc_cache.get(pid)
            try:
                if p is None:
                    p = psutil.Process(pid)
                    self._proc_cache[pid] = p
                c = p.cpu_times()
                total += c.user + c.system
                ok = True
            except psutil.Error:
                self._proc_cache.pop(pid, None)
        return total if ok else None


def _s(x) -> str:
    return str(x) if x is not None else ""


def _cmd(pid: int) -> str:
    try:
        p = psutil.Process(pid)
        name = p.name()
        exe = ""
        try:
            exe = p.cmdline()[0] if p.cmdline() else ""
        except Exception:
            pass
        return exe or name
    except Exception:
        return f"pid{pid}"


def _gib_used(ram: RamSnapshot) -> float | None:
    if ram.total_gib.value is None or ram.available_gib.value is None:
        return None
    return ram.total_gib.value - ram.available_gib.value
