"""Core data model: runtime states, hardware/runtime snapshots, capability flags.

Snapshots carry numeric values with a source, an observation time and an
optional unavailable reason. ``None`` always means "unknown / not measured",
never zero — a field that is genuinely 0 is represented as ``0.0``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# --- Runtime lifecycle states ---------------------------------------------
STOPPED = "STOPPED"
STARTING = "STARTING"
READY = "READY"
STOPPING = "STOPPING"
FAILED = "FAILED"
UNHEALTHY = "UNHEALTHY"
ALL_STATES = (STOPPED, STARTING, READY, STOPPING, FAILED, UNHEALTHY)

# --- Capability / problem flags -------------------------------------------
OBSERVED = "OBSERVED"
MISSING_FILE = "MISSING FILE"
PORT_CONFLICT = "PORT CONFLICT"


@dataclass
class Metric:
    """A scalar with provenance. value None == unknown, not zero."""

    value: float | None = None
    source: str = ""
    observed_at: float = field(default_factory=time.time)
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


# --- GPU ------------------------------------------------------------------
@dataclass
class GpuProcess:
    pid: int
    command: str
    vram_mib: float
    util_pct: float | None
    create_time: float | None


@dataclass
class GpuSnapshot:
    index: int
    name: str
    uuid: str
    util_pct: Metric = field(default_factory=Metric)
    vram_used_mib: Metric = field(default_factory=Metric)
    vram_total_mib: Metric = field(default_factory=Metric)
    vram_free_mib: Metric = field(default_factory=Metric)
    temp_c: Metric = field(default_factory=Metric)
    power_w: Metric = field(default_factory=Metric)
    power_limit_w: Metric = field(default_factory=Metric)
    fan_pct: Metric = field(default_factory=Metric)
    graphics_clock_mhz: Metric = field(default_factory=Metric)
    memory_clock_mhz: Metric = field(default_factory=Metric)
    throttle_reasons: list[str] = field(default_factory=list)
    processes: list[GpuProcess] = field(default_factory=list)
    model_pids: set[int] = field(default_factory=set)


# --- CPU / RAM / disk -----------------------------------------------------
@dataclass
class CpuSnapshot:
    total_pct: Metric = field(default_factory=Metric)
    per_core_pct: list[Metric] = field(default_factory=list)
    frequency_mhz: Metric = field(default_factory=Metric)
    load1: Metric = field(default_factory=Metric)
    load5: Metric = field(default_factory=Metric)
    load15: Metric = field(default_factory=Metric)
    temperatures: dict[str, Metric] = field(default_factory=dict)
    core_count: int = 0
    threads: int = 0


@dataclass
class RamSnapshot:
    total_gib: Metric = field(default_factory=Metric)
    available_gib: Metric = field(default_factory=Metric)
    free_gib: Metric = field(default_factory=Metric)
    cache_gib: Metric = field(default_factory=Metric)
    swap_total_gib: Metric = field(default_factory=Metric)
    swap_free_gib: Metric = field(default_factory=Metric)


@dataclass
class DiskSnapshot:
    mount: str
    used_gib: Metric = field(default_factory=Metric)
    free_gib: Metric = field(default_factory=Metric)
    total_gib: Metric = field(default_factory=Metric)


# --- Hardware aggregate ---------------------------------------------------
@dataclass
class HardwareSnapshot:
    ts: float = field(default_factory=time.time)
    gpus: list[GpuSnapshot] = field(default_factory=list)
    nvml_available: bool = True
    nvml_reason: str | None = None
    cpu: CpuSnapshot = field(default_factory=CpuSnapshot)
    ram: RamSnapshot = field(default_factory=RamSnapshot)
    disks: list[DiskSnapshot] = field(default_factory=list)


# --- Per-model cgroup resources -------------------------------------------
@dataclass
class ModelResources:
    cgroup_path: str | None = None
    memory_current: Metric = field(default_factory=Metric)  # bytes
    memory_max: Metric = field(default_factory=Metric)      # bytes
    swap_current: Metric = field(default_factory=Metric)    # bytes
    oom_kill: Metric = field(default_factory=Metric)        # count
    oom_group_kill: Metric = field(default_factory=Metric)  # count
    cpu_total: Metric = field(default_factory=Metric)       # seconds
    gpu_vram_mib: Metric = field(default_factory=Metric)    # summed over cgroup pids


# --- Runtime identity + metrics -------------------------------------------
@dataclass(frozen=True)
class RuntimeIdentity:
    profile_id: str
    invocation_id: str  # systemd InvocationID ("" when unknown/observe)


# Normalized runtime metrics. Every field is a Metric so source/freshness is
# tracked. Gauge fields are the latest observed value; token/latency fields are
# observed-window deltas computed by metrics.py.
@dataclass
class RuntimeSnapshot:
    identity: RuntimeIdentity
    ts: float = field(default_factory=time.time)
    healthy: Metric = field(default_factory=Metric)
    running_requests: Metric = field(default_factory=Metric)
    queue_requests: Metric = field(default_factory=Metric)
    prompt_tokens_total: Metric = field(default_factory=Metric)
    completion_tokens_total: Metric = field(default_factory=Metric)
    requests_total: Metric = field(default_factory=Metric)
    # observed-window token rates (delta_tokens / monotonic_elapsed_seconds)
    prompt_rate: Metric = field(default_factory=Metric)
    completion_rate: Metric = field(default_factory=Metric)
    observed_prompt_tokens: Metric = field(default_factory=Metric)
    observed_completion_tokens: Metric = field(default_factory=Metric)
    # KV/SSM pool occupancy (0-1 fraction) — not VRAM, not context tokens
    kv_usage_frac: Metric = field(default_factory=Metric)
    cache_hit_rate: Metric = field(default_factory=Metric)
    prefix_cache_hits: Metric = field(default_factory=Metric)
    prefix_cache_queries: Metric = field(default_factory=Metric)
    spec_accept_rate: Metric = field(default_factory=Metric)
    spec_accept_length: Metric = field(default_factory=Metric)
    # mean over observed window (delta_sum/delta_count); None when count is 0
    ttft_mean_s: Metric = field(default_factory=Metric)
    e2e_mean_s: Metric = field(default_factory=Metric)
    ttft_samples: Metric = field(default_factory=Metric)
    e2e_samples: Metric = field(default_factory=Metric)
    # engine-reported decode/prefill speed (distinct from wall-clock rate)
    engine_prefill_tps: Metric = field(default_factory=Metric)
    engine_decode_tps: Metric = field(default_factory=Metric)
    # scheduler / queue gauges from NInfer throughput (interval measurements)
    scheduler_running: Metric = field(default_factory=Metric)
    scheduler_waiting: Metric = field(default_factory=Metric)
    extra: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


# --- Benchmark ------------------------------------------------------------
@dataclass
class BenchmarkResult:
    profile_id: str
    ts: float = field(default_factory=time.time)
    ok: bool = False
    ttft_s: Metric = field(default_factory=Metric)
    total_s: Metric = field(default_factory=Metric)
    e2e_tps: Metric = field(default_factory=Metric)  # completion_tokens / total_elapsed
    prompt_tokens: Metric = field(default_factory=Metric)
    completion_tokens: Metric = field(default_factory=Metric)
    engine_prefill_tps: Metric = field(default_factory=Metric)
    engine_decode_tps: Metric = field(default_factory=Metric)
    spec_accept_rate: Metric = field(default_factory=Metric)
    detail: str | None = None
