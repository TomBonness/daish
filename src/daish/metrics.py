"""Runtime metrics: poll a ready instance and normalize to a RuntimeSnapshot.

Uses ``prometheus_client.parser`` (never regex-splitting exposition text).
Counters are processed per full label set + instance: the first sample sets a
baseline; a reset, identity change, or collection gap sets a fresh baseline
instead of producing a spike / negative rate. Observed token throughput is
``delta_tokens / monotonic_elapsed_seconds``. Mean TTFT/latency is
``delta_sum / delta_count`` over the observed window (None when count is 0) and
is labelled **mean**, not p95.
"""
from __future__ import annotations

import time
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

from .state import Metric, RuntimeIdentity, RuntimeSnapshot, READY

# Collection gap beyond this many seconds -> establish a fresh baseline.
GAP_RESET_S = 6.0

# Names to look up, trying the base name first then the ``_total`` variant
# (installed vLLM exposes e.g. ``vllm:prompt_tokens``).
def _variants(name: str):
    return (name, name + "_total") if not name.endswith("_total") else (name,)


def _parse(text: str) -> dict[str, list[tuple[dict, float]]]:
    """Return {sample_name: [(labels, value), ...]}."""
    out: dict[str, list[tuple[dict, float]]] = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            out.setdefault(s.name, []).append((dict(s.labels), float(s.value)))
    return out


def _first(samples: list[tuple[dict, float]]) -> float | None:
    return samples[0][1] if samples else None


def _sum(samples: list[tuple[dict, float]]) -> float | None:
    return sum(v for _, v in samples) if samples else None


def _max(samples: list[tuple[dict, float]]) -> float | None:
    return max(v for _, v in samples) if samples else None


def _mean(samples: list[tuple[dict, float]]) -> float | None:
    return (sum(v for _, v in samples) / len(samples)) if samples else None


def _clamp01(v: float | None) -> float | None:
    return None if v is None else max(0.0, min(1.0, v))


class _Counter:
    """Monotonic counter with reset/gap/identity-aware baselining."""

    __slots__ = ("val", "ts")

    def __init__(self) -> None:
        self.val: float | None = None
        self.ts: float | None = None

    def update(self, value: float, now: float) -> tuple[float, float | None]:
        """Return (delta, rate_per_s). First call -> (0, None)."""
        if self.val is None:
            self.val, self.ts = value, now
            return 0.0, None
        delta = value - self.val
        if value < self.val or (now - self.ts) > GAP_RESET_S:
            # reset or gap: establish a fresh baseline, no negative spike
            self.val, self.ts = value, now
            return 0.0, None
        elapsed = now - self.ts
        self.val, self.ts = value, now
        rate = delta / elapsed if elapsed > 0 else None
        return delta, rate


class MetricPoller:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(follow_redirects=False,
                                         timeout=httpx.Timeout(2.0, connect=1.0))
        # keyed by (profile_id, invocation_id)
        self._counters: dict[str, dict[str, dict[tuple, _Counter]]] = {}
        self._hist: dict[str, dict[str, dict[tuple, tuple[float, float, float]]]] = {}
        self._llamacpp_probe: dict[str, bool] = {}
        self._last_poll: dict[str, float] = {}

    async def close(self) -> None:
        await self._client.aclose()

    def _key(self, identity: RuntimeIdentity) -> str:
        return f"{identity.profile_id}:{identity.invocation_id}"

    def _counters_for(self, identity: RuntimeIdentity) -> dict[str, dict[tuple, _Counter]]:
        return self._counters.setdefault(self._key(identity), {})

    def _hist_for(self, identity: RuntimeIdentity) -> dict[str, dict[tuple, tuple[float, float, float]]]:
        return self._hist.setdefault(self._key(identity), {})

    async def _get_metrics(self, base: str) -> str | None:
        r = await self._client.get(base + "/metrics",
                                   headers={"Accept": "text/plain"})
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            return None
        return r.text[:2 * 1024 * 1024]

    async def poll(self, profile, identity: RuntimeIdentity) -> RuntimeSnapshot:
        snap = RuntimeSnapshot(identity=identity, ts=time.time())
        base = profile.base_url
        if not base:
            return snap
        if profile.backend == "ninfer":
            return snap  # filled by the NInfer JSONL reader (merged by caller)
        text = await self._get_metrics(base)
        if text is None:
            snap.sources = ["metrics-404" if profile.backend == "llamacpp" else "metrics-unavailable"]
            return snap
        data = _parse(text)
        now = time.monotonic()
        counters = self._counters_for(identity)
        hist = self._hist_for(identity)
        snap.sources = ["prometheus"]
        self._fill(profile.backend, data, snap, counters, hist, now)
        return snap

    # --- per-backend -----------------------------------------------------
    def _fill(self, backend: str, data, snap: RuntimeSnapshot, counters, hist, now) -> None:
        if backend == "vllm":
            self._vllm(data, snap, counters, hist, now)
        elif backend == "sglang":
            self._sglang(data, snap, counters, hist, now)
        elif backend == "llamacpp":
            self._llamacpp(data, snap, counters, hist, now)
        elif backend == "openai":
            pass  # generic: identity + cgroup + benchmark only

    def _counter(self, data, counters, metric: str, now: float) -> tuple[float, float | None]:
        samples = self._lookup(data, metric)
        if not samples:
            return 0.0, None
        total = 0.0
        rate: float | None = None
        for labels, val in samples:
            c = counters.setdefault(metric, {}).setdefault(tuple(sorted(labels.items())), _Counter())
            d, r = c.update(val, now)
            total += d
            if r is not None:
                rate = r
        return total, rate

    @staticmethod
    def _lookup(data: dict, metric: str) -> list[tuple[dict, float]]:
        for name in _variants(metric):
            if name in data:
                return data[name]
        return []

    def _vllm(self, data, snap: RuntimeSnapshot, counters, hist, now) -> None:
        running = _max(self._lookup(data, "vllm:num_requests_running"))
        waiting = _max(self._lookup(data, "vllm:num_requests_waiting"))
        snap.running_requests = Metric(running, "vllm", time.time())
        snap.queue_requests = Metric(waiting, "vllm", time.time())
        d_prompt, _ = self._counter(data, counters, "vllm:prompt_tokens", now)
        d_gen, _ = self._counter(data, counters, "vllm:generation_tokens", now)
        d_req, _ = self._counter(data, counters, "vllm:request_success", now)
        snap.observed_prompt_tokens = Metric(d_prompt, "vllm", time.time())
        snap.observed_completion_tokens = Metric(d_gen, "vllm", time.time())
        snap.requests_total = Metric(d_req, "vllm", time.time())
        snap.prompt_rate = Metric(d_prompt, "vllm", time.time())
        snap.completion_rate = Metric(d_gen, "vllm", time.time())
        snap.kv_usage_frac = Metric(_clamp01(_first(self._lookup(data, "vllm:kv_cache_usage_perc"))), "vllm", time.time())
        d_hits, _ = self._counter(data, counters, "vllm:prefix_cache_hits", now)
        d_queries, _ = self._counter(data, counters, "vllm:prefix_cache_queries", now)
        snap.prefix_cache_hits = Metric(d_hits, "vllm", time.time())
        snap.prefix_cache_queries = Metric(d_queries, "vllm", time.time())
        self._hist_mean(data, hist, "vllm:time_to_first_token_seconds", snap.ttft_mean_s, snap.ttft_samples)
        self._hist_mean(data, hist, "vllm:e2e_request_latency_seconds", snap.e2e_mean_s, snap.e2e_samples)

    def _sglang(self, data, snap: RuntimeSnapshot, counters, hist, now) -> None:
        snap.running_requests = Metric(_max(self._lookup(data, "sglang:num_running_reqs")), "sglang", time.time())
        snap.queue_requests = Metric(_max(self._lookup(data, "sglang:num_queue_reqs")), "sglang", time.time())
        d_prompt, _ = self._counter(data, counters, "sglang:prompt_tokens", now)
        d_gen, _ = self._counter(data, counters, "sglang:generation_tokens", now)
        d_req, _ = self._counter(data, counters, "sglang:num_requests", now)
        snap.observed_prompt_tokens = Metric(d_prompt, "sglang", time.time())
        snap.observed_completion_tokens = Metric(d_gen, "sglang", time.time())
        snap.requests_total = Metric(d_req, "sglang", time.time())
        snap.prompt_rate = Metric(d_prompt, "sglang", time.time())
        snap.completion_rate = Metric(d_gen, "sglang", time.time())
        # token_usage is KV/SSM pool occupancy (0-1), not VRAM/context tokens
        snap.kv_usage_frac = Metric(_clamp01(_first(self._lookup(data, "sglang:token_usage"))), "sglang", time.time())
        snap.cache_hit_rate = Metric(_clamp01(_first(self._lookup(data, "sglang:cache_hit_rate"))), "sglang", time.time())
        snap.spec_accept_rate = Metric(_clamp01(_first(self._lookup(data, "sglang:spec_accept_rate"))), "sglang", time.time())
        snap.spec_accept_length = Metric(_first(self._lookup(data, "sglang:spec_accept_length")), "sglang", time.time())
        self._hist_mean(data, hist, "sglang:time_to_first_token_seconds", snap.ttft_mean_s, snap.ttft_samples)
        self._hist_mean(data, hist, "sglang:e2e_request_latency_seconds", snap.e2e_mean_s, snap.e2e_samples)

    def _llamacpp(self, data, snap: RuntimeSnapshot, counters, hist, now) -> None:
        self._llamacpp_probe["llamacpp"] = True
        d_prompt, _ = self._counter(data, counters, "llamacpp:prompt_tokens", now)
        d_pred, _ = self._counter(data, counters, "llamacpp:tokens_predicted", now)
        snap.observed_prompt_tokens = Metric(d_prompt, "llamacpp", time.time())
        snap.observed_completion_tokens = Metric(d_pred, "llamacpp", time.time())
        snap.engine_prefill_tps = Metric(_first(self._lookup(data, "llamacpp:prompt_tokens_seconds")), "llamacpp", time.time())
        snap.engine_decode_tps = Metric(_first(self._lookup(data, "llamacpp:predicted_tokens_seconds")), "llamacpp", time.time())
        snap.running_requests = Metric(_max(self._lookup(data, "llamacpp:requests_processing")), "llamacpp", time.time())
        snap.queue_requests = Metric(_max(self._lookup(data, "llamacpp:requests_deferred")), "llamacpp", time.time())

    def _hist_mean(self, data, hist: dict, metric: str, out_mean: Metric, out_count: Metric) -> None:
        samples = data.get(metric + "_sum", [])
        counts = data.get(metric + "_count", [])
        if not samples or not counts:
            return
        sum_val = sum(v for _, v in samples)
        cnt = sum(v for _, v in counts)
        # single label set assumed; store baseline sum/count
        key = tuple(sorted((counts[0][0] if counts else {}).items()))
        entry = hist.setdefault(metric, {}).setdefault(key, (0.0, 0.0))
        bsum, bcnt = entry
        dsum = sum_val - bsum
        dcnt = cnt - bcnt
        if dcnt < 0 or sum_val < bsum:
            hist[metric][key] = (sum_val, cnt)
            out_mean.value, out_mean.reason = None, "baseline"
            out_count.value = 0.0
            return
        hist[metric][key] = (sum_val, cnt)
        if dcnt > 0:
            out_mean.value, out_mean.reason = dsum / dcnt, None
            out_count.value = float(dcnt)
        else:
            out_mean.value, out_mean.reason = None, "no samples in window"
            out_count.value = 0.0


def _m(v: float | None, source: str) -> Metric:
    return Metric(v, source, time.time())
