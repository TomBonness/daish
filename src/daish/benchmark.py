"""Small explicit inference benchmark.

Sends exactly one streamed ``POST /v1/chat/completions`` request to a ready
OpenAI-compatible endpoint. No warmup, concurrency test, or automatic retry.
Reports time to the first nonempty content/reasoning delta (TTFT), total
elapsed time, the exact returned prompt/completion token counts, and
``completion_tokens / total_elapsed`` explicitly labelled **end-to-end tok/s**.
NInfer's final ``timings.prompt_per_second`` / ``predicted_per_second`` supply
engine-reported prefill/decode speed when present; ``draft_n_accepted /
draft_n`` supplies acceptance. SSE chunks/characters are never counted as
tokens, and response-start latency is not presented as TTFT.
"""
from __future__ import annotations

import json
import time

import httpx

from .state import BenchmarkResult, Metric

PROMPT = "List the integers from 1 to 20, separated by spaces. Do not explain."
MAX_TOKENS = 128
TOTAL_TIMEOUT_S = 120.0


class BenchmarkRunner:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(follow_redirects=False)
        self.active = False

    async def close(self) -> None:
        await self._client.aclose()

    async def benchmark(self, profile) -> BenchmarkResult:
        res = BenchmarkResult(profile_id=profile.id, ts=time.time())
        base = profile.base_url
        if not base:
            res.detail = "no endpoint"
            return res
        payload = {
            "model": profile.model_id,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        self.active = True
        t0 = time.monotonic()
        ttft: float | None = None
        prompt_tokens = completion_tokens = None
        engine_prefill = engine_decode = None
        spec_accept = None
        try:
            async with self._client.stream(
                "POST", base + "/v1/chat/completions", json=payload,
                timeout=httpx.Timeout(TOTAL_TIMEOUT_S, connect=2.0)) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread())[:256]
                    res.total_s = Metric(time.monotonic() - t0, "wall")
                    res.detail = f"HTTP {resp.status_code}: {body!r}"
                    return res
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content") or delta.get("reasoning_content")
                        if content and ttft is None:
                            ttft = time.monotonic() - t0
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                    timings = chunk.get("timings")
                    if isinstance(timings, dict):
                        engine_prefill = _num(timings.get("prompt_per_second"))
                        engine_decode = _num(timings.get("predicted_per_second"))
                    dn = chunk.get("draft_n")
                    dna = chunk.get("draft_n_accepted")
                    if dn and dna is not None:
                        spec_accept = _num(dna) / _num(dn) if _num(dn) else None
        except httpx.TimeoutException:
            res.total_s = Metric(time.monotonic() - t0, "wall")
            res.detail = "timeout"
            return res
        except httpx.HTTPError as e:
            res.total_s = Metric(time.monotonic() - t0, "wall")
            res.detail = f"{type(e).__name__}: {e}"
            return res
        finally:
            self.active = False
        total = time.monotonic() - t0
        res.total_s = Metric(total, "wall")
        res.ttft_s = Metric(ttft, "wall", reason=None if ttft is not None else "no content delta")
        res.prompt_tokens = Metric(prompt_tokens, "usage", reason=None if prompt_tokens is not None else "no usage")
        res.completion_tokens = Metric(completion_tokens, "usage", reason=None if completion_tokens is not None else "no usage")
        if isinstance(completion_tokens, (int, float)) and total > 0:
            res.e2e_tps = Metric(completion_tokens / total, "derived", "completion_tokens/total_elapsed")
        res.engine_prefill_tps = Metric(engine_prefill, "engine", reason=None if engine_prefill is not None else "not reported")
        res.engine_decode_tps = Metric(engine_decode, "engine", reason=None if engine_decode is not None else "not reported")
        res.spec_accept_rate = Metric(spec_accept, "engine", reason=None if spec_accept is not None else "not reported")
        res.ok = ttft is not None
        if not res.ok:
            res.detail = res.detail or "no content received"
        return res


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None
