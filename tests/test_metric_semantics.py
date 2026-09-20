"""Metric semantics regression tests: counter reset, label attribution,
unsupported-vs-zero, and duplicate NInfer completion events.

These target the *semantics*, not source text: counter baselines, histogram
mean windows, per-label isolation, and NInfer request dedup.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from daish.metrics import _Counter, _parse
from daish.ninfer_log import NInferLogReader


# --- Counter baselines -----------------------------------------------------
def test_counter_first_sample_is_baseline():
    c = _Counter()
    d, r = c.update(100.0, 10.0)
    assert d == 0.0 and r is None


def test_counter_delta_and_rate():
    c = _Counter()
    c.update(100.0, 10.0)
    d, r = c.update(130.0, 12.0)
    assert d == 30.0          # 30 observed tokens
    assert r == 15.0          # 15 tok/s


def test_counter_reset_no_negative_rate():
    c = _Counter()
    c.update(100.0, 10.0)
    c.update(130.0, 12.0)
    d, r = c.update(5.0, 14.0)   # counter reset
    assert d == 0.0 and r is None  # no negative spike, no extra tokens


def test_counter_gap_resets_baseline():
    c = _Counter()
    c.update(10.0, 100.0)
    d, r = c.update(20.0, 200.0)  # > GAP_RESET_S gap
    assert d == 0.0 and r is None


def test_parse_labels_preserved():
    data = _parse(
        '# HELP a a\n# TYPE a counter\n'
        'a{model="m1"} 10\na{model="m2"} 20\n')
    # prometheus parser normalizes counter names with a _total suffix
    key = "a_total" if "a_total" in data else "a"
    vals = sorted(v for _, v in data[key])
    assert vals == [10.0, 20.0]

# --- Histogram mean window -------------------------------------------------
def test_histogram_mean_zero_observations_is_na():
    # No _count increment -> mean must be None, not 0
    from daish.metrics import MetricPoller
    from daish.state import RuntimeSnapshot, Metric, RuntimeIdentity
    pol = MetricPoller()
    idn = RuntimeIdentity("p", "inv")
    hist = pol._hist_for(idn)
    snap = RuntimeSnapshot(identity=idn)
    # sum present, count unchanged
    data = {"h_sum": [({"x": "1"}, 5.0)], "h_count": [({"x": "1"}, 10.0)]}
    pol._hist_mean(data, hist, "h", snap.ttft_mean_s, snap.ttft_samples)
    assert snap.ttft_mean_s.value is not None  # first window has 10 samples
    # second call with same count -> N/A
    pol._hist_mean(data, hist, "h", snap.ttft_mean_s, snap.ttft_samples)
    assert snap.ttft_mean_s.value is None


# --- NInfer dedup + schema -------------------------------------------------
def _req(sid, rid, prompt, comp, cached):
    return {"artifact_type": "ninfer_serve_request_log", "schema_version": 21,
            "event": "request_done", "server_instance_id": sid,
            "timestamp_unix_ms": 1, "request": {"request_id": rid},
            "result": {"prompt_tokens": prompt, "completion_tokens": comp,
                       "prefix_cache_hit_tokens": cached},
            "timings_seconds": {"ttft": 0.1, "total": 1.0},
            "speculative": {"drafted_tokens": 10, "accepted_tokens": 4}}


def _thr():
    return {"artifact_type": "ninfer_serve_request_log", "schema_version": 21,
            "event": "throughput", "server_instance_id": "s1",
            "throughput_tokens_per_second": {"prefill": 10.0, "decode": 5.0},
            "scheduler": {"running": 1, "waiting": 0}, "interval_seconds": 5.0}


def test_ninfer_duplicate_request_counted_once(tmp_path: Path):
    f = tmp_path / "log.jsonl"
    f.write_text(json.dumps(_req("s1", "1", 100, 20, 40)) + "\n")
    r = NInferLogReader(f)
    r.read()  # establishes EOF
    with f.open("a") as fh:
        fh.write(json.dumps(_req("s1", "1", 100, 20, 40)) + "\n")  # duplicate
    r.read()
    t = r.totals
    assert t["requests"] == 1
    assert t["prompt_tokens"] == 100
    assert t["completion_tokens"] == 20
    assert abs(r.cache_hit_rate - 0.4) < 1e-9


def test_ninfer_throughput_does_not_increment_totals(tmp_path: Path):
    f = tmp_path / "log.jsonl"
    f.write_text("")
    r = NInferLogReader(f)
    r.read()  # baseline at EOF (empty)
    with f.open("a") as fh:
        fh.write(json.dumps(_req("s1", "1", 100, 20, 40)) + "\n")
    r.read()
    assert r.totals["requests"] == 1
    with f.open("a") as fh:
        fh.write(json.dumps(_thr()) + "\n")  # throughput is interval data, not a token counter
    c = r.read()
    assert r.totals["requests"] == 1
    assert c["d_requests"] == 0
    assert c["throughput"]["decode_tps"] == 5.0


def test_ninfer_same_request_id_new_server_is_new_request(tmp_path: Path):
    f = tmp_path / "log.jsonl"
    f.write_text("")
    r = NInferLogReader(f)
    r.read()
    with f.open("a") as fh:
        fh.write(json.dumps(_req("s1", "1", 10, 5, 0)) + "\n")
        fh.write(json.dumps(_req("s2", "1", 10, 5, 0)) + "\n")
    r.read()
    assert r.totals["requests"] == 2


def test_ninfer_split_and_unknown_schema(tmp_path: Path):
    f = tmp_path / "log.jsonl"
    f.write_text("")
    r = NInferLogReader(f)
    r.read()
    good = json.dumps(_req("s1", "7", 30, 12, 0))
    bad = json.dumps({"artifact_type": "ninfer_serve_request_log",
                      "schema_version": 99, "event": "request_done",
                      "request": {"request_id": 1}})
    with f.open("a") as fh:
        fh.write(good[:12]); fh.flush()
        r.read()  # partial trailing line must not corrupt state
        fh.write(good[12:] + "\n" + bad + "\n" + json.dumps(_req("s1", "8", 5, 2, 0)) + "\n")
    c = r.read()
    assert r.totals["requests"] == 2  # s1/7 and s1/8; the schema-99 record ignored
    assert "schema_version=99" in c["diagnostics"]
    assert r.totals["prompt_tokens"] == 35
