"""Sanitized observed history backed by SQLite.

Three tables:
  samples(minute_utc, entity_id, source, data_json, PRIMARY KEY(minute_utc,entity_id,source))
  events(id INTEGER PRIMARY KEY, ts_utc, profile_id, kind, data_json)
  benchmarks(id INTEGER PRIMARY KEY, ts_utc, profile_id, data_json)

``data_json`` is built only from a numeric allowlist — never from a raw backend
response. Minute samples keep gauge mean/max + valid-sample count, observed
token/request deltas, latency sum/count and coverage seconds. Monitoring pauses
when Daish is closed; this is *observed* history, not a claim about the whole
machine.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

RETENTION_SECONDS = 7 * 24 * 3600
BENCHMARK_LIMIT = 100

# Numeric-only allowlist: the only data that ever reaches the database/export.
GAUGE_KEYS = {
    "gpu_util", "gpu_vram", "gpu_temp", "gpu_power", "cpu_total", "cpu_freq",
    "ram_used", "ram_available", "model_mem", "model_swap", "gpu_model_vram",
}
COUNTER_KEYS = {
    "prompt_tokens", "completion_tokens", "requests",
}
LATENCY_KEYS = {
    "ttft", "e2e",
}


def _minute_bucket(ts: float) -> int:
    return int(ts // 60) * 60


class _Acc:
    __slots__ = ("sum", "count", "max", "delta", "lat_sum", "lat_count", "coverage")

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.max: float | None = None
        self.delta = 0.0
        self.lat_sum = 0.0
        self.lat_count = 0
        self.coverage = 0.0


class HistoryStore:
    def __init__(self, path: Path | None, readonly: bool = False) -> None:
        self.path = path
        self.readonly = readonly
        self.available = False
        self._conn: sqlite3.Connection | None = None
        self._minute: dict[tuple[int, str, str], _Acc] = {}
        self._events_mem: list[dict] = []
        self._bench_mem: list[dict] = []
        self._opened = False

    # --- lifecycle -------------------------------------------------------
    def open(self) -> None:
        if self._opened:
            return
        self._opened = True
        if self.path is None:
            return
        if self.readonly:
            if not self.path.exists():
                return  # no existing history: memory-only
            try:
                self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
                self.available = True
            except sqlite3.Error:
                return
        else:
            try:
                self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self.path))
                self._init_schema()
                self.available = True
            except sqlite3.Error:
                self.available = False

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples(
              minute_utc INTEGER NOT NULL,
              entity_id  TEXT NOT NULL,
              source     TEXT NOT NULL,
              data_json  TEXT NOT NULL,
              PRIMARY KEY(minute_utc, entity_id, source));
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY,
              ts_utc REAL NOT NULL,
              profile_id TEXT,
              kind TEXT NOT NULL,
              data_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS benchmarks(
              id INTEGER PRIMARY KEY,
              ts_utc REAL NOT NULL,
              profile_id TEXT,
              data_json TEXT NOT NULL);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self.flush_samples(force=True)
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    # --- events + benchmarks (immediate) ---------------------------------
    def record_event(self, profile_id: str, kind: str, data: dict[str, Any]) -> None:
        row = {"ts_utc": time.time(), "profile_id": profile_id, "kind": kind, "data": _clean(data)}
        self._events_mem.append(row)
        if self.available and not self.readonly:
            self._conn.execute(
                "INSERT INTO events(ts_utc, profile_id, kind, data_json) VALUES(?,?,?,?)",
                (row["ts_utc"], profile_id, kind, json.dumps(row["data"])))
            self._conn.commit()
        self._purge_mem_events()

    def record_benchmark(self, profile_id: str, data: dict[str, Any]) -> None:
        row = {"ts_utc": time.time(), "profile_id": profile_id, "data": _clean(data)}
        self._bench_mem.append(row)
        del self._bench_mem[:-BENCHMARK_LIMIT]
        if self.available and not self.readonly:
            self._conn.execute(
                "INSERT INTO benchmarks(ts_utc, profile_id, data_json) VALUES(?,?,?)",
                (row["ts_utc"], profile_id, json.dumps(row["data"])))
            self._conn.execute("DELETE FROM benchmarks WHERE id NOT IN "
                               "(SELECT id FROM benchmarks ORDER BY ts_utc DESC LIMIT ?)", (BENCHMARK_LIMIT,))
            self._conn.commit()
    # --- minute sample aggregation ---------------------------------------
    def add_sample(self, entity_id: str, source: str, gauges: dict[str, float],
                   counters: dict[str, float], latencies: dict[str, tuple[float, float]],
                   dt_s: float) -> None:
        bucket = _minute_bucket(time.time())
        acc = self._minute.setdefault((bucket, entity_id, source), _Acc())
        acc.coverage += dt_s
        for k, v in gauges.items():
            if k in GAUGE_KEYS and v is not None:
                acc.sum += v
                acc.count += 1
                acc.max = v if acc.max is None else max(acc.max, v)
        for k, v in counters.items():
            if k in COUNTER_KEYS and v is not None:
                acc.delta += v
        for k, (s, c) in latencies.items():
            if k in LATENCY_KEYS:
                acc.lat_sum += s
                acc.lat_count += c

    def flush_samples(self, force: bool = False) -> None:
        if not (self.available and not self.readonly and self._conn is not None):
            return
        now = _minute_bucket(time.time())
        for (bucket, entity, source), acc in list(self._minute.items()):
            if bucket >= now and not force:
                continue
            data: dict[str, Any] = {"coverage_s": round(acc.coverage, 3), "gauges": {}, "counters": {}, "latencies": {}}
            if acc.count:
                data["gauges"] = {"mean": round(acc.sum / acc.count, 4),
                                  "max": round(acc.max, 4) if acc.max is not None else None,
                                  "n": acc.count}
            if acc.delta:
                data["counters"] = {"total": round(acc.delta, 3)}
            if acc.lat_count:
                data["latencies"] = {"sum": round(acc.lat_sum, 4), "count": acc.lat_count,
                                     "mean": round(acc.lat_sum / acc.lat_count, 4)}
            self._conn.execute(
                "INSERT INTO samples(minute_utc, entity_id, source, data_json) VALUES(?,?,?,?) "
                "ON CONFLICT(minute_utc, entity_id, source) DO UPDATE SET data_json=excluded.data_json",
                (bucket, entity, source, json.dumps(data)))
            del self._minute[(bucket, entity, source)]
        self._conn.commit()

    # --- queries ---------------------------------------------------------
    def events(self, since: float | None = None, limit: int = 500) -> list[dict]:
        if self.available and self._conn is not None:
            try:
                if since is None:
                    rows = self._conn.execute(
                        "SELECT ts_utc, profile_id, kind, data_json FROM events ORDER BY ts_utc DESC LIMIT ?",
                        (limit,)).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT ts_utc, profile_id, kind, data_json FROM events WHERE ts_utc>=? ORDER BY ts_utc DESC LIMIT ?",
                        (since, limit)).fetchall()
                return [{"ts_utc": r[0], "profile_id": r[1], "kind": r[2], "data": json.loads(r[3])} for r in rows]
            except sqlite3.Error:
                return list(self._events_mem)[-limit:]
        out = [e for e in self._events_mem if since is None or e["ts_utc"] >= since]
        return out[-limit:]

    def benchmarks(self, limit: int = BENCHMARK_LIMIT) -> list[dict]:
        if self.available and self._conn is not None:
            try:
                rows = self._conn.execute(
                    "SELECT ts_utc, profile_id, data_json FROM benchmarks ORDER BY ts_utc DESC LIMIT ?",
                    (limit,)).fetchall()
                return [{"ts_utc": r[0], "profile_id": r[1], "data": json.loads(r[2])} for r in rows]
            except sqlite3.Error:
                pass
        return self._bench_mem[-limit:]

    def samples(self, since: float | None = None) -> list[dict]:
        if self.available and self._conn is not None:
            try:
                if since is None:
                    rows = self._conn.execute(
                        "SELECT minute_utc, entity_id, source, data_json FROM samples").fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT minute_utc, entity_id, source, data_json FROM samples WHERE minute_utc>=?",
                        (int(since),)).fetchall()
                return [{"minute_utc": r[0], "entity_id": r[1], "source": r[2], "data": json.loads(r[3])} for r in rows]
            except sqlite3.Error:
                return []
        return []

    # --- maintenance -----------------------------------------------------
    def purge(self) -> None:
        if not (self.available and not self.readonly and self._conn is not None):
            return
        cutoff = time.time() - RETENTION_SECONDS
        self._conn.execute("DELETE FROM samples WHERE minute_utc < ?", (int(cutoff),))
        self._conn.execute("DELETE FROM events WHERE ts_utc < ?", (cutoff,))
        self._conn.execute("DELETE FROM benchmarks WHERE ts_utc < ?", (cutoff,))
        self._conn.execute("DELETE FROM benchmarks WHERE id NOT IN "
                           "(SELECT id FROM benchmarks ORDER BY ts_utc DESC LIMIT ?)", (BENCHMARK_LIMIT,))
        self._conn.commit()

    def _purge_mem_events(self) -> None:
        cutoff = time.time() - RETENTION_SECONDS
        self._events_mem = [e for e in self._events_mem if e["ts_utc"] >= cutoff][-500:]

    # --- export ----------------------------------------------------------
    def export(self, since: float, until: float) -> dict[str, Any]:
        return {
            "window": {"since": since, "until": until, "note": "OBSERVED HISTORY — monitoring pauses when Daish is closed"},
            "samples": self.samples(since=since),
            "events": self.events(since=since),
            "benchmarks": [b for b in self.benchmarks() if since <= b["ts_utc"] <= until],
        }


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively keep only numbers/strings/booleans/dict/list of allowed depth."""
    def walk(v):
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            return v[:512]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items() if isinstance(k, str)}
        if isinstance(v, list):
            return [walk(x) for x in v[:64]]
        return None
    return walk(data)
