"""NInfer schema-21 JSONL request-log reader.

Reads only a configured ``ninfer_jsonl`` file, validating
``artifact_type="ninfer_serve_request_log"`` and ``schema_version=21``.
``request_done.result`` supplies prompt/completion/prefix-cache tokens;
``timings_seconds`` supplies ttft/prefill/decode/total; ``speculative`` supplies
drafted/accepted tokens; ``throughput`` supplies interval speed/queue only —
never a second token-total counter. Completed-request identity is
``(server_instance_id, request.request_id)`` so a reused request id under a new
server instance is a new request, and duplicates are counted once.

An already-active log is first read from its current EOF; history means
"observed by Daish", not a retrospective claim. Partial trailing lines,
replacement/truncation and unreadable paths are handled. Records are capped at
256 KiB and each read batch at 1 MiB. Malformed/unknown-schema records produce
one diagnostic per file/schema, not a crash or a stream of repeated alerts.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ARTIFACT_TYPE = "ninfer_serve_request_log"
SCHEMA_VERSION = 21
RECORD_MAX = 256 * 1024
BATCH_MAX = 1024 * 1024
THROUGHPUT_STALE_S = 15.0  # older than this -> "no recent report"


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


class NInferLogReader:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._offset: int | None = None
        self._inode: int | None = None
        self._partial = b""
        self._seen: set[tuple[str, str]] = set()
        # cumulative observed totals
        self.total_prompt = 0.0
        self.total_completion = 0.0
        self.total_prefix_hit = 0.0
        self.total_requests = 0
        self.total_drafted = 0.0
        self.total_accepted = 0.0
        self._diag: dict[str, str] = {}
        self._throughput: dict | None = None
        self._throughput_ts: float = 0.0
        self.available: bool = False

    def _stat(self):
        try:
            return os.stat(self.path)
        except OSError:
            return None

    def _ingest(self, line: bytes) -> None:
        raw = line[:RECORD_MAX]
        try:
            d = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._diag.setdefault("malformed", "malformed JSONL record")
            return
        if d.get("artifact_type") != ARTIFACT_TYPE:
            self._diag.setdefault("artifact", f"artifact_type={d.get('artifact_type')!r}")
            return
        if d.get("schema_version") != SCHEMA_VERSION:
            self._diag.setdefault("schema", f"schema_version={d.get('schema_version')!r}")
            return
        event = d.get("event")
        server = str(d.get("server_instance_id", ""))
        if event == "request_done":
            req = d.get("request", {}) or {}
            rid = str(req.get("request_id", ""))
            key = (server, rid)
            if key in self._seen:
                return  # count once
            self._seen.add(key)
            res = d.get("result", {}) or {}
            self.total_prompt += _num(res.get("prompt_tokens")) or 0.0
            self.total_completion += _num(res.get("completion_tokens")) or 0.0
            self.total_prefix_hit += _num(res.get("prefix_cache_hit_tokens")) or 0.0
            self.total_requests += 1
            spec = d.get("speculative", {}) or {}
            self.total_drafted += _num(spec.get("drafted_tokens")) or 0.0
            self.total_accepted += _num(spec.get("accepted_tokens")) or 0.0
        elif event == "throughput":
            self._throughput = d
            self._throughput_ts = time.time()

    def read(self) -> dict:
        """Read new records since the last call. Returns a contribution dict."""
        out = {"d_prompt": 0.0, "d_completion": 0.0, "d_prefix_hit": 0.0,
               "d_requests": 0, "d_drafted": 0.0, "d_accepted": 0.0,
               "throughput": None, "stale": True, "diagnostics": [], "available": False}
        st = self._stat()
        if st is None:
            self.available = False
            return out
        self.available = True
        # detect replacement / truncation by inode + size
        if self._inode is None:
            # first read: start at EOF (observe only, no retrospective claim)
            self._offset = st.st_size
            self._inode = st.st_ino
        elif st.st_ino != self._inode:
            self._offset = 0
            self._inode = st.st_ino
            self._partial = b""
            self._seen = set()
        elif st.st_size < self._offset:
            # truncated in place
            self._offset = 0
            self._partial = b""
        before = (self.total_prompt, self.total_completion, self.total_prefix_hit,
                  self.total_requests, self.total_drafted, self.total_accepted)
        try:
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read(BATCH_MAX)
                self._offset = f.tell()
        except OSError as e:
            self._diag.setdefault("unreadable", str(e))
            return out
        data = self._partial + chunk
        last_nl = data.rfind(b"\n")
        if last_nl == -1:
            self._partial = data[:RECORD_MAX]
        else:
            complete, self._partial = data[:last_nl + 1], data[last_nl + 1:]
            for line in complete.split(b"\n"):
                if line.strip():
                    self._ingest(line)
        out["d_prompt"] = self.total_prompt - before[0]
        out["d_completion"] = self.total_completion - before[1]
        out["d_prefix_hit"] = self.total_prefix_hit - before[2]
        out["d_requests"] = self.total_requests - before[3]
        out["d_drafted"] = self.total_drafted - before[4]
        out["d_accepted"] = self.total_accepted - before[5]
        # latest throughput / scheduler (interval measurements, not token totals)
        if self._throughput is not None:
            tps = self._throughput.get("throughput_tokens_per_second", {}) or {}
            sched = self._throughput.get("scheduler", {}) or {}
            out["throughput"] = {
                "prefill_tps": _num(tps.get("prefill")),
                "decode_tps": _num(tps.get("decode")),
                "running": _num(sched.get("running")),
                "waiting": _num(sched.get("waiting")),
                "prefilling": _num(sched.get("prefilling")),
                "decode_ready": _num(sched.get("decode_ready")),
            }
            out["stale"] = (time.time() - self._throughput_ts) > THROUGHPUT_STALE_S
        out["diagnostics"] = [d for d in self._diag.values()]
        return out

    # --- convenience aggregate getters -----------------------------------
    @property
    def totals(self) -> dict:
        return {"prompt_tokens": self.total_prompt, "completion_tokens": self.total_completion,
                "prefix_cache_hit_tokens": self.total_prefix_hit, "requests": self.total_requests,
                "drafted_tokens": self.total_drafted, "accepted_tokens": self.total_accepted}

    @property
    def spec_accept_rate(self) -> float | None:
        if self.total_drafted > 0:
            return self.total_accepted / self.total_drafted
        return None

    @property
    def cache_hit_rate(self) -> float | None:
        if self.total_prompt > 0:
            return self.total_prefix_hit / self.total_prompt
        return None

    @property
    def file_size(self) -> int:
        st = self._stat()
        return st.st_size if st else 0
