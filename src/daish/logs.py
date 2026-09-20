"""Live user-journal log follower for a selected unit.

Follows only the selected unit via
``journalctl --user -u <unit> -n 200 -f -o json --no-pager`` and parses
MESSAGE/timestamp/priority. Visible retention is bounded to 2000 lines and
individual lines to 8 KiB. A denied/empty/unavailable journal is an explicit
source state, not proof the model is idle or healthy.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field

MAX_LINES = 2000
MAX_LINE = 8 * 1024

# Strip terminal control + OSC sequences so log text renders literally.
_CTL_RE = re.compile(r"(\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0b\x0c\x0e-\x1f])")


def strip_control(s: str) -> str:
    return _CTL_RE.sub("", s)


@dataclass
class LogLine:
    ts: float
    priority: int
    message: str
    pid: int | None = None

    @property
    def severity(self) -> str:
        return {0: "EMERG", 1: "ALERT", 2: "CRIT", 3: "ERR", 4: "WARN",
                5: "NOTICE", 6: "INFO", 7: "DEBUG"}.get(self.priority, "?")


class JournalFollower:
    def __init__(self) -> None:
        self.lines: deque[LogLine] = deque(maxlen=MAX_LINES)
        self.unit: str | None = None
        self.state: str = "idle"  # idle|following|paused|unavailable|denied|empty
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._read_head: int = 0  # index of first visible line

    async def follow(self, unit: str) -> None:
        """Start following a unit, reaping any previous follower."""
        await self.stop()
        self.unit = unit
        self.lines.clear()
        self._read_head = 0
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "journalctl", "--user", "-u", unit, "-n", "200", "-f", "-o", "json", "--no-pager",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL)
        except Exception:
            self.state = "unavailable"
            return
        if self._proc.stdout is None:
            self.state = "unavailable"
            return
        self.state = "following"
        self._task = asyncio.create_task(self._reader())

    async def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        buf = b""
        try:
            while True:
                chunk = await self._proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        self._ingest(line)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            if self.state == "following":
                self.state = "unavailable"

    def _ingest(self, raw: bytes) -> None:
        try:
            data = json.loads(raw[:65536])
        except json.JSONDecodeError:
            return
        msg = data.get("MESSAGE", "")
        if not msg:
            return
        ts = data.get("__REALTIME_TIMESTAMP")
        try:
            ts_f = int(ts) / 1_000_000 if ts else time.time()
        except (TypeError, ValueError):
            ts_f = time.time()
        prio = data.get("PRIORITY", 6)
        try:
            prio = int(prio)
        except (TypeError, ValueError):
            prio = 6
        pid = data.get("_PID")
        try:
            pid = int(pid) if pid else None
        except (TypeError, ValueError):
            pid = None
        msg = strip_control(msg)[:MAX_LINE]
        self.lines.append(LogLine(ts=ts_f, priority=prio, message=msg, pid=pid))
        if self.state == "empty":
            self.state = "following"

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._proc is not None:
            try:
                if self._proc.returncode is None:
                    self._proc.terminate()
                    await asyncio.wait_for(self._proc.wait(), timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self.state in ("following", "paused"):
            self.state = "idle"

    # --- queries ---------------------------------------------------------
    def search(self, text: str) -> list[LogLine]:
        if not text:
            return list(self.lines)
        low = text.lower()
        return [ln for ln in self.lines if low in ln.message.lower()]

    def by_severity(self, max_priority: int) -> list[LogLine]:
        return [ln for ln in self.lines if ln.priority <= max_priority]
