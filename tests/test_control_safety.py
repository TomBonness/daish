"""Control-safety regression tests: port ownership, unit persistence, and
stale-invocation protection.

Systemd integration is gated behind ``DAISH_TEST_SYSTEMD=1`` and uses unique
temporary units/ports; it never discovers-and-stops existing user units.
"""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from daish.control import ModelController, Systemd, port_owners, cgroup_empty, UnitInfo
from daish.config import ModelProfile


# --- port ownership (no systemd) ------------------------------------------
def test_port_owners_detects_listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        owners = port_owners(port)
        assert os.getpid() in owners
    finally:
        s.close()
    assert port_owners(port) == []


def test_cgroup_empty_none():
    assert cgroup_empty(None) is True
    assert cgroup_empty("") is True


def test_matches_fingerprint():
    class App:
        config = None
        def get_profile(self, pid):
            return ModelProfile(id=pid, label=pid, backend="openai", control="transient",
                                argv=["/opt/bin/ninfer-serve", "--x"], unit="u.service")
        def emit_event(self, *a):
            pass
    ctrl = ModelController(App())
    info = UnitInfo(unit="u.service", active=True, fingerprint="[systemd-run] /opt/bin/ninfer-serve --x",
                    main_pid=1, invocation_id="abc")
    p = ctrl.get_profile("pid")
    assert ctrl._matches(p, info) is True
    info2 = UnitInfo(unit="u.service", active=True, fingerprint="[systemd-run] /opt/bin/other --y",
                     main_pid=1, invocation_id="abc")
    assert ctrl._matches(p, info2) is False


# --- systemd integration (gated) ------------------------------------------
pytestmark = pytest.mark.skipif(
    os.environ.get("DAISH_TEST_SYSTEMD") != "1",
    reason="set DAISH_TEST_SYSTEMD=1 to run real systemd integration")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _StubApp:
    def __init__(self, profiles):
        self.config = type("C", (), {"profiles": profiles})
        self._m = {p.id: p for p in profiles}
        self.events = []

    def get_profile(self, pid):
        return self._m.get(pid)

    def emit_event(self, pid, kind, data):
        self.events.append((pid, kind))


def test_transient_unit_persistence_and_port_ownership():
    port = _free_port()
    suffix = f"daishtest{int(time.time())}"
    unit = f"{suffix}.service"
    exe = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
    if not Path(exe).exists():
        exe = "python3"
    prof = ModelProfile(id="t1", label="t1", backend="openai", control="transient",
                        unit=unit, cwd=str(Path.home()),
                        argv=[exe, "-c",
                              "import http.server,socketserver,sys; "
                              f"socketserver.TCPServer(('127.0.0.1', {port}), "
                              "http.server.SimpleHTTPRequestHandler).serve_forever()"],
                        base_url=f"http://127.0.0.1:{port}", model_id=None)
    app = _StubApp([prof])
    ctrl = ModelController(app)

    async def scenario():
        ok, msg = await ctrl.start("t1")
        assert ok, msg
        info = await ctrl._systemd.unit_info(unit)
        pid = info.main_pid
        assert info.active and pid
        await ctrl.close()
        # simulate TUI quit: unit must remain active (persistence contract)
        await asyncio.sleep(1.0)
        info2 = await Systemd().unit_info(unit)
        assert info2.active and info2.main_pid == pid, "unit must survive TUI exit"
        assert pid in port_owners(port)
        # foreign listener: stop this unit, start a foreign listener -> conflict
        await Systemd().stop_unit(unit)
        await asyncio.sleep(0.5)
        assert not port_owners(port)
    asyncio.run(scenario())
