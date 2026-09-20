"""Lifecycle smoke test against real user-systemd transient units.

Uses a tiny python HTTP fixture (no GPU/VRAM). Exercises:
  STARTING->READY, repeated-start no-op, quit-persistence, reattach,
  PORT CONFLICT, missing-file disabled start, wrong-model-id -> UNHEALTHY.
Run from the project root:  .venv/bin/python tests/smoke_lifecycle.py
"""
from __future__ import annotations

import asyncio
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daish.config import ModelProfile, AppConfig
from daish.control import ModelController, Systemd, port_owners

PY = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable

SERVER = r'''
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "18081"))
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MODEL", "daish-test-model")
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": MODEL}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a):
        pass
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
'''


class StubApp:
    def __init__(self, profiles):
        self.config = AppConfig(profiles=profiles)
        self.events = []

    def get_profile(self, pid):
        return next((p for p in self.config.profiles if p.id == pid), None)

    def emit_event(self, pid, kind, data):
        self.events.append((pid, kind, data))


def free_port(prefer):
    for port in [prefer] + list(range(18082, 18091)):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port)); s.close(); return port
        except OSError:
            s.close()
    raise RuntimeError("no free test port")


def make_profile(pid, unit, port, model_id, model_paths=None, fixture_model=None):
    argv = [PY, "-c", SERVER, str(port), fixture_model or model_id]
    return ModelProfile(id=pid, label=pid, backend="openai", control="transient",
                        unit=unit, cwd=str(Path.home()), env={},
                        argv=argv, base_url=f"http://127.0.0.1:{port}",
                        model_id=model_id, model_paths=model_paths or [])


async def poll_ready(ctrl, app, pid, timeout=30.0):
    """Drive the same health+reconcile cadence the TUI worker does."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        p = app.get_profile(pid)
        ok, msg = await ctrl._check_ready(p)
        ctrl.set_health(pid, ok, msg)
        statuses = await ctrl.reconcile()
        last = next((s for s in statuses if s.profile_id == pid), None)
        if last and last.state == "READY":
            return last
        await asyncio.sleep(0.5)
    return last


async def main():
    suffix = f"daishtest{int(time.time())}"
    port = free_port(18081)
    unit = f"{suffix}.service"
    profile = make_profile("t1", unit, port, "daish-test-model")
    app = StubApp([profile])
    ctrl = ModelController(app)
    sd = Systemd()
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    try:
        # 1. STARTING -> READY
        print("1) start -> STARTING -> READY")
        ok, msg = await ctrl.start("t1")
        check("start submitted", ok, msg)
        st = ctrl.status["t1"]
        check("state STARTING after start", st.state == "STARTING", st.state)
        st = await poll_ready(ctrl, app, "t1")
        check("reaches READY", st is not None and st.state == "READY",
              st.state if st else "None")
        check("port owned by unit", st.main_pid in port_owners(port))

        # 2. repeated start no-op
        print("2) repeated start is a no-op")
        ok2, msg2 = await ctrl.start("t1")
        check("second start refused", not ok2, msg2)

        # 3. quit-persistence: close controller (TUI exit) without stopping unit
        print("3) quit-persistence + reattach")
        await ctrl.close()
        await asyncio.sleep(0.5)
        info = await sd.unit_info(unit)
        pid_before = info.main_pid
        check("unit survives TUI exit", info.active and info.main_pid == pid_before)
        check("port still owned after exit", pid_before in port_owners(port))
        # new controller instance -> explicit start() reattaches to the active unit
        ctrl2 = ModelController(app)
        rok, rmsg = await ctrl2.start("t1")
        check("start() reports reattach", rok and "reattach" in rmsg, rmsg)
        check("reattach event", any(k == "reattach" for (_, k, _) in app.events))
        st = None
        for _ in range(20):
            p = app.get_profile("t1")
            ok, msg = await ctrl2._check_ready(p)
            ctrl2.set_health("t1", ok, msg)
            statuses = await ctrl2.reconcile()
            st = next((s for s in statuses if s.profile_id == "t1"), None)
            if st and st.state == "READY":
                break
            await asyncio.sleep(0.5)
        check("reattached to READY", st is not None and st.state == "READY",
              st.state if st else "None")
        ctrl = ctrl2

        # 4. PORT CONFLICT: second profile same port
        print("4) PORT CONFLICT")
        port2 = port  # same port
        unit2 = f"{suffix}b.service"
        prof2 = make_profile("t2", unit2, port2, "daish-test-model")
        app.config.profiles.append(prof2)
        ok3, msg3 = await ctrl.start("t2")
        check("conflicting start refused", not ok3, msg3)
        st2 = ctrl.status["t2"]
        check("port_conflict flag set", st2.port_conflict, f"foreign={st2.foreign_pid}")

        # 5. missing file -> start disabled
        print("5) missing-file disables start")
        unit3 = f"{suffix}c.service"
        port3 = free_port(18090)
        prof3 = make_profile("t3", unit3, port3, "m",
                             model_paths=[str(Path.home() / "no-such-weights-xyz")])
        app.config.profiles.append(prof3)
        ok4, msg4 = await ctrl.start("t3")
        check("missing-file start refused", not ok4, msg4)

        # 6. wrong model-id -> UNHEALTHY
        print("6) wrong model-id -> UNHEALTHY")
        unit4 = f"{suffix}d.service"
        port4 = free_port(18089)
        prof4 = make_profile("t4", unit4, port4, "wrong-model-id",
                             fixture_model="daish-test-model")
        # fixture MODEL is 'daish-test-model'; profile wants a different id
        prof4.startup_timeout_seconds = 6.0
        app.config.profiles.append(prof4)
        ok5, msg5 = await ctrl.start("t4")
        check("t4 start submitted", ok5, msg5)
        st4 = None
        deadline = time.time() + 15
        while time.time() < deadline:
            p = app.get_profile("t4")
            ok, msg = await ctrl._check_ready(p)
            ctrl.set_health("t4", ok, msg)
            statuses = await ctrl.reconcile()
            st4 = next((s for s in statuses if s.profile_id == "t4"), None)
            if st4 and st4.state in ("UNHEALTHY",):
                break
            await asyncio.sleep(0.5)
        check("t4 reaches UNHEALTHY", st4 is not None and st4.state == "UNHEALTHY",
              st4.state if st4 else "None")
    finally:
        print("cleanup: stopping test units")
        for u in (unit, unit2, unit3, unit4):
            await sd.stop_unit(u)
        await ctrl.close()

    print("\nSUMMARY")
    failed = [r for r in results if not r[1]]
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
