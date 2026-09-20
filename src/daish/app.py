"""DaishApp: the Textual application, service wiring and background workers.

Collectors run as application-owned workers (``exit_on_error=False``) and
publish snapshots to widgets via worker messages — no blocking I/O in event
handlers, no worker-thread widget mutation. Quitting cancels collectors and
journal followers, **not** model services.
"""
from __future__ import annotations

import asyncio
import os
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.worker import Worker

from . import art
from .benchmark import BenchmarkRunner
from .config import (AppConfig, default_config_path, default_history_path,
                     default_lock_path, load_config, ninfer_log_dir,
                     save_config, validate_profiles)
from .control import ModelController
from .discovery import (ProfileCandidate, discover_profiles, scan_inventory)
from .history import HistoryStore
from .logs import JournalFollower
from .metrics import MetricPoller
from .ninfer_log import NInferLogReader
from .state import (HardwareSnapshot, READY, STARTING, STOPPED,
                    RuntimeSnapshot)
from .telemetry import HardwareSampler


class HwMessage(Message):
    def __init__(self, snap: HardwareSnapshot) -> None:
        super().__init__()
        self.snap = snap


class StatusMessage(Message):
    def __init__(self, statuses) -> None:
        super().__init__()
        self.statuses = statuses


class RuntimeMessage(Message):
    def __init__(self, profile_id: str, snap: RuntimeSnapshot, ninfer: dict | None) -> None:
        super().__init__()
        self.profile_id = profile_id
        self.snap = snap
        self.ninfer = ninfer


class EventMessage(Message):
    def __init__(self, profile_id: str, kind: str, data: dict) -> None:
        super().__init__()
        self.profile_id = profile_id
        self.kind = kind
        self.data = data


class Alert:
    __slots__ = ("ts", "text", "level", "sticky")

    def __init__(self, text: str, level: str = "info", sticky: bool = True) -> None:
        self.ts = time.time()
        self.text = text
        self.level = level
        self.sticky = sticky


class DaishApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "DAISH / LOCAL AI COMMAND CENTER"
    BINDINGS = [
        ("1", "tab(0)", "Overview"), ("2", "tab(1)", "Models"),
        ("3", "tab(2)", "Hardware"), ("4", "tab(3)", "Activity"),
        ("down", "down_row", "Down"), ("up", "up_row", "Up"),
        ("j", "down_row", ""), ("k", "up_row", ""),
        ("enter", "details", "Details"), ("s", "start_model", "Start"),
        ("x", "stop_model", "Stop"), ("r", "restart_model", "Restart"),
        ("w", "switch_model", "Switch"), ("l", "logs_model", "Logs"),
        ("b", "benchmark_model", "Benchmark"), ("a", "add_model", "Add"),
        ("i", "import_profiles", "Import"), ("e", "edit_model", "Edit"),
        ("f1", "help", "Help"), ("question_mark", "help", "Help"),
        ("q", "quit_app", "Quit"), ("ctrl+q", "quit_app", "Quit"),
    ]
    SCREENS: dict = {}

    @classmethod
    def _register_screens(cls) -> None:
        if not cls.SCREENS:
            from .ui import BenchmarkScreen, ConfirmScreen, DetailsScreen
            cls.SCREENS = {"details": DetailsScreen, "benchmark": BenchmarkScreen,
                           "confirm": ConfirmScreen}

    def __init__(self, config_path: Path | None, readonly: bool, ascii_only: bool,
                 local_root: Path | None = None) -> None:
        super().__init__()
        self.config_path = config_path or default_config_path()
        self.readonly = readonly
        self.ascii_only = ascii_only
        self.local_root = local_root or (Path.home() / "Local")
        self.config: AppConfig | None = None
        self.config_error: str | None = None
        self.candidates: list[ProfileCandidate] = []
        self.profiles_loaded = False
        self.selected: str | None = None
        self._problems: list[str] = []
        # live state
        self.statuses: dict[str, Any] = {}
        self.snapshots: dict[str, RuntimeSnapshot] = {}
        self.hw_snap: HardwareSnapshot | None = None
        self.inventory = []
        self.alerts: deque[Alert] = deque(maxlen=200)
        self.events: deque = deque(maxlen=500)
        self.ninfer_totals: dict[str, dict] = {}
        self._starting_any = False
        self._lock_fd: int | None = None
        self._controlling = not readonly
        # services
        self.sampler: HardwareSampler | None = None
        self.controller: ModelController | None = None
        self.poller: MetricPoller | None = None
        self.history: HistoryStore | None = None
        self.logs = JournalFollower()
        self.bench: BenchmarkRunner | None = None
        self._ninfer: dict[str, NInferLogReader] = {}
        self._worker_stop = threading.Event()
        self._worker_handles: list = []

    # --- config / startup ------------------------------------------------
    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                self.config = load_config(self.config_path)
                self._problems = validate_profiles(self.config.profiles)
            except Exception as e:
                self.config_error = f"{self.config_path}: {e}"
        else:
            self.config = AppConfig()

    def _acquire_lock(self) -> None:
        """Nonblocking flock for the controlling TUI. Second instance -> read-only."""
        import fcntl
        path = default_lock_path()
        if path is None or self.readonly:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            self._lock_fd = None
            self._controlling = False
            self.notify_alert("Another Daish controls this machine — read-only", "warn")

    def on_mount(self) -> None:
        self._load_config()
        if self.config is None:
            self.config = AppConfig()
        self._acquire_lock()
        self._init_services()
        from .ui import _install_series
        _install_series(self)
        self._refresh_inventory()
        if not self.config.profiles:
            self.candidates = discover_profiles(self.local_root)
        self.call_later(self._start_workers)

    def _init_services(self) -> None:
        roots = [r for r in (self.config.model_roots if self.config else []) if r]
        self.sampler = HardwareSampler(model_roots=roots)
        self.sampler.init()
        self.controller = ModelController(self)
        self.poller = MetricPoller()
        self.bench = BenchmarkRunner()
        hpath = default_history_path() if self._controlling else default_history_path()
        if self.readonly:
            self.history = HistoryStore(hpath if hpath.exists() else None, readonly=True)
        else:
            self.history = HistoryStore(hpath)
        self.history.open()
        if self.history.available and not self.readonly:
            self.history.purge()
        # NInfer log readers for configured profiles
        for p in self.config.profiles:
            if p.backend == "ninfer" and p.ninfer_jsonl:
                self._ninfer[p.id] = NInferLogReader(p.ninfer_jsonl)
    def _start_workers(self) -> None:
        self._worker_handles.append(self.run_worker(self._hardware_loop(), name="hardware", group="collect",
                                                    exclusive=False, exit_on_error=False, thread=True))
        self._worker_handles.append(self.run_worker(self._reconcile_loop(), name="reconcile", group="collect",
                                                    exclusive=False, exit_on_error=False))
        self._worker_handles.append(self.run_worker(self._metrics_loop(), name="metrics", group="collect",
                                                    exclusive=False, exit_on_error=False))
        self._worker_handles.append(self.run_worker(self._history_loop(), name="history", group="collect",
                                                    exclusive=False, exit_on_error=False))
        # initial sample + reconcile
        self.call_later(self._do_reconcile)

    def on_unmount(self) -> None:
        self._worker_stop.set()
        for w in self._worker_handles:
            try:
                w.cancel()
            except Exception:
                pass

    def _refresh_inventory(self) -> None:
        if not self.config:
            return
        roots = [Path(r) for r in self.config.model_roots if r]
        if not roots:
            roots = [self.local_root / "models"]
        self.inventory = scan_inventory(roots, self.config.profiles)

    def get_profile(self, profile_id: str):
        if not self.config:
            return None
        for p in self.config.profiles:
            if p.id == profile_id:
                return p
        return None

    def emit_event(self, profile_id: str, kind: str, data: dict) -> None:
        self.events.appendleft({"ts": time.time(), "profile_id": profile_id, "kind": kind, "data": data})
        if self.history is not None:
            self.history.record_event(profile_id, kind, data)
        self.post_message(EventMessage(profile_id, kind, data))
        if kind in ("ready", "unhealthy", "failed", "start", "stop", "stopped"):
            self.notify_alert(f"{profile_id}: {kind}", "info" if kind == "ready" else "warn")

    def notify_alert(self, text: str, level: str = "info") -> None:
        self.alerts.appendleft(Alert(text, level))

    # --- workers ---------------------------------------------------------
    async def _hardware_loop(self) -> None:
        assert self.sampler is not None
        while not self._worker_stop.is_set():
            try:
                snap = self.sampler.sample()
                self.post_message(HwMessage(snap))
            except Exception as e:
                self.call_from_thread(self.notify_alert, f"hardware sampler: {e}", "warn")
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

    async def _reconcile_loop(self) -> None:
        assert self.controller is not None
        while not self._worker_stop.is_set():
            try:
                await self._do_reconcile()
            except Exception as e:
                self.notify_alert(f"reconcile: {e}", "warn")
            cadence = 1.0 if self._starting_any else 2.0
            await asyncio.sleep(cadence)

    async def _do_reconcile(self) -> None:
        assert self.controller is not None
        # health checks for active/starting profiles (feeds the controller)
        for p in self.config.profiles:
            st = self.statuses.get(p.id)
            if st is None or not p.base_url:
                continue
            if st.state in (STARTING, READY) or (st.active):
                ok, msg = await self.controller._check_ready(p)
                self.controller.set_health(p.id, ok, msg)
        statuses = await self.controller.reconcile()
        self._starting_any = any(s.state == STARTING for s in statuses)
        for s in statuses:
            self.statuses[s.profile_id] = s
        self.post_message(StatusMessage(statuses))

    async def _metrics_loop(self) -> None:
        assert self.poller is not None and self.controller is not None
        from .state import RuntimeIdentity
        while not self._worker_stop.is_set():
            for p in self.config.profiles:
                st = self.statuses.get(p.id)
                if st is None or st.state != READY:
                    continue
                identity = RuntimeIdentity(p.id, st.invocation_id)
                snap = await self.poller.poll(p, identity)
                ninfer = None
                reader = self._ninfer.get(p.id)
                if reader is not None:
                    try:
                        ninfer = reader.read()
                    except Exception:
                        ninfer = None
                self.snapshots[p.id] = snap
                if ninfer is not None:
                    self._merge_ninfer(p, snap, ninfer)
                self.post_message(RuntimeMessage(p.id, snap, ninfer))
            await asyncio.sleep(2.0)

    def _merge_ninfer(self, p, snap: RuntimeSnapshot, ninfer: dict) -> None:
        from .state import Metric
        snap.sources = sorted(set(snap.sources) | {"ninfer-jsonl"})
        snap.prompt_rate = Metric(ninfer.get("d_prompt"), "ninfer", time.time())
        snap.completion_rate = Metric(ninfer.get("d_completion"), "ninfer", time.time())
        snap.observed_prompt_tokens = Metric(ninfer.get("d_prompt"), "ninfer", time.time())
        snap.observed_completion_tokens = Metric(ninfer.get("d_completion"), "ninfer", time.time())
        snap.requests_total = Metric(float(ninfer.get("d_requests", 0)), "ninfer", time.time())
        reader = self._ninfer.get(p.id)
        if reader is not None:
            self.ninfer_totals[p.id] = reader.totals
            snap.spec_accept_rate = Metric(reader.spec_accept_rate, "ninfer",
                                           time.time(), None if reader.spec_accept_rate is not None else "no drafts")
            snap.cache_hit_rate = Metric(reader.cache_hit_rate, "ninfer", time.time(),
                                         None if reader.cache_hit_rate is not None else "no prompts")
        t = ninfer.get("throughput")
        if t:
            snap.engine_prefill_tps = Metric(t.get("prefill_tps"), "ninfer", time.time(),
                                             "no recent report" if ninfer.get("stale") else None)
            snap.engine_decode_tps = Metric(t.get("decode_tps"), "ninfer", time.time(),
                                            "no recent report" if ninfer.get("stale") else None)
            snap.scheduler_running = Metric(t.get("running"), "ninfer", time.time())
            snap.scheduler_waiting = Metric(t.get("waiting"), "ninfer", time.time())

    async def _history_loop(self) -> None:
        assert self.history is not None
        while not self._worker_stop.is_set():
            await asyncio.sleep(60.0)
            try:
                self._record_minute_sample()
                self.history.flush_samples()
                self.history.purge()
            except Exception:
                pass

    def _record_minute_sample(self) -> None:
        """Accumulate the current host hardware gauges into history."""
        if self.hw_snap is None or self.history is None:
            return
        gauges: dict[str, float] = {}
        ram = self.hw_snap.ram
        if ram.total_gib.value is not None and ram.available_gib.value is not None:
            gauges["ram_used"] = ram.total_gib.value - ram.available_gib.value
        if self.hw_snap.cpu.total_pct.value is not None:
            gauges["cpu_total"] = self.hw_snap.cpu.total_pct.value
        for g in self.hw_snap.gpus:
            if g.util_pct.value is not None:
                gauges[f"gpu_util"] = g.util_pct.value
            if g.vram_used_mib.value is not None:
                gauges["gpu_vram"] = g.vram_used_mib.value
        if gauges:
            self.history.add_sample("host", "hardware", gauges, {}, {}, 1.0)

    # --- key handlers ----------------------------------------------------
    def action_tab(self, index: int) -> None:
        tabs = self.query_one("TabbedContent")
        tabs.active = index

    def action_down_row(self) -> None:
        self._move(1)

    def action_up_row(self) -> None:
        self._move(-1)

    def _move(self, delta: int) -> None:
        if not self.config:
            return
        ids = [p.id for p in self.config.profiles]
        if not ids:
            return
        idx = ids.index(self.selected) if self.selected in ids else -1
        idx = (idx + delta) % len(ids)
        self.selected = ids[idx]
        self.call_later(self._refresh_panels)

    def action_details(self) -> None:
        if self.selected:
            self._register_screens()
            self.push_screen("details")

    async def action_start_model(self) -> None:
        await self._lifecycle("start")

    async def action_stop_model(self) -> None:
        await self._lifecycle("stop")

    async def action_restart_model(self) -> None:
        await self._lifecycle("restart")

    async def action_switch_model(self) -> None:
        await self._lifecycle("switch")

    async def _lifecycle(self, action: str) -> None:
        if self.readonly:
            self.notify_alert("read-only: monitoring available, no lifecycle actions", "warn")
            return
        if not self.selected:
            self.notify_alert("select a model first", "warn")
            return
        p = self.get_profile(self.selected)
        if p is None:
            return
        if not self._controlling:
            self.notify_alert("observing only (another Daish controls this machine)", "warn")
            return
        if p.control == "observe":
            self.notify_alert(f"{p.id} is observed; import its unit to control it", "warn")
            return
        if action == "start" and self._missing(p):
            self.notify_alert(f"{p.id}: missing required artifact", "warn")
            return
        self._register_screens()
        if action == "switch":
            src = self._switch_source(p)
            if src is None:
                self.notify_alert("no active source to switch from", "warn")
                return
            confirmed = await self.push_screen_wait("confirm", "switch", p, src)
            if confirmed:
                await self._do_lifecycle("switch", p, src)
            return
        confirmed = await self.push_screen_wait("confirm", action, p)
        if confirmed:
            await self._do_lifecycle(action, p)

    def _missing(self, p) -> bool:
        from .discovery import _missing_artifacts
        return bool(_missing_artifacts(p))

    def _switch_source(self, target):
        src = None
        for p in self.config.profiles:
            if p.id == target.id:
                continue
            st = self.statuses.get(p.id)
            if st and st.state == READY and st.active and p.port() == target.port():
                src = p
                break
        return src

    async def _do_lifecycle(self, action: str, p, src=None) -> None:
        assert self.controller is not None
        try:
            if action == "start":
                ok, msg = await self.controller.start(p.id)
            elif action == "stop":
                ok, msg = await self.controller.stop(p.id)
            elif action == "restart":
                ok, msg = await self.controller.restart(p.id)
            else:  # switch
                ok, msg = await self.controller.switch(src.id, p.id)
            self.notify_alert(f"{action}: {msg}", "info" if ok else "warn")
        except Exception as e:
            self.notify_alert(f"{action} error: {e}", "error")

    def action_logs_model(self) -> None:
        if self.selected:
            p = self.get_profile(self.selected)
            if p and p.unit:
                self.logs.follow(p.unit)
                self.notify_alert(f"following {p.unit}")

    def action_benchmark_model(self) -> None:
        if self.selected and self.bench:
            p = self.get_profile(self.selected)
            if p:
                self._register_screens()
                self.push_screen_wait("benchmark", profile=p)

    def action_tab(self, index: int) -> None:
        tabs = self.query_one("#tabs")
        tabs.active = f"tab-{index + 1}"

    def action_import_profiles(self) -> None:
        self._import_candidates()

    def _import_candidates(self) -> None:
        cands = self.candidates
        if not cands:
            cands = discover_profiles(self.local_root)
            self.candidates = cands
        if not self.config:
            self.config = AppConfig()
        existing = {p.id for p in self.config.profiles}
        added = 0
        for c in cands:
            if c.profile.id not in existing:
                self.config.profiles.append(c.profile)
                added += 1
        self._problems = validate_profiles(self.config.profiles)
        if self._controlling and not self.readonly:
            self.config.model_roots = [str(self.local_root / "models")]
            save_config(self.config_path, self.config)
        self.profiles_loaded = True
        self._init_ninfer()
        self._refresh_inventory()
        self.notify_alert(f"imported {added} profile(s)")
        self._refresh_panels()

    def _init_ninfer(self) -> None:
        for p in self.config.profiles:
            if p.backend == "ninfer" and p.ninfer_jsonl and p.id not in self._ninfer:
                self._ninfer[p.id] = NInferLogReader(p.ninfer_jsonl)

    def action_quit_app(self) -> None:
        self._cleanup()
        self.exit()

    def _cleanup(self) -> None:
        self._worker_stop.set()
        for w in self._worker_handles:
            try:
                w.cancel()
            except Exception:
                pass
        try:
            self.call_later(self._stop_logs)
        except Exception:
            pass
        if self.history is not None:
            try:
                self._record_minute_sample()
                self.history.flush_samples(force=True)
                self.history.close()
            except Exception:
                pass
        if self.sampler is not None:
            self.sampler.shutdown()

    async def _stop_logs(self) -> None:
        try:
            await self.logs.stop()
        except Exception:
            pass

    # --- UI composition --------------------------------------------------
    def compose(self) -> ComposeResult:
        from .ui import (ActivityPanel, HardwarePanel, HeaderWidget,
                         ModelsPanel, OverviewPanel, AlertsBar)
        yield HeaderWidget(self)
        from textual.widgets import TabbedContent
        with TabbedContent("Overview", "Models", "Hardware", "Activity", id="tabs"):
            yield OverviewPanel(self)
            yield ModelsPanel(self)
            yield HardwarePanel(self)
            yield ActivityPanel(self)
        yield AlertsBar(self)
        from textual.widgets import Footer
        yield Footer()

    # --- data -> UI ------------------------------------------------------
    def on_hw_message(self, msg: HwMessage) -> None:
        self.hw_snap = msg.snap
        self.call_later(self._refresh_panels)

    def on_status_message(self, msg: StatusMessage) -> None:
        self._refresh_panels()

    def on_runtime_message(self, msg: RuntimeMessage) -> None:
        self._refresh_panels()

    def _refresh_panels(self) -> None:
        for w in self.query(".daish-panel"):
            try:
                w.refresh_panels()
            except Exception:
                pass
