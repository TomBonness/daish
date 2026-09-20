"""Daish UI: panels, widgets, and modal screens.

Panels are ``Static`` renderables rebuilt on each data message — only changed
panels repaint. Status always has a word/icon in addition to color.
"""
from __future__ import annotations

import time
from pathlib import Path

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import DataTable, Input, Static, TabbedContent, Tab

from . import art

# --- palette ---------------------------------------------------------------
# Aetheria — deep-purple "celestial" base16 (base16-tarot derived, JJDizz1L/aetheria).
C_BG = "#0e091d"      # deep purple background
C_PANEL = "#16102a"   # raised panel
C_FG = "#e7e1f2"      # light lavender primary text
C_MUTED = "#8b7fa8"   # muted lavender
C_CYAN = "#14b9b5"    # teal — primary accent (aetheria foreground)
C_AMBER = "#ff7f41"   # orange — warm accent (aetheria cursor)
C_GREEN = "#7cd699"   # mint — success / running
C_RED = "#e20342"     # crimson — failed / error
C_PINK = "#fd3e6a"    # raspberry — strong accent / alerts
C_PURPLE = "#9147a8"  # draft / decoration
C_BLUE = "#04c5f0"    # selection / links
C_LIME = "#c8e967"    # bright highlight

STATE_STYLE = {
    "READY": C_GREEN, "STARTING": C_CYAN, "STOPPED": C_MUTED,
    "STOPPING": C_AMBER, "FAILED": C_RED, "UNHEALTHY": C_AMBER,
    "OBSERVED": C_MUTED,
}
STATE_ICON = {
    "READY": "RUN", "STARTING": "LOAD", "STOPPED": "OFF",
    "STOPPING": "STOP", "FAILED": "FAIL", "UNHEALTHY": "WARN",
    "OBSERVED": "OBS",
}


# Genuine pure-ASCII box (+, -, |) for --ascii; Rich's box.SQUARE is ┌┐└┘│─.
# Template mirrors rich.box.SQUARE layout with ASCII glyphs.
_ASCII_BOX = box.Box('+-++\n| ||\n+-++\n| ||\n+-++\n+-++\n| ||\n+-++\n')


def _pbox(app):
    """Panel border box: pure ASCII vs rounded."""
    return _ASCII_BOX if getattr(app, "ascii_only", False) else box.ROUNDED


def _tbox(app):
    """Table border box: pure ASCII vs simple."""
    return _ASCII_BOX if getattr(app, "ascii_only", False) else box.SIMPLE


def _bar(frac, width: int, ascii_only: bool = False) -> str:
    if frac is None:
        return "." * width
    frac = max(0.0, min(1.0, float(frac)))
    filled = int(round(width * frac))
    return "#" * filled + "." * (width - filled)


def _spark(values, width: int = 30, ascii_only: bool = False) -> str:
    if not values:
        return "." * width
    # take the last `width` samples, drop Nones
    vals = [v for v in values[-width:] if v is not None]
    if not vals:
        return "." * width
    hi = max(vals)
    lo = min(vals)
    span = (hi - lo) or 1.0
    blocks = " ▁▂▃▄▅▆▇█" if not ascii_only else " .:-=+*#@"
    out = []
    for v in vals:
        idx = int((v - lo) / span * (len(blocks) - 1))
        out.append(blocks[idx])
    return "".join(out).rjust(width)


def _fmt_gib(v) -> str:
    return "N/A" if v is None else f"{v:.1f} GiB"


def _fmt(v, unit: str = "", nd: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{nd}f}{unit}"


def _val(m) -> float | None:
    return m.value if m is not None else None


class HeaderWidget(Static):
    def __init__(self, app) -> None:
        super().__init__(classes="daish-panel header")
        self._app = app

    def refresh_panels(self) -> None:
        w = self.size.width if self.size else 100
        h = self.app.screen.size.height if self.app else 24
        if not self._app.ascii_only and w >= 140 and h >= 42:
            self.update(Text(art.WORDMARK, style=f"bold {C_CYAN}"))
        else:
            sub = "OBSERVED" if not self._app._controlling else "CONTROL"
            self.update(Text.assemble(
                ("DAISH", f"bold {C_CYAN}"), (" / LOCAL AI COMMAND CENTER   ", C_FG),
                (sub, f"bold {C_AMBER}")))

    


def _gpu_table(app, hw) -> Table:
    t = Table(box=_tbox(app), expand=True, pad_edge=False)
    t.add_column("GPU", style=C_FG)
    t.add_column("LOAD", no_wrap=True)
    t.add_column("VRAM", no_wrap=True)
    t.add_column("TEMP", justify="right", no_wrap=True)
    t.add_column("POWER", justify="right", no_wrap=True)
    if hw is None or not hw.gpus:
        reason = hw.nvml_reason if hw else "nvml"
        t.add_row("GPU", Text(f"N/A · {reason}", style=C_MUTED), "", "N/A", "N/A")
        return t
    ascii_only = app.ascii_only
    for g in hw.gpus:
        load_bar = _bar(_val(g.util_pct), 16, ascii_only)
        vram = _val(g.vram_used_mib)
        vtot = _val(g.vram_total_mib)
        vram_txt = f"[{load_bar}] {_fmt(vram/1024.0 if vram is not None else None, ' GiB', 2)}"
        if vtot:
            vram_txt += f"/{_fmt(vtot/1024.0, ' GiB', 2)}"
        name = f"GPU{g.index} / {g.name}"
        t.add_row(
            name,
            Text(load_bar, style=C_CYAN) if _val(g.util_pct) is not None else Text("N/A", style=C_MUTED),
            vram_txt,
            _fmt(_val(g.temp_c), " C", 0),
            _fmt(_val(g.power_w), " W", 0),
        )
    return t


def _host_lines(app, hw) -> list[Text]:
    lines: list[Text] = []
    if hw is None:
        return [Text("N/A · no sample", style=C_MUTED)]
    c, r = hw.cpu, hw.ram
    cpu_bar = _bar(_val(c.total_pct) / 100.0 if _val(c.total_pct) is not None else None, 16, app.ascii_only)
    lines.append(Text.assemble(
        ("CPU ", C_FG), (cpu_bar, C_CYAN), ("  ", ""),
        (_fmt(_val(c.total_pct), "%", 0), C_FG),
        (f"  {c.threads} thr  load {_fmt(_val(c.load1), nd=2)}", C_MUTED)))
    ram_total = _val(r.total_gib)
    ram_avail = _val(r.available_gib)
    ram_used = (ram_total - ram_avail) if (ram_total is not None and ram_avail is not None) else None
    lines.append(Text.assemble(
        ("RAM ", C_FG),
        (f"{_fmt(ram_used, ' GiB')}/{_fmt(ram_total, ' GiB')}", C_FG),
        (f"  avail {_fmt(ram_avail, ' GiB')}", C_MUTED),
        (f"  SWAP {_fmt((_val(r.swap_total_gib) or 0) - (_val(r.swap_free_gib) or 0), ' GiB')}", C_MUTED)))
    temps = "  ".join(f"{k}:{_fmt(v.value, 'C', 0)}" for k, v in list(c.temperatures.items())[:3])
    if temps:
        lines.append(Text(temps, style=C_MUTED))
    if hw.disks:
        d = hw.disks[0]
        lines.append(Text.assemble(
            ("DISK ", C_FG),
            (f"{_fmt(_val(d.free_gib), ' GiB')} free", C_FG),
            (f"  ({d.mount})", C_MUTED)))
    return lines


def _visible_profiles(app):
    """Config profiles if any imported, else discovered candidates as OBSERVED."""
    if app.config and app.config.profiles:
        return [(p, False) for p in app.config.profiles]
    return [(c.profile, True) for c in app.candidates]


def _cand_flags(app, pid) -> str:
    for c in app.candidates:
        if c.profile.id == pid:
            return c.status
    return ""


def _model_row(app, p, observed=False) -> Text:
    pid = p.id
    st = None if observed else app.statuses.get(pid)
    state = "OBSERVED" if observed else (st.state if st else "STOPPED")
    label = p.label
    port = f":{p.port()}" if p.port() else ""
    color = STATE_STYLE.get(state, C_MUTED)
    icon = STATE_ICON.get(state, "?")
    flags = _cand_flags(app, pid) if observed else (" ".join(st.flags) if st and st.flags else "")
    sel = ">" if app.selected == pid else " "
    parts = [(sel, C_BLUE), (f" {icon:5}", color), (f" {label:<24}", C_FG), (f"{port}", C_MUTED)]
    if flags:
        parts.append((f"  {flags}", C_AMBER))
    return Text.assemble(*parts)


def _selected_panel(app) -> Group:
    pid = app.selected
    if not pid:
        return Group(Text("(no model selected)", style=C_MUTED))
    p = app.get_profile(pid)
    st = app.statuses.get(pid)
    snap = app.snapshots.get(pid)
    rows = []
    rows.append(Text.assemble(
        (p.label if p else pid, f"bold {C_CYAN}"),
        (f"  [{(st.state if st else 'STOPPED')}", f"bold {STATE_STYLE.get(st.state if st else 'STOPPED', C_MUTED)}"), ("]", C_FG)))
    if p:
        ep = p.base_url or "N/A"
        rows.append(Text.assemble(("endpoint ", C_MUTED), (ep, C_FG),
                                  (f"  model {p.model_id or 'N/A'}", C_MUTED)))
        meta = []
        if p.backend == "ninfer":
            meta.append("KV k8v4")
        rows.append(Text.assemble(
            (f"backend {p.backend}  control {p.control}  unit {p.unit or 'N/A'}", C_MUTED)))
    if st:
        if st.main_pid:
            rows.append(Text.assemble(("pid ", C_MUTED), (str(st.main_pid), C_FG),
                                      (f"  invocation {st.invocation_id[:8] or 'n/a'}", C_MUTED)))
        if st.port_conflict:
            rows.append(Text(f"PORT CONFLICT — foreign pid {st.foreign_pid}", style=C_AMBER))
        if st.message:
            rows.append(Text(st.message, style=C_MUTED))
    if snap:
        cards = []
        def card(title, val):
            cards.append(Text.assemble((title + " ", C_MUTED), (val, C_FG)))
        card("run/queue", f"{_fmt(_val(snap.running_requests), nd=0)}/{_fmt(_val(snap.queue_requests), nd=0)}")
        card("in tok/s", _fmt(_val(snap.prompt_rate), nd=1))
        card("out tok/s", _fmt(_val(snap.completion_rate), nd=1))
        card("TTFT", _fmt(_val(snap.ttft_mean_s), "s", 3) if _val(snap.ttft_mean_s) is not None else "N/A")
        card("KV", f"{_val(snap.kv_usage_frac):.0%}" if _val(snap.kv_usage_frac) is not None else "N/A")
        card("decode", _fmt(_val(snap.engine_decode_tps), " tok/s", 0) if _val(snap.engine_decode_tps) is not None else "N/A")
        # lay out in pairs
        for i in range(0, len(cards), 2):
            left = cards[i]
            right = cards[i + 1] if i + 1 < len(cards) else Text("")
            rows.append(Text.assemble(left.plain, C_MUTED) if False else _pair(left, right))
        if snap.spec_accept_rate.value is not None:
            rows.append(Text.assemble(("spec accept ", C_MUTED), (f"{snap.spec_accept_rate.value:.0%}", C_FG)))
    nin = app.ninfer_totals.get(pid)
    if nin:
        rows.append(Text.assemble(
            ("ninfer observed ", C_MUTED),
            (f"req {int(nin['requests'])}  prompt {int(nin['prompt_tokens'])}  comp {int(nin['completion_tokens'])}", C_FG)))
    return Group(*rows)


def _pair(left: Text, right: Text) -> Text:
    l = left.copy()
    l.pad_right(10)
    return l + right


class OverviewPanel(Static):
    def __init__(self, app) -> None:
        super().__init__(classes="daish-panel")
        self._app = app

    def refresh_panels(self) -> None:
        app = self._app
        hw = app.hw_snap
        header = Table(box=_tbox(app), expand=True, pad_edge=False)
        header.add_column("GPU / ENGINE ROOM", ratio=3)
        header.add_column("HOST", ratio=3)
        if hw and hw.gpus and not app.ascii_only:
            board = art.gpu_board(40,
                                  (hw.gpus[0].vram_used_mib.value / hw.gpus[0].vram_total_mib.value
                                   if hw.gpus[0].vram_total_mib.value else None),
                                  (hw.gpus[0].util_pct.value / 100.0 if hw.gpus[0].util_pct.value is not None else None),
                                  _fmt(_val(hw.gpus[0].temp_c), "C", 0),
                                  _fmt(_val(hw.gpus[0].power_w), "W", 0), app.ascii_only)
            gpu_cell = Group(Text("\n".join(board), style=C_CYAN), _gpu_table(app, hw))
        else:
            gpu_cell = _gpu_table(app, hw)
        header.add_row(gpu_cell, Group(*_host_lines(app, hw)))
        # model list
        models = Table(box=_tbox(app), expand=True, pad_edge=False)
        models.add_column("", ratio=1)
        for p, observed in _visible_profiles(app):
            models.add_row(_model_row(app, p, observed))
        if not _visible_profiles(app):
            models.add_row(Text("  (no models — press i to import local profiles)", style=C_MUTED))
        sel = _selected_panel(app)
        self.update(Group(
            Panel(header, title="OVERVIEW", box=_pbox(app), border_style=C_CYAN, padding=(0, 1)),
            Panel(models, title="MODELS", box=_pbox(app), border_style=C_PURPLE, padding=(0, 1)),
            Panel(sel, title="SELECTED", box=_pbox(app), border_style=C_BLUE, padding=(0, 1)),
        ))

    


class ModelsPanel(Static):
    def __init__(self, app) -> None:
        super().__init__(classes="daish-panel")
        self._app = app

    def refresh_panels(self) -> None:
        app = self._app
        prof = Table(box=_tbox(app), expand=True, pad_edge=False)
        for col in ("ID", "LABEL", "BACKEND", "CONTROL", "STATE", "UNIT", "PORT", "FLAGS"):
            prof.add_column(col)
        for p, observed in _visible_profiles(app):
            st = None if observed else app.statuses.get(p.id)
            state = "OBSERVED" if observed else (st.state if st else "STOPPED")
            flags = _cand_flags(app, p.id) if observed else (" ".join(st.flags) if st and st.flags else "")
            problem = "" if observed else (app._problems.get(p.id) if p.id in app._problems else "")
            prof.add_row(
                p.id, p.label, p.backend, p.control,
                Text(state, style=STATE_STYLE.get(state, C_MUTED)),
                p.unit or "", f":{p.port()}" if p.port() else "",
                Text(flags or problem, style=C_AMBER if (flags or problem) else C_MUTED))
        inv = Table(box=_tbox(app), expand=True, pad_edge=False)
        for col in ("KIND", "LABEL", "BYTES", "DRAFT", "PROFILES"):
            inv.add_column(col)
        for a in app.inventory:
            gb = a.bytes / (1024 ** 3)
            inv.add_row(a.kind, a.label, f"{gb:.1f} GiB", "yes" if a.draft else "",
                        ",".join(a.related_profiles) or "NO LAUNCH PROFILE")
        self.update(Group(
            Panel(prof, title="PROFILES", box=_pbox(app), border_style=C_CYAN, padding=(0, 1)),
            Panel(inv, title="ARTIFACT INVENTORY", box=_pbox(app), border_style=C_PURPLE, padding=(0, 1)),
        ))

    


class HardwarePanel(Static):
    def __init__(self, app) -> None:
        super().__init__(classes="daish-panel")
        self._app = app

    def refresh_panels(self) -> None:
        app = self._app
        hw = app.hw_snap
        if hw is None:
            self.update(Text("N/A · no hardware sample", style=C_MUTED))
            return
        g = hw.gpus[0] if hw.gpus else None
        top = Table(box=_tbox(app), expand=True, pad_edge=False)
        top.add_column("", ratio=1)
        if g:
            top.add_row("GPU LOAD", _spark([s if s is not None else 0 for s in
                                            app._gpu_util_hist(g.index)] if False else app._series(f'gpu{g.index}.util'), 40, app.ascii_only))
            top.add_row("GPU VRAM", _spark(app._series(f'gpu{g.index}.vram'), 40, app.ascii_only))
            top.add_row("GPU TEMP", _spark(app._series(f'gpu{g.index}.temp'), 40, app.ascii_only))
            top.add_row("CPU TOTAL", _spark(app._series("cpu.total"), 40, app.ascii_only))
        top2 = Table(box=_tbox(app), expand=True, pad_edge=False)
        top2.add_column("", ratio=1)
        if g:
            top2.add_row("clocks", f"gfx {_fmt(_val(g.graphics_clock_mhz), ' MHz', 0)}  mem {_fmt(_val(g.memory_clock_mhz), ' MHz', 0)}  fan {_fmt(_val(g.fan_pct), '%', 0)}")
            top2.add_row("throttle", ", ".join(g.throttle_reasons) or "none")
        procs = Table(box=_tbox(app), expand=True, pad_edge=False)
        procs.add_column("PID"); procs.add_column("VRAM"); procs.add_column("UTIL"); procs.add_column("COMMAND")
        if g:
            for p in g.processes:
                mark = "M" if p.pid in g.model_pids else " "
                procs.add_row(f"{mark} {p.pid}", f"{p.vram_mib:.0f} MiB", _fmt(p.util_pct, "%", 0),
                              p.command[:40])
        self.update(Group(
            Panel(top, title="GPU HISTORY (last 900s)", box=_pbox(app), border_style=C_CYAN, padding=(0, 1)),
            Panel(top2, title="GPU DETAIL", box=_pbox(app), border_style=C_AMBER, padding=(0, 1)),
            Panel(procs, title="GPU PROCESSES (M = model cgroup)", box=_pbox(app), border_style=C_PURPLE, padding=(0, 1)),
        ))

    


class ActivityPanel(Static):
    def __init__(self, app) -> None:
        super().__init__(classes="daish-panel")
        self._app = app

    def refresh_panels(self) -> None:
        app = self._app
        ev = Table(box=_tbox(app), expand=True, pad_edge=False)
        ev.add_column("TIME"); ev.add_column("PROFILE"); ev.add_column("KIND"); ev.add_column("DATA")
        for e in list(app.events)[:30]:
            ev.add_row(time.strftime("%H:%M:%S", time.localtime(e["ts"])), e["profile_id"],
                       e["kind"], str(e["data"])[:60])
        log_lines = []
        for ln in list(app.logs.lines)[-30:]:
            log_lines.append(Text.assemble(
                (time.strftime("%H:%M:%S", time.localtime(ln.ts)), C_MUTED),
                (f" [{ln.severity}] ", C_AMBER if ln.priority <= 3 else C_MUTED),
                (ln.message[:120], C_FG)))
        if not log_lines:
            log_lines = [Text(f"logs: {app.logs.state} (select a model and press l)", style=C_MUTED)]
        hist = Table(box=_tbox(app), expand=True, pad_edge=False)
        hist.add_column("TIME"); hist.add_column("PROFILE"); hist.add_column("RESULT")
        if app.history is not None:
            for b in app.history.benchmarks()[:10]:
                d = b["data"]
                hist.add_row(time.strftime("%H:%M:%S", time.localtime(b["ts_utc"])), b["profile_id"],
                             f"ttft {_fmt(d.get('ttft_s'), 's', 3)}  e2e {_fmt(d.get('e2e_tps'), ' tok/s', 1)}")
        self.update(Group(
            Panel(ev, title="EVENTS", box=_pbox(app), border_style=C_CYAN, padding=(0, 1)),
            Panel(Group(*log_lines), title="LOGS (selected unit)", box=_pbox(app), border_style=C_PURPLE, padding=(0, 1)),
            Panel(hist, title="BENCHMARKS", box=_pbox(app), border_style=C_LIME, padding=(0, 1)),
            Text("OBSERVED HISTORY — monitoring pauses when Daish is closed", style=C_MUTED),
        ))

    


class AlertsBar(Static):
    def __init__(self, app) -> None:
        super().__init__(classes="daish-panel alerts")
        self._app = app

    def refresh_panels(self) -> None:
        app = self._app
        if not app.alerts:
            self.update(Text("no alerts", style=C_MUTED))
            return
        parts = []
        for a in list(app.alerts)[:4]:
            color = {"warn": C_AMBER, "error": C_RED, "info": C_CYAN}.get(a.level, C_MUTED)
            parts.append(Text(f"[{a.text}] ", style=color))
        self.update(Text.assemble(*parts))

    


# --- series accessor (delegated to sampler history) -----------------------
def _series_get(app, key: str):
    if app.sampler is None:
        return []
    dq = app.sampler.history.get(key)
    return list(dq) if dq else []


# attach a helper to the app for hardware panels
def _install_series(app) -> None:
    app._series = lambda k: list(app.sampler.history.get(k) or []) if app.sampler else []
    app._gpu_util_hist = lambda i: list(app.sampler.history.get(f"gpu{i}.util") or []) if app.sampler else []


# --- modal screens ---------------------------------------------------------
class BenchmarkScreen(Screen):
    CSS = """
    BenchmarkScreen { alignment: center_middle; }
    #bm-box { width: 60; height: 12; border: heavy $cyan; }
    """

    def __init__(self, profile) -> None:
        super().__init__()
        self._profile = profile
        self._result = None
        self._text = Static("Benchmarking…", id="bm-box")

    def compose(self) -> ComposeResult:
        yield self._text

    def on_mount(self) -> None:
        app = self.app
        self.call_later(self._run)

    async def _run(self) -> None:
        app = self.app
        try:
            res = await app.bench.benchmark(self._profile)
        except Exception as e:
            res = None
            self._text.update(Text(f"benchmark error: {e}", style=C_RED))
            return
        self._result = res
        lines = []
        lines.append(Text.assemble(("Benchmark ", f"bold {C_CYAN}"), (self._profile.id, C_FG)))
        lines.append(Text(f"  TTFT: {_fmt(_val(res.ttft_s), 's', 3)}"))
        lines.append(Text(f"  total: {_fmt(_val(res.total_s), 's', 3)}"))
        lines.append(Text(f"  end-to-end tok/s: {_fmt(_val(res.e2e_tps), nd=1)}"))
        lines.append(Text(f"  prompt/completion tokens: {_val(res.prompt_tokens)}/{_val(res.completion_tokens)}"))
        lines.append(Text(f"  engine prefill/decode: {_fmt(_val(res.engine_prefill_tps), ' tps', 0)}/{_fmt(_val(res.engine_decode_tps), ' tps', 0)}"))
        if res.detail:
            lines.append(Text(f"  {res.detail}", style=C_MUTED))
        if app.history is not None:
            app.history.record_benchmark(self._profile.id, {
                "ttft_s": res.ttft_s.value, "total_s": res.total_s.value,
                "e2e_tps": res.e2e_tps.value,
                "prompt_tokens": res.prompt_tokens.value,
                "completion_tokens": res.completion_tokens.value})
        self._text.update(Group(*lines))

    def key_q(self) -> None:
        self.dismiss()

    def action_default(self) -> None:
        self.dismiss()


class DetailsScreen(Screen):
    CSS = """
    DetailsScreen { alignment: center_middle; }
    #det-box { width: 70; height: 20; border: heavy $cyan; }
    """

    def compose(self) -> ComposeResult:
        app = self.app
        pid = app.selected
        p = app.get_profile(pid) if pid else None
        st = app.statuses.get(pid) if pid else None
        lines = []
        if p:
            lines.append(Text.assemble((p.label, f"bold {C_CYAN}")))
            lines.append(Text(f"executable: {p.argv[0] if p.argv else 'N/A'}", style=C_MUTED))
            lines.append(Text("arguments:", style=C_MUTED))
            for a in p.argv:
                lines.append(Text(f"  {a}", style=C_FG))
            if p.env:
                lines.append(Text("environment:", style=C_MUTED))
                for k, v in p.env.items():
                    lines.append(Text(f"  {k}={v}", style=C_MUTED))
            lines.append(Text(f"model paths: {', '.join(p.model_paths) or 'N/A'}", style=C_MUTED))
            lines.append(Text(f"draft paths: {', '.join(p.draft_paths) or 'N/A'}", style=C_MUTED))
        if st:
            lines.append(Text.assemble(
                ("state ", C_MUTED), (st.state, f"bold {STATE_STYLE.get(st.state, C_MUTED)}"),
                (f"  pid {st.main_pid or 'n/a'}  unit {st.unit or 'n/a'}", C_MUTED)))
            if st.control_group:
                lines.append(Text(f"cgroup: {st.control_group}", style=C_MUTED))
            if st.fingerprint:
                lines.append(Text(f"fingerprint: {st.fingerprint[:100]}", style=C_MUTED))
        self._lines = lines
        yield Static(Group(*lines), id="det-box")

    def key_q(self) -> None:
        self.dismiss()


class ConfirmScreen(Screen):
    CSS = """
    ConfirmScreen { alignment: center_middle; }
    #confirm-box { width: 64; height: 10; border: heavy $amber; }
    """

    def __init__(self, action: str, profile, source=None) -> None:
        super().__init__()
        self._action = action
        self._profile = profile
        self._source = source

    def compose(self) -> ComposeResult:
        app = self.app
        st = app.statuses.get(self._profile.id)
        inflight = _fmt(_val(st) if False else None) if False else ""
        if self._action == "switch":
            title = f"SWITCH {self._source.id} -> {self._profile.id}"
            body = (f"Stop {self._source.id} then start {self._profile.id} on the same port.\n"
                    f"Default: Cancel")
        else:
            state = st.state if st else "STOPPED"
            body = (f"{self._action.upper()} {self._profile.label}\n"
                    f"endpoint {self._profile.base_url or 'N/A'}  unit {self._profile.unit or 'N/A'}\n"
                    f"current state {state}  activity unknown\n"
                    f"Default: Cancel")
        yield Static(
            Group(Text.assemble((title, f"bold {C_AMBER}"), Text("\n"), Text(body, style=C_FG),
                                 Text("\n[y]es  [n]o(default)  [esc]cancel", style=C_MUTED)),
                  id="confirm-box"),
        )

    def key_y(self) -> None:
        app = self.app
        self.dismiss(True)

    def key_n(self) -> None:
        self.dismiss(False)

    def key_escape(self) -> None:
        self.dismiss(False)
