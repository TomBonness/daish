# Daish

Local AI command center: a terminal UI that centralizes starting and stopping this
machine's local AI models, shows GPU/CPU/RAM activity, and adds practical
local-LLM inference monitoring.

## Run

```
uv sync --python 3.12            # project venv (dev)
uv tool install -e .             # install a global `daish` on PATH (editable)
daish --read-only                # monitoring only: no config writes, lifecycle, or benchmarks
daish                            # full command center
daish --ascii                    # pure-ASCII borders/charts/glyphs
NO_COLOR=1 daish                 # strip color
```

`uv tool install -e .` puts `daish` in `~/.local/bin` and points it at the project
source, so code edits are picked up without reinstalling. `uv tool upgrade -e .`
syncs dependencies; `uv tool uninstall daish` removes it.

A second instance detects the controlling instance (flock) and runs read-only.

## Tabs & keys

- `1` Overview · `2` Models · `3` Hardware · `4` Activity
- `j`/`k` select a model, `⏎` details, `l` follow its journal logs
- `s` start, `x` stop, `r` restart, `b` benchmark (selected model)
- `i` import discovered profiles, `p` keybinding palette, `q` quit

Lifecycle actions (start/stop/restart) open a confirm screen defaulting to Cancel.

## Config

Default: `$XDG_CONFIG_HOME/daish/config.json` (or `--config PATH`), mode `0600`.
Five curated recipes ship in `src/daish/data/local-profiles.json` and are offered
for import on first launch. Each recipe carries exact argv/env transcribed from
the production launchers.

## Monitoring model

- Hardware: NVML (GPU util/VRAM/temp/power/clocks, per-process) + psutil (CPU
  per-core, RAM available-semantics, disks) at 1 Hz, sampled in a worker thread.
- Per-model resources come from the unit's cgroup (mem/swap/oom/pids/CPU).
- Inference metrics: engine Prometheus endpoints (vLLM / SGLang / llama.cpp)
  parsed with counter-reset and gap baselining; histogram mean from `_sum`/`_count`
  deltas; NInfer's structured `request_log.jsonl` parsed with
  `(server_instance_id, request_id)` dedup. Observed only while Daish runs.
- Benchmarks: one streamed chat request; TTFT, exact usage token counts, and
  `completion_tokens / total_elapsed` labelled end-to-end tok/s, plus
  engine-reported prefill/decode when present.

## History

`$XDG_STATE_HOME/daish/history.db3` (SQLite): events, benchmark results, and
minute-bucketed hardware/inference samples (gauge mean/max/n, counter deltas,
latency sum/count). 7-day retention. Gaps in the timeline mark time Daish was
closed. `Activity` tab shows recent events/benchmarks.

## Tests

```
uv run pytest tests/                          # unit + metric/control semantics
DAISH_TEST_SYSTEMD=1 uv run pytest tests/     # + real user-systemd lifecycle
.venv/bin/python tests/smoke_lifecycle.py     # standalone lifecycle smoke (HTTP fixture)
```

## Layout

`src/daish/` — `config.py` (profiles/config), `state.py` (snapshots),
`telemetry.py` (hardware sampler), `discovery.py` (profile + artifact discovery),
`control.py` (systemd lifecycle + reconciliation), `logs.py` (journal follower),
`metrics.py` (Prometheus parser), `ninfer_log.py` (NInfer JSONL), `benchmark.py`,
`history.py` (SQLite store), `ui.py` (panels), `app.py` (TUI + workers),
`cli.py` (entry point).
