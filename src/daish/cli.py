"""Command-line entry point.

``daish [--config PATH] [--read-only] [--ascii]``. No arguments opens the TUI.
``--read-only`` disables imports/config writes, lifecycle actions and inference
benchmarks; monitoring remains available.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _no_color() -> bool:
    return os.environ.get("NO_COLOR", "") != ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="daish",
        description="Local AI command center: start/stop local models, GPU/CPU/RAM dashboard, inference monitoring.")
    parser.add_argument("--config", type=Path, default=None,
                        help="path to config.json (default: $XDG_CONFIG_HOME/daish/config.json)")
    parser.add_argument("--read-only", action="store_true",
                        help="monitoring only: no config writes, lifecycle actions, or benchmarks")
    parser.add_argument("--ascii", action="store_true",
                        help="pure-ASCII borders/charts/status glyphs")
    args = parser.parse_args(argv)

    from .app import DaishApp
    app = DaishApp(config_path=args.config, readonly=args.read_only,
                   ascii_only=args.ascii)
    app.run()


if __name__ == "__main__":
    main()
