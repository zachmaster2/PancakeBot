"""CLI entrypoint: parses --sync/--backtest/--dry/--live flags and invokes run_from_config."""
from __future__ import annotations

import argparse
import atexit
import os
import sys
from pathlib import Path

from pancakebot import paths
from pancakebot.app import run_from_config
from pancakebot.log import info
from pancakebot.runtime.process_health import (
    clear_pid_file,
    read_last_heartbeat,
    write_crash,
    write_pid_file,
)


def _resolve_process_health_paths(dry: bool) -> tuple[Path, Path, Path]:
    """Return (pid_path, heartbeat_path, crash_path) for the given mode.

    Only meaningful for dry/live (backtest and sync don't need runtime health
    artifacts). Live-mode triple mirrors the dry-mode layout under var/live/.
    """
    if dry:
        return (
            Path(paths.DRY_PID_PATH),
            Path(paths.DRY_HEARTBEAT_PATH),
            Path(paths.DRY_CRASH_PATH),
        )
    return (
        Path(paths.LIVE_PID_PATH),
        Path(paths.LIVE_HEARTBEAT_PATH),
        Path(paths.LIVE_CRASH_PATH),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="PancakeBot -- automated trading for PancakeSwap Prediction V2",
    )
    p.add_argument("--config", type=str, default="config.toml",
                   help="Path to config file (default: config.toml)")
    p.add_argument("--sync", action="store_true",
                   help="Fetch rounds + klines + contract constants from chain/OKX/Graph")
    p.add_argument("--backtest", action="store_true",
                   help="Replay historical data and compute PnL")
    p.add_argument("--dry", action="store_true",
                   help="Real-time paper trading (no on-chain transactions)")
    p.add_argument("--live", action="store_true",
                   help="Real-time trading with real BNB")
    p.add_argument("--fresh", action="store_true",
                   help="Archive existing dry state and start fresh (--dry only)")
    p.add_argument("--no-archive", action="store_true",
                   help="Delete (don't archive) existing state on --fresh (--dry only)")
    args = p.parse_args(argv)

    selected = [args.sync, args.backtest, args.dry, args.live]
    if not any(selected):
        p.print_help()
        sys.exit(1)
    if sum(bool(s) for s in selected) > 1:
        p.error("--sync, --backtest, --dry, and --live are mutually exclusive")

    if args.fresh and not args.dry:
        p.error("--fresh is only valid with --dry")
    if args.no_archive and not args.fresh:
        p.error("--no-archive requires --fresh")

    return args


def main() -> None:
    args = _parse_args()
    if args.dry or args.live:
        from pancakebot.runtime.single_instance import find_duplicate_bots
        from pancakebot.log import error
        dupes = find_duplicate_bots()
        if dupes:
            for p in dupes:
                error("CORE", "START", "DUP_PROC",
                      msg="another dry/live bot already running",
                      pid=p["pid"], cmdline=p["cmdline"], started_at=p["started_at"])
            sys.exit(2)

    # Process-health instrumentation: write PID file + register atexit cleanup
    # + catch any top-level exception into crash.json before re-raising. Only
    # for dry/live -- backtest and sync complete quickly without supervision.
    pid_path: Path | None = None
    heartbeat_path: Path | None = None
    crash_path: Path | None = None
    if args.dry or args.live:
        pid_path, heartbeat_path, crash_path = _resolve_process_health_paths(args.dry)
        write_pid_file(pid_path, os.getpid())
        atexit.register(clear_pid_file, pid_path)

    try:
        run_from_config(
            config_path=args.config,
            dry=args.dry,
            backtest=args.backtest,
            sync=args.sync,
            live=args.live,
            fresh=args.fresh,
            no_archive=args.no_archive,
        )
    except KeyboardInterrupt:
        info("CORE", "RUN", "EXIT", msg="Caught KeyboardInterrupt: shutting down")
    except Exception as e:
        # Narrow-catch: Exception (NOT BaseException) so KeyboardInterrupt /
        # SystemExit still propagate cleanly without being treated as crashes.
        # Dry/live only -- backtest/sync run to completion or fail fast.
        if crash_path is not None:
            last_epoch: int | None = None
            if heartbeat_path is not None:
                hb = read_last_heartbeat(heartbeat_path)
                if hb is not None:
                    raw = hb.get("last_epoch")
                    if isinstance(raw, int):
                        last_epoch = raw
            write_crash(crash_path, e, last_epoch=last_epoch)
        raise


if __name__ == "__main__":
    main()
