"""PancakeBot supervisor -- classify bot health and log one line per invocation.

Phase 2b: observability only. No alerting (Phase 2c), no auto-restart (Phase 2d).

One-shot invocation. Reads the Phase 2a health artifacts for one mode
(``var/<mode>/heartbeat.json``, ``bot.pid``, ``crash.json``), classifies state,
appends a single line to ``var/<mode>/supervisor.log``, and exits. Idempotent;
safe to invoke on a schedule (e.g. ``schtasks`` every 3 min).

Classification order (first match wins):
  UP            - fresh heartbeat (mtime < ``--stale-threshold``) + PID alive
  STARTING      - PID file alive, no/stale heartbeat, PID file age < ``--startup-grace``
  STALE         - PID alive but heartbeat stale past threshold (past grace)
  CRASHED       - ``crash.json`` present (regardless of process state)
  UNINSTRUMENTED- bot process alive (psutil match on ``run.py --<mode>``)
                  but no heartbeat AND no PID file (legacy pre-Phase-2a bot)
  DOWN          - none of the above

Exit codes: UP=0, STARTING=1, STALE=2, CRASHED=3, UNINSTRUMENTED=4, DOWN=5,
supervisor error=99.

Usage:
    python scripts/supervisor.py --mode dry
    python scripts/supervisor.py --mode live
    python scripts/supervisor.py --mode dry --stale-threshold 5 --startup-grace 90
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure the repo root is importable when this script runs from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import paths  # noqa: E402


# -- Exit codes (per spec) ---------------------------------------------------

EXIT_UP = 0
EXIT_STARTING = 1
EXIT_STALE = 2
EXIT_CRASHED = 3
EXIT_UNINSTRUMENTED = 4
EXIT_DOWN = 5
EXIT_ERROR = 99

_STATUS_TO_EXIT = {
    "UP": EXIT_UP,
    "STARTING": EXIT_STARTING,
    "STALE": EXIT_STALE,
    "CRASHED": EXIT_CRASHED,
    "UNINSTRUMENTED": EXIT_UNINSTRUMENTED,
    "DOWN": EXIT_DOWN,
}


# -- Artifact resolution -----------------------------------------------------

def _artifacts_for_mode(mode: str) -> dict[str, Path]:
    """Return {heartbeat, pid, crash, supervisor_log, trades} paths for *mode*."""
    if mode == "dry":
        return {
            "heartbeat": Path(paths.DRY_HEARTBEAT_PATH),
            "pid": Path(paths.DRY_PID_PATH),
            "crash": Path(paths.DRY_CRASH_PATH),
            "supervisor_log": Path("var/dry/supervisor.log"),
            "trades": Path(paths.DRY_TRADES_PATH),
        }
    if mode == "live":
        return {
            "heartbeat": Path(paths.LIVE_HEARTBEAT_PATH),
            "pid": Path(paths.LIVE_PID_PATH),
            "crash": Path(paths.LIVE_CRASH_PATH),
            "supervisor_log": Path("var/live/supervisor.log"),
            "trades": Path(paths.LIVE_TRADES_PATH),
        }
    raise ValueError(f"unknown_mode: {mode!r}")


# -- Safe reads (BLOCKER #2: atomic read, never hold handle) -----------------

def _safe_read_json(path: Path) -> dict | None:
    """Atomic open-read-close. Never holds a file handle. None on any error."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (PermissionError, OSError):
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _safe_read_pid_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (PermissionError, OSError):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _safe_stat_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _safe_count_trades(path: Path) -> int | None:
    """Count non-header rows in trades.csv. Best-effort; None on any error."""
    try:
        with path.open(encoding="utf-8") as f:
            count = 0
            for i, _ in enumerate(f):
                count = i
            return count if count >= 0 else 0
    except FileNotFoundError:
        return None
    except (PermissionError, OSError):
        return None


# -- Process liveness -------------------------------------------------------

def _pid_is_our_bot(pid: int, mode: str) -> bool:
    """True iff PID is alive AND its cmdline looks like a ``run.py --<mode>`` process.

    We can't use plain ``psutil.pid_exists`` because Windows aggressively reuses
    PIDs -- a stale bot.pid could point at a PID now owned by svchost.exe or
    any other unrelated process. The cmdline check ensures we only report a
    PID as "our bot alive" when it actually is one.
    """
    try:
        import psutil
        if not psutil.pid_exists(int(pid)):
            return False
        proc = psutil.Process(int(pid))
        cmdline = " ".join(proc.cmdline() or [])
    except Exception:
        return False
    return f"run.py --{mode}" in cmdline


def _find_legacy_bot_pid(mode: str) -> int | None:
    """Scan process list for ``run.py --<mode>``. Excludes self.

    Used only for UNINSTRUMENTED detection -- finds bots that predate the
    Phase 2a heartbeat/PID-file instrumentation.
    """
    needle = f"run.py --{mode}"
    self_pid = os.getpid()
    try:
        import psutil
    except Exception:
        return None
    try:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if proc.info["pid"] == self_pid:
                    continue
                cmdline = proc.info.get("cmdline") or []
                if not cmdline:
                    continue
                cmd = " ".join(cmdline)
                if needle in cmd:
                    return int(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        return None
    return None


# -- Classification ---------------------------------------------------------

def _classify(
    mode: str,
    *,
    stale_threshold_s: float,
    startup_grace_s: float,
) -> tuple[str, dict[str, Any]]:
    """Return (status, fields) for the given mode.

    Fields are all optional — callers format with "-" when absent.
    """
    art = _artifacts_for_mode(mode)
    now = time.time()
    fields: dict[str, Any] = {}

    hb = _safe_read_json(art["heartbeat"])
    hb_mtime = _safe_stat_mtime(art["heartbeat"])
    hb_age: float | None = (now - hb_mtime) if hb_mtime is not None else None

    pid_from_file = _safe_read_pid_file(art["pid"])
    pid_file_mtime = _safe_stat_mtime(art["pid"])
    pid_file_age: float | None = (now - pid_file_mtime) if pid_file_mtime is not None else None
    # A pid file with a dead PID is effectively absent -- stale leftovers from
    # old infrastructure or from a bot that crashed without atexit running
    # should not mask the UNINSTRUMENTED signal or wrongly imply STARTING/STALE.
    pid_file_is_live = pid_from_file is not None and _pid_is_our_bot(pid_from_file, mode)

    # Select the PID to check: prefer heartbeat.pid when fresh, else pid file.
    heartbeat_pid: int | None = None
    if hb is not None:
        raw = hb.get("pid")
        if isinstance(raw, int):
            heartbeat_pid = raw

    # Populate diagnostic fields from heartbeat (best-effort).
    if hb is not None:
        if heartbeat_pid is not None:
            fields["pid"] = heartbeat_pid
        br = hb.get("bankroll_bnb")
        if isinstance(br, (int, float)):
            fields["bankroll"] = f"{float(br):.4f}"
        ic = hb.get("iteration_count")
        if isinstance(ic, int):
            fields["iterations"] = ic
        le = hb.get("last_epoch")
        if isinstance(le, int):
            fields["last_epoch"] = le
    # Only surface pid_from_file when it's actually live, otherwise it's
    # misleading noise in the log line.
    if heartbeat_pid is None and pid_file_is_live:
        fields["pid"] = pid_from_file

    if hb_age is not None:
        fields["hb_age"] = f"{hb_age:.1f}s"

    # Bets count (best-effort, read trades.csv).
    bets = _safe_count_trades(art["trades"])
    if bets is not None:
        fields["bets"] = bets

    # -- State precedence (first match wins) --

    # 1. UP: fresh heartbeat + PID alive.
    if (
        hb is not None
        and hb_age is not None
        and hb_age <= stale_threshold_s
        and heartbeat_pid is not None
        and _pid_is_our_bot(heartbeat_pid, mode)
    ):
        return "UP", fields

    # 2. STARTING: pid file alive + fresh, no/stale heartbeat.
    if (
        pid_file_is_live
        and (hb is None or (hb_age is not None and hb_age > stale_threshold_s))
        and pid_file_age is not None
        and pid_file_age <= startup_grace_s
    ):
        fields["since_pid_ts"] = f"{pid_file_age:.0f}s"
        return "STARTING", fields

    # 3. STALE: PID alive but heartbeat stale past threshold + past grace.
    #    Prefer heartbeat.pid for the liveness probe; fall back to pid file.
    stale_pid_alive = (
        (heartbeat_pid is not None and _pid_is_our_bot(heartbeat_pid, mode))
        or pid_file_is_live
    )
    if (
        stale_pid_alive
        and hb is not None
        and hb_age is not None
        and hb_age > stale_threshold_s
        and (pid_file_age is None or pid_file_age > startup_grace_s)
    ):
        return "STALE", fields

    # 4. CRASHED: crash.json present (regardless of process state).
    crash = _safe_read_json(art["crash"])
    crash_mtime = _safe_stat_mtime(art["crash"])
    if crash is not None and crash_mtime is not None:
        crash_age = now - crash_mtime
        fields["crash_age"] = f"{crash_age:.1f}s"
        exc_type = crash.get("exc_type")
        if isinstance(exc_type, str):
            fields["exc"] = exc_type
        last_epoch = crash.get("last_epoch")
        if isinstance(last_epoch, int):
            fields["last_epoch"] = last_epoch
        return "CRASHED", fields

    # 5. UNINSTRUMENTED: bot process running but no heartbeat AND no valid
    #    pid file (a stale pid file with a dead PID is treated as absent).
    if hb is None and not pid_file_is_live:
        legacy_pid = _find_legacy_bot_pid(mode)
        if legacy_pid is not None:
            fields.clear()
            fields["pid"] = legacy_pid
            fields["note"] = "legacy_no_heartbeat"
            return "UNINSTRUMENTED", fields

    # 6. DOWN -- no useful context to report.
    fields.clear()
    return "DOWN", fields


# -- Logging ---------------------------------------------------------------

def _iso_utc_now() -> str:
    """2026-04-22T19:55:00Z-style ISO-8601 UTC timestamp."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_fields(fields: dict[str, Any]) -> str:
    # Deterministic field order -- readability + easy grep/parse.
    order = (
        "pid", "hb_age", "bankroll", "bets", "iterations",
        "last_epoch", "since_pid_ts", "crash_age", "exc", "note",
    )
    parts = []
    for k in order:
        if k in fields:
            parts.append(f"{k}={fields[k]}")
    # Any keys we didn't enumerate go last, sorted for stability.
    extras = sorted(k for k in fields if k not in order)
    for k in extras:
        parts.append(f"{k}={fields[k]}")
    return " ".join(parts)


def _write_supervisor_line(log_path: Path, mode: str, status: str, fields: dict[str, Any]) -> None:
    """Append one line. Creates parent dir if missing. Never raises."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = _iso_utc_now()
        body = _format_fields(fields)
        line = f"{ts} STATUS={status} mode={mode}"
        if body:
            line = f"{line} {body}"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        # Also mirror to stdout for interactive / schtasks visibility.
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:
        # Last-ditch: if logging itself fails, at least stderr gets something.
        sys.stderr.write(f"supervisor_log_write_failed: {log_path}\n")


# -- Entrypoint ------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="supervisor.py",
        description="PancakeBot health supervisor (one-shot classify+log).",
    )
    p.add_argument(
        "--mode", required=True, choices=("dry", "live"),
        help="Which bot mode to supervise.",
    )
    p.add_argument(
        "--stale-threshold", type=float, default=5.0,
        help="Seconds after which a heartbeat is considered stale (default: 5.0).",
    )
    p.add_argument(
        "--startup-grace", type=float, default=90.0,
        help="Seconds after PID-file write during which a missing or stale heartbeat "
             "is tolerated (STARTING state; default: 90.0).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        status, fields = _classify(
            args.mode,
            stale_threshold_s=float(args.stale_threshold),
            startup_grace_s=float(args.startup_grace),
        )
    except Exception as e:
        # Supervisor errored during classification -- log to stderr, exit 99.
        sys.stderr.write(f"supervisor_classify_failed: {type(e).__name__}: {e}\n")
        sys.stderr.write(traceback.format_exc())
        return EXIT_ERROR

    art = _artifacts_for_mode(args.mode)
    _write_supervisor_line(art["supervisor_log"], args.mode, status, fields)
    return _STATUS_TO_EXIT.get(status, EXIT_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
