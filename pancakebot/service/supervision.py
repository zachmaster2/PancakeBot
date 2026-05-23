"""Pure bot-health classification + restart-history helpers (no Win32 deps).

Extracted from ``scripts/supervisor.py`` so the Windows Service in
``pancakebot.service.common`` can reuse the exact same classification
semantics. Pure-Python; safe to import on any platform; covered by unit
tests under ``tests/test_service_lifecycle.py``.

Classification order matches the legacy supervisor (first match wins):
  UP            - fresh heartbeat (mtime < stale_threshold) + PID alive
  STARTING      - PID file alive, no/stale heartbeat, PID file age < startup_grace
  STALE         - PID alive but heartbeat stale past threshold (past grace)
  CRASHED       - crash.json present (regardless of process state)
  UNINSTRUMENTED- bot process alive (psutil match on run.py --<mode>) but
                  no heartbeat AND no PID file (legacy pre-Phase-2a bot)
  DOWN          - none of the above
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from pancakebot import paths

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Match supervisor.py defaults exactly so behavior is unchanged across the
# migration. Tunable per-service if we ever need to.
DEFAULT_STALE_THRESHOLD_S: float = 5.0
DEFAULT_STARTUP_GRACE_S: float = 90.0

# Retry-once backoff for transient read failures (Windows AV file-lock race,
# atomic-rename windows, psutil process-iter transients). Mirrors the legacy
# supervisor's 500ms backoff.
_TRANSIENT_READ_BACKOFF_S: float = 0.5


def artifacts_for_mode(mode: str) -> dict[str, Path]:
    """Return all supervisor-visible paths for ``mode`` (``dry`` or ``live``).

    All paths are resolved against the repo root so the service works
    correctly regardless of CWD (SCM starts services with arbitrary CWDs).
    """
    if mode == "dry":
        return {
            "heartbeat": _REPO_ROOT / paths.DRY_HEARTBEAT_PATH,
            "pid": _REPO_ROOT / paths.DRY_PID_PATH,
            "crash": _REPO_ROOT / paths.DRY_CRASH_PATH,
            "supervisor_log": _REPO_ROOT / "var/dry/supervisor.log",
            "trades": _REPO_ROOT / paths.DRY_TRADES_PATH,
            "last_alert": _REPO_ROOT / "var/dry/last_alert.json",
            "restart_history": _REPO_ROOT / "var/dry/restart_history.jsonl",
            "logs_dir": _REPO_ROOT / "var/dry/logs",
        }
    if mode == "live":
        return {
            "heartbeat": _REPO_ROOT / paths.LIVE_HEARTBEAT_PATH,
            "pid": _REPO_ROOT / paths.LIVE_PID_PATH,
            "crash": _REPO_ROOT / paths.LIVE_CRASH_PATH,
            "supervisor_log": _REPO_ROOT / "var/live/supervisor.log",
            "trades": _REPO_ROOT / paths.LIVE_TRADES_PATH,
            "last_alert": _REPO_ROOT / "var/live/last_alert.json",
            "restart_history": _REPO_ROOT / "var/live/restart_history.jsonl",
            "logs_dir": _REPO_ROOT / "var/live/logs",
        }
    raise ValueError(f"unknown_mode: {mode!r}")


# -- Safe reads (atomic open-read-close, never hold handles) ----------------

def _safe_read_json(path: Path) -> dict | None:
    """Atomic read with retry-once on transient failure."""
    def _once() -> dict | None:
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

    first = _once()
    if first is not None:
        return first
    time.sleep(_TRANSIENT_READ_BACKOFF_S)
    return _once()


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
    """True iff PID is alive AND its cmdline contains ``run.py --<mode>``.

    Windows aggressively reuses PIDs; a stale bot.pid could point at an
    unrelated process. The cmdline check ensures we only report a PID as
    "our bot alive" when it actually is one. Retry-once on transient
    psutil failures.
    """
    needle = f"run.py --{mode}"
    for attempt in range(2):
        try:
            import psutil
            if not psutil.pid_exists(int(pid)):
                return False
            proc = psutil.Process(int(pid))
            cmdline = " ".join(proc.cmdline() or [])
            return needle in cmdline
        except Exception:
            if attempt == 0:
                time.sleep(_TRANSIENT_READ_BACKOFF_S)
                continue
            return False
    return False


def find_legacy_bot_pid(mode: str) -> int | None:
    """Scan process list for ``run.py --<mode>``. Used for UNINSTRUMENTED detection."""
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
            except Exception:
                continue
    except Exception:
        return None
    return None


# -- Classification ---------------------------------------------------------

def classify_state(
    mode: str,
    *,
    stale_threshold_s: float = DEFAULT_STALE_THRESHOLD_S,
    startup_grace_s: float = DEFAULT_STARTUP_GRACE_S,
) -> tuple[str, dict[str, Any]]:
    """Return (status, fields) for the given mode.

    Pure function. Reads filesystem artifacts and process list. Never raises;
    fields populated best-effort.
    """
    art = artifacts_for_mode(mode)
    now = time.time()
    fields: dict[str, Any] = {}

    hb = _safe_read_json(art["heartbeat"])
    hb_mtime = _safe_stat_mtime(art["heartbeat"])
    hb_age: float | None = (now - hb_mtime) if hb_mtime is not None else None

    pid_from_file = _safe_read_pid_file(art["pid"])
    pid_file_mtime = _safe_stat_mtime(art["pid"])
    pid_file_age: float | None = (now - pid_file_mtime) if pid_file_mtime is not None else None
    pid_file_is_live = pid_from_file is not None and _pid_is_our_bot(pid_from_file, mode)

    heartbeat_pid: int | None = None
    if hb is not None:
        raw = hb.get("pid")
        if isinstance(raw, int):
            heartbeat_pid = raw

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
    if heartbeat_pid is None and pid_file_is_live:
        fields["pid"] = pid_from_file

    if hb_age is not None:
        fields["hb_age"] = f"{hb_age:.1f}s"

    bets = _safe_count_trades(art["trades"])
    if bets is not None:
        fields["bets"] = bets

    # State precedence — first match wins.

    # 1. UP
    if (
        hb is not None
        and hb_age is not None
        and hb_age <= stale_threshold_s
        and heartbeat_pid is not None
        and _pid_is_our_bot(heartbeat_pid, mode)
    ):
        return "UP", fields

    # 2. STARTING
    if (
        pid_file_is_live
        and (hb is None or (hb_age is not None and hb_age > stale_threshold_s))
        and pid_file_age is not None
        and pid_file_age <= startup_grace_s
    ):
        fields["since_pid_ts"] = f"{pid_file_age:.0f}s"
        return "STARTING", fields

    # 3. STALE
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

    # 4. CRASHED
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

    # 5. UNINSTRUMENTED
    if hb is None and not pid_file_is_live:
        legacy_pid = find_legacy_bot_pid(mode)
        if legacy_pid is not None:
            fields.clear()
            fields["pid"] = legacy_pid
            fields["note"] = "legacy_no_heartbeat"
            return "UNINSTRUMENTED", fields

    # 6. DOWN
    fields.clear()
    return "DOWN", fields


# -- Restart-history (crashloop limiter) -----------------------------------

def read_restart_history(path: Path) -> list[dict]:
    entries: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return entries
    except (PermissionError, OSError):
        return entries
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def prune_history(entries: list[dict], now: float, window_s: float) -> list[dict]:
    """Drop entries older than ``window_s``."""
    kept = []
    for e in entries:
        ts_wall = e.get("ts_wall")
        if isinstance(ts_wall, (int, float)) and (now - float(ts_wall)) <= window_s:
            kept.append(e)
    return kept


def count_within(entries: list[dict], now: float, window_s: float) -> int:
    n = 0
    for e in entries:
        ts_wall = e.get("ts_wall")
        if isinstance(ts_wall, (int, float)) and (now - float(ts_wall)) <= window_s:
            n += 1
    return n


def write_restart_history(path: Path, entries: list[dict]) -> None:
    """Atomic write of JSONL. Parent dir created on demand. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.name + ".tmp")
        body = "\n".join(
            json.dumps(e, sort_keys=True, separators=(",", ":")) for e in entries
        )
        if body:
            body += "\n"
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Best-effort. The supervisor must not crash on log-write failure.
        sys.stderr.write(f"restart_history_write_failed: {path}\n")
