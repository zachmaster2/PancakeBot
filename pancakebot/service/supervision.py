"""Pure bot-health classification + restart-history helpers (no Win32 deps).

Two classifier entry points:

  classify_running_bot(proc, proc_started_at, art, ...)
      The authoritative in-loop classifier used by the Windows Service.
      Liveness is determined by ``Popen.poll()`` — zero filesystem race.
      This is what the service's supervision loop calls every tick.

  classify_state(mode, ...)
      Legacy artifact-only classifier (bot.pid + crash.json). Kept for
      first-run / no-Popen-handle use cases (e.g., checking whether a
      bot is somehow already running outside the service). Vulnerable
      to the post-spawn DOWN race that classify_running_bot eliminates
      — do not use in the supervision loop after the initial spawn.

Status values returned by both:
  UP            - bot alive
  STARTING      - bot alive, within startup grace
  CRASHED       - bot dead, crash.json present
  DOWN          - bot dead (or absent), no crash.json
  UNINSTRUMENTED- (classify_state only) legacy bot detected outside service

Heartbeat-staleness STALE classification removed 2026-05-27 (Step 27a full
cleanup). The 5s heartbeat-age threshold was firing on transient BSC RPC
hedged-timeouts that auto-resolve next round (~12 false-positive restarts/24h
with no real bot dysfunction). Process-death is the only restart trigger now.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pancakebot import paths

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Legacy artifact-classifier grace (PID-file mtime based).
DEFAULT_STARTUP_GRACE_S: float = 90.0

# New Popen-based classifier grace. Tighter than the legacy 90s because we
# now know the bot's actual start time directly — no need to wait for a
# PID-file write. 30s is comfortably longer than the bot's ~5s init path
# (RPC poller + bankroll fetch + first round catch-up).
DEFAULT_RUN_GRACE_S: float = 30.0

# Retry-once backoff for transient read failures (Windows AV file-lock race,
# atomic-rename windows, psutil process-iter transients). Mirrors the legacy
# supervisor's 500ms backoff.
_TRANSIENT_READ_BACKOFF_S: float = 0.5


# ---------------------------------------------------------------------------
# Service-safe stderr (sys.stderr is None when hosted by pythonservice.exe)
# ---------------------------------------------------------------------------

def safe_stderr_write(msg: str) -> None:
    """Write to sys.stderr when available; silently drop when it's None.

    Windows Services hosted by pythonservice.exe run with NO stdio handles
    attached, so ``sys.stderr`` is ``None`` and ``sys.stderr.write(...)``
    raises ``AttributeError: 'NoneType' object has no attribute 'write'``
    — a hazard that previously crashed the supervisor (2026-05-23 boot).

    Routes through ``servicemanager.LogErrorMsg`` when running inside a
    service (importable + sys.stderr is None), so error-path diagnostics
    still land somewhere visible (Windows Event Log → Application,
    ProviderName = service name). Falls back to plain ``sys.stderr`` for
    interactive / test invocation.
    """
    if sys.stderr is not None:
        try:
            sys.stderr.write(msg if msg.endswith("\n") else msg + "\n")
            return
        except Exception:
            pass  # stderr present but broken — fall through to servicemanager
    # No stderr → try Windows Event Log via servicemanager (only available
    # when pywin32 is installed AND we're inside a service host).
    try:
        import servicemanager
        servicemanager.LogErrorMsg(msg.rstrip())
    except Exception:
        # Last resort: silently swallow. The alternative is letting an
        # AttributeError propagate up the call stack and crash the
        # service, which is exactly the bug we're guarding against.
        pass


def artifacts_for_mode(mode: str) -> dict[str, Path]:
    """Return all supervisor-visible paths for ``mode`` (``dry`` or ``live``).

    All paths are resolved against the repo root so the service works
    correctly regardless of CWD (SCM starts services with arbitrary CWDs).
    """
    if mode == "dry":
        return {
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
    startup_grace_s: float = DEFAULT_STARTUP_GRACE_S,
) -> tuple[str, dict[str, Any]]:
    """Return (status, fields) for the given mode.

    Pure function. Reads filesystem artifacts and process list. Never raises;
    fields populated best-effort.

    Heartbeat reads removed 2026-05-27 (Step 27a). Liveness inferred from
    bot.pid + crash.json + process-listing only.
    """
    art = artifacts_for_mode(mode)
    now = time.time()
    fields: dict[str, Any] = {}

    pid_from_file = _safe_read_pid_file(art["pid"])
    pid_file_mtime = _safe_stat_mtime(art["pid"])
    pid_file_age: float | None = (now - pid_file_mtime) if pid_file_mtime is not None else None
    pid_file_is_live = pid_from_file is not None and _pid_is_our_bot(pid_from_file, mode)

    if pid_file_is_live:
        fields["pid"] = pid_from_file

    bets = _safe_count_trades(art["trades"])
    if bets is not None:
        fields["bets"] = bets

    # State precedence — first match wins.

    # 1. UP — PID file points to a live process past startup grace.
    if pid_file_is_live and pid_file_age is not None and pid_file_age > startup_grace_s:
        return "UP", fields

    # 2. STARTING — PID file points to a live process inside startup grace.
    if pid_file_is_live and pid_file_age is not None and pid_file_age <= startup_grace_s:
        fields["since_pid_ts"] = f"{pid_file_age:.0f}s"
        return "STARTING", fields

    # 3. CRASHED — process dead, crash.json present.
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

    # 4. UNINSTRUMENTED — bot running outside service control.
    if not pid_file_is_live:
        legacy_pid = find_legacy_bot_pid(mode)
        if legacy_pid is not None:
            fields.clear()
            fields["pid"] = legacy_pid
            fields["note"] = "legacy_no_pid_file"
            return "UNINSTRUMENTED", fields

    # 5. DOWN
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
        # safe_stderr_write handles sys.stderr=None (service-hosted) cleanly.
        safe_stderr_write(f"restart_history_write_failed: {path}")


# ---------------------------------------------------------------------------
# Authoritative Popen-based classifier (new — used by the supervision loop)
# ---------------------------------------------------------------------------

def classify_running_bot(
    proc: subprocess.Popen | None,
    proc_started_at: float | None,
    art: dict[str, Path],
    *,
    startup_grace_s: float = DEFAULT_RUN_GRACE_S,
) -> tuple[str, dict[str, Any]]:
    """Classify the bot child using the Popen handle as the truth source.

    Eliminates the post-spawn DOWN race that the artifact-only classifier
    suffers from. ``proc.poll()`` is the authoritative liveness signal.

    Args:
        proc: the Popen object for the bot child, or None if no bot has
            been spawned yet (treated as DOWN).
        proc_started_at: wall-clock time when proc was spawned (from
            ``time.time()``). Used for the startup-grace window.
        art: artifacts dict (from ``artifacts_for_mode``) — used to read
            crash.json on process death.
        startup_grace_s: window after spawn during which a slow-init bot
            is reported as STARTING rather than UP.

    Returns:
        (status, fields) where status is one of UP, STARTING, CRASHED, DOWN.
        Fields are best-effort diagnostics.

    Heartbeat-staleness STALE classification removed 2026-05-27 (Step 27a).

    Never raises.
    """
    fields: dict[str, Any] = {}

    # Decide liveness from Popen — authoritative, zero race.
    proc_alive: bool = False
    if proc is not None:
        try:
            poll_result = proc.poll()
        except Exception:
            poll_result = None  # be conservative on any read error
        proc_alive = (poll_result is None)
        fields["pid"] = proc.pid

    if proc_alive:
        proc_uptime: float = (
            (time.time() - proc_started_at) if proc_started_at is not None else 0.0
        )
        fields["proc_uptime"] = f"{proc_uptime:.1f}s"

        if proc_uptime < startup_grace_s:
            return "STARTING", fields

        return "UP", fields

    # Process is dead (or proc is None). Distinguish CRASHED (crash.json
    # present) from DOWN (no signal).
    crash = _safe_read_json(art["crash"])
    if crash is not None:
        exc_type = crash.get("exc_type")
        if isinstance(exc_type, str):
            fields["exc"] = exc_type
        last_epoch = crash.get("last_epoch")
        if isinstance(last_epoch, int):
            fields["last_epoch"] = last_epoch
        return "CRASHED", fields

    return "DOWN", fields
