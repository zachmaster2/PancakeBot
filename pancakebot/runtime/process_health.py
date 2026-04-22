"""Process-health instrumentation: heartbeat file, PID file, crash dump.

Written by the runtime (dry/live) and consumed by an out-of-process supervisor
(see scripts/dry_supervisor.py in Phase 2b).

Three artifacts per mode:
  - ``var/<mode>/heartbeat.json``: updated every iteration + every second
    during sleeps. Its mtime is the supervisor's primary liveness signal.
    Contents: {pid, ts_wall, last_epoch, bankroll_bnb, iteration_count}.
  - ``var/<mode>/bot.pid``: contains the OS pid as plain text. Written at
    startup, best-effort removed at clean exit. NOT reliable as a sole
    liveness signal (the process may die without clearing it).
  - ``var/<mode>/crash.json``: written only when the top-level entrypoint
    catches an unhandled exception. Contents:
    {ts_wall, exc_type, exc_repr, traceback_str, last_epoch}.

All writes are atomic (tempfile + fsync + os.replace) with a bounded retry
on PermissionError (Windows file-lock race with antivirus / indexer). After
five consecutive heartbeat-write failures, ``write_heartbeat`` raises
InvariantError so the bot exits loudly rather than appearing alive to the
supervisor while silently bleeding.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from pathlib import Path

from pancakebot.log import warn
from pancakebot.util import InvariantError, ensure_parent_dir


# Max consecutive heartbeat write failures before we give up and hard-exit.
# Rationale: one transient PermissionError is survivable, but sustained
# failure means the supervisor will treat us as dead anyway -- better to die
# cleanly than drift.
_MAX_CONSECUTIVE_HEARTBEAT_FAILURES = 5

# Module-level failure counter (reset on every successful heartbeat).
_consecutive_heartbeat_failures: int = 0

# PermissionError retry schedule (seconds). Short enough not to push the
# critical bet path, long enough to get past typical AV scan windows.
_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.05, 0.1, 0.2)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically.

    - Creates parent dir if missing.
    - tempfile.mkstemp in the same directory (so os.replace is a rename, not a
      cross-device copy).
    - flush + fsync on the file descriptor before close so the data hits disk
      before the replace.
    - os.replace for atomic swap.
    - Retries up to 3 times on PermissionError (Windows file-lock race with
      antivirus / indexer).

    Raises OSError / PermissionError after exhausting retries.
    """
    ensure_parent_dir(str(path))
    attempt = 0
    last_exc: BaseException | None = None
    while True:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
            return
        except PermissionError as e:
            last_exc = e
            # Best-effort tempfile cleanup.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if attempt >= len(_RETRY_BACKOFF_SECONDS):
                raise
            time.sleep(_RETRY_BACKOFF_SECONDS[attempt])
            attempt += 1
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    # unreachable
    if last_exc is not None:
        raise last_exc


# -- PID file ----------------------------------------------------------------

def write_pid_file(path: Path, pid: int) -> None:
    """Write the PID as plain text atomically."""
    _atomic_write_text(path, f"{int(pid)}\n")


def clear_pid_file(path: Path) -> None:
    """Best-effort removal of the PID file. Registered via atexit.

    Silently ignores missing file, permission errors, etc. -- called during
    interpreter shutdown where raising isn't useful.
    """
    # noinspection PyBroadException
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


# -- Heartbeat ---------------------------------------------------------------

def write_heartbeat(
    path: Path,
    *,
    pid: int,
    ts_wall: float,
    last_epoch: int | None,
    bankroll_bnb: float | None,
    iteration_count: int,
) -> bool:
    """Write a heartbeat JSON atomically. Return True on success.

    On write failure: logs a WARN, increments the module-level consecutive-
    failure counter, and returns False. After
    ``_MAX_CONSECUTIVE_HEARTBEAT_FAILURES`` consecutive failures, raises
    ``InvariantError`` so the bot exits loudly.
    """
    global _consecutive_heartbeat_failures
    record = {
        "pid": int(pid),
        "ts_wall": float(ts_wall),
        "last_epoch": (int(last_epoch) if last_epoch is not None else None),
        "bankroll_bnb": (float(bankroll_bnb) if bankroll_bnb is not None else None),
        "iteration_count": int(iteration_count),
    }
    content = json.dumps(record, separators=(",", ":"), sort_keys=True)
    try:
        _atomic_write_text(path, content)
        _consecutive_heartbeat_failures = 0
        return True
    except Exception as e:
        _consecutive_heartbeat_failures += 1
        warn(
            "HEALTH", "HRTBT", "WRITE_FAIL",
            msg=(
                f"heartbeat write failed (consecutive={_consecutive_heartbeat_failures}/"
                f"{_MAX_CONSECUTIVE_HEARTBEAT_FAILURES}): {type(e).__name__}: {e}"
            ),
        )
        if _consecutive_heartbeat_failures >= _MAX_CONSECUTIVE_HEARTBEAT_FAILURES:
            raise InvariantError(
                f"heartbeat_write_failed_{_consecutive_heartbeat_failures}_times_consecutively"
            ) from e
        return False


def read_last_heartbeat(path: Path) -> dict | None:
    """Read + parse the heartbeat JSON. Returns None if absent or malformed.

    Used by the crash handler to populate ``last_epoch`` and by supervisors
    to classify staleness. Never raises on malformed input -- supervisors
    need to degrade gracefully.
    """
    if not path.exists():
        return None
    # noinspection PyBroadException
    try:
        text = path.read_text(encoding="utf-8")
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


# -- Crash dump --------------------------------------------------------------

def write_crash(path: Path, exc: BaseException, *, last_epoch: int | None) -> None:
    """Write a crash.json atomically. Swallows its own errors.

    This is a last-ditch write on a dying process; if it fails the process
    is exiting anyway and stderr will still carry the original traceback.
    """
    record = {
        "ts_wall": time.time(),
        "exc_type": type(exc).__name__,
        "exc_repr": repr(exc),
        "traceback_str": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "last_epoch": (int(last_epoch) if last_epoch is not None else None),
    }
    # noinspection PyBroadException
    try:
        _atomic_write_text(path, json.dumps(record, separators=(",", ":"), sort_keys=True))
    except Exception:
        # Last-ditch: can't do anything useful if this fails.
        pass
