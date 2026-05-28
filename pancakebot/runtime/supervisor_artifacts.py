"""Supervisor artifacts: PID file + crash dump.

Written by the runtime (dry/live) and consumed by the Windows Service
supervisor.

Two artifacts per mode:
  - ``var/<mode>/bot.pid``: contains the OS pid as plain text. Written at
    startup, best-effort removed at clean exit. NOT reliable as a sole
    liveness signal (the process may die without clearing it).
  - ``var/<mode>/crash.json``: written only when the top-level entrypoint
    catches an unhandled exception. Contents:
    {ts_wall, exc_type, exc_repr, traceback_str, last_epoch}.

All writes are atomic (tempfile + fsync + os.replace) with a bounded retry
on PermissionError (Windows file-lock race with antivirus / indexer).
The supervisor uses ``Popen.poll()`` as the authoritative liveness signal;
these artifacts are passive forensic records, not heartbeat polls.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from pathlib import Path

from pancakebot.util import ensure_parent_dir


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
    while True:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
            return
        except PermissionError:
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


# Default: 0 (always archive on startup). Any crash.json present when a
# new bot is calling this helper must be from a previous bot incarnation;
# the new bot has just acquired the PID file slot, so it isn't its own
# crash. Leaving a lingering crash.json in place caused 2026-04-25
# false-CRASHED events where the supervisor saw an old crash.json an
# hour after the original crash and killed a healthy bot. The previous
# 60s threshold was justified as "give the writer time to finish" --
# but the writer is the dead previous bot, and the supervisor's CRASHED
# alert (the only other reader) has already fired by the time it
# triggers --restart.
_LINGERING_CRASH_MIN_AGE_SECONDS: float = 0.0


def archive_lingering_crash_file(crash_path: Path, *, min_age_seconds: float = _LINGERING_CRASH_MIN_AGE_SECONDS) -> Path | None:
    """Rename an existing crash.json to crash_archive_<ts>.json on bot startup.

    Preserves forensic data (no deletion) while preventing the supervisor from
    classifying a fresh bot as CRASHED based on a leftover crash marker from a
    previous bot. Called from run.py immediately after the bot writes its PID
    file, before the main runtime loop begins.

    Returns the archive path on success, ``None`` if:
      - the crash file doesn't exist (no-op, the common case), or
      - the file is younger than ``min_age_seconds`` (default 0 -- always
        archive; can be raised by the caller for special cases), or
      - any error occurred (silently swallowed; bot startup must not be blocked
        by a cleanup failure).

    Default policy: always archive. Any crash.json present at bot startup
    must be from a previous bot (we wouldn't be running run.py if our own
    process had crashed). The supervisor's CRASHED alert is fired BEFORE
    it triggers a --restart, so by the time this helper runs the alert
    has already gone out -- archiving the file doesn't lose any signal.

    The archive filename uses the ORIGINAL crash timestamp (file mtime), so
    the filename remains chronologically meaningful across many archives:
    ``crash_archive_YYYYMMDD-HHMMSS.json``. Collisions (multiple crashes in
    the same second, or re-archive) get a numeric suffix.

    Concurrent-startup safety: run.py calls ``find_duplicate_bots`` BEFORE
    this helper runs (see run.py), so two supervisor-spawned bots can't both
    reach this line. A residual TOCTOU between ``archive.exists()`` and
    ``os.rename`` exists only in the theoretical path where the dup-check
    misses a truly simultaneous spawn -- the swallow-all handler then loses
    at most one archive, and the winner still archives successfully.
    """
    # noinspection PyBroadException
    try:
        if not crash_path.exists():
            return None
        st = crash_path.stat()
        age = time.time() - st.st_mtime
        if age < min_age_seconds:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(st.st_mtime))
        archive = crash_path.parent / f"crash_archive_{stamp}.json"
        # Collision handling: multiple archives at the same second-granularity
        # stamp (rare; only if a previous archive was created from a crash at
        # the same wall-clock second) get a numeric suffix.
        suffix = 1
        while archive.exists():
            archive = crash_path.parent / f"crash_archive_{stamp}_{suffix}.json"
            suffix += 1
        os.rename(str(crash_path), str(archive))
        return archive
    except Exception:
        # Never block bot startup on a cleanup failure.
        return None
