"""Make a STOPPED sync look different from a QUIET one.

THE PROBLEM THIS EXISTS FOR. Every failure this project hit in August
became invisible the same way: something stopped, and the stopping looked
like quiet. The getLogs outage ran four days with zero bets and zero
alerts. The breaker fix was dead for days with no signal. repair_torn_tail
was correct, tested five times, and called by nothing. In each case the
system emitted NOTHING, and nothing is exactly what a healthy idle system
emits too.

A scheduled job on a personal machine has that failure mode by default.
There is no server, no cron mail, and nobody watching a dashboard. If the
task is disabled, or Task Scheduler never fires it, or the machine sleeps
through every window, the observable result is an absence -- and an absence
is not a signal.

THE DESIGN. Health is a POSITIVE, TIMESTAMPED assertion written on every
attempt, not a log line emitted on failure. The question an operator asks
is never "was there an error", it is "when did this last work". A file that
answers that question is readable even when the thing that writes it is
dead -- and its own staleness IS the alarm, because a stale heartbeat and a
missing heartbeat both mean the same thing and both are loud.

WHAT THIS DELIBERATELY DOES NOT CLAIM. A purely local heartbeat cannot
detect its own death: if the whole machine is off, nothing here notices.
That blind spot is irreducible without an external observer. What this does
is make the blind spot VISIBLE the moment the machine is used again,
instead of silent forever.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

HEALTH_PATH = "var/sync_health.json"

# A daily job that has not succeeded in this long is not "between runs".
# 36h allows one missed window plus the catch-up run before it complains,
# so a laptop closed overnight does not cry wolf.
STALE_AFTER_HOURS = 36.0

# Two consecutive misses is no longer bad luck.
DEAD_AFTER_HOURS = 60.0


def _utc_now() -> float:
    return time.time()


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def load(path: str = HEALTH_PATH) -> dict[str, Any]:
    """Read the heartbeat. A missing or corrupt file reads as NEVER RUN.

    Corruption degrading to "never ran" is deliberate: the alternative is
    raising, and a health file that can crash the health check is worse
    than one that reports the worst case.
    """
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError("not an object")
        return d
    except Exception:
        return {}


def _write(d: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        json.dump(d, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def record_attempt(path: str = HEALTH_PATH) -> None:
    """Called BEFORE the sync runs.

    Recording the attempt separately from the outcome is what distinguishes
    "ran and failed" from "never started" -- a task that is disabled or
    never fires leaves last_attempt untouched, which is a different and
    more serious diagnosis than a run that errored.
    """
    d = load(path)
    now = _utc_now()
    d["last_attempt_ts"] = now
    d["last_attempt_utc"] = _iso(now)
    d["attempts"] = int(d.get("attempts", 0)) + 1
    _write(d, path)


def record_success(*, detail: str = "", path: str = HEALTH_PATH) -> None:
    d = load(path)
    now = _utc_now()
    d["last_success_ts"] = now
    d["last_success_utc"] = _iso(now)
    d["last_exit_code"] = 0
    d["last_error"] = None
    d["consecutive_failures"] = 0
    d["last_detail"] = detail
    d["successes"] = int(d.get("successes", 0)) + 1
    _write(d, path)


def record_failure(*, exit_code: int, error: str = "",
                   path: str = HEALTH_PATH) -> None:
    d = load(path)
    now = _utc_now()
    d["last_failure_ts"] = now
    d["last_failure_utc"] = _iso(now)
    d["last_exit_code"] = int(exit_code)
    d["last_error"] = (error or "")[:2000]
    d["consecutive_failures"] = int(d.get("consecutive_failures", 0)) + 1
    d["failures"] = int(d.get("failures", 0)) + 1
    _write(d, path)


def assess(path: str = HEALTH_PATH, *, now: float | None = None) -> dict:
    """Classify current health. Never raises.

    Statuses:
      NEVER_RUN -- no heartbeat at all. The task may not be registered.
      OK        -- succeeded within STALE_AFTER_HOURS.
      STALE     -- no SUCCESS recently. This is the state that used to be
                   invisible, and it fires whether the cause is failure,
                   the task never firing, or the machine being off.
      DEAD      -- stale well past any legitimate gap.

    Staleness is measured from last SUCCESS, never from last attempt. A job
    that runs daily and fails daily is not healthy, and measuring from the
    attempt would report it as fine.
    """
    now = _utc_now() if now is None else now
    d = load(path)
    if not d or d.get("last_success_ts") is None:
        # An attempt with no success ever is still NEVER_RUN for the purpose
        # of "has this ever worked", but say which of the two it is.
        return {
            "status": "NEVER_RUN",
            "hours_since_success": None,
            "last_success_utc": None,
            "last_attempt_utc": d.get("last_attempt_utc"),
            "consecutive_failures": int(d.get("consecutive_failures", 0)),
            "last_error": d.get("last_error"),
            "detail": ("no successful sync has ever been recorded"
                       if d else
                       "no health file -- the scheduled task may not be "
                       "registered, or has never fired"),
        }

    hours = (now - float(d["last_success_ts"])) / 3600.0
    if hours >= DEAD_AFTER_HOURS:
        status = "DEAD"
    elif hours >= STALE_AFTER_HOURS:
        status = "STALE"
    else:
        status = "OK"
    return {
        "status": status,
        "hours_since_success": round(hours, 2),
        "last_success_utc": d.get("last_success_utc"),
        "last_attempt_utc": d.get("last_attempt_utc"),
        "consecutive_failures": int(d.get("consecutive_failures", 0)),
        "last_error": d.get("last_error"),
        "detail": d.get("last_detail", ""),
    }


def summary_line(a: dict) -> str:
    """One line an operator can read without context."""
    s = a["status"]
    if s == "NEVER_RUN":
        return f"SYNC {s}: {a['detail']}"
    return (
        f"SYNC {s}: last success {a['last_success_utc']} "
        f"({a['hours_since_success']}h ago), "
        f"consecutive_failures={a['consecutive_failures']}"
        + (f", last_error={a['last_error']}" if a.get("last_error") else "")
    )
