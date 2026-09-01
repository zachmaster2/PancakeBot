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

# An attempt with no outcome after this long is ORPHANED: a run started and
# never reported success or failure.
#
# WHY THIS EXISTS, and why it is the more important half of the 2026-09-01
# fix. The wrapper died mid-stream that morning, so record_failure never
# ran, and the health file showed consecutive_failures=0 with a stale
# last_exit_code=0 -- a FAILED run was indistinguishable from a run that had
# not happened yet. Making record_failure unconditional fixes THAT way of
# dying. It cannot fix the next one.
#
# This rule needs no theory about how the wrapper died. If an attempt is
# newer than every recorded outcome and enough time has passed, something
# began and never finished, whatever the mechanism. It would have caught
# 2026-09-01 with nobody having anticipated PowerShell's stderr semantics,
# which is exactly the category of failure that keeps hurting this project.
#
# 2 hours: observed syncs run 5-7 minutes, so this is ~20x a normal run, and
# it is comfortably inside the 2.5h between the sync's 06:30 and the
# watchdog's 09:00 -- so a morning orphan is caught the SAME day rather than
# a day later. A sync legitimately running longer than 2h would be abnormal
# in its own right and worth surfacing.
ORPHAN_AFTER_HOURS = 2.0


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

    orphan_hours = _orphan_age_hours(d, now)
    detail = d.get("last_detail", "")
    if orphan_hours is not None:
        # An orphan outranks OK -- a run that started and vanished is a real
        # condition, not a quiet day. It does NOT outrank STALE/DEAD, which
        # are strictly more serious; there the orphan rides in the detail.
        if status == "OK":
            status = "ORPHANED"
        detail = (f"a run started {orphan_hours:.1f}h ago and never reported "
                  f"an outcome (attempt newer than every success and failure)"
                  + (f"; {detail}" if detail else ""))

    return {
        "status": status,
        "hours_since_success": round(hours, 2),
        "hours_since_orphaned_attempt": (None if orphan_hours is None
                                         else round(orphan_hours, 2)),
        "last_success_utc": d.get("last_success_utc"),
        "last_attempt_utc": d.get("last_attempt_utc"),
        "consecutive_failures": int(d.get("consecutive_failures", 0)),
        "last_error": d.get("last_error"),
        "detail": detail,
    }


def _orphan_age_hours(d: dict, now: float) -> float | None:
    """Age of an attempt that never reported an outcome, else None.

    Deliberately makes no assumption about HOW the outcome went missing.
    The only inputs are three timestamps.
    """
    at = d.get("last_attempt_ts")
    if at is None:
        return None
    at = float(at)
    for k in ("last_success_ts", "last_failure_ts"):
        v = d.get(k)
        if v is not None and float(v) >= at:
            return None            # an outcome was recorded for this attempt
    age = (now - at) / 3600.0
    return age if age >= ORPHAN_AFTER_HOURS else None


def summary_line(a: dict) -> str:
    """One line an operator can read without context."""
    s = a["status"]
    if s == "NEVER_RUN":
        return f"SYNC {s}: {a['detail']}"
    return (
        f"SYNC {s}: last success {a['last_success_utc']} "
        f"({a['hours_since_success']}h ago), "
        f"consecutive_failures={a['consecutive_failures']}"
        + (f", ORPHANED ATTEMPT {a['hours_since_orphaned_attempt']}h old"
           if a.get("hours_since_orphaned_attempt") is not None else "")
        + (f", last_error={a['last_error']}" if a.get("last_error") else "")
    )
