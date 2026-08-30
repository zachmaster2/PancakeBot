"""A stopped sync must look different from a quiet one.

THE FAILURE CLASS THIS ADDRESSES. Every August failure became invisible the
same way: something stopped, and the stopping looked like quiet. The
getLogs outage ran four days with zero bets and zero alerts. The breaker
fix was dead for days. repair_torn_tail was correct, tested five times, and
called by nothing. In each case the system emitted NOTHING -- and nothing
is exactly what a healthy idle system emits.

So health is a POSITIVE, TIMESTAMPED assertion written on every attempt,
never a log line emitted on failure. The operator's real question is not
"was there an error" but "when did this last work", and a file answering
that is readable even when the thing that writes it is dead. Its own
staleness IS the alarm.

STALENESS IS MEASURED FROM LAST SUCCESS, NEVER FROM LAST ATTEMPT. A job
that runs daily and fails daily is not healthy, and measuring from the
attempt would report it as fine. That distinction is tested here because it
is the one that would quietly invert the whole design.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from pancakebot.ops import sync_health as H  # noqa: E402


def _p(tmp_path) -> str:
    return str(tmp_path / "health.json")


# ---- the state machine -----------------------------------------------------

def test_no_health_file_reads_as_never_run(tmp_path):
    """The task may not be registered at all. That must be a LOUD state,
    not an empty one."""
    a = H.assess(_p(tmp_path))
    assert a["status"] == "NEVER_RUN"
    assert "not be registered" in a["detail"]


def test_attempted_but_never_succeeded_is_distinguishable(tmp_path):
    """Different diagnosis from 'never fired': the task IS running, the
    sync is failing. Conflating them sends the operator to the wrong place."""
    p = _p(tmp_path)
    H.record_attempt(p)
    H.record_failure(exit_code=1, error="boom", path=p)
    a = H.assess(p)
    assert a["status"] == "NEVER_RUN"
    assert a["detail"] == "no successful sync has ever been recorded", (
        "this must NOT read as 'the task may not be registered' -- the task "
        "clearly is running; the sync is what is failing")
    assert a["last_attempt_utc"] is not None, (
        "the attempt must be recorded even when the run failed -- it is "
        "what separates 'ran and failed' from 'never fired'")


def test_success_makes_it_ok(tmp_path):
    p = _p(tmp_path)
    H.record_attempt(p)
    H.record_success(path=p)
    assert H.assess(p)["status"] == "OK"


@pytest.mark.parametrize("hours,expected", [
    (0, "OK"), (12, "OK"), (35.9, "OK"),
    (36, "STALE"), (50, "STALE"), (59.9, "STALE"),
    (60, "DEAD"), (200, "DEAD"),
])
def test_the_staleness_thresholds(tmp_path, hours, expected):
    p = _p(tmp_path)
    H.record_success(path=p)
    now = time.time() + hours * 3600
    assert H.assess(p, now=now)["status"] == expected


def test_staleness_is_measured_from_SUCCESS_not_from_ATTEMPT(tmp_path):
    """THE inversion that would break the design. A job attempting daily
    and failing daily must read STALE, not OK."""
    p = _p(tmp_path)
    H.record_success(path=p)
    d = json.load(open(p))
    d["last_success_ts"] = time.time() - 50 * 3600      # succeeded 50h ago
    with open(p, "w") as f:
        json.dump(d, f)
    H.record_attempt(p)                                  # attempted JUST NOW
    H.record_failure(exit_code=1, path=p)
    a = H.assess(p)
    assert a["status"] == "STALE", (
        "a job failing every day reported healthy because it was measured "
        "from the attempt")


def test_consecutive_failures_accumulate_and_reset(tmp_path):
    p = _p(tmp_path)
    H.record_success(path=p)
    for _ in range(3):
        H.record_failure(exit_code=1, path=p)
    assert H.assess(p)["consecutive_failures"] == 3
    H.record_success(path=p)
    assert H.assess(p)["consecutive_failures"] == 0


def test_a_corrupt_health_file_degrades_and_never_raises(tmp_path):
    """A health file that can crash the health check is worse than none."""
    p = _p(tmp_path)
    with open(p, "w") as f:
        f.write("{not json at all")
    a = H.assess(p)
    assert a["status"] == "NEVER_RUN"


def test_the_write_is_atomic(tmp_path):
    """A torn health file is the same class of bug as a torn store."""
    src = (Path(_REPO_ROOT) / "pancakebot" / "ops" / "sync_health.py").read_text(
        encoding="utf-8")
    assert "os.replace(tmp, path)" in src
    assert "os.fsync" in src


# ---- the watchdog ----------------------------------------------------------

def _assess_stale(hours: float) -> dict:
    return {
        "status": "STALE", "hours_since_success": hours,
        "last_success_utc": "2026-08-28T06:30:00Z",
        "last_attempt_utc": None, "consecutive_failures": 2,
        "last_error": "network down", "detail": "",
    }


def test_the_marker_filename_carries_the_diagnosis():
    """A file that must be OPENED to be understood is one step from being
    ignored. The filename alone has to say what is wrong."""
    import sync_watchdog as W
    assert W._marker_name(_assess_stale(50)) == \
        "PANCAKEBOT_SYNC_STOPPED_2_DAYS_AGO.txt"
    assert W._marker_name(_assess_stale(5)) == \
        "PANCAKEBOT_SYNC_STOPPED_5_HOURS_AGO.txt"
    assert W._marker_name({"status": "NEVER_RUN", "hours_since_success": None}) \
        == "PANCAKEBOT_SYNC_HAS_NEVER_RUN.txt"


def test_the_marker_appears_when_stale_and_clears_on_recovery(tmp_path):
    import sync_watchdog as W
    desk = tmp_path / "desk"
    desk.mkdir()
    health = str(tmp_path / "h.json")

    H.record_success(path=health)
    d = json.load(open(health))
    d["last_success_ts"] = time.time() - 50 * 3600
    with open(health, "w") as f:
        json.dump(d, f)

    rc = W.main.__wrapped__ if hasattr(W.main, "__wrapped__") else None
    sys.argv = ["w", "--health", health, "--desktop", str(desk), "--no-discord"]
    assert W.main() == 1
    markers = [f for f in os.listdir(desk) if f.startswith("PANCAKEBOT_SYNC")]
    assert markers, "no Desktop marker was raised for a stale sync"

    H.record_success(path=health)
    sys.argv = ["w", "--health", health, "--desktop", str(desk), "--no-discord"]
    assert W.main() == 0
    assert not [f for f in os.listdir(desk) if f.startswith("PANCAKEBOT_SYNC")], (
        "the marker survived a successful sync -- a stale alarm trains the "
        "operator to ignore it")


def test_the_watchdog_exits_nonzero_when_unhealthy(tmp_path):
    """Task Scheduler's Last Result must carry the signal too."""
    import sync_watchdog as W
    desk = tmp_path / "d2"
    desk.mkdir()
    sys.argv = ["w", "--health", str(tmp_path / "nope.json"),
                "--desktop", str(desk), "--no-discord"]
    assert W.main() == 1


def test_the_marker_body_names_what_is_at_risk(tmp_path):
    import sync_watchdog as W
    body = W._marker_body(_assess_stale(50))
    assert "171.6 days" in body, (
        "the marker must say WHY it matters -- the OKX horizon is what "
        "makes a stalled sync permanent data loss rather than a delay")
    assert "run.py --sync" in body, "it must say what to do"


def test_discord_is_best_effort_and_never_blocks_the_marker(tmp_path):
    """The Desktop marker is primary precisely because it does not need
    the network, which is one of the things that can be broken."""
    import sync_watchdog as W
    old = os.environ.pop("PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL", None)
    old2 = os.environ.pop("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", None)
    try:
        assert "not configured" in W._try_discord("x")
    finally:
        if old:
            os.environ["PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL"] = old
        if old2:
            os.environ["PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL"] = old2


# ---- the wrapper contract --------------------------------------------------

def test_lock_held_is_not_recorded_as_a_failure():
    """Exit 3 means the single-instance lock did its job. Recording it as a
    failure would make a correctly-behaving system look broken and train
    the operator to ignore the alarm."""
    ps = (Path(_REPO_ROOT) / "scripts" / "daily_sync.ps1").read_text(
        encoding="utf-8")
    # The elseif branch ONLY -- ending where the else branch begins, so the
    # else branch's legitimate record_failure call is not scanned.
    i = ps.index("$code -eq 3")
    j = ps.index("else {", i)
    window = ps[i:j]
    assert "record_failure" not in window, (
        "the lock-held path records a failure -- a working single-instance "
        "guard would look like a broken sync")
    assert "SKIPPED" in window


def test_the_wrapper_records_the_attempt_before_running():
    ps = (Path(_REPO_ROOT) / "scripts" / "daily_sync.ps1").read_text(
        encoding="utf-8")
    assert ps.index("record_attempt") < ps.index("run.py --sync"), (
        "the attempt must be recorded BEFORE the run, or a crash mid-sync "
        "is indistinguishable from the task never firing")


def test_the_repair_runs_before_the_network_is_needed():
    """Found by simulation: the sync died at the RPC step before the repair
    ran, so a torn tail would survive every network-down day. The repair is
    local and free and must not be gated behind connectivity."""
    src = (Path(_REPO_ROOT) / "pancakebot" / "app.py").read_text(encoding="utf-8")
    assert src.index("repair_torn_tail(_store_path)") < \
        src.index("refreshing contract_constants cache from chain"), (
        "the torn-tail repair is gated behind the network again")
