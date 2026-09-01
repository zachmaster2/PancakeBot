"""A wrapper must not be able to fail without saying so. 2026-09-01.

THE INCIDENT. The first genuinely unattended sync fired at 06:30, ran for
two minutes, and died mid-stream. Task Scheduler recorded LastResult=1, but
the health file still read:

    consecutive_failures : 0
    last_exit_code       : 0        (stale, from the previous success)
    last_error           : null

record_failure never ran, so a FAILED run was indistinguishable from one
that had not happened yet.

CAUSE. `$ErrorActionPreference = 'Stop'` at script scope plus `*>&1` on a
native command. In PowerShell 5.1 merging a native command's stderr wraps
each line in an ErrorRecord; under 'Stop' that is a TERMINATING error, so
the wrapper aborted the first time anything wrote one line to stderr. It
had survived two earlier runs purely because nothing happened to write any.

TWO LAYERS OF FIX, AND THE SECOND MATTERS MORE.

  1. Specific: stop dying that way -- 'Continue' around the native call,
     stderr captured as data, outcome recorded in a `finally`.

  2. General: make "started and never finished" a STATE. If an attempt is
     newer than every recorded outcome, a run began and vanished, whatever
     the mechanism. That needs no theory about how the wrapper died, so it
     holds for failure modes nobody has thought of -- which is the category
     that produced this one. It would have caught 2026-09-01 with nobody
     having anticipated PowerShell's stderr semantics.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from pancakebot.ops.sync_health import (  # noqa: E402
    ORPHAN_AFTER_HOURS,
    assess,
    record_attempt,
    record_failure,
    record_success,
    summary_line,
)

def _code_only(path: Path) -> str:
    """Strip the leading <# ... #> comment block.

    The incident notes in these wrappers deliberately QUOTE the constructs
    that caused the failure, so a naive text search finds them in the
    documentation and fails. These tests are about what the script DOES.
    """
    src = path.read_text(encoding="utf-8")
    return re.sub(r"<#.*?#>", "", src, flags=re.DOTALL)


_SYNC_WRAPPER = _REPO_ROOT / "scripts" / "daily_sync.ps1"
_WD_WRAPPER = _REPO_ROOT / "scripts" / "watchdog_run.ps1"


# ---- LAYER 2: the general rule --------------------------------------------

def test_an_orphaned_attempt_is_not_healthy(tmp_path):
    """THE regression, expressed generally. An attempt newer than every
    outcome means a run started and vanished."""
    p = str(tmp_path / "h.json")
    record_success(path=p)
    time.sleep(0.01)
    record_attempt(p)                       # ... and then nothing
    a = assess(p, now=time.time() + (ORPHAN_AFTER_HOURS + 0.1) * 3600)
    assert a["status"] == "ORPHANED", a
    assert a["hours_since_orphaned_attempt"] is not None


def test_the_2026_09_01_health_file_would_have_been_caught(tmp_path):
    """The EXACT file the incident produced -- an attempt at 10:30:01Z, a
    success from the previous evening, consecutive_failures=0 -- must read
    as not-OK at the 13:00Z watchdog run."""
    p = tmp_path / "h.json"
    p.write_text(json.dumps({
        "attempts": 5,
        "consecutive_failures": 0,
        "last_attempt_ts": 1788258601.457329,
        "last_attempt_utc": "2026-09-01T10:30:01Z",
        "last_detail": "scheduled run",
        "last_error": None,
        "last_exit_code": 0,
        "last_success_ts": 1788206103.926293,
        "last_success_utc": "2026-08-31T19:55:03Z",
        "successes": 4,
    }), encoding="utf-8")
    watchdog_time = 1788258601.457329 + 2.5 * 3600     # 13:00Z
    a = assess(str(p), now=watchdog_time)
    assert a["status"] == "ORPHANED", (
        "the incident's own health file still reads healthy")
    assert "never reported an outcome" in a["detail"]


def test_a_run_still_in_flight_is_not_called_orphaned(tmp_path):
    """A sync takes 5-7 minutes. It must not be accused of vanishing while
    it is simply working."""
    p = str(tmp_path / "h.json")
    record_success(path=p)
    time.sleep(0.01)
    record_attempt(p)
    assert assess(p, now=time.time() + 600)["status"] == "OK"   # 10 min in


def test_a_recorded_failure_is_not_an_orphan(tmp_path):
    """Failing loudly is a different, better state than vanishing."""
    p = str(tmp_path / "h.json")
    record_success(path=p)
    time.sleep(0.01)
    record_attempt(p)
    record_failure(exit_code=1, error="boom", path=p)
    a = assess(p, now=time.time() + 10 * 3600)
    assert a["status"] != "ORPHANED"
    assert a["consecutive_failures"] == 1


def test_a_recorded_success_is_not_an_orphan(tmp_path):
    p = str(tmp_path / "h.json")
    record_attempt(p)
    record_success(path=p)
    assert assess(p, now=time.time() + 10 * 3600)["status"] == "OK"


def test_orphaned_does_not_mask_a_more_serious_stale(tmp_path):
    """STALE/DEAD outrank ORPHANED, but the orphan still rides in detail."""
    p = str(tmp_path / "h.json")
    record_success(path=p)
    d = json.loads(Path(p).read_text())
    d["last_success_ts"] = time.time() - 70 * 3600
    Path(p).write_text(json.dumps(d))
    record_attempt(p)
    a = assess(p, now=time.time() + 3 * 3600)
    assert a["status"] == "DEAD"
    assert a["hours_since_orphaned_attempt"] is not None
    assert "never reported an outcome" in a["detail"]


def test_the_summary_line_names_the_orphan(tmp_path):
    p = str(tmp_path / "h.json")
    record_success(path=p)
    time.sleep(0.01)
    record_attempt(p)
    a = assess(p, now=time.time() + 3 * 3600)
    assert "ORPHANED ATTEMPT" in summary_line(a)


def test_the_marker_names_a_vanished_run_not_staleness():
    """Calling this staleness would send the reader to the wrong diagnosis:
    the sync IS firing, it is dying without saying so."""
    import sync_watchdog as W
    a = {"status": "ORPHANED", "hours_since_success": 17.0,
         "hours_since_orphaned_attempt": 2.5}
    assert W._marker_name(a, None) == "PANCAKEBOT_SYNC_RUN_VANISHED_2_HOURS_AGO.txt"


def test_the_orphan_marker_body_says_the_task_is_firing():
    import sync_watchdog as W
    body = W._marker_body({
        "status": "ORPHANED", "hours_since_success": 17.0,
        "hours_since_orphaned_attempt": 2.5,
        "last_success_utc": "x", "consecutive_failures": 0,
        "last_error": None, "detail": "d"}, None)
    assert "STARTED AND VANISHED" in body
    assert "no RESULT line" in body


# ---- LAYER 1: the wrappers survive stderr ---------------------------------

_PS = ["powershell.exe", "-NoProfile", "-NonInteractive",
       "-ExecutionPolicy", "Bypass", "-Command"]


def _run_ps(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(_PS + [script], capture_output=True, text=True,
                          timeout=120)


@pytest.mark.parametrize("pref,expect_survive", [
    ("Continue", True),
    ("Stop", False),
])
def test_the_powershell_semantics_that_caused_the_incident(pref, expect_survive):
    """Pins the actual behaviour, so nobody has to rediscover it. Under
    'Stop', one stderr line from a native command terminates the pipeline."""
    py = str(_REPO_ROOT / ".venv" / "Scripts" / "python.exe")
    script = (
        f"$ErrorActionPreference = '{pref}'; "
        f"try {{ & '{py}' -c \"import sys; print('a'); "
        f"sys.stderr.write('e\\n'); print('b')\" 2>&1 | "
        f"ForEach-Object {{ [string]$_ }} | Out-Null; "
        f"Write-Output 'SURVIVED' }} catch {{ Write-Output 'ABORTED' }}"
    )
    out = _run_ps(script).stdout
    assert ("SURVIVED" in out) is expect_survive, out


def test_both_wrappers_do_not_set_erroractionpreference_stop():
    """THE fix. Either wrapper set to 'Stop' aborts on any stderr line."""
    for w in (_SYNC_WRAPPER, _WD_WRAPPER):
        src = _code_only(w)
        assert "$ErrorActionPreference = 'Stop'" not in src, (
            f"{w.name} sets ErrorActionPreference to Stop -- it will abort "
            f"mid-run the first time anything writes to stderr")
        assert "$ErrorActionPreference = 'Continue'" in src


def test_neither_wrapper_uses_the_all_stream_merge_operator():
    """`*>&1` merges every stream including error records. `2>&1` is what
    is wanted, and only with Continue."""
    for w in (_SYNC_WRAPPER, _WD_WRAPPER):
        src = _code_only(w)
        assert "*>&1" not in src, f"{w.name} still uses *>&1"


def test_the_outcome_is_recorded_in_a_finally_block():
    """No failure mode -- including ones nobody has thought of -- may leave
    the health file reading clean."""
    src = _code_only(_SYNC_WRAPPER)
    assert "finally {" in src
    i_try, i_finally = src.index("try {"), src.index("finally {")
    assert i_try < src.index("run.py --sync") < i_finally
    assert src.index("record_failure") > i_finally, (
        "record_failure must live in the finally block")


def test_the_wrapper_has_a_fallback_exit_code():
    """If the wrapper dies before $LASTEXITCODE is read, the outcome must
    still be recorded rather than silently skipped."""
    src = _code_only(_SYNC_WRAPPER)
    assert "$code = 90" in src and "$code = 91" in src


def test_the_diagnostic_tail_is_passed_by_file_not_interpolated():
    """Log lines are arbitrary text; quoting them into a command line is
    its own failure mode."""
    src = _code_only(_SYNC_WRAPPER)
    assert "pb_sync_tail_" in src
    assert "'''$esc'''" not in src, "the tail is still interpolated into python -c"


# ---- the retry claim -------------------------------------------------------

def test_restartcount_is_not_claimed_as_retry_protection():
    """It was set to 3/PT20M and read like retry. On 2026-09-01 the sync
    exited 1 and NOTHING retried -- RestartCount covers unexpected
    termination, not a non-zero exit. Honest one-shot instead."""
    src = (_REPO_ROOT / "scripts" / "register_sync_tasks.ps1").read_text(
        encoding="utf-8")
    assert "-RestartCount" not in src, (
        "RestartCount is back; it does not retry a non-zero exit")
    assert "retry is TOMORROW" in src.replace("\n", " ").replace("  ", " ") \
        or "retry is TOMORROW'S RUN" in src
