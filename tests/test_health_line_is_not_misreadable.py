"""A number printed beside a sample count must not be readable as a rate.

On 2026-08-25 the getLogs health line read

    getlogs health: p99=>=122ms samples=200 censored=21 errors=30 ...

and was reported up the chain as a 15% error rate. It was not: `samples`
was the fixed size of a latency ring, while `censored`/`errors` were
MONOTONIC counters since process start. The true rate was 30 / ~8,500
calls = 0.35%. Worse, the counters restarting at zero across a deploy
restart were read as "recovered, then degrading again" -- two process
lifetimes mistaken for two episodes of a real condition.

Both errors are properties of the RENDERED LINE, so that is what these
test. The rule: every count on the line is either
  * windowed to a denominator printed beside it (suffix ``_win``), or
  * named to disclaim being a rate (suffix ``_life``), or
  * itself a denominator / non-count field.
"""
from __future__ import annotations

import collections
import re
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain import rpc_poller as rp  # noqa: E402

# Fields that are denominators or not counts at all.
_NOT_A_COUNT = {"p99", "lat_n", "calls_win", "uptime", "head_fetch", "host",
                "window_rounds"}
_FIELD_RE = re.compile(r"(\w+)=")


def _render(*, outcomes, cens_life, err_life, uptime_s, lat=200):
    p = object.__new__(rp.RpcPoller)
    p._lock = threading.Lock()
    p._getlogs_latency_ms = collections.deque([50.0] * lat, maxlen=200)
    p._getlogs_censored_total = cens_life
    p._getlogs_errors_total = err_life
    p._getlogs_outcomes = collections.deque(outcomes, maxlen=200)
    p._proc_start_monotonic = time.monotonic() - uptime_s
    p._last_head_fetch_error = None
    p._health_extra = {}
    out: list[str] = []
    with patch.object(rp, "info", lambda tag, msg: out.append(msg)):
        p._log_getlogs_health()
    return out[-1]


def _fields(line: str) -> set[str]:
    return set(_FIELD_RE.findall(line.split(":", 1)[1]))


# ---- the naming rule ------------------------------------------------------

def test_every_count_is_windowed_or_disclaims_being_a_rate():
    """THE regression. A bare `errors=30` beside `samples=200` is exactly
    what got misread, so the line must not be able to contain one."""
    line = _render(outcomes=["ok"] * 200, cens_life=21, err_life=30,
                   uptime_s=66300)
    offenders = [f for f in _fields(line)
                 if f not in _NOT_A_COUNT
                 and not f.endswith("_win") and not f.endswith("_life")]
    assert not offenders, (
        f"{offenders} are counts printed beside a sample count with no "
        f"suffix saying whether they are windowed (_win) or monotonic "
        f"(_life); a reader will take them for a rate. Line: {line}")


@pytest.mark.parametrize("banned", ["censored=", "errors=", "samples="])
def test_the_exact_misread_field_names_are_gone(banned):
    """The old names, verbatim. `samples=200 censored=21 errors=30` must
    not be reconstructible."""
    line = _render(outcomes=["ok"] * 200, cens_life=21, err_life=30,
                   uptime_s=66300)
    assert banned not in line, f"{banned!r} is back in: {line}"


# ---- the windowed numbers are real rates ---------------------------------

def test_windowed_counts_are_rates_over_the_printed_denominator():
    outcomes = ["ok"] * 190 + ["error"] * 8 + ["censored"] * 2
    line = _render(outcomes=outcomes, cens_life=99, err_life=99,
                   uptime_s=3600)
    assert "calls_win=200" in line
    assert "err_win=8(4.0%)" in line
    assert "cens_win=2(1.0%)" in line


def test_the_window_denominator_is_attempts_not_the_latency_ring():
    """They are genuinely different populations: a fast RPC rejection
    increments errors but appends NO latency sample, so windowing errors
    to the latency ring would have been dishonest rather than merely
    imprecise."""
    line = _render(outcomes=["error"] * 10, cens_life=0, err_life=10,
                   uptime_s=60, lat=200)
    assert "lat_n=200" in line          # ring is full
    assert "calls_win=10" in line       # but only 10 attempts observed
    assert "err_win=10(100.0%)" in line


def test_an_empty_window_reports_na_not_a_fake_zero():
    """Before any attempt is observed, there is no rate to report, and
    `0.0%` would read as 'healthy' rather than 'unknown'."""
    line = _render(outcomes=[], cens_life=0, err_life=0, uptime_s=5)
    assert "calls_win=0" in line
    assert "err_win=0(n/a)" in line


# ---- restart cannot be mistaken for a trend ------------------------------

def test_two_lines_from_different_lifetimes_are_distinguishable():
    """The second half of the misdiagnosis: counters restarting at zero
    across a deploy were read as recovery followed by fresh degradation.
    Uptime makes that impossible to infer."""
    before = _render(outcomes=["ok"] * 200, cens_life=18, err_life=24,
                     uptime_s=66300)
    after = _render(outcomes=["ok"] * 5, cens_life=0, err_life=0,
                    uptime_s=61)
    assert "uptime=18h25m" in before
    assert "uptime=0h01m" in after
    # a reader comparing the two sees the process restarted, so the drop
    # in *_life is explained rather than looking like recovery
    assert "err_life=24" in before and "err_life=0" in after


def test_uptime_is_always_present():
    for secs in (0, 61, 3600, 66300, 400000):
        assert "uptime=" in _render(outcomes=["ok"], cens_life=0,
                                    err_life=0, uptime_s=secs)


# ---- the header path, checked rather than assumed ------------------------

def test_no_other_health_line_prints_a_bare_monotonic_counter():
    """The brief asked whether header-path failures share this shape. They
    do not: `_last_rs_block_error` is a CAUSE STRING logged inline on the
    failure, and `rs_block_error_count` (added for ENDPOINT_MOVE_TRIGGER)
    is consumed by an alarm, never rendered. Asserted so that changing
    either is a deliberate act."""
    src = Path(rp.__file__).read_text(encoding="utf-8")
    info_calls = re.findall(r'info\(\s*\n?\s*"POLL",(.{0,900}?)\)\n', src,
                            re.S)
    assert info_calls, "no POLL log lines found — the emitter moved"
    for body in info_calls:
        for field in _FIELD_RE.findall(body):
            assert (field in _NOT_A_COUNT
                    or field.endswith("_win") or field.endswith("_life")), (
                f"POLL line renders {field}= with no _win/_life suffix")
