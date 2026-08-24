"""ENDPOINT_MOVE_TRIGGER: one alarm, two independent detectors.

SCOPE OF THESE TESTS, stated because the distinction matters. They drive
the DETECTOR against synthetic sequences at stated hit-rates: given a
sustained share of X, does the window cross the bar. They do NOT validate
the THRESHOLDS against the real Aug 20-24 series — that rests on an
offline sweep which is not in this repo, so "0 of 158 windows" and "fires
on Aug 20" are not falsifiable here.

That distinction is not pedantry: this repo shipped a replay table on
2026-08-22 whose n and trailing-gap columns claimed to be measured from
the canonical stream and turned out not to reproduce. Treat any
day-level claim below as a label on the fixture, not as evidence.

The daily shares are recorded as the SOURCE of the chosen bar (0.15 sits
between the benign 0.056 and the degraded 0.161-0.212), which the tests
do check. Whether a real day's per-round sequence crosses it is a
different question and needs the real series.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from pancakebot.runtime import engine
from pancakebot.runtime.pool_gate_alarm import (
    KIND_ENDPOINT_MOVE_CLEARED,
    KIND_ENDPOINT_MOVE_TRIGGERED,
)

# Real daily static-wake shares, 2026-08-20..24. These are DAILY
# AVERAGES from the offline sweep; the per-round series they came from is
# not in the repo, so tests below feed them as sustained rates.
DAILY_STATIC_SHARE = {
    "2026-08-20": 0.074,
    "2026-08-21": 0.212,
    "2026-08-22": 0.198,
    "2026-08-23": 0.161,
    "2026-08-24": 0.056,
}


class _Poller:
    def __init__(self):
        self.rs_block_error_count = 0

    def fail(self, n=1):
        self.rs_block_error_count += n


def _cfg(poller=None):
    return types.SimpleNamespace(
        dry=False, rpc_poller=poller if poller is not None else _Poller(),
        max_consecutive_kline_fetch_failures=5,
    )


def _feed(cfg, *, statics, headers=None, pool_uncovered=None, start=1000):
    """Drive N rounds. `statics` is a list of bools (wake_mode static?);
    `headers` a list of bools (did the header call fail this round?)."""
    n = len(statics)

    def _fit(seq):
        seq = list(seq or [])
        return (seq + [False] * n)[:n]

    headers = _fit(headers)
    pool_uncovered = _fit(pool_uncovered)
    for i in range(n):
        if headers[i]:
            cfg.rpc_poller.fail()
        engine._note_endpoint_move_outcome(
            cfg,
            wake_mode="static" if statics[i] else "dynamic",
            pool_ready=not pool_uncovered[i],
            epoch=start + i,
        )
    return list(engine._PENDING_POOL_GATE_EVENTS)


def _pattern(share, n=160):
    """A round sequence whose static share is `share`, EVENLY SPREAD.

    Bresenham rather than a front-loaded block: a leading run of hits
    would push the trailing window over the bar transiently and make a
    benign day look like a firing one, which is a property of the fixture,
    not of the detector."""
    return [int((i + 1) * share) > int(i * share) for i in range(n)]


def _triggered(events):
    return [e for e in events if e.kind == KIND_ENDPOINT_MOVE_TRIGGERED]


def setup_function():
    engine._reset_pool_gate_alarm()


# ---- Trigger A, against the real series ---------------------------------

@pytest.mark.parametrize("day", ["2026-08-21", "2026-08-22", "2026-08-23"])
def test_a_sustained_degraded_day_share_crosses_the_bar(day):
    """SYNTHETIC: feeds each degraded day's DAILY share as a sustained
    per-round rate. Establishes that the bar sits below those shares and
    that the window crosses it — not that the real per-round sequence for
    that date did."""
    share = DAILY_STATIC_SHARE[day]
    assert share > engine._ENDPOINT_STATIC_ENTER
    events = _feed(_cfg(), statics=_pattern(share))
    assert _triggered(events), f"{day} (share {share}) must trigger"


def test_a_sustained_benign_day_share_does_not_cross_the_bar():
    """SYNTHETIC, same caveat: Aug 24's 0.056 fed as a sustained rate stays
    under the 0.15 bar with a 2.7x margin. The offline sweep's stronger
    claim — 0 of 158 real windows — is NOT checked here."""
    events = _feed(_cfg(), statics=_pattern(DAILY_STATIC_SHARE["2026-08-24"]))
    assert not _triggered(events)


def test_an_intraday_burst_triggers_even_when_the_daily_average_is_low():
    """Why day-one detection is possible at all. Aug 20's DAILY average
    was 0.074, under the bar — the offline sweep reports it firing because
    the rolling window sees an intra-day burst. This test feeds 0.20
    directly and therefore demonstrates only the MECHANISM (a rolling
    window fires on a burst a daily average would hide), not Aug 20."""
    assert DAILY_STATIC_SHARE["2026-08-20"] < engine._ENDPOINT_STATIC_ENTER
    events = _feed(_cfg(), statics=_pattern(0.20))
    assert _triggered(events)


def test_hourly_buckets_would_have_been_the_wrong_instrument():
    """At ~12 rounds/h an hourly share quantises to multiples of 1/12, so
    every threshold in (0.084, 0.166) is the same rule. The rolling window
    resolves shares inside that band — 0.10 and 0.14 must be
    distinguishable from 0.20, which hourly buckets cannot do."""
    quantised = {round(k / 12.0, 4) for k in range(13)}
    assert not any(0.084 < q < 0.166 for q in quantised), quantised
    assert engine._ENDPOINT_WINDOW_ROUNDS == 72        # ~6h, not 12
    engine._reset_pool_gate_alarm()
    assert not _triggered(_feed(_cfg(), statics=_pattern(0.10)))
    engine._reset_pool_gate_alarm()
    assert _triggered(_feed(_cfg(), statics=_pattern(0.20)))


# ---- Trigger B, independent of A ----------------------------------------

def test_trigger_b_fires_on_header_failures_with_no_static_wakes():
    """The decoupling case: a degradation that surfaces as outright RPC
    failure without a timing fallback is caught by exactly this one."""
    n = 80
    headers = [(i % 10) == 0 for i in range(n)]        # 10% > 0.05 bar
    events = _feed(_cfg(), statics=[False] * n, headers=headers)
    trig = _triggered(events)
    assert trig, "header-rate detector must fire on its own"
    assert trig[0].fields["trigger"] == "header_failure_rate"


def test_a_low_header_rate_does_not_fire():
    n = 80
    headers = [(i % 50) == 0 for i in range(n)]        # 2% < 0.05
    assert not _triggered(_feed(_cfg(), statics=[False] * n, headers=headers))


def test_the_first_sample_after_restart_cannot_manufacture_a_hit():
    """The counter is monotonic and survives a restart of this process's
    view; the first observation must only establish the baseline."""
    poller = _Poller()
    poller.fail(500)                       # a long history of failures
    cfg = _cfg(poller)
    events = _feed(cfg, statics=[False] * 80)
    assert not _triggered(events)


# ---- the OR, and both metrics on every alert ----------------------------

def test_either_detector_alone_raises_the_same_single_alarm():
    engine._reset_pool_gate_alarm()
    a = _triggered(_feed(_cfg(), statics=_pattern(0.20)))
    engine._reset_pool_gate_alarm()
    n = 80
    b = _triggered(_feed(_cfg(), statics=[False] * n,
                         headers=[(i % 10) == 0 for i in range(n)]))
    assert a and b
    assert a[0].kind == b[0].kind == KIND_ENDPOINT_MOVE_TRIGGERED


def test_both_metrics_are_reported_whichever_fired():
    """One number without the other does not support a migrate/don't
    decision."""
    trig = _triggered(_feed(_cfg(), statics=_pattern(0.20)))
    extra = trig[0].fields
    assert extra["trigger"] == "static_wake_share"
    assert "static_wake_share" in extra and "header_failure_rate" in extra
    assert extra["header_failure_rate"] == 0.0


def test_both_firing_together_is_named_as_such():
    n = 80
    trig = _triggered(_feed(_cfg(), statics=_pattern(0.20),
                            headers=[(i % 10) == 0 for i in range(n)]))
    assert trig[0].fields["trigger"] == "static_wake_share+header_failure_rate"


# ---- pool_uncovered: reported, never a trigger --------------------------

def test_pool_uncovered_never_triggers_on_its_own():
    """It expresses HARM, not mechanism. A harm-based trigger fires later
    by construction — the wrong direction for an alarm whose purpose is to
    open a diagnostic window while the fault is still active."""
    n = 80
    events = _feed(_cfg(), statics=[False] * n, pool_uncovered=[True] * n)
    assert not _triggered(events)


def test_pool_uncovered_is_reported_in_the_body():
    n = 80
    trig = _triggered(_feed(_cfg(), statics=_pattern(0.20),
                            pool_uncovered=[(i % 2) == 0 for i in range(n)]))
    assert trig[0].fields["pool_uncovered_rate"] > 0.4


def test_the_call_site_sits_before_the_pool_gate_early_return():
    """THE REAL PLACEMENT GUARD (MED-1).

    The behavioural test below feeds the detector directly, so it passes
    no matter where the production call site is — the reviewer moved the
    call to the kline dispatch and NOTHING failed. Placement is the most
    consequential decision in this change, so it is asserted at the
    source: the call must sit AFTER _note_pool_gate_outcome and BEFORE
    the `if not ready:` block that returns out of the round.
    """
    src = Path(engine.__file__).read_text(encoding="utf-8")
    # anchor on the CALL arguments, not the bare name: the function
    # definitions appear earlier in the file and would match first.
    i_pool = src.index("cfg, ready=ready, reason=ready_reason")
    i_ours = src.index("cfg, wake_mode=wake_mode, pool_ready=ready")
    i_ret = src.index("if not ready:", i_pool)
    assert i_pool < i_ours < i_ret, (
        "endpoint detector must be called between the pool-gate outcome "
        "and the early return; a pool-gate skip returns before the kline "
        "dispatch, so sampling there goes blind exactly when the fault "
        "is worst")


def test_pool_blocked_rounds_stay_in_the_denominator():
    """Behavioural half of the placement rule: pool-blocked rounds must
    reach the detector. Guards the CONSEQUENCE; the source assertion
    above guards the placement itself."""
    n = 80
    trig = _triggered(_feed(_cfg(), statics=_pattern(0.20),
                            pool_uncovered=[True] * n))
    assert trig, "pool-blocked rounds must still reach the detector"
    assert trig[0].fields["pool_uncovered_rate"] == 1.0
    # fires as soon as the window reaches min_samples, so the count at
    # first fire is the warm-up size, not the full window
    assert trig[0].fields["window_rounds"] >= engine._ENDPOINT_MIN_SAMPLES


# ---- warm-up, hysteresis, cadence, kinds --------------------------------

def test_silent_while_the_window_is_warming():
    """min_samples encodes the '3 consecutive hours' intent."""
    assert engine._ENDPOINT_MIN_SAMPLES == 36
    events = _feed(_cfg(), statics=[True] * (engine._ENDPOINT_MIN_SAMPLES - 1))
    assert not _triggered(events)


def test_hysteresis_holds_the_alarm_between_the_bars():
    assert engine._ENDPOINT_STATIC_EXIT < engine._ENDPOINT_STATIC_ENTER
    assert engine._ENDPOINT_HEADER_EXIT < engine._ENDPOINT_HEADER_ENTER
    cfg = _cfg()
    _feed(cfg, statics=_pattern(0.30))
    del engine._PENDING_POOL_GATE_EVENTS[:]
    # drop to 0.10: below ENTER but above EXIT -> must NOT clear yet
    _feed(cfg, statics=_pattern(0.10), start=5000)
    assert not [e for e in engine._PENDING_POOL_GATE_EVENTS
                if e.kind == KIND_ENDPOINT_MOVE_CLEARED]


def test_clearing_requires_dropping_under_the_exit_bar():
    cfg = _cfg()
    _feed(cfg, statics=_pattern(0.30))
    del engine._PENDING_POOL_GATE_EVENTS[:]
    _feed(cfg, statics=[False] * 80, start=5000)
    cleared = [e for e in engine._PENDING_POOL_GATE_EVENTS
               if e.kind == KIND_ENDPOINT_MOVE_CLEARED]
    assert cleared


def test_cadence_is_slower_than_the_kline_alarm():
    """A days-long condition whose response is a planned migration; nobody
    executes an endpoint move at 03:00."""
    assert engine._ENDPOINT_REALERT_S == 12 * 3600.0
    assert engine._ENDPOINT_REALERT_S > engine._PUBLISH_REALERT_S


def test_kinds_are_distinct_from_the_pool_and_kline_alarms():
    from pancakebot.runtime import pool_gate_alarm as pga
    assert KIND_ENDPOINT_MOVE_TRIGGERED not in (
        pga.KIND_BLOCKED, pga.KIND_KLINE_BLOCKED, pga.KIND_FETCH_FAILING)
    assert KIND_ENDPOINT_MOVE_CLEARED not in (
        pga.KIND_RECOVERED, pga.KIND_KLINE_RECOVERED, pga.KIND_FETCH_RECOVERED)


# ---- the alert must carry the discriminator -----------------------------

def test_the_alert_body_carries_the_discriminator_instruction():
    """THE failure this alarm must not have: firing without telling the
    reader to run the test while the window is open. That is what happened
    between 2026-08-23 and 08-24."""
    from pancakebot.ops.notifications import build_message
    msg = build_message(mode="live", kind=KIND_ENDPOINT_MOVE_TRIGGERED,
                        fields={"trigger": "static_wake_share",
                                "static_wake_share": 0.161,
                                "header_failure_rate": 0.0,
                                "pool_uncovered_rate": 0.08})
    assert "DISCRIMINATOR" in msg
    assert "eth_getBlockByNumber" in msg
    assert "ONLY WORKS WHILE THIS IS ACTIVE" in msg
    # both metrics rendered
    assert "static_wake_share" in msg and "header_failure_rate" in msg


def test_the_alert_carries_the_banked_constants_and_their_caveats():
    """The decision must not require finding the report. Measured values,
    the derived-vs-current verdict, the gates already passed, and the two
    caveats that separate 'ready to decide' from 'ready to execute'."""
    from pancakebot.ops.notifications import build_message
    msg = build_message(mode="live", kind=KIND_ENDPOINT_MOVE_TRIGGERED,
                        fields={"trigger": "static_wake_share"})

    # measured latency, both hosts
    assert "BANKED LATENCY" in msg and "n=35" in msg
    assert "4/5/15/15" in msg and "19/29/52/52" in msg

    # the verdict that actually drives the change
    assert "NO CONSTANT NEEDS TO CHANGE" in msg
    for derived in ("182", "114", "236"):
        assert derived in msg, derived
    assert "KEEP the current values" in msg
    assert "wake_mode=static risk" in msg

    # gates already passed
    assert "zero 429s" in msg
    assert "200/200 byte-identical" in msg
    assert "NEVER behind" in msg

    # the two caveats must travel with the numbers
    assert "ready to DECIDE, not ready to" in msg
    assert "re-soak" in msg
    assert "BURST tolerance, not multi-day sustained" in msg
    assert "DOUBLES sustained load" in msg
    assert "403" in msg

    # and the thing nobody should argue the move on
    assert "DO NOT LEAN ON THE SECOND-ORDER BENEFIT" in msg
    assert "0.6-1.4pp" in msg
    assert "33.3% -> 29.2%" in msg

    # the earlier getLogs measurement is kept but clearly separated
    assert "SUPPORTING (separate, earlier measurement" in msg
    assert "2,865ms" in msg


def test_cleared_says_the_window_closed_not_that_things_are_fine():
    from pancakebot.ops.notifications import build_message
    msg = build_message(mode="live", kind=KIND_ENDPOINT_MOVE_CLEARED,
                        fields={"trigger": "none"})
    assert "WINDOW HAS CLOSED" in msg
    assert "CIRCUMSTANTIAL" in msg
    assert "fine now" in msg          # explicitly disclaimed


def test_recording_sends_nothing_on_the_critical_path():
    """Queued only — no Discord POST before the lock."""
    posted = []
    import pancakebot.ops.notifications as notifications

    # L3: patch BOTH bindings. The engine imports `notify` at module load,
    # so patching only the notifications module cannot intercept it and
    # the test would pass without proving anything.
    patched = []
    for mod, name in ((notifications, "notify"), (engine, "notify")):
        if hasattr(mod, name):
            patched.append((mod, name, getattr(mod, name)))
            setattr(mod, name, lambda **kw: posted.append(kw))
    assert patched, "no notify binding found to patch"
    try:
        _feed(_cfg(), statics=_pattern(0.30))
    finally:
        for mod, name, orig in patched:
            setattr(mod, name, orig)
    assert posted == []
    assert engine._PENDING_POOL_GATE_EVENTS, "event must be QUEUED"
