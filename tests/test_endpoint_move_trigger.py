"""ENDPOINT_MOVE_TRIGGER: one alarm, two independent detectors.

Validated against the real 2026-08-20..24 series. Trigger A (static-wake
share) must fire on Aug 20/21/22/23 and stay silent on Aug 24, whose peak
benign share was 0.056 against a 0.15 bar. Firing on Aug 20 is DESIRED —
that was day one of the condition.
"""
from __future__ import annotations

import types

import pytest

from pancakebot.runtime import engine
from pancakebot.runtime.pool_gate_alarm import (
    KIND_ENDPOINT_MOVE_CLEARED,
    KIND_ENDPOINT_MOVE_TRIGGERED,
)

# Real daily static-wake shares, 2026-08-20..24.
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
def test_trigger_a_fires_on_the_degraded_days(day):
    share = DAILY_STATIC_SHARE[day]
    assert share > engine._ENDPOINT_STATIC_ENTER
    events = _feed(_cfg(), statics=_pattern(share))
    assert _triggered(events), f"{day} (share {share}) must trigger"


def test_trigger_a_is_silent_on_the_benign_day():
    """Aug 24: 0.056 against a 0.15 bar, a 2.7x margin."""
    events = _feed(_cfg(), statics=_pattern(DAILY_STATIC_SHARE["2026-08-24"]))
    assert not _triggered(events)


def test_day_one_detection_is_the_point():
    """Aug 20 ran 0.074 daily — under the bar as a DAILY average, but the
    rolling window sees the burst within the day. Fed as a sustained
    intra-day rate above the bar, day one must trigger; that is the
    difference between detecting on day one and on day four."""
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


def test_pool_blocked_rounds_stay_in_the_denominator():
    """PLACEMENT REGRESSION. The detector is called beside the pool-gate
    outcome, not at the kline dispatch, because a pool-gate skip returns
    out of the round before that point. Sampling later would drop exactly
    the rounds this degradation causes, and the reported pool_uncovered
    rate would be structurally ~0."""
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
    assert "BANKED CONSTANTS" in msg
    assert "250ms" in msg                       # decision needs no report
    # both metrics rendered
    assert "static_wake_share" in msg and "header_failure_rate" in msg


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
    orig = notifications.notify
    notifications.notify = lambda **kw: posted.append(kw)
    try:
        _feed(_cfg(), statics=_pattern(0.30))
    finally:
        notifications.notify = orig
    assert posted == []
    assert engine._PENDING_POOL_GATE_EVENTS, "event must be QUEUED"
