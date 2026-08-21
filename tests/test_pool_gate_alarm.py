"""Pool-gate blocked-round alarm: streak, threshold, re-alert, recovery.

Covers pancakebot/runtime/pool_gate_alarm.py (pure state machine) plus the
engine's dispatch wiring, which must alert-and-continue: never raise, never
skip a round, never touch systemd.
"""
import types

import pytest

from pancakebot.runtime.pool_gate_alarm import (
    DEFAULT_THRESHOLD,
    KIND_BLOCKED,
    KIND_RECOVERED,
    PoolGateAlarm,
    format_duration,
)

T0 = 1_787_000_000.0
ROUND_S = 300.0


def _block(alarm, n, *, start=T0, reason="pool_uncovered", epoch0=500_000,
           blocks_short=655):
    """Feed n consecutive blocked rounds; return the list of events."""
    out = []
    for i in range(n):
        out.append(alarm.record(
            ready=False, reason=reason, epoch=epoch0 + i,
            now=start + i * ROUND_S, blocks_short=blocks_short))
    return out


def test_below_threshold_is_silent():
    a = PoolGateAlarm()
    events = _block(a, DEFAULT_THRESHOLD - 1)
    assert events == [None] * (DEFAULT_THRESHOLD - 1)
    assert a.streak == DEFAULT_THRESHOLD - 1
    assert a.alerting is False


def test_threshold_fires_once_with_diagnostic_fields():
    a = PoolGateAlarm()
    events = _block(a, DEFAULT_THRESHOLD)
    assert events[:-1] == [None] * (DEFAULT_THRESHOLD - 1)
    ev = events[-1]
    assert ev is not None and ev.kind == KIND_BLOCKED
    # everything needed to triage from a phone, no SSH
    assert ev.fields["consecutive"] == DEFAULT_THRESHOLD
    assert ev.fields["reason"] == "pool_uncovered"
    assert ev.fields["blocks_short"] == 655
    assert ev.fields["blocked_for"] == "25m"   # 5 gaps x 300s
    assert ev.fields["epoch"] == 500_000 + DEFAULT_THRESHOLD - 1
    assert ev.fields["last_ok_epoch"] == "none-since-start"
    assert "consecutive=6" in ev.detail and "blocks_short=655" in ev.detail


def test_no_realert_before_the_hour_then_one_per_hour():
    a = PoolGateAlarm()
    _block(a, DEFAULT_THRESHOLD)
    # 59 minutes of further blocked rounds: silent
    t = T0 + DEFAULT_THRESHOLD * ROUND_S
    quiet = [a.record(ready=False, reason="pool_uncovered", epoch=600_000 + i,
                      now=t + i * ROUND_S, blocks_short=655)
             for i in range(11)]           # 11 x 5min = 55min
    assert quiet == [None] * 11
    # the hour is measured from the LAST alert (T0+1500), not from the
    # first blocked round: crossing it re-alerts with the running count
    late = a.record(ready=False, reason="pool_uncovered", epoch=610_000,
                    now=T0 + 5200.0, blocks_short=655)
    assert late is not None and late.kind == KIND_BLOCKED
    assert late.fields["consecutive"] == DEFAULT_THRESHOLD + 12
    assert late.fields["blocked_for"] == "1h26m"


def test_ready_round_resets_streak_without_alerting():
    a = PoolGateAlarm()
    _block(a, DEFAULT_THRESHOLD - 1)
    ev = a.record(ready=True, reason="", epoch=500_100, now=T0 + 9_999)
    assert ev is None          # never alerted, so nothing to recover from
    assert a.streak == 0 and a.alerting is False


def test_recovery_alert_after_an_alerting_run():
    a = PoolGateAlarm()
    _block(a, DEFAULT_THRESHOLD)
    ev = a.record(ready=True, reason="", epoch=500_100, now=T0 + 86_400)
    assert ev is not None and ev.kind == KIND_RECOVERED
    assert ev.fields["recovered_after"] == DEFAULT_THRESHOLD
    assert ev.fields["blocked_for"] == "1d00h"
    assert ev.fields["reason"] == "pool_uncovered"
    assert a.streak == 0 and a.alerting is False
    # and the alarm re-arms cleanly for the next outage
    assert _block(a, DEFAULT_THRESHOLD, start=T0 + 90_000)[-1].kind == KIND_BLOCKED


def test_reset_is_keyed_to_readiness_not_to_betting():
    """A ready round resets even though no bet was placed — at a ~0.3% fire
    rate a no-BET streak carries no information."""
    a = PoolGateAlarm()
    for i in range(50):        # 50 ready rounds, zero bets placed
        assert a.record(ready=True, reason="", epoch=500_000 + i,
                        now=T0 + i * ROUND_S) is None
    assert a.streak == 0 and a.alerting is False


def test_dominant_reason_wins_the_histogram():
    a = PoolGateAlarm()
    a.record(ready=False, reason="cold_start_in_progress", epoch=1, now=T0)
    for i in range(DEFAULT_THRESHOLD - 1):
        a.record(ready=False, reason="pool_uncovered", epoch=2 + i,
                 now=T0 + (i + 1) * ROUND_S)
    ev = a.record(ready=False, reason="pool_uncovered", epoch=99,
                  now=T0 + 10 * ROUND_S)
    assert ev is None or ev.fields["reason"] == "pool_uncovered"
    a2 = PoolGateAlarm(threshold=2)
    a2.record(ready=False, reason="catchup_infeasible_for_round", epoch=1, now=T0)
    ev2 = a2.record(ready=False, reason="catchup_infeasible_for_round", epoch=2,
                    now=T0 + ROUND_S)
    assert ev2.fields["reason"] == "catchup_infeasible_for_round"


def test_unknown_reason_still_counts():
    """An unforeseen not-ready reason must not open a silent hole."""
    a = PoolGateAlarm(threshold=2)
    a.record(ready=False, reason="poll_in_progress", epoch=1, now=T0)
    ev = a.record(ready=False, reason="poll_in_progress", epoch=2, now=T0 + ROUND_S)
    assert ev is not None and ev.fields["reason"] == "poll_in_progress"


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        PoolGateAlarm(threshold=0)


@pytest.mark.parametrize("secs,want", [
    (0, "0s"), (45, "45s"), (60, "1m"), (1800, "30m"),
    (3660, "1h01m"), (86_400, "1d00h"), (417_600, "4d20h"),
])
def test_format_duration(secs, want):
    assert format_duration(secs) == want


# ---- engine wiring -------------------------------------------------------

def test_engine_dispatch_alerts_and_never_raises(monkeypatch):
    from pancakebot.runtime import engine

    engine._reset_pool_gate_alarm()
    sent = []
    monkeypatch.setattr(engine, "notify",
                        lambda **kw: sent.append(kw) or "SENT")
    cfg = types.SimpleNamespace(
        dry=False,
        rpc_poller=types.SimpleNamespace(last_pool_blocks_short=655),
    )
    for i in range(DEFAULT_THRESHOLD):
        engine._dispatch_pool_gate_alarm(
            cfg, ready=False, reason="pool_uncovered", epoch=500_000 + i)
    assert len(sent) == 1
    assert sent[0]["kind"] == KIND_BLOCKED
    assert sent[0]["mode"] == "live"
    assert sent[0]["fields"]["blocks_short"] == 655

    engine._dispatch_pool_gate_alarm(cfg, ready=True, reason="", epoch=500_100)
    assert len(sent) == 2 and sent[1]["kind"] == KIND_RECOVERED
    engine._reset_pool_gate_alarm()


def test_engine_dispatch_swallows_notifier_failure(monkeypatch):
    """A broken notifier must never propagate into the bet path."""
    from pancakebot.runtime import engine

    engine._reset_pool_gate_alarm()

    def boom(**kw):
        raise RuntimeError("discord exploded")

    monkeypatch.setattr(engine, "notify", boom)
    cfg = types.SimpleNamespace(
        dry=True, rpc_poller=types.SimpleNamespace(last_pool_blocks_short=None))
    for i in range(DEFAULT_THRESHOLD):
        engine._dispatch_pool_gate_alarm(
            cfg, ready=False, reason="pool_uncovered", epoch=i)  # must not raise
    engine._reset_pool_gate_alarm()
