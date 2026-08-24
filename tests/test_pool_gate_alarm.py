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

FAR_FROM_LOCK = 300.0   # a fresh round: lock is ~5 minutes out


def _cfg(dry=False, blocks_short=655):
    return types.SimpleNamespace(
        dry=dry,
        rpc_poller=types.SimpleNamespace(
            last_pool_blocks_short=blocks_short, getlogs_p99_ms=250),
    )


def test_recording_on_the_critical_path_sends_nothing(monkeypatch):
    """THE guard: _note_pool_gate_outcome runs ~2.5s before lock, so it must
    do no I/O at all. Even at the threshold the alert is only QUEUED."""
    from pancakebot.runtime import engine

    engine._reset_pool_gate_alarm()
    sent = []
    monkeypatch.setattr(engine, "notify", lambda **kw: sent.append(kw) or "SENT")
    cfg = _cfg()
    for i in range(DEFAULT_THRESHOLD):
        engine._note_pool_gate_outcome(
            cfg, ready=False, reason="pool_uncovered", epoch=500_000 + i)
    assert sent == []
    assert len(engine._PENDING_POOL_GATE_EVENTS) == 1
    engine._reset_pool_gate_alarm()


def test_flush_refuses_inside_the_pre_lock_window(monkeypatch):
    """A queued alert must never be sent close to lock, whoever calls it."""
    from pancakebot.runtime import engine

    engine._reset_pool_gate_alarm()
    sent = []
    monkeypatch.setattr(engine, "notify", lambda **kw: sent.append(kw) or "SENT")
    cfg = _cfg()
    for i in range(DEFAULT_THRESHOLD):
        engine._note_pool_gate_outcome(
            cfg, ready=False, reason="pool_uncovered", epoch=i)

    now = 1_000_000.0
    # every point inside the wake ladder, up to the slack margin itself
    for slack in (0.0, 1.195, 2.5, 7.0, engine._POOL_GATE_ALERT_MIN_SLACK_S - 0.001):
        engine._flush_pool_gate_alerts(cfg, lock_ts=now + slack, now=now)
        assert sent == [], f"alert sent with only {slack}s to lock"
        assert len(engine._PENDING_POOL_GATE_EVENTS) == 1  # still queued

    # ...and delivered once there is ample room
    engine._flush_pool_gate_alerts(cfg, lock_ts=now + FAR_FROM_LOCK, now=now)
    assert len(sent) == 1 and sent[0]["kind"] == KIND_BLOCKED
    assert engine._PENDING_POOL_GATE_EVENTS == []
    engine._reset_pool_gate_alarm()


def test_queued_alert_carries_the_diagnostic_fields(monkeypatch):
    from pancakebot.runtime import engine

    engine._reset_pool_gate_alarm()
    sent = []
    monkeypatch.setattr(engine, "notify", lambda **kw: sent.append(kw) or "SENT")
    cfg = _cfg()
    now = 1_000_000.0
    for i in range(DEFAULT_THRESHOLD):
        engine._note_pool_gate_outcome(
            cfg, ready=False, reason="pool_uncovered", epoch=500_000 + i)
    engine._flush_pool_gate_alerts(cfg, lock_ts=now + FAR_FROM_LOCK, now=now)
    assert sent[0]["mode"] == "live"
    assert sent[0]["fields"]["blocks_short"] == 655
    assert sent[0]["fields"]["getlogs_p99_ms"] == ">=250"

    # recovery is queued the same way, never sent from the critical path
    engine._note_pool_gate_outcome(cfg, ready=True, reason="", epoch=500_100)
    assert len(sent) == 1
    engine._flush_pool_gate_alerts(cfg, lock_ts=now + FAR_FROM_LOCK, now=now)
    assert len(sent) == 2 and sent[1]["kind"] == KIND_RECOVERED
    engine._reset_pool_gate_alarm()


def test_engine_dispatch_swallows_notifier_failure(monkeypatch):
    """A broken notifier must never propagate into the bet path."""
    from pancakebot.runtime import engine

    engine._reset_pool_gate_alarm()

    def boom(**kw):
        raise RuntimeError("discord exploded")

    monkeypatch.setattr(engine, "notify", boom)
    cfg = _cfg(dry=True, blocks_short=None)
    now = 1_000_000.0
    for i in range(DEFAULT_THRESHOLD):
        engine._note_pool_gate_outcome(
            cfg, ready=False, reason="pool_uncovered", epoch=i)  # must not raise
    engine._flush_pool_gate_alerts(cfg, lock_ts=now + FAR_FROM_LOCK, now=now)
    # the failed send is dropped, not retried forever
    assert engine._PENDING_POOL_GATE_EVENTS == []
    engine._reset_pool_gate_alarm()

# ---- kline-gate alarm (second instance) -----------------------------------

def _kcfg(threshold=5):
    return types.SimpleNamespace(
        dry=False, rpc_poller=None,
        max_consecutive_kline_fetch_failures=threshold,
    )


def test_second_instance_uses_its_own_kinds_and_threshold():
    from pancakebot.runtime.pool_gate_alarm import (
        KIND_KLINE_BLOCKED, KIND_KLINE_RECOVERED)
    a = PoolGateAlarm(threshold=3, kind_blocked=KIND_KLINE_BLOCKED,
                      kind_recovered=KIND_KLINE_RECOVERED)
    assert a.record(ready=False, reason="x", epoch=1, now=0.0) is None
    assert a.record(ready=False, reason="x", epoch=2, now=300.0) is None
    ev = a.record(ready=False, reason="x", epoch=3, now=600.0)
    assert ev.kind == KIND_KLINE_BLOCKED
    rec = a.record(ready=True, reason="", epoch=4, now=900.0)
    assert rec.kind == KIND_KLINE_RECOVERED


def _feed(engine, cfg, classes):
    """Feed a sequence of transient classes; return queued events."""
    for i, c in enumerate(classes):
        engine._note_kline_gate_outcome(cfg, transient_class=c, epoch=i)
    return list(engine._PENDING_POOL_GATE_EVENTS)


def test_rate_signal_fires_at_todays_33_percent():
    """Aug 23 ran a 31% daily rate with rolling-60 windows at p50 .300 --
    the regime the run-length alarm was structurally blind to."""
    from pancakebot.runtime import engine
    from pancakebot.runtime.pool_gate_alarm import KIND_KLINE_BLOCKED

    engine._reset_pool_gate_alarm()
    cfg = _kcfg()
    # 1 in 3 rounds is a publish delay -> rate 0.333 > enter 0.30
    seq = [("publish_delay" if i % 3 == 0 else None) for i in range(60)]
    events = _feed(engine, cfg, seq)
    blocked = [e for e in events if e.kind == KIND_KLINE_BLOCKED]
    assert blocked, "33% publish-delay rate must alert"
    ev = blocked[0]
    assert ev.fields["signal"] == "rate"
    assert ev.fields["rate"] >= engine._PUBLISH_RATE_ENTER
    assert ev.fields["window_rounds"] >= engine._RATE_MIN_SAMPLES
    engine._reset_pool_gate_alarm()


def test_rate_signal_is_quiet_at_the_measured_benign_baselines():
    """Aug 21 (7.1%/day, worst window .150) and Aug 22 (11.3%/day, worst
    window .233) must not alert -- entry sits above both."""
    from pancakebot.runtime import engine
    from pancakebot.runtime.pool_gate_alarm import KIND_KLINE_BLOCKED

    for period in (7, 5):        # 1-in-7 = 14%, 1-in-5 = 20%
        engine._reset_pool_gate_alarm()
        seq = [("publish_delay" if i % period == 0 else None) for i in range(120)]
        events = _feed(engine, _kcfg(), seq)
        assert not [e for e in events if e.kind == KIND_KLINE_BLOCKED], \
            f"1-in-{period} must stay under the bar"
    engine._reset_pool_gate_alarm()


def test_window_must_fill_before_any_rate_alert():
    """A cold start must not alarm off a handful of rounds."""
    from pancakebot.runtime import engine
    engine._reset_pool_gate_alarm()
    events = _feed(engine, _kcfg(),
                   ["publish_delay"] * (engine._RATE_MIN_SAMPLES - 1))
    assert events == []
    engine._reset_pool_gate_alarm()


def test_interleaved_genuine_and_tail_raises_BOTH_rates():
    """The silence case: alternating classes left both run-length counters
    pinned at 1 forever while every round was skipped. Rates cannot cancel
    -- every round lands in both denominators."""
    from pancakebot.runtime import engine
    from pancakebot.runtime.pool_gate_alarm import (
        KIND_FETCH_FAILING, KIND_KLINE_BLOCKED)

    engine._reset_pool_gate_alarm()
    seq = [("publish_delay" if i % 2 == 0 else "fetch_failure")
           for i in range(60)]
    events = _feed(engine, _kcfg(), seq)
    kinds = {e.kind for e in events}
    assert KIND_KLINE_BLOCKED in kinds     # publish rate 0.5
    assert KIND_FETCH_FAILING in kinds     # genuine rate 0.5
    engine._reset_pool_gate_alarm()


def test_alternating_sequence_reaches_the_genuine_BURST_threshold():
    """THE neutral-input pin. A publish-delay round carries no evidence
    about genuine fetches, so it must neither advance nor RESET the burst
    streak -- otherwise the mutual-reset bug returns one layer down."""
    from pancakebot.runtime import engine
    from pancakebot.runtime.pool_gate_alarm import KIND_FETCH_FAILING

    engine._reset_pool_gate_alarm()
    cfg = _kcfg(threshold=5)
    # strictly alternating, and only 5 genuine rounds in total
    seq = []
    for _ in range(5):
        seq += ["fetch_failure", "publish_delay"]
    events = _feed(engine, cfg, seq)
    burst = [e for e in events
             if e.kind == KIND_FETCH_FAILING and e.fields.get("signal") == "burst"]
    assert burst, "5 genuine rounds interleaved with tails must still burst"
    assert burst[0].fields["consecutive"] == 5
    engine._reset_pool_gate_alarm()


def test_a_genuinely_healthy_round_does_reset_the_burst_streak():
    """Neutral is not the same as healthy: a clean round still resets."""
    from pancakebot.runtime import engine
    from pancakebot.runtime.pool_gate_alarm import KIND_FETCH_FAILING

    engine._reset_pool_gate_alarm()
    cfg = _kcfg(threshold=5)
    seq = ["fetch_failure"] * 4 + [None] + ["fetch_failure"] * 4
    events = _feed(engine, cfg, seq)
    assert not [e for e in events if e.kind == KIND_FETCH_FAILING]
    engine._reset_pool_gate_alarm()


def test_publish_realerts_six_hourly_and_genuine_hourly():
    """Publish delay is a days-long known condition; hourly would be 24
    messages a day about something already known."""
    from pancakebot.runtime import engine
    engine._reset_pool_gate_alarm()
    (_pw, _gw, pub, gen_rate, burst) = engine._kline_rate_state(_kcfg())
    assert pub.realert_interval_s == 6 * 3600.0
    assert gen_rate.realert_interval_s == 3600.0
    assert burst.realert_interval_s == 3600.0
    engine._reset_pool_gate_alarm()


def test_rate_alarms_are_level_detectors():
    """RateWindow owns the hysteresis, so the alarms fire on the first
    over-threshold round."""
    from pancakebot.runtime import engine
    engine._reset_pool_gate_alarm()
    (_pw, _gw, pub, gen_rate, burst) = engine._kline_rate_state(_kcfg(threshold=5))
    assert pub.threshold == 1 and gen_rate.threshold == 1
    assert burst.threshold == 5
    engine._reset_pool_gate_alarm()


def test_kline_note_never_raises():
    """Runs on the critical path; a failure here must not reach the bet."""
    from pancakebot.runtime import engine
    engine._reset_pool_gate_alarm()
    bad = types.SimpleNamespace(dry=False, rpc_poller=None,
                                max_consecutive_kline_fetch_failures="nope")
    engine._note_kline_gate_outcome(
        bad, transient_class="publish_delay", epoch=1)  # must not raise
    engine._reset_pool_gate_alarm()

# ---- RateWindow ----------------------------------------------------------

def test_rate_window_needs_min_samples_then_reports():
    from pancakebot.runtime.pool_gate_alarm import RateWindow
    w = RateWindow(enter_rate=0.30, exit_rate=0.15, window=60, min_samples=30)
    for _ in range(29):
        assert w.observe(True) is None
    assert w.observe(True) is True
    assert w.n == 30 and w.rate == 1.0


def test_rate_window_hysteresis_does_not_flap():
    """Enters at 0.30, and does NOT clear until the rate falls under 0.15."""
    from pancakebot.runtime.pool_gate_alarm import RateWindow
    w = RateWindow(enter_rate=0.30, exit_rate=0.15, window=10, min_samples=10)
    for _ in range(10):
        w.observe(True)
    assert w.over is True
    # drift down to 0.20 -- above exit, so still alerting
    for _ in range(8):
        w.observe(False)
    assert 0.15 <= w.rate <= 0.25 and w.over is True
    for _ in range(2):
        w.observe(False)
    assert w.rate == 0.0 and w.over is False


def test_rate_window_rejects_inverted_or_oversized_config():
    import pytest as _pytest
    from pancakebot.runtime.pool_gate_alarm import RateWindow
    with _pytest.raises(ValueError):
        RateWindow(enter_rate=0.15, exit_rate=0.30)      # inverted
    with _pytest.raises(ValueError):
        RateWindow(enter_rate=0.30, exit_rate=0.15, window=10, min_samples=20)


def test_rate_window_length_and_warmup_are_independent():
    """W was doubled to 120 to cut sampling variance (sqrt(p(1-p)/W): at
    p=0.22 it drops false entry over the 0.30 bar from 6.7% to 1.7% per
    window), NOT to move the bar -- entry 0.25 would have flagged 28.7% of
    windows in a 22% regime. Warm-up is governed by min_samples, which is
    unchanged, so the signal still goes live after 30 rounds."""
    from pancakebot.runtime import engine
    assert engine._RATE_WINDOW_ROUNDS == 120
    assert engine._RATE_MIN_SAMPLES == 30
    assert engine._PUBLISH_RATE_ENTER == 0.30
    assert engine._PUBLISH_RATE_EXIT == 0.15

    engine._reset_pool_gate_alarm()
    # live after min_samples even though the window holds far more
    seq = ["publish_delay"] * engine._RATE_MIN_SAMPLES
    events = _feed(engine, _kcfg(), seq)
    assert events, "must alert once min_samples is reached, not at full W"
    engine._reset_pool_gate_alarm()


def test_window_rounds_counts_the_round_being_processed():
    """Off-by-one fix: the health line read pub_win.n BEFORE observe(), so
    it excluded the round it was reporting on and showed 0/120(warming)
    immediately after a restart had already seen one round. Reported after
    the observe loop now, so the count includes the current round."""
    from pancakebot.runtime import engine

    engine._reset_pool_gate_alarm()
    seen = []

    class _Poller:
        def set_health_extra(self, **kw):
            seen.append(kw.get("window_rounds"))

    cfg = types.SimpleNamespace(
        dry=False, rpc_poller=_Poller(),
        max_consecutive_kline_fetch_failures=5,
    )
    for i in range(3):
        engine._note_kline_gate_outcome(cfg, transient_class=None, epoch=i)

    assert seen[0].startswith("1/"), seen
    assert seen[1].startswith("2/"), seen
    assert seen[2].startswith("3/"), seen
    assert "(warming)" in seen[0]
