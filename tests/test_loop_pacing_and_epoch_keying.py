"""A safety quantity denominated in ROUNDS must not advance per ITERATION.

THE INCIDENT (2026-08-30). A stale chain read handed _epoch_handshake a
round whose lock had ALREADY passed. It validated lock_ts > 0 and accepted
it. Every wake in _run_one_iteration is an offset before that lock, and
_sleep_until_ts returns immediately for a target already past -- so the
whole wake schedule became a no-op and the outer `while True` free-ran at
~1.4s per iteration against a ~306s round.

The drawdown cooldown is a count of ROUNDS decremented once per pipeline
decision, i.e. once per ITERATION. It fell ~220x real time, hit zero,
extended the suspension by 288 rounds, and repeated -- FIVE spurious
extensions in 27 minutes, adding ~22h to a live-money risk stand-down.

Two more counters had the identical shape and were spared only by which
gate happened to be failing:
  * PoolGateAlarm._streak  -- threshold 6 consecutive blocked rounds is
    reasoned about as ~30 minutes; under the spin it would have tripped in
    ~9 seconds.
  * RateWindow.observe     -- a rate window denominated in rounds; 1,255
    repeats of one round would have flushed it entirely and REPLACED the
    measured rate with the spin's own.

Three defences, tested here: reject the stale round, floor the loop, and
key every one of those counters on the round anchor.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from audit_reader import dedup_by_epoch, spin_report  # noqa: E402
from pancakebot.bankroll_tracker import InMemoryBankrollTracker  # noqa: E402
from pancakebot.runtime import engine  # noqa: E402
from pancakebot.runtime.pool_gate_alarm import PoolGateAlarm, RateWindow  # noqa: E402


def _tracker() -> InMemoryBankrollTracker:
    return InMemoryBankrollTracker(initial_bankroll=2.0,
                                   drawdown_peak_window_days=7)


# ---- (3a) the cooldown ----------------------------------------------------

def test_cooldown_ticks_once_per_round_not_once_per_call():
    """THE regression. 1,000 calls for the SAME round must consume ONE
    round of cooldown, not 1,000."""
    t = _tracker()
    t.set_paused(288, triggered_at=1000)
    for _ in range(1000):
        t.tick_cooldown(1000)
    assert t.cooldown_remaining() == 287, (
        f"cooldown fell to {t.cooldown_remaining()} on repeats of one round "
        f"— a spinning loop can still burn a risk timer")


def test_cooldown_advances_on_each_distinct_round():
    t = _tracker()
    t.set_paused(288, triggered_at=1000)
    for start_at in (1000, 1306, 1612, 1918):
        t.tick_cooldown(start_at)
    assert t.cooldown_remaining() == 284


def test_a_spin_cannot_drive_the_cooldown_to_zero():
    """The concrete harm: reaching zero is what triggered the spurious
    +288 extensions."""
    t = _tracker()
    t.set_paused(5, triggered_at=1000)
    for _ in range(10_000):
        t.tick_cooldown(1000)
    assert t.cooldown_remaining() == 4
    assert t.is_paused(1000) is True


def test_no_anchor_keeps_the_old_unconditional_behaviour():
    """Backtest and tests have no round anchor; they must not change."""
    t = _tracker()
    t.set_paused(10, triggered_at=1000)
    for _ in range(4):
        t.tick_cooldown()
    assert t.cooldown_remaining() == 6


# ---- (3b) the blocked-round streak ---------------------------------------

def test_pool_gate_streak_counts_rounds_not_calls():
    a = PoolGateAlarm(threshold=6, realert_interval_s=3600,
                      kind_blocked="B", kind_recovered="R")
    for _ in range(500):
        a.record(ready=False, reason="pool_uncovered", epoch=100, now=1.0)
    st = a.snapshot() if hasattr(a, "snapshot") else None
    # The streak must not have crossed a 6-round threshold on one round.
    assert a._streak == 1, (
        f"streak reached {a._streak} on repeats of ONE round — a 30-minute "
        f"threshold would trip in seconds under a spinning loop")


def test_pool_gate_streak_advances_across_distinct_rounds():
    a = PoolGateAlarm(threshold=6, realert_interval_s=3600,
                      kind_blocked="B", kind_recovered="R")
    for e in range(100, 106):
        a.record(ready=False, reason="pool_uncovered", epoch=e, now=1.0)
    assert a._streak == 6


# ---- (3c) the rate window -------------------------------------------------

def test_rate_window_folds_one_sample_per_round():
    w = RateWindow(enter_rate=0.3, exit_rate=0.15, window=60, min_samples=10)
    for _ in range(500):
        w.observe(True, epoch=42)
    assert w.n == 1, (
        f"window holds {w.n} samples from ONE round — a spin would flush it "
        f"and replace the measured rate with its own")


def test_rate_window_still_measures_a_real_rate():
    w = RateWindow(enter_rate=0.3, exit_rate=0.15, window=60, min_samples=10)
    for i in range(40):
        w.observe(i % 4 == 0, epoch=1000 + i)        # 25% hit rate
    assert w.n == 40
    assert w.rate == pytest.approx(0.25, abs=1e-9)


def test_rate_window_without_epoch_is_unchanged():
    w = RateWindow(enter_rate=0.3, exit_rate=0.15, window=60, min_samples=5)
    for _ in range(10):
        w.observe(True)
    assert w.n == 10


# ---- (1) the handshake invariant -----------------------------------------

def test_handshake_rejects_an_already_locked_open_round():
    """Source-level: lock_ts > 0 was never sufficient; the lock must also
    be in the FUTURE, or the wake schedule is already stale."""
    src = Path(engine.__file__).read_text(encoding="utf-8")
    assert "open_round_already_locked" in src, (
        "the handshake no longer rejects a stale open round")
    i = src.index("open_round_already_locked")
    window = src[max(0, i - 400):i]
    assert "_utc_now()" in window, (
        "the staleness check must compare the lock against wall time")


# ---- (2) the loop floor ---------------------------------------------------

def test_the_outer_loop_has_a_minimum_iteration_time():
    src = Path(engine.__file__).read_text(encoding="utf-8")
    assert "_MIN_ITERATION_FRACTION_OF_ROUND" in src
    assert "min_iter_s" in src, "no floor applied in the outer loop"
    assert 'warn("LOOP"' in src, (
        "a floor that fires silently gives no more signal than the spin "
        "did — the whole point is that there WAS no signal")


def test_the_floor_is_far_from_both_the_spin_and_a_healthy_round():
    """Derived from data: the spin ran at ~1.4s, a round is ~306s. The
    floor must sit well above the former and well below the latter."""
    frac = engine._MIN_ITERATION_FRACTION_OF_ROUND
    floor_s = 306 * frac
    assert floor_s > 1.4 * 10, f"floor {floor_s}s is too close to the spin"
    assert floor_s < 306 * 0.5, f"floor {floor_s}s could fire in normal use"


# ---- (5) audit consumers --------------------------------------------------

def test_audit_dedup_collapses_a_spin_to_one_row_per_round():
    rows = ([{"locked_epoch": "100", "skip_reason": "a"}]
            + [{"locked_epoch": "101", "skip_reason": "spin"}] * 1255
            + [{"locked_epoch": "102", "skip_reason": "b"}])
    ded = dedup_by_epoch(rows)
    assert [r["locked_epoch"] for r in ded] == ["100", "101", "102"]
    assert spin_report(rows) == [("101", 1255)]


def test_audit_dedup_keeps_the_last_row_for_a_round():
    """The engine overwrites its view of a round as the iteration
    progresses; the final row is the decision that stood."""
    rows = [{"locked_epoch": "7", "skip_reason": "early"},
            {"locked_epoch": "7", "skip_reason": "final"}]
    assert dedup_by_epoch(rows)[0]["skip_reason"] == "final"


def test_audit_dedup_is_a_no_op_on_a_healthy_file():
    rows = [{"locked_epoch": str(e), "skip_reason": "x"} for e in range(50)]
    assert len(dedup_by_epoch(rows)) == 50
    assert spin_report(rows) == []
