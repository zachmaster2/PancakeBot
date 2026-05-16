"""Unit tests for the absolute-ratchet peak mode in InMemoryBankrollTracker.

Per p2a Refinement #3: ratchet invariant.
For any sequence [b_0, b_1, ..., b_n] of bankroll updates,
the absolute_peak after step k must equal max(b_0, b_1, ..., b_k).
The peak is monotonically non-decreasing across the sequence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.bankroll_tracker import InMemoryBankrollTracker  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


_FAR_FUTURE_START = 2_000_000_000  # past the canonical floor; arbitrary


def _make_tracker_absolute(initial: float = 50.0) -> InMemoryBankrollTracker:
    return InMemoryBankrollTracker(
        initial_bankroll=initial, drawdown_peak_window_days=7, peak_mode="absolute_ratchet",
    )


def _make_tracker_rolling(initial: float = 50.0) -> InMemoryBankrollTracker:
    return InMemoryBankrollTracker(
        initial_bankroll=initial, drawdown_peak_window_days=7, peak_mode="rolling_7d",
    )


# ---------------- ratchet invariant ----------------

def test_initial_peak_equals_initial_bankroll():
    t = _make_tracker_absolute(initial=100.0)
    # Before any settlement, peak should equal initial.
    assert t.peak_bankroll(_FAR_FUTURE_START) == 100.0


def test_ratchet_grows_with_increasing_bankroll():
    t = _make_tracker_absolute(initial=50.0)
    sequence = [50.0, 51.0, 52.0, 53.0, 54.0]
    for i, b in enumerate(sequence):
        t.record_settlement(bankroll=b, start_at=_FAR_FUTURE_START + i * 300)
        expected_peak = max(sequence[: i + 1] + [50.0])
        actual_peak = t.peak_bankroll(_FAR_FUTURE_START + i * 300)
        assert actual_peak == expected_peak, (
            f"step {i}: expected peak {expected_peak}, got {actual_peak}"
        )


def test_ratchet_holds_through_drawdown():
    """Peak ratchets up to high water mark, then stays put through drawdown."""
    t = _make_tracker_absolute(initial=50.0)
    # Climb to 70.
    for i, b in enumerate([55.0, 60.0, 65.0, 70.0]):
        t.record_settlement(bankroll=b, start_at=_FAR_FUTURE_START + i * 300)
    assert t.peak_bankroll(_FAR_FUTURE_START + 1200) == 70.0
    # Now drain to 55.
    for i, b in enumerate([68.0, 65.0, 60.0, 55.0]):
        t.record_settlement(bankroll=b, start_at=_FAR_FUTURE_START + (4 + i) * 300)
    # Peak must still be 70 — does NOT drop with bankroll.
    assert t.peak_bankroll(_FAR_FUTURE_START + 2400) == 70.0


def test_ratchet_invariant_random_sequence():
    """For any update sequence, peak_at(k) = max(b_0..b_k)."""
    import random
    rng = random.Random(42)
    seq = [50.0]
    for _ in range(200):
        # Random walk in [+/- 5].
        seq.append(max(0.5, seq[-1] + rng.uniform(-5.0, 5.0)))
    t = _make_tracker_absolute(initial=50.0)
    running_max = 50.0
    for i, b in enumerate(seq[1:], start=1):  # seq[0] is the init value, skip
        t.record_settlement(bankroll=b, start_at=_FAR_FUTURE_START + i * 300)
        running_max = max(running_max, b)
        actual = t.peak_bankroll(_FAR_FUTURE_START + i * 300)
        assert actual == running_max, (
            f"step {i}: expected running_max {running_max}, got {actual}"
        )


def test_ratchet_handles_dedup():
    """Dedup (record_settlement with same value as last) must not break ratchet."""
    t = _make_tracker_absolute(initial=50.0)
    t.record_settlement(bankroll=60.0, start_at=_FAR_FUTURE_START)
    t.record_settlement(bankroll=60.0, start_at=_FAR_FUTURE_START + 300)  # dedup
    t.record_settlement(bankroll=70.0, start_at=_FAR_FUTURE_START + 600)
    t.record_settlement(bankroll=70.0, start_at=_FAR_FUTURE_START + 900)  # dedup
    assert t.peak_bankroll(_FAR_FUTURE_START + 1200) == 70.0


# ---------------- mode parity ----------------

def test_rolling_mode_unaffected_by_absolute_peak_field():
    """Rolling mode must produce the same peak regardless of the (newly added)
    absolute peak field. Verifies backward compatibility."""
    t = _make_tracker_rolling(initial=50.0)
    # Climb to 70 then drop to 55, all within window.
    seq = [(60.0, 0), (70.0, 300), (65.0, 600), (60.0, 900), (55.0, 1200)]
    for b, off in seq:
        t.record_settlement(bankroll=b, start_at=_FAR_FUTURE_START + off)
    # Within rolling window: peak should be 70 (highest in window).
    assert t.peak_bankroll(_FAR_FUTURE_START + 1200) == 70.0


def test_default_peak_mode_is_rolling():
    """Default peak_mode (no kwarg) must be rolling_7d for backward compat."""
    t = InMemoryBankrollTracker(initial_bankroll=50.0, drawdown_peak_window_days=7)
    # Climb then drop, drop should reduce window peak after entries leave window.
    # We won't simulate that here; just confirm that the mode field defaults right.
    assert getattr(t, "_peak_mode") == "rolling_7d"


# ---------------- validation ----------------

def test_invalid_peak_mode_raises():
    with pytest.raises(InvariantError) as exc_info:
        InMemoryBankrollTracker(
            initial_bankroll=50.0, drawdown_peak_window_days=7, peak_mode="bogus_mode",
        )
    assert "peak_mode_invalid" in str(exc_info.value)


def test_absolute_ratchet_drawdown_above_threshold_on_slow_drain():
    """End-to-end demonstration: slow drain that the rolling-7d misses, the
    absolute-ratchet catches.

    Simulate a slow drain spanning > drawdown_peak_window_days. The rolling-7d peak follows
    the decline downward; the absolute peak remembers the launch high.
    """
    DAY = 86400
    rolling = _make_tracker_rolling(initial=100.0)
    absolute = _make_tracker_absolute(initial=100.0)
    # Linear drain from 100 to 80 over 30 days (one update per day).
    for day in range(31):
        b = 100.0 - day * (20.0 / 30.0)
        ts = _FAR_FUTURE_START + day * DAY
        rolling.record_settlement(bankroll=b, start_at=ts)
        absolute.record_settlement(bankroll=b, start_at=ts)

    final_ts = _FAR_FUTURE_START + 30 * DAY
    rolling_peak = rolling.peak_bankroll(final_ts)
    absolute_peak = absolute.peak_bankroll(final_ts)
    # Rolling-7d peak: the highest bankroll seen in the last 7 days, which is
    # ~7-day-old value somewhere around 84-85 BNB.
    # Absolute peak: 100 (the launch high).
    assert absolute_peak == 100.0
    # rolling peak under slow drain should be MUCH less than 100 (proving the bug).
    assert rolling_peak < 90.0
    # And absolute peak gives a much higher dd_frac than rolling.
    current = 100.0 - 30 * (20.0 / 30.0)  # = 80.0
    rolling_dd = (rolling_peak - current) / rolling_peak
    absolute_dd = (absolute_peak - current) / absolute_peak
    assert absolute_dd > rolling_dd, (
        f"expected absolute_dd > rolling_dd, got {absolute_dd:.4f} vs {rolling_dd:.4f}"
    )
    # Concretely: absolute_dd ≈ 20%, would trigger 15% breaker. rolling_dd ≈ 5-7%, would not.
    assert absolute_dd >= 0.15
    assert rolling_dd < 0.15
