"""Tests for ``engine._sleep_until_ts`` and the dynamic-wake divergence alert.

``_sleep_until_ts`` must honor arbitrarily-small future sleeps: it has no
minimum-sleep short-circuit, so a sub-500ms dynamic critical-path wake is
actually slept (it was previously skipped by a ``remaining <= 0.5`` guard,
which silently bypassed the dynamic timing optimization every round).

``_wake_divergence_alert_message`` surfaces the bypass: when the actual
fetch-fire offset exceeds the computed dynamic-wake offset beyond the
tolerance, it returns an ALERT prose string (else ``None``).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime import engine  # noqa: E402
from pancakebot.runtime.engine import _WAKE_DIVERGENCE_ALERT_TOLERANCE_MS  # noqa: E402


# --------------------------------------------------------------------------
# _sleep_until_ts
# --------------------------------------------------------------------------


def test_sleep_until_ts_honors_short_sleeps():
    """A 100ms future target is actually slept, not short-circuited."""
    # _utc_now: guard check at t0, loop check 1 at t0 (still 0.1 remaining),
    # loop check 2 at t0+0.1 (remaining 0 -> return).
    clock = iter([100.0, 100.0, 100.1])
    with mock.patch.object(engine, "_utc_now", side_effect=lambda: next(clock)), \
            mock.patch.object(engine, "sleep_seconds") as m_sleep:
        engine._sleep_until_ts(100.1, reason="t", epoch=1)
    assert m_sleep.call_count == 1
    # slept min(1.0, 0.1) ~= 0.1
    (slept,), _ = m_sleep.call_args
    assert abs(slept - 0.1) < 1e-9


def test_sleep_until_ts_honors_sub_500ms_target():
    """THE bug-fix contract: a 0.3s future target (inside the old 0.5s guard)
    must NOT be skipped — it is slept until."""
    clock = iter([10.0, 10.0, 10.3])
    with mock.patch.object(engine, "_utc_now", side_effect=lambda: next(clock)), \
            mock.patch.object(engine, "sleep_seconds") as m_sleep:
        engine._sleep_until_ts(10.3, reason="wait_for_critical_path", epoch=2)
    assert m_sleep.call_count == 1, "sub-500ms target was bypassed (regression)"
    (slept,), _ = m_sleep.call_args
    assert abs(slept - 0.3) < 1e-9


def test_sleep_until_ts_returns_on_past_target():
    """A target already in the past returns immediately, never sleeps."""
    with mock.patch.object(engine, "_utc_now", return_value=500.0), \
            mock.patch.object(engine, "sleep_seconds") as m_sleep:
        engine._sleep_until_ts(499.0, reason="t", epoch=3)
    assert m_sleep.call_count == 0


def test_sleep_until_ts_returns_on_exact_target():
    """remaining == 0 returns immediately (boundary of the > 0 guard)."""
    with mock.patch.object(engine, "_utc_now", return_value=500.0), \
            mock.patch.object(engine, "sleep_seconds") as m_sleep:
        engine._sleep_until_ts(500.0, reason="t", epoch=4)
    assert m_sleep.call_count == 0


# --------------------------------------------------------------------------
# _wake_divergence_alert_message
# --------------------------------------------------------------------------


def test_divergence_alert_fires_on_bypass():
    """Regime B: actual fire ~360ms earlier than computed -> ALERT prose."""
    msg = engine._wake_divergence_alert_message(
        actual_offset_ms=1236.8, computed_offset_ms=877.0,
    )
    assert msg is not None
    assert "DYNAMIC_WAKE_BYPASS" in msg
    assert "divergence_ms=360" in msg
    assert "wake_target_offset=877" in msg
    assert "actual_offset=1237" in msg
    assert "reason=sleep_threshold_or_late_arrival" in msg


def test_divergence_alert_silent_when_aligned():
    """Regime A: actual ~= computed (within a few ms) -> no alert."""
    msg = engine._wake_divergence_alert_message(
        actual_offset_ms=880.0, computed_offset_ms=877.0,
    )
    assert msg is None


def test_divergence_alert_silent_at_tolerance_boundary():
    """Exactly at the tolerance -> no alert (only strictly beyond fires)."""
    msg = engine._wake_divergence_alert_message(
        actual_offset_ms=877.0 + _WAKE_DIVERGENCE_ALERT_TOLERANCE_MS,
        computed_offset_ms=877.0,
    )
    assert msg is None


def test_divergence_alert_fires_just_beyond_tolerance():
    """One ms beyond the tolerance fires."""
    msg = engine._wake_divergence_alert_message(
        actual_offset_ms=877.0 + _WAKE_DIVERGENCE_ALERT_TOLERANCE_MS + 1.0,
        computed_offset_ms=877.0,
    )
    assert msg is not None


def test_divergence_alert_silent_when_actual_later_than_computed():
    """Actual fires LATER than computed (negative divergence) -> no alert.
    Only the early-fire bypass is the failure mode we surface."""
    msg = engine._wake_divergence_alert_message(
        actual_offset_ms=800.0, computed_offset_ms=877.0,
    )
    assert msg is None
