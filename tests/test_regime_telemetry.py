"""Tests for the rolling-window regime-divergence monitors."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime.regime_telemetry import (  # noqa: E402
    RollingMedianDriftMonitor,
    RollingP99Monitor,
    RollingRateMonitor,
    _nearest_rank,
)


# --------------------------------------------------------------------------
# _nearest_rank
# --------------------------------------------------------------------------


def test_nearest_rank_single():
    assert _nearest_rank([42.0], 0.99) == 42.0


def test_nearest_rank_p99_of_100():
    vals = sorted(float(i) for i in range(1, 101))  # 1..100
    # ceil(0.99 * 100) = 99 -> 1-indexed 99th value = 99.0
    assert _nearest_rank(vals, 0.99) == 99.0


def test_nearest_rank_max_clamped():
    vals = [1.0, 2.0, 3.0]
    assert _nearest_rank(vals, 1.0) == 3.0


# --------------------------------------------------------------------------
# RollingP99Monitor
# --------------------------------------------------------------------------


def test_p99_silent_below_min_samples():
    m = RollingP99Monitor(name="x", constant_ms=352, tolerance_ms=50, window=100, min_samples=30)
    for _ in range(29):
        assert m.observe(9999) is None  # huge, but under min_samples -> silent


def test_p99_fires_on_breach_edge_only():
    m = RollingP99Monitor(name="okx", constant_ms=352, tolerance_ms=50, window=100, min_samples=30)
    msgs = [m.observe(500) for _ in range(40)]  # 500 > 352+50
    fired = [x for x in msgs if x is not None]
    assert len(fired) == 1, "should be edge-triggered, exactly one alert"
    assert "REGIME_DRIFT" in fired[0]
    assert "monitor=okx" in fired[0]
    assert "observed_p99=500ms" in fired[0]
    assert "constant=352ms" in fired[0]


def test_p99_silent_within_tolerance():
    m = RollingP99Monitor(name="okx", constant_ms=352, tolerance_ms=50, window=100, min_samples=30)
    # 400 <= 352+50=402 -> no breach
    fired = [m.observe(400) for _ in range(50)]
    assert all(x is None for x in fired)


def test_p99_recovery_after_breach():
    m = RollingP99Monitor(name="okx", constant_ms=352, tolerance_ms=50, window=10, min_samples=5)
    breach = [m.observe(500) for _ in range(10)]
    assert any("REGIME_DRIFT monitor=okx" in (x or "") for x in breach)
    # flush window with healthy values <= constant
    rec = [m.observe(300) for _ in range(10)]
    assert any("REGIME_DRIFT_RECOVERED" in (x or "") for x in rec)


# --------------------------------------------------------------------------
# RollingRateMonitor
# --------------------------------------------------------------------------


def test_rate_fires_when_fallback_rate_high():
    m = RollingRateMonitor(name="anchor_static_fallback", max_rate=0.10, window=50, min_samples=50)
    msgs = [m.observe(True, detail="reason=timeout") for _ in range(50)]
    fired = [x for x in msgs if x is not None]
    assert len(fired) == 1
    assert "monitor=anchor_static_fallback" in fired[0]
    assert "rate=1.00" in fired[0]
    assert "reason=timeout" in fired[0]


def test_rate_silent_when_below_threshold():
    m = RollingRateMonitor(name="anchor", max_rate=0.10, window=50, min_samples=50)
    # 4 of 50 = 0.08 < 0.10
    seq = [True, True, True, True] + [False] * 46
    fired = [m.observe(h) for h in seq]
    assert all(x is None for x in fired)


# --------------------------------------------------------------------------
# RollingMedianDriftMonitor
# --------------------------------------------------------------------------


def test_median_drift_fires_on_block_time_shift():
    m = RollingMedianDriftMonitor(name="bsc_block_time", expected=450, tolerance=20, window=20, min_samples=10)
    fired = [m.observe(480) for _ in range(20)]  # 480 - 450 = 30 > 20
    hits = [x for x in fired if x is not None]
    assert len(hits) == 1
    assert "monitor=bsc_block_time" in hits[0]
    assert "observed_median=480" in hits[0]
    assert "drift=+30" in hits[0]


def test_median_drift_silent_within_tolerance():
    m = RollingMedianDriftMonitor(name="bsc", expected=450, tolerance=20, window=20, min_samples=10)
    fired = [m.observe(455) for _ in range(20)]  # within 20
    assert all(x is None for x in fired)


def test_median_drift_recovers():
    m = RollingMedianDriftMonitor(name="bsc", expected=450, tolerance=20, window=10, min_samples=5)
    [m.observe(490) for _ in range(10)]
    rec = [m.observe(450) for _ in range(10)]
    assert any("REGIME_DRIFT_RECOVERED" in (x or "") for x in rec)
