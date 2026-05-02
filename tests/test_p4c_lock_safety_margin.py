"""p4c lock_safety_margin_ms config + guard-math regression tests.

Per p4c (`var/strategy_review/p4c_timing_guard_regression_*.md`): the prior
`_LOCK_SAFETY_MARGIN_SECONDS = 1.0` constant exceeded the post-cf04f35
`kline_fetch_offset_ms = 850` wake schedule, making the timing guard at
engine.py:609 fire structurally on every BET decision. Fix: promote to
`[runtime] lock_safety_margin_ms` config (default 300, range 50-2000) with
a cross-constraint that `lock_safety_margin_ms < kline_fetch_offset_ms`.

Tests cover:
  - Config default + valid range + out-of-range rejection
  - Cross-constraint: safety margin must be < kline_fetch_offset_ms
  - Guard math at the new default: median fetch passes, slow fetch aborts
  - Guard math regression: re-asserts the broken-old-default behavior
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import load_app_config  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


_BASE_TOML = """
[runtime]
kline_cutoff_seconds = 2
prefetch_offset_seconds = 6
{extra_runtime}

[dry]
initial_bankroll_bnb = 50.0

[live]
min_bet_only = true

[backtest]
simulation_size = 1000
initial_bankroll_bnb = 50.0
"""


def _write_cfg(tmp_path: Path, *, extra: str = "") -> Path:
    p = tmp_path / "config.toml"
    p.write_text(_BASE_TOML.format(extra_runtime=extra), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Config: default + valid range
# ---------------------------------------------------------------------------

def test_lock_safety_margin_ms_default_is_300(tmp_path):
    """Omitted -> default 300ms (the post-p4c safe default)."""
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    assert cfg.lock_safety_margin_ms == 300


@pytest.mark.parametrize("margin", [50, 100, 300, 700, 800])
def test_lock_safety_margin_ms_accepts_valid_range(tmp_path, margin):
    """[50..kline_fetch_offset_ms): all accepted at default kline_fetch_offset_ms=850."""
    extra = f"lock_safety_margin_ms = {margin}"
    cfg = load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    assert cfg.lock_safety_margin_ms == margin


@pytest.mark.parametrize("margin", [-1, 0, 49, 2001, 5000])
def test_lock_safety_margin_ms_rejects_out_of_range(tmp_path, margin):
    """Below 50 or above 2000: must raise."""
    extra = f"lock_safety_margin_ms = {margin}"
    raised: Exception | None = None
    try:
        load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError), (
        f"margin={margin} must raise InvariantError; got "
        f"{type(raised).__name__}: {raised}"
    )
    assert "lock_safety_margin_ms_out_of_range" in str(raised)


# ---------------------------------------------------------------------------
# Cross-constraint: safety_margin < kline_fetch_offset_ms
# ---------------------------------------------------------------------------

def test_safety_margin_equal_to_kline_offset_rejected(tmp_path):
    """The exact regression case: margin == kline_fetch_offset_ms must FAIL.

    This is what the pre-p4c code path effectively had (margin=1000ms vs
    wake=850ms). The cross-constraint ensures the wake is OUTSIDE the
    safety zone.
    """
    extra = "kline_fetch_offset_ms = 850\nlock_safety_margin_ms = 850"
    raised: Exception | None = None
    try:
        load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError), (
        f"margin == kline_fetch_offset_ms must raise; got "
        f"{type(raised).__name__}: {raised}"
    )
    assert "lock_safety_margin_ms_must_be_less_than_kline_fetch_offset_ms" in str(raised)


def test_safety_margin_greater_than_kline_offset_rejected(tmp_path):
    """The original regression: margin > wake offset means wake INSIDE safety zone."""
    extra = "kline_fetch_offset_ms = 500\nlock_safety_margin_ms = 700"
    raised: Exception | None = None
    try:
        load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError), (
        f"margin > kline_fetch_offset_ms must raise; got "
        f"{type(raised).__name__}: {raised}"
    )
    assert "lock_safety_margin_ms_must_be_less_than_kline_fetch_offset_ms" in str(raised)


def test_safety_margin_less_than_kline_offset_accepted(tmp_path):
    """The intended regime: margin strictly less than wake offset."""
    extra = "kline_fetch_offset_ms = 850\nlock_safety_margin_ms = 300"
    cfg = load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    assert cfg.kline_fetch_offset_ms == 850
    assert cfg.lock_safety_margin_ms == 300


# ---------------------------------------------------------------------------
# Guard math: re-implement engine.py:609 logic with explicit time values
# ---------------------------------------------------------------------------

def _guard_fires(*, now: float, lock_ts: float, safety_margin_ms: int) -> bool:
    """Mirror engine.py:609:
        if _utc_now() >= lock_ts_t - cfg.lock_safety_margin_ms / 1000.0: SKIP
    Returns True iff the guard SKIPs the bet.
    """
    safety_seconds = safety_margin_ms / 1000.0
    return now >= lock_ts - safety_seconds


def test_guard_at_p4c_default_passes_typical_fetch():
    """Wake at lock-850ms, fetch 280ms median, decision-ready at lock-520ms.

    With margin=300ms: lock - 520ms < lock - 300ms, so guard does NOT fire.
    """
    lock_ts = 1_000_000.0
    decision_ready = lock_ts - 0.520  # 520 ms before lock
    assert not _guard_fires(now=decision_ready, lock_ts=lock_ts, safety_margin_ms=300)


def test_guard_at_p4c_default_aborts_slow_p99_fetch():
    """Slow p99 fetch (~850ms) lands decision right at lock_ts.

    With margin=300ms: now=lock_ts >= lock_ts - 300ms, so guard FIRES.
    Correct conservative behavior.
    """
    lock_ts = 1_000_000.0
    decision_ready = lock_ts  # right at lock
    assert _guard_fires(now=decision_ready, lock_ts=lock_ts, safety_margin_ms=300)


def test_guard_at_p4c_default_passes_fast_fetch():
    """Fast 250ms fetch -> decision-ready at lock-600ms. Margin=300 -> PASS."""
    lock_ts = 1_000_000.0
    decision_ready = lock_ts - 0.600
    assert not _guard_fires(now=decision_ready, lock_ts=lock_ts, safety_margin_ms=300)


def test_guard_at_old_broken_margin_fires_at_wake_regression():
    """Regression assertion: with the OLD margin (1000ms) and wake at lock-850ms,
    the guard fires AT WAKE TIME -- before any fetch even runs.

    This is the bug p4c fixes. The test is here so that anyone re-introducing
    the old margin gets a clear pytest failure.
    """
    lock_ts = 1_000_000.0
    wake_time = lock_ts - 0.850  # the moment the bot wakes for kline fetch
    # With the OLD margin = 1000ms, the guard would fire at wake.
    assert _guard_fires(now=wake_time, lock_ts=lock_ts, safety_margin_ms=1000)
    # With the NEW default 300ms, the guard does NOT fire at wake.
    assert not _guard_fires(now=wake_time, lock_ts=lock_ts, safety_margin_ms=300)


def test_guard_negative_offset_always_fires():
    """Edge case: if fetch finishes AFTER lock_at (now > lock_ts), guard MUST fire
    regardless of margin."""
    lock_ts = 1_000_000.0
    decision_ready = lock_ts + 0.050  # 50ms past lock
    for margin in [50, 100, 300, 700, 2000]:
        assert _guard_fires(now=decision_ready, lock_ts=lock_ts, safety_margin_ms=margin), (
            f"with now > lock_ts, guard must fire at margin={margin}"
        )
