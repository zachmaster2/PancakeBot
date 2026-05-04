"""Derived-timing-config tests.

The timing wakes are NOT user-tunable. They derive from empirical
constants in pancakebot/timing_constants.py at config load. This file
tests:

1. The derivation chain produces the expected values from the locked
   constants (regression: catch accidental constant edits).
2. Cross-validations fire when the kline-fetch wake offset exceeds
   `kline_cutoff_seconds * 1000 - OKX_KLINE_PUBLISH_DELAY_P95_MS`
   or the pool-read wake offset exceeds
   `pool_cutoff_seconds * 1000 - WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS`.
   The cutoffs are fixed by strategy; the wake offsets must fit.
3. Inclusion-math chain remains satisfied at the locked constants
   (median fetch lands block before lock_ts).
4. Engine timing-guard math at the locked bet_submit_deadline_offset_ms
   behaves correctly across the fetch-RTT distribution.
5. User-tunable knobs ``pool_cutoff_seconds`` and
   ``max_consecutive_fetch_failures`` accept their valid ranges.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import load_app_config  # noqa: E402
from pancakebot import timing_constants as tc  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


_BASE_TOML = """
[runtime]
kline_cutoff_seconds = {cutoff}
{extra_runtime}

[dry]
initial_bankroll_bnb = 50.0

[live]
min_bet_only = true

[backtest]
simulation_size = 1000
initial_bankroll_bnb = 50.0
"""


def _write_cfg(tmp_path: Path, *, cutoff: int = 2, extra: str = "") -> Path:
    p = tmp_path / "config.toml"
    p.write_text(
        _BASE_TOML.format(cutoff=cutoff, extra_runtime=extra),
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# 1. Derivation chain produces expected values from locked constants
# ---------------------------------------------------------------------------

def test_bet_submit_deadline_offset_derived_correctly(tmp_path):
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    expected = (
        tc.BSC_BET_SUBMIT_RTT_P95_MS
        + tc.BSC_BLOCK_TIME_MS
        + tc.BET_SUBMIT_SAFETY_BUFFER_MS
    )
    assert cfg.bet_submit_deadline_offset_ms == expected
    assert cfg.bet_submit_deadline_offset_ms == 750  # locked snapshot


def test_kline_fetch_wakeup_offset_derived_correctly(tmp_path):
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    expected = (
        cfg.bet_submit_deadline_offset_ms
        + tc.OKX_KLINE_FETCH_RTT_P95_MS
        + tc.SIGNAL_COMPUTE_TIME_MS
    )
    assert cfg.kline_fetch_wakeup_offset_ms == expected
    assert cfg.kline_fetch_wakeup_offset_ms == 1090  # locked snapshot


def test_pool_read_wakeup_offset_derived_correctly(tmp_path):
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    expected = cfg.kline_fetch_wakeup_offset_ms + tc.POOL_READ_TIME_MS
    assert cfg.pool_read_wakeup_offset_ms == expected
    assert cfg.pool_read_wakeup_offset_ms == 1095  # locked snapshot


def test_skew_sync_wakeup_offset_derived_correctly(tmp_path):
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    expected = (
        cfg.pool_read_wakeup_offset_ms
        + tc.OKX_SKEW_SYNC_TIME_P99_MS
        + tc.SKEW_SYNC_SAFETY_BUFFER_MS
    )
    assert cfg.skew_sync_wakeup_offset_ms == expected
    assert cfg.skew_sync_wakeup_offset_ms == 3645  # locked snapshot


def test_wake_chain_strictly_increasing(tmp_path):
    """Wake offsets must be ordered:
    skew_sync > pool_read > kline_fetch > bet_submit_deadline."""
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    assert cfg.skew_sync_wakeup_offset_ms > cfg.pool_read_wakeup_offset_ms
    assert cfg.pool_read_wakeup_offset_ms > cfg.kline_fetch_wakeup_offset_ms
    assert cfg.kline_fetch_wakeup_offset_ms > cfg.bet_submit_deadline_offset_ms


# ---------------------------------------------------------------------------
# 2. Cross-validations fire when cutoffs are too small
# ---------------------------------------------------------------------------

def test_kline_fetch_wakeup_exceeds_cutoff_publish_budget_rejected(tmp_path):
    """kline_fetch_wakeup > cutoff*1000 - P95 (looser bound) must raise.

    Tier ladder: P99 first, P95 fallback, error if even P95 fails.
    With cutoff=1 (=1000ms), P95=700, P99=1300:
      P99 budget = 1000 - 1300 = -300ms (already negative).
      P95 budget = 1000 - 700  =  300ms.
      kline_fetch_wakeup_offset=1090ms > both -> InvariantError.
    """
    raised: Exception | None = None
    try:
        load_app_config(str(_write_cfg(tmp_path, cutoff=1)))
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError)
    assert "config_kline_fetch_wakeup_exceeds_cutoff_publish_budget" in str(raised)


def test_canonical_cutoff_2_falls_back_to_p95_tier(tmp_path):
    """Strategy-canonical cutoff=2 lands in P95 tier (P99 budget too tight).

    cutoff=2 (2000ms), kline_fetch_wakeup=1090, P95=700, P99=1300:
      P99 budget = 2000 - 1300 = 700ms;  1090 > 700 -> P99 fails.
      P95 budget = 2000 - 700  = 1300ms; 1090 <= 1300 -> P95 passes.
    Expected tier: "P95".
    """
    cfg = load_app_config(str(_write_cfg(tmp_path, cutoff=2)))
    assert cfg.kline_cutoff_seconds == 2
    assert cfg.kline_publish_tier == "P95"
    # Sanity: at this tier the wake offset fits the P95 budget.
    assert cfg.kline_fetch_wakeup_offset_ms <= (
        2 * 1000 - tc.OKX_KLINE_PUBLISH_DELAY_P95_MS
    )
    # And it does NOT fit the strict P99 budget (else tier would be P99).
    assert cfg.kline_fetch_wakeup_offset_ms > (
        2 * 1000 - tc.OKX_KLINE_PUBLISH_DELAY_P99_MS
    )


def test_cutoff_3_promotes_to_p99_tier(tmp_path):
    """Larger cutoff auto-promotes to P99 tier without code change.

    cutoff=3 (3000ms), kline_fetch_wakeup=1090, P99=1300:
      P99 budget = 3000 - 1300 = 1700ms; 1090 <= 1700 -> P99 passes.
    Expected tier: "P99".
    """
    cfg = load_app_config(str(_write_cfg(tmp_path, cutoff=3)))
    assert cfg.kline_cutoff_seconds == 3
    assert cfg.kline_publish_tier == "P99"
    assert cfg.kline_fetch_wakeup_offset_ms <= (
        3 * 1000 - tc.OKX_KLINE_PUBLISH_DELAY_P99_MS
    )


def test_p95_le_p99_invariant_holds():
    """Module-load assert in timing_constants.py: P95 must be <= P99.

    If a future probe update accidentally inverts the percentile order,
    the assert at module load fires immediately. This test re-asserts
    the invariant for explicit regression coverage.
    """
    assert tc.OKX_KLINE_PUBLISH_DELAY_P95_MS <= tc.OKX_KLINE_PUBLISH_DELAY_P99_MS


def test_pool_read_wakeup_exceeds_cutoff_arrival_budget_rejected(tmp_path):
    """pool_read_wakeup > pool_cutoff*1000 - WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS must raise.

    With pool_cutoff=4 (=4000ms) and WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS=3500,
    the budget is 4000 - 3500 = 500ms but pool_read_wakeup_offset=1095ms.
    1095 > 500 → fires.
    """
    extra = "pool_cutoff_seconds = 4"
    raised: Exception | None = None
    try:
        load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError)
    assert "config_pool_read_wakeup_exceeds_cutoff_arrival_budget" in str(raised)


def test_pool_cutoff_default_is_6(tmp_path):
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    assert cfg.pool_cutoff_seconds == 6


def test_max_consecutive_fetch_failures_default_is_5(tmp_path):
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    assert cfg.max_consecutive_fetch_failures == 5


@pytest.mark.parametrize("n", [1, 5, 10, 100])
def test_max_consecutive_fetch_failures_accepts_valid_range(tmp_path, n):
    extra = f"max_consecutive_fetch_failures = {n}"
    cfg = load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    assert cfg.max_consecutive_fetch_failures == n


@pytest.mark.parametrize("n", [-1, 0, 101, 500])
def test_max_consecutive_fetch_failures_rejects_out_of_range(tmp_path, n):
    extra = f"max_consecutive_fetch_failures = {n}"
    raised: Exception | None = None
    try:
        load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError)
    assert "max_consecutive_fetch_failures_out_of_range" in str(raised)


# ---------------------------------------------------------------------------
# 3. Inclusion-math chain
# ---------------------------------------------------------------------------

def _wake_to_block_landing_ms(*, kline_fetch_wakeup_offset_ms: int, fetch_rtt_ms: int) -> int:
    """Worst-case ms past lock_at when a TX broadcast at wake+fetch+compute+sign
    lands in the next BSC block. Uses canonical timing constants.

    Returns NEGATIVE if block lands BEFORE lock_ts (= INCLUDED).
    """
    sign_overhead_ms = 5
    decision_ready_ms_after_wake = (
        fetch_rtt_ms + tc.SIGNAL_COMPUTE_TIME_MS + sign_overhead_ms
    )
    mempool_ms_after_wake = decision_ready_ms_after_wake + tc.BSC_BET_SUBMIT_RTT_P95_MS
    worst_case_block_landing_ms_after_wake = mempool_ms_after_wake + tc.BSC_BLOCK_TIME_MS
    return worst_case_block_landing_ms_after_wake - kline_fetch_wakeup_offset_ms


def test_inclusion_math_at_locked_constants_median_fetch(tmp_path):
    """Median-fetch rounds at the locked constants land block BEFORE lock_ts."""
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    delta_ms = _wake_to_block_landing_ms(
        kline_fetch_wakeup_offset_ms=cfg.kline_fetch_wakeup_offset_ms,
        fetch_rtt_ms=tc.OKX_KLINE_FETCH_RTT_P95_MS - 50,  # ~median (slightly tighter than p95)
    )
    assert delta_ms < 0, (
        f"At median fetch RTT and locked constants, worst-case block landing "
        f"must precede lock_ts. delta_ms={delta_ms}."
    )


def test_inclusion_math_at_locked_constants_p95_fetch_aborts_safely(tmp_path):
    """At p95 fetch RTT, decision-ready hits the timing guard cleanly --
    no submission, no gas burn.
    """
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    # decision_ready_ms_after_wake at p95 fetch
    p95_decision_ms_after_wake = (
        tc.OKX_KLINE_FETCH_RTT_P95_MS + tc.SIGNAL_COMPUTE_TIME_MS
    )
    decision_ready_offset_ms = (
        cfg.kline_fetch_wakeup_offset_ms - p95_decision_ms_after_wake
    )
    # Guard fires when remaining_to_lock <= safety_margin
    guard_fires = decision_ready_offset_ms <= cfg.bet_submit_deadline_offset_ms
    # Note: at the locked constants kline_fetch_wakeup=1090, p95_decision=340,
    # so decision_ready_offset = 750 = exactly the bet-submit-deadline. The
    # guard condition is `>=`, so equality fires the guard.
    assert guard_fires, (
        f"At p95 fetch RTT, decision-ready offset = {decision_ready_offset_ms}ms "
        f"vs bet_submit_deadline_offset_ms = {cfg.bet_submit_deadline_offset_ms}ms. "
        f"Guard must fire to abort the round before risk of late inclusion."
    )


# ---------------------------------------------------------------------------
# 4. Engine timing-guard math
# ---------------------------------------------------------------------------

def _guard_fires(*, now: float, lock_ts: float, deadline_ms: int) -> bool:
    """Mirror engine.py timing guard:
        if _utc_now() >= lock_ts - cfg.bet_submit_deadline_offset_ms / 1000.0: SKIP
    """
    deadline_seconds = deadline_ms / 1000.0
    return now >= lock_ts - deadline_seconds


def test_guard_does_not_fire_at_wake_with_locked_constants():
    """Wake fires at lock - kline_fetch_wakeup_offset_ms. Guard fires at
    lock - bet_submit_deadline_offset_ms. Wake must be OUTSIDE the safety
    zone (kline_fetch_wakeup > bet_submit_deadline).

    With locked constants: wake at lock-1090ms, guard at lock-750ms.
    -1090 < -750 -> wake is BEFORE guard threshold -> wake doesn't fire guard.
    """
    lock_ts = 1_000_000.0
    wake_at = lock_ts - 1090 / 1000.0
    assert not _guard_fires(now=wake_at, lock_ts=lock_ts, deadline_ms=750)


def test_guard_fires_at_p99_fetch_decision_ready():
    """At p99 fetch RTT, decision-ready is right at the safety margin
    boundary; guard fires (skips the round).
    """
    lock_ts = 1_000_000.0
    # Decision-ready at p99 fetch = wake + fetch_p99 + compute = lock - 1090 + 363 + 50
    decision_ready = lock_ts - (1090 - 363 - 50) / 1000.0  # = lock - 0.677s
    # Guard at lock - 0.75. -0.677 >= -0.75 -> TRUE -> SKIP
    assert _guard_fires(now=decision_ready, lock_ts=lock_ts, deadline_ms=750)


def test_guard_negative_offset_always_fires():
    """If fetch finishes AFTER lock_ts (now > lock_ts), guard MUST fire."""
    lock_ts = 1_000_000.0
    decision_ready = lock_ts + 0.050
    for deadline in [50, 100, 300, 750, 2000]:
        assert _guard_fires(now=decision_ready, lock_ts=lock_ts, deadline_ms=deadline)
