"""Derived-timing-config tests.

The timing wakes are NOT user-tunable. They derive from empirical
constants in pancakebot/timing_constants.py at config load. This file
tests:

1. The derivation chain produces the expected values from the locked
   constants (regression: catch accidental constant edits).
2. Cross-validations fire when the derived final-RPC-poll offset
   doesn't leave room for the RPC roundtrip + safety before the
   critical-path wake (Era 11 replacement for the WSS-arrival
   cross-validation). The cutoffs are fixed by strategy; the wake
   offsets must fit.
3. Inclusion-math chain remains satisfied at the locked constants
   (median fetch lands block before lock_ts).
4. Engine timing-guard math at the locked bet_submit_deadline_offset_before_lock_ms
   behaves correctly across the fetch-RTT distribution.
5. User-tunable knobs ``pool_cutoff_seconds`` and
   ``max_consecutive_kline_fetch_failures`` accept their valid ranges.

The prior P95/P99 publish-tier ladder + its config-load gate (removed
2026-05-17) is no longer covered here: it was a one-shot config-load
check that didn't gate runtime behavior, and the dynamic-anchor wake
fires at whatever offset the per-round anchor dictates anyway.
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
backtest_round_count = 1000
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
    """Bundle 4 (2026-05-14): derivation is
    (BSC_QUANTUM_MS + BSC_BLOCK_TIME_MS + VALIDATOR_ASSEMBLY_WINDOW_MS + BSC_BET_SUBMIT_ONE_WAY_MS) = 625.
    These constants reflect BEP-520 ms-encoding awareness and the
    correct one-way RPC framing (vs the prior round-trip overestimate). Used
    only as static fallback; the live decision path uses
    RpcPoller.compute_dynamic_submit_deadline_ms() per round.
    """
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    expected = (
        tc.BSC_QUANTUM_MS
        + tc.BSC_BLOCK_TIME_MS
        + tc.VALIDATOR_ASSEMBLY_WINDOW_MS
        + tc.BSC_BET_SUBMIT_ONE_WAY_MS
    )
    assert cfg.bet_submit_deadline_offset_before_lock_ms == expected
    assert cfg.bet_submit_deadline_offset_before_lock_ms == 625  # 2026-05-20 re-measurement


def test_critical_path_wakeup_offset_derived_correctly(tmp_path):
    """critical_path_wakeup_offset_before_lock_ms is the SINGLE entry point for the
    bet-decision sequence. Inside the wake the engine sequences pool
    snapshot -> kline fetch -> signal compute -> bet submit; all the
    operation-time constants roll up into the one wake offset.

    2026-05-20: 1045 -> 970ms (BSC_BET_SUBMIT_ONE_WAY_MS 150 -> 75ms).
    2026-06-06 VM re-baseline: 970 -> 1031ms — the static fallback now uses
    OKX P99 (351), the same statistic the dynamic wake uses; the P95 tier is
    retired.
    """
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    expected = (
        cfg.bet_submit_deadline_offset_before_lock_ms
        + tc.OKX_KLINE_FETCH_RTT_P99_MS
        + tc.SIGNAL_COMPUTE_TIME_MS
        + tc.POOL_READ_TIME_MS
    )
    assert cfg.critical_path_wakeup_offset_before_lock_ms == expected
    assert cfg.critical_path_wakeup_offset_before_lock_ms == 1031  # 2026-06-06 VM re-baseline


def test_preflight_wakeup_offset_derived_correctly(tmp_path):
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    expected = (
        cfg.critical_path_wakeup_offset_before_lock_ms
        + tc.PREFLIGHT_WAKEUP_OFFSET_BEFORE_CRITICAL_PATH_MS
    )
    assert cfg.preflight_wakeup_offset_before_lock_ms == expected
    assert cfg.preflight_wakeup_offset_before_lock_ms == 6031  # 2026-06-06 VM re-baseline


def test_wake_chain_strictly_increasing(tmp_path):
    """Bundle 5 v2 (2026-05-14): wake offsets must be ordered
    preflight > critical_path > bet_submit_deadline. The prior
    ``ntp_sync_wakeup_offset_ms`` is retired alongside the
    application-level NTP layer."""
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    assert cfg.preflight_wakeup_offset_before_lock_ms > cfg.critical_path_wakeup_offset_before_lock_ms
    assert cfg.critical_path_wakeup_offset_before_lock_ms > cfg.bet_submit_deadline_offset_before_lock_ms


# ---------------------------------------------------------------------------
# 2. Cross-validations fire when cutoffs are too small
# ---------------------------------------------------------------------------

def test_pool_cutoff_too_small_for_single_poll_capture_rejected(tmp_path):
    """The single-poll CAPTURE invariant must reject too-small pool_cutoff.

    2026-06-06 VM re-baseline: the single poll is a FIXED 2500ms rail. At
    pool_cutoff=2 the cutoff block sits at lock-2000, available ~lock-1375,
    but the rail fires at lock-2500 — before the cutoff block even exists.
    capture bound = 2000 - 450 - 625 - 200 = 725ms; single_poll=2500 > 725
    -> single_poll_fires_before_cutoff_available fires at config load.
    """
    extra = "pool_cutoff_seconds = 2"
    raised: Exception | None = None
    try:
        load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    except InvariantError as e:
        raised = e
    assert isinstance(raised, InvariantError)
    assert "single_poll_fires_before_cutoff_available" in str(raised)


def test_pool_cutoff_default_is_6(tmp_path):
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    assert cfg.pool_cutoff_seconds == 6


def test_max_consecutive_fetch_failures_default_is_5(tmp_path):
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    assert cfg.max_consecutive_kline_fetch_failures == 5


@pytest.mark.parametrize("n", [1, 5, 10, 100])
def test_max_consecutive_fetch_failures_accepts_valid_range(tmp_path, n):
    extra = f"max_consecutive_kline_fetch_failures = {n}"
    cfg = load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    assert cfg.max_consecutive_kline_fetch_failures == n


@pytest.mark.parametrize("n", [-1, 0, 101, 500])
def test_max_consecutive_fetch_failures_rejects_out_of_range(tmp_path, n):
    extra = f"max_consecutive_kline_fetch_failures = {n}"
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
    mempool_ms_after_wake = decision_ready_ms_after_wake + tc.BSC_BET_SUBMIT_ONE_WAY_MS
    worst_case_block_landing_ms_after_wake = mempool_ms_after_wake + tc.BSC_BLOCK_TIME_MS
    return worst_case_block_landing_ms_after_wake - kline_fetch_wakeup_offset_ms


def test_inclusion_math_at_locked_constants_median_fetch(tmp_path):
    """Median-fetch rounds at the locked constants land block BEFORE lock_ts."""
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    # kline fetch fires at lock - (critical_path_wakeup - POOL_READ_TIME)
    kline_fetch_offset = cfg.critical_path_wakeup_offset_before_lock_ms - tc.POOL_READ_TIME_MS
    delta_ms = _wake_to_block_landing_ms(
        kline_fetch_wakeup_offset_ms=kline_fetch_offset,
        fetch_rtt_ms=208,  # VM cycle_audit p50 max-of-3, 2026-06-06 (well below the p99 tail)
    )
    assert delta_ms < 0, (
        f"At median fetch RTT and locked constants, worst-case block landing "
        f"must precede lock_ts. delta_ms={delta_ms}."
    )


def test_inclusion_math_at_design_p99_fetch_aborts_safely(tmp_path):
    """At the design (p99) fetch RTT — the same statistic critical_path is
    sized from — decision-ready lands exactly at the bet-submit deadline, so
    the timing guard fires cleanly: no submission, no gas burn.
    """
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    # decision_ready_ms_after_wake at the design p99 fetch
    p99_decision_ms_after_wake = (
        tc.OKX_KLINE_FETCH_RTT_P99_MS + tc.SIGNAL_COMPUTE_TIME_MS
    )
    kline_fetch_offset = cfg.critical_path_wakeup_offset_before_lock_ms - tc.POOL_READ_TIME_MS
    decision_ready_offset_ms = kline_fetch_offset - p99_decision_ms_after_wake
    # Guard fires when remaining_to_lock <= safety_margin
    guard_fires = decision_ready_offset_ms <= cfg.bet_submit_deadline_offset_before_lock_ms
    # By construction kline_fetch_offset = bet_submit + p99 + signal, so at a
    # p99 fetch decision_ready_offset = bet_submit_deadline exactly (1026 - 401
    # = 625). The guard condition is `<=`, so equality fires the guard.
    assert guard_fires, (
        f"At the design p99 fetch RTT, decision-ready offset = {decision_ready_offset_ms}ms "
        f"vs bet_submit_deadline_offset_before_lock_ms = {cfg.bet_submit_deadline_offset_before_lock_ms}ms. "
        f"Guard must fire to abort the round before risk of late inclusion."
    )


# ---------------------------------------------------------------------------
# 4. Engine timing-guard math
# ---------------------------------------------------------------------------

def _guard_fires(*, now: float, lock_ts: float, deadline_ms: int) -> bool:
    """Mirror engine.py timing guard:
        if _utc_now() >= lock_ts - cfg.bet_submit_deadline_offset_before_lock_ms / 1000.0: SKIP
    """
    deadline_seconds = deadline_ms / 1000.0
    return now >= lock_ts - deadline_seconds


def test_guard_does_not_fire_at_wake_with_locked_constants(tmp_path):
    """Critical-path wake fires at lock - critical_path_wakeup_offset_before_lock_ms.
    Guard fires at lock - bet_submit_deadline_offset_before_lock_ms. Wake must be
    OUTSIDE the safety zone (critical_path_wakeup > bet_submit_deadline).

    Bundle 4 reviewer Y3: was hardcoded to 1095/750; now reads canonical
    config values so the assertion tracks the derivation (currently
    1045/700 post-Bundle 4).
    """
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    lock_ts = 1_000_000.0
    wake_at = lock_ts - cfg.critical_path_wakeup_offset_before_lock_ms / 1000.0
    assert not _guard_fires(
        now=wake_at, lock_ts=lock_ts,
        deadline_ms=cfg.bet_submit_deadline_offset_before_lock_ms,
    )


def test_guard_fires_at_p99_fetch_decision_ready(tmp_path):
    """At p99 fetch RTT, decision-ready is right at the safety margin
    boundary; guard fires (skips the round).

    Bundle 4 reviewer Y3: pulls canonical offsets via load_app_config
    rather than hardcoding pre-Bundle-4 1090/750 magic numbers.
    """
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    lock_ts = 1_000_000.0
    # Decision-ready = critical_path wake (lock - critical_path_wakeup)
    #                 + pool_read + kline_fetch_p99 + signal_compute.
    # Equivalent: lock - (critical_path - kline_fetch - signal_compute - pool_read)
    # but critical_path - pool_read - signal_compute = kline_fetch_offset, so
    # decision_ready = lock - kline_fetch_offset + kline_fetch_p99
    kline_fetch_offset_ms = cfg.critical_path_wakeup_offset_before_lock_ms - tc.POOL_READ_TIME_MS
    decision_ready = lock_ts - (kline_fetch_offset_ms - 363 - tc.SIGNAL_COMPUTE_TIME_MS) / 1000.0
    # At canonical Bundle 4 timing: decision_ready = lock - 0.627s; guard at
    # lock - 0.700s. -0.627 >= -0.700 -> TRUE -> SKIP.
    assert _guard_fires(
        now=decision_ready, lock_ts=lock_ts,
        deadline_ms=cfg.bet_submit_deadline_offset_before_lock_ms,
    )


def test_guard_negative_offset_always_fires():
    """If fetch finishes AFTER lock_ts (now > lock_ts), guard MUST fire."""
    lock_ts = 1_000_000.0
    decision_ready = lock_ts + 0.050
    for deadline in [50, 100, 300, 750, 2000]:
        assert _guard_fires(now=decision_ready, lock_ts=lock_ts, deadline_ms=deadline)
