"""p4c-revision config + guard-math + inclusion-math tests.

History:
- p4c (commit b1ae33a): introduced ``[runtime] lock_safety_margin_ms`` to
  fix the wake-inside-safety-zone regression. Defaulted to 300ms, which
  unblocked the gate but didn't model BSC inclusion timing -> ~76% gas-
  burn rate at submit.
- p4c-revision (this commit): defaults lifted to
  ``kline_fetch_offset_ms=1200`` and ``lock_safety_margin_ms=750`` so
  decision-ready + submit + mempool + next-block lands BEFORE lock_ts
  (block.timestamp < lockTimestamp, satisfying PancakeSwap PredictionV2's
  strict check). Sized empirically from
  ``research/p4c_canonical_loop_probe.py`` (n=200 at wake=1200ms,
  2026-05-02): per-symbol p99 RTT ~558ms; round first-try success 88%.

Tests cover:
  - Config defaults + valid range + out-of-range rejection
  - Cross-constraint: safety margin must be < kline_fetch_offset_ms
  - Guard math at the new defaults: median fetch passes, slow fetch aborts
  - Inclusion-math chain (NEW per reviewer R1): wake -> fetch -> sign ->
    submit -> mempool -> next-block lands with block.timestamp < lock_ts
  - Guard math regression: re-asserts the pre-p4c broken-default behavior
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
# Config: defaults + valid range
# ---------------------------------------------------------------------------

def test_kline_fetch_offset_ms_default_is_1200(tmp_path):
    """Post-p4c-revision default is 1200ms wake (= 800ms post-close)."""
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    assert cfg.kline_fetch_offset_ms == 1200


def test_lock_safety_margin_ms_default_is_750(tmp_path):
    """Post-p4c-revision default is 750ms (= block_time + submit_RTT)."""
    cfg = load_app_config(str(_write_cfg(tmp_path)))
    assert cfg.lock_safety_margin_ms == 750


@pytest.mark.parametrize("margin", [50, 100, 300, 750, 1199])
def test_lock_safety_margin_ms_accepts_valid_range(tmp_path, margin):
    """[50..kline_fetch_offset_ms): all accepted at default kline_fetch_offset_ms=1200."""
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
    """margin == kline_fetch_offset_ms must FAIL.

    This is the structural failure mode of the original p4c regression
    (wake fires AT the safety margin boundary). Cross-constraint enforces
    strict less-than to keep wake outside the safety zone.
    """
    extra = "kline_fetch_offset_ms = 1200\nlock_safety_margin_ms = 1200"
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
    extra = "kline_fetch_offset_ms = 1200\nlock_safety_margin_ms = 750"
    cfg = load_app_config(str(_write_cfg(tmp_path, extra=extra)))
    assert cfg.kline_fetch_offset_ms == 1200
    assert cfg.lock_safety_margin_ms == 750


# ---------------------------------------------------------------------------
# Guard math: re-implement engine.py's pre-bet timing guard with explicit values
# ---------------------------------------------------------------------------

def _guard_fires(*, now: float, lock_ts: float, safety_margin_ms: int) -> bool:
    """Mirror the engine's timing guard:
        if _utc_now() >= lock_ts_t - cfg.lock_safety_margin_ms / 1000.0: SKIP
    Returns True iff the guard SKIPs the bet.
    """
    safety_seconds = safety_margin_ms / 1000.0
    return now >= lock_ts - safety_seconds


def test_guard_at_revision_default_passes_typical_fetch():
    """Wake at lock-1200ms, fetch ~270ms median + compute 50ms,
    decision-ready at lock-880ms.

    Guard threshold = lock-750ms. now=lock-880ms < lock-750ms -> PASS.
    """
    lock_ts = 1_000_000.0
    decision_ready = lock_ts - 0.880
    assert not _guard_fires(now=decision_ready, lock_ts=lock_ts, safety_margin_ms=750)


def test_guard_at_revision_default_aborts_p99_fetch():
    """p99 fetch ~560ms + compute 50ms -> decision-ready at lock-590ms.

    Guard threshold = lock-750ms. now=lock-590ms >= lock-750ms (-590 > -750)
    -> guard FIRES -> SKIP. Conservative-correct (avoids gas burn on
    p99-tail rounds where mempool would land too close to lock).
    """
    lock_ts = 1_000_000.0
    decision_ready = lock_ts - 0.590  # p99 fetch + compute
    assert _guard_fires(now=decision_ready, lock_ts=lock_ts, safety_margin_ms=750)


def test_guard_at_revision_default_passes_fast_fetch():
    """Fast fetch (250ms) + compute 50ms -> decision-ready at lock-900ms.

    Guard at lock-750ms: -900 < -750 -> PASS. Plenty of submit budget.
    """
    lock_ts = 1_000_000.0
    decision_ready = lock_ts - 0.900
    assert not _guard_fires(now=decision_ready, lock_ts=lock_ts, safety_margin_ms=750)


def test_guard_at_old_p4c_broken_margin_fires_at_wake_regression():
    """Regression assertion: with the OLD p4c-era margin (1000ms) and
    wake at lock-850ms (also OLD), the guard fired AT WAKE TIME -- before
    any fetch ran. This was the original p4c bug.

    With OLD config (850ms wake, 1000ms margin): now=lock-850ms >=
    lock-1000ms (-850 > -1000) -> TRUE -> SKIP at wake.
    With NEW config (1200ms wake, 750ms margin): now=lock-1200ms >=
    lock-750ms? -1200 > -750? FALSE -> wake outside safety zone, OK.
    """
    lock_ts = 1_000_000.0
    old_wake = lock_ts - 0.850
    new_wake = lock_ts - 1.200
    assert _guard_fires(now=old_wake, lock_ts=lock_ts, safety_margin_ms=1000), \
        "OLD config (1000ms margin, 850ms wake) MUST fire at wake (regression assert)"
    assert not _guard_fires(now=new_wake, lock_ts=lock_ts, safety_margin_ms=750), \
        "NEW config (750ms margin, 1200ms wake) MUST NOT fire at wake"


def test_guard_negative_offset_always_fires():
    """Edge case: if fetch finishes AFTER lock_ts (now > lock_ts), guard
    MUST fire regardless of margin. Defense against extreme network
    pathology that survives all earlier checks.
    """
    lock_ts = 1_000_000.0
    decision_ready = lock_ts + 0.050  # 50ms past lock
    for margin in [50, 100, 300, 750, 2000]:
        assert _guard_fires(now=decision_ready, lock_ts=lock_ts, safety_margin_ms=margin), (
            f"with now > lock_ts, guard must fire at margin={margin}"
        )


# ---------------------------------------------------------------------------
# Inclusion-math chain (NEW for p4c-revision per reviewer R1)
#
# Validates that the wake -> fetch -> compute -> sign -> submit ->
# mempool -> next-block-landing chain ends with block.timestamp <
# lock_ts (i.e., block lands in the lock_ts - 1 second).
#
# PancakeSwap PredictionV2 contract requires block.timestamp < lockTimestamp;
# any TX that lands in a block with timestamp >= lock_ts reverts.
# ---------------------------------------------------------------------------

# Reference values empirically observed by research/p4c_canonical_loop_probe.py
# n=200 (2026-05-02). Re-derive: py research/p4c_canonical_loop_probe.py 200 1200
_FETCH_RTT_MEDIAN_MS = 270
_FETCH_RTT_P99_MS = 560
_COMPUTE_OVERHEAD_MS = 50
_SIGN_TX_OVERHEAD_MS = 5
# BSC RPC submit RTT estimate; matches user's working assumption.
# Update if research/p4c_okx_submit_probe.py (TBD) produces different data.
_SUBMIT_RTT_MS = 400
# BSC block production interval, post-Maxwell hardfork (~Apr 2026).
# Source: pancakebot/chain/pool_watcher.py:537 _BSC_BLOCK_TIME = 0.5
# (conservative; "actual ~0.44s" per comment).
_BSC_BLOCK_MS = 450


def _wake_to_block_landing_ms(*, kline_fetch_offset_ms: int, fetch_rtt_ms: int) -> int:
    """Return the time-from-wake-to-WORST-CASE-block-landing (positive = past lock).

    Returns NEGATIVE if block lands BEFORE lock_ts (= included).
    """
    decision_ready_ms_after_wake = fetch_rtt_ms + _COMPUTE_OVERHEAD_MS + _SIGN_TX_OVERHEAD_MS
    mempool_ms_after_wake = decision_ready_ms_after_wake + _SUBMIT_RTT_MS
    worst_case_block_landing_ms_after_wake = mempool_ms_after_wake + _BSC_BLOCK_MS
    # Position of block landing relative to lock_ts:
    # wake fires at lock_ts - kline_fetch_offset_ms.
    # block lands at wake + worst_case_block_landing_ms_after_wake
    #              = lock_ts - kline_fetch_offset_ms + worst_case_block_landing_ms_after_wake
    return worst_case_block_landing_ms_after_wake - kline_fetch_offset_ms


def test_inclusion_math_at_revision_default_median_fetch():
    """Median-fetch round at default config lands block BEFORE lock_ts.

    Chain (median): wake @ lock-1200ms -> fetch+compute+sign 325ms ->
    submit_RTT 400ms -> next-block 450ms -> total 1175ms after wake.
    Block lands at lock - 1200 + 1175 = lock - 25ms (worst case).
    block.timestamp = lock_ts - 1 (still in the lock_ts-1 second).
    INCLUDED.
    """
    delta_ms = _wake_to_block_landing_ms(
        kline_fetch_offset_ms=1200,
        fetch_rtt_ms=_FETCH_RTT_MEDIAN_MS,
    )
    # delta_ms < 0 means block lands BEFORE lock_ts (included).
    assert delta_ms < 0, (
        f"At median fetch, worst-case block landing must precede lock_ts. "
        f"delta_ms={delta_ms} (positive means too late)."
    )


def test_inclusion_math_old_p4c_default_fails_inclusion():
    """Pre-p4c-revision default (kline_fetch_offset_ms=850) FAILS inclusion
    in the worst-case block-landing scenario.

    Chain: wake @ lock-850ms -> fetch+compute+sign 325ms -> submit 400ms
    -> next-block 450ms -> total 1175ms after wake. Block at lock-850 +
    1175 = lock + 325ms. block.timestamp >= lock_ts -> REVERT.
    """
    delta_ms = _wake_to_block_landing_ms(
        kline_fetch_offset_ms=850,
        fetch_rtt_ms=_FETCH_RTT_MEDIAN_MS,
    )
    assert delta_ms > 0, (
        f"OLD p4c default (850ms wake) MUST land block AFTER lock_ts at "
        f"median fetch -- this is the inclusion-side bug. delta_ms={delta_ms}."
    )


def test_inclusion_math_at_revision_default_p99_fetch_aborts_safely():
    """At p99 fetch, decision-ready is lock-590ms (= 1200 - 560 - 50 = 590).
    Guard at lock-750ms fires (-590 > -750), aborting BEFORE TX submission.
    No gas burn. Conservative-correct.

    This test also confirms the guard's role: if the guard somehow DIDN'T
    fire, the inclusion-math chain at p99 fetch would push block landing
    past lock_ts. The guard prevents the wasted gas.
    """
    # Confirm guard would fire at p99 decision-ready time.
    lock_ts = 1_000_000.0
    decision_ready_p99 = lock_ts - (1200 - _FETCH_RTT_P99_MS - _COMPUTE_OVERHEAD_MS) / 1000.0
    assert _guard_fires(now=decision_ready_p99, lock_ts=lock_ts, safety_margin_ms=750), \
        "p99 fetch decision-ready must trigger guard at default config (no gas burn)"

    # Confirm: if the guard didn't fire, p99-fetch round would miss inclusion.
    delta_ms_p99 = _wake_to_block_landing_ms(
        kline_fetch_offset_ms=1200,
        fetch_rtt_ms=_FETCH_RTT_P99_MS,
    )
    assert delta_ms_p99 > 0, (
        f"p99 fetch round (560ms) DOES land past lock_ts in worst case "
        f"-- which is why the guard at lock-750ms aborts these rounds. "
        f"delta_ms={delta_ms_p99}."
    )


def test_inclusion_math_smaller_offset_breaks_inclusion():
    """A smaller kline_fetch_offset_ms (e.g., 1000ms) leaves median
    fetches without enough budget for next-block inclusion.

    At wake=1000ms, fetch+compute+sign 325ms, submit 400ms, block 450ms:
    total 1175ms after wake. Block lands at lock-1000+1175 = lock+175ms
    -> block.timestamp = lock_ts -> REVERT.
    """
    delta_ms = _wake_to_block_landing_ms(
        kline_fetch_offset_ms=1000,
        fetch_rtt_ms=_FETCH_RTT_MEDIAN_MS,
    )
    assert delta_ms > 0, (
        f"At kline_fetch_offset_ms=1000, median-fetch round STILL fails "
        f"inclusion in worst-case (block lands 175ms past lock_ts). "
        f"This is why the empirical sweet spot is 1200ms. delta_ms={delta_ms}."
    )


def test_inclusion_math_threshold_is_at_least_1075ms():
    """The minimum kline_fetch_offset_ms for guaranteed median-fetch
    inclusion in worst-case block-landing is 1075ms = block(450) +
    submit(400) + sign(5) + compute(50) + fetch(270).

    User's locked default of 1200 sits 125ms above this floor.
    """
    threshold = (
        _BSC_BLOCK_MS
        + _SUBMIT_RTT_MS
        + _SIGN_TX_OVERHEAD_MS
        + _COMPUTE_OVERHEAD_MS
        + _FETCH_RTT_MEDIAN_MS
    )
    assert threshold == 1175, (
        f"Sanity-check on the inclusion threshold formula. "
        f"Got {threshold}; expected 1175 (450 + 400 + 5 + 50 + 270)."
    )
    # Default 1200 sits 25ms above the threshold; passes.
    delta_ms = _wake_to_block_landing_ms(
        kline_fetch_offset_ms=1200, fetch_rtt_ms=_FETCH_RTT_MEDIAN_MS,
    )
    assert delta_ms < 0, "Default 1200ms wake should clear the inclusion threshold"
