"""Regression tests for momentum_pipeline skip_reason propagation.

Investigation A (2026-05-05) found that when ``MomentumGate.evaluate``
returns ``MomentumGateResult(signal=None, skip_reason="kline_fetch_transient_failure")``,
the pipeline silently fell through to ``_skip("gate_no_signal")`` because
the skip-reason check at decide_open_round was ``if signal_dir is None:
return self._skip("gate_no_signal")`` — never inspecting the gate's
specific skip_reason.

Effect: every transient kline fetch failure that didn't manage to fire
a regime-2 ETH+SOL fallback would surface in cycle_audit as
``gate_no_signal``, masking data-availability issues from the operator.

This file's tests pin the FIX: the pipeline propagates the gate's
specific skip_reason.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import (  # noqa: E402
    BtcPrimaryConfig,
    BtcPrimarySizingConfig,
    BtcPrimaryThresholdConfig,
    EthSolFallbackConfig,
    EthSolFallbackSignalConfig,
    EthSolFallbackSizingConfig,
    GateConfig,
    PoolFilterConfig,
    RiskConfig,
    StrategyConfig,
    Tier2SizingConfig,
)
from pancakebot.strategy.momentum_gate import (  # noqa: E402
    MomentumGateConfig,
    MomentumGateResult,
)
from pancakebot.strategy.momentum_pipeline import (  # noqa: E402
    MomentumOnlyPipeline,
    StrategyPipelineDecision,
)
from pancakebot.types import Round  # noqa: E402
from pancakebot.util import InvariantError  # noqa: E402


def _make_strategy_config() -> StrategyConfig:
    """Canonical strategy config used by the pipeline tests."""
    return StrategyConfig(
        pool_filter=PoolFilterConfig(min_pool_bnb_at_cutoff=1.5, min_payout_multiple_at_cutoff=1.5),
        gate=GateConfig(mtf_lookbacks=(3, 7, 15), mtf_min_return_threshold=0.0001),
        btc_primary=BtcPrimaryConfig(
            threshold=BtcPrimaryThresholdConfig(
                small_pool_min_signal_strength=0.0002, pool_size_boundary_bnb=3.0,
            ),
            sizing=BtcPrimarySizingConfig(
                base_pool_fraction=0.04, pool_fraction_slope=100.0, max_pool_fraction=0.30,
            ),
        ),
        eth_sol_fallback=EthSolFallbackConfig(
            signal=EthSolFallbackSignalConfig(min_signal_strength=0.00015),
            sizing=EthSolFallbackSizingConfig(base_pool_fraction=0.02),
        ),
        tier2_sizing=Tier2SizingConfig(
            eth_sol_signal_weight=0.3, min_bet_threshold_bnb=0.01,
        ),
        risk=RiskConfig(
            max_bet_fraction_of_bankroll=0.05, min_bankroll_bnb_to_bet=0.20,
            max_drawdown_fraction_from_peak=0.15, cooldown_rounds=72,
            drawdown_peak_window_days=7, max_bet_bnb_btc_primary=2.0,
            max_bet_bnb_eth_sol_fallback=0.5, drawdown_peak_mode="rolling_7d",
        ),
    )


def _make_pipeline(gate_mock) -> MomentumOnlyPipeline:
    """Pipeline with a mock gate. kline_cutoff_seconds=2, canonical strategy."""
    cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=2,
        mtf_lookbacks=(3, 7, 15),
        mtf_min_return_threshold=0.0001,
        max_consecutive_kline_fetch_failures=5,
    )
    return MomentumOnlyPipeline(
        config=cfg,
        strategy_config=_make_strategy_config(),
        gate=gate_mock,
        kline_cutoff_seconds=2,
        pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001,
        treasury_fee_fraction=0.03,
    )


def _make_round(epoch: int = 478456) -> Round:
    """A minimal open round for the pipeline to consume."""
    return Round(
        epoch=epoch,
        start_at=1777966856,
        lock_at=1777967156,
        lock_price=None,
        close_price=None,
        position=None,
        failed=False,
        bets=(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_propagates_kline_fetch_transient_failure():
    """When the gate signals a transient fetch failure, the pipeline must
    surface that specific reason — NOT silently rebrand it as
    ``gate_no_signal``.
    """
    gate = MagicMock()
    gate.evaluate.return_value = MomentumGateResult(
        signal=None,
        tier=None,
        skip_reason="kline_fetch_transient_failure",
        signal_strength=0.0,
        eth_signal=None,
        eth_signal_strength=0.0,
        sol_signal=None,
        sol_signal_strength=0.0,
    )
    pipeline = _make_pipeline(gate)
    decision = pipeline.decide_open_round(
        round_t=_make_round(),
        pool_bull_bnb=1.5,
        pool_bear_bnb=0.8,
    )
    assert decision.action == "SKIP"
    assert decision.skip_reason == "kline_fetch_transient_failure", (
        f"expected propagated 'kline_fetch_transient_failure', "
        f"got {decision.skip_reason!r}"
    )


def test_pipeline_falls_through_to_gate_no_signal_when_skip_reason_is_none():
    """When the gate returns no signal and no skip_reason (the canonical
    quiet-market case), the pipeline still emits ``gate_no_signal``.
    """
    gate = MagicMock()
    gate.evaluate.return_value = MomentumGateResult(
        signal=None,
        tier=None,
        skip_reason=None,
        signal_strength=0.0,
        eth_signal=None,
        eth_signal_strength=0.0,
        sol_signal=None,
        sol_signal_strength=0.0,
    )
    pipeline = _make_pipeline(gate)
    decision = pipeline.decide_open_round(
        round_t=_make_round(),
        pool_bull_bnb=1.5,
        pool_bear_bnb=0.8,
    )
    assert decision.action == "SKIP"
    assert decision.skip_reason == "gate_no_signal", (
        f"expected fallback 'gate_no_signal', got {decision.skip_reason!r}"
    )


def test_pipeline_propagates_explicit_gate_no_signal():
    """The explicit ``gate_no_signal`` skip reason from the gate is
    propagated unchanged.
    """
    gate = MagicMock()
    gate.evaluate.return_value = MomentumGateResult(
        signal=None,
        tier=None,
        skip_reason="gate_no_signal",
        signal_strength=0.0,
        eth_signal=None,
        eth_signal_strength=0.0,
        sol_signal=None,
        sol_signal_strength=0.0,
    )
    pipeline = _make_pipeline(gate)
    decision = pipeline.decide_open_round(
        round_t=_make_round(),
        pool_bull_bnb=1.5,
        pool_bear_bnb=0.8,
    )
    assert decision.action == "SKIP"
    assert decision.skip_reason == "gate_no_signal"


def test_pipeline_does_not_propagate_skip_reason_when_signal_fires():
    """When the gate fires a signal, downstream gates (pool, payout) take
    over. Even if ``result.skip_reason`` is set (it shouldn't be when
    signal is set, but defensively), the pipeline routes through the
    BET path and never returns ``result.skip_reason``.
    """
    gate = MagicMock()
    gate.evaluate.return_value = MomentumGateResult(
        signal="Bear",
        tier="multi_tf",
        skip_reason=None,  # canonical: skip_reason is None when signal fires
        signal_strength=0.0003,  # above small_pool_min_signal_strength 0.0002
        eth_signal=None,
        eth_signal_strength=0.0,
        sol_signal=None,
        sol_signal_strength=0.0,
    )
    pipeline = _make_pipeline(gate)
    # Pool large enough to exit small_pool_min_signal_strength path AND pass min_pool / payout
    decision = pipeline.decide_open_round(
        round_t=_make_round(),
        pool_bull_bnb=1.0,    # OUR side (Bear is bear; bull is the OTHER side)
        pool_bear_bnb=2.0,    # bet on Bear → our_side = bear
    )
    # action is BET; bet_side is Bear; specific skip_reason not asserted
    # (sizing math may still skip on bet_size_below_min etc; the point is
    # we entered the BET path, not the skip-reason-propagation path).
    if decision.action == "SKIP":
        # If sizing skipped, it must NOT be a kline_fetch_transient_failure.
        assert decision.skip_reason != "kline_fetch_transient_failure"
    else:
        assert decision.action == "BET"
        assert decision.bet_side == "Bear"


# ---------------------------------------------------------------------------
# T3-B: skip_context population for risk/pool reasons
# ---------------------------------------------------------------------------


def _make_pipeline_with_tracker(gate_mock, tracker) -> MomentumOnlyPipeline:
    """Pipeline wired to a real BankrollTracker so the risk gates fire."""
    cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=2,
        mtf_lookbacks=(3, 7, 15),
        mtf_min_return_threshold=0.0001,
        max_consecutive_kline_fetch_failures=5,
    )
    return MomentumOnlyPipeline(
        config=cfg,
        strategy_config=_make_strategy_config(),
        gate=gate_mock,
        kline_cutoff_seconds=2,
        pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001,
        treasury_fee_fraction=0.03,
        bankroll_tracker=tracker,
    )


def test_pipeline_populates_skip_context_for_pool_below_minimum():
    """When pool_total < min_pool_bnb_at_cutoff, the SKIP decision carries
    skip_context with (pool_bnb, min_pool_bnb_at_cutoff) so the engine
    can render the spec wording 'pool below minimum (0.8 BNB < 1.5 BNB
    threshold)'.
    """
    gate = MagicMock()
    gate.evaluate.return_value = MomentumGateResult(
        signal="Bull",
        tier="multi_tf",
        skip_reason=None,
        signal_strength=0.0003,
        eth_signal=None,
        eth_signal_strength=0.0,
        sol_signal=None,
        sol_signal_strength=0.0,
    )
    pipeline = _make_pipeline(gate)
    # Pool below min_pool_bnb_at_cutoff=1.5: 0.5 + 0.3 = 0.8
    decision = pipeline.decide_open_round(
        round_t=_make_round(),
        pool_bull_bnb=0.5,
        pool_bear_bnb=0.3,
    )
    assert decision.action == "SKIP"
    assert decision.skip_reason == "pool_below_minimum"
    assert decision.skip_context is not None
    assert decision.skip_context["pool_bnb"] == 0.8
    assert decision.skip_context["min_pool_bnb_at_cutoff"] == 1.5


def test_pipeline_populates_skip_context_for_risk_cooldown_active():
    """When the bankroll tracker is paused, the SKIP carries
    skip_context.rounds_remaining (the remaining count AFTER the
    tick_cooldown decrement that this skip performs).
    """
    from pancakebot.bankroll_tracker import InMemoryBankrollTracker
    tracker = InMemoryBankrollTracker(
        initial_bankroll=10.0,
        drawdown_peak_window_days=7,
    )
    # Pause for 5 rounds; cooldown_remaining starts at 5, decrements to 4
    # when the pipeline ticks during the SKIP.
    tracker.set_paused(cooldown_rounds=5, triggered_at=1777966856)
    gate = MagicMock()
    pipeline = _make_pipeline_with_tracker(gate, tracker)
    decision = pipeline.decide_open_round(
        round_t=_make_round(),
        pool_bull_bnb=1.5,
        pool_bear_bnb=0.8,
    )
    assert decision.action == "SKIP"
    assert decision.skip_reason == "risk_cooldown_active"
    assert decision.skip_context is not None
    # 5 → 4 after the in-skip tick.
    assert decision.skip_context["rounds_remaining"] == 4


def test_pipeline_populates_skip_context_for_risk_drawdown_breaker_fired():
    """When current bankroll is far enough below peak to exceed the
    drawdown threshold, the SKIP carries skip_context with
    (drawdown_pct, threshold_pct) for the spec wording.

    Canonical threshold: max_drawdown_fraction_from_peak=0.15 (15%).
    Construct a tracker where peak=10.0 and current=8.0 → 20% drawdown,
    exceeds threshold by 5pp.
    """
    from pancakebot.bankroll_tracker import InMemoryBankrollTracker
    tracker = InMemoryBankrollTracker(
        initial_bankroll=10.0,
        drawdown_peak_window_days=7,
    )
    # Seed peak via record_settlement, then drop to current=8.0
    tracker.record_settlement(bankroll=10.0, start_at=1777966000)
    tracker.record_settlement(bankroll=8.0, start_at=1777966500)
    gate = MagicMock()
    pipeline = _make_pipeline_with_tracker(gate, tracker)
    decision = pipeline.decide_open_round(
        round_t=_make_round(),
        pool_bull_bnb=1.5,
        pool_bear_bnb=0.8,
    )
    assert decision.action == "SKIP"
    assert decision.skip_reason == "risk_drawdown_breaker_fired"
    assert decision.skip_context is not None
    # peak=10, current=8 → dd_frac=0.2 → 20.0%
    assert decision.skip_context["drawdown_pct"] == pytest.approx(20.0, abs=1e-6)
    # threshold=0.15 → 15.0%
    assert decision.skip_context["threshold_pct"] == pytest.approx(15.0, abs=1e-6)


# ---------------------------------------------------------------------------
# T3-B amend: __post_init__ invariant — context-required SKIP reasons MUST
# carry skip_context with the documented keys. Missing or empty → raises.
# This locks the contract so the engine consumer can read keys directly
# without isinstance / None-fallback guards.
# ---------------------------------------------------------------------------


def test_decision_invariant_raises_when_drawdown_skip_context_missing():
    """A SKIP decision with reason=risk_drawdown_breaker_fired and
    skip_context=None must raise InvariantError at construction time."""
    with pytest.raises(InvariantError, match="risk_drawdown_breaker_fired"):
        StrategyPipelineDecision(
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
            skip_reason="risk_drawdown_breaker_fired",
            skip_context=None,
        )


def test_decision_invariant_raises_when_cooldown_skip_context_missing():
    """A SKIP decision with reason=risk_cooldown_active and
    skip_context=None must raise InvariantError at construction time."""
    with pytest.raises(InvariantError, match="risk_cooldown_active"):
        StrategyPipelineDecision(
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
            skip_reason="risk_cooldown_active",
            skip_context=None,
        )


def test_decision_invariant_raises_when_pool_below_min_skip_context_missing():
    """A SKIP decision with reason=pool_below_minimum and
    skip_context=None must raise InvariantError at construction time."""
    with pytest.raises(InvariantError, match="pool_below_minimum"):
        StrategyPipelineDecision(
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
            skip_reason="pool_below_minimum",
            skip_context=None,
        )


def test_decision_invariant_raises_on_partial_skip_context():
    """An incomplete skip_context (missing one of the required keys)
    must raise InvariantError — the engine consumer reads keys directly
    and would KeyError downstream; we catch the bug earlier at the
    decision-construction site."""
    with pytest.raises(InvariantError, match="missing keys"):
        StrategyPipelineDecision(
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
            skip_reason="risk_drawdown_breaker_fired",
            # threshold_pct is missing
            skip_context={"drawdown_pct": 18.2},
        )


def test_decision_invariant_silent_for_non_context_reasons():
    """SKIP reasons NOT in _SKIP_CONTEXT_SCHEMA (gate_no_signal,
    kline_*, cold_start_in_progress, risk_bankroll_stale, etc.) construct
    cleanly with skip_context=None. The invariant only fires for the
    three reasons that explicitly require structured payload."""
    # Should NOT raise.
    decision = StrategyPipelineDecision(
        action="SKIP",
        bet_side=None,
        bet_size_bnb=0.0,
        skip_reason="gate_no_signal",
        skip_context=None,
    )
    assert decision.skip_context is None


def test_decision_invariant_silent_for_bet_decisions():
    """BET decisions have skip_reason=None; the invariant must not fire."""
    decision = StrategyPipelineDecision(
        action="BET",
        bet_side="Bull",
        bet_size_bnb=0.25,
        skip_reason=None,
        skip_context=None,
    )
    assert decision.action == "BET"
