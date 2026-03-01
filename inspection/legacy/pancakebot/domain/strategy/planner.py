from __future__ import annotations

import time
from dataclasses import dataclass

from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.domain.types import Kline, Round
from pancakebot.domain.features.feature_builder import build_features, vectorize
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei_at_or_before
from pancakebot.domain.features.schema import FEATURE_SCHEMA, max_required_context_klines_size, max_required_prior_context_rounds_size
from pancakebot.domain.models.walk_forward import WalkForwardState, predict_probabilities, predict_tradeable_probability
from pancakebot.domain.strategy.ev_math import ChainPolicyParams
from pancakebot.domain.strategy.policy import PolicyDecision, decide
from pancakebot.core.errors import InvariantError


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    epoch: int
    lock_ts: int
    cutoff_ts: int

    # Observed at decision time (hard): pools at or before cutoff_ts.
    cutoff_total_bnb: float
    cutoff_bull_bnb: float
    cutoff_bear_bnb: float

    # Model inputs.
    x_price: list[list[float]]
    x_pool: list[list[float]]
    feat_ms: int


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    epoch: int

    # Probabilities:
    # - p_final: calibrated probability used for EV/sizing
    # - p_tradeable: predictability gate score
    p_final: float
    p_tradeable: float

    # Observed pools at cutoff.
    cutoff_total_bnb: float
    cutoff_bull_bnb: float
    cutoff_bear_bnb: float

    # Model-predicted primitives (frozen pool forecast contract).
    pred_late_inflow_total_bnb: float
    pred_late_inflow_bull_frac: float

    # Derived canonical pool forecast outputs.
    final_total_bnb: float
    final_bull_bnb: float
    final_bear_bnb: float

    model_ms: int


@dataclass(frozen=True, slots=True)
class BetDecision:
    action: str  # "BET" or "SKIP"
    bet_side: str | None
    amount_bnb: float

    expected_profit_bnb: float
    post_impact_payout_multiple: float | None

    bet_cap_bnb: float
    best_expected_profit_bnb: float

    skip_reason: str | None


def build_inputs(
    *,
    cfg,
    prior_context_rounds: list[Round],
    context_klines: list[Kline],
    target_round: Round,
) -> FeatureBundle:
    """Build cutoff-time features for the target round.

    Canonical v1: prior_context_rounds is the only prior-context input.
    """

    epoch = int(target_round.epoch)
    if target_round.lock_at is None:
        raise InvariantError("target_round_lock_at_missing")
    lock_ts = int(target_round.lock_at)
    if int(lock_ts) <= 0:
        raise InvariantError("target_round_lock_at_invalid")

    cutoff_ts = int(lock_ts) - int(cfg.cutoff_seconds)

    k = int(max_required_prior_context_rounds_size())
    if len(prior_context_rounds) != int(k):
        raise InvariantError(
            f"prior_context_rounds_len_mismatch: got={len(prior_context_rounds)} expected={int(k)}"
        )

    if prior_context_rounds and int(prior_context_rounds[-1].epoch) >= int(target_round.epoch):
        raise InvariantError("prior_context_not_strictly_before_target")

    kk = int(max_required_context_klines_size())
    if len(context_klines) != int(kk):
        raise InvariantError(
            f"context_klines_len_mismatch: got={len(context_klines)} expected={int(kk)}"
        )

    t0 = time.perf_counter()
    x_feats = build_features(
        target_round=target_round,
        prior_context_rounds=prior_context_rounds,
        context_klines=context_klines,
        cutoff_seconds=int(cfg.cutoff_seconds),
    )
    feat_ms = int((time.perf_counter() - t0) * 1000)

    x_price = [vectorize(features=x_feats, schema=FEATURE_SCHEMA)]
    x_pool = x_price

    pools_cutoff_wei = compute_pool_amounts_wei_at_or_before(bets=target_round.bets, cutoff_ts=int(cutoff_ts))
    cutoff_bull_bnb = float(pools_cutoff_wei.bull_wei) / float(BNB_WEI)
    cutoff_bear_bnb = float(pools_cutoff_wei.bear_wei) / float(BNB_WEI)
    cutoff_total_bnb = float(cutoff_bull_bnb) + float(cutoff_bear_bnb)

    return FeatureBundle(
        epoch=int(epoch),
        lock_ts=int(lock_ts),
        cutoff_ts=int(cutoff_ts),
        cutoff_total_bnb=float(cutoff_total_bnb),
        cutoff_bull_bnb=float(cutoff_bull_bnb),
        cutoff_bear_bnb=float(cutoff_bear_bnb),
        x_price=x_price,
        x_pool=x_pool,
        feat_ms=int(feat_ms),
    )


def predict(*, state: WalkForwardState, feats: FeatureBundle) -> PredictionBundle:
    """Predict calibrated probability and pool forecast primitives; derive final pools."""

    if state.models is None:
        raise InvariantError("predict_without_models")

    t0 = time.perf_counter()

    mu = float(state.models.price_model.predict(feats.x_price)[0])
    p_final = predict_probabilities(state=state, mu=float(mu))
    p_tradeable = predict_tradeable_probability(state=state, x_row=list(feats.x_price[0]))

    late_total_pred, late_bull_frac_pred = state.models.pool_model.predict(feats.x_pool)[0]

    late_total_pred = float(late_total_pred)
    late_bull_frac_pred = float(late_bull_frac_pred)

    if late_total_pred < 0.0:
        raise InvariantError("pred_late_inflow_total_negative")
    if not (0.0 <= late_bull_frac_pred <= 1.0):
        raise InvariantError("pred_late_inflow_bull_frac_out_of_range")

    # Derived canonical outputs (frozen):
    #   final_total_bnb = cutoff_total_bnb + pred_late_inflow_total_bnb
    #   final_bull_bnb  = cutoff_bull_bnb  + pred_late_inflow_total_bnb * pred_late_inflow_bull_frac
    #   final_bear_bnb  = cutoff_bear_bnb  + pred_late_inflow_total_bnb * (1 - pred_late_inflow_bull_frac)
    #
    # IMPORTANT: The spec requires the equality invariant:
    #   final_bull_bnb + final_bear_bnb == final_total_bnb
    # We enforce this by construction (not by tolerant comparison).
    late_bull_bnb = float(late_total_pred) * float(late_bull_frac_pred)
    late_bear_bnb = float(late_total_pred) - float(late_bull_bnb)

    final_bull = float(feats.cutoff_bull_bnb) + float(late_bull_bnb)
    final_bear = float(feats.cutoff_bear_bnb) + float(late_bear_bnb)
    final_total = float(final_bull) + float(final_bear)

    # Validity invariants (hard).
    if final_total < float(feats.cutoff_total_bnb):
        raise InvariantError("final_total_below_cutoff_total")
    if final_bull < float(feats.cutoff_bull_bnb):
        raise InvariantError("final_bull_below_cutoff_bull")
    if final_bear < float(feats.cutoff_bear_bnb):
        raise InvariantError("final_bear_below_cutoff_bear")
    if float(final_bull) + float(final_bear) != float(final_total):
        raise InvariantError("final_pool_sum_mismatch")

    model_ms = int((time.perf_counter() - t0) * 1000)

    return PredictionBundle(
        epoch=int(feats.epoch),
        p_final=float(p_final),
        p_tradeable=float(p_tradeable),
        cutoff_total_bnb=float(feats.cutoff_total_bnb),
        cutoff_bull_bnb=float(feats.cutoff_bull_bnb),
        cutoff_bear_bnb=float(feats.cutoff_bear_bnb),
        pred_late_inflow_total_bnb=float(late_total_pred),
        pred_late_inflow_bull_frac=float(late_bull_frac_pred),
        final_total_bnb=float(final_total),
        final_bull_bnb=float(final_bull),
        final_bear_bnb=float(final_bear),
        model_ms=int(model_ms),
    )


def size_bet(*, cfg, pred: PredictionBundle, bankroll_bnb: float) -> BetDecision:
    """Run the canonical impact-aware sizing policy using shared chain parameters."""
    min_bet_amount_bnb = float(getattr(cfg, "min_bet_amount_bnb", 0.0))
    if float(min_bet_amount_bnb) <= 0.0:
        raise InvariantError("min_bet_amount_bnb_missing_or_nonpositive")

    chain = ChainPolicyParams(
        min_bet_amount=float(min_bet_amount_bnb),
        treasury_fee_rate=float(cfg.treasury_fee_fraction),
        gas_bet_bnb=float(GAS_COST_BET_BNB),
        gas_claim_bnb=float(GAS_COST_CLAIM_BNB),
    )

    decision: PolicyDecision = decide(
        epoch=int(pred.epoch),
        p_bull=float(pred.p_final),
        final_bull_bnb=float(pred.final_bull_bnb),
        final_bear_bnb=float(pred.final_bear_bnb),
        bankroll_bnb=float(bankroll_bnb),
        cfg=cfg.policy_cfg,
        chain=chain,
    )

    if decision.action != "BET":
        return BetDecision(
            action="SKIP",
            bet_side=None,
            amount_bnb=0.0,
            expected_profit_bnb=0.0,
            post_impact_payout_multiple=None,
            bet_cap_bnb=float(decision.bet_cap_bnb),
            best_expected_profit_bnb=float(decision.best_expected_profit_bnb),
            skip_reason=str(decision.reason),
        )

    return BetDecision(
        action="BET",
        bet_side=str(decision.bet_side),
        amount_bnb=float(decision.bet_bnb),
        expected_profit_bnb=float(decision.expected_profit_bnb),
        post_impact_payout_multiple=float(decision.post_impact_payout_multiple)
        if decision.post_impact_payout_multiple is not None
        else None,
        bet_cap_bnb=float(decision.bet_cap_bnb),
        best_expected_profit_bnb=float(decision.best_expected_profit_bnb),
        skip_reason=None,
    )

