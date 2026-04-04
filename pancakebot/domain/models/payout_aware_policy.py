from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei_at_or_before
from pancakebot.domain.types import Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round


@dataclass(frozen=True, slots=True)
class PayoutAwarePolicyTraceRow:
    target_epoch: int
    predicted_ev_bull: float
    predicted_ev_bear: float
    bull_threshold: float
    bear_threshold: float
    action: str
    selected_side: str | None
    selected_predicted_ev: float | None
    realized_profit_bnb: float
    cumulative_profit_bnb: float
    bankroll_bnb: float
    outcome: str | None


@dataclass(frozen=True, slots=True)
class PayoutAwarePolicyResult:
    num_rounds: int
    num_bets: int
    num_bull_bets: int
    num_bear_bets: int
    num_wins: int
    num_losses: int
    num_refunds: int
    num_skips_below_threshold: int
    num_skips_insufficient_bankroll: int
    bull_threshold: float
    bear_threshold: float
    bet_size_bnb: float
    initial_bankroll_bnb: float
    final_bankroll_bnb: float
    net_profit_bnb: float
    gross_profit_bnb: float
    gross_loss_bnb: float
    max_drawdown_bnb: float
    bet_rate: float
    win_rate: float
    profit_per_500_bnb: float
    selected_mean_predicted_ev: float
    selected_min_predicted_ev: float | None
    selected_max_predicted_ev: float | None


@dataclass(frozen=True, slots=True)
class PayoutAwareThresholdChoice:
    bull_threshold: float
    bear_threshold: float
    result: PayoutAwarePolicyResult
    met_min_bet_rate: bool


def realized_profit_for_side(
    *,
    round_closed: Round,
    bet_size_bnb: float,
    bet_side: str,
    treasury_fee_fraction: float,
) -> float:
    if float(bet_size_bnb) <= 0.0:
        raise InvariantError("payout_aware_realized_profit_bet_size_nonpositive")
    outcome = settle_bet_against_closed_round(
        bet_bnb=float(bet_size_bnb),
        bet_side=str(bet_side),
        round_closed=round_closed,
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    return float(outcome.credit_bnb) - float(bet_size_bnb) - float(GAS_COST_BET_BNB)


def naive_cutoff_profit_if_side_wins(
    *,
    round_closed: Round,
    bet_size_bnb: float,
    bet_side: str,
    treasury_fee_fraction: float,
    cutoff_seconds: int,
) -> float:
    if float(bet_size_bnb) <= 0.0:
        raise InvariantError("payout_aware_naive_cutoff_profit_bet_size_nonpositive")
    if round_closed.lock_at is None:
        raise InvariantError("payout_aware_naive_cutoff_profit_round_unlocked")
    if int(cutoff_seconds) < 0:
        raise InvariantError("payout_aware_naive_cutoff_profit_cutoff_seconds_negative")
    bet_side_u = str(bet_side).upper()
    if bet_side_u not in ("BULL", "BEAR"):
        raise InvariantError("payout_aware_naive_cutoff_profit_side_invalid")
    cutoff_ts = int(round_closed.lock_at) - int(cutoff_seconds)
    pools_wei = compute_pool_amounts_wei_at_or_before(
        bets=round_closed.bets,
        cutoff_ts=int(cutoff_ts),
    )
    bull_pool_bnb = float(pools_wei.bull_wei) / float(BNB_WEI)
    bear_pool_bnb = float(pools_wei.bear_wei) / float(BNB_WEI)
    bull_after = float(bull_pool_bnb) + (float(bet_size_bnb) if bet_side_u == "BULL" else 0.0)
    bear_after = float(bear_pool_bnb) + (float(bet_size_bnb) if bet_side_u == "BEAR" else 0.0)
    total_after = float(bull_after) + float(bear_after)
    denom = float(bull_after) if bet_side_u == "BULL" else float(bear_after)
    if float(denom) <= 0.0 or float(total_after) <= 0.0:
        return -float(GAS_COST_CLAIM_BNB) - float(GAS_COST_BET_BNB)
    payout_multiple = float(total_after) * (1.0 - float(treasury_fee_fraction)) / float(denom)
    credit_bnb = float(bet_size_bnb) * float(payout_multiple) - float(GAS_COST_CLAIM_BNB)
    return float(credit_bnb) - float(bet_size_bnb) - float(GAS_COST_BET_BNB)


def simulate_payout_aware_policy(
    *,
    rounds: Sequence[Round],
    predicted_ev_bull: np.ndarray,
    predicted_ev_bear: np.ndarray,
    bull_threshold: float,
    bear_threshold: float,
    bet_size_bnb: float,
    initial_bankroll_bnb: float,
    treasury_fee_fraction: float,
) -> tuple[PayoutAwarePolicyResult, list[PayoutAwarePolicyTraceRow]]:
    sim_rounds = list(rounds)
    ev_bull = np.asarray(predicted_ev_bull, dtype=np.float32)
    ev_bear = np.asarray(predicted_ev_bear, dtype=np.float32)
    if ev_bull.ndim != 1 or ev_bear.ndim != 1:
        raise InvariantError("payout_aware_policy_ev_rank_invalid")
    if len(sim_rounds) != len(ev_bull) or len(sim_rounds) != len(ev_bear):
        raise InvariantError("payout_aware_policy_len_mismatch")
    if len(sim_rounds) <= 0:
        raise InvariantError("payout_aware_policy_rounds_empty")
    if float(bet_size_bnb) <= 0.0:
        raise InvariantError("payout_aware_policy_bet_size_nonpositive")
    if float(initial_bankroll_bnb) <= 0.0:
        raise InvariantError("payout_aware_policy_initial_bankroll_nonpositive")

    bankroll = float(initial_bankroll_bnb)
    peak_bankroll = float(bankroll)
    max_drawdown = 0.0
    num_bets = 0
    num_bull_bets = 0
    num_bear_bets = 0
    num_wins = 0
    num_losses = 0
    num_refunds = 0
    num_skips_below_threshold = 0
    num_skips_insufficient_bankroll = 0
    gross_profit_bnb = 0.0
    gross_loss_bnb = 0.0
    selected_predicted_evs: list[float] = []
    traces: list[PayoutAwarePolicyTraceRow] = []

    for round_t, ev_bull_raw, ev_bear_raw in zip(sim_rounds, ev_bull, ev_bear, strict=True):
        ev_bull_value = float(ev_bull_raw)
        ev_bear_value = float(ev_bear_raw)
        action = "skip_below_threshold"
        selected_side: str | None = None
        selected_predicted_ev: float | None = None
        realized_profit = 0.0
        realized_outcome: str | None = None

        bull_allowed = float(ev_bull_value) >= float(bull_threshold)
        bear_allowed = float(ev_bear_value) >= float(bear_threshold)
        if bull_allowed or bear_allowed:
            if bull_allowed and bear_allowed:
                if float(ev_bull_value) >= float(ev_bear_value):
                    selected_side = "Bull"
                    selected_predicted_ev = float(ev_bull_value)
                else:
                    selected_side = "Bear"
                    selected_predicted_ev = float(ev_bear_value)
            elif bull_allowed:
                selected_side = "Bull"
                selected_predicted_ev = float(ev_bull_value)
            else:
                selected_side = "Bear"
                selected_predicted_ev = float(ev_bear_value)

        if selected_side is None:
            num_skips_below_threshold += 1
        else:
            total_cost_bnb = float(bet_size_bnb) + float(GAS_COST_BET_BNB)
            if float(bankroll) < float(total_cost_bnb):
                action = "skip_insufficient_bankroll"
                num_skips_insufficient_bankroll += 1
                selected_side = None
                selected_predicted_ev = None
            else:
                action = f"bet_{str(selected_side).lower()}"
                num_bets += 1
                if str(selected_side) == "Bull":
                    num_bull_bets += 1
                else:
                    num_bear_bets += 1
                selected_predicted_evs.append(float(selected_predicted_ev))
                bankroll -= float(total_cost_bnb)
                outcome = settle_bet_against_closed_round(
                    bet_bnb=float(bet_size_bnb),
                    bet_side=str(selected_side),
                    round_closed=round_t,
                    treasury_fee_fraction=float(treasury_fee_fraction),
                )
                bankroll += float(outcome.credit_bnb)
                realized_profit = float(outcome.credit_bnb) - float(bet_size_bnb) - float(GAS_COST_BET_BNB)
                realized_outcome = str(outcome.outcome)
                if str(outcome.outcome) == "win":
                    num_wins += 1
                elif str(outcome.outcome) == "refund":
                    num_refunds += 1
                else:
                    num_losses += 1
                if float(realized_profit) > 0.0:
                    gross_profit_bnb += float(realized_profit)
                elif float(realized_profit) < 0.0:
                    gross_loss_bnb += float(-float(realized_profit))

        peak_bankroll = max(float(peak_bankroll), float(bankroll))
        max_drawdown = max(float(max_drawdown), float(peak_bankroll) - float(bankroll))
        traces.append(
            PayoutAwarePolicyTraceRow(
                target_epoch=int(round_t.epoch),
                predicted_ev_bull=float(ev_bull_value),
                predicted_ev_bear=float(ev_bear_value),
                bull_threshold=float(bull_threshold),
                bear_threshold=float(bear_threshold),
                action=str(action),
                selected_side=None if selected_side is None else str(selected_side),
                selected_predicted_ev=selected_predicted_ev,
                realized_profit_bnb=float(realized_profit),
                cumulative_profit_bnb=float(bankroll) - float(initial_bankroll_bnb),
                bankroll_bnb=float(bankroll),
                outcome=realized_outcome,
            )
        )

    selected_mean_predicted_ev = (
        float(np.mean(np.asarray(selected_predicted_evs, dtype=np.float32)))
        if selected_predicted_evs
        else 0.0
    )
    selected_min_predicted_ev = (
        float(np.min(np.asarray(selected_predicted_evs, dtype=np.float32)))
        if selected_predicted_evs
        else None
    )
    selected_max_predicted_ev = (
        float(np.max(np.asarray(selected_predicted_evs, dtype=np.float32)))
        if selected_predicted_evs
        else None
    )
    num_rounds = int(len(sim_rounds))
    net_profit_bnb = float(bankroll) - float(initial_bankroll_bnb)
    result = PayoutAwarePolicyResult(
        num_rounds=int(num_rounds),
        num_bets=int(num_bets),
        num_bull_bets=int(num_bull_bets),
        num_bear_bets=int(num_bear_bets),
        num_wins=int(num_wins),
        num_losses=int(num_losses),
        num_refunds=int(num_refunds),
        num_skips_below_threshold=int(num_skips_below_threshold),
        num_skips_insufficient_bankroll=int(num_skips_insufficient_bankroll),
        bull_threshold=float(bull_threshold),
        bear_threshold=float(bear_threshold),
        bet_size_bnb=float(bet_size_bnb),
        initial_bankroll_bnb=float(initial_bankroll_bnb),
        final_bankroll_bnb=float(bankroll),
        net_profit_bnb=float(net_profit_bnb),
        gross_profit_bnb=float(gross_profit_bnb),
        gross_loss_bnb=float(gross_loss_bnb),
        max_drawdown_bnb=float(max_drawdown),
        bet_rate=float(num_bets / float(num_rounds)),
        win_rate=0.0 if int(num_bets) <= 0 else float(num_wins / float(num_bets)),
        profit_per_500_bnb=float(net_profit_bnb) * 500.0 / float(num_rounds),
        selected_mean_predicted_ev=float(selected_mean_predicted_ev),
        selected_min_predicted_ev=selected_min_predicted_ev,
        selected_max_predicted_ev=selected_max_predicted_ev,
    )
    return result, traces


def tune_side_thresholds(
    *,
    rounds: Sequence[Round],
    predicted_ev_bull: np.ndarray,
    predicted_ev_bear: np.ndarray,
    threshold_grid: Sequence[float],
    bet_size_bnb: float,
    initial_bankroll_bnb: float,
    treasury_fee_fraction: float,
    min_bet_rate: float,
) -> PayoutAwareThresholdChoice:
    grid = [float(value) for value in threshold_grid]
    if not grid:
        raise InvariantError("payout_aware_policy_threshold_grid_empty")
    feasible_choices: list[PayoutAwareThresholdChoice] = []
    all_choices: list[PayoutAwareThresholdChoice] = []
    for bull_threshold in grid:
        for bear_threshold in grid:
            result, _ = simulate_payout_aware_policy(
                rounds=rounds,
                predicted_ev_bull=predicted_ev_bull,
                predicted_ev_bear=predicted_ev_bear,
                bull_threshold=float(bull_threshold),
                bear_threshold=float(bear_threshold),
                bet_size_bnb=float(bet_size_bnb),
                initial_bankroll_bnb=float(initial_bankroll_bnb),
                treasury_fee_fraction=float(treasury_fee_fraction),
            )
            choice = PayoutAwareThresholdChoice(
                bull_threshold=float(bull_threshold),
                bear_threshold=float(bear_threshold),
                result=result,
                met_min_bet_rate=bool(float(result.bet_rate) >= float(min_bet_rate)),
            )
            all_choices.append(choice)
            if choice.met_min_bet_rate:
                feasible_choices.append(choice)
    pool = feasible_choices if feasible_choices else all_choices
    return max(
        pool,
        key=lambda item: (
            float(item.result.net_profit_bnb),
            float(item.result.profit_per_500_bnb),
            float(item.result.bet_rate),
            -float(item.bull_threshold + item.bear_threshold),
        ),
    )
