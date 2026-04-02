from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.types import Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round


@dataclass(frozen=True, slots=True)
class NeuralDirectionThresholdPolicyResult:
    num_rounds: int
    num_bets: int
    num_wins: int
    num_skips_below_threshold: int
    num_skips_insufficient_bankroll: int
    threshold_used: float
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
    selected_mean_confidence: float
    selected_min_confidence: float | None
    selected_max_confidence: float | None


def confidence_threshold_for_target_coverage(
    *,
    chosen_side_confidence: np.ndarray,
    target_coverage_fraction: float,
) -> float:
    confidence = np.asarray(chosen_side_confidence, dtype=np.float32)
    if confidence.ndim != 1:
        raise InvariantError("neural_direction_policy_confidence_rank_invalid")
    if len(confidence) <= 0:
        raise InvariantError("neural_direction_policy_confidence_empty")
    fraction = float(target_coverage_fraction)
    if not (0.0 < float(fraction) <= 1.0):
        raise InvariantError("neural_direction_policy_coverage_fraction_invalid")
    selected_count = max(1, int(np.ceil(float(len(confidence)) * float(fraction))))
    ordered = np.sort(confidence)[::-1]
    return float(ordered[int(selected_count) - 1])


def simulate_confidence_threshold_policy(
    *,
    rounds: Sequence[Round],
    calibrated_bull_probs: np.ndarray,
    threshold: float,
    bet_size_bnb: float,
    initial_bankroll_bnb: float,
    treasury_fee_fraction: float,
) -> NeuralDirectionThresholdPolicyResult:
    sim_rounds = list(rounds)
    probs = np.asarray(calibrated_bull_probs, dtype=np.float32)
    if probs.ndim != 1:
        raise InvariantError("neural_direction_policy_probs_rank_invalid")
    if len(sim_rounds) != len(probs):
        raise InvariantError("neural_direction_policy_len_mismatch")
    if len(sim_rounds) <= 0:
        raise InvariantError("neural_direction_policy_rounds_empty")
    if float(threshold) <= 0.0 or float(threshold) > 1.0:
        raise InvariantError("neural_direction_policy_threshold_out_of_range")
    if float(bet_size_bnb) <= 0.0:
        raise InvariantError("neural_direction_policy_bet_size_nonpositive")
    if float(initial_bankroll_bnb) <= 0.0:
        raise InvariantError("neural_direction_policy_initial_bankroll_nonpositive")

    bankroll = float(initial_bankroll_bnb)
    peak_bankroll = float(bankroll)
    max_drawdown = 0.0
    num_bets = 0
    num_wins = 0
    num_skips_below_threshold = 0
    num_skips_insufficient_bankroll = 0
    gross_profit_bnb = 0.0
    gross_loss_bnb = 0.0
    selected_confidences: list[float] = []

    for round_t, bull_prob in zip(sim_rounds, probs, strict=True):
        p_bull = float(bull_prob)
        if not (0.0 <= float(p_bull) <= 1.0):
            raise InvariantError("neural_direction_policy_prob_out_of_range")
        bet_side = "Bull" if float(p_bull) >= 0.5 else "Bear"
        confidence = float(max(float(p_bull), 1.0 - float(p_bull)))
        if float(confidence) < float(threshold):
            num_skips_below_threshold += 1
            peak_bankroll = max(float(peak_bankroll), float(bankroll))
            max_drawdown = max(float(max_drawdown), float(peak_bankroll) - float(bankroll))
            continue
        total_cost_bnb = float(bet_size_bnb) + float(GAS_COST_BET_BNB)
        if float(bankroll) < float(total_cost_bnb):
            num_skips_insufficient_bankroll += 1
            peak_bankroll = max(float(peak_bankroll), float(bankroll))
            max_drawdown = max(float(max_drawdown), float(peak_bankroll) - float(bankroll))
            continue

        num_bets += 1
        selected_confidences.append(float(confidence))
        bankroll -= float(total_cost_bnb)
        outcome = settle_bet_against_closed_round(
            bet_bnb=float(bet_size_bnb),
            bet_side=str(bet_side),
            round_closed=round_t,
            treasury_fee_fraction=float(treasury_fee_fraction),
        )
        credit_bnb = float(outcome.credit_bnb)
        bankroll += float(credit_bnb)
        profit_bnb = float(credit_bnb) - float(bet_size_bnb) - float(GAS_COST_BET_BNB)
        if str(outcome.outcome) == "win":
            num_wins += 1
        if float(profit_bnb) > 0.0:
            gross_profit_bnb += float(profit_bnb)
        elif float(profit_bnb) < 0.0:
            gross_loss_bnb += float(-float(profit_bnb))

        peak_bankroll = max(float(peak_bankroll), float(bankroll))
        max_drawdown = max(float(max_drawdown), float(peak_bankroll) - float(bankroll))

    selected_mean_confidence = (
        float(np.mean(np.asarray(selected_confidences, dtype=np.float32)))
        if selected_confidences
        else 0.0
    )
    selected_min_confidence = (
        float(np.min(np.asarray(selected_confidences, dtype=np.float32)))
        if selected_confidences
        else None
    )
    selected_max_confidence = (
        float(np.max(np.asarray(selected_confidences, dtype=np.float32)))
        if selected_confidences
        else None
    )
    num_rounds = int(len(sim_rounds))
    net_profit_bnb = float(bankroll) - float(initial_bankroll_bnb)
    return NeuralDirectionThresholdPolicyResult(
        num_rounds=int(num_rounds),
        num_bets=int(num_bets),
        num_wins=int(num_wins),
        num_skips_below_threshold=int(num_skips_below_threshold),
        num_skips_insufficient_bankroll=int(num_skips_insufficient_bankroll),
        threshold_used=float(threshold),
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
        selected_mean_confidence=float(selected_mean_confidence),
        selected_min_confidence=selected_min_confidence,
        selected_max_confidence=selected_max_confidence,
    )
