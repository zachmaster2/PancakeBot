"""Compute win/loss/refund settlement for a bet against on-chain round data or a closed Round.

Includes the bet's self-impact on the winning-side denominator and subtracts claim gas.
"""
from __future__ import annotations

from dataclasses import dataclass

from pancakebot.constants import BNB_WEI, GAS_COST_CLAIM_BNB
from pancakebot.types import Round
from pancakebot.pool_amounts import compute_pool_amounts_wei
from pancakebot.util import InvariantError


def settle_from_round_data(
    *,
    bet_bnb: float,
    bet_side: str,
    lock_price_usd: float,
    close_price_usd: float,
    bull_amount_wei: int,
    bear_amount_wei: int,
    oracle_called: bool,
    treasury_fee_fraction: float,
) -> "SettlementResult":
    """Compute settlement from on-chain round data (no bets list needed).

    Used by dry-mode settlement via contract RPC (replaces settle_bet_against_closed_round
    in the live/dry loop where bets are not available).
    """
    if bet_bnb < 0.0:
        raise InvariantError("settle_bet_bnb_negative")
    if not (0.0 <= treasury_fee_fraction < 1.0):
        raise InvariantError("settle_treasury_fee_fraction_out_of_range")

    bet_side_u = bet_side.upper()
    if bet_side_u not in ("BULL", "BEAR"):
        raise InvariantError("settle_bet_side_invalid")

    if not oracle_called:
        return SettlementResult(outcome="refund", credit_bnb=bet_bnb - GAS_COST_CLAIM_BNB, payout_multiple_after_fee=0.0)

    if close_price_usd > lock_price_usd:
        winner_u = "BULL"
    elif close_price_usd < lock_price_usd:
        winner_u = "BEAR"
    else:
        return SettlementResult(outcome="refund", credit_bnb=bet_bnb - GAS_COST_CLAIM_BNB, payout_multiple_after_fee=0.0)

    if winner_u != bet_side_u:
        return SettlementResult(outcome="loss", credit_bnb=0.0, payout_multiple_after_fee=0.0)

    bull_bnb = bull_amount_wei / BNB_WEI
    bear_bnb = bear_amount_wei / BNB_WEI

    # Apply our simulated bet impact to pools.
    bull_after = bull_bnb + (bet_bnb if bet_side_u == "BULL" else 0.0)
    bear_after = bear_bnb + (bet_bnb if bet_side_u == "BEAR" else 0.0)
    total_after = bull_after + bear_after

    denom = bull_after if bet_side_u == "BULL" else bear_after
    if denom <= 0.0 or total_after <= 0.0:
        return SettlementResult(outcome="win", credit_bnb=-GAS_COST_CLAIM_BNB, payout_multiple_after_fee=0.0)

    mult = (total_after * (1.0 - treasury_fee_fraction)) / denom
    credit = bet_bnb * mult - GAS_COST_CLAIM_BNB
    return SettlementResult(outcome="win", credit_bnb=credit, payout_multiple_after_fee=mult)


@dataclass(frozen=True, slots=True)
class SettlementResult:
    outcome: str  # "win" | "loss" | "refund"
    credit_bnb: float  # amount credited to bankroll AFTER close (net of claim gas)
    payout_multiple_after_fee: float  # 0 for loss/refund


def settle_bet_against_closed_round(
    *,
    bet_bnb: float,
    bet_side: str,
    round_closed: Round,
    treasury_fee_fraction: float,
) -> SettlementResult:
    """Compute the settlement credit for a bet, using only the closed round data.

    This is used by:
      - dry mode: simulating claim settlement
      - backtest: deterministic replay (no RPC calls)

    Convention:
      - Bet principal and bet gas are paid at bet-time (outside this function).
      - This function returns the net credit applied at claim-time:
          - WIN: bet_bnb * payout_multiple_after_fee - GAS_COST_CLAIM_BNB
          - REFUND (failed): bet_bnb - GAS_COST_CLAIM_BNB
          - LOSS: 0.0

    Important (impact-aware):
      - Historical closed-round pools do NOT include our simulated bet.
      - To match live execution, settlement payout math MUST include the bet's impact
        on the final pools (total and winner-side denominator).
    """
    if bet_bnb < 0.0:
        raise InvariantError("settle_bet_bnb_negative")
    if round_closed.position is None:
        raise InvariantError("settle_round_not_closed")
    if not (0.0 <= treasury_fee_fraction < 1.0):
        raise InvariantError("settle_treasury_fee_fraction_out_of_range")

    bet_side_u = bet_side.upper()
    if bet_side_u not in ("BULL", "BEAR"):
        raise InvariantError("settle_bet_side_invalid")

    winner_u = round_closed.position.upper()

    # Use ALL bets in the round (no timestamp filter) -- matches on-chain
    # settlement which uses the final pool totals regardless of bet timing.
    pools_wei = compute_pool_amounts_wei(bets=round_closed.bets)
    bull_pool_bnb = pools_wei.bull_wei / BNB_WEI
    bear_pool_bnb = pools_wei.bear_wei / BNB_WEI

    if round_closed.failed:
        return SettlementResult(outcome="refund", credit_bnb=bet_bnb - GAS_COST_CLAIM_BNB, payout_multiple_after_fee=0.0)

    if winner_u not in ("BULL", "BEAR", "HOUSE"):
        raise InvariantError("settle_winner_invalid")

    if winner_u != bet_side_u:
        return SettlementResult(outcome="loss", credit_bnb=0.0, payout_multiple_after_fee=0.0)

    # Apply our simulated bet impact to pools (live-consistent settlement math).
    bull_after = bull_pool_bnb + (bet_bnb if bet_side_u == "BULL" else 0.0)
    bear_after = bear_pool_bnb + (bet_bnb if bet_side_u == "BEAR" else 0.0)
    total_after = bull_after + bear_after

    denom = bull_after if bet_side_u == "BULL" else bear_after
    if denom <= 0.0 or total_after <= 0.0:
        return SettlementResult(outcome="win", credit_bnb=-GAS_COST_CLAIM_BNB, payout_multiple_after_fee=0.0)

    mult = (total_after * (1.0 - treasury_fee_fraction)) / denom
    credit = bet_bnb * mult - GAS_COST_CLAIM_BNB
    return SettlementResult(outcome="win", credit_bnb=credit, payout_multiple_after_fee=mult)
