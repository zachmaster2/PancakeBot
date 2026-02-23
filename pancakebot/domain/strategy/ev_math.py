from __future__ import annotations

from dataclasses import dataclass

from pancakebot.core.errors import InvariantError


@dataclass(frozen=True, slots=True)
class ChainPolicyParams:
    """Parameters sourced from chain / locked constants.

    - min_bet_amount: on-chain minBetAmount (BNB)
    - treasury_fee_rate: treasuryFee expressed as a fraction in [0,1)
    - gas_bet_bnb: deterministic bet gas cost (BNB) used for EV/backtest accounting
    - gas_claim_bnb: deterministic claim gas cost (BNB) subtracted on win/refund paths
      to match settlement accounting.
    """

    min_bet_amount: float
    treasury_fee_rate: float
    gas_bet_bnb: float
    gas_claim_bnb: float


def _require_finite_nonneg(x: float, name: str) -> None:
    if not (x >= 0.0):
        raise InvariantError(f"{name}_must_be_nonnegative")
    if x != x or x == float("inf") or x == float("-inf"):
        raise InvariantError(f"{name}_not_finite")


def _require_fee_fraction(x: float) -> None:
    if not (0.0 <= x < 1.0):
        raise InvariantError("treasury_fee_fraction_out_of_range_0_1")


def post_impact_payout_multiple(
    *,
    bet_side: str,
    bet_bnb: float,
    final_bull_bnb: float,
    final_bear_bnb: float,
    treasury_fee_fraction: float,
) -> float:
    """Return the post-impact payout multiple for the chosen side.

    This is the per-unit gross payout multiple including stake return, net of treasury fee:

      payout_multiple =
        ( (adj_final_total_bnb * (1 - fee)) / adj_final_side_bnb )

    where adj_final_* include the candidate bet_size_bnb on the chosen side.
    """
    _require_finite_nonneg(final_bull_bnb, "final_bull_bnb")
    _require_finite_nonneg(final_bear_bnb, "final_bear_bnb")
    _require_fee_fraction(float(treasury_fee_fraction))

    if bet_bnb < 0.0:
        raise InvariantError("bet_bnb_negative")
    if bet_side not in ("Bull", "Bear"):
        raise InvariantError("bet_side_invalid")

    bull_after = float(final_bull_bnb) + (float(bet_bnb) if bet_side == "Bull" else 0.0)
    bear_after = float(final_bear_bnb) + (float(bet_bnb) if bet_side == "Bear" else 0.0)

    total_after = float(bull_after) + float(bear_after)
    side_after = float(bull_after) if bet_side == "Bull" else float(bear_after)

    if not (side_after > 0.0):
        raise InvariantError("side_pool_after_bet_nonpositive")
    if not (total_after > 0.0):
        raise InvariantError("total_pool_after_bet_nonpositive")

    effective_total = float(total_after) * (1.0 - float(treasury_fee_fraction))
    return float(effective_total / float(side_after))


def ev_for_side(
    *,
    bet_side: str,
    p_win: float,
    bet_bnb: float,
    final_bull_bnb: float,
    final_bear_bnb: float,
    chain: ChainPolicyParams,
) -> tuple[float, float]:
    """Return (EV_bnb, payout_multiple) for the given side and stake.

    Canonical EV (settlement-consistent):
      - impact-adjusted pools
      - subtract bet gas in all outcomes
      - subtract claim gas on win path (same convention as settlement credit)
    """
    if not (0.0 <= p_win <= 1.0):
        raise InvariantError("p_win_out_of_range")
    if bet_bnb < 0.0:
        raise InvariantError("bet_bnb_negative")

    _require_finite_nonneg(chain.gas_bet_bnb, "gas_bet_bnb")
    _require_finite_nonneg(chain.gas_claim_bnb, "gas_claim_bnb")
    if not (chain.min_bet_amount > 0.0):
        raise InvariantError("min_bet_amount_must_be_positive")

    payout_mult = post_impact_payout_multiple(
        bet_side=str(bet_side),
        bet_bnb=float(bet_bnb),
        final_bull_bnb=float(final_bull_bnb),
        final_bear_bnb=float(final_bear_bnb),
        treasury_fee_fraction=float(chain.treasury_fee_rate),
    )

    # profit_win_bnb = gross_payout_bnb - bet_size_bnb - bet_gas - claim_gas
    profit_if_win = (
        float(bet_bnb) * (float(payout_mult) - 1.0)
        - float(chain.gas_bet_bnb)
        - float(chain.gas_claim_bnb)
    )
    profit_if_lose = -float(bet_bnb) - float(chain.gas_bet_bnb)

    ev = float(p_win) * float(profit_if_win) + (1.0 - float(p_win)) * float(profit_if_lose)
    return float(ev), float(payout_mult)
