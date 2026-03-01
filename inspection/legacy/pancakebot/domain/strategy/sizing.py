from __future__ import annotations

from dataclasses import dataclass
import math

from pancakebot.config.policy_config import PolicyConfig
from pancakebot.domain.strategy.ev_math import ChainPolicyParams, ev_for_side, post_impact_payout_multiple
from pancakebot.core.errors import InvariantError


@dataclass(frozen=True, slots=True)
class BetSizingInputs:
    epoch: int
    bankroll_bnb: float
    p_bull: float
    final_bull_bnb: float
    final_bear_bnb: float
    cfg: PolicyConfig
    chain: ChainPolicyParams


@dataclass(frozen=True, slots=True)
class BetSizingDecision:
    epoch: int
    bet_side: str | None  # "Bull" or "Bear" when betting; None when skipping
    bet_bnb: float
    expected_profit_bnb: float
    post_impact_payout_multiple: float | None
    bet_cap_bnb: float
    best_expected_profit_bnb: float
    skip_reason: str | None


def _require_finite_nonneg(x: float, name: str) -> None:
    if not (x >= 0.0):
        raise InvariantError(f"{name}_must_be_nonnegative")
    if x != x or x == float("inf") or x == float("-inf"):
        raise InvariantError(f"{name}_not_finite")


def _kelly_fraction(*, p: float, b: float) -> float:
    """Return full Kelly fraction f_full.

    f_full = (p*b - (1-p)) / b
    """
    if not (0.0 <= p <= 1.0):
        raise InvariantError("p_out_of_range")
    if not (b > 0.0):
        return 0.0

    f = (float(p) * float(b) - (1.0 - float(p))) / float(b)
    if not math.isfinite(float(f)):
        return 0.0
    return float(max(0.0, f))


def _cap_bnb(*, bankroll_bnb: float, final_side_bnb: float, cfg: PolicyConfig) -> float:
    # Caps (policy-configured):
    #  - bankroll fraction cap
    #  - predicted final side pool fraction cap (pre-impact)
    #  - absolute cap
    cap_bankroll = float(cfg.bankroll_cap_fraction) * float(bankroll_bnb)
    cap_pool = float(cfg.pool_cap_fraction) * float(final_side_bnb)
    cap_abs = float(cfg.max_bet_bnb)
    return float(max(0.0, min(cap_bankroll, cap_pool, cap_abs)))


def _gas_adjusted_kelly_bet(
    *,
    p: float,
    b_raw: float,
    bankroll_bnb: float,
    gas_bet_bnb: float,
    gas_claim_bnb: float,
    cfg: PolicyConfig,
    cap_bnb: float,
) -> float:
    """Return Kelly bet size incorporating fixed gas cost into odds.

    We approximate fixed costs via an odds adjustment:
        b_net = b_raw - expected_cost / s
    where expected_cost = gas_bet_bnb + p * gas_claim_bnb
    using a single deterministic refinement step.
    """
    if not (cap_bnb > 0.0):
        return 0.0
    if not (bankroll_bnb > 0.0):
        return 0.0
    if gas_bet_bnb < 0.0:
        raise InvariantError("gas_bet_bnb_negative")
    if gas_claim_bnb < 0.0:
        raise InvariantError("gas_claim_bnb_negative")

    f0 = _kelly_fraction(p=float(p), b=float(b_raw))
    s0 = float(min(float(cfg.kelly_multiplier) * float(f0) * float(bankroll_bnb), float(cap_bnb)))
    if not (s0 > 0.0):
        return 0.0

    expected_cost = float(gas_bet_bnb) + float(p) * float(gas_claim_bnb)
    b_net = float(b_raw) - float(expected_cost) / float(s0)
    f1 = _kelly_fraction(p=float(p), b=float(b_net))
    s1 = float(min(float(cfg.kelly_multiplier) * float(f1) * float(bankroll_bnb), float(cap_bnb)))
    return float(max(0.0, s1))


def _two_pass_kelly_candidate(
    *,
    bet_side: str,
    p_win: float,
    bankroll_bnb: float,
    final_bull_bnb: float,
    final_bear_bnb: float,
    final_side_bnb: float,
    chain: ChainPolicyParams,
    cfg: PolicyConfig,
) -> tuple[float, float, float | None]:
    """Deterministic 2-pass approximation for s when b depends on s.

    Pass 1: assume s=0 for impact; compute payout0 -> b0 -> s1 (gas-adjusted).
    Pass 2: compute payout1 using impact with s1; compute b1 -> s2 (gas-adjusted).

    Returns (s2, ev2, payout2) or (0,0,None) when no bet.
    """
    cap = _cap_bnb(bankroll_bnb=float(bankroll_bnb), final_side_bnb=float(final_side_bnb), cfg=cfg)
    if not (cap > 0.0):
        return 0.0, 0.0, None

    # Pass 1 (impact assumed zero)
    payout0 = post_impact_payout_multiple(
        bet_side=str(bet_side),
        bet_bnb=0.0,
        final_bull_bnb=float(final_bull_bnb),
        final_bear_bnb=float(final_bear_bnb),
        treasury_fee_fraction=float(chain.treasury_fee_rate),
    )
    b0 = float(payout0 - 1.0)

    s1 = _gas_adjusted_kelly_bet(
        p=float(p_win),
        b_raw=float(b0),
        bankroll_bnb=float(bankroll_bnb),
        gas_bet_bnb=float(chain.gas_bet_bnb),
        gas_claim_bnb=float(chain.gas_claim_bnb),
        cfg=cfg,
        cap_bnb=float(cap),
    )
    if not (s1 > 0.0):
        return 0.0, 0.0, None

    # Pass 2 (impact with s1)
    payout1 = post_impact_payout_multiple(
        bet_side=str(bet_side),
        bet_bnb=float(s1),
        final_bull_bnb=float(final_bull_bnb),
        final_bear_bnb=float(final_bear_bnb),
        treasury_fee_fraction=float(chain.treasury_fee_rate),
    )
    b1 = float(payout1 - 1.0)

    s2 = _gas_adjusted_kelly_bet(
        p=float(p_win),
        b_raw=float(b1),
        bankroll_bnb=float(bankroll_bnb),
        gas_bet_bnb=float(chain.gas_bet_bnb),
        gas_claim_bnb=float(chain.gas_claim_bnb),
        cfg=cfg,
        cap_bnb=float(cap),
    )
    if not (s2 > 0.0):
        return 0.0, 0.0, None
    if float(s2) + 1e-18 < float(chain.min_bet_amount):
        return 0.0, 0.0, None

    ev2, payout2 = ev_for_side(
        bet_side=str(bet_side),
        p_win=float(p_win),
        bet_bnb=float(s2),
        final_bull_bnb=float(final_bull_bnb),
        final_bear_bnb=float(final_bear_bnb),
        chain=chain,
    )
    return float(s2), float(ev2), float(payout2)


def size_bet_impact_aware(inputs: BetSizingInputs) -> BetSizingDecision:
    """Canonical impact-aware EV sizing (v1.0 frozen)."""
    if inputs.epoch < 0:
        raise InvariantError("epoch_negative")
    if not (0.0 <= inputs.p_bull <= 1.0):
        raise InvariantError("p_bull_out_of_range")

    _require_finite_nonneg(inputs.final_bull_bnb, "final_bull_bnb")
    _require_finite_nonneg(inputs.final_bear_bnb, "final_bear_bnb")

    if not (inputs.bankroll_bnb >= 0.0) or not math.isfinite(float(inputs.bankroll_bnb)):
        raise InvariantError("bankroll_bnb_not_finite_nonnegative")

    final_bull = float(inputs.final_bull_bnb)
    final_bear = float(inputs.final_bear_bnb)

    # Candidate sizing for Bull
    s2_bull, ev2_bull, payout2_bull = _two_pass_kelly_candidate(
        bet_side="Bull",
        p_win=float(inputs.p_bull),
        bankroll_bnb=float(inputs.bankroll_bnb),
        final_bull_bnb=float(final_bull),
        final_bear_bnb=float(final_bear),
        final_side_bnb=float(final_bull),
        chain=inputs.chain,
        cfg=inputs.cfg,
    )

    # Candidate sizing for Bear
    p_bear = 1.0 - float(inputs.p_bull)
    s2_bear, ev2_bear, payout2_bear = _two_pass_kelly_candidate(
        bet_side="Bear",
        p_win=float(p_bear),
        bankroll_bnb=float(inputs.bankroll_bnb),
        final_bull_bnb=float(final_bull),
        final_bear_bnb=float(final_bear),
        final_side_bnb=float(final_bear),
        chain=inputs.chain,
        cfg=inputs.cfg,
    )

    best_side: str | None
    best_bet: float
    best_ev: float
    best_payout: float | None

    if ev2_bull > ev2_bear:
        best_side = "Bull" if (s2_bull > 0.0 and ev2_bull > 0.0) else None
        best_bet = float(s2_bull) if best_side is not None else 0.0
        best_ev = float(ev2_bull) if best_side is not None else 0.0
        best_payout = float(payout2_bull) if best_side is not None and payout2_bull is not None else None
    elif ev2_bear > ev2_bull:
        best_side = "Bear" if (s2_bear > 0.0 and ev2_bear > 0.0) else None
        best_bet = float(s2_bear) if best_side is not None else 0.0
        best_ev = float(ev2_bear) if best_side is not None else 0.0
        best_payout = float(payout2_bear) if best_side is not None and payout2_bear is not None else None
    else:
        best_side = None
        best_bet = 0.0
        best_ev = 0.0
        best_payout = None

    best_expected_profit_bnb = float(max(ev2_bull, ev2_bear, 0.0))

    bet_cap_bnb = float(
        max(
            _cap_bnb(bankroll_bnb=float(inputs.bankroll_bnb), final_side_bnb=float(final_bull), cfg=inputs.cfg),
            _cap_bnb(bankroll_bnb=float(inputs.bankroll_bnb), final_side_bnb=float(final_bear), cfg=inputs.cfg),
        )
    )

    if best_side is None:
        return BetSizingDecision(
            epoch=int(inputs.epoch),
            bet_side=None,
            bet_bnb=0.0,
            expected_profit_bnb=0.0,
            post_impact_payout_multiple=None,
            bet_cap_bnb=float(bet_cap_bnb),
            best_expected_profit_bnb=float(best_expected_profit_bnb),
            skip_reason="no_positive_ev",
        )

    if float(inputs.bankroll_bnb) < float(best_bet) + float(inputs.chain.gas_bet_bnb):
        return BetSizingDecision(
            epoch=int(inputs.epoch),
            bet_side=None,
            bet_bnb=0.0,
            expected_profit_bnb=0.0,
            post_impact_payout_multiple=None,
            bet_cap_bnb=float(bet_cap_bnb),
            best_expected_profit_bnb=float(best_expected_profit_bnb),
            skip_reason="insufficient_bankroll_for_gas",
        )

    return BetSizingDecision(
        epoch=int(inputs.epoch),
        bet_side=str(best_side),
        bet_bnb=float(best_bet),
        expected_profit_bnb=float(best_ev),
        post_impact_payout_multiple=float(best_payout) if best_payout is not None else None,
        bet_cap_bnb=float(bet_cap_bnb),
        best_expected_profit_bnb=float(best_expected_profit_bnb),
        skip_reason=None,
    )
