from __future__ import annotations

from dataclasses import dataclass

from pancakebot.config.policy_config import PolicyConfig
from pancakebot.domain.strategy.ev_math import ChainPolicyParams
from pancakebot.domain.strategy.sizing import BetSizingInputs, size_bet_impact_aware
from pancakebot.core.errors import InvariantError


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: str  # "BET" or "SKIP"
    bet_side: str | None  # "Bull" or "Bear" when betting; None when skipping
    bet_bnb: float

    expected_profit_bnb: float
    post_impact_payout_multiple: float | None

    bet_cap_bnb: float
    best_expected_profit_bnb: float

    reason: str  # skip reason or "ok" when betting


def decide(
    *,
    epoch: int,
    p_bull: float,
    final_bull_bnb: float,
    final_bear_bnb: float,
    bankroll_bnb: float,
    cfg: PolicyConfig,
    chain: ChainPolicyParams,
) -> PolicyDecision:
    """Locked policy decision (canonical impact-aware EV sizing)."""
    if epoch < 0:
        raise InvariantError("epoch_negative")
    if bankroll_bnb < 0.0:
        raise InvariantError("bankroll_negative")

    sizing = size_bet_impact_aware(
        BetSizingInputs(
            epoch=int(epoch),
            bankroll_bnb=float(bankroll_bnb),
            p_bull=float(p_bull),
            final_bull_bnb=float(final_bull_bnb),
            final_bear_bnb=float(final_bear_bnb),
            cfg=cfg,
            chain=chain,
        )
    )

    if sizing.bet_side is None:
        return PolicyDecision(
            action="SKIP",
            bet_side=None,
            bet_bnb=0.0,
            expected_profit_bnb=0.0,
            post_impact_payout_multiple=None,
            bet_cap_bnb=float(sizing.bet_cap_bnb),
            best_expected_profit_bnb=float(sizing.best_expected_profit_bnb),
            reason=str(sizing.skip_reason),
        )

    return PolicyDecision(
        action="BET",
        bet_side=str(sizing.bet_side),
        bet_bnb=float(sizing.bet_bnb),
        expected_profit_bnb=float(sizing.expected_profit_bnb),
        post_impact_payout_multiple=float(sizing.post_impact_payout_multiple)
        if sizing.post_impact_payout_multiple is not None
        else None,
        bet_cap_bnb=float(sizing.bet_cap_bnb),
        best_expected_profit_bnb=float(sizing.best_expected_profit_bnb),
        reason="ok",
    )
