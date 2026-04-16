"""Aggregate bet amounts into total/bull/bear wei pools, optionally filtered by timestamp."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pancakebot.types import Bet
from pancakebot.util import InvariantError


@dataclass(frozen=True, slots=True)
class PoolAmountsWei:
    total_wei: int
    bull_wei: int
    bear_wei: int


def compute_pool_amounts_wei(*, bets: Iterable[Bet]) -> PoolAmountsWei:
    """Compute pool amounts from a bets list.

    Locked requirements:
      - Do not query {total,bull,bear}Amount from The Graph.
      - The feature builder computes totals from bets.
      - Enforce bull_amt + bear_amt <= total_amt.
    """
    total = 0
    bull = 0
    bear = 0

    for b in bets:
        amt = b.amount_wei
        if amt <= 0:
            raise InvariantError("bet_amount_wei_nonpositive")

        total += amt
        if b.position == "Bull":
            bull += amt
        elif b.position == "Bear":
            bear += amt
        else:
            raise InvariantError(f"unexpected_bet_position_in_pool_amounts: {b.position}")

    if bull + bear > total:
        raise InvariantError("bull_bear_exceed_total")

    return PoolAmountsWei(total_wei=total, bull_wei=bull, bear_wei=bear)


def compute_pool_amounts_wei_before(*, bets: Iterable[Bet], cutoff_ts: int) -> PoolAmountsWei:
    """Compute pool amounts using only bets strictly before cutoff_ts.

    Strict < avoids boundary ambiguity between The Graph's createdAt
    and BSC block timestamps.
    """
    eligible = (b for b in bets if b.created_at < cutoff_ts)
    return compute_pool_amounts_wei(bets=eligible)
