from __future__ import annotations

import math
from dataclasses import dataclass

from pancakebot.core.constants import BNB_WEI
from pancakebot.domain.types import Round
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei, compute_pool_amounts_wei_at_or_before
from pancakebot.core.errors import InvariantError


@dataclass(frozen=True, slots=True)
class PriceTargets:
    ret_open: float
    up: int


def compute_price_targets(*, round_t: Round) -> PriceTargets:
    if round_t.lock_price is None or round_t.close_price is None:
        raise InvariantError("target_missing_prices")
    lock_price = float(round_t.lock_price)
    close_price = float(round_t.close_price)
    if lock_price <= 0.0:
        raise InvariantError("target_lock_price_nonpositive")
    ret_open = float(math.log(close_price / lock_price))
    up = 1 if ret_open > 0.0 else 0
    return PriceTargets(ret_open=float(ret_open), up=int(up))


@dataclass(frozen=True, slots=True)
class PoolForecastTargets:
    """Canonical pool forecast labels for a target round.

    These targets are defined relative to cutoff_ts (lock_ts - cutoff_seconds):
    - late_inflow_total_bnb: nonnegative late inflow after cutoff_ts.
    - late_inflow_bull_frac: fraction of late inflow that is Bull.

    Corner rule (frozen): if late_inflow_total_bnb == 0, late_inflow_bull_frac = 0.5.
    """

    late_inflow_total_bnb: float
    late_inflow_bull_frac: float


def compute_pool_forecast_targets(*, round_t: Round, cutoff_seconds: int) -> PoolForecastTargets:
    if round_t.lock_at is None:
        raise InvariantError("target_missing_lock_at")
    lock_ts = int(round_t.lock_at)
    cutoff_ts = int(lock_ts) - int(cutoff_seconds)

    cutoff_pools = compute_pool_amounts_wei_at_or_before(bets=round_t.bets, cutoff_ts=int(cutoff_ts))
    final_pools = compute_pool_amounts_wei(bets=round_t.bets)

    late_total_wei = int(final_pools.total_wei) - int(cutoff_pools.total_wei)
    late_bull_wei = int(final_pools.bull_wei) - int(cutoff_pools.bull_wei)

    # Defensive clamp: labels must not be negative.
    late_total_wei = max(0, int(late_total_wei))
    late_bull_wei = max(0, int(late_bull_wei))

    late_total_bnb = float(late_total_wei) / float(BNB_WEI)
    late_bull_bnb = float(late_bull_wei) / float(BNB_WEI)

    if late_total_bnb > 0.0:
        frac = float(late_bull_bnb / late_total_bnb)
        frac = min(1.0, max(0.0, frac))
    else:
        frac = 0.5

    return PoolForecastTargets(late_inflow_total_bnb=float(late_total_bnb), late_inflow_bull_frac=float(frac))
