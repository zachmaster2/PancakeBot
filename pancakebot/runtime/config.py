"""RuntimeConfig dataclass binding the round store, contract, gate, pool watcher, and runtime knobs."""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.chain.prediction_contract import Web3PredictionContract
from pancakebot.strategy.momentum_gate import MomentumGate
from pancakebot.chain.pool_watcher import PoolEventWatcher


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    # Closed rounds store (JSONL; used by backtest only; None in live/dry)
    round_store: ClosedRoundsStore | None

    # Momentum strategy config (always present; MomentumGateConfig)
    momentum_gate_config: object

    # Momentum gate (OKX 1s live client; None in backtest mode)
    momentum_gate: MomentumGate | None

    # On-chain / identity
    contract: Web3PredictionContract
    wallet_address: str

    # Feature cutoff
    cutoff_seconds: int

    # Prefetch offset: how many seconds before cutoff to wake for housekeeping
    prefetch_offset_seconds: int

    # Protocol constants (from chain via contract_constants.json)
    min_bet_amount_bnb: float
    treasury_fee_fraction: float
    interval_seconds: int
    buffer_seconds: int

    # Dry-mode initial bankroll
    dry_initial_bankroll_bnb: float | None

    # Execution
    dry: bool

    # Live: clamp all bet sizes to contract minimum for safe testing
    live_min_bet_only: bool

    # Fresh start: archive existing dry state before starting
    dry_fresh_start: bool

    # No-archive: delete (don't archive) existing dry state on --fresh
    dry_no_archive: bool

    # Pool event watcher: accumulates BetBull/BetBear events for accurate pools
    pool_watcher: PoolEventWatcher | None = None


_MOMENTUM_CACHE_N = 10  # retained for sync_mode compatibility


def required_runtime_sync_cache_n() -> int:
    return _MOMENTUM_CACHE_N
