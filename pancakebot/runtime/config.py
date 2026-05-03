"""RuntimeConfig dataclass binding the round store, contract, gate, pool watcher, and runtime knobs."""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.config import StrategyConfig
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

    # On-chain / identity (backtest passes a stub; dry/live use real contract)
    contract: Web3PredictionContract
    wallet_address: str

    # Feature cutoff
    cutoff_seconds: int

    # Pre-lock wake schedule (all DERIVED from pancakebot/timing_constants.py
    # at config load; not user-tunable). All in milliseconds before lock_at.
    #
    #   skew_sync_wakeup_offset_ms  >  pool_wakeup_offset_ms
    #                               >  kline_wakeup_offset_ms
    #                               >  lock_safety_margin_ms
    #
    # Engine fires three distinct _sleep_until_ts wakes per round:
    # skew_sync -> pool -> kline. Timing guard at lock_safety_margin_ms.
    skew_sync_wakeup_offset_ms: int
    pool_wakeup_offset_ms: int
    kline_wakeup_offset_ms: int
    lock_safety_margin_ms: int

    # User-tunable. Streak counter for OKX transient failures; bot
    # crashes (-> supervisor restart + Discord alert) after this many
    # consecutive `kline_fetch_transient_failure` rounds.
    max_consecutive_fetch_failures: int

    # User-tunable. Pool cutoff: only bets with on-chain block_timestamp
    # < lock_at - pool_cutoff_seconds are counted in the pool aggregate.
    # Cross-validated at config load to be >= WSS_ARRIVAL_DELAY_P99.
    pool_cutoff_seconds: int

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

    # Strategy config (10 knobs; loaded from config.toml [strategy.*] sections)
    strategy: StrategyConfig

    # Pool event watcher: accumulates BetBull/BetBear events for accurate pools
    pool_watcher: PoolEventWatcher | None = None
