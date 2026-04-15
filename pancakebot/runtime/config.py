"""Runtime configuration dataclass and constants."""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.config import RuntimeStatePathsConfig
from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.chain.prediction_contract import Web3PredictionContract
from pancakebot.strategy.momentum_gate import MomentumGate
from pancakebot.chain.pool_watcher import PoolEventWatcher


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    # Closed rounds store (JSONL; used by backtest only; None in live/dry)
    round_store: ClosedRoundsStore | None

    # Momentum strategy config (always present)
    momentum_gate_config: object  # MomentumGateConfig

    # Momentum gate (OKX 1s live client; None in backtest mode)
    momentum_gate: MomentumGate | None

    # On-chain / identity
    contract: Web3PredictionContract
    wallet_address: str

    # Feature cutoff
    cutoff_seconds: int

    # Protocol constants (cached at startup)
    min_bet_amount_bnb: float
    treasury_fee_fraction: float

    # Runtime latency telemetry.
    latency_log_path: str
    dry_initial_bankroll_bnb: float | None
    wait_for_bet_receipt: bool
    bet_receipt_timeout_seconds: int

    # Execution
    dry: bool

    # Pool event watcher: accumulates BetBull/BetBear events for accurate pools
    pool_watcher: PoolEventWatcher | None = None

    # Mutable runtime state paths used by live/dry loops.
    runtime_state_paths: RuntimeStatePathsConfig = RuntimeStatePathsConfig(
        claim_scan_cursor_path="var/runtime/claim_scan_cursor.txt",
        dry_bets_path="var/runtime/dry_bets.jsonl",
        dry_settled_epochs_path="var/runtime/dry_settled_epochs.txt",
        dry_audit_trades_path="var/runtime/dry_audit_trades.csv",
        dry_cycle_audit_path="var/runtime/dry_cycle_audit.csv",
        dry_bankroll_state_path="var/runtime/dry_bankroll_state.json",
        dry_archive_root="../PancakeBot_var_exp",
        dry_fresh_start=True,
    )


_MOMENTUM_CACHE_N = 10  # retained for sync_mode compatibility


def required_runtime_sync_cache_n() -> int:
    return _MOMENTUM_CACHE_N
