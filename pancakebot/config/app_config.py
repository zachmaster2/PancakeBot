from __future__ import annotations

from dataclasses import dataclass

from pancakebot.backtest.config import BacktestConfig
from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig


@dataclass(frozen=True, slots=True)
class RuntimeStatePathsConfig:
    """Filesystem paths for mutable runtime state shared by live and dry modes."""

    claim_scan_cursor_path: str
    dry_bets_path: str
    dry_settled_epochs_path: str
    dry_audit_trades_path: str
    dry_cycle_audit_path: str
    dry_bankroll_state_path: str
    dry_pipeline_bootstrap_state_path: str
    live_pipeline_bootstrap_state_path: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    """User-facing configuration loaded from config.toml."""

    # Paths
    closed_rounds_path: str
    market_data_db_path: str
    abi_json_path: str

    # Runtime
    cutoff_seconds: int
    latency_log_path: str
    dry_initial_bankroll_bnb: float | None
    wait_for_bet_receipt: bool
    bet_receipt_timeout_seconds: int

    # Runtime state paths
    runtime_state_paths: RuntimeStatePathsConfig

    # OKX momentum gate (live/dry only; ignored by backtest).
    momentum_gate: MomentumGateConfig

    # Protocol constants (sourced from chain on first live run; kept in config for backtest).
    min_bet_amount_bnb: float
    treasury_fee_fraction: float

    # Backtest options. Live and dry ignore these.
    backtest: BacktestConfig
