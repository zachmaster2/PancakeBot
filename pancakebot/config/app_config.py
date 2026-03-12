from __future__ import annotations

from dataclasses import dataclass

from pancakebot.backtest.config import BacktestConfig
from pancakebot.config.strategy_config import StrategyConfig


@dataclass(frozen=True, slots=True)
class AppConfig:
    """User-facing configuration loaded from config.toml."""

    # Paths
    closed_rounds_path: str
    klines_path: str
    feature_cache_path: str
    backtest_state_cache_dir: str
    market_data_db_path: str
    projection_cache_db_path: str
    run_registry_db_path: str
    abi_json_path: str

    # Runtime
    cutoff_seconds: int
    random_seed: int
    use_onchain_event_bets: bool
    event_lookback_blocks: int
    latency_log_path: str
    wait_for_bet_receipt: bool
    bet_receipt_timeout_seconds: int

    # Strategy
    strategy: StrategyConfig

    # Backtest options. Live and dry ignore these.
    backtest: BacktestConfig
