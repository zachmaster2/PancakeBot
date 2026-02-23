from __future__ import annotations

from dataclasses import dataclass

from pancakebot.backtest.config import BacktestConfig
from pancakebot.config.policy_config import PolicyConfig


@dataclass(frozen=True, slots=True)
class AppConfig:
    """User-facing configuration loaded from config.toml."""

    # Paths
    closed_rounds_path: str
    klines_path: str
    abi_json_path: str

    # Runtime
    cutoff_seconds: int
    train_size: int
    retrain_interval: int
    calibrate_size: int
    recalibrate_interval: int
    recency_weight_floor: float
    recency_weight_power: float
    random_seed: int

    # Model hyperparameters
    price_alpha: float
    pool_alpha_total: float
    pool_alpha_ratio: float

    # Policy
    policy: PolicyConfig

    # Backtest options. Live and dry ignore these.
    backtest: BacktestConfig
