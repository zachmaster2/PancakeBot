"""Legacy inspection AppConfig.

This config model is consumed only by legacy inspection scripts.
Canonical runtime/backtest config remains in `pancakebot/config/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pancakebot.backtest.config import BacktestConfig
from pancakebot.config.policy_config import PolicyConfig


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuration contract expected by legacy inspection tooling."""

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

    # Event + latency knobs
    use_onchain_event_bets: bool
    event_lookback_blocks: int
    event_freshness_slack_seconds: int
    latency_log_path: str
    wait_for_bet_receipt: bool
    bet_receipt_timeout_seconds: int

    # Predictability gate knobs
    predictability_gate_enabled: bool
    predictability_gate_threshold: float
    predictability_baseline_bet_bnb: float

    # Strategy/policy knobs
    policy: PolicyConfig
    strategy: Any

    # Backtest knobs
    backtest: BacktestConfig
