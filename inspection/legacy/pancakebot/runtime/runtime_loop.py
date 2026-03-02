"""Minimal legacy RuntimeConfig for inspection backtests.

This file intentionally contains only the dataclass shape consumed by
`inspection.legacy.run_backtest_scenario` and legacy backtest modules.
It is not used by canonical live/dry runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime/backtest dependency bundle for legacy inspection experiments."""

    # Data clients and stores.
    graph_client: Any
    round_store: Any
    klines_store: Any
    binance_us_client: Any
    binance_us_symbol: str

    # On-chain identity/contract handles.
    contract: Any
    wallet_address: str

    # Core timing and walk-forward sizes.
    cutoff_seconds: int
    train_size: int
    retrain_interval: int
    calibrate_size: int
    recalibrate_interval: int

    # Recency weighting used by legacy walk-forward training.
    recency_weight_floor: float
    recency_weight_power: float

    # Event replacement / telemetry controls.
    use_onchain_event_bets: bool
    event_lookback_blocks: int
    event_freshness_slack_seconds: int
    latency_log_path: str
    wait_for_bet_receipt: bool
    bet_receipt_timeout_seconds: int

    # Predictability gate controls.
    predictability_gate_enabled: bool
    predictability_gate_threshold: float
    predictability_baseline_bet_bnb: float

    # Strategy/policy/model knobs.
    policy_cfg: Any
    strategy_cfg: Any

    # Mode flag.
    dry: bool

    # Chain constants.
    treasury_fee_fraction: float
    buffer_seconds: int
    min_bet_amount_bnb: float

    # Model hyperparameters.
    price_alpha: float
    pool_alpha_total: float
    pool_alpha_ratio: float
    random_seed: int
