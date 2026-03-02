"""Compatibility config loader for legacy inspection scripts.

This loader intentionally accepts both:
- legacy ML-era config shape (`runtime` + `models` + `policy` sections), and
- current canonical config shape (dislocation-focused runtime/backtest sections).

It is used only by `inspection/legacy` entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from pancakebot.backtest.config import BacktestConfig
from pancakebot.config.app_config import AppConfig
from pancakebot.config.policy_config import PolicyConfig
from pancakebot.core.errors import InvariantError


def _expect_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    val = raw.get(key)
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise InvariantError(f"config_section_not_dict: {key}")
    return val


def _as_str(section: dict[str, Any], key: str, *, default: str | None = None) -> str:
    if key not in section:
        if default is None:
            raise InvariantError(f"missing_config_key: {key}")
        return str(default)
    val = section[key]
    if not isinstance(val, str) or not val.strip():
        raise InvariantError(f"config_key_not_nonempty_str: {key}")
    return str(val).strip()


def _as_int(section: dict[str, Any], key: str, *, default: int | None = None) -> int:
    if key not in section:
        if default is None:
            raise InvariantError(f"missing_config_key: {key}")
        return int(default)
    val = section[key]
    try:
        out = int(val)
    except (TypeError, ValueError) as exc:
        raise InvariantError(f"config_key_not_int: {key} err={exc}") from exc
    return int(out)


def _as_float(section: dict[str, Any], key: str, *, default: float | None = None) -> float:
    if key not in section:
        if default is None:
            raise InvariantError(f"missing_config_key: {key}")
        return float(default)
    val = section[key]
    if not isinstance(val, (int, float)):
        raise InvariantError(f"config_key_not_number: {key}")
    return float(val)


def _as_bool(section: dict[str, Any], key: str, *, default: bool | None = None) -> bool:
    if key not in section:
        if default is None:
            raise InvariantError(f"missing_config_key: {key}")
        return bool(default)
    val = section[key]
    if not isinstance(val, bool):
        raise InvariantError(f"config_key_not_bool: {key}")
    return bool(val)


def load_app_config(path: str) -> AppConfig:
    """Load inspection config with permissive compatibility defaults."""

    config_path = Path(path)
    if not config_path.exists():
        raise InvariantError(f"config_file_missing: {path}")

    try:
        raw = tomllib.loads(config_path.read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise InvariantError(f"config_toml_parse_failed: {exc}") from exc

    if not isinstance(raw, dict):
        raise InvariantError("config_root_not_dict")

    paths = _expect_dict(raw, "paths")
    graph = _expect_dict(raw, "graph")
    runtime = _expect_dict(raw, "runtime")
    models = _expect_dict(raw, "models")
    policy = _expect_dict(raw, "policy")
    backtest = _expect_dict(raw, "backtest")

    closed_rounds_path = _as_str(paths, "closed_rounds_path")
    klines_path = _as_str(paths, "klines_path", default="var/klines.jsonl")
    abi_json_path = _as_str(graph, "abi_json_path")

    cutoff_seconds = _as_int(runtime, "cutoff_seconds", default=17)
    if cutoff_seconds <= 0:
        raise InvariantError("cutoff_seconds_must_be_positive")

    train_size = _as_int(runtime, "train_size", default=30000)
    if train_size <= 0:
        raise InvariantError("train_size_must_be_positive")

    retrain_interval = _as_int(runtime, "retrain_interval", default=100)
    if retrain_interval <= 0:
        raise InvariantError("retrain_interval_must_be_positive")

    calibrate_size = _as_int(runtime, "calibrate_size", default=30000)
    if calibrate_size <= 0:
        raise InvariantError("calibrate_size_must_be_positive")

    recalibrate_interval = _as_int(runtime, "recalibrate_interval", default=100)
    if recalibrate_interval < 0:
        raise InvariantError("recalibrate_interval_must_be_nonnegative")

    recency_weight_floor = _as_float(runtime, "recency_weight_floor", default=0.10)
    if not (0.0 < float(recency_weight_floor) <= 1.0):
        raise InvariantError("recency_weight_floor_out_of_range")

    recency_weight_power = _as_float(runtime, "recency_weight_power", default=2.0)
    if float(recency_weight_power) <= 0.0:
        raise InvariantError("recency_weight_power_must_be_positive")

    random_seed = _as_int(runtime, "random_seed", default=1337)
    if random_seed < 0:
        raise InvariantError("random_seed_must_be_nonnegative")

    use_onchain_event_bets = _as_bool(runtime, "use_onchain_event_bets", default=False)

    event_lookback_blocks = _as_int(runtime, "event_lookback_blocks", default=600)
    if event_lookback_blocks <= 0:
        raise InvariantError("event_lookback_blocks_must_be_positive")

    event_freshness_slack_seconds = _as_int(runtime, "event_freshness_slack_seconds", default=0)
    if event_freshness_slack_seconds < 0:
        raise InvariantError("event_freshness_slack_seconds_negative")

    latency_log_path = _as_str(runtime, "latency_log_path", default="var/live_latency.jsonl")

    wait_for_bet_receipt = _as_bool(runtime, "wait_for_bet_receipt", default=True)

    bet_receipt_timeout_seconds = _as_int(runtime, "bet_receipt_timeout_seconds", default=45)
    if bet_receipt_timeout_seconds <= 0:
        raise InvariantError("bet_receipt_timeout_seconds_must_be_positive")

    predictability_gate_enabled = _as_bool(runtime, "predictability_gate_enabled", default=True)

    predictability_gate_threshold = _as_float(runtime, "predictability_gate_threshold", default=0.60)
    if not (0.0 <= float(predictability_gate_threshold) <= 1.0):
        raise InvariantError("predictability_gate_threshold_out_of_range")

    predictability_baseline_bet_bnb = _as_float(runtime, "predictability_baseline_bet_bnb", default=0.05)
    if float(predictability_baseline_bet_bnb) <= 0.0:
        raise InvariantError("predictability_baseline_bet_bnb_must_be_positive")

    price_alpha = _as_float(models, "price_alpha", default=1.0)
    if price_alpha <= 0.0:
        raise InvariantError("price_alpha_must_be_positive")

    pool_alpha_total = _as_float(models, "pool_alpha_total", default=1.0)
    if pool_alpha_total <= 0.0:
        raise InvariantError("pool_alpha_total_must_be_positive")

    pool_alpha_ratio = _as_float(models, "pool_alpha_ratio", default=1.0)
    if pool_alpha_ratio <= 0.0:
        raise InvariantError("pool_alpha_ratio_must_be_positive")

    policy_cfg = PolicyConfig(
        kelly_multiplier=_as_float(policy, "kelly_multiplier", default=0.9),
        bankroll_cap_fraction=_as_float(policy, "bankroll_cap_fraction", default=0.20),
        pool_cap_fraction=_as_float(policy, "pool_cap_fraction", default=0.15),
        max_bet_bnb=_as_float(policy, "max_bet_bnb", default=0.50),
    )

    if float(policy_cfg.kelly_multiplier) <= 0.0:
        raise InvariantError("kelly_multiplier_must_be_positive")
    if not (0.0 < float(policy_cfg.bankroll_cap_fraction) <= 1.0):
        raise InvariantError("bankroll_cap_fraction_out_of_range")
    if not (0.0 < float(policy_cfg.pool_cap_fraction) <= 1.0):
        raise InvariantError("pool_cap_fraction_out_of_range")
    if float(policy_cfg.max_bet_bnb) <= 0.0:
        raise InvariantError("max_bet_bnb_must_be_positive")

    simulation_size = _as_int(backtest, "simulation_size", default=5000)
    if simulation_size <= 0:
        raise InvariantError("backtest_simulation_size_must_be_positive")

    initial_bankroll_bnb = _as_float(backtest, "initial_bankroll_bnb", default=0.5)
    if initial_bankroll_bnb <= 0.0:
        raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")

    reset_mode = _as_str(backtest, "reset_mode", default="continuous")
    reset_every_rounds = _as_int(backtest, "reset_every_rounds", default=0)

    backtest_cfg = BacktestConfig(
        simulation_size=int(simulation_size),
        initial_bankroll_bnb=float(initial_bankroll_bnb),
        reset_mode=str(reset_mode),
        reset_every_rounds=int(reset_every_rounds),
    )
    backtest_cfg.validate()

    return AppConfig(
        closed_rounds_path=str(closed_rounds_path),
        klines_path=str(klines_path),
        abi_json_path=str(abi_json_path),
        cutoff_seconds=int(cutoff_seconds),
        train_size=int(train_size),
        retrain_interval=int(retrain_interval),
        calibrate_size=int(calibrate_size),
        recalibrate_interval=int(recalibrate_interval),
        recency_weight_floor=float(recency_weight_floor),
        recency_weight_power=float(recency_weight_power),
        random_seed=int(random_seed),
        price_alpha=float(price_alpha),
        pool_alpha_total=float(pool_alpha_total),
        pool_alpha_ratio=float(pool_alpha_ratio),
        use_onchain_event_bets=bool(use_onchain_event_bets),
        event_lookback_blocks=int(event_lookback_blocks),
        event_freshness_slack_seconds=int(event_freshness_slack_seconds),
        latency_log_path=str(latency_log_path),
        wait_for_bet_receipt=bool(wait_for_bet_receipt),
        bet_receipt_timeout_seconds=int(bet_receipt_timeout_seconds),
        predictability_gate_enabled=bool(predictability_gate_enabled),
        predictability_gate_threshold=float(predictability_gate_threshold),
        predictability_baseline_bet_bnb=float(predictability_baseline_bet_bnb),
        policy=policy_cfg,
        strategy=raw.get("strategy", {}),
        backtest=backtest_cfg,
    )
