from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from pancakebot.backtest.config import BacktestConfig
from pancakebot.config.app_config import AppConfig
from pancakebot.config.policy_config import PolicyConfig
from pancakebot.core.errors import InvariantError


def _req(obj: dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise InvariantError(f"missing_config_key: {key}")
    return obj[key]


def _req_str(obj: dict[str, Any], key: str) -> str:
    v = _req(obj, key)
    if not isinstance(v, str) or not v.strip():
        raise InvariantError(f"config_key_not_nonempty_str: {key}")
    return v.strip()


def _opt_str(obj: dict[str, Any], key: str, default: str) -> str:
    if key not in obj:
        return str(default)
    v = obj[key]
    if not isinstance(v, str) or not v.strip():
        raise InvariantError(f"config_key_not_nonempty_str: {key}")
    return v.strip()


def _req_int(obj: dict[str, Any], key: str) -> int:
    v = _req(obj, key)
    try:
        i = int(v)
    except (TypeError, ValueError) as e:
        raise InvariantError(f"config_key_not_int: {key} err={e}") from e
    return i


def _opt_int(obj: dict[str, Any], key: str, default: int) -> int:
    if key not in obj:
        return default
    v = obj[key]
    try:
        i = int(v)
    except (TypeError, ValueError) as e:
        raise InvariantError(f"config_key_not_int: {key} err={e}") from e
    return i


def _opt_float(obj: dict[str, Any], key: str, default: float) -> float:
    if key not in obj:
        return float(default)
    v = obj[key]
    if not isinstance(v, (int, float)):
        raise InvariantError(f"config_key_not_number: {key}")
    return float(v)


def load_app_config(path: str) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise InvariantError(f"config_file_missing: {path}")

    try:
        raw = tomllib.loads(p.read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise InvariantError(f"config_toml_parse_failed: {e}") from e

    if not isinstance(raw, dict):
        raise InvariantError("config_root_not_dict")

    paths = raw.get("paths")
    graph = raw.get("graph")
    runtime = raw.get("runtime")
    policy = raw.get("policy")
    models = raw.get("models", {})
    backtest = raw.get("backtest", {})

    if not isinstance(paths, dict):
        raise InvariantError("config_section_missing_or_not_dict: paths")
    if not isinstance(graph, dict):
        raise InvariantError("config_section_missing_or_not_dict: graph")
    if not isinstance(runtime, dict):
        raise InvariantError("config_section_missing_or_not_dict: runtime")
    if not isinstance(policy, dict):
        raise InvariantError("config_section_missing_or_not_dict: policy")

    if models is None:
        models = {}
    if not isinstance(models, dict):
        raise InvariantError("config_section_not_dict: models")

    if backtest is None:
        backtest = {}
    if not isinstance(backtest, dict):
        raise InvariantError("config_section_not_dict: backtest")

    closed_rounds_path = _req_str(paths, "closed_rounds_path")
    klines_path = _opt_str(paths, "klines_path", "var/klines.jsonl")
    abi_json_path = _req_str(graph, "abi_json_path")

    allowed_runtime_keys = {
        "cutoff_seconds",
        "train_size",
        "retrain_interval",
        "calibrate_size",
        "recalibrate_interval",
        "recency_weight_floor",
        "recency_weight_power",
        "random_seed",
    }
    unknown_runtime_keys = sorted([k for k in runtime.keys() if k not in allowed_runtime_keys])
    if unknown_runtime_keys:
        raise InvariantError(f"unknown_runtime_config_keys: {unknown_runtime_keys}")

    cutoff_seconds = _req_int(runtime, "cutoff_seconds")
    if cutoff_seconds <= 0:
        raise InvariantError("cutoff_seconds_must_be_positive")

    train_size = _req_int(runtime, "train_size")
    if train_size <= 0:
        raise InvariantError("train_size_must_be_positive")

    retrain_interval = _req_int(runtime, "retrain_interval")
    if retrain_interval <= 0:
        raise InvariantError("retrain_interval_must_be_positive")

    calibrate_size = _req_int(runtime, "calibrate_size")
    if calibrate_size <= 0:
        raise InvariantError("calibrate_size_must_be_positive")

    recalibrate_interval = _opt_int(runtime, "recalibrate_interval", 0)
    if recalibrate_interval < 0:
        raise InvariantError("recalibrate_interval_must_be_nonnegative")

    recency_weight_floor = _opt_float(runtime, "recency_weight_floor", 1.0)
    if not (0.0 < recency_weight_floor <= 1.0):
        raise InvariantError("recency_weight_floor_out_of_range")

    recency_weight_power = _opt_float(runtime, "recency_weight_power", 1.0)
    if recency_weight_power <= 0.0:
        raise InvariantError("recency_weight_power_must_be_positive")

    random_seed = _req_int(runtime, "random_seed")
    if random_seed < 0:
        raise InvariantError("random_seed_must_be_nonnegative")

    # Models
    allowed_model_keys = {"price_alpha", "pool_alpha_total", "pool_alpha_ratio"}
    unknown_model_keys = sorted([k for k in models.keys() if k not in allowed_model_keys])
    if unknown_model_keys:
        raise InvariantError(f"unknown_models_config_keys: {unknown_model_keys}")

    price_alpha = float(models.get("price_alpha", 1.0))
    if price_alpha <= 0.0:
        raise InvariantError("price_alpha_must_be_positive")

    pool_alpha_total = float(models.get("pool_alpha_total", 1.0))
    if pool_alpha_total <= 0.0:
        raise InvariantError("pool_alpha_total_must_be_positive")

    pool_alpha_ratio = float(models.get("pool_alpha_ratio", 1.0))
    if pool_alpha_ratio <= 0.0:
        raise InvariantError("pool_alpha_ratio_must_be_positive")

    # Policy (v1.0 frozen knobs; strictly validated)
    allowed_policy_keys = {"kelly_multiplier", "bankroll_cap_fraction", "pool_cap_fraction", "max_bet_bnb"}
    unknown_policy_keys = sorted([k for k in policy.keys() if k not in allowed_policy_keys])
    if unknown_policy_keys:
        raise InvariantError(f"unknown_policy_config_keys: {unknown_policy_keys}")

    kelly_multiplier = _opt_float(policy, "kelly_multiplier", 0.5)
    if kelly_multiplier <= 0.0:
        raise InvariantError("kelly_multiplier_must_be_positive")

    bankroll_cap_fraction = _opt_float(policy, "bankroll_cap_fraction", 0.10)
    if not (0.0 < bankroll_cap_fraction <= 1.0):
        raise InvariantError("bankroll_cap_fraction_out_of_range")

    pool_cap_fraction = _opt_float(policy, "pool_cap_fraction", 0.10)
    if not (0.0 < pool_cap_fraction <= 1.0):
        raise InvariantError("pool_cap_fraction_out_of_range")

    max_bet_bnb = _opt_float(policy, "max_bet_bnb", 0.25)
    if max_bet_bnb <= 0.0:
        raise InvariantError("max_bet_bnb_must_be_positive")

    policy_cfg = PolicyConfig(
        kelly_multiplier=kelly_multiplier,
        bankroll_cap_fraction=bankroll_cap_fraction,
        pool_cap_fraction=pool_cap_fraction,
        max_bet_bnb=max_bet_bnb,
    )

    # Backtest
    allowed_bt_keys = {"simulation_size", "initial_bankroll_bnb"}
    unknown_bt_keys = sorted([k for k in backtest.keys() if k not in allowed_bt_keys])
    if unknown_bt_keys:
        raise InvariantError(f"unknown_backtest_config_keys: {unknown_bt_keys}")

    simulation_size_v = backtest.get("simulation_size", 5000)
    if not isinstance(simulation_size_v, int):
        raise InvariantError("backtest_simulation_size_not_int")
    if simulation_size_v <= 0:
        raise InvariantError("backtest_simulation_size_must_be_positive")

    initial_bankroll_bnb_v = backtest.get("initial_bankroll_bnb", 0.5)
    if not isinstance(initial_bankroll_bnb_v, (int, float)):
        raise InvariantError("backtest_initial_bankroll_bnb_not_number")
    initial_bankroll_bnb = float(initial_bankroll_bnb_v)
    if initial_bankroll_bnb <= 0.0:
        raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")

    return AppConfig(
        closed_rounds_path=closed_rounds_path,
        klines_path=klines_path,
        abi_json_path=abi_json_path,
        cutoff_seconds=cutoff_seconds,
        train_size=train_size,
        retrain_interval=retrain_interval,
        calibrate_size=calibrate_size,
        recalibrate_interval=recalibrate_interval,
        recency_weight_floor=recency_weight_floor,
        recency_weight_power=recency_weight_power,
        random_seed=random_seed,
        price_alpha=price_alpha,
        pool_alpha_total=pool_alpha_total,
        pool_alpha_ratio=pool_alpha_ratio,
        policy=policy_cfg,
        backtest=BacktestConfig(
            simulation_size=simulation_size_v,
            initial_bankroll_bnb=initial_bankroll_bnb,
        ),
    )
