from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from pancakebot.backtest.config import BacktestConfig
from pancakebot.config.app_config import AppConfig, RuntimeStatePathsConfig
from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
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
        return int(default)
    v = obj[key]
    try:
        i = int(v)
    except (TypeError, ValueError) as e:
        raise InvariantError(f"config_key_not_int: {key} err={e}") from e
    return i


def _req_float(obj: dict[str, Any], key: str) -> float:
    v = _req(obj, key)
    if not isinstance(v, (int, float)):
        raise InvariantError(f"config_key_not_number: {key}")
    return float(v)


def _opt_float(obj: dict[str, Any], key: str, default: float) -> float:
    if key not in obj:
        return float(default)
    v = obj[key]
    if not isinstance(v, (int, float)):
        raise InvariantError(f"config_key_not_number: {key}")
    return float(v)


def _opt_float_or_none(obj: dict[str, Any], key: str) -> float | None:
    if key not in obj:
        return None
    v = obj[key]
    if v is None:
        return None
    if not isinstance(v, (int, float)):
        raise InvariantError(f"config_key_not_number: {key}")
    return float(v)


def _opt_bool(obj: dict[str, Any], key: str, default: bool) -> bool:
    if key not in obj:
        return bool(default)
    v = obj[key]
    if not isinstance(v, bool):
        raise InvariantError(f"config_key_not_bool: {key}")
    return bool(v)


def _req_bool(obj: dict[str, Any], key: str) -> bool:
    v = _req(obj, key)
    if not isinstance(v, bool):
        raise InvariantError(f"config_key_not_bool: {key}")
    return bool(v)


def _validate_unknown_keys(section_name: str, obj: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted([k for k in obj.keys() if k not in allowed])
    if unknown:
        raise InvariantError(f"unknown_{section_name}_config_keys: {unknown}")


def _validate_distinct_paths(section_name: str, paths: dict[str, str]) -> None:
    seen: dict[str, str] = {}
    for key, raw_path in paths.items():
        normalized = str(Path(str(raw_path))).replace("\\", "/").lower()
        if normalized in seen:
            other = seen[normalized]
            raise InvariantError(
                f"{section_name}_paths_must_be_distinct: {other} vs {key} -> {raw_path}"
            )
        seen[normalized] = str(key)


def load_app_config(path: str) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise InvariantError(f"config_file_missing: {path}")

    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise InvariantError(f"config_toml_parse_failed: {e}") from e

    if not isinstance(raw, dict):
        raise InvariantError("config_root_not_dict")

    paths = raw.get("paths")
    graph = raw.get("graph")
    runtime = raw.get("runtime")
    contract_raw = raw.get("contract", {})
    backtest = raw.get("backtest", {})
    momentum_gate_raw = raw.get("momentum_gate", {})

    if not isinstance(paths, dict):
        raise InvariantError("config_section_missing_or_not_dict: paths")
    if not isinstance(graph, dict):
        raise InvariantError("config_section_missing_or_not_dict: graph")
    if not isinstance(runtime, dict):
        raise InvariantError("config_section_missing_or_not_dict: runtime")
    if backtest is None:
        backtest = {}
    if not isinstance(backtest, dict):
        raise InvariantError("config_section_not_dict: backtest")
    if contract_raw is None:
        contract_raw = {}
    if not isinstance(contract_raw, dict):
        raise InvariantError("config_section_not_dict: contract")
    if momentum_gate_raw is None:
        momentum_gate_raw = {}
    if not isinstance(momentum_gate_raw, dict):
        raise InvariantError("config_section_not_dict: momentum_gate")

    allowed_path_keys = {
        "closed_rounds_path",
        "klines_path",
        "market_data_db_path",
        "claim_scan_cursor_path",
        "dry_bets_path",
        "dry_settled_epochs_path",
        "dry_audit_trades_path",
        "dry_cycle_audit_path",
        "dry_bankroll_state_path",
        "dry_pipeline_bootstrap_state_path",
        "live_pipeline_bootstrap_state_path",
    }
    _validate_unknown_keys("paths", paths, allowed_path_keys)

    closed_rounds_path = _req_str(paths, "closed_rounds_path")
    klines_path = _opt_str(paths, "klines_path", "var/klines.jsonl")
    market_data_db_path = _opt_str(
        paths,
        "market_data_db_path",
        "../PancakeBot_var_exp/market_data_v1.sqlite",
    )
    claim_scan_cursor_path = _opt_str(
        paths,
        "claim_scan_cursor_path",
        "var/runtime/claim_scan_cursor.txt",
    )
    dry_bets_path = _opt_str(
        paths,
        "dry_bets_path",
        "var/runtime/dry_bets.jsonl",
    )
    dry_settled_epochs_path = _opt_str(
        paths,
        "dry_settled_epochs_path",
        "var/runtime/dry_settled_epochs.txt",
    )
    dry_audit_trades_path = _opt_str(
        paths,
        "dry_audit_trades_path",
        "var/runtime/dry_audit_trades.csv",
    )
    dry_cycle_audit_path = _opt_str(
        paths,
        "dry_cycle_audit_path",
        "var/runtime/dry_cycle_audit.csv",
    )
    dry_bankroll_state_path = _opt_str(
        paths,
        "dry_bankroll_state_path",
        "var/runtime/dry_bankroll_state.json",
    )
    dry_pipeline_bootstrap_state_path = _opt_str(
        paths,
        "dry_pipeline_bootstrap_state_path",
        "var/runtime/dry_pipeline_bootstrap_state.pkl.gz",
    )
    live_pipeline_bootstrap_state_path = _opt_str(
        paths,
        "live_pipeline_bootstrap_state_path",
        "var/runtime/live_pipeline_bootstrap_state.pkl.gz",
    )
    _validate_distinct_paths(
        "runtime_state",
        {
            "claim_scan_cursor_path": str(claim_scan_cursor_path),
            "dry_bets_path": str(dry_bets_path),
            "dry_settled_epochs_path": str(dry_settled_epochs_path),
            "dry_audit_trades_path": str(dry_audit_trades_path),
            "dry_cycle_audit_path": str(dry_cycle_audit_path),
            "dry_bankroll_state_path": str(dry_bankroll_state_path),
            "dry_pipeline_bootstrap_state_path": str(dry_pipeline_bootstrap_state_path),
            "live_pipeline_bootstrap_state_path": str(live_pipeline_bootstrap_state_path),
        },
    )

    _validate_unknown_keys("graph", graph, {"abi_json_path"})
    abi_json_path = _req_str(graph, "abi_json_path")

    allowed_runtime_keys = {
        "cutoff_seconds",
        "latency_log_path",
        "dry_initial_bankroll_bnb",
        "wait_for_bet_receipt",
        "bet_receipt_timeout_seconds",
    }
    _validate_unknown_keys("runtime", runtime, allowed_runtime_keys)

    cutoff_seconds = _req_int(runtime, "cutoff_seconds")
    if cutoff_seconds <= 0:
        raise InvariantError("cutoff_seconds_must_be_positive")

    latency_log_path = _opt_str(runtime, "latency_log_path", "var/live_latency.jsonl")

    dry_initial_bankroll_bnb = _opt_float_or_none(runtime, "dry_initial_bankroll_bnb")
    if dry_initial_bankroll_bnb is not None and float(dry_initial_bankroll_bnb) <= 0.0:
        raise InvariantError("dry_initial_bankroll_bnb_must_be_positive")

    wait_for_bet_receipt = _opt_bool(runtime, "wait_for_bet_receipt", True)

    bet_receipt_timeout_seconds = _opt_int(runtime, "bet_receipt_timeout_seconds", 45)
    if bet_receipt_timeout_seconds <= 0:
        raise InvariantError("bet_receipt_timeout_seconds_must_be_positive")

    _validate_unknown_keys("contract", contract_raw, {
        "min_bet_amount_bnb", "treasury_fee_fraction", "buffer_seconds",
    })
    min_bet_amount_bnb = _opt_float(contract_raw, "min_bet_amount_bnb", 0.001)
    if float(min_bet_amount_bnb) <= 0.0:
        raise InvariantError("contract_min_bet_amount_bnb_must_be_positive")
    treasury_fee_fraction = _opt_float(contract_raw, "treasury_fee_fraction", 0.03)
    if not (0.0 <= float(treasury_fee_fraction) < 1.0):
        raise InvariantError("contract_treasury_fee_fraction_out_of_range")
    buffer_seconds_cfg = _opt_int(contract_raw, "buffer_seconds", 30)
    if int(buffer_seconds_cfg) <= 0:
        raise InvariantError("contract_buffer_seconds_must_be_positive")

    allowed_bt_keys = {
        "simulation_size",
        "initial_bankroll_bnb",
        "reset_mode",
        "reset_every_rounds",
        "tail_offset_rounds",
        "epoch_start",
        "epoch_end",
    }
    _validate_unknown_keys("backtest", backtest, allowed_bt_keys)

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

    reset_mode = _opt_str(backtest, "reset_mode", "continuous")
    reset_every_rounds = _opt_int(backtest, "reset_every_rounds", 0)
    tail_offset_rounds = _opt_int(backtest, "tail_offset_rounds", 0)

    epoch_start_raw = backtest.get("epoch_start")
    epoch_end_raw = backtest.get("epoch_end")
    epoch_start = None if epoch_start_raw is None else int(epoch_start_raw)
    epoch_end = None if epoch_end_raw is None else int(epoch_end_raw)

    backtest_cfg = BacktestConfig(
        simulation_size=simulation_size_v,
        initial_bankroll_bnb=initial_bankroll_bnb,
        reset_mode=str(reset_mode),
        reset_every_rounds=int(reset_every_rounds),
        tail_offset_rounds=int(tail_offset_rounds),
        epoch_start=epoch_start,
        epoch_end=epoch_end,
    )
    backtest_cfg.validate()

    _validate_unknown_keys("momentum_gate", momentum_gate_raw, {
        "enabled", "symbol", "threshold", "max_staleness_seconds", "bet_size_bnb",
    })
    mg_enabled = _opt_bool(momentum_gate_raw, "enabled", False)
    mg_symbol = _opt_str(momentum_gate_raw, "symbol", "BNB-USDT")
    mg_threshold = _opt_float(momentum_gate_raw, "threshold", 0.0001)
    if float(mg_threshold) <= 0:
        raise InvariantError("momentum_gate_threshold_must_be_positive")
    mg_max_staleness = _opt_int(momentum_gate_raw, "max_staleness_seconds", 120)
    if int(mg_max_staleness) <= 0:
        raise InvariantError("momentum_gate_max_staleness_must_be_positive")
    mg_bet_size_bnb = _opt_float(momentum_gate_raw, "bet_size_bnb", 0.05)
    if float(mg_bet_size_bnb) <= 0:
        raise InvariantError("momentum_gate_bet_size_bnb_must_be_positive")
    momentum_gate_cfg = MomentumGateConfig(
        enabled=bool(mg_enabled),
        symbol=str(mg_symbol),
        threshold=float(mg_threshold),
        max_staleness_seconds=int(mg_max_staleness),
        bet_size_bnb=float(mg_bet_size_bnb),
    )

    return AppConfig(
        closed_rounds_path=closed_rounds_path,
        klines_path=klines_path,
        market_data_db_path=market_data_db_path,
        abi_json_path=abi_json_path,
        cutoff_seconds=int(cutoff_seconds),
        latency_log_path=str(latency_log_path),
        dry_initial_bankroll_bnb=(
            None if dry_initial_bankroll_bnb is None else float(dry_initial_bankroll_bnb)
        ),
        wait_for_bet_receipt=bool(wait_for_bet_receipt),
        bet_receipt_timeout_seconds=int(bet_receipt_timeout_seconds),
        runtime_state_paths=RuntimeStatePathsConfig(
            claim_scan_cursor_path=str(claim_scan_cursor_path),
            dry_bets_path=str(dry_bets_path),
            dry_settled_epochs_path=str(dry_settled_epochs_path),
            dry_audit_trades_path=str(dry_audit_trades_path),
            dry_cycle_audit_path=str(dry_cycle_audit_path),
            dry_bankroll_state_path=str(dry_bankroll_state_path),
            dry_pipeline_bootstrap_state_path=str(dry_pipeline_bootstrap_state_path),
            live_pipeline_bootstrap_state_path=str(live_pipeline_bootstrap_state_path),
        ),
        momentum_gate=momentum_gate_cfg,
        min_bet_amount_bnb=float(min_bet_amount_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
        buffer_seconds=int(buffer_seconds_cfg),
        backtest=backtest_cfg,
    )
