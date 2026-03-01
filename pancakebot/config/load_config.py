from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from pancakebot.backtest.config import BacktestConfig
from pancakebot.config.app_config import AppConfig
from pancakebot.config.strategy_config import (
    DislocationCandidateConfig,
    DislocationSelectorConfig,
    DislocationStrategyConfig,
    StrategyConfig,
)
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


def _opt_bool(obj: dict[str, Any], key: str, default: bool) -> bool:
    if key not in obj:
        return bool(default)
    v = obj[key]
    if not isinstance(v, bool):
        raise InvariantError(f"config_key_not_bool: {key}")
    return bool(v)


def _req_list_str(obj: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = _req(obj, key)
    if not isinstance(raw, list):
        raise InvariantError(f"config_key_not_list: {key}")
    out: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise InvariantError(f"config_key_list_item_not_nonempty_str: {key}[{i}]")
        out.append(item.strip())
    return tuple(out)


def _validate_unknown_keys(section_name: str, obj: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted([k for k in obj.keys() if k not in allowed])
    if unknown:
        raise InvariantError(f"unknown_{section_name}_config_keys: {unknown}")


def _parse_dislocation_selector(selector: dict[str, Any]) -> DislocationSelectorConfig:
    defaults = DislocationSelectorConfig()
    allowed = {
        "warmup_rounds",
        "num_quantile_bins",
        "min_cell_obs",
        "score_threshold",
        "use_direction_split",
        "shadow_initial_bankroll_bnb",
    }
    _validate_unknown_keys("dislocation_selector", selector, allowed)

    warmup_rounds = _opt_int(selector, "warmup_rounds", int(defaults.warmup_rounds))
    if warmup_rounds <= 0:
        raise InvariantError("dislocation_warmup_rounds_must_be_positive")

    num_quantile_bins = _opt_int(selector, "num_quantile_bins", int(defaults.num_quantile_bins))
    if num_quantile_bins <= 1:
        raise InvariantError("dislocation_num_quantile_bins_invalid")

    min_cell_obs = _opt_int(selector, "min_cell_obs", int(defaults.min_cell_obs))
    if min_cell_obs <= 0:
        raise InvariantError("dislocation_min_cell_obs_must_be_positive")

    score_threshold = _opt_float(selector, "score_threshold", float(defaults.score_threshold))

    use_direction_split = _opt_bool(selector, "use_direction_split", bool(defaults.use_direction_split))

    shadow_initial_bankroll_bnb = _opt_float(
        selector,
        "shadow_initial_bankroll_bnb",
        float(defaults.shadow_initial_bankroll_bnb),
    )
    if shadow_initial_bankroll_bnb <= 0.0:
        raise InvariantError("dislocation_shadow_initial_bankroll_bnb_must_be_positive")

    return DislocationSelectorConfig(
        warmup_rounds=int(warmup_rounds),
        num_quantile_bins=int(num_quantile_bins),
        min_cell_obs=int(min_cell_obs),
        score_threshold=float(score_threshold),
        use_direction_split=bool(use_direction_split),
        shadow_initial_bankroll_bnb=float(shadow_initial_bankroll_bnb),
    )


def _parse_dislocation_candidate(candidate: dict[str, Any], idx: int) -> DislocationCandidateConfig:
    candidate_name = f"strategy.dislocation.candidates[{int(idx)}]"
    allowed = {
        "name",
        "lookback1_seconds",
        "lookback2_seconds",
        "lookback3_seconds",
        "weight1",
        "weight2",
        "weight3",
        "temperature_bps",
        "fixed_bet_bnb",
        "dislocation_threshold_pp",
        "nowcast_confidence_min",
        "cutoff_pool_total_min_bnb",
        "expected_net_min_bnb",
        "side_selection_mode",
        "market_extreme_min",
        "flow_window_seconds",
        "flow_min_imbalance",
        "flow_gate_mode",
        "adaptive_candidate_modes",
        "adaptive_window",
        "adaptive_min_history",
        "adaptive_score",
        "adaptive_fallback_mode",
        "stake_mode",
        "stake_min_bnb",
        "stake_max_bnb",
        "stake_ev_ref_bnb",
        "stake_max_side_pool_frac",
        "perf_adapt_mode",
        "perf_gate_window",
        "perf_gate_min_history",
        "perf_gate_min_win_rate",
        "perf_gate_min_mean_profit_bnb",
    }
    _validate_unknown_keys(candidate_name, candidate, allowed)

    name = _req_str(candidate, "name")

    lookback1_seconds = _req_int(candidate, "lookback1_seconds")
    lookback2_seconds = _req_int(candidate, "lookback2_seconds")
    lookback3_seconds = _req_int(candidate, "lookback3_seconds")
    if lookback1_seconds <= 0 or lookback2_seconds <= 0 or lookback3_seconds <= 0:
        raise InvariantError("dislocation_candidate_lookback_must_be_positive")

    weight1 = _req_float(candidate, "weight1")
    weight2 = _req_float(candidate, "weight2")
    weight3 = _req_float(candidate, "weight3")

    temperature_bps = _req_float(candidate, "temperature_bps")
    if temperature_bps <= 0.0:
        raise InvariantError("dislocation_candidate_temperature_bps_must_be_positive")

    fixed_bet_bnb = _req_float(candidate, "fixed_bet_bnb")
    if fixed_bet_bnb <= 0.0:
        raise InvariantError("dislocation_candidate_fixed_bet_bnb_must_be_positive")

    dislocation_threshold_pp = _req_float(candidate, "dislocation_threshold_pp")
    if dislocation_threshold_pp < 0.0:
        raise InvariantError("dislocation_candidate_dislocation_threshold_pp_negative")

    nowcast_confidence_min = _req_float(candidate, "nowcast_confidence_min")
    if nowcast_confidence_min < 0.0:
        raise InvariantError("dislocation_candidate_nowcast_confidence_min_negative")

    cutoff_pool_total_min_bnb = _req_float(candidate, "cutoff_pool_total_min_bnb")
    if cutoff_pool_total_min_bnb < 0.0:
        raise InvariantError("dislocation_candidate_cutoff_pool_total_min_bnb_negative")

    expected_net_min_bnb = _req_float(candidate, "expected_net_min_bnb")

    side_selection_mode = _req_str(candidate, "side_selection_mode")

    market_extreme_min = _req_float(candidate, "market_extreme_min")
    if market_extreme_min < 0.0:
        raise InvariantError("dislocation_candidate_market_extreme_min_negative")

    flow_window_seconds = _req_int(candidate, "flow_window_seconds")
    if flow_window_seconds < 0:
        raise InvariantError("dislocation_candidate_flow_window_seconds_negative")

    flow_min_imbalance = _req_float(candidate, "flow_min_imbalance")
    if flow_min_imbalance < 0.0:
        raise InvariantError("dislocation_candidate_flow_min_imbalance_negative")

    flow_gate_mode = _req_str(candidate, "flow_gate_mode")

    adaptive_candidate_modes = _req_list_str(candidate, "adaptive_candidate_modes")

    adaptive_window = _req_int(candidate, "adaptive_window")
    if adaptive_window <= 0:
        raise InvariantError("dislocation_candidate_adaptive_window_must_be_positive")

    adaptive_min_history = _req_int(candidate, "adaptive_min_history")
    if adaptive_min_history <= 0:
        raise InvariantError("dislocation_candidate_adaptive_min_history_must_be_positive")

    adaptive_score = _req_str(candidate, "adaptive_score")
    adaptive_fallback_mode = _req_str(candidate, "adaptive_fallback_mode")

    stake_mode = _req_str(candidate, "stake_mode")

    stake_min_bnb = _req_float(candidate, "stake_min_bnb")
    stake_max_bnb = _req_float(candidate, "stake_max_bnb")
    if stake_min_bnb <= 0.0 or stake_max_bnb <= 0.0:
        raise InvariantError("dislocation_candidate_stake_bounds_must_be_positive")

    stake_ev_ref_bnb = _req_float(candidate, "stake_ev_ref_bnb")
    if stake_ev_ref_bnb <= 0.0:
        raise InvariantError("dislocation_candidate_stake_ev_ref_bnb_must_be_positive")

    stake_max_side_pool_frac = _req_float(candidate, "stake_max_side_pool_frac")
    if stake_max_side_pool_frac <= 0.0:
        raise InvariantError("dislocation_candidate_stake_max_side_pool_frac_must_be_positive")

    perf_adapt_mode = _req_str(candidate, "perf_adapt_mode")

    perf_gate_window = _req_int(candidate, "perf_gate_window")
    if perf_gate_window < 0:
        raise InvariantError("dislocation_candidate_perf_gate_window_negative")

    perf_gate_min_history = _req_int(candidate, "perf_gate_min_history")
    if perf_gate_min_history < 0:
        raise InvariantError("dislocation_candidate_perf_gate_min_history_negative")

    perf_gate_min_win_rate = _req_float(candidate, "perf_gate_min_win_rate")
    if not (0.0 <= perf_gate_min_win_rate <= 1.0):
        raise InvariantError("dislocation_candidate_perf_gate_min_win_rate_out_of_range")

    perf_gate_min_mean_profit_bnb = _req_float(candidate, "perf_gate_min_mean_profit_bnb")

    return DislocationCandidateConfig(
        name=str(name),
        lookback1_seconds=int(lookback1_seconds),
        lookback2_seconds=int(lookback2_seconds),
        lookback3_seconds=int(lookback3_seconds),
        weight1=float(weight1),
        weight2=float(weight2),
        weight3=float(weight3),
        temperature_bps=float(temperature_bps),
        fixed_bet_bnb=float(fixed_bet_bnb),
        dislocation_threshold_pp=float(dislocation_threshold_pp),
        nowcast_confidence_min=float(nowcast_confidence_min),
        cutoff_pool_total_min_bnb=float(cutoff_pool_total_min_bnb),
        expected_net_min_bnb=float(expected_net_min_bnb),
        side_selection_mode=str(side_selection_mode),
        market_extreme_min=float(market_extreme_min),
        flow_window_seconds=int(flow_window_seconds),
        flow_min_imbalance=float(flow_min_imbalance),
        flow_gate_mode=str(flow_gate_mode),
        adaptive_candidate_modes=tuple(adaptive_candidate_modes),
        adaptive_window=int(adaptive_window),
        adaptive_min_history=int(adaptive_min_history),
        adaptive_score=str(adaptive_score),
        adaptive_fallback_mode=str(adaptive_fallback_mode),
        stake_mode=str(stake_mode),
        stake_min_bnb=float(stake_min_bnb),
        stake_max_bnb=float(stake_max_bnb),
        stake_ev_ref_bnb=float(stake_ev_ref_bnb),
        stake_max_side_pool_frac=float(stake_max_side_pool_frac),
        perf_adapt_mode=str(perf_adapt_mode),
        perf_gate_window=int(perf_gate_window),
        perf_gate_min_history=int(perf_gate_min_history),
        perf_gate_min_win_rate=float(perf_gate_min_win_rate),
        perf_gate_min_mean_profit_bnb=float(perf_gate_min_mean_profit_bnb),
    )


def _parse_strategy(strategy: dict[str, Any]) -> StrategyConfig:
    _validate_unknown_keys("strategy", strategy, {"dislocation"})

    dislocation = strategy.get("dislocation", {})
    if dislocation is None:
        dislocation = {}
    if not isinstance(dislocation, dict):
        raise InvariantError("config_section_not_dict: strategy.dislocation")

    _validate_unknown_keys("strategy_dislocation", dislocation, {"selector", "candidates"})

    selector_obj = dislocation.get("selector", {})
    if selector_obj is None:
        selector_obj = {}
    if not isinstance(selector_obj, dict):
        raise InvariantError("config_section_not_dict: strategy.dislocation.selector")
    selector_cfg = _parse_dislocation_selector(selector_obj)

    candidates_obj = _req(dislocation, "candidates")
    if not isinstance(candidates_obj, list):
        raise InvariantError("config_section_not_list: strategy.dislocation.candidates")
    if not candidates_obj:
        raise InvariantError("dislocation_candidates_empty")

    parsed: list[DislocationCandidateConfig] = []
    seen_names: set[str] = set()
    for i, item in enumerate(candidates_obj):
        if not isinstance(item, dict):
            raise InvariantError(f"dislocation_candidate_not_dict: idx={i}")
        cfg = _parse_dislocation_candidate(item, idx=i)
        key = str(cfg.name)
        if key in seen_names:
            raise InvariantError(f"dislocation_candidate_name_duplicate: {key}")
        seen_names.add(key)
        parsed.append(cfg)
    candidate_cfgs = tuple(parsed)

    return StrategyConfig(
        dislocation=DislocationStrategyConfig(
            selector=selector_cfg,
            candidates=candidate_cfgs,
        )
    )


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
    strategy = raw.get("strategy", {})
    backtest = raw.get("backtest", {})

    if not isinstance(paths, dict):
        raise InvariantError("config_section_missing_or_not_dict: paths")
    if not isinstance(graph, dict):
        raise InvariantError("config_section_missing_or_not_dict: graph")
    if not isinstance(runtime, dict):
        raise InvariantError("config_section_missing_or_not_dict: runtime")
    if strategy is None:
        strategy = {}
    if not isinstance(strategy, dict):
        raise InvariantError("config_section_not_dict: strategy")
    if backtest is None:
        backtest = {}
    if not isinstance(backtest, dict):
        raise InvariantError("config_section_not_dict: backtest")

    closed_rounds_path = _req_str(paths, "closed_rounds_path")
    klines_path = _opt_str(paths, "klines_path", "var/klines.jsonl")
    abi_json_path = _req_str(graph, "abi_json_path")

    allowed_runtime_keys = {
        "cutoff_seconds",
        "random_seed",
        "use_onchain_event_bets",
        "event_lookback_blocks",
        "latency_log_path",
        "wait_for_bet_receipt",
        "bet_receipt_timeout_seconds",
    }
    _validate_unknown_keys("runtime", runtime, allowed_runtime_keys)

    cutoff_seconds = _req_int(runtime, "cutoff_seconds")
    if cutoff_seconds <= 0:
        raise InvariantError("cutoff_seconds_must_be_positive")

    random_seed = _req_int(runtime, "random_seed")
    if random_seed < 0:
        raise InvariantError("random_seed_must_be_nonnegative")

    use_onchain_event_bets = _opt_bool(runtime, "use_onchain_event_bets", False)

    event_lookback_blocks = _opt_int(runtime, "event_lookback_blocks", 600)
    if event_lookback_blocks <= 0:
        raise InvariantError("event_lookback_blocks_must_be_positive")

    latency_log_path = _opt_str(runtime, "latency_log_path", "var/live_latency.jsonl")

    wait_for_bet_receipt = _opt_bool(runtime, "wait_for_bet_receipt", True)

    bet_receipt_timeout_seconds = _opt_int(runtime, "bet_receipt_timeout_seconds", 45)
    if bet_receipt_timeout_seconds <= 0:
        raise InvariantError("bet_receipt_timeout_seconds_must_be_positive")

    strategy_cfg = _parse_strategy(strategy)

    allowed_bt_keys = {
        "simulation_size",
        "initial_bankroll_bnb",
        "reset_mode",
        "reset_every_rounds",
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

    backtest_cfg = BacktestConfig(
        simulation_size=simulation_size_v,
        initial_bankroll_bnb=initial_bankroll_bnb,
        reset_mode=str(reset_mode),
        reset_every_rounds=int(reset_every_rounds),
    )
    backtest_cfg.validate()

    return AppConfig(
        closed_rounds_path=closed_rounds_path,
        klines_path=klines_path,
        abi_json_path=abi_json_path,
        cutoff_seconds=int(cutoff_seconds),
        random_seed=int(random_seed),
        use_onchain_event_bets=bool(use_onchain_event_bets),
        event_lookback_blocks=int(event_lookback_blocks),
        latency_log_path=str(latency_log_path),
        wait_for_bet_receipt=bool(wait_for_bet_receipt),
        bet_receipt_timeout_seconds=int(bet_receipt_timeout_seconds),
        strategy=strategy_cfg,
        backtest=backtest_cfg,
    )
