from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from pancakebot.backtest.config import BacktestConfig
from pancakebot.config.app_config import AppConfig, RuntimeStatePathsConfig
from pancakebot.config.strategy_config import (
    DislocationCandidateConfig,
    DislocationSelectorConfig,
    DislocationStrategyConfig,
    FlowCandidateConfig,
    MlCandidateConfig,
    StrategyConfig,
    StrategyRouterConfig,
)
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.predictability_modes import (
    DEFAULT_PREDICTABILITY_FEATURE_MODE,
    DEFAULT_PREDICTABILITY_LABEL_MODE,
    validate_predictability_feature_mode,
    validate_predictability_label_mode,
)


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


def _opt_list_str(obj: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if key not in obj:
        return tuple(default)
    raw = obj[key]
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


def _parse_strategy_router(router: dict[str, Any]) -> StrategyRouterConfig:
    defaults = StrategyRouterConfig()
    allowed = {
        "mode",
        "score_threshold_bnb",
        "online_warmup_rounds",
        "online_num_quantile_bins",
        "online_min_cell_obs",
        "online_score_threshold_bnb",
        "online_use_direction_split",
    }
    _validate_unknown_keys("strategy_router", router, allowed)

    mode = _opt_str(router, "mode", str(defaults.mode))
    if str(mode) not in (
        "selector_max_score",
        "skip_only",
        "oracle_skip",
        "online_cellmean",
        "online_cellmean_side_gap",
        "online_cellmean_backoff",
        "online_cellmean_selector_fallback",
        "online_cellmean_selector_gate",
        "online_selector_score_fallback",
        "online_selector_score_gate",
        "online_selector_score_side_gap",
        "online_selector_score_late_imb_fallback",
        "online_selector_score_late_imb_gate",
        "online_selector_score_side_late_fallback",
        "online_selector_score_side_late_gate",
        "online_selector_score_side_support_fallback",
        "online_selector_score_side_support_gate",
    ):
        raise InvariantError("strategy_router_mode_invalid")

    score_threshold_bnb = _opt_float(
        router,
        "score_threshold_bnb",
        float(defaults.score_threshold_bnb),
    )

    online_warmup_rounds = _opt_int(
        router,
        "online_warmup_rounds",
        int(defaults.online_warmup_rounds),
    )
    if int(online_warmup_rounds) <= 0:
        raise InvariantError("strategy_router_online_warmup_rounds_must_be_positive")

    online_num_quantile_bins = _opt_int(
        router,
        "online_num_quantile_bins",
        int(defaults.online_num_quantile_bins),
    )
    if int(online_num_quantile_bins) <= 1:
        raise InvariantError("strategy_router_online_num_quantile_bins_invalid")

    online_min_cell_obs = _opt_int(
        router,
        "online_min_cell_obs",
        int(defaults.online_min_cell_obs),
    )
    if int(online_min_cell_obs) <= 0:
        raise InvariantError("strategy_router_online_min_cell_obs_must_be_positive")

    online_score_threshold_bnb = _opt_float(
        router,
        "online_score_threshold_bnb",
        float(defaults.online_score_threshold_bnb),
    )
    online_use_direction_split = _opt_bool(
        router,
        "online_use_direction_split",
        bool(defaults.online_use_direction_split),
    )

    return StrategyRouterConfig(
        mode=str(mode),
        score_threshold_bnb=float(score_threshold_bnb),
        online_warmup_rounds=int(online_warmup_rounds),
        online_num_quantile_bins=int(online_num_quantile_bins),
        online_min_cell_obs=int(online_min_cell_obs),
        online_score_threshold_bnb=float(online_score_threshold_bnb),
        online_use_direction_split=bool(online_use_direction_split),
    )


def _parse_ml_candidate(candidate: dict[str, Any]) -> MlCandidateConfig:
    allowed = {
        "enabled",
        "name",
        "fixed_bet_bnb",
        "min_tradeable_prob",
        "min_prob_edge",
        "cutoff_pool_total_min_bnb",
        "expected_net_min_bnb",
        "expected_net_max_bnb",
        "train_size",
        "calibrate_size",
        "calibration_size",
        "retrain_interval",
        "recalibrate_interval",
        "recalibration_interval",
        "price_alpha",
        "pool_alpha_total",
        "pool_alpha_ratio",
        "recency_weight_floor",
        "recency_weight_power",
        "predictability_baseline_bet_bnb",
        "predictability_feature_mode",
        "predictability_label_mode",
        "emit_candidate",
        "veto_opposite_side_candidates",
        "veto_untradeable_candidates",
        "veto_candidate_expected_net_below_min",
        "rescore_baseline_candidates_with_expected_net",
        "candidate_profit_model_enabled",
        "candidate_profit_model_warmup_rounds",
        "candidate_profit_model_num_quantile_bins",
        "candidate_profit_model_min_cell_obs",
        "random_seed",
    }
    _validate_unknown_keys("strategy_ml_candidate", candidate, allowed)

    enabled = _req_bool(candidate, "enabled")
    name = _req_str(candidate, "name")
    fixed_bet_bnb = _req_float(candidate, "fixed_bet_bnb")
    if float(fixed_bet_bnb) <= 0.0:
        raise InvariantError("strategy_ml_candidate_fixed_bet_bnb_must_be_positive")

    min_tradeable_prob = _req_float(candidate, "min_tradeable_prob")
    if not (0.0 <= float(min_tradeable_prob) <= 1.0):
        raise InvariantError("strategy_ml_candidate_min_tradeable_prob_out_of_range")

    min_prob_edge = _req_float(candidate, "min_prob_edge")
    if float(min_prob_edge) < 0.0:
        raise InvariantError("strategy_ml_candidate_min_prob_edge_negative")

    cutoff_pool_total_min_bnb = _req_float(candidate, "cutoff_pool_total_min_bnb")
    if float(cutoff_pool_total_min_bnb) < 0.0:
        raise InvariantError("strategy_ml_candidate_cutoff_pool_total_min_bnb_negative")

    expected_net_min_bnb = _req_float(candidate, "expected_net_min_bnb")
    expected_net_max_bnb: float | None = None
    if "expected_net_max_bnb" in candidate:
        expected_net_max_bnb = _req_float(candidate, "expected_net_max_bnb")
        if float(expected_net_max_bnb) < 0.0:
            raise InvariantError("strategy_ml_candidate_expected_net_max_bnb_negative")
        if float(expected_net_max_bnb) < float(expected_net_min_bnb):
            raise InvariantError("strategy_ml_candidate_expected_net_max_below_min")

    def _req_int_with_alias(*, primary: str, alias: str) -> int:
        has_primary = primary in candidate
        has_alias = alias in candidate
        if bool(has_primary) and bool(has_alias):
            primary_value = _req_int(candidate, primary)
            alias_value = _req_int(candidate, alias)
            if int(primary_value) != int(alias_value):
                raise InvariantError(f"strategy_ml_candidate_alias_conflict: {primary} vs {alias}")
            return int(primary_value)
        if bool(has_primary):
            return int(_req_int(candidate, primary))
        if bool(has_alias):
            return int(_req_int(candidate, alias))
        raise InvariantError(f"missing_config_key: {primary}")

    train_size = _req_int(candidate, "train_size")
    calibrate_size = _req_int_with_alias(primary="calibrate_size", alias="calibration_size")
    retrain_interval = _req_int(candidate, "retrain_interval")
    recalibrate_interval = _req_int_with_alias(primary="recalibrate_interval", alias="recalibration_interval")
    if int(train_size) <= 0 or int(calibrate_size) < 0:
        raise InvariantError("strategy_ml_candidate_train_or_calibrate_size_invalid")
    if int(retrain_interval) <= 0 or int(recalibrate_interval) < 0:
        raise InvariantError("strategy_ml_candidate_retrain_or_recalibrate_interval_invalid")

    price_alpha = _req_float(candidate, "price_alpha")
    pool_alpha_total = _req_float(candidate, "pool_alpha_total")
    pool_alpha_ratio = _req_float(candidate, "pool_alpha_ratio")
    if float(price_alpha) <= 0.0 or float(pool_alpha_total) <= 0.0 or float(pool_alpha_ratio) <= 0.0:
        raise InvariantError("strategy_ml_candidate_alpha_must_be_positive")

    recency_weight_floor = _req_float(candidate, "recency_weight_floor")
    recency_weight_power = _req_float(candidate, "recency_weight_power")
    if not (0.0 < float(recency_weight_floor) <= 1.0):
        raise InvariantError("strategy_ml_candidate_recency_weight_floor_out_of_range")
    if float(recency_weight_power) <= 0.0:
        raise InvariantError("strategy_ml_candidate_recency_weight_power_must_be_positive")

    predictability_baseline_bet_bnb = _req_float(candidate, "predictability_baseline_bet_bnb")
    if float(predictability_baseline_bet_bnb) <= 0.0:
        raise InvariantError("strategy_ml_candidate_predictability_baseline_bet_bnb_must_be_positive")

    predictability_feature_mode = validate_predictability_feature_mode(
        _opt_str(candidate, "predictability_feature_mode", DEFAULT_PREDICTABILITY_FEATURE_MODE)
    )
    predictability_label_mode = validate_predictability_label_mode(
        _opt_str(candidate, "predictability_label_mode", DEFAULT_PREDICTABILITY_LABEL_MODE)
    )
    emit_candidate = _opt_bool(
        candidate,
        "emit_candidate",
        bool(MlCandidateConfig.__dataclass_fields__["emit_candidate"].default),
    )
    veto_opposite_side_candidates = _opt_bool(
        candidate,
        "veto_opposite_side_candidates",
        bool(MlCandidateConfig.__dataclass_fields__["veto_opposite_side_candidates"].default),
    )
    veto_untradeable_candidates = _opt_bool(
        candidate,
        "veto_untradeable_candidates",
        bool(MlCandidateConfig.__dataclass_fields__["veto_untradeable_candidates"].default),
    )
    veto_candidate_expected_net_below_min = _opt_bool(
        candidate,
        "veto_candidate_expected_net_below_min",
        bool(MlCandidateConfig.__dataclass_fields__["veto_candidate_expected_net_below_min"].default),
    )
    rescore_baseline_candidates_with_expected_net = _opt_bool(
        candidate,
        "rescore_baseline_candidates_with_expected_net",
        bool(MlCandidateConfig.__dataclass_fields__["rescore_baseline_candidates_with_expected_net"].default),
    )
    candidate_profit_model_enabled = _opt_bool(
        candidate,
        "candidate_profit_model_enabled",
        bool(MlCandidateConfig.__dataclass_fields__["candidate_profit_model_enabled"].default),
    )
    candidate_profit_model_warmup_rounds = _opt_int(
        candidate,
        "candidate_profit_model_warmup_rounds",
        int(MlCandidateConfig.__dataclass_fields__["candidate_profit_model_warmup_rounds"].default),
    )
    if int(candidate_profit_model_warmup_rounds) <= 0:
        raise InvariantError("strategy_ml_candidate_profit_model_warmup_rounds_nonpositive")
    candidate_profit_model_num_quantile_bins = _opt_int(
        candidate,
        "candidate_profit_model_num_quantile_bins",
        int(MlCandidateConfig.__dataclass_fields__["candidate_profit_model_num_quantile_bins"].default),
    )
    if int(candidate_profit_model_num_quantile_bins) <= 1:
        raise InvariantError("strategy_ml_candidate_profit_model_num_quantile_bins_invalid")
    candidate_profit_model_min_cell_obs = _opt_int(
        candidate,
        "candidate_profit_model_min_cell_obs",
        int(MlCandidateConfig.__dataclass_fields__["candidate_profit_model_min_cell_obs"].default),
    )
    if int(candidate_profit_model_min_cell_obs) <= 0:
        raise InvariantError("strategy_ml_candidate_profit_model_min_cell_obs_nonpositive")

    random_seed = _req_int(candidate, "random_seed")
    if int(random_seed) < 0:
        raise InvariantError("strategy_ml_candidate_random_seed_negative")

    return MlCandidateConfig(
        enabled=bool(enabled),
        name=str(name),
        fixed_bet_bnb=float(fixed_bet_bnb),
        min_tradeable_prob=float(min_tradeable_prob),
        min_prob_edge=float(min_prob_edge),
        cutoff_pool_total_min_bnb=float(cutoff_pool_total_min_bnb),
        expected_net_min_bnb=float(expected_net_min_bnb),
        train_size=int(train_size),
        calibrate_size=int(calibrate_size),
        retrain_interval=int(retrain_interval),
        recalibrate_interval=int(recalibrate_interval),
        price_alpha=float(price_alpha),
        pool_alpha_total=float(pool_alpha_total),
        pool_alpha_ratio=float(pool_alpha_ratio),
        recency_weight_floor=float(recency_weight_floor),
        recency_weight_power=float(recency_weight_power),
        predictability_baseline_bet_bnb=float(predictability_baseline_bet_bnb),
        random_seed=int(random_seed),
        expected_net_max_bnb=(
            None if expected_net_max_bnb is None else float(expected_net_max_bnb)
        ),
        predictability_feature_mode=str(predictability_feature_mode),
        predictability_label_mode=str(predictability_label_mode),
        emit_candidate=bool(emit_candidate),
        veto_opposite_side_candidates=bool(veto_opposite_side_candidates),
        veto_untradeable_candidates=bool(veto_untradeable_candidates),
        veto_candidate_expected_net_below_min=bool(veto_candidate_expected_net_below_min),
        rescore_baseline_candidates_with_expected_net=bool(rescore_baseline_candidates_with_expected_net),
        candidate_profit_model_enabled=bool(candidate_profit_model_enabled),
        candidate_profit_model_warmup_rounds=int(candidate_profit_model_warmup_rounds),
        candidate_profit_model_num_quantile_bins=int(candidate_profit_model_num_quantile_bins),
        candidate_profit_model_min_cell_obs=int(candidate_profit_model_min_cell_obs),
    )

def _parse_flow_candidate(candidate: dict[str, Any]) -> FlowCandidateConfig:
    defaults = FlowCandidateConfig()
    allowed = {
        "enabled",
        "name",
        "shadow_initial_bankroll_bnb",
        "train_size",
        "retrain_interval",
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "subsample",
        "colsample_bytree",
        "random_seed",
        "ev_threshold",
        "kelly_fraction",
        "max_fraction",
        "max_bet_abs",
        "min_bet_size",
        "round_to",
        "min_total_pool_c",
        "max_total_pool_share",
        "max_side_pool_share",
        "min_bull_ratio",
        "max_bull_ratio",
        "allowed_sides",
        "selector_score_penalty_bnb",
        "vol_mid",
        "drawdown_stop_pct",
        "drawdown_throttle_start_pct",
        "drawdown_throttle_min_scale",
        "roll_window",
        "roll_edge_min",
        "roll_winrate_min",
        "cooldown_trades",
    }
    _validate_unknown_keys("flow_candidate", candidate, allowed)

    enabled = _opt_bool(candidate, "enabled", bool(defaults.enabled))
    name = _opt_str(candidate, "name", str(defaults.name))
    shadow_initial_bankroll_bnb = _opt_float(
        candidate,
        "shadow_initial_bankroll_bnb",
        float(defaults.shadow_initial_bankroll_bnb),
    )
    if float(shadow_initial_bankroll_bnb) <= 0.0:
        raise InvariantError("strategy_flow_candidate_shadow_initial_bankroll_bnb_must_be_positive")
    train_size = _opt_int(candidate, "train_size", int(defaults.train_size))
    if int(train_size) <= 0:
        raise InvariantError("strategy_flow_candidate_train_size_nonpositive")
    retrain_interval = _opt_int(candidate, "retrain_interval", int(defaults.retrain_interval))
    if int(retrain_interval) <= 0:
        raise InvariantError("strategy_flow_candidate_retrain_interval_nonpositive")
    n_estimators = _opt_int(candidate, "n_estimators", int(defaults.n_estimators))
    if int(n_estimators) <= 0:
        raise InvariantError("strategy_flow_candidate_n_estimators_nonpositive")
    learning_rate = _opt_float(candidate, "learning_rate", float(defaults.learning_rate))
    if float(learning_rate) <= 0.0:
        raise InvariantError("strategy_flow_candidate_learning_rate_nonpositive")
    num_leaves = _opt_int(candidate, "num_leaves", int(defaults.num_leaves))
    if int(num_leaves) <= 1:
        raise InvariantError("strategy_flow_candidate_num_leaves_invalid")
    subsample = _opt_float(candidate, "subsample", float(defaults.subsample))
    if not (0.0 < float(subsample) <= 1.0):
        raise InvariantError("strategy_flow_candidate_subsample_out_of_range")
    colsample_bytree = _opt_float(candidate, "colsample_bytree", float(defaults.colsample_bytree))
    if not (0.0 < float(colsample_bytree) <= 1.0):
        raise InvariantError("strategy_flow_candidate_colsample_bytree_out_of_range")
    random_seed = _opt_int(candidate, "random_seed", int(defaults.random_seed))
    if int(random_seed) < 0:
        raise InvariantError("strategy_flow_candidate_random_seed_negative")
    ev_threshold = _opt_float(candidate, "ev_threshold", float(defaults.ev_threshold))
    if float(ev_threshold) < 0.0:
        raise InvariantError("strategy_flow_candidate_ev_threshold_negative")
    kelly_fraction = _opt_float(candidate, "kelly_fraction", float(defaults.kelly_fraction))
    if float(kelly_fraction) < 0.0:
        raise InvariantError("strategy_flow_candidate_kelly_fraction_negative")
    max_fraction = _opt_float(candidate, "max_fraction", float(defaults.max_fraction))
    if not (0.0 <= float(max_fraction) <= 1.0):
        raise InvariantError("strategy_flow_candidate_max_fraction_out_of_range")
    max_bet_abs = _opt_float(candidate, "max_bet_abs", float(defaults.max_bet_abs))
    if float(max_bet_abs) <= 0.0:
        raise InvariantError("strategy_flow_candidate_max_bet_abs_nonpositive")
    min_bet_size = _opt_float(candidate, "min_bet_size", float(defaults.min_bet_size))
    if float(min_bet_size) <= 0.0:
        raise InvariantError("strategy_flow_candidate_min_bet_size_nonpositive")
    round_to = _opt_float(candidate, "round_to", float(defaults.round_to))
    if float(round_to) <= 0.0:
        raise InvariantError("strategy_flow_candidate_round_to_nonpositive")
    min_total_pool_c = _opt_float(candidate, "min_total_pool_c", float(defaults.min_total_pool_c))
    if float(min_total_pool_c) < 0.0:
        raise InvariantError("strategy_flow_candidate_min_total_pool_c_negative")
    max_total_pool_share = _opt_float(
        candidate,
        "max_total_pool_share",
        float(defaults.max_total_pool_share),
    )
    if not (0.0 < float(max_total_pool_share) <= 1.0):
        raise InvariantError("strategy_flow_candidate_max_total_pool_share_out_of_range")
    max_side_pool_share = _opt_float(
        candidate,
        "max_side_pool_share",
        float(defaults.max_side_pool_share),
    )
    if not (0.0 < float(max_side_pool_share) <= 1.0):
        raise InvariantError("strategy_flow_candidate_max_side_pool_share_out_of_range")
    min_bull_ratio = _opt_float(candidate, "min_bull_ratio", float(defaults.min_bull_ratio))
    max_bull_ratio = _opt_float(candidate, "max_bull_ratio", float(defaults.max_bull_ratio))
    if not (0.0 <= float(min_bull_ratio) <= float(max_bull_ratio) <= 1.0):
        raise InvariantError("strategy_flow_candidate_bull_ratio_out_of_range")
    allowed_sides = _opt_str(candidate, "allowed_sides", str(defaults.allowed_sides))
    if str(allowed_sides) not in ("both", "bull_only", "bear_only"):
        raise InvariantError("strategy_flow_candidate_allowed_sides_invalid")
    selector_score_penalty_bnb = _opt_float(
        candidate,
        "selector_score_penalty_bnb",
        float(defaults.selector_score_penalty_bnb),
    )
    if float(selector_score_penalty_bnb) < 0.0:
        raise InvariantError("strategy_flow_candidate_selector_score_penalty_negative")
    vol_mid = _opt_float(candidate, "vol_mid", float(defaults.vol_mid))
    if float(vol_mid) < 0.0:
        raise InvariantError("strategy_flow_candidate_vol_mid_negative")
    drawdown_stop_pct = _opt_float(
        candidate,
        "drawdown_stop_pct",
        float(defaults.drawdown_stop_pct),
    )
    if not (0.0 < float(drawdown_stop_pct) <= 1.0):
        raise InvariantError("strategy_flow_candidate_drawdown_stop_pct_out_of_range")
    drawdown_throttle_start_pct = _opt_float(
        candidate,
        "drawdown_throttle_start_pct",
        float(defaults.drawdown_throttle_start_pct),
    )
    if not (0.0 <= float(drawdown_throttle_start_pct) <= float(drawdown_stop_pct)):
        raise InvariantError("strategy_flow_candidate_drawdown_throttle_start_pct_out_of_range")
    drawdown_throttle_min_scale = _opt_float(
        candidate,
        "drawdown_throttle_min_scale",
        float(defaults.drawdown_throttle_min_scale),
    )
    if not (0.0 < float(drawdown_throttle_min_scale) <= 1.0):
        raise InvariantError("strategy_flow_candidate_drawdown_throttle_min_scale_out_of_range")
    roll_window = _opt_int(candidate, "roll_window", int(defaults.roll_window))
    if int(roll_window) <= 0:
        raise InvariantError("strategy_flow_candidate_roll_window_nonpositive")
    roll_edge_min = _opt_float(candidate, "roll_edge_min", float(defaults.roll_edge_min))
    roll_winrate_min = _opt_float(candidate, "roll_winrate_min", float(defaults.roll_winrate_min))
    if not (0.0 <= float(roll_winrate_min) <= 1.0):
        raise InvariantError("strategy_flow_candidate_roll_winrate_min_out_of_range")
    cooldown_trades = _opt_int(candidate, "cooldown_trades", int(defaults.cooldown_trades))
    if int(cooldown_trades) < 0:
        raise InvariantError("strategy_flow_candidate_cooldown_trades_negative")

    return FlowCandidateConfig(
        enabled=bool(enabled),
        name=str(name),
        shadow_initial_bankroll_bnb=float(shadow_initial_bankroll_bnb),
        train_size=int(train_size),
        retrain_interval=int(retrain_interval),
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=int(num_leaves),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        random_seed=int(random_seed),
        ev_threshold=float(ev_threshold),
        kelly_fraction=float(kelly_fraction),
        max_fraction=float(max_fraction),
        max_bet_abs=float(max_bet_abs),
        min_bet_size=float(min_bet_size),
        round_to=float(round_to),
        min_total_pool_c=float(min_total_pool_c),
        max_total_pool_share=float(max_total_pool_share),
        max_side_pool_share=float(max_side_pool_share),
        min_bull_ratio=float(min_bull_ratio),
        max_bull_ratio=float(max_bull_ratio),
        allowed_sides=str(allowed_sides),
        selector_score_penalty_bnb=float(selector_score_penalty_bnb),
        vol_mid=float(vol_mid),
        drawdown_stop_pct=float(drawdown_stop_pct),
        drawdown_throttle_start_pct=float(drawdown_throttle_start_pct),
        drawdown_throttle_min_scale=float(drawdown_throttle_min_scale),
        roll_window=int(roll_window),
        roll_edge_min=float(roll_edge_min),
        roll_winrate_min=float(roll_winrate_min),
        cooldown_trades=int(cooldown_trades),
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
        "pool_total_gate_mode",
        "projected_final_pool_multiplier",
        "projected_final_pool_total_min_bnb",
        "expected_net_min_bnb",
        "bull_expected_net_extra_min_bnb",
        "bear_expected_net_extra_min_bnb",
        "bull_late_min_ratio",
        "bull_late_min_imbalance",
        "bear_late_min_ratio",
        "bear_late_max_imbalance",
        "late_support_ev_scale_bnb",
        "side_selection_mode",
        "allowed_sides",
        "market_extreme_min",
        "nowcast_market_gap_min",
        "flow_window_seconds",
        "flow_min_imbalance",
        "flow_gate_mode",
        "flow_gate_relax_dislocation_min",
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
        "drawdown_stake_guard_enabled",
        "drawdown_stake_guard_start_bnb",
        "drawdown_stake_guard_full_bnb",
        "drawdown_stake_guard_min_scale",
        "anti_martingale_enabled",
        "anti_martingale_win_multiplier",
        "anti_martingale_loss_multiplier",
        "anti_martingale_min_scale",
        "anti_martingale_max_scale",
        "circuit_breaker_enabled",
        "circuit_breaker_drawdown_trigger_bnb",
        "circuit_breaker_base_skip_rounds",
        "circuit_breaker_escalation_multiplier",
        "circuit_breaker_escalation_window_rounds",
        "circuit_breaker_max_level",
        "circuit_breaker_max_skip_rounds",
        "circuit_breaker_reentry_rounds",
        "circuit_breaker_reentry_scale",
        "perf_adapt_mode",
        "perf_gate_window",
        "perf_gate_min_history",
        "perf_gate_min_win_rate",
        "perf_gate_min_mean_profit_bnb",
        "robust_ev_veto_enabled",
        "robust_ev_veto_min_history",
        "robust_ev_veto_window",
        "robust_ev_veto_low_inflow_mult",
        "robust_ev_veto_extreme_inflow_mult",
        "robust_ev_veto_adverse_skew",
        "robust_ev_veto_min_expected_net_bnb",
        "shock_filter_enabled",
        "shock_filter_window_seconds",
        "shock_filter_min_window_total_bnb",
        "shock_filter_min_abs_imbalance",
        "shock_filter_min_surge_ratio",
        "late_model_conflict_flip_enabled",
        "late_model_veto_enabled",
        "late_model_veto_min_late_ratio",
        "late_model_veto_min_abs_imbalance",
        "late_model_neutral_filter_enabled",
        "late_model_neutral_min_late_ratio",
        "late_model_neutral_max_abs_imbalance",
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

    pool_total_gate_mode = _opt_str(candidate, "pool_total_gate_mode", "cutoff_only")
    if str(pool_total_gate_mode) not in (
        "cutoff_only",
        "projected_final_only",
        "projected_final_model_only",
    ):
        raise InvariantError("dislocation_candidate_pool_total_gate_mode_invalid")

    projected_final_pool_multiplier = _opt_float(
        candidate,
        "projected_final_pool_multiplier",
        1.0,
    )
    if float(projected_final_pool_multiplier) <= 0.0:
        raise InvariantError(
            "dislocation_candidate_projected_final_pool_multiplier_must_be_positive"
        )

    projected_final_pool_total_min_bnb = _opt_float(
        candidate,
        "projected_final_pool_total_min_bnb",
        0.0,
    )
    if float(projected_final_pool_total_min_bnb) < 0.0:
        raise InvariantError("dislocation_candidate_projected_final_pool_total_min_bnb_negative")

    expected_net_min_bnb = _req_float(candidate, "expected_net_min_bnb")
    bull_expected_net_extra_min_bnb = _opt_float(candidate, "bull_expected_net_extra_min_bnb", 0.0)
    if float(bull_expected_net_extra_min_bnb) < 0.0:
        raise InvariantError("dislocation_candidate_bull_expected_net_extra_min_bnb_negative")
    bear_expected_net_extra_min_bnb = _opt_float(candidate, "bear_expected_net_extra_min_bnb", 0.0)
    if float(bear_expected_net_extra_min_bnb) < 0.0:
        raise InvariantError("dislocation_candidate_bear_expected_net_extra_min_bnb_negative")
    bull_late_min_ratio = _opt_float(candidate, "bull_late_min_ratio", 0.0)
    if not (0.0 <= float(bull_late_min_ratio) <= 1.0):
        raise InvariantError("dislocation_candidate_bull_late_min_ratio_out_of_range")
    bull_late_min_imbalance = _opt_float(candidate, "bull_late_min_imbalance", -1.0)
    if not (-1.0 <= float(bull_late_min_imbalance) <= 1.0):
        raise InvariantError("dislocation_candidate_bull_late_min_imbalance_out_of_range")
    bear_late_min_ratio = _opt_float(candidate, "bear_late_min_ratio", 0.0)
    if not (0.0 <= float(bear_late_min_ratio) <= 1.0):
        raise InvariantError("dislocation_candidate_bear_late_min_ratio_out_of_range")
    bear_late_max_imbalance = _opt_float(candidate, "bear_late_max_imbalance", 1.0)
    if not (-1.0 <= float(bear_late_max_imbalance) <= 1.0):
        raise InvariantError("dislocation_candidate_bear_late_max_imbalance_out_of_range")
    late_support_ev_scale_bnb = _opt_float(candidate, "late_support_ev_scale_bnb", 0.0)
    if float(late_support_ev_scale_bnb) < 0.0:
        raise InvariantError("dislocation_candidate_late_support_ev_scale_bnb_negative")

    side_selection_mode = _req_str(candidate, "side_selection_mode")
    allowed_sides = _opt_str(candidate, "allowed_sides", "both")
    if str(allowed_sides) not in ("both", "bull_only", "bear_only"):
        raise InvariantError("dislocation_candidate_allowed_sides_invalid")

    market_extreme_min = _req_float(candidate, "market_extreme_min")
    if market_extreme_min < 0.0:
        raise InvariantError("dislocation_candidate_market_extreme_min_negative")

    nowcast_market_gap_min = _opt_float(candidate, "nowcast_market_gap_min", 0.0)
    if float(nowcast_market_gap_min) < 0.0:
        raise InvariantError("dislocation_candidate_nowcast_market_gap_min_negative")

    flow_window_seconds = _req_int(candidate, "flow_window_seconds")
    if flow_window_seconds < 0:
        raise InvariantError("dislocation_candidate_flow_window_seconds_negative")

    flow_min_imbalance = _req_float(candidate, "flow_min_imbalance")
    if flow_min_imbalance < 0.0:
        raise InvariantError("dislocation_candidate_flow_min_imbalance_negative")

    flow_gate_mode = _req_str(candidate, "flow_gate_mode")
    flow_gate_relax_dislocation_min = _opt_float(candidate, "flow_gate_relax_dislocation_min", 1.0)
    if not (0.0 <= float(flow_gate_relax_dislocation_min) <= 1.0):
        raise InvariantError("dislocation_candidate_flow_gate_relax_dislocation_min_out_of_range")

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

    drawdown_stake_guard_enabled = _opt_bool(candidate, "drawdown_stake_guard_enabled", False)

    drawdown_stake_guard_start_bnb = _opt_float(candidate, "drawdown_stake_guard_start_bnb", 0.0)
    if float(drawdown_stake_guard_start_bnb) < 0.0:
        raise InvariantError("dislocation_candidate_drawdown_stake_guard_start_bnb_negative")

    drawdown_stake_guard_full_bnb = _opt_float(candidate, "drawdown_stake_guard_full_bnb", 0.0)
    if float(drawdown_stake_guard_full_bnb) < 0.0:
        raise InvariantError("dislocation_candidate_drawdown_stake_guard_full_bnb_negative")

    if float(drawdown_stake_guard_full_bnb) > 0.0 and float(drawdown_stake_guard_full_bnb) < float(
        drawdown_stake_guard_start_bnb
    ):
        raise InvariantError("dislocation_candidate_drawdown_stake_guard_full_below_start")

    drawdown_stake_guard_min_scale = _opt_float(candidate, "drawdown_stake_guard_min_scale", 1.0)
    if not (0.0 < float(drawdown_stake_guard_min_scale) <= 1.0):
        raise InvariantError("dislocation_candidate_drawdown_stake_guard_min_scale_out_of_range")

    anti_martingale_enabled = _opt_bool(candidate, "anti_martingale_enabled", False)

    anti_martingale_win_multiplier = _opt_float(candidate, "anti_martingale_win_multiplier", 1.15)
    if float(anti_martingale_win_multiplier) <= 0.0:
        raise InvariantError("dislocation_candidate_anti_martingale_win_multiplier_nonpositive")

    anti_martingale_loss_multiplier = _opt_float(candidate, "anti_martingale_loss_multiplier", 0.9)
    if float(anti_martingale_loss_multiplier) <= 0.0:
        raise InvariantError("dislocation_candidate_anti_martingale_loss_multiplier_nonpositive")

    anti_martingale_min_scale = _opt_float(candidate, "anti_martingale_min_scale", 0.5)
    anti_martingale_max_scale = _opt_float(candidate, "anti_martingale_max_scale", 1.5)
    if float(anti_martingale_min_scale) <= 0.0:
        raise InvariantError("dislocation_candidate_anti_martingale_min_scale_nonpositive")
    if float(anti_martingale_max_scale) <= 0.0:
        raise InvariantError("dislocation_candidate_anti_martingale_max_scale_nonpositive")
    if float(anti_martingale_min_scale) > float(anti_martingale_max_scale):
        raise InvariantError("dislocation_candidate_anti_martingale_scale_bounds_invalid")

    circuit_breaker_enabled = _opt_bool(candidate, "circuit_breaker_enabled", False)
    circuit_breaker_drawdown_trigger_bnb = _opt_float(candidate, "circuit_breaker_drawdown_trigger_bnb", 0.0)
    if float(circuit_breaker_drawdown_trigger_bnb) < 0.0:
        raise InvariantError("dislocation_candidate_circuit_breaker_drawdown_trigger_bnb_negative")

    circuit_breaker_base_skip_rounds = _opt_int(candidate, "circuit_breaker_base_skip_rounds", 0)
    if int(circuit_breaker_base_skip_rounds) < 0:
        raise InvariantError("dislocation_candidate_circuit_breaker_base_skip_rounds_negative")

    circuit_breaker_escalation_multiplier = _opt_float(candidate, "circuit_breaker_escalation_multiplier", 1.5)
    if float(circuit_breaker_escalation_multiplier) < 1.0:
        raise InvariantError("dislocation_candidate_circuit_breaker_escalation_multiplier_invalid")

    circuit_breaker_escalation_window_rounds = _opt_int(candidate, "circuit_breaker_escalation_window_rounds", 200)
    if int(circuit_breaker_escalation_window_rounds) <= 0:
        raise InvariantError("dislocation_candidate_circuit_breaker_escalation_window_rounds_nonpositive")

    circuit_breaker_max_level = _opt_int(candidate, "circuit_breaker_max_level", 6)
    if int(circuit_breaker_max_level) <= 0:
        raise InvariantError("dislocation_candidate_circuit_breaker_max_level_nonpositive")

    circuit_breaker_max_skip_rounds = _opt_int(candidate, "circuit_breaker_max_skip_rounds", 0)
    if int(circuit_breaker_max_skip_rounds) < 0:
        raise InvariantError("dislocation_candidate_circuit_breaker_max_skip_rounds_negative")

    circuit_breaker_reentry_rounds = _opt_int(candidate, "circuit_breaker_reentry_rounds", 0)
    if int(circuit_breaker_reentry_rounds) < 0:
        raise InvariantError("dislocation_candidate_circuit_breaker_reentry_rounds_negative")

    circuit_breaker_reentry_scale = _opt_float(candidate, "circuit_breaker_reentry_scale", 1.0)
    if not (0.0 < float(circuit_breaker_reentry_scale) <= 1.0):
        raise InvariantError("dislocation_candidate_circuit_breaker_reentry_scale_out_of_range")

    if bool(circuit_breaker_enabled):
        if float(circuit_breaker_drawdown_trigger_bnb) <= 0.0:
            raise InvariantError("dislocation_candidate_circuit_breaker_drawdown_trigger_bnb_nonpositive")
        if int(circuit_breaker_base_skip_rounds) <= 0:
            raise InvariantError("dislocation_candidate_circuit_breaker_base_skip_rounds_nonpositive")

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

    robust_ev_veto_enabled = _opt_bool(candidate, "robust_ev_veto_enabled", False)

    robust_ev_veto_min_history = _opt_int(candidate, "robust_ev_veto_min_history", 200)
    if int(robust_ev_veto_min_history) <= 0:
        raise InvariantError("dislocation_candidate_robust_ev_veto_min_history_nonpositive")

    robust_ev_veto_window = _opt_int(candidate, "robust_ev_veto_window", 4000)
    if int(robust_ev_veto_window) <= 0:
        raise InvariantError("dislocation_candidate_robust_ev_veto_window_nonpositive")

    robust_ev_veto_low_inflow_mult = _opt_float(candidate, "robust_ev_veto_low_inflow_mult", 0.5)
    if float(robust_ev_veto_low_inflow_mult) < 0.0:
        raise InvariantError("dislocation_candidate_robust_ev_veto_low_inflow_mult_negative")

    robust_ev_veto_extreme_inflow_mult = _opt_float(candidate, "robust_ev_veto_extreme_inflow_mult", 0.2)
    if float(robust_ev_veto_extreme_inflow_mult) < 0.0:
        raise InvariantError("dislocation_candidate_robust_ev_veto_extreme_inflow_mult_negative")

    robust_ev_veto_adverse_skew = _opt_float(candidate, "robust_ev_veto_adverse_skew", 0.15)
    if not (0.0 <= float(robust_ev_veto_adverse_skew) <= 0.49):
        raise InvariantError("dislocation_candidate_robust_ev_veto_adverse_skew_out_of_range")

    robust_ev_veto_min_expected_net_bnb = _opt_float(
        candidate,
        "robust_ev_veto_min_expected_net_bnb",
        0.0,
    )

    shock_filter_enabled = _opt_bool(candidate, "shock_filter_enabled", False)

    shock_filter_window_seconds = _opt_int(candidate, "shock_filter_window_seconds", 20)
    if int(shock_filter_window_seconds) <= 0:
        raise InvariantError("dislocation_candidate_shock_filter_window_seconds_nonpositive")

    shock_filter_min_window_total_bnb = _opt_float(
        candidate,
        "shock_filter_min_window_total_bnb",
        0.25,
    )
    if float(shock_filter_min_window_total_bnb) < 0.0:
        raise InvariantError("dislocation_candidate_shock_filter_min_window_total_bnb_negative")

    shock_filter_min_abs_imbalance = _opt_float(
        candidate,
        "shock_filter_min_abs_imbalance",
        0.8,
    )
    if not (0.0 <= float(shock_filter_min_abs_imbalance) <= 1.0):
        raise InvariantError("dislocation_candidate_shock_filter_min_abs_imbalance_out_of_range")

    shock_filter_min_surge_ratio = _opt_float(
        candidate,
        "shock_filter_min_surge_ratio",
        2.5,
    )
    if float(shock_filter_min_surge_ratio) < 0.0:
        raise InvariantError("dislocation_candidate_shock_filter_min_surge_ratio_negative")

    late_model_conflict_flip_enabled = _opt_bool(candidate, "late_model_conflict_flip_enabled", False)

    late_model_veto_enabled = _opt_bool(candidate, "late_model_veto_enabled", False)

    late_model_veto_min_late_ratio = _opt_float(
        candidate,
        "late_model_veto_min_late_ratio",
        0.45,
    )
    if float(late_model_veto_min_late_ratio) < 0.0:
        raise InvariantError("dislocation_candidate_late_model_veto_min_late_ratio_negative")

    late_model_veto_min_abs_imbalance = _opt_float(
        candidate,
        "late_model_veto_min_abs_imbalance",
        0.35,
    )
    if not (0.0 <= float(late_model_veto_min_abs_imbalance) <= 1.0):
        raise InvariantError("dislocation_candidate_late_model_veto_min_abs_imbalance_out_of_range")

    late_model_neutral_filter_enabled = _opt_bool(candidate, "late_model_neutral_filter_enabled", False)

    late_model_neutral_min_late_ratio = _opt_float(
        candidate,
        "late_model_neutral_min_late_ratio",
        0.0,
    )
    if float(late_model_neutral_min_late_ratio) < 0.0:
        raise InvariantError("dislocation_candidate_late_model_neutral_min_late_ratio_negative")

    late_model_neutral_max_abs_imbalance = _opt_float(
        candidate,
        "late_model_neutral_max_abs_imbalance",
        1.0,
    )
    if not (0.0 <= float(late_model_neutral_max_abs_imbalance) <= 1.0):
        raise InvariantError("dislocation_candidate_late_model_neutral_max_abs_imbalance_out_of_range")

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
        pool_total_gate_mode=str(pool_total_gate_mode),
        projected_final_pool_multiplier=float(projected_final_pool_multiplier),
        projected_final_pool_total_min_bnb=float(projected_final_pool_total_min_bnb),
        expected_net_min_bnb=float(expected_net_min_bnb),
        bull_expected_net_extra_min_bnb=float(bull_expected_net_extra_min_bnb),
        bear_expected_net_extra_min_bnb=float(bear_expected_net_extra_min_bnb),
        bull_late_min_ratio=float(bull_late_min_ratio),
        bull_late_min_imbalance=float(bull_late_min_imbalance),
        bear_late_min_ratio=float(bear_late_min_ratio),
        bear_late_max_imbalance=float(bear_late_max_imbalance),
        late_support_ev_scale_bnb=float(late_support_ev_scale_bnb),
        side_selection_mode=str(side_selection_mode),
        allowed_sides=str(allowed_sides),
        market_extreme_min=float(market_extreme_min),
        nowcast_market_gap_min=float(nowcast_market_gap_min),
        flow_window_seconds=int(flow_window_seconds),
        flow_min_imbalance=float(flow_min_imbalance),
        flow_gate_mode=str(flow_gate_mode),
        flow_gate_relax_dislocation_min=float(flow_gate_relax_dislocation_min),
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
        drawdown_stake_guard_enabled=bool(drawdown_stake_guard_enabled),
        drawdown_stake_guard_start_bnb=float(drawdown_stake_guard_start_bnb),
        drawdown_stake_guard_full_bnb=float(drawdown_stake_guard_full_bnb),
        drawdown_stake_guard_min_scale=float(drawdown_stake_guard_min_scale),
        anti_martingale_enabled=bool(anti_martingale_enabled),
        anti_martingale_win_multiplier=float(anti_martingale_win_multiplier),
        anti_martingale_loss_multiplier=float(anti_martingale_loss_multiplier),
        anti_martingale_min_scale=float(anti_martingale_min_scale),
        anti_martingale_max_scale=float(anti_martingale_max_scale),
        circuit_breaker_enabled=bool(circuit_breaker_enabled),
        circuit_breaker_drawdown_trigger_bnb=float(circuit_breaker_drawdown_trigger_bnb),
        circuit_breaker_base_skip_rounds=int(circuit_breaker_base_skip_rounds),
        circuit_breaker_escalation_multiplier=float(circuit_breaker_escalation_multiplier),
        circuit_breaker_escalation_window_rounds=int(circuit_breaker_escalation_window_rounds),
        circuit_breaker_max_level=int(circuit_breaker_max_level),
        circuit_breaker_max_skip_rounds=int(circuit_breaker_max_skip_rounds),
        circuit_breaker_reentry_rounds=int(circuit_breaker_reentry_rounds),
        circuit_breaker_reentry_scale=float(circuit_breaker_reentry_scale),
        perf_adapt_mode=str(perf_adapt_mode),
        perf_gate_window=int(perf_gate_window),
        perf_gate_min_history=int(perf_gate_min_history),
        perf_gate_min_win_rate=float(perf_gate_min_win_rate),
        perf_gate_min_mean_profit_bnb=float(perf_gate_min_mean_profit_bnb),
        robust_ev_veto_enabled=bool(robust_ev_veto_enabled),
        robust_ev_veto_min_history=int(robust_ev_veto_min_history),
        robust_ev_veto_window=int(robust_ev_veto_window),
        robust_ev_veto_low_inflow_mult=float(robust_ev_veto_low_inflow_mult),
        robust_ev_veto_extreme_inflow_mult=float(robust_ev_veto_extreme_inflow_mult),
        robust_ev_veto_adverse_skew=float(robust_ev_veto_adverse_skew),
        robust_ev_veto_min_expected_net_bnb=float(robust_ev_veto_min_expected_net_bnb),
        shock_filter_enabled=bool(shock_filter_enabled),
        shock_filter_window_seconds=int(shock_filter_window_seconds),
        shock_filter_min_window_total_bnb=float(shock_filter_min_window_total_bnb),
        shock_filter_min_abs_imbalance=float(shock_filter_min_abs_imbalance),
        shock_filter_min_surge_ratio=float(shock_filter_min_surge_ratio),
        late_model_conflict_flip_enabled=bool(late_model_conflict_flip_enabled),
        late_model_veto_enabled=bool(late_model_veto_enabled),
        late_model_veto_min_late_ratio=float(late_model_veto_min_late_ratio),
        late_model_veto_min_abs_imbalance=float(late_model_veto_min_abs_imbalance),
        late_model_neutral_filter_enabled=bool(late_model_neutral_filter_enabled),
        late_model_neutral_min_late_ratio=float(late_model_neutral_min_late_ratio),
        late_model_neutral_max_abs_imbalance=float(late_model_neutral_max_abs_imbalance),
    )


def _parse_strategy(strategy: dict[str, Any]) -> StrategyConfig:
    _validate_unknown_keys("strategy", strategy, {"dislocation", "router", "ml_candidate", "flow_candidate"})

    dislocation = strategy.get("dislocation", {})
    if dislocation is None:
        dislocation = {}
    if not isinstance(dislocation, dict):
        raise InvariantError("config_section_not_dict: strategy.dislocation")

    _validate_unknown_keys(
        "strategy_dislocation",
        dislocation,
        {"selector", "candidates", "active_candidate_names"},
    )

    router_obj = strategy.get("router", {})
    if router_obj is None:
        router_obj = {}
    if not isinstance(router_obj, dict):
        raise InvariantError("config_section_not_dict: strategy.router")
    router_cfg = _parse_strategy_router(router_obj)

    ml_candidate_obj = _req(strategy, "ml_candidate")
    if not isinstance(ml_candidate_obj, dict):
        raise InvariantError("config_section_not_dict: strategy.ml_candidate")
    ml_candidate_cfg = _parse_ml_candidate(ml_candidate_obj)

    flow_candidate_obj = strategy.get("flow_candidate", {})
    if flow_candidate_obj is None:
        flow_candidate_obj = {}
    if not isinstance(flow_candidate_obj, dict):
        raise InvariantError("config_section_not_dict: strategy.flow_candidate")
    flow_candidate_cfg = _parse_flow_candidate(flow_candidate_obj)

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
    candidate_cfgs_all = tuple(parsed)

    active_candidate_names = _opt_list_str(dislocation, "active_candidate_names", ())
    if active_candidate_names:
        active_set = set(active_candidate_names)
        if len(active_set) != len(active_candidate_names):
            raise InvariantError("dislocation_active_candidate_names_duplicate")

        missing = sorted(str(x) for x in active_set if str(x) not in seen_names)
        if missing:
            raise InvariantError(f"dislocation_active_candidate_missing: {missing}")

        by_name = {str(c.name): c for c in candidate_cfgs_all}
        candidate_cfgs = tuple(by_name[str(name)] for name in active_candidate_names)
        if not candidate_cfgs:
            raise InvariantError("dislocation_active_candidates_empty")
    else:
        candidate_cfgs = tuple(candidate_cfgs_all)

    return StrategyConfig(
        dislocation=DislocationStrategyConfig(
            selector=selector_cfg,
            candidates=candidate_cfgs,
        ),
        router=router_cfg,
        ml_candidate=ml_candidate_cfg,
        flow_candidate=flow_candidate_cfg,
    )


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

    allowed_path_keys = {
        "closed_rounds_path",
        "klines_path",
        "feature_cache_path",
        "backtest_state_cache_dir",
        "market_data_db_path",
        "projection_cache_db_path",
        "run_registry_db_path",
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
    feature_cache_path = _opt_str(paths, "feature_cache_path", "../PancakeBot_var_exp/feature_cache_v8.sqlite")
    backtest_state_cache_dir = _opt_str(
        paths,
        "backtest_state_cache_dir",
        "../PancakeBot_var_exp/backtest_state_cache",
    )
    market_data_db_path = _opt_str(
        paths,
        "market_data_db_path",
        "../PancakeBot_var_exp/market_data_v1.sqlite",
    )
    projection_cache_db_path = _opt_str(
        paths,
        "projection_cache_db_path",
        "../PancakeBot_var_exp/projection_cache_v1.sqlite",
    )
    run_registry_db_path = _opt_str(
        paths,
        "run_registry_db_path",
        "../PancakeBot_var_exp/run_registry_v1.sqlite",
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
        "random_seed",
        "use_onchain_event_bets",
        "event_lookback_blocks",
        "latency_log_path",
        "dry_initial_bankroll_bnb",
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

    dry_initial_bankroll_bnb = _opt_float_or_none(runtime, "dry_initial_bankroll_bnb")
    if dry_initial_bankroll_bnb is not None and float(dry_initial_bankroll_bnb) <= 0.0:
        raise InvariantError("dry_initial_bankroll_bnb_must_be_positive")

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
        "tail_offset_rounds",
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

    backtest_cfg = BacktestConfig(
        simulation_size=simulation_size_v,
        initial_bankroll_bnb=initial_bankroll_bnb,
        reset_mode=str(reset_mode),
        reset_every_rounds=int(reset_every_rounds),
        tail_offset_rounds=int(tail_offset_rounds),
    )
    backtest_cfg.validate()

    return AppConfig(
        closed_rounds_path=closed_rounds_path,
        klines_path=klines_path,
        feature_cache_path=feature_cache_path,
        backtest_state_cache_dir=backtest_state_cache_dir,
        market_data_db_path=market_data_db_path,
        projection_cache_db_path=projection_cache_db_path,
        run_registry_db_path=run_registry_db_path,
        abi_json_path=abi_json_path,
        cutoff_seconds=int(cutoff_seconds),
        random_seed=int(random_seed),
        use_onchain_event_bets=bool(use_onchain_event_bets),
        event_lookback_blocks=int(event_lookback_blocks),
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
        strategy=strategy_cfg,
        backtest=backtest_cfg,
    )
