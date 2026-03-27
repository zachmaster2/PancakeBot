from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyRouterConfig:
    """Shared strategy router configuration used by live/dry/backtest."""

    mode: str = "selector_max_score"
    score_threshold_bnb: float = -1e9
    online_warmup_rounds: int = 10_000
    online_num_quantile_bins: int = 12
    online_min_cell_obs: int = 5
    online_score_threshold_bnb: float = 0.008
    online_use_direction_split: bool = False


@dataclass(frozen=True, slots=True)
class DislocationSelectorConfig:
    """Selector controls shared across all dislocation candidate profiles."""

    warmup_rounds: int = 10_000
    num_quantile_bins: int = 12
    min_cell_obs: int = 5
    score_threshold: float = -0.01
    use_direction_split: bool = True
    shadow_initial_bankroll_bnb: float = 0.5


@dataclass(frozen=True, slots=True)
class DislocationCandidateConfig:
    """One dislocation candidate profile used by the selector ensemble."""

    name: str
    lookback1_seconds: int
    lookback2_seconds: int
    lookback3_seconds: int
    weight1: float
    weight2: float
    weight3: float
    temperature_bps: float
    fixed_bet_bnb: float
    dislocation_threshold_pp: float
    nowcast_confidence_min: float
    cutoff_pool_total_min_bnb: float
    pool_total_gate_mode: str
    projected_final_pool_multiplier: float
    projected_final_pool_total_min_bnb: float
    expected_net_min_bnb: float
    bull_expected_net_extra_min_bnb: float
    bear_expected_net_extra_min_bnb: float
    bull_late_min_ratio: float
    bull_late_min_imbalance: float
    bear_late_min_ratio: float
    bear_late_max_imbalance: float
    late_support_ev_scale_bnb: float
    side_selection_mode: str
    allowed_sides: str
    market_extreme_min: float
    nowcast_market_gap_min: float
    flow_window_seconds: int
    flow_min_imbalance: float
    flow_gate_mode: str
    flow_gate_relax_dislocation_min: float
    adaptive_candidate_modes: tuple[str, ...]
    adaptive_window: int
    adaptive_min_history: int
    adaptive_score: str
    adaptive_fallback_mode: str
    stake_mode: str
    stake_min_bnb: float
    stake_max_bnb: float
    stake_ev_ref_bnb: float
    stake_max_side_pool_frac: float
    drawdown_stake_guard_enabled: bool
    drawdown_stake_guard_start_bnb: float
    drawdown_stake_guard_full_bnb: float
    drawdown_stake_guard_min_scale: float
    anti_martingale_enabled: bool
    anti_martingale_win_multiplier: float
    anti_martingale_loss_multiplier: float
    anti_martingale_min_scale: float
    anti_martingale_max_scale: float
    circuit_breaker_enabled: bool
    circuit_breaker_drawdown_trigger_bnb: float
    circuit_breaker_base_skip_rounds: int
    circuit_breaker_escalation_multiplier: float
    circuit_breaker_escalation_window_rounds: int
    circuit_breaker_max_level: int
    circuit_breaker_max_skip_rounds: int
    circuit_breaker_reentry_rounds: int
    circuit_breaker_reentry_scale: float
    perf_adapt_mode: str
    perf_gate_window: int
    perf_gate_min_history: int
    perf_gate_min_win_rate: float
    perf_gate_min_mean_profit_bnb: float
    robust_ev_veto_enabled: bool
    robust_ev_veto_min_history: int
    robust_ev_veto_window: int
    robust_ev_veto_low_inflow_mult: float
    robust_ev_veto_extreme_inflow_mult: float
    robust_ev_veto_adverse_skew: float
    robust_ev_veto_min_expected_net_bnb: float
    shock_filter_enabled: bool
    shock_filter_window_seconds: int
    shock_filter_min_window_total_bnb: float
    shock_filter_min_abs_imbalance: float
    shock_filter_min_surge_ratio: float
    late_model_conflict_flip_enabled: bool
    late_model_veto_enabled: bool
    late_model_veto_min_late_ratio: float
    late_model_veto_min_abs_imbalance: float
    late_model_neutral_filter_enabled: bool
    late_model_neutral_min_late_ratio: float
    late_model_neutral_max_abs_imbalance: float


@dataclass(frozen=True, slots=True)
class DislocationStrategyConfig:
    """Top-level config for the production dislocation strategy."""

    selector: DislocationSelectorConfig
    candidates: tuple[DislocationCandidateConfig, ...]


@dataclass(frozen=True, slots=True)
class MlCandidateConfig:
    """ML walk-forward candidate configuration used by the shared router."""

    enabled: bool
    name: str
    fixed_bet_bnb: float
    min_tradeable_prob: float
    min_prob_edge: float
    cutoff_pool_total_min_bnb: float
    expected_net_min_bnb: float
    train_size: int
    calibrate_size: int
    retrain_interval: int
    recalibrate_interval: int
    price_alpha: float
    pool_alpha_total: float
    pool_alpha_ratio: float
    recency_weight_floor: float
    recency_weight_power: float
    predictability_baseline_bet_bnb: float
    random_seed: int
    expected_net_max_bnb: float | None = None
    predictability_feature_mode: str = "all_features"
    predictability_label_mode: str = "baseline_log_imbalance_side"
    emit_candidate: bool = True
    veto_opposite_side_candidates: bool = False
    veto_untradeable_candidates: bool = False
    veto_candidate_expected_net_below_min: bool = False
    rescore_baseline_candidates_with_expected_net: bool = False
    candidate_profit_model_enabled: bool = False
    candidate_profit_model_warmup_rounds: int = 5000
    candidate_profit_model_num_quantile_bins: int = 8
    candidate_profit_model_min_cell_obs: int = 5


@dataclass(frozen=True, slots=True)
class FlowCandidateConfig:
    """Simple flow/LGBM candidate configuration used by the shared router."""

    enabled: bool = False
    name: str = "flow_lgbm_recent_t12k_r1k_regime40_v1"
    shadow_initial_bankroll_bnb: float = 50.0
    train_size: int = 12_000
    retrain_interval: int = 1_000
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_seed: int = 42
    ev_threshold: float = 0.0025
    kelly_fraction: float = 0.10
    max_fraction: float = 0.25
    max_bet_abs: float = 0.50
    min_bet_size: float = 0.05
    round_to: float = 0.01
    min_total_pool_c: float = 1.0
    max_total_pool_share: float = 0.05
    max_side_pool_share: float = 0.50
    min_bull_ratio: float = 0.05
    max_bull_ratio: float = 0.95
    vol_mid: float = 0.030
    drawdown_stop_pct: float = 0.75
    drawdown_throttle_start_pct: float = 0.35
    drawdown_throttle_min_scale: float = 0.35
    roll_window: int = 40
    roll_edge_min: float = -0.002
    roll_winrate_min: float = 0.48
    cooldown_trades: int = 40


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Top-level strategy configuration root."""

    dislocation: DislocationStrategyConfig
    ml_candidate: MlCandidateConfig
    flow_candidate: FlowCandidateConfig = FlowCandidateConfig()
    router: StrategyRouterConfig = StrategyRouterConfig()
