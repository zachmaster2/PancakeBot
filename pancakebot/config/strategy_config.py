from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyRouterConfig:
    """Shared strategy router configuration used by live/dry/backtest."""

    mode: str = "selector_max_score"
    score_threshold_bnb: float = -1e9
    online_warmup_rounds: int = 50_000
    online_num_quantile_bins: int = 12
    online_min_cell_obs: int = 5
    online_score_threshold_bnb: float = 0.0
    online_use_direction_split: bool = True


@dataclass(frozen=True, slots=True)
class DislocationSelectorConfig:
    """Selector controls shared across all dislocation candidate profiles."""

    warmup_rounds: int = 20_000
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
    bear_expected_net_extra_min_bnb: float
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
    late_model_veto_enabled: bool
    late_model_veto_min_late_ratio: float
    late_model_veto_min_abs_imbalance: float


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


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Top-level strategy configuration root."""

    dislocation: DislocationStrategyConfig
    ml_candidate: MlCandidateConfig
    router: StrategyRouterConfig = StrategyRouterConfig()
