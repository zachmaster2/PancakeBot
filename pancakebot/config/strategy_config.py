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
    expected_net_min_bnb: float
    side_selection_mode: str
    market_extreme_min: float
    flow_window_seconds: int
    flow_min_imbalance: float
    flow_gate_mode: str
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
    perf_adapt_mode: str
    perf_gate_window: int
    perf_gate_min_history: int
    perf_gate_min_win_rate: float
    perf_gate_min_mean_profit_bnb: float


@dataclass(frozen=True, slots=True)
class DislocationStrategyConfig:
    """Top-level config for the production dislocation strategy."""

    selector: DislocationSelectorConfig
    candidates: tuple[DislocationCandidateConfig, ...]


@dataclass(frozen=True, slots=True)
class MlCandidateConfig:
    """ML walk-forward candidate configuration used by the shared router."""

    enabled: bool = False
    name: str = "ml_walkforward_v1"
    fixed_bet_bnb: float = 0.2
    min_tradeable_prob: float = 0.55
    min_prob_edge: float = 0.02
    cutoff_pool_total_min_bnb: float = 1.2
    expected_net_min_bnb: float = 0.0
    train_size: int = 20_000
    calibrate_size: int = 2_000
    retrain_interval: int = 500
    recalibrate_interval: int = 250
    price_alpha: float = 1.0
    pool_alpha_total: float = 1.0
    pool_alpha_ratio: float = 1.0
    recency_weight_floor: float = 0.7
    recency_weight_power: float = 1.0
    predictability_baseline_bet_bnb: float = 0.05
    random_seed: int = 1337


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Top-level strategy configuration root."""

    dislocation: DislocationStrategyConfig
    router: StrategyRouterConfig = StrategyRouterConfig()
    ml_candidate: MlCandidateConfig = MlCandidateConfig()
