from __future__ import annotations

from dataclasses import dataclass


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
class StrategyConfig:
    """Top-level strategy configuration root."""

    dislocation: DislocationStrategyConfig
