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


def _default_candidates() -> tuple[DislocationCandidateConfig, ...]:
    """Return default dislocation candidate profiles for strategy bootstrapping."""

    return (
        DislocationCandidateConfig(
            name="disloc_best_20260227_x80",
            lookback1_seconds=26,
            lookback2_seconds=150,
            lookback3_seconds=300,
            weight1=1.0,
            weight2=0.0,
            weight3=0.0,
            temperature_bps=3.0,
            fixed_bet_bnb=0.3,
            dislocation_threshold_pp=0.15,
            nowcast_confidence_min=0.0,
            cutoff_pool_total_min_bnb=1.2,
            expected_net_min_bnb=0.18,
            side_selection_mode="nowcast_when_market_disagree",
            market_extreme_min=0.0,
            flow_window_seconds=30,
            flow_min_imbalance=0.1,
            flow_gate_mode="against_side",
            adaptive_candidate_modes=tuple(),
            adaptive_window=200,
            adaptive_min_history=100,
            adaptive_score="mean_profit_per_round",
            adaptive_fallback_mode="nowcast_when_market_disagree",
            stake_mode="ev_optimal",
            stake_min_bnb=0.05,
            stake_max_bnb=0.35,
            stake_ev_ref_bnb=0.1,
            stake_max_side_pool_frac=1_000_000.0,
            perf_adapt_mode="skip",
            perf_gate_window=120,
            perf_gate_min_history=60,
            perf_gate_min_win_rate=0.55,
            perf_gate_min_mean_profit_bnb=0.001,
        ),
        DislocationCandidateConfig(
            name="disloc_altA_20260227_x80",
            lookback1_seconds=26,
            lookback2_seconds=150,
            lookback3_seconds=300,
            weight1=1.4,
            weight2=0.0,
            weight3=0.0,
            temperature_bps=2.5,
            fixed_bet_bnb=0.3,
            dislocation_threshold_pp=0.15,
            nowcast_confidence_min=0.0,
            cutoff_pool_total_min_bnb=1.2,
            expected_net_min_bnb=0.18,
            side_selection_mode="nowcast_when_market_disagree",
            market_extreme_min=0.0,
            flow_window_seconds=30,
            flow_min_imbalance=0.1,
            flow_gate_mode="against_side",
            adaptive_candidate_modes=tuple(),
            adaptive_window=200,
            adaptive_min_history=100,
            adaptive_score="mean_profit_per_round",
            adaptive_fallback_mode="nowcast_when_market_disagree",
            stake_mode="fixed",
            stake_min_bnb=0.05,
            stake_max_bnb=0.3,
            stake_ev_ref_bnb=0.1,
            stake_max_side_pool_frac=1_000_000.0,
            perf_adapt_mode="skip",
            perf_gate_window=80,
            perf_gate_min_history=40,
            perf_gate_min_win_rate=0.52,
            perf_gate_min_mean_profit_bnb=0.0,
        ),
        DislocationCandidateConfig(
            name="disloc_altB_20260227_x80",
            lookback1_seconds=26,
            lookback2_seconds=150,
            lookback3_seconds=300,
            weight1=1.4,
            weight2=0.0,
            weight3=0.0,
            temperature_bps=2.5,
            fixed_bet_bnb=0.3,
            dislocation_threshold_pp=0.15,
            nowcast_confidence_min=0.0,
            cutoff_pool_total_min_bnb=1.2,
            expected_net_min_bnb=0.18,
            side_selection_mode="nowcast_when_market_disagree",
            market_extreme_min=0.0,
            flow_window_seconds=30,
            flow_min_imbalance=0.1,
            flow_gate_mode="off",
            adaptive_candidate_modes=tuple(),
            adaptive_window=200,
            adaptive_min_history=100,
            adaptive_score="mean_profit_per_round",
            adaptive_fallback_mode="nowcast_when_market_disagree",
            stake_mode="fixed",
            stake_min_bnb=0.05,
            stake_max_bnb=0.3,
            stake_ev_ref_bnb=0.1,
            stake_max_side_pool_frac=1_000_000.0,
            perf_adapt_mode="skip",
            perf_gate_window=80,
            perf_gate_min_history=40,
            perf_gate_min_win_rate=0.52,
            perf_gate_min_mean_profit_bnb=0.0,
        ),
        DislocationCandidateConfig(
            name="disloc_cons_20260227_x80",
            lookback1_seconds=26,
            lookback2_seconds=150,
            lookback3_seconds=300,
            weight1=1.0,
            weight2=0.0,
            weight3=0.0,
            temperature_bps=3.0,
            fixed_bet_bnb=0.3,
            dislocation_threshold_pp=0.1,
            nowcast_confidence_min=0.0,
            cutoff_pool_total_min_bnb=2.0,
            expected_net_min_bnb=0.18,
            side_selection_mode="nowcast_when_market_disagree",
            market_extreme_min=0.0,
            flow_window_seconds=30,
            flow_min_imbalance=0.1,
            flow_gate_mode="against_side",
            adaptive_candidate_modes=tuple(),
            adaptive_window=200,
            adaptive_min_history=100,
            adaptive_score="mean_profit_per_round",
            adaptive_fallback_mode="nowcast_when_market_disagree",
            stake_mode="ev_optimal",
            stake_min_bnb=0.05,
            stake_max_bnb=0.35,
            stake_ev_ref_bnb=0.1,
            stake_max_side_pool_frac=1_000_000.0,
            perf_adapt_mode="skip",
            perf_gate_window=120,
            perf_gate_min_history=60,
            perf_gate_min_win_rate=0.55,
            perf_gate_min_mean_profit_bnb=0.001,
        ),
        DislocationCandidateConfig(
            name="disloc_stageG2_r37_x80",
            lookback1_seconds=30,
            lookback2_seconds=150,
            lookback3_seconds=300,
            weight1=1.4,
            weight2=0.0,
            weight3=0.0,
            temperature_bps=6.0,
            fixed_bet_bnb=0.4,
            dislocation_threshold_pp=0.5,
            nowcast_confidence_min=0.0,
            cutoff_pool_total_min_bnb=2.0,
            expected_net_min_bnb=0.146,
            side_selection_mode="market_contra",
            market_extreme_min=0.0,
            flow_window_seconds=0,
            flow_min_imbalance=0.0,
            flow_gate_mode="off",
            adaptive_candidate_modes=tuple(),
            adaptive_window=200,
            adaptive_min_history=100,
            adaptive_score="mean_profit_per_round",
            adaptive_fallback_mode="nowcast_when_market_disagree",
            stake_mode="fixed",
            stake_min_bnb=0.05,
            stake_max_bnb=0.4,
            stake_ev_ref_bnb=0.1,
            stake_max_side_pool_frac=1_000_000.0,
            perf_adapt_mode="off",
            perf_gate_window=0,
            perf_gate_min_history=0,
            perf_gate_min_win_rate=0.0,
            perf_gate_min_mean_profit_bnb=0.0,
        ),
        DislocationCandidateConfig(
            name="disloc_stageH_sidenowcast_when_market_disagree_perfflip_w80_h40_wr0p5_mnm0p001_x80",
            lookback1_seconds=26,
            lookback2_seconds=150,
            lookback3_seconds=300,
            weight1=1.4,
            weight2=0.0,
            weight3=0.0,
            temperature_bps=2.5,
            fixed_bet_bnb=0.3,
            dislocation_threshold_pp=0.15,
            nowcast_confidence_min=0.0,
            cutoff_pool_total_min_bnb=1.2,
            expected_net_min_bnb=0.18,
            side_selection_mode="nowcast_when_market_disagree",
            market_extreme_min=0.0,
            flow_window_seconds=30,
            flow_min_imbalance=0.1,
            flow_gate_mode="against_side",
            adaptive_candidate_modes=tuple(),
            adaptive_window=200,
            adaptive_min_history=100,
            adaptive_score="mean_profit_per_round",
            adaptive_fallback_mode="nowcast_when_market_disagree",
            stake_mode="fixed",
            stake_min_bnb=0.05,
            stake_max_bnb=0.3,
            stake_ev_ref_bnb=0.1,
            stake_max_side_pool_frac=1_000_000.0,
            perf_adapt_mode="flip",
            perf_gate_window=80,
            perf_gate_min_history=40,
            perf_gate_min_win_rate=0.5,
            perf_gate_min_mean_profit_bnb=-0.001,
        ),
        DislocationCandidateConfig(
            name="disloc_stageB_side_adaptive_shadow_ev0p146_skip_w80_h40_wr0p52_mn0p0_x80",
            lookback1_seconds=18,
            lookback2_seconds=150,
            lookback3_seconds=300,
            weight1=2.2,
            weight2=0.0,
            weight3=0.0,
            temperature_bps=5.0,
            fixed_bet_bnb=0.2,
            dislocation_threshold_pp=0.35,
            nowcast_confidence_min=0.0,
            cutoff_pool_total_min_bnb=1.2,
            expected_net_min_bnb=0.146,
            side_selection_mode="adaptive_shadow",
            market_extreme_min=0.0,
            flow_window_seconds=0,
            flow_min_imbalance=0.0,
            flow_gate_mode="off",
            adaptive_candidate_modes=("nowcast_when_market_disagree", "ev_max", "nowcast_contra"),
            adaptive_window=200,
            adaptive_min_history=100,
            adaptive_score="mean_profit_per_round",
            adaptive_fallback_mode="nowcast_when_market_disagree",
            stake_mode="fixed",
            stake_min_bnb=0.05,
            stake_max_bnb=0.2,
            stake_ev_ref_bnb=0.1,
            stake_max_side_pool_frac=1_000_000.0,
            perf_adapt_mode="skip",
            perf_gate_window=80,
            perf_gate_min_history=40,
            perf_gate_min_win_rate=0.52,
            perf_gate_min_mean_profit_bnb=0.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class DislocationStrategyConfig:
    """Top-level config for the production dislocation strategy."""

    selector: DislocationSelectorConfig = DislocationSelectorConfig()
    candidates: tuple[DislocationCandidateConfig, ...] = _default_candidates()


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Top-level strategy configuration root."""

    dislocation: DislocationStrategyConfig = DislocationStrategyConfig()
