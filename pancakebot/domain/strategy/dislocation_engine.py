from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Protocol

from pancakebot.config.strategy_config import DislocationCandidateConfig, DislocationSelectorConfig
from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei_at_or_before
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.types import Kline, Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round

_STATIC_SIDE_SELECTION_MODES = (
    "ev_max",
    "nowcast",
    "nowcast_contra",
    "dislocation",
    "dislocation_contra",
    "market_follow",
    "market_contra",
    "nowcast_when_market_disagree",
    "nowcast_when_market_agree",
)
_PROJECTED_EV_STAKE_MODES = (
    "ev_scaled_projected",
    "ev_optimal_projected",
)


@dataclass(frozen=True, slots=True)
class LiveStrategyDecision:
    action: str
    bet_side: str | None
    amount_bnb: float
    expected_profit_bnb: float
    skip_reason: str | None
    p_bull: float | None = None
    selected_strategy: str | None = None


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    warmup_rounds: int
    num_quantile_bins: int
    min_cell_obs: int
    score_threshold: float
    use_direction_split: bool


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    name: str
    cutoff_seconds: int
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
class _KlineIndex:
    close_times_ms: tuple[int, ...]
    close_prices: tuple[float, ...]

    def spot_at_or_before(self, ts_ms: int) -> float | None:
        idx = bisect.bisect_right(self.close_times_ms, int(ts_ms)) - 1
        if int(idx) < 0:
            return None
        return float(self.close_prices[int(idx)])


class _ProjectedPoolProvider(Protocol):
    """Provider contract for model-based final-pool projections."""

    def predict_final_pools_for_round(self, *, round_t: Round) -> tuple[float, float, float] | None:
        """Return (final_total_bnb, final_bull_bnb, final_bear_bnb) or None if unavailable."""


@dataclass(frozen=True, slots=True)
class _CoreDecision:
    side: str | None
    reason: str
    p_nowcast_bull: float | None
    p_market_bull: float | None
    dislocation_bull: float | None
    expected_net_bull: float | None
    expected_net_bear: float | None
    expected_net_selected: float | None
    ev_pool_bull_bnb: float | None = None
    ev_pool_bear_bnb: float | None = None


@dataclass(frozen=True, slots=True)
class _SelectorBetRow:
    ev_selected: float
    abs_dislocation: float
    side_idx: int
    profit_bnb: float


@dataclass(slots=True)
class _CandidateState:
    cfg: CandidateConfig
    shadow_bankroll_bnb: float
    shadow_peak_bankroll_bnb: float
    anti_martingale_scale: float
    circuit_breaker_skip_rounds_remaining: int
    circuit_breaker_level: int
    circuit_breaker_last_trigger_settled_round: int | None
    circuit_breaker_reentry_rounds_remaining: int
    adaptive_shadow_round_profit: dict[str, deque[float]]
    adaptive_shadow_bet_profit: dict[str, deque[float]]
    adaptive_shadow_bet_wins: dict[str, deque[int]]
    perf_shadow_profits: deque[float]
    perf_shadow_wins: deque[int]
    robust_late_inflow_ratio: deque[float]
    robust_late_bull_share: deque[float]


@dataclass(slots=True)
class _CandidateRoundDecision:
    action: str
    side: str | None
    bet_bnb: float
    skip_reason: str
    p_nowcast_bull: float | None
    dislocation_bull: float | None
    expected_net_selected: float | None
    adaptive_mode_decisions: dict[str, _CoreDecision] | None
    base_side: str | None
    perf_shadow_track: bool
    projected_late_ratio: float | None = None
    projected_late_imbalance: float | None = None


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(float(hi), max(float(lo), float(x))))


def _drawdown_stake_scale(
    *,
    cfg: CandidateConfig,
    shadow_bankroll_bnb: float,
    shadow_peak_bankroll_bnb: float,
) -> float:
    if not bool(cfg.drawdown_stake_guard_enabled):
        return 1.0

    min_scale = _clamp(float(cfg.drawdown_stake_guard_min_scale), 0.0, 1.0)
    if float(min_scale) >= 1.0:
        return 1.0

    start_bnb = max(0.0, float(cfg.drawdown_stake_guard_start_bnb))
    full_bnb = max(0.0, float(cfg.drawdown_stake_guard_full_bnb))
    if float(full_bnb) <= 0.0:
        return 1.0

    drawdown_bnb = max(0.0, float(shadow_peak_bankroll_bnb) - float(shadow_bankroll_bnb))
    if float(drawdown_bnb) <= float(start_bnb):
        return 1.0

    if float(full_bnb) <= float(start_bnb):
        return float(min_scale)

    if float(drawdown_bnb) >= float(full_bnb):
        return float(min_scale)

    frac = float(drawdown_bnb - float(start_bnb)) / float(full_bnb - float(start_bnb))
    return float(max(float(min_scale), 1.0 - float(frac) * (1.0 - float(min_scale))))


def _anti_martingale_next_scale(
    *,
    cfg: CandidateConfig,
    prev_scale: float,
    realized_profit_bnb: float,
) -> float:
    if not bool(cfg.anti_martingale_enabled):
        return 1.0
    scale = float(prev_scale)
    if float(realized_profit_bnb) > 0.0:
        scale *= float(cfg.anti_martingale_win_multiplier)
    elif float(realized_profit_bnb) < 0.0:
        scale *= float(cfg.anti_martingale_loss_multiplier)
    return float(
        _clamp(
            float(scale),
            float(cfg.anti_martingale_min_scale),
            float(cfg.anti_martingale_max_scale),
        )
    )


def _circuit_breaker_skip_rounds_for_level(*, cfg: CandidateConfig, level: int) -> int:
    base = max(0, int(cfg.circuit_breaker_base_skip_rounds))
    if int(base) <= 0:
        return 0
    lvl = max(1, int(level))
    mult = max(1.0, float(cfg.circuit_breaker_escalation_multiplier))
    rounds = int(math.ceil(float(base) * (float(mult) ** float(max(0, int(lvl - 1))))))
    max_skip = max(0, int(cfg.circuit_breaker_max_skip_rounds))
    if int(max_skip) > 0:
        rounds = min(int(rounds), int(max_skip))
    return max(0, int(rounds))


def _median_deque(values: deque[float]) -> float | None:
    if not values:
        return None
    vv = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vv:
        return None
    n = int(len(vv))
    mid = int(n // 2)
    if int(n % 2) == 1:
        return float(vv[mid])
    return float((float(vv[mid - 1]) + float(vv[mid])) * 0.5)


def _sigmoid(x: float) -> float:
    xx = float(x)
    if xx >= 0.0:
        z = math.exp(-xx)
        return float(1.0 / (1.0 + z))
    z = math.exp(xx)
    return float(z / (1.0 + z))


def _opposite_side(side: str) -> str:
    s = str(side).upper()
    if s == "BULL":
        return "BEAR"
    if s == "BEAR":
        return "BULL"
    raise InvariantError("dislocation_side_invalid")


def _side_allowed(*, side: str, allowed_sides: str) -> bool:
    mode = str(allowed_sides)
    side_u = str(side).upper()
    if mode == "both":
        return side_u in ("BULL", "BEAR")
    if mode == "bull_only":
        return side_u == "BULL"
    if mode == "bear_only":
        return side_u == "BEAR"
    raise InvariantError("dislocation_allowed_sides_unknown")


def _expected_net_min_for_side(*, cfg: CandidateConfig, side: str) -> float:
    side_u = str(side).upper()
    base = float(cfg.expected_net_min_bnb)
    if side_u == "BULL":
        base += max(0.0, float(cfg.bull_expected_net_extra_min_bnb))
    if side_u == "BEAR":
        base += max(0.0, float(cfg.bear_expected_net_extra_min_bnb))
    return float(base)


def _late_projection_metrics(
    *,
    bull_pool_cutoff_bnb: float,
    bear_pool_cutoff_bnb: float,
    projected_final_pool_bull_bnb: float | None,
    projected_final_pool_bear_bnb: float | None,
) -> tuple[float, float] | None:
    if projected_final_pool_bull_bnb is None or projected_final_pool_bear_bnb is None:
        return None

    cut_bull = max(0.0, float(bull_pool_cutoff_bnb))
    cut_bear = max(0.0, float(bear_pool_cutoff_bnb))
    cut_total = float(cut_bull) + float(cut_bear)
    if float(cut_total) <= 0.0:
        return None

    final_bull = float(projected_final_pool_bull_bnb)
    final_bear = float(projected_final_pool_bear_bnb)
    if (
        not math.isfinite(float(final_bull))
        or not math.isfinite(float(final_bear))
        or float(final_bull) <= 0.0
        or float(final_bear) <= 0.0
    ):
        return None

    late_bull = max(0.0, float(final_bull) - float(cut_bull))
    late_bear = max(0.0, float(final_bear) - float(cut_bear))
    late_total = float(late_bull) + float(late_bear)
    if float(late_total) <= 0.0:
        return None

    late_ratio = float(late_total) / float(cut_total)
    late_imb = (float(late_bull) - float(late_bear)) / float(late_total)
    return float(late_ratio), float(late_imb)


def _flow_gate_relaxed_for_dislocation(*, cfg: CandidateConfig, dislocation_bull: float) -> bool:
    return abs(float(dislocation_bull)) >= float(cfg.flow_gate_relax_dislocation_min)


def _quantile_edges(values: list[float], n_bins: int) -> list[float]:
    if int(n_bins) <= 1:
        raise InvariantError("selector_num_bins_invalid")
    if not values:
        return [0.0 for _ in range(int(n_bins) + 1)]
    vv = sorted(float(x) for x in values)
    out: list[float] = []
    for i in range(int(n_bins) + 1):
        q = float(i) / float(n_bins)
        idx = int(round((len(vv) - 1) * q))
        idx = max(0, min(len(vv) - 1, idx))
        out.append(float(vv[idx]))
    return out


def _bin_index(x: float, edges: list[float]) -> int:
    n_bins = int(len(edges) - 1)
    if int(n_bins) <= 1:
        return 0
    for i in range(int(n_bins)):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if float(x) >= float(lo) and (float(x) < float(hi) or int(i) == int(n_bins - 1)):
            return int(i)
    return int(n_bins - 1)


def _nowcast_probability_bull(
    *,
    round_t: Round,
    kidx: _KlineIndex,
    cutoff_seconds: int,
    lookback1_seconds: int,
    lookback2_seconds: int,
    lookback3_seconds: int,
    weight1: float,
    weight2: float,
    weight3: float,
    temperature_bps: float,
) -> float | None:
    if round_t.lock_at is None:
        return None
    if float(temperature_bps) <= 0.0:
        raise InvariantError("dislocation_temperature_bps_must_be_positive")

    cutoff_ts_ms = int(int(round_t.lock_at) - int(cutoff_seconds)) * 1000
    p0 = kidx.spot_at_or_before(int(cutoff_ts_ms))
    if p0 is None or float(p0) <= 0.0:
        return None

    lbs = [int(lookback1_seconds), int(lookback2_seconds), int(lookback3_seconds)]
    ws = [float(weight1), float(weight2), float(weight3)]
    score_bps = 0.0
    used = 0
    for lb, w in zip(lbs, ws):
        if int(lb) <= 0 or float(w) == 0.0:
            continue
        past = kidx.spot_at_or_before(int(cutoff_ts_ms) - int(lb) * 1000)
        if past is None or float(past) <= 0.0:
            return None
        score_bps += float(math.log(float(p0) / float(past)) * 10000.0) * float(w)
        used += 1
    if int(used) <= 0:
        raise InvariantError("dislocation_no_lookbacks_enabled")
    return float(_sigmoid(float(score_bps) / float(temperature_bps)))


def _expected_net_from_cutoff(
    *,
    p_nowcast_bull: float,
    bull_pool_cutoff_bnb: float,
    bear_pool_cutoff_bnb: float,
    side: str,
    fixed_bet_bnb: float,
    treasury_fee_fraction: float,
) -> float:
    bull_cut = float(bull_pool_cutoff_bnb)
    bear_cut = float(bear_pool_cutoff_bnb)
    stake = float(fixed_bet_bnb)
    if float(stake) <= 0.0:
        raise InvariantError("dislocation_stake_nonpositive")
    if not (0.0 <= float(treasury_fee_fraction) < 1.0):
        raise InvariantError("dislocation_treasury_fee_out_of_range")

    side_u = str(side).upper()
    if side_u == "BULL":
        side_pool_after = float(bull_cut) + float(stake)
        total_after = float(bull_cut) + float(stake) + float(bear_cut)
        p_win = float(p_nowcast_bull)
    elif side_u == "BEAR":
        side_pool_after = float(bear_cut) + float(stake)
        total_after = float(bear_cut) + float(stake) + float(bull_cut)
        p_win = 1.0 - float(p_nowcast_bull)
    else:
        raise InvariantError("dislocation_side_invalid")

    if float(side_pool_after) <= 0.0 or float(total_after) <= 0.0:
        return float("-inf")

    payout_multiple = (float(total_after) * (1.0 - float(treasury_fee_fraction))) / float(side_pool_after)
    win_credit = float(stake) * float(payout_multiple) - float(GAS_COST_CLAIM_BNB)
    expected_credit = float(p_win) * float(win_credit)
    expected_net = float(expected_credit) - (float(stake) + float(GAS_COST_BET_BNB))
    return float(expected_net)


def _precutoff_flow_imbalance(
    *,
    round_t: Round,
    cutoff_seconds: int,
    flow_window_seconds: int,
) -> float | None:
    if round_t.lock_at is None or int(round_t.lock_at) <= 0:
        return None
    if int(flow_window_seconds) <= 0:
        return None
    cutoff_ts = int(round_t.lock_at) - int(cutoff_seconds)
    start_ts = int(cutoff_ts) - int(flow_window_seconds)
    bull_wei = 0
    bear_wei = 0
    for b in round_t.bets:
        created = int(b.created_at)
        if int(created) < int(start_ts) or int(created) > int(cutoff_ts):
            continue
        pos = str(b.position)
        if pos == "Bull":
            bull_wei += int(b.amount_wei)
        elif pos == "Bear":
            bear_wei += int(b.amount_wei)
    total = int(bull_wei) + int(bear_wei)
    if int(total) <= 0:
        return None
    return float((int(bull_wei) - int(bear_wei)) / float(total))


def _precutoff_shock_filter_triggers(
    *,
    round_t: Round,
    cfg: CandidateConfig,
) -> bool:
    if not bool(cfg.shock_filter_enabled):
        return False
    if round_t.lock_at is None or int(round_t.lock_at) <= 0:
        return False
    if int(round_t.start_at) <= 0:
        return False

    cutoff_ts = int(round_t.lock_at) - int(cfg.cutoff_seconds)
    win_secs = max(1, int(cfg.shock_filter_window_seconds))
    win_start = int(cutoff_ts) - int(win_secs)

    bull_win = 0
    bear_win = 0
    bull_prev = 0
    bear_prev = 0
    for b in round_t.bets:
        created = int(b.created_at)
        if int(created) > int(cutoff_ts) or int(created) < int(round_t.start_at):
            continue
        pos = str(b.position)
        if int(created) >= int(win_start):
            if pos == "Bull":
                bull_win += int(b.amount_wei)
            elif pos == "Bear":
                bear_win += int(b.amount_wei)
            continue
        if pos == "Bull":
            bull_prev += int(b.amount_wei)
        elif pos == "Bear":
            bear_prev += int(b.amount_wei)

    win_total = int(bull_win) + int(bear_win)
    if int(win_total) <= 0:
        return False
    win_total_bnb = float(win_total) / float(BNB_WEI)
    if float(win_total_bnb) < float(cfg.shock_filter_min_window_total_bnb):
        return False

    win_imb = float(int(bull_win) - int(bear_win)) / float(win_total)
    if abs(float(win_imb)) < float(cfg.shock_filter_min_abs_imbalance):
        return False

    prev_total = int(bull_prev) + int(bear_prev)
    prev_dur = max(1, int(win_start) - int(round_t.start_at))
    win_rate = float(win_total_bnb) / float(max(1, int(win_secs)))
    if int(prev_total) <= 0:
        return float(win_rate) > 0.0
    prev_rate = (float(prev_total) / float(BNB_WEI)) / float(prev_dur)
    if float(prev_rate) <= 0.0:
        return float(win_rate) > 0.0
    surge_ratio = float(win_rate) / float(prev_rate)
    return float(surge_ratio) >= float(cfg.shock_filter_min_surge_ratio)


def _late_model_veto_triggers(
    *,
    cfg: CandidateConfig,
    side: str,
    bull_pool_cutoff_bnb: float,
    bear_pool_cutoff_bnb: float,
    projected_final_pool_bull_bnb: float | None,
    projected_final_pool_bear_bnb: float | None,
) -> bool:
    if not bool(cfg.late_model_veto_enabled):
        return False
    late_metrics = _late_projection_metrics(
        bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
        bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    )
    if late_metrics is None:
        return False

    side_u = str(side).upper()
    if side_u not in ("BULL", "BEAR"):
        return False

    late_ratio, late_imb = late_metrics
    if float(late_ratio) < float(cfg.late_model_veto_min_late_ratio):
        return False

    if abs(float(late_imb)) < float(cfg.late_model_veto_min_abs_imbalance):
        return False

    late_sign = 1 if float(late_imb) > 0.0 else -1
    side_sign = 1 if str(side_u) == "BULL" else -1
    return int(late_sign) != int(side_sign)


def _late_model_conflict_flip_side(
    *,
    cfg: CandidateConfig,
    side: str,
    bull_pool_cutoff_bnb: float,
    bear_pool_cutoff_bnb: float,
    projected_final_pool_bull_bnb: float | None,
    projected_final_pool_bear_bnb: float | None,
) -> str:
    if not bool(cfg.late_model_conflict_flip_enabled):
        return str(side)
    side_u = str(side).upper()
    if side_u not in ("BULL", "BEAR"):
        return str(side)
    if not _late_model_veto_triggers(
        cfg=cfg,
        side=str(side),
        bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
        bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    ):
        return str(side)
    return "BEAR" if side_u == "BULL" else "BULL"


def _late_model_neutral_filter_triggers(
    *,
    cfg: CandidateConfig,
    bull_pool_cutoff_bnb: float,
    bear_pool_cutoff_bnb: float,
    projected_final_pool_bull_bnb: float | None,
    projected_final_pool_bear_bnb: float | None,
) -> bool:
    if not bool(cfg.late_model_neutral_filter_enabled):
        return False
    late_metrics = _late_projection_metrics(
        bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
        bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    )
    if late_metrics is None:
        return False

    late_ratio, late_imb = late_metrics
    if float(late_ratio) < float(cfg.late_model_neutral_min_late_ratio):
        return False

    return abs(float(late_imb)) <= float(cfg.late_model_neutral_max_abs_imbalance)


def _late_side_support_skip_reason(
    *,
    cfg: CandidateConfig,
    side: str,
    bull_pool_cutoff_bnb: float,
    bear_pool_cutoff_bnb: float,
    projected_final_pool_bull_bnb: float | None,
    projected_final_pool_bear_bnb: float | None,
) -> str | None:
    side_u = str(side).upper()
    if side_u not in ("BULL", "BEAR"):
        return None
    late_metrics = _late_projection_metrics(
        bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
        bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    )
    if late_metrics is None:
        return None
    late_ratio, late_imb = late_metrics

    if side_u == "BULL":
        if float(late_ratio) < float(cfg.bull_late_min_ratio):
            return "projected_late_ratio_below_bull_min"
        if float(late_imb) < float(cfg.bull_late_min_imbalance):
            return "projected_late_bull_imbalance_below_min"
        return None

    if float(late_ratio) < float(cfg.bear_late_min_ratio):
        return "projected_late_ratio_below_bear_min"
    if float(late_imb) > float(cfg.bear_late_max_imbalance):
        return "projected_late_bear_imbalance_above_max"
    return None


def _late_support_ev_adjustment(
    *,
    cfg: CandidateConfig,
    side: str,
    bull_pool_cutoff_bnb: float,
    bear_pool_cutoff_bnb: float,
    projected_final_pool_bull_bnb: float | None,
    projected_final_pool_bear_bnb: float | None,
) -> float:
    scale = float(cfg.late_support_ev_scale_bnb)
    if float(scale) <= 0.0:
        return 0.0
    side_u = str(side).upper()
    if side_u not in ("BULL", "BEAR"):
        return 0.0
    late_metrics = _late_projection_metrics(
        bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
        bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    )
    if late_metrics is None:
        return 0.0
    late_ratio, late_imb = late_metrics
    aligned_imb = float(late_imb) if side_u == "BULL" else -float(late_imb)
    support_weight = max(0.0, min(1.0, float(late_ratio)))
    return float(scale) * float(support_weight) * float(aligned_imb)


def _pool_total_gate_skip_reason(
    *,
    cfg: CandidateConfig,
    cutoff_pool_total_bnb: float,
    projected_final_pool_total_bnb: float | None,
) -> str | None:
    gate_mode = str(cfg.pool_total_gate_mode)
    cutoff_total = float(cutoff_pool_total_bnb)

    if gate_mode == "cutoff_only":
        if float(cutoff_total) < float(cfg.cutoff_pool_total_min_bnb):
            return "cutoff_pool_below_min_total"
        return None

    if gate_mode == "projected_final_only":
        projected_total = float(cutoff_total) * float(cfg.projected_final_pool_multiplier)
        if float(projected_total) < float(cfg.projected_final_pool_total_min_bnb):
            return "projected_final_pool_below_min_total"
        return None

    if gate_mode == "projected_final_model_only":
        if projected_final_pool_total_bnb is None:
            return "projected_final_pool_model_unavailable"
        projected_total = float(projected_final_pool_total_bnb)
        if not math.isfinite(float(projected_total)) or float(projected_total) <= 0.0:
            return "projected_final_pool_model_unavailable"
        if float(projected_total) < float(cfg.projected_final_pool_total_min_bnb):
            return "projected_final_pool_below_min_total"
        return None

    raise InvariantError("dislocation_pool_total_gate_mode_unknown")


def _stake_mode_uses_projected_pool_ev(*, stake_mode: str) -> bool:
    return str(stake_mode) in _PROJECTED_EV_STAKE_MODES


def _effective_ev_pools(
    *,
    cfg: CandidateConfig,
    bull_pool_cutoff_bnb: float,
    bear_pool_cutoff_bnb: float,
    projected_final_pool_bull_bnb: float | None,
    projected_final_pool_bear_bnb: float | None,
) -> tuple[float, float]:
    bull_cut = float(bull_pool_cutoff_bnb)
    bear_cut = float(bear_pool_cutoff_bnb)
    if not _stake_mode_uses_projected_pool_ev(stake_mode=str(cfg.stake_mode)):
        return float(bull_cut), float(bear_cut)
    if projected_final_pool_bull_bnb is None or projected_final_pool_bear_bnb is None:
        return float(bull_cut), float(bear_cut)

    bull_final = float(projected_final_pool_bull_bnb)
    bear_final = float(projected_final_pool_bear_bnb)
    if (
        not math.isfinite(float(bull_final))
        or not math.isfinite(float(bear_final))
        or float(bull_final) <= 0.0
        or float(bear_final) <= 0.0
    ):
        return float(bull_cut), float(bear_cut)

    # Reuse projected_final_pool_multiplier as late-inflow confidence scaling.
    # 1.0 = full model inflow, 0.0 = cutoff-only fallback, >1.0 = aggressive.
    inflow_mult = max(0.0, float(cfg.projected_final_pool_multiplier))
    late_bull = max(0.0, float(bull_final) - float(bull_cut))
    late_bear = max(0.0, float(bear_final) - float(bear_cut))
    ev_bull = float(bull_cut) + float(late_bull) * float(inflow_mult)
    ev_bear = float(bear_cut) + float(late_bear) * float(inflow_mult)
    if float(ev_bull) <= 0.0 or float(ev_bear) <= 0.0:
        return float(bull_cut), float(bear_cut)
    return float(ev_bull), float(ev_bear)


def _final_pools_bnb_for_round(*, round_t: Round, lock_at: int) -> tuple[float, float]:
    bull_wei = 0
    bear_wei = 0
    lock_ts = int(lock_at)
    for b in round_t.bets:
        created = int(b.created_at)
        if int(created) > int(lock_ts):
            continue
        pos = str(b.position)
        if pos == "Bull":
            bull_wei += int(b.amount_wei)
        elif pos == "Bear":
            bear_wei += int(b.amount_wei)
    bull = float(bull_wei) / float(BNB_WEI)
    bear = float(bear_wei) / float(BNB_WEI)
    return float(bull), float(bear)


def _robust_selected_ev_min(
    *,
    cfg: CandidateConfig,
    side: str,
    p_nowcast_bull: float,
    bull_pool_cutoff_bnb: float,
    bear_pool_cutoff_bnb: float,
    robust_late_inflow_ratio: float,
    robust_late_bull_share: float,
    treasury_fee_fraction: float,
) -> float:
    side_u = str(side).upper()
    if side_u not in ("BULL", "BEAR"):
        raise InvariantError("dislocation_side_invalid")

    cut_bull = float(bull_pool_cutoff_bnb)
    cut_bear = float(bear_pool_cutoff_bnb)
    cut_total = float(cut_bull) + float(cut_bear)
    if float(cut_bull) <= 0.0 or float(cut_bear) <= 0.0 or float(cut_total) <= 0.0:
        return float("-inf")

    late_ratio = max(0.0, float(robust_late_inflow_ratio))
    late_total_base = float(cut_total) * float(late_ratio)
    late_share_base = _clamp(float(robust_late_bull_share), 0.0, 1.0)

    low_mult = max(0.0, float(cfg.robust_ev_veto_low_inflow_mult))
    extreme_mult = max(0.0, float(cfg.robust_ev_veto_extreme_inflow_mult))
    adverse_skew = _clamp(float(cfg.robust_ev_veto_adverse_skew), 0.0, 0.49)

    scenarios: list[tuple[float, float]] = []
    for mult, is_extreme in ((1.0, False), (float(low_mult), False), (float(extreme_mult), True)):
        late_total = float(late_total_base) * float(mult)
        late_share = float(late_share_base)
        if bool(is_extreme):
            if side_u == "BULL":
                late_share = _clamp(float(late_share) - float(adverse_skew), 0.0, 1.0)
            else:
                late_share = _clamp(float(late_share) + float(adverse_skew), 0.0, 1.0)

        bull_pool = float(cut_bull) + float(late_total) * float(late_share)
        bear_pool = float(cut_bear) + float(late_total) * (1.0 - float(late_share))
        scenarios.append((float(bull_pool), float(bear_pool)))

    evs: list[float] = []
    for bull_pool, bear_pool in scenarios:
        evs.append(
            _expected_net_from_cutoff(
                p_nowcast_bull=float(p_nowcast_bull),
                bull_pool_cutoff_bnb=float(bull_pool),
                bear_pool_cutoff_bnb=float(bear_pool),
                side=str(side_u),
                fixed_bet_bnb=float(cfg.fixed_bet_bnb),
                treasury_fee_fraction=float(treasury_fee_fraction),
            )
        )
    if not evs:
        return float("-inf")
    return float(min(evs))


def _stake_bnb_for_decision(
    *,
    stake_mode: str,
    fixed_bet_bnb: float,
    expected_net_selected: float | None,
    stake_min_bnb: float,
    stake_max_bnb: float,
    stake_ev_ref_bnb: float,
    side: str | None,
    p_nowcast_bull: float | None,
    bull_pool_cutoff_bnb: float | None,
    bear_pool_cutoff_bnb: float | None,
    bull_pool_ev_bnb: float | None,
    bear_pool_ev_bnb: float | None,
    treasury_fee_fraction: float,
    stake_max_side_pool_frac: float,
    stake_scale: float = 1.0,
) -> float:
    mn = float(stake_min_bnb)
    mx = float(stake_max_bnb)
    if float(mx) < float(mn):
        mn, mx = float(mx), float(mn)

    pool_frac = float(stake_max_side_pool_frac)
    if float(pool_frac) > 0.0 and side is not None:
        side_u = str(side).upper()
        side_pool = None
        if side_u == "BULL" and bull_pool_cutoff_bnb is not None:
            side_pool = float(bull_pool_cutoff_bnb)
        elif side_u == "BEAR" and bear_pool_cutoff_bnb is not None:
            side_pool = float(bear_pool_cutoff_bnb)
        if side_pool is not None and math.isfinite(float(side_pool)) and float(side_pool) > 0.0:
            mx = min(float(mx), float(side_pool) * float(pool_frac))

    if float(mx) <= 0.0:
        return 0.0
    mn = min(float(mn), float(mx))

    base_bet_bnb = float(mn)
    mode = str(stake_mode)
    if mode == "fixed":
        base_bet_bnb = float(min(float(mx), max(float(mn), float(fixed_bet_bnb))))

    elif mode in ("ev_scaled", "ev_scaled_projected"):
        ref = max(1e-9, float(stake_ev_ref_bnb))
        ev = float(expected_net_selected) if expected_net_selected is not None else float("nan")
        if not math.isfinite(float(ev)):
            frac = 0.0
        else:
            frac = float(ev) / float(ref)
        frac = max(0.0, min(1.0, float(frac)))
        base_bet_bnb = float(mn + (float(mx) - float(mn)) * float(frac))

    elif mode in ("ev_optimal", "ev_optimal_projected"):
        if side is None or p_nowcast_bull is None or bull_pool_cutoff_bnb is None or bear_pool_cutoff_bnb is None:
            base_bet_bnb = float(mn)
        elif float(mx) <= float(mn):
            base_bet_bnb = float(mn)
        else:
            ev_bull_pool = float(bull_pool_cutoff_bnb)
            ev_bear_pool = float(bear_pool_cutoff_bnb)
            if (
                mode == "ev_optimal_projected"
                and bull_pool_ev_bnb is not None
                and bear_pool_ev_bnb is not None
                and math.isfinite(float(bull_pool_ev_bnb))
                and math.isfinite(float(bear_pool_ev_bnb))
                and float(bull_pool_ev_bnb) > 0.0
                and float(bear_pool_ev_bnb) > 0.0
            ):
                ev_bull_pool = float(bull_pool_ev_bnb)
                ev_bear_pool = float(bear_pool_ev_bnb)

            side_u = str(side).upper()
            if side_u not in ("BULL", "BEAR"):
                base_bet_bnb = float(mn)
            else:
                points = 31
                best_s = float(mn)
                best_ev = float("-inf")
                span = float(mx) - float(mn)
                for i in range(points):
                    s = float(mn + (float(span) * float(i) / float(points - 1)))
                    ev = _expected_net_from_cutoff(
                        p_nowcast_bull=float(p_nowcast_bull),
                        bull_pool_cutoff_bnb=float(ev_bull_pool),
                        bear_pool_cutoff_bnb=float(ev_bear_pool),
                        side=str(side_u),
                        fixed_bet_bnb=float(s),
                        treasury_fee_fraction=float(treasury_fee_fraction),
                    )
                    if float(ev) > float(best_ev):
                        best_ev = float(ev)
                        best_s = float(s)
                base_bet_bnb = float(best_s)

    else:
        raise InvariantError("dislocation_stake_mode_unknown")

    scale = max(0.0, float(stake_scale))
    if float(scale) <= 0.0:
        return 0.0
    return float(max(0.0, min(float(mx), float(base_bet_bnb) * float(scale))))


def _adaptive_mode_score(
    *,
    mode: str,
    adaptive_score: str,
    adaptive_min_history: int,
    shadow_round_profit: dict[str, deque[float]],
    shadow_bet_profit: dict[str, deque[float]],
    shadow_bet_wins: dict[str, deque[int]],
) -> float | None:
    min_hist = max(1, int(adaptive_min_history))
    score_mode = str(adaptive_score)
    m = str(mode)

    if score_mode == "mean_profit_per_round":
        seq = shadow_round_profit[m]
        if len(seq) < int(min_hist):
            return None
        return float(sum(seq)) / float(len(seq))

    if score_mode == "mean_profit_per_bet":
        seq = shadow_bet_profit[m]
        if len(seq) < int(min_hist):
            return None
        return float(sum(seq)) / float(len(seq))

    if score_mode == "win_rate":
        seq = shadow_bet_wins[m]
        if len(seq) < int(min_hist):
            return None
        return float(sum(seq)) / float(len(seq))

    raise InvariantError("dislocation_adaptive_score_unknown")


def _decide_core(
    *,
    round_t: Round,
    kidx: _KlineIndex,
    cfg: CandidateConfig,
    treasury_fee_fraction: float,
    expected_net_gate_bnb: float,
    projected_final_pool_total_bnb: float | None,
    projected_final_pool_bull_bnb: float | None,
    projected_final_pool_bear_bnb: float | None,
    robust_late_inflow_ratio: float | None,
    robust_late_bull_share: float | None,
) -> _CoreDecision:
    if round_t.lock_at is None:
        return _CoreDecision(
            side=None,
            reason="round_lock_at_missing",
            p_nowcast_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
        )

    cutoff_ts = int(round_t.lock_at) - int(cfg.cutoff_seconds)
    pools = compute_pool_amounts_wei_at_or_before(bets=round_t.bets, cutoff_ts=int(cutoff_ts))
    if int(pools.total_wei) <= 0:
        return _CoreDecision(
            side=None,
            reason="cutoff_pool_empty",
            p_nowcast_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
        )

    pool_total_bnb = float(pools.total_wei) / float(BNB_WEI)
    pool_gate_skip_reason = _pool_total_gate_skip_reason(
        cfg=cfg,
        cutoff_pool_total_bnb=float(pool_total_bnb),
        projected_final_pool_total_bnb=projected_final_pool_total_bnb,
    )
    if pool_gate_skip_reason is not None:
        return _CoreDecision(
            side=None,
            reason=str(pool_gate_skip_reason),
            p_nowcast_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
        )

    if _precutoff_shock_filter_triggers(round_t=round_t, cfg=cfg):
        return _CoreDecision(
            side=None,
            reason="precutoff_shock_filter",
            p_nowcast_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
        )

    p_now = _nowcast_probability_bull(
        round_t=round_t,
        kidx=kidx,
        cutoff_seconds=int(cfg.cutoff_seconds),
        lookback1_seconds=int(cfg.lookback1_seconds),
        lookback2_seconds=int(cfg.lookback2_seconds),
        lookback3_seconds=int(cfg.lookback3_seconds),
        weight1=float(cfg.weight1),
        weight2=float(cfg.weight2),
        weight3=float(cfg.weight3),
        temperature_bps=float(cfg.temperature_bps),
    )
    if p_now is None:
        return _CoreDecision(
            side=None,
            reason="nowcast_unavailable",
            p_nowcast_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
        )

    p_market_bull = float(pools.bull_wei) / float(pools.total_wei)
    dislocation_bull = float(p_now) - float(p_market_bull)

    if abs(float(p_now) - 0.5) < float(cfg.nowcast_confidence_min):
        return _CoreDecision(
            side=None,
            reason="nowcast_confidence_below_min",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
        )

    if abs(float(dislocation_bull)) < float(cfg.dislocation_threshold_pp) / 100.0:
        return _CoreDecision(
            side=None,
            reason="dislocation_below_threshold",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
        )

    bull_cut_bnb = float(pools.bull_wei) / float(BNB_WEI)
    bear_cut_bnb = float(pools.bear_wei) / float(BNB_WEI)
    ev_pool_bull_bnb, ev_pool_bear_bnb = _effective_ev_pools(
        cfg=cfg,
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    )
    ev_bull = _expected_net_from_cutoff(
        p_nowcast_bull=float(p_now),
        bull_pool_cutoff_bnb=float(ev_pool_bull_bnb),
        bear_pool_cutoff_bnb=float(ev_pool_bear_bnb),
        side="BULL",
        fixed_bet_bnb=float(cfg.fixed_bet_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    ev_bear = _expected_net_from_cutoff(
        p_nowcast_bull=float(p_now),
        bull_pool_cutoff_bnb=float(ev_pool_bull_bnb),
        bear_pool_cutoff_bnb=float(ev_pool_bear_bnb),
        side="BEAR",
        fixed_bet_bnb=float(cfg.fixed_bet_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    nowcast_market_gap = abs(float(p_now) - float(p_market_bull))
    if float(nowcast_market_gap) < float(cfg.nowcast_market_gap_min):
        return _CoreDecision(
            side=None,
            reason="nowcast_market_gap_below_min",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=float(ev_bull),
            expected_net_bear=float(ev_bear),
            expected_net_selected=None,
            ev_pool_bull_bnb=float(ev_pool_bull_bnb),
            ev_pool_bear_bnb=float(ev_pool_bear_bnb),
        )

    side_mode = str(cfg.side_selection_mode)
    if side_mode == "ev_max":
        side = "BULL" if float(ev_bull) >= float(ev_bear) else "BEAR"
    elif side_mode == "nowcast":
        side = "BULL" if float(p_now) >= 0.5 else "BEAR"
    elif side_mode == "nowcast_contra":
        side = "BEAR" if float(p_now) >= 0.5 else "BULL"
    elif side_mode == "market_follow":
        if abs(float(p_market_bull) - 0.5) < float(cfg.market_extreme_min):
            return _CoreDecision(
                side=None,
                reason="market_not_extreme",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
            )
        side = "BULL" if float(p_market_bull) >= 0.5 else "BEAR"
    elif side_mode == "market_contra":
        if abs(float(p_market_bull) - 0.5) < float(cfg.market_extreme_min):
            return _CoreDecision(
                side=None,
                reason="market_not_extreme",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
            )
        side = "BEAR" if float(p_market_bull) >= 0.5 else "BULL"
    elif side_mode == "nowcast_when_market_disagree":
        if abs(float(p_market_bull) - 0.5) < float(cfg.market_extreme_min):
            return _CoreDecision(
                side=None,
                reason="market_not_extreme",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
            )
        now_sign = 1 if float(p_now) >= 0.5 else -1
        market_sign = 1 if float(p_market_bull) >= 0.5 else -1
        if int(now_sign) == int(market_sign):
            return _CoreDecision(
                side=None,
                reason="nowcast_market_agree",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
            )
        side = "BULL" if int(now_sign) > 0 else "BEAR"
    elif side_mode == "nowcast_when_market_agree":
        if abs(float(p_market_bull) - 0.5) < float(cfg.market_extreme_min):
            return _CoreDecision(
                side=None,
                reason="market_not_extreme",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
            )
        now_sign = 1 if float(p_now) >= 0.5 else -1
        market_sign = 1 if float(p_market_bull) >= 0.5 else -1
        if int(now_sign) != int(market_sign):
            return _CoreDecision(
                side=None,
                reason="nowcast_market_disagree",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
            )
        side = "BULL" if int(now_sign) > 0 else "BEAR"
    elif side_mode == "dislocation":
        side = "BULL" if float(dislocation_bull) >= 0.0 else "BEAR"
    elif side_mode == "dislocation_contra":
        side = "BEAR" if float(dislocation_bull) >= 0.0 else "BULL"
    else:
        raise InvariantError("dislocation_side_selection_mode_unknown")

    flow_mode = str(cfg.flow_gate_mode)
    if flow_mode != "off" and _flow_gate_relaxed_for_dislocation(
        cfg=cfg,
        dislocation_bull=float(dislocation_bull),
    ):
        flow_mode = "off"
    if flow_mode != "off":
        flow_imb = _precutoff_flow_imbalance(
            round_t=round_t,
            cutoff_seconds=int(cfg.cutoff_seconds),
            flow_window_seconds=int(cfg.flow_window_seconds),
        )
        if flow_imb is None:
            return _CoreDecision(
                side=None,
                reason="flow_unavailable",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
            )
        flow_abs = abs(float(flow_imb))
        if float(flow_abs) < float(cfg.flow_min_imbalance):
            return _CoreDecision(
                side=None,
                reason="flow_below_min_imbalance",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
            )
        flow_sign = 1 if float(flow_imb) > 0.0 else (-1 if float(flow_imb) < 0.0 else 0)
        side_sign = 1 if str(side).upper() == "BULL" else -1
        if flow_mode == "with_side":
            if int(flow_sign) != int(side_sign):
                return _CoreDecision(
                    side=None,
                    reason="flow_not_with_side",
                    p_nowcast_bull=float(p_now),
                    p_market_bull=float(p_market_bull),
                    dislocation_bull=float(dislocation_bull),
                    expected_net_bull=float(ev_bull),
                    expected_net_bear=float(ev_bear),
                    expected_net_selected=None,
                )
        elif flow_mode == "against_side":
            if int(flow_sign) == int(side_sign):
                return _CoreDecision(
                    side=None,
                    reason="flow_not_against_side",
                    p_nowcast_bull=float(p_now),
                    p_market_bull=float(p_market_bull),
                    dislocation_bull=float(dislocation_bull),
                    expected_net_bull=float(ev_bull),
                    expected_net_bear=float(ev_bear),
                    expected_net_selected=None,
                )
        else:
            raise InvariantError("dislocation_flow_gate_mode_unknown")

    side = _late_model_conflict_flip_side(
        cfg=cfg,
        side=str(side),
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    )

    selected_ev = float(ev_bull) if str(side) == "BULL" else float(ev_bear)
    selected_ev += _late_support_ev_adjustment(
        cfg=cfg,
        side=str(side),
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    )
    side_ev_gate_bnb = float(expected_net_gate_bnb)
    if math.isfinite(float(side_ev_gate_bnb)):
        side_ev_gate_bnb = max(
            float(side_ev_gate_bnb),
            _expected_net_min_for_side(cfg=cfg, side=str(side)),
        )
    if _late_model_neutral_filter_triggers(
        cfg=cfg,
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    ):
        return _CoreDecision(
            side=None,
            reason="projected_late_flow_neutral",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=float(ev_bull),
            expected_net_bear=float(ev_bear),
            expected_net_selected=float(selected_ev),
            ev_pool_bull_bnb=float(ev_pool_bull_bnb),
            ev_pool_bear_bnb=float(ev_pool_bear_bnb),
        )
    if _late_model_veto_triggers(
        cfg=cfg,
        side=str(side),
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    ):
        return _CoreDecision(
            side=None,
            reason="projected_late_flow_against_side",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=float(ev_bull),
            expected_net_bear=float(ev_bear),
            expected_net_selected=float(selected_ev),
            ev_pool_bull_bnb=float(ev_pool_bull_bnb),
            ev_pool_bear_bnb=float(ev_pool_bear_bnb),
        )
    late_side_skip_reason = _late_side_support_skip_reason(
        cfg=cfg,
        side=str(side),
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
        projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
    )
    if late_side_skip_reason is not None:
        return _CoreDecision(
            side=None,
            reason=str(late_side_skip_reason),
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=float(ev_bull),
            expected_net_bear=float(ev_bear),
            expected_net_selected=float(selected_ev),
            ev_pool_bull_bnb=float(ev_pool_bull_bnb),
            ev_pool_bear_bnb=float(ev_pool_bear_bnb),
        )

    if bool(cfg.robust_ev_veto_enabled):
        if robust_late_inflow_ratio is None or robust_late_bull_share is None:
            return _CoreDecision(
                side=None,
                reason="robust_ev_history_insufficient",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=float(selected_ev),
                ev_pool_bull_bnb=float(ev_pool_bull_bnb),
                ev_pool_bear_bnb=float(ev_pool_bear_bnb),
            )

        robust_min_ev = _robust_selected_ev_min(
            cfg=cfg,
            side=str(side),
            p_nowcast_bull=float(p_now),
            bull_pool_cutoff_bnb=float(bull_cut_bnb),
            bear_pool_cutoff_bnb=float(bear_cut_bnb),
            robust_late_inflow_ratio=float(robust_late_inflow_ratio),
            robust_late_bull_share=float(robust_late_bull_share),
            treasury_fee_fraction=float(treasury_fee_fraction),
        )
        robust_gate = max(
            0.0,
            float(side_ev_gate_bnb),
            float(cfg.robust_ev_veto_min_expected_net_bnb),
        )
        if float(robust_min_ev) < float(robust_gate):
            return _CoreDecision(
                side=None,
                reason="robust_ev_below_min",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=float(robust_min_ev),
                ev_pool_bull_bnb=float(ev_pool_bull_bnb),
                ev_pool_bear_bnb=float(ev_pool_bear_bnb),
            )
        selected_ev = float(min(float(selected_ev), float(robust_min_ev)))

    if float(selected_ev) < float(side_ev_gate_bnb):
        return _CoreDecision(
            side=None,
            reason="expected_net_below_min",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=float(ev_bull),
            expected_net_bear=float(ev_bear),
            expected_net_selected=float(selected_ev),
            ev_pool_bull_bnb=float(ev_pool_bull_bnb),
            ev_pool_bear_bnb=float(ev_pool_bear_bnb),
        )

    return _CoreDecision(
        side=str(side),
        reason="bet",
        p_nowcast_bull=float(p_now),
        p_market_bull=float(p_market_bull),
        dislocation_bull=float(dislocation_bull),
        expected_net_bull=float(ev_bull),
        expected_net_bear=float(ev_bear),
        expected_net_selected=float(selected_ev),
        ev_pool_bull_bnb=float(ev_pool_bull_bnb),
        ev_pool_bear_bnb=float(ev_pool_bear_bnb),
    )


def _shadow_profit_for_decision(
    *,
    round_t: Round,
    dec: _CoreDecision,
    cfg: CandidateConfig,
    treasury_fee_fraction: float,
    bull_pool_cutoff_bnb: float | None,
    bear_pool_cutoff_bnb: float | None,
) -> tuple[float, bool, bool]:
    if dec.side is None:
        return 0.0, False, False
    if dec.p_nowcast_bull is None:
        return 0.0, False, False
    if bull_pool_cutoff_bnb is None or bear_pool_cutoff_bnb is None:
        return 0.0, False, False
    ev_bull_pool = (
        float(dec.ev_pool_bull_bnb)
        if dec.ev_pool_bull_bnb is not None and math.isfinite(float(dec.ev_pool_bull_bnb))
        else float(bull_pool_cutoff_bnb)
    )
    ev_bear_pool = (
        float(dec.ev_pool_bear_bnb)
        if dec.ev_pool_bear_bnb is not None and math.isfinite(float(dec.ev_pool_bear_bnb))
        else float(bear_pool_cutoff_bnb)
    )
    if float(ev_bull_pool) <= 0.0 or float(ev_bear_pool) <= 0.0:
        ev_bull_pool = float(bull_pool_cutoff_bnb)
        ev_bear_pool = float(bear_pool_cutoff_bnb)

    bet_bnb = _stake_bnb_for_decision(
        stake_mode=str(cfg.stake_mode),
        fixed_bet_bnb=float(cfg.fixed_bet_bnb),
        expected_net_selected=dec.expected_net_selected,
        stake_min_bnb=float(cfg.stake_min_bnb),
        stake_max_bnb=float(cfg.stake_max_bnb),
        stake_ev_ref_bnb=float(cfg.stake_ev_ref_bnb),
        side=str(dec.side),
        p_nowcast_bull=dec.p_nowcast_bull,
        bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
        bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
        bull_pool_ev_bnb=float(ev_bull_pool),
        bear_pool_ev_bnb=float(ev_bear_pool),
        treasury_fee_fraction=float(treasury_fee_fraction),
        stake_max_side_pool_frac=float(cfg.stake_max_side_pool_frac),
    )
    if float(bet_bnb) <= 0.0:
        return 0.0, False, False

    selected_ev_actual = _expected_net_from_cutoff(
        p_nowcast_bull=float(dec.p_nowcast_bull),
        bull_pool_cutoff_bnb=float(ev_bull_pool),
        bear_pool_cutoff_bnb=float(ev_bear_pool),
        side=str(dec.side),
        fixed_bet_bnb=float(bet_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    expected_net_min_side = _expected_net_min_for_side(cfg=cfg, side=str(dec.side))
    if float(selected_ev_actual) < float(expected_net_min_side):
        return 0.0, False, False

    total_cost = float(bet_bnb) + float(GAS_COST_BET_BNB)
    outcome = settle_bet_against_closed_round(
        bet_bnb=float(bet_bnb),
        bet_side=str(dec.side),
        round_closed=round_t,
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    profit = -float(total_cost) + float(outcome.credit_bnb)
    is_win = str(outcome.outcome) == "win"
    return float(profit), True, bool(is_win)


class DislocationEngine:
    def __init__(
        self,
        *,
        selector_cfg: SelectorConfig,
        candidate_cfgs: list[CandidateConfig],
        treasury_fee_fraction: float,
        shadow_initial_bankroll_bnb: float,
        projected_pool_provider: _ProjectedPoolProvider | None = None,
    ) -> None:
        if not candidate_cfgs:
            raise InvariantError("dislocation_candidates_empty")
        if not (0.0 <= float(treasury_fee_fraction) < 1.0):
            raise InvariantError("dislocation_treasury_fee_out_of_range")
        if float(shadow_initial_bankroll_bnb) <= 0.0:
            raise InvariantError("dislocation_shadow_initial_bankroll_nonpositive")

        self._selector_cfg = selector_cfg
        self._treasury_fee_fraction = float(treasury_fee_fraction)
        self._projected_pool_provider = projected_pool_provider
        self._candidate_order = [str(c.name) for c in candidate_cfgs]

        self._candidate_states: dict[str, _CandidateState] = {}
        for cfg in candidate_cfgs:
            self._candidate_states[str(cfg.name)] = self._new_candidate_state(
                cfg=cfg,
                shadow_initial_bankroll_bnb=float(shadow_initial_bankroll_bnb),
            )

        self._kidx = _KlineIndex(close_times_ms=tuple(), close_prices=tuple())
        self._pending_decisions_by_epoch: dict[int, dict[str, _CandidateRoundDecision]] = {}
        self._last_settled_epoch: int | None = None

        self._selector_ready = False
        self._settled_round_count = 0
        self._warmup_rows_by_candidate: dict[str, list[_SelectorBetRow]] = {s: [] for s in self._candidate_order}
        self._selector_edges_by_candidate: dict[str, tuple[list[float], list[float]]] | None = None
        self._selector_sum_profit: dict[str, dict[tuple[int, int, int], float]] = {s: {} for s in self._candidate_order}
        self._selector_cnt_profit: dict[str, dict[tuple[int, int, int], int]] = {s: {} for s in self._candidate_order}

    @staticmethod
    def _new_candidate_state(*, cfg: CandidateConfig, shadow_initial_bankroll_bnb: float) -> _CandidateState:
        adaptive_shadow_round_profit: dict[str, deque[float]] = {}
        adaptive_shadow_bet_profit: dict[str, deque[float]] = {}
        adaptive_shadow_bet_wins: dict[str, deque[int]] = {}

        if str(cfg.side_selection_mode) == "adaptive_shadow":
            if int(cfg.adaptive_window) <= 0:
                raise InvariantError("adaptive_window_must_be_positive")
            for mode in cfg.adaptive_candidate_modes:
                m = str(mode)
                adaptive_shadow_round_profit[m] = deque(maxlen=int(cfg.adaptive_window))
                adaptive_shadow_bet_profit[m] = deque(maxlen=int(cfg.adaptive_window))
                adaptive_shadow_bet_wins[m] = deque(maxlen=int(cfg.adaptive_window))

        perf_len = int(cfg.perf_gate_window) if int(cfg.perf_gate_window) > 0 else None
        perf_shadow_profits: deque[float] = deque(maxlen=perf_len)
        perf_shadow_wins: deque[int] = deque(maxlen=perf_len)
        robust_hist_len = max(1, int(cfg.robust_ev_veto_window))
        robust_late_inflow_ratio: deque[float] = deque(maxlen=int(robust_hist_len))
        robust_late_bull_share: deque[float] = deque(maxlen=int(robust_hist_len))

        return _CandidateState(
            cfg=cfg,
            shadow_bankroll_bnb=float(shadow_initial_bankroll_bnb),
            shadow_peak_bankroll_bnb=float(shadow_initial_bankroll_bnb),
            anti_martingale_scale=1.0,
            circuit_breaker_skip_rounds_remaining=0,
            circuit_breaker_level=0,
            circuit_breaker_last_trigger_settled_round=None,
            circuit_breaker_reentry_rounds_remaining=0,
            adaptive_shadow_round_profit=adaptive_shadow_round_profit,
            adaptive_shadow_bet_profit=adaptive_shadow_bet_profit,
            adaptive_shadow_bet_wins=adaptive_shadow_bet_wins,
            perf_shadow_profits=perf_shadow_profits,
            perf_shadow_wins=perf_shadow_wins,
            robust_late_inflow_ratio=robust_late_inflow_ratio,
            robust_late_bull_share=robust_late_bull_share,
        )

    def refresh_klines(self, klines: list[Kline]) -> None:
        if not klines:
            return
        close_times: list[int] = []
        close_prices: list[float] = []
        prev_ct: int | None = None
        for k in klines:
            ct = int(k.close_time_ms)
            cp = float(k.close_price)
            if prev_ct is not None and int(ct) <= int(prev_ct):
                raise InvariantError("dislocation_klines_not_strictly_ascending")
            if float(cp) <= 0.0:
                raise InvariantError("dislocation_kline_close_nonpositive")
            close_times.append(int(ct))
            close_prices.append(float(cp))
            prev_ct = int(ct)
        self._kidx = _KlineIndex(close_times_ms=tuple(close_times), close_prices=tuple(close_prices))

    def bootstrap_from_closed_rounds(self, rounds: list[Round]) -> None:
        self.settle_closed_rounds(rounds)

    def settle_closed_rounds(self, rounds: list[Round]) -> None:
        if not rounds:
            return

        ordered = sorted((r for r in rounds), key=lambda r: int(r.epoch))
        for r in ordered:
            epoch = int(r.epoch)
            if self._last_settled_epoch is not None and int(epoch) <= int(self._last_settled_epoch):
                continue

            pending = self._pending_decisions_by_epoch.pop(int(epoch), None)
            decisions = pending if pending is not None else self._compute_decisions_for_round(round_t=r)

            selector_rows: dict[str, _SelectorBetRow] = {}
            for name in self._candidate_order:
                state = self._candidate_states[name]
                dec = decisions[name]
                row = self._settle_candidate_decision(state=state, dec=dec, round_t=r)
                if row is not None:
                    selector_rows[name] = row

            self._consume_selector_rows(selector_rows)
            self._last_settled_epoch = int(epoch)

    def decide_open_round(self, *, round_t: Round, bankroll_bnb: float) -> LiveStrategyDecision:
        if round_t.lock_at is None:
            return LiveStrategyDecision(
                action="SKIP",
                bet_side=None,
                amount_bnb=0.0,
                expected_profit_bnb=0.0,
                skip_reason="round_lock_at_missing",
                p_bull=None,
                selected_strategy=None,
            )
        if float(bankroll_bnb) < 0.0:
            raise InvariantError("bankroll_negative")

        epoch = int(round_t.epoch)
        decisions = self._compute_decisions_for_round(round_t=round_t)
        self._pending_decisions_by_epoch[int(epoch)] = decisions

        if not self._selector_ready:
            return LiveStrategyDecision(
                action="SKIP",
                bet_side=None,
                amount_bnb=0.0,
                expected_profit_bnb=0.0,
                skip_reason="selector_warmup",
                p_bull=None,
                selected_strategy=None,
            )

        best_name: str | None = None
        best_score = float("-inf")

        for name in self._candidate_order:
            dec = decisions[name]
            if dec.action != "BET":
                continue
            score = self._selector_score_for_decision(candidate=name, dec=dec)
            if score is None:
                continue
            if float(score) > float(best_score):
                best_score = float(score)
                best_name = str(name)

        if best_name is None:
            return LiveStrategyDecision(
                action="SKIP",
                bet_side=None,
                amount_bnb=0.0,
                expected_profit_bnb=0.0,
                skip_reason="selector_no_candidate",
                p_bull=None,
                selected_strategy=None,
            )

        chosen = decisions[best_name]
        if chosen.action != "BET" or chosen.side is None:
            raise InvariantError("selector_chosen_non_bet")
        total_cost = float(chosen.bet_bnb) + float(GAS_COST_BET_BNB)
        if float(total_cost) > float(bankroll_bnb):
            return LiveStrategyDecision(
                action="SKIP",
                bet_side=None,
                amount_bnb=0.0,
                expected_profit_bnb=0.0,
                skip_reason="insufficient_bankroll_real",
                p_bull=float(chosen.p_nowcast_bull) if chosen.p_nowcast_bull is not None else None,
                selected_strategy=str(best_name),
            )

        side_out = "Bull" if str(chosen.side).upper() == "BULL" else "Bear"
        return LiveStrategyDecision(
            action="BET",
            bet_side=str(side_out),
            amount_bnb=float(chosen.bet_bnb),
            expected_profit_bnb=float(chosen.expected_net_selected or 0.0),
            skip_reason=None,
            p_bull=float(chosen.p_nowcast_bull) if chosen.p_nowcast_bull is not None else None,
            selected_strategy=str(best_name),
        )

    def selector_ready(self) -> bool:
        """Return whether selector warmup has completed."""

        return bool(self._selector_ready)

    def export_kline_index_state(self) -> dict[str, object]:
        """Export kline index state for backtest cache reuse."""

        return {
            "close_times_ms": [int(x) for x in self._kidx.close_times_ms],
            "close_prices": [float(x) for x in self._kidx.close_prices],
        }

    def import_kline_index_state(self, *, state: dict[str, object]) -> None:
        """Restore kline index state from backtest cache."""

        times_raw = list(state.get("close_times_ms", []))
        prices_raw = list(state.get("close_prices", []))
        if len(times_raw) != len(prices_raw):
            raise InvariantError("dislocation_kline_index_len_mismatch")
        if not times_raw:
            raise InvariantError("dislocation_kline_index_empty")

        times = tuple(int(x) for x in times_raw)
        prices = tuple(float(x) for x in prices_raw)
        self._kidx = _KlineIndex(close_times_ms=times, close_prices=prices)

    def export_bootstrap_state(self) -> dict[str, object]:
        """Export selector/candidate state snapshot for backtest bootstrap cache."""

        return {
            "candidate_order": list(self._candidate_order),
            "candidate_states": self._candidate_states,
            "last_settled_epoch": self._last_settled_epoch,
            "selector_ready": bool(self._selector_ready),
            "settled_round_count": int(self._settled_round_count),
            "warmup_rows_by_candidate": self._warmup_rows_by_candidate,
            "selector_edges_by_candidate": self._selector_edges_by_candidate,
            "selector_sum_profit": self._selector_sum_profit,
            "selector_cnt_profit": self._selector_cnt_profit,
        }

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        """Restore selector/candidate state snapshot for backtest bootstrap cache."""

        candidate_order = list(state.get("candidate_order", []))
        if candidate_order != list(self._candidate_order):
            raise InvariantError("dislocation_snapshot_candidate_order_mismatch")

        candidate_states = state.get("candidate_states")
        if not isinstance(candidate_states, dict):
            raise InvariantError("dislocation_snapshot_candidate_states_missing")
        if set(candidate_states.keys()) != set(self._candidate_order):
            raise InvariantError("dislocation_snapshot_candidate_set_mismatch")
        self._candidate_states = {str(k): candidate_states[str(k)] for k in self._candidate_order}

        last_settled_epoch = state.get("last_settled_epoch")
        if last_settled_epoch is None:
            self._last_settled_epoch = None
        else:
            self._last_settled_epoch = int(last_settled_epoch)

        self._selector_ready = bool(state.get("selector_ready", False))
        self._settled_round_count = int(state.get("settled_round_count", 0))
        warmup_rows_raw = dict(state.get("warmup_rows_by_candidate", {}))
        self._warmup_rows_by_candidate = {
            str(name): list(warmup_rows_raw.get(str(name), []))
            for name in self._candidate_order
        }
        self._selector_edges_by_candidate = state.get("selector_edges_by_candidate")
        selector_sum_raw = dict(state.get("selector_sum_profit", {}))
        selector_cnt_raw = dict(state.get("selector_cnt_profit", {}))
        self._selector_sum_profit = {
            str(name): dict(selector_sum_raw.get(str(name), {}))
            for name in self._candidate_order
        }
        self._selector_cnt_profit = {
            str(name): dict(selector_cnt_raw.get(str(name), {}))
            for name in self._candidate_order
        }
        self._pending_decisions_by_epoch = {}

    def candidate_signals_for_open_round(self, *, round_t: Round) -> dict[str, StrategyCandidateSignal]:
        """Return per-candidate routing signals for the target open round."""

        decisions = self._compute_decisions_for_round(round_t=round_t)
        self._pending_decisions_by_epoch[int(round_t.epoch)] = decisions
        return self._candidate_signals_from_decisions(decisions=decisions)

    def _candidate_signals_from_decisions(
        self,
        *,
        decisions: dict[str, _CandidateRoundDecision],
    ) -> dict[str, StrategyCandidateSignal]:
        """Convert internal candidate decisions to a shared signal contract."""

        out: dict[str, StrategyCandidateSignal] = {}
        for name in self._candidate_order:
            dec = decisions[str(name)]
            selector_score = None
            if bool(self._selector_ready) and str(dec.action) == "BET":
                selector_score = self._selector_score_for_decision(
                    candidate=str(name),
                    dec=dec,
                )
            out[str(name)] = self._candidate_signal_from_decision(
                candidate_name=str(name),
                dec=dec,
                selector_score_bnb=selector_score,
            )
        return out

    @staticmethod
    def _candidate_signal_from_decision(
        *,
        candidate_name: str,
        dec: _CandidateRoundDecision,
        selector_score_bnb: float | None,
    ) -> StrategyCandidateSignal:
        """Map one dislocation candidate decision to a strategy-agnostic signal."""

        is_bet = str(dec.action) == "BET" and dec.side is not None and float(dec.bet_bnb) > 0.0
        bet_side = None
        if bool(is_bet):
            bet_side = "Bull" if str(dec.side).upper() == "BULL" else "Bear"
        skip_reason = None if bool(is_bet) else str(dec.skip_reason or "unknown_skip_reason")
        return StrategyCandidateSignal(
            candidate_name=str(candidate_name),
            action="BET" if bool(is_bet) else "SKIP",
            bet_side=bet_side,
            bet_size_bnb=float(dec.bet_bnb) if bool(is_bet) else 0.0,
            expected_profit_bnb=(
                float(dec.expected_net_selected)
                if dec.expected_net_selected is not None
                else None
            ),
            selector_score_bnb=(
                float(selector_score_bnb) if selector_score_bnb is not None else None
            ),
            skip_reason=skip_reason,
            p_bull=float(dec.p_nowcast_bull) if dec.p_nowcast_bull is not None else None,
            dislocation_bull=(
                float(dec.dislocation_bull) if dec.dislocation_bull is not None else None
            ),
            projected_late_ratio=(
                float(dec.projected_late_ratio)
                if dec.projected_late_ratio is not None
                else None
            ),
            projected_late_imbalance=(
                float(dec.projected_late_imbalance)
                if dec.projected_late_imbalance is not None
                else None
            ),
        )

    def _compute_decisions_for_round(self, *, round_t: Round) -> dict[str, _CandidateRoundDecision]:
        out: dict[str, _CandidateRoundDecision] = {}
        for name in self._candidate_order:
            state = self._candidate_states[name]
            out[name] = self._candidate_decision_for_round(round_t=round_t, state=state)
        return out

    def _candidate_decision_for_round(
        self,
        *,
        round_t: Round,
        state: _CandidateState,
    ) -> _CandidateRoundDecision:
        cfg = state.cfg
        if int(state.circuit_breaker_skip_rounds_remaining) > 0:
            state.circuit_breaker_skip_rounds_remaining = int(state.circuit_breaker_skip_rounds_remaining) - 1
            if (
                int(state.circuit_breaker_skip_rounds_remaining) == 0
                and int(cfg.circuit_breaker_reentry_rounds) > 0
            ):
                state.circuit_breaker_reentry_rounds_remaining = int(cfg.circuit_breaker_reentry_rounds)
            return _CandidateRoundDecision(
                action="SKIP",
                side=None,
                bet_bnb=0.0,
                skip_reason="circuit_breaker_cooldown",
                p_nowcast_bull=None,
                dislocation_bull=None,
                expected_net_selected=None,
                adaptive_mode_decisions=None,
                base_side=None,
                perf_shadow_track=False,
            )

        in_reentry = int(state.circuit_breaker_reentry_rounds_remaining) > 0
        if bool(in_reentry):
            state.circuit_breaker_reentry_rounds_remaining = int(state.circuit_breaker_reentry_rounds_remaining) - 1

        gate_ev_min = float(cfg.expected_net_min_bnb) if str(cfg.stake_mode) == "fixed" else float("-inf")
        projected_final_pool_total_bnb: float | None = None
        projected_final_pool_bull_bnb: float | None = None
        projected_final_pool_bear_bnb: float | None = None
        needs_projection = (
            str(cfg.pool_total_gate_mode) == "projected_final_model_only"
            or _stake_mode_uses_projected_pool_ev(stake_mode=str(cfg.stake_mode))
            or bool(cfg.late_model_conflict_flip_enabled)
            or bool(cfg.late_model_veto_enabled)
            or bool(cfg.late_model_neutral_filter_enabled)
            or float(cfg.late_support_ev_scale_bnb) > 0.0
            or float(cfg.bull_late_min_ratio) > 0.0
            or float(cfg.bull_late_min_imbalance) > -1.0
            or float(cfg.bear_late_min_ratio) > 0.0
            or float(cfg.bear_late_max_imbalance) < 1.0
        )
        if bool(needs_projection) and self._projected_pool_provider is not None:
            projected = self._projected_pool_provider.predict_final_pools_for_round(round_t=round_t)
            if projected is not None:
                projected_final_pool_total_bnb = float(projected[0])
                projected_final_pool_bull_bnb = float(projected[1])
                projected_final_pool_bear_bnb = float(projected[2])

        robust_late_inflow_ratio: float | None = None
        robust_late_bull_share: float | None = None
        if bool(cfg.robust_ev_veto_enabled):
            have_n = int(len(state.robust_late_inflow_ratio))
            need_n = max(1, int(cfg.robust_ev_veto_min_history))
            if int(have_n) >= int(need_n):
                robust_late_inflow_ratio = _median_deque(state.robust_late_inflow_ratio)
                robust_late_bull_share = _median_deque(state.robust_late_bull_share)

        adaptive_decisions: dict[str, _CoreDecision] | None = None
        if str(cfg.side_selection_mode) == "adaptive_shadow":
            adaptive_decisions = {}
            for mode in cfg.adaptive_candidate_modes:
                m = str(mode)
                mode_cfg = replace(cfg, side_selection_mode=str(m))
                adaptive_decisions[m] = _decide_core(
                    round_t=round_t,
                    kidx=self._kidx,
                    cfg=mode_cfg,
                    treasury_fee_fraction=float(self._treasury_fee_fraction),
                    expected_net_gate_bnb=float(gate_ev_min),
                    projected_final_pool_total_bnb=projected_final_pool_total_bnb,
                    projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
                    projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
                    robust_late_inflow_ratio=robust_late_inflow_ratio,
                    robust_late_bull_share=robust_late_bull_share,
                )

            fallback_mode = str(cfg.adaptive_fallback_mode)
            if fallback_mode not in adaptive_decisions:
                fallback_mode = str(cfg.adaptive_candidate_modes[0])

            scored: list[tuple[float, str]] = []
            for mode in cfg.adaptive_candidate_modes:
                mm = str(mode)
                score = _adaptive_mode_score(
                    mode=str(mm),
                    adaptive_score=str(cfg.adaptive_score),
                    adaptive_min_history=int(cfg.adaptive_min_history),
                    shadow_round_profit=state.adaptive_shadow_round_profit,
                    shadow_bet_profit=state.adaptive_shadow_bet_profit,
                    shadow_bet_wins=state.adaptive_shadow_bet_wins,
                )
                if score is not None:
                    scored.append((float(score), str(mm)))
            if scored:
                scored.sort(key=lambda x: float(x[0]), reverse=True)
                selected_mode = str(scored[0][1])
            else:
                selected_mode = str(fallback_mode)
            core = adaptive_decisions[str(selected_mode)]
        else:
            core = _decide_core(
                round_t=round_t,
                kidx=self._kidx,
                cfg=cfg,
                treasury_fee_fraction=float(self._treasury_fee_fraction),
                expected_net_gate_bnb=float(gate_ev_min),
                projected_final_pool_total_bnb=projected_final_pool_total_bnb,
                projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
                projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
                robust_late_inflow_ratio=robust_late_inflow_ratio,
                robust_late_bull_share=robust_late_bull_share,
            )

        if core.side is None:
            return _CandidateRoundDecision(
                action="SKIP",
                side=None,
                bet_bnb=0.0,
                skip_reason=str(core.reason),
                p_nowcast_bull=core.p_nowcast_bull,
                dislocation_bull=core.dislocation_bull,
                expected_net_selected=core.expected_net_selected,
                adaptive_mode_decisions=adaptive_decisions,
                base_side=None,
                perf_shadow_track=False,
            )

        if not _side_allowed(side=str(core.side), allowed_sides=str(cfg.allowed_sides)):
            return _CandidateRoundDecision(
                action="SKIP",
                side=None,
                bet_bnb=0.0,
                skip_reason="side_not_allowed",
                p_nowcast_bull=core.p_nowcast_bull,
                dislocation_bull=core.dislocation_bull,
                expected_net_selected=core.expected_net_selected,
                adaptive_mode_decisions=adaptive_decisions,
                base_side=None,
                perf_shadow_track=False,
            )

        if round_t.lock_at is None:
            return _CandidateRoundDecision(
                action="SKIP",
                side=None,
                bet_bnb=0.0,
                skip_reason="round_lock_at_missing_recheck",
                p_nowcast_bull=core.p_nowcast_bull,
                dislocation_bull=core.dislocation_bull,
                expected_net_selected=core.expected_net_selected,
                adaptive_mode_decisions=adaptive_decisions,
                base_side=None,
                perf_shadow_track=False,
            )

        cutoff_ts = int(round_t.lock_at) - int(cfg.cutoff_seconds)
        pools_now = compute_pool_amounts_wei_at_or_before(bets=round_t.bets, cutoff_ts=int(cutoff_ts))
        if int(pools_now.total_wei) <= 0:
            return _CandidateRoundDecision(
                action="SKIP",
                side=None,
                bet_bnb=0.0,
                skip_reason="cutoff_pool_empty_recheck",
                p_nowcast_bull=core.p_nowcast_bull,
                dislocation_bull=core.dislocation_bull,
                expected_net_selected=core.expected_net_selected,
                adaptive_mode_decisions=adaptive_decisions,
                base_side=None,
                perf_shadow_track=False,
            )

        bull_cut = float(pools_now.bull_wei) / float(BNB_WEI)
        bear_cut = float(pools_now.bear_wei) / float(BNB_WEI)
        late_metrics = _late_projection_metrics(
            bull_pool_cutoff_bnb=float(bull_cut),
            bear_pool_cutoff_bnb=float(bear_cut),
            projected_final_pool_bull_bnb=projected_final_pool_bull_bnb,
            projected_final_pool_bear_bnb=projected_final_pool_bear_bnb,
        )
        projected_late_ratio = None
        projected_late_imbalance = None
        if late_metrics is not None:
            projected_late_ratio = float(late_metrics[0])
            projected_late_imbalance = float(late_metrics[1])
        ev_pool_bull = (
            float(core.ev_pool_bull_bnb)
            if core.ev_pool_bull_bnb is not None and math.isfinite(float(core.ev_pool_bull_bnb))
            else float(bull_cut)
        )
        ev_pool_bear = (
            float(core.ev_pool_bear_bnb)
            if core.ev_pool_bear_bnb is not None and math.isfinite(float(core.ev_pool_bear_bnb))
            else float(bear_cut)
        )
        if float(ev_pool_bull) <= 0.0 or float(ev_pool_bear) <= 0.0:
            ev_pool_bull = float(bull_cut)
            ev_pool_bear = float(bear_cut)

        stake_scale = _drawdown_stake_scale(
            cfg=cfg,
            shadow_bankroll_bnb=float(state.shadow_bankroll_bnb),
            shadow_peak_bankroll_bnb=float(state.shadow_peak_bankroll_bnb),
        )
        anti_scale = 1.0
        if bool(cfg.anti_martingale_enabled):
            anti_scale = float(state.anti_martingale_scale)
            if bool(in_reentry):
                anti_scale = min(float(anti_scale), 1.0)

        reentry_scale = 1.0
        if bool(in_reentry):
            reentry_scale = float(cfg.circuit_breaker_reentry_scale)

        effective_stake_scale = float(stake_scale) * float(anti_scale) * float(reentry_scale)
        bet_bnb = _stake_bnb_for_decision(
            stake_mode=str(cfg.stake_mode),
            fixed_bet_bnb=float(cfg.fixed_bet_bnb),
            expected_net_selected=core.expected_net_selected,
            stake_min_bnb=float(cfg.stake_min_bnb),
            stake_max_bnb=float(cfg.stake_max_bnb),
            stake_ev_ref_bnb=float(cfg.stake_ev_ref_bnb),
            side=str(core.side),
            p_nowcast_bull=core.p_nowcast_bull,
            bull_pool_cutoff_bnb=float(bull_cut),
            bear_pool_cutoff_bnb=float(bear_cut),
            bull_pool_ev_bnb=float(ev_pool_bull),
            bear_pool_ev_bnb=float(ev_pool_bear),
            treasury_fee_fraction=float(self._treasury_fee_fraction),
            stake_max_side_pool_frac=float(cfg.stake_max_side_pool_frac),
            stake_scale=float(effective_stake_scale),
        )
        if float(bet_bnb) <= 0.0:
            return _CandidateRoundDecision(
                action="SKIP",
                side=None,
                bet_bnb=0.0,
                skip_reason="stake_nonpositive",
                p_nowcast_bull=core.p_nowcast_bull,
                dislocation_bull=core.dislocation_bull,
                expected_net_selected=core.expected_net_selected,
                adaptive_mode_decisions=adaptive_decisions,
                base_side=None,
                perf_shadow_track=False,
            )

        if core.p_nowcast_bull is None:
            return _CandidateRoundDecision(
                action="SKIP",
                side=None,
                bet_bnb=0.0,
                skip_reason="p_nowcast_missing_recheck",
                p_nowcast_bull=None,
                dislocation_bull=core.dislocation_bull,
                expected_net_selected=core.expected_net_selected,
                adaptive_mode_decisions=adaptive_decisions,
                base_side=None,
                perf_shadow_track=False,
            )

        selected_ev_actual = _expected_net_from_cutoff(
            p_nowcast_bull=float(core.p_nowcast_bull),
            bull_pool_cutoff_bnb=float(ev_pool_bull),
            bear_pool_cutoff_bnb=float(ev_pool_bear),
            side=str(core.side),
            fixed_bet_bnb=float(bet_bnb),
            treasury_fee_fraction=float(self._treasury_fee_fraction),
        )
        expected_net_min_side = _expected_net_min_for_side(cfg=cfg, side=str(core.side))
        if float(selected_ev_actual) < float(expected_net_min_side):
            return _CandidateRoundDecision(
                action="SKIP",
                side=None,
                bet_bnb=0.0,
                skip_reason="expected_net_below_min_dynamic",
                p_nowcast_bull=core.p_nowcast_bull,
                dislocation_bull=core.dislocation_bull,
                expected_net_selected=float(selected_ev_actual),
                adaptive_mode_decisions=adaptive_decisions,
                base_side=None,
                perf_shadow_track=False,
                projected_late_ratio=projected_late_ratio,
                projected_late_imbalance=projected_late_imbalance,
            )

        effective_side = str(core.side)
        effective_ev = float(selected_ev_actual)
        action = "BET"
        skip_reason = ""
        perf_shadow_track = True

        perf_gate_history_needed = max(1, int(cfg.perf_gate_min_history))
        adapt_mode = str(cfg.perf_adapt_mode)
        if (
            adapt_mode != "off"
            and int(cfg.perf_gate_window) > 0
            and len(state.perf_shadow_profits) >= int(perf_gate_history_needed)
        ):
            recent_win_rate = float(sum(state.perf_shadow_wins)) / float(len(state.perf_shadow_wins))
            recent_mean_profit = float(sum(state.perf_shadow_profits)) / float(len(state.perf_shadow_profits))
            perf_gate_fail_win = float(recent_win_rate) < float(cfg.perf_gate_min_win_rate)
            perf_gate_fail_mean = float(recent_mean_profit) < float(cfg.perf_gate_min_mean_profit_bnb)

            perf_fail_reason = ""
            if bool(perf_gate_fail_win):
                perf_fail_reason = "perf_gate_below_min_win_rate"
            elif bool(perf_gate_fail_mean):
                perf_fail_reason = "perf_gate_below_min_mean_profit"

            if perf_fail_reason:
                if adapt_mode == "skip":
                    action = "SKIP"
                    skip_reason = str(perf_fail_reason)
                elif adapt_mode == "flip":
                    flipped_side = _opposite_side(str(core.side))
                    flipped_ev = _expected_net_from_cutoff(
                        p_nowcast_bull=float(core.p_nowcast_bull),
                        bull_pool_cutoff_bnb=float(ev_pool_bull),
                        bear_pool_cutoff_bnb=float(ev_pool_bear),
                        side=str(flipped_side),
                        fixed_bet_bnb=float(bet_bnb),
                        treasury_fee_fraction=float(self._treasury_fee_fraction),
                    )
                    flipped_ev_min = _expected_net_min_for_side(cfg=cfg, side=str(flipped_side))
                    if float(flipped_ev) < float(flipped_ev_min):
                        action = "SKIP"
                        skip_reason = "perf_flip_expected_net_below_min"
                    else:
                        effective_side = str(flipped_side)
                        effective_ev = float(flipped_ev)
                else:
                    raise InvariantError("dislocation_perf_adapt_mode_unknown")

        if action == "BET":
            return _CandidateRoundDecision(
                action="BET",
                side=str(effective_side),
                bet_bnb=float(bet_bnb),
                skip_reason="",
                p_nowcast_bull=core.p_nowcast_bull,
                dislocation_bull=core.dislocation_bull,
                expected_net_selected=float(effective_ev),
                adaptive_mode_decisions=adaptive_decisions,
                base_side=str(core.side),
                perf_shadow_track=bool(perf_shadow_track),
                projected_late_ratio=projected_late_ratio,
                projected_late_imbalance=projected_late_imbalance,
            )

        return _CandidateRoundDecision(
            action="SKIP",
            side=None,
            bet_bnb=float(bet_bnb),
            skip_reason=str(skip_reason),
            p_nowcast_bull=core.p_nowcast_bull,
            dislocation_bull=core.dislocation_bull,
            expected_net_selected=float(effective_ev),
            adaptive_mode_decisions=adaptive_decisions,
            base_side=str(core.side),
            perf_shadow_track=bool(perf_shadow_track),
            projected_late_ratio=projected_late_ratio,
            projected_late_imbalance=projected_late_imbalance,
        )

    def _settle_candidate_decision(
        self,
        *,
        state: _CandidateState,
        dec: _CandidateRoundDecision,
        round_t: Round,
    ) -> _SelectorBetRow | None:
        cfg = state.cfg
        settled_round_idx = int(self._settled_round_count) + 1
        self._update_robust_ev_history(state=state, round_t=round_t)

        bull_cut_shadow: float | None = None
        bear_cut_shadow: float | None = None
        if round_t.lock_at is not None:
            cutoff_ts_shadow = int(round_t.lock_at) - int(cfg.cutoff_seconds)
            pools_shadow = compute_pool_amounts_wei_at_or_before(
                bets=round_t.bets,
                cutoff_ts=int(cutoff_ts_shadow),
            )
            if int(pools_shadow.total_wei) > 0:
                bull_cut_shadow = float(pools_shadow.bull_wei) / float(BNB_WEI)
                bear_cut_shadow = float(pools_shadow.bear_wei) / float(BNB_WEI)

        if (
            bool(dec.perf_shadow_track)
            and dec.base_side is not None
            and dec.p_nowcast_bull is not None
            and bull_cut_shadow is not None
            and bear_cut_shadow is not None
            and float(dec.bet_bnb) > 0.0
        ):
            total_cost = float(dec.bet_bnb) + float(GAS_COST_BET_BNB)
            base_shadow_outcome = settle_bet_against_closed_round(
                bet_bnb=float(dec.bet_bnb),
                bet_side=str(dec.base_side),
                round_closed=round_t,
                treasury_fee_fraction=float(self._treasury_fee_fraction),
            )
            base_shadow_profit = -float(total_cost) + float(base_shadow_outcome.credit_bnb)
            base_shadow_is_win = 1 if str(base_shadow_outcome.outcome) == "win" else 0
            state.perf_shadow_profits.append(float(base_shadow_profit))
            state.perf_shadow_wins.append(int(base_shadow_is_win))

        self._update_adaptive_shadows(
            state=state,
            round_t=round_t,
            decisions_by_mode=dec.adaptive_mode_decisions,
            bull_pool_cutoff_bnb=bull_cut_shadow,
            bear_pool_cutoff_bnb=bear_cut_shadow,
        )

        if dec.action != "BET" or dec.side is None or float(dec.bet_bnb) <= 0.0:
            return None

        total_cost = float(dec.bet_bnb) + float(GAS_COST_BET_BNB)
        outcome = settle_bet_against_closed_round(
            bet_bnb=float(dec.bet_bnb),
            bet_side=str(dec.side),
            round_closed=round_t,
            treasury_fee_fraction=float(self._treasury_fee_fraction),
        )
        profit = -float(total_cost) + float(outcome.credit_bnb)
        state.shadow_bankroll_bnb += float(profit)
        if float(state.shadow_bankroll_bnb) > float(state.shadow_peak_bankroll_bnb):
            state.shadow_peak_bankroll_bnb = float(state.shadow_bankroll_bnb)

        if bool(cfg.anti_martingale_enabled):
            state.anti_martingale_scale = _anti_martingale_next_scale(
                cfg=cfg,
                prev_scale=float(state.anti_martingale_scale),
                realized_profit_bnb=float(profit),
            )
        else:
            state.anti_martingale_scale = 1.0

        if (
            bool(cfg.circuit_breaker_enabled)
            and float(profit) < 0.0
            and int(state.circuit_breaker_skip_rounds_remaining) <= 0
            and float(cfg.circuit_breaker_drawdown_trigger_bnb) > 0.0
            and int(cfg.circuit_breaker_base_skip_rounds) > 0
        ):
            drawdown_bnb = max(
                0.0,
                float(state.shadow_peak_bankroll_bnb) - float(state.shadow_bankroll_bnb),
            )
            if float(drawdown_bnb) >= float(cfg.circuit_breaker_drawdown_trigger_bnb):
                prev_round = state.circuit_breaker_last_trigger_settled_round
                prev_level = int(state.circuit_breaker_level)
                if (
                    prev_round is not None
                    and int(settled_round_idx - int(prev_round))
                    <= int(cfg.circuit_breaker_escalation_window_rounds)
                ):
                    next_level = min(int(prev_level + 1), int(cfg.circuit_breaker_max_level))
                else:
                    next_level = 1
                state.circuit_breaker_level = int(next_level)
                state.circuit_breaker_last_trigger_settled_round = int(settled_round_idx)
                state.circuit_breaker_skip_rounds_remaining = _circuit_breaker_skip_rounds_for_level(
                    cfg=cfg,
                    level=int(next_level),
                )
                if int(state.circuit_breaker_skip_rounds_remaining) > 0:
                    state.circuit_breaker_reentry_rounds_remaining = 0
                    state.anti_martingale_scale = 1.0

        if dec.expected_net_selected is None or dec.dislocation_bull is None:
            return None
        side_idx = 0 if str(dec.side).upper() == "BULL" else 1
        return _SelectorBetRow(
            ev_selected=float(dec.expected_net_selected),
            abs_dislocation=abs(float(dec.dislocation_bull)),
            side_idx=int(side_idx),
            profit_bnb=float(profit),
        )

    def _update_adaptive_shadows(
        self,
        *,
        state: _CandidateState,
        round_t: Round,
        decisions_by_mode: dict[str, _CoreDecision] | None,
        bull_pool_cutoff_bnb: float | None,
        bear_pool_cutoff_bnb: float | None,
    ) -> None:
        if str(state.cfg.side_selection_mode) != "adaptive_shadow":
            return
        if decisions_by_mode is None:
            return

        for mode_name, mode_dec in decisions_by_mode.items():
            m = str(mode_name)
            shadow_profit, shadow_is_bet, shadow_is_win = _shadow_profit_for_decision(
                round_t=round_t,
                dec=mode_dec,
                cfg=state.cfg,
                treasury_fee_fraction=float(self._treasury_fee_fraction),
                bull_pool_cutoff_bnb=bull_pool_cutoff_bnb,
                bear_pool_cutoff_bnb=bear_pool_cutoff_bnb,
            )
            state.adaptive_shadow_round_profit[m].append(float(shadow_profit))
            if bool(shadow_is_bet):
                state.adaptive_shadow_bet_profit[m].append(float(shadow_profit))
                state.adaptive_shadow_bet_wins[m].append(1 if bool(shadow_is_win) else 0)

    @staticmethod
    def _update_robust_ev_history(*, state: _CandidateState, round_t: Round) -> None:
        cfg = state.cfg
        if round_t.lock_at is None:
            return

        lock_at = int(round_t.lock_at)
        cutoff_ts = int(lock_at) - int(cfg.cutoff_seconds)
        cutoff = compute_pool_amounts_wei_at_or_before(
            bets=round_t.bets,
            cutoff_ts=int(cutoff_ts),
        )
        if int(cutoff.total_wei) <= 0:
            return

        bull_final_bnb, bear_final_bnb = _final_pools_bnb_for_round(
            round_t=round_t,
            lock_at=int(lock_at),
        )
        final_total_bnb = float(bull_final_bnb) + float(bear_final_bnb)

        cut_bull_bnb = float(cutoff.bull_wei) / float(BNB_WEI)
        cut_bear_bnb = float(cutoff.bear_wei) / float(BNB_WEI)
        cut_total_bnb = float(cutoff.total_wei) / float(BNB_WEI)

        if float(final_total_bnb) <= 0.0 or float(cut_total_bnb) <= 0.0:
            return

        late_total_bnb = max(0.0, float(final_total_bnb) - float(cut_total_bnb))
        late_ratio = float(late_total_bnb) / float(cut_total_bnb)
        if not math.isfinite(float(late_ratio)):
            return

        late_bull_bnb = max(0.0, float(bull_final_bnb) - float(cut_bull_bnb))
        late_bear_bnb = max(0.0, float(bear_final_bnb) - float(cut_bear_bnb))
        late_side_total = float(late_bull_bnb) + float(late_bear_bnb)
        if float(late_side_total) <= 0.0:
            late_bull_share = 0.5
        else:
            late_bull_share = float(late_bull_bnb) / float(late_side_total)
        late_bull_share = _clamp(float(late_bull_share), 0.0, 1.0)

        state.robust_late_inflow_ratio.append(float(max(0.0, float(late_ratio))))
        state.robust_late_bull_share.append(float(late_bull_share))

    def _consume_selector_rows(self, rows_by_candidate: dict[str, _SelectorBetRow]) -> None:
        self._settled_round_count += 1

        if not self._selector_ready:
            for name, row in rows_by_candidate.items():
                self._warmup_rows_by_candidate[name].append(row)
            if int(self._settled_round_count) >= int(self._selector_cfg.warmup_rounds):
                self._freeze_selector_edges_and_seed()
            return

        if self._selector_edges_by_candidate is None:
            raise InvariantError("selector_edges_missing")

        for name, row in rows_by_candidate.items():
            ev_edges, dis_edges = self._selector_edges_by_candidate[name]
            key = self._selector_cell_key(
                row=row,
                ev_edges=ev_edges,
                dis_edges=dis_edges,
                use_direction_split=bool(self._selector_cfg.use_direction_split),
            )
            self._selector_sum_profit[name][key] = float(self._selector_sum_profit[name].get(key, 0.0) + float(row.profit_bnb))
            self._selector_cnt_profit[name][key] = int(self._selector_cnt_profit[name].get(key, 0) + 1)

    def _freeze_selector_edges_and_seed(self) -> None:
        edges: dict[str, tuple[list[float], list[float]]] = {}

        for name in self._candidate_order:
            rows = self._warmup_rows_by_candidate[name]
            ev_vals = [float(r.ev_selected) for r in rows]
            dis_vals = [float(r.abs_dislocation) for r in rows]
            ev_edges = _quantile_edges(ev_vals, int(self._selector_cfg.num_quantile_bins))
            dis_edges = _quantile_edges(dis_vals, int(self._selector_cfg.num_quantile_bins))
            edges[name] = (ev_edges, dis_edges)

        self._selector_edges_by_candidate = edges

        for name in self._candidate_order:
            for row in self._warmup_rows_by_candidate[name]:
                ev_edges, dis_edges = edges[name]
                key = self._selector_cell_key(
                    row=row,
                    ev_edges=ev_edges,
                    dis_edges=dis_edges,
                    use_direction_split=bool(self._selector_cfg.use_direction_split),
                )
                self._selector_sum_profit[name][key] = float(self._selector_sum_profit[name].get(key, 0.0) + float(row.profit_bnb))
                self._selector_cnt_profit[name][key] = int(self._selector_cnt_profit[name].get(key, 0) + 1)

        self._selector_ready = True

    def _selector_score_for_decision(self, *, candidate: str, dec: _CandidateRoundDecision) -> float | None:
        if dec.action != "BET" or dec.side is None:
            return None
        if dec.expected_net_selected is None or dec.dislocation_bull is None:
            return None
        if self._selector_edges_by_candidate is None:
            return None

        side_idx = 0 if str(dec.side).upper() == "BULL" else 1
        row = _SelectorBetRow(
            ev_selected=float(dec.expected_net_selected),
            abs_dislocation=abs(float(dec.dislocation_bull)),
            side_idx=int(side_idx),
            profit_bnb=0.0,
        )

        ev_edges, dis_edges = self._selector_edges_by_candidate[str(candidate)]
        key = self._selector_cell_key(
            row=row,
            ev_edges=ev_edges,
            dis_edges=dis_edges,
            use_direction_split=bool(self._selector_cfg.use_direction_split),
        )
        c = int(self._selector_cnt_profit[str(candidate)].get(key, 0))
        if int(c) < int(self._selector_cfg.min_cell_obs):
            return None

        est = float(self._selector_sum_profit[str(candidate)][key]) / float(c)
        if float(est) < float(self._selector_cfg.score_threshold):
            return None
        return float(est)

    @staticmethod
    def _selector_cell_key(
        *,
        row: _SelectorBetRow,
        ev_edges: list[float],
        dis_edges: list[float],
        use_direction_split: bool,
    ) -> tuple[int, int, int]:
        eb = _bin_index(float(row.ev_selected), ev_edges)
        db = _bin_index(float(row.abs_dislocation), dis_edges)
        sb = int(row.side_idx) if bool(use_direction_split) else 0
        return int(eb), int(db), int(sb)


def _to_selector_config(cfg: DislocationSelectorConfig) -> SelectorConfig:
    """Convert user-config selector settings into engine selector settings."""

    selector = SelectorConfig(
        warmup_rounds=int(cfg.warmup_rounds),
        num_quantile_bins=int(cfg.num_quantile_bins),
        min_cell_obs=int(cfg.min_cell_obs),
        score_threshold=float(cfg.score_threshold),
        use_direction_split=bool(cfg.use_direction_split),
    )
    if int(selector.warmup_rounds) <= 0:
        raise InvariantError("selector_warmup_rounds_must_be_positive")
    if int(selector.num_quantile_bins) <= 1:
        raise InvariantError("selector_num_quantile_bins_invalid")
    if int(selector.min_cell_obs) <= 0:
        raise InvariantError("selector_min_cell_obs_must_be_positive")
    return selector


def _to_candidate_config(
    *,
    cfg: DislocationCandidateConfig,
    cutoff_seconds: int,
) -> CandidateConfig:
    """Convert one user-config candidate profile into an engine candidate."""

    return CandidateConfig(
        name=str(cfg.name),
        cutoff_seconds=int(cutoff_seconds),
        lookback1_seconds=int(cfg.lookback1_seconds),
        lookback2_seconds=int(cfg.lookback2_seconds),
        lookback3_seconds=int(cfg.lookback3_seconds),
        weight1=float(cfg.weight1),
        weight2=float(cfg.weight2),
        weight3=float(cfg.weight3),
        temperature_bps=float(cfg.temperature_bps),
        fixed_bet_bnb=float(cfg.fixed_bet_bnb),
        dislocation_threshold_pp=float(cfg.dislocation_threshold_pp),
        nowcast_confidence_min=float(cfg.nowcast_confidence_min),
        cutoff_pool_total_min_bnb=float(cfg.cutoff_pool_total_min_bnb),
        pool_total_gate_mode=str(cfg.pool_total_gate_mode),
        projected_final_pool_multiplier=float(cfg.projected_final_pool_multiplier),
        projected_final_pool_total_min_bnb=float(cfg.projected_final_pool_total_min_bnb),
        expected_net_min_bnb=float(cfg.expected_net_min_bnb),
        bull_expected_net_extra_min_bnb=float(cfg.bull_expected_net_extra_min_bnb),
        bear_expected_net_extra_min_bnb=float(cfg.bear_expected_net_extra_min_bnb),
        bull_late_min_ratio=float(cfg.bull_late_min_ratio),
        bull_late_min_imbalance=float(cfg.bull_late_min_imbalance),
        bear_late_min_ratio=float(cfg.bear_late_min_ratio),
        bear_late_max_imbalance=float(cfg.bear_late_max_imbalance),
        late_support_ev_scale_bnb=float(cfg.late_support_ev_scale_bnb),
        side_selection_mode=str(cfg.side_selection_mode),
        allowed_sides=str(cfg.allowed_sides),
        market_extreme_min=float(cfg.market_extreme_min),
        nowcast_market_gap_min=float(cfg.nowcast_market_gap_min),
        flow_window_seconds=int(cfg.flow_window_seconds),
        flow_min_imbalance=float(cfg.flow_min_imbalance),
        flow_gate_mode=str(cfg.flow_gate_mode),
        flow_gate_relax_dislocation_min=float(cfg.flow_gate_relax_dislocation_min),
        adaptive_candidate_modes=tuple(str(m) for m in cfg.adaptive_candidate_modes),
        adaptive_window=int(cfg.adaptive_window),
        adaptive_min_history=int(cfg.adaptive_min_history),
        adaptive_score=str(cfg.adaptive_score),
        adaptive_fallback_mode=str(cfg.adaptive_fallback_mode),
        stake_mode=str(cfg.stake_mode),
        stake_min_bnb=float(cfg.stake_min_bnb),
        stake_max_bnb=float(cfg.stake_max_bnb),
        stake_ev_ref_bnb=float(cfg.stake_ev_ref_bnb),
        stake_max_side_pool_frac=float(cfg.stake_max_side_pool_frac),
        drawdown_stake_guard_enabled=bool(cfg.drawdown_stake_guard_enabled),
        drawdown_stake_guard_start_bnb=float(cfg.drawdown_stake_guard_start_bnb),
        drawdown_stake_guard_full_bnb=float(cfg.drawdown_stake_guard_full_bnb),
        drawdown_stake_guard_min_scale=float(cfg.drawdown_stake_guard_min_scale),
        anti_martingale_enabled=bool(cfg.anti_martingale_enabled),
        anti_martingale_win_multiplier=float(cfg.anti_martingale_win_multiplier),
        anti_martingale_loss_multiplier=float(cfg.anti_martingale_loss_multiplier),
        anti_martingale_min_scale=float(cfg.anti_martingale_min_scale),
        anti_martingale_max_scale=float(cfg.anti_martingale_max_scale),
        circuit_breaker_enabled=bool(cfg.circuit_breaker_enabled),
        circuit_breaker_drawdown_trigger_bnb=float(cfg.circuit_breaker_drawdown_trigger_bnb),
        circuit_breaker_base_skip_rounds=int(cfg.circuit_breaker_base_skip_rounds),
        circuit_breaker_escalation_multiplier=float(cfg.circuit_breaker_escalation_multiplier),
        circuit_breaker_escalation_window_rounds=int(cfg.circuit_breaker_escalation_window_rounds),
        circuit_breaker_max_level=int(cfg.circuit_breaker_max_level),
        circuit_breaker_max_skip_rounds=int(cfg.circuit_breaker_max_skip_rounds),
        circuit_breaker_reentry_rounds=int(cfg.circuit_breaker_reentry_rounds),
        circuit_breaker_reentry_scale=float(cfg.circuit_breaker_reentry_scale),
        perf_adapt_mode=str(cfg.perf_adapt_mode),
        perf_gate_window=int(cfg.perf_gate_window),
        perf_gate_min_history=int(cfg.perf_gate_min_history),
        perf_gate_min_win_rate=float(cfg.perf_gate_min_win_rate),
        perf_gate_min_mean_profit_bnb=float(cfg.perf_gate_min_mean_profit_bnb),
        robust_ev_veto_enabled=bool(cfg.robust_ev_veto_enabled),
        robust_ev_veto_min_history=int(cfg.robust_ev_veto_min_history),
        robust_ev_veto_window=int(cfg.robust_ev_veto_window),
        robust_ev_veto_low_inflow_mult=float(cfg.robust_ev_veto_low_inflow_mult),
        robust_ev_veto_extreme_inflow_mult=float(cfg.robust_ev_veto_extreme_inflow_mult),
        robust_ev_veto_adverse_skew=float(cfg.robust_ev_veto_adverse_skew),
        robust_ev_veto_min_expected_net_bnb=float(cfg.robust_ev_veto_min_expected_net_bnb),
        shock_filter_enabled=bool(cfg.shock_filter_enabled),
        shock_filter_window_seconds=int(cfg.shock_filter_window_seconds),
        shock_filter_min_window_total_bnb=float(cfg.shock_filter_min_window_total_bnb),
        shock_filter_min_abs_imbalance=float(cfg.shock_filter_min_abs_imbalance),
        shock_filter_min_surge_ratio=float(cfg.shock_filter_min_surge_ratio),
        late_model_conflict_flip_enabled=bool(cfg.late_model_conflict_flip_enabled),
        late_model_veto_enabled=bool(cfg.late_model_veto_enabled),
        late_model_veto_min_late_ratio=float(cfg.late_model_veto_min_late_ratio),
        late_model_veto_min_abs_imbalance=float(cfg.late_model_veto_min_abs_imbalance),
        late_model_neutral_filter_enabled=bool(cfg.late_model_neutral_filter_enabled),
        late_model_neutral_min_late_ratio=float(cfg.late_model_neutral_min_late_ratio),
        late_model_neutral_max_abs_imbalance=float(cfg.late_model_neutral_max_abs_imbalance),
    )


def build_dislocation_engine_from_config(
    *,
    selector_cfg: DislocationSelectorConfig,
    candidate_cfgs: tuple[DislocationCandidateConfig, ...],
    cutoff_seconds: int,
    treasury_fee_fraction: float,
    projected_pool_provider: _ProjectedPoolProvider | None = None,
) -> DislocationEngine:
    """Build the production dislocation engine from app config."""

    if int(cutoff_seconds) <= 0:
        raise InvariantError("dislocation_cutoff_seconds_invalid")
    if not candidate_cfgs:
        raise InvariantError("dislocation_candidates_empty")

    selector = _to_selector_config(selector_cfg)
    candidates = [
        _to_candidate_config(cfg=cfg, cutoff_seconds=int(cutoff_seconds))
        for cfg in candidate_cfgs
    ]
    return DislocationEngine(
        selector_cfg=selector,
        candidate_cfgs=candidates,
        treasury_fee_fraction=float(treasury_fee_fraction),
        shadow_initial_bankroll_bnb=float(selector_cfg.shadow_initial_bankroll_bnb),
        projected_pool_provider=projected_pool_provider,
    )
