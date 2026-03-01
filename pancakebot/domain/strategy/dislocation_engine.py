from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, replace

from pancakebot.config.strategy_config import DislocationCandidateConfig, DislocationSelectorConfig
from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei_at_or_before
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
class _KlineIndex:
    close_times_ms: tuple[int, ...]
    close_prices: tuple[float, ...]

    def spot_at_or_before(self, ts_ms: int) -> float | None:
        idx = bisect.bisect_right(self.close_times_ms, int(ts_ms)) - 1
        if int(idx) < 0:
            return None
        return float(self.close_prices[int(idx)])


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
    adaptive_shadow_round_profit: dict[str, deque[float]]
    adaptive_shadow_bet_profit: dict[str, deque[float]]
    adaptive_shadow_bet_wins: dict[str, deque[int]]
    perf_shadow_profits: deque[float]
    perf_shadow_wins: deque[int]


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


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


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
    treasury_fee_fraction: float,
    stake_max_side_pool_frac: float,
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

    mode = str(stake_mode)
    if mode == "fixed":
        return float(min(float(mx), max(float(mn), float(fixed_bet_bnb))))

    if mode == "ev_scaled":
        ref = max(1e-9, float(stake_ev_ref_bnb))
        ev = float(expected_net_selected) if expected_net_selected is not None else float("nan")
        if not math.isfinite(float(ev)):
            frac = 0.0
        else:
            frac = float(ev) / float(ref)
        frac = max(0.0, min(1.0, float(frac)))
        return float(mn + (float(mx) - float(mn)) * float(frac))

    if mode == "ev_optimal":
        if side is None or p_nowcast_bull is None or bull_pool_cutoff_bnb is None or bear_pool_cutoff_bnb is None:
            return float(mn)
        if float(mx) <= float(mn):
            return float(mn)

        side_u = str(side).upper()
        if side_u not in ("BULL", "BEAR"):
            return float(mn)

        points = 31
        best_s = float(mn)
        best_ev = float("-inf")
        span = float(mx) - float(mn)
        for i in range(points):
            s = float(mn + (float(span) * float(i) / float(points - 1)))
            ev = _expected_net_from_cutoff(
                p_nowcast_bull=float(p_nowcast_bull),
                bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
                bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
                side=str(side_u),
                fixed_bet_bnb=float(s),
                treasury_fee_fraction=float(treasury_fee_fraction),
            )
            if float(ev) > float(best_ev):
                best_ev = float(ev)
                best_s = float(s)
        return float(best_s)

    raise InvariantError("dislocation_stake_mode_unknown")


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
    if float(pool_total_bnb) < float(cfg.cutoff_pool_total_min_bnb):
        return _CoreDecision(
            side=None,
            reason="cutoff_pool_below_min_total",
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
    ev_bull = _expected_net_from_cutoff(
        p_nowcast_bull=float(p_now),
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        side="BULL",
        fixed_bet_bnb=float(cfg.fixed_bet_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    ev_bear = _expected_net_from_cutoff(
        p_nowcast_bull=float(p_now),
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        side="BEAR",
        fixed_bet_bnb=float(cfg.fixed_bet_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
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

    selected_ev = float(ev_bull) if str(side) == "BULL" else float(ev_bear)
    if float(selected_ev) < float(expected_net_gate_bnb):
        return _CoreDecision(
            side=None,
            reason="expected_net_below_min",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=float(ev_bull),
            expected_net_bear=float(ev_bear),
            expected_net_selected=float(selected_ev),
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
        treasury_fee_fraction=float(treasury_fee_fraction),
        stake_max_side_pool_frac=float(cfg.stake_max_side_pool_frac),
    )
    if float(bet_bnb) <= 0.0:
        return 0.0, False, False

    selected_ev_actual = _expected_net_from_cutoff(
        p_nowcast_bull=float(dec.p_nowcast_bull),
        bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
        bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
        side=str(dec.side),
        fixed_bet_bnb=float(bet_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    if float(selected_ev_actual) < float(cfg.expected_net_min_bnb):
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
    ) -> None:
        if not candidate_cfgs:
            raise InvariantError("dislocation_candidates_empty")
        if not (0.0 <= float(treasury_fee_fraction) < 1.0):
            raise InvariantError("dislocation_treasury_fee_out_of_range")
        if float(shadow_initial_bankroll_bnb) <= 0.0:
            raise InvariantError("dislocation_shadow_initial_bankroll_nonpositive")

        self._selector_cfg = selector_cfg
        self._treasury_fee_fraction = float(treasury_fee_fraction)
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

        return _CandidateState(
            cfg=cfg,
            shadow_bankroll_bnb=float(shadow_initial_bankroll_bnb),
            adaptive_shadow_round_profit=adaptive_shadow_round_profit,
            adaptive_shadow_bet_profit=adaptive_shadow_bet_profit,
            adaptive_shadow_bet_wins=adaptive_shadow_bet_wins,
            perf_shadow_profits=perf_shadow_profits,
            perf_shadow_wins=perf_shadow_wins,
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
        gate_ev_min = float(cfg.expected_net_min_bnb) if str(cfg.stake_mode) == "fixed" else float("-inf")

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
            treasury_fee_fraction=float(self._treasury_fee_fraction),
            stake_max_side_pool_frac=float(cfg.stake_max_side_pool_frac),
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
            bull_pool_cutoff_bnb=float(bull_cut),
            bear_pool_cutoff_bnb=float(bear_cut),
            side=str(core.side),
            fixed_bet_bnb=float(bet_bnb),
            treasury_fee_fraction=float(self._treasury_fee_fraction),
        )
        if float(selected_ev_actual) < float(cfg.expected_net_min_bnb):
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
                        bull_pool_cutoff_bnb=float(bull_cut),
                        bear_pool_cutoff_bnb=float(bear_cut),
                        side=str(flipped_side),
                        fixed_bet_bnb=float(bet_bnb),
                        treasury_fee_fraction=float(self._treasury_fee_fraction),
                    )
                    if float(flipped_ev) < float(cfg.expected_net_min_bnb):
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
        )

    def _settle_candidate_decision(
        self,
        *,
        state: _CandidateState,
        dec: _CandidateRoundDecision,
        round_t: Round,
    ) -> _SelectorBetRow | None:
        cfg = state.cfg

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
        expected_net_min_bnb=float(cfg.expected_net_min_bnb),
        side_selection_mode=str(cfg.side_selection_mode),
        market_extreme_min=float(cfg.market_extreme_min),
        flow_window_seconds=int(cfg.flow_window_seconds),
        flow_min_imbalance=float(cfg.flow_min_imbalance),
        flow_gate_mode=str(cfg.flow_gate_mode),
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
        perf_adapt_mode=str(cfg.perf_adapt_mode),
        perf_gate_window=int(cfg.perf_gate_window),
        perf_gate_min_history=int(cfg.perf_gate_min_history),
        perf_gate_min_win_rate=float(cfg.perf_gate_min_win_rate),
        perf_gate_min_mean_profit_bnb=float(cfg.perf_gate_min_mean_profit_bnb),
    )


def build_dislocation_engine_from_config(
    *,
    selector_cfg: DislocationSelectorConfig,
    candidate_cfgs: tuple[DislocationCandidateConfig, ...],
    cutoff_seconds: int,
    treasury_fee_fraction: float,
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
    )
