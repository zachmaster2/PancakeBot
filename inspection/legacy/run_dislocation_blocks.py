from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei_at_or_before
from pancakebot.domain.types import Round
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
class KlineIndex:
    close_times_ms: list[int]
    close_prices: list[float]

    def spot_at_or_before(self, ts_ms: int) -> float | None:
        idx = bisect.bisect_right(self.close_times_ms, int(ts_ms)) - 1
        if int(idx) < 0:
            return None
        return float(self.close_prices[int(idx)])


@dataclass(frozen=True, slots=True)
class Decision:
    side: str | None
    reason: str
    p_nowcast_bull: float | None
    p_market_bull: float | None
    dislocation_bull: float | None
    expected_net_bull: float | None
    expected_net_bear: float | None
    expected_net_selected: float | None
    pool_total_bnb_cutoff: float | None


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _sigmoid(x: float) -> float:
    xx = float(x)
    if float(xx) >= 0.0:
        z = math.exp(-float(xx))
        return float(1.0 / (1.0 + float(z)))
    z = math.exp(float(xx))
    return float(float(z) / (1.0 + float(z)))


def _load_rounds(path: Path) -> list[Round]:
    if not path.exists():
        raise FileNotFoundError(f"missing_rounds_jsonl: {path}")
    out: list[Round] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            r = Round.from_json(json.loads(s))
            if bool(r.failed):
                continue
            if r.lock_at is None or r.close_at is None or r.lock_price is None or r.close_price is None:
                continue
            if float(r.lock_price) <= 0.0 or float(r.close_price) <= 0.0:
                continue
            out.append(r)
    if not out:
        raise InvariantError("dislocation_rounds_empty")
    prev = int(out[0].epoch)
    for r in out[1:]:
        e = int(r.epoch)
        if int(e) <= int(prev):
            raise InvariantError("dislocation_rounds_not_strictly_ascending")
        prev = int(e)
    return out


def _load_kline_index(path: Path) -> KlineIndex:
    if not path.exists():
        raise FileNotFoundError(f"missing_klines_jsonl: {path}")
    times: list[int] = []
    prices: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            obj = json.loads(s)
            ct = int(obj["close_time_ms"])
            cp = float(obj["close_price"])
            if times and int(ct) <= int(times[-1]):
                raise InvariantError("dislocation_klines_not_strictly_ascending")
            times.append(int(ct))
            prices.append(float(cp))
    if not times:
        raise InvariantError("dislocation_klines_empty")
    return KlineIndex(close_times_ms=times, close_prices=prices)


def _nowcast_probability_bull(
    *,
    round_t: Round,
    kidx: KlineIndex,
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
    if int(round_t.lock_at) <= 0:
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


def _opposite_side(side: str) -> str:
    s = str(side).upper()
    if s == "BULL":
        return "BEAR"
    if s == "BEAR":
        return "BULL"
    raise InvariantError("dislocation_side_invalid")


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


def _shadow_profit_for_decision(
    *,
    round_t: Round,
    dec: Decision,
    stake_mode: str,
    fixed_bet_bnb: float,
    stake_min_bnb: float,
    stake_max_bnb: float,
    stake_ev_ref_bnb: float,
    stake_max_side_pool_frac: float,
    expected_net_min_bnb: float,
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
        stake_mode=str(stake_mode),
        fixed_bet_bnb=float(fixed_bet_bnb),
        expected_net_selected=dec.expected_net_selected,
        stake_min_bnb=float(stake_min_bnb),
        stake_max_bnb=float(stake_max_bnb),
        stake_ev_ref_bnb=float(stake_ev_ref_bnb),
        side=str(dec.side),
        p_nowcast_bull=dec.p_nowcast_bull,
        bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
        bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
        stake_max_side_pool_frac=float(stake_max_side_pool_frac),
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
    if float(selected_ev_actual) < float(expected_net_min_bnb):
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


def _stake_bnb_for_decision(
    *,
    stake_mode: str,
    fixed_bet_bnb: float,
    expected_net_selected: float | None,
    stake_min_bnb: float,
    stake_max_bnb: float,
    stake_ev_ref_bnb: float,
    side: str | None = None,
    p_nowcast_bull: float | None = None,
    bull_pool_cutoff_bnb: float | None = None,
    bear_pool_cutoff_bnb: float | None = None,
    treasury_fee_fraction: float = 0.03,
    stake_max_side_pool_frac: float = 1.0,
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
        if (
            side is None
            or p_nowcast_bull is None
            or bull_pool_cutoff_bnb is None
            or bear_pool_cutoff_bnb is None
        ):
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


def _decide(
    *,
    round_t: Round,
    kidx: KlineIndex,
    cutoff_seconds: int,
    lookback1_seconds: int,
    lookback2_seconds: int,
    lookback3_seconds: int,
    weight1: float,
    weight2: float,
    weight3: float,
    temperature_bps: float,
    fixed_bet_bnb: float,
    treasury_fee_fraction: float,
    dislocation_threshold_pp: float,
    nowcast_confidence_min: float,
    cutoff_pool_total_min_bnb: float,
    expected_net_min_bnb: float,
    side_selection_mode: str,
    market_extreme_min: float,
    flow_window_seconds: int,
    flow_min_imbalance: float,
    flow_gate_mode: str,
) -> Decision:
    if round_t.lock_at is None:
        return Decision(
            side=None,
            reason="round_lock_at_missing",
            p_nowcast_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=None,
        )

    cutoff_ts = int(round_t.lock_at) - int(cutoff_seconds)
    pools = compute_pool_amounts_wei_at_or_before(bets=round_t.bets, cutoff_ts=int(cutoff_ts))
    if int(pools.total_wei) <= 0:
        return Decision(
            side=None,
            reason="cutoff_pool_empty",
            p_nowcast_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=0.0,
        )
    pool_total_bnb = float(pools.total_wei) / float(BNB_WEI)
    if float(pool_total_bnb) < float(cutoff_pool_total_min_bnb):
        return Decision(
            side=None,
            reason="cutoff_pool_below_min_total",
            p_nowcast_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=float(pool_total_bnb),
        )

    p_now = _nowcast_probability_bull(
        round_t=round_t,
        kidx=kidx,
        cutoff_seconds=int(cutoff_seconds),
        lookback1_seconds=int(lookback1_seconds),
        lookback2_seconds=int(lookback2_seconds),
        lookback3_seconds=int(lookback3_seconds),
        weight1=float(weight1),
        weight2=float(weight2),
        weight3=float(weight3),
        temperature_bps=float(temperature_bps),
    )
    if p_now is None:
        return Decision(
            side=None,
            reason="nowcast_unavailable",
            p_nowcast_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=float(pool_total_bnb),
        )

    p_market_bull = float(pools.bull_wei) / float(pools.total_wei)
    dislocation_bull = float(p_now) - float(p_market_bull)

    if abs(float(p_now) - 0.5) < float(nowcast_confidence_min):
        return Decision(
            side=None,
            reason="nowcast_confidence_below_min",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=float(pool_total_bnb),
        )
    if abs(float(dislocation_bull)) < float(dislocation_threshold_pp) / 100.0:
        return Decision(
            side=None,
            reason="dislocation_below_threshold",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=float(pool_total_bnb),
        )

    bull_cut_bnb = float(pools.bull_wei) / float(BNB_WEI)
    bear_cut_bnb = float(pools.bear_wei) / float(BNB_WEI)
    ev_bull = _expected_net_from_cutoff(
        p_nowcast_bull=float(p_now),
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        side="BULL",
        fixed_bet_bnb=float(fixed_bet_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    ev_bear = _expected_net_from_cutoff(
        p_nowcast_bull=float(p_now),
        bull_pool_cutoff_bnb=float(bull_cut_bnb),
        bear_pool_cutoff_bnb=float(bear_cut_bnb),
        side="BEAR",
        fixed_bet_bnb=float(fixed_bet_bnb),
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    side_mode = str(side_selection_mode)
    if side_mode == "ev_max":
        side = "BULL" if float(ev_bull) >= float(ev_bear) else "BEAR"
    elif side_mode == "nowcast":
        side = "BULL" if float(p_now) >= 0.5 else "BEAR"
    elif side_mode == "nowcast_contra":
        side = "BEAR" if float(p_now) >= 0.5 else "BULL"
    elif side_mode == "market_follow":
        if abs(float(p_market_bull) - 0.5) < float(market_extreme_min):
            return Decision(
                side=None,
                reason="market_not_extreme",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
                pool_total_bnb_cutoff=float(pool_total_bnb),
            )
        side = "BULL" if float(p_market_bull) >= 0.5 else "BEAR"
    elif side_mode == "market_contra":
        if abs(float(p_market_bull) - 0.5) < float(market_extreme_min):
            return Decision(
                side=None,
                reason="market_not_extreme",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
                pool_total_bnb_cutoff=float(pool_total_bnb),
            )
        side = "BEAR" if float(p_market_bull) >= 0.5 else "BULL"
    elif side_mode == "nowcast_when_market_disagree":
        if abs(float(p_market_bull) - 0.5) < float(market_extreme_min):
            return Decision(
                side=None,
                reason="market_not_extreme",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
                pool_total_bnb_cutoff=float(pool_total_bnb),
            )
        now_sign = 1 if float(p_now) >= 0.5 else -1
        market_sign = 1 if float(p_market_bull) >= 0.5 else -1
        if int(now_sign) == int(market_sign):
            return Decision(
                side=None,
                reason="nowcast_market_agree",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
                pool_total_bnb_cutoff=float(pool_total_bnb),
            )
        side = "BULL" if int(now_sign) > 0 else "BEAR"
    elif side_mode == "nowcast_when_market_agree":
        if abs(float(p_market_bull) - 0.5) < float(market_extreme_min):
            return Decision(
                side=None,
                reason="market_not_extreme",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
                pool_total_bnb_cutoff=float(pool_total_bnb),
            )
        now_sign = 1 if float(p_now) >= 0.5 else -1
        market_sign = 1 if float(p_market_bull) >= 0.5 else -1
        if int(now_sign) != int(market_sign):
            return Decision(
                side=None,
                reason="nowcast_market_disagree",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
                pool_total_bnb_cutoff=float(pool_total_bnb),
            )
        side = "BULL" if int(now_sign) > 0 else "BEAR"
    elif side_mode == "dislocation":
        side = "BULL" if float(dislocation_bull) >= 0.0 else "BEAR"
    elif side_mode == "dislocation_contra":
        side = "BEAR" if float(dislocation_bull) >= 0.0 else "BULL"
    else:
        raise InvariantError("dislocation_side_selection_mode_unknown")

    flow_mode = str(flow_gate_mode)
    if flow_mode != "off":
        flow_imb = _precutoff_flow_imbalance(
            round_t=round_t,
            cutoff_seconds=int(cutoff_seconds),
            flow_window_seconds=int(flow_window_seconds),
        )
        if flow_imb is None:
            return Decision(
                side=None,
                reason="flow_unavailable",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
                pool_total_bnb_cutoff=float(pool_total_bnb),
            )
        flow_abs = abs(float(flow_imb))
        if float(flow_abs) < float(flow_min_imbalance):
            return Decision(
                side=None,
                reason="flow_below_min_imbalance",
                p_nowcast_bull=float(p_now),
                p_market_bull=float(p_market_bull),
                dislocation_bull=float(dislocation_bull),
                expected_net_bull=float(ev_bull),
                expected_net_bear=float(ev_bear),
                expected_net_selected=None,
                pool_total_bnb_cutoff=float(pool_total_bnb),
            )
        flow_sign = 1 if float(flow_imb) > 0.0 else (-1 if float(flow_imb) < 0.0 else 0)
        side_sign = 1 if str(side).upper() == "BULL" else -1
        if flow_mode == "with_side":
            if int(flow_sign) != int(side_sign):
                return Decision(
                    side=None,
                    reason="flow_not_with_side",
                    p_nowcast_bull=float(p_now),
                    p_market_bull=float(p_market_bull),
                    dislocation_bull=float(dislocation_bull),
                    expected_net_bull=float(ev_bull),
                    expected_net_bear=float(ev_bear),
                    expected_net_selected=None,
                    pool_total_bnb_cutoff=float(pool_total_bnb),
                )
        elif flow_mode == "against_side":
            if int(flow_sign) == int(side_sign):
                return Decision(
                    side=None,
                    reason="flow_not_against_side",
                    p_nowcast_bull=float(p_now),
                    p_market_bull=float(p_market_bull),
                    dislocation_bull=float(dislocation_bull),
                    expected_net_bull=float(ev_bull),
                    expected_net_bear=float(ev_bear),
                    expected_net_selected=None,
                    pool_total_bnb_cutoff=float(pool_total_bnb),
                )
        else:
            raise InvariantError("dislocation_flow_gate_mode_unknown")

    selected_ev = float(ev_bull) if str(side) == "BULL" else float(ev_bear)
    if float(selected_ev) < float(expected_net_min_bnb):
        return Decision(
            side=None,
            reason="expected_net_below_min",
            p_nowcast_bull=float(p_now),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=float(ev_bull),
            expected_net_bear=float(ev_bear),
            expected_net_selected=float(selected_ev),
            pool_total_bnb_cutoff=float(pool_total_bnb),
        )

    return Decision(
        side=str(side),
        reason="bet",
        p_nowcast_bull=float(p_now),
        p_market_bull=float(p_market_bull),
        dislocation_bull=float(dislocation_bull),
        expected_net_bull=float(ev_bull),
        expected_net_bear=float(ev_bear),
        expected_net_selected=float(selected_ev),
        pool_total_bnb_cutoff=float(pool_total_bnb),
    )


def _slice_with_offset(*, rounds: list[Round], block_size: int, offset_rounds: int) -> list[Round]:
    if int(block_size) <= 0:
        raise InvariantError("block_size_must_be_positive")
    if int(offset_rounds) < 0:
        raise InvariantError("offset_rounds_negative")
    required = int(block_size) + int(offset_rounds)
    if len(rounds) < int(required):
        raise InvariantError("insufficient_rounds_for_offset_slice")
    tail = list(rounds[-int(required):])
    if int(offset_rounds) > 0:
        tail = tail[:-int(offset_rounds)]
    out = tail[-int(block_size):]
    if len(out) != int(block_size):
        raise InvariantError("offset_slice_len_mismatch")
    return out


def _simulate_block(
    *,
    rounds_block: list[Round],
    kidx: KlineIndex,
    cutoff_seconds: int,
    lookback1_seconds: int,
    lookback2_seconds: int,
    lookback3_seconds: int,
    weight1: float,
    weight2: float,
    weight3: float,
    temperature_bps: float,
    fixed_bet_bnb: float,
    treasury_fee_fraction: float,
    dislocation_threshold_pp: float,
    nowcast_confidence_min: float,
    cutoff_pool_total_min_bnb: float,
    expected_net_min_bnb: float,
    side_selection_mode: str,
    market_extreme_min: float,
    flow_window_seconds: int,
    flow_min_imbalance: float,
    flow_gate_mode: str,
    adaptive_candidate_modes: tuple[str, ...],
    adaptive_window: int,
    adaptive_min_history: int,
    adaptive_score: str,
    adaptive_fallback_mode: str,
    stake_mode: str,
    stake_min_bnb: float,
    stake_max_bnb: float,
    stake_ev_ref_bnb: float,
    stake_max_side_pool_frac: float,
    perf_adapt_mode: str,
    perf_gate_window: int,
    perf_gate_min_history: int,
    perf_gate_min_win_rate: float,
    perf_gate_min_mean_profit_bnb: float,
    initial_bankroll_bnb: float,
    write_trades_path: Path | None,
) -> dict[str, Any]:
    bankroll = float(initial_bankroll_bnb)
    wins = 0
    bets = 0
    bets_bull = 0
    bets_bear = 0
    gross_profit = 0.0
    gross_loss = 0.0
    skip_counts: dict[str, int] = {}
    bet_bnb_sum = 0.0
    bet_bnb_min = float("inf")
    bet_bnb_max = 0.0
    perf_flip_count = 0
    adaptive_enabled = str(side_selection_mode) == "adaptive_shadow"
    adaptive_selected_mode_counts: dict[str, int] = {}
    adaptive_shadow_round_profit: dict[str, deque[float]] = {}
    adaptive_shadow_bet_profit: dict[str, deque[float]] = {}
    adaptive_shadow_bet_wins: dict[str, deque[int]] = {}
    if bool(adaptive_enabled):
        for m in adaptive_candidate_modes:
            mm = str(m)
            adaptive_shadow_round_profit[mm] = deque(maxlen=int(adaptive_window))
            adaptive_shadow_bet_profit[mm] = deque(maxlen=int(adaptive_window))
            adaptive_shadow_bet_wins[mm] = deque(maxlen=int(adaptive_window))
    perf_shadow_profits: deque[float] = deque(
        maxlen=(int(perf_gate_window) if int(perf_gate_window) > 0 else None)
    )
    perf_shadow_wins: deque[int] = deque(
        maxlen=(int(perf_gate_window) if int(perf_gate_window) > 0 else None)
    )

    trades_rows: list[list[Any]] = []
    if write_trades_path is not None:
        trades_rows.append(
            [
                "epoch",
                "action",
                "skip_reason",
                "direction",
                "p_nowcast_bull",
                "p_market_bull",
                "dislocation_bull",
                "expected_net_bull",
                "expected_net_bear",
                "expected_net_selected",
                "pool_total_bnb_cutoff",
                "bet_size_bnb",
                "profit_bnb",
                "bankroll_bnb",
            ]
        )

    def _update_adaptive_shadows(*, round_t: Round, decisions_by_mode: dict[str, Decision] | None) -> None:
        if not bool(adaptive_enabled):
            return
        if decisions_by_mode is None:
            return

        bull_cut_shadow: float | None = None
        bear_cut_shadow: float | None = None
        if round_t.lock_at is not None:
            cutoff_ts_shadow = int(round_t.lock_at) - int(cutoff_seconds)
            pools_shadow = compute_pool_amounts_wei_at_or_before(
                bets=round_t.bets,
                cutoff_ts=int(cutoff_ts_shadow),
            )
            if int(pools_shadow.total_wei) > 0:
                bull_cut_shadow = float(pools_shadow.bull_wei) / float(BNB_WEI)
                bear_cut_shadow = float(pools_shadow.bear_wei) / float(BNB_WEI)

        for mode_name, mode_dec in decisions_by_mode.items():
            m = str(mode_name)
            shadow_profit, shadow_is_bet, shadow_is_win = _shadow_profit_for_decision(
                round_t=round_t,
                dec=mode_dec,
                stake_mode=str(stake_mode),
                fixed_bet_bnb=float(fixed_bet_bnb),
                stake_min_bnb=float(stake_min_bnb),
                stake_max_bnb=float(stake_max_bnb),
                stake_ev_ref_bnb=float(stake_ev_ref_bnb),
                stake_max_side_pool_frac=float(stake_max_side_pool_frac),
                expected_net_min_bnb=float(expected_net_min_bnb),
                treasury_fee_fraction=float(treasury_fee_fraction),
                bull_pool_cutoff_bnb=bull_cut_shadow,
                bear_pool_cutoff_bnb=bear_cut_shadow,
            )
            adaptive_shadow_round_profit[m].append(float(shadow_profit))
            if bool(shadow_is_bet):
                adaptive_shadow_bet_profit[m].append(float(shadow_profit))
                adaptive_shadow_bet_wins[m].append(1 if bool(shadow_is_win) else 0)

    for idx in range(1, len(rounds_block)):
        r = rounds_block[idx]
        adaptive_decisions: dict[str, Decision] | None = None
        gate_ev_min = float(expected_net_min_bnb) if str(stake_mode) == "fixed" else float("-inf")
        if bool(adaptive_enabled):
            adaptive_decisions = {}
            for mode in adaptive_candidate_modes:
                mode_u = str(mode)
                adaptive_decisions[mode_u] = _decide(
                    round_t=r,
                    kidx=kidx,
                    cutoff_seconds=int(cutoff_seconds),
                    lookback1_seconds=int(lookback1_seconds),
                    lookback2_seconds=int(lookback2_seconds),
                    lookback3_seconds=int(lookback3_seconds),
                    weight1=float(weight1),
                    weight2=float(weight2),
                    weight3=float(weight3),
                    temperature_bps=float(temperature_bps),
                    fixed_bet_bnb=float(fixed_bet_bnb),
                    treasury_fee_fraction=float(treasury_fee_fraction),
                    dislocation_threshold_pp=float(dislocation_threshold_pp),
                    nowcast_confidence_min=float(nowcast_confidence_min),
                    cutoff_pool_total_min_bnb=float(cutoff_pool_total_min_bnb),
                    expected_net_min_bnb=float(gate_ev_min),
                    side_selection_mode=str(mode_u),
                    market_extreme_min=float(market_extreme_min),
                    flow_window_seconds=int(flow_window_seconds),
                    flow_min_imbalance=float(flow_min_imbalance),
                    flow_gate_mode=str(flow_gate_mode),
                )

            fallback_mode = str(adaptive_fallback_mode)
            if fallback_mode not in adaptive_decisions:
                fallback_mode = str(adaptive_candidate_modes[0])

            scored: list[tuple[float, str]] = []
            for mode in adaptive_candidate_modes:
                mm = str(mode)
                score = _adaptive_mode_score(
                    mode=str(mm),
                    adaptive_score=str(adaptive_score),
                    adaptive_min_history=int(adaptive_min_history),
                    shadow_round_profit=adaptive_shadow_round_profit,
                    shadow_bet_profit=adaptive_shadow_bet_profit,
                    shadow_bet_wins=adaptive_shadow_bet_wins,
                )
                if score is None:
                    continue
                scored.append((float(score), str(mm)))

            if scored:
                scored.sort(key=lambda x: float(x[0]), reverse=True)
                selected_mode = str(scored[0][1])
            else:
                selected_mode = str(fallback_mode)

            adaptive_selected_mode_counts[selected_mode] = int(
                adaptive_selected_mode_counts.get(selected_mode, 0)
            ) + 1
            dec = adaptive_decisions[str(selected_mode)]
        else:
            dec = _decide(
                round_t=r,
                kidx=kidx,
                cutoff_seconds=int(cutoff_seconds),
                lookback1_seconds=int(lookback1_seconds),
                lookback2_seconds=int(lookback2_seconds),
                lookback3_seconds=int(lookback3_seconds),
                weight1=float(weight1),
                weight2=float(weight2),
                weight3=float(weight3),
                temperature_bps=float(temperature_bps),
                fixed_bet_bnb=float(fixed_bet_bnb),
                treasury_fee_fraction=float(treasury_fee_fraction),
                dislocation_threshold_pp=float(dislocation_threshold_pp),
                nowcast_confidence_min=float(nowcast_confidence_min),
                cutoff_pool_total_min_bnb=float(cutoff_pool_total_min_bnb),
                expected_net_min_bnb=float(gate_ev_min),
                side_selection_mode=str(side_selection_mode),
                market_extreme_min=float(market_extreme_min),
                flow_window_seconds=int(flow_window_seconds),
                flow_min_imbalance=float(flow_min_imbalance),
                flow_gate_mode=str(flow_gate_mode),
            )

        if dec.side is None:
            key = str(dec.reason)
            skip_counts[key] = int(skip_counts.get(key, 0)) + 1
            if trades_rows:
                trades_rows.append(
                    [
                        int(r.epoch),
                        "SKIP",
                        str(dec.reason),
                        "",
                        dec.p_nowcast_bull,
                        dec.p_market_bull,
                        dec.dislocation_bull,
                        dec.expected_net_bull,
                        dec.expected_net_bear,
                        dec.expected_net_selected,
                        dec.pool_total_bnb_cutoff,
                        0.0,
                        0.0,
                        float(bankroll),
                    ]
                )
            _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)
            continue

        if r.lock_at is None:
            skip_counts["round_lock_at_missing_recheck"] = int(skip_counts.get("round_lock_at_missing_recheck", 0)) + 1
            if trades_rows:
                trades_rows.append(
                    [
                        int(r.epoch),
                        "SKIP",
                        "round_lock_at_missing_recheck",
                        str(dec.side),
                        dec.p_nowcast_bull,
                        dec.p_market_bull,
                        dec.dislocation_bull,
                        dec.expected_net_bull,
                        dec.expected_net_bear,
                        dec.expected_net_selected,
                        dec.pool_total_bnb_cutoff,
                        0.0,
                        0.0,
                        float(bankroll),
                    ]
                )
            _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)
            continue

        cutoff_ts = int(r.lock_at) - int(cutoff_seconds)
        pools_now = compute_pool_amounts_wei_at_or_before(bets=r.bets, cutoff_ts=int(cutoff_ts))
        if int(pools_now.total_wei) <= 0:
            skip_counts["cutoff_pool_empty_recheck"] = int(skip_counts.get("cutoff_pool_empty_recheck", 0)) + 1
            if trades_rows:
                trades_rows.append(
                    [
                        int(r.epoch),
                        "SKIP",
                        "cutoff_pool_empty_recheck",
                        str(dec.side),
                        dec.p_nowcast_bull,
                        dec.p_market_bull,
                        dec.dislocation_bull,
                        dec.expected_net_bull,
                        dec.expected_net_bear,
                        dec.expected_net_selected,
                        0.0,
                        0.0,
                        0.0,
                        float(bankroll),
                    ]
                )
            _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)
            continue
        bull_cut_bnb_now = float(pools_now.bull_wei) / float(BNB_WEI)
        bear_cut_bnb_now = float(pools_now.bear_wei) / float(BNB_WEI)

        bet_bnb = _stake_bnb_for_decision(
            stake_mode=str(stake_mode),
            fixed_bet_bnb=float(fixed_bet_bnb),
            expected_net_selected=dec.expected_net_selected,
            stake_min_bnb=float(stake_min_bnb),
            stake_max_bnb=float(stake_max_bnb),
            stake_ev_ref_bnb=float(stake_ev_ref_bnb),
            side=str(dec.side),
            p_nowcast_bull=dec.p_nowcast_bull,
            bull_pool_cutoff_bnb=float(bull_cut_bnb_now),
            bear_pool_cutoff_bnb=float(bear_cut_bnb_now),
            treasury_fee_fraction=float(treasury_fee_fraction),
            stake_max_side_pool_frac=float(stake_max_side_pool_frac),
        )
        if float(bet_bnb) <= 0.0:
            skip_counts["stake_nonpositive"] = int(skip_counts.get("stake_nonpositive", 0)) + 1
            if trades_rows:
                trades_rows.append(
                    [
                        int(r.epoch),
                        "SKIP",
                        "stake_nonpositive",
                        str(dec.side),
                        dec.p_nowcast_bull,
                        dec.p_market_bull,
                        dec.dislocation_bull,
                        dec.expected_net_bull,
                        dec.expected_net_bear,
                        dec.expected_net_selected,
                        float(pools_now.total_wei) / float(BNB_WEI),
                        0.0,
                        0.0,
                        float(bankroll),
                    ]
                )
            _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)
            continue

        if dec.p_nowcast_bull is None:
            skip_counts["p_nowcast_missing_recheck"] = int(skip_counts.get("p_nowcast_missing_recheck", 0)) + 1
            if trades_rows:
                trades_rows.append(
                    [
                        int(r.epoch),
                        "SKIP",
                        "p_nowcast_missing_recheck",
                        str(dec.side),
                        dec.p_nowcast_bull,
                        dec.p_market_bull,
                        dec.dislocation_bull,
                        dec.expected_net_bull,
                        dec.expected_net_bear,
                        dec.expected_net_selected,
                        float(pools_now.total_wei) / float(BNB_WEI),
                        float(bet_bnb),
                        0.0,
                        float(bankroll),
                    ]
                )
            _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)
            continue

        selected_ev_actual = _expected_net_from_cutoff(
            p_nowcast_bull=float(dec.p_nowcast_bull),
            bull_pool_cutoff_bnb=float(bull_cut_bnb_now),
            bear_pool_cutoff_bnb=float(bear_cut_bnb_now),
            side=str(dec.side),
            fixed_bet_bnb=float(bet_bnb),
            treasury_fee_fraction=float(treasury_fee_fraction),
        )
        if float(selected_ev_actual) < float(expected_net_min_bnb):
            skip_counts["expected_net_below_min_dynamic"] = int(skip_counts.get("expected_net_below_min_dynamic", 0)) + 1
            if trades_rows:
                trades_rows.append(
                    [
                        int(r.epoch),
                        "SKIP",
                        "expected_net_below_min_dynamic",
                        str(dec.side),
                        dec.p_nowcast_bull,
                        dec.p_market_bull,
                        dec.dislocation_bull,
                        dec.expected_net_bull,
                        dec.expected_net_bear,
                        float(selected_ev_actual),
                        float(pools_now.total_wei) / float(BNB_WEI),
                        float(bet_bnb),
                        0.0,
                        float(bankroll),
                    ]
                )
            _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)
            continue

        total_cost = float(bet_bnb) + float(GAS_COST_BET_BNB)
        base_shadow_outcome = settle_bet_against_closed_round(
            bet_bnb=float(bet_bnb),
            bet_side=str(dec.side),
            round_closed=r,
            treasury_fee_fraction=float(treasury_fee_fraction),
        )
        base_shadow_profit = -float(total_cost) + float(base_shadow_outcome.credit_bnb)
        base_shadow_is_win = 1 if str(base_shadow_outcome.outcome) == "win" else 0

        effective_side = str(dec.side)
        effective_selected_ev = float(selected_ev_actual)
        effective_outcome = base_shadow_outcome

        perf_gate_history_needed = max(1, int(perf_gate_min_history))
        adapt_mode = str(perf_adapt_mode)
        if (
            adapt_mode != "off"
            and int(perf_gate_window) > 0
            and len(perf_shadow_profits) >= int(perf_gate_history_needed)
        ):
            recent_win_rate = float(sum(perf_shadow_wins)) / float(len(perf_shadow_wins))
            recent_mean_profit = float(sum(perf_shadow_profits)) / float(len(perf_shadow_profits))
            perf_gate_fail_win = float(recent_win_rate) < float(perf_gate_min_win_rate)
            perf_gate_fail_mean = float(recent_mean_profit) < float(perf_gate_min_mean_profit_bnb)

            perf_fail_reason = ""
            if bool(perf_gate_fail_win):
                perf_fail_reason = "perf_gate_below_min_win_rate"
            elif bool(perf_gate_fail_mean):
                perf_fail_reason = "perf_gate_below_min_mean_profit"

            if perf_fail_reason:
                if adapt_mode == "skip":
                    skip_counts[str(perf_fail_reason)] = int(skip_counts.get(str(perf_fail_reason), 0)) + 1
                    perf_shadow_profits.append(float(base_shadow_profit))
                    perf_shadow_wins.append(int(base_shadow_is_win))
                    if trades_rows:
                        trades_rows.append(
                            [
                                int(r.epoch),
                                "SKIP",
                                str(perf_fail_reason),
                                str(dec.side),
                                dec.p_nowcast_bull,
                                dec.p_market_bull,
                                dec.dislocation_bull,
                                dec.expected_net_bull,
                                dec.expected_net_bear,
                                float(selected_ev_actual),
                                float(pools_now.total_wei) / float(BNB_WEI),
                                float(bet_bnb),
                                0.0,
                                float(bankroll),
                            ]
                        )
                    _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)
                    continue
                if adapt_mode == "flip":
                    flipped_side = _opposite_side(str(dec.side))
                    flipped_ev_actual = _expected_net_from_cutoff(
                        p_nowcast_bull=float(dec.p_nowcast_bull),
                        bull_pool_cutoff_bnb=float(bull_cut_bnb_now),
                        bear_pool_cutoff_bnb=float(bear_cut_bnb_now),
                        side=str(flipped_side),
                        fixed_bet_bnb=float(bet_bnb),
                        treasury_fee_fraction=float(treasury_fee_fraction),
                    )
                    if float(flipped_ev_actual) < float(expected_net_min_bnb):
                        skip_counts["perf_flip_expected_net_below_min"] = int(
                            skip_counts.get("perf_flip_expected_net_below_min", 0)
                        ) + 1
                        perf_shadow_profits.append(float(base_shadow_profit))
                        perf_shadow_wins.append(int(base_shadow_is_win))
                        if trades_rows:
                            trades_rows.append(
                                [
                                    int(r.epoch),
                                    "SKIP",
                                    "perf_flip_expected_net_below_min",
                                    str(flipped_side),
                                    dec.p_nowcast_bull,
                                    dec.p_market_bull,
                                    dec.dislocation_bull,
                                    dec.expected_net_bull,
                                    dec.expected_net_bear,
                                    float(flipped_ev_actual),
                                    float(pools_now.total_wei) / float(BNB_WEI),
                                    float(bet_bnb),
                                    0.0,
                                    float(bankroll),
                                ]
                            )
                        _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)
                        continue

                    effective_side = str(flipped_side)
                    effective_selected_ev = float(flipped_ev_actual)
                    effective_outcome = settle_bet_against_closed_round(
                        bet_bnb=float(bet_bnb),
                        bet_side=str(effective_side),
                        round_closed=r,
                        treasury_fee_fraction=float(treasury_fee_fraction),
                    )
                    perf_flip_count += 1

        if float(bankroll) < float(total_cost):
            skip_counts["insufficient_bankroll"] = int(skip_counts.get("insufficient_bankroll", 0)) + 1
            perf_shadow_profits.append(float(base_shadow_profit))
            perf_shadow_wins.append(int(base_shadow_is_win))
            if trades_rows:
                trades_rows.append(
                    [
                        int(r.epoch),
                        "SKIP",
                        "insufficient_bankroll",
                        str(dec.side),
                        dec.p_nowcast_bull,
                        dec.p_market_bull,
                        dec.dislocation_bull,
                        dec.expected_net_bull,
                        dec.expected_net_bear,
                        float(effective_selected_ev),
                        float(pools_now.total_wei) / float(BNB_WEI),
                        float(bet_bnb),
                        0.0,
                        float(bankroll),
                    ]
                )
            _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)
            continue

        bankroll -= float(total_cost)
        outcome = effective_outcome
        bankroll += float(outcome.credit_bnb)
        profit = -float(total_cost) + float(outcome.credit_bnb)
        if float(profit) >= 0.0:
            gross_profit += float(profit)
        else:
            gross_loss += -float(profit)

        bets += 1
        bet_bnb_sum += float(bet_bnb)
        bet_bnb_min = min(float(bet_bnb_min), float(bet_bnb))
        bet_bnb_max = max(float(bet_bnb_max), float(bet_bnb))
        if str(effective_side) == "BULL":
            bets_bull += 1
        else:
            bets_bear += 1
        if str(outcome.outcome) == "win":
            wins += 1
        perf_shadow_profits.append(float(base_shadow_profit))
        perf_shadow_wins.append(int(base_shadow_is_win))

        if trades_rows:
            trades_rows.append(
                [
                    int(r.epoch),
                    "BET",
                    "",
                    str(effective_side),
                    dec.p_nowcast_bull,
                    dec.p_market_bull,
                    dec.dislocation_bull,
                    dec.expected_net_bull,
                    dec.expected_net_bear,
                    float(effective_selected_ev),
                    float(pools_now.total_wei) / float(BNB_WEI),
                    float(bet_bnb),
                    float(profit),
                    float(bankroll),
                ]
            )
        _update_adaptive_shadows(round_t=r, decisions_by_mode=adaptive_decisions)

    if write_trades_path is not None:
        write_trades_path.parent.mkdir(parents=True, exist_ok=True)
        with write_trades_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(trades_rows)

    n_rounds = int(len(rounds_block))
    n_eval = int(max(0, n_rounds - 1))
    net = float(bankroll - float(initial_bankroll_bnb))
    perf_recent_count = int(len(perf_shadow_profits))
    perf_recent_mean_profit = float("nan")
    perf_recent_win_rate = float("nan")
    if int(perf_recent_count) > 0:
        perf_recent_mean_profit = float(sum(perf_shadow_profits)) / float(perf_recent_count)
        perf_recent_win_rate = float(sum(perf_shadow_wins)) / float(perf_recent_count)
    return {
        "num_rounds": int(n_rounds),
        "num_eval_rounds": int(n_eval),
        "num_bets": int(bets),
        "num_wins": int(wins),
        "num_bets_bull": int(bets_bull),
        "num_bets_bear": int(bets_bear),
        "bet_rate": float(_safe_rate(int(bets), int(n_eval))),
        "win_rate": float(_safe_rate(int(wins), int(bets))),
        "gross_profit_bnb": float(gross_profit),
        "gross_loss_bnb": float(gross_loss),
        "net_profit_bnb": float(net),
        "initial_bankroll_bnb": float(initial_bankroll_bnb),
        "final_bankroll_bnb": float(bankroll),
        "avg_bet_bnb": float(0.0 if int(bets) <= 0 else float(bet_bnb_sum) / float(bets)),
        "min_bet_bnb": float(0.0 if int(bets) <= 0 else float(bet_bnb_min)),
        "max_bet_bnb": float(0.0 if int(bets) <= 0 else float(bet_bnb_max)),
        "perf_gate_window": int(perf_gate_window),
        "perf_gate_recent_count": int(perf_recent_count),
        "perf_gate_recent_mean_profit_bnb": float(perf_recent_mean_profit),
        "perf_gate_recent_win_rate": float(perf_recent_win_rate),
        "perf_flip_count": int(perf_flip_count),
        "adaptive_enabled": bool(adaptive_enabled),
        "adaptive_selected_mode_counts": {
            str(k): int(v) for k, v in sorted(adaptive_selected_mode_counts.items())
        },
        "skip_reason_counts": {str(k): int(v) for k, v in sorted(skip_counts.items())},
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--closed-rounds-path", type=str, default="var/closed_rounds.jsonl")
    p.add_argument("--klines-path", type=str, default="var/klines.jsonl")
    p.add_argument("--block-size", type=int, default=10000)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--skip-most-recent-blocks", type=int, default=0)

    p.add_argument("--cutoff-seconds", type=int, default=20)
    p.add_argument("--lookback1-seconds", type=int, default=60)
    p.add_argument("--lookback2-seconds", type=int, default=180)
    p.add_argument("--lookback3-seconds", type=int, default=300)
    p.add_argument("--weight1", type=float, default=0.8)
    p.add_argument("--weight2", type=float, default=0.6)
    p.add_argument("--weight3", type=float, default=0.3)
    p.add_argument("--temperature-bps", type=float, default=2.0)

    p.add_argument("--fixed-bet-bnb", type=float, default=0.05)
    p.add_argument("--initial-bankroll-bnb", type=float, default=0.5)
    p.add_argument("--treasury-fee-fraction", type=float, default=0.03)
    p.add_argument("--stake-mode", type=str, choices=("fixed", "ev_scaled", "ev_optimal"), default="fixed")
    p.add_argument("--stake-min-bnb", type=float, default=0.05)
    p.add_argument("--stake-max-bnb", type=float, default=None)
    p.add_argument("--stake-ev-ref-bnb", type=float, default=0.10)
    p.add_argument("--stake-max-side-pool-frac", type=float, default=1000000.0)

    p.add_argument("--dislocation-threshold-pp", type=float, default=3.0)
    p.add_argument("--nowcast-confidence-min", type=float, default=0.02)
    p.add_argument("--cutoff-pool-total-min-bnb", type=float, default=3.0)
    p.add_argument("--expected-net-min-bnb", type=float, default=-0.001)
    p.add_argument("--flow-window-seconds", type=int, default=0)
    p.add_argument("--flow-min-imbalance", type=float, default=0.0)
    p.add_argument("--flow-gate-mode", type=str, choices=("off", "with_side", "against_side"), default="off")
    p.add_argument(
        "--side-selection-mode",
        type=str,
        choices=tuple(_STATIC_SIDE_SELECTION_MODES) + ("adaptive_shadow",),
        default="ev_max",
    )
    p.add_argument("--market-extreme-min", type=float, default=0.0)
    p.add_argument(
        "--adaptive-candidate-modes",
        type=str,
        default="nowcast_when_market_disagree,ev_max,nowcast_contra",
    )
    p.add_argument("--adaptive-window", type=int, default=200)
    p.add_argument("--adaptive-min-history", type=int, default=100)
    p.add_argument(
        "--adaptive-score",
        type=str,
        choices=("mean_profit_per_round", "mean_profit_per_bet", "win_rate"),
        default="mean_profit_per_round",
    )
    p.add_argument("--adaptive-fallback-mode", type=str, default="nowcast_when_market_disagree")
    p.add_argument("--perf-adapt-mode", type=str, choices=("off", "skip", "flip"), default="off")
    p.add_argument("--perf-gate-window", type=int, default=0)
    p.add_argument("--perf-gate-min-history", type=int, default=0)
    p.add_argument("--perf-gate-min-win-rate", type=float, default=0.0)
    p.add_argument("--perf-gate-min-mean-profit-bnb", type=float, default=0.0)

    p.add_argument("--write-trades", action="store_true", default=False)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if int(args.block_size) <= 10:
        raise ValueError("block_size_too_small")
    if int(args.num_blocks) <= 0:
        raise ValueError("num_blocks_must_be_positive")
    if int(args.skip_most_recent_blocks) < 0:
        raise ValueError("skip_most_recent_blocks_negative")
    if int(args.cutoff_seconds) < 0:
        raise ValueError("cutoff_seconds_negative")
    if int(args.lookback1_seconds) <= 0 or int(args.lookback2_seconds) <= 0 or int(args.lookback3_seconds) <= 0:
        raise ValueError("lookback_seconds_must_be_positive")
    if float(args.temperature_bps) <= 0.0:
        raise ValueError("temperature_bps_must_be_positive")
    if float(args.fixed_bet_bnb) <= 0.0:
        raise ValueError("fixed_bet_bnb_must_be_positive")
    if float(args.initial_bankroll_bnb) <= 0.0:
        raise ValueError("initial_bankroll_bnb_must_be_positive")
    if not (0.0 <= float(args.treasury_fee_fraction) < 1.0):
        raise ValueError("treasury_fee_fraction_out_of_range")
    if float(args.dislocation_threshold_pp) < 0.0:
        raise ValueError("dislocation_threshold_pp_negative")
    if not (0.0 <= float(args.nowcast_confidence_min) <= 0.5):
        raise ValueError("nowcast_confidence_min_out_of_range")
    if float(args.cutoff_pool_total_min_bnb) < 0.0:
        raise ValueError("cutoff_pool_total_min_bnb_negative")
    if int(args.flow_window_seconds) < 0:
        raise ValueError("flow_window_seconds_negative")
    if not (0.0 <= float(args.flow_min_imbalance) <= 1.0):
        raise ValueError("flow_min_imbalance_out_of_range")
    if str(args.flow_gate_mode) != "off" and int(args.flow_window_seconds) <= 0:
        raise ValueError("flow_gate_requires_positive_window")
    if not (0.0 <= float(args.market_extreme_min) <= 0.5):
        raise ValueError("market_extreme_min_out_of_range")
    adaptive_candidate_modes = tuple(
        str(x).strip() for x in str(args.adaptive_candidate_modes).split(",") if str(x).strip()
    )
    if str(args.side_selection_mode) == "adaptive_shadow":
        if not adaptive_candidate_modes:
            raise ValueError("adaptive_candidate_modes_empty")
        for m in adaptive_candidate_modes:
            if str(m) not in _STATIC_SIDE_SELECTION_MODES:
                raise ValueError(f"adaptive_candidate_mode_invalid: {m}")
        if int(args.adaptive_window) <= 0:
            raise ValueError("adaptive_window_must_be_positive")
        if int(args.adaptive_min_history) <= 0:
            raise ValueError("adaptive_min_history_must_be_positive")
        if int(args.adaptive_min_history) > int(args.adaptive_window):
            raise ValueError("adaptive_min_history_exceeds_window")
        if str(args.adaptive_fallback_mode) not in adaptive_candidate_modes:
            raise ValueError("adaptive_fallback_mode_not_in_candidates")
    else:
        adaptive_candidate_modes = tuple()
    if str(args.perf_adapt_mode) not in ("off", "skip", "flip"):
        raise ValueError("perf_adapt_mode_invalid")
    if int(args.perf_gate_window) < 0:
        raise ValueError("perf_gate_window_negative")
    if int(args.perf_gate_min_history) < 0:
        raise ValueError("perf_gate_min_history_negative")
    if int(args.perf_gate_window) > 0 and int(args.perf_gate_min_history) > int(args.perf_gate_window):
        raise ValueError("perf_gate_min_history_exceeds_window")
    if not (0.0 <= float(args.perf_gate_min_win_rate) <= 1.0):
        raise ValueError("perf_gate_min_win_rate_out_of_range")
    if not math.isfinite(float(args.perf_gate_min_mean_profit_bnb)):
        raise ValueError("perf_gate_min_mean_profit_bnb_non_finite")
    if float(args.stake_min_bnb) <= 0.0:
        raise ValueError("stake_min_bnb_must_be_positive")
    stake_max_bnb = float(args.stake_max_bnb) if args.stake_max_bnb is not None else float(args.fixed_bet_bnb)
    if float(stake_max_bnb) <= 0.0:
        raise ValueError("stake_max_bnb_must_be_positive")
    if float(args.stake_ev_ref_bnb) <= 0.0:
        raise ValueError("stake_ev_ref_bnb_must_be_positive")
    if float(args.stake_max_side_pool_frac) <= 0.0:
        raise ValueError("stake_max_side_pool_frac_must_be_positive")

    rounds = _load_rounds(Path(str(args.closed_rounds_path)))
    kidx = _load_kline_index(Path(str(args.klines_path)))

    skip = int(args.skip_most_recent_blocks)
    offsets = [
        int(args.block_size) * i
        for i in range(int(args.num_blocks) + int(skip) - 1, int(skip) - 1, -1)
    ]

    out_dir = Path("var/exp")
    out_dir.mkdir(parents=True, exist_ok=True)

    block_rows: list[dict[str, Any]] = []
    nets: list[float] = []
    bets_total = 0
    wins_total = 0

    for block_idx, offset in enumerate(offsets, start=1):
        scenario_name = f"{args.name_prefix}_b{int(block_idx)}of{int(args.num_blocks)}_off{int(offset)}"
        block = _slice_with_offset(rounds=rounds, block_size=int(args.block_size), offset_rounds=int(offset))
        trades_path = (out_dir / scenario_name / "dislocation_trades.csv") if bool(args.write_trades) else None

        summary = _simulate_block(
            rounds_block=block,
            kidx=kidx,
            cutoff_seconds=int(args.cutoff_seconds),
            lookback1_seconds=int(args.lookback1_seconds),
            lookback2_seconds=int(args.lookback2_seconds),
            lookback3_seconds=int(args.lookback3_seconds),
            weight1=float(args.weight1),
            weight2=float(args.weight2),
            weight3=float(args.weight3),
            temperature_bps=float(args.temperature_bps),
            fixed_bet_bnb=float(args.fixed_bet_bnb),
            treasury_fee_fraction=float(args.treasury_fee_fraction),
            dislocation_threshold_pp=float(args.dislocation_threshold_pp),
            nowcast_confidence_min=float(args.nowcast_confidence_min),
            cutoff_pool_total_min_bnb=float(args.cutoff_pool_total_min_bnb),
            expected_net_min_bnb=float(args.expected_net_min_bnb),
            side_selection_mode=str(args.side_selection_mode),
            market_extreme_min=float(args.market_extreme_min),
            flow_window_seconds=int(args.flow_window_seconds),
            flow_min_imbalance=float(args.flow_min_imbalance),
            flow_gate_mode=str(args.flow_gate_mode),
            adaptive_candidate_modes=tuple(str(x) for x in adaptive_candidate_modes),
            adaptive_window=int(args.adaptive_window),
            adaptive_min_history=int(args.adaptive_min_history),
            adaptive_score=str(args.adaptive_score),
            adaptive_fallback_mode=str(args.adaptive_fallback_mode),
            stake_mode=str(args.stake_mode),
            stake_min_bnb=float(args.stake_min_bnb),
            stake_max_bnb=float(stake_max_bnb),
            stake_ev_ref_bnb=float(args.stake_ev_ref_bnb),
            stake_max_side_pool_frac=float(args.stake_max_side_pool_frac),
            perf_adapt_mode=str(args.perf_adapt_mode),
            perf_gate_window=int(args.perf_gate_window),
            perf_gate_min_history=int(args.perf_gate_min_history),
            perf_gate_min_win_rate=float(args.perf_gate_min_win_rate),
            perf_gate_min_mean_profit_bnb=float(args.perf_gate_min_mean_profit_bnb),
            initial_bankroll_bnb=float(args.initial_bankroll_bnb),
            write_trades_path=trades_path,
        )

        row = {
            "scenario": str(scenario_name),
            "block_index": int(block_idx),
            "sim_offset_rounds": int(offset),
            "epoch_first": int(block[0].epoch),
            "epoch_last": int(block[-1].epoch),
            "net": float(summary["net_profit_bnb"]),
            "bets": int(summary["num_bets"]),
            "wins": int(summary["num_wins"]),
            "bet_rate": float(summary["bet_rate"]),
            "win_rate": float(summary["win_rate"]),
        }
        block_rows.append(row)
        nets.append(float(summary["net_profit_bnb"]))
        bets_total += int(summary["num_bets"])
        wins_total += int(summary["num_wins"])

        scenario_out = out_dir / scenario_name / "dislocation_summary.json"
        scenario_out.parent.mkdir(parents=True, exist_ok=True)
        scenario_out.write_text(
            json.dumps(
                {
                    "scenario": {
                        "name": str(scenario_name),
                        "cutoff_seconds": int(args.cutoff_seconds),
                        "lookback1_seconds": int(args.lookback1_seconds),
                        "lookback2_seconds": int(args.lookback2_seconds),
                        "lookback3_seconds": int(args.lookback3_seconds),
                        "weight1": float(args.weight1),
                        "weight2": float(args.weight2),
                        "weight3": float(args.weight3),
                        "temperature_bps": float(args.temperature_bps),
                        "fixed_bet_bnb": float(args.fixed_bet_bnb),
                        "initial_bankroll_bnb": float(args.initial_bankroll_bnb),
                        "treasury_fee_fraction": float(args.treasury_fee_fraction),
                        "stake_mode": str(args.stake_mode),
                        "stake_min_bnb": float(args.stake_min_bnb),
                        "stake_max_bnb": float(stake_max_bnb),
                        "stake_ev_ref_bnb": float(args.stake_ev_ref_bnb),
                        "stake_max_side_pool_frac": float(args.stake_max_side_pool_frac),
                        "dislocation_threshold_pp": float(args.dislocation_threshold_pp),
                        "nowcast_confidence_min": float(args.nowcast_confidence_min),
                        "cutoff_pool_total_min_bnb": float(args.cutoff_pool_total_min_bnb),
                        "expected_net_min_bnb": float(args.expected_net_min_bnb),
                        "flow_window_seconds": int(args.flow_window_seconds),
                        "flow_min_imbalance": float(args.flow_min_imbalance),
                        "flow_gate_mode": str(args.flow_gate_mode),
                        "side_selection_mode": str(args.side_selection_mode),
                        "market_extreme_min": float(args.market_extreme_min),
                        "adaptive_candidate_modes": [str(x) for x in adaptive_candidate_modes],
                        "adaptive_window": int(args.adaptive_window),
                        "adaptive_min_history": int(args.adaptive_min_history),
                        "adaptive_score": str(args.adaptive_score),
                        "adaptive_fallback_mode": str(args.adaptive_fallback_mode),
                        "perf_adapt_mode": str(args.perf_adapt_mode),
                        "perf_gate_window": int(args.perf_gate_window),
                        "perf_gate_min_history": int(args.perf_gate_min_history),
                        "perf_gate_min_win_rate": float(args.perf_gate_min_win_rate),
                        "perf_gate_min_mean_profit_bnb": float(args.perf_gate_min_mean_profit_bnb),
                        "block_size": int(args.block_size),
                        "sim_offset_rounds": int(offset),
                    },
                    **summary,
                },
                indent=2,
                sort_keys=True,
            )
        )

        print(
            "BLOCK_DONE "
            + f"block={block_idx}/{args.num_blocks} "
            + f"offset={offset} "
            + f"net={row['net']} bets={row['bets']} win={row['win_rate']}"
        )

    agg = {
        "blocks": int(len(block_rows)),
        "net_total": float(sum(nets)),
        "net_mean": float(sum(nets) / len(nets)),
        "net_median": float(statistics.median(nets)),
        "net_worst": float(min(nets)),
        "net_best": float(max(nets)),
        "positive_blocks": int(sum(1 for x in nets if float(x) > 0.0)),
        "positive_block_frac": float(sum(1 for x in nets if float(x) > 0.0) / len(nets)),
        "bets_total": int(bets_total),
        "wins_total": int(wins_total),
        "win_rate_weighted": float(_safe_rate(int(wins_total), int(bets_total))),
        "bet_rate_mean": float(sum(float(r["bet_rate"]) for r in block_rows) / len(block_rows)),
        "net_per_500_rounds": float((sum(nets) / (int(args.block_size) * len(block_rows))) * 500.0),
        "cutoff_seconds": int(args.cutoff_seconds),
        "lookback1_seconds": int(args.lookback1_seconds),
        "lookback2_seconds": int(args.lookback2_seconds),
        "lookback3_seconds": int(args.lookback3_seconds),
        "weight1": float(args.weight1),
        "weight2": float(args.weight2),
        "weight3": float(args.weight3),
        "temperature_bps": float(args.temperature_bps),
        "fixed_bet_bnb": float(args.fixed_bet_bnb),
        "stake_mode": str(args.stake_mode),
        "stake_min_bnb": float(args.stake_min_bnb),
        "stake_max_bnb": float(stake_max_bnb),
        "stake_ev_ref_bnb": float(args.stake_ev_ref_bnb),
        "stake_max_side_pool_frac": float(args.stake_max_side_pool_frac),
        "dislocation_threshold_pp": float(args.dislocation_threshold_pp),
        "nowcast_confidence_min": float(args.nowcast_confidence_min),
        "cutoff_pool_total_min_bnb": float(args.cutoff_pool_total_min_bnb),
        "expected_net_min_bnb": float(args.expected_net_min_bnb),
        "flow_window_seconds": int(args.flow_window_seconds),
        "flow_min_imbalance": float(args.flow_min_imbalance),
        "flow_gate_mode": str(args.flow_gate_mode),
        "side_selection_mode": str(args.side_selection_mode),
        "market_extreme_min": float(args.market_extreme_min),
        "adaptive_candidate_modes": ",".join(str(x) for x in adaptive_candidate_modes),
        "adaptive_window": int(args.adaptive_window),
        "adaptive_min_history": int(args.adaptive_min_history),
        "adaptive_score": str(args.adaptive_score),
        "adaptive_fallback_mode": str(args.adaptive_fallback_mode),
        "perf_adapt_mode": str(args.perf_adapt_mode),
        "perf_gate_window": int(args.perf_gate_window),
        "perf_gate_min_history": int(args.perf_gate_min_history),
        "perf_gate_min_win_rate": float(args.perf_gate_min_win_rate),
        "perf_gate_min_mean_profit_bnb": float(args.perf_gate_min_mean_profit_bnb),
    }

    blocks_csv = out_dir / f"{args.name_prefix}_blocks.csv"
    agg_csv = out_dir / f"{args.name_prefix}_aggregate.csv"
    agg_json = out_dir / f"{args.name_prefix}_aggregate.json"

    with blocks_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "block_index",
                "sim_offset_rounds",
                "epoch_first",
                "epoch_last",
                "scenario",
                "net",
                "bets",
                "wins",
                "bet_rate",
                "win_rate",
            ],
        )
        w.writeheader()
        for row in block_rows:
            w.writerow(row)

    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cutoff_seconds",
                "lookback1_seconds",
                "lookback2_seconds",
                "lookback3_seconds",
                "weight1",
                "weight2",
                "weight3",
                "temperature_bps",
                "fixed_bet_bnb",
                "stake_mode",
                "stake_min_bnb",
                "stake_max_bnb",
                "stake_ev_ref_bnb",
                "stake_max_side_pool_frac",
                "dislocation_threshold_pp",
                "nowcast_confidence_min",
                "cutoff_pool_total_min_bnb",
                "expected_net_min_bnb",
                "flow_window_seconds",
                "flow_min_imbalance",
                "flow_gate_mode",
                "side_selection_mode",
                "market_extreme_min",
                "adaptive_candidate_modes",
                "adaptive_window",
                "adaptive_min_history",
                "adaptive_score",
                "adaptive_fallback_mode",
                "perf_adapt_mode",
                "perf_gate_window",
                "perf_gate_min_history",
                "perf_gate_min_win_rate",
                "perf_gate_min_mean_profit_bnb",
                "blocks",
                "net_total",
                "net_mean",
                "net_median",
                "net_worst",
                "net_best",
                "positive_blocks",
                "positive_block_frac",
                "bets_total",
                "wins_total",
                "win_rate_weighted",
                "bet_rate_mean",
                "net_per_500_rounds",
            ],
        )
        w.writeheader()
        w.writerow(agg)

    agg_json.write_text(json.dumps({"aggregate": agg, "blocks": block_rows}, indent=2, sort_keys=True))

    print(f"BLOCKS_CSV={blocks_csv}")
    print(f"AGG_CSV={agg_csv}")
    print(f"AGG_JSON={agg_json}")
    print(
        "SUMMARY "
        + f"net_total={agg['net_total']} "
        + f"net_median={agg['net_median']} "
        + f"positive_frac={agg['positive_block_frac']} "
        + f"net_per_500={agg['net_per_500_rounds']}"
    )


if __name__ == "__main__":
    main()
