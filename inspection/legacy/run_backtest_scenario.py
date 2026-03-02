from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

# Legacy compatibility island: make `inspection/legacy/pancakebot` resolve first
# while still allowing fallback to canonical modules via package __path__ overlay.
_THIS_DIR = Path(__file__).resolve().parent
_LEGACY_ROOT = str(_THIS_DIR)
if _LEGACY_ROOT not in sys.path:
    sys.path.insert(0, _LEGACY_ROOT)

from pancakebot.backtest.config import BacktestConfig
from pancakebot.backtest.runner import run_backtest
from pancakebot.config.load_config import load_app_config
from pancakebot.core.constants import (
    BNB_WEI,
    GAS_COST_BET_BNB,
    GAS_COST_CLAIM_BNB,
)
from pancakebot.domain.strategy.planner import BetDecision
from pancakebot.domain.strategy.ev_math import ChainPolicyParams, ev_for_side
from pancakebot.core.determinism import set_global_determinism
from pancakebot.core.errors import InvariantError
from pancakebot.infra.binance_us_client import BinanceUsClient
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.klines_store import KlinesStore
from pancakebot.runtime.runtime_loop import RuntimeConfig
from pancakebot.runtime.contract_constants_cache import load_contract_constants


_BINANCE_US_SYMBOL = "BNBUSDT"
_WINRATE_PROBE_PROFILES = ("none", "minimal_lock_lag1", "oracle_lock_close", "price_flow_divergence")
_FLOW_WINDOWS = ("w_p_0_to_p_50", "w_p_50_to_p_100", "w_p_0_to_p_100")
_FLOW_EPS = 1e-12
_PRICE_METRIC_PREFIXES = {
    "price_log_return_mean",
    "price_log_return_std",
    "price_log_return_abs_mean",
    "price_log_return_abs_max",
    "price_range_mean",
    "price_range_max",
    "price_volume_mean",
    "price_volume_std",
    "price_volume_max",
    "price_trade_count_mean",
    "price_trade_count_std",
    "price_trade_count_max",
}
_SUPPORTED_PRICE_WINDOWS = {15, 30, 60, 120}
_FLOW_GROUPS = {"bet_amounts", "bet_counts", "imbalance", "dynamics", "concentration", "flags"}
_DYNAMICS_COLUMNS = {
    "bull_sum_ratio_w_p_50_to_p_100_over_w_p_0_to_p_50",
    "bear_sum_ratio_w_p_50_to_p_100_over_w_p_0_to_p_50",
    "total_sum_ratio_w_p_50_to_p_100_over_w_p_0_to_p_50",
}
_CORR_STATUS_OK = "ok"
_CORR_STATUS_INSUFFICIENT = "insufficient_sample"
_CORR_STATUS_DEGENERATE = "degenerate"


def _to_bnb_from_wei(amount_wei: int) -> float:
    return float(amount_wei) / float(BNB_WEI)


def _sparse_mean(values: list[float]) -> float:
    if not values:
        raise InvariantError("sparse_mean_empty")
    return float(sum(values)) / float(len(values))


def _sparse_std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    acc = 0.0
    for v in values:
        d = float(v) - float(mean)
        acc += d * d
    var = float(acc) / float(len(values) - 1)
    return float(math.sqrt(var))


def _sparse_max(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(max(values))


def _sparse_hhi(values: list[float]) -> float:
    if not values:
        return float("nan")
    valid = [float(v) for v in values if float(v) >= 0.0]
    if not valid:
        return float("nan")
    total = float(sum(valid))
    if total <= 0.0:
        return float("nan")
    return float(sum((float(v) / float(total)) ** 2 for v in valid))


def _sparse_gini(values: list[float]) -> float:
    if not values:
        return float("nan")
    valid = [float(v) for v in values if float(v) >= 0.0]
    if not valid:
        return float("nan")
    total = float(sum(valid))
    if total <= 0.0:
        return float("nan")
    sorted_vals = sorted(valid)
    n = len(sorted_vals)
    cum = 0.0
    for i, x in enumerate(sorted_vals, start=1):
        cum += float(i) * float(x)
    return (2.0 * float(cum)) / (float(n) * float(total)) - (float(n) + 1.0) / float(n)


def _sparse_log_imb(*, bull: float, bear: float) -> float:
    return float(math.log((float(bull) + _FLOW_EPS) / (float(bear) + _FLOW_EPS)))


def _flow_window_bucket(*, start_ts: int, cutoff_ts: int, created_at: int) -> str | None:
    if int(created_at) < int(start_ts) or int(created_at) > int(cutoff_ts):
        return None
    span = int(cutoff_ts) - int(start_ts)
    if span <= 0:
        return None
    if (int(created_at) - int(start_ts)) * 2 < int(span):
        return "w_p_0_to_p_50"
    return "w_p_50_to_p_100"


def _parse_price_col(col: str) -> tuple[str, int] | None:
    if "_k_" not in str(col):
        return None
    head, sep, tail = str(col).rpartition("_k_")
    if sep != "_k_":
        return None
    try:
        n = int(tail)
    except ValueError:
        return None
    if head not in _PRICE_METRIC_PREFIXES:
        return None
    if int(n) not in _SUPPORTED_PRICE_WINDOWS:
        return None
    return str(head), int(n)


def _parse_window_col(col: str) -> tuple[str, str] | None:
    name = str(col)
    for w in _FLOW_WINDOWS:
        suffix = "_" + str(w)
        if name.endswith(suffix):
            return str(name[: -len(suffix)]), str(w)
    return None


def _compute_sparse_price_features(*, context_klines, selected_price_cols: Sequence[str]) -> dict[str, float]:
    if not selected_price_cols:
        return {}
    if not context_klines:
        raise InvariantError("context_klines_empty")

    by_window: dict[int, set[str]] = {}
    for col in selected_price_cols:
        parsed = _parse_price_col(str(col))
        if parsed is None:
            raise InvariantError(f"sparse_price_column_unrecognized: {col}")
        metric_prefix, n = parsed
        if int(n) not in by_window:
            by_window[int(n)] = set()
        by_window[int(n)].add(str(metric_prefix))

    out: dict[str, float] = {}
    for n, metric_prefixes in by_window.items():
        need = int(n) + 1
        if len(context_klines) < int(need):
            raise InvariantError("context_klines_insufficient_for_window")

        window = list(context_klines[-int(need) :])
        last_n = window[-int(n) :]

        need_returns = any(
            str(prefix).startswith("price_log_return_")
            for prefix in metric_prefixes
        )
        need_ranges = "price_range_mean" in metric_prefixes or "price_range_max" in metric_prefixes
        need_volumes = "price_volume_mean" in metric_prefixes or "price_volume_std" in metric_prefixes or "price_volume_max" in metric_prefixes
        need_trade_counts = (
            "price_trade_count_mean" in metric_prefixes
            or "price_trade_count_std" in metric_prefixes
            or "price_trade_count_max" in metric_prefixes
        )

        rets: list[float] | None = None
        if need_returns:
            closes = [float(k.close_price) for k in window]
            rets = []
            for i in range(1, len(closes)):
                prev = float(closes[i - 1])
                cur = float(closes[i])
                if prev <= 0.0 or cur <= 0.0:
                    raise InvariantError("kline_close_non_positive")
                rets.append(float(math.log(cur / prev)))
            if len(rets) != int(n):
                raise InvariantError("kline_returns_len_mismatch")

        ranges: list[float] | None = None
        if need_ranges:
            ranges = [float(k.high_price) - float(k.low_price) for k in last_n]

        volumes: list[float] | None = None
        if need_volumes:
            volumes = [float(k.volume) for k in last_n]

        trades: list[float] | None = None
        if need_trade_counts:
            trades = [float(int(k.number_of_trades)) for k in last_n]

        for prefix in metric_prefixes:
            col = f"{prefix}_k_{int(n)}"
            if prefix == "price_log_return_mean":
                if rets is None:
                    raise InvariantError("sparse_price_returns_uninitialized")
                out[col] = _sparse_mean(rets)
            elif prefix == "price_log_return_std":
                if rets is None:
                    raise InvariantError("sparse_price_returns_uninitialized")
                mu = _sparse_mean(rets)
                out[col] = _sparse_std(rets, mu)
            elif prefix == "price_log_return_abs_mean":
                if rets is None:
                    raise InvariantError("sparse_price_returns_uninitialized")
                abs_rets = [abs(float(v)) for v in rets]
                out[col] = _sparse_mean(abs_rets)
            elif prefix == "price_log_return_abs_max":
                if rets is None:
                    raise InvariantError("sparse_price_returns_uninitialized")
                abs_rets = [abs(float(v)) for v in rets]
                out[col] = _sparse_max(abs_rets)
            elif prefix == "price_range_mean":
                if ranges is None:
                    raise InvariantError("sparse_price_ranges_uninitialized")
                out[col] = _sparse_mean(ranges)
            elif prefix == "price_range_max":
                if ranges is None:
                    raise InvariantError("sparse_price_ranges_uninitialized")
                out[col] = _sparse_max(ranges)
            elif prefix == "price_volume_mean":
                if volumes is None:
                    raise InvariantError("sparse_price_volumes_uninitialized")
                out[col] = _sparse_mean(volumes)
            elif prefix == "price_volume_std":
                if volumes is None:
                    raise InvariantError("sparse_price_volumes_uninitialized")
                vol_mu = _sparse_mean(volumes)
                out[col] = _sparse_std(volumes, vol_mu)
            elif prefix == "price_volume_max":
                if volumes is None:
                    raise InvariantError("sparse_price_volumes_uninitialized")
                out[col] = _sparse_max(volumes)
            elif prefix == "price_trade_count_mean":
                if trades is None:
                    raise InvariantError("sparse_price_trade_counts_uninitialized")
                out[col] = _sparse_mean(trades)
            elif prefix == "price_trade_count_std":
                if trades is None:
                    raise InvariantError("sparse_price_trade_counts_uninitialized")
                trade_mu = _sparse_mean(trades)
                out[col] = _sparse_std(trades, trade_mu)
            elif prefix == "price_trade_count_max":
                if trades is None:
                    raise InvariantError("sparse_price_trade_counts_uninitialized")
                out[col] = _sparse_max(trades)
            else:  # pragma: no cover - defensive branch
                raise InvariantError(f"sparse_price_metric_unhandled: {prefix}")

    return out


def _compute_sparse_probe_features(
    *,
    target_round,
    prior_context_rounds,
    context_klines,
    cutoff_seconds: int,
    selected_columns: Sequence[str],
    column_groups: dict[str, str],
) -> dict[str, float]:
    out: dict[str, float] = {str(c): float("nan") for c in selected_columns}
    selected_set = set(str(c) for c in selected_columns)

    for idx in range(1, len(prior_context_rounds)):
        prev = int(prior_context_rounds[idx - 1].epoch)
        cur = int(prior_context_rounds[idx].epoch)
        if cur <= prev:
            raise InvariantError("prior_context_rounds_not_strictly_increasing")
    for idx in range(1, len(context_klines)):
        prev = int(context_klines[idx - 1].open_time_ms)
        cur = int(context_klines[idx].open_time_ms)
        if cur <= prev:
            raise InvariantError("context_klines_not_strictly_increasing")

    if target_round.lock_at is None:
        raise InvariantError("target_round_lock_at_missing")
    lock_ts = int(target_round.lock_at)
    if int(lock_ts) <= 0:
        raise InvariantError("target_round_lock_at_invalid")
    if int(target_round.start_at) <= 0:
        raise InvariantError("target_round_start_at_missing")

    flow_cols = [c for c in selected_set if column_groups[str(c)] in _FLOW_GROUPS]
    if flow_cols:
        start_ts = int(target_round.start_at)
        cutoff_ts = int(lock_ts) - int(cutoff_seconds)
        sums: dict[str, dict[str, float]] = {
            "w_p_0_to_p_50": {"Bull": 0.0, "Bear": 0.0},
            "w_p_50_to_p_100": {"Bull": 0.0, "Bear": 0.0},
            "w_p_0_to_p_100": {"Bull": 0.0, "Bear": 0.0},
        }
        counts: dict[str, dict[str, float]] = {
            "w_p_0_to_p_50": {"Bull": 0.0, "Bear": 0.0},
            "w_p_50_to_p_100": {"Bull": 0.0, "Bear": 0.0},
            "w_p_0_to_p_100": {"Bull": 0.0, "Bear": 0.0},
        }
        bull_vals: dict[str, list[float]] = {"w_p_0_to_p_50": [], "w_p_50_to_p_100": [], "w_p_0_to_p_100": []}
        bear_vals: dict[str, list[float]] = {"w_p_0_to_p_50": [], "w_p_50_to_p_100": [], "w_p_0_to_p_100": []}

        for b in target_round.bets:
            created = int(b.created_at)
            if int(created) > int(cutoff_ts):
                continue
            bucket = _flow_window_bucket(start_ts=int(start_ts), cutoff_ts=int(cutoff_ts), created_at=int(created))
            if bucket is None:
                continue
            pos = str(b.position)
            if pos not in ("Bull", "Bear"):
                continue
            amt_bnb = _to_bnb_from_wei(int(b.amount_wei))
            sums[bucket][pos] += float(amt_bnb)
            counts[bucket][pos] += 1.0
            sums["w_p_0_to_p_100"][pos] += float(amt_bnb)
            counts["w_p_0_to_p_100"][pos] += 1.0
            if pos == "Bull":
                bull_vals[bucket].append(float(amt_bnb))
                bull_vals["w_p_0_to_p_100"].append(float(amt_bnb))
            else:
                bear_vals[bucket].append(float(amt_bnb))
                bear_vals["w_p_0_to_p_100"].append(float(amt_bnb))

        for col in flow_cols:
            name = str(col)
            if name in _DYNAMICS_COLUMNS:
                side = str(name).split("_sum_ratio_")[0]
                hi = "w_p_50_to_p_100"
                lo = "w_p_0_to_p_50"
                if side == "bull":
                    num = float(sums[hi]["Bull"])
                    den = float(sums[lo]["Bull"])
                elif side == "bear":
                    num = float(sums[hi]["Bear"])
                    den = float(sums[lo]["Bear"])
                elif side == "total":
                    num = float(sums[hi]["Bull"] + sums[hi]["Bear"])
                    den = float(sums[lo]["Bull"] + sums[lo]["Bear"])
                else:
                    raise InvariantError(f"sparse_flow_side_unrecognized: {name}")
                out[name] = float("nan") if float(den) == 0.0 else float(num / den)
                continue

            parsed = _parse_window_col(name)
            if parsed is None:
                raise InvariantError(f"sparse_flow_column_unrecognized: {name}")
            prefix, window = parsed
            bull = float(sums[window]["Bull"])
            bear = float(sums[window]["Bear"])
            total = float(bull) + float(bear)
            bull_n = float(counts[window]["Bull"])
            bear_n = float(counts[window]["Bear"])
            total_n = float(bull_n) + float(bear_n)
            if prefix == "bull_sum":
                out[name] = float(bull)
            elif prefix == "bear_sum":
                out[name] = float(bear)
            elif prefix == "total_sum":
                out[name] = float(total)
            elif prefix == "bull_n":
                out[name] = float(bull_n)
            elif prefix == "bear_n":
                out[name] = float(bear_n)
            elif prefix == "total_n":
                out[name] = float(total_n)
            elif prefix == "has_any_bets":
                out[name] = 1.0 if float(total_n) > 0.0 else 0.0
            elif prefix == "has_bull_bets":
                out[name] = 1.0 if float(bull_n) > 0.0 else 0.0
            elif prefix == "has_bear_bets":
                out[name] = 1.0 if float(bear_n) > 0.0 else 0.0
            elif prefix == "log_imb":
                out[name] = _sparse_log_imb(bull=float(bull), bear=float(bear))
            elif prefix == "max_bet_bull":
                out[name] = _sparse_max(bull_vals[window])
            elif prefix == "max_bet_bear":
                out[name] = _sparse_max(bear_vals[window])
            elif prefix == "hhi_bull":
                out[name] = _sparse_hhi(bull_vals[window])
            elif prefix == "hhi_bear":
                out[name] = _sparse_hhi(bear_vals[window])
            elif prefix == "gini_bull":
                out[name] = _sparse_gini(bull_vals[window])
            elif prefix == "gini_bear":
                out[name] = _sparse_gini(bear_vals[window])
            else:
                raise InvariantError(f"sparse_flow_prefix_unrecognized: {prefix}")

    late_cols = [c for c in selected_set if str(column_groups[str(c)]) == "late_phase"]
    if late_cols:
        if not prior_context_rounds:
            raise InvariantError("late_phase_requires_prior_context_round")
        prior_last = prior_context_rounds[-1]
        if prior_last.lock_at is None or int(prior_last.lock_at) <= 0:
            late_map = {
                "late_bull_sum": float("nan"),
                "late_bear_sum": float("nan"),
                "late_total_sum": float("nan"),
                "late_bull_n": float("nan"),
                "late_bear_n": float("nan"),
                "late_total_n": float("nan"),
                "late_log_imb": float("nan"),
            }
        else:
            late_lock = int(prior_last.lock_at)
            late_cutoff = int(late_lock) - int(cutoff_seconds)
            bull = 0.0
            bear = 0.0
            bull_n = 0.0
            bear_n = 0.0
            for b in prior_last.bets:
                created = int(b.created_at)
                if int(created) <= int(late_cutoff) or int(created) > int(late_lock):
                    continue
                pos = str(b.position)
                if pos not in ("Bull", "Bear"):
                    continue
                amt_bnb = _to_bnb_from_wei(int(b.amount_wei))
                if pos == "Bull":
                    bull += float(amt_bnb)
                    bull_n += 1.0
                else:
                    bear += float(amt_bnb)
                    bear_n += 1.0
            total = float(bull) + float(bear)
            total_n = float(bull_n) + float(bear_n)
            late_map = {
                "late_bull_sum": float(bull),
                "late_bear_sum": float(bear),
                "late_total_sum": float(total),
                "late_bull_n": float(bull_n),
                "late_bear_n": float(bear_n),
                "late_total_n": float(total_n),
                "late_log_imb": _sparse_log_imb(bull=float(bull), bear=float(bear)),
            }
        for col in late_cols:
            out[str(col)] = float(late_map[str(col)])

    regime_cols = [c for c in selected_set if str(column_groups[str(c)]) == "regime"]
    if regime_cols:
        w = list(prior_context_rounds[:-1])

        def _outcomes_last_n(n: int) -> list[str]:
            if int(n) <= 0:
                return []
            tail = w[-int(n) :] if len(w) >= int(n) else []
            return [str(r.position) for r in tail if r.position is not None]

        def _bull_bear_only(seq: list[str]) -> list[str]:
            return [str(s) for s in seq if str(s) in ("Bull", "Bear")]

        def _frac(seq: list[str], side: str) -> float:
            bb = _bull_bear_only(seq)
            denom = len(bb)
            if int(denom) == 0:
                return 0.0
            num = sum(1 for s in bb if str(s) == str(side))
            return float(num) / float(denom)

        def _flip_rate(seq: list[str]) -> float:
            bb = _bull_bear_only(seq)
            if len(bb) < 2:
                return 0.0
            flips = 0
            denom = 0
            for a, b in zip(bb[:-1], bb[1:]):
                denom += 1
                if a != b:
                    flips += 1
            if int(denom) == 0:
                return 0.0
            return float(flips) / float(denom)

        seq20 = _outcomes_last_n(20)
        seq60 = _outcomes_last_n(60)
        bb_rev = [s for s in reversed([str(r.position) for r in w if r.position is not None]) if s in ("Bull", "Bear")]
        if bb_rev:
            direction = bb_rev[0]
            streak = 0
            for s in bb_rev:
                if str(s) == str(direction):
                    streak += 1
                else:
                    break
            streak_len = float(streak)
        else:
            streak_len = 0.0

        regime_map = {
            "regime_bull_frac_r_20": _frac(seq20, "Bull"),
            "regime_bear_frac_r_20": _frac(seq20, "Bear"),
            "regime_flip_rate_r_20": _flip_rate(seq20),
            "regime_bull_frac_r_60": _frac(seq60, "Bull"),
            "regime_bear_frac_r_60": _frac(seq60, "Bear"),
            "regime_flip_rate_r_60": _flip_rate(seq60),
            "regime_streak_len": float(streak_len),
        }
        for col in regime_cols:
            out[str(col)] = float(regime_map[str(col)])

    price_cols = [c for c in selected_set if str(column_groups[str(c)]) == "price"]
    if price_cols:
        out.update(_compute_sparse_price_features(context_klines=context_klines, selected_price_cols=price_cols))

    for col in selected_columns:
        c = str(col)
        if c not in out:
            raise InvariantError(f"sparse_probe_feature_missing: {c}")
        v = float(out[c])
        if math.isinf(v):
            raise InvariantError(f"sparse_probe_feature_inf: {c}")
        if not math.isfinite(v):
            out[c] = float("nan")
        else:
            out[c] = float(v)

    return out


def _build_runtime_cfg(
    *,
    config_path: str,
    train_size: int,
    calibrate_size: int,
    rw_floor: float,
    rw_power: float,
    predictability_gate_mode: str,
    predictability_gate_threshold_override: float | None,
    predictability_baseline_bet_bnb_override: float | None,
):
    cfg = load_app_config(config_path)
    set_global_determinism(seed=int(cfg.random_seed))
    round_store = ClosedRoundsStore(cfg.closed_rounds_path)
    klines_store = KlinesStore(cfg.klines_path)
    constants = load_contract_constants()
    binance_us_client = BinanceUsClient(timeout_seconds=10.0)

    gate_mode = str(predictability_gate_mode)
    gate_enabled = bool(cfg.predictability_gate_enabled)
    if gate_mode == "off":
        gate_enabled = False
    elif gate_mode == "on":
        gate_enabled = True
    elif gate_mode != "config":
        raise ValueError("predictability_gate_mode_invalid")

    gate_threshold = (
        float(cfg.predictability_gate_threshold)
        if predictability_gate_threshold_override is None
        else float(predictability_gate_threshold_override)
    )
    if not (0.0 <= float(gate_threshold) <= 1.0):
        raise ValueError("predictability_gate_threshold_out_of_range")

    gate_baseline_bet = (
        float(cfg.predictability_baseline_bet_bnb)
        if predictability_baseline_bet_bnb_override is None
        else float(predictability_baseline_bet_bnb_override)
    )
    if (not math.isfinite(float(gate_baseline_bet))) or float(gate_baseline_bet) <= 0.0:
        raise ValueError("predictability_baseline_bet_bnb_invalid")

    runtime_cfg = RuntimeConfig(
        graph_client=None,
        round_store=round_store,
        klines_store=klines_store,
        binance_us_client=binance_us_client,
        binance_us_symbol=_BINANCE_US_SYMBOL,
        contract=None,
        wallet_address="",
        cutoff_seconds=cfg.cutoff_seconds,
        train_size=int(train_size),
        retrain_interval=cfg.retrain_interval,
        calibrate_size=int(calibrate_size),
        recalibrate_interval=cfg.recalibrate_interval,
        recency_weight_floor=float(rw_floor),
        recency_weight_power=float(rw_power),
        use_onchain_event_bets=False,
        event_lookback_blocks=cfg.event_lookback_blocks,
        event_freshness_slack_seconds=cfg.event_freshness_slack_seconds,
        latency_log_path=cfg.latency_log_path,
        wait_for_bet_receipt=False,
        bet_receipt_timeout_seconds=cfg.bet_receipt_timeout_seconds,
        predictability_gate_enabled=bool(gate_enabled),
        predictability_gate_threshold=float(gate_threshold),
        predictability_baseline_bet_bnb=float(gate_baseline_bet),
        policy_cfg=cfg.policy,
        strategy_cfg=cfg.strategy,
        dry=False,
        treasury_fee_fraction=float(constants.treasury_fee_fraction),
        buffer_seconds=int(constants.buffer_seconds),
        min_bet_amount_bnb=float(constants.min_bet_amount_bnb),
        price_alpha=cfg.price_alpha,
        pool_alpha_total=cfg.pool_alpha_total,
        pool_alpha_ratio=cfg.pool_alpha_ratio,
        random_seed=cfg.random_seed,
    )
    return cfg, runtime_cfg


def _p_stats(trades_csv: Path) -> dict[str, float | int]:
    with trades_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    p = [float(r["p_final"]) for r in rows]
    if not p:
        return {
            "p_min": float("nan"),
            "p_max": float("nan"),
            "p_mean": float("nan"),
            "p_q10": float("nan"),
            "p_q50": float("nan"),
            "p_q90": float("nan"),
            "p_spread_q90_q10": float("nan"),
            "p_unique_rounded_10dp": 0,
        }
    p_sorted = sorted(p)
    unique = len({round(x, 10) for x in p})

    def q(v: float) -> float:
        idx = min(len(p_sorted) - 1, max(0, int(round((len(p_sorted) - 1) * v))))
        return float(p_sorted[idx])

    return {
        "p_min": float(min(p)),
        "p_max": float(max(p)),
        "p_mean": float(sum(p) / len(p)),
        "p_q10": q(0.10),
        "p_q50": q(0.50),
        "p_q90": q(0.90),
        "p_spread_q90_q10": float(q(0.90) - q(0.10)),
        "p_unique_rounded_10dp": int(unique),
    }


def _quantile_from_sorted(values_sorted: list[float], q: float) -> float:
    if not values_sorted:
        raise ValueError("quantile_values_empty")
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError("quantile_q_out_of_range")
    idx = min(len(values_sorted) - 1, max(0, int(round((len(values_sorted) - 1) * float(q)))))
    return float(values_sorted[idx])


def _pool_imbalance(*, final_bull_bnb: float, final_bear_bnb: float, final_total_bnb: float) -> float:
    total = float(final_total_bnb)
    if not math.isfinite(total) or total <= 0.0:
        return 0.0
    bull = float(final_bull_bnb)
    bear = float(final_bear_bnb)
    if not math.isfinite(bull) or not math.isfinite(bear):
        return 0.0
    return abs(float(bull) - float(bear)) / float(total)


def _pool_imbalance_from_trade_row(row: dict[str, str]) -> float:
    return _pool_imbalance(
        final_bull_bnb=float(row["final_bull_bnb"]),
        final_bear_bnb=float(row["final_bear_bnb"]),
        final_total_bnb=float(row["final_total_bnb"]),
    )


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    n = int(len(xs))
    if n != len(ys) or n < 2:
        return float("nan")
    mx = float(sum(xs) / n)
    my = float(sum(ys) / n)
    cov = 0.0
    vx = 0.0
    vy = 0.0
    for x, y in zip(xs, ys):
        dx = float(x) - float(mx)
        dy = float(y) - float(my)
        cov += dx * dy
        vx += dx * dx
        vy += dy * dy
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return float(cov / math.sqrt(vx * vy))


def _corr_with_guard(*, xs: list[float], ys: list[float], min_samples: int) -> dict[str, Any]:
    if int(min_samples) < 2:
        raise ValueError("corr_min_samples_must_be_at_least_2")
    if len(xs) != len(ys):
        raise ValueError("corr_xy_len_mismatch")
    n = int(len(xs))
    if n < int(min_samples):
        return {
            "value": None,
            "status": str(_CORR_STATUS_INSUFFICIENT),
            "sample_count": int(n),
            "min_samples": int(min_samples),
        }
    v = float(_pearson_corr(xs, ys))
    if not math.isfinite(v):
        return {
            "value": None,
            "status": str(_CORR_STATUS_DEGENERATE),
            "sample_count": int(n),
            "min_samples": int(min_samples),
        }
    return {
        "value": float(v),
        "status": str(_CORR_STATUS_OK),
        "sample_count": int(n),
        "min_samples": int(min_samples),
    }


def _max_drawdown_bnb(trades_csv: Path) -> float:
    with trades_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0
    peak = float("-inf")
    max_dd = 0.0
    for row in rows:
        bankroll = float(row["bankroll_bnb"])
        if bankroll > peak:
            peak = float(bankroll)
        dd = float(peak - bankroll)
        if dd > max_dd:
            max_dd = float(dd)
    return float(max_dd)


def _chunk_stability(
    *,
    trades_csv: Path,
    chunks: int,
    min_bets_per_chunk: int,
    max_dominance_share: float,
    min_positive_chunk_fraction: float,
) -> dict[str, Any]:
    if int(chunks) <= 0:
        raise ValueError("chunk_stability_chunks_must_be_positive")
    if int(min_bets_per_chunk) < 0:
        raise ValueError("chunk_stability_min_bets_per_chunk_negative")
    if not (0.0 <= float(max_dominance_share) <= 1.0):
        raise ValueError("chunk_stability_max_dominance_share_out_of_range")
    if not (0.0 <= float(min_positive_chunk_fraction) <= 1.0):
        raise ValueError("chunk_stability_min_positive_fraction_out_of_range")

    with trades_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    n_rows = int(len(rows))
    if n_rows <= 0:
        return {
            "enabled": True,
            "num_chunks_requested": int(chunks),
            "num_chunks_actual": 0,
            "max_abs_profit_share": 0.0,
            "dominance_ok": True,
            "chunks_with_bets": 0,
            "chunks_meeting_min_bets": 0,
            "all_bet_chunks_meet_min_bets": True,
            "positive_chunk_fraction": 0.0,
            "positive_chunk_fraction_ok": True,
            "stability_ok": True,
            "chunks": [],
        }

    actual_chunks = min(int(chunks), int(n_rows))
    chunk_rows: list[dict[str, Any]] = []
    for idx in range(actual_chunks):
        lo = (idx * n_rows) // int(actual_chunks)
        hi = ((idx + 1) * n_rows) // int(actual_chunks)
        if hi <= lo:
            continue
        rows_c = rows[lo:hi]
        bet_rows_c = [r for r in rows_c if str(r.get("action", "")).strip().upper() == "BET"]
        profits_c = [float(r["profit_bnb"]) for r in rows_c]
        wins_c = int(sum(1 for r in bet_rows_c if float(r["profit_bnb"]) > 0.0))
        chunk_rows.append(
            {
                "chunk_index": int(idx + 1),
                "num_rounds": int(len(rows_c)),
                "num_bets": int(len(bet_rows_c)),
                "net_profit_bnb": float(sum(profits_c)),
                "win_rate": float(wins_c / len(bet_rows_c)) if bet_rows_c else float("nan"),
                "start_epoch": int(rows_c[0]["epoch"]),
                "end_epoch": int(rows_c[-1]["epoch"]),
            }
        )

    if not chunk_rows:
        raise InvariantError("chunk_stability_empty_after_partition")

    total_profit = float(sum(float(c["net_profit_bnb"]) for c in chunk_rows))
    abs_total_profit = abs(float(total_profit))
    max_abs_profit_share = 0.0
    if abs_total_profit > 0.0:
        max_abs_profit_share = max(
            float(abs(float(c["net_profit_bnb"])) / abs_total_profit) for c in chunk_rows
        )
    dominance_ok = bool(abs_total_profit <= 0.0 or float(max_abs_profit_share) <= float(max_dominance_share))

    chunks_with_bets = int(sum(1 for c in chunk_rows if int(c["num_bets"]) > 0))
    chunks_meeting_min_bets = int(
        sum(
            1
            for c in chunk_rows
            if int(c["num_bets"]) > 0 and int(c["num_bets"]) >= int(min_bets_per_chunk)
        )
    )
    all_bet_chunks_meet_min_bets = bool(
        chunks_with_bets == 0 or chunks_meeting_min_bets == chunks_with_bets
    )
    positive_chunks = int(sum(1 for c in chunk_rows if float(c["net_profit_bnb"]) > 0.0))
    positive_chunk_fraction = float(positive_chunks / len(chunk_rows))
    positive_chunk_fraction_ok = bool(
        float(positive_chunk_fraction) >= float(min_positive_chunk_fraction)
    )

    return {
        "enabled": True,
        "num_chunks_requested": int(chunks),
        "num_chunks_actual": int(len(chunk_rows)),
        "max_abs_profit_share": float(max_abs_profit_share),
        "max_dominance_share_limit": float(max_dominance_share),
        "dominance_ok": bool(dominance_ok),
        "min_bets_per_chunk": int(min_bets_per_chunk),
        "chunks_with_bets": int(chunks_with_bets),
        "chunks_meeting_min_bets": int(chunks_meeting_min_bets),
        "all_bet_chunks_meet_min_bets": bool(all_bet_chunks_meet_min_bets),
        "positive_chunk_fraction": float(positive_chunk_fraction),
        "min_positive_chunk_fraction": float(min_positive_chunk_fraction),
        "positive_chunk_fraction_ok": bool(positive_chunk_fraction_ok),
        "stability_ok": bool(dominance_ok and all_bet_chunks_meet_min_bets and positive_chunk_fraction_ok),
        "chunks": chunk_rows,
    }


def _canonical_skip_reason_counts(raw_skip_counts: dict[str, int]) -> dict[str, int]:
    out = {
        "no_positive_ev": 0,
        "direction_filter_no_signal": 0,
        "direction_filter_side_mismatch": 0,
        "pool_imbalance": 0,
        "insufficient_edge": 0,
        "other": 0,
    }
    for raw_reason, raw_count in raw_skip_counts.items():
        reason = str(raw_reason)
        count = int(raw_count)
        if reason == "no_positive_ev":
            out["no_positive_ev"] += count
            continue
        if reason in ("direction_filter_no_signal", "direction_filter_no_bull_signal", "direction_filter_no_bear_signal"):
            out["direction_filter_no_signal"] += count
            continue
        if reason == "direction_filter_side_mismatch":
            out["direction_filter_side_mismatch"] += count
            continue
        if reason == "insufficient_edge":
            out["insufficient_edge"] += count
            continue
        if "pool_imbalance" in reason:
            out["pool_imbalance"] += count
            continue
        out["other"] += count
    return out


def _direction_viability(
    *,
    direction_gate: dict[str, Any],
    p_stats: dict[str, float | int],
    min_expected_signals: int,
    min_expected_signal_rate: float,
) -> dict[str, Any]:
    enabled = bool(direction_gate.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "actionable": True,
            "reasons": [],
        }

    rounds_seen = int(direction_gate.get("rounds_seen", 0))
    expected_signals = int(direction_gate.get("expected_signal_rounds", 0))
    realized_signals = int(direction_gate.get("realized_signal_bets", 0))
    expected_rate = float(expected_signals / rounds_seen) if rounds_seen > 0 else 0.0
    realized_rate = float(realized_signals / rounds_seen) if rounds_seen > 0 else 0.0
    p_spread = float(p_stats.get("p_spread_q90_q10", float("nan")))
    expected_bull = int(direction_gate.get("expected_bull_signals", 0))
    expected_bear = int(direction_gate.get("expected_bear_signals", 0))
    both_side_coverage = bool(expected_bull > 0 and expected_bear > 0)

    reasons: list[str] = []
    if expected_signals < int(min_expected_signals):
        reasons.append("expected_signal_count_below_min")
    if expected_rate < float(min_expected_signal_rate):
        reasons.append("expected_signal_rate_below_min")
    if realized_signals <= 0:
        reasons.append("no_realized_signal_bets")

    return {
        "enabled": True,
        "rounds_seen": int(rounds_seen),
        "p_spread_q90_q10": float(p_spread),
        "expected_signal_count": int(expected_signals),
        "expected_signal_rate": float(expected_rate),
        "realized_signal_count": int(realized_signals),
        "realized_signal_rate": float(realized_rate),
        "expected_bull_signals": int(expected_bull),
        "expected_bear_signals": int(expected_bear),
        "both_side_coverage": bool(both_side_coverage),
        "min_expected_signals": int(min_expected_signals),
        "min_expected_signal_rate": float(min_expected_signal_rate),
        "actionable": bool(not reasons),
        "reasons": list(reasons),
    }


def _promotion_assessment(
    *,
    summary: dict[str, Any],
    viability: dict[str, Any],
    stability: dict[str, Any],
    max_drawdown_bnb: float,
    max_allowed_drawdown_bnb: float | None,
    require_both_side_coverage: bool,
) -> dict[str, Any]:
    checks = {
        "positive_net_profit": bool(float(summary.get("net_profit_bnb", 0.0)) > 0.0),
        "viability_actionable": bool(viability.get("actionable", True)),
        "stability_ok": bool(stability.get("stability_ok", True)),
    }
    if bool(require_both_side_coverage):
        checks["both_side_coverage"] = bool(viability.get("both_side_coverage", True))
    if max_allowed_drawdown_bnb is not None:
        checks["max_drawdown_within_limit"] = bool(float(max_drawdown_bnb) <= float(max_allowed_drawdown_bnb))
    failing = [str(k) for k, ok in checks.items() if not bool(ok)]
    return {
        "checks": checks,
        "candidate": bool(not failing),
        "failing_checks": failing,
    }


def _bet_deciles(*, rows: list[dict[str, str]], key: str, buckets: int = 10) -> list[dict[str, float | int]]:
    if int(buckets) <= 0:
        raise ValueError("buckets_must_be_positive")
    n = int(len(rows))
    if n <= 0:
        return []

    ordered = sorted(rows, key=lambda r: float(r[key]))
    out: list[dict[str, float | int]] = []
    for idx in range(int(buckets)):
        lo = (idx * n) // int(buckets)
        hi = ((idx + 1) * n) // int(buckets)
        if hi <= lo:
            continue
        bucket_rows = ordered[lo:hi]
        vals = [float(r[key]) for r in bucket_rows]
        profits = [float(r["profit_bnb"]) for r in bucket_rows]
        evs = [float(r["ev_bnb"]) for r in bucket_rows]
        wins = int(sum(1 for p in profits if float(p) > 0.0))
        out.append(
            {
                "decile": int(idx + 1),
                "n": int(len(bucket_rows)),
                f"{key}_min": float(min(vals)),
                f"{key}_max": float(max(vals)),
                f"{key}_mean": float(sum(vals) / len(vals)),
                "ev_mean_bnb": float(sum(evs) / len(evs)),
                "profit_mean_bnb": float(sum(profits) / len(profits)),
                "win_rate": float(wins / len(bucket_rows)),
            }
        )
    return out


def _bet_diagnostics(trades_csv: Path, *, corr_min_samples: int) -> dict[str, Any]:
    if int(corr_min_samples) < 2:
        raise ValueError("corr_min_samples_must_be_at_least_2")
    with trades_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    bet_rows = [r for r in rows if str(r.get("action", "")).strip().upper() == "BET"]
    if not bet_rows:
        return {
            "num_bets_analyzed": 0,
            "corr_min_samples": int(corr_min_samples),
            "ev_deciles": [],
            "p_final_deciles": [],
            "by_side": {},
            "by_pool_imbalance_regime": {},
        }

    profits = [float(r["profit_bnb"]) for r in bet_rows]
    evs = [float(r["ev_bnb"]) for r in bet_rows]
    edges = [abs(float(r["p_final"]) - 0.5) for r in bet_rows]
    profitable = [1.0 if float(p) > 0.0 else 0.0 for p in profits]
    wins = int(sum(1 for p in profits if float(p) > 0.0))
    corr_ev = _corr_with_guard(xs=evs, ys=profits, min_samples=int(corr_min_samples))
    corr_edge = _corr_with_guard(xs=edges, ys=profitable, min_samples=int(corr_min_samples))
    out: dict[str, Any] = {
        "num_bets_analyzed": int(len(bet_rows)),
        "corr_min_samples": int(corr_min_samples),
        "overall_profit_mean_bnb": float(sum(profits) / len(profits)),
        "overall_win_rate": float(wins / len(profits)),
        "overall_ev_profit_corr": corr_ev["value"],
        "overall_ev_profit_corr_status": str(corr_ev["status"]),
        "overall_ev_profit_corr_n": int(corr_ev["sample_count"]),
        "overall_edge_win_corr": corr_edge["value"],
        "overall_edge_win_corr_status": str(corr_edge["status"]),
        "overall_edge_win_corr_n": int(corr_edge["sample_count"]),
        "ev_deciles": _bet_deciles(rows=bet_rows, key="ev_bnb", buckets=10),
        "p_final_deciles": _bet_deciles(rows=bet_rows, key="p_final", buckets=10),
    }

    by_side: dict[str, Any] = {}
    for side in ("Bull", "Bear"):
        side_rows = [r for r in bet_rows if str(r.get("direction", "")).strip() == side]
        if not side_rows:
            by_side[side] = {
                "num_bets": 0,
                "ev_profit_corr": None,
                "ev_profit_corr_status": str(_CORR_STATUS_INSUFFICIENT),
                "ev_profit_corr_n": 0,
                "edge_win_corr": None,
                "edge_win_corr_status": str(_CORR_STATUS_INSUFFICIENT),
                "edge_win_corr_n": 0,
                "ev_deciles": [],
                "p_final_deciles": [],
            }
            continue
        side_profits = [float(r["profit_bnb"]) for r in side_rows]
        side_evs = [float(r["ev_bnb"]) for r in side_rows]
        side_edges = [abs(float(r["p_final"]) - 0.5) for r in side_rows]
        side_profitable = [1.0 if float(p) > 0.0 else 0.0 for p in side_profits]
        side_corr_ev = _corr_with_guard(xs=side_evs, ys=side_profits, min_samples=int(corr_min_samples))
        side_corr_edge = _corr_with_guard(
            xs=side_edges,
            ys=side_profitable,
            min_samples=int(corr_min_samples),
        )
        side_wins = int(sum(1 for p in side_profits if float(p) > 0.0))
        by_side[side] = {
            "num_bets": int(len(side_rows)),
            "win_rate": float(side_wins / len(side_rows)),
            "profit_mean_bnb": float(sum(side_profits) / len(side_profits)),
            "ev_profit_corr": side_corr_ev["value"],
            "ev_profit_corr_status": str(side_corr_ev["status"]),
            "ev_profit_corr_n": int(side_corr_ev["sample_count"]),
            "edge_win_corr": side_corr_edge["value"],
            "edge_win_corr_status": str(side_corr_edge["status"]),
            "edge_win_corr_n": int(side_corr_edge["sample_count"]),
            "ev_deciles": _bet_deciles(rows=side_rows, key="ev_bnb", buckets=10),
            "p_final_deciles": _bet_deciles(rows=side_rows, key="p_final", buckets=10),
        }
    out["by_side"] = by_side

    regime_rows: dict[str, list[dict[str, str]]] = {"low": [], "mid": [], "high": []}
    for r in bet_rows:
        imb = _pool_imbalance_from_trade_row(r)
        if float(imb) < 0.10:
            regime_rows["low"].append(r)
        elif float(imb) < 0.20:
            regime_rows["mid"].append(r)
        else:
            regime_rows["high"].append(r)

    by_regime: dict[str, Any] = {
        "thresholds": {"low_lt": 0.10, "mid_lt": 0.20, "high_gte": 0.20},
        "buckets": {},
    }
    for bucket_name in ("low", "mid", "high"):
        rows_b = regime_rows[bucket_name]
        if not rows_b:
            by_regime["buckets"][bucket_name] = {
                "num_bets": 0,
                "ev_profit_corr": None,
                "ev_profit_corr_status": str(_CORR_STATUS_INSUFFICIENT),
                "ev_profit_corr_n": 0,
                "edge_win_corr": None,
                "edge_win_corr_status": str(_CORR_STATUS_INSUFFICIENT),
                "edge_win_corr_n": 0,
                "ev_deciles": [],
                "p_final_deciles": [],
            }
            continue
        p_b = [float(r["profit_bnb"]) for r in rows_b]
        e_b = [float(r["ev_bnb"]) for r in rows_b]
        edge_b = [abs(float(r["p_final"]) - 0.5) for r in rows_b]
        profitable_b = [1.0 if float(p) > 0.0 else 0.0 for p in p_b]
        corr_ev_b = _corr_with_guard(xs=e_b, ys=p_b, min_samples=int(corr_min_samples))
        corr_edge_b = _corr_with_guard(xs=edge_b, ys=profitable_b, min_samples=int(corr_min_samples))
        w_b = int(sum(1 for p in p_b if float(p) > 0.0))
        by_regime["buckets"][bucket_name] = {
            "num_bets": int(len(rows_b)),
            "win_rate": float(w_b / len(rows_b)),
            "profit_mean_bnb": float(sum(p_b) / len(p_b)),
            "ev_profit_corr": corr_ev_b["value"],
            "ev_profit_corr_status": str(corr_ev_b["status"]),
            "ev_profit_corr_n": int(corr_ev_b["sample_count"]),
            "edge_win_corr": corr_edge_b["value"],
            "edge_win_corr_status": str(corr_edge_b["status"]),
            "edge_win_corr_n": int(corr_edge_b["sample_count"]),
            "ev_deciles": _bet_deciles(rows=rows_b, key="ev_bnb", buckets=10),
            "p_final_deciles": _bet_deciles(rows=rows_b, key="p_final", buckets=10),
        }
    out["by_pool_imbalance_regime"] = by_regime
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="var/tmp_config.toml")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--calibrate-size", type=int, required=True)
    parser.add_argument("--rw-floor", type=float, required=True)
    parser.add_argument("--rw-power", type=float, required=True)
    parser.add_argument("--sim-size", type=int, default=None)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    parser.add_argument("--reset-mode", type=str, choices=("continuous", "chunk_reset"), default=None)
    parser.add_argument("--reset-every-rounds", type=int, default=None)
    parser.add_argument("--sim-offset-rounds", type=int, default=0)
    parser.add_argument("--calibration-mode", type=str, choices=("isotonic", "raw", "platt"), default="isotonic")
    parser.add_argument("--direction-model-type", type=str, choices=("lgbm", "logistic"), default="lgbm")
    parser.add_argument(
        "--window-order",
        type=str,
        choices=("cal_train", "train_cal", "train_in_sample"),
        default="cal_train",
    )
    parser.add_argument("--raw-prob", action="store_true", default=False)
    parser.add_argument("--force-no-positive-ev", action="store_true", default=False)
    parser.add_argument("--no-positive-ev-floor-bnb", type=float, default=None)
    parser.add_argument("--fixed-bet-bnb", type=float, default=None)
    parser.add_argument("--fixed-bet-ignore-cap", action="store_true", default=False)
    parser.add_argument(
        "--predictability-gate-mode",
        type=str,
        choices=("config", "off", "on"),
        default="config",
    )
    parser.add_argument("--predictability-gate-threshold", type=float, default=None)
    parser.add_argument("--predictability-baseline-bet-bnb", type=float, default=None)
    parser.add_argument("--zero-feature-groups", type=str, default="")
    parser.add_argument("--winrate-only", action="store_true", default=False)
    parser.add_argument("--winrate-probe-profile", type=str, choices=_WINRATE_PROBE_PROFILES, default="none")
    parser.add_argument("--winrate-probe-columns", type=str, default="")
    parser.add_argument("--sparse-probe-columns", type=str, default="")
    parser.add_argument(
        "--direction-filter-mode",
        type=str,
        choices=("none", "bull_only", "bear_only", "both_sides", "adaptive_side"),
        default="none",
    )
    parser.add_argument(
        "--direction-threshold-mode",
        type=str,
        choices=("fixed", "quantile"),
        default="fixed",
    )
    parser.add_argument("--direction-threshold-bull", type=float, default=0.5)
    parser.add_argument("--direction-threshold-bear", type=float, default=0.5)
    parser.add_argument("--direction-target-bull-rate", type=float, default=0.03)
    parser.add_argument("--direction-target-bear-rate", type=float, default=0.03)
    parser.add_argument("--direction-threshold-window", type=int, default=1000)
    parser.add_argument("--direction-threshold-min-history", type=int, default=100)
    parser.add_argument(
        "--direction-center-mode",
        type=str,
        choices=("fixed_0p5", "rolling_median", "rolling_mean"),
        default="fixed_0p5",
    )
    parser.add_argument("--direction-center-window", type=int, default=1000)
    parser.add_argument("--direction-edge-floor-pp", type=float, default=0.0)
    parser.add_argument("--direction-edge-floor-ratio", type=float, default=None)
    parser.add_argument("--direction-edge-floor-quantile", type=float, default=None)
    parser.add_argument("--direction-edge-window", type=int, default=1000)
    parser.add_argument("--direction-adaptive-window", type=int, default=300)
    parser.add_argument("--direction-adaptive-min-history", type=int, default=40)
    parser.add_argument("--direction-adaptive-switch-margin-bnb", type=float, default=0.0)
    parser.add_argument(
        "--direction-adaptive-score",
        type=str,
        choices=("mean_profit_per_bet", "win_rate"),
        default="mean_profit_per_bet",
    )
    parser.add_argument(
        "--direction-adaptive-default-side",
        type=str,
        choices=("signal", "bull", "bear"),
        default="signal",
    )
    parser.add_argument("--direction-adaptive-allow-signal-fallback", action="store_true", default=False)
    parser.add_argument("--direction-adaptive-counterfactual", action="store_true", default=False)
    parser.add_argument("--ev-reliability-window", type=int, default=0)
    parser.add_argument("--ev-reliability-min-bets", type=int, default=20)
    parser.add_argument("--ev-reliability-quantile", type=float, default=0.70)
    parser.add_argument("--ev-reliability-min-mean-profit", type=float, default=0.0)
    parser.add_argument("--regime-filter", type=str, choices=("none", "pool_imbalance"), default="none")
    parser.add_argument("--regime-min-imbalance", type=float, default=0.12)
    parser.add_argument("--corr-min-samples", type=int, default=30)
    parser.add_argument("--direction-viability-min-expected-signals", type=int, default=10)
    parser.add_argument("--direction-viability-min-expected-rate", type=float, default=0.01)
    parser.add_argument("--direction-viability-hard-fail", action="store_true", default=False)
    parser.add_argument("--chunk-stability-chunks", type=int, default=10)
    parser.add_argument("--chunk-stability-min-bets-per-chunk", type=int, default=0)
    parser.add_argument("--chunk-stability-max-dominance-share", type=float, default=0.65)
    parser.add_argument("--chunk-stability-min-positive-fraction", type=float, default=0.40)
    parser.add_argument("--promotion-max-drawdown-bnb", type=float, default=None)
    parser.add_argument("--promotion-require-both-side-coverage", action="store_true", default=False)
    args = parser.parse_args()

    if bool(args.raw_prob) and str(args.calibration_mode) != "isotonic":
        raise ValueError("raw_prob_flag_conflicts_with_calibration_mode")

    calibration_mode = "raw" if bool(args.raw_prob) else str(args.calibration_mode)
    direction_model_type = str(args.direction_model_type)
    window_order = str(args.window_order)
    zero_feature_groups = [g.strip() for g in str(args.zero_feature_groups).split(",") if g.strip()]
    ev_reliability_window = int(args.ev_reliability_window)
    ev_reliability_min_bets = int(args.ev_reliability_min_bets)
    ev_reliability_quantile = float(args.ev_reliability_quantile)
    ev_reliability_min_mean_profit = float(args.ev_reliability_min_mean_profit)
    regime_filter = str(args.regime_filter)
    regime_min_imbalance = float(args.regime_min_imbalance)
    corr_min_samples = int(args.corr_min_samples)
    direction_viability_min_expected_signals = int(args.direction_viability_min_expected_signals)
    direction_viability_min_expected_rate = float(args.direction_viability_min_expected_rate)
    chunk_stability_chunks = int(args.chunk_stability_chunks)
    chunk_stability_min_bets_per_chunk = int(args.chunk_stability_min_bets_per_chunk)
    chunk_stability_max_dominance_share = float(args.chunk_stability_max_dominance_share)
    chunk_stability_min_positive_fraction = float(args.chunk_stability_min_positive_fraction)
    promotion_max_drawdown_bnb = (
        None if args.promotion_max_drawdown_bnb is None else float(args.promotion_max_drawdown_bnb)
    )
    no_positive_ev_floor_bnb = (
        None if args.no_positive_ev_floor_bnb is None else float(args.no_positive_ev_floor_bnb)
    )
    fixed_bet_bnb = None if args.fixed_bet_bnb is None else float(args.fixed_bet_bnb)
    fixed_bet_ignore_cap = bool(args.fixed_bet_ignore_cap)
    predictability_gate_mode = str(args.predictability_gate_mode)
    predictability_gate_threshold_override = (
        None if args.predictability_gate_threshold is None else float(args.predictability_gate_threshold)
    )
    predictability_baseline_bet_bnb_override = (
        None if args.predictability_baseline_bet_bnb is None else float(args.predictability_baseline_bet_bnb)
    )
    winrate_probe_profile = str(args.winrate_probe_profile)
    winrate_probe_columns = [c.strip() for c in str(args.winrate_probe_columns).split(",") if c.strip()]
    sparse_probe_columns = [c.strip() for c in str(args.sparse_probe_columns).split(",") if c.strip()]
    direction_filter_mode = str(args.direction_filter_mode)
    direction_threshold_mode = str(args.direction_threshold_mode)
    direction_threshold_bull = float(args.direction_threshold_bull)
    direction_threshold_bear = float(args.direction_threshold_bear)
    direction_target_bull_rate = float(args.direction_target_bull_rate)
    direction_target_bear_rate = float(args.direction_target_bear_rate)
    direction_threshold_window = int(args.direction_threshold_window)
    direction_threshold_min_history = int(args.direction_threshold_min_history)
    direction_center_mode = str(args.direction_center_mode)
    direction_center_window = int(args.direction_center_window)
    direction_edge_floor_pp = float(args.direction_edge_floor_pp)
    direction_edge_floor_ratio = (
        None if args.direction_edge_floor_ratio is None else float(args.direction_edge_floor_ratio)
    )
    direction_edge_floor_quantile = (
        None if args.direction_edge_floor_quantile is None else float(args.direction_edge_floor_quantile)
    )
    direction_edge_window = int(args.direction_edge_window)
    direction_adaptive_window = int(args.direction_adaptive_window)
    direction_adaptive_min_history = int(args.direction_adaptive_min_history)
    direction_adaptive_switch_margin_bnb = float(args.direction_adaptive_switch_margin_bnb)
    direction_adaptive_score = str(args.direction_adaptive_score)
    direction_adaptive_default_side = str(args.direction_adaptive_default_side)
    direction_adaptive_allow_signal_fallback = bool(args.direction_adaptive_allow_signal_fallback)
    direction_adaptive_counterfactual = bool(args.direction_adaptive_counterfactual)
    reset_mode_override = None if args.reset_mode is None else str(args.reset_mode)
    reset_every_rounds_override = None if args.reset_every_rounds is None else int(args.reset_every_rounds)
    sim_offset_rounds = int(args.sim_offset_rounds)

    if ev_reliability_window < 0:
        raise ValueError("ev_reliability_window_must_be_nonnegative")
    if ev_reliability_min_bets < 1:
        raise ValueError("ev_reliability_min_bets_must_be_positive")
    if not (0.0 <= ev_reliability_quantile <= 1.0):
        raise ValueError("ev_reliability_quantile_out_of_range")
    if regime_filter == "pool_imbalance" and not (0.0 <= regime_min_imbalance <= 1.0):
        raise ValueError("regime_min_imbalance_out_of_range")
    if corr_min_samples < 2:
        raise ValueError("corr_min_samples_must_be_at_least_2")
    if direction_viability_min_expected_signals < 0:
        raise ValueError("direction_viability_min_expected_signals_negative")
    if not (0.0 <= direction_viability_min_expected_rate <= 1.0):
        raise ValueError("direction_viability_min_expected_rate_out_of_range")
    if chunk_stability_chunks <= 0:
        raise ValueError("chunk_stability_chunks_must_be_positive")
    if chunk_stability_min_bets_per_chunk < 0:
        raise ValueError("chunk_stability_min_bets_per_chunk_negative")
    if not (0.0 <= chunk_stability_max_dominance_share <= 1.0):
        raise ValueError("chunk_stability_max_dominance_share_out_of_range")
    if not (0.0 <= chunk_stability_min_positive_fraction <= 1.0):
        raise ValueError("chunk_stability_min_positive_fraction_out_of_range")
    if promotion_max_drawdown_bnb is not None and (
        not math.isfinite(float(promotion_max_drawdown_bnb)) or float(promotion_max_drawdown_bnb) < 0.0
    ):
        raise ValueError("promotion_max_drawdown_bnb_invalid")
    if no_positive_ev_floor_bnb is not None and not math.isfinite(no_positive_ev_floor_bnb):
        raise ValueError("no_positive_ev_floor_bnb_non_finite")
    if fixed_bet_bnb is not None and (not math.isfinite(float(fixed_bet_bnb)) or float(fixed_bet_bnb) <= 0.0):
        raise ValueError("fixed_bet_bnb_must_be_positive_finite")
    if predictability_gate_threshold_override is not None and (
        not (0.0 <= float(predictability_gate_threshold_override) <= 1.0)
    ):
        raise ValueError("predictability_gate_threshold_out_of_range")
    if predictability_baseline_bet_bnb_override is not None and (
        (not math.isfinite(float(predictability_baseline_bet_bnb_override)))
        or float(predictability_baseline_bet_bnb_override) <= 0.0
    ):
        raise ValueError("predictability_baseline_bet_bnb_invalid")
    if bool(args.winrate_only) and bool(args.force_no_positive_ev):
        raise ValueError("winrate_only_conflicts_with_force_no_positive_ev")
    if bool(args.winrate_only) and no_positive_ev_floor_bnb is not None:
        raise ValueError("winrate_only_conflicts_with_no_positive_ev_floor")
    if winrate_probe_profile != "none" and zero_feature_groups:
        raise ValueError("winrate_probe_profile_conflicts_with_zero_feature_groups")
    if winrate_probe_columns and zero_feature_groups:
        raise ValueError("winrate_probe_columns_conflicts_with_zero_feature_groups")
    if sparse_probe_columns and zero_feature_groups:
        raise ValueError("sparse_probe_columns_conflicts_with_zero_feature_groups")
    if winrate_probe_profile != "none" and winrate_probe_columns:
        raise ValueError("winrate_probe_profile_conflicts_with_winrate_probe_columns")
    if winrate_probe_profile != "none" and sparse_probe_columns:
        raise ValueError("winrate_probe_profile_conflicts_with_sparse_probe_columns")
    if sparse_probe_columns and winrate_probe_columns:
        sparse_set = set(str(c) for c in sparse_probe_columns)
        winrate_set = set(str(c) for c in winrate_probe_columns)
        missing = sorted(winrate_set - sparse_set)
        if missing:
            raise ValueError(
                "winrate_probe_columns_not_subset_of_sparse_probe_columns: "
                + ",".join(missing)
            )
    if not (0.0 <= direction_threshold_bull <= 1.0):
        raise ValueError("direction_threshold_bull_out_of_range")
    if not (0.0 <= direction_threshold_bear <= 1.0):
        raise ValueError("direction_threshold_bear_out_of_range")
    if not (0.0 <= direction_target_bull_rate <= 1.0):
        raise ValueError("direction_target_bull_rate_out_of_range")
    if not (0.0 <= direction_target_bear_rate <= 1.0):
        raise ValueError("direction_target_bear_rate_out_of_range")
    if direction_threshold_window <= 0:
        raise ValueError("direction_threshold_window_must_be_positive")
    if direction_threshold_min_history <= 0:
        raise ValueError("direction_threshold_min_history_must_be_positive")
    if direction_center_window <= 0:
        raise ValueError("direction_center_window_must_be_positive")
    if not (0.0 <= direction_edge_floor_pp < 0.5):
        raise ValueError("direction_edge_floor_pp_out_of_range")
    if direction_edge_floor_ratio is not None and direction_edge_floor_ratio < 0.0:
        raise ValueError("direction_edge_floor_ratio_negative")
    if direction_edge_floor_quantile is not None and not (0.0 <= direction_edge_floor_quantile <= 1.0):
        raise ValueError("direction_edge_floor_quantile_out_of_range")
    if direction_edge_window <= 0:
        raise ValueError("direction_edge_window_must_be_positive")
    if direction_adaptive_window <= 0:
        raise ValueError("direction_adaptive_window_must_be_positive")
    if direction_adaptive_min_history <= 0:
        raise ValueError("direction_adaptive_min_history_must_be_positive")
    if (not math.isfinite(float(direction_adaptive_switch_margin_bnb))) or float(direction_adaptive_switch_margin_bnb) < 0.0:
        raise ValueError("direction_adaptive_switch_margin_bnb_invalid")
    if reset_every_rounds_override is not None and int(reset_every_rounds_override) <= 0:
        raise ValueError("reset_every_rounds_must_be_positive")
    if sim_offset_rounds < 0:
        raise ValueError("sim_offset_rounds_must_be_nonnegative")
    if args.initial_bankroll_bnb is not None and float(args.initial_bankroll_bnb) <= 0.0:
        raise ValueError("initial_bankroll_bnb_must_be_positive")

    cfg, runtime_cfg = _build_runtime_cfg(
        config_path=str(args.config),
        train_size=int(args.train_size),
        calibrate_size=int(args.calibrate_size),
        rw_floor=float(args.rw_floor),
        rw_power=float(args.rw_power),
        predictability_gate_mode=str(predictability_gate_mode),
        predictability_gate_threshold_override=predictability_gate_threshold_override,
        predictability_baseline_bet_bnb_override=predictability_baseline_bet_bnb_override,
    )
    if fixed_bet_bnb is not None and float(fixed_bet_bnb) < float(runtime_cfg.min_bet_amount_bnb):
        raise ValueError("fixed_bet_bnb_below_min_bet_amount")

    out_dir = Path("var/exp") / str(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_mode_applied = False
    orig_fit_final_calibrator = None
    window_order_applied = False
    orig_train_and_maybe_calibrate = None
    feature_profile_applied = False
    feature_profile_slots: dict[str, str] = {}
    orig_feature_builder_build_features = None
    orig_walk_forward_build_features = None
    orig_planner_build_features = None
    feature_group_zeroing_applied = False
    orig_feature_builder_vectorize = None
    orig_walk_forward_vectorize = None
    orig_planner_vectorize = None
    winrate_only_sizing_applied = False
    orig_winrate_size_bet = None
    orig_winrate_backtest_runner_size_bet = None
    direction_filter_applied = False
    orig_direction_filter_size_bet = None
    orig_direction_filter_backtest_runner_size_bet = None
    orig_direction_filter_settle = None
    force_no_positive_ev_applied = False
    orig_size_bet = None
    orig_backtest_runner_size_bet = None
    fixed_bet_sizing_applied = False
    orig_fixed_bet_size_bet = None
    orig_fixed_bet_backtest_runner_size_bet = None
    decision_gate_applied = False
    orig_gate_size_bet = None
    orig_gate_backtest_size_bet = None
    orig_backtest_settle = None
    sim_offset_applied = False
    orig_backtest_tail_rounds = None
    direction_model_type_applied = False
    orig_direction_model_class = None
    sparse_probe_applied = False
    orig_sparse_schema_feature_schema = None
    orig_sparse_schema_max_prior = None
    orig_sparse_schema_max_klines = None
    orig_sparse_feature_builder_feature_schema = None
    orig_sparse_walk_forward_feature_schema = None
    orig_sparse_planner_feature_schema = None
    orig_sparse_feature_builder_max_prior = None
    orig_sparse_feature_builder_max_klines = None
    orig_sparse_walk_forward_max_prior = None
    orig_sparse_walk_forward_max_klines = None
    orig_sparse_planner_max_prior = None
    orig_sparse_planner_max_klines = None
    orig_sparse_backtest_runner_max_prior = None
    orig_sparse_backtest_runner_max_klines = None
    orig_sparse_feature_builder_vectorize = None
    orig_sparse_walk_forward_vectorize = None
    orig_sparse_planner_vectorize = None
    direction_gate_stats: dict[str, Any] = {"enabled": False}

    if str(direction_model_type) == "logistic":
        import numpy as np
        import warnings
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        import pancakebot.domain.models.walk_forward as walk_forward
        from pancakebot.core.errors import InvariantError

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        orig_direction_model_class = walk_forward.PriceReturnModel

        class _LogisticDirectionModel:
            def __init__(self, *, alpha: float, seed: int):
                if float(alpha) <= 0.0:
                    raise InvariantError("logistic_direction_alpha_nonpositive")
                if int(seed) < 0:
                    raise InvariantError("logistic_direction_seed_negative")
                c_value = max(1e-9, 1.0 / float(alpha))
                self._model = LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    C=float(c_value),
                    random_state=int(seed),
                )
                self._n_features: int | None = None

            @staticmethod
            def _to_2d_array(x) -> np.ndarray:
                arr = np.asarray(x, dtype=float)
                if arr.ndim != 2:
                    raise InvariantError("logistic_direction_x_not_2d")
                return arr

            @staticmethod
            def _sanitize_x(arr: np.ndarray) -> np.ndarray:
                if arr.ndim != 2:
                    raise InvariantError("logistic_direction_x_not_2d")
                # LogisticRegression cannot consume NaN/Inf. Match sparse-probe behavior
                # by replacing non-finite values with zeros.
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            def fit(self, x, y_up, *, x_eval=None, y_eval=None, sample_weight=None) -> None:
                del x_eval, y_eval
                x_arr = self._sanitize_x(self._to_2d_array(x))
                if x_arr.ndim != 2 or x_arr.shape[0] <= 1 or x_arr.shape[1] <= 0:
                    raise InvariantError("logistic_direction_fit_x_shape_invalid")

                y_arr = np.asarray(list(y_up), dtype=int)
                if y_arr.ndim != 1:
                    raise InvariantError("logistic_direction_fit_y_not_1d")
                if len(y_arr) < 2 or len(y_arr) != int(x_arr.shape[0]):
                    raise InvariantError("logistic_direction_fit_requires_at_least_2_rows")
                if not np.all((y_arr == 0) | (y_arr == 1)):
                    raise InvariantError("logistic_direction_fit_y_not_binary")
                pos = int(np.sum(y_arr))
                if pos == 0 or pos == int(len(y_arr)):
                    raise InvariantError("logistic_direction_fit_requires_both_classes")

                sample_weight_arr = None
                if sample_weight is not None:
                    sample_weight_arr = np.asarray(list(sample_weight), dtype=float)
                    if sample_weight_arr.ndim != 1:
                        raise InvariantError("logistic_direction_fit_sample_weight_not_1d")
                    if len(sample_weight_arr) != int(len(y_arr)):
                        raise InvariantError("logistic_direction_fit_sample_weight_len_mismatch")
                    if not np.all(np.isfinite(sample_weight_arr)):
                        raise InvariantError("logistic_direction_fit_sample_weight_non_finite")
                    if np.any(sample_weight_arr <= 0.0):
                        raise InvariantError("logistic_direction_fit_sample_weight_nonpositive")

                self._model.fit(x_arr, y_arr, sample_weight=sample_weight_arr)
                self._n_features = int(x_arr.shape[1])

            def predict(self, x):
                if self._n_features is None:
                    raise InvariantError("logistic_direction_predict_without_fit")

                x_arr = self._sanitize_x(self._to_2d_array(x))
                if x_arr.ndim != 2 or x_arr.shape[0] <= 0:
                    raise InvariantError("logistic_direction_predict_x_shape_invalid")
                if int(x_arr.shape[1]) != int(self._n_features):
                    raise InvariantError("logistic_direction_predict_feature_count_mismatch")

                proba = np.asarray(self._model.predict_proba(x_arr), dtype=float)
                if proba.ndim != 2 or proba.shape[1] < 2:
                    raise InvariantError("logistic_direction_predict_proba_shape_invalid")
                out = proba[:, 1]
                if not np.all(np.isfinite(out)):
                    raise InvariantError("logistic_direction_predict_non_finite")
                return out

        walk_forward.PriceReturnModel = _LogisticDirectionModel
        direction_model_type_applied = True

    if str(calibration_mode) in ("raw", "platt"):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        import pancakebot.domain.models.walk_forward as walk_forward
        from pancakebot.domain.models.calibration import IsotonicCalibrator
        from pancakebot.core.errors import InvariantError

        orig_fit_final_calibrator = walk_forward._fit_final_calibrator

        class _RawIdentityCalibrator:
            def predict_proba_up(self, mu):
                if isinstance(mu, (float, int)):
                    m = float(mu)
                    if m < 0.0:
                        return 0.0
                    if m > 1.0:
                        return 1.0
                    return float(m)
                arr = np.asarray(list(mu), dtype=float)
                arr = np.clip(arr, 0.0, 1.0)
                return arr

        class _PlattCalibrator:
            def __init__(self) -> None:
                self._model = LogisticRegression(solver="lbfgs", max_iter=1000)
                self._fitted = False

            def fit(self, mu, y_up, sample_weight=None) -> None:
                x = np.asarray(list(mu), dtype=float).reshape(-1, 1)
                y = np.asarray(list(y_up), dtype=int)
                if x.shape[0] < 2:
                    raise InvariantError("platt_fit_requires_at_least_2")
                if x.shape[0] != y.shape[0]:
                    raise InvariantError("platt_fit_len_mismatch")
                if not np.all(np.isfinite(x)):
                    raise InvariantError("platt_fit_mu_non_finite")
                if np.any((y != 0) & (y != 1)):
                    raise InvariantError("platt_fit_y_not_binary")
                if len(np.unique(y)) < 2:
                    raise InvariantError("platt_fit_requires_both_classes")

                sw = None
                if sample_weight is not None:
                    sw = np.asarray(list(sample_weight), dtype=float)
                    if sw.shape[0] != y.shape[0]:
                        raise InvariantError("platt_fit_sample_weight_len_mismatch")
                    if not np.all(np.isfinite(sw)):
                        raise InvariantError("platt_fit_sample_weight_non_finite")
                    if np.any(sw <= 0.0):
                        raise InvariantError("platt_fit_sample_weight_nonpositive")

                self._model.fit(x, y, sample_weight=sw)
                self._fitted = True

            def predict_proba_up(self, mu):
                if not self._fitted:
                    raise InvariantError("platt_predict_without_fit")
                if isinstance(mu, (float, int)):
                    x = np.asarray([[float(mu)]], dtype=float)
                    return float(self._model.predict_proba(x)[0, 1])
                x = np.asarray(list(mu), dtype=float).reshape(-1, 1)
                out = self._model.predict_proba(x)[:, 1]
                return out

        def _fit_final_calibrator_override(*, mu_cal, y_up_cal, sample_weight=None):
            if str(calibration_mode) == "raw":
                del y_up_cal, sample_weight
                # Keep sanity checks for non-finite mu inputs.
                _ = [float(v) for v in mu_cal]
                return _RawIdentityCalibrator()
            if str(calibration_mode) == "platt":
                cal = _PlattCalibrator()
                cal.fit(mu_cal, y_up_cal, sample_weight=sample_weight)
                return cal
            cal = IsotonicCalibrator()
            cal.fit(mu_cal, y_up_cal, sample_weight=sample_weight)
            return cal

        walk_forward._fit_final_calibrator = _fit_final_calibrator_override
        calibration_mode_applied = True

    if str(window_order) in ("train_cal", "train_in_sample"):
        import pancakebot.domain.models.walk_forward as walk_forward
        from pancakebot.core.errors import InvariantError

        orig_train_and_maybe_calibrate = walk_forward._train_and_maybe_calibrate
        mode_window_order = str(window_order)

        def _train_and_maybe_calibrate_train_cal(
            *,
            cfg,
            closed_rounds,
            train_size: int,
            calibrate_size: int,
        ):
            k = int(walk_forward.max_required_prior_context_rounds_size())
            if str(mode_window_order) == "train_in_sample":
                required = int(k + int(train_size))
            else:
                required = int(k + int(train_size) + int(calibrate_size))
            if len(closed_rounds) < int(required):
                raise InvariantError("insufficient_closed_rounds_for_walk_forward_train")

            tail = list(closed_rounds[-int(required):])

            (
                x_price_train,
                _y_ret_train,
                y_up_train,
                x_pool_train,
                y_late_inflow_total,
                y_late_inflow_bull_frac,
            ) = walk_forward._build_training_rows(
                cfg=cfg,
                rounds=tail,
                target_begin=int(k),
                target_end=int(k + int(train_size)),
                prior_context_rounds_required=int(k),
            )

            train_sample_weight = walk_forward._build_recency_weights_for_rows(
                cfg=cfg,
                n_rows=int(len(y_up_train)),
            )

            price_model = walk_forward.PriceReturnModel(alpha=float(cfg.price_alpha), seed=int(cfg.random_seed))
            pool_model = walk_forward.FinalPoolModel(
                alpha_total=float(cfg.pool_alpha_total),
                alpha_ratio=float(cfg.pool_alpha_ratio),
                seed=int(cfg.random_seed),
            )

            if int(calibrate_size) <= 0:
                price_model.fit(
                    x_price_train,
                    y_up_train,
                    sample_weight=train_sample_weight,
                )
                pool_model.fit(
                    x_pool_train,
                    y_late_inflow_total,
                    y_late_inflow_bull_frac,
                    sample_weight=train_sample_weight,
                )
                models = walk_forward.WalkForwardModels(price_model=price_model, pool_model=pool_model)
                calibrator_final = walk_forward._fit_final_calibrator(
                    mu_cal=[0.0, 1.0],
                    y_up_cal=[0, 1],
                    sample_weight=[1.0, 1.0],
                )
                return models, calibrator_final

            if str(mode_window_order) == "train_in_sample":
                x_price_cal = x_price_train
                y_up_cal = y_up_train
                x_pool_cal = x_pool_train
                y_late_inflow_total_cal = y_late_inflow_total
                y_late_inflow_bull_frac_cal = y_late_inflow_bull_frac
                cal_sample_weight = train_sample_weight
            else:
                (
                    x_price_cal,
                    _y_ret_cal,
                    y_up_cal,
                    x_pool_cal,
                    y_late_inflow_total_cal,
                    y_late_inflow_bull_frac_cal,
                ) = walk_forward._build_training_rows(
                    cfg=cfg,
                    rounds=tail,
                    target_begin=int(k + int(train_size)),
                    target_end=int(k + int(train_size) + int(calibrate_size)),
                    prior_context_rounds_required=int(k),
                )

                cal_sample_weight = walk_forward._build_recency_weights_for_rows(
                    cfg=cfg,
                    n_rows=int(len(y_up_cal)),
                )

            price_model.fit(
                x_price_train,
                y_up_train,
                x_eval=x_price_cal,
                y_eval=y_up_cal,
                sample_weight=train_sample_weight,
            )
            pool_model.fit(
                x_pool_train,
                y_late_inflow_total,
                y_late_inflow_bull_frac,
                x_eval=x_pool_cal,
                y_total_eval=y_late_inflow_total_cal,
                y_frac_eval=y_late_inflow_bull_frac_cal,
                sample_weight=train_sample_weight,
            )

            models = walk_forward.WalkForwardModels(price_model=price_model, pool_model=pool_model)
            mu_cal = list(models.price_model.predict(x_price_cal))
            calibrator_final = walk_forward._fit_final_calibrator(
                mu_cal=mu_cal,
                y_up_cal=y_up_cal,
                sample_weight=cal_sample_weight,
            )
            pool_preds_cal = list(models.pool_model.predict(x_pool_cal))
            p_cal = calibrator_final.predict_proba_up(mu_cal)
            walk_forward._log_model_diagnostics(
                y_up_cal=y_up_cal,
                p_up_cal=p_cal,
                y_late_inflow_total_cal=y_late_inflow_total_cal,
                y_late_inflow_bull_frac_cal=y_late_inflow_bull_frac_cal,
                pool_preds_cal=pool_preds_cal,
            )
            return models, calibrator_final

        walk_forward._train_and_maybe_calibrate = _train_and_maybe_calibrate_train_cal
        window_order_applied = True

    if str(winrate_probe_profile) != "none" or winrate_probe_columns or sparse_probe_columns:
        import pancakebot.backtest.runner as backtest_runner
        import pancakebot.domain.features.feature_builder as feature_builder
        import pancakebot.domain.features.schema as schema_module
        import pancakebot.domain.models.walk_forward as walk_forward
        import pancakebot.domain.strategy.planner as planner
        from pancakebot.domain.features.schema import FeatureSchema

        profile = str(winrate_probe_profile)
        base_schema = schema_module.FEATURE_SCHEMA
        slot_a = str(base_schema.columns[0])
        slot_b = str(base_schema.columns[1]) if len(base_schema.columns) > 1 else str(slot_a)
        slot_c = str(base_schema.columns[2]) if len(base_schema.columns) > 2 else str(slot_b)
        selected_winrate_probe_columns = tuple(str(c) for c in winrate_probe_columns)
        selected_sparse_probe_columns = tuple(str(c) for c in sparse_probe_columns)
        if selected_sparse_probe_columns and selected_winrate_probe_columns:
            selected_sparse_effective_columns = tuple(str(c) for c in selected_winrate_probe_columns)
        else:
            selected_sparse_effective_columns = tuple(str(c) for c in selected_sparse_probe_columns)
        if selected_winrate_probe_columns:
            unknown_cols = sorted(set(selected_winrate_probe_columns) - set(base_schema.columns))
            if unknown_cols:
                raise ValueError(f"unknown_winrate_probe_columns: {','.join(unknown_cols)}")
        if selected_sparse_probe_columns:
            unknown_cols = sorted(set(selected_sparse_probe_columns) - set(base_schema.columns))
            if unknown_cols:
                raise ValueError(f"unknown_sparse_probe_columns: {','.join(unknown_cols)}")
        feature_profile_slots = {
            "slot_a": str(slot_a),
            "slot_b": str(slot_b),
            "slot_c": str(slot_c),
            "custom_columns": list(selected_winrate_probe_columns)
            if selected_winrate_probe_columns
            else list(selected_sparse_probe_columns),
            "winrate_custom_columns": list(selected_winrate_probe_columns),
            "sparse_custom_columns": list(selected_sparse_probe_columns),
        }

        if selected_sparse_probe_columns:
            selected_set = set(selected_sparse_effective_columns)
            selected_defs = tuple(f for f in base_schema.features if str(f.name) in selected_set)
            if len(selected_defs) != len(selected_sparse_effective_columns):
                raise InvariantError("sparse_probe_selected_defs_size_mismatch")
            sparse_schema = FeatureSchema(name=f"{base_schema.name}_sparse_probe", features=selected_defs)
            sparse_prior_required = int(sparse_schema.required_prior_context_rounds_size)
            sparse_klines_required = int(sparse_schema.required_context_klines_size)
            sparse_klines_required_runtime = int(sparse_klines_required) if int(sparse_klines_required) > 0 else 1
            col_group_map = {str(f.name): str(f.group) for f in selected_defs}

            def _sparse_max_prior() -> int:
                return int(sparse_prior_required)

            def _sparse_max_klines() -> int:
                return int(sparse_klines_required_runtime)

            def _sparse_vectorize(*, features, schema):
                del schema
                out: list[float] = []
                for c in sparse_schema.columns:
                    if c not in features:
                        raise InvariantError(f"sparse_probe_vectorize_missing_column: {c}")
                    out.append(float(features[c]))
                return out

            def _sparse_build_features(*, target_round, prior_context_rounds, context_klines, cutoff_seconds):
                if len(prior_context_rounds) != int(sparse_prior_required):
                    raise InvariantError(
                        f"prior_context_rounds_size_mismatch: got={len(prior_context_rounds)} expected={int(sparse_prior_required)}"
                    )
                if len(context_klines) != int(sparse_klines_required_runtime):
                    raise InvariantError(
                        f"context_klines_size_mismatch: got={len(context_klines)} expected={int(sparse_klines_required_runtime)}"
                    )
                return _compute_sparse_probe_features(
                    target_round=target_round,
                    prior_context_rounds=prior_context_rounds,
                    context_klines=context_klines,
                    cutoff_seconds=int(cutoff_seconds),
                    selected_columns=sparse_schema.columns,
                    column_groups=col_group_map,
                )

            orig_sparse_schema_feature_schema = schema_module.FEATURE_SCHEMA
            orig_sparse_schema_max_prior = schema_module.max_required_prior_context_rounds_size
            orig_sparse_schema_max_klines = schema_module.max_required_context_klines_size
            orig_sparse_feature_builder_feature_schema = feature_builder.FEATURE_SCHEMA
            orig_sparse_walk_forward_feature_schema = walk_forward.FEATURE_SCHEMA
            orig_sparse_planner_feature_schema = planner.FEATURE_SCHEMA
            orig_sparse_feature_builder_max_prior = feature_builder.max_required_prior_context_rounds_size
            orig_sparse_feature_builder_max_klines = feature_builder.max_required_context_klines_size
            orig_sparse_walk_forward_max_prior = walk_forward.max_required_prior_context_rounds_size
            orig_sparse_walk_forward_max_klines = walk_forward.max_required_context_klines_size
            orig_sparse_planner_max_prior = planner.max_required_prior_context_rounds_size
            orig_sparse_planner_max_klines = planner.max_required_context_klines_size
            orig_sparse_backtest_runner_max_prior = backtest_runner.max_required_prior_context_rounds_size
            orig_sparse_backtest_runner_max_klines = backtest_runner.max_required_context_klines_size
            orig_feature_builder_build_features = feature_builder.build_features
            orig_walk_forward_build_features = walk_forward.build_features
            orig_planner_build_features = planner.build_features
            orig_sparse_feature_builder_vectorize = feature_builder.vectorize
            orig_sparse_walk_forward_vectorize = walk_forward.vectorize
            orig_sparse_planner_vectorize = planner.vectorize

            schema_module.FEATURE_SCHEMA = sparse_schema
            schema_module.max_required_prior_context_rounds_size = _sparse_max_prior
            schema_module.max_required_context_klines_size = _sparse_max_klines

            feature_builder.FEATURE_SCHEMA = sparse_schema
            walk_forward.FEATURE_SCHEMA = sparse_schema
            planner.FEATURE_SCHEMA = sparse_schema

            feature_builder.max_required_prior_context_rounds_size = _sparse_max_prior
            feature_builder.max_required_context_klines_size = _sparse_max_klines
            walk_forward.max_required_prior_context_rounds_size = _sparse_max_prior
            walk_forward.max_required_context_klines_size = _sparse_max_klines
            planner.max_required_prior_context_rounds_size = _sparse_max_prior
            planner.max_required_context_klines_size = _sparse_max_klines
            backtest_runner.max_required_prior_context_rounds_size = _sparse_max_prior
            backtest_runner.max_required_context_klines_size = _sparse_max_klines

            feature_builder.build_features = _sparse_build_features
            walk_forward.build_features = _sparse_build_features
            planner.build_features = _sparse_build_features
            feature_builder.vectorize = _sparse_vectorize
            walk_forward.vectorize = _sparse_vectorize
            planner.vectorize = _sparse_vectorize
            sparse_probe_applied = True
        else:
            orig_feature_builder_build_features = feature_builder.build_features
            orig_walk_forward_build_features = walk_forward.build_features
            orig_planner_build_features = planner.build_features

            def _profile_build_features(*, target_round, prior_context_rounds, context_klines, cutoff_seconds):
                if str(profile) == "minimal_lock_lag1":
                    del context_klines, cutoff_seconds
                    out = {str(col): 0.0 for col in base_schema.columns}
                    lag1 = 0.0
                    if prior_context_rounds:
                        lock_lag1 = prior_context_rounds[-1].lock_price
                        if lock_lag1 is not None:
                            lock_lag1_f = float(lock_lag1)
                            if math.isfinite(lock_lag1_f) and lock_lag1_f > 0.0:
                                lag1 = float(math.log(lock_lag1_f))
                    out[str(slot_a)] = float(lag1)
                    return out

                if str(profile) == "oracle_lock_close":
                    del context_klines, cutoff_seconds, prior_context_rounds
                    out = {str(col): 0.0 for col in base_schema.columns}
                    lock_f = float(target_round.lock_price) if target_round.lock_price is not None else float("nan")
                    close_f = float(target_round.close_price) if target_round.close_price is not None else float("nan")
                    if math.isfinite(lock_f) and math.isfinite(close_f) and lock_f > 0.0 and close_f > 0.0:
                        out[str(slot_a)] = float(math.log(close_f / lock_f))
                        out[str(slot_b)] = float(math.log(lock_f))
                    return out

                if str(profile) == "price_flow_divergence":
                    raw = orig_feature_builder_build_features(
                        target_round=target_round,
                        prior_context_rounds=prior_context_rounds,
                        context_klines=context_klines,
                        cutoff_seconds=int(cutoff_seconds),
                    )
                    out = {str(col): 0.0 for col in base_schema.columns}
                    flow_raw = float(raw.get("log_imb_w_p_0_to_p_100", 0.0))
                    price_raw = float(raw.get("price_log_return_mean_k_15", 0.0))
                    flow = float(math.tanh(flow_raw)) if math.isfinite(flow_raw) else 0.0
                    price = float(max(-0.05, min(0.05, price_raw))) if math.isfinite(price_raw) else 0.0
                    divergence = float(-flow * price)
                    out[str(slot_a)] = float(divergence)
                    out[str(slot_b)] = float(flow)
                    out[str(slot_c)] = float(price)
                    return out

                if selected_winrate_probe_columns:
                    raw = orig_feature_builder_build_features(
                        target_round=target_round,
                        prior_context_rounds=prior_context_rounds,
                        context_klines=context_klines,
                        cutoff_seconds=int(cutoff_seconds),
                    )
                    out = {str(col): 0.0 for col in base_schema.columns}
                    for col in selected_winrate_probe_columns:
                        val = float(raw[str(col)])
                        out[str(col)] = float(val) if math.isfinite(val) else 0.0
                    return out

                return orig_feature_builder_build_features(
                    target_round=target_round,
                    prior_context_rounds=prior_context_rounds,
                    context_klines=context_klines,
                    cutoff_seconds=int(cutoff_seconds),
                )

            feature_builder.build_features = _profile_build_features
            walk_forward.build_features = _profile_build_features
            planner.build_features = _profile_build_features
            feature_profile_applied = True

    if zero_feature_groups:
        import pancakebot.domain.features.feature_builder as feature_builder
        import pancakebot.domain.models.walk_forward as walk_forward
        import pancakebot.domain.strategy.planner as planner
        from pancakebot.domain.features.schema import FEATURE_SCHEMA

        valid_groups = {str(f.group) for f in FEATURE_SCHEMA.features}
        unknown = sorted(set(zero_feature_groups) - valid_groups)
        if unknown:
            raise ValueError(f"unknown_feature_groups: {','.join(unknown)}")

        selected = set(zero_feature_groups)
        orig_feature_builder_vectorize = feature_builder.vectorize
        orig_walk_forward_vectorize = walk_forward.vectorize
        orig_planner_vectorize = planner.vectorize

        def _vectorize_with_group_zeroing(*, features, schema):
            out = orig_feature_builder_vectorize(features=features, schema=schema)
            for idx, f in enumerate(schema.features):
                if str(f.group) in selected:
                    out[idx] = 0.0
            return out

        feature_builder.vectorize = _vectorize_with_group_zeroing
        walk_forward.vectorize = _vectorize_with_group_zeroing
        planner.vectorize = _vectorize_with_group_zeroing
        feature_group_zeroing_applied = True

    if bool(args.winrate_only):
        import pancakebot.backtest.runner as backtest_runner
        import pancakebot.domain.strategy.planner as planner

        orig_winrate_size_bet = planner.size_bet
        orig_winrate_backtest_runner_size_bet = backtest_runner.size_bet

        def _winrate_only_size_bet(*, cfg, pred, bankroll_bnb: float) -> BetDecision:
            del bankroll_bnb
            min_bet_bnb = float(cfg.min_bet_amount_bnb)
            forced_side = "Bull" if float(pred.p_final) >= 0.5 else "Bear"
            return BetDecision(
                action="BET",
                bet_side=str(forced_side),
                amount_bnb=float(min_bet_bnb),
                expected_profit_bnb=0.0,
                post_impact_payout_multiple=None,
                bet_cap_bnb=float(min_bet_bnb),
                best_expected_profit_bnb=0.0,
                skip_reason=None,
            )

        planner.size_bet = _winrate_only_size_bet
        backtest_runner.size_bet = _winrate_only_size_bet
        winrate_only_sizing_applied = True

    no_positive_ev_override_enabled = bool(args.force_no_positive_ev) or (no_positive_ev_floor_bnb is not None)
    if no_positive_ev_override_enabled and not bool(args.winrate_only):
        import pancakebot.backtest.runner as backtest_runner
        import pancakebot.domain.strategy.planner as planner

        orig_size_bet = planner.size_bet
        orig_backtest_runner_size_bet = backtest_runner.size_bet
        override_floor_bnb = (
            float(no_positive_ev_floor_bnb) if no_positive_ev_floor_bnb is not None else float("-inf")
        )

        def _forced_size_bet(*, cfg, pred, bankroll_bnb: float) -> BetDecision:
            decision = orig_size_bet(cfg=cfg, pred=pred, bankroll_bnb=float(bankroll_bnb))
            if str(decision.action) != "SKIP" or str(decision.skip_reason) != "no_positive_ev":
                return decision
            if float(decision.best_expected_profit_bnb) < float(override_floor_bnb):
                return decision

            min_bet_bnb = float(cfg.min_bet_amount_bnb)
            max_affordable = float(bankroll_bnb) - float(GAS_COST_BET_BNB)
            cap_bnb = min(float(decision.bet_cap_bnb), float(max_affordable))
            if cap_bnb < float(min_bet_bnb):
                return decision

            forced_side = "Bull" if float(pred.p_final) >= 0.5 else "Bear"
            return BetDecision(
                action="BET",
                bet_side=str(forced_side),
                amount_bnb=float(min_bet_bnb),
                expected_profit_bnb=float(decision.best_expected_profit_bnb),
                post_impact_payout_multiple=None,
                bet_cap_bnb=float(decision.bet_cap_bnb),
                best_expected_profit_bnb=float(decision.best_expected_profit_bnb),
                skip_reason=None,
            )

        planner.size_bet = _forced_size_bet
        backtest_runner.size_bet = _forced_size_bet
        force_no_positive_ev_applied = True

    if str(direction_filter_mode) != "none":
        import pancakebot.backtest.runner as backtest_runner
        import pancakebot.domain.strategy.planner as planner

        orig_direction_filter_size_bet = planner.size_bet
        orig_direction_filter_backtest_runner_size_bet = backtest_runner.size_bet
        orig_direction_filter_settle = backtest_runner.settle_bet_against_closed_round
        th_bull = float(direction_threshold_bull)
        th_bear = float(direction_threshold_bear)
        p_history: list[float] = []
        adaptive_side_settled_history: list[dict[str, float | int | str]] = []
        last_adaptive_preference: str | None = None
        direction_gate_stats = {
            "enabled": True,
            "mode": str(direction_filter_mode),
            "rounds_seen": 0,
            "p_sum": 0.0,
            "center_sum": 0.0,
            "raw_bull_threshold_sum": 0.0,
            "raw_bear_threshold_sum": 0.0,
            "bull_gate_sum": 0.0,
            "bear_gate_sum": 0.0,
            "edge_floor_sum": 0.0,
            "bull_signal_rounds": 0,
            "bear_signal_rounds": 0,
            "overlap_signal_rounds": 0,
            "no_signal_rounds": 0,
            "expected_signal_rounds": 0,
            "expected_bull_signals": 0,
            "expected_bear_signals": 0,
            "realized_signal_bets": 0,
            "realized_signal_bets_bull": 0,
            "realized_signal_bets_bear": 0,
            "blocked_by_reason": {},
            "adaptive_window": int(direction_adaptive_window),
            "adaptive_min_history": int(direction_adaptive_min_history),
            "adaptive_switch_margin_bnb": float(direction_adaptive_switch_margin_bnb),
            "adaptive_score": str(direction_adaptive_score),
            "adaptive_default_side": str(direction_adaptive_default_side),
            "adaptive_allow_signal_fallback": bool(direction_adaptive_allow_signal_fallback),
            "adaptive_counterfactual": bool(direction_adaptive_counterfactual),
            "adaptive_preferred_bull_rounds": 0,
            "adaptive_preferred_bear_rounds": 0,
            "adaptive_no_preference_rounds": 0,
            "adaptive_fallback_to_signal_rounds": 0,
            "adaptive_preference_switches": 0,
            "adaptive_score_obs": 0,
            "adaptive_score_bull_sum": 0.0,
            "adaptive_score_bear_sum": 0.0,
            "adaptive_score_diff_sum": 0.0,
            "adaptive_side_history_size": 0,
        }

        def _history_tail(*, window: int) -> list[float]:
            if int(window) <= 0:
                return list(p_history)
            if len(p_history) <= int(window):
                return list(p_history)
            return list(p_history[-int(window):])

        def _direction_center_from_history() -> float:
            mode = str(direction_center_mode)
            if mode == "fixed_0p5":
                return 0.5
            hist = _history_tail(window=int(direction_center_window))
            if not hist:
                return 0.5
            if mode == "rolling_mean":
                return float(sum(hist) / len(hist))
            sorted_hist = sorted(float(v) for v in hist)
            return float(_quantile_from_sorted(sorted_hist, 0.5))

        def _direction_thresholds_from_history() -> tuple[float, float]:
            bull = float(th_bull)
            bear = float(th_bear)
            if str(direction_threshold_mode) != "quantile":
                return float(bull), float(bear)

            hist = _history_tail(window=int(direction_threshold_window))
            if len(hist) < int(direction_threshold_min_history):
                return float(bull), float(bear)

            sorted_hist = sorted(float(v) for v in hist)
            bull_q = 1.0 - float(direction_target_bull_rate)
            bear_q = float(direction_target_bear_rate)
            bull = float(_quantile_from_sorted(sorted_hist, bull_q))
            bear = float(_quantile_from_sorted(sorted_hist, bear_q))
            return float(bull), float(bear)

        def _direction_edge_floor_from_history(*, center: float) -> float:
            eps = float(direction_edge_floor_pp)
            ratio = direction_edge_floor_ratio
            q_spread = 0.9 if direction_edge_floor_quantile is None else float(direction_edge_floor_quantile)
            use_dynamic = (direction_edge_floor_quantile is not None) or (ratio is not None)
            if not bool(use_dynamic):
                return float(min(0.499999, max(0.0, eps)))

            hist = _history_tail(window=int(direction_edge_window))
            if len(hist) < int(direction_threshold_min_history):
                return float(min(0.499999, max(0.0, eps)))

            deviations = sorted(abs(float(v) - float(center)) for v in hist)
            spread = float(_quantile_from_sorted(deviations, float(q_spread)))
            if math.isfinite(float(spread)):
                if direction_edge_floor_quantile is not None:
                    eps = max(float(eps), float(spread))
                if ratio is not None:
                    eps = max(float(eps), float(ratio) * float(spread))
            return float(min(0.499999, max(0.0, eps)))

        def _adaptive_side_preference() -> tuple[str | None, dict[str, float | int | str | bool | None]]:
            window = int(direction_adaptive_window)
            if int(window) <= 0:
                recent = list(adaptive_side_settled_history)
            elif len(adaptive_side_settled_history) <= int(window):
                recent = list(adaptive_side_settled_history)
            else:
                recent = list(adaptive_side_settled_history[-int(window):])

            bull_rows = [x for x in recent if str(x.get("side", "")) == "Bull"]
            bear_rows = [x for x in recent if str(x.get("side", "")) == "Bear"]

            meta: dict[str, float | int | str | bool | None] = {
                "recent_total": int(len(recent)),
                "recent_bull": int(len(bull_rows)),
                "recent_bear": int(len(bear_rows)),
                "score_mode": str(direction_adaptive_score),
                "bull_score": None,
                "bear_score": None,
                "score_diff": None,
                "enough_history": False,
                "preferred_side": None,
            }

            if int(len(recent)) < int(direction_adaptive_min_history):
                return None, meta
            if not bull_rows or not bear_rows:
                return None, meta

            score_mode = str(direction_adaptive_score)
            if score_mode == "win_rate":
                bull_score = float(sum(int(x.get("win", 0)) for x in bull_rows) / len(bull_rows))
                bear_score = float(sum(int(x.get("win", 0)) for x in bear_rows) / len(bear_rows))
            else:
                bull_score = float(sum(float(x.get("profit", 0.0)) for x in bull_rows) / len(bull_rows))
                bear_score = float(sum(float(x.get("profit", 0.0)) for x in bear_rows) / len(bear_rows))

            diff = float(bull_score) - float(bear_score)
            margin = float(direction_adaptive_switch_margin_bnb)

            preferred: str | None
            if float(diff) > float(margin):
                preferred = "Bull"
            elif float(diff) < -float(margin):
                preferred = "Bear"
            else:
                preferred = None

            meta["bull_score"] = float(bull_score)
            meta["bear_score"] = float(bear_score)
            meta["score_diff"] = float(diff)
            meta["enough_history"] = True
            meta["preferred_side"] = preferred
            return preferred, meta

        def _skip_from_decision_reason(decision: BetDecision, reason: str) -> BetDecision:
            return BetDecision(
                action="SKIP",
                bet_side=None,
                amount_bnb=0.0,
                expected_profit_bnb=0.0,
                post_impact_payout_multiple=None,
                bet_cap_bnb=float(decision.bet_cap_bnb),
                best_expected_profit_bnb=float(decision.best_expected_profit_bnb),
                skip_reason=str(reason),
            )

        def _record_gate_round(
            *,
            decision: BetDecision,
            p_value: float,
            center: float,
            raw_bull_th: float,
            raw_bear_th: float,
            bull_gate: float,
            bear_gate: float,
            edge_floor: float,
            bull_sig: bool,
            bear_sig: bool,
            expected_signal: bool,
            desired_side: str | None,
            blocked_reason: str | None,
            realized_signal_bet: bool,
        ) -> BetDecision:
            if math.isfinite(float(p_value)):
                p_history.append(min(1.0, max(0.0, float(p_value))))
            direction_gate_stats["rounds_seen"] = int(direction_gate_stats["rounds_seen"]) + 1
            direction_gate_stats["p_sum"] = float(direction_gate_stats["p_sum"]) + float(p_value)
            direction_gate_stats["center_sum"] = float(direction_gate_stats["center_sum"]) + float(center)
            direction_gate_stats["raw_bull_threshold_sum"] = (
                float(direction_gate_stats["raw_bull_threshold_sum"]) + float(raw_bull_th)
            )
            direction_gate_stats["raw_bear_threshold_sum"] = (
                float(direction_gate_stats["raw_bear_threshold_sum"]) + float(raw_bear_th)
            )
            direction_gate_stats["bull_gate_sum"] = float(direction_gate_stats["bull_gate_sum"]) + float(bull_gate)
            direction_gate_stats["bear_gate_sum"] = float(direction_gate_stats["bear_gate_sum"]) + float(bear_gate)
            direction_gate_stats["edge_floor_sum"] = float(direction_gate_stats["edge_floor_sum"]) + float(edge_floor)

            if bool(bull_sig):
                direction_gate_stats["bull_signal_rounds"] = int(direction_gate_stats["bull_signal_rounds"]) + 1
            if bool(bear_sig):
                direction_gate_stats["bear_signal_rounds"] = int(direction_gate_stats["bear_signal_rounds"]) + 1
            if bool(bull_sig) and bool(bear_sig):
                direction_gate_stats["overlap_signal_rounds"] = int(direction_gate_stats["overlap_signal_rounds"]) + 1
            if (not bool(bull_sig)) and (not bool(bear_sig)):
                direction_gate_stats["no_signal_rounds"] = int(direction_gate_stats["no_signal_rounds"]) + 1

            if bool(expected_signal):
                direction_gate_stats["expected_signal_rounds"] = int(direction_gate_stats["expected_signal_rounds"]) + 1
                if str(desired_side) == "Bull":
                    direction_gate_stats["expected_bull_signals"] = (
                        int(direction_gate_stats["expected_bull_signals"]) + 1
                    )
                elif str(desired_side) == "Bear":
                    direction_gate_stats["expected_bear_signals"] = (
                        int(direction_gate_stats["expected_bear_signals"]) + 1
                    )

            if bool(realized_signal_bet):
                direction_gate_stats["realized_signal_bets"] = int(direction_gate_stats["realized_signal_bets"]) + 1
                if str(desired_side) == "Bull":
                    direction_gate_stats["realized_signal_bets_bull"] = (
                        int(direction_gate_stats["realized_signal_bets_bull"]) + 1
                    )
                elif str(desired_side) == "Bear":
                    direction_gate_stats["realized_signal_bets_bear"] = (
                        int(direction_gate_stats["realized_signal_bets_bear"]) + 1
                    )

            if blocked_reason is not None:
                key = str(blocked_reason).strip() or "unknown_skip_reason"
                blocked = direction_gate_stats["blocked_by_reason"]
                blocked[str(key)] = int(blocked.get(str(key), 0)) + 1
                direction_gate_stats["blocked_by_reason"] = blocked

            return decision

        def _direction_filtered_size_bet(*, cfg, pred, bankroll_bnb: float) -> BetDecision:
            nonlocal last_adaptive_preference
            decision = orig_direction_filter_size_bet(cfg=cfg, pred=pred, bankroll_bnb=float(bankroll_bnb))
            p = min(1.0, max(0.0, float(pred.p_final)))
            center = float(_direction_center_from_history())
            raw_bull_th, raw_bear_th = _direction_thresholds_from_history()
            edge_floor = float(_direction_edge_floor_from_history(center=float(center)))

            bull_gate = max(float(raw_bull_th), float(center) + float(edge_floor))
            bear_gate = min(float(raw_bear_th), float(center) - float(edge_floor))
            if float(bull_gate) < float(bear_gate):
                split = float(center)
                bull_gate = max(float(bull_gate), float(split))
                bear_gate = min(float(bear_gate), float(split))

            bull_sig = float(p) >= float(bull_gate)
            bear_sig = float(p) <= float(bear_gate)
            edge_insufficient = abs(float(p) - float(center)) < float(edge_floor)
            mode = str(direction_filter_mode)

            if mode == "adaptive_side":
                preferred, pref_meta = _adaptive_side_preference()
                direction_gate_stats["adaptive_side_history_size"] = int(pref_meta.get("recent_total", 0))
                bull_score = pref_meta.get("bull_score")
                bear_score = pref_meta.get("bear_score")
                diff_score = pref_meta.get("score_diff")
                if isinstance(bull_score, (int, float)) and isinstance(bear_score, (int, float)) and isinstance(diff_score, (int, float)):
                    direction_gate_stats["adaptive_score_obs"] = int(direction_gate_stats["adaptive_score_obs"]) + 1
                    direction_gate_stats["adaptive_score_bull_sum"] = (
                        float(direction_gate_stats["adaptive_score_bull_sum"]) + float(bull_score)
                    )
                    direction_gate_stats["adaptive_score_bear_sum"] = (
                        float(direction_gate_stats["adaptive_score_bear_sum"]) + float(bear_score)
                    )
                    direction_gate_stats["adaptive_score_diff_sum"] = (
                        float(direction_gate_stats["adaptive_score_diff_sum"]) + float(diff_score)
                    )

                if str(preferred) == "Bull":
                    direction_gate_stats["adaptive_preferred_bull_rounds"] = (
                        int(direction_gate_stats["adaptive_preferred_bull_rounds"]) + 1
                    )
                elif str(preferred) == "Bear":
                    direction_gate_stats["adaptive_preferred_bear_rounds"] = (
                        int(direction_gate_stats["adaptive_preferred_bear_rounds"]) + 1
                    )
                else:
                    direction_gate_stats["adaptive_no_preference_rounds"] = (
                        int(direction_gate_stats["adaptive_no_preference_rounds"]) + 1
                    )

                if str(preferred) in ("Bull", "Bear"):
                    if last_adaptive_preference is not None and str(last_adaptive_preference) != str(preferred):
                        direction_gate_stats["adaptive_preference_switches"] = (
                            int(direction_gate_stats["adaptive_preference_switches"]) + 1
                        )
                    last_adaptive_preference = str(preferred)

                desired: str | None = None
                blocked_reason: str | None = None

                if str(preferred) == "Bull":
                    if bool(bull_sig):
                        desired = "Bull"
                    elif bool(direction_adaptive_allow_signal_fallback) and bool(bear_sig):
                        desired = "Bear"
                        direction_gate_stats["adaptive_fallback_to_signal_rounds"] = (
                            int(direction_gate_stats["adaptive_fallback_to_signal_rounds"]) + 1
                        )
                    else:
                        blocked_reason = "direction_filter_adaptive_no_bull_signal"
                elif str(preferred) == "Bear":
                    if bool(bear_sig):
                        desired = "Bear"
                    elif bool(direction_adaptive_allow_signal_fallback) and bool(bull_sig):
                        desired = "Bull"
                        direction_gate_stats["adaptive_fallback_to_signal_rounds"] = (
                            int(direction_gate_stats["adaptive_fallback_to_signal_rounds"]) + 1
                        )
                    else:
                        blocked_reason = "direction_filter_adaptive_no_bear_signal"
                else:
                    cold = str(direction_adaptive_default_side)
                    if cold == "bull":
                        if bool(bull_sig):
                            desired = "Bull"
                        elif bool(direction_adaptive_allow_signal_fallback) and bool(bear_sig):
                            desired = "Bear"
                            direction_gate_stats["adaptive_fallback_to_signal_rounds"] = (
                                int(direction_gate_stats["adaptive_fallback_to_signal_rounds"]) + 1
                            )
                        else:
                            blocked_reason = "direction_filter_adaptive_default_bull_no_signal"
                    elif cold == "bear":
                        if bool(bear_sig):
                            desired = "Bear"
                        elif bool(direction_adaptive_allow_signal_fallback) and bool(bull_sig):
                            desired = "Bull"
                            direction_gate_stats["adaptive_fallback_to_signal_rounds"] = (
                                int(direction_gate_stats["adaptive_fallback_to_signal_rounds"]) + 1
                            )
                        else:
                            blocked_reason = "direction_filter_adaptive_default_bear_no_signal"
                    else:
                        if bool(bull_sig) and not bool(bear_sig):
                            desired = "Bull"
                        elif bool(bear_sig) and not bool(bull_sig):
                            desired = "Bear"
                        elif bool(bull_sig) and bool(bear_sig):
                            desired = "Bull" if float(p) >= float(center) else "Bear"
                        else:
                            blocked_reason = "insufficient_edge" if bool(edge_insufficient) else "direction_filter_no_signal"

                if desired is None:
                    reason = str(blocked_reason or ("insufficient_edge" if bool(edge_insufficient) else "direction_filter_no_signal"))
                    return _record_gate_round(
                        decision=_skip_from_decision_reason(decision, str(reason)),
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=False,
                        desired_side=None,
                        blocked_reason=str(reason),
                        realized_signal_bet=False,
                    )

                if str(decision.action) != "BET":
                    blocked = str(decision.skip_reason or "unknown_skip_reason")
                    return _record_gate_round(
                        decision=decision,
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=True,
                        desired_side=str(desired),
                        blocked_reason=str(blocked),
                        realized_signal_bet=False,
                    )
                if str(decision.bet_side) != str(desired):
                    return _record_gate_round(
                        decision=_skip_from_decision_reason(decision, "direction_filter_side_mismatch"),
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=True,
                        desired_side=str(desired),
                        blocked_reason="direction_filter_side_mismatch",
                        realized_signal_bet=False,
                    )
                return _record_gate_round(
                    decision=decision,
                    p_value=float(p),
                    center=float(center),
                    raw_bull_th=float(raw_bull_th),
                    raw_bear_th=float(raw_bear_th),
                    bull_gate=float(bull_gate),
                    bear_gate=float(bear_gate),
                    edge_floor=float(edge_floor),
                    bull_sig=bool(bull_sig),
                    bear_sig=bool(bear_sig),
                    expected_signal=True,
                    desired_side=str(desired),
                    blocked_reason=None,
                    realized_signal_bet=True,
                )

            if mode == "bull_only":
                if not bool(bull_sig):
                    reason = "insufficient_edge" if bool(edge_insufficient) else "direction_filter_no_bull_signal"
                    return _record_gate_round(
                        decision=_skip_from_decision_reason(decision, str(reason)),
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=False,
                        desired_side=None,
                        blocked_reason=str(reason),
                        realized_signal_bet=False,
                    )
                if str(decision.action) != "BET":
                    blocked = str(decision.skip_reason or "unknown_skip_reason")
                    return _record_gate_round(
                        decision=decision,
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=True,
                        desired_side="Bull",
                        blocked_reason=str(blocked),
                        realized_signal_bet=False,
                    )
                if str(decision.bet_side) != "Bull":
                    return _record_gate_round(
                        decision=_skip_from_decision_reason(decision, "direction_filter_side_mismatch"),
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=True,
                        desired_side="Bull",
                        blocked_reason="direction_filter_side_mismatch",
                        realized_signal_bet=False,
                    )
                return _record_gate_round(
                    decision=decision,
                    p_value=float(p),
                    center=float(center),
                    raw_bull_th=float(raw_bull_th),
                    raw_bear_th=float(raw_bear_th),
                    bull_gate=float(bull_gate),
                    bear_gate=float(bear_gate),
                    edge_floor=float(edge_floor),
                    bull_sig=bool(bull_sig),
                    bear_sig=bool(bear_sig),
                    expected_signal=True,
                    desired_side="Bull",
                    blocked_reason=None,
                    realized_signal_bet=True,
                )

            if mode == "bear_only":
                if not bool(bear_sig):
                    reason = "insufficient_edge" if bool(edge_insufficient) else "direction_filter_no_bear_signal"
                    return _record_gate_round(
                        decision=_skip_from_decision_reason(decision, str(reason)),
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=False,
                        desired_side=None,
                        blocked_reason=str(reason),
                        realized_signal_bet=False,
                    )
                if str(decision.action) != "BET":
                    blocked = str(decision.skip_reason or "unknown_skip_reason")
                    return _record_gate_round(
                        decision=decision,
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=True,
                        desired_side="Bear",
                        blocked_reason=str(blocked),
                        realized_signal_bet=False,
                    )
                if str(decision.bet_side) != "Bear":
                    return _record_gate_round(
                        decision=_skip_from_decision_reason(decision, "direction_filter_side_mismatch"),
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=True,
                        desired_side="Bear",
                        blocked_reason="direction_filter_side_mismatch",
                        realized_signal_bet=False,
                    )
                return _record_gate_round(
                    decision=decision,
                    p_value=float(p),
                    center=float(center),
                    raw_bull_th=float(raw_bull_th),
                    raw_bear_th=float(raw_bear_th),
                    bull_gate=float(bull_gate),
                    bear_gate=float(bear_gate),
                    edge_floor=float(edge_floor),
                    bull_sig=bool(bull_sig),
                    bear_sig=bool(bear_sig),
                    expected_signal=True,
                    desired_side="Bear",
                    blocked_reason=None,
                    realized_signal_bet=True,
                )

            if mode == "both_sides":
                desired = None
                if bool(bull_sig) and not bool(bear_sig):
                    desired = "Bull"
                elif bool(bear_sig) and not bool(bull_sig):
                    desired = "Bear"
                elif bool(bull_sig) and bool(bear_sig):
                    desired = "Bull" if float(p) >= float(center) else "Bear"
                else:
                    reason = "insufficient_edge" if bool(edge_insufficient) else "direction_filter_no_signal"
                    return _record_gate_round(
                        decision=_skip_from_decision_reason(decision, str(reason)),
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=False,
                        desired_side=None,
                        blocked_reason=str(reason),
                        realized_signal_bet=False,
                    )

                if str(decision.action) != "BET":
                    blocked = str(decision.skip_reason or "unknown_skip_reason")
                    return _record_gate_round(
                        decision=decision,
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=True,
                        desired_side=str(desired),
                        blocked_reason=str(blocked),
                        realized_signal_bet=False,
                    )
                if str(decision.bet_side) != str(desired):
                    return _record_gate_round(
                        decision=_skip_from_decision_reason(decision, "direction_filter_side_mismatch"),
                        p_value=float(p),
                        center=float(center),
                        raw_bull_th=float(raw_bull_th),
                        raw_bear_th=float(raw_bear_th),
                        bull_gate=float(bull_gate),
                        bear_gate=float(bear_gate),
                        edge_floor=float(edge_floor),
                        bull_sig=bool(bull_sig),
                        bear_sig=bool(bear_sig),
                        expected_signal=True,
                        desired_side=str(desired),
                        blocked_reason="direction_filter_side_mismatch",
                        realized_signal_bet=False,
                    )
                return _record_gate_round(
                    decision=decision,
                    p_value=float(p),
                    center=float(center),
                    raw_bull_th=float(raw_bull_th),
                    raw_bear_th=float(raw_bear_th),
                    bull_gate=float(bull_gate),
                    bear_gate=float(bear_gate),
                    edge_floor=float(edge_floor),
                    bull_sig=bool(bull_sig),
                    bear_sig=bool(bear_sig),
                    expected_signal=True,
                    desired_side=str(desired),
                    blocked_reason=None,
                    realized_signal_bet=True,
                )

            return _record_gate_round(
                decision=decision,
                p_value=float(p),
                center=float(center),
                raw_bull_th=float(raw_bull_th),
                raw_bear_th=float(raw_bear_th),
                bull_gate=float(bull_gate),
                bear_gate=float(bear_gate),
                edge_floor=float(edge_floor),
                bull_sig=bool(bull_sig),
                bear_sig=bool(bear_sig),
                expected_signal=False,
                desired_side=None,
                blocked_reason=None,
                realized_signal_bet=False,
            )

        def _direction_settle_with_history(*, bet_bnb: float, bet_side: str, round_closed, treasury_fee_fraction: float):
            res = orig_direction_filter_settle(
                bet_bnb=float(bet_bnb),
                bet_side=str(bet_side),
                round_closed=round_closed,
                treasury_fee_fraction=float(treasury_fee_fraction),
            )
            if str(direction_filter_mode) == "adaptive_side":
                side_u = str(bet_side).strip().upper()
                if side_u in ("BULL", "BEAR"):
                    chosen_side = "Bull" if side_u == "BULL" else "Bear"
                    chosen_profit = float(res.credit_bnb) - float(bet_bnb) - float(GAS_COST_BET_BNB)

                    def _append_side(side_name: str, profit_value: float) -> None:
                        adaptive_side_settled_history.append(
                            {
                                "side": str(side_name),
                                "profit": float(profit_value),
                                "win": 1 if float(profit_value) > 0.0 else 0,
                            }
                        )

                    _append_side(str(chosen_side), float(chosen_profit))

                    if bool(direction_adaptive_counterfactual):
                        other_side = "Bear" if str(chosen_side) == "Bull" else "Bull"
                        other_res = orig_direction_filter_settle(
                            bet_bnb=float(bet_bnb),
                            bet_side=str(other_side),
                            round_closed=round_closed,
                            treasury_fee_fraction=float(treasury_fee_fraction),
                        )
                        other_profit = float(other_res.credit_bnb) - float(bet_bnb) - float(GAS_COST_BET_BNB)
                        _append_side(str(other_side), float(other_profit))
            return res

        planner.size_bet = _direction_filtered_size_bet
        backtest_runner.size_bet = _direction_filtered_size_bet
        backtest_runner.settle_bet_against_closed_round = _direction_settle_with_history
        direction_filter_applied = True

    if (int(ev_reliability_window) > 0 or str(regime_filter) != "none") and not bool(args.winrate_only):
        import pancakebot.backtest.runner as backtest_runner
        import pancakebot.domain.strategy.planner as planner

        orig_gate_size_bet = planner.size_bet
        orig_gate_backtest_size_bet = backtest_runner.size_bet
        orig_backtest_settle = backtest_runner.settle_bet_against_closed_round

        pending_bets: list[dict[str, float | str]] = []
        settled_history: list[dict[str, float | str]] = []

        def _skip_from_decision(decision: BetDecision, reason: str) -> BetDecision:
            return BetDecision(
                action="SKIP",
                bet_side=None,
                amount_bnb=0.0,
                expected_profit_bnb=0.0,
                post_impact_payout_multiple=None,
                bet_cap_bnb=float(decision.bet_cap_bnb),
                best_expected_profit_bnb=float(decision.best_expected_profit_bnb),
                skip_reason=str(reason),
            )

        def _gated_size_bet(*, cfg, pred, bankroll_bnb: float) -> BetDecision:
            decision = orig_gate_size_bet(cfg=cfg, pred=pred, bankroll_bnb=float(bankroll_bnb))
            if str(decision.action) != "BET" or float(decision.amount_bnb) <= 0.0:
                return decision

            pool_imb = _pool_imbalance(
                final_bull_bnb=float(pred.final_bull_bnb),
                final_bear_bnb=float(pred.final_bear_bnb),
                final_total_bnb=float(pred.final_total_bnb),
            )

            if str(regime_filter) == "pool_imbalance":
                if float(pool_imb) < float(regime_min_imbalance):
                    return _skip_from_decision(decision, "regime_pool_imbalance_below_threshold")

            if int(ev_reliability_window) > 0:
                recent = settled_history[-int(ev_reliability_window):]
                if len(recent) >= int(ev_reliability_min_bets):
                    recent_evs_sorted = sorted(float(x["ev"]) for x in recent)
                    ev_threshold = _quantile_from_sorted(recent_evs_sorted, float(ev_reliability_quantile))
                    if float(decision.expected_profit_bnb) < float(ev_threshold):
                        return _skip_from_decision(decision, "ev_reliability_below_recent_quantile")

                    recent_top = [x for x in recent if float(x["ev"]) >= float(ev_threshold)]
                    if recent_top:
                        mean_profit_top = float(sum(float(x["profit"]) for x in recent_top) / len(recent_top))
                        if float(mean_profit_top) <= float(ev_reliability_min_mean_profit):
                            return _skip_from_decision(decision, "ev_reliability_recent_top_not_profitable")

            pending_bets.append(
                {
                    "ev": float(decision.expected_profit_bnb),
                    "side": str(decision.bet_side or ""),
                    "pool_imbalance": float(pool_imb),
                }
            )
            return decision

        def _settle_with_history(*, bet_bnb: float, bet_side: str, round_closed, treasury_fee_fraction: float):
            res = orig_backtest_settle(
                bet_bnb=float(bet_bnb),
                bet_side=str(bet_side),
                round_closed=round_closed,
                treasury_fee_fraction=float(treasury_fee_fraction),
            )
            if pending_bets:
                p = pending_bets.pop(0)
                profit = float(res.credit_bnb) - float(bet_bnb) - float(GAS_COST_BET_BNB)
                settled_history.append(
                    {
                        "ev": float(p["ev"]),
                        "profit": float(profit),
                        "side": str(p["side"]),
                        "pool_imbalance": float(p["pool_imbalance"]),
                    }
                )
            return res

        planner.size_bet = _gated_size_bet
        backtest_runner.size_bet = _gated_size_bet
        backtest_runner.settle_bet_against_closed_round = _settle_with_history
        decision_gate_applied = True

    if fixed_bet_bnb is not None:
        import pancakebot.backtest.runner as backtest_runner
        import pancakebot.domain.strategy.planner as planner

        orig_fixed_bet_size_bet = planner.size_bet
        orig_fixed_bet_backtest_runner_size_bet = backtest_runner.size_bet
        fixed_bet_amount_bnb = float(fixed_bet_bnb)
        ignore_cap = bool(fixed_bet_ignore_cap)

        def _fixed_bet_size_bet(*, cfg, pred, bankroll_bnb: float) -> BetDecision:
            decision = orig_fixed_bet_size_bet(cfg=cfg, pred=pred, bankroll_bnb=float(bankroll_bnb))
            if str(decision.action) != "BET" or float(decision.amount_bnb) <= 0.0:
                return decision

            max_affordable = float(bankroll_bnb) - float(GAS_COST_BET_BNB)
            cap_bnb = float(max_affordable) if bool(ignore_cap) else min(float(decision.bet_cap_bnb), float(max_affordable))
            if float(fixed_bet_amount_bnb) > float(cap_bnb):
                return BetDecision(
                    action="SKIP",
                    bet_side=None,
                    amount_bnb=0.0,
                    expected_profit_bnb=0.0,
                    post_impact_payout_multiple=None,
                    bet_cap_bnb=float(decision.bet_cap_bnb),
                    best_expected_profit_bnb=float(decision.best_expected_profit_bnb),
                    skip_reason=(
                        "fixed_bet_exceeds_bankroll"
                        if bool(ignore_cap)
                        else "fixed_bet_exceeds_cap_or_bankroll"
                    ),
                )

            bet_side = str(decision.bet_side or "")
            if bet_side not in ("Bull", "Bear"):
                return BetDecision(
                    action="SKIP",
                    bet_side=None,
                    amount_bnb=0.0,
                    expected_profit_bnb=0.0,
                    post_impact_payout_multiple=None,
                    bet_cap_bnb=float(decision.bet_cap_bnb),
                    best_expected_profit_bnb=float(decision.best_expected_profit_bnb),
                    skip_reason="fixed_bet_missing_side",
                )

            chain = ChainPolicyParams(
                min_bet_amount=float(cfg.min_bet_amount_bnb),
                treasury_fee_rate=float(cfg.treasury_fee_fraction),
                gas_bet_bnb=float(GAS_COST_BET_BNB),
                gas_claim_bnb=float(GAS_COST_CLAIM_BNB),
            )
            p_win = float(pred.p_final) if bet_side == "Bull" else (1.0 - float(pred.p_final))
            ev_fixed, payout_fixed = ev_for_side(
                bet_side=str(bet_side),
                p_win=float(p_win),
                bet_bnb=float(fixed_bet_amount_bnb),
                final_bull_bnb=float(pred.final_bull_bnb),
                final_bear_bnb=float(pred.final_bear_bnb),
                chain=chain,
            )

            return BetDecision(
                action="BET",
                bet_side=str(bet_side),
                amount_bnb=float(fixed_bet_amount_bnb),
                expected_profit_bnb=float(ev_fixed),
                post_impact_payout_multiple=float(payout_fixed),
                bet_cap_bnb=float(decision.bet_cap_bnb),
                best_expected_profit_bnb=float(decision.best_expected_profit_bnb),
                skip_reason=None,
            )

        planner.size_bet = _fixed_bet_size_bet
        backtest_runner.size_bet = _fixed_bet_size_bet
        fixed_bet_sizing_applied = True

    if int(sim_offset_rounds) > 0:
        import pancakebot.backtest.runner as backtest_runner

        orig_backtest_tail_rounds = backtest_runner._tail_rounds
        offset = int(sim_offset_rounds)

        def _tail_rounds_with_offset(store, *, n: int):
            if n <= 0:
                raise InvariantError("tail_rounds_n_invalid")

            required = int(n) + int(offset)
            dq = deque(maxlen=int(required))
            for r in store.iter_closed_rounds():
                dq.append(r)
            all_rounds = list(dq)

            if len(all_rounds) < int(required):
                raise InvariantError("backtest_insufficient_closed_rounds_for_offset")

            out = all_rounds[-int(required): -int(offset)] if int(offset) > 0 else all_rounds[-int(n):]
            if len(out) != int(n):
                raise InvariantError("backtest_offset_slice_len_mismatch")
            return list(out)

        backtest_runner._tail_rounds = _tail_rounds_with_offset
        sim_offset_applied = True

    bt_cfg = cfg.backtest
    if (
        args.sim_size is not None
        or args.initial_bankroll_bnb is not None
        or reset_mode_override is not None
        or reset_every_rounds_override is not None
    ):
        bt_cfg = BacktestConfig(
            simulation_size=int(args.sim_size) if args.sim_size is not None else int(cfg.backtest.simulation_size),
            initial_bankroll_bnb=(
                float(args.initial_bankroll_bnb)
                if args.initial_bankroll_bnb is not None
                else float(cfg.backtest.initial_bankroll_bnb)
            ),
            reset_mode=(
                str(reset_mode_override)
                if reset_mode_override is not None
                else str(cfg.backtest.reset_mode)
            ),
            reset_every_rounds=(
                int(reset_every_rounds_override)
                if reset_every_rounds_override is not None
                else int(cfg.backtest.reset_every_rounds)
            ),
        )
    bt_cfg.validate()

    try:
        run_backtest(runtime_cfg=runtime_cfg, backtest_cfg=bt_cfg, out_dir=out_dir)
    finally:
        if sparse_probe_applied:
            import pancakebot.backtest.runner as backtest_runner
            import pancakebot.domain.features.feature_builder as feature_builder
            import pancakebot.domain.features.schema as schema_module
            import pancakebot.domain.models.walk_forward as walk_forward
            import pancakebot.domain.strategy.planner as planner

            feature_builder.build_features = orig_feature_builder_build_features
            walk_forward.build_features = orig_walk_forward_build_features
            planner.build_features = orig_planner_build_features
            feature_builder.vectorize = orig_sparse_feature_builder_vectorize
            walk_forward.vectorize = orig_sparse_walk_forward_vectorize
            planner.vectorize = orig_sparse_planner_vectorize

            schema_module.FEATURE_SCHEMA = orig_sparse_schema_feature_schema
            schema_module.max_required_prior_context_rounds_size = orig_sparse_schema_max_prior
            schema_module.max_required_context_klines_size = orig_sparse_schema_max_klines

            feature_builder.FEATURE_SCHEMA = orig_sparse_feature_builder_feature_schema
            walk_forward.FEATURE_SCHEMA = orig_sparse_walk_forward_feature_schema
            planner.FEATURE_SCHEMA = orig_sparse_planner_feature_schema

            feature_builder.max_required_prior_context_rounds_size = orig_sparse_feature_builder_max_prior
            feature_builder.max_required_context_klines_size = orig_sparse_feature_builder_max_klines
            walk_forward.max_required_prior_context_rounds_size = orig_sparse_walk_forward_max_prior
            walk_forward.max_required_context_klines_size = orig_sparse_walk_forward_max_klines
            planner.max_required_prior_context_rounds_size = orig_sparse_planner_max_prior
            planner.max_required_context_klines_size = orig_sparse_planner_max_klines
            backtest_runner.max_required_prior_context_rounds_size = orig_sparse_backtest_runner_max_prior
            backtest_runner.max_required_context_klines_size = orig_sparse_backtest_runner_max_klines
        if feature_profile_applied:
            import pancakebot.domain.features.feature_builder as feature_builder
            import pancakebot.domain.models.walk_forward as walk_forward
            import pancakebot.domain.strategy.planner as planner

            feature_builder.build_features = orig_feature_builder_build_features
            walk_forward.build_features = orig_walk_forward_build_features
            planner.build_features = orig_planner_build_features
        if feature_group_zeroing_applied:
            import pancakebot.domain.features.feature_builder as feature_builder
            import pancakebot.domain.models.walk_forward as walk_forward
            import pancakebot.domain.strategy.planner as planner

            feature_builder.vectorize = orig_feature_builder_vectorize
            walk_forward.vectorize = orig_walk_forward_vectorize
            planner.vectorize = orig_planner_vectorize
        if calibration_mode_applied:
            import pancakebot.domain.models.walk_forward as walk_forward

            walk_forward._fit_final_calibrator = orig_fit_final_calibrator
        if direction_model_type_applied:
            import pancakebot.domain.models.walk_forward as walk_forward

            walk_forward.PriceReturnModel = orig_direction_model_class
        if window_order_applied:
            import pancakebot.domain.models.walk_forward as walk_forward

            walk_forward._train_and_maybe_calibrate = orig_train_and_maybe_calibrate
        if fixed_bet_sizing_applied:
            import pancakebot.backtest.runner as backtest_runner
            import pancakebot.domain.strategy.planner as planner

            planner.size_bet = orig_fixed_bet_size_bet
            backtest_runner.size_bet = orig_fixed_bet_backtest_runner_size_bet
        if sim_offset_applied:
            import pancakebot.backtest.runner as backtest_runner

            backtest_runner._tail_rounds = orig_backtest_tail_rounds
        if decision_gate_applied:
            import pancakebot.backtest.runner as backtest_runner
            import pancakebot.domain.strategy.planner as planner

            planner.size_bet = orig_gate_size_bet
            backtest_runner.size_bet = orig_gate_backtest_size_bet
            backtest_runner.settle_bet_against_closed_round = orig_backtest_settle
        if force_no_positive_ev_applied:
            import pancakebot.backtest.runner as backtest_runner
            import pancakebot.domain.strategy.planner as planner

            planner.size_bet = orig_size_bet
            backtest_runner.size_bet = orig_backtest_runner_size_bet
        if direction_filter_applied:
            import pancakebot.backtest.runner as backtest_runner
            import pancakebot.domain.strategy.planner as planner

            planner.size_bet = orig_direction_filter_size_bet
            backtest_runner.size_bet = orig_direction_filter_backtest_runner_size_bet
            backtest_runner.settle_bet_against_closed_round = orig_direction_filter_settle
        if winrate_only_sizing_applied:
            import pancakebot.backtest.runner as backtest_runner
            import pancakebot.domain.strategy.planner as planner

            planner.size_bet = orig_winrate_size_bet
            backtest_runner.size_bet = orig_winrate_backtest_runner_size_bet

    summary_path = out_dir / "backtest_summary.json"
    trades_path = out_dir / "backtest_trades.csv"
    summary = json.loads(summary_path.read_text())
    summary["scenario"] = {
        "name": str(args.name),
        "train_size": int(args.train_size),
        "calibrate_size": int(args.calibrate_size),
        "recency_weight_floor": float(args.rw_floor),
        "recency_weight_power": float(args.rw_power),
        "sim_size": int(bt_cfg.simulation_size),
        "initial_bankroll_bnb": float(bt_cfg.initial_bankroll_bnb),
        "reset_mode": str(bt_cfg.reset_mode),
        "reset_every_rounds": int(bt_cfg.reset_every_rounds),
        "sim_offset_rounds": int(sim_offset_rounds),
        "initial_bankroll_bnb_override": (
            float(args.initial_bankroll_bnb)
            if args.initial_bankroll_bnb is not None
            else None
        ),
        "direction_model_type": str(direction_model_type),
        "calibration_mode": str(calibration_mode),
        "window_order": str(window_order),
        "zero_feature_groups": list(zero_feature_groups),
        "ev_reliability_window": int(ev_reliability_window),
        "ev_reliability_min_bets": int(ev_reliability_min_bets),
        "ev_reliability_quantile": float(ev_reliability_quantile),
        "ev_reliability_min_mean_profit": float(ev_reliability_min_mean_profit),
        "regime_filter": str(regime_filter),
        "regime_min_imbalance": float(regime_min_imbalance),
        "raw_prob": bool(args.raw_prob),
        "winrate_only": bool(args.winrate_only),
        "winrate_probe_profile": str(winrate_probe_profile),
        "winrate_probe_columns": list(winrate_probe_columns),
        "sparse_probe_columns": list(sparse_probe_columns),
        "sparse_probe_enabled": bool(sparse_probe_applied),
        "direction_filter_mode": str(direction_filter_mode),
        "direction_threshold_mode": str(direction_threshold_mode),
        "direction_threshold_bull": float(direction_threshold_bull),
        "direction_threshold_bear": float(direction_threshold_bear),
        "direction_target_bull_rate": float(direction_target_bull_rate),
        "direction_target_bear_rate": float(direction_target_bear_rate),
        "direction_threshold_window": int(direction_threshold_window),
        "direction_threshold_min_history": int(direction_threshold_min_history),
        "direction_center_mode": str(direction_center_mode),
        "direction_center_window": int(direction_center_window),
        "direction_edge_floor_pp": float(direction_edge_floor_pp),
        "direction_edge_floor_ratio": direction_edge_floor_ratio,
        "direction_edge_floor_quantile": direction_edge_floor_quantile,
        "direction_edge_window": int(direction_edge_window),
        "direction_adaptive_window": int(direction_adaptive_window),
        "direction_adaptive_min_history": int(direction_adaptive_min_history),
        "direction_adaptive_switch_margin_bnb": float(direction_adaptive_switch_margin_bnb),
        "direction_adaptive_score": str(direction_adaptive_score),
        "direction_adaptive_default_side": str(direction_adaptive_default_side),
        "direction_adaptive_allow_signal_fallback": bool(direction_adaptive_allow_signal_fallback),
        "direction_adaptive_counterfactual": bool(direction_adaptive_counterfactual),
        "winrate_probe_slots": dict(feature_profile_slots),
        "force_no_positive_ev": bool(args.force_no_positive_ev),
        "no_positive_ev_floor_bnb": no_positive_ev_floor_bnb,
        "fixed_bet_bnb": fixed_bet_bnb,
        "fixed_bet_mode": "constant_stake" if fixed_bet_bnb is not None else "policy_sizing",
        "fixed_bet_ignore_cap": bool(fixed_bet_ignore_cap),
        "predictability_gate_mode": str(predictability_gate_mode),
        "predictability_gate_enabled": bool(runtime_cfg.predictability_gate_enabled),
        "predictability_gate_threshold": float(runtime_cfg.predictability_gate_threshold),
        "predictability_baseline_bet_bnb": float(runtime_cfg.predictability_baseline_bet_bnb),
        "predictability_gate_threshold_override": predictability_gate_threshold_override,
        "predictability_baseline_bet_bnb_override": predictability_baseline_bet_bnb_override,
        "corr_min_samples": int(corr_min_samples),
        "direction_viability_min_expected_signals": int(direction_viability_min_expected_signals),
        "direction_viability_min_expected_rate": float(direction_viability_min_expected_rate),
        "direction_viability_hard_fail": bool(args.direction_viability_hard_fail),
        "chunk_stability_chunks": int(chunk_stability_chunks),
        "chunk_stability_min_bets_per_chunk": int(chunk_stability_min_bets_per_chunk),
        "chunk_stability_max_dominance_share": float(chunk_stability_max_dominance_share),
        "chunk_stability_min_positive_fraction": float(chunk_stability_min_positive_fraction),
        "promotion_max_drawdown_bnb": promotion_max_drawdown_bnb,
        "promotion_require_both_side_coverage": bool(args.promotion_require_both_side_coverage),
    }
    p_stats = _p_stats(trades_path)
    summary["p_stats"] = p_stats
    summary["bet_diagnostics"] = _bet_diagnostics(
        trades_path,
        corr_min_samples=int(corr_min_samples),
    )
    raw_skip_counts = {
        str(k): int(v) for k, v in sorted(dict(summary.get("num_skips_by_reason", {})).items())
    }
    summary["skip_reason_groups"] = _canonical_skip_reason_counts(raw_skip_counts)

    if bool(direction_gate_stats.get("enabled", False)):
        rounds_seen = int(direction_gate_stats.get("rounds_seen", 0))
        denom = float(rounds_seen) if rounds_seen > 0 else 1.0
        direction_gate_summary = {
            "enabled": True,
            "mode": str(direction_gate_stats.get("mode", "")),
            "rounds_seen": int(rounds_seen),
            "p_mean_seen": float(float(direction_gate_stats.get("p_sum", 0.0)) / denom),
            "center_mean": float(float(direction_gate_stats.get("center_sum", 0.0)) / denom),
            "raw_bull_threshold_mean": float(float(direction_gate_stats.get("raw_bull_threshold_sum", 0.0)) / denom),
            "raw_bear_threshold_mean": float(float(direction_gate_stats.get("raw_bear_threshold_sum", 0.0)) / denom),
            "bull_gate_mean": float(float(direction_gate_stats.get("bull_gate_sum", 0.0)) / denom),
            "bear_gate_mean": float(float(direction_gate_stats.get("bear_gate_sum", 0.0)) / denom),
            "edge_floor_mean": float(float(direction_gate_stats.get("edge_floor_sum", 0.0)) / denom),
            "bull_signal_rounds": int(direction_gate_stats.get("bull_signal_rounds", 0)),
            "bear_signal_rounds": int(direction_gate_stats.get("bear_signal_rounds", 0)),
            "overlap_signal_rounds": int(direction_gate_stats.get("overlap_signal_rounds", 0)),
            "no_signal_rounds": int(direction_gate_stats.get("no_signal_rounds", 0)),
            "expected_signal_rounds": int(direction_gate_stats.get("expected_signal_rounds", 0)),
            "expected_bull_signals": int(direction_gate_stats.get("expected_bull_signals", 0)),
            "expected_bear_signals": int(direction_gate_stats.get("expected_bear_signals", 0)),
            "realized_signal_bets": int(direction_gate_stats.get("realized_signal_bets", 0)),
            "realized_signal_bets_bull": int(direction_gate_stats.get("realized_signal_bets_bull", 0)),
            "realized_signal_bets_bear": int(direction_gate_stats.get("realized_signal_bets_bear", 0)),
            "adaptive_window": int(direction_gate_stats.get("adaptive_window", 0)),
            "adaptive_min_history": int(direction_gate_stats.get("adaptive_min_history", 0)),
            "adaptive_switch_margin_bnb": float(direction_gate_stats.get("adaptive_switch_margin_bnb", 0.0)),
            "adaptive_score": str(direction_gate_stats.get("adaptive_score", "")),
            "adaptive_default_side": str(direction_gate_stats.get("adaptive_default_side", "")),
            "adaptive_allow_signal_fallback": bool(direction_gate_stats.get("adaptive_allow_signal_fallback", False)),
            "adaptive_counterfactual": bool(direction_gate_stats.get("adaptive_counterfactual", False)),
            "adaptive_preferred_bull_rounds": int(direction_gate_stats.get("adaptive_preferred_bull_rounds", 0)),
            "adaptive_preferred_bear_rounds": int(direction_gate_stats.get("adaptive_preferred_bear_rounds", 0)),
            "adaptive_no_preference_rounds": int(direction_gate_stats.get("adaptive_no_preference_rounds", 0)),
            "adaptive_fallback_to_signal_rounds": int(direction_gate_stats.get("adaptive_fallback_to_signal_rounds", 0)),
            "adaptive_preference_switches": int(direction_gate_stats.get("adaptive_preference_switches", 0)),
            "adaptive_side_history_size": int(direction_gate_stats.get("adaptive_side_history_size", 0)),
            "adaptive_score_obs": int(direction_gate_stats.get("adaptive_score_obs", 0)),
            "adaptive_score_bull_mean": (
                float(direction_gate_stats.get("adaptive_score_bull_sum", 0.0))
                / float(max(1, int(direction_gate_stats.get("adaptive_score_obs", 0))))
            ),
            "adaptive_score_bear_mean": (
                float(direction_gate_stats.get("adaptive_score_bear_sum", 0.0))
                / float(max(1, int(direction_gate_stats.get("adaptive_score_obs", 0))))
            ),
            "adaptive_score_diff_mean": (
                float(direction_gate_stats.get("adaptive_score_diff_sum", 0.0))
                / float(max(1, int(direction_gate_stats.get("adaptive_score_obs", 0))))
            ),
            "blocked_by_reason": {
                str(k): int(v)
                for k, v in sorted(dict(direction_gate_stats.get("blocked_by_reason", {})).items())
            },
        }
    else:
        direction_gate_summary = {"enabled": False}
    summary["direction_gate"] = direction_gate_summary

    viability = _direction_viability(
        direction_gate=direction_gate_summary,
        p_stats=p_stats,
        min_expected_signals=int(direction_viability_min_expected_signals),
        min_expected_signal_rate=float(direction_viability_min_expected_rate),
    )
    summary["gate_viability"] = viability

    chunk_stability = _chunk_stability(
        trades_csv=trades_path,
        chunks=int(chunk_stability_chunks),
        min_bets_per_chunk=int(chunk_stability_min_bets_per_chunk),
        max_dominance_share=float(chunk_stability_max_dominance_share),
        min_positive_chunk_fraction=float(chunk_stability_min_positive_fraction),
    )
    summary["chunk_stability"] = chunk_stability

    max_drawdown_bnb = _max_drawdown_bnb(trades_path)
    summary["risk"] = {
        "max_drawdown_bnb": float(max_drawdown_bnb),
    }
    summary["promotion"] = _promotion_assessment(
        summary=summary,
        viability=viability,
        stability=chunk_stability,
        max_drawdown_bnb=float(max_drawdown_bnb),
        max_allowed_drawdown_bnb=promotion_max_drawdown_bnb,
        require_both_side_coverage=bool(args.promotion_require_both_side_coverage),
    )
    if bool(args.winrate_only):
        summary["winrate_probe"] = {
            "mode": "fixed_min_bet_directional",
            "scored_rounds": int(summary["num_bets"]),
            "correct": int(summary["num_wins"]),
            "win_rate": float(summary["win_rate"]),
            "pred_bull": int(summary["num_bets_bull"]),
            "pred_bear": int(summary["num_bets_bear"]),
            "num_skips": int(summary["num_skips"]),
            "num_skips_by_reason": {
                str(k): int(v) for k, v in sorted(dict(summary["num_skips_by_reason"]).items())
            },
        }
    hard_fail_viability = bool(
        bool(args.direction_viability_hard_fail)
        and bool(viability.get("enabled", False))
        and (not bool(viability.get("actionable", True)))
    )
    summary["status"] = "non_actionable_viability_failed" if hard_fail_viability else "ok"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if hard_fail_viability:
        raise InvariantError("direction_viability_failed")

    print(f"SCENARIO={args.name}")
    print(f"SUMMARY={summary_path}")
    print(f"TRADES={trades_path}")
    print(f"NET={summary['net_profit_bnb']}")
    print(f"BETS={summary['num_bets']}")
    print(f"BET_RATE={summary['bet_rate']}")
    if bool(args.winrate_only):
        print(f"WINRATE={summary['win_rate']}")
        print(f"SCORED_ROUNDS={summary['num_bets']}")
        print(f"CORRECT={summary['num_wins']}")


if __name__ == "__main__":
    main()
