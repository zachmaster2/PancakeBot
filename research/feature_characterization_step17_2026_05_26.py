"""Step 17 — descriptive feature characterization, no parameter sweep.

Five feature families × 7 cohorts. Statistical separation: extension vs CV5
via Cohen's d + permutation-based Mann-Whitney p-value.

  F1: Return autocorrelation (1-min returns, lags 1/5/15/60)
  F2: Cross-asset correlation (hourly rolling 24h Pearson, 6 pairs)
  F3: Vol term structure (sigma_1h / sigma_24h ratio, hourly)
  F4: Time-of-day / day-of-week bet stats (requires baseline bet stream)
  F5: Round-level pool features (bull/total ratio, bet count, total BNB)

Uses cached klines + rounds. One baseline backtest (cd=72/dd=0.15) for F4.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
import research.in_process_runner as ipr  # noqa: E402
ipr._EXT_CLOSED_ROUNDS_PATH = EXT_DIR / "closed_rounds.jsonl"
ipr._EXT_BTC_KLINES_PATH = EXT_DIR / "btc_spot_prices.jsonl"
ipr._EXT_ETH_KLINES_PATH = EXT_DIR / "eth_spot_prices.jsonl"
ipr._EXT_SOL_KLINES_PATH = EXT_DIR / "sol_spot_prices.jsonl"

from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.constants import MAX_GAS_COST_BET_BNB, BNB_WEI  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402
from pancakebot.settlement import settle_bet_against_closed_round  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402
from pancakebot.bankroll_tracker import InMemoryBankrollTracker  # noqa: E402


EPOCH_MIN = 422298
EPOCH_MAX = 484999
CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2
POOL_CUTOFF = 6
TREASURY_FEE = 0.03
MIN_BET = 0.001
COOLDOWN_ROUNDS = 72
DRAWDOWN_PEAK_WINDOW_DAYS = 7
ABS_DD_FRAC = 0.15
INITIAL_BANKROLL = 5.0
PERM_SEEDS = 1000

COHORT_DEFS = [
    ("extension", 422298, 437561),
    ("cv5", 437562, 474086),
    ("gap_post_cv5_pre_holdout", 474087, 474879),
    ("holdout", 474880, 475311),
    ("ext_v2", 475312, 479952),
    ("fresh_oos", 479953, 483191),
    ("post_fresh", 483192, 999999),
]
COHORT_ORDER = [c[0] for c in COHORT_DEFS]
ASSETS = ("BTC", "ETH", "SOL", "BNB")


def cohort_of(epoch: int) -> str:
    for name, lo, hi in COHORT_DEFS:
        if lo <= epoch <= hi:
            return name
    return "unknown"


# ============================================================
# Statistics helpers
# ============================================================

def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size between samples a and b."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_mean, b_mean = float(np.mean(a_arr)), float(np.mean(b_arr))
    a_var, b_var = float(np.var(a_arr, ddof=1)), float(np.var(b_arr, ddof=1))
    pooled_std = math.sqrt(((len(a) - 1) * a_var + (len(b) - 1) * b_var) /
                            (len(a) + len(b) - 2))
    if pooled_std == 0:
        return 0.0
    return (a_mean - b_mean) / pooled_std


def perm_mann_whitney_p(a: list[float], b: list[float],
                         n_seeds: int = PERM_SEEDS, seed: int = 42) -> float:
    """Permutation-based two-sided test on mean difference."""
    if len(a) < 2 or len(b) < 2:
        return 1.0
    obs_diff = abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(n_seeds)
    for i in range(n_seeds):
        rng.shuffle(combined)
        perm_diffs[i] = abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
    return float(np.mean(perm_diffs >= obs_diff))


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "stdev": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


# ============================================================
# Single-pass kline loader: emits 1-min samples + pipeline slice
# ============================================================

_SAMPLE_INDICES = (59, 119, 179, 239, 299)


def load_klines_for_step17(
    path: Path, ext_path: Path | None,
    *,
    want_pipeline_slice: bool,
    pipeline_earliest_offset: int,
    pipeline_latest_offset: int,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[list]]]:
    """Single pass over the JSONL kline file(s).

    Returns:
      samples_per_epoch: {epoch: [(ts_sec, close_price), ...]} — 1-min samples
                          extracted at positions 59/119/179/239/299 of each
                          300-candle round (5 samples covering ~5 min span).
      pipeline_klines_per_epoch: {epoch: 16-candle sliced kline} for use by
                                  the F4 pipeline. Empty if want_pipeline_slice=False.
    """
    samples_out: dict[int, list[tuple[int, float]]] = {}
    pipeline_out: dict[int, list[list]] = {}
    if want_pipeline_slice:
        if pipeline_latest_offset < 2:
            raise ValueError("pipeline_latest_offset_must_be_ge_2")
        start_neg = -(pipeline_earliest_offset - 1)
        end_neg = None if pipeline_latest_offset == 2 else -(pipeline_latest_offset - 2)

    def _ingest(p: Path) -> None:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error") or rec.get("klines_1s") is None:
                    continue
                kl = rec["klines_1s"]
                if not kl:
                    continue
                ep = int(rec["epoch"])
                if ep in samples_out:
                    continue  # canonical wins
                samples: list[tuple[int, float]] = []
                for i in _SAMPLE_INDICES:
                    if i < len(kl):
                        ts_ms = int(kl[i][0])
                        price = float(kl[i][4])
                        if price > 0:
                            samples.append((ts_ms // 1000, price))
                if samples:
                    samples_out[ep] = samples
                if want_pipeline_slice:
                    if end_neg is None:
                        pipeline_out[ep] = kl[start_neg:]
                    else:
                        pipeline_out[ep] = kl[start_neg:end_neg]

    if path.exists():
        _ingest(path)
    if ext_path is not None and ext_path.exists():
        _ingest(ext_path)
    return samples_out, pipeline_out


# ============================================================
# F1: 1-min returns per cohort per asset, autocorrelation
# ============================================================

def build_1min_returns_per_cohort(
    samples_by_asset: dict[str, dict[int, list[tuple[int, float]]]],
    all_rounds: list,
) -> dict[str, dict[str, list[float]]]:
    """Returns {asset: {cohort: [1-min log returns]}}.

    samples_by_asset[asset][epoch] = list of (ts_sec, price) at every 60s
    within the round. Log-return between consecutive samples.
    """
    out = {asset: {coh: [] for coh in COHORT_ORDER} for asset in ASSETS}
    for r in all_rounds:
        ep = int(r.epoch)
        coh = cohort_of(ep)
        for asset in ASSETS:
            samples = samples_by_asset[asset].get(ep)
            if not samples or len(samples) < 2:
                continue
            for j in range(1, len(samples)):
                p_prev, p_cur = samples[j - 1][1], samples[j][1]
                if p_prev > 0 and p_cur > 0:
                    out[asset][coh].append(math.log(p_cur / p_prev))
    return out


def autocorrelation_at_lag(returns: list[float], lag: int) -> float:
    if len(returns) < lag + 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    if len(arr) < lag + 1:
        return 0.0
    a = arr[:-lag]
    b = arr[lag:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# F2: cross-asset correlations (hourly rolling 24h)
# ============================================================

def build_global_1min_timeline(
    samples_by_asset: dict[str, dict[int, list[tuple[int, float]]]],
) -> dict[str, list[tuple[int, float]]]:
    """Per asset, build sorted (ts_sec, price) list at 1-min resolution."""
    out: dict[str, list[tuple[int, float]]] = {a: [] for a in ASSETS}
    for asset in ASSETS:
        for ep, samples in samples_by_asset[asset].items():
            for ts_sec, price in samples:
                out[asset].append((ts_sec, price))
        out[asset].sort()
    return out


def build_returns_from_timeline(
    timeline: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Compute log returns from a sorted (ts, price) timeline. Returns
    list of (ts_at_return_end, log_return)."""
    out = []
    for i in range(1, len(timeline)):
        t0, p0 = timeline[i - 1]
        t1, p1 = timeline[i]
        if p0 > 0 and p1 > 0:
            out.append((t1, math.log(p1 / p0)))
    return out


def cross_asset_correlations_hourly(
    asset_returns: dict[str, list[tuple[int, float]]],
    cohort_ts_ranges: dict[str, tuple[int, int]],
) -> dict[str, list[float]]:
    """For each hour-aligned window in the dataset, compute rolling 24h
    Pearson correlation across asset pairs. Return {cohort: [mean of 6 pair correlations]}."""
    # Build sorted ts arrays per asset for fast indexing
    ts_per_asset: dict[str, np.ndarray] = {}
    ret_per_asset: dict[str, np.ndarray] = {}
    for a in ASSETS:
        ts_per_asset[a] = np.array([t for t, _ in asset_returns[a]])
        ret_per_asset[a] = np.array([r for _, r in asset_returns[a]])

    # Determine common ts range
    if not all(len(ts_per_asset[a]) > 0 for a in ASSETS):
        return {c: [] for c in COHORT_ORDER}
    ts_min = max(int(ts_per_asset[a][0]) for a in ASSETS) + 24 * 3600
    ts_max = min(int(ts_per_asset[a][-1]) for a in ASSETS)
    if ts_max <= ts_min:
        return {c: [] for c in COHORT_ORDER}

    pairs = [("BTC", "ETH"), ("BTC", "SOL"), ("BTC", "BNB"),
             ("ETH", "SOL"), ("ETH", "BNB"), ("SOL", "BNB")]

    out: dict[str, list[float]] = {c: [] for c in COHORT_ORDER}

    # Iterate hourly
    window_s = 24 * 3600
    hour_s = 3600
    for hr_end in range(ts_min, ts_max + 1, hour_s):
        hr_start = hr_end - window_s
        # Determine cohort of this hour endpoint via ts -> epoch lookup approximation
        # Use the epoch_to_ts inverse implicitly: find which cohort by ts ranges
        coh = None
        for c, (lo_ts, hi_ts) in cohort_ts_ranges.items():
            if lo_ts <= hr_end <= hi_ts:
                coh = c; break
        if coh is None:
            continue

        # For each asset, get returns within [hr_start, hr_end]
        rets_in_window: dict[str, np.ndarray] = {}
        n_min = 999_999_999
        for a in ASSETS:
            ts_arr = ts_per_asset[a]
            idx_lo = np.searchsorted(ts_arr, hr_start, side="left")
            idx_hi = np.searchsorted(ts_arr, hr_end, side="right")
            rets_in_window[a] = ret_per_asset[a][idx_lo:idx_hi]
            n_min = min(n_min, len(rets_in_window[a]))
        if n_min < 50:
            continue
        # Truncate each to min length so all aligned
        n_use = n_min
        pair_corrs = []
        for a1, a2 in pairs:
            r1 = rets_in_window[a1][-n_use:]
            r2 = rets_in_window[a2][-n_use:]
            if r1.std() == 0 or r2.std() == 0:
                continue
            c12 = float(np.corrcoef(r1, r2)[0, 1])
            pair_corrs.append(c12)
        if pair_corrs:
            out[coh].append(float(np.mean(pair_corrs)))
    return out


# ============================================================
# F3: vol term structure (hourly sigma_1h / sigma_24h)
# ============================================================

def vol_term_structure_hourly(
    asset_returns: dict[str, list[tuple[int, float]]],
    cohort_ts_ranges: dict[str, tuple[int, int]],
) -> dict[str, list[float]]:
    """For BTC, compute hourly ratio sigma_1h / sigma_24h. Return {cohort: [ratios]}."""
    if not asset_returns["BTC"]:
        return {c: [] for c in COHORT_ORDER}
    ts_arr = np.array([t for t, _ in asset_returns["BTC"]])
    ret_arr = np.array([r for _, r in asset_returns["BTC"]])
    if len(ts_arr) < 1440 + 60:
        return {c: [] for c in COHORT_ORDER}

    ts_min = int(ts_arr[0]) + 24 * 3600
    ts_max = int(ts_arr[-1])
    out: dict[str, list[float]] = {c: [] for c in COHORT_ORDER}
    for hr_end in range(ts_min, ts_max + 1, 3600):
        idx_24h_lo = np.searchsorted(ts_arr, hr_end - 24 * 3600, side="left")
        idx_1h_lo = np.searchsorted(ts_arr, hr_end - 3600, side="left")
        idx_hi = np.searchsorted(ts_arr, hr_end, side="right")
        if idx_24h_lo >= idx_1h_lo or idx_1h_lo >= idx_hi:
            continue
        ret_24h = ret_arr[idx_24h_lo:idx_hi]
        ret_1h = ret_arr[idx_1h_lo:idx_hi]
        if len(ret_24h) < 50 or len(ret_1h) < 5:
            continue
        sigma_24h = float(np.std(ret_24h, ddof=1))
        sigma_1h = float(np.std(ret_1h, ddof=1))
        if sigma_24h <= 0:
            continue
        coh = None
        for c, (lo_ts, hi_ts) in cohort_ts_ranges.items():
            if lo_ts <= hr_end <= hi_ts:
                coh = c; break
        if coh is None:
            continue
        out[coh].append(sigma_1h / sigma_24h)
    return out


# ============================================================
# F4: Time-of-day / day-of-week (requires baseline bet stream)
# ============================================================

class GateTracker(InMemoryBankrollTracker):
    """Production-bit-identical pre-decision drawdown + +1 cooldown comp."""

    def __init__(self, *, initial_bankroll, drawdown_peak_window_days, peak_mode,
                  cooldown_rounds, abs_dd_frac):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cd_total = int(cooldown_rounds)
        self._abs_dd = float(abs_dd_frac)
        self.n_pauses_fired = 0

    def is_paused(self, as_of_start_at):
        if self._cooldown > 0:
            return True
        current = self.current_bankroll()
        peak = self.peak_bankroll(as_of_start_at)
        if peak > 0:
            dd = (peak - current) / peak
            if dd >= self._abs_dd:
                if self._cd_total > 0:
                    self.set_paused(self._cd_total + 1, as_of_start_at)
                self.n_pauses_fired += 1
                return self._cd_total > 0
        return False


def run_baseline_for_bets(all_rounds, btc_klines, eth_klines, sol_klines,
                            earliest_offset) -> list[dict[str, Any]]:
    """Single backtest at cd=72/dd=0.15 @5BNB. Returns per-bet records with
    epoch, ts, win, profit, cohort."""
    overrides = {
        "gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)},
        "risk": {"max_drawdown_fraction_from_peak": 1.0},
    }
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    tracker = GateTracker(
        initial_bankroll=INITIAL_BANKROLL,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        cooldown_rounds=COOLDOWN_ROUNDS,
        abs_dd_frac=ABS_DD_FRAC,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_cfg, strategy_config=sc, gate=None,
        kline_cutoff_seconds=CANONICAL_CUTOFF, pool_cutoff_seconds=POOL_CUTOFF,
        min_bet_amount_bnb=MIN_BET, treasury_fee_fraction=TREASURY_FEE,
        bankroll_tracker=tracker,
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX]
    bankroll = float(INITIAL_BANKROLL)
    bet_records = []
    for round_t in sim_rounds:
        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue
        bet_size = float(decision.bet_size_bnb)
        side = str(decision.bet_side)
        bankroll -= bet_size + MAX_GAS_COST_BET_BNB
        outcome = settle_bet_against_closed_round(
            bet_bnb=bet_size, bet_side=side, round_closed=round_t,
            treasury_fee_fraction=TREASURY_FEE,
        )
        bankroll += outcome.credit_bnb
        profit = outcome.credit_bnb - bet_size - MAX_GAS_COST_BET_BNB
        bet_records.append({
            "epoch": int(round_t.epoch),
            "start_at": int(round_t.start_at),
            "cohort": cohort_of(int(round_t.epoch)),
            "won": outcome.outcome == "win",
            "profit": profit,
            "side": side,
        })
        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])
    return bet_records


def f4_time_of_day_analysis(bet_records: list[dict[str, Any]],
                              all_rounds: list) -> dict[str, Any]:
    """Group bets + rounds by hour-of-day (UTC) and day-of-week."""
    # Build per-round (hour, weekday) from all_rounds
    round_count_by_hour = defaultdict(int)
    round_count_by_dow = defaultdict(int)
    round_count_by_hour_cohort = defaultdict(lambda: defaultdict(int))

    for r in all_rounds:
        if not (EPOCH_MIN <= r.epoch <= EPOCH_MAX):
            continue
        dt = datetime.fromtimestamp(int(r.start_at), tz=timezone.utc)
        round_count_by_hour[dt.hour] += 1
        round_count_by_dow[dt.weekday()] += 1
        round_count_by_hour_cohort[cohort_of(int(r.epoch))][dt.hour] += 1

    # Per-bet stats per hour
    bet_by_hour: dict[int, list[dict]] = defaultdict(list)
    bet_by_dow: dict[int, list[dict]] = defaultdict(list)
    for b in bet_records:
        dt = datetime.fromtimestamp(b["start_at"], tz=timezone.utc)
        bet_by_hour[dt.hour].append(b)
        bet_by_dow[dt.weekday()].append(b)

    # Build hour table
    hour_stats = {}
    for h in range(24):
        bets = bet_by_hour[h]
        rounds = round_count_by_hour[h]
        n_bets = len(bets)
        wins = sum(1 for b in bets if b["won"])
        pnl = sum(b["profit"] for b in bets)
        hour_stats[h] = {
            "n_rounds": rounds, "n_bets": n_bets, "n_wins": wins,
            "bet_rate": n_bets / rounds if rounds else 0.0,
            "win_rate": wins / n_bets if n_bets else 0.0,
            "total_pnl_bnb": pnl,
            "mean_pnl_per_bet": pnl / n_bets if n_bets else 0.0,
        }

    # Per-cohort by hour: PnL summary
    bet_by_cohort_hour: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for b in bet_records:
        dt = datetime.fromtimestamp(b["start_at"], tz=timezone.utc)
        bet_by_cohort_hour[b["cohort"]][dt.hour].append(b)

    cohort_hour_pnl = {c: {h: 0.0 for h in range(24)} for c in COHORT_ORDER}
    cohort_hour_bets = {c: {h: 0 for h in range(24)} for c in COHORT_ORDER}
    for coh, by_hour in bet_by_cohort_hour.items():
        for h, bets in by_hour.items():
            cohort_hour_pnl[coh][h] = sum(b["profit"] for b in bets)
            cohort_hour_bets[coh][h] = len(bets)

    # Day-of-week table
    dow_stats = {}
    for d in range(7):
        bets = bet_by_dow[d]
        rounds = round_count_by_dow[d]
        n_bets = len(bets)
        wins = sum(1 for b in bets if b["won"])
        pnl = sum(b["profit"] for b in bets)
        dow_stats[d] = {
            "n_rounds": rounds, "n_bets": n_bets, "n_wins": wins,
            "bet_rate": n_bets / rounds if rounds else 0.0,
            "win_rate": wins / n_bets if n_bets else 0.0,
            "total_pnl_bnb": pnl,
            "mean_pnl_per_bet": pnl / n_bets if n_bets else 0.0,
        }

    return {
        "hour_stats": hour_stats,
        "dow_stats": dow_stats,
        "cohort_hour_pnl": cohort_hour_pnl,
        "cohort_hour_bets": cohort_hour_bets,
    }


# ============================================================
# F5: Round-level pool features
# ============================================================

def f5_pool_features(all_rounds: list) -> dict[str, dict[str, Any]]:
    """Per round, compute pool ratio, bet count, total BNB. Aggregate per cohort."""
    per_cohort: dict[str, dict[str, list[float]]] = {
        c: {"bull_pool_ratio": [], "n_bets": [], "total_bnb": []}
        for c in COHORT_ORDER
    }

    for r in all_rounds:
        ep = int(r.epoch)
        if not (EPOCH_MIN <= ep <= EPOCH_MAX):
            continue
        if not r.bets:
            continue
        coh = cohort_of(ep)
        pools_wei = compute_pool_amounts_wei(bets=r.bets)
        bull_bnb = pools_wei.bull_wei / BNB_WEI
        bear_bnb = pools_wei.bear_wei / BNB_WEI
        total = bull_bnb + bear_bnb
        if total <= 0:
            continue
        per_cohort[coh]["bull_pool_ratio"].append(bull_bnb / total)
        per_cohort[coh]["n_bets"].append(float(len(r.bets)))
        per_cohort[coh]["total_bnb"].append(total)

    summary = {c: {} for c in COHORT_ORDER}
    for c, d in per_cohort.items():
        for feature, values in d.items():
            summary[c][feature] = summarize(values)
    return {"summary": summary, "raw": per_cohort}


# ============================================================
# Main
# ============================================================

def main():
    t_all = time.time()
    print("--- loading rounds + klines ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  {len(all_rounds)} rounds", flush=True)

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1

    bnb_path = REPO / "var" / "bnb_spot_prices.jsonl"
    bnb_ext_path = EXT_DIR / "bnb_spot_prices.jsonl"

    t = time.time()
    btc_samples, btc_pipe = load_klines_for_step17(
        ipr._BTC_KLINES_PATH, ipr._EXT_BTC_KLINES_PATH,
        want_pipeline_slice=True,
        pipeline_earliest_offset=earliest_offset, pipeline_latest_offset=latest_offset,
    )
    eth_samples, eth_pipe = load_klines_for_step17(
        ipr._ETH_KLINES_PATH, ipr._EXT_ETH_KLINES_PATH,
        want_pipeline_slice=True,
        pipeline_earliest_offset=earliest_offset, pipeline_latest_offset=latest_offset,
    )
    sol_samples, sol_pipe = load_klines_for_step17(
        ipr._SOL_KLINES_PATH, ipr._EXT_SOL_KLINES_PATH,
        want_pipeline_slice=True,
        pipeline_earliest_offset=earliest_offset, pipeline_latest_offset=latest_offset,
    )
    bnb_samples, _ = load_klines_for_step17(
        bnb_path, bnb_ext_path,
        want_pipeline_slice=False,
        pipeline_earliest_offset=earliest_offset, pipeline_latest_offset=latest_offset,
    )
    print(f"  klines: BTC samples={len(btc_samples)} pipe={len(btc_pipe)} "
          f"ETH samples={len(eth_samples)} pipe={len(eth_pipe)} "
          f"SOL samples={len(sol_samples)} pipe={len(sol_pipe)} "
          f"BNB samples={len(bnb_samples)} ({time.time()-t:.1f}s)", flush=True)

    samples_by_asset = {"BTC": btc_samples, "ETH": eth_samples, "SOL": sol_samples, "BNB": bnb_samples}

    # Pre-build epoch_to_ts and cohort_ts_ranges
    epoch_to_ts = {int(r.epoch): int(r.start_at) for r in all_rounds}
    cohort_ts_ranges = {}
    for c, lo, hi in COHORT_DEFS:
        ts_in_cohort = [epoch_to_ts.get(e) for e in range(lo, hi + 1)
                        if e in epoch_to_ts]
        if ts_in_cohort:
            cohort_ts_ranges[c] = (min(ts_in_cohort), max(ts_in_cohort))

    # --- F1: 1-min returns per cohort per asset; autocorr ---
    print("\n--- F1: 1-min returns + autocorrelation ---", flush=True)
    t = time.time()
    returns_by_asset_cohort = build_1min_returns_per_cohort(samples_by_asset, all_rounds)
    print(f"  Built 1-min returns ({time.time()-t:.1f}s). Sample sizes:", flush=True)
    for a in ASSETS:
        sizes = {c: len(returns_by_asset_cohort[a][c]) for c in COHORT_ORDER}
        print(f"    {a}: {sizes}", flush=True)

    f1_autocorr = {}
    for asset in ASSETS:
        f1_autocorr[asset] = {}
        for coh in COHORT_ORDER:
            rets = returns_by_asset_cohort[asset][coh]
            f1_autocorr[asset][coh] = {
                "n": len(rets),
                "lag_1": autocorrelation_at_lag(rets, 1),
                "lag_5": autocorrelation_at_lag(rets, 5),
                "lag_15": autocorrelation_at_lag(rets, 15),
                "lag_60": autocorrelation_at_lag(rets, 60),
            }
    print(f"  Autocorrelation computed", flush=True)

    # --- F2: cross-asset correlations (hourly rolling 24h) ---
    print("\n--- F2: cross-asset correlations (hourly rolling 24h) ---", flush=True)
    t = time.time()
    # Build global 1-min timelines for ASSETS, then returns
    timelines = build_global_1min_timeline(samples_by_asset)
    asset_returns_for_f2 = {a: build_returns_from_timeline(timelines[a]) for a in ASSETS}
    f2_by_cohort = cross_asset_correlations_hourly(asset_returns_for_f2, cohort_ts_ranges)
    f2_summary = {c: summarize(vals) for c, vals in f2_by_cohort.items()}
    print(f"  F2 computed in {time.time()-t:.1f}s. Sample sizes:", flush=True)
    for c in COHORT_ORDER:
        print(f"    {c}: n={f2_summary[c]['n']} mean={f2_summary[c].get('mean')}", flush=True)

    # --- F3: vol term structure ---
    print("\n--- F3: vol term structure (sigma_1h / sigma_24h, BTC) ---", flush=True)
    t = time.time()
    f3_by_cohort = vol_term_structure_hourly(asset_returns_for_f2, cohort_ts_ranges)
    f3_summary = {c: summarize(vals) for c, vals in f3_by_cohort.items()}
    print(f"  F3 computed in {time.time()-t:.1f}s. Sample sizes:", flush=True)
    for c in COHORT_ORDER:
        print(f"    {c}: n={f3_summary[c]['n']} mean={f3_summary[c].get('mean')}", flush=True)

    # --- F4: time-of-day / dow (requires bet stream) ---
    print("\n--- F4: baseline backtest for time-of-day ---", flush=True)
    t = time.time()
    btc_klines_for_pipe = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                                     max_lookback=max_lookback,
                                                     earliest_offset=earliest_offset)
                            for ep, kl in btc_pipe.items()}
    eth_klines_for_pipe = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                                     max_lookback=max_lookback,
                                                     earliest_offset=earliest_offset)
                            for ep, kl in eth_pipe.items()}
    sol_klines_for_pipe = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                                     max_lookback=max_lookback,
                                                     earliest_offset=earliest_offset)
                            for ep, kl in sol_pipe.items()}
    bet_records = run_baseline_for_bets(all_rounds, btc_klines_for_pipe,
                                          eth_klines_for_pipe, sol_klines_for_pipe,
                                          earliest_offset)
    print(f"  Baseline backtest: {len(bet_records)} bets ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    f4 = f4_time_of_day_analysis(bet_records, all_rounds)
    print(f"  F4 grouped ({time.time()-t:.1f}s)", flush=True)

    # --- F5: pool features ---
    print("\n--- F5: round-level pool features ---", flush=True)
    t = time.time()
    f5 = f5_pool_features(all_rounds)
    print(f"  F5 computed ({time.time()-t:.1f}s)", flush=True)

    # ============================================================
    # Statistical tests: extension vs CV5
    # ============================================================
    print("\n--- Statistical tests: extension vs CV5 ---", flush=True)

    test_results: list[dict[str, Any]] = []

    # F1: per asset per lag
    for asset in ASSETS:
        for lag_label, lag_n in [("lag_1", 1), ("lag_5", 5), ("lag_15", 15), ("lag_60", 60)]:
            ext_rets = returns_by_asset_cohort[asset]["extension"]
            cv5_rets = returns_by_asset_cohort[asset]["cv5"]
            if len(ext_rets) < lag_n + 2 or len(cv5_rets) < lag_n + 2:
                continue
            # We're comparing autocorrelation, but autocorrelation is a scalar per cohort.
            # The proper test compares the distribution of return PAIRS.
            # For simplicity: build per-cohort lag-N return pairs, compute Cohen's d
            # on the PRODUCT of consecutive returns (a proxy for autocorr signal).
            ext_pairs = [ext_rets[i] * ext_rets[i + lag_n] for i in range(len(ext_rets) - lag_n)]
            cv5_pairs = [cv5_rets[i] * cv5_rets[i + lag_n] for i in range(len(cv5_rets) - lag_n)]
            d = cohens_d(ext_pairs, cv5_pairs)
            p = perm_mann_whitney_p(ext_pairs, cv5_pairs, n_seeds=PERM_SEEDS)
            test_results.append({
                "feature": f"F1_{asset}_autocorr_{lag_label}",
                "ext_mean": f1_autocorr[asset]["extension"][lag_label],
                "cv5_mean": f1_autocorr[asset]["cv5"][lag_label],
                "cohens_d": d,
                "p_value": p,
            })

    # F2: cross-asset correlation mean
    ext_f2 = f2_by_cohort["extension"]
    cv5_f2 = f2_by_cohort["cv5"]
    if len(ext_f2) >= 2 and len(cv5_f2) >= 2:
        test_results.append({
            "feature": "F2_cross_asset_corr_mean",
            "ext_mean": float(np.mean(ext_f2)),
            "cv5_mean": float(np.mean(cv5_f2)),
            "cohens_d": cohens_d(ext_f2, cv5_f2),
            "p_value": perm_mann_whitney_p(ext_f2, cv5_f2, n_seeds=PERM_SEEDS),
        })

    # F3: vol term structure
    ext_f3 = f3_by_cohort["extension"]
    cv5_f3 = f3_by_cohort["cv5"]
    if len(ext_f3) >= 2 and len(cv5_f3) >= 2:
        test_results.append({
            "feature": "F3_vol_term_ratio",
            "ext_mean": float(np.mean(ext_f3)),
            "cv5_mean": float(np.mean(cv5_f3)),
            "cohens_d": cohens_d(ext_f3, cv5_f3),
            "p_value": perm_mann_whitney_p(ext_f3, cv5_f3, n_seeds=PERM_SEEDS),
        })

    # F5: pool features
    for feature in ("bull_pool_ratio", "n_bets", "total_bnb"):
        ext_vals = f5["raw"]["extension"][feature]
        cv5_vals = f5["raw"]["cv5"][feature]
        if len(ext_vals) >= 2 and len(cv5_vals) >= 2:
            test_results.append({
                "feature": f"F5_{feature}",
                "ext_mean": float(np.mean(ext_vals)),
                "cv5_mean": float(np.mean(cv5_vals)),
                "cohens_d": cohens_d(ext_vals, cv5_vals),
                "p_value": perm_mann_whitney_p(ext_vals, cv5_vals, n_seeds=PERM_SEEDS),
            })

    # Sort by |Cohen's d|
    test_results_sorted = sorted(test_results, key=lambda x: -abs(x["cohens_d"]))
    print("\n  TOP 10 features by |Cohen's d|:", flush=True)
    print(f"    {'feature':>32s} {'ext_mean':>12s} {'cv5_mean':>12s} {'cohens_d':>10s} {'p_value':>10s}", flush=True)
    for tr in test_results_sorted[:10]:
        ext_m = tr["ext_mean"] if tr["ext_mean"] is not None else 0.0
        cv5_m = tr["cv5_mean"] if tr["cv5_mean"] is not None else 0.0
        print(f"    {tr['feature']:>32s} {ext_m:>+12.6f} {cv5_m:>+12.6f} "
              f"{tr['cohens_d']:>+10.4f} {tr['p_value']:>10.4f}", flush=True)

    # Save
    # Strip large raw lists from F5 before save
    f5_save = {"summary": f5["summary"]}

    out_path = REPO / "var" / "strategy_review" / "feature_characterization_step17_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "cohort_defs": [list(c) for c in COHORT_DEFS],
                "permutation_seeds": PERM_SEEDS,
            },
            "F1_autocorrelation": f1_autocorr,
            "F2_cross_asset_corr_summary": f2_summary,
            "F3_vol_term_summary": f3_summary,
            "F4_time_of_day": f4,
            "F5_pool_features": f5_save,
            "test_results_sorted": test_results_sorted,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
