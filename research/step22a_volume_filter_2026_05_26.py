"""Step 22a — backtest the volume-Z filter from Step 21b finding.

Point-in-time Z: at decision day D, use vol[D-1] standardized against
trailing window vol[D-N..D-1]. No lookahead. Test N in {30, 60, 90}.

Filter families:
  A: skip if btc_vol_z_30d > X  for X in {0, 0.25, 0.5, 0.75, 1.0, 1.25}
  B: skip if bnb_vol_z_30d > X  same thresholds
  C: skip if btc_vol_z_30d > X OR bnb_vol_z_30d > X  same thresholds
Lookback sensitivity: best A-family threshold at N=60, N=90.
Scale check: best A-family threshold also at 50 BNB.

CoinGecko pull window: 2025-07-01 to 2026-05-26 (covers 90d lookback +
backtest range). Reuse Step 21b's pull if available — re-fetch otherwise.

Permutation null on best 5 BNB variant: 1000 seeds, within-cohort label
permutation, same methodology as Steps 15/16/19.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore
import requests  # type: ignore

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
from pancakebot.constants import MAX_GAS_COST_BET_BNB  # noqa: E402
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
DRAWDOWN_PEAK_WINDOW_DAYS = 7
COOLDOWN_ROUNDS = 72
ABS_DD_FRAC = 0.15
PERMUTATION_SEEDS = 1000

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

# CoinGecko pull window: extend back to July 2025 to support N=90 lookback
COINGECKO_FROM = datetime(2025, 7, 1, tzinfo=timezone.utc)
COINGECKO_TO = datetime(2026, 5, 26, tzinfo=timezone.utc)

THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
LOOKBACKS_TO_TEST = (30, 60, 90)
DEFAULT_LOOKBACK = 30


def cohort_of(epoch: int) -> str:
    for name, lo, hi in COHORT_DEFS:
        if lo <= epoch <= hi:
            return name
    return "unknown"


# ============================================================
# Tracker
# ============================================================

class Step22Tracker(InMemoryBankrollTracker):
    def __init__(self, *, initial_bankroll, drawdown_peak_window_days, peak_mode,
                  cooldown_rounds, abs_dd_frac):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cd_total = int(cooldown_rounds)
        self._abs_dd_frac = float(abs_dd_frac)
        self.n_pauses_fired = 0

    def is_paused(self, as_of_start_at):
        if self._cooldown > 0:
            return True
        current = self.current_bankroll()
        peak = self.peak_bankroll(as_of_start_at)
        if peak > 0:
            dd = (peak - current) / peak
            if dd >= self._abs_dd_frac:
                if self._cd_total > 0:
                    self.set_paused(self._cd_total + 1, as_of_start_at)
                self.n_pauses_fired += 1
                return self._cd_total > 0
        return False


# ============================================================
# CoinGecko pull
# ============================================================

def fetch_coingecko_market_chart(coin_id, from_ts, to_ts, retries=3):
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
           f"?vs_currency=usd&from={from_ts}&to={to_ts}")
    for i in range(retries):
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                return r.json()
            print(f"    {coin_id}: HTTP {r.status_code} attempt {i+1}", flush=True)
            time.sleep(5)
        except Exception as e:
            print(f"    {coin_id}: {e!r} attempt {i+1}", flush=True)
            time.sleep(5)
    return None


def daily_series_from_chart(chart, key="total_volumes"):
    if not chart or key not in chart:
        return {}
    out = {}
    for ts_ms, v in chart[key]:
        d_iso = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()
        out[d_iso] = float(v)
    return out


# ============================================================
# Point-in-time Z-score (trailing window, no lookahead)
# ============================================================

def build_trailing_z_per_round(daily_volume_series, all_rounds, lookback_days):
    """For each round, compute Z = (vol[D-1] - mean(vol[D-N..D-1])) / stdev(...).

    Returns {epoch: z_value or None}.
    """
    # Sort dates
    sorted_dates = sorted(daily_volume_series.keys())
    date_to_pos = {d: i for i, d in enumerate(sorted_dates)}
    vol_arr = np.array([daily_volume_series[d] for d in sorted_dates], dtype=float)

    out = {}
    for r in all_rounds:
        ep = int(r.epoch)
        if not (EPOCH_MIN <= ep <= EPOCH_MAX):
            continue
        round_dt = datetime.fromtimestamp(int(r.start_at), tz=timezone.utc)
        d_minus_1 = (round_dt.date() - timedelta(days=1)).isoformat()
        if d_minus_1 not in date_to_pos:
            out[ep] = None
            continue
        end_idx = date_to_pos[d_minus_1]
        start_idx = end_idx - (lookback_days - 1)
        if start_idx < 0:
            out[ep] = None
            continue
        window = vol_arr[start_idx:end_idx + 1]
        target = float(vol_arr[end_idx])
        mean = float(window.mean())
        stdev = float(window.std(ddof=1)) if len(window) > 1 else 0.0
        if stdev == 0:
            out[ep] = 0.0
        else:
            out[ep] = (target - mean) / stdev
    return out


# ============================================================
# Backtest with volume filter
# ============================================================

def run_backtest(*, initial_bankroll, all_rounds, btc_klines, eth_klines, sol_klines,
                  btc_z_per_round, bnb_z_per_round,
                  filter_mode="none", threshold=None, label="baseline"):
    """filter_mode: 'none' | 'btc' | 'bnb' | 'or' """
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
    tracker = Step22Tracker(
        initial_bankroll=initial_bankroll,
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
    per_cohort = {c: {"n_rounds": 0, "n_bets": 0, "n_wins": 0, "pnl_bnb": 0.0,
                       "n_filter_vetoed": 0,
                       "btc_z_sum": 0.0, "btc_z_n": 0} for c in COHORT_ORDER}
    bankroll = float(initial_bankroll); peak = bankroll; max_dd_frac = 0.0
    bet_records = []
    n_filter_vetoes = 0

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1
        z_btc = btc_z_per_round.get(ep)
        z_bnb = bnb_z_per_round.get(ep)
        if z_btc is not None:
            per_cohort[coh]["btc_z_sum"] += z_btc
            per_cohort[coh]["btc_z_n"] += 1

        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Volume-Z filter
        skip = False
        if filter_mode == "btc" and z_btc is not None and z_btc > threshold:
            skip = True
        elif filter_mode == "bnb" and z_bnb is not None and z_bnb > threshold:
            skip = True
        elif filter_mode == "or":
            if (z_btc is not None and z_btc > threshold) or \
               (z_bnb is not None and z_bnb > threshold):
                skip = True
        if skip:
            n_filter_vetoes += 1
            per_cohort[coh]["n_filter_vetoed"] += 1
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
        per_cohort[coh]["n_bets"] += 1
        per_cohort[coh]["pnl_bnb"] += profit
        if outcome.outcome == "win":
            per_cohort[coh]["n_wins"] += 1
        bet_records.append({"epoch": ep, "cohort": coh, "profit": profit,
                             "won": outcome.outcome == "win"})

        if bankroll > peak: peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac: max_dd_frac = dd
        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])

    for cd in per_cohort.values():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
        cd["btc_z_mean"] = cd["btc_z_sum"] / cd["btc_z_n"] if cd["btc_z_n"] else 0.0
    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())
    return {
        "label": label,
        "filter_mode": filter_mode, "threshold": threshold,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - initial_bankroll,
            "final_bankroll": bankroll,
        },
        "max_drawdown_frac": max_dd_frac,
        "n_filter_vetoes": n_filter_vetoes,
        "per_cohort": per_cohort,
        "bet_records": bet_records,
    }


# ============================================================
# Permutation null
# ============================================================

def permutation_null(*, bets_candidate, bets_baseline, n_seeds, base_seed=42):
    cohort_bets = {}
    for b in bets_candidate:
        c = b["cohort"]
        cohort_bets.setdefault(c, {"cand": [], "base": []})["cand"].append(b["profit"])
    for b in bets_baseline:
        c = b["cohort"]
        cohort_bets.setdefault(c, {"cand": [], "base": []})["base"].append(b["profit"])

    obs_D = sum(b["profit"] for b in bets_candidate) - sum(b["profit"] for b in bets_baseline)
    rng = random.Random(base_seed)
    perm_Ds = []
    for _ in range(n_seeds):
        perm_D = 0.0
        for coh, d in cohort_bets.items():
            pool = d["cand"] + d["base"]
            n_cand = len(d["cand"]); n_base = len(d["base"])
            if not pool: continue
            rng.shuffle(pool)
            perm_D += sum(pool[:n_cand]) - sum(pool[n_cand:n_cand + n_base])
        perm_Ds.append(perm_D)
    perm_Ds_sorted = sorted(perm_Ds)
    n_geq = sum(1 for d in perm_Ds if d >= obs_D)
    return {
        "observed_D": obs_D, "n_seeds": n_seeds, "p_value": n_geq / n_seeds,
        "perm_D_mean": statistics.mean(perm_Ds),
        "perm_D_stdev": statistics.stdev(perm_Ds) if len(perm_Ds) > 1 else 0.0,
        "perm_D_p05": perm_Ds_sorted[int(0.05 * n_seeds)],
        "perm_D_p50": perm_Ds_sorted[int(0.50 * n_seeds)],
        "perm_D_p95": perm_Ds_sorted[int(0.95 * n_seeds)],
        "perm_D_p99": perm_Ds_sorted[int(0.99 * n_seeds)],
    }


# ============================================================
# Main
# ============================================================

def main():
    t_all = time.time()

    # ----- CoinGecko pull -----
    print("--- CoinGecko volume pulls (Jul 2025 - May 2026) ---", flush=True)
    t = time.time()
    bnb_chart = fetch_coingecko_market_chart(
        "binancecoin", int(COINGECKO_FROM.timestamp()), int(COINGECKO_TO.timestamp()))
    time.sleep(1.5)
    btc_chart = fetch_coingecko_market_chart(
        "bitcoin", int(COINGECKO_FROM.timestamp()), int(COINGECKO_TO.timestamp()))
    print(f"  CoinGecko done ({time.time()-t:.1f}s)", flush=True)
    bnb_vol_series = daily_series_from_chart(bnb_chart, "total_volumes")
    btc_vol_series = daily_series_from_chart(btc_chart, "total_volumes")
    print(f"  BNB daily volumes: {len(bnb_vol_series)} points  "
          f"({min(bnb_vol_series)} .. {max(bnb_vol_series)})", flush=True)
    print(f"  BTC daily volumes: {len(btc_vol_series)} points  "
          f"({min(btc_vol_series)} .. {max(btc_vol_series)})", flush=True)

    # ----- Load rounds + klines -----
    print("\n--- loading rounds + klines ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  {len(all_rounds)} rounds", flush=True)

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1
    t_kl = time.time()
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  klines loaded ({time.time()-t_kl:.1f}s)", flush=True)
    btc_klines = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                  for ep, kl in btc.items()}
    eth_klines = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                  for ep, kl in eth.items()}
    sol_klines = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                  for ep, kl in sol.items()}

    # ----- Build point-in-time Z per round -----
    print("\n--- building point-in-time Z per round (N=30) ---", flush=True)
    t = time.time()
    btc_z_30 = build_trailing_z_per_round(btc_vol_series, all_rounds, 30)
    bnb_z_30 = build_trailing_z_per_round(bnb_vol_series, all_rounds, 30)
    n_z_valid_btc = sum(1 for v in btc_z_30.values() if v is not None)
    n_z_valid_bnb = sum(1 for v in bnb_z_30.values() if v is not None)
    print(f"  btc_z_30: {n_z_valid_btc} valid / {len(btc_z_30)} total", flush=True)
    print(f"  bnb_z_30: {n_z_valid_bnb} valid / {len(bnb_z_30)} total ({time.time()-t:.1f}s)", flush=True)

    # ----- Run baseline -----
    print("\n--- baseline backtest (no filter) at 5 BNB ---", flush=True)
    t = time.time()
    baseline = run_backtest(
        initial_bankroll=5.0, all_rounds=all_rounds,
        btc_klines=btc_klines, eth_klines=eth_klines, sol_klines=sol_klines,
        btc_z_per_round=btc_z_30, bnb_z_per_round=bnb_z_30,
        filter_mode="none", label="baseline_5bnb",
    )
    s = baseline["summary"]
    print(f"  baseline: pnl={s['net_pnl_bnb']:+.4f} bets={s['num_bets']} "
          f"wr={s['win_rate']*100:.2f}% max_dd={baseline['max_drawdown_frac']*100:.2f}% "
          f"({time.time()-t:.1f}s)", flush=True)

    # ----- Run filter families -----
    print("\n--- Filter families at 5 BNB (N=30) ---", flush=True)
    results: dict[str, dict] = {"baseline": baseline}
    for fam_mode, fam_label in [("btc", "A_btc"), ("bnb", "B_bnb"), ("or", "C_or")]:
        for thr in THRESHOLDS:
            label = f"{fam_label}_thr{thr}"
            t = time.time()
            r = run_backtest(
                initial_bankroll=5.0, all_rounds=all_rounds,
                btc_klines=btc_klines, eth_klines=eth_klines, sol_klines=sol_klines,
                btc_z_per_round=btc_z_30, bnb_z_per_round=bnb_z_30,
                filter_mode=fam_mode, threshold=thr, label=label,
            )
            s = r["summary"]
            delta = s["net_pnl_bnb"] - baseline["summary"]["net_pnl_bnb"]
            print(f"  {label:>20s}: pnl={s['net_pnl_bnb']:+.4f} delta={delta:+.4f} "
                  f"bets={s['num_bets']} vetoes={r['n_filter_vetoes']} "
                  f"wr={s['win_rate']*100:.2f}% max_dd={r['max_drawdown_frac']*100:.2f}% "
                  f"({time.time()-t:.1f}s)", flush=True)
            results[label] = r

    # ----- Identify best A-family variant -----
    a_variants = [(k, v) for k, v in results.items() if k.startswith("A_btc")]
    best_a_name, best_a = max(a_variants, key=lambda kv: kv[1]["summary"]["net_pnl_bnb"])
    print(f"\n  best A-family: {best_a_name} (pnl={best_a['summary']['net_pnl_bnb']:+.4f})", flush=True)

    # Best overall variant across all families
    all_variants = [(k, v) for k, v in results.items() if k != "baseline"]
    best_overall_name, best_overall = max(all_variants, key=lambda kv: kv[1]["summary"]["net_pnl_bnb"])
    print(f"  best overall: {best_overall_name} (pnl={best_overall['summary']['net_pnl_bnb']:+.4f})", flush=True)

    # ----- Lookback sensitivity (N=60, 90) on best A threshold -----
    best_a_thr = best_a["threshold"]
    print(f"\n--- Lookback sensitivity (N=60, 90) at best A threshold ({best_a_thr}) ---", flush=True)
    lookback_results = {}
    for N in (60, 90):
        t = time.time()
        btc_z_N = build_trailing_z_per_round(btc_vol_series, all_rounds, N)
        bnb_z_N = build_trailing_z_per_round(bnb_vol_series, all_rounds, N)
        r = run_backtest(
            initial_bankroll=5.0, all_rounds=all_rounds,
            btc_klines=btc_klines, eth_klines=eth_klines, sol_klines=sol_klines,
            btc_z_per_round=btc_z_N, bnb_z_per_round=bnb_z_N,
            filter_mode="btc", threshold=best_a_thr, label=f"A_btc_thr{best_a_thr}_N{N}",
        )
        s = r["summary"]
        delta = s["net_pnl_bnb"] - baseline["summary"]["net_pnl_bnb"]
        print(f"  N={N}: pnl={s['net_pnl_bnb']:+.4f} delta={delta:+.4f} "
              f"bets={s['num_bets']} vetoes={r['n_filter_vetoes']} "
              f"({time.time()-t:.1f}s)", flush=True)
        lookback_results[f"N_{N}"] = r

    # ----- 50 BNB scale check on best A threshold -----
    print(f"\n--- 50 BNB scale (best A threshold {best_a_thr}, N=30) ---", flush=True)
    t = time.time()
    baseline_50 = run_backtest(
        initial_bankroll=50.0, all_rounds=all_rounds,
        btc_klines=btc_klines, eth_klines=eth_klines, sol_klines=sol_klines,
        btc_z_per_round=btc_z_30, bnb_z_per_round=bnb_z_30,
        filter_mode="none", label="baseline_50bnb",
    )
    s = baseline_50["summary"]
    print(f"  baseline 50BNB: pnl={s['net_pnl_bnb']:+.4f} bets={s['num_bets']} "
          f"max_dd={baseline_50['max_drawdown_frac']*100:.2f}% ({time.time()-t:.1f}s)", flush=True)
    t = time.time()
    filt_50 = run_backtest(
        initial_bankroll=50.0, all_rounds=all_rounds,
        btc_klines=btc_klines, eth_klines=eth_klines, sol_klines=sol_klines,
        btc_z_per_round=btc_z_30, bnb_z_per_round=bnb_z_30,
        filter_mode="btc", threshold=best_a_thr, label=f"A_btc_thr{best_a_thr}_50bnb",
    )
    s = filt_50["summary"]
    delta_50 = s["net_pnl_bnb"] - baseline_50["summary"]["net_pnl_bnb"]
    print(f"  filter 50BNB:   pnl={s['net_pnl_bnb']:+.4f} delta={delta_50:+.4f} "
          f"bets={s['num_bets']} vetoes={filt_50['n_filter_vetoes']} "
          f"max_dd={filt_50['max_drawdown_frac']*100:.2f}% ({time.time()-t:.1f}s)", flush=True)

    # ----- Per-cohort breakdown of best 5 BNB variant -----
    print(f"\n--- Per-cohort breakdown: {best_overall_name} @ 5 BNB ---", flush=True)
    print(f"  {'cohort':>30s} {'rounds':>7s} {'bets':>5s} {'vetoed':>7s} {'WR':>7s} {'PnL':>10s} {'btc_z_mn':>9s}", flush=True)
    for c in COHORT_ORDER:
        bc = best_overall["per_cohort"][c]
        bs = baseline["per_cohort"][c]
        print(f"  {c:>30s} {bc['n_rounds']:>7d} {bc['n_bets']:>5d} {bc['n_filter_vetoed']:>7d} "
              f"{bc['win_rate']*100:>6.2f}% {bc['pnl_bnb']:>+10.4f} {bc['btc_z_mean']:>+9.4f}", flush=True)
    print(f"  baseline cohort PnL for comparison:", flush=True)
    for c in COHORT_ORDER:
        bs = baseline["per_cohort"][c]
        print(f"    {c:>28s}: bets={bs['n_bets']:>5d}  pnl={bs['pnl_bnb']:>+10.4f}", flush=True)

    # ----- Permutation null on best overall 5 BNB variant -----
    print(f"\n--- Permutation null on {best_overall_name} ({PERMUTATION_SEEDS} seeds) ---", flush=True)
    t = time.time()
    null = permutation_null(
        bets_candidate=best_overall["bet_records"],
        bets_baseline=baseline["bet_records"],
        n_seeds=PERMUTATION_SEEDS,
    )
    print(f"  Observed D: {null['observed_D']:+.4f}", flush=True)
    print(f"  Null mean: {null['perm_D_mean']:+.4f}  stdev: {null['perm_D_stdev']:.4f}", flush=True)
    print(f"  Null p05/p50/p95/p99: {null['perm_D_p05']:+.4f} / {null['perm_D_p50']:+.4f} / "
          f"{null['perm_D_p95']:+.4f} / {null['perm_D_p99']:+.4f}", flush=True)
    print(f"  p-value: {null['p_value']:.4f}", flush=True)
    print(f"  ({time.time()-t:.1f}s)", flush=True)

    # ----- Save -----
    def strip(r):
        return {k: v for k, v in r.items() if k != "bet_records"}
    out_path = REPO / "var" / "strategy_review" / "step22a_volume_filter_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "thresholds": list(THRESHOLDS),
                "lookbacks_to_test": list(LOOKBACKS_TO_TEST),
                "default_lookback": DEFAULT_LOOKBACK,
                "permutation_seeds": PERMUTATION_SEEDS,
            },
            "results_5bnb": {k: strip(v) for k, v in results.items()},
            "lookback_results": {k: strip(v) for k, v in lookback_results.items()},
            "scale_50bnb": {
                "baseline": strip(baseline_50),
                "filter": strip(filt_50),
                "delta_50": delta_50,
            },
            "best_a_threshold": best_a_thr,
            "best_overall_name": best_overall_name,
            "permutation_null_best": null,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
