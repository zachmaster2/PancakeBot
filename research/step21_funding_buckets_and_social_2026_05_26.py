"""Step 21 — per-bet funding buckets + CoinGecko social/activity proxy.

21a: Per-bet funding-rate bucket analysis.
     Reuse Step 20's OKX BNB-USDT-SWAP funding data (3-month coverage).
     Run canonical baseline backtest (cd=72, dd=0.15, 5 BNB).
     For each bet with funding coverage, bucket by:
       - signed funding rate (5 quintiles)
       - abs(funding rate) (5 quintiles, tests "extreme funding = bad" hypothesis)
     Per-bucket: n_bets, WR, mean PnL, total PnL, 95% bootstrap CI on per-bet PnL.
     Also: direction-conditional split (BULL vs BEAR).

21b: CoinGecko daily volumes (substitute for blocked CryptoCompare social).
     Free, no-auth endpoint /coins/{id}/market_chart/range covers full
     backtest window (Oct 2025 - May 2026). 237 daily points per coin.
     Pulls: BNB + BTC daily volumes. Computes per-cohort mean Z-score.
     Cohen's d extension vs CV5.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
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
INITIAL_BANKROLL = 5.0
N_BUCKETS = 5
BOOTSTRAP_SEEDS = 1000

BACKTEST_START = datetime(2025, 10, 1, tzinfo=timezone.utc)
BACKTEST_END = datetime(2026, 5, 26, tzinfo=timezone.utc)

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


def cohort_of(epoch: int) -> str:
    for name, lo, hi in COHORT_DEFS:
        if lo <= epoch <= hi:
            return name
    return "unknown"


# ============================================================
# Tracker (gate-validated)
# ============================================================

class Step21Tracker(InMemoryBankrollTracker):
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


def run_baseline(all_rounds, btc_klines, eth_klines, sol_klines):
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
    tracker = Step21Tracker(
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
            "cohort": cohort_of(int(round_t.epoch)),
            "side": side,
            "won": outcome.outcome == "win",
            "profit": profit,
            "start_at_s": int(round_t.start_at),
            "lock_at_ms": int(round_t.start_at) * 1000 + 300_000,
        })
        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])
    return bet_records, bankroll - INITIAL_BANKROLL


# ============================================================
# 21a: Funding bucket analysis
# ============================================================

def bootstrap_ci_mean(values, n_boot=BOOTSTRAP_SEEDS, alpha=0.05, seed=42):
    if not values:
        return (None, None)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    means.sort()
    lo_idx = int(alpha / 2 * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot)
    return (float(means[lo_idx]), float(means[hi_idx]))


def attach_funding_to_bets(bet_records, funding_rows):
    """For each bet, lookup most-recent funding rate at lock_at_ms (the
    decision moment). Bets without funding coverage get funding_rate=None."""
    if not funding_rows:
        for b in bet_records:
            b["funding_rate"] = None
            b["abs_funding_rate"] = None
        return
    ts_arr = np.array([r["ts_ms"] for r in funding_rows], dtype=np.int64)
    rate_arr = np.array([r["rate"] for r in funding_rows], dtype=float)
    for b in bet_records:
        idx = int(np.searchsorted(ts_arr, b["lock_at_ms"], side="right")) - 1
        if idx < 0:
            b["funding_rate"] = None
            b["abs_funding_rate"] = None
        else:
            b["funding_rate"] = float(rate_arr[idx])
            b["abs_funding_rate"] = float(abs(rate_arr[idx]))


def quintile_buckets(values):
    """Compute quintile boundaries (4 cuts → 5 buckets) on a sorted array."""
    arr = np.asarray(values, dtype=float)
    return [float(np.quantile(arr, q)) for q in (0.2, 0.4, 0.6, 0.8)]


def bucket_index(value, cuts):
    """0..4 inclusive."""
    i = 0
    for c in cuts:
        if value <= c:
            return i
        i += 1
    return i  # 4


def bucket_analysis(bets, key, direction_split=False):
    """Returns dict of per-bucket stats. key='funding_rate' or 'abs_funding_rate'."""
    rated = [b for b in bets if b.get(key) is not None]
    if len(rated) < 5:
        return {"buckets": [], "n_rated": len(rated), "cuts": None}
    cuts = quintile_buckets([b[key] for b in rated])
    buckets = [{"i": i, "lo": None, "hi": None, "bets": []} for i in range(5)]
    # Bucket bounds
    sorted_vals = sorted(b[key] for b in rated)
    for i in range(5):
        if i == 0:
            buckets[i]["lo"] = sorted_vals[0]
            buckets[i]["hi"] = cuts[0]
        elif i == 4:
            buckets[i]["lo"] = cuts[3]
            buckets[i]["hi"] = sorted_vals[-1]
        else:
            buckets[i]["lo"] = cuts[i - 1]
            buckets[i]["hi"] = cuts[i]
    for b in rated:
        bi = bucket_index(b[key], cuts)
        buckets[bi]["bets"].append(b)

    out = []
    for bk in buckets:
        bets_in = bk["bets"]
        n = len(bets_in)
        if n == 0:
            out.append({"i": bk["i"], "lo": bk["lo"], "hi": bk["hi"],
                         "n": 0, "wr": None, "mean_pnl": None,
                         "ci_lo": None, "ci_hi": None, "total_pnl": 0.0})
            continue
        wr = sum(1 for b in bets_in if b["won"]) / n
        profits = [b["profit"] for b in bets_in]
        mean_pnl = float(np.mean(profits))
        ci_lo, ci_hi = bootstrap_ci_mean(profits)
        total_pnl = float(sum(profits))
        entry = {"i": bk["i"], "lo": bk["lo"], "hi": bk["hi"],
                  "n": n, "wr": wr, "mean_pnl": mean_pnl,
                  "ci_lo": ci_lo, "ci_hi": ci_hi, "total_pnl": total_pnl}
        if direction_split:
            for d in ("bull", "bear"):
                sub = [b for b in bets_in if b["side"].lower() == d]
                if sub:
                    sub_profits = [b["profit"] for b in sub]
                    entry[f"n_{d}"] = len(sub)
                    entry[f"wr_{d}"] = sum(1 for b in sub if b["won"]) / len(sub)
                    entry[f"mean_pnl_{d}"] = float(np.mean(sub_profits))
                else:
                    entry[f"n_{d}"] = 0
                    entry[f"wr_{d}"] = None
                    entry[f"mean_pnl_{d}"] = None
        out.append(entry)
    return {"buckets": out, "n_rated": len(rated), "cuts": cuts}


# ============================================================
# 21b: CoinGecko daily volumes
# ============================================================

def fetch_coingecko_market_chart(coin_id):
    start_s = int(BACKTEST_START.timestamp())
    end_s = int(BACKTEST_END.timestamp())
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
           f"?vs_currency=usd&from={start_s}&to={end_s}")
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                print(f"    {coin_id}: HTTP {r.status_code} attempt {attempt+1}", flush=True)
                time.sleep(5)
                continue
            d = r.json()
            return d
        except Exception as e:
            print(f"    {coin_id}: exception {e!r} attempt {attempt+1}", flush=True)
            time.sleep(5)
    return None


def daily_series_from_chart(chart, key="total_volumes"):
    """Returns dict {date_iso: value} from CoinGecko response."""
    if not chart or key not in chart:
        return {}
    out = {}
    for ts_ms, val in chart[key]:
        d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()
        out[d] = float(val)
    return out


def assign_volume_to_rounds(daily_series, all_rounds):
    """Map round -> day's volume Z-score (per coin)."""
    if not daily_series:
        return {}
    vals = np.array(list(daily_series.values()), dtype=float)
    mean = float(vals.mean()); stdev = float(vals.std(ddof=1)) if len(vals) > 1 else 1.0
    date_to_z = {}
    for d, v in daily_series.items():
        z = (v - mean) / (stdev if stdev > 0 else 1.0)
        date_to_z[d] = {"raw": v, "z": z}
    out = {}
    for r in all_rounds:
        ep = int(r.epoch)
        if not (EPOCH_MIN <= ep <= EPOCH_MAX):
            continue
        d = datetime.fromtimestamp(int(r.start_at), tz=timezone.utc).date().isoformat()
        if d in date_to_z:
            out[ep] = date_to_z[d]
    return out


def per_cohort_means(round_to_value, key="z"):
    per_cohort = {c: [] for c in COHORT_ORDER}
    for ep, d in round_to_value.items():
        per_cohort[cohort_of(ep)].append(float(d[key]))
    out = {}
    for c, vals in per_cohort.items():
        if not vals:
            out[c] = {"n": 0, "mean": None, "stdev": None}
        else:
            arr = np.asarray(vals)
            out[c] = {"n": len(vals), "mean": float(arr.mean()),
                       "stdev": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
    return out, per_cohort


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    a_arr = np.asarray(a, dtype=float); b_arr = np.asarray(b, dtype=float)
    am, bm = float(a_arr.mean()), float(b_arr.mean())
    av, bv = float(a_arr.var(ddof=1)), float(b_arr.var(ddof=1))
    pooled = math.sqrt(((len(a) - 1) * av + (len(b) - 1) * bv) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (am - bm) / pooled


# ============================================================
# Main
# ============================================================

def main():
    t_all = time.time()

    # ----- Load Step 20's OKX funding data -----
    print("--- loading Step 20 OKX funding data ---", flush=True)
    step20_path = REPO / "var" / "strategy_review" / "step20_external_signals_data.json"
    if not step20_path.exists():
        print(f"  ERROR: {step20_path} not found", flush=True)
        sys.exit(1)
    with step20_path.open(encoding="utf-8") as f:
        step20 = json.load(f)
    funding_rows = step20.get("20C_okx_funding", {}).get("raw_funding", [])
    funding_rows.sort(key=lambda r: r["ts_ms"])
    print(f"  {len(funding_rows)} funding intervals loaded", flush=True)

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

    # ----- Run baseline backtest -----
    print("\n--- canonical baseline backtest (cd=72, dd=0.15, 5 BNB) ---", flush=True)
    t = time.time()
    bet_records, total_pnl = run_baseline(all_rounds, btc_klines, eth_klines, sol_klines)
    print(f"  bets={len(bet_records)} total_pnl={total_pnl:+.4f} ({time.time()-t:.1f}s)", flush=True)

    # ----- 21a: attach funding + bucket analysis -----
    print("\n=== 21a: per-bet funding bucket analysis ===", flush=True)
    attach_funding_to_bets(bet_records, funding_rows)
    rated = [b for b in bet_records if b.get("funding_rate") is not None]
    print(f"  bets with funding coverage: {len(rated)} / {len(bet_records)}", flush=True)

    # Cohort distribution of rated bets
    rated_by_cohort = {c: 0 for c in COHORT_ORDER}
    for b in rated:
        rated_by_cohort[b["cohort"]] += 1
    print(f"  rated bets per cohort: {rated_by_cohort}", flush=True)

    if len(rated) >= 5:
        print("\n  Signed funding_rate quintiles (Q1 most-negative .. Q5 most-positive):", flush=True)
        signed_buckets = bucket_analysis(rated, "funding_rate", direction_split=True)
        print(f"    cuts: {[f'{c:+.2e}' for c in signed_buckets['cuts']]}", flush=True)
        print(f"    {'bucket':>6s} {'lo':>10s} {'hi':>10s} {'n':>4s} {'WR':>7s} {'mean_PnL':>10s} "
              f"{'95% CI':>22s} {'tot_PnL':>9s}", flush=True)
        for bk in signed_buckets["buckets"]:
            ci_str = f"[{bk['ci_lo']:+.4f}, {bk['ci_hi']:+.4f}]" if bk['ci_lo'] is not None else "n/a"
            wr_str = f"{bk['wr']*100:.2f}%" if bk['wr'] is not None else "n/a"
            mp_str = f"{bk['mean_pnl']:+.5f}" if bk['mean_pnl'] is not None else "n/a"
            print(f"    Q{bk['i']+1:>5d} {bk['lo']:+.2e} {bk['hi']:+.2e} {bk['n']:>4d} {wr_str:>7s} "
                  f"{mp_str:>10s} {ci_str:>22s} {bk['total_pnl']:>+9.4f}", flush=True)

        print("\n  abs(funding_rate) quintiles (Q1 most-neutral .. Q5 most-extreme):", flush=True)
        abs_buckets = bucket_analysis(rated, "abs_funding_rate", direction_split=True)
        print(f"    cuts: {[f'{c:+.2e}' for c in abs_buckets['cuts']]}", flush=True)
        print(f"    {'bucket':>6s} {'lo':>10s} {'hi':>10s} {'n':>4s} {'WR':>7s} {'mean_PnL':>10s} "
              f"{'95% CI':>22s} {'tot_PnL':>9s}", flush=True)
        for bk in abs_buckets["buckets"]:
            ci_str = f"[{bk['ci_lo']:+.4f}, {bk['ci_hi']:+.4f}]" if bk['ci_lo'] is not None else "n/a"
            wr_str = f"{bk['wr']*100:.2f}%" if bk['wr'] is not None else "n/a"
            mp_str = f"{bk['mean_pnl']:+.5f}" if bk['mean_pnl'] is not None else "n/a"
            print(f"    Q{bk['i']+1:>5d} {bk['lo']:+.2e} {bk['hi']:+.2e} {bk['n']:>4d} {wr_str:>7s} "
                  f"{mp_str:>10s} {ci_str:>22s} {bk['total_pnl']:>+9.4f}", flush=True)
    else:
        signed_buckets = {"buckets": [], "n_rated": len(rated), "cuts": None}
        abs_buckets = {"buckets": [], "n_rated": len(rated), "cuts": None}

    # ----- 21b: CoinGecko volume pulls -----
    print("\n=== 21b: CoinGecko daily volumes (BNB + BTC) ===", flush=True)
    t = time.time()
    bnb_chart = fetch_coingecko_market_chart("binancecoin")
    time.sleep(1.0)
    btc_chart = fetch_coingecko_market_chart("bitcoin")
    print(f"  CoinGecko pulls done ({time.time()-t:.1f}s)", flush=True)

    bnb_vol_series = daily_series_from_chart(bnb_chart, "total_volumes")
    bnb_px_series = daily_series_from_chart(bnb_chart, "prices")
    btc_vol_series = daily_series_from_chart(btc_chart, "total_volumes")
    btc_px_series = daily_series_from_chart(btc_chart, "prices")
    print(f"  BNB: {len(bnb_vol_series)} daily volumes, {len(bnb_px_series)} prices", flush=True)
    print(f"  BTC: {len(btc_vol_series)} daily volumes, {len(btc_px_series)} prices", flush=True)

    bnb_vol_per_round = assign_volume_to_rounds(bnb_vol_series, all_rounds)
    btc_vol_per_round = assign_volume_to_rounds(btc_vol_series, all_rounds)

    bnb_vol_summary, bnb_vol_raw = per_cohort_means(bnb_vol_per_round, "z")
    btc_vol_summary, btc_vol_raw = per_cohort_means(btc_vol_per_round, "z")

    print("\n  BNB daily volume Z-score per cohort:", flush=True)
    for c in COHORT_ORDER:
        s = bnb_vol_summary[c]
        if s["n"] > 0:
            print(f"    {c:>28s}: n={s['n']:>5d} mean_z={s['mean']:+.4f} stdev_z={s['stdev']:.4f}", flush=True)
    print("\n  BTC daily volume Z-score per cohort:", flush=True)
    for c in COHORT_ORDER:
        s = btc_vol_summary[c]
        if s["n"] > 0:
            print(f"    {c:>28s}: n={s['n']:>5d} mean_z={s['mean']:+.4f} stdev_z={s['stdev']:.4f}", flush=True)

    # Cohen's d extension vs CV5
    print("\n=== Effect sizes (extension vs CV5) ===", flush=True)
    d_bnb = cohens_d(bnb_vol_raw["extension"], bnb_vol_raw["cv5"])
    d_btc = cohens_d(btc_vol_raw["extension"], btc_vol_raw["cv5"])
    print(f"  21b BNB_volume_z   extension vs CV5: d={d_bnb:+.4f}" if d_bnb is not None else
          f"  21b BNB_volume_z   d not computable", flush=True)
    print(f"  21b BTC_volume_z   extension vs CV5: d={d_btc:+.4f}" if d_btc is not None else
          f"  21b BTC_volume_z   d not computable", flush=True)

    # ----- Save -----
    def strip_bets(b):
        return {k: v for k, v in b.items() if k not in ()}
    out_path = REPO / "var" / "strategy_review" / "step21_funding_buckets_and_social_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "initial_bankroll": INITIAL_BANKROLL,
                "cooldown_rounds": COOLDOWN_ROUNDS,
                "abs_dd_frac": ABS_DD_FRAC,
                "n_buckets": N_BUCKETS,
                "bootstrap_seeds": BOOTSTRAP_SEEDS,
            },
            "baseline": {"total_pnl": total_pnl, "n_bets": len(bet_records)},
            "21a_funding_buckets": {
                "n_rated": len(rated),
                "rated_by_cohort": rated_by_cohort,
                "signed_buckets": signed_buckets,
                "abs_buckets": abs_buckets,
            },
            "21b_coingecko": {
                "bnb_vol_per_cohort": bnb_vol_summary,
                "btc_vol_per_cohort": btc_vol_summary,
                "d_bnb_volume_z_ext_vs_cv5": d_bnb,
                "d_btc_volume_z_ext_vs_cv5": d_btc,
            },
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
