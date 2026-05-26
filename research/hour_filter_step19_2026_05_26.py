"""Step 19 — F4 hour-of-day filter test at 5 BNB.

6 variants vs canonical baseline (cd=72, dd=0.15) @ 5 BNB:
  v1: skip_h16_h17       — veto bets during UTC hours 16, 17 (negative PnL)
  v2: skip_h16_only      — veto only hour 16 (worst)
  v3: skip_h16_h17_h18   — extend to include h18 (marginal +1.76 BNB)
  v4: keep_h11_only      — opposite test: bet ONLY hour 11 (+9.11, 68% WR)
  v5: keep_top_4_hours   — bet only during hours 11, 19, 15, 9
  v6: skip_h16_h17 + dd=0.08 + vol_24h_thr30 — stack with Step 12b winner

Permutation null on best variant (1000 seeds, within-cohort outcome shuffle).
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
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
VOL_LOOKBACK_HOURS = 24
VOL_THRESHOLD_PCT = 30.0
PERMUTATION_SEEDS = 1000
INITIAL_BANKROLL = 5.0

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


class Step19Tracker(InMemoryBankrollTracker):
    """Same gate-validated pattern as Step 16Tracker."""

    def __init__(self, *, initial_bankroll, drawdown_peak_window_days, peak_mode,
                  cooldown_rounds, abs_dd_frac):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cd_total = int(cooldown_rounds)
        self._abs_dd_frac = float(abs_dd_frac)
        self.n_pauses_fired = 0
        self.n_cooldown_skips = 0

    def is_paused(self, as_of_start_at):
        if self._cooldown > 0:
            self.n_cooldown_skips += 1
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


def compute_vol_cache(btc_timeline, lookback_seconds):
    if not btc_timeline:
        return {}
    epochs = np.array([r[0] for r in btc_timeline])
    ts = np.array([r[1] for r in btc_timeline])
    closes = np.array([r[2] for r in btc_timeline])
    n = len(ts)
    log_closes = np.log(closes)
    log_returns = np.diff(log_closes)
    end_ts_of_returns = ts[1:]
    out = {}
    PER_YEAR_5MIN = 288 * 365
    for i in range(n):
        target_ts = ts[i]
        cutoff_low = target_ts - lookback_seconds
        cutoff_high = target_ts - 2
        idx_lo = np.searchsorted(end_ts_of_returns, cutoff_low, side="left")
        idx_hi = np.searchsorted(end_ts_of_returns, cutoff_high, side="right")
        if idx_hi - idx_lo < 3:
            continue
        window = log_returns[idx_lo:idx_hi]
        sd = float(np.std(window, ddof=1))
        vol_ann_pct = sd * math.sqrt(PER_YEAR_5MIN) * 100.0
        out[int(epochs[i])] = vol_ann_pct
    return out


def run_backtest(*, dd_frac, hour_filter_mode, hour_set,
                  vol_cache, vol_threshold,
                  all_rounds, btc_klines, eth_klines, sol_klines,
                  label):
    """hour_filter_mode: 'none' | 'skip' | 'keep'
       hour_set: set of UTC hours (int 0-23) — when mode='skip', these are vetoed;
                 when mode='keep', only these hours are bet.
       vol_cache + vol_threshold: optional vol filter (None = disabled)
    """
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
    tracker = Step19Tracker(
        initial_bankroll=INITIAL_BANKROLL,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        cooldown_rounds=COOLDOWN_ROUNDS,
        abs_dd_frac=dd_frac,
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
                       "n_hour_vetoed": 0, "n_vol_vetoed": 0} for c in COHORT_ORDER}
    per_hour = {h: {"n_bets": 0, "n_wins": 0, "pnl_bnb": 0.0, "n_rounds_seen": 0,
                     "n_hour_vetoed": 0} for h in range(24)}
    bankroll = float(INITIAL_BANKROLL); peak = bankroll; max_dd_frac = 0.0
    bet_records = []
    n_hour_vetoes = 0
    n_vol_vetoes = 0

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1
        dt = datetime.fromtimestamp(int(round_t.start_at), tz=timezone.utc)
        hr = dt.hour
        per_hour[hr]["n_rounds_seen"] += 1

        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Hour-of-day filter
        if hour_filter_mode == "skip" and hr in hour_set:
            n_hour_vetoes += 1
            per_cohort[coh]["n_hour_vetoed"] += 1
            per_hour[hr]["n_hour_vetoed"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue
        elif hour_filter_mode == "keep" and hr not in hour_set:
            n_hour_vetoes += 1
            per_cohort[coh]["n_hour_vetoed"] += 1
            per_hour[hr]["n_hour_vetoed"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Vol filter veto
        if vol_cache is not None:
            vol = vol_cache.get(ep)
            if vol is None or vol < vol_threshold:
                n_vol_vetoes += 1
                per_cohort[coh]["n_vol_vetoed"] += 1
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
        per_hour[hr]["n_bets"] += 1
        per_hour[hr]["pnl_bnb"] += profit
        if outcome.outcome == "win":
            per_cohort[coh]["n_wins"] += 1
            per_hour[hr]["n_wins"] += 1
        bet_records.append({"epoch": ep, "cohort": coh, "hour": hr,
                             "profit": profit, "won": outcome.outcome == "win"})

        if bankroll > peak: peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac: max_dd_frac = dd

        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])

    for cd in per_cohort.values():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0
    for hd in per_hour.values():
        hd["win_rate"] = hd["n_wins"] / hd["n_bets"] if hd["n_bets"] else 0.0

    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())
    return {
        "label": label,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - INITIAL_BANKROLL,
            "final_bankroll": bankroll,
        },
        "max_drawdown_frac": max_dd_frac,
        "n_pauses_fired": tracker.n_pauses_fired,
        "n_cooldown_skips": tracker.n_cooldown_skips,
        "n_hour_vetoes": n_hour_vetoes,
        "n_vol_vetoes": n_vol_vetoes,
        "per_cohort": per_cohort,
        "per_hour": per_hour,
        "bet_records": bet_records,
    }


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
            if not pool:
                continue
            rng.shuffle(pool)
            perm_D += sum(pool[:n_cand]) - sum(pool[n_cand:n_cand + n_base])
        perm_Ds.append(perm_D)

    perm_Ds_sorted = sorted(perm_Ds)
    n_geq = sum(1 for d in perm_Ds if d >= obs_D)
    return {
        "observed_D": obs_D,
        "n_seeds": n_seeds,
        "p_value": n_geq / n_seeds,
        "perm_D_mean": statistics.mean(perm_Ds),
        "perm_D_stdev": statistics.stdev(perm_Ds) if len(perm_Ds) > 1 else 0.0,
        "perm_D_min": min(perm_Ds),
        "perm_D_max": max(perm_Ds),
        "perm_D_p05": perm_Ds_sorted[int(0.05 * n_seeds)],
        "perm_D_p50": perm_Ds_sorted[int(0.50 * n_seeds)],
        "perm_D_p95": perm_Ds_sorted[int(0.95 * n_seeds)],
        "perm_D_p99": perm_Ds_sorted[int(0.99 * n_seeds)],
    }


def main():
    t_all = time.time()
    print("--- loading rounds + klines ---", flush=True)
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

    # Vol cache for variant 6
    print("--- building 24h vol cache ---", flush=True)
    btc_timeline = []
    for ep, kl in btc.items():
        if not kl: continue
        last_candle = kl[-1]
        ts_ms = int(last_candle[0]); last_close = float(last_candle[4])
        if last_close > 0:
            btc_timeline.append((int(ep), ts_ms // 1000, last_close))
    btc_timeline.sort(key=lambda x: x[1])
    t_v = time.time()
    vol_cache = compute_vol_cache(btc_timeline, VOL_LOOKBACK_HOURS * 3600)
    print(f"  vol_24h: {len(vol_cache)} epochs ({time.time()-t_v:.1f}s)", flush=True)

    # Variants
    print("\n--- 7 backtests at 5 BNB ---", flush=True)

    common = dict(
        all_rounds=all_rounds, btc_klines=btc_klines,
        eth_klines=eth_klines, sol_klines=sol_klines,
    )

    t = time.time()
    baseline = run_backtest(
        dd_frac=0.15, hour_filter_mode="none", hour_set=set(),
        vol_cache=None, vol_threshold=0.0, label="baseline",
        **common,
    )
    s = baseline["summary"]
    print(f"  baseline (cd=72, dd=0.15): pnl={s['net_pnl_bnb']:+.4f} bets={s['num_bets']} "
          f"wr={s['win_rate']*100:.2f}% max_dd={baseline['max_drawdown_frac']*100:.2f}% "
          f"({time.time()-t:.1f}s)", flush=True)

    variants_specs = [
        ("v1_skip_h16_h17",        dict(hour_filter_mode="skip", hour_set={16, 17},
                                         dd_frac=0.15, vol_cache=None, vol_threshold=0.0)),
        ("v2_skip_h16_only",       dict(hour_filter_mode="skip", hour_set={16},
                                         dd_frac=0.15, vol_cache=None, vol_threshold=0.0)),
        ("v3_skip_h16_h17_h18",    dict(hour_filter_mode="skip", hour_set={16, 17, 18},
                                         dd_frac=0.15, vol_cache=None, vol_threshold=0.0)),
        ("v4_keep_h11_only",       dict(hour_filter_mode="keep", hour_set={11},
                                         dd_frac=0.15, vol_cache=None, vol_threshold=0.0)),
        ("v5_keep_top4",           dict(hour_filter_mode="keep", hour_set={11, 19, 15, 9},
                                         dd_frac=0.15, vol_cache=None, vol_threshold=0.0)),
        ("v6_skip_h16_h17_dd08_vol", dict(hour_filter_mode="skip", hour_set={16, 17},
                                            dd_frac=0.08, vol_cache=vol_cache,
                                            vol_threshold=VOL_THRESHOLD_PCT)),
    ]

    results = {"baseline": baseline}
    for name, kw in variants_specs:
        t = time.time()
        r = run_backtest(label=name, **kw, **common)
        s = r["summary"]
        delta = s["net_pnl_bnb"] - baseline["summary"]["net_pnl_bnb"]
        print(f"  {name:>30s}: pnl={s['net_pnl_bnb']:+.4f} delta={delta:+.4f} "
              f"bets={s['num_bets']} wr={s['win_rate']*100:.2f}% "
              f"hour_vetoed={r['n_hour_vetoes']} vol_vetoed={r['n_vol_vetoes']} "
              f"max_dd={r['max_drawdown_frac']*100:.2f}% ({time.time()-t:.1f}s)", flush=True)
        results[name] = r

    # Identify best variant by net PnL
    variant_names = [n for n, _ in variants_specs]
    best_name = max(variant_names, key=lambda n: results[n]["summary"]["net_pnl_bnb"])
    best = results[best_name]
    print(f"\n  best variant by PnL: {best_name} ({best['summary']['net_pnl_bnb']:+.4f})", flush=True)

    # Permutation null on best variant
    print(f"\n--- Permutation null on best variant ({PERMUTATION_SEEDS} seeds) ---", flush=True)
    t = time.time()
    null = permutation_null(
        bets_candidate=best["bet_records"],
        bets_baseline=baseline["bet_records"],
        n_seeds=PERMUTATION_SEEDS,
    )
    print(f"  Observed D: {null['observed_D']:+.4f}", flush=True)
    print(f"  Null mean: {null['perm_D_mean']:+.4f}  stdev: {null['perm_D_stdev']:.4f}", flush=True)
    print(f"  Null p05/p50/p95/p99: {null['perm_D_p05']:+.4f} / {null['perm_D_p50']:+.4f} / "
          f"{null['perm_D_p95']:+.4f} / {null['perm_D_p99']:+.4f}", flush=True)
    print(f"  p-value: {null['p_value']:.4f}", flush=True)
    print(f"  ({time.time()-t:.1f}s)", flush=True)

    # Persist
    def strip(r):
        return {k: v for k, v in r.items() if k != "bet_records"}

    out_path = REPO / "var" / "strategy_review" / "hour_filter_step19_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "initial_bankroll": INITIAL_BANKROLL,
                "cooldown_rounds": COOLDOWN_ROUNDS,
                "permutation_seeds": PERMUTATION_SEEDS,
                "vol_threshold_pct": VOL_THRESHOLD_PCT,
            },
            "results": {k: strip(v) for k, v in results.items()},
            "best_variant_name": best_name,
            "permutation_null_best": null,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
