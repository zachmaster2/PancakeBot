"""Step 15 — cooldown_rounds sweep + permutation null on 5 BNB optimum.

Step 14c finding: production cooldown=72 over-pauses. random_resume at
{18, 24, 36} all beat baseline by +1.9-2.98 BNB. Cleanly deployable change:
shorten cooldown_rounds.

Step 15 confirms with fine-grained sweep at both scales + permutation null
significance test.

Sweep: cooldown_rounds ∈ {0, 3, 5, 8, 10, 12, 18, 24, 36, 48, 60, 72} = 12 values
Scales: 5 BNB and 50 BNB = 24 backtests + 2 references
Permutation null: 1000 iterations on 5 BNB optimum vs cd=72 baseline

Uses gate-validated Step14Tracker pattern (production-bit-identical breaker
with +1 cooldown compensation).
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
import tempfile
import time
from collections import deque
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
ABS_DD_FRAC = 0.15

COOLDOWN_VALUES = (0, 3, 5, 8, 10, 12, 18, 24, 36, 48, 60, 72)
SCALES = (5.0, 50.0)
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


def cohort_of(epoch: int) -> str:
    for name, lo, hi in COHORT_DEFS:
        if lo <= epoch <= hi:
            return name
    return "unknown"


class Step15Tracker(InMemoryBankrollTracker):
    """Same as Step 14c's Step14cTracker baseline path: production-faithful
    pre-decision drawdown check + +1 cooldown compensation. Just varies
    cooldown_rounds.
    """

    def __init__(self, *, initial_bankroll: float, drawdown_peak_window_days: int,
                  peak_mode: str, cooldown_rounds: int,
                  abs_dd_frac: float = ABS_DD_FRAC):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cd_total = int(cooldown_rounds)
        self._abs_dd_frac = abs_dd_frac
        self.n_pauses_fired = 0
        self.n_cooldown_skips = 0

    def is_paused(self, as_of_start_at: int) -> bool:
        if self._cooldown > 0:
            self.n_cooldown_skips += 1
            return True
        current = self.current_bankroll()
        peak = self.peak_bankroll(as_of_start_at)
        if peak > 0:
            dd = (peak - current) / peak
            if dd >= self._abs_dd_frac:
                # +1 compensation for pipeline's tick after is_paused True
                # Edge case: if cooldown_rounds=0, don't pause at all
                if self._cd_total > 0:
                    self.set_paused(self._cd_total + 1, as_of_start_at)
                self.n_pauses_fired += 1
                return self._cd_total > 0
        return False


def run_cd_backtest(*, initial_bankroll: float, cooldown_rounds: int,
                     all_rounds, btc_klines, eth_klines, sol_klines,
                     earliest_offset: int) -> dict[str, Any]:
    overrides = {
        "gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)},
        "risk": {"max_drawdown_fraction_from_peak": 1.0},  # neutered; tracker owns it
    }
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    tracker = Step15Tracker(
        initial_bankroll=initial_bankroll,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        cooldown_rounds=cooldown_rounds,
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
                      "n_skip": 0, "n_breaker_skip": 0, "n_cooldown_skip": 0}
                  for c in COHORT_ORDER}
    bankroll = float(initial_bankroll); peak = bankroll; max_dd_frac = 0.0
    bet_records: list[dict[str, Any]] = []  # for permutation null

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1
        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            sr = decision.skip_reason or ""
            if sr == "risk_drawdown_breaker_fired":
                per_cohort[coh]["n_breaker_skip"] += 1
            elif sr == "risk_cooldown_active":
                per_cohort[coh]["n_cooldown_skip"] += 1
            else:
                per_cohort[coh]["n_skip"] += 1
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
        bet_records.append({
            "epoch": ep, "cohort": coh, "profit": profit, "won": outcome.outcome == "win",
        })

        if bankroll > peak: peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac: max_dd_frac = dd

        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])

    for cd in per_cohort.values():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0

    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())

    # Sanity: mean cooldown rounds per fire
    mean_cd_per_fire = (tracker.n_cooldown_skips / tracker.n_pauses_fired
                        if tracker.n_pauses_fired > 0 else 0.0)

    return {
        "cooldown_rounds": cooldown_rounds,
        "initial_bankroll": initial_bankroll,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - initial_bankroll,
            "final_bankroll": bankroll,
        },
        "max_drawdown_frac": max_dd_frac,
        "n_pauses_fired": tracker.n_pauses_fired,
        "n_cooldown_skips": tracker.n_cooldown_skips,
        "mean_cooldown_per_fire": mean_cd_per_fire,
        "per_cohort": per_cohort,
        "bet_records": bet_records,
    }


def permutation_null(*, bets_optimum: list[dict], bets_baseline: list[dict],
                       n_seeds: int = PERMUTATION_SEEDS, base_seed: int = 42) -> dict[str, Any]:
    """Permutation null: within each cohort, take union of bets from both runs.
    Randomly split into 'optimum-like' (size n_opt) and 'baseline-like' (size n_base).
    Compute permuted D. p-value = #(perm_D >= obs_D) / n_seeds.
    """
    # Group by cohort
    cohort_bets: dict[str, dict[str, list[float]]] = {}
    for b in bets_optimum:
        c = b["cohort"]
        if c not in cohort_bets:
            cohort_bets[c] = {"opt": [], "base": []}
        cohort_bets[c]["opt"].append(b["profit"])
    for b in bets_baseline:
        c = b["cohort"]
        if c not in cohort_bets:
            cohort_bets[c] = {"opt": [], "base": []}
        cohort_bets[c]["base"].append(b["profit"])

    obs_D = sum(b["profit"] for b in bets_optimum) - sum(b["profit"] for b in bets_baseline)

    rng = random.Random(base_seed)
    perm_Ds: list[float] = []
    for seed_idx in range(n_seeds):
        perm_D = 0.0
        for coh, d in cohort_bets.items():
            pool = d["opt"] + d["base"]
            n_opt = len(d["opt"])
            n_base = len(d["base"])
            if not pool or (n_opt == 0 and n_base == 0):
                continue
            rng.shuffle(pool)
            sum_opt = sum(pool[:n_opt])
            sum_base = sum(pool[n_opt:n_opt + n_base])
            perm_D += sum_opt - sum_base
        perm_Ds.append(perm_D)

    n_geq = sum(1 for d in perm_Ds if d >= obs_D)
    p_value = n_geq / n_seeds
    return {
        "observed_D": obs_D,
        "n_seeds": n_seeds,
        "p_value": p_value,
        "perm_D_mean": statistics.mean(perm_Ds),
        "perm_D_stdev": statistics.stdev(perm_Ds) if len(perm_Ds) > 1 else 0.0,
        "perm_D_min": min(perm_Ds),
        "perm_D_max": max(perm_Ds),
        "perm_D_p05": sorted(perm_Ds)[int(0.05 * n_seeds)],
        "perm_D_p95": sorted(perm_Ds)[int(0.95 * n_seeds)],
        "perm_D_p99": sorted(perm_Ds)[int(0.99 * n_seeds)],
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
    print(f"  klines: BTC={len(btc)} ETH={len(eth)} SOL={len(sol)} ({time.time()-t_kl:.1f}s)", flush=True)

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

    all_results: dict[float, list[dict[str, Any]]] = {5.0: [], 50.0: []}

    print(f"\n--- Running 24 backtests (12 cd values x 2 scales) ---", flush=True)
    for scale in SCALES:
        print(f"\n  --- {scale} BNB scale ---", flush=True)
        for cd in COOLDOWN_VALUES:
            t = time.time()
            r = run_cd_backtest(
                initial_bankroll=scale, cooldown_rounds=cd,
                all_rounds=all_rounds, btc_klines=btc_klines,
                eth_klines=eth_klines, sol_klines=sol_klines,
                earliest_offset=earliest_offset,
            )
            r["elapsed_seconds"] = time.time() - t
            all_results[scale].append(r)
            s = r["summary"]
            print(f"    cd={cd:>2d}: pnl={s['net_pnl_bnb']:+8.4f} bets={s['num_bets']:>4d} "
                  f"fires={r['n_pauses_fired']:>4d} cd_skips={r['n_cooldown_skips']:>5d} "
                  f"mean_cd_per_fire={r['mean_cooldown_per_fire']:.1f} "
                  f"max_dd={r['max_drawdown_frac']*100:.2f}% ({r['elapsed_seconds']:.1f}s)",
                  flush=True)

    # Find optimum per scale
    print(f"\n--- Optima ---", flush=True)
    for scale in SCALES:
        results = all_results[scale]
        best = max(results, key=lambda r: r["summary"]["net_pnl_bnb"])
        baseline = next(r for r in results if r["cooldown_rounds"] == 72)
        delta = best["summary"]["net_pnl_bnb"] - baseline["summary"]["net_pnl_bnb"]
        print(f"  @ {scale} BNB: optimum cd={best['cooldown_rounds']} "
              f"pnl={best['summary']['net_pnl_bnb']:+.4f} "
              f"vs baseline cd=72 pnl={baseline['summary']['net_pnl_bnb']:+.4f} "
              f"delta={delta:+.4f}", flush=True)

    # Permutation null on 5 BNB optimum
    print(f"\n--- Permutation null (5 BNB, 1000 seeds) ---", flush=True)
    results_5 = all_results[5.0]
    optimum_5 = max(results_5, key=lambda r: r["summary"]["net_pnl_bnb"])
    baseline_5 = next(r for r in results_5 if r["cooldown_rounds"] == 72)
    null_result = permutation_null(
        bets_optimum=optimum_5["bet_records"],
        bets_baseline=baseline_5["bet_records"],
        n_seeds=PERMUTATION_SEEDS,
    )
    print(f"  Observed D: {null_result['observed_D']:+.4f}", flush=True)
    print(f"  Null mean: {null_result['perm_D_mean']:+.4f}  stdev: {null_result['perm_D_stdev']:.4f}", flush=True)
    print(f"  Null p05/p50/p95/p99: {null_result['perm_D_p05']:+.4f} / "
          f"{statistics.median([null_result['perm_D_min'], null_result['perm_D_max']]):+.4f} / "
          f"{null_result['perm_D_p95']:+.4f} / {null_result['perm_D_p99']:+.4f}", flush=True)
    print(f"  p-value: {null_result['p_value']:.4f}", flush=True)

    # Save
    # Strip bet_records from JSON output (large)
    save_results: dict[str, list[dict[str, Any]]] = {}
    for scale, results in all_results.items():
        save_results[f"{int(scale)}bnb"] = []
        for r in results:
            r_copy = {k: v for k, v in r.items() if k != "bet_records"}
            save_results[f"{int(scale)}bnb"].append(r_copy)

    out_path = REPO / "var" / "strategy_review" / "cooldown_sweep_step15_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "cooldown_values": list(COOLDOWN_VALUES),
                "scales": list(SCALES),
                "permutation_seeds": PERMUTATION_SEEDS,
            },
            "results_per_scale": save_results,
            "permutation_null_5bnb": null_result,
            "optimum_5bnb_cd": optimum_5["cooldown_rounds"],
            "optimum_5bnb_pnl": optimum_5["summary"]["net_pnl_bnb"],
            "baseline_5bnb_pnl": baseline_5["summary"]["net_pnl_bnb"],
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
