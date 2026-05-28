"""Step 10b — post-CV5 to current backtest (expanded post-sync).

After --sync brought closed_rounds + klines current (now epochs 437562..484408),
this driver runs canonical (3, 7, 15) cs=2 on the full post-CV5 slice:
epochs 474087..484408 (10,322 rounds total).

Cohort breakdown is EXPLICIT for the previously-mislabeled gap:
  gap_post_cv5_pre_holdout : 474087..474879  (793 rounds)
  holdout                  : 474880..475311  (432 rounds)
  ext_v2                   : 475312..479952  (4,641 rounds)
  fresh_oos                : 479953..483191  (3,239 rounds)
  post_fresh               : 483192..484408  (1,217 rounds, expanded by sync)

Two scales: dynamic 5 BNB AND dynamic 50 BNB (deployable config, risk gates engaged).
Output reports: per-cohort n_bets/WR/PnL/max_dd, total skip-reason distribution,
historical comparison to CV5.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Use canonical data only post-sync; no need for extended data (post-CV5 epochs
# are all in the canonical store now).
import research.in_process_runner as ipr  # noqa: E402
from pancakebot.config import load_strategy_config_from_dict  # noqa: E402


EPOCH_START = 474087   # immediately after CV5 ends (CV5: 437562..474086)
EPOCH_END_CONFIG = 484999  # safety upper bound; actual cap is dataset max

CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2

# Cohort definitions (explicit gap bucket for accuracy)
COHORT_DEFS = [
    ("gap_post_cv5_pre_holdout", 474087, 474879),
    ("holdout", 474880, 475311),
    ("ext_v2", 475312, 479952),
    ("fresh_oos", 479953, 483191),
    ("post_fresh", 483192, 999999),  # everything ≥ 483192
]
COHORT_ORDER = [c[0] for c in COHORT_DEFS]


def cohort_of(epoch: int) -> str:
    for name, lo, hi in COHORT_DEFS:
        if lo <= epoch <= hi:
            return name
    return "unknown"


def run_post_cv5(*, initial_bankroll: float, label: str,
                  all_rounds: list, btc: dict, eth: dict, sol: dict,
                  earliest_offset: int, out_root: Path,
                  epoch_max: int) -> dict[str, Any]:
    sc = load_strategy_config_from_dict(
        {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    )
    spec = ipr.FoldSpec(
        name=f"post_cv5_{label}",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        epoch_start=EPOCH_START,
        epoch_end=epoch_max,
        strategy_overrides={"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}},
    )
    print(f"\n--- post-CV5 @ {initial_bankroll} BNB ({label}): "
          f"epochs [{EPOCH_START}..{epoch_max}] ---")
    t0 = time.time()
    summary = ipr.run_fold(
        spec=spec,
        strategy_cfg=sc,
        all_rounds=all_rounds,
        btc_unified=btc,
        eth_unified=eth,
        sol_unified=sol,
        earliest_offset=earliest_offset,
        output_base_dir=out_root,
        initial_bankroll_bnb=initial_bankroll,
        treasury_fee_fraction=0.03,
        min_bet_amount_bnb=0.001,
    )
    elapsed = time.time() - t0
    print(f"  bets={summary['num_bets']} wins={summary['num_wins']} "
          f"WR={summary['win_rate']:.4f} pnl={summary['net_pnl_bnb']:+.4f} BNB "
          f"({elapsed:.1f}s)")

    trades_csv = out_root / spec.name / "trades.csv"

    per_cohort: dict[str, dict[str, Any]] = {
        c: {
            "n_rounds": 0, "n_bets": 0, "n_wins": 0, "pnl_bnb": 0.0,
            "total_bet_size_bnb": 0.0, "max_bet_size_bnb": 0.0,
            "skip_drawdown_breaker": 0, "skip_cooldown": 0,
            "skip_bankroll_below_min": 0, "skip_gate_no_signal": 0,
            "skip_pool_below_minimum": 0, "skip_payout_below_floor": 0,
            "skip_other": 0,
            "per_bet_profits": [],
        } for c in COHORT_ORDER
    }
    max_dd_frac = 0.0
    peak = initial_bankroll

    with open(trades_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            coh = cohort_of(epoch)
            if coh not in per_cohort:
                continue
            per_cohort[coh]["n_rounds"] += 1
            bankroll_after = float(row["bankroll_bnb"])
            if bankroll_after > peak:
                peak = bankroll_after
            if peak > 0:
                dd = (peak - bankroll_after) / peak
                if dd > max_dd_frac:
                    max_dd_frac = dd
            action = row.get("action")
            if action == "BET":
                profit = float(row["profit_bnb"])
                bet_size = float(row["bet_size_bnb"])
                per_cohort[coh]["n_bets"] += 1
                per_cohort[coh]["pnl_bnb"] += profit
                per_cohort[coh]["total_bet_size_bnb"] += bet_size
                if bet_size > per_cohort[coh]["max_bet_size_bnb"]:
                    per_cohort[coh]["max_bet_size_bnb"] = bet_size
                if profit > 0:
                    per_cohort[coh]["n_wins"] += 1
                per_cohort[coh]["per_bet_profits"].append(profit)
            else:
                sr = (row.get("skip_reason") or "").strip()
                if sr == "risk_drawdown_breaker_fired":
                    per_cohort[coh]["skip_drawdown_breaker"] += 1
                elif sr == "risk_cooldown_active":
                    per_cohort[coh]["skip_cooldown"] += 1
                elif sr == "risk_bankroll_below_min":
                    per_cohort[coh]["skip_bankroll_below_min"] += 1
                elif sr == "gate_no_signal":
                    per_cohort[coh]["skip_gate_no_signal"] += 1
                elif sr == "pool_below_minimum":
                    per_cohort[coh]["skip_pool_below_minimum"] += 1
                elif sr == "payout_below_floor":
                    per_cohort[coh]["skip_payout_below_floor"] += 1
                else:
                    per_cohort[coh]["skip_other"] += 1

    # Derive WR + mean bet / per-bet stats
    for coh, cd in per_cohort.items():
        if cd["n_bets"] > 0:
            cd["win_rate"] = cd["n_wins"] / cd["n_bets"]
            cd["mean_bet_size_bnb"] = cd["total_bet_size_bnb"] / cd["n_bets"]
            cd["mean_pnl_per_bet"] = cd["pnl_bnb"] / cd["n_bets"]
            cd["stdev_profit_per_bet"] = (statistics.stdev(cd["per_bet_profits"])
                                          if cd["n_bets"] > 1 else 0.0)
            cd["bet_rate"] = cd["n_bets"] / cd["n_rounds"] if cd["n_rounds"] else 0.0
        else:
            cd["win_rate"] = 0.0
            cd["mean_bet_size_bnb"] = 0.0
            cd["mean_pnl_per_bet"] = 0.0
            cd["stdev_profit_per_bet"] = 0.0
            cd["bet_rate"] = 0.0
        del cd["per_bet_profits"]  # strip for JSON

    return {
        "initial_bankroll": initial_bankroll,
        "label": label,
        "epoch_start": EPOCH_START,
        "epoch_end": epoch_max,
        "summary": {k: summary[k] for k in
                    ("backtest_round_count", "num_bets", "num_wins", "win_rate",
                     "net_pnl_bnb", "final_bankroll_bnb", "first_epoch", "last_epoch",
                     "gross_profit_bnb", "gross_loss_bnb", "bet_rate", "num_skips")
                    if k in summary},
        "max_drawdown_frac": max_dd_frac,
        "skip_counts_by_reason": summary.get("skip_counts_by_reason", {}),
        "per_cohort": per_cohort,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    t_all = time.time()

    print("--- loading rounds (canonical only — post-CV5 is all in canonical store) ---")
    all_rounds = ipr._load_all_rounds(use_extended_data=False)
    max_epoch_in_dataset = max(r.epoch for r in all_rounds)
    print(f"  loaded {len(all_rounds)} rounds; range "
          f"[{all_rounds[0].epoch}..{max_epoch_in_dataset}]")

    n_post_cv5 = sum(1 for r in all_rounds if r.epoch >= EPOCH_START)
    print(f"  post-CV5 slice: epochs [{EPOCH_START}..{max_epoch_in_dataset}] "
          f"({n_post_cv5} actual rounds)")

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1

    print("--- loading klines unified ---")
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
    )
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
    )
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
    )
    print(f"  BTC={len(btc)} ETH={len(eth)} SOL={len(sol)}")

    out_root = Path(tempfile.mkdtemp(prefix="step10b_"))

    run_5 = run_post_cv5(
        initial_bankroll=5.0, label="5bnb",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset, out_root=out_root,
        epoch_max=max_epoch_in_dataset,
    )
    run_50 = run_post_cv5(
        initial_bankroll=50.0, label="50bnb",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset, out_root=out_root,
        epoch_max=max_epoch_in_dataset,
    )

    # --- Print per-cohort table ---
    print(f"\n=== Per-cohort breakdown (post-CV5 to current, expanded with sync) ===")
    print(f"{'cohort':>28s} {'rounds':>7s}  "
          f"{'C5:bets':>8s} {'WR':>7s} {'PnL':>10s} {'mean_bet':>9s} {'bet_rate':>9s} {'dd':>4s} {'cd':>5s}  "
          f"{'C50:bets':>9s} {'WR':>7s} {'PnL':>10s} {'mean_bet':>9s} {'bet_rate':>9s} {'dd':>4s} {'cd':>5s}")
    for coh in COHORT_ORDER:
        c5 = run_5["per_cohort"][coh]
        c50 = run_50["per_cohort"][coh]
        print(f"{coh:>28s} {c5['n_rounds']:>7d}  "
              f"{c5['n_bets']:>8d} {c5['win_rate']:>7.4f} {c5['pnl_bnb']:>+10.4f} "
              f"{c5['mean_bet_size_bnb']:>9.4f} {c5['bet_rate']*100:>8.2f}% "
              f"{c5['skip_drawdown_breaker']:>4d} {c5['skip_cooldown']:>5d}  "
              f"{c50['n_bets']:>9d} {c50['win_rate']:>7.4f} {c50['pnl_bnb']:>+10.4f} "
              f"{c50['mean_bet_size_bnb']:>9.4f} {c50['bet_rate']*100:>8.2f}% "
              f"{c50['skip_drawdown_breaker']:>4d} {c50['skip_cooldown']:>5d}")

    # Totals
    s5 = run_5["summary"]
    s50 = run_50["summary"]
    print(f"\n{'TOTAL':>28s} "
          f"{'':>7s}  "
          f"{s5['num_bets']:>8d} {s5['win_rate']:>7.4f} {s5['net_pnl_bnb']:>+10.4f}  "
          f"{'max_dd=':>9s}{run_5['max_drawdown_frac']*100:>6.2f}%  "
          f"{s50['num_bets']:>9d} {s50['win_rate']:>7.4f} {s50['net_pnl_bnb']:>+10.4f}  "
          f"{'max_dd=':>9s}{run_50['max_drawdown_frac']*100:>6.2f}%")

    # --- Skip-reason aggregate ---
    print(f"\n=== Aggregate skip reasons (5 BNB) ===")
    skip5 = run_5["skip_counts_by_reason"]
    known_keys = {"gate_no_signal", "pool_below_minimum", "payout_below_floor",
                  "risk_cooldown_active", "risk_drawdown_breaker_fired",
                  "risk_bankroll_below_min", "bet_size_below_min",
                  "kline_fetch_transient_failure",
                  "gate_no_btc_klines", "gate_no_eth_klines", "gate_no_sol_klines"}
    known_total = 0
    for k, v in sorted(skip5.items(), key=lambda x: -x[1]):
        if k in known_keys:
            print(f"  {k:>35s}: {v:>6d}")
            known_total += v
    unknown = sum(v for k, v in skip5.items() if k not in known_keys)
    if unknown > 0:
        print(f"  {'(other singletons)':>35s}: {unknown:>6d}")

    out_path = REPO / "var" / "strategy_review" / "post_cv5_to_current_step10b_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_start": EPOCH_START,
                "epoch_max_in_dataset": max_epoch_in_dataset,
                "cohort_defs": [list(c) for c in COHORT_DEFS],
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
            },
            "run_5bnb": run_5,
            "run_50bnb": run_50,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
