"""Step 9 — max bet cap 0.5 BNB at both bankroll scales.

User asks: cap max bet size at 0.5 BNB (a NEW tighter constraint than canonical's
2.0 BNB absolute cap for btc_primary). Run dynamic-bankroll (real risk gates) at
5 BNB and 50 BNB initial. Compare per-cohort to Step 7's baseline.

Implementation:
  - Override StrategyConfig.risk.max_bet_bnb_btc_primary from 2.0 -> 0.5.
    (eth_sol_fallback's cap is already 0.5; leave it.)
  - The sizer's `cap_bnb` parameter for the btc_primary path becomes 0.5.
  - Everything else identical to canonical: same gate (3, 7, 15) cs=2, same
    pool filter, same bankroll cap fraction (0.05), same drawdown breaker,
    same cooldown semantics, real impact-aware settlement.
  - Bankroll-cap interaction:
      @ 5 BNB:  bankroll cap = 0.25 (binds tighter than 0.5; no change)
      @ 50 BNB: bankroll cap = 2.5 (so 0.5 binds; bets DROP)
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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


EPOCH_MIN = 422298
EPOCH_MAX = 484000

CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2

NEW_MAX_BET_BNB = 0.5  # the experiment

COHORT_ORDER = ("extension", "cv5", "holdout", "ext_v2", "fresh_oos", "post_fresh")


def cohort_of(epoch: int) -> str:
    if 422298 <= epoch <= 437561: return "extension"
    if 437562 <= epoch <= 474086: return "cv5"
    if 474880 <= epoch <= 475311: return "holdout"
    if 475312 <= epoch <= 479952: return "ext_v2"
    if 479953 <= epoch <= 483191: return "fresh_oos"
    return "post_fresh"


def run_capped_and_accumulate(*, initial_bankroll: float, label: str,
                                all_rounds: list, btc: dict, eth: dict, sol: dict,
                                earliest_offset: int, out_root: Path) -> dict[str, Any]:
    overrides = {
        "gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)},
        "risk": {"max_bet_bnb_btc_primary": NEW_MAX_BET_BNB},
    }
    sc = load_strategy_config_from_dict(overrides)
    # Sanity check
    assert sc.risk.max_bet_bnb_btc_primary == NEW_MAX_BET_BNB
    assert sc.gate.mtf_lookbacks == CANONICAL_LOOKBACKS

    spec = ipr.FoldSpec(
        name=f"capped_{label}",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        epoch_start=EPOCH_MIN,
        epoch_end=EPOCH_MAX,
        strategy_overrides=overrides,
    )
    print(f"\n--- capped backtest @ {initial_bankroll} BNB ({label}), "
          f"max_bet_bnb_btc_primary={NEW_MAX_BET_BNB} ---")
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
          f"WR={summary['win_rate']:.4f} pnl={summary['net_pnl_bnb']:+.4f} BNB  "
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
        } for c in COHORT_ORDER
    }

    with open(trades_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            coh = cohort_of(epoch)
            per_cohort[coh]["n_rounds"] += 1
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

    for c in COHORT_ORDER:
        cd = per_cohort[c]
        if cd["n_bets"] > 0:
            cd["win_rate"] = cd["n_wins"] / cd["n_bets"]
            cd["mean_bet_size_bnb"] = cd["total_bet_size_bnb"] / cd["n_bets"]
            cd["mean_pnl_per_bet"] = cd["pnl_bnb"] / cd["n_bets"]
        else:
            cd["win_rate"] = 0.0
            cd["mean_bet_size_bnb"] = 0.0
            cd["mean_pnl_per_bet"] = 0.0

    return {
        "initial_bankroll": initial_bankroll,
        "label": label,
        "max_bet_cap_bnb": NEW_MAX_BET_BNB,
        "summary": {k: summary[k] for k in
                    ("num_bets", "num_wins", "win_rate", "net_pnl_bnb",
                     "final_bankroll_bnb", "first_epoch", "last_epoch",
                     "gross_profit_bnb", "gross_loss_bnb")
                    if k in summary},
        "skip_counts_by_reason": summary.get("skip_counts_by_reason", {}),
        "per_cohort": per_cohort,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    t_all = time.time()

    print("--- loading rounds (canonical + extended) ---")
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  loaded {len(all_rounds)} rounds; range "
          f"[{all_rounds[0].epoch}..{all_rounds[-1].epoch}]")

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1
    print(f"  earliest_offset={earliest_offset} latest_offset={latest_offset}")

    print("--- loading klines unified ---")
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  BTC={len(btc)} ETH={len(eth)} SOL={len(sol)}")

    out_root = Path(tempfile.mkdtemp(prefix="step9_"))
    print(f"--- temp output: {out_root} ---")

    run_5 = run_capped_and_accumulate(
        initial_bankroll=5.0, label="5bnb_cap0.5",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset, out_root=out_root,
    )
    run_50 = run_capped_and_accumulate(
        initial_bankroll=50.0, label="50bnb_cap0.5",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset, out_root=out_root,
    )

    # --- Print per-cohort table ---
    print(f"\n=== Per-cohort: max_bet_bnb_btc_primary=0.5 ===")
    print(f"{'cohort':>12s} {'rounds':>7s}  "
          f"{'C5:bets':>8s} {'WR':>7s} {'PnL':>10s} {'mean_bet':>9s} {'max_bet':>8s} {'dd':>4s} {'cd':>5s}  "
          f"{'C50:bets':>9s} {'WR':>7s} {'PnL':>10s} {'mean_bet':>9s} {'max_bet':>8s} {'dd':>4s} {'cd':>5s}")
    for coh in COHORT_ORDER:
        c5 = run_5["per_cohort"][coh]
        c50 = run_50["per_cohort"][coh]
        print(f"{coh:>12s} {c5['n_rounds']:>7d}  "
              f"{c5['n_bets']:>8d} {c5['win_rate']:>7.4f} {c5['pnl_bnb']:>+10.4f} "
              f"{c5['mean_bet_size_bnb']:>9.4f} {c5['max_bet_size_bnb']:>8.4f} "
              f"{c5['skip_drawdown_breaker']:>4d} {c5['skip_cooldown']:>5d}  "
              f"{c50['n_bets']:>9d} {c50['win_rate']:>7.4f} {c50['pnl_bnb']:>+10.4f} "
              f"{c50['mean_bet_size_bnb']:>9.4f} {c50['max_bet_size_bnb']:>8.4f} "
              f"{c50['skip_drawdown_breaker']:>4d} {c50['skip_cooldown']:>5d}")

    # --- Compare to Step 7 baseline ---
    step7_path = REPO / "var" / "strategy_review" / "bankroll_scale_rerun_step7_data.json"
    if step7_path.exists():
        step7_data = json.loads(step7_path.read_text(encoding="utf-8"))
        print(f"\n=== Baseline (Step 7) vs Cap-0.5 (Step 9) comparison ===")
        print(f"{'cohort':>12s}  {'dyn5_base':>10s} {'dyn5_cap':>10s} {'d-b':>8s}  "
              f"{'dyn50_base':>11s} {'dyn50_cap':>10s} {'d-b':>8s}")
        d5b = step7_data["run_5bnb"]["per_cohort"]
        d50b = step7_data["run_50bnb"]["per_cohort"]
        c5 = run_5["per_cohort"]
        c50 = run_50["per_cohort"]
        for coh in COHORT_ORDER:
            print(f"{coh:>12s}  "
                  f"{d5b[coh]['pnl_bnb']:>+10.4f} {c5[coh]['pnl_bnb']:>+10.4f} "
                  f"{c5[coh]['pnl_bnb'] - d5b[coh]['pnl_bnb']:>+8.4f}  "
                  f"{d50b[coh]['pnl_bnb']:>+11.4f} {c50[coh]['pnl_bnb']:>+10.4f} "
                  f"{c50[coh]['pnl_bnb'] - d50b[coh]['pnl_bnb']:>+8.4f}")
        d5_total = step7_data["run_5bnb"]["summary"]["net_pnl_bnb"]
        d50_total = step7_data["run_50bnb"]["summary"]["net_pnl_bnb"]
        c5_total = run_5["summary"]["net_pnl_bnb"]
        c50_total = run_50["summary"]["net_pnl_bnb"]
        print(f"{'TOTAL':>12s}  "
              f"{d5_total:>+10.4f} {c5_total:>+10.4f} {c5_total - d5_total:>+8.4f}  "
              f"{d50_total:>+11.4f} {c50_total:>+10.4f} {c50_total - d50_total:>+8.4f}")

    # --- Persist ---
    out_path = REPO / "var" / "strategy_review" / "max_bet_cap_step9_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "new_max_bet_bnb_btc_primary": NEW_MAX_BET_BNB,
                "previous_max_bet_bnb_btc_primary": 2.0,
            },
            "run_5bnb_cap": run_5,
            "run_50bnb_cap": run_50,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
