"""Step 10 — post_fresh+ backtest (recent regime).

Run canonical (3, 7, 15) cs=2 on the post_fresh cohort (epochs 483192+) only,
at both dynamic-5-BNB and dynamic-50-BNB scales. Goal: characterize the recent
data slice that the live bot has been operating against.

Reports:
  - n_bets, WR, total PnL, max drawdown, mean bet, per-bet detail
  - FULL skip-reason breakdown (gate_no_signal, pool_below_min, payout_below_floor,
    kline issues, risk gates, etc.) — to expose why bet rate is what it is
  - Bet-by-bet table for the post_fresh slice
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

EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
import research.in_process_runner as ipr  # noqa: E402
ipr._EXT_CLOSED_ROUNDS_PATH = EXT_DIR / "closed_rounds.jsonl"
ipr._EXT_BTC_KLINES_PATH = EXT_DIR / "btc_spot_prices.jsonl"
ipr._EXT_ETH_KLINES_PATH = EXT_DIR / "eth_spot_prices.jsonl"
ipr._EXT_SOL_KLINES_PATH = EXT_DIR / "sol_spot_prices.jsonl"

from pancakebot.config import load_strategy_config_from_dict  # noqa: E402


POST_FRESH_START = 483192  # from cohort_of: post_fresh = 483192+
CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2


def run_post_fresh(*, initial_bankroll: float, label: str,
                    all_rounds: list, btc: dict, eth: dict, sol: dict,
                    earliest_offset: int, out_root: Path,
                    epoch_max: int) -> dict[str, Any]:
    sc = load_strategy_config_from_dict(
        {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    )
    spec = ipr.FoldSpec(
        name=f"post_fresh_{label}",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        epoch_start=POST_FRESH_START,
        epoch_end=epoch_max,
        strategy_overrides={"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}},
    )
    print(f"\n--- post_fresh @ {initial_bankroll} BNB ({label}): "
          f"epochs [{POST_FRESH_START}..{epoch_max}] ---")
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

    # Per-bet detail + max drawdown tracking
    bet_detail: list[dict[str, Any]] = []
    bankrolls: list[float] = [initial_bankroll]
    peak = initial_bankroll
    max_dd_frac = 0.0
    with open(trades_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            br = float(row["bankroll_bnb"])
            bankrolls.append(br)
            if br > peak:
                peak = br
            if peak > 0:
                dd = (peak - br) / peak
                if dd > max_dd_frac:
                    max_dd_frac = dd
            if row.get("action") == "BET":
                bet_detail.append({
                    "epoch": int(row["epoch"]),
                    "side": row["direction"],
                    "bet_size_bnb": float(row["bet_size_bnb"]),
                    "profit_bnb": float(row["profit_bnb"]),
                    "bankroll_after": br,
                    "outcome": "win" if float(row["profit_bnb"]) > 0 else "loss",
                })

    profits = [b["profit_bnb"] for b in bet_detail]
    bet_sizes = [b["bet_size_bnb"] for b in bet_detail]

    return {
        "initial_bankroll": initial_bankroll,
        "label": label,
        "epoch_start": POST_FRESH_START,
        "epoch_end": epoch_max,
        "summary": {k: summary[k] for k in
                    ("backtest_round_count", "num_bets", "num_wins", "win_rate",
                     "net_pnl_bnb", "final_bankroll_bnb", "first_epoch", "last_epoch",
                     "gross_profit_bnb", "gross_loss_bnb", "bet_rate", "num_skips")
                    if k in summary},
        "max_drawdown_frac": max_dd_frac,
        "skip_counts_by_reason": summary.get("skip_counts_by_reason", {}),
        "bet_detail": bet_detail,
        "mean_bet_size_bnb": statistics.mean(bet_sizes) if bet_sizes else 0.0,
        "median_bet_size_bnb": statistics.median(bet_sizes) if bet_sizes else 0.0,
        "max_bet_size_bnb": max(bet_sizes) if bet_sizes else 0.0,
        "mean_profit_per_bet": statistics.mean(profits) if profits else 0.0,
        "stdev_profit_per_bet": statistics.stdev(profits) if len(profits) > 1 else 0.0,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    t_all = time.time()

    print("--- loading rounds (canonical + extended) ---")
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    max_epoch_in_dataset = max(r.epoch for r in all_rounds)
    print(f"  loaded {len(all_rounds)} rounds; range "
          f"[{all_rounds[0].epoch}..{max_epoch_in_dataset}]")
    print(f"  post_fresh slice: [{POST_FRESH_START}..{max_epoch_in_dataset}] "
          f"= {max_epoch_in_dataset - POST_FRESH_START + 1} epochs in range")

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1

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

    out_root = Path(tempfile.mkdtemp(prefix="step10_"))

    run_5 = run_post_fresh(
        initial_bankroll=5.0, label="5bnb",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset, out_root=out_root,
        epoch_max=max_epoch_in_dataset,
    )
    run_50 = run_post_fresh(
        initial_bankroll=50.0, label="50bnb",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset, out_root=out_root,
        epoch_max=max_epoch_in_dataset,
    )

    # --- Print per-scale comparison ---
    for run in (run_5, run_50):
        s = run["summary"]
        print(f"\n=== Summary @ {run['initial_bankroll']} BNB ===")
        print(f"  epochs={s['backtest_round_count']}  bets={s['num_bets']}  "
              f"wins={s['num_wins']}  WR={s['win_rate']:.4f}")
        print(f"  PnL={s['net_pnl_bnb']:+.4f} BNB  "
              f"final={s['final_bankroll_bnb']:.4f}  "
              f"max_dd={run['max_drawdown_frac']*100:.2f}%")
        print(f"  bet_rate={s['bet_rate']*100:.2f}%  "
              f"gross_profit={s['gross_profit_bnb']:+.4f}  "
              f"gross_loss={s['gross_loss_bnb']:+.4f}")
        print(f"  mean_bet={run['mean_bet_size_bnb']:.4f}  "
              f"median_bet={run['median_bet_size_bnb']:.4f}  "
              f"max_bet={run['max_bet_size_bnb']:.4f}")
        print(f"  per-bet: mean={run['mean_profit_per_bet']:+.5f}  "
              f"stdev={run['stdev_profit_per_bet']:.5f}")
        skips = s.get("skip_counts_by_reason", run["skip_counts_by_reason"])
        # Get top skip reasons
        skip_items = sorted(skips.items(), key=lambda x: -x[1])
        top_skips = [(k, v) for k, v in skip_items if v >= 5]  # threshold
        # Show only well-known reasons, group the rest as "other_singletons"
        known_keys = {
            "gate_no_signal", "pool_below_minimum", "payout_below_floor",
            "risk_cooldown_active", "risk_drawdown_breaker_fired",
            "risk_bankroll_below_min", "bet_size_below_min",
            "gate_no_btc_klines", "gate_no_eth_klines", "gate_no_sol_klines",
            "kline_fetch_transient_failure",
        }
        print(f"  skip reasons:")
        shown_total = 0
        for k, v in skip_items:
            if k in known_keys:
                print(f"    {k:>35s}: {v:>5d}")
                shown_total += v
        unknown_total = sum(v for k, v in skip_items if k not in known_keys)
        if unknown_total > 0:
            print(f"    {'(other reasons)':>35s}: {unknown_total:>5d}")

    # Print per-bet table for 5 BNB scale (same epochs/bets as 50 BNB, just different sizes)
    print(f"\n=== Per-bet detail (5 BNB scale) ===")
    print(f"  {'epoch':>7s} {'side':>5s} {'bet':>8s} {'profit':>10s} {'bankroll':>10s} {'outcome':>7s}")
    for b in run_5["bet_detail"]:
        print(f"  {b['epoch']:>7d} {b['side']:>5s} "
              f"{b['bet_size_bnb']:>8.4f} {b['profit_bnb']:>+10.5f} "
              f"{b['bankroll_after']:>10.4f} {b['outcome']:>7s}")

    print(f"\n=== Per-bet detail (50 BNB scale) ===")
    print(f"  {'epoch':>7s} {'side':>5s} {'bet':>8s} {'profit':>10s} {'bankroll':>10s} {'outcome':>7s}")
    for b in run_50["bet_detail"]:
        print(f"  {b['epoch']:>7d} {b['side']:>5s} "
              f"{b['bet_size_bnb']:>8.4f} {b['profit_bnb']:>+10.5f} "
              f"{b['bankroll_after']:>10.4f} {b['outcome']:>7s}")

    out_path = REPO / "var" / "strategy_review" / "post_fresh_backtest_step10_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "post_fresh_start": POST_FRESH_START,
                "epoch_max_in_dataset": max_epoch_in_dataset,
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
