"""Step 7 — bankroll scale rerun + WR meta-filter at 50 BNB.

Three deliverables in one driver:

1. Canonical (3, 7, 15) cs=2 full-range backtest at 5.0 BNB scale,
   per-cohort breakdown including drawdown trips + cooldown skips.

2. Same at 50.0 BNB scale.

3. WR meta-filter sweep (Step 5's 30-variant grid) applied to the 50 BNB
   bet timeline. Compare best variant at 50 BNB vs Step 5's best at 5 BNB.

Output:
  - var/strategy_review/bankroll_scale_rerun_step7_data.json
  - var/strategy_review/2026_05_26_bankroll_scale_rerun_step7.md (written separately)

Frozen invariants:
  - kline_cutoff_seconds=2 (HARD).
  - Canonical (3, 7, 15) lookbacks.
  - Real impact-aware settlement.
  - Same shadow-betting filter semantics as Step 5.
"""
from __future__ import annotations

import collections
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


EPOCH_MIN = 422298
EPOCH_MAX = 484000

CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2

# WR meta-filter grid (same as Step 5)
N_GRID = (10, 20, 50, 100, 200)
X_GRID = (0.45, 0.50, 0.52, 0.55, 0.57, 0.60)

COHORT_ORDER = ("extension", "cv5", "holdout", "ext_v2", "fresh_oos", "post_fresh")


def cohort_of(epoch: int) -> str:
    if 422298 <= epoch <= 437561: return "extension"
    if 437562 <= epoch <= 474086: return "cv5"
    if 474880 <= epoch <= 475311: return "holdout"
    if 475312 <= epoch <= 479952: return "ext_v2"
    if 479953 <= epoch <= 483191: return "fresh_oos"
    return "post_fresh"


# ---------------------------------------------------------------------------
# Phase 1: canonical backtest + per-cohort accounting
# ---------------------------------------------------------------------------

def run_canonical_and_accumulate(*, initial_bankroll: float, label: str,
                                  all_rounds: list, btc: dict, eth: dict, sol: dict,
                                  earliest_offset: int, out_root: Path) -> dict[str, Any]:
    from pancakebot.config import load_strategy_config_from_dict
    spec = ipr.FoldSpec(
        name=f"canonical_{label}",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        epoch_start=EPOCH_MIN,
        epoch_end=EPOCH_MAX,
        strategy_overrides={"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}},
    )
    sc = load_strategy_config_from_dict(spec.strategy_overrides)

    print(f"\n--- canonical backtest @ {initial_bankroll} BNB ({label}) ---")
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

    # Read trades.csv, build per-cohort stats AND per-bet timeline
    per_cohort: dict[str, dict[str, Any]] = {
        c: {
            "n_rounds": 0, "n_bets": 0, "n_wins": 0, "pnl_bnb": 0.0,
            "skip_drawdown_breaker": 0, "skip_cooldown": 0,
            "skip_bankroll_below_min": 0, "skip_gate_no_signal": 0,
            "skip_pool_below_minimum": 0, "skip_payout_below_floor": 0,
            "skip_other": 0,
        } for c in COHORT_ORDER
    }
    bet_timeline: list[dict[str, Any]] = []
    bull_pool_dummy_skipped = 0

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
                side = row["direction"]
                win = profit > 0
                per_cohort[coh]["n_bets"] += 1
                per_cohort[coh]["pnl_bnb"] += profit
                if win:
                    per_cohort[coh]["n_wins"] += 1
                bet_timeline.append({
                    "epoch": epoch, "cohort": coh, "side": side,
                    "win": win, "profit_bnb": profit, "bet_size_bnb": bet_size,
                })
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

    return {
        "initial_bankroll": initial_bankroll,
        "label": label,
        "summary": {k: summary[k] for k in
                    ("num_bets", "num_wins", "win_rate", "net_pnl_bnb",
                     "final_bankroll_bnb", "first_epoch", "last_epoch",
                     "gross_profit_bnb", "gross_loss_bnb")
                    if k in summary},
        "skip_counts_by_reason": summary.get("skip_counts_by_reason", {}),
        "per_cohort": per_cohort,
        "bet_timeline": bet_timeline,
        "trades_csv": str(trades_csv),
    }


# ---------------------------------------------------------------------------
# Phase 2: WR meta-filter (shadow-betting, identical to Step 5)
# ---------------------------------------------------------------------------

def apply_wr_filter(bet_timeline: list[dict[str, Any]], N: int, X: float) -> dict[str, Any]:
    window: collections.deque = collections.deque(maxlen=N)
    paused = False

    taken_profit = 0.0
    taken_bets = 0
    taken_wins = 0
    skipped_bets = 0
    by_cohort: dict[str, dict[str, Any]] = {
        c: {
            "taken_profit_bnb": 0.0, "taken_bets": 0, "taken_wins": 0,
            "skipped_bets": 0, "skipped_profit_hypothetical": 0.0,
            "total_bets_offered": 0,
        } for c in COHORT_ORDER
    }
    n_pause_events = 0
    pause_run_lengths: list[int] = []
    current_pause_run = 0

    for b in bet_timeline:
        coh = b["cohort"]
        by_cohort[coh]["total_bets_offered"] += 1

        if paused:
            skipped_bets += 1
            current_pause_run += 1
            by_cohort[coh]["skipped_bets"] += 1
            by_cohort[coh]["skipped_profit_hypothetical"] += b["profit_bnb"]
        else:
            taken_profit += b["profit_bnb"]
            taken_bets += 1
            if b["win"]:
                taken_wins += 1
            by_cohort[coh]["taken_profit_bnb"] += b["profit_bnb"]
            by_cohort[coh]["taken_bets"] += 1
            if b["win"]:
                by_cohort[coh]["taken_wins"] += 1
            if current_pause_run > 0:
                pause_run_lengths.append(current_pause_run)
                current_pause_run = 0

        window.append(1 if b["win"] else 0)

        if len(window) >= N:
            wr = sum(window) / N
            if not paused and wr < X:
                paused = True
                n_pause_events += 1
            elif paused and wr >= X:
                paused = False

    if current_pause_run > 0:
        pause_run_lengths.append(current_pause_run)

    return {
        "N": N, "X": X,
        "total_taken_profit_bnb": taken_profit,
        "total_taken_bets": taken_bets,
        "total_taken_wins": taken_wins,
        "total_taken_win_rate": taken_wins / taken_bets if taken_bets else 0.0,
        "total_skipped_bets": skipped_bets,
        "total_offered_bets": taken_bets + skipped_bets,
        "n_pause_events": n_pause_events,
        "mean_pause_run_len": statistics.mean(pause_run_lengths) if pause_run_lengths else 0.0,
        "max_pause_run_len": max(pause_run_lengths) if pause_run_lengths else 0,
        "by_cohort": by_cohort,
    }


def baseline_unfiltered(bet_timeline: list[dict[str, Any]]) -> dict[str, Any]:
    total_profit = 0.0
    total_bets = len(bet_timeline)
    total_wins = sum(1 for b in bet_timeline if b["win"])
    by_cohort: dict[str, dict[str, Any]] = {
        c: {"taken_profit_bnb": 0.0, "taken_bets": 0, "taken_wins": 0}
        for c in COHORT_ORDER
    }
    for b in bet_timeline:
        coh = b["cohort"]
        by_cohort[coh]["taken_profit_bnb"] += b["profit_bnb"]
        by_cohort[coh]["taken_bets"] += 1
        if b["win"]:
            by_cohort[coh]["taken_wins"] += 1
        total_profit += b["profit_bnb"]
    return {
        "total_taken_profit_bnb": total_profit,
        "total_taken_bets": total_bets,
        "total_taken_wins": total_wins,
        "total_taken_win_rate": total_wins / total_bets if total_bets else 0.0,
        "by_cohort": by_cohort,
    }


# ---------------------------------------------------------------------------
# Phase 3: main
# ---------------------------------------------------------------------------

def main() -> None:
    t_all = time.time()

    # Load everything ONCE
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

    out_root = Path(tempfile.mkdtemp(prefix="step7_"))
    print(f"--- temp output: {out_root} ---")

    # --- Run canonical at 5 BNB ---
    run_5 = run_canonical_and_accumulate(
        initial_bankroll=5.0, label="5bnb",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset, out_root=out_root,
    )

    # --- Run canonical at 50 BNB ---
    run_50 = run_canonical_and_accumulate(
        initial_bankroll=50.0, label="50bnb",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset, out_root=out_root,
    )

    # --- Print per-cohort comparison ---
    print(f"\n=== Per-cohort comparison ({len(COHORT_ORDER)} cohorts) ===")
    print(f"{'cohort':>12s} {'rounds':>7s}  "
          f"{'5BNB:bets':>10s} {'WR':>7s} {'PnL':>10s} {'dd':>4s} {'cd':>5s}  "
          f"{'50BNB:bets':>11s} {'WR':>7s} {'PnL':>10s} {'dd':>4s} {'cd':>5s}")
    for coh in COHORT_ORDER:
        c5 = run_5["per_cohort"][coh]
        c50 = run_50["per_cohort"][coh]
        wr5 = c5["n_wins"] / c5["n_bets"] if c5["n_bets"] else 0
        wr50 = c50["n_wins"] / c50["n_bets"] if c50["n_bets"] else 0
        print(f"{coh:>12s} {c5['n_rounds']:>7d}  "
              f"{c5['n_bets']:>10d} {wr5:>7.4f} {c5['pnl_bnb']:>+10.4f} "
              f"{c5['skip_drawdown_breaker']:>4d} {c5['skip_cooldown']:>5d}  "
              f"{c50['n_bets']:>11d} {wr50:>7.4f} {c50['pnl_bnb']:>+10.4f} "
              f"{c50['skip_drawdown_breaker']:>4d} {c50['skip_cooldown']:>5d}")

    # --- WR meta-filter sweep at 50 BNB ---
    print(f"\n=== WR meta-filter sweep @ 50 BNB scale ===")
    print(f"--- grid: {len(N_GRID)*len(X_GRID)} variants ---")
    timeline_50 = run_50["bet_timeline"]
    baseline_50 = baseline_unfiltered(timeline_50)
    print(f"--- baseline (no filter) @ 50 BNB: total PnL = "
          f"{baseline_50['total_taken_profit_bnb']:+.4f} BNB, "
          f"{baseline_50['total_taken_bets']} bets, "
          f"WR={baseline_50['total_taken_win_rate']:.4f} ---")
    for coh in COHORT_ORDER:
        cd = baseline_50["by_cohort"][coh]
        wr = cd["taken_wins"] / cd["taken_bets"] if cd["taken_bets"] else 0
        print(f"  {coh:>12}: PnL={cd['taken_profit_bnb']:+8.4f}  "
              f"bets={cd['taken_bets']:>4}  WR={wr:.4f}")

    variants_50: list[dict[str, Any]] = []
    for N in N_GRID:
        for X in X_GRID:
            v = apply_wr_filter(timeline_50, N, X)
            variants_50.append(v)
            delta = v["total_taken_profit_bnb"] - baseline_50["total_taken_profit_bnb"]
            ext = v["by_cohort"]["extension"]["taken_profit_bnb"]
            cv5 = v["by_cohort"]["cv5"]["taken_profit_bnb"]
            print(f"  N={N:>3} X={X:.2f}: filt={v['total_taken_profit_bnb']:+9.4f}  "
                  f"d={delta:+8.4f}  bets={v['total_taken_bets']:>4}/{v['total_offered_bets']:<4}  "
                  f"ext={ext:+8.4f}  cv5={cv5:+8.4f}  "
                  f"events={v['n_pause_events']}  mean_pause={v['mean_pause_run_len']:.1f}")

    # Top 5 variants by total PnL
    variants_50_sorted = sorted(variants_50, key=lambda v: -v["total_taken_profit_bnb"])
    print(f"\n--- top 5 variants @ 50 BNB by total filtered PnL ---")
    for i, v in enumerate(variants_50_sorted[:5]):
        delta = v["total_taken_profit_bnb"] - baseline_50["total_taken_profit_bnb"]
        print(f"  #{i+1} N={v['N']} X={v['X']:.2f}: total={v['total_taken_profit_bnb']:+.4f}  "
              f"(d={delta:+.4f})")
        for coh in COHORT_ORDER:
            cd = v["by_cohort"][coh]
            tot = cd["taken_profit_bnb"] + cd["skipped_profit_hypothetical"]
            if cd["total_bets_offered"] == 0:
                continue
            print(f"    {coh:>12}: taken_PnL={cd['taken_profit_bnb']:+7.4f} "
                  f"({cd['taken_bets']}/{cd['total_bets_offered']} bets); "
                  f"hypo skipped={cd['skipped_profit_hypothetical']:+7.4f} "
                  f"({cd['skipped_bets']} skipped); "
                  f"unfiltered would be {tot:+7.4f}")

    # --- Load Step 5 (5 BNB) data for side-by-side ---
    step5_path = REPO / "var" / "strategy_review" / "wr_meta_filter_step5_data.json"
    step5_data: dict[str, Any] | None = None
    step5_best: dict[str, Any] | None = None
    if step5_path.exists():
        step5_data = json.loads(step5_path.read_text(encoding="utf-8"))
        # Find best variant (max total_taken_profit_bnb)
        s5_variants = step5_data.get("variants", [])
        if s5_variants:
            step5_best = max(s5_variants, key=lambda v: v["total_taken_profit_bnb"])

    if step5_best is not None:
        print(f"\n=== side-by-side: best variant @ 5 BNB vs 50 BNB ===")
        b5 = step5_best
        b50 = variants_50_sorted[0]
        print(f"  5 BNB best:  N={b5['N']} X={b5['X']:.2f}  "
              f"filt={b5['total_taken_profit_bnb']:+.4f}  "
              f"d={b5['total_taken_profit_bnb'] - step5_data['baseline']['total_taken_profit_bnb']:+.4f}")
        print(f"  50 BNB best: N={b50['N']} X={b50['X']:.2f}  "
              f"filt={b50['total_taken_profit_bnb']:+.4f}  "
              f"d={b50['total_taken_profit_bnb'] - baseline_50['total_taken_profit_bnb']:+.4f}")

    # --- Persist ---
    # Strip bet_timeline from JSON output (large)
    run_5_persist = {k: v for k, v in run_5.items() if k != "bet_timeline"}
    run_50_persist = {k: v for k, v in run_50.items() if k != "bet_timeline"}

    out_path = REPO / "var" / "strategy_review" / "bankroll_scale_rerun_step7_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "N_grid": list(N_GRID), "X_grid": list(X_GRID),
                "pause_semantics": "shadow_betting_during_pause",
            },
            "run_5bnb": run_5_persist,
            "run_50bnb": run_50_persist,
            "wr_filter_50bnb": {
                "baseline": baseline_50,
                "variants": variants_50,
                "top5": variants_50_sorted[:5],
            },
            "step5_best_5bnb": step5_best,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
