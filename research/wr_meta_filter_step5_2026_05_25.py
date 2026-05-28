"""WR-based runtime meta-filter — Step 5 of regime characterization.

Question: can a "pause if recent rolling WR < X" filter improve total PnL by
skipping bad regimes (like extension cohort) without sacrificing CV5's edge?

Filter semantics (shadow-betting during pause):
  - Maintain rolling deque of last N bet outcomes (1=win, 0=loss).
  - State: paused (bool), initially False.
  - For each canonical gate-fire (i.e., each bet the unfiltered strategy would
    take), in chronological order:
      * If paused: skip the bet (don't commit BNB), but record this round's
        hypothetical outcome into the rolling window. Profit = 0 for this bet.
      * If not paused: take the bet, record outcome, profit accrues.
    Then update state:
      * Once window has N samples, compute WR = sum/N.
      * If WR < X → set paused=True.
      * If WR >= X → set paused=False.

This gives a clean reactive filter with no arbitrary cooldown parameter —
recovery is driven by the same statistic that triggered the pause.

Frozen invariants:
  - kline_cutoff_seconds=2 (canonical, HARD INVARIANT).
  - Canonical (3, 7, 15) lookbacks.
  - Real impact-aware settlement (in_process_runner uses pancakebot.settlement).

Output:
  - var/strategy_review/wr_meta_filter_step5_data.json
  - var/strategy_review/2026_05_25_wr_meta_filter_step5.md (written separately)

Wall-clock: ~60s for backtest + <1s for 30-variant grid sweep.
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

from pancakebot.config import load_strategy_config_from_dict  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPOCH_MIN = 422298
EPOCH_MAX = 484000

CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2

# Grid (30 variants)
N_GRID = (10, 20, 50, 100, 200)
X_GRID = (0.45, 0.50, 0.52, 0.55, 0.57, 0.60)

# Cohort tagging (same as Step 3)
def cohort_of(epoch: int) -> str:
    if 422298 <= epoch <= 437561: return "extension"
    if 437562 <= epoch <= 474086: return "cv5"
    if 474880 <= epoch <= 475311: return "holdout"
    if 475312 <= epoch <= 479952: return "ext_v2"
    if 479953 <= epoch <= 483191: return "fresh_oos"
    return "post_fresh"


# ---------------------------------------------------------------------------
# Phase 1: canonical backtest full range
# ---------------------------------------------------------------------------

def run_canonical_full_backtest() -> Path:
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

    out_root = Path(tempfile.mkdtemp(prefix="step5_"))
    print(f"--- temp output: {out_root} ---")

    spec = ipr.FoldSpec(
        name="canonical_full_range",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        epoch_start=EPOCH_MIN,
        epoch_end=EPOCH_MAX,
        strategy_overrides={"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}},
    )
    sc = load_strategy_config_from_dict(spec.strategy_overrides)

    print(f"--- running canonical {CANONICAL_LOOKBACKS} cs={CANONICAL_CUTOFF} "
          f"on epochs [{EPOCH_MIN}..{EPOCH_MAX}] ---")
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
        initial_bankroll_bnb=5.0,
        treasury_fee_fraction=0.03,
        min_bet_amount_bnb=0.001,
    )
    elapsed = time.time() - t0
    print(f"  bets={summary['num_bets']} wins={summary['num_wins']} "
          f"wr={summary['win_rate']:.4f} pnl={summary['net_pnl_bnb']:+.4f} BNB  "
          f"({elapsed:.1f}s)")

    return out_root / spec.name / "trades.csv"


# ---------------------------------------------------------------------------
# Phase 2: extract bets
# ---------------------------------------------------------------------------

def load_bets(trades_csv: Path) -> list[dict[str, Any]]:
    """Load BET rows from trades.csv. Each row -> dict with epoch, win, profit."""
    bets = []
    with open(trades_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("action") != "BET":
                continue
            profit = float(row["profit_bnb"])
            bets.append({
                "epoch": int(row["epoch"]),
                "win": profit > 0,
                "profit_bnb": profit,
                "cohort": cohort_of(int(row["epoch"])),
            })
    # Already in chronological epoch order from the backtest, but be safe:
    bets.sort(key=lambda b: b["epoch"])
    return bets


# ---------------------------------------------------------------------------
# Phase 3: filter simulator
# ---------------------------------------------------------------------------

def simulate_filter(bets: list[dict[str, Any]], N: int, X: float) -> dict[str, Any]:
    """Apply shadow-betting filter with rolling-WR window of size N, threshold X.

    Returns aggregate stats + per-cohort breakdown.
    """
    window: collections.deque = collections.deque(maxlen=N)
    paused = False

    # Aggregate accumulators
    taken_profit = 0.0
    taken_bets = 0
    skipped_bets = 0
    taken_wins = 0

    # Per-cohort accumulators
    by_cohort: dict[str, dict[str, float]] = {}

    # Pause-event accounting
    n_pause_events = 0  # number of False->True transitions
    pause_run_lengths: list[int] = []
    current_pause_run = 0
    bets_paused_during_window_eval = 0  # bets where we were paused at decision

    for b in bets:
        coh = b["cohort"]
        if coh not in by_cohort:
            by_cohort[coh] = {
                "taken_profit_bnb": 0.0,
                "taken_bets": 0,
                "taken_wins": 0,
                "skipped_bets": 0,
                "skipped_profit_bnb_hypothetical": 0.0,
                "total_bets_offered": 0,
            }
        by_cohort[coh]["total_bets_offered"] += 1

        if paused:
            # Skip the bet, but record outcome shadow-style.
            skipped_bets += 1
            current_pause_run += 1
            by_cohort[coh]["skipped_bets"] += 1
            by_cohort[coh]["skipped_profit_bnb_hypothetical"] += b["profit_bnb"]
            bets_paused_during_window_eval += 1
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

        # Update rolling window with this bet's outcome (shadow or real).
        window.append(1 if b["win"] else 0)

        # Update pause state for next bet (only after window has N samples).
        if len(window) >= N:
            wr = sum(window) / N
            if not paused and wr < X:
                paused = True
                n_pause_events += 1
            elif paused and wr >= X:
                paused = False

    # Tail pause run (if we end paused, capture the last episode)
    if current_pause_run > 0:
        pause_run_lengths.append(current_pause_run)

    mean_pause_len = (statistics.mean(pause_run_lengths)
                      if pause_run_lengths else 0.0)
    median_pause_len = (statistics.median(pause_run_lengths)
                        if pause_run_lengths else 0.0)
    max_pause_len = max(pause_run_lengths) if pause_run_lengths else 0

    return {
        "N": N,
        "X": X,
        "total_taken_profit_bnb": taken_profit,
        "total_taken_bets": taken_bets,
        "total_taken_wins": taken_wins,
        "total_taken_win_rate": taken_wins / taken_bets if taken_bets else 0.0,
        "total_skipped_bets": skipped_bets,
        "total_offered_bets": taken_bets + skipped_bets,
        "n_pause_events": n_pause_events,
        "n_pause_episodes_completed": len(pause_run_lengths),
        "mean_pause_run_len": mean_pause_len,
        "median_pause_run_len": median_pause_len,
        "max_pause_run_len": max_pause_len,
        "by_cohort": by_cohort,
    }


def aggregate_no_filter(bets: list[dict[str, Any]]) -> dict[str, Any]:
    """Unfiltered canonical baseline aggregates (for comparison)."""
    by_cohort: dict[str, dict[str, Any]] = {}
    total_profit = 0.0
    total_bets = 0
    total_wins = 0
    for b in bets:
        coh = b["cohort"]
        if coh not in by_cohort:
            by_cohort[coh] = {
                "taken_profit_bnb": 0.0,
                "taken_bets": 0,
                "taken_wins": 0,
                "skipped_bets": 0,
                "skipped_profit_bnb_hypothetical": 0.0,
                "total_bets_offered": 0,
            }
        by_cohort[coh]["taken_profit_bnb"] += b["profit_bnb"]
        by_cohort[coh]["taken_bets"] += 1
        by_cohort[coh]["total_bets_offered"] += 1
        if b["win"]:
            by_cohort[coh]["taken_wins"] += 1
            total_wins += 1
        total_profit += b["profit_bnb"]
        total_bets += 1
    return {
        "total_taken_profit_bnb": total_profit,
        "total_taken_bets": total_bets,
        "total_taken_wins": total_wins,
        "total_taken_win_rate": total_wins / total_bets if total_bets else 0.0,
        "total_skipped_bets": 0,
        "total_offered_bets": total_bets,
        "by_cohort": by_cohort,
    }


# ---------------------------------------------------------------------------
# Phase 4: grid sweep
# ---------------------------------------------------------------------------

def main() -> None:
    t_all = time.time()

    trades_csv = run_canonical_full_backtest()
    bets = load_bets(trades_csv)
    print(f"\n--- extracted {len(bets)} BET rows from full backtest ---")

    # Print canonical baseline + per-cohort breakdown
    baseline = aggregate_no_filter(bets)
    print(f"\n=== unfiltered canonical baseline ===")
    print(f"  total PnL = {baseline['total_taken_profit_bnb']:+.4f} BNB "
          f"({baseline['total_taken_bets']} bets, "
          f"WR={baseline['total_taken_win_rate']:.4f})")
    for coh in ("extension", "cv5", "holdout", "ext_v2", "fresh_oos", "post_fresh"):
        cd = baseline["by_cohort"].get(coh)
        if cd is None:
            continue
        wr = cd['taken_wins'] / cd['taken_bets'] if cd['taken_bets'] else 0
        print(f"  {coh:>12}: PnL={cd['taken_profit_bnb']:+8.4f}  "
              f"bets={cd['taken_bets']:>4}  wins={cd['taken_wins']:>4}  "
              f"WR={wr:.4f}")

    # Sweep grid
    print(f"\n--- grid sweep: {len(N_GRID)} N x {len(X_GRID)} X = "
          f"{len(N_GRID)*len(X_GRID)} variants ---")
    variants = []
    for N in N_GRID:
        for X in X_GRID:
            t_v = time.time()
            v = simulate_filter(bets, N, X)
            v["elapsed_seconds"] = time.time() - t_v
            variants.append(v)
            delta_pnl = v["total_taken_profit_bnb"] - baseline["total_taken_profit_bnb"]
            ext_pnl = v["by_cohort"].get("extension", {}).get("taken_profit_bnb", 0.0)
            cv5_pnl = v["by_cohort"].get("cv5", {}).get("taken_profit_bnb", 0.0)
            print(f"  N={N:>3} X={X:.2f}: filt PnL={v['total_taken_profit_bnb']:+8.4f}  "
                  f"d={delta_pnl:+8.4f}  bets={v['total_taken_bets']:>4}/{v['total_offered_bets']:<4}  "
                  f"ext={ext_pnl:+7.4f}  cv5={cv5_pnl:+7.4f}  "
                  f"pause_events={v['n_pause_events']}  "
                  f"mean_pause={v['mean_pause_run_len']:.1f}")

    # Find "improvement" variants: total filtered PnL > baseline AND extension PnL > -14.97
    print(f"\n--- candidates with total>baseline AND extension>-14.97 ---")
    improving = []
    for v in variants:
        total_better = v["total_taken_profit_bnb"] > baseline["total_taken_profit_bnb"]
        ext = v["by_cohort"].get("extension", {}).get("taken_profit_bnb", 0.0)
        ext_better = ext > -14.97
        if total_better and ext_better:
            improving.append(v)
            cv5 = v["by_cohort"].get("cv5", {}).get("taken_profit_bnb", 0.0)
            delta = v["total_taken_profit_bnb"] - baseline["total_taken_profit_bnb"]
            print(f"  N={v['N']} X={v['X']:.2f}: total={v['total_taken_profit_bnb']:+.4f} "
                  f"(d={delta:+.4f}) ext={ext:+.4f} cv5={cv5:+.4f}")
    if not improving:
        print(f"  NONE — filter strictly underperforms canonical on this objective")

    # Persist
    out_path = REPO / "var" / "strategy_review" / "wr_meta_filter_step5_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "N_grid": list(N_GRID),
                "X_grid": list(X_GRID),
                "pause_semantics": "shadow_betting_during_pause",
            },
            "baseline": baseline,
            "variants": variants,
            "improving_variants_count": len(improving),
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
