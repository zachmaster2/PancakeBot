"""Random-shuffle k-fold CV research — Step 4 of regime characterization.

Question: does CV5's +50 BNB edge survive when we swap the contiguous-block
fold structure for random-shuffle folds?

If the edge is fold-structure-sensitive (e.g., one "good" time block carries
the rest), shuffling the fold structure should destroy or weaken it.
If the edge is uniformly distributed across the CV5 epoch range, shuffling
should give per-fold PnLs tightly centered around total/K with all folds
positive.

Methodology (single canonical run + permutation analysis):

1. Run canonical (3, 7, 15) cs=2 ONCE on CV5 range [437562..474086].
2. Extract per-epoch profit from the resulting trades.csv (BET rows only).
3. Compute the contiguous-block 5-fold baseline (sort epochs, slice into 5
   contiguous chunks, sum profits per chunk).
4. For each seed in 0..(N_SEEDS-1):
   - Shuffle epoch list with that seed.
   - Partition into K=5 chunks → per-fold PnL.
   - Partition into K=10 chunks → per-fold PnL.
5. Aggregate per-fold positive-rate, mean, std, min, max across all
   (seed × fold) draws.

Output: var/strategy_review/random_shuffle_cv_step4_data.json +
        var/strategy_review/2026_05_25_random_shuffle_cv_step4.md

Frozen invariants:
  - kline_cutoff_seconds=2 (canonical, per feedback_kline_cutoff_strategy_invariant).
  - Canonical (3, 7, 15) lookbacks.
  - Real impact-aware settlement via in_process_runner.run_fold.

Wall-clock: ~60s one-time backtest + ~5s permutation analysis.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.in_process_runner as ipr  # noqa: E402
from pancakebot.config import load_strategy_config_from_dict  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# CV5 epoch range (per project_holdout_slice memory).
CV5_EPOCH_START = 437562
CV5_EPOCH_END = 474086

# Canonical strategy.
CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2

# Permutation parameters.
N_SEEDS = 100
K_VARIANTS = (5, 10)


# ---------------------------------------------------------------------------
# Phase 1: canonical CV5 backtest → trades.csv
# ---------------------------------------------------------------------------

def run_canonical_cv5_backtest() -> tuple[Path, dict[str, Any]]:
    """Run one canonical backtest across the CV5 epoch range. Returns
    (trades_csv_path, summary_dict).
    """
    print("--- loading rounds (canonical only — CV5 range is all-canonical) ---")
    all_rounds = ipr._load_all_rounds(use_extended_data=False)
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
    )
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
    )
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
    )
    print(f"  BTC={len(btc)} ETH={len(eth)} SOL={len(sol)} epochs loaded")

    out_root = Path(tempfile.mkdtemp(prefix="cv5_step4_"))
    print(f"--- temp output: {out_root} ---")

    spec = ipr.FoldSpec(
        name="canonical_cv5_full_range",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        epoch_start=CV5_EPOCH_START,
        epoch_end=CV5_EPOCH_END,
        strategy_overrides={"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}},
    )
    sc = load_strategy_config_from_dict(spec.strategy_overrides)

    print(f"--- running canonical {CANONICAL_LOOKBACKS} cs={CANONICAL_CUTOFF} "
          f"on epochs [{CV5_EPOCH_START}..{CV5_EPOCH_END}] ---")
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

    trades_csv = out_root / spec.name / "trades.csv"
    return trades_csv, summary


# ---------------------------------------------------------------------------
# Phase 2: extract per-bet records
# ---------------------------------------------------------------------------

def load_bets(trades_csv: Path) -> list[tuple[int, float]]:
    """Read trades.csv, return list of (epoch, profit_bnb) for action==BET rows."""
    bets: list[tuple[int, float]] = []
    with open(trades_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("action") == "BET":
                bets.append((int(row["epoch"]), float(row["profit_bnb"])))
    return bets


# ---------------------------------------------------------------------------
# Phase 3: partitioning logic
# ---------------------------------------------------------------------------

def contiguous_block_partition(items: list[tuple[int, float]], k: int) -> list[list[tuple[int, float]]]:
    """Sort by epoch, split into k contiguous chunks of (near-)equal size."""
    items_sorted = sorted(items, key=lambda x: x[0])
    n = len(items_sorted)
    chunks: list[list[tuple[int, float]]] = []
    base = n // k
    extras = n % k
    idx = 0
    for i in range(k):
        size = base + (1 if i < extras else 0)
        chunks.append(items_sorted[idx:idx + size])
        idx += size
    return chunks


def random_shuffle_partition(items: list[tuple[int, float]], k: int, seed: int) -> list[list[tuple[int, float]]]:
    """Shuffle deterministically with given seed, then split into k chunks."""
    items_shuffled = list(items)
    rng = random.Random(seed)
    rng.shuffle(items_shuffled)
    n = len(items_shuffled)
    chunks: list[list[tuple[int, float]]] = []
    base = n // k
    extras = n % k
    idx = 0
    for i in range(k):
        size = base + (1 if i < extras else 0)
        chunks.append(items_shuffled[idx:idx + size])
        idx += size
    return chunks


def chunk_stats(chunks: list[list[tuple[int, float]]]) -> list[dict[str, Any]]:
    out = []
    for i, ch in enumerate(chunks):
        profits = [p for _, p in ch]
        epochs = [e for e, _ in ch]
        out.append({
            "fold_idx": i,
            "n_bets": len(ch),
            "epoch_min": min(epochs) if epochs else None,
            "epoch_max": max(epochs) if epochs else None,
            "pnl_bnb": sum(profits),
            "n_wins": sum(1 for p in profits if p > 0),
        })
    return out


# ---------------------------------------------------------------------------
# Phase 4: permutation analysis
# ---------------------------------------------------------------------------

def aggregate_random_shuffle(bets: list[tuple[int, float]], k: int, n_seeds: int) -> dict[str, Any]:
    """Run n_seeds random shuffles partitioned into k folds. Aggregate
    per-fold PnL distribution across all seeds.
    """
    all_fold_pnls: list[float] = []
    per_seed_fold_pnls: list[list[float]] = []
    n_pos_per_seed: list[int] = []
    seed_total_pnls: list[float] = []

    for seed in range(n_seeds):
        chunks = random_shuffle_partition(bets, k, seed)
        fold_pnls = [sum(p for _, p in ch) for ch in chunks]
        per_seed_fold_pnls.append(fold_pnls)
        all_fold_pnls.extend(fold_pnls)
        n_pos_per_seed.append(sum(1 for p in fold_pnls if p > 0))
        seed_total_pnls.append(sum(fold_pnls))

    return {
        "k": k,
        "n_seeds": n_seeds,
        "total_folds": len(all_fold_pnls),
        "per_fold_pnl_mean": statistics.mean(all_fold_pnls),
        "per_fold_pnl_stdev": statistics.stdev(all_fold_pnls) if len(all_fold_pnls) > 1 else 0.0,
        "per_fold_pnl_min": min(all_fold_pnls),
        "per_fold_pnl_max": max(all_fold_pnls),
        "per_fold_positive_rate": sum(1 for p in all_fold_pnls if p > 0) / len(all_fold_pnls),
        "all_folds_positive_seed_count": sum(1 for c in n_pos_per_seed if c == k),
        "any_fold_negative_seed_count": sum(1 for c in n_pos_per_seed if c < k),
        "seed_total_pnl_mean": statistics.mean(seed_total_pnls),
        "seed_total_pnl_stdev": statistics.stdev(seed_total_pnls) if len(seed_total_pnls) > 1 else 0.0,
        "seed_total_pnl_min": min(seed_total_pnls),
        "seed_total_pnl_max": max(seed_total_pnls),
        # First 5 seeds for spot-check
        "first_5_seeds": [
            {"seed": s, "fold_pnls": per_seed_fold_pnls[s], "total": seed_total_pnls[s]}
            for s in range(min(5, n_seeds))
        ],
    }


# ---------------------------------------------------------------------------
# Phase 5: main
# ---------------------------------------------------------------------------

def main() -> None:
    t_all = time.time()

    # Phase 1: canonical backtest
    trades_csv, full_summary = run_canonical_cv5_backtest()

    # Phase 2: load bets
    bets = load_bets(trades_csv)
    total_bets = len(bets)
    total_pnl = sum(p for _, p in bets)
    print(f"\n--- extracted {total_bets} BET rows; "
          f"sum profit = {total_pnl:+.4f} BNB ---")

    # Phase 3a: contiguous-block baselines for K=5 and K=10
    contiguous_results: dict[int, list[dict[str, Any]]] = {}
    for k in K_VARIANTS:
        chunks = contiguous_block_partition(bets, k)
        stats = chunk_stats(chunks)
        contiguous_results[k] = stats
        total_pos = sum(1 for s in stats if s["pnl_bnb"] > 0)
        print(f"\n--- contiguous-block K={k} baseline ---")
        for s in stats:
            print(f"  f{s['fold_idx']+1}: epochs [{s['epoch_min']}..{s['epoch_max']}]  "
                  f"n_bets={s['n_bets']}  PnL={s['pnl_bnb']:+.4f}  wins={s['n_wins']}")
        print(f"  total positive folds: {total_pos}/{k}")

    # Phase 4: random-shuffle aggregates
    shuffle_results: dict[int, dict[str, Any]] = {}
    for k in K_VARIANTS:
        print(f"\n--- random-shuffle K={k}, N={N_SEEDS} seeds ---")
        t0 = time.time()
        agg = aggregate_random_shuffle(bets, k, N_SEEDS)
        elapsed = time.time() - t0
        shuffle_results[k] = agg
        print(f"  elapsed={elapsed:.1f}s")
        print(f"  per-fold PnL: mean={agg['per_fold_pnl_mean']:+.4f}  "
              f"stdev={agg['per_fold_pnl_stdev']:.4f}  "
              f"min={agg['per_fold_pnl_min']:+.4f}  "
              f"max={agg['per_fold_pnl_max']:+.4f}")
        print(f"  per-fold positive rate: {agg['per_fold_positive_rate']*100:.1f}% "
              f"(of {agg['total_folds']} folds)")
        print(f"  seeds where all {k} folds positive: "
              f"{agg['all_folds_positive_seed_count']}/{N_SEEDS}")
        print(f"  seeds with at least one negative fold: "
              f"{agg['any_fold_negative_seed_count']}/{N_SEEDS}")
        print(f"  seed total PnL: mean={agg['seed_total_pnl_mean']:+.4f}  "
              f"stdev={agg['seed_total_pnl_stdev']:.4f}  "
              f"min={agg['seed_total_pnl_min']:+.4f}  "
              f"max={agg['seed_total_pnl_max']:+.4f}")

    # Phase 5: persist
    out_path = REPO / "var" / "strategy_review" / "random_shuffle_cv_step4_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "cv5_epoch_start": CV5_EPOCH_START,
                "cv5_epoch_end": CV5_EPOCH_END,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "n_seeds": N_SEEDS,
                "k_variants": list(K_VARIANTS),
            },
            "canonical_backtest_summary": {
                "num_bets": full_summary["num_bets"],
                "num_wins": full_summary["num_wins"],
                "win_rate": full_summary["win_rate"],
                "net_pnl_bnb": full_summary["net_pnl_bnb"],
                "first_epoch": full_summary["first_epoch"],
                "last_epoch": full_summary["last_epoch"],
            },
            "bet_level_total_pnl": total_pnl,
            "bet_level_total_count": total_bets,
            "contiguous_block_results": {
                str(k): contiguous_results[k] for k in K_VARIANTS
            },
            "random_shuffle_results": {
                str(k): shuffle_results[k] for k in K_VARIANTS
            },
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
