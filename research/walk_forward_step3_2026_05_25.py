"""Walk-forward retune research — Step 3 of regime characterization.

For each walk-forward window:
  1. Sweep MTF lookback tuples (cutoff=2 frozen per invariant) on the
     training portion; pick the best by net PnL.
  2. Apply that optimal tuple to the test portion; record realized PnL.
  3. Apply canonical (3, 7, 15) to the same test portion; record baseline.

Cumulative walk-forward PnL (using each window's chosen-from-training
tuple) vs cumulative canonical PnL = answer to "does runtime parameter
retuning beat the fixed canonical?"

Frozen invariants:
  - kline_cutoff_seconds=2 (canonical, per feedback_kline_cutoff_strategy_invariant)
  - Real impact-aware settlement (in_process_runner uses pancakebot.settlement)

Background-runnable. Wall-clock estimate: 20–40 min for ~1600 folds
(load klines once, ~0.5–1.0 s per fold after).
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Monkey-patch extended paths BEFORE importing dependents from the runner.
# The /tmp/ext/extended/ tree contains the recovered extension cohort data.
EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
import research.in_process_runner as ipr  # noqa: E402
ipr._EXT_CLOSED_ROUNDS_PATH = EXT_DIR / "closed_rounds.jsonl"
ipr._EXT_BTC_KLINES_PATH = EXT_DIR / "btc_spot_prices.jsonl"
ipr._EXT_ETH_KLINES_PATH = EXT_DIR / "eth_spot_prices.jsonl"
ipr._EXT_SOL_KLINES_PATH = EXT_DIR / "sol_spot_prices.jsonl"

from pancakebot.config import load_strategy_config_from_dict  # noqa: E402


# ---------------------------------------------------------------------------
# Walk-forward design
# ---------------------------------------------------------------------------

EPOCH_MIN = 422298      # extension start
EPOCH_MAX = 484000      # ~current top of dataset
TRAIN_ROUNDS = 15000    # ~7.5 weeks
TEST_ROUNDS = 3000      # ~1.5 weeks (non-overlapping with next test)
STEP_ROUNDS = 3000

# Lookback grid (cutoff frozen at 2 per project invariant).
# Reuses the historical 153-variant sweep grid with cutoff dropped.
LOOKBACK_GRID = []
for a in (2, 3, 4):
    for b in (5, 7, 10, 15):
        for c in (10, 15, 20, 25, 30):
            if a < b < c:
                LOOKBACK_GRID.append((a, b, c))

CANONICAL = (3, 7, 15)

# Tiny cohort-tag function for reporting
def cohort_of(epoch: int) -> str:
    if 422298 <= epoch <= 437561: return "extension"
    if 437562 <= epoch <= 474086: return "cv5"
    if 474880 <= epoch <= 475311: return "holdout"
    if 475312 <= epoch <= 479952: return "ext_v2"
    if 479953 <= epoch <= 483191: return "fresh_oos"
    return "post_fresh"


def build_windows() -> list[dict[str, int]]:
    windows: list[dict[str, int]] = []
    train_start = EPOCH_MIN
    while True:
        train_end = train_start + TRAIN_ROUNDS - 1
        test_start = train_end + 1
        test_end = test_start + TEST_ROUNDS - 1
        if test_end > EPOCH_MAX:
            break
        windows.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
        })
        train_start += STEP_ROUNDS
    return windows


def make_spec(name: str, lookbacks: tuple[int, int, int],
              epoch_start: int, epoch_end: int) -> ipr.FoldSpec:
    return ipr.FoldSpec(
        name=name,
        kline_cutoff_seconds=2,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        strategy_overrides={"gate": {"mtf_lookbacks": list(lookbacks)}},
    )


def run_one_fold(spec: ipr.FoldSpec,
                  all_rounds: list,
                  btc: dict, eth: dict, sol: dict,
                  earliest_offset: int,
                  output_dir: Path) -> dict[str, Any]:
    # Resolve strategy config per spec (overrides nest within "strategy" wrapper)
    sc = load_strategy_config_from_dict(spec.strategy_overrides)
    return ipr.run_fold(
        spec=spec,
        strategy_cfg=sc,
        all_rounds=all_rounds,
        btc_unified=btc,
        eth_unified=eth,
        sol_unified=sol,
        earliest_offset=earliest_offset,
        output_base_dir=output_dir,
        initial_bankroll_bnb=5.0,
        treasury_fee_fraction=0.03,
        min_bet_amount_bnb=0.001,
    )


def main() -> None:
    t0 = time.time()
    print("--- loading rounds (canonical + extended) ---")
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  loaded {len(all_rounds)} rounds; range "
          f"[{all_rounds[0].epoch}..{all_rounds[-1].epoch}]")

    # Determine kline-load extent: use the widest variant in the grid (c=30)
    # so a single load covers every fold's needs.
    print("--- computing kline load extent (max lookback = 30) ---")
    max_lookback = max(c for _, _, c in LOOKBACK_GRID)
    earliest_offset = 2 + max_lookback + 1   # cutoff + max_lookback + 1
    latest_offset = 3  # cutoff + 1 (for cutoff=2)
    print(f"  earliest_offset={earliest_offset} latest_offset={latest_offset}")

    print("--- loading klines unified ---")
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    print(f"  BTC: {len(btc)} epochs")
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    print(f"  ETH: {len(eth)} epochs")
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  SOL: {len(sol)} epochs")

    windows = build_windows()
    print(f"--- walk-forward design: {len(windows)} windows "
          f"(train={TRAIN_ROUNDS} test={TEST_ROUNDS} step={STEP_ROUNDS}) ---")
    for i, w in enumerate(windows[:5]):
        print(f"  wf{i:02d}: train {w['train_start']}..{w['train_end']}  "
              f"test {w['test_start']}..{w['test_end']}")
    if len(windows) > 5:
        print(f"  ... and {len(windows) - 5} more")

    print(f"--- lookback variants: {len(LOOKBACK_GRID)} (cutoff=2 frozen) ---")
    print(f"--- total folds to run: {len(windows) * len(LOOKBACK_GRID) * 2} ---")
    print()

    # Temporary output dir (we throw away per-fold files; only need summary)
    out_root = Path(tempfile.mkdtemp(prefix="wf_step3_"))
    print(f"--- temp output: {out_root} ---")

    results: list[dict[str, Any]] = []
    t_fold_start = time.time()
    fold_count = 0

    for i, w in enumerate(windows):
        print(f"\n=== window {i:02d}: train [{w['train_start']}..{w['train_end']}]  "
              f"test [{w['test_start']}..{w['test_end']}] "
              f"(test cohorts: {cohort_of(w['test_start'])}..{cohort_of(w['test_end'])}) ===")

        # Phase 1: train sweep
        train_results: list[tuple[tuple[int, int, int], float, int, int]] = []
        for lb in LOOKBACK_GRID:
            spec = make_spec(
                name=f"wf{i:02d}_train_a{lb[0]}_b{lb[1]}_c{lb[2]}",
                lookbacks=lb,
                epoch_start=w["train_start"], epoch_end=w["train_end"],
            )
            try:
                summary = run_one_fold(spec, all_rounds, btc, eth, sol, earliest_offset, out_root)
                pnl = float(summary.get("net_pnl_bnb", 0.0))
                bets = int(summary.get("num_bets", 0))
                wins = int(summary.get("num_wins", 0))
                train_results.append((lb, pnl, bets, wins))
            except Exception as e:
                print(f"  train fold {lb} FAILED: {type(e).__name__}: {e}")
                train_results.append((lb, float("-inf"), 0, 0))
            fold_count += 1

        # Pick best (max PnL) — break ties by more bets (more samples = more confidence)
        train_results.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_lb, best_train_pnl, best_train_bets, best_train_wins = train_results[0]
        print(f"  train winner: {best_lb}  PnL={best_train_pnl:+.4f} BNB  "
              f"bets={best_train_bets} wins={best_train_wins}")

        # Phase 2: test the winner + canonical on the test window
        test_winner_spec = make_spec(
            name=f"wf{i:02d}_test_winner",
            lookbacks=best_lb,
            epoch_start=w["test_start"], epoch_end=w["test_end"],
        )
        test_canonical_spec = make_spec(
            name=f"wf{i:02d}_test_canonical",
            lookbacks=CANONICAL,
            epoch_start=w["test_start"], epoch_end=w["test_end"],
        )
        win_summary = run_one_fold(test_winner_spec, all_rounds, btc, eth, sol, earliest_offset, out_root)
        can_summary = run_one_fold(test_canonical_spec, all_rounds, btc, eth, sol, earliest_offset, out_root)
        fold_count += 2

        win_pnl = float(win_summary.get("net_pnl_bnb", 0.0))
        win_bets = int(win_summary.get("num_bets", 0))
        win_wins = int(win_summary.get("num_wins", 0))
        can_pnl = float(can_summary.get("net_pnl_bnb", 0.0))
        can_bets = int(can_summary.get("num_bets", 0))
        can_wins = int(can_summary.get("num_wins", 0))
        print(f"  test winner ({best_lb}):  PnL={win_pnl:+.4f} bets={win_bets} wins={win_wins}")
        print(f"  test canonical ({CANONICAL}):  PnL={can_pnl:+.4f} bets={can_bets} wins={can_wins}")

        results.append({
            "window_idx": i,
            "train_start": w["train_start"], "train_end": w["train_end"],
            "test_start": w["test_start"], "test_end": w["test_end"],
            "test_cohort_start": cohort_of(w["test_start"]),
            "test_cohort_end": cohort_of(w["test_end"]),
            "best_lb": list(best_lb),
            "train_pnl": best_train_pnl,
            "train_bets": best_train_bets, "train_wins": best_train_wins,
            "test_winner_pnl": win_pnl,
            "test_winner_bets": win_bets, "test_winner_wins": win_wins,
            "test_canonical_pnl": can_pnl,
            "test_canonical_bets": can_bets, "test_canonical_wins": can_wins,
            "winner_minus_canonical_pnl": win_pnl - can_pnl,
            "winner_is_canonical": (best_lb == CANONICAL),
        })

        elapsed = time.time() - t_fold_start
        rate = fold_count / max(1e-6, elapsed)
        print(f"  cumulative {fold_count} folds in {elapsed:.0f}s ({rate:.1f} folds/s)")

    # Aggregate
    print("\n--- aggregate ---")
    total_winner_pnl = sum(r["test_winner_pnl"] for r in results)
    total_canonical_pnl = sum(r["test_canonical_pnl"] for r in results)
    delta = total_winner_pnl - total_canonical_pnl
    n_pos_winner = sum(1 for r in results if r["test_winner_pnl"] > 0)
    n_pos_canonical = sum(1 for r in results if r["test_canonical_pnl"] > 0)
    n_winner_is_canonical = sum(1 for r in results if r["winner_is_canonical"])
    print(f"  total walk-forward PnL: winner={total_winner_pnl:+.4f} canonical={total_canonical_pnl:+.4f} delta={delta:+.4f}")
    print(f"  positive test windows: winner={n_pos_winner}/{len(results)} canonical={n_pos_canonical}/{len(results)}")
    print(f"  windows where winner == canonical (3,7,15): {n_winner_is_canonical}/{len(results)}")

    # Parameter selection histogram
    from collections import Counter
    winner_counts = Counter(tuple(r["best_lb"]) for r in results)
    print(f"  winner lookback histogram:")
    for lb, ct in winner_counts.most_common():
        print(f"    {lb}: {ct}")

    out_path = REPO / "var" / "strategy_review" / "walk_forward_step3_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "train_rounds": TRAIN_ROUNDS, "test_rounds": TEST_ROUNDS,
                "step_rounds": STEP_ROUNDS,
                "lookback_grid": [list(lb) for lb in LOOKBACK_GRID],
                "canonical": list(CANONICAL),
            },
            "windows": results,
            "aggregate": {
                "n_windows": len(results),
                "total_winner_pnl": total_winner_pnl,
                "total_canonical_pnl": total_canonical_pnl,
                "delta_winner_minus_canonical": delta,
                "n_pos_winner": n_pos_winner,
                "n_pos_canonical": n_pos_canonical,
                "n_winner_is_canonical": n_winner_is_canonical,
                "winner_lookback_histogram": {str(lb): ct for lb, ct in winner_counts.items()},
                "total_folds": fold_count,
                "elapsed_seconds": time.time() - t0,
            },
        }, f, indent=2)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t0:.0f}s  ({fold_count} folds)")


if __name__ == "__main__":
    main()
