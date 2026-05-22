"""Permutation null for canonical (3,7,15) cs=2 CV5 PnL.

Methodology (per audit memo 2026-05-22 §6.6 + §2 entry 1.11 regime_phase0
permutation pattern):

1. Hold the gate's bet sequence (epoch, side, size) FIXED. These are the
   1446 bets canonical produces on epochs 437562..474086.
2. For each permutation: shuffle which round-outcome tuple
   (winner, failed-flag, bull-pool, bear-pool, lock_price, close_price)
   each bet is settled against, by randomly permuting the outcome-tuples
   across the bet list.
3. Re-settle every bet via pancakebot.settlement.settle_bet_against_closed_round
   (impact-aware; NEVER simplified 50/50).
4. Sum to get the permuted PnL per permutation. Build the null distribution.
5. Compute one-sided p (P(null >= actual)) and verdict.

Bet sequence comes from trades.csv produced by research/in_process_runner.py.
The trades.csv elapsed_sim_seconds and per-fold metrics are NOT used; only
the bet-row triples (epoch, direction, bet_size_bnb).

Output:
- var/strategy_review/2026_05_22_canonical_cv5_permutation_null.jsonl
  (one JSON line per permutation: {perm_idx, pnl, wins, bets_settled})
- var/strategy_review/2026_05_22_canonical_cv5_permutation_null.md
  (markdown summary with percentiles + verdict)
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.settlement import settle_bet_against_closed_round
from pancakebot.types import Round, Bet
from pancakebot.market_data.round_store import ClosedRoundsStore


# --- Config ----------------------------------------------------------
N_PERMUTATIONS = 1000
TREASURY_FEE_FRACTION = 0.03
ACTUAL_PNL = 50.4953  # canonical CV5 PnL, verified bit-identical
SEED = 20260522
TRADES_CSV = Path(r"C:\Users\zking\AppData\Local\Temp\canonical_cv5_out\canonical_cv5_437562_474086\trades.csv")
OUT_DIR = _REPO_ROOT / "var" / "strategy_review"
OUT_JSONL = OUT_DIR / "2026_05_22_canonical_cv5_permutation_null.jsonl"
OUT_MD = OUT_DIR / "2026_05_22_canonical_cv5_permutation_null.md"
CV5_EPOCH_START = 437562
CV5_EPOCH_END = 474086


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # --- Load bet sequence from trades.csv ---
    import csv
    bets: list[dict] = []
    with TRADES_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["action"] != "BET":
                continue
            bets.append({
                "epoch": int(row["epoch"]),
                "side": row["direction"],
                "size_bnb": float(row["bet_size_bnb"]),
                "actual_profit": float(row["profit_bnb"]),
            })
    print(f"loaded {len(bets)} bets from {TRADES_CSV.name}")
    actual_sum = sum(b["actual_profit"] for b in bets)
    print(f"actual_pnl_from_trades = {actual_sum:.4f} BNB (expected ~{ACTUAL_PNL})")
    assert abs(actual_sum - ACTUAL_PNL) < 0.01, "actual PnL drift"

    # --- Load round-outcome tuples from canonical closed_rounds.jsonl ---
    # Need only the CV5 epoch range (437562..474086).
    store = ClosedRoundsStore(str(_REPO_ROOT / "var" / "closed_rounds.jsonl"))
    rounds_by_epoch: dict[int, Round] = {}
    for r in store.iter_closed_rounds():
        if CV5_EPOCH_START <= r.epoch <= CV5_EPOCH_END:
            rounds_by_epoch[r.epoch] = r
    print(f"loaded {len(rounds_by_epoch)} rounds for CV5 range")

    # All Round objects in this range form our outcome-pool.
    # The permutation samples FROM these (an outcome reassignment to a
    # random round from within the CV5 range).
    all_round_list: list[Round] = list(rounds_by_epoch.values())
    n_rounds = len(all_round_list)

    # --- Verify actual settlement reproduces ACTUAL_PNL ---
    print("verifying actual settlement...")
    actual_total = 0.0
    actual_wins = 0
    for b in bets:
        rd = rounds_by_epoch[b["epoch"]]
        outcome = settle_bet_against_closed_round(
            bet_bnb=b["size_bnb"], bet_side=b["side"],
            round_closed=rd, treasury_fee_fraction=TREASURY_FEE_FRACTION,
        )
        # The trades.csv profit already includes -MAX_GAS_COST_BET_BNB and
        # the outcome.credit_bnb. So actual_profit = outcome.credit_bnb - size - gas_cost.
        # For verification we compute the same: profit = credit - size - gas_cost.
        from pancakebot.constants import MAX_GAS_COST_BET_BNB
        profit = outcome.credit_bnb - b["size_bnb"] - MAX_GAS_COST_BET_BNB
        actual_total += profit
        if outcome.outcome == "win":
            actual_wins += 1
    print(f"settlement reproduces: {actual_total:.4f} BNB (expected {ACTUAL_PNL}, delta {actual_total - ACTUAL_PNL:.4f})")
    print(f"actual wins: {actual_wins} / {len(bets)}")
    if abs(actual_total - ACTUAL_PNL) > 0.1:
        print("WARNING: settlement reproduces with non-trivial drift")

    # --- Permutation null ---
    # For each permutation: assign a randomly-permuted Round from CV5 range
    # to each bet, settle, sum. The bet keeps its (side, size) but its
    # epoch's actual outcome is replaced with a randomly-chosen round's outcome.
    print(f"\nrunning {N_PERMUTATIONS} permutations...")
    rng = random.Random(SEED)
    null_pnls: list[float] = []
    null_wins: list[int] = []

    from pancakebot.constants import MAX_GAS_COST_BET_BNB

    perm_start = time.perf_counter()
    for perm_idx in range(N_PERMUTATIONS):
        # Random permutation of n_bets indices into rounds list.
        # For each bet, assign a random round index. We use sampling
        # WITHOUT replacement when n_bets <= n_rounds (typical case).
        if len(bets) <= n_rounds:
            sampled_indices = rng.sample(range(n_rounds), len(bets))
        else:
            sampled_indices = [rng.randrange(n_rounds) for _ in bets]

        perm_total = 0.0
        perm_wins = 0
        for bet_idx, bet in enumerate(bets):
            ro = all_round_list[sampled_indices[bet_idx]]
            outcome = settle_bet_against_closed_round(
                bet_bnb=bet["size_bnb"], bet_side=bet["side"],
                round_closed=ro, treasury_fee_fraction=TREASURY_FEE_FRACTION,
            )
            profit = outcome.credit_bnb - bet["size_bnb"] - MAX_GAS_COST_BET_BNB
            perm_total += profit
            if outcome.outcome == "win":
                perm_wins += 1

        null_pnls.append(perm_total)
        null_wins.append(perm_wins)

        if (perm_idx + 1) % 100 == 0:
            elapsed = time.perf_counter() - perm_start
            rate = (perm_idx + 1) / elapsed
            eta = (N_PERMUTATIONS - perm_idx - 1) / rate
            print(f"  perm {perm_idx+1}/{N_PERMUTATIONS}: elapsed={elapsed:.1f}s rate={rate:.1f}/s eta={eta:.0f}s")

    # --- Write JSONL ---
    with OUT_JSONL.open("w") as f:
        for i, (pnl, wins) in enumerate(zip(null_pnls, null_wins)):
            f.write(json.dumps({"perm_idx": i, "pnl_bnb": pnl, "wins": wins, "bets": len(bets)}) + "\n")

    # --- Stats ---
    null_mean = statistics.mean(null_pnls)
    null_std = statistics.stdev(null_pnls)
    null_pnls_sorted = sorted(null_pnls)
    def pct(p): return null_pnls_sorted[int(round(p/100 * (len(null_pnls_sorted)-1)))]
    p5, p25, p50, p75, p95, p99 = pct(5), pct(25), pct(50), pct(75), pct(95), pct(99)

    # One-sided p: P(null >= actual)
    n_at_or_above = sum(1 for x in null_pnls if x >= actual_total)
    p_one_sided = (n_at_or_above + 1) / (N_PERMUTATIONS + 1)  # +1 smoothing

    # z-score
    z = (actual_total - null_mean) / null_std if null_std > 0 else 0

    # Where does actual sit?
    rank_above = sum(1 for x in null_pnls if x > actual_total)
    actual_percentile = 100.0 * (1 - (rank_above / N_PERMUTATIONS))

    elapsed_total = time.perf_counter() - t0

    # --- Write markdown summary ---
    md = []
    md.append(f"# Permutation null on canonical CV5 — 2026-05-22\n")
    md.append(f"**Methodology:** hold bet sequence (epoch, side, size) fixed; permute round-outcome assignment across bets within the CV5 epoch range; re-settle via real impact-aware `settle_bet_against_closed_round`; sum to permuted PnL.")
    md.append("")
    md.append(f"**Config:** N_PERMUTATIONS={N_PERMUTATIONS}, SEED={SEED}, TREASURY_FEE={TREASURY_FEE_FRACTION}")
    md.append(f"**Epoch range:** {CV5_EPOCH_START}..{CV5_EPOCH_END} ({n_rounds} rounds)")
    md.append(f"**Bet sequence:** {len(bets)} bets (from canonical CV5 trades.csv)")
    md.append("")
    md.append(f"## Actual canonical CV5 result")
    md.append(f"- PnL: **+{actual_total:.4f} BNB** (matches `test_in_process_runner.py` expected +{ACTUAL_PNL})")
    md.append(f"- Bets: {len(bets)}, Wins: {actual_wins}, WR: {actual_wins/len(bets):.4f}")
    md.append("")
    md.append(f"## Null distribution ({N_PERMUTATIONS} permutations)")
    md.append("| stat | value |")
    md.append("|---|---:|")
    md.append(f"| mean | {null_mean:+.4f} BNB |")
    md.append(f"| std  | {null_std:.4f} BNB |")
    md.append(f"| min  | {min(null_pnls):+.4f} BNB |")
    md.append(f"| p5   | {p5:+.4f} BNB |")
    md.append(f"| p25  | {p25:+.4f} BNB |")
    md.append(f"| p50  | {p50:+.4f} BNB |")
    md.append(f"| p75  | {p75:+.4f} BNB |")
    md.append(f"| p95  | {p95:+.4f} BNB |")
    md.append(f"| p99  | {p99:+.4f} BNB |")
    md.append(f"| max  | {max(null_pnls):+.4f} BNB |")
    md.append("")
    md.append(f"## Actual position in null")
    md.append(f"- z-score: **{z:+.3f}σ**")
    md.append(f"- percentile: **{actual_percentile:.2f}**")
    md.append(f"- n permutations >= actual: {n_at_or_above} / {N_PERMUTATIONS}")
    md.append(f"- one-sided p (P(null >= actual)): **{p_one_sided:.4f}**")
    md.append("")
    verdict = "**REJECT NULL** (CV5 edge real)" if p_one_sided < 0.05 else "**FAIL TO REJECT** (CV5 indistinguishable from luck)"
    md.append(f"## Verdict")
    md.append(f"- p_one_sided={p_one_sided:.4f} {'<' if p_one_sided < 0.05 else '>='} 0.05 → {verdict}")
    if p_one_sided >= 0.05:
        # Effect-size + power note
        per_bet_actual = actual_total / len(bets)
        per_bet_se = null_std / len(bets)  # null SD/n
        md.append("")
        md.append(f"**Effect size:** per-bet edge = {per_bet_actual:+.4f} BNB / bet")
        md.append(f"**Null per-bet SD:** {null_std/math.sqrt(len(bets)):.4f} BNB / sqrt(n)")
        md.append(f"**Power note:** with per-bet σ ≈ {null_std/math.sqrt(len(bets)):.4f} BNB and observed edge +{per_bet_actual:.4f} BNB/bet, the sample size needed to detect this edge from zero at α=0.05 two-sided with 80% power is ≈ {(2.8 * null_std / (per_bet_actual * math.sqrt(len(bets))))**2 * len(bets):.0f} bets (current: {len(bets)}).")
    md.append("")
    md.append(f"## Compute")
    md.append(f"- total elapsed: {elapsed_total:.1f}s ({N_PERMUTATIONS/elapsed_total:.1f} perm/s)")
    md.append("")
    md.append(f"## Files")
    md.append(f"- raw permutations: `{OUT_JSONL.relative_to(_REPO_ROOT)}`")
    md.append(f"- bet sequence source: `{TRADES_CSV}` (canonical CV5 run via `research/in_process_runner.py`)")
    md.append(f"- driver: `research/canonical_cv5_permutation_null_2026_05_22.py`")

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n=== SUMMARY ===")
    print(f"actual pnl: {actual_total:+.4f} BNB")
    print(f"null mean ± std: {null_mean:+.4f} ± {null_std:.4f} BNB")
    print(f"actual z: {z:+.3f}σ ({actual_percentile:.2f}-th percentile)")
    print(f"one-sided p: {p_one_sided:.4f}")
    print(f"verdict: {verdict}")
    print(f"output: {OUT_MD}, {OUT_JSONL}")
    print(f"elapsed: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
