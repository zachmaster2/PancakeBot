"""p1c Step 3 — placebo permutation test (post-Step-2 reviewer follow-on).

Pre-registered design (per implementer entry at 2026-05-01T01:51:22-04:00):

Null: BNB confirmation provides no information about A's per-bet outcome.
Under this null, the partition of A's bets into confirming/rejecting is
exchangeable — any 1061/385 split of CV5 and any 168/395 split of extension
has expected paired-diff = 0 and a sampling distribution determined by
the underlying per-bet PnLs.

Procedure (1,000 seeds):
  1. Permute CV5's 1,446 A-bet PnLs. First 1,061 -> placebo_confirmed,
     remaining 385 -> placebo_rejected.
  2. Permute extension's 563 A-bet PnLs. First 168 -> placebo_confirmed,
     remaining 395 -> placebo_rejected.
  3. Pool: confirmed = CV5_c + ext_c (1,229 bets), rejected = CV5_r + ext_r (780).
  4. placebo_diff_s = mean(confirmed) − mean(rejected).

Pre-registered PASS: actual +0.03353 ≥ 99th percentile of placebo distribution.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")

# Step 2 actual numbers (locked from p1c_paired_test_results.json)
ACTUAL_POOLED_DIFF = 0.033530553861031806
ACTUAL_CV5_C, ACTUAL_CV5_R = 1061, 385
ACTUAL_EXT_C, ACTUAL_EXT_R = 168, 395
N_SEEDS = 1000
PASS_PERCENTILE = 99.0

OUT = REPO / "var" / "extended" / "p1c_placebo_results.json"


def load_a_bets_cv5() -> np.ndarray:
    """Load per-bet PnLs from canonical CV5 fold trades.csv files (BET only)."""
    base = REPO / "var" / "sweep" / "_post_sync_baseline"
    pnls: list[float] = []
    for fold in ["f1", "f2", "f3", "f4", "f5"]:
        with open(base / f"fold_{fold}" / "trades.csv") as f:
            r = csv.DictReader(f)
            for row in r:
                if row["action"] == "BET":
                    pnls.append(float(row["profit_bnb"]))
    return np.array(pnls)


def load_a_bets_extension() -> np.ndarray:
    """Load per-bet PnLs from Step 2's canonical-extension trades.csv (BET only)."""
    p = Path(r"C:\Users\zking\AppData\Local\Temp\p1c_step2_runs\extension_A\p1c_step2_A\extension\trades.csv")
    pnls: list[float] = []
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["action"] == "BET":
                pnls.append(float(row["profit_bnb"]))
    return np.array(pnls)


def main():
    print("=" * 100, flush=True)
    print("p1c Step 3 — placebo permutation test", flush=True)
    print(f"  pre-registered PASS gate: actual >= {PASS_PERCENTILE}th percentile of placebo", flush=True)
    print(f"  actual paired diff (Step 2): +{ACTUAL_POOLED_DIFF:.5f}", flush=True)
    print(f"  seeds: {N_SEEDS}", flush=True)
    print("=" * 100, flush=True)
    t_start = time.time()

    cv5_pnls = load_a_bets_cv5()
    ext_pnls = load_a_bets_extension()
    print(f"\n  CV5 A-bets loaded: n={len(cv5_pnls)} (expected 1446)", flush=True)
    print(f"  extension A-bets loaded: n={len(ext_pnls)} (expected 563)", flush=True)
    assert len(cv5_pnls) == 1446, f"CV5 bet count mismatch: {len(cv5_pnls)}"
    assert len(ext_pnls) == 563, f"extension bet count mismatch: {len(ext_pnls)}"

    print(f"\n  CV5 mean per-bet: {cv5_pnls.mean():+.5f}, std: {cv5_pnls.std(ddof=1):.5f}", flush=True)
    print(f"  extension mean per-bet: {ext_pnls.mean():+.5f}, std: {ext_pnls.std(ddof=1):.5f}", flush=True)

    placebo_diffs = np.zeros(N_SEEDS)

    for s in range(N_SEEDS):
        rng = np.random.default_rng(s + 1)  # seed in 1..1000
        cv5_perm = rng.permutation(cv5_pnls)
        ext_perm = rng.permutation(ext_pnls)
        cv5_c = cv5_perm[:ACTUAL_CV5_C]
        cv5_r = cv5_perm[ACTUAL_CV5_C:ACTUAL_CV5_C + ACTUAL_CV5_R]
        ext_c = ext_perm[:ACTUAL_EXT_C]
        ext_r = ext_perm[ACTUAL_EXT_C:ACTUAL_EXT_C + ACTUAL_EXT_R]
        # Pool
        confirmed = np.concatenate([cv5_c, ext_c])
        rejected = np.concatenate([cv5_r, ext_r])
        placebo_diffs[s] = float(confirmed.mean() - rejected.mean())

    # Distribution stats
    mean_p = float(placebo_diffs.mean())
    median_p = float(np.median(placebo_diffs))
    std_p = float(placebo_diffs.std(ddof=1))
    p1 = float(np.percentile(placebo_diffs, 1))
    p5 = float(np.percentile(placebo_diffs, 5))
    p50 = float(np.percentile(placebo_diffs, 50))
    p95 = float(np.percentile(placebo_diffs, 95))
    p99 = float(np.percentile(placebo_diffs, 99))
    p99_5 = float(np.percentile(placebo_diffs, 99.5))
    max_p = float(placebo_diffs.max())
    min_p = float(placebo_diffs.min())

    # One-sided empirical p-value: P(placebo >= actual)
    n_ge_actual = int(np.sum(placebo_diffs >= ACTUAL_POOLED_DIFF))
    empirical_p = n_ge_actual / N_SEEDS

    # Percentile rank of actual
    actual_rank = float(np.sum(placebo_diffs <= ACTUAL_POOLED_DIFF) / N_SEEDS * 100)

    # Pre-registered pass check
    placebo_pass = (ACTUAL_POOLED_DIFF >= p99)

    print(f"\nPlacebo distribution (n={N_SEEDS}):", flush=True)
    print(f"  mean:    {mean_p:+.5f}", flush=True)
    print(f"  median:  {median_p:+.5f}", flush=True)
    print(f"  std:     {std_p:.5f}", flush=True)
    print(f"  min:     {min_p:+.5f}", flush=True)
    print(f"  1st %ile: {p1:+.5f}", flush=True)
    print(f"  5th %ile: {p5:+.5f}", flush=True)
    print(f"  50th:    {p50:+.5f}", flush=True)
    print(f"  95th:    {p95:+.5f}", flush=True)
    print(f"  99th %ile: {p99:+.5f}  <-- PRE-REGISTERED PASS GATE", flush=True)
    print(f"  99.5th:  {p99_5:+.5f}", flush=True)
    print(f"  max:     {max_p:+.5f}", flush=True)
    print(f"\nActual: +{ACTUAL_POOLED_DIFF:.5f}", flush=True)
    print(f"  Percentile rank of actual: {actual_rank:.2f}", flush=True)
    print(f"  Placebo seeds >= actual: {n_ge_actual}/{N_SEEDS} (one-sided empirical p={empirical_p:.4f})",
          flush=True)
    print(f"\n  Pre-registered PASS: actual >= 99th percentile placebo", flush=True)
    print(f"  Result: {'PASS' if placebo_pass else 'FAIL (does NOT clear 99th percentile)'}", flush=True)

    out = {
        "spec": {
            "n_seeds": N_SEEDS,
            "pass_percentile": PASS_PERCENTILE,
            "actual_pooled_diff": ACTUAL_POOLED_DIFF,
            "cv5_split": {"confirmed": ACTUAL_CV5_C, "rejected": ACTUAL_CV5_R},
            "extension_split": {"confirmed": ACTUAL_EXT_C, "rejected": ACTUAL_EXT_R},
            "cv5_n_bets": int(len(cv5_pnls)),
            "extension_n_bets": int(len(ext_pnls)),
        },
        "placebo_distribution": {
            "n": N_SEEDS,
            "mean": mean_p, "median": median_p, "std": std_p,
            "min": min_p, "max": max_p,
            "p1": p1, "p5": p5, "p50": p50, "p95": p95, "p99": p99, "p99_5": p99_5,
        },
        "comparison": {
            "actual_pooled_diff": ACTUAL_POOLED_DIFF,
            "placebo_99th_percentile": p99,
            "actual_clears_99th": placebo_pass,
            "n_placebos_geq_actual": n_ge_actual,
            "one_sided_empirical_p": empirical_p,
            "actual_percentile_rank": actual_rank,
        },
        "verdict": "PASS" if placebo_pass else "FAIL",
        "elapsed_seconds": time.time() - t_start,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults JSON: {OUT}", flush=True)
    print(f"Total elapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
