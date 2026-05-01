"""p1c Step 2 — paired-difference test on A's existing bets, BNB as classifier.

Per orchestrator v2.1 (after reviewer R8 arithmetic correction):

  For each FROZEN HOLDOUT slice and CV5:
    1. Run A = canonical BTC-primary at the slice's pre-registered initial_bankroll.
    2. Run B = BNB-as-primary (monkey-patched _BTC_KLINES_PATH) at same bankroll.
    3. For every round where A bet (BULL or BEAR), look up B's decision.
       - CONFIRMING: B bet AND same direction.
       - REJECTING: B SKIP, OR B bet opposite direction.
    4. Compute mean per-bet PnL on confirming subset and rejecting subset.
    5. Compute paired-difference (confirming − rejecting), SE, 95% CI.

  Pooled gating test: pool CV5 + extension's per-bet PnLs into a single
  paired-difference computation. Apply v2.1 success criteria:
    - PASS: pooled diff >= +0.030 BNB/bet AND CI lower bound >= 0
            AND same-sign on CV5 alone AND extension alone (R9)
    - INCONCLUSIVE: pooled CI spans 0 OR signs disagree
    - HARD FAIL: pooled diff <= -0.020 OR extension diff < -0.030

  v3 + post-v1: descriptive only (n too small for gated PASS).

Per-step bankrolls (R7):
  CV5 at 50 BNB (matches canonical hash test)
  extension/v3/post-v1 at 100 BNB (matches p1a/p2a holdout protocol)
"""
from __future__ import annotations

import csv as csvmod
import json
import math
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")
sys.path.insert(0, str(REPO))

# Slices: (name, epoch_start, epoch_end, use_extended_data, initial_bankroll)
CV5_FOLDS = [
    ("f1", 437562, 444866, False, 50.0),
    ("f2", 444867, 452171, False, 50.0),
    ("f3", 452172, 459476, False, 50.0),
    ("f4", 459477, 466781, False, 50.0),
    ("f5", 466782, 474086, False, 50.0),
]
HOLDOUT_SLICES = [
    ("extension", 422298, 437561, True,  100.0),
    ("v3",        474880, 477254, False, 100.0),
    ("post_v1",   475312, 477254, False, 100.0),
]

# Pre-registered v2.1 thresholds
PASS_DIFF_MIN = 0.030
HARD_FAIL_POOLED_MIN = -0.020
HARD_FAIL_EXT_MIN = -0.030

OUT = REPO / "var" / "extended" / "p1c_paired_test_results.json"
TMP_BASE = Path(r"C:\Users\zking\AppData\Local\Temp\p1c_step2_runs")
TMP_BASE.mkdir(parents=True, exist_ok=True)


def run_canonical(slice_name: str, ep_start: int, ep_end: int,
                  use_extended: bool, initial_bankroll: float, label: str):
    """Run A = canonical BTC-primary on the slice. Returns out_dir."""
    # Re-import (or reload) ipr with original BTC paths.
    import importlib
    import research.in_process_runner as ipr
    importlib.reload(ipr)
    out_dir = TMP_BASE / f"{slice_name}_A"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = ipr.FoldSpec(
        name=f"{label}/{slice_name}",
        cutoff_seconds=2,
        epoch_start=ep_start,
        epoch_end=ep_end,
        strategy_overrides={},
    )
    ipr.run_experiment(
        experiment_specs=[spec], output_base_dir=out_dir,
        initial_bankroll_bnb=initial_bankroll, use_extended_data=use_extended,
    )
    return out_dir / label / slice_name


def run_bnb_as_primary(slice_name: str, ep_start: int, ep_end: int,
                        use_extended: bool, initial_bankroll: float, label: str):
    """Run B = BNB-as-primary by monkey-patching the BTC kline path."""
    import importlib
    import research.in_process_runner as ipr
    importlib.reload(ipr)
    ipr._BTC_KLINES_PATH = REPO / "var" / "bnb_spot_prices.jsonl"
    ipr._EXT_BTC_KLINES_PATH = REPO / "var" / "extended" / "bnb_spot_prices.jsonl"
    out_dir = TMP_BASE / f"{slice_name}_B"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = ipr.FoldSpec(
        name=f"{label}/{slice_name}",
        cutoff_seconds=2,
        epoch_start=ep_start,
        epoch_end=ep_end,
        strategy_overrides={},
    )
    ipr.run_experiment(
        experiment_specs=[spec], output_base_dir=out_dir,
        initial_bankroll_bnb=initial_bankroll, use_extended_data=use_extended,
    )
    return out_dir / label / slice_name


def load_decisions(trades_csv: Path) -> dict[int, tuple[str, str, float]]:
    """Return {epoch: (action, direction, profit_bnb)} from a trades.csv."""
    out: dict[int, tuple[str, str, float]] = {}
    with open(trades_csv, "r", encoding="utf-8") as f:
        r = csvmod.DictReader(f)
        for row in r:
            ep = int(row["epoch"])
            out[ep] = (
                row["action"],
                row["direction"].strip(),
                float(row["profit_bnb"]),
            )
    return out


def classify_and_compute(a_decisions: dict, b_decisions: dict, slice_name: str):
    """For each round where A bet, classify by B's decision and split per-bet PnL."""
    confirming: list[float] = []
    rejecting: list[float] = []
    n_disagree_direction = 0
    n_b_skip = 0
    for ep, (a_act, a_dir, a_pnl) in a_decisions.items():
        if a_act != "BET":
            continue
        b = b_decisions.get(ep)
        if b is None:
            continue
        b_act, b_dir, _b_pnl = b
        if b_act == "BET" and b_dir == a_dir:
            confirming.append(a_pnl)
        else:
            rejecting.append(a_pnl)
            if b_act == "BET":
                n_disagree_direction += 1
            else:
                n_b_skip += 1
    return confirming, rejecting, n_disagree_direction, n_b_skip


def stats_summary(confirming: list[float], rejecting: list[float]) -> dict:
    n_c = len(confirming)
    n_r = len(rejecting)
    mean_c = sum(confirming) / n_c if n_c else 0.0
    mean_r = sum(rejecting) / n_r if n_r else 0.0
    diff = mean_c - mean_r
    if n_c < 2 or n_r < 2:
        return {
            "n_confirming": n_c, "n_rejecting": n_r,
            "mean_confirming": mean_c, "mean_rejecting": mean_r,
            "diff": diff,
            "se_diff": None, "ci95_half": None, "ci95_lower": None, "ci95_upper": None,
            "underpowered": True,
        }
    var_c = sum((x - mean_c) ** 2 for x in confirming) / (n_c - 1)
    var_r = sum((x - mean_r) ** 2 for x in rejecting) / (n_r - 1)
    se_c = math.sqrt(var_c / n_c)
    se_r = math.sqrt(var_r / n_r)
    se_diff = math.sqrt(se_c * se_c + se_r * se_r)
    ci95_half = 1.96 * se_diff
    return {
        "n_confirming": n_c, "n_rejecting": n_r,
        "mean_confirming": mean_c, "mean_rejecting": mean_r,
        "diff": diff,
        "se_diff": se_diff, "ci95_half": ci95_half,
        "ci95_lower": diff - ci95_half, "ci95_upper": diff + ci95_half,
        "underpowered": (n_c < 30 or n_r < 30),
    }


def main():
    print("=" * 100, flush=True)
    print("p1c Step 2 — paired-difference test (A's bets, BNB as classifier)", flush=True)
    print("=" * 100, flush=True)
    t_start = time.time()

    per_slice: dict[str, dict] = {}
    cv5_pooled_confirming: list[float] = []
    cv5_pooled_rejecting: list[float] = []

    # ---- CV5: reuse existing canonical trades.csv + Step 0's BNB trades.csv ----
    print("\n[CV5] Reusing existing trades.csv from var/sweep/_post_sync_baseline + p1c_step0_runs", flush=True)
    a_dir_cv5 = REPO / "var" / "sweep" / "_post_sync_baseline"
    b_dir_cv5 = Path(r"C:\Users\zking\AppData\Local\Temp\p1c_step0_runs\bnb_primary\p1c_bnb_primary")
    for fold_name, _, _, _, _ in CV5_FOLDS:
        a = load_decisions(a_dir_cv5 / f"fold_{fold_name}" / "trades.csv")
        b = load_decisions(b_dir_cv5 / fold_name / "trades.csv")
        c, r, n_dis, n_skip = classify_and_compute(a, b, fold_name)
        s = stats_summary(c, r)
        s["n_disagree_direction"] = n_dis
        s["n_b_skip"] = n_skip
        per_slice[fold_name] = s
        cv5_pooled_confirming.extend(c)
        cv5_pooled_rejecting.extend(r)
        print(f"  {fold_name}: n_c={s['n_confirming']:4d} n_r={s['n_rejecting']:4d}  "
              f"mean_c={s['mean_confirming']:+.5f} mean_r={s['mean_rejecting']:+.5f}  "
              f"diff={s['diff']:+.5f}",
              flush=True)

    cv5_summary = stats_summary(cv5_pooled_confirming, cv5_pooled_rejecting)
    print(f"  CV5 pooled: n_c={cv5_summary['n_confirming']} n_r={cv5_summary['n_rejecting']}  "
          f"diff={cv5_summary['diff']:+.5f}  CI=[{cv5_summary.get('ci95_lower', 0):+.5f}, {cv5_summary.get('ci95_upper', 0):+.5f}]",
          flush=True)
    per_slice["cv5_pooled"] = cv5_summary

    # ---- HOLDOUT slices: run both arms fresh ----
    holdout_results: dict[str, dict] = {}
    for slice_name, ep_start, ep_end, use_ext, init_br in HOLDOUT_SLICES:
        print(f"\n[{slice_name}] Running A (canonical BTC-primary) at {init_br} BNB...", flush=True)
        t0 = time.time()
        a_out = run_canonical(slice_name, ep_start, ep_end, use_ext, init_br,
                                "p1c_step2_A")
        print(f"  A done in {time.time()-t0:.1f}s", flush=True)
        print(f"[{slice_name}] Running B (BNB-as-primary) at {init_br} BNB...", flush=True)
        t0 = time.time()
        b_out = run_bnb_as_primary(slice_name, ep_start, ep_end, use_ext, init_br,
                                     "p1c_step2_B")
        print(f"  B done in {time.time()-t0:.1f}s", flush=True)

        a_dec = load_decisions(a_out / "trades.csv")
        b_dec = load_decisions(b_out / "trades.csv")
        c, r, n_dis, n_skip = classify_and_compute(a_dec, b_dec, slice_name)
        s = stats_summary(c, r)
        s["n_disagree_direction"] = n_dis
        s["n_b_skip"] = n_skip
        holdout_results[slice_name] = s
        per_slice[slice_name] = s
        print(f"  {slice_name}: n_c={s['n_confirming']} n_r={s['n_rejecting']}  "
              f"diff={s['diff']:+.5f} "
              f"CI=[{s.get('ci95_lower') if s.get('ci95_lower') is not None else float('nan'):+.5f}, "
              f"{s.get('ci95_upper') if s.get('ci95_upper') is not None else float('nan'):+.5f}]",
              flush=True)

    # ---- Pooled CV5 + extension (load-bearing) ----
    extension_confirming = []
    extension_rejecting = []
    if "extension" in holdout_results:
        # Re-derive raw CV5 lists; we already have cv5_pooled_*
        # Get extension confirming/rejecting from the per-round classification
        # We don't have them stored, so re-classify
        a_dec = load_decisions(TMP_BASE / "extension_A" / "p1c_step2_A" / "extension" / "trades.csv")
        b_dec = load_decisions(TMP_BASE / "extension_B" / "p1c_step2_B" / "extension" / "trades.csv")
        ec, er, _, _ = classify_and_compute(a_dec, b_dec, "extension")
        extension_confirming = ec
        extension_rejecting = er

    pooled_confirming = cv5_pooled_confirming + extension_confirming
    pooled_rejecting = cv5_pooled_rejecting + extension_rejecting
    pooled_summary = stats_summary(pooled_confirming, pooled_rejecting)
    print(f"\n[POOLED CV5 + extension]: n_c={pooled_summary['n_confirming']} n_r={pooled_summary['n_rejecting']}",
          flush=True)
    print(f"  diff={pooled_summary['diff']:+.5f}  "
          f"CI=[{pooled_summary.get('ci95_lower', 0):+.5f}, {pooled_summary.get('ci95_upper', 0):+.5f}]",
          flush=True)

    # ---- Verdict per v2.1 ----
    cv5_diff = cv5_summary["diff"]
    ext_diff = per_slice.get("extension", {}).get("diff", 0.0)
    pooled_diff = pooled_summary["diff"]
    pooled_ci_lower = pooled_summary.get("ci95_lower", -float("inf"))
    pooled_ci_upper = pooled_summary.get("ci95_upper", float("inf"))

    same_sign_cv5_ext = (cv5_diff >= 0 and ext_diff >= 0) or (cv5_diff < 0 and ext_diff < 0)
    pass_pooled_thresh = (pooled_diff >= PASS_DIFF_MIN and pooled_ci_lower >= 0)
    hard_fail = (pooled_diff <= HARD_FAIL_POOLED_MIN) or (ext_diff < HARD_FAIL_EXT_MIN)

    if hard_fail:
        verdict = "HARD_FAIL"
    elif pass_pooled_thresh and same_sign_cv5_ext:
        verdict = "PASS"
    else:
        # CI spans 0 or signs disagree
        if not same_sign_cv5_ext:
            verdict = "INCONCLUSIVE_SIGN_DISAGREE"
        else:
            verdict = "INCONCLUSIVE_INSUFFICIENT_POWER"

    print(f"\n  CV5 diff:        {cv5_diff:+.5f}", flush=True)
    print(f"  extension diff:  {ext_diff:+.5f}", flush=True)
    print(f"  pooled diff:     {pooled_diff:+.5f}", flush=True)
    print(f"  same-sign CV5/ext: {same_sign_cv5_ext}", flush=True)
    print(f"  pooled CI lower >= 0: {pooled_ci_lower >= 0}", flush=True)
    print(f"  pooled diff >= +{PASS_DIFF_MIN}: {pooled_diff >= PASS_DIFF_MIN}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    out = {
        "spec": {
            "pass_diff_min": PASS_DIFF_MIN,
            "hard_fail_pooled_min": HARD_FAIL_POOLED_MIN,
            "hard_fail_ext_min": HARD_FAIL_EXT_MIN,
            "cv5_folds": CV5_FOLDS,
            "holdout_slices": HOLDOUT_SLICES,
        },
        "per_slice": per_slice,
        "pooled_cv5_extension": pooled_summary,
        "verdict": verdict,
        "verdict_reasons": {
            "cv5_diff": cv5_diff,
            "ext_diff": ext_diff,
            "pooled_diff": pooled_diff,
            "same_sign_cv5_ext": same_sign_cv5_ext,
            "pooled_ci_lower_geq_zero": pooled_ci_lower >= 0,
            "pooled_diff_geq_pass": pooled_diff >= PASS_DIFF_MIN,
            "hard_fail": hard_fail,
        },
        "elapsed_seconds": time.time() - t_start,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResult JSON: {OUT}", flush=True)
    print(f"Total elapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
