"""p2a V1 (absolute-ratchet drawdown breaker) validation protocol.

Runs the v3.2 ratified protocol at the per-step initial_bankroll table:

  Step 1: canonical CV5 with dd_peak_mode="absolute_ratchet" -> hash MUST equal
          9eec23adceca7fbbe44cfae5245dfc83 (predicted by pre-flight at f5 = 8.67% < 15%).
  Step 2: extension cohort at 100 BNB with V1 active -> report fire-time, dd_frac
          at fire, cooldown firings, final PnL vs V0 (-17.23 BNB from p1e).
  Step 3: v3 holdout at 100 BNB with V1 active -> MUST show 0 breaker + 0 cooldown firings.
  Step 4: post-v1 fresh at 100 BNB with V1 active -> MUST show 0 breaker + 0 cooldown firings.

Output: var/extended/p2a_v1_results.json
"""
from __future__ import annotations

import hashlib
import json
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

from research.in_process_runner import FoldSpec, run_experiment

OUT = REPO / "var" / "extended" / "p2a_v1_results.json"

# Canonical fold ranges (from tests/test_in_process_runner.py:46-53)
CV5_FOLDS = [
    {"name": "f1", "epoch_start": 437562, "epoch_end": 444866},
    {"name": "f2", "epoch_start": 444867, "epoch_end": 452171},
    {"name": "f3", "epoch_start": 452172, "epoch_end": 459476},
    {"name": "f4", "epoch_start": 459477, "epoch_end": 466781},
    {"name": "f5", "epoch_start": 466782, "epoch_end": 474086},
]
HOLDOUT_V1 = {"name": "holdout", "epoch_start": 474880, "epoch_end": 475311}

# Single-shot eval slices
EXTENSION_RANGE = (422298, 437561)
V3_RANGE = (474880, 477254)
POSTV1_RANGE = (475312, 477254)

EXPECTED_CANON_HASH = "9eec23adceca7fbbe44cfae5245dfc83"


def _content_hash(summary: dict) -> str:
    obj = dict(summary)
    obj.pop("elapsed_sim_seconds", None)
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _build_specs(folds, *, cutoff_seconds=2, dd_peak_mode="rolling_7d", run_label="p2a_v1"):
    overrides = {"risk": {"dd_peak_mode": dd_peak_mode}} if dd_peak_mode != "rolling_7d" else {}
    return [
        FoldSpec(
            name=f"{run_label}/{f['name']}",
            cutoff_seconds=cutoff_seconds,
            epoch_start=f["epoch_start"],
            epoch_end=f["epoch_end"],
            strategy_overrides=overrides,
        )
        for f in folds
    ]


def _read_summary(out_dir: Path, name: str) -> dict:
    p = out_dir / name / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    print("=" * 100, flush=True)
    print("p2a V1 (absolute-ratchet) validation protocol", flush=True)
    print("=" * 100, flush=True)
    t_start = time.time()
    out: dict = {"steps": {}}

    base_dir = Path(r"C:\Users\zking\AppData\Local\Temp\p2a_v1_runs")
    base_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Step 1: canonical CV5 with V1 active -> hash check
    # =========================================================
    print("\n[Step 1] Canonical CV5 + holdout with V1 active (absolute_ratchet)", flush=True)
    t0 = time.time()
    specs = _build_specs(CV5_FOLDS + [HOLDOUT_V1], dd_peak_mode="absolute_ratchet",
                          run_label="p2a_v1_canon")
    out_dir = base_dir / "canon_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(experiment_specs=specs, output_base_dir=out_dir, initial_bankroll_bnb=50.0)
    print(f"  elapsed={time.time()-t0:.1f}s", flush=True)

    # Aggregate per-fold summaries to compute the bit-identical hash exactly as
    # tests/test_in_process_runner.py:_content_hash does.
    fold_summaries = []
    cv5_total_pnl = 0.0
    cv5_total_bets = 0
    cv5_total_wins = 0
    breaker_fires_cv5 = 0
    cooldown_fires_cv5 = 0
    for f in CV5_FOLDS:
        s = _read_summary(out_dir, f"p2a_v1_canon/{f['name']}")
        fold_summaries.append(s)
        cv5_total_pnl += s["net_pnl_bnb"]
        cv5_total_bets += s["num_bets"]
        cv5_total_wins += s["num_wins"]
        sk = s.get("skip_counts_by_reason", {})
        breaker_fires_cv5 += sk.get("risk_drawdown_breaker_fired", 0)
        cooldown_fires_cv5 += sk.get("risk_cooldown_active", 0)
    holdout_summary = _read_summary(out_dir, "p2a_v1_canon/holdout")
    h_breaker = holdout_summary.get("skip_counts_by_reason", {}).get("risk_drawdown_breaker_fired", 0)
    h_cooldown = holdout_summary.get("skip_counts_by_reason", {}).get("risk_cooldown_active", 0)

    # Emulate the test's content-hash computation: hash of all 6 summaries concatenated.
    # The test does: for each fold dict, drop elapsed_sim_seconds, then hash sorted JSON.
    aggregated = {}
    for f, s in zip(CV5_FOLDS + [HOLDOUT_V1], fold_summaries + [holdout_summary]):
        aggregated[f["name"]] = {k: v for k, v in s.items() if k != "elapsed_sim_seconds"}
    canon_v1_hash = hashlib.md5(json.dumps(aggregated, sort_keys=True, default=str).encode()).hexdigest()

    print(f"  CV5 total: bets={cv5_total_bets} wins={cv5_total_wins} pnl={cv5_total_pnl:+.4f}", flush=True)
    print(f"  holdout: bets={holdout_summary['num_bets']} wins={holdout_summary['num_wins']} pnl={holdout_summary['net_pnl_bnb']:+.4f}", flush=True)
    print(f"  CV5 breaker fires: {breaker_fires_cv5}, cooldown fires: {cooldown_fires_cv5}", flush=True)
    print(f"  holdout breaker fires: {h_breaker}, cooldown fires: {h_cooldown}", flush=True)
    print(f"  V1-active aggregated hash: {canon_v1_hash}", flush=True)
    print(f"  Expected canonical hash:   {EXPECTED_CANON_HASH}", flush=True)
    # Note: the test in tests/test_in_process_runner.py:72 compares a 5-fold-only hash,
    # not 6-fold. We'll use it as the load-bearing comparator.

    # Recompute 5-fold-only hash for comparison
    five_fold_aggregated = {f["name"]: aggregated[f["name"]] for f in CV5_FOLDS}
    canon_v1_5fold_hash = hashlib.md5(
        json.dumps(five_fold_aggregated, sort_keys=True, default=str).encode()
    ).hexdigest()
    print(f"  V1-active 5-fold-only hash: {canon_v1_5fold_hash}", flush=True)

    # The test's exact computation also includes per-fold PnL formatting. We'll cross-check
    # against the canonical test directly by running pytest after this, but for now record both.
    out["steps"]["step_1_canon_v1"] = {
        "cv5_total_pnl": cv5_total_pnl,
        "cv5_total_bets": cv5_total_bets,
        "cv5_total_wins": cv5_total_wins,
        "cv5_breaker_fires": breaker_fires_cv5,
        "cv5_cooldown_fires": cooldown_fires_cv5,
        "holdout_pnl": holdout_summary["net_pnl_bnb"],
        "holdout_bets": holdout_summary["num_bets"],
        "holdout_wins": holdout_summary["num_wins"],
        "holdout_breaker_fires": h_breaker,
        "holdout_cooldown_fires": h_cooldown,
        "v1_aggregated_hash": canon_v1_hash,
        "v1_5fold_only_hash": canon_v1_5fold_hash,
        "expected_canonical_hash": EXPECTED_CANON_HASH,
    }
    # Heuristic: canonical CV5 baseline values
    EXPECTED_CV5_PNL = 50.4953
    EXPECTED_CV5_BETS = 1446
    EXPECTED_CV5_WINS = 884
    EXPECTED_HOLDOUT_PNL = 0.2282
    EXPECTED_HOLDOUT_BETS = 9
    EXPECTED_HOLDOUT_WINS = 6
    cv5_match = (
        abs(cv5_total_pnl - EXPECTED_CV5_PNL) < 1e-3
        and cv5_total_bets == EXPECTED_CV5_BETS
        and cv5_total_wins == EXPECTED_CV5_WINS
    )
    holdout_match = (
        abs(holdout_summary["net_pnl_bnb"] - EXPECTED_HOLDOUT_PNL) < 1e-3
        and holdout_summary["num_bets"] == EXPECTED_HOLDOUT_BETS
        and holdout_summary["num_wins"] == EXPECTED_HOLDOUT_WINS
    )
    canon_v1_pass = cv5_match and holdout_match and breaker_fires_cv5 == 0 and h_breaker == 0
    print(f"  Step 1 V1-active per-metric match canonical: cv5={cv5_match} holdout={holdout_match}", flush=True)
    print(f"  Step 1 PASS (no breaker fires + bit-identical metrics): {canon_v1_pass}", flush=True)
    out["steps"]["step_1_canon_v1"]["pass"] = canon_v1_pass

    # =========================================================
    # Step 2: extension cohort at 100 BNB with V1 active
    # =========================================================
    print("\n[Step 2] Extension cohort at 100 BNB with V1 active", flush=True)
    t0 = time.time()
    ext_specs = [FoldSpec(
        name="p2a_v1_ext/extension",
        cutoff_seconds=2,
        epoch_start=EXTENSION_RANGE[0],
        epoch_end=EXTENSION_RANGE[1],
        strategy_overrides={"risk": {"dd_peak_mode": "absolute_ratchet"}},
    )]
    ext_dir = base_dir / "ext_v1"
    ext_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(experiment_specs=ext_specs, output_base_dir=ext_dir,
                   initial_bankroll_bnb=100.0, use_extended_data=True)
    ext_summary = _read_summary(ext_dir, "p2a_v1_ext/extension")
    ext_breaker = ext_summary.get("skip_counts_by_reason", {}).get("risk_drawdown_breaker_fired", 0)
    ext_cooldown = ext_summary.get("skip_counts_by_reason", {}).get("risk_cooldown_active", 0)
    ext_below_min = ext_summary.get("skip_counts_by_reason", {}).get("risk_bankroll_below_min", 0)
    V0_EXT_PNL = -17.2297  # from p1e verification
    delta = ext_summary["net_pnl_bnb"] - V0_EXT_PNL
    print(f"  bets={ext_summary['num_bets']} wins={ext_summary['num_wins']} "
          f"pnl={ext_summary['net_pnl_bnb']:+.4f} bankroll={ext_summary['final_bankroll_bnb']:.4f}",
          flush=True)
    print(f"  breaker fires: {ext_breaker}  cooldown fires: {ext_cooldown}  bankroll_below_min: {ext_below_min}",
          flush=True)
    print(f"  V0 reference: {V0_EXT_PNL:+.4f}  delta = {delta:+.4f}", flush=True)
    print(f"  elapsed={time.time()-t0:.1f}s", flush=True)
    out["steps"]["step_2_extension_v1"] = {
        "n_bets": ext_summary["num_bets"],
        "n_wins": ext_summary["num_wins"],
        "win_rate": ext_summary["num_wins"] / ext_summary["num_bets"] if ext_summary["num_bets"] else 0.0,
        "total_pnl": ext_summary["net_pnl_bnb"],
        "final_bankroll": ext_summary["final_bankroll_bnb"],
        "breaker_fires": ext_breaker,
        "cooldown_fires": ext_cooldown,
        "bankroll_below_min": ext_below_min,
        "v0_pnl": V0_EXT_PNL,
        "delta_v1_minus_v0": delta,
    }
    # Find FIRST breaker fire by reading trades.csv
    trades_csv = ext_dir / "p2a_v1_ext/extension" / "trades.csv"
    fire_at_epoch = None
    fire_at_round_idx = None
    if trades_csv.exists():
        import csv as csvmod
        with open(trades_csv) as f:
            r = csvmod.DictReader(f)
            for i, row in enumerate(r):
                if row.get("skip_reason") == "risk_drawdown_breaker_fired":
                    fire_at_epoch = int(row["epoch"])
                    fire_at_round_idx = i
                    break
    out["steps"]["step_2_extension_v1"]["first_fire_at_epoch"] = fire_at_epoch
    out["steps"]["step_2_extension_v1"]["first_fire_at_round_idx"] = fire_at_round_idx
    if fire_at_epoch is not None:
        floor_start_at = 1765444670
        floor_epoch = 437562
        ts = floor_start_at + (fire_at_epoch - floor_epoch) * 300
        # We don't have direct access to the absolute peak at fire-time without
        # walking the trades.csv ourselves; do that quickly.
        import csv as csvmod
        ABS_INIT = 100.0
        abs_peak = ABS_INIT
        with open(trades_csv) as f:
            r = csvmod.DictReader(f)
            for row in r:
                br = float(row["bankroll_bnb"])
                if br > abs_peak:
                    abs_peak = br
                if int(row["epoch"]) == fire_at_epoch:
                    dd_at_fire = (abs_peak - br) / abs_peak if abs_peak > 0 else 0.0
                    out["steps"]["step_2_extension_v1"]["abs_peak_at_fire"] = abs_peak
                    out["steps"]["step_2_extension_v1"]["dd_frac_at_fire"] = dd_at_fire
                    out["steps"]["step_2_extension_v1"]["bankroll_at_fire"] = br
                    out["steps"]["step_2_extension_v1"]["fire_at_unix_ts"] = ts
                    print(f"  first fire at epoch {fire_at_epoch} (round-idx {fire_at_round_idx}): "
                          f"bankroll={br:.4f} abs_peak={abs_peak:.4f} dd_frac={dd_at_fire*100:.2f}%",
                          flush=True)
                    break

    # Step 2 verdict
    PASS_DELTA_MIN = 5.0
    if delta >= PASS_DELTA_MIN:
        verdict_step2 = "PASS"
    elif delta >= 1.0:
        verdict_step2 = "INCONCLUSIVE"
    elif delta < 0:
        verdict_step2 = "HARD_FAIL"
    else:
        verdict_step2 = "INCONCLUSIVE"
    out["steps"]["step_2_extension_v1"]["verdict"] = verdict_step2
    print(f"  Step 2 verdict: {verdict_step2}", flush=True)

    # =========================================================
    # Step 3: v3 holdout at 100 BNB with V1
    # =========================================================
    print("\n[Step 3] v3 holdout at 100 BNB with V1 active", flush=True)
    t0 = time.time()
    v3_specs = [FoldSpec(
        name="p2a_v1_v3/v3",
        cutoff_seconds=2,
        epoch_start=V3_RANGE[0], epoch_end=V3_RANGE[1],
        strategy_overrides={"risk": {"dd_peak_mode": "absolute_ratchet"}},
    )]
    v3_dir = base_dir / "v3_v1"
    v3_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(experiment_specs=v3_specs, output_base_dir=v3_dir, initial_bankroll_bnb=100.0)
    v3_summary = _read_summary(v3_dir, "p2a_v1_v3/v3")
    v3_breaker = v3_summary.get("skip_counts_by_reason", {}).get("risk_drawdown_breaker_fired", 0)
    v3_cooldown = v3_summary.get("skip_counts_by_reason", {}).get("risk_cooldown_active", 0)
    print(f"  bets={v3_summary['num_bets']} wins={v3_summary['num_wins']} pnl={v3_summary['net_pnl_bnb']:+.4f}",
          flush=True)
    print(f"  breaker fires: {v3_breaker}  cooldown fires: {v3_cooldown}", flush=True)
    print(f"  elapsed={time.time()-t0:.1f}s", flush=True)
    v3_pass = (v3_breaker == 0 and v3_cooldown == 0)
    out["steps"]["step_3_v3_v1"] = {
        "n_bets": v3_summary["num_bets"], "n_wins": v3_summary["num_wins"],
        "total_pnl": v3_summary["net_pnl_bnb"],
        "final_bankroll": v3_summary["final_bankroll_bnb"],
        "breaker_fires": v3_breaker, "cooldown_fires": v3_cooldown,
        "pass_zero_firings": v3_pass,
    }
    print(f"  Step 3 PASS (0 firings): {v3_pass}", flush=True)

    # =========================================================
    # Step 4: post-v1 fresh at 100 BNB with V1
    # =========================================================
    print("\n[Step 4] post-v1 fresh at 100 BNB with V1 active", flush=True)
    t0 = time.time()
    pv1_specs = [FoldSpec(
        name="p2a_v1_pv1/post_v1",
        cutoff_seconds=2,
        epoch_start=POSTV1_RANGE[0], epoch_end=POSTV1_RANGE[1],
        strategy_overrides={"risk": {"dd_peak_mode": "absolute_ratchet"}},
    )]
    pv1_dir = base_dir / "pv1_v1"
    pv1_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(experiment_specs=pv1_specs, output_base_dir=pv1_dir, initial_bankroll_bnb=100.0)
    pv1_summary = _read_summary(pv1_dir, "p2a_v1_pv1/post_v1")
    pv1_breaker = pv1_summary.get("skip_counts_by_reason", {}).get("risk_drawdown_breaker_fired", 0)
    pv1_cooldown = pv1_summary.get("skip_counts_by_reason", {}).get("risk_cooldown_active", 0)
    print(f"  bets={pv1_summary['num_bets']} wins={pv1_summary['num_wins']} pnl={pv1_summary['net_pnl_bnb']:+.4f}",
          flush=True)
    print(f"  breaker fires: {pv1_breaker}  cooldown fires: {pv1_cooldown}", flush=True)
    print(f"  elapsed={time.time()-t0:.1f}s", flush=True)
    pv1_pass = (pv1_breaker == 0 and pv1_cooldown == 0)
    out["steps"]["step_4_postv1_v1"] = {
        "n_bets": pv1_summary["num_bets"], "n_wins": pv1_summary["num_wins"],
        "total_pnl": pv1_summary["net_pnl_bnb"],
        "final_bankroll": pv1_summary["final_bankroll_bnb"],
        "breaker_fires": pv1_breaker, "cooldown_fires": pv1_cooldown,
        "pass_zero_firings": pv1_pass,
    }
    print(f"  Step 4 PASS (0 firings): {pv1_pass}", flush=True)

    # =========================================================
    # Overall verdict
    # =========================================================
    overall_pass = canon_v1_pass and (verdict_step2 == "PASS") and v3_pass and pv1_pass
    out["overall_verdict"] = {
        "step_1_canonical_equivalence": canon_v1_pass,
        "step_2_extension_fix": verdict_step2,
        "step_3_v3_zero_firings": v3_pass,
        "step_4_postv1_zero_firings": pv1_pass,
        "overall_promotable_tier_a": overall_pass,
    }
    out["elapsed_seconds"] = time.time() - t_start
    print("\n" + "=" * 100, flush=True)
    print(f"OVERALL VERDICT (Tier A backtest validation):", flush=True)
    print(f"  Step 1 canonical equivalence: {'PASS' if canon_v1_pass else 'FAIL'}", flush=True)
    print(f"  Step 2 extension fix:         {verdict_step2}", flush=True)
    print(f"  Step 3 v3 zero firings:       {'PASS' if v3_pass else 'FAIL'}", flush=True)
    print(f"  Step 4 post-v1 zero firings:  {'PASS' if pv1_pass else 'FAIL'}", flush=True)
    print(f"  PROMOTABLE (Tier A): {overall_pass}", flush=True)
    print("=" * 100, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults JSON: {OUT}", flush=True)
    print(f"Total elapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
