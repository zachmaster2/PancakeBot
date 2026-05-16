"""Re-aggregate sweep_2026_05_08 with fixed key names.

The original harness used `net_profit_bnb`, but in_process_runner writes
`net_pnl_bnb`. All 918 fold summaries are valid; only aggregation was wrong.
This script re-reads them, picks the correct top-5, runs Phase 4 (extension),
and writes the memo.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force-reload the (now-patched) module so we use the corrected aggregator.
import importlib  # noqa: E402
import research.cutoff_lookback_sweep_2026_05_08 as sweep  # noqa: E402
importlib.reload(sweep)

from research.in_process_runner import run_experiment  # noqa: E402


def main() -> None:
    variants = sweep._generate_variants()
    print(f"Re-aggregating {len(variants)} variants from existing fold summaries...")

    aggregated: list[dict] = []
    for cutoff, a, b, c in variants:
        label = sweep._variant_label(cutoff, a, b, c)
        agg = sweep._aggregate_variant(label, output_dir=sweep.OUTPUT_BASE)
        agg["cutoff"] = cutoff
        agg["a"] = a
        agg["b"] = b
        agg["c"] = c
        aggregated.append(agg)
        sweep._concat_cv5_trades(label, output_dir=sweep.OUTPUT_BASE)

    import json
    comparison = {
        "canonical": {
            "cv5_pnl_bnb": sweep.CANONICAL_CV5_PNL,
            "f5_floor_bnb": sweep.CANONICAL_F5_FLOOR,
            "holdout_pnl_bnb": sweep.CANONICAL_HOLDOUT_PNL,
        },
        "variants": aggregated,
    }
    comp_path = sweep.OUTPUT_BASE / "comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"  wrote {comp_path}")

    # Rank
    ranked = sorted(
        aggregated,
        key=lambda v: (v["gates_passed"], v["cv5_pnl_bnb"]),
        reverse=True,
    )
    print(f"\nTop 15 by (gates_passed, cv5_pnl_bnb):")
    print(
        f"  {'label':18s}  {'gates':>5s}  {'cv5':>9s}  "
        f"{'folds+':>6s}  {'f5':>8s}  {'holdout':>8s}  "
        f"{'bets':>6s}  {'wr':>6s}"
    )
    for v in ranked[:15]:
        print(
            f"  {v['label']:18s}  {v['gates_passed']:5d}  "
            f"{v['cv5_pnl_bnb']:>9.2f}  {v['folds_positive']:>6d}  "
            f"{v['f5_floor_bnb']:>8.3f}  {v['holdout_pnl_bnb']:>8.3f}  "
            f"{v['cv5_bets']:>6d}  {v['cv5_winrate']*100:>5.2f}%"
        )

    n_pass4 = sum(1 for v in aggregated if v["gates_passed"] == 4)
    n_pass3 = sum(1 for v in aggregated if v["gates_passed"] >= 3)
    print(f"\nPass all 4 gates: {n_pass4}/{len(aggregated)}")
    print(f"Pass >=3 gates:   {n_pass3}/{len(aggregated)}")

    top5 = ranked[:5]
    top5_labels = [v["label"] for v in top5]
    print(f"\nTOP 5 (re-running on extension): {top5_labels}")

    # Phase 4: extension top-5
    import time
    print(f"\n--- Phase 4: Extension top-5 ---")
    ext_dir = sweep.OUTPUT_BASE / "extension_top5"
    ext_dir.mkdir(parents=True, exist_ok=True)
    ext_specs = []
    for v in top5:
        ext_specs.extend(sweep._build_specs_for_variant(
            v["cutoff"], v["a"], v["b"], v["c"], include_extension=True,
        ))
    print(f"  ext specs: {len(ext_specs)}")
    t0 = time.perf_counter()
    run_experiment(
        experiment_specs=ext_specs,
        output_base_dir=ext_dir,
        use_extended_data=True,
    )
    print(f"  Phase 4 elapsed: {time.perf_counter() - t0:.1f}s")

    # Aggregate extension results
    ext_summaries: list[dict] = []
    for v in top5:
        label = v["label"]
        ext_path = ext_dir / label / "extension" / "summary.json"
        if not ext_path.exists():
            ext_summaries.append({"label": label, "ext_pnl_bnb": None})
            continue
        s = sweep._read_summary(ext_path)
        ext_summaries.append({
            "label": label,
            "ext_pnl_bnb": s.get("net_pnl_bnb", 0.0),
            "ext_bets": s.get("num_bets", 0),
            "ext_wins": s.get("num_wins", 0),
            "ext_breaker_trips": s.get(
                "skip_counts_by_reason", {}).get("risk_drawdown_breaker", 0),
        })

    print(f"\n  Extension cohort top-5 results:")
    print(f"  {'label':18s}  {'ext_pnl':>9s}  {'ext_bets':>9s}  {'ext_wr':>8s}")
    for e in ext_summaries:
        if e.get("ext_pnl_bnb") is None:
            print(f"  {e['label']:18s}  FAILED")
            continue
        wr = (e["ext_wins"] / e["ext_bets"]) if e["ext_bets"] > 0 else 0.0
        print(
            f"  {e['label']:18s}  {e['ext_pnl_bnb']:>9.3f}  "
            f"{e['ext_bets']:>9d}  {wr*100:>7.2f}%"
        )

    ext_comp = {"canonical_baseline": True, "top5": ext_summaries}
    (ext_dir / "extension_results.json").write_text(
        json.dumps(ext_comp, indent=2), encoding="utf-8",
    )

    # Memo
    sweep._write_memo(
        aggregated=aggregated,
        top5=top5,
        ext_summaries=ext_summaries,
    )

    print(f"\nDone. Output: {sweep.OUTPUT_BASE}")


if __name__ == "__main__":
    main()
