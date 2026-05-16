"""Cutoff x mtf_lookbacks sweep — 2026-05-08.

User-greenlit grid:
  cutoffs = {2, 3, 4}
  a in {2, 3, 4}
  b in {5, 7, 10, 15}
  c in {10, 15, 20, 25, 30}
  filter: a < b < c

Per variant: CV5 folds + holdout. Top 5 by promotion-gate-pass-count
re-run on extension cohort (epochs 422298..437561).

Output layout::

    var/sweep_2026_05_08/
      c2_a3_b7_c10/
        f1/trades.csv
        f1/summary.json
        ...
        f5/...
        holdout/...
        summary.json         <- aggregated per-variant
        trades.csv           <- CV5 concatenated
      ...
      comparison.json
      plots/
        primary_heatmap.png
        scoreboard.png
        parallel_coords.png
      extension_top5/
        c2_a3_b7_c15/
          extension/...
          summary.json
        ...
      summary.md             <- promotion gate analysis

Usage::

    python research/cutoff_lookback_sweep_2026_05_08.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.in_process_runner import FoldSpec, run_experiment  # noqa: E402


# Canonical CV5 + holdout boundaries (from project_holdout_slice.md).
_CV5_FOLDS = [
    {"name": "f1", "epoch_start": 437562, "epoch_end": 444866},
    {"name": "f2", "epoch_start": 444867, "epoch_end": 452171},
    {"name": "f3", "epoch_start": 452172, "epoch_end": 459476},
    {"name": "f4", "epoch_start": 459477, "epoch_end": 466781},
    {"name": "f5", "epoch_start": 466782, "epoch_end": 474086},
]
_HOLDOUT = {"name": "holdout", "epoch_start": 474880, "epoch_end": 475311}
_EXTENSION = {"name": "extension", "epoch_start": 422298, "epoch_end": 437561}

# Canonical baseline numbers (from project_holdout_slice.md).
CANONICAL_CV5_PNL = 50.4953
CANONICAL_F5_FLOOR = 1.5703
CANONICAL_HOLDOUT_PNL = 0.2282

OUTPUT_BASE = REPO_ROOT / "var" / "sweep_2026_05_08"


def _generate_variants() -> list[tuple[int, int, int, int]]:
    """Return list of (cutoff, a, b, c) per the user-greenlit grid."""
    cutoffs = [2, 3, 4]
    a_vals = [2, 3, 4]
    b_vals = [5, 7, 10, 15]
    c_vals = [10, 15, 20, 25, 30]
    out: list[tuple[int, int, int, int]] = []
    for cutoff in cutoffs:
        for a in a_vals:
            for b in b_vals:
                for c in c_vals:
                    if a < b < c:
                        out.append((cutoff, a, b, c))
    return out


def _variant_label(cutoff: int, a: int, b: int, c: int) -> str:
    return f"c{cutoff}_a{a}_b{b}_c{c}"


def _build_specs_for_variant(
    cutoff: int, a: int, b: int, c: int, *, include_extension: bool = False,
) -> list[FoldSpec]:
    label = _variant_label(cutoff, a, b, c)
    overrides = {"gate": {"mtf_lookbacks": [a, b, c]}}
    folds = list(_CV5_FOLDS) + [_HOLDOUT]
    if include_extension:
        folds = [_EXTENSION]
    return [
        FoldSpec(
            name=f"{label}/{fold['name']}",
            cutoff_seconds=cutoff,
            epoch_start=fold["epoch_start"],
            epoch_end=fold["epoch_end"],
            strategy_overrides=overrides,
            plot=False,
        )
        for fold in folds
    ]


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trades_count(path: Path) -> tuple[int, int]:
    """Return (total_bets, num_wins) from a fold's trades.csv."""
    if not path.exists():
        return (0, 0)
    bets = 0
    wins = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("action") == "BET":
                bets += 1
                if float(row.get("profit_bnb", 0)) > 0:
                    wins += 1
    return (bets, wins)


def _aggregate_variant(label: str, *, output_dir: Path) -> dict:
    """Read each fold's summary.json and produce an aggregate per-variant
    summary."""
    variant_dir = output_dir / label
    folds_data: list[dict] = []
    for fold in _CV5_FOLDS:
        fold_dir = variant_dir / fold["name"]
        s_path = fold_dir / "summary.json"
        if not s_path.exists():
            continue
        s = _read_summary(s_path)
        folds_data.append({"name": fold["name"], **s})
    holdout_path = variant_dir / "holdout" / "summary.json"
    holdout_data = _read_summary(holdout_path) if holdout_path.exists() else None

    cv5_pnl = sum(d.get("net_pnl_bnb", 0.0) for d in folds_data)
    folds_positive = sum(1 for d in folds_data if d.get("net_pnl_bnb", 0.0) > 0)
    f5_floor = (
        folds_data[-1].get("net_pnl_bnb", 0.0) if len(folds_data) >= 5 else 0.0
    )
    cv5_bets = sum(d.get("num_bets", 0) for d in folds_data)
    cv5_wins = sum(d.get("num_wins", 0) for d in folds_data)
    cv5_winrate = (cv5_wins / cv5_bets) if cv5_bets > 0 else 0.0
    breaker_trips = sum(
        d.get("skip_counts_by_reason", {}).get("risk_drawdown_breaker", 0)
        for d in folds_data
    )

    holdout_pnl = (
        holdout_data.get("net_pnl_bnb", 0.0) if holdout_data else 0.0
    )
    holdout_bets = holdout_data.get("num_bets", 0) if holdout_data else 0
    holdout_wins = holdout_data.get("num_wins", 0) if holdout_data else 0

    # Promotion gates
    gate_cv5 = cv5_pnl >= CANONICAL_CV5_PNL
    gate_folds = folds_positive == 5
    gate_holdout = holdout_pnl > 0
    gate_f5 = f5_floor >= 0.0
    gates_passed = sum(
        1 for g in (gate_cv5, gate_folds, gate_holdout, gate_f5) if g
    )

    return {
        "label": label,
        "cv5_pnl_bnb": cv5_pnl,
        "folds_positive": folds_positive,
        "f5_floor_bnb": f5_floor,
        "cv5_bets": cv5_bets,
        "cv5_winrate": cv5_winrate,
        "breaker_trips": breaker_trips,
        "holdout_pnl_bnb": holdout_pnl,
        "holdout_bets": holdout_bets,
        "holdout_wins": holdout_wins,
        "gates": {
            "cv5_ge_canonical": gate_cv5,
            "all_folds_positive": gate_folds,
            "holdout_positive": gate_holdout,
            "f5_floor_nonneg": gate_f5,
        },
        "gates_passed": gates_passed,
        "per_fold_pnl": [d.get("net_pnl_bnb", 0.0) for d in folds_data],
    }


def _concat_cv5_trades(label: str, *, output_dir: Path) -> None:
    """Concat each variant's 5 CV folds into a single trades.csv at the
    variant dir."""
    variant_dir = output_dir / label
    out_path = variant_dir / "trades.csv"
    rows: list[dict] = []
    header: list[str] | None = None
    for fold in _CV5_FOLDS:
        fold_csv = variant_dir / fold["name"] / "trades.csv"
        if not fold_csv.exists():
            continue
        with open(fold_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = ["fold_name"] + (reader.fieldnames or [])
            for row in reader:
                rows.append({"fold_name": fold["name"], **row})
    if header is None:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    variants = _generate_variants()
    print(f"\n=== Sweep 2026-05-08: {len(variants)} variants ===")
    print(f"  cutoffs={2,3,4}  a={2,3,4}  b={5,7,10,15}  c={10,15,20,25,30}")
    print(f"  output: {OUTPUT_BASE}")

    # --- Phase 1: main sweep (CV5 + holdout) ---
    print(f"\n--- Phase 1: Main sweep ({len(variants)} variants × 6 folds) ---")
    all_specs: list[FoldSpec] = []
    for cutoff, a, b, c in variants:
        all_specs.extend(_build_specs_for_variant(cutoff, a, b, c))
    print(f"  total specs: {len(all_specs)}")

    t0 = time.perf_counter()
    summaries = run_experiment(
        experiment_specs=all_specs,
        output_base_dir=OUTPUT_BASE,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Phase 1 elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  per-spec wall-clock: {elapsed/len(all_specs):.2f}s")

    # --- Phase 2: aggregate per-variant ---
    print(f"\n--- Phase 2: Aggregate per-variant ---")
    aggregated: list[dict] = []
    for cutoff, a, b, c in variants:
        label = _variant_label(cutoff, a, b, c)
        agg = _aggregate_variant(label, output_dir=OUTPUT_BASE)
        agg["cutoff"] = cutoff
        agg["a"] = a
        agg["b"] = b
        agg["c"] = c
        aggregated.append(agg)
        _concat_cv5_trades(label, output_dir=OUTPUT_BASE)

    comparison = {
        "canonical": {
            "cv5_pnl_bnb": CANONICAL_CV5_PNL,
            "f5_floor_bnb": CANONICAL_F5_FLOOR,
            "holdout_pnl_bnb": CANONICAL_HOLDOUT_PNL,
        },
        "variants": aggregated,
    }
    comp_path = OUTPUT_BASE / "comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"  wrote {comp_path}")

    # --- Phase 3: rank + identify top 5 by promotion-gate-pass-count ---
    print(f"\n--- Phase 3: Top variants by gates passed ---")
    ranked = sorted(
        aggregated,
        key=lambda v: (v["gates_passed"], v["cv5_pnl_bnb"]),
        reverse=True,
    )
    print(f"  All {len(ranked)} variants ranked. Top 10:")
    print(
        f"  {'label':18s}  {'gates':>5s}  {'cv5':>9s}  "
        f"{'folds+':>6s}  {'f5':>8s}  {'holdout':>8s}"
    )
    for v in ranked[:10]:
        print(
            f"  {v['label']:18s}  {v['gates_passed']:5d}  "
            f"{v['cv5_pnl_bnb']:>9.2f}  {v['folds_positive']:>6d}  "
            f"{v['f5_floor_bnb']:>8.3f}  {v['holdout_pnl_bnb']:>8.3f}"
        )

    top5 = ranked[:5]
    top5_labels = [v["label"] for v in top5]
    print(f"\n  TOP 5 (re-running on extension): {top5_labels}")

    # --- Phase 4: extension top-5 ---
    print(f"\n--- Phase 4: Extension top-5 ---")
    ext_dir = OUTPUT_BASE / "extension_top5"
    ext_dir.mkdir(parents=True, exist_ok=True)
    ext_specs: list[FoldSpec] = []
    for v in top5:
        ext_specs.extend(_build_specs_for_variant(
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
        s = _read_summary(ext_path)
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

    # Persist extension results
    ext_comp = {"canonical_baseline": True, "top5": ext_summaries}
    (ext_dir / "extension_results.json").write_text(
        json.dumps(ext_comp, indent=2), encoding="utf-8",
    )

    # --- Phase 5: summary memo ---
    _write_memo(aggregated=aggregated, top5=top5, ext_summaries=ext_summaries)

    print(f"\nDone. Wall-clock: {time.perf_counter() - t0:.0f}s.")
    print(f"Output: {OUTPUT_BASE}")


def _write_memo(*, aggregated: list[dict], top5: list[dict],
                ext_summaries: list[dict]) -> None:
    memo_path = REPO_ROOT / "var" / "incident_reports" / "2026_05_08_cutoff_lookback_sweep_results.md"
    memo_path.parent.mkdir(parents=True, exist_ok=True)

    # Patterns
    by_cutoff: dict[int, list[dict]] = {}
    for v in aggregated:
        by_cutoff.setdefault(v["cutoff"], []).append(v)

    lines = [
        "# Cutoff x lookbacks sweep — results (2026-05-08)",
        "",
        "## Summary",
        "",
        f"Grid: cutoffs={{2,3,4}} × a={{2,3,4}} × b={{5,7,10,15}} × c={{10,15,20,25,30}}, filter a<b<c.",
        f"Total variants: {len(aggregated)}.",
        f"Canonical baseline: cutoff=2, lookbacks=(3,7,15), CV5={CANONICAL_CV5_PNL} BNB, "
        f"f5={CANONICAL_F5_FLOOR}, holdout={CANONICAL_HOLDOUT_PNL}.",
        "",
        "## Promotion gate scoreboard (top 10 by gates passed)",
        "",
        "| label | gates | CV5 | folds+ | f5 floor | holdout |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        aggregated,
        key=lambda v: (v["gates_passed"], v["cv5_pnl_bnb"]),
        reverse=True,
    )
    for v in ranked[:10]:
        lines.append(
            f"| {v['label']} | {v['gates_passed']}/4 | "
            f"{v['cv5_pnl_bnb']:.2f} | {v['folds_positive']}/5 | "
            f"{v['f5_floor_bnb']:.3f} | {v['holdout_pnl_bnb']:.3f} |"
        )
    lines += [
        "",
        "## Extension cohort top-5 results",
        "",
        "| label | extension PnL | extension bets | extension WR |",
        "|---|---:|---:|---:|",
    ]
    for e in ext_summaries:
        if e.get("ext_pnl_bnb") is None:
            lines.append(f"| {e['label']} | FAILED | — | — |")
            continue
        wr = (e["ext_wins"] / e["ext_bets"]) if e["ext_bets"] > 0 else 0.0
        lines.append(
            f"| {e['label']} | {e['ext_pnl_bnb']:.3f} | "
            f"{e['ext_bets']} | {wr*100:.2f}% |"
        )

    lines += [
        "",
        "## Patterns by cutoff",
        "",
    ]
    for cutoff in sorted(by_cutoff.keys()):
        vs = by_cutoff[cutoff]
        n_pass4 = sum(1 for v in vs if v["gates_passed"] == 4)
        n_pass3 = sum(1 for v in vs if v["gates_passed"] >= 3)
        best = max(vs, key=lambda v: v["cv5_pnl_bnb"])
        worst = min(vs, key=lambda v: v["cv5_pnl_bnb"])
        n_breaker = sum(1 for v in vs if v["breaker_trips"] > 0)
        lines.append(
            f"- **cutoff={cutoff}** (n={len(vs)}): "
            f"{n_pass4} variants pass all 4 gates, "
            f"{n_pass3} pass ≥3. "
            f"best CV5 = {best['cv5_pnl_bnb']:.2f} BNB ({best['label']}). "
            f"worst CV5 = {worst['cv5_pnl_bnb']:.2f} BNB ({worst['label']}). "
            f"{n_breaker}/{len(vs)} variants tripped drawdown breaker."
        )

    lines += [
        "",
        "## Verdict / recommendation",
        "",
    ]
    if any(v["gates_passed"] == 4 for v in ranked):
        winners = [v for v in ranked if v["gates_passed"] == 4]
        lines.append(f"**{len(winners)} variant(s) cleared all 4 promotion gates.** "
                     f"Recommend: review the extension cohort PnL above; if any "
                     f"top-5 variant beats canonical on extension AND has positive "
                     f"holdout, consider it a promotion candidate. "
                     f"Otherwise close exploration — the canonical (cutoff=2, (3,7,15)) "
                     f"remains binding.")
    elif any(v["gates_passed"] >= 3 for v in ranked):
        lines.append("**No variant cleared all 4 promotion gates.** "
                     "Some near-misses (3/4) exist; review the extension PnL "
                     "to see if the missing gate (most often a single-fold "
                     "negativity or below-canonical CV5) is offset by extension "
                     "outperformance. Likely close exploration.")
    else:
        lines.append("**No variant came close to passing.** "
                     "Canonical (cutoff=2, (3,7,15)) remains the binding "
                     "baseline. The 4-gate promotion rule is doing its job: "
                     "no neighborhood in this sparse pass beats canonical "
                     "on the strict criteria. Recommend close exploration.")
    lines += [
        "",
        "## Files",
        "",
        f"- Per-variant artifacts: `{OUTPUT_BASE.relative_to(REPO_ROOT)}/<label>/`",
        f"- Aggregate comparison: `{OUTPUT_BASE.relative_to(REPO_ROOT)}/comparison.json`",
        f"- Extension top-5 results: `{OUTPUT_BASE.relative_to(REPO_ROOT)}/extension_top5/extension_results.json`",
        "- Plots (if generated): `var/sweep_2026_05_08/plots/`",
        "",
    ]

    memo_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote memo: {memo_path}")


if __name__ == "__main__":
    main()
