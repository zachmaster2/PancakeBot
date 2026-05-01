"""p1d Step 0: B = (4,8,16) cs=2 standalone on canonical CV5 + holdout.

Verifies B produces sensible per-fold totals and captures B's identity baseline
hash. Compared side-by-side with A = canonical (3,7,15) cs=2 baseline metrics.

This is non-controversial diagnostic prep that runs before reviewer ratifies
the v2 protocol. No portfolio harness, no decision correlation — just B's
standalone behavior under the canonical test setup (initial_bankroll=50.0,
five fold ranges + holdout from tests/test_in_process_runner.py).
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

OUT = REPO / "var" / "extended" / "p1d_step0_b_standalone_results.json"

# Canonical fold ranges (from tests/test_in_process_runner.py:46-53)
CV5_FOLDS = [
    {"name": "f1", "epoch_start": 437562, "epoch_end": 444866},
    {"name": "f2", "epoch_start": 444867, "epoch_end": 452171},
    {"name": "f3", "epoch_start": 452172, "epoch_end": 459476},
    {"name": "f4", "epoch_start": 459477, "epoch_end": 466781},
    {"name": "f5", "epoch_start": 466782, "epoch_end": 474086},
]
HOLDOUT_V1 = {"name": "holdout", "epoch_start": 474880, "epoch_end": 475311}

# A = canonical (3,7,15) baseline values from tests/test_in_process_runner.py:55-71
A_FOLD_STATS = {
    "f1":      {"bets": 129, "wins": 85,  "pnl": 4.2602},
    "f2":      {"bets": 196, "wins": 120, "pnl": 7.3128},
    "f3":      {"bets": 473, "wins": 291, "pnl": 20.2876},
    "f4":      {"bets": 411, "wins": 251, "pnl": 17.0644},
    "f5":      {"bets": 237, "wins": 137, "pnl": 1.5703},
    "holdout": {"bets": 9,   "wins": 6,   "pnl": 0.2282},
}
A_CV5_TOTAL_PNL = 50.4953
A_HASH = "9eec23adceca7fbbe44cfae5245dfc83"


def main():
    print("=" * 100, flush=True)
    print("p1d Step 0 — B = (4,8,16) cs=2 standalone on canonical CV5 + holdout", flush=True)
    print("=" * 100, flush=True)
    t_start = time.time()

    base_dir = Path(r"C:\Users\zking\AppData\Local\Temp\p1d_step0_runs")
    base_dir.mkdir(parents=True, exist_ok=True)

    # B specs: same fold ranges as canonical, with mtf_lookbacks override
    overrides_B = {"gate": {"mtf_lookbacks": [4, 8, 16]}}
    specs = [
        FoldSpec(
            name=f"p1d_b_alone/{f['name']}",
            cutoff_seconds=2,
            epoch_start=f["epoch_start"],
            epoch_end=f["epoch_end"],
            strategy_overrides=overrides_B,
        )
        for f in CV5_FOLDS + [HOLDOUT_V1]
    ]
    out_dir = base_dir / "b_alone"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running B on {len(specs)} folds at initial_bankroll=50.0...", flush=True)
    run_experiment(experiment_specs=specs, output_base_dir=out_dir, initial_bankroll_bnb=50.0)

    # Aggregate
    fold_summaries = {}
    for f in CV5_FOLDS + [HOLDOUT_V1]:
        s = json.loads((out_dir / f"p1d_b_alone/{f['name']}" / "summary.json").read_text())
        fold_summaries[f["name"]] = s

    # Print A-vs-B side-by-side
    print()
    print(f"{'fold':<10}{'A_bets':>8}{'A_wins':>8}{'A_pnl':>10}    "
          f"{'B_bets':>8}{'B_wins':>8}{'B_pnl':>10}{'B_wr':>8}    "
          f"{'Δbets':>7}{'Δpnl':>9}", flush=True)
    print("-" * 110, flush=True)
    cv5_a_bets = cv5_a_wins = 0
    cv5_a_pnl = 0.0
    cv5_b_bets = cv5_b_wins = 0
    cv5_b_pnl = 0.0
    per_bet_b: list[float] = []  # accumulate per-bet PnL across CV5 for σ
    rows = []
    for f in CV5_FOLDS + [HOLDOUT_V1]:
        name = f["name"]
        a = A_FOLD_STATS[name]
        s = fold_summaries[name]
        b_bets = s["num_bets"]
        b_wins = s["num_wins"]
        b_pnl = s["net_pnl_bnb"]
        b_wr = b_wins / b_bets if b_bets > 0 else 0.0
        d_bets = b_bets - a["bets"]
        d_pnl = b_pnl - a["pnl"]
        rows.append({
            "fold": name,
            "a_bets": a["bets"], "a_wins": a["wins"], "a_pnl": a["pnl"],
            "b_bets": b_bets, "b_wins": b_wins, "b_pnl": b_pnl, "b_wr": b_wr,
            "delta_bets": d_bets, "delta_pnl": d_pnl,
        })
        print(f"{name:<10}{a['bets']:>8}{a['wins']:>8}{a['pnl']:>+10.4f}    "
              f"{b_bets:>8}{b_wins:>8}{b_pnl:>+10.4f}{b_wr*100:>7.2f}%    "
              f"{d_bets:>+7}{d_pnl:>+9.4f}", flush=True)
        if name != "holdout":
            cv5_a_bets += a["bets"]; cv5_a_wins += a["wins"]; cv5_a_pnl += a["pnl"]
            cv5_b_bets += b_bets; cv5_b_wins += b_wins; cv5_b_pnl += b_pnl
    print("-" * 110, flush=True)
    print(f"{'CV5 sum':<10}{cv5_a_bets:>8}{cv5_a_wins:>8}{cv5_a_pnl:>+10.4f}    "
          f"{cv5_b_bets:>8}{cv5_b_wins:>8}{cv5_b_pnl:>+10.4f}"
          f"{(cv5_b_wins/cv5_b_bets)*100 if cv5_b_bets else 0:>7.2f}%    "
          f"{cv5_b_bets - cv5_a_bets:>+7}{cv5_b_pnl - cv5_a_pnl:>+9.4f}", flush=True)

    # B identity hash (5-fold-only, mirrors test_in_process_runner.py:72 semantics)
    aggregated = {}
    for f in CV5_FOLDS:
        s = fold_summaries[f["name"]]
        aggregated[f["name"]] = {k: v for k, v in s.items() if k != "elapsed_sim_seconds"}
    b_5fold_hash = hashlib.md5(
        json.dumps(aggregated, sort_keys=True, default=str).encode()
    ).hexdigest()
    print(f"\nB identity hash (5-fold-only): {b_5fold_hash}", flush=True)
    print(f"A canonical hash (reference):  {A_HASH}", flush=True)

    # Per-bet stats on CV5 from B's trades.csv (for σ comparison vs A's 0.354)
    import csv as csvmod
    b_per_bet_pnls = []
    for f in CV5_FOLDS:
        trades = out_dir / f"p1d_b_alone/{f['name']}" / "trades.csv"
        with open(trades) as fp:
            r = csvmod.DictReader(fp)
            for row in r:
                if row["action"] == "BET":
                    b_per_bet_pnls.append(float(row["profit_bnb"]))
    if b_per_bet_pnls:
        import statistics
        mean = statistics.mean(b_per_bet_pnls)
        std = statistics.stdev(b_per_bet_pnls) if len(b_per_bet_pnls) > 1 else 0.0
        print(f"\nB per-bet stats (CV5, n={len(b_per_bet_pnls)}):", flush=True)
        print(f"  mean: {mean:+.5f} BNB/bet  (A reference: +0.04047)", flush=True)
        print(f"  std:  {std:.5f} BNB/bet  (A reference: 0.35413)", flush=True)
    else:
        mean = std = 0.0

    # Save
    out = {
        "spec": {
            "variant_B_lookbacks": [4, 8, 16],
            "cutoff_seconds": 2,
            "initial_bankroll_bnb": 50.0,
            "fold_ranges": [{"name": f["name"], "epoch_start": f["epoch_start"], "epoch_end": f["epoch_end"]}
                              for f in CV5_FOLDS + [HOLDOUT_V1]],
        },
        "per_fold": rows,
        "B_cv5_total": {
            "bets": cv5_b_bets, "wins": cv5_b_wins, "pnl": cv5_b_pnl,
            "win_rate": (cv5_b_wins / cv5_b_bets) if cv5_b_bets else 0.0,
        },
        "A_cv5_reference": {
            "bets": cv5_a_bets, "wins": cv5_a_wins, "pnl": cv5_a_pnl,
        },
        "B_per_bet_cv5_stats": {
            "n": len(b_per_bet_pnls),
            "mean": mean,
            "std": std,
        },
        "A_per_bet_cv5_reference": {
            "n": 1209,
            "mean": 0.04047,
            "std": 0.35413,
        },
        "B_identity_5fold_hash": b_5fold_hash,
        "A_canonical_hash_reference": A_HASH,
        "elapsed_seconds": time.time() - t_start,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResult JSON: {OUT}", flush=True)
    print(f"Total elapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
