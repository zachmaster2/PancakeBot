"""Post-sync 5-fold CV baseline on the extended dataset.

The strong-bypass removal plus today's +432 round sync define a new baseline.
This script runs the canonical 5-fold boundaries (437562..474086) with empty
overrides and reports per-fold PnL/WR/bets plus total sum. The fold-5 figure
is the key anomaly check -- a negative or sharply degraded f5 would signal
the edge eroded during the holdout period and we should NOT restart the dry
bot before surfacing the concern.

The 5 fold boundaries do NOT change across the sync. The new rounds
(474880..475311) are designated the holdout slice (see
research/holdout_2026_04_24.md) and are excluded from CV.

Expected result: identical to today's earlier SB-KILL sweep (proven
byte-equivalent by research/verify_strong_bypass_removal.py on 2026-04-24):
total_pnl ~+50.4953 BNB, 1446 bets, WR 61.13%, fold-5 +1.5703 BNB.

If the result matches: restart dry bot, update memory baseline.
If the result diverges: STOP, surface to user, investigate.

Usage:
    python research/post_sync_baseline_cv.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.sweep_harness import run_one  # noqa: E402

FOLDS = [
    {"name": "f1", "epoch_start": 437562, "epoch_end": 444866},
    {"name": "f2", "epoch_start": 444867, "epoch_end": 452171},
    {"name": "f3", "epoch_start": 452172, "epoch_end": 459476},
    {"name": "f4", "epoch_start": 459477, "epoch_end": 466781},
    {"name": "f5", "epoch_start": 466782, "epoch_end": 474086},
]

OUTPUT_DIR = REPO_ROOT / "var" / "sweep" / "_post_sync_baseline"


def main() -> int:
    per_fold = []
    total_pnl = 0.0
    total_bets = 0
    total_wins = 0
    for fold in FOLDS:
        fname = fold["name"]
        print(f"=== fold {fname} [{fold['epoch_start']}..{fold['epoch_end']}] ===",
              flush=True)
        summary = run_one(
            name=f"_post_sync_baseline/fold_{fname}",
            overrides={},
            epoch_start=fold["epoch_start"],
            epoch_end=fold["epoch_end"],
        )
        pnl = float(summary["net_pnl_bnb"])
        bets = int(summary["num_bets"])
        wins = int(summary["num_wins"])
        wr = float(summary["win_rate"]) if bets else 0.0
        per_fold.append({
            "fold": fname,
            "bets": bets,
            "wins": wins,
            "pnl": pnl,
            "wr": wr,
        })
        total_pnl += pnl
        total_bets += bets
        total_wins += wins
        print(f"  bets={bets} wins={wins} wr={wr:.4f} pnl={pnl:+.4f}", flush=True)

    print()
    print("=" * 66, flush=True)
    print(f"{'fold':<6} {'bets':>6} {'wins':>6} {'wr':>8} {'pnl_bnb':>10}",
          flush=True)
    for row in per_fold:
        print(f"{row['fold']:<6} {row['bets']:>6d} {row['wins']:>6d} "
              f"{row['wr']:>8.4f} {row['pnl']:>+10.4f}", flush=True)
    print("-" * 66, flush=True)
    total_wr = total_wins / total_bets if total_bets else 0.0
    print(f"{'TOTAL':<6} {total_bets:>6d} {total_wins:>6d} "
          f"{total_wr:>8.4f} {total_pnl:>+10.4f}", flush=True)
    print("=" * 66, flush=True)
    print(f"fold-5 check: {per_fold[-1]['pnl']:+.4f} BNB "
          f"({'PASS' if per_fold[-1]['pnl'] > 0 else 'NEGATIVE -- INVESTIGATE'})",
          flush=True)

    # Persist summary for the commit.
    out = OUTPUT_DIR / "cv_summary.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "per_fold": per_fold,
        "total_pnl_bnb": total_pnl,
        "total_bets": total_bets,
        "total_wins": total_wins,
        "total_wr": total_wr,
        "n_positive_folds": sum(1 for r in per_fold if r["pnl"] > 0),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
