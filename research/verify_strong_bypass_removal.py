"""One-shot verification: post-removal code hash must match SB-KILL per-fold.

The strong-bypass sweep's SB_KILL_strength_0.01 variant set the threshold
min_strength = 0.01, far above any observed BTC signal strength. That turned
the regime off at the admission gate; the bypass branch never fired. The
removal commit (516c67d) deletes the regime entirely from the code.

These two approaches (regime-dead-by-threshold vs regime-nonexistent) should
produce byte-identical backtest outputs round-for-round. This script checks
that on the 5 folds from phase_strong_bypass_sweep.py.

If any fold's content_hash(post-removal) != content_hash(SB-KILL) we STOP and
investigate. The expected outcome is 5/5 matches.

Usage:
    python research/verify_strong_bypass_removal.py

Outputs a table of per-fold hashes and a PASS/FAIL line.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.sweep_harness import run_one  # noqa: E402

# Same 5 folds as phase_strong_bypass_sweep.py.
FOLDS = [
    {"name": "f1", "epoch_start": 437562, "epoch_end": 444866},
    {"name": "f2", "epoch_start": 444867, "epoch_end": 452171},
    {"name": "f3", "epoch_start": 452172, "epoch_end": 459476},
    {"name": "f4", "epoch_start": 459477, "epoch_end": 466781},
    {"name": "f5", "epoch_start": 466782, "epoch_end": 474086},
]

SB_KILL_DIR = REPO_ROOT / "var" / "sweep" / "phase_strong_bypass" / "SB_KILL_strength_0.01"
VERIFY_DIR = REPO_ROOT / "var" / "sweep" / "_verify_strong_bypass_removal"


def content_hash(summary_path: Path) -> str:
    """Hash summary.json minus elapsed_sim_seconds, matching sweep_harness."""
    with summary_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    obj.pop("elapsed_sim_seconds", None)
    return hashlib.md5(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()


def main() -> int:
    results = []
    for fold in FOLDS:
        fname = fold["name"]
        print(f"=== fold {fname} [{fold['epoch_start']}..{fold['epoch_end']}] ===", flush=True)

        # Expected hash: SB-KILL output from the sweep.
        kill_summary = SB_KILL_DIR / f"fold_{fname}" / "summary.json"
        if not kill_summary.exists():
            print(f"  MISSING: {kill_summary}")
            return 2
        expected = content_hash(kill_summary)
        print(f"  expected (SB-KILL): {expected}", flush=True)

        # Run post-removal code with empty overrides on the same fold.
        run_one(
            name=f"_verify_strong_bypass_removal/fold_{fname}",
            overrides={},
            epoch_start=fold["epoch_start"],
            epoch_end=fold["epoch_end"],
        )
        actual = content_hash(
            VERIFY_DIR / f"fold_{fname}" / "summary.json"
        )
        print(f"  actual   (post-rm): {actual}", flush=True)

        match = (expected == actual)
        results.append((fname, expected, actual, match))
        print(f"  -> {'MATCH' if match else 'MISMATCH'}", flush=True)
        print(flush=True)

    print("=" * 66, flush=True)
    print(f"{'fold':<6} {'SB-KILL':<34} {'post-removal':<34} match", flush=True)
    for fname, exp, act, match in results:
        print(f"{fname:<6} {exp:<34} {act:<34} {match}", flush=True)
    n_match = sum(1 for _, _, _, m in results if m)
    print("=" * 66, flush=True)
    print(f"RESULT: {n_match}/{len(results)} folds match", flush=True)
    return 0 if n_match == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
