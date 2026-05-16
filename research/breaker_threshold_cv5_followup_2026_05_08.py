"""CV5 follow-up: spot-check V1 baseline (cd=72) at all 3 scales + absolute_scaled
variant at all 3 scales, since the main sweep's CV5 logic only ran cd=12 at 5 BNB.

This reuses the variant-tracker patches from breaker_threshold_cooldown_sweep_2026_05_08.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.breaker_threshold_cooldown_sweep_2026_05_08 import (
    run_one,
    _spec_v1_baseline,
    _spec_absolute_scaled,
    _spec_hybrid,
    CV5_FOLDS,
    SCALES,
)


def main() -> int:
    out_root = REPO_ROOT / "var" / "extended" / "breaker_sweep_2026_05_08_cv5_followup"
    out_root.mkdir(parents=True, exist_ok=True)
    json_out = REPO_ROOT / "var" / "extended" / "breaker_sweep_2026_05_08_cv5_followup.json"

    results: dict = {"cv5": []}
    t0 = time.perf_counter()

    candidates = []
    for scale in SCALES:
        # V1 baseline (dd=0.15, cd=72)
        spec = _spec_v1_baseline(scale)
        candidates.append(("v1_baseline_cd72", scale, spec))
        # absolute_scaled (cd=72)
        spec = _spec_absolute_scaled(scale)
        candidates.append(("absolute_scaled_5pct_init", scale, spec))
        # hybrid
        spec = _spec_hybrid(scale)
        candidates.append(("hybrid_15rel_or_5pct_init", scale, spec))

    # Also: best Part2 cooldowns at 5 BNB. Since cd=72 is identical-pnl to cd=12,
    # we're already covered by V1 baseline.
    # Plus: cd=144 at 5 BNB with V1 baseline.
    candidates.append((
        "v1_dd0.15_cd144", 5.0,
        {"name": "v1_dd0.15_cd144",
         "overrides": {"risk": {"dd_peak_mode": "absolute_ratchet",
                                 "max_drawdown_frac_from_peak": 0.15,
                                 "cooldown_rounds": 144}},
         "variant": "v1_native",
         "abs_thresh": None}
    ))

    for label, scale, spec in candidates:
        for fold_name, ep_lo, ep_hi in CV5_FOLDS:
            r = run_one(
                name=f"cv5_{label}_{fold_name}",
                initial=scale,
                overrides=spec["overrides"],
                variant=spec["variant"],
                abs_thresh=spec["abs_thresh"],
                output_root=out_root,
                epoch_start=ep_lo,
                epoch_end=ep_hi,
                use_extended=False,
            )
            r["candidate"] = label
            r["scale"] = scale
            r["fold"] = fold_name
            results["cv5"].append(r)
            print(
                f"  CV5 {label:<35s} init={scale:>4.0f} {fold_name}: "
                f"bets={r['bets']:>4d} pnl={r['net_pnl_bnb']:+.4f} "
                f"breaker={r['breaker_fires']:>3d} cd={r['cooldown_fires']:>5d}",
                flush=True,
            )
            json_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    results["wallclock_s"] = elapsed
    json_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n=== Total: {elapsed:.1f}s. Results: {json_out} ===\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
