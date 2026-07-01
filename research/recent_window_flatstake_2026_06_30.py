"""Flat-stake + permutation significance for the 2026-06-30 recent windows.

The 2026-06-30 standard backtest showed the canonical gate turning POSITIVE
in the last 1-2 weeks (last_2w WR 61.2%, last_1w WR 64.4%) — materially
different from the 2026-06-17 null run. This quantifies whether that is
statistically real vs the structural permutation null, on the SAME flat-
stake risk-free signal stream the monitor's T1 wire uses (so it is directly
comparable and not inflated by sizing).

Reuses build_canonical_bets + window_stats from
research/recent_window_flatstake_2026_06_17.py (canonical gate, cutoff=2,
lookbacks 3/7/15, risk OFF). Windows:
  last_2w : epochs 490379..494321  (trailing 14 days)
  last_1w : epochs 492350..494321  (trailing  7 days)

Run:  cd <repo> && .venv/Scripts/python.exe research/recent_window_flatstake_2026_06_30.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.recent_window_flatstake_2026_06_17 import (  # noqa: E402
    build_canonical_bets,
    window_stats,
)

OUT = REPO / "var" / "strategy_review" / "monitor_runs" / "2026-06-30"
WINDOWS = [("last_2w", 490379, 494321), ("last_1w", 492350, 494321)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("--- building canonical bets (risk-free signal stream) ---", flush=True)
    bets = build_canonical_bets()
    results = {}
    for name, e0, e1 in WINDOWS:
        sub = [b for b in bets if e0 <= b["epoch"] <= e1]
        results[name] = window_stats(sub, name)

    (OUT / "recent_window_flatstake.json").write_text(
        json.dumps(dict(windows=results,
                        total_canonical_bets=len(bets),
                        run_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime())),
                   indent=2), encoding="utf-8")

    print("\n=== flat-stake canonical-gate windows (risk-free) — 2026-06-30 ===")
    print(f"{'window':>10} {'n':>4} {'gateWR':>7} {'flat/bet':>9} {'PnL@0.001':>10} "
          f"{'deficit':>8} {'nullMean':>9} {'p_up':>6} {'p_lo':>6}")
    for name, _, _ in WINDOWS:
        r = results[name]
        if r.get("n_bets", 0) == 0:
            print(f"{name:>10}  no bets"); continue
        print(f"{name:>10} {r['n_bets']:>4} {r['gate_wr']:>7.4f} "
              f"{r['flat_mean_pnl_per_bet']:>+9.4f} {r['flat_pnl_at_0p001_bnb']:>+10.5f} "
              f"{r['deficit_vs_null']:>+8.4f} {r['null_mean']:>+9.4f} "
              f"{r['p_upper']:>6.3f} {r['p_lower']:>6.3f}")
    print(f"\n[done] {time.time()-t0:.0f}s; artifacts -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
