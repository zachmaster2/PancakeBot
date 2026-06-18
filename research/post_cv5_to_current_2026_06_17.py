"""Post-CV5 -> current backtest, 2026-06-17 sync (wait-and-monitor cadence).

Reuses the tested driver in
``research/post_cv5_to_current_step10b_2026_05_26.py`` (canonical cutoff=2,
lookbacks (3,7,15); BOTH 5 BNB and 50 BNB dynamic scales with risk gates;
per-cohort n_bets/WR/PnL/mean_bet/bet_rate/max_dd + skip distribution),
exactly like the 06-06 and 06-10 dated wrappers did for their syncs.

The 2026-06-17 sync brought the dataset from 488832 to 490743 (+1911
rounds, ~6.6 days). The 06-10 run's open-ended ``vm_live_era`` (>= 487687)
is split so the previously-backtested portion (the OKX-probe ``dead_vmlive``
cohort, 487687..488832) stays a clean comparison baseline and the freshly
synced rounds are isolated:
  latest        : 484409..487686  (bounded)
  dead_vmlive   : 487687..488832  (the 06-10 sync's vm_live tail = the OKX
                  probe's dead_vmlive; the last-run comparison baseline)
  synced_new    : 488833..490743  (the 1911 rounds THIS sync added)

The trailing-14-day ``recent_2w`` window (which overlaps these disjoint
cohorts) is analysed separately at flat stake + permutation in
``research/recent_window_flatstake_2026_06_17.py``.

Run:  cd <repo> && .venv/Scripts/python.exe research/post_cv5_to_current_2026_06_17.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.post_cv5_to_current_step10b_2026_05_26 as step10b  # noqa: E402

step10b.COHORT_DEFS = [
    ("gap_post_cv5_pre_holdout", 474087, 474879),
    ("holdout", 474880, 475311),
    ("ext_v2", 475312, 479952),
    ("fresh_oos", 479953, 483191),
    ("post_fresh", 483192, 484408),
    ("latest", 484409, 487686),
    ("dead_vmlive", 487687, 488832),   # OKX-probe dead_vmlive = last-run baseline
    ("synced_new", 488833, 999999),    # the 1911 rounds this 06-17 sync added
]
step10b.COHORT_ORDER = [c[0] for c in step10b.COHORT_DEFS]


if __name__ == "__main__":
    step10b.main()
