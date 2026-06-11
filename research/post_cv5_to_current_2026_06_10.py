"""Post-CV5 -> current backtest, 2026-06-10 sync (adds the 'vm_live_era' cohort).

Reuses the tested driver in
``research/post_cv5_to_current_step10b_2026_05_26.py`` (canonical cutoff=2,
lookbacks (3,7,15); BOTH 5 BNB and 50 BNB scales; per-cohort
n_bets/WR/PnL/mean_bet/bet_rate/max_dd + skip-reason distribution), exactly
like ``research/post_cv5_to_current_2026_06_06.py`` did for its sync.

The 2026-06-10 sync brought the dataset from 487718 to current. The 06-06
run's open-ended ``latest`` cohort (>= 484409) is bounded at its actual
backtested end (487686), and everything after splits into:
  latest      : 484409..487686  (the 06-06 run's slice, now bounded)
  vm_live_era : 487687..current (never backtested; the Frankfurt-VM live
                era — off350 broadcast-lead fix 06-08, Era 12b single-source
                read path + systemd-direct cutover 06-10 all live here)

so the freshest, live-traded slice is isolated to test whether the edge
persists.

Run:  cd <repo> && .venv/Scripts/python.exe research/post_cv5_to_current_2026_06_10.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.post_cv5_to_current_step10b_2026_05_26 as step10b  # noqa: E402

# Bound the 06-06 run's open-ended `latest` at its actual backtested end and
# isolate the rounds this sync added. The driver's cohort_of() / run_post_cv5()
# read these module globals at call time, so reassigning them re-buckets cleanly.
step10b.COHORT_DEFS = [
    ("gap_post_cv5_pre_holdout", 474087, 474879),
    ("holdout", 474880, 475311),
    ("ext_v2", 475312, 479952),
    ("fresh_oos", 479953, 483191),
    ("post_fresh", 483192, 484408),
    ("latest", 484409, 487686),
    ("vm_live_era", 487687, 999999),  # everything the 2026-06-10 sync added
]
step10b.COHORT_ORDER = [c[0] for c in step10b.COHORT_DEFS]


if __name__ == "__main__":
    step10b.main()
