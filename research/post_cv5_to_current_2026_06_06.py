"""Post-CV5 -> current backtest, 2026-06-06 expanded sync (adds the 'latest' cohort).

Reuses the tested driver in
``research/post_cv5_to_current_step10b_2026_05_26.py`` (canonical cutoff=2,
lookbacks (3,7,15); BOTH 5 BNB and 50 BNB scales; per-cohort
n_bets/WR/PnL/mean_bet/bet_rate/max_dd + skip-reason distribution). The only
change: the 2026-06-06 sync brought the dataset from 484408 to ~487686, so
step10b's open-ended ``post_fresh`` (>= 483192) is SPLIT into:
  post_fresh : 483192..484408  (step10b's prior newest)
  latest     : 484409..current (the ~3300 rounds NEVER backtested; live-traded)

so the freshest, live-traded slice is isolated to test whether the edge persists.

Run:  cd <repo> && .venv/Scripts/python.exe research/post_cv5_to_current_2026_06_06.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.post_cv5_to_current_step10b_2026_05_26 as step10b  # noqa: E402

# Split the open-ended post_fresh into a bounded post_fresh + a fresh `latest`
# cohort (the rounds this sync added). The driver's cohort_of() / run_post_cv5()
# read these module globals at call time, so reassigning them re-buckets cleanly.
step10b.COHORT_DEFS = [
    ("gap_post_cv5_pre_holdout", 474087, 474879),
    ("holdout", 474880, 475311),
    ("ext_v2", 475312, 479952),
    ("fresh_oos", 479953, 483191),
    ("post_fresh", 483192, 484408),
    ("latest", 484409, 999999),  # everything the 2026-06-06 sync added
]
step10b.COHORT_ORDER = [c[0] for c in step10b.COHORT_DEFS]


if __name__ == "__main__":
    step10b.main()
