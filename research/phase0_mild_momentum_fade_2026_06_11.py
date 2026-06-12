"""Phase-0: mild-momentum FADE — the single pre-registered C4 follow-up.

Pre-registration (pinned BEFORE this run, from
research/phase0_candidates_runtime_2026_06_11_findings.md):
  - Band: BTC tri-agreement (lookbacks 3/7/15 same sign) with
    min|r| in [2e-5, 5e-5)  — the C4 cell that showed -0.108/bet (z=-2.0)
    in the dead era when betting WITH the impulse.
  - Hypothesis: betting AGAINST the impulse (fade) in this band is +EV in
    the dead era.
  - Discovery data: the ENTIRE dead era was used in the C4 sweep, so there
    is no unburned holdout. This test therefore reports:
      (1) the multiple-comparisons-adjusted significance of the discovery
          itself (15 cells examined; Sidak adjustment),
      (2) cross-era sign consistency (golden/fade eras were NOT part of
          the discovery direction — a real reversal mechanism should not
          be wildly sign-inconsistent),
      (3) a within-dead split (latest 484409..487686 vs vm_live_era
          487687+) as a CONSISTENCY check, explicitly NOT a holdout,
      (4) a permutation null on the dead-era fade PnL.
  - Runtime feasibility: TRIVIAL (identical data + path to the canonical
    strategy; the band test is one comparison on already-computed values).

Settlement: flat stake 1, realized final-pool payouts, 3% fee, no gas.

Outputs: var/strategy_review/phase0_fade_2026_06_11/findings.json.
Findings doc: research/phase0_wallet_fade_2026_06_11_findings.md (shared).

Run:  cd <repo> && .venv/Scripts/python.exe research/phase0_mild_momentum_fade_2026_06_11.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.in_process_runner as ipr  # noqa: E402
from pancakebot.constants import BNB_WEI  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402

OUT = REPO / "var" / "strategy_review" / "phase0_fade_2026_06_11"
CUTOFF = 2
LOOKBACKS = (3, 7, 15)
FEE = 0.03
BAND_LO, BAND_HI = 2e-5, 5e-5          # pre-registered
N_CELLS_EXAMINED = 15                  # C4 sweep: 5 bands x 3 eras
RAW_Z = -2.04                          # discovery cell

SLICES = [
    ("golden", 437562, 479952),
    ("fade_era", 479953, 484408),
    ("dead_all", 484409, 488832),
    ("dead_latest", 484409, 487686),   # within-dead consistency split
    ("dead_vmlive", 487687, 488832),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("--- loading ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=False)
    all_rounds = [r for r in all_rounds if r.position in ("Bull", "Bear")
                  and r.epoch >= 437562]
    max_lb = max(LOOKBACKS)
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH, earliest_offset=CUTOFF + max_lb + 1,
        latest_offset=CUTOFF + 1)
    btc = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CUTOFF,
                                    max_lookback=max_lb,
                                    earliest_offset=CUTOFF + max_lb + 1)
           for ep, kl in btc.items()}

    # band rounds + fade settlement
    fade_rows = []
    for r in all_rounds:
        entry = btc.get(int(r.epoch))
        if not entry or len(entry) < max_lb + 1:
            continue
        closes = np.array([float(k[4]) for k in entry])
        rets = {lb: float(closes[-1] / closes[-1 - lb] - 1.0) for lb in LOOKBACKS}
        signs = {np.sign(v) for v in rets.values()}
        if len(signs) != 1 or np.sign(rets[3]) == 0:
            continue
        minabs = min(abs(v) for v in rets.values())
        if not (BAND_LO <= minabs < BAND_HI):
            continue
        pools = compute_pool_amounts_wei(bets=r.bets)
        f_bull = pools.bull_wei / BNB_WEI
        f_bear = pools.bear_wei / BNB_WEI
        if f_bull <= 0 or f_bear <= 0:
            continue
        impulse_bull = rets[3] > 0
        fade_bull = not impulse_bull           # the pre-registered FADE side
        win = (r.position == "Bull") == fade_bull
        pay = ((f_bull + f_bear) * (1 - FEE) / f_bull if fade_bull
               else (f_bull + f_bear) * (1 - FEE) / f_bear)
        fade_rows.append(dict(
            epoch=int(r.epoch), win=win, pnl=(pay - 1.0) if win else -1.0,
            outcome_bull=r.position == "Bull", fade_bull=fade_bull,
            payout_bull=(f_bull + f_bear) * (1 - FEE) / f_bull,
            payout_bear=(f_bull + f_bear) * (1 - FEE) / f_bear,
        ))

    # per-slice stats
    res = {}
    for name, lo, hi in SLICES:
        sub = [x for x in fade_rows if lo <= x["epoch"] <= hi]
        n = len(sub)
        if n == 0:
            res[name] = dict(n=0)
            continue
        pnls = [x["pnl"] for x in sub]
        mean = float(np.mean(pnls))
        se = float(np.std(pnls) / np.sqrt(n)) if n > 1 else 0.0
        res[name] = dict(n=n, wr=round(float(np.mean([x["win"] for x in sub])), 4),
                         mean_pnl=round(mean, 4),
                         z=round(mean / se, 2) if se > 0 else None)

    # multiple-comparisons adjustment of the DISCOVERY
    from math import erf, sqrt
    p_raw = 2 * (1 - 0.5 * (1 + erf(abs(RAW_Z) / sqrt(2))))   # two-sided
    p_sidak = 1 - (1 - p_raw) ** N_CELLS_EXAMINED
    adjustment = dict(raw_z=RAW_Z, p_raw=round(p_raw, 4),
                      n_cells=N_CELLS_EXAMINED, p_sidak=round(p_sidak, 4),
                      note="discovery data = ENTIRE dead era (burned); no "
                           "unburned holdout exists")

    # permutation null on dead-era fade PnL
    sub = [x for x in fade_rows if 484409 <= x["epoch"] <= 488832]
    rng = random.Random(20260611)
    obs = float(np.mean([x["pnl"] for x in sub])) if sub else None
    perm = None
    if sub and len(sub) >= 20:
        outs = [(x["outcome_bull"], x["payout_bull"], x["payout_bear"]) for x in sub]
        sides = [x["fade_bull"] for x in sub]
        null = []
        for _ in range(1000):
            sh = outs[:]
            rng.shuffle(sh)
            tot = 0.0
            for (ob, pb, pr), side in zip(sh, sides):
                win = ob == side
                pay = pb if side else pr
                tot += (pay - 1.0) if win else -1.0
            null.append(tot / len(sides))
        p_upper = sum(1 for x in null if x >= obs) / len(null)
        perm = dict(n=len(sub), obs_mean_pnl=round(obs, 4),
                    p_upper=round(p_upper, 4))

    findings = dict(band=[BAND_LO, BAND_HI], slices=res,
                    discovery_adjustment=adjustment, permutation_dead=perm)
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2),
                                       encoding="utf-8")
    print(json.dumps(findings, indent=1))
    print(f"\n[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
