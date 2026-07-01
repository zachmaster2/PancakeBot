"""Full Phase-0 gauntlet on the 2026-06-30 recent positive signal.

Stress-tests whether the last-1-2-week WR bump (last_2w 61.2%, last_1w
64.4%) is real edge or noise, using the same discipline that ruled out
every prior candidate. ONE shared canonical-bet stream (risk-free, cutoff=2,
lookbacks 3/7/15) with rich per-fire metadata feeds all tests, so every
number is mutually consistent.

Tests (dispatch 2026-06-30):
  1. CV5            golden era split into 5 contiguous folds (the validated
                   +EV baseline) — does recent resemble the validated folds?
  2. holdout       frozen holdout 474880..475311 reference baseline.
  3. ext_v2        475312..479952 (the last p=0.002 finding) — direction /
                   regime / pool comparison to recent.
  4. permutation   N>=1000 structural null (outcomes+payouts shuffled, sides
                   fixed) on last_2w (n=67) and last_1w (n=45): null mean +
                   distribution + p_upper.
  5. Sidak/Bonf    over all windows + wires examined: raw p and adjusted p.
  6. consistency   does recent direction/mechanism agree with the positive
                   cohorts (golden/ext_v2/fresh_oos)?
  7. mechanism     what changed: momentum magnitude at fire, cutoff-pool
                   size, bull/bear balance, time-of-day / day-of-week —
                   recent vs dead vs golden.
  8. OOS           hold out the most recent 3 days; validate the signal
                   persists on data outside the discovery window.

Settlement: flat stake, realized final-pool payouts, 3% fee, no gas.
Output: var/strategy_review/monitor_runs/2026-06-30/gauntlet/{findings.json}.

Run: cd <repo> && .venv/Scripts/python.exe research/gauntlet_recent_signal_2026_06_30.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.in_process_runner as ipr  # noqa: E402
from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.constants import BNB_WEI  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import (  # noqa: E402
    MomentumOnlyPipeline,
    _pools_from_bets,
)

OUT = REPO / "var" / "strategy_review" / "monitor_runs" / "2026-06-30" / "gauntlet"
CUTOFF = 2
LOOKBACKS = (3, 7, 15)
FEE = 0.03
SEED = 20260630
N_PERM = 10_000

# discovery max lock 2026-06-30 14:35Z, max epoch 494321
LAST_2W = (490379, 494321)
LAST_1W = (492350, 494321)
OOS_3D_START = 493457   # ~2026-06-27 14:35Z (max_lock - 3d); refined at runtime

GOLDEN = (437562, 474086)
COHORTS = [
    ("golden_f1", 437562, 444866), ("golden_f2", 444867, 452171),
    ("golden_f3", 452172, 459475), ("golden_f4", 459476, 466780),
    ("golden_f5", 466781, 474086),
    ("holdout", 474880, 475311), ("ext_v2", 475312, 479952),
    ("fresh_oos", 479953, 483191), ("post_fresh", 483192, 484408),
    ("dead_latest", 484409, 487686), ("dead_vmlive", 487687, 488832),
    ("gap_pre2w", 488833, 490378),
    ("last_2w_excl_1w", 490379, 492349), ("last_1w", 492350, 494321),
]


def build_bets():
    """Canonical gate (risk OFF) -> per-fire records with metadata."""
    rounds = [r for r in ipr._load_all_rounds(use_extended_data=False)
              if r.position in ("Bull", "Bear")]
    rounds.sort(key=lambda r: r.epoch)
    max_lb = max(LOOKBACKS)
    sliced = {}
    raw_btc = {}
    for sym, path in (("btc", ipr._BTC_KLINES_PATH), ("eth", ipr._ETH_KLINES_PATH),
                      ("sol", ipr._SOL_KLINES_PATH)):
        uni = ipr._load_klines_unified(
            path, earliest_offset=CUTOFF + max_lb + 1, latest_offset=CUTOFF + 1)
        sliced[sym] = {ep: ipr._slice_per_entry(
            kl, kline_cutoff_seconds=CUTOFF, max_lookback=max_lb,
            earliest_offset=CUTOFF + max_lb + 1) for ep, kl in uni.items()}
        if sym == "btc":
            raw_btc = sliced[sym]
    sc = load_strategy_config_from_dict({})
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT", kline_cutoff_seconds=CUTOFF,
        mtf_lookbacks=sc.gate.mtf_lookbacks,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold)
    pipe = MomentumOnlyPipeline(
        config=gate_cfg, strategy_config=sc, gate=None, kline_cutoff_seconds=CUTOFF,
        pool_cutoff_seconds=6, min_bet_amount_bnb=0.001, treasury_fee_fraction=FEE,
        bankroll_tracker=None)
    pipe.refresh_btc_klines(btc_klines_by_epoch=sliced["btc"])
    pipe.refresh_eth_klines(eth_klines_by_epoch=sliced["eth"])
    pipe.refresh_sol_klines(sol_klines_by_epoch=sliced["sol"])
    pipe.refresh_bnb_klines(bnb_klines_by_epoch={})

    bets = []
    for r in rounds:
        d = pipe.decide_open_round(round_t=r)
        if d.action != "BET":
            continue
        pools = compute_pool_amounts_wei(bets=r.bets)
        fb = pools.bull_wei / BNB_WEI
        fbe = pools.bear_wei / BNB_WEI
        if fb <= 0 or fbe <= 0:
            continue
        tot = fb + fbe
        bull = d.bet_side == "Bull"
        outcome_bull = r.position == "Bull"
        pay = (tot * (1 - FEE) / fb) if bull else (tot * (1 - FEE) / fbe)
        win = bull == outcome_bull
        lock = int(r.lock_at)
        # cutoff-pool total (what the strategy sees at lock-6s)
        cb, cbe = _pools_from_bets(r, lock - 6)
        # BTC momentum magnitude (min|r| across lookbacks) at fire
        entry = raw_btc.get(int(r.epoch))
        mom = None
        if entry and len(entry) >= max_lb + 1:
            closes = np.array([float(k[4]) for k in entry])
            mom = float(min(abs(closes[-1] / closes[-1 - lb] - 1.0) for lb in LOOKBACKS))
        bets.append(dict(
            epoch=int(r.epoch), lock=lock, side_bull=bull, outcome_bull=outcome_bull,
            payout_bull=tot * (1 - FEE) / fb, payout_bear=tot * (1 - FEE) / fbe,
            win=win, pnl=(pay - 1.0) if win else -1.0,
            cutoff_pool=float(cb + cbe), final_pool=float(tot), mom=mom,
            hour=time.gmtime(lock).tm_hour, dow=time.gmtime(lock).tm_wday))
    return bets


def perm(bets, n_iter=N_PERM, seed=SEED):
    if len(bets) < 5:
        return dict(n=len(bets))
    obs = float(np.mean([b["pnl"] for b in bets]))
    out = np.array([b["outcome_bull"] for b in bets])
    pb = np.array([b["payout_bull"] for b in bets])
    pr = np.array([b["payout_bear"] for b in bets])
    side = np.array([b["side_bull"] for b in bets])
    rng = np.random.default_rng(seed)
    null = np.empty(n_iter)
    for i in range(n_iter):
        p = rng.permutation(len(out))
        pay = np.where(side, pb[p], pr[p])
        null[i] = np.where(out[p] == side, pay - 1.0, -1.0).mean()
    return dict(
        n=len(bets), wr=round(float(np.mean([b["win"] for b in bets])), 4),
        obs_mean_pnl=round(obs, 4), null_mean=round(float(null.mean()), 4),
        null_std=round(float(null.std()), 4),
        null_p05=round(float(np.percentile(null, 5)), 4),
        null_p95=round(float(np.percentile(null, 95)), 4),
        deficit_vs_null=round(obs - float(null.mean()), 4),
        p_upper=round(float((null >= obs).mean()), 5),
        bull_frac=round(float(np.mean(side)), 3))


def desc(bets):
    if not bets:
        return dict(n=0)
    moms = [b["mom"] for b in bets if b["mom"] is not None]
    return dict(
        n=len(bets), wr=round(float(np.mean([b["win"] for b in bets])), 4),
        bull_bet_frac=round(float(np.mean([b["side_bull"] for b in bets])), 3),
        bull_outcome_frac=round(float(np.mean([b["outcome_bull"] for b in bets])), 3),
        median_cutoff_pool=round(float(np.median([b["cutoff_pool"] for b in bets])), 3),
        median_final_pool=round(float(np.median([b["final_pool"] for b in bets])), 3),
        median_mom=round(float(np.median(moms)), 6) if moms else None,
        hour_hist={h: sum(1 for b in bets if b["hour"] == h) for h in range(0, 24, 6)},
        dow_hist={d: sum(1 for b in bets if b["dow"] == d) for d in range(7)})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("--- building canonical bet stream + metadata ---", flush=True)
    bets = build_bets()
    by = lambda lo, hi: [b for b in bets if lo <= b["epoch"] <= hi]
    print(f"  {len(bets)} canonical bets total", flush=True)

    # refine OOS 3-day boundary from actual locks
    max_lock = max(b["lock"] for b in bets)
    oos_start_ep = min((b["epoch"] for b in bets if b["lock"] >= max_lock - 3 * 86400),
                       default=OOS_3D_START)

    findings = {}

    # 1-3,6: cohort backbone (CV5 folds + holdout + ext_v2 + ... + recent)
    print("--- cohort backbone (CV5 / holdout / ext_v2 / ... / recent) ---", flush=True)
    cohort_tbl = {}
    for name, lo, hi in COHORTS:
        cohort_tbl[name] = perm(by(lo, hi))
    findings["cohorts"] = cohort_tbl

    # 4: permutation on last_2w + last_1w (full union windows)
    print("--- permutation: last_2w + last_1w ---", flush=True)
    windows = {"last_2w": perm(by(*LAST_2W)), "last_1w": perm(by(*LAST_1W))}
    findings["windows"] = windows

    # 5: Sidak/Bonferroni over examined cells (cohorts with n>=20 + 2 windows + 3 wires)
    examinable = [c for c in cohort_tbl.values() if c.get("n", 0) >= 20 and "p_upper" in c]
    n_examined = len(examinable) + 2 + 3   # + last_2w/last_1w + T1/T2/T3
    best_recent_p = min(windows["last_2w"]["p_upper"], windows["last_1w"]["p_upper"])
    findings["multiple_comparisons"] = dict(
        n_examined=n_examined, best_recent_raw_p=best_recent_p,
        sidak=round(1 - (1 - best_recent_p) ** n_examined, 4),
        bonferroni=round(min(1.0, best_recent_p * n_examined), 4))

    # 7: mechanism — recent vs dead vs golden
    print("--- mechanism diagnostic ---", flush=True)
    findings["mechanism"] = dict(
        golden=desc(by(*GOLDEN)),
        dead=desc(by(484409, 488832)),
        recent_2w=desc(by(*LAST_2W)),
        recent_1w=desc(by(*LAST_1W)))

    # 8: OOS — discovery (last_2w minus last 3d) vs OOS (last 3d)
    print(f"--- OOS: hold out last 3 days (epoch >= {oos_start_ep}) ---", flush=True)
    disc = [b for b in bets if LAST_2W[0] <= b["epoch"] < oos_start_ep]
    oos = [b for b in bets if oos_start_ep <= b["epoch"] <= LAST_2W[1]]
    findings["oos"] = dict(oos_start_epoch=oos_start_ep,
                           discovery_2w_minus_3d=perm(disc), oos_last_3d=perm(oos))

    (OUT / "findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    # ---- console digest ----
    print("\n=== COHORT BACKBONE (flat-stake, risk-free) ===")
    print(f"{'cohort':>16} {'n':>5} {'WR':>7} {'deficit':>8} {'p_up':>6} {'bull%':>6}")
    for name, _, _ in COHORTS:
        c = cohort_tbl[name]
        if c.get("n", 0) < 5:
            print(f"{name:>16} {c.get('n',0):>5}  (sparse)"); continue
        print(f"{name:>16} {c['n']:>5} {c['wr']:>7.4f} {c['deficit_vs_null']:>+8.4f} "
              f"{c['p_upper']:>6.3f} {c['bull_frac']*100:>5.1f}%")
    print("\n=== RECENT WINDOWS (permutation N=10k) ===")
    for w in ("last_2w", "last_1w"):
        x = windows[w]
        print(f"  {w}: n={x['n']} WR={x['wr']} obs={x['obs_mean_pnl']} "
              f"null={x['null_mean']}±{x['null_std']} deficit={x['deficit_vs_null']} "
              f"p_up={x['p_upper']}")
    mc = findings["multiple_comparisons"]
    print(f"\n=== MULTI-COMPARISONS: best recent raw p={mc['best_recent_raw_p']} over "
          f"{mc['n_examined']} examined -> Sidak {mc['sidak']} / Bonf {mc['bonferroni']}")
    print("\n=== OOS ===")
    o = findings["oos"]
    for k in ("discovery_2w_minus_3d", "oos_last_3d"):
        x = o[k]
        print(f"  {k}: n={x.get('n')} WR={x.get('wr')} deficit={x.get('deficit_vs_null')} "
              f"p_up={x.get('p_upper')}")
    print("\n=== MECHANISM (recent vs dead vs golden) ===")
    for k in ("golden", "dead", "recent_2w", "recent_1w"):
        m = findings["mechanism"][k]
        print(f"  {k:>10}: n={m['n']} WR={m.get('wr')} bullBet={m.get('bull_bet_frac')} "
              f"medPool={m.get('median_cutoff_pool')} medMom={m.get('median_mom')}")
    print(f"\n[done] {time.time()-t0:.0f}s; artifacts -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
