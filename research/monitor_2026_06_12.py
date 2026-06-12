"""Offline regime monitor — the wait-and-monitor posture (2026-06-12).

UNLIKE other research/ scripts this is a LIVING ops script, expected to be
re-run on a cadence (weekly recommended, monthly minimum — OKX tape
retention ~3 months bounds the maximum gap). It detects a regime turn that
would justify revisiting the paused strategy machinery. Docs:
docs/monitoring.md.

What it computes (newest synced data; run `python run.py --sync` first,
or pass --sync to have this script invoke it):
  T1  Canonical gate, latest 500 risk-free bets: WR + flat-stake PnL vs
      the payout-weighted breakeven, against a Bernoulli(1/payout)
      market-implied null (the exact "no edge" hypothesis in a
      pari-mutuel).  TRIP: p < 0.01.
  T2  Contrarian @ lock-6s (threshold 0.4, the golden-era best), trailing
      15 days: PnL deficit vs permutation null.  TRIP: permutation
      z > 2 AND n > 500.
  T3  Perp tape imbalance (BNB-USDT-SWAP, 1m/5m/15m x cutoff 2s/6s),
      trailing 15 days: deficit vs permutation null per cell, Sidak over
      the 6 cells.  TRIP: any Sidak-adjusted two-sided p < 0.01.
      (The tape file is extended incrementally before computing: walk
      newest->back until overlapping the file's oldest contiguous
      segment, which also heals holes left by interrupted runs.)

Pre-registered: thresholds, windows, and statistics above are PINNED.  A
tripped wire is NOT a deploy signal — it is a "rerun the full Phase-0
gauntlet on fresh data" signal (CV-style validation, holdout, sweep
discounts all still apply; see the 2026-06 probes for how often raw
p<0.01 cells dissolve).

Tripwire silence is also informative: each quiet run extends the
no-regime-turn record.

Output: var/strategy_review/monitor_runs/<YYYY-MM-DD>/findings.json +
summary.txt + console digest.

Run:  cd <repo> && .venv/Scripts/python.exe research/monitor_2026_06_12.py [--sync] [--no-fetch]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

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
from research.phase0_okx_perp_2026_06_11 import (  # noqa: E402
    load_tape,
    sign_strategy,
    tape_imbalance,
)

EXT = REPO / "var" / "extended"
PERP_TAPE = EXT / "okx_swap_trades_BNB-USDT-SWAP.jsonl"
OUT_ROOT = REPO / "var" / "strategy_review" / "monitor_runs"
FEE = 0.03
CUTOFF = 2
LOOKBACKS = (3, 7, 15)
SEED = 20260612

# --- pre-registered monitor parameters (PINNED 2026-06-12; do not tune) ---
T1_N_BETS = 500          # latest canonical bets
T1_P_TRIP = 0.01
T2_TRAIL_DAYS = 15
T2_THRESHOLD = 0.4       # golden-era best contrarian threshold
T2_Z_TRIP = 2.0
T2_MIN_N = 500
T3_TRAIL_DAYS = 15
T3_WINDOWS_S = (60, 300, 900)
T3_CUTOFFS_S = (2, 6)
T3_P_SIDAK_TRIP = 0.01
N_PERM = 10_000


# ---------------------------------------------------------------------------
# tape extension: walk newest -> back until overlapping the oldest
# contiguous segment (heals interruption holes by construction)
# ---------------------------------------------------------------------------

def extend_perp_tape() -> dict:
    tape = load_tape([PERP_TAPE])
    if tape is None:
        return dict(error="tape file missing; run the phase0 capture first")
    target_s = tape["segments"][0][1]   # hi of the OLDEST contiguous segment
    sess = requests.Session()
    sess.headers["User-Agent"] = "monitor/1.0"
    after = None
    added = 0
    t0 = time.time()
    with open(PERP_TAPE, "a", encoding="utf-8") as f:
        while True:
            params = {"instId": "BNB-USDT-SWAP", "limit": "100", "type": "1"}
            if after:
                params["after"] = after
            d = []
            for attempt in range(5):
                try:
                    r = sess.get("https://www.okx.com/api/v5/market/history-trades",
                                 params=params, timeout=15)
                    j = r.json()
                    if j.get("code") == "0":
                        d = j.get("data", [])
                        break
                except Exception:
                    pass
                time.sleep(1.0 + attempt)
            if not d:
                return dict(error="empty page before overlap — tape NOT "
                                  "contiguous; rerun, or re-capture if the "
                                  "retention floor passed the gap",
                            added=added)
            for rec in d:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            added += len(d)
            after = d[-1]["tradeId"]
            oldest = int(d[-1]["ts"]) / 1000.0
            if oldest <= target_s:
                break
            time.sleep(0.25)
    return dict(added=added, walk_s=round(time.time() - t0),
                reached=time.strftime("%Y-%m-%d %H:%M", time.gmtime(oldest)))


# ---------------------------------------------------------------------------
# shared loading
# ---------------------------------------------------------------------------

def load_rounds_with_payouts():
    rounds = ipr._load_all_rounds(use_extended_data=False)
    rows = []
    for r in rounds:
        if r.position not in ("Bull", "Bear"):
            continue
        pools = compute_pool_amounts_wei(bets=r.bets)
        f_bull = pools.bull_wei / BNB_WEI
        f_bear = pools.bear_wei / BNB_WEI
        if f_bull <= 0 or f_bear <= 0:
            continue
        tot = f_bull + f_bear
        rows.append(dict(
            epoch=int(r.epoch), lock=int(r.lock_at), round_t=r,
            outcome_bull=r.position == "Bull",
            payout_bull=tot * (1 - FEE) / f_bull,
            payout_bear=tot * (1 - FEE) / f_bear,
        ))
    rows.sort(key=lambda x: x["epoch"])
    return rows


def perm_null_stats(bets, n_iter=N_PERM, seed=SEED) -> dict:
    """Per-round outcome-shuffle null -> mean/std/p (sides fixed)."""
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
    mu, sd = float(null.mean()), float(null.std())
    return dict(
        n=len(bets), obs_mean_pnl=round(obs, 4), null_mean=round(mu, 4),
        deficit_vs_null=round(obs - mu, 4),
        perm_z=round((obs - mu) / sd, 2) if sd > 0 else None,
        p_upper=round(float((null >= obs).mean()), 5),
        p_lower=round(float((null <= obs).mean()), 5))


# ---------------------------------------------------------------------------
# T1: canonical latest-500 vs market-implied null
# ---------------------------------------------------------------------------

def t1_canonical(rows) -> dict:
    strategy_cfg = load_strategy_config_from_dict({})
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CUTOFF,
        mtf_lookbacks=strategy_cfg.gate.mtf_lookbacks,
        mtf_min_return_threshold=strategy_cfg.gate.mtf_min_return_threshold,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_cfg, strategy_config=strategy_cfg, gate=None,
        kline_cutoff_seconds=CUTOFF, pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001, treasury_fee_fraction=FEE,
        bankroll_tracker=None,   # risk-free signal stream
    )
    max_lb = max(LOOKBACKS)
    sliced = {}
    for sym, path in (("btc", ipr._BTC_KLINES_PATH), ("eth", ipr._ETH_KLINES_PATH),
                      ("sol", ipr._SOL_KLINES_PATH)):
        uni = ipr._load_klines_unified(
            path, earliest_offset=CUTOFF + max_lb + 1, latest_offset=CUTOFF + 1)
        sliced[sym] = {
            ep: ipr._slice_per_entry(
                kl, kline_cutoff_seconds=CUTOFF, max_lookback=max_lb,
                earliest_offset=CUTOFF + max_lb + 1)
            for ep, kl in uni.items()}
    pipeline.refresh_btc_klines(btc_klines_by_epoch=sliced["btc"])
    pipeline.refresh_eth_klines(eth_klines_by_epoch=sliced["eth"])
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sliced["sol"])
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    bets = []
    for x in rows:
        d = pipeline.decide_open_round(round_t=x["round_t"])
        if d.action != "BET":
            continue
        bull = d.bet_side == "Bull"
        pay = x["payout_bull"] if bull else x["payout_bear"]
        win = bull == x["outcome_bull"]
        bets.append(dict(epoch=x["epoch"], win=win, payout=pay,
                         pnl=(pay - 1.0) if win else -1.0))
    latest = bets[-T1_N_BETS:]
    n = len(latest)
    if n < 200:
        return dict(skipped=f"only {n} canonical bets total; need >=200",
                    n=n, tripped=False)
    wr = float(np.mean([b["win"] for b in latest]))
    pnl = float(np.mean([b["pnl"] for b in latest]))
    pays = np.array([b["payout"] for b in latest])
    breakeven_wr = float(np.mean(1.0 / pays))
    # market-implied null: win_i ~ Bernoulli(1/payout_i)
    rng = np.random.default_rng(SEED)
    p_i = 1.0 / pays
    sims = rng.random((N_PERM, n)) < p_i
    null_pnl = np.where(sims, pays - 1.0, -1.0).mean(axis=1)
    p = float((null_pnl >= pnl).mean())
    return dict(n=n, epochs=[latest[0]["epoch"], latest[-1]["epoch"]],
                wr=round(wr, 4), breakeven_wr=round(breakeven_wr, 4),
                mean_pnl=round(pnl, 4),
                null_mean=round(float(null_pnl.mean()), 4),
                p_value=round(p, 5), tripped=bool(p < T1_P_TRIP))


# ---------------------------------------------------------------------------
# T2: contrarian @ lock-6s, trailing window
# ---------------------------------------------------------------------------

def t2_contrarian(rows, newest_lock: int) -> dict:
    lo = newest_lock - T2_TRAIL_DAYS * 86400
    bets = []
    for x in rows:
        if x["lock"] < lo:
            continue
        c_bull, c_bear = _pools_from_bets(x["round_t"], x["lock"] - 6)
        tot = c_bull + c_bear
        if tot <= 0:
            continue
        imb = (c_bull - c_bear) / tot
        if abs(imb) < T2_THRESHOLD:
            continue
        bull = imb < 0            # bet AGAINST the cutoff crowd
        pay = x["payout_bull"] if bull else x["payout_bear"]
        win = bull == x["outcome_bull"]
        bets.append(dict(epoch=x["epoch"], win=win, side_bull=bull,
                         pnl=(pay - 1.0) if win else -1.0,
                         outcome_bull=x["outcome_bull"],
                         payout_bull=x["payout_bull"],
                         payout_bear=x["payout_bear"]))
    if len(bets) < 50:
        return dict(skipped=f"n={len(bets)} < 50", n=len(bets), tripped=False)
    st = perm_null_stats(bets)
    st["wr"] = round(float(np.mean([b["win"] for b in bets])), 4)
    st["tripped"] = bool(st["perm_z"] is not None
                         and st["perm_z"] > T2_Z_TRIP and st["n"] > T2_MIN_N)
    return st


# ---------------------------------------------------------------------------
# T3: perp tape imbalance cells, trailing window
# ---------------------------------------------------------------------------

def t3_tape(rows, newest_lock: int) -> dict:
    tape = load_tape([PERP_TAPE])
    if tape is None:
        return dict(skipped="no perp tape", tripped=False)
    lo = newest_lock - T3_TRAIL_DAYS * 86400
    sub = [x for x in rows if x["lock"] >= lo]
    cells = {}
    best_p2 = None
    for w in T3_WINDOWS_S:
        for c in T3_CUTOFFS_S:
            key = f"imb_perp_{w}_{c}"
            for x in sub:
                x[key] = tape_imbalance(tape, x["lock"] - c, w)
            bets = sign_strategy(sub, key)
            if len(bets) < 50:
                cells[key] = dict(n=len(bets), skipped=True)
                continue
            st = perm_null_stats(bets)
            p2 = min(1.0, 2 * min(st["p_upper"], st["p_lower"]))
            st["p_two_sided"] = round(p2, 5)
            cells[key] = st
            if best_p2 is None or p2 < best_p2[1]:
                best_p2 = (key, p2)
    sidak = None
    tripped = False
    if best_p2:
        k = len([c for c in cells.values() if not c.get("skipped")])
        p_sidak = 1 - (1 - best_p2[1]) ** k
        sidak = dict(best_cell=best_p2[0], p_raw=round(best_p2[1], 5),
                     n_cells=k, p_sidak=round(p_sidak, 5))
        tripped = bool(p_sidak < T3_P_SIDAK_TRIP)
    return dict(cells=cells, sidak=sidak, tripped=tripped,
                tape_segments=[
                    (time.strftime("%Y-%m-%d %H:%M", time.gmtime(a)),
                     time.strftime("%Y-%m-%d %H:%M", time.gmtime(b)))
                    for a, b in tape["segments"]])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = set(sys.argv[1:])
    t0 = time.time()
    out_dir = OUT_ROOT / time.strftime("%Y-%m-%d", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)

    if "--sync" in args:
        print("--- syncing rounds (run.py --sync) ---", flush=True)
        rc = subprocess.run([sys.executable, str(REPO / "run.py"), "--sync"],
                            cwd=REPO).returncode
        if rc != 0:
            print(f"sync FAILED rc={rc}; continuing on existing store", flush=True)

    tape_ext = dict(skipped=True)
    if "--no-fetch" not in args:
        print("--- extending perp tape ---", flush=True)
        tape_ext = extend_perp_tape()
        print(f"  {tape_ext}", flush=True)

    print("--- loading rounds ---", flush=True)
    rows = load_rounds_with_payouts()
    newest = rows[-1]
    currency = dict(
        rounds=len(rows), newest_epoch=newest["epoch"],
        newest_lock_utc=time.strftime("%Y-%m-%d %H:%M",
                                      time.gmtime(newest["lock"])),
        stale_days=round((time.time() - newest["lock"]) / 86400, 1))
    print(f"  {currency}", flush=True)

    print("--- T1 canonical latest-500 ---", flush=True)
    t1 = t1_canonical(rows)
    print("--- T2 contrarian @-6s ---", flush=True)
    t2 = t2_contrarian(rows, newest["lock"])
    print("--- T3 perp tape cells ---", flush=True)
    t3 = t3_tape(rows, newest["lock"])

    tripped = [k for k, v in (("T1", t1), ("T2", t2), ("T3", t3))
               if v.get("tripped")]
    findings = dict(
        run_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        data_currency=currency, tape_extension=tape_ext,
        t1_canonical=t1, t2_contrarian=t2, t3_tape=t3,
        tripwires=dict(t1=t1.get("tripped", False),
                       t2=t2.get("tripped", False),
                       t3=t3.get("tripped", False)),
        verdict=("TRIPPED: " + ",".join(tripped)) if tripped else "quiet",
    )
    (out_dir / "findings.json").write_text(
        json.dumps(findings, indent=2, default=str), encoding="utf-8")

    lines = [
        f"monitor run {findings['run_at_utc']}Z — data through "
        f"{currency['newest_lock_utc']}Z ({currency['stale_days']}d old)",
        f"T1 canonical latest-{t1.get('n')}: WR={t1.get('wr')} vs "
        f"breakeven {t1.get('breakeven_wr')}, pnl={t1.get('mean_pnl')}, "
        f"p={t1.get('p_value')} -> {'TRIP' if t1.get('tripped') else 'quiet'}",
        f"T2 contrarian@-6s thr{T2_THRESHOLD} n={t2.get('n')}: "
        f"deficit={t2.get('deficit_vs_null')}, z={t2.get('perm_z')} "
        f"-> {'TRIP' if t2.get('tripped') else 'quiet'}",
        f"T3 tape best={t3.get('sidak', {}) and t3['sidak'].get('best_cell')}: "
        f"p_sidak={(t3.get('sidak') or {}).get('p_sidak')} "
        f"-> {'TRIP' if t3.get('tripped') else 'quiet'}",
        f"VERDICT: {findings['verdict']}",
    ]
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\n[done] {time.time()-t0:.0f}s; artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
