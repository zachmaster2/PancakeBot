"""Recent-window flat-stake + permutation analysis (2026-06-17 sync).

Companion to research/post_cv5_to_current_2026_06_17.py (which gives the
deployable 5/50 BNB dynamic-sizing cohort table). This script answers the
"past few weeks" focus at FLAT stake with a permutation null, reusing the
monitor's canonical-pipeline + permutation machinery so the numbers are
directly comparable to the monitor's T1 wire.

Windows analysed (all on the canonical gate, cutoff=2, lookbacks 3/7/15,
risk gates OFF = pure signal stream):
  dead_vmlive : epochs 487687..488832  (last run's baseline / OKX-probe cohort)
  synced_new  : epochs 488833..490743  (the 1911 rounds THIS sync added)
  recent_2w   : trailing 14 days of rounds (overlaps the above; the dispatch's
                explicit focus window)

For each: n canonical bets, gate WR, flat mean PnL per unit-stake bet,
flat PnL at 0.001 BNB stake, and a permutation p vs the STRUCTURAL null
(outcomes shuffled holding payouts — a sign-strategy's null expectation is
the fee + majority discount, not zero).

Settlement: flat stake, realized final-pool payouts, 3% fee, no gas.

Run:  cd <repo> && .venv/Scripts/python.exe research/recent_window_flatstake_2026_06_17.py
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
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402
from research.monitor_2026_06_12 import perm_null_stats  # noqa: E402

OUT = REPO / "var" / "strategy_review" / "monitor_runs" / "2026-06-17"
CUTOFF = 2
LOOKBACKS = (3, 7, 15)
FEE = 0.03
FLAT_STAKE_BNB = 0.001
TRAIL_DAYS = 14


def build_canonical_bets():
    """Replay the canonical gate (risk-free) -> per-bet records with payouts."""
    rounds = ipr._load_all_rounds(use_extended_data=False)
    rounds = [r for r in rounds if r.position in ("Bull", "Bear")]
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
        bets.append(dict(
            epoch=int(r.epoch), lock=int(r.lock_at), side_bull=bull,
            outcome_bull=outcome_bull, payout_bull=tot * (1 - FEE) / fb,
            payout_bear=tot * (1 - FEE) / fbe, win=win,
            pnl=(pay - 1.0) if win else -1.0))
    bets.sort(key=lambda b: b["epoch"])
    return bets


def window_stats(bets, name):
    n = len(bets)
    if n == 0:
        return dict(window=name, n_bets=0)
    wr = float(np.mean([b["win"] for b in bets]))
    unit = float(np.mean([b["pnl"] for b in bets]))
    out = dict(window=name, n_bets=n,
               epoch_range=[bets[0]["epoch"], bets[-1]["epoch"]],
               gate_wr=round(wr, 4),
               flat_mean_pnl_per_bet=round(unit, 4),
               flat_pnl_at_0p001_bnb=round(FLAT_STAKE_BNB * sum(b["pnl"] for b in bets), 6),
               total_unit_pnl=round(sum(b["pnl"] for b in bets), 3))
    perm = perm_null_stats(bets)
    out.update(null_mean=perm["null_mean"], deficit_vs_null=perm["deficit_vs_null"],
               perm_z=perm["perm_z"], p_upper=perm["p_upper"], p_lower=perm["p_lower"])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("--- building canonical bets (risk-free signal stream) ---", flush=True)
    bets = build_canonical_bets()
    max_lock = max(b["lock"] for b in bets)
    e14_cutoff = max_lock - TRAIL_DAYS * 86400
    recent_epochs = [b["epoch"] for b in bets if b["lock"] >= e14_cutoff]
    e14 = min(recent_epochs) if recent_epochs else None
    print(f"  {len(bets)} canonical bets; newest lock "
          f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(max_lock))}; "
          f"recent_2w starts ~epoch {e14}", flush=True)

    windows = {
        "dead_vmlive": [b for b in bets if 487687 <= b["epoch"] <= 488832],
        "synced_new": [b for b in bets if 488833 <= b["epoch"] <= 490743],
        "recent_2w": [b for b in bets if b["lock"] >= e14_cutoff],
    }
    results = {name: window_stats(bs, name) for name, bs in windows.items()}

    findings = dict(
        run_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        newest_lock_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime(max_lock)),
        recent_2w_start_epoch=e14, trail_days=TRAIL_DAYS,
        total_canonical_bets=len(bets), windows=results)
    (OUT / "recent_window_flatstake.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8")

    print("\n=== flat-stake canonical-gate windows (risk-free) ===")
    print(f"{'window':>12} {'n':>5} {'gateWR':>7} {'flat/bet':>9} "
          f"{'PnL@0.001':>10} {'deficit':>8} {'p_up':>6} {'p_lo':>6}")
    for name in ("dead_vmlive", "synced_new", "recent_2w"):
        r = results[name]
        if r["n_bets"] == 0:
            print(f"{name:>12} {0:>5}"); continue
        print(f"{name:>12} {r['n_bets']:>5} {r['gate_wr']:>7.4f} "
              f"{r['flat_mean_pnl_per_bet']:>+9.4f} {r['flat_pnl_at_0p001_bnb']:>+10.5f} "
              f"{r['deficit_vs_null']:>+8.4f} {r['p_upper']:>6.3f} {r['p_lower']:>6.3f}")
    print(f"\n[done] {time.time()-t0:.0f}s; artifacts -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
