"""Step 22b: Combined filter + sizing sweep.

Phase 1 showed filter relaxation increases bet rate + linearity but lowers PnL/2k.
Phase 2 showed base_frac=0.04 adds ~+0.09/2k.

This tests: can improved sizing compensate for the PnL loss from filter relaxation,
giving us BOTH more bets AND equal/better PnL with better linearity?
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pancakebot.domain.strategy.momentum_gate as _gate_mod
from pancakebot.core.constants import (
    BNB_WEI, GAS_COST_BET_BNB, POOL_CUTOFF_SECONDS, TREASURY_FEE_FRACTION,
)
from pancakebot.domain.strategy.momentum_gate import compute_signal_from_klines
from pancakebot.domain.strategy.momentum_pipeline import _pools_from_bets
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2
N_FOLDS = 5
INITIAL_BANKROLL = 50.0
MIN_BET_AMOUNT = 0.001


@dataclass
class PR:
    rnd: object
    signal: str | None
    signal_strength: float
    eth_confirm: float
    sol_confirm: float
    pool_bull: float
    pool_bear: float
    pool_total: float


@dataclass
class Cfg:
    min_pool: float
    thresh_mode: str
    uniform_thresh: float
    small_thresh: float
    large_thresh: float
    thresh_boundary: float
    min_payout: float
    base_frac: float
    sizing_slope: float
    payout_slope: float
    eth_w: float
    sol_w: float
    max_frac: float
    cap_bnb: float


def load_and_precompute():
    print("Loading data...", end=" ", flush=True)
    t0 = time.time()
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())

    def lk(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    bnb = lk("var/bnb_spot_prices.jsonl")
    btc = lk("var/btc_spot_prices.jsonl")
    eth = lk("var/eth_spot_prices.jsonl")
    sol = lk("var/sol_spot_prices.jsonl")
    print(f"{len(rounds)} rounds, {time.time()-t0:.1f}s")

    orig = _gate_mod._MTF_THRESH
    _gate_mod._MTF_THRESH = 0.00005
    try:
        print("Pre-computing signals...", end=" ", flush=True)
        t0 = time.time()
        result = []
        for rnd in rounds:
            ep = int(rnd.epoch)
            la = int(rnd.lock_at)
            cms = (la - CUTOFF_S) * 1000
            b_raw = bnb.get(ep)
            t_raw = btc.get(ep)
            if b_raw is None or t_raw is None:
                result.append(PR(rnd=rnd, signal=None, signal_strength=0, eth_confirm=0, sol_confirm=0, pool_bull=0, pool_bear=0, pool_total=0))
                continue
            sig = compute_signal_from_klines(b_raw, t_raw, cms, eth_klines=eth.get(ep), sol_klines=sol.get(ep))
            pb, pe = _pools_from_bets(rnd, la - POOL_CUTOFF_SECONDS)
            result.append(PR(rnd=rnd, signal=sig.signal, signal_strength=sig.signal_strength, eth_confirm=sig.eth_confirmation_strength, sol_confirm=sig.sol_confirmation_strength, pool_bull=pb, pool_bear=pe, pool_total=pb+pe))
        n_sig = sum(1 for p in result if p.signal is not None)
        print(f"{n_sig} signals, {time.time()-t0:.1f}s")
        return result
    finally:
        _gate_mod._MTF_THRESH = orig


def sim_fold(fold, cfg):
    bankroll = INITIAL_BANKROLL
    bets = wins = 0
    pnl = 0.0
    for pr in fold:
        if pr.signal is None:
            continue
        if cfg.thresh_mode == "uniform":
            if pr.signal_strength < cfg.uniform_thresh:
                continue
        else:
            t = cfg.large_thresh if pr.pool_total >= cfg.thresh_boundary else cfg.small_thresh
            if pr.signal_strength < t:
                continue
        if pr.pool_total < cfg.min_pool:
            continue
        our = pr.pool_bull if pr.signal == "Bull" else pr.pool_bear
        if our > 0 and pr.pool_total > 0:
            pay = pr.pool_total * 0.97 / our
            if pay < cfg.min_payout:
                continue
        elif our <= 0:
            pay = 99.0
        else:
            continue
        eff = pr.signal_strength + (pr.eth_confirm * cfg.eth_w if pr.eth_confirm > 0 else 0) + (pr.sol_confirm * cfg.sol_w if pr.sol_confirm > 0 else 0)
        frac = min(cfg.base_frac + cfg.sizing_slope * eff, cfg.max_frac)
        if our > 0:
            pm = max(0.5, 1.0 + cfg.payout_slope * (pay - 2.0))
            frac = min(frac * pm, cfg.max_frac)
        bet = max(0.01, min(cfg.cap_bnb, pr.pool_total * frac))
        if bet < MIN_BET_AMOUNT:
            continue
        bankroll -= bet + GAS_COST_BET_BNB
        out = settle_bet_against_closed_round(bet_bnb=bet, bet_side=pr.signal, round_closed=pr.rnd, treasury_fee_fraction=0.03)
        bankroll += out.credit_bnb
        p = out.credit_bnb - bet - GAS_COST_BET_BNB
        pnl += p
        bets += 1
        if p > 0:
            wins += 1
    return bets, wins, pnl, len(fold)


def run_5f(data, cfg):
    fs = len(data) // N_FOLDS
    folds = [data[i*fs:(i+1)*fs] for i in range(N_FOLDS)]
    results = [sim_fold(f, cfg) for f in folds]
    tb = sum(r[0] for r in results)
    tw = sum(r[1] for r in results)
    tp = sum(r[2] for r in results)
    tr = sum(r[3] for r in results)
    pnl_2ks = [r[2] / r[3] * 2000 if r[3] > 0 else 0 for r in results]
    avg = sum(pnl_2ks) / len(pnl_2ks)
    std = (sum((p - avg)**2 for p in pnl_2ks) / len(pnl_2ks)) ** 0.5
    npos = sum(1 for p in pnl_2ks if p > 0)
    return {
        "bets_2k": tb / tr * 2000 if tr > 0 else 0,
        "wr": tw / tb * 100 if tb > 0 else 0,
        "pnl_2k": avg,
        "std": std,
        "npos": npos,
        "pnl_2ks": pnl_2ks,
        "total_bets": tb,
    }


def main():
    data = load_and_precompute()

    # Define configs: combine filter relaxation with sizing improvements
    configs = []

    # Base filters to try
    filter_combos = [
        # (label, min_pool, thresh_mode, uniform_thresh, min_payout)
        ("prod",             2.0,  "adaptive", 0.0001, 1.5),
        ("pool>=1.5",        1.5,  "adaptive", 0.0001, 1.5),
        ("pool>=1.5 pay1.3", 1.5,  "adaptive", 0.0001, 1.3),
        ("pool>=1.25",       1.25, "adaptive", 0.0001, 1.5),
        ("pool>=1.25 pay1.3",1.25, "adaptive", 0.0001, 1.3),
        ("pool>=1.0",        1.0,  "adaptive", 0.0001, 1.5),
        ("pool>=1.0 pay1.3", 1.0,  "adaptive", 0.0001, 1.3),
    ]

    # Sizing combos to try
    sizing_combos = [
        # (label, base_frac, slope, cap)
        ("prod",       0.03, 100, 2.0),
        ("b=.04",      0.04, 100, 2.0),
        ("b=.05",      0.05, 100, 2.0),
        ("b=.04 s=75", 0.04,  75, 2.0),
        ("b=.04 c1.5", 0.04, 100, 1.5),
        ("b=.05 c1.5", 0.05, 100, 1.5),
    ]

    print(f"\n{'='*120}")
    print(f"COMBINED SWEEP: {len(filter_combos)} filter x {len(sizing_combos)} sizing = {len(filter_combos)*len(sizing_combos)} configs")
    print(f"{'='*120}")

    all_results = []
    for fl, mp, tm, ut, pay in filter_combos:
        for sl, bf, slope, cap in sizing_combos:
            cfg = Cfg(
                min_pool=mp, thresh_mode=tm, uniform_thresh=ut,
                small_thresh=0.0002, large_thresh=0.0001, thresh_boundary=3.0,
                min_payout=pay, base_frac=bf, sizing_slope=slope,
                payout_slope=1.0, eth_w=0.3, sol_w=0.3, max_frac=0.30, cap_bnb=cap,
            )
            s = run_5f(data, cfg)
            label = f"{fl:20s} {sl:12s}"
            all_results.append((label, s))

    # Sort by PnL/2k
    all_results.sort(key=lambda x: x[1]["pnl_2k"], reverse=True)

    print(f"\n{'Config':<34} {'bets/2k':>7} {'WR%':>6} {'PnL/2k':>8} {'std':>6} {'pos':>4}  {'f1':>7} {'f2':>7} {'f3':>7} {'f4':>7} {'f5':>7}")
    print("-" * 120)

    for label, s in all_results:
        p = s["pnl_2ks"]
        m = " ***" if s["npos"] >= 5 else " **" if s["npos"] >= 4 else ""
        print(f"{label:<34} {s['bets_2k']:7.1f} {s['wr']:5.1f}% {s['pnl_2k']:+7.2f} {s['std']:6.2f} {s['npos']:>3}/5  {p[0]:+7.2f} {p[1]:+7.2f} {p[2]:+7.2f} {p[3]:+7.2f} {p[4]:+7.2f}{m}")

    # Find best 5/5 with more bets than baseline
    baseline = [r for r in all_results if "prod" in r[0] and "prod" in r[0].split()[1]][0]
    b_bets = baseline[1]["bets_2k"]
    better = [r for r in all_results if r[1]["npos"] == 5 and r[1]["bets_2k"] > b_bets * 1.1]
    if better:
        print(f"\n>>> BEST with >10% more bets AND 5/5:")
        for label, s in better[:5]:
            print(f"    {label}: bets/2k={s['bets_2k']:.1f}, PnL/2k={s['pnl_2k']:+.2f}, std={s['std']:.2f}")

    # Best overall linearity (5/5, lowest std)
    best_linear = sorted([r for r in all_results if r[1]["npos"] == 5], key=lambda x: x[1]["std"])
    if best_linear:
        print(f"\n>>> BEST linearity (lowest fold_std, 5/5):")
        for label, s in best_linear[:5]:
            print(f"    {label}: bets/2k={s['bets_2k']:.1f}, PnL/2k={s['pnl_2k']:+.2f}, std={s['std']:.2f}")


if __name__ == "__main__":
    main()
