"""Step 26: Three remaining angles with existing data.

1. Regime-2 specific sizing (separate base_frac/slope/cap for ETH+SOL bets)
2. Signal-strength WR analysis (non-linear WR vs strength, optimal sizing)
3. Bet data as confirmation (whale/late-bet alignment with signal direction)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pancakebot.domain.strategy.momentum_gate as _gate_mod
from pancakebot.core.constants import (
    BNB_WEI, GAS_COST_BET_BNB, POOL_CUTOFF_SECONDS,
)
from pancakebot.domain.strategy.momentum_gate import _trim_to_window
from pancakebot.domain.strategy.momentum_pipeline import _pools_from_bets
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2
CANDLE_COUNT = 31
N_FOLDS = 5
TREASURY_FEE = 0.03
INITIAL_BANKROLL = 50.0
MIN_BET = 0.001
MIN_POOL = 1.5
MIN_PAYOUT = 1.5
SMALL_POOL_THRESH = 0.0002
LARGE_POOL_THRESH = 0.0001
POOL_THRESH_BOUNDARY = 3.0
REGIME2_MIN_STR = 0.00015


def load_data():
    print("Loading data...", end=" ", flush=True)
    t0 = time.time()
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    def lk(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out
    bnb = lk("var/bnb_spot_prices.jsonl")
    btc = lk("var/btc_spot_prices.jsonl")
    eth = lk("var/eth_spot_prices.jsonl")
    sol = lk("var/sol_spot_prices.jsonl")
    print(f"{len(rounds)} rounds, {time.time()-t0:.1f}s")
    return rounds, bnb, btc, eth, sol


def get_closes(raw, cutoff_ms):
    if raw is None: return None
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT: return None
    return [c[4] for c in trimmed[-CANDLE_COUNT:]]


def ret(closes, lb):
    if closes is None or len(closes) < lb + 1 or closes[-(lb+1)] == 0:
        return None
    return (closes[-1] - closes[-(lb+1)]) / closes[-(lb+1)]


def multi_tf(closes, lookbacks=(3,7,15)):
    rets = [ret(closes, lb) for lb in lookbacks]
    if any(r is None for r in rets):
        return False, None, 0.0
    if all(r > 0 for r in rets):
        return True, "Bull", min(abs(r) for r in rets)
    if all(r < 0 for r in rets):
        return True, "Bear", min(abs(r) for r in rets)
    return False, None, 0.0


def precompute(rounds, bnb_kl, btc_kl, eth_kl, sol_kl):
    print("Pre-computing...", end=" ", flush=True)
    t0 = time.time()

    orig = _gate_mod._MTF_THRESH
    _gate_mod._MTF_THRESH = 0.00003

    data = []
    for rnd in rounds:
        ep = int(rnd.epoch)
        la = int(rnd.lock_at)
        cms = (la - CUTOFF_S) * 1000

        btc_c = get_closes(btc_kl.get(ep), cms)
        eth_c = get_closes(eth_kl.get(ep), cms)
        sol_c = get_closes(sol_kl.get(ep), cms)

        pb, pe = _pools_from_bets(rnd, la - POOL_CUTOFF_SECONDS)
        pt = pb + pe

        btc_fires, btc_dir, btc_str = multi_tf(btc_c)
        eth_fires, eth_dir, eth_str = multi_tf(eth_c)
        sol_fires, sol_dir, sol_str = multi_tf(sol_c)

        # Bet data features
        late_bull_wei = 0
        late_bear_wei = 0
        whale_side = None
        max_bet_wei = 0
        pool_cutoff_ts = la - POOL_CUTOFF_SECONDS
        for bet in rnd.bets:
            ca = int(bet.created_at)
            if ca > pool_cutoff_ts:
                continue
            amt = int(bet.amount_wei)
            # Late bets: within last 30s before pool cutoff
            if ca >= pool_cutoff_ts - 30:
                if bet.position == "Bull":
                    late_bull_wei += amt
                else:
                    late_bear_wei += amt
            # Whale: largest single bet
            if amt > max_bet_wei:
                max_bet_wei = amt
                whale_side = bet.position

        # Determine signal source
        source = None
        signal_dir = None
        signal_str = 0.0
        eth_confirm = 0.0
        sol_confirm = 0.0

        # Primary
        if btc_fires:
            pool_thresh = LARGE_POOL_THRESH if pt >= POOL_THRESH_BOUNDARY else SMALL_POOL_THRESH
            if btc_str >= pool_thresh:
                source = "primary"
                signal_dir = btc_dir
                signal_str = btc_str
                if eth_fires and eth_dir == btc_dir:
                    eth_confirm = eth_str
                if sol_fires and sol_dir == btc_dir:
                    sol_confirm = sol_str

        # Regime-2
        if source is None:
            if eth_fires and sol_fires and eth_dir == sol_dir:
                r2_str = min(eth_str, sol_str)
                if r2_str >= REGIME2_MIN_STR:
                    source = "regime2"
                    signal_dir = eth_dir
                    signal_str = r2_str
                    eth_confirm = eth_str
                    sol_confirm = sol_str

        data.append({
            "rnd": rnd, "epoch": ep, "lock_at": la,
            "pb": pb, "pe": pe, "pt": pt,
            "source": source, "signal_dir": signal_dir,
            "signal_str": signal_str, "eth_confirm": eth_confirm, "sol_confirm": sol_confirm,
            "btc_str": btc_str, "btc_dir": btc_dir, "btc_fires": btc_fires,
            "eth_str": eth_str, "sol_str": sol_str,
            "late_bull_bnb": late_bull_wei / 1e18,
            "late_bear_bnb": late_bear_wei / 1e18,
            "whale_side": whale_side,
            "whale_bnb": max_bet_wei / 1e18,
        })

    _gate_mod._MTF_THRESH = orig
    n_pri = sum(1 for d in data if d["source"] == "primary")
    n_r2 = sum(1 for d in data if d["source"] == "regime2")
    print(f"{len(data)} rounds, {n_pri} primary, {n_r2} regime-2, {time.time()-t0:.1f}s")
    return data


def compute_bet(strength, pool, our_side, base_frac, slope, cap, payout_slope=1.0, max_frac=0.30):
    if pool <= 0: return 0.01
    frac = min(base_frac + slope * strength, max_frac)
    if our_side > 0:
        pay = pool * 0.97 / our_side
        pm = max(0.5, 1.0 + payout_slope * (pay - 2.0))
        frac = min(frac * pm, max_frac)
    return max(0.01, min(cap, pool * frac))


def sim_fold(fold, pri_bf, pri_slope, pri_cap, r2_bf, r2_slope, r2_cap,
             whale_filter=False, late_filter=False, payout_slope=1.0):
    bankroll = INITIAL_BANKROLL
    bets = wins = 0
    pnl = 0.0
    bets_pri = wins_pri = bets_r2 = wins_r2 = 0

    for d in fold:
        if d["source"] is None:
            continue
        pt = d["pt"]
        if pt < MIN_POOL:
            continue

        signal = d["signal_dir"]
        our = d["pb"] if signal == "Bull" else d["pe"]
        if our > 0 and pt > 0:
            pay = pt * 0.97 / our
            if pay < MIN_PAYOUT:
                continue
        elif our <= 0:
            pay = 99.0
        else:
            continue

        # Bet data filters
        if whale_filter and d["whale_side"] is not None:
            if d["whale_side"] != signal and d["whale_bnb"] >= 0.1:
                continue  # whale disagrees with our signal

        if late_filter:
            late_our = d["late_bull_bnb"] if signal == "Bull" else d["late_bear_bnb"]
            late_opp = d["late_bear_bnb"] if signal == "Bull" else d["late_bull_bnb"]
            if late_opp > late_our * 2.0 and late_opp > 0.05:
                continue  # late money strongly disagrees

        # Sizing
        if d["source"] == "primary":
            eff = d["signal_str"]
            if d["eth_confirm"] > 0: eff += d["eth_confirm"] * 0.3
            if d["sol_confirm"] > 0: eff += d["sol_confirm"] * 0.3
            bet = compute_bet(eff, pt, our, pri_bf, pri_slope, pri_cap, payout_slope)
        else:
            eff = d["eth_confirm"] * 0.3 + d["sol_confirm"] * 0.3
            bet = compute_bet(eff, pt, our, r2_bf, r2_slope, r2_cap, payout_slope)

        if bet < MIN_BET:
            continue

        bankroll -= bet + GAS_COST_BET_BNB
        out = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=signal,
            round_closed=d["rnd"], treasury_fee_fraction=TREASURY_FEE,
        )
        bankroll += out.credit_bnb
        profit = out.credit_bnb - bet - GAS_COST_BET_BNB
        pnl += profit; bets += 1
        if profit > 0: wins += 1
        if d["source"] == "primary":
            bets_pri += 1
            if profit > 0: wins_pri += 1
        else:
            bets_r2 += 1
            if profit > 0: wins_r2 += 1

    return {"pnl": pnl, "n": len(fold), "bets": bets, "wins": wins,
            "bets_pri": bets_pri, "wins_pri": wins_pri,
            "bets_r2": bets_r2, "wins_r2": wins_r2}


def run_5f(data, **kwargs):
    fs = len(data) // N_FOLDS
    folds = [data[i*fs:(i+1)*fs] for i in range(N_FOLDS)]
    results = [sim_fold(f, **kwargs) for f in folds]
    tb = sum(r["bets"] for r in results)
    tw = sum(r["wins"] for r in results)
    tr = sum(r["n"] for r in results)
    pnl_2ks = [r["pnl"] / r["n"] * 2000 for r in results]
    avg = sum(pnl_2ks) / len(pnl_2ks)
    std = (sum((p - avg)**2 for p in pnl_2ks) / len(pnl_2ks)) ** 0.5
    npos = sum(1 for p in pnl_2ks if p > 0)
    return {
        "bets_2k": tb / tr * 2000, "wr": tw / tb * 100 if tb > 0 else 0,
        "pnl_2k": avg, "std": std, "npos": npos, "pnl_2ks": pnl_2ks,
        "bets_pri": sum(r["bets_pri"] for r in results),
        "bets_r2": sum(r["bets_r2"] for r in results),
        "wr_pri": sum(r["wins_pri"] for r in results) / max(1, sum(r["bets_pri"] for r in results)) * 100,
        "wr_r2": sum(r["wins_r2"] for r in results) / max(1, sum(r["bets_r2"] for r in results)) * 100,
    }


def pr(label, s):
    p = s["pnl_2ks"]
    m = " ***" if s["npos"] >= 5 else " **" if s["npos"] >= 4 else ""
    print(f"  {label:<50s} {s['bets_2k']:5.1f} {s['wr']:5.1f}% {s['pnl_2k']:+7.2f} {s['std']:5.2f} {s['npos']}/5  "
          f"{p[0]:+6.2f} {p[1]:+6.2f} {p[2]:+6.2f} {p[3]:+6.2f} {p[4]:+6.2f}{m}")


def main():
    rounds, bnb_kl, btc_kl, eth_kl, sol_kl = load_data()
    data = precompute(rounds, bnb_kl, btc_kl, eth_kl, sol_kl)

    # Baseline
    print(f"\n{'='*130}")
    print("BASELINE")
    print(f"{'='*130}")
    bl = run_5f(data, pri_bf=0.04, pri_slope=100, pri_cap=2.0,
                r2_bf=0.04, r2_slope=100, r2_cap=2.0)
    pr("Current production (same sizing for both)", bl)

    # =========================================================
    # ANGLE 1: Regime-2 specific sizing
    # =========================================================
    print(f"\n{'='*130}")
    print("ANGLE 1: Regime-2 specific sizing (primary unchanged at bf=.04, s=100, cap=2.0)")
    print(f"{'='*130}")

    for r2_bf in [0.02, 0.03, 0.04, 0.05]:
        for r2_slope in [50, 75, 100, 150]:
            for r2_cap in [0.5, 1.0, 2.0]:
                s = run_5f(data, pri_bf=0.04, pri_slope=100, pri_cap=2.0,
                           r2_bf=r2_bf, r2_slope=r2_slope, r2_cap=r2_cap)
                if s["npos"] >= 4:
                    pr(f"r2: bf={r2_bf} s={r2_slope} cap={r2_cap}", s)

    # =========================================================
    # ANGLE 2: Signal-strength WR analysis
    # =========================================================
    print(f"\n{'='*130}")
    print("ANGLE 2: Signal strength vs WR (primary signal)")
    print(f"{'='*130}")

    # Bin primary bets by signal strength and compute WR per bin
    primary_bets = []
    for d in data:
        if d["source"] != "primary" or d["pt"] < MIN_POOL:
            continue
        signal = d["signal_dir"]
        our = d["pb"] if signal == "Bull" else d["pe"]
        if our > 0 and d["pt"] > 0:
            pay = d["pt"] * 0.97 / our
            if pay < MIN_PAYOUT:
                continue

        eff = d["signal_str"]
        if d["eth_confirm"] > 0: eff += d["eth_confirm"] * 0.3
        if d["sol_confirm"] > 0: eff += d["sol_confirm"] * 0.3
        bet = compute_bet(eff, d["pt"], our, 0.04, 100, 2.0)
        out = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=signal,
            round_closed=d["rnd"], treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - bet - GAS_COST_BET_BNB
        primary_bets.append((d["signal_str"], eff, profit, bet, d["pt"]))

    # Bin by raw BTC signal strength
    bins = [(0.0001, 0.00015), (0.00015, 0.0002), (0.0002, 0.0003),
            (0.0003, 0.0005), (0.0005, 0.001), (0.001, 1.0)]
    print(f"\n  {'Strength bin':<20s} {'Count':>6} {'WR%':>6} {'Avg profit':>10} {'Avg bet':>8} {'PnL':>8}")
    print(f"  {'-'*65}")
    for lo, hi in bins:
        in_bin = [(s, e, p, b, pt) for s, e, p, b, pt in primary_bets if lo <= s < hi]
        if not in_bin:
            continue
        n = len(in_bin)
        w = sum(1 for _, _, p, _, _ in in_bin if p > 0)
        avg_p = sum(p for _, _, p, _, _ in in_bin) / n
        avg_b = sum(b for _, _, _, b, _ in in_bin) / n
        total_pnl = sum(p for _, _, p, _, _ in in_bin)
        print(f"  [{lo:.5f}, {hi:.5f})  {n:6d} {w/n*100:5.1f}% {avg_p:+10.4f} {avg_b:8.4f} {total_pnl:+8.2f}")

    # Same for regime-2
    print(f"\n  Regime-2 signal strength vs WR:")
    r2_bets = []
    for d in data:
        if d["source"] != "regime2" or d["pt"] < MIN_POOL:
            continue
        signal = d["signal_dir"]
        our = d["pb"] if signal == "Bull" else d["pe"]
        if our > 0 and d["pt"] > 0:
            pay = d["pt"] * 0.97 / our
            if pay < MIN_PAYOUT:
                continue
        eff = d["eth_confirm"] * 0.3 + d["sol_confirm"] * 0.3
        bet = compute_bet(eff, d["pt"], our, 0.04, 100, 2.0)
        out = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=signal,
            round_closed=d["rnd"], treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - bet - GAS_COST_BET_BNB
        r2_bets.append((d["signal_str"], eff, profit, bet, d["pt"]))

    r2_bins = [(0.00015, 0.0002), (0.0002, 0.0003), (0.0003, 0.0005), (0.0005, 1.0)]
    print(f"  {'Strength bin':<20s} {'Count':>6} {'WR%':>6} {'Avg profit':>10} {'Avg bet':>8} {'PnL':>8}")
    print(f"  {'-'*65}")
    for lo, hi in r2_bins:
        in_bin = [(s, e, p, b, pt) for s, e, p, b, pt in r2_bets if lo <= s < hi]
        if not in_bin:
            continue
        n = len(in_bin)
        w = sum(1 for _, _, p, _, _ in in_bin if p > 0)
        avg_p = sum(p for _, _, p, _, _ in in_bin) / n
        avg_b = sum(b for _, _, _, b, _ in in_bin) / n
        total_pnl = sum(p for _, _, p, _, _ in in_bin)
        print(f"  [{lo:.5f}, {hi:.5f})  {n:6d} {w/n*100:5.1f}% {avg_p:+10.4f} {avg_b:8.4f} {total_pnl:+8.2f}")

    # =========================================================
    # ANGLE 3: Bet data as confirmation
    # =========================================================
    print(f"\n{'='*130}")
    print("ANGLE 3: Bet data as confirmation (whale alignment, late-money alignment)")
    print(f"{'='*130}")

    # Baseline without filters
    pr("No bet-data filters (baseline)", bl)

    # Whale filter: skip if whale disagrees
    s = run_5f(data, pri_bf=0.04, pri_slope=100, pri_cap=2.0,
               r2_bf=0.04, r2_slope=100, r2_cap=2.0, whale_filter=True)
    pr("Skip if whale (>=0.1 BNB) disagrees", s)

    # Late money filter: skip if late money strongly disagrees
    s = run_5f(data, pri_bf=0.04, pri_slope=100, pri_cap=2.0,
               r2_bf=0.04, r2_slope=100, r2_cap=2.0, late_filter=True)
    pr("Skip if late money >2x on other side", s)

    # Both filters
    s = run_5f(data, pri_bf=0.04, pri_slope=100, pri_cap=2.0,
               r2_bf=0.04, r2_slope=100, r2_cap=2.0,
               whale_filter=True, late_filter=True)
    pr("Skip if whale OR late money disagrees", s)

    # Whale alignment analysis (informational)
    print(f"\n  Whale alignment analysis (primary signal only):")
    whale_agree = whale_disagree = whale_none = 0
    wins_agree = wins_disagree = 0
    for d in data:
        if d["source"] != "primary" or d["pt"] < MIN_POOL:
            continue
        if d["whale_side"] is None or d["whale_bnb"] < 0.1:
            whale_none += 1
            continue
        if d["whale_side"] == d["signal_dir"]:
            whale_agree += 1
            # Check if this bet would win
            rnd = d["rnd"]
            if rnd.close_price is not None and rnd.lock_price is not None:
                actual = "Bull" if rnd.close_price > rnd.lock_price else "Bear"
                if actual == d["signal_dir"]:
                    wins_agree += 1
        else:
            whale_disagree += 1
            rnd = d["rnd"]
            if rnd.close_price is not None and rnd.lock_price is not None:
                actual = "Bull" if rnd.close_price > rnd.lock_price else "Bear"
                if actual == d["signal_dir"]:
                    wins_disagree += 1

    print(f"  Whale agrees with signal:    {whale_agree:5d} bets, WR={wins_agree/max(1,whale_agree)*100:.1f}%")
    print(f"  Whale disagrees with signal: {whale_disagree:5d} bets, WR={wins_disagree/max(1,whale_disagree)*100:.1f}%")
    print(f"  No whale (< 0.1 BNB):        {whale_none:5d} bets")


if __name__ == "__main__":
    main()
