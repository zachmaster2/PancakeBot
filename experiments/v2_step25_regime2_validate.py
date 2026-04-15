"""Step 25: Validate ETH+SOL regime-2 signal with production-identical sizing.

Step 24 showed that when BTC multi-TF is silent but ETH+SOL multi-TF
both fire same direction, adding this as a regime-2 signal:
  - Adds 41% more bets
  - Maintains PnL
  - Fixes fold 5 from negative to positive (linearity)

This script validates with FULL production sizing (adaptive strength,
payout boost, pool filters) and sweeps the regime-2 threshold.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pancakebot.domain.strategy.momentum_gate as _gate_mod
from pancakebot.core.constants import (
    BNB_WEI, GAS_COST_BET_BNB, POOL_CUTOFF_SECONDS, TREASURY_FEE_FRACTION,
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

# Production sizing constants
BASE_FRAC = 0.04
SIZING_SLOPE = 100
PAYOUT_SLOPE = 1.0
ETH_SIZING_W = 0.3
SOL_SIZING_W = 0.3
MAX_FRAC = 0.30
FLOOR_BNB = 0.01
CAP_BNB = 2.0
MIN_POOL = 1.5
MIN_PAYOUT = 1.5
SMALL_POOL_THRESH = 0.0002
LARGE_POOL_THRESH = 0.0001
POOL_THRESH_BOUNDARY = 3.0


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
    """Returns (fires, direction, min_abs_strength)."""
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

    # Lower gate threshold to capture weak BTC signals
    orig = _gate_mod._MTF_THRESH
    _gate_mod._MTF_THRESH = 0.00003

    data = []
    for rnd in rounds:
        ep = int(rnd.epoch)
        la = int(rnd.lock_at)
        cms = (la - CUTOFF_S) * 1000

        btc_c = get_closes(btc_kl.get(ep), cms)
        bnb_c = get_closes(bnb_kl.get(ep), cms)
        eth_c = get_closes(eth_kl.get(ep), cms)
        sol_c = get_closes(sol_kl.get(ep), cms)

        pb, pe = _pools_from_bets(rnd, la - POOL_CUTOFF_SECONDS)

        btc_fires, btc_dir, btc_str = multi_tf(btc_c)
        eth_fires, eth_dir, eth_str = multi_tf(eth_c)
        sol_fires, sol_dir, sol_str = multi_tf(sol_c)

        data.append({
            "rnd": rnd, "epoch": ep,
            "bnb_c": bnb_c, "btc_c": btc_c, "eth_c": eth_c, "sol_c": sol_c,
            "pb": pb, "pe": pe, "pt": pb + pe,
            "btc_fires": btc_fires, "btc_dir": btc_dir, "btc_str": btc_str,
            "eth_fires": eth_fires, "eth_dir": eth_dir, "eth_str": eth_str,
            "sol_fires": sol_fires, "sol_dir": sol_dir, "sol_str": sol_str,
        })

    _gate_mod._MTF_THRESH = orig
    n_btc = sum(1 for d in data if d["btc_fires"] and d["btc_str"] >= LARGE_POOL_THRESH)
    n_eth_sol = sum(1 for d in data
                    if d["eth_fires"] and d["sol_fires"]
                    and d["eth_dir"] == d["sol_dir"]
                    and not (d["btc_fires"] and d["btc_str"] >= LARGE_POOL_THRESH))
    print(f"{len(data)} rounds, {n_btc} primary signals, {n_eth_sol} ETH+SOL regime-2 candidates, {time.time()-t0:.1f}s")
    return data


def compute_bet_size(signal_strength, pool_bnb, our_side_bnb):
    """Production-identical sizing."""
    if pool_bnb <= 0:
        return FLOOR_BNB
    frac = min(BASE_FRAC + SIZING_SLOPE * signal_strength, MAX_FRAC)
    if our_side_bnb > 0:
        payout = pool_bnb * 0.97 / our_side_bnb
        payout_mult = max(0.5, 1.0 + PAYOUT_SLOPE * (payout - 2.0))
        frac = min(frac * payout_mult, MAX_FRAC)
    return max(FLOOR_BNB, min(CAP_BNB, pool_bnb * frac))


def simulate_fold(fold, regime2_thresh, regime2_enabled, weak_btc_thresh=None):
    """Simulate one fold with full production logic + optional regime-2.

    If weak_btc_thresh is set, also bet on weak BTC (below primary thresh)
    confirmed by ETH or SOL.
    """
    bankroll = INITIAL_BANKROLL
    bets_primary = bets_regime2 = bets_weak = 0
    wins_primary = wins_regime2 = wins_weak = 0
    pnl = 0.0

    for d in fold:
        pt = d["pt"]
        if pt < MIN_POOL:
            continue

        signal = None
        strength = 0.0
        source = None

        # --- Primary: BTC multi-TF(3,7,15) ---
        if d["btc_fires"]:
            pool_thresh = LARGE_POOL_THRESH if pt >= POOL_THRESH_BOUNDARY else SMALL_POOL_THRESH
            if d["btc_str"] >= pool_thresh:
                signal = d["btc_dir"]
                # Effective strength with ETH/SOL confirmation (production logic)
                strength = d["btc_str"]
                if d["eth_fires"] and d["eth_dir"] == d["btc_dir"]:
                    strength += d["eth_str"] * ETH_SIZING_W
                if d["sol_fires"] and d["sol_dir"] == d["btc_dir"]:
                    strength += d["sol_str"] * SOL_SIZING_W
                source = "primary"

        # --- Weak BTC + ETH|SOL confirmation ---
        if signal is None and weak_btc_thresh is not None and d["btc_fires"]:
            if d["btc_str"] >= weak_btc_thresh:
                eth_ok = d["eth_fires"] and d["eth_dir"] == d["btc_dir"]
                sol_ok = d["sol_fires"] and d["sol_dir"] == d["btc_dir"]
                if eth_ok or sol_ok:
                    signal = d["btc_dir"]
                    strength = d["btc_str"]
                    if eth_ok: strength += d["eth_str"] * ETH_SIZING_W
                    if sol_ok: strength += d["sol_str"] * SOL_SIZING_W
                    source = "weak_btc"

        # --- Regime-2: ETH+SOL both multi-TF, BTC silent ---
        if signal is None and regime2_enabled:
            if d["eth_fires"] and d["sol_fires"] and d["eth_dir"] == d["sol_dir"]:
                # Use min of ETH and SOL strength
                r2_str = min(d["eth_str"], d["sol_str"])
                if r2_str >= regime2_thresh:
                    signal = d["eth_dir"]
                    # For sizing, use combined ETH+SOL strength
                    strength = d["eth_str"] * ETH_SIZING_W + d["sol_str"] * SOL_SIZING_W
                    source = "regime2"

        if signal is None:
            continue

        # Payout floor
        our_side = d["pb"] if signal == "Bull" else d["pe"]
        if our_side > 0 and pt > 0:
            payout = pt * 0.97 / our_side
            if payout < MIN_PAYOUT:
                continue

        bet = compute_bet_size(strength, pt, our_side)
        if bet < MIN_BET:
            continue

        # Settle
        bankroll -= bet + GAS_COST_BET_BNB
        out = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=signal,
            round_closed=d["rnd"], treasury_fee_fraction=TREASURY_FEE,
        )
        bankroll += out.credit_bnb
        profit = out.credit_bnb - bet - GAS_COST_BET_BNB
        pnl += profit

        if source == "primary":
            bets_primary += 1
            if profit > 0: wins_primary += 1
        elif source == "weak_btc":
            bets_weak += 1
            if profit > 0: wins_weak += 1
        else:
            bets_regime2 += 1
            if profit > 0: wins_regime2 += 1

    return {
        "pnl": pnl,
        "n_rounds": len(fold),
        "bets_primary": bets_primary, "wins_primary": wins_primary,
        "bets_regime2": bets_regime2, "wins_regime2": wins_regime2,
        "bets_weak": bets_weak, "wins_weak": wins_weak,
    }


def run_5fold(data, regime2_thresh, regime2_enabled, weak_btc_thresh=None):
    fold_size = len(data) // N_FOLDS
    folds = [data[i*fold_size:(i+1)*fold_size] for i in range(N_FOLDS)]
    results = [simulate_fold(f, regime2_thresh, regime2_enabled, weak_btc_thresh) for f in folds]

    total_primary = sum(r["bets_primary"] for r in results)
    total_regime2 = sum(r["bets_regime2"] for r in results)
    total_weak = sum(r["bets_weak"] for r in results)
    total_bets = total_primary + total_regime2 + total_weak
    total_wins = sum(r["wins_primary"] + r["wins_regime2"] + r["wins_weak"] for r in results)
    total_rounds = sum(r["n_rounds"] for r in results)
    pnl_2ks = [r["pnl"] / r["n_rounds"] * 2000 for r in results]
    avg = sum(pnl_2ks) / len(pnl_2ks)
    std = (sum((p - avg)**2 for p in pnl_2ks) / len(pnl_2ks)) ** 0.5
    npos = sum(1 for p in pnl_2ks if p > 0)

    # Per-source WR
    wr_primary = sum(r["wins_primary"] for r in results) / total_primary * 100 if total_primary > 0 else 0
    wr_regime2 = sum(r["wins_regime2"] for r in results) / total_regime2 * 100 if total_regime2 > 0 else 0
    wr_weak = sum(r["wins_weak"] for r in results) / total_weak * 100 if total_weak > 0 else 0

    return {
        "total_bets": total_bets,
        "bets_primary": total_primary, "bets_regime2": total_regime2, "bets_weak": total_weak,
        "wr": total_wins / total_bets * 100 if total_bets > 0 else 0,
        "wr_primary": wr_primary, "wr_regime2": wr_regime2, "wr_weak": wr_weak,
        "bets_2k": total_bets / total_rounds * 2000,
        "pnl_2k": avg, "fold_std": std, "npos": npos,
        "pnl_2ks": pnl_2ks,
    }


def main():
    rounds, bnb_kl, btc_kl, eth_kl, sol_kl = load_data()
    data = precompute(rounds, bnb_kl, btc_kl, eth_kl, sol_kl)

    # ===== BASELINE (primary only) =====
    print(f"\n{'='*130}")
    print("BASELINE: Primary BTC multi-TF only (production sizing)")
    print(f"{'='*130}")
    baseline = run_5fold(data, 0, False)
    print(f"  Bets/2k: {baseline['bets_2k']:.1f} (all primary)")
    print(f"  WR: {baseline['wr']:.1f}%")
    print(f"  PnL/2k: {baseline['pnl_2k']:+.2f}, fold_std: {baseline['fold_std']:.2f}, pos: {baseline['npos']}/5")
    print(f"  Per-fold: {' '.join(f'{p:+.2f}' for p in baseline['pnl_2ks'])}")

    # ===== REGIME-2 SWEEP =====
    print(f"\n{'='*130}")
    print("REGIME-2: Primary + ETH&SOL multi-TF on flat rounds")
    print(f"{'='*130}")

    hdr = f"  {'Config':<40s} {'b/2k':>5} {'pri':>4} {'r2':>4} {'WR%':>5} {'WR_r2':>5} {'PnL/2k':>7} {'std':>5} {'pos':>4}  {'f1':>6} {'f2':>6} {'f3':>6} {'f4':>6} {'f5':>6}"
    print(hdr)
    print(f"  {'-'*120}")

    for r2_thresh in [0.00003, 0.00005, 0.00008, 0.0001, 0.00015, 0.0002]:
        s = run_5fold(data, r2_thresh, True)
        m = " ***" if s["npos"] >= 5 else " **" if s["npos"] >= 4 else ""
        p = s["pnl_2ks"]
        print(f"  ETH+SOL r2_thresh={str(r2_thresh):<22s} {s['bets_2k']:5.1f} {s['bets_primary']:4d} {s['bets_regime2']:4d} "
              f"{s['wr']:5.1f} {s['wr_regime2']:5.1f} {s['pnl_2k']:+7.2f} {s['fold_std']:5.2f} {s['npos']:>3}/5  "
              f"{p[0]:+6.2f} {p[1]:+6.2f} {p[2]:+6.2f} {p[3]:+6.2f} {p[4]:+6.2f}{m}")

    # ===== WEAK BTC + CONFIRMATION =====
    print(f"\n{'='*130}")
    print("WEAK BTC: Primary + weak BTC confirmed by ETH|SOL")
    print(f"{'='*130}")
    print(hdr.replace("r2", "wk"))
    print(f"  {'-'*120}")

    for wbt in [0.00003, 0.00005, 0.00008]:
        s = run_5fold(data, 0, False, weak_btc_thresh=wbt)
        m = " ***" if s["npos"] >= 5 else " **" if s["npos"] >= 4 else ""
        p = s["pnl_2ks"]
        print(f"  Weak BTC>{wbt} + ETH|SOL            {s['bets_2k']:5.1f} {s['bets_primary']:4d} {s['bets_weak']:4d} "
              f"{s['wr']:5.1f} {s['wr_weak']:5.1f} {s['pnl_2k']:+7.2f} {s['fold_std']:5.2f} {s['npos']:>3}/5  "
              f"{p[0]:+6.2f} {p[1]:+6.2f} {p[2]:+6.2f} {p[3]:+6.2f} {p[4]:+6.2f}{m}")

    # ===== COMBINED: Primary + regime-2 + weak BTC =====
    print(f"\n{'='*130}")
    print("COMBINED: Primary + regime-2 + weak BTC")
    print(f"{'='*130}")

    for r2t in [0.00005, 0.0001]:
        for wbt in [0.00005, 0.00008]:
            s = run_5fold(data, r2t, True, weak_btc_thresh=wbt)
            m = " ***" if s["npos"] >= 5 else " **" if s["npos"] >= 4 else ""
            p = s["pnl_2ks"]
            total_extra = s["bets_regime2"] + s["bets_weak"]
            print(f"  r2>{r2t} + wBTC>{wbt}                   "
                  f"{s['bets_2k']:5.1f} {s['bets_primary']:4d} {s['bets_regime2']:4d}+{s['bets_weak']:3d} "
                  f"{s['wr']:5.1f} {s['pnl_2k']:+7.2f} {s['fold_std']:5.2f} {s['npos']:>3}/5  "
                  f"{p[0]:+6.2f} {p[1]:+6.2f} {p[2]:+6.2f} {p[3]:+6.2f} {p[4]:+6.2f}{m}")

    # ===== FINAL COMPARISON =====
    print(f"\n{'='*130}")
    print("SUMMARY")
    print(f"{'='*130}")
    print(f"  Baseline (primary only):         bets/2k={baseline['bets_2k']:.1f}, PnL/2k={baseline['pnl_2k']:+.2f}, std={baseline['fold_std']:.2f}, {baseline['npos']}/5")

    # Best regime-2 only
    best_r2 = None
    for r2t in [0.00005, 0.0001, 0.00015]:
        s = run_5fold(data, r2t, True)
        if s["npos"] >= 5 and (best_r2 is None or s["pnl_2k"] > best_r2[1]["pnl_2k"]):
            best_r2 = (r2t, s)
    if best_r2:
        s = best_r2[1]
        print(f"  Best regime-2 (r2>{best_r2[0]}):    bets/2k={s['bets_2k']:.1f}, PnL/2k={s['pnl_2k']:+.2f}, std={s['fold_std']:.2f}, {s['npos']}/5")
        print(f"    Primary: {s['bets_primary']} bets, {s['wr_primary']:.1f}% WR")
        print(f"    Regime-2: {s['bets_regime2']} bets, {s['wr_regime2']:.1f}% WR")
        print(f"    Fold delta: {' '.join(f'{b-a:+.2f}' for a, b in zip(baseline['pnl_2ks'], s['pnl_2ks']))}")


if __name__ == "__main__":
    main()
