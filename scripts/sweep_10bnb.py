"""Comprehensive sweep targeting +10 BNB per 5000 rounds.

Strategy: stack multiple independent signal sources, each with its own
WR and confidence-adjusted bet sizing. The goal is to find enough
high-WR rounds to accumulate +10 BNB total.

Sources:
  A) BNB 7+10 acceleration (proven ~60.5% WR, ~600 rounds)
  B) Loose BNB (any nonzero return) + BTC confirmation
  C) BTC-only acceleration on all rounds (including BNB-flat)
  D) BNB micro-move + BTC agreement (looser threshold)
  E) Multi-timeframe BTC consensus
  F) Payout-extreme filter (skip low-payout rounds)

All with pool-aware + payout-aware + confidence-tiered sizing.
"""

from __future__ import annotations

import json
from pathlib import Path

BNB_DATA_PATH = Path("var/cutoff_spot_prices.jsonl")
BTC_DATA_PATH = Path("var/btc_spot_prices.jsonl")
ETH_DATA_PATH = Path("var/eth_spot_prices.jsonl")
ROUNDS_PATH = Path("var/closed_rounds.jsonl")

BNB_WEI = 10**18
GAS_BET = 0.0002
GAS_CLAIM = 0.00025
FEE = 0.03
CUTOFF_SECONDS = 4


def find_closest(klines, target_ms):
    best, bd = None, float("inf")
    for k in klines:
        d = abs(k[0] - target_ms)
        if d < bd:
            bd, best = d, k
    return best if best and bd <= 2000 else None


def compute_pools(rnd):
    lock_at = rnd["lockAt"]
    bw, ew = 0, 0
    for b in rnd.get("bets", []):
        if b["createdAt"] > lock_at:
            continue
        if b["position"] == "Bull":
            bw += b["amountWei"]
        else:
            ew += b["amountWei"]
    return bw, ew


def payout_multiple(bull_wei, bear_wei, side, bet_bnb):
    bet_wei = int(bet_bnb * BNB_WEI)
    bw = bull_wei + (bet_wei if side == "Bull" else 0)
    ew = bear_wei + (bet_wei if side == "Bear" else 0)
    tw = bw + ew
    my = bw if side == "Bull" else ew
    return (tw * (1 - FEE)) / my if my > 0 else 0


def net_profit(bw, ew, side, outcome, bet_bnb):
    m = payout_multiple(bw, ew, side, bet_bnb)
    return bet_bnb * m - GAS_CLAIM - bet_bnb - GAS_BET if outcome == side else -bet_bnb - GAS_BET


def get_return(klines, cutoff_ms, lookback_s):
    kn = find_closest(klines, cutoff_ms)
    ka = find_closest(klines, cutoff_ms - lookback_s * 1000)
    if not kn or not ka or ka[4] <= 0:
        return None
    return (kn[4] / ka[4]) - 1


def load_data():
    bnb_by_epoch = {}
    for line in BNB_DATA_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                bnb_by_epoch[r["epoch"]] = r

    btc_by_epoch = {}
    if BTC_DATA_PATH.exists():
        for line in BTC_DATA_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    btc_by_epoch[r["epoch"]] = r

    eth_by_epoch = {}
    if ETH_DATA_PATH.exists():
        for line in ETH_DATA_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    eth_by_epoch[r["epoch"]] = r

    rounds_by_epoch = {}
    for line in ROUNDS_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rounds_by_epoch[r["epoch"]] = r

    return bnb_by_epoch, btc_by_epoch, eth_by_epoch, rounds_by_epoch


def build_rounds(bnb_by_epoch, btc_by_epoch, eth_by_epoch, rounds_by_epoch):
    """Build feature set for all valid rounds."""
    rounds = []
    for epoch, bnb_rec in bnb_by_epoch.items():
        rnd = rounds_by_epoch.get(epoch)
        if not rnd or rnd.get("failed") or rnd["position"] not in ("Bull", "Bear"):
            continue
        lock_ms = bnb_rec["lock_at"] * 1000
        cutoff_ms = lock_ms - CUTOFF_SECONDS * 1000
        bull_wei, bear_wei = compute_pools(rnd)
        if bull_wei + bear_wei == 0:
            continue

        pool_bnb = (bull_wei + bear_wei) / BNB_WEI

        bnb_rets = {}
        for lb in [3, 5, 7, 10, 15, 20, 30]:
            bnb_rets[lb] = get_return(bnb_rec["klines_1s"], cutoff_ms, lb)

        btc_rets = {}
        btc_rec = btc_by_epoch.get(epoch)
        if btc_rec:
            for lb in [3, 5, 7, 10, 15, 20, 30, 45, 60]:
                btc_rets[lb] = get_return(btc_rec["klines_1s"], cutoff_ms, lb)

        eth_rets = {}
        eth_rec = eth_by_epoch.get(epoch)
        if eth_rec:
            for lb in [5, 7, 10, 15, 20, 30]:
                eth_rets[lb] = get_return(eth_rec["klines_1s"], cutoff_ms, lb)

        # Pre-compute useful flags
        bnb_any_move = any(
            bnb_rets.get(lb) is not None and bnb_rets[lb] != 0
            for lb in [5, 7, 10]
        )

        # Payout multiple at small bet
        pm_bull = payout_multiple(bull_wei, bear_wei, "Bull", 0.001)
        pm_bear = payout_multiple(bull_wei, bear_wei, "Bear", 0.001)

        rounds.append({
            "epoch": epoch,
            "outcome": rnd["position"],
            "bull_wei": bull_wei, "bear_wei": bear_wei,
            "pool_bnb": pool_bnb,
            "pm_bull": pm_bull, "pm_bear": pm_bear,
            "bnb_rets": bnb_rets,
            "btc_rets": btc_rets,
            "eth_rets": eth_rets,
            "bnb_any_move": bnb_any_move,
            "has_btc": bool(btc_rets),
            "has_eth": bool(eth_rets),
        })
    return rounds


# ===================================================================
# Signal functions — each returns (direction, confidence) or (None, 0)
# ===================================================================

def sig_bnb_accel(r, short=7, long=10, thresh=0.0002):
    """BNB acceleration: short+long agree, max >= thresh."""
    rs = r["bnb_rets"].get(short)
    rl = r["bnb_rets"].get(long)
    if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
        if max(abs(rs), abs(rl)) >= thresh:
            strength = max(abs(rs), abs(rl))
            return ("Bull" if rs > 0 else "Bear", strength)
    return (None, 0)


def sig_btc_accel(r, short=5, long=30, thresh=0.0003):
    """BTC acceleration signal."""
    rs = r["btc_rets"].get(short)
    rl = r["btc_rets"].get(long)
    if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
        if max(abs(rs), abs(rl)) >= thresh:
            return ("Bull" if rs > 0 else "Bear", max(abs(rs), abs(rl)))
    return (None, 0)


def sig_bnb_any_btc_confirm(r, bnb_lb=7, btc_lb=30, btc_thresh=0.0003):
    """Any nonzero BNB return + BTC agreement."""
    bnb_r = r["bnb_rets"].get(bnb_lb)
    if bnb_r is None or bnb_r == 0:
        return (None, 0)
    bnb_dir = "Bull" if bnb_r > 0 else "Bear"
    btc_r = r["btc_rets"].get(btc_lb)
    if btc_r is None or btc_r == 0 or abs(btc_r) < btc_thresh:
        return (None, 0)
    btc_dir = "Bull" if btc_r > 0 else "Bear"
    if bnb_dir != btc_dir:
        return (None, 0)
    return (bnb_dir, abs(bnb_r) + abs(btc_r))


def sig_btc_multi_tf(r, lookbacks=(5, 10, 20), min_agree=3, thresh=0.0001):
    """Multiple BTC timeframes agree."""
    dirs = []
    for lb in lookbacks:
        ret = r["btc_rets"].get(lb)
        if ret is not None and ret != 0 and abs(ret) >= thresh:
            dirs.append("Bull" if ret > 0 else "Bear")
    if len(dirs) < min_agree:
        return (None, 0)
    bull_ct = sum(1 for d in dirs if d == "Bull")
    bear_ct = len(dirs) - bull_ct
    if bull_ct >= min_agree:
        return ("Bull", bull_ct / len(dirs))
    elif bear_ct >= min_agree:
        return ("Bear", bear_ct / len(dirs))
    return (None, 0)


def sig_bnb_micro_btc(r, bnb_lb=7, bnb_thresh=0.00005, btc_lb=20, btc_thresh=0.0002):
    """Very small BNB move + BTC confirmation (lower BNB bar than accel)."""
    bnb_r = r["bnb_rets"].get(bnb_lb)
    if bnb_r is None or abs(bnb_r) < bnb_thresh:
        return (None, 0)
    bnb_dir = "Bull" if bnb_r > 0 else "Bear"
    btc_r = r["btc_rets"].get(btc_lb)
    if btc_r is None or abs(btc_r) < btc_thresh:
        return (None, 0)
    if ("Bull" if btc_r > 0 else "Bear") != bnb_dir:
        return (None, 0)
    return (bnb_dir, abs(bnb_r) + abs(btc_r))


# ===================================================================
# Sizing functions
# ===================================================================

def size_flat(r, direction, base=0.10):
    return base


def size_pool(r, direction, pct=0.05, floor=0.05, cap=0.40):
    return min(cap, max(floor, r["pool_bnb"] * pct))


def size_pool_payout(r, direction, pct=0.05, floor=0.05, cap=0.40):
    bet = max(floor, r["pool_bnb"] * pct)
    pm = r["pm_bull"] if direction == "Bull" else r["pm_bear"]
    if pm >= 2.0:
        bet *= 1.3
    elif pm < 1.7:
        bet *= 0.6
    return min(cap, bet)


def size_full(r, direction, base_pct=0.05, cap=0.50, btc_agrees=False,
              payout_boost=True):
    bet = max(0.05, r["pool_bnb"] * base_pct)
    if payout_boost:
        pm = r["pm_bull"] if direction == "Bull" else r["pm_bear"]
        if pm >= 2.0:
            bet *= 1.3
        elif pm < 1.7:
            bet *= 0.6
    if btc_agrees:
        bet *= 1.5
    return min(cap, bet)


def eval_strategy(rounds, signal_fn, size_fn, label, skip_epochs=None):
    """Evaluate a signal+sizing strategy. Returns dict with stats."""
    bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
    bet_epochs = set()
    for r in rounds:
        if skip_epochs and r["epoch"] in skip_epochs:
            continue
        d, conf = signal_fn(r)
        if d is None:
            continue
        bs = size_fn(r, d)
        if bs < 0.001:
            continue
        bets += 1
        wagered += bs
        bet_epochs.add(r["epoch"])
        if d == r["outcome"]:
            wins += 1
        pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
    if bets == 0:
        return None
    return {
        "label": label,
        "bets": bets,
        "wins": wins,
        "wr": wins / bets,
        "wagered": wagered,
        "pnl": pnl,
        "roi": pnl / wagered * 100,
        "avg_bet": wagered / bets,
        "pnl_per_bet": pnl / bets,
        "epochs": bet_epochs,
    }


def print_result(res, total_rounds=5000):
    if res is None:
        return
    proj = res["pnl"] / len(res["epochs"]) * total_rounds if total_rounds else res["pnl"]
    print(f"  {res['label']:>55}: bets={res['bets']:>5} WR={res['wr']:.1%} "
          f"avg={res['avg_bet']:.3f} PnL={res['pnl']:+.3f} ROI={res['roi']:+.1f}%")


def main():
    bnb_by_epoch, btc_by_epoch, eth_by_epoch, rounds_by_epoch = load_data()
    rounds = build_rounds(bnb_by_epoch, btc_by_epoch, eth_by_epoch, rounds_by_epoch)
    total = len(rounds)
    with_btc = sum(1 for r in rounds if r["has_btc"])
    with_eth = sum(1 for r in rounds if r["has_eth"])
    print(f"Total valid rounds: {total}")
    print(f"  With BTC data: {with_btc}")
    print(f"  With ETH data: {with_eth}")
    print()

    # =========================================================
    # SECTION 1: Individual signal evaluation
    # =========================================================
    print("=" * 85)
    print("SECTION 1: INDIVIDUAL SIGNALS (flat 0.10 bet)")
    print("=" * 85)

    signals = [
        # BNB acceleration variants
        ("BNB accel 7+10 @0.0002", lambda r: sig_bnb_accel(r, 7, 10, 0.0002)),
        ("BNB accel 5+10 @0.0002", lambda r: sig_bnb_accel(r, 5, 10, 0.0002)),
        ("BNB accel 7+10 @0.00015", lambda r: sig_bnb_accel(r, 7, 10, 0.00015)),
        ("BNB accel 5+7 @0.0002", lambda r: sig_bnb_accel(r, 5, 7, 0.0002)),
        ("BNB accel 3+7 @0.0002", lambda r: sig_bnb_accel(r, 3, 7, 0.0002)),
        ("BNB accel 3+10 @0.0002", lambda r: sig_bnb_accel(r, 3, 10, 0.0002)),
        ("BNB accel 7+15 @0.0002", lambda r: sig_bnb_accel(r, 7, 15, 0.0002)),
        ("BNB accel 7+20 @0.0002", lambda r: sig_bnb_accel(r, 7, 20, 0.0002)),
        # BTC acceleration variants
        ("BTC accel 5+30 @0.0003", lambda r: sig_btc_accel(r, 5, 30, 0.0003)),
        ("BTC accel 7+30 @0.0003", lambda r: sig_btc_accel(r, 7, 30, 0.0003)),
        ("BTC accel 5+20 @0.0003", lambda r: sig_btc_accel(r, 5, 20, 0.0003)),
        ("BTC accel 10+30 @0.0003", lambda r: sig_btc_accel(r, 10, 30, 0.0003)),
        ("BTC accel 5+30 @0.0002", lambda r: sig_btc_accel(r, 5, 30, 0.0002)),
        ("BTC accel 5+30 @0.0004", lambda r: sig_btc_accel(r, 5, 30, 0.0004)),
        ("BTC accel 5+45 @0.0003", lambda r: sig_btc_accel(r, 5, 45, 0.0003)),
        ("BTC accel 5+60 @0.0003", lambda r: sig_btc_accel(r, 5, 60, 0.0003)),
        # BNB any + BTC confirmation
        ("BNB any(7s) + BTC(30s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 7, 30, 0.0003)),
        ("BNB any(7s) + BTC(20s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 7, 20, 0.0003)),
        ("BNB any(7s) + BTC(30s@0.0002)", lambda r: sig_bnb_any_btc_confirm(r, 7, 30, 0.0002)),
        ("BNB any(10s) + BTC(30s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 10, 30, 0.0003)),
        ("BNB any(5s) + BTC(30s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 5, 30, 0.0003)),
        ("BNB any(7s) + BTC(10s@0.0002)", lambda r: sig_bnb_any_btc_confirm(r, 7, 10, 0.0002)),
        ("BNB any(7s) + BTC(45s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 7, 45, 0.0003)),
        # BTC multi-timeframe
        ("BTC multi 5+10+20 agree=3 @0.0001", lambda r: sig_btc_multi_tf(r, (5,10,20), 3, 0.0001)),
        ("BTC multi 5+10+20+30 agree=3 @0.0001", lambda r: sig_btc_multi_tf(r, (5,10,20,30), 3, 0.0001)),
        ("BTC multi 5+10+20+30 agree=4 @0.0001", lambda r: sig_btc_multi_tf(r, (5,10,20,30), 4, 0.0001)),
        ("BTC multi 5+10+30+60 agree=3 @0.0001", lambda r: sig_btc_multi_tf(r, (5,10,30,60), 3, 0.0001)),
        ("BTC multi 5+10+30+60 agree=4 @0.0001", lambda r: sig_btc_multi_tf(r, (5,10,30,60), 4, 0.0001)),
        # BNB micro + BTC
        ("BNB micro(7s@5e-5) + BTC(20s@0.0002)", lambda r: sig_bnb_micro_btc(r, 7, 5e-5, 20, 0.0002)),
        ("BNB micro(7s@5e-5) + BTC(30s@0.0003)", lambda r: sig_bnb_micro_btc(r, 7, 5e-5, 30, 0.0003)),
        ("BNB micro(10s@5e-5) + BTC(30s@0.0003)", lambda r: sig_bnb_micro_btc(r, 10, 5e-5, 30, 0.0003)),
    ]

    results = []
    for label, sig_fn in signals:
        res = eval_strategy(rounds, sig_fn, lambda r, d: 0.10, label)
        if res and res["bets"] >= 20:
            results.append(res)
            print_result(res)

    # =========================================================
    # SECTION 2: Best signals with smart sizing
    # =========================================================
    print("\n" + "=" * 85)
    print("SECTION 2: TOP SIGNALS + SMART SIZING")
    print("=" * 85)

    # Sort by PnL
    results.sort(key=lambda r: r["pnl"], reverse=True)
    top_signals = results[:8]
    print("Top signals by PnL:")
    for r in top_signals:
        print(f"  {r['label']}: PnL={r['pnl']:+.3f}")
    print()

    # Re-test top signals with different sizing
    sizing_options = [
        ("flat 0.10", lambda r, d: 0.10),
        ("pool 5% cap 0.40", lambda r, d: size_pool(r, d, 0.05, 0.05, 0.40)),
        ("pool 5% cap 0.50", lambda r, d: size_pool(r, d, 0.05, 0.05, 0.50)),
        ("pool+pay 5% cap 0.40", lambda r, d: size_pool_payout(r, d, 0.05, 0.05, 0.40)),
        ("pool+pay 5% cap 0.50", lambda r, d: size_pool_payout(r, d, 0.05, 0.05, 0.50)),
        ("pool 7% cap 0.50", lambda r, d: size_pool(r, d, 0.07, 0.05, 0.50)),
    ]

    for label, sig_fn in signals[:6]:  # Top BNB signals
        print(f"\n  Signal: {label}")
        for sz_label, sz_fn in sizing_options:
            res = eval_strategy(rounds, sig_fn, sz_fn, f"{label} | {sz_label}")
            if res:
                print(f"    {sz_label:>25}: bets={res['bets']:>5} WR={res['wr']:.1%} "
                      f"PnL={res['pnl']:+.3f} ROI={res['roi']:+.1f}%")

    # =========================================================
    # SECTION 3: STACKED STRATEGIES (non-overlapping rounds)
    # =========================================================
    print("\n" + "=" * 85)
    print("SECTION 3: STACKED STRATEGIES (additive, non-overlapping)")
    print("=" * 85)

    # Layer 1: BNB acceleration (highest WR signal)
    def sig_bnb_any_accel(r):
        for short, long, thresh in [(7, 10, 0.0002), (5, 10, 0.0002)]:
            d, c = sig_bnb_accel(r, short, long, thresh)
            if d:
                return (d, c)
        return (None, 0)

    layer1 = eval_strategy(rounds, sig_bnb_any_accel,
                          lambda r, d: size_pool_payout(r, d, 0.05, 0.05, 0.40),
                          "L1: BNB accel (pool+pay)")
    print(f"\n  Layer 1 (BNB accel + pool+pay sizing):")
    if layer1:
        print(f"    bets={layer1['bets']} WR={layer1['wr']:.1%} "
              f"PnL={layer1['pnl']:+.3f} avg_bet={layer1['avg_bet']:.3f}")

    # Layer 2 options: signals on NON-layer1 rounds
    l1_epochs = layer1["epochs"] if layer1 else set()

    print(f"\n  Layer 2 candidates (non-overlapping with L1, {len(l1_epochs)} epochs):")
    layer2_options = [
        ("BTC accel 5+30@0.0003", lambda r: sig_btc_accel(r, 5, 30, 0.0003)),
        ("BTC accel 7+30@0.0003", lambda r: sig_btc_accel(r, 7, 30, 0.0003)),
        ("BTC accel 5+30@0.0002", lambda r: sig_btc_accel(r, 5, 30, 0.0002)),
        ("BTC accel 5+45@0.0003", lambda r: sig_btc_accel(r, 5, 45, 0.0003)),
        ("BTC accel 5+60@0.0003", lambda r: sig_btc_accel(r, 5, 60, 0.0003)),
        ("BNB any(7s)+BTC(30s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 7, 30, 0.0003)),
        ("BNB any(7s)+BTC(30s@0.0002)", lambda r: sig_bnb_any_btc_confirm(r, 7, 30, 0.0002)),
        ("BNB any(7s)+BTC(20s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 7, 20, 0.0003)),
        ("BNB any(7s)+BTC(45s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 7, 45, 0.0003)),
        ("BNB micro(7s)+BTC(20s@0.0002)", lambda r: sig_bnb_micro_btc(r, 7, 5e-5, 20, 0.0002)),
        ("BNB micro(7s)+BTC(30s@0.0003)", lambda r: sig_bnb_micro_btc(r, 7, 5e-5, 30, 0.0003)),
        ("BTC multi 5+10+20 agree=3", lambda r: sig_btc_multi_tf(r, (5,10,20), 3, 0.0001)),
        ("BTC multi 5+10+20+30 agree=3", lambda r: sig_btc_multi_tf(r, (5,10,20,30), 3, 0.0001)),
        ("BTC multi 5+10+20+30 agree=4", lambda r: sig_btc_multi_tf(r, (5,10,20,30), 4, 0.0001)),
    ]

    l2_results = []
    for label, sig_fn in layer2_options:
        for sz_label, sz_fn in [
            ("flat 0.10", lambda r, d: 0.10),
            ("pool+pay 5%", lambda r, d: size_pool_payout(r, d, 0.05, 0.05, 0.40)),
        ]:
            res = eval_strategy(rounds, sig_fn, sz_fn,
                              f"L2: {label} | {sz_label}",
                              skip_epochs=l1_epochs)
            if res and res["bets"] >= 20:
                l2_results.append(res)
                print(f"    {label:>40} {sz_label:>15}: "
                      f"bets={res['bets']:>5} WR={res['wr']:.1%} "
                      f"PnL={res['pnl']:+.3f}")

    # =========================================================
    # SECTION 4: COMBINED STACKED PnL
    # =========================================================
    print("\n" + "=" * 85)
    print("SECTION 4: BEST COMBINED STACKS")
    print("=" * 85)

    # Sort L2 by PnL
    l2_results.sort(key=lambda r: r["pnl"], reverse=True)

    # Try combining L1 with each L2
    if layer1:
        for l2 in l2_results[:15]:
            total_pnl = layer1["pnl"] + l2["pnl"]
            total_bets = layer1["bets"] + l2["bets"]
            total_wagered = layer1["wagered"] + l2["wagered"]
            combined_wr = (layer1["wins"] + l2["wins"]) / total_bets
            # Check epoch overlap
            overlap = layer1["epochs"] & l2["epochs"]
            proj_5k = total_pnl / total * 5000 if total > 0 else total_pnl
            print(f"  L1+{l2['label'][4:]:>55}: "
                  f"bets={total_bets:>5} WR={combined_wr:.1%} "
                  f"PnL={total_pnl:+.3f} proj5k={proj_5k:+.1f}")

    # =========================================================
    # SECTION 5: TRIPLE STACK (L1 + L2 + L3)
    # =========================================================
    print("\n" + "=" * 85)
    print("SECTION 5: TRIPLE STACK (L1 + best L2 + L3)")
    print("=" * 85)

    if l2_results and layer1:
        best_l2 = l2_results[0]
        l12_epochs = layer1["epochs"] | best_l2["epochs"]
        print(f"  L1: {layer1['label']} => {layer1['pnl']:+.3f}")
        print(f"  L2: {best_l2['label']} => {best_l2['pnl']:+.3f}")
        print(f"  L1+L2 epochs: {len(l12_epochs)}")
        print()

        # L3: anything left
        l3_signals = [
            ("BTC accel 5+30@0.0003", lambda r: sig_btc_accel(r, 5, 30, 0.0003)),
            ("BTC accel 5+30@0.0002", lambda r: sig_btc_accel(r, 5, 30, 0.0002)),
            ("BTC accel 5+45@0.0003", lambda r: sig_btc_accel(r, 5, 45, 0.0003)),
            ("BTC multi 5+10+20 agree=3", lambda r: sig_btc_multi_tf(r, (5,10,20), 3, 0.0001)),
            ("BTC multi 5+10+30+60 agree=3", lambda r: sig_btc_multi_tf(r, (5,10,30,60), 3, 0.0001)),
            ("BNB any(7s)+BTC(20s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 7, 20, 0.0003)),
            ("BNB any(7s)+BTC(45s@0.0003)", lambda r: sig_bnb_any_btc_confirm(r, 7, 45, 0.0003)),
            ("BNB micro(7s)+BTC(20s@0.0002)", lambda r: sig_bnb_micro_btc(r, 7, 5e-5, 20, 0.0002)),
        ]
        for label, sig_fn in l3_signals:
            res = eval_strategy(rounds, sig_fn,
                              lambda r, d: size_pool_payout(r, d, 0.05, 0.05, 0.40),
                              f"L3: {label}",
                              skip_epochs=l12_epochs)
            if res and res["bets"] >= 10:
                total = layer1["pnl"] + best_l2["pnl"] + res["pnl"]
                total_bets = layer1["bets"] + best_l2["bets"] + res["bets"]
                print(f"    L3: {label:>40}: +{res['bets']} bets, "
                      f"WR={res['wr']:.1%} PnL={res['pnl']:+.3f}  "
                      f"TOTAL={total:+.3f} ({total_bets} bets)")

    # =========================================================
    # SECTION 6: UNIFIED PRIORITY SIGNAL (smart dispatch)
    # =========================================================
    print("\n" + "=" * 85)
    print("SECTION 6: UNIFIED PRIORITY SIGNAL")
    print("   Per-round: pick best signal, size by confidence")
    print("=" * 85)

    # For each round, try signals in priority order
    # Higher confidence -> bigger bet
    def unified_signal(r, config):
        """Try signals in priority order, return (direction, tier, base_bet)."""
        # Tier 1: BNB accel (highest WR)
        d, c = sig_bnb_any_accel(r)
        if d:
            # Check BTC agreement for confidence boost
            btc_d, btc_c = sig_btc_accel(r, 5, 30, 0.0003)
            if btc_d == d:
                return (d, "bnb_accel+btc", config.get("t1_btc", 0.15))
            elif btc_d and btc_d != d:
                return (d, "bnb_accel-btc", config.get("t1_nobtc", 0.05))
            return (d, "bnb_accel", config.get("t1_base", 0.10))

        # Tier 2: BNB any move + BTC confirmation
        if r["has_btc"]:
            for bnb_lb, btc_lb, btc_th in [(7, 30, 0.0003), (7, 20, 0.0003),
                                             (10, 30, 0.0003)]:
                d, c = sig_bnb_any_btc_confirm(r, bnb_lb, btc_lb, btc_th)
                if d:
                    return (d, "bnb_any+btc", config.get("t2", 0.08))

        # Tier 3: BTC-only accel
        if r["has_btc"]:
            for s, l, th in [(5, 30, 0.0003), (7, 30, 0.0003), (5, 45, 0.0003)]:
                d, c = sig_btc_accel(r, s, l, th)
                if d:
                    return (d, "btc_only", config.get("t3", 0.06))

        # Tier 4: BTC multi-tf consensus
        if r["has_btc"]:
            d, c = sig_btc_multi_tf(r, (5, 10, 20, 30), 4, 0.0001)
            if d:
                return (d, "btc_multi4", config.get("t4", 0.05))

        return (None, None, 0)

    configs = [
        {"name": "conservative",
         "t1_btc": 0.15, "t1_base": 0.10, "t1_nobtc": 0.05,
         "t2": 0.08, "t3": 0.05, "t4": 0.05},
        {"name": "moderate",
         "t1_btc": 0.20, "t1_base": 0.12, "t1_nobtc": 0.07,
         "t2": 0.10, "t3": 0.07, "t4": 0.05},
        {"name": "aggressive",
         "t1_btc": 0.30, "t1_base": 0.15, "t1_nobtc": 0.10,
         "t2": 0.12, "t3": 0.08, "t4": 0.06},
        {"name": "t1_heavy",
         "t1_btc": 0.40, "t1_base": 0.20, "t1_nobtc": 0.10,
         "t2": 0.10, "t3": 0.06, "t4": 0.05},
    ]

    for config in configs:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        tier_stats = {}
        for r in rounds:
            d, tier, base = unified_signal(r, config)
            if d is None:
                continue
            # Apply pool sizing on top
            pool_mult = min(2.0, max(0.5, r["pool_bnb"] / 3.0))
            bs = base * pool_mult
            # Payout boost
            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
            if pm >= 2.0:
                bs *= 1.2
            elif pm < 1.6:
                bs *= 0.7
            bs = min(0.50, max(0.03, bs))

            bets += 1
            wagered += bs
            won = d == r["outcome"]
            if won:
                wins += 1
            pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
            tier_stats.setdefault(tier, [0, 0, 0.0])
            tier_stats[tier][0] += 1
            tier_stats[tier][1] += 1 if won else 0
            tier_stats[tier][2] += net_profit(r["bull_wei"], r["bear_wei"],
                                               d, r["outcome"], bs)

        if bets > 0:
            proj = pnl / total * 5000
            print(f"\n  {config['name'].upper()}: bets={bets} WR={wins/bets:.1%} "
                  f"avg={wagered/bets:.3f} PnL={pnl:+.3f} proj5k={proj:+.1f}")
            for tier, (n, w, p) in sorted(tier_stats.items(),
                                           key=lambda x: -x[1][0]):
                print(f"    {tier:>20}: n={n:>4} WR={w/n:.1%} PnL={p:+.3f}")

    # =========================================================
    # SECTION 7: BANKROLL SIMULATION with best unified
    # =========================================================
    print("\n" + "=" * 85)
    print("SECTION 7: BANKROLL SIMULATION (50 BNB start)")
    print("=" * 85)

    for config in configs:
        bankroll = 50.0
        peak = 50.0
        max_dd = 0.0
        bets, wins, pnl = 0, 0, 0.0
        for r in rounds:
            d, tier, base = unified_signal(r, config)
            if d is None:
                continue
            pool_mult = min(2.0, max(0.5, r["pool_bnb"] / 3.0))
            bs = base * pool_mult
            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
            if pm >= 2.0:
                bs *= 1.2
            elif pm < 1.6:
                bs *= 0.7
            bs = min(0.50, max(0.03, bs))
            # Don't bet more than 2% of bankroll
            bs = min(bs, bankroll * 0.02)
            if bs < 0.01:
                continue

            bets += 1
            won = d == r["outcome"]
            if won:
                wins += 1
            p = net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
            pnl += p
            bankroll += p
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak
            if dd > max_dd:
                max_dd = dd

        print(f"  {config['name']:>15}: bets={bets} WR={wins/bets:.1%} "
              f"PnL={pnl:+.3f} final={bankroll:.2f} maxDD={max_dd:.1%}")

    # =========================================================
    # SECTION 8: PROJECTION & GAP ANALYSIS
    # =========================================================
    print("\n" + "=" * 85)
    print("SECTION 8: PROJECTION & GAP ANALYSIS")
    print("=" * 85)

    # What if we had full BTC data?
    btc_coverage = with_btc / total if total > 0 else 0
    print(f"\n  BTC coverage: {with_btc}/{total} = {btc_coverage:.1%}")
    print(f"  If full BTC data were available:")

    for config in configs:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for r in rounds:
            d, tier, base = unified_signal(r, config)
            if d is None:
                continue
            pool_mult = min(2.0, max(0.5, r["pool_bnb"] / 3.0))
            bs = base * pool_mult
            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
            if pm >= 2.0:
                bs *= 1.2
            elif pm < 1.6:
                bs *= 0.7
            bs = min(0.50, max(0.03, bs))
            bets += 1
            wagered += bs
            if d == r["outcome"]:
                wins += 1
            pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)

        if bets > 0:
            # Project BTC-dependent bets to full coverage
            # BNB-only bets stay same, BTC bets scale by 1/coverage
            btc_dep_pnl = 0
            bnb_only_pnl = 0
            btc_dep_bets = 0
            bnb_only_bets = 0
            for r in rounds:
                d, tier, base = unified_signal(r, config)
                if d is None:
                    continue
                pool_mult = min(2.0, max(0.5, r["pool_bnb"] / 3.0))
                bs = base * pool_mult
                pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                if pm >= 2.0:
                    bs *= 1.2
                elif pm < 1.6:
                    bs *= 0.7
                bs = min(0.50, max(0.03, bs))
                p = net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
                if tier and "btc" in tier.lower():
                    btc_dep_pnl += p
                    btc_dep_bets += 1
                else:
                    bnb_only_pnl += p
                    bnb_only_bets += 1

            if btc_coverage > 0 and btc_dep_bets > 0:
                proj_btc_pnl = btc_dep_pnl / btc_coverage
                proj_total = bnb_only_pnl + proj_btc_pnl
                proj_bets = bnb_only_bets + int(btc_dep_bets / btc_coverage)
                print(f"    {config['name']:>15}: bnb_only={bnb_only_pnl:+.2f} "
                      f"btc_proj={proj_btc_pnl:+.2f} total_proj={proj_total:+.2f} "
                      f"(~{proj_bets} bets)")


if __name__ == "__main__":
    main()
