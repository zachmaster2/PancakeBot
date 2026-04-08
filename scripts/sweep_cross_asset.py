"""Sweep cross-asset signals: BTC/USDT momentum predicting BNB round outcomes.

Hypothesis: BTC is more liquid than BNB, moves first, and BNB follows.
If BTC shows clear momentum at cutoff time, BNB will likely follow,
even when BNB's own 1s klines show zero movement.

This could dramatically increase bet count (the main bottleneck).

Signals tested:
1. BTC momentum alone (does BTC predict BNB round outcomes?)
2. BTC as fallback (use BNB signal when available, BTC otherwise)
3. BTC + BNB agreement (both must agree for higher confidence)
4. BTC acceleration (short+long BTC returns agree)
5. Combined: BNB acceleration + BTC fallback for no-signal rounds
"""

from __future__ import annotations

import json
from pathlib import Path

BNB_DATA_PATH = Path("var/cutoff_spot_prices.jsonl")
BTC_DATA_PATH = Path("var/btc_spot_prices.jsonl")
ROUNDS_PATH = Path("var/closed_rounds.jsonl")

BNB_WEI = 10**18
BET = 0.05
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


def payout_multiple(bull_wei, bear_wei, side, bet_bnb=0.05):
    bet_wei = int(bet_bnb * BNB_WEI)
    bw = bull_wei + (bet_wei if side == "Bull" else 0)
    ew = bear_wei + (bet_wei if side == "Bear" else 0)
    tw = bw + ew
    my = bw if side == "Bull" else ew
    return (tw * (1 - FEE)) / my if my > 0 else 0


def net_profit(bw, ew, side, outcome, bet_bnb=0.05):
    m = payout_multiple(bw, ew, side, bet_bnb)
    return bet_bnb * m - GAS_CLAIM - bet_bnb - GAS_BET if outcome == side else -bet_bnb - GAS_BET


def get_return(klines, cutoff_ms, lookback_s):
    kn = find_closest(klines, cutoff_ms)
    ka = find_closest(klines, cutoff_ms - lookback_s * 1000)
    if not kn or not ka or ka[4] <= 0:
        return None
    return (kn[4] / ka[4]) - 1


def main():
    # Load BNB data
    bnb_by_epoch = {}
    for line in BNB_DATA_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                bnb_by_epoch[r["epoch"]] = r

    # Load BTC data
    btc_by_epoch = {}
    for line in BTC_DATA_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                btc_by_epoch[r["epoch"]] = r

    # Load rounds
    rounds_by_epoch = {}
    for line in ROUNDS_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rounds_by_epoch[r["epoch"]] = r

    # Build features — only rounds where we have both BNB and BTC data
    features = []
    for epoch, bnb_rec in bnb_by_epoch.items():
        btc_rec = btc_by_epoch.get(epoch)
        rnd = rounds_by_epoch.get(epoch)
        if btc_rec is None or rnd is None:
            continue
        if rnd.get("failed") or rnd["position"] not in ("Bull", "Bear"):
            continue

        lock_ms = bnb_rec["lock_at"] * 1000
        cutoff_ms = lock_ms - CUTOFF_SECONDS * 1000
        bull_wei, bear_wei = compute_pools(rnd)
        if bull_wei + bear_wei == 0:
            continue

        feat = {
            "epoch": epoch,
            "outcome": rnd["position"],
            "bull_wei": bull_wei,
            "bear_wei": bear_wei,
            "cutoff_ms": cutoff_ms,
            "bnb_klines": bnb_rec["klines_1s"],
            "btc_klines": btc_rec["klines_1s"],
        }

        # Pre-compute returns
        for lb in [3, 5, 7, 10, 15, 20, 30, 45, 60]:
            feat[f"bnb_ret_{lb}"] = get_return(bnb_rec["klines_1s"], cutoff_ms, lb)
            feat[f"btc_ret_{lb}"] = get_return(btc_rec["klines_1s"], cutoff_ms, lb)

        features.append(feat)

    print(f"Rounds with both BNB + BTC data: {len(features)}\n")

    if len(features) < 100:
        print("Not enough data yet. Wait for BTC fetch to complete.")
        return

    # =========================================================
    # 0. BTC SIGNAL FIRING RATE vs BNB
    # =========================================================
    print("=" * 85)
    print("0. SIGNAL FIRING RATES: BTC vs BNB")
    print("=" * 85)
    for lb in [5, 7, 10, 15, 20, 30, 60]:
        bnb_fires = sum(1 for f in features if f.get(f"bnb_ret_{lb}") is not None
                        and abs(f[f"bnb_ret_{lb}"]) > 0.0003)
        btc_fires = sum(1 for f in features if f.get(f"btc_ret_{lb}") is not None
                        and abs(f[f"btc_ret_{lb}"]) > 0.0003)
        bnb_nonzero = sum(1 for f in features if f.get(f"bnb_ret_{lb}") is not None
                          and f[f"bnb_ret_{lb}"] != 0)
        btc_nonzero = sum(1 for f in features if f.get(f"btc_ret_{lb}") is not None
                          and f[f"btc_ret_{lb}"] != 0)
        print(f"  lb={lb:>2}s: BNB nonzero={bnb_nonzero:>5} ({bnb_nonzero/len(features)*100:.1f}%)  "
              f"BTC nonzero={btc_nonzero:>5} ({btc_nonzero/len(features)*100:.1f}%)  |  "
              f"BNB>0.0003={bnb_fires:>5} ({bnb_fires/len(features)*100:.1f}%)  "
              f"BTC>0.0003={btc_fires:>5} ({btc_fires/len(features)*100:.1f}%)")
    print()

    # =========================================================
    # 1. BTC MOMENTUM ALONE
    # =========================================================
    print("=" * 85)
    print("1. BTC MOMENTUM ALONE: does BTC direction predict BNB round outcome?")
    print("=" * 85)

    results1 = []
    for lb in [5, 7, 10, 15, 20, 30, 60]:
        for thresh in [0.0, 0.0001, 0.0003, 0.0005, 0.001]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                ret = f.get(f"btc_ret_{lb}")
                if ret is None or ret == 0:
                    continue
                if abs(ret) < thresh:
                    continue
                direction = "Bull" if ret > 0 else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 30:
                results1.append({
                    "lb": lb, "thresh": thresh,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                })

    results1.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'lb':>3} {'thresh':>7} {'bets':>5} {'wins':>5} {'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 55)
    for r in results1[:20]:
        print(f"{r['lb']:>2}s {r['thresh']:>7.4f} {r['bets']:>5} {r['wins']:>5} "
              f"{r['wr']:>6.1%} {r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 2. BTC AS FALLBACK: use BNB when signal fires, BTC otherwise
    # =========================================================
    print("\n" + "=" * 85)
    print("2. BTC AS FALLBACK: BNB acceleration first, BTC when BNB is flat")
    print("=" * 85)

    results2 = []
    for bnb_short, bnb_long, bnb_thresh in [(7, 10, 0.0002), (5, 10, 0.0002)]:
        for btc_lb in [5, 7, 10, 15, 20, 30]:
            for btc_thresh in [0.0003, 0.0005, 0.001]:
                bets, wins, pnl = 0, 0, 0.0
                bnb_fired, btc_fired = 0, 0
                for f in features:
                    direction = None
                    # Try BNB acceleration first
                    rs = f.get(f"bnb_ret_{bnb_short}")
                    rl = f.get(f"bnb_ret_{bnb_long}")
                    if (rs is not None and rl is not None and rs != 0 and rl != 0
                            and (rs > 0) == (rl > 0)
                            and max(abs(rs), abs(rl)) >= bnb_thresh):
                        direction = "Bull" if rs > 0 else "Bear"
                        bnb_fired += 1
                    else:
                        # Fall back to BTC
                        btc_ret = f.get(f"btc_ret_{btc_lb}")
                        if btc_ret is not None and abs(btc_ret) >= btc_thresh:
                            direction = "Bull" if btc_ret > 0 else "Bear"
                            btc_fired += 1
                    if direction is None:
                        continue
                    bets += 1
                    if direction == f["outcome"]:
                        wins += 1
                    pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
                if bets >= 50:
                    results2.append({
                        "bnb": f"{bnb_short}+{bnb_long}@{bnb_thresh:.4f}",
                        "btc": f"{btc_lb}s@{btc_thresh:.4f}",
                        "bets": bets, "bnb_fired": bnb_fired, "btc_fired": btc_fired,
                        "wins": wins, "wr": wins / bets,
                        "pnl": pnl, "ppb": pnl / bets,
                    })

    results2.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'bnb_sig':>18} {'btc_sig':>12} {'bets':>5} {'bnb':>4} {'btc':>4} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 80)
    for r in results2[:20]:
        print(f"{r['bnb']:>18} {r['btc']:>12} {r['bets']:>5} {r['bnb_fired']:>4} {r['btc_fired']:>4} "
              f"{r['wr']:>6.1%} {r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 3. BTC + BNB AGREEMENT: both must agree for high confidence
    # =========================================================
    print("\n" + "=" * 85)
    print("3. BTC + BNB AGREEMENT: both assets' momentum must agree")
    print("=" * 85)

    results3 = []
    for bnb_lb in [5, 7, 10]:
        for btc_lb in [5, 7, 10, 15, 20]:
            for min_thresh in [0.0, 0.0001, 0.0003]:
                bets, wins, pnl = 0, 0, 0.0
                for f in features:
                    bnb_ret = f.get(f"bnb_ret_{bnb_lb}")
                    btc_ret = f.get(f"btc_ret_{btc_lb}")
                    if bnb_ret is None or btc_ret is None:
                        continue
                    if bnb_ret == 0 or btc_ret == 0:
                        continue
                    if (bnb_ret > 0) != (btc_ret > 0):
                        continue  # disagree
                    if max(abs(bnb_ret), abs(btc_ret)) < min_thresh:
                        continue
                    direction = "Bull" if bnb_ret > 0 else "Bear"
                    bets += 1
                    if direction == f["outcome"]:
                        wins += 1
                    pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
                if bets >= 30:
                    results3.append({
                        "bnb_lb": bnb_lb, "btc_lb": btc_lb, "thresh": min_thresh,
                        "bets": bets, "wins": wins,
                        "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                    })

    results3.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'bnb':>4} {'btc':>4} {'thresh':>7} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 60)
    for r in results3[:20]:
        print(f"{r['bnb_lb']:>3}s {r['btc_lb']:>3}s {r['thresh']:>7.4f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 4. BTC ACCELERATION (short+long BTC agreement)
    # =========================================================
    print("\n" + "=" * 85)
    print("4. BTC ACCELERATION: short + long BTC returns agree")
    print("=" * 85)

    results4 = []
    for short_lb in [5, 7, 10]:
        for long_lb in [15, 20, 30, 45, 60]:
            for thresh in [0.0, 0.0003, 0.0005]:
                bets, wins, pnl = 0, 0, 0.0
                for f in features:
                    rs = f.get(f"btc_ret_{short_lb}")
                    rl = f.get(f"btc_ret_{long_lb}")
                    if rs is None or rl is None or rs == 0 or rl == 0:
                        continue
                    if (rs > 0) != (rl > 0):
                        continue
                    if max(abs(rs), abs(rl)) < thresh:
                        continue
                    direction = "Bull" if rs > 0 else "Bear"
                    bets += 1
                    if direction == f["outcome"]:
                        wins += 1
                    pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
                if bets >= 30:
                    results4.append({
                        "short": short_lb, "long": long_lb, "thresh": thresh,
                        "bets": bets, "wins": wins,
                        "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                    })

    results4.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'short':>5} {'long':>5} {'thresh':>7} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 60)
    for r in results4[:20]:
        print(f"{r['short']:>4}s {r['long']:>4}s {r['thresh']:>7.4f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 5. COMBINED: BNB accel + BTC accel fallback
    # =========================================================
    print("\n" + "=" * 85)
    print("5. COMBINED: BNB 7+10 acceleration primary, BTC acceleration fallback")
    print("=" * 85)

    # First find the best BTC acceleration config from results4
    if results4:
        best_btc = results4[0]
        print(f"  Best BTC accel: {best_btc['short']}+{best_btc['long']}s "
              f"thresh={best_btc['thresh']:.4f}")

    results5 = []
    for btc_short, btc_long in [(5, 20), (7, 20), (7, 30), (10, 30), (5, 30), (7, 60), (10, 60)]:
        for btc_thresh in [0.0, 0.0003, 0.0005, 0.001]:
            bets, wins, pnl = 0, 0, 0.0
            bnb_ct, btc_ct = 0, 0
            for f in features:
                direction = None
                # BNB 7+10 acceleration
                r7 = f.get("bnb_ret_7")
                r10 = f.get("bnb_ret_10")
                if (r7 is not None and r10 is not None and r7 != 0 and r10 != 0
                        and (r7 > 0) == (r10 > 0)
                        and max(abs(r7), abs(r10)) >= 0.0002):
                    direction = "Bull" if r7 > 0 else "Bear"
                    bnb_ct += 1
                else:
                    # BTC acceleration fallback
                    rs = f.get(f"btc_ret_{btc_short}")
                    rl = f.get(f"btc_ret_{btc_long}")
                    if (rs is not None and rl is not None and rs != 0 and rl != 0
                            and (rs > 0) == (rl > 0)
                            and max(abs(rs), abs(rl)) >= btc_thresh):
                        direction = "Bull" if rs > 0 else "Bear"
                        btc_ct += 1
                if direction is None:
                    continue
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 50:
                results5.append({
                    "btc_sig": f"{btc_short}+{btc_long}@{btc_thresh:.4f}",
                    "bets": bets, "bnb_ct": bnb_ct, "btc_ct": btc_ct,
                    "wins": wins, "wr": wins / bets,
                    "pnl": pnl, "ppb": pnl / bets,
                })

    results5.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'btc_fallback':>16} {'bets':>5} {'bnb':>4} {'btc':>4} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 65)
    for r in results5[:20]:
        print(f"{r['btc_sig']:>16} {r['bets']:>5} {r['bnb_ct']:>4} {r['btc_ct']:>4} "
              f"{r['wr']:>6.1%} {r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 6. HEAD-TO-HEAD: best of each approach
    # =========================================================
    print("\n" + "=" * 85)
    print("6. HEAD-TO-HEAD COMPARISON")
    print("=" * 85)

    all_approaches = [
        ("BTC alone", results1),
        ("BTC fallback", results2),
        ("BTC+BNB agree", results3),
        ("BTC accel", results4),
        ("BNB accel + BTC fallback", results5),
    ]

    # Also include the BNB-only baseline for reference
    bets_base, wins_base, pnl_base = 0, 0, 0.0
    for f in features:
        r7 = f.get("bnb_ret_7")
        r10 = f.get("bnb_ret_10")
        if (r7 is not None and r10 is not None and r7 != 0 and r10 != 0
                and (r7 > 0) == (r10 > 0)
                and max(abs(r7), abs(r10)) >= 0.0002):
            d = "Bull" if r7 > 0 else "Bear"
            bets_base += 1
            if d == f["outcome"]:
                wins_base += 1
            pnl_base += net_profit(f["bull_wei"], f["bear_wei"], d, f["outcome"])
    print(f"\n  {'BNB 7+10 accel (baseline)':>30}: bets={bets_base:>5}  "
          f"WR={wins_base/bets_base:.1%}  PnL={pnl_base:+.4f}")

    for label, res_list in all_approaches:
        if not res_list:
            continue
        # Show best by PnL with reasonable bet count
        close = [r for r in res_list if r["bets"] >= 100]
        if not close:
            close = res_list
        best = max(close, key=lambda r: r["pnl"])
        print(f"  {label:>30}: bets={best['bets']:>5}  "
              f"WR={best['wr']:.1%}  PnL={best['pnl']:+.4f}  config={best}")


if __name__ == "__main__":
    main()
