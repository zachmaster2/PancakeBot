"""Gap analysis: leaked signal vs honest signal.

Reconstructs what the leaked 1m kline signal was doing, quantifies
the gap, and explores what it would take to close it.

The leaked signal used the 1m kline close price at cutoff time, but
that close price actually reflects the price at END of that minute —
i.e., it contained ~0-60 seconds of future information.

We can reconstruct this: the 1s klines cover lockAt-99s to lockAt.
The cutoff is lockAt-4s. The "future" information available was
roughly the 1s klines AFTER cutoff (i.e., between cutoff and lockAt).

This script:
1. Reconstructs the leaked signal using post-cutoff 1s klines
2. Compares honest vs leaked signal performance
3. Tests how much future lookahead is needed to match
4. Explores untapped features: volume, high-low range, longer lookbacks
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROUNDS_PATH = Path("var/closed_rounds.jsonl")
DATA_PATH = Path("var/cutoff_spot_prices.jsonl")

BNB_WEI = 10**18
BET = 0.05
GAS_BET = 0.0002
GAS_CLAIM = 0.00025
FEE = 0.03
CUTOFF_SECONDS = 4


def load_data():
    records = []
    for line in DATA_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                records.append(r)
    rounds_by_epoch = {}
    for line in ROUNDS_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rounds_by_epoch[r["epoch"]] = r
    return records, rounds_by_epoch


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
    if side == "Bull":
        bw = bull_wei + bet_wei
        ew = bear_wei
    else:
        bw = bull_wei
        ew = bear_wei + bet_wei
    tw = bw + ew
    my_pool = bw if side == "Bull" else ew
    if my_pool <= 0:
        return 0.0
    return (tw * (1 - FEE)) / my_pool


def net_profit(bull_wei, bear_wei, side, outcome, bet_bnb=0.05):
    mult = payout_multiple(bull_wei, bear_wei, side, bet_bnb)
    if outcome == side:
        return bet_bnb * mult - GAS_CLAIM - bet_bnb - GAS_BET
    return -bet_bnb - GAS_BET


def get_return(klines, cutoff_ms, lookback_s):
    kn = find_closest(klines, cutoff_ms)
    ka = find_closest(klines, cutoff_ms - lookback_s * 1000)
    if not kn or not ka or ka[4] <= 0:
        return None
    return (kn[4] / ka[4]) - 1


def main():
    records, rounds_by_epoch = load_data()
    print(f"Loaded {len(records)} rounds\n")

    features = []
    for rec in records:
        rnd = rounds_by_epoch.get(rec["epoch"])
        if not rnd or rnd.get("failed") or rnd["position"] not in ("Bull", "Bear"):
            continue
        kl = rec["klines_1s"]
        lock_ms = rec["lock_at"] * 1000
        cutoff_ms = lock_ms - CUTOFF_SECONDS * 1000
        bull_wei, bear_wei = compute_pools(rnd)
        if bull_wei + bear_wei == 0:
            continue

        feat = {
            "epoch": rec["epoch"],
            "outcome": rnd["position"],
            "bull_wei": bull_wei,
            "bear_wei": bear_wei,
            "cutoff_ms": cutoff_ms,
            "lock_ms": lock_ms,
            "klines": kl,
        }
        features.append(feat)

    print(f"Features: {len(features)} rounds\n")

    # =========================================================
    # 1. RECONSTRUCT LEAKED SIGNAL
    # =========================================================
    print("=" * 85)
    print("1. SIMULATING LOOKAHEAD: what if we peek N seconds past cutoff?")
    print("   This reconstructs what the leaked 1m kline was doing.")
    print("   ret = spot(cutoff + N) / spot(cutoff - lookback) - 1")
    print("=" * 85)

    # The leaked signal computed ret using the 1m kline close price.
    # If cutoff was at second 45 of a minute, the close was at second 60 = 15s lookahead.
    # On average, ~30s lookahead.

    for lookback in [5, 10, 20]:
        for lookahead in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30]:
            for threshold in [0.0003]:
                bets, wins, pnl = 0, 0, 0.0
                for f in features:
                    # "now" price includes lookahead
                    kn = find_closest(f["klines"], f["cutoff_ms"] + lookahead * 1000)
                    ka = find_closest(f["klines"], f["cutoff_ms"] - lookback * 1000)
                    if not kn or not ka or ka[4] <= 0:
                        continue
                    ret = (kn[4] / ka[4]) - 1
                    if abs(ret) < threshold:
                        continue
                    direction = "Bull" if ret > 0 else "Bear"
                    bets += 1
                    if direction == f["outcome"]:
                        wins += 1
                    pnl += net_profit(f["bull_wei"], f["bear_wei"],
                                      direction, f["outcome"])
                if bets >= 20:
                    wr = wins / bets
                    print(f"  lb={lookback:>2}s  ahead={lookahead:>2}s  "
                          f"thresh={threshold:.4f}  bets={bets:>5}  "
                          f"WR={wr:.1%}  PnL={pnl:>+8.2f}  PnL/bet={pnl/bets:>+.4f}")
        print()

    # =========================================================
    # 2. HONEST SIGNAL COMPARISON: our best configs
    # =========================================================
    print("=" * 85)
    print("2. HONEST SIGNAL (no lookahead) — our current best")
    print("=" * 85)

    # 7s+10s acceleration
    bets, wins, pnl = 0, 0, 0.0
    for f in features:
        r7 = get_return(f["klines"], f["cutoff_ms"], 7)
        r10 = get_return(f["klines"], f["cutoff_ms"], 10)
        if r7 is None or r10 is None or r7 == 0 or r10 == 0:
            continue
        if (r7 > 0) != (r10 > 0):
            continue
        if max(abs(r7), abs(r10)) < 0.0002:
            continue
        direction = "Bull" if r7 > 0 else "Bear"
        bets += 1
        if direction == f["outcome"]:
            wins += 1
        pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
    print(f"  7+10 accel: bets={bets}  WR={wins/bets:.1%}  PnL={pnl:+.2f}")

    # Flat 0.10 BNB
    bets2, wins2, pnl2 = 0, 0, 0.0
    for f in features:
        r7 = get_return(f["klines"], f["cutoff_ms"], 7)
        r10 = get_return(f["klines"], f["cutoff_ms"], 10)
        if r7 is None or r10 is None or r7 == 0 or r10 == 0:
            continue
        if (r7 > 0) != (r10 > 0):
            continue
        if max(abs(r7), abs(r10)) < 0.0002:
            continue
        direction = "Bull" if r7 > 0 else "Bear"
        bets2 += 1
        if direction == f["outcome"]:
            wins2 += 1
        pnl2 += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"], 0.10)
    print(f"  7+10 accel @0.10: bets={bets2}  WR={wins2/bets2:.1%}  PnL={pnl2:+.2f}")
    print()

    # =========================================================
    # 3. UNTAPPED FEATURES: volume, volatility, longer lookbacks
    # =========================================================
    print("=" * 85)
    print("3. UNTAPPED FEATURES FROM EXISTING DATA")
    print("=" * 85)

    # 3a. Longer lookback (we have 99s of history)
    print("\n  3a. LONGER LOOKBACK (45s, 60s, 90s):")
    for lookback in [30, 45, 60, 75, 90]:
        for threshold in [0.0003, 0.0005, 0.001]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                ret = get_return(f["klines"], f["cutoff_ms"], lookback)
                if ret is None or abs(ret) < threshold:
                    continue
                direction = "Bull" if ret > 0 else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                wr = wins / bets
                print(f"    lb={lookback:>2}s  thresh={threshold:.4f}  "
                      f"bets={bets:>5}  WR={wr:.1%}  PnL={pnl:>+8.4f}")

    # 3b. Volume-weighted momentum
    print("\n  3b. VOLUME-WEIGHTED MOMENTUM:")
    for window in [10, 20, 30, 60]:
        for min_vwap_ret in [0.0001, 0.0003, 0.0005]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                cutoff_ms = f["cutoff_ms"]
                start_ms = cutoff_ms - window * 1000
                # Compute VWAP and volume-weighted direction
                total_vol = 0.0
                vwap_num = 0.0
                spot_now = None
                for k in f["klines"]:
                    if start_ms <= k[0] <= cutoff_ms:
                        mid = (k[1] + k[4]) / 2  # (open + close) / 2
                        vol = k[5]
                        vwap_num += mid * vol
                        total_vol += vol
                    if spot_now is None or abs(k[0] - cutoff_ms) < abs(spot_now[0] - cutoff_ms):
                        spot_now = k
                if total_vol <= 0 or spot_now is None or spot_now[4] <= 0:
                    continue
                vwap = vwap_num / total_vol
                ret = (spot_now[4] / vwap) - 1
                if abs(ret) < min_vwap_ret:
                    continue
                direction = "Bull" if ret > 0 else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                wr = wins / bets
                print(f"    window={window:>2}s  min_ret={min_vwap_ret:.4f}  "
                      f"bets={bets:>5}  WR={wr:.1%}  PnL={pnl:>+8.4f}")

    # 3c. Volume spike as a signal
    print("\n  3c. VOLUME SPIKE: bet when recent volume surge + momentum agree:")
    for window in [5, 10, 20]:
        for vol_mult in [1.5, 2.0, 3.0]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                cutoff_ms = f["cutoff_ms"]
                # Recent volume (last `window` seconds)
                recent_vol = 0.0
                recent_count = 0
                for k in f["klines"]:
                    if cutoff_ms - window * 1000 <= k[0] <= cutoff_ms:
                        recent_vol += k[5]
                        recent_count += 1
                # Background volume (window before that)
                bg_vol = 0.0
                bg_count = 0
                for k in f["klines"]:
                    if cutoff_ms - 2 * window * 1000 <= k[0] < cutoff_ms - window * 1000:
                        bg_vol += k[5]
                        bg_count += 1
                if bg_count == 0 or bg_vol <= 0:
                    continue
                vol_ratio = recent_vol / bg_vol
                if vol_ratio < vol_mult:
                    continue
                # Direction from momentum
                ret = get_return(f["klines"], f["cutoff_ms"], window)
                if ret is None or ret == 0:
                    continue
                direction = "Bull" if ret > 0 else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                wr = wins / bets
                print(f"    window={window:>2}s  vol_mult>={vol_mult:.1f}  "
                      f"bets={bets:>5}  WR={wr:.1%}  PnL={pnl:>+8.4f}")

    # 3d. High-low range as volatility signal
    print("\n  3d. HIGH-LOW RANGE: bet only when recent volatility is above threshold:")
    for window in [10, 20, 30, 60]:
        for min_range_pct in [0.0005, 0.001, 0.002]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                cutoff_ms = f["cutoff_ms"]
                highs, lows = [], []
                for k in f["klines"]:
                    if cutoff_ms - window * 1000 <= k[0] <= cutoff_ms:
                        highs.append(k[2])
                        lows.append(k[3])
                if not highs:
                    continue
                max_h = max(highs)
                min_l = min(lows)
                if min_l <= 0:
                    continue
                range_pct = (max_h - min_l) / min_l
                if range_pct < min_range_pct:
                    continue
                # Direction from endpoint momentum
                ret = get_return(f["klines"], f["cutoff_ms"], window)
                if ret is None or ret == 0:
                    continue
                direction = "Bull" if ret > 0 else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                wr = wins / bets
                print(f"    window={window:>2}s  min_range>={min_range_pct:.4f}  "
                      f"bets={bets:>5}  WR={wr:.1%}  PnL={pnl:>+8.4f}")

    # 3e. Trend consistency (linear regression slope of 1s closes)
    print("\n  3e. TREND SLOPE: linear regression slope of 1s close prices:")
    for window in [10, 20, 30, 60]:
        for min_r2 in [0.0, 0.3, 0.5]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                cutoff_ms = f["cutoff_ms"]
                xs, ys = [], []
                for k in f["klines"]:
                    if cutoff_ms - window * 1000 <= k[0] <= cutoff_ms:
                        xs.append((k[0] - cutoff_ms) / 1000)  # seconds before cutoff
                        ys.append(k[4])
                if len(xs) < 3:
                    continue
                # Simple linear regression
                n = len(xs)
                sx = sum(xs)
                sy = sum(ys)
                sxx = sum(x*x for x in xs)
                sxy = sum(x*y for x, y in zip(xs, ys))
                denom = n * sxx - sx * sx
                if denom == 0:
                    continue
                slope = (n * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / n
                # R-squared
                y_mean = sy / n
                ss_tot = sum((y - y_mean)**2 for y in ys)
                ss_res = sum((y - (slope*x + intercept))**2 for x, y in zip(xs, ys))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                if r2 < min_r2:
                    continue
                # Normalize slope: per-second return as fraction of price
                if y_mean <= 0:
                    continue
                norm_slope = slope / y_mean  # fractional price change per second
                if abs(norm_slope) < 0.00001:
                    continue
                direction = "Bull" if norm_slope > 0 else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                wr = wins / bets
                print(f"    window={window:>2}s  min_r2={min_r2:.1f}  "
                      f"bets={bets:>5}  WR={wr:.1%}  PnL={pnl:>+8.4f}")

    # 3f. Acceleration signal + longer lookback combos
    print("\n  3f. ACCELERATION with longer lookbacks:")
    for short_lb in [5, 7, 10, 15]:
        for long_lb in [30, 45, 60, 90]:
            for thresh in [0.0002, 0.0003, 0.0005]:
                bets, wins, pnl = 0, 0, 0.0
                for f in features:
                    rs = get_return(f["klines"], f["cutoff_ms"], short_lb)
                    rl = get_return(f["klines"], f["cutoff_ms"], long_lb)
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
                if bets >= 50:
                    wr = wins / bets
                    if pnl > 0:
                        print(f"    {short_lb:>2}s+{long_lb:>2}s  thresh={thresh:.4f}  "
                              f"bets={bets:>5}  WR={wr:.1%}  PnL={pnl:>+8.4f}")

    # =========================================================
    # 4. THE GAP — what does it take?
    # =========================================================
    print("\n" + "=" * 85)
    print("4. THE GAP: how much lookahead is needed to reach +10/+20 BNB?")
    print("   Shows PnL at each lookahead level for lb=5, thresh=0.0003")
    print("=" * 85)
    for lookahead in range(0, 31):
        bets, wins, pnl = 0, 0, 0.0
        for f in features:
            kn = find_closest(f["klines"], f["cutoff_ms"] + lookahead * 1000)
            ka = find_closest(f["klines"], f["cutoff_ms"] - 5 * 1000)
            if not kn or not ka or ka[4] <= 0:
                continue
            ret = (kn[4] / ka[4]) - 1
            if abs(ret) < 0.0003:
                continue
            direction = "Bull" if ret > 0 else "Bear"
            bets += 1
            if direction == f["outcome"]:
                wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
        if bets >= 20:
            wr = wins / bets
            bar = "█" * int(max(0, pnl) / 0.5)
            print(f"  +{lookahead:>2}s: bets={bets:>5}  WR={wr:.1%}  "
                  f"PnL={pnl:>+8.2f}  {bar}")


if __name__ == "__main__":
    main()
