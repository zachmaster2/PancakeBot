"""Sweep advanced momentum signal constructions on cached 1s kline data.

Ideas tested:
1. Adaptive threshold: scale threshold proportionally to lookback
2. Cascade with decreasing threshold
3. Momentum acceleration: compare short vs long return
4. Directional consistency: how many lookback windows agree
5. Max-move signal: use max deviation within window instead of endpoint
6. Cascade + payout asymmetry combined

All use cutoff=4s. No API calls.
"""

from __future__ import annotations

import json
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


def payout_multiple(bull_wei, bear_wei, side):
    bet_wei = int(BET * BNB_WEI)
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


def net_profit(bull_wei, bear_wei, side, outcome):
    mult = payout_multiple(bull_wei, bear_wei, side)
    if outcome == side:
        return BET * mult - GAS_CLAIM - BET - GAS_BET
    return -BET - GAS_BET


def get_return(klines, cutoff_ms, lookback_s):
    """Get endpoint return for a given lookback."""
    kn = find_closest(klines, cutoff_ms)
    ka = find_closest(klines, cutoff_ms - lookback_s * 1000)
    if not kn or not ka or ka[4] <= 0:
        return None
    return (kn[4] / ka[4]) - 1


def main():
    records, rounds_by_epoch = load_data()
    print(f"Loaded {len(records)} rounds\n")

    # Build features
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
            "klines": kl,
            "cutoff_ms": cutoff_ms,
        }

        # Compute returns for various lookbacks
        for lb in [3, 5, 7, 10, 15, 20, 30]:
            ret = get_return(kl, cutoff_ms, lb)
            feat[f"ret_{lb}"] = ret

        # Max deviation in window: max(close) and min(close) relative to spot_now
        kn = find_closest(kl, cutoff_ms)
        if kn:
            spot_now = kn[4]
            for window in [5, 10, 15, 20]:
                window_start_ms = cutoff_ms - window * 1000
                closes = []
                for k in kl:
                    if window_start_ms <= k[0] <= cutoff_ms:
                        closes.append(k[4])
                if closes and spot_now > 0:
                    max_dev_up = max((c / spot_now - 1) for c in closes)
                    max_dev_down = min((c / spot_now - 1) for c in closes)
                    feat[f"maxdev_up_{window}"] = max_dev_up
                    feat[f"maxdev_down_{window}"] = max_dev_down
                    # Net max deviation: which direction had the bigger extreme?
                    feat[f"maxdev_net_{window}"] = abs(max_dev_up) - abs(max_dev_down)
                else:
                    feat[f"maxdev_up_{window}"] = 0.0
                    feat[f"maxdev_down_{window}"] = 0.0
                    feat[f"maxdev_net_{window}"] = 0.0

        features.append(feat)

    print(f"Features: {len(features)} rounds\n")

    # =========================================================
    # 1. ADAPTIVE THRESHOLD: threshold scales with sqrt(lookback)
    # =========================================================
    print("=" * 85)
    print("1. ADAPTIVE THRESHOLD: threshold = base * sqrt(lookback / 5)")
    print("=" * 85)

    import math
    results1 = []
    for base_thresh in [0.0002, 0.0003, 0.0004, 0.0005]:
        for lb in [5, 7, 10, 15, 20, 30]:
            scaled_thresh = base_thresh * math.sqrt(lb / 5)
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                ret = f.get(f"ret_{lb}")
                if ret is None:
                    continue
                if abs(ret) < scaled_thresh:
                    continue
                direction = "Bull" if ret > 0 else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                results1.append({
                    "base_thresh": base_thresh, "lookback": lb,
                    "eff_thresh": scaled_thresh,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                })

    results1.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'base':>7} {'lb':>3} {'eff_thresh':>10} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 70)
    for r in results1[:20]:
        print(f"{r['base_thresh']:>7.4f} {r['lookback']:>3}s {r['eff_thresh']:>10.6f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 2. CASCADE WITH DECREASING THRESHOLD
    # =========================================================
    print("\n" + "=" * 85)
    print("2. CASCADE WITH DECREASING THRESHOLD")
    print("    Try short lookback first with high threshold,")
    print("    fall back to longer lookback with lower threshold")
    print("=" * 85)

    results2 = []
    cascade_configs = [
        # (label, [(lookback, threshold), ...])
        ("5@3,10@3", [(5, 0.0003), (10, 0.0003)]),
        ("5@3,10@2", [(5, 0.0003), (10, 0.0002)]),
        ("5@3,10@2,20@1", [(5, 0.0003), (10, 0.0002), (20, 0.0001)]),
        ("5@5,10@3", [(5, 0.0005), (10, 0.0003)]),
        ("5@5,10@3,20@2", [(5, 0.0005), (10, 0.0003), (20, 0.0002)]),
        ("5@3,10@3,15@3,20@3", [(5, 0.0003), (10, 0.0003), (15, 0.0003), (20, 0.0003)]),
        ("5@3,10@2,15@2,20@1", [(5, 0.0003), (10, 0.0002), (15, 0.0002), (20, 0.0001)]),
        ("5@3,7@3,10@3", [(5, 0.0003), (7, 0.0003), (10, 0.0003)]),
        ("3@3,5@3,7@3,10@3", [(3, 0.0003), (5, 0.0003), (7, 0.0003), (10, 0.0003)]),
        ("5@3,10@3,20@3,30@3", [(5, 0.0003), (10, 0.0003), (20, 0.0003), (30, 0.0003)]),
    ]
    for label, steps in cascade_configs:
        bets, wins, pnl = 0, 0, 0.0
        for f in features:
            direction = None
            for lb, thresh in steps:
                ret = f.get(f"ret_{lb}")
                if ret is not None and abs(ret) >= thresh:
                    direction = "Bull" if ret > 0 else "Bear"
                    break
            if direction is None:
                continue
            bets += 1
            if direction == f["outcome"]:
                wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
        if bets >= 20:
            results2.append({
                "cascade": label, "bets": bets, "wins": wins,
                "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
            })

    results2.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'cascade':>25} {'bets':>5} {'wins':>5} {'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 72)
    for r in results2:
        print(f"{r['cascade']:>25} {r['bets']:>5} {r['wins']:>5} "
              f"{r['wr']:>6.1%} {r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 3. MOMENTUM ACCELERATION: short_ret vs long_ret
    # =========================================================
    print("\n" + "=" * 85)
    print("3. MOMENTUM ACCELERATION: require short-term and long-term to agree")
    print("   Signal fires when both 5s and 10s returns agree on direction")
    print("   and at least one exceeds threshold")
    print("=" * 85)

    results3 = []
    for short_lb, long_lb in [(5, 10), (5, 15), (5, 20), (7, 15), (7, 20), (10, 20)]:
        for min_thresh in [0.0, 0.0001, 0.0002, 0.0003]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                ret_short = f.get(f"ret_{short_lb}")
                ret_long = f.get(f"ret_{long_lb}")
                if ret_short is None or ret_long is None:
                    continue
                # Both must agree on direction (nonzero)
                if ret_short == 0 or ret_long == 0:
                    continue
                if (ret_short > 0) != (ret_long > 0):
                    continue  # disagreement
                # At least one must exceed threshold
                if max(abs(ret_short), abs(ret_long)) < min_thresh:
                    continue
                direction = "Bull" if ret_short > 0 else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                results3.append({
                    "short": short_lb, "long": long_lb,
                    "min_thresh": min_thresh,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                })

    results3.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'short':>5} {'long':>5} {'thresh':>7} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 65)
    for r in results3[:20]:
        print(f"{r['short']:>4}s {r['long']:>4}s {r['min_thresh']:>7.4f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 4. DIRECTIONAL CONSISTENCY: how many lookback windows agree
    # =========================================================
    print("\n" + "=" * 85)
    print("4. DIRECTIONAL CONSISTENCY: count how many lookbacks agree on direction")
    print("   Uses lookbacks: 3s, 5s, 7s, 10s, 15s, 20s")
    print("=" * 85)

    LOOKBACKS = [3, 5, 7, 10, 15, 20]
    results4 = []
    for min_agree in [3, 4, 5, 6]:
        for min_any_thresh in [0.0, 0.0001, 0.0003]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                bull_count = 0
                bear_count = 0
                any_exceeds = False
                for lb in LOOKBACKS:
                    ret = f.get(f"ret_{lb}")
                    if ret is None or ret == 0:
                        continue
                    if ret > 0:
                        bull_count += 1
                    else:
                        bear_count += 1
                    if abs(ret) >= min_any_thresh:
                        any_exceeds = True
                if not any_exceeds:
                    continue
                total_votes = bull_count + bear_count
                if total_votes < min_agree:
                    continue
                majority = max(bull_count, bear_count)
                if majority < min_agree:
                    continue
                direction = "Bull" if bull_count > bear_count else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                results4.append({
                    "min_agree": min_agree, "min_thresh": min_any_thresh,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                })

    results4.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'agree':>5} {'thresh':>7} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 55)
    for r in results4[:15]:
        print(f"{r['min_agree']:>5} {r['min_thresh']:>7.4f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 5. MAX-DEVIATION SIGNAL
    # =========================================================
    print("\n" + "=" * 85)
    print("5. MAX-DEVIATION: direction = which extreme (up/down) was bigger in window")
    print("   Captures intra-window momentum even when endpoints are equal")
    print("=" * 85)

    results5 = []
    for window in [5, 10, 15, 20]:
        for min_dev in [0.0001, 0.0002, 0.0003, 0.0005]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                up = f.get(f"maxdev_up_{window}", 0)
                down = f.get(f"maxdev_down_{window}", 0)
                # Direction = which extreme was larger
                if abs(up) > abs(down):
                    # Upward deviation dominated → Bull
                    if abs(up) < min_dev:
                        continue
                    direction = "Bull"
                elif abs(down) > abs(up):
                    if abs(down) < min_dev:
                        continue
                    direction = "Bear"
                else:
                    continue
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                results5.append({
                    "window": window, "min_dev": min_dev,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                })

    results5.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'win':>4} {'min_dev':>8} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 55)
    for r in results5[:15]:
        print(f"{r['window']:>3}s {r['min_dev']:>8.4f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 6. CASCADE + PAYOUT ASYMMETRY COMBINED
    # =========================================================
    print("\n" + "=" * 85)
    print("6. CASCADE + PAYOUT ASYMMETRY COMBINED")
    print("   Best cascade configs + min payout multiple filter")
    print("=" * 85)

    results6 = []
    cascade_configs_6 = [
        ("5@3", [(5, 0.0003)]),
        ("5>10@3", [(5, 0.0003), (10, 0.0003)]),
        ("5>10>15@3", [(5, 0.0003), (10, 0.0003), (15, 0.0003)]),
        ("5>10>20@3", [(5, 0.0003), (10, 0.0003), (20, 0.0003)]),
    ]
    for label, steps in cascade_configs_6:
        for min_pay in [0.0, 1.8, 1.9, 2.0, 2.1, 2.5]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                direction = None
                for lb, thresh in steps:
                    ret = f.get(f"ret_{lb}")
                    if ret is not None and abs(ret) >= thresh:
                        direction = "Bull" if ret > 0 else "Bear"
                        break
                if direction is None:
                    continue
                # Check payout multiple
                mult = payout_multiple(f["bull_wei"], f["bear_wei"], direction)
                if mult < min_pay:
                    continue
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                results6.append({
                    "cascade": label, "min_pay": min_pay,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                })

    results6.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'cascade':>12} {'min_pay':>8} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 65)
    for r in results6:
        print(f"{r['cascade']:>12} {r['min_pay']:>8.2f} {r['bets']:>5} {r['wins']:>5} "
              f"{r['wr']:>6.1%} {r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 7. BEST OVERALL COMPARISON
    # =========================================================
    print("\n" + "=" * 85)
    print("7. BEST OVERALL — top configs from each approach (150-800 bets)")
    print("=" * 85)

    all_results = []
    for label, rl in [
        ("adaptive_thresh", results1),
        ("cascade_thresh", results2),
        ("acceleration", results3),
        ("consistency", results4),
        ("max_deviation", results5),
        ("cascade+payout", results6),
    ]:
        close = [r for r in rl if 100 <= r["bets"] <= 1500]
        if close:
            best = max(close, key=lambda r: r["pnl"])
            all_results.append((label, best))

    for label, best in sorted(all_results, key=lambda x: x[1]["pnl"], reverse=True):
        print(f"\n  {label:>18}: bets={best['bets']:>5}  WR={best['wr']:.1%}  "
              f"PnL={best['pnl']:+.4f}  PnL/bet={best['ppb']:+.6f}")
        print(f"                     config: {best}")


if __name__ == "__main__":
    main()
