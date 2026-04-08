"""Final sweep: refine the top signal approaches and test combinations.

Winner from prior sweeps: Momentum Acceleration (7s+20s agreement)
  699 bets, 60.8% WR, +1.56 BNB

Tests here:
1. Fine-grain acceleration sweep (more short/long pairs, thresholds)
2. Triple confirmation: short + mid + long all agree
3. Acceleration + payout asymmetry combined
4. Acceleration + directional consistency combined
5. Time-series split: is the edge stable across early/late halves?

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
            "klines": kl,
        }
        for lb in [3, 5, 7, 10, 12, 15, 20, 25, 30]:
            feat[f"ret_{lb}"] = get_return(kl, cutoff_ms, lb)
        features.append(feat)

    print(f"Features: {len(features)} rounds\n")
    half = len(features) // 2

    # =========================================================
    # 1. FINE-GRAIN ACCELERATION SWEEP
    # =========================================================
    print("=" * 85)
    print("1. ACCELERATION: fine-grain sweep of short+long pairs")
    print("=" * 85)

    results1 = []
    for short_lb in [3, 5, 7, 10]:
        for long_lb in [10, 12, 15, 20, 25, 30]:
            if long_lb <= short_lb:
                continue
            for min_thresh in [0.0, 0.0001, 0.0002, 0.0003, 0.0005]:
                bets, wins, pnl = 0, 0, 0.0
                for f in features:
                    rs = f.get(f"ret_{short_lb}")
                    rl = f.get(f"ret_{long_lb}")
                    if rs is None or rl is None or rs == 0 or rl == 0:
                        continue
                    if (rs > 0) != (rl > 0):
                        continue
                    if max(abs(rs), abs(rl)) < min_thresh:
                        continue
                    direction = "Bull" if rs > 0 else "Bear"
                    bets += 1
                    if direction == f["outcome"]:
                        wins += 1
                    pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
                if bets >= 30:
                    results1.append({
                        "short": short_lb, "long": long_lb, "thresh": min_thresh,
                        "bets": bets, "wins": wins,
                        "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                    })

    results1.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'short':>5} {'long':>5} {'thresh':>7} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 65)
    for r in results1[:30]:
        print(f"{r['short']:>4}s {r['long']:>4}s {r['thresh']:>7.4f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 2. TRIPLE CONFIRMATION: short + mid + long
    # =========================================================
    print("\n" + "=" * 85)
    print("2. TRIPLE CONFIRMATION: require 3 lookback windows to agree")
    print("=" * 85)

    results2 = []
    triples = [
        (3, 7, 20), (3, 10, 20), (5, 7, 20), (5, 10, 20), (5, 10, 30),
        (5, 15, 30), (3, 7, 15), (5, 7, 15), (7, 10, 20), (7, 15, 30),
        (3, 5, 10), (3, 5, 20), (3, 10, 30), (5, 12, 25),
    ]
    for s, m, l in triples:
        for min_thresh in [0.0, 0.0002, 0.0003]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                rs = f.get(f"ret_{s}")
                rm = f.get(f"ret_{m}")
                rl = f.get(f"ret_{l}")
                if rs is None or rm is None or rl is None:
                    continue
                if rs == 0 or rm == 0 or rl == 0:
                    continue
                # All three must agree
                if not ((rs > 0 and rm > 0 and rl > 0) or
                        (rs < 0 and rm < 0 and rl < 0)):
                    continue
                if max(abs(rs), abs(rm), abs(rl)) < min_thresh:
                    continue
                direction = "Bull" if rs > 0 else "Bear"
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                results2.append({
                    "triple": f"{s},{m},{l}", "thresh": min_thresh,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                })

    results2.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'triple':>10} {'thresh':>7} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 60)
    for r in results2[:20]:
        print(f"{r['triple']:>10} {r['thresh']:>7.4f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 3. ACCELERATION + PAYOUT ASYMMETRY
    # =========================================================
    print("\n" + "=" * 85)
    print("3. ACCELERATION + PAYOUT ASYMMETRY COMBINED")
    print("   Best acceleration configs + payout gate")
    print("=" * 85)

    # Top acceleration configs from prior sweep
    accel_configs = [
        ("7+20@2", 7, 20, 0.0002),
        ("5+10@2", 5, 10, 0.0002),
        ("7+15@2", 7, 15, 0.0002),
        ("5+20@2", 5, 20, 0.0002),
        ("5+10@0", 5, 10, 0.0),
        ("7+20@0", 7, 20, 0.0),
    ]
    results3 = []
    for label, short_lb, long_lb, min_thresh in accel_configs:
        for min_pay in [0.0, 1.8, 1.9, 2.0, 2.5]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                rs = f.get(f"ret_{short_lb}")
                rl = f.get(f"ret_{long_lb}")
                if rs is None or rl is None or rs == 0 or rl == 0:
                    continue
                if (rs > 0) != (rl > 0):
                    continue
                if max(abs(rs), abs(rl)) < min_thresh:
                    continue
                direction = "Bull" if rs > 0 else "Bear"
                mult = payout_multiple(f["bull_wei"], f["bear_wei"], direction)
                if mult < min_pay:
                    continue
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                results3.append({
                    "accel": label, "min_pay": min_pay,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                })

    results3.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'accel':>10} {'min_pay':>8} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 65)
    for r in results3:
        print(f"{r['accel']:>10} {r['min_pay']:>8.2f} {r['bets']:>5} {r['wins']:>5} "
              f"{r['wr']:>6.1%} {r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 4. TIME-SERIES STABILITY: early vs late half
    # =========================================================
    print("\n" + "=" * 85)
    print("4. TIME-SERIES STABILITY: first half vs second half")
    print("   Tests whether edge is regime-dependent or persistent")
    print("=" * 85)

    top_configs = [
        ("7+20@2 accel", lambda f: _accel_signal(f, 7, 20, 0.0002)),
        ("5+10@2 accel", lambda f: _accel_signal(f, 5, 10, 0.0002)),
        ("5@3 baseline", lambda f: _baseline_signal(f, 5, 0.0003)),
        ("6/6 consist", lambda f: _consistency_signal(f, 6, 0.0003)),
        ("5+10@0 accel", lambda f: _accel_signal(f, 5, 10, 0.0)),
    ]

    for label, signal_fn in top_configs:
        for split_label, subset in [("all", features), ("1st half", features[:half]),
                                    ("2nd half", features[half:])]:
            bets, wins, pnl = 0, 0, 0.0
            for f in subset:
                direction = signal_fn(f)
                if direction is None:
                    continue
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets > 0:
                print(f"  {label:>16} [{split_label:>8}]: bets={bets:>5}  "
                      f"WR={wins/bets:.1%}  PnL={pnl:+.4f}  PnL/bet={pnl/bets:+.6f}")
        print()

    # =========================================================
    # 5. QUARTILE STABILITY
    # =========================================================
    print("=" * 85)
    print("5. QUARTILE STABILITY: Q1/Q2/Q3/Q4 for top configs")
    print("=" * 85)

    q_size = len(features) // 4
    quartiles = [
        ("Q1", features[:q_size]),
        ("Q2", features[q_size:2*q_size]),
        ("Q3", features[2*q_size:3*q_size]),
        ("Q4", features[3*q_size:]),
    ]

    for label, signal_fn in top_configs[:3]:
        print(f"\n  {label}:")
        for q_label, subset in quartiles:
            bets, wins, pnl = 0, 0, 0.0
            for f in subset:
                direction = signal_fn(f)
                if direction is None:
                    continue
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets > 0:
                print(f"    {q_label}: bets={bets:>4}  WR={wins/bets:.1%}  "
                      f"PnL={pnl:+.4f}")
            else:
                print(f"    {q_label}: bets=   0")


def _accel_signal(f, short_lb, long_lb, min_thresh):
    rs = f.get(f"ret_{short_lb}")
    rl = f.get(f"ret_{long_lb}")
    if rs is None or rl is None or rs == 0 or rl == 0:
        return None
    if (rs > 0) != (rl > 0):
        return None
    if max(abs(rs), abs(rl)) < min_thresh:
        return None
    return "Bull" if rs > 0 else "Bear"


def _baseline_signal(f, lookback, threshold):
    ret = f.get(f"ret_{lookback}")
    if ret is None or abs(ret) < threshold:
        return None
    return "Bull" if ret > 0 else "Bear"


def _consistency_signal(f, min_agree, min_thresh):
    LOOKBACKS = [3, 5, 7, 10, 15, 20]
    bull, bear = 0, 0
    any_exceeds = False
    for lb in LOOKBACKS:
        ret = f.get(f"ret_{lb}")
        if ret is None or ret == 0:
            continue
        if ret > 0:
            bull += 1
        else:
            bear += 1
        if abs(ret) >= min_thresh:
            any_exceeds = True
    if not any_exceeds:
        return None
    majority = max(bull, bear)
    if majority < min_agree:
        return None
    return "Bull" if bull > bear else "Bear"


if __name__ == "__main__":
    main()
