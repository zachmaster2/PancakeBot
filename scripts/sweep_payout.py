"""Sweep momentum signal + payout asymmetry gate.

Since pools are frozen by lockAt-4s, we can compute exact payout
multiples at decision time. This sweep tests whether filtering on
the payout multiple improves profitability.

Payout multiple = total_pool * (1 - treasury_fee) / my_side_pool
  - If we bet Bull:  payout = total * 0.97 / bull_pool
  - If we bet Bear:  payout = total * 0.97 / bear_pool

A payout > 2.0 means the pool is skewed in our favour.
A payout < 1.5 means we need very high win rate to be profitable.

Uses cached 1s kline data + round pool data — no API calls.
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

# Base signal
CUTOFF_SECONDS = 4
LOOKBACK_SECONDS = 5


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


def find_closest(klines_1s, target_ms):
    best, best_d = None, float("inf")
    for k in klines_1s:
        d = abs(k[0] - target_ms)
        if d < best_d:
            best_d, best = d, k
    return best if best and best_d <= 2000 else None


def compute_pools(rnd):
    """Compute bull/bear pools from bets placed before lockAt."""
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
    """What we'd get per BNB wagered if we win (before gas)."""
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
    """Net profit/loss for a single bet."""
    mult = payout_multiple(bull_wei, bear_wei, side)
    if outcome == side:
        return BET * mult - GAS_CLAIM - BET - GAS_BET
    return -BET - GAS_BET


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
        ago_ms = cutoff_ms - LOOKBACK_SECONDS * 1000

        kn = find_closest(kl, cutoff_ms)
        ka = find_closest(kl, ago_ms)
        if not kn or not ka or ka[4] <= 0:
            continue

        ret = (kn[4] / ka[4]) - 1
        direction = "Bull" if ret > 0 else "Bear"
        bull_wei, bear_wei = compute_pools(rnd)

        if bull_wei + bear_wei == 0:
            continue

        mult = payout_multiple(bull_wei, bear_wei, direction)
        bull_frac = bull_wei / (bull_wei + bear_wei)

        features.append({
            "epoch": rec["epoch"],
            "ret": ret,
            "abs_ret": abs(ret),
            "direction": direction,
            "outcome": rnd["position"],
            "bull_wei": bull_wei,
            "bear_wei": bear_wei,
            "payout_mult": mult,
            "bull_frac": bull_frac,
        })

    print(f"Features: {len(features)} rounds\n")

    # --- Distribution of payout multiples ---
    mults = sorted([f["payout_mult"] for f in features if f["abs_ret"] >= 0.0003])
    n = len(mults)
    print("Payout multiple distribution (for bets passing threshold=0.0003):")
    print(f"  min={mults[0]:.2f}  p10={mults[int(n*0.1)]:.2f}  p25={mults[int(n*0.25)]:.2f}  "
          f"median={mults[n//2]:.2f}  p75={mults[int(n*0.75)]:.2f}  p90={mults[int(n*0.9)]:.2f}  "
          f"max={mults[-1]:.2f}")
    print()

    # --- Sweep: threshold x min_payout_multiple ---
    print("=" * 85)
    print("SWEEP: threshold x min_payout_multiple")
    print("=" * 85)

    THRESHOLDS = [0.0003, 0.0005, 0.0008]
    MIN_PAYOUTS = [0.0, 1.5, 1.7, 1.8, 1.9, 1.95, 2.0, 2.1, 2.3, 2.5, 3.0]

    results = []
    for thresh in THRESHOLDS:
        for min_pay in MIN_PAYOUTS:
            bets = 0
            wins = 0
            total_pnl = 0.0

            for f in features:
                if f["abs_ret"] < thresh:
                    continue
                if f["payout_mult"] < min_pay:
                    continue

                bets += 1
                if f["direction"] == f["outcome"]:
                    wins += 1
                total_pnl += net_profit(f["bull_wei"], f["bear_wei"],
                                        f["direction"], f["outcome"])

            if bets >= 10:
                wr = wins / bets
                # Breakeven WR for this payout: 1/avg_mult
                avg_mult = 0.0
                count = 0
                for f in features:
                    if f["abs_ret"] < thresh and f["payout_mult"] >= min_pay:
                        continue
                    if f["abs_ret"] >= thresh and f["payout_mult"] >= min_pay:
                        avg_mult += f["payout_mult"]
                        count += 1
                avg_mult = avg_mult / count if count > 0 else 1.94
                breakeven = 1.0 / avg_mult if avg_mult > 0 else 0.5

                results.append({
                    "thresh": thresh,
                    "min_pay": min_pay,
                    "bets": bets,
                    "wins": wins,
                    "wr": wr,
                    "pnl": total_pnl,
                    "ppb": total_pnl / bets,
                    "avg_mult": avg_mult,
                    "breakeven": breakeven,
                })

    results.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'thresh':>7} {'min_pay':>8} {'bets':>5} {'wins':>5} {'wr':>7} "
          f"{'avg_mult':>9} {'breakeven':>9} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 82)
    for r in results:
        print(f"{r['thresh']:>7.4f} {r['min_pay']:>8.2f} {r['bets']:>5} {r['wins']:>5} "
              f"{r['wr']:>6.1%} {r['avg_mult']:>9.2f} {r['breakeven']:>8.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # --- Contrarian angle: bet AGAINST momentum when payout is very high ---
    print("\n" + "=" * 85)
    print("CONTRARIAN: bet AGAINST momentum when payout is high")
    print("=" * 85)

    results2 = []
    for thresh in THRESHOLDS:
        for min_pay in [2.0, 2.5, 3.0, 4.0, 5.0]:
            bets = 0
            wins = 0
            total_pnl = 0.0

            for f in features:
                if f["abs_ret"] < thresh:
                    continue
                # Flip direction
                contra_dir = "Bear" if f["direction"] == "Bull" else "Bull"
                contra_mult = payout_multiple(f["bull_wei"], f["bear_wei"], contra_dir)
                if contra_mult < min_pay:
                    continue

                bets += 1
                if contra_dir == f["outcome"]:
                    wins += 1
                total_pnl += net_profit(f["bull_wei"], f["bear_wei"],
                                        contra_dir, f["outcome"])

            if bets >= 10:
                results2.append({
                    "thresh": thresh,
                    "min_pay": min_pay,
                    "bets": bets,
                    "wins": wins,
                    "wr": wins / bets,
                    "pnl": total_pnl,
                    "ppb": total_pnl / bets,
                })

    if results2:
        results2.sort(key=lambda r: r["pnl"], reverse=True)
        print(f"\n{'thresh':>7} {'min_pay':>8} {'bets':>5} {'wins':>5} {'wr':>7} "
              f"{'pnl':>10} {'pnl/bet':>10}")
        print("-" * 60)
        for r in results2:
            print(f"{r['thresh']:>7.4f} {r['min_pay']:>8.2f} {r['bets']:>5} {r['wins']:>5} "
                  f"{r['wr']:>6.1%} {r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")
    else:
        print("  (no combos with >= 10 bets)")

    # --- Expected value per bet: WR * payout - (1-WR) * bet ---
    print("\n" + "=" * 85)
    print("EXPECTED VALUE ANALYSIS — WR * avg_win - (1-WR) * avg_loss")
    print("=" * 85)
    for thresh in THRESHOLDS:
        for min_pay in [0.0, 1.9, 2.0, 2.5]:
            wins_pnl = []
            losses_pnl = []
            for f in features:
                if f["abs_ret"] < thresh:
                    continue
                if f["payout_mult"] < min_pay:
                    continue
                p = net_profit(f["bull_wei"], f["bear_wei"], f["direction"], f["outcome"])
                if f["direction"] == f["outcome"]:
                    wins_pnl.append(p)
                else:
                    losses_pnl.append(p)
            if len(wins_pnl) >= 5 and len(losses_pnl) >= 5:
                n = len(wins_pnl) + len(losses_pnl)
                wr = len(wins_pnl) / n
                avg_win = sum(wins_pnl) / len(wins_pnl)
                avg_loss = sum(losses_pnl) / len(losses_pnl)
                ev = wr * avg_win + (1 - wr) * avg_loss
                print(f"  thresh={thresh:.4f} min_pay={min_pay:.1f}: "
                      f"bets={n} WR={wr:.1%} "
                      f"avg_win={avg_win:+.4f} avg_loss={avg_loss:.4f} "
                      f"EV/bet={ev:+.6f}")


if __name__ == "__main__":
    main()
