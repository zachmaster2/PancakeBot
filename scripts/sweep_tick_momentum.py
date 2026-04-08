"""Sweep alternative momentum signals on cached 1s kline data.

Tests three approaches to increase bet count without diluting WR:
1. Tick-direction: count up/down ticks in last N seconds
2. Cascade lookback: try 5s, fall back to 10s, then 15s
3. Hybrid: tick-direction + endpoint return confirmation

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


def find_n_klines_before(klines, target_ms, n):
    """Return the N klines ending at or just before target_ms, oldest first."""
    # klines are oldest-first
    best_idx = None
    best_d = float("inf")
    for i, k in enumerate(klines):
        d = abs(k[0] - target_ms)
        if d < best_d:
            best_d = d
            best_idx = i
    if best_idx is None or best_d > 2000:
        return None
    start = max(0, best_idx - n + 1)
    result = klines[start:best_idx + 1]
    return result if len(result) >= n else None


def payout(bull_wei, bear_wei, side, outcome):
    bw, ew = int(bull_wei), int(bear_wei)
    mw = int(BET * BNB_WEI)
    if side == "Bull":
        bw += mw
    else:
        ew += mw
    tw = bw + ew
    if outcome == side:
        mp = bw if side == "Bull" else ew
        if mp <= 0:
            return -BET - GAS_BET
        return BET * (tw * (1 - FEE) / mp) - GAS_CLAIM - BET - GAS_BET
    return -BET - GAS_BET


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


def tick_direction_score(klines_window):
    """Count up vs down ticks. Returns (score, n_ticks).

    score = (up_ticks - down_ticks) / n_ticks, range [-1, 1]
    An up-tick is when close > open for a 1s candle.
    """
    up = 0
    down = 0
    for k in klines_window:
        o, c = k[1], k[4]  # open, close
        if c > o:
            up += 1
        elif c < o:
            down += 1
        # flat ticks don't count
    total = up + down
    if total == 0:
        return 0.0, 0
    return (up - down) / total, total


def endpoint_return(klines_window):
    """Return based on first and last kline close prices."""
    if not klines_window or len(klines_window) < 2:
        return None
    first_close = klines_window[0][4]
    last_close = klines_window[-1][4]
    if first_close <= 0:
        return None
    return (last_close / first_close) - 1


def main():
    records, rounds_by_epoch = load_data()
    print(f"Loaded {len(records)} rounds\n")

    # Build features for all rounds
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

        # Pre-compute tick scores and endpoint returns for various windows
        for window in [5, 7, 10, 15, 20]:
            w = find_n_klines_before(kl, cutoff_ms, window)
            if w is not None:
                score, n_active = tick_direction_score(w)
                ret = endpoint_return(w)
                feat[f"tick_{window}"] = score
                feat[f"tick_{window}_n"] = n_active
                feat[f"ret_{window}"] = ret if ret is not None else 0.0
            else:
                feat[f"tick_{window}"] = 0.0
                feat[f"tick_{window}_n"] = 0
                feat[f"ret_{window}"] = 0.0

        features.append(feat)

    print(f"Features: {len(features)} rounds\n")

    # --- How often do tick scores fire vs endpoint returns? ---
    print("Signal firing rates:")
    for window in [5, 7, 10, 15, 20]:
        tick_fires = sum(1 for f in features if abs(f[f"tick_{window}"]) > 0)
        ret_fires = sum(1 for f in features if abs(f[f"ret_{window}"]) > 0.0003)
        print(f"  window={window:>2}s: tick_nonzero={tick_fires:>4} ({tick_fires/len(features)*100:.1f}%)  "
              f"ret>0.0003={ret_fires:>4} ({ret_fires/len(features)*100:.1f}%)")
    print()

    # =========================================================
    # 1. TICK-DIRECTION SIGNAL
    # =========================================================
    print("=" * 85)
    print("1. TICK-DIRECTION SIGNAL: direction = sign(tick_score)")
    print("=" * 85)

    results = []
    for window in [5, 7, 10, 15, 20]:
        for min_score in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            for min_active in [2, 3, 4, 5]:
                bets, wins, pnl = 0, 0, 0.0
                for f in features:
                    score = f[f"tick_{window}"]
                    n_active = f[f"tick_{window}_n"]
                    if abs(score) < min_score or n_active < min_active:
                        continue
                    direction = "Bull" if score > 0 else "Bear"
                    bets += 1
                    if direction == f["outcome"]:
                        wins += 1
                    pnl += payout(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
                if bets >= 20:
                    results.append({
                        "window": window, "min_score": min_score,
                        "min_active": min_active,
                        "bets": bets, "wins": wins,
                        "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                    })

    results.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'win':>4} {'score':>6} {'active':>6} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 65)
    for r in results[:25]:
        print(f"{r['window']:>3}s {r['min_score']:>6.1f} {r['min_active']:>6} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 2. CASCADE LOOKBACK
    # =========================================================
    print("\n" + "=" * 85)
    print("2. CASCADE LOOKBACK: try 5s, fallback to 10s, then 15s")
    print("=" * 85)

    results2 = []
    for threshold in [0.0003, 0.0005, 0.0008]:
        cascade_configs = [
            ("5", [5]),
            ("5>10", [5, 10]),
            ("5>10>15", [5, 10, 15]),
            ("5>10>15>20", [5, 10, 15, 20]),
        ]
        for label, lookbacks in cascade_configs:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                direction = None
                for lb in lookbacks:
                    ret = f[f"ret_{lb}"]
                    if abs(ret) >= threshold:
                        direction = "Bull" if ret > 0 else "Bear"
                        break
                if direction is None:
                    continue
                bets += 1
                if direction == f["outcome"]:
                    wins += 1
                pnl += payout(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
            if bets >= 20:
                results2.append({
                    "cascade": label, "thresh": threshold,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                })

    results2.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'cascade':>12} {'thresh':>7} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 65)
    for r in results2:
        print(f"{r['cascade']:>12} {r['thresh']:>7.4f} {r['bets']:>5} {r['wins']:>5} "
              f"{r['wr']:>6.1%} {r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 3. HYBRID: tick direction + endpoint return confirmation
    # =========================================================
    print("\n" + "=" * 85)
    print("3. HYBRID: tick_direction must agree with endpoint return")
    print("=" * 85)

    results3 = []
    for window in [5, 7, 10, 15]:
        for min_score in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for min_ret in [0.0, 0.0001, 0.0003]:
                bets, wins, pnl = 0, 0, 0.0
                for f in features:
                    score = f[f"tick_{window}"]
                    ret = f[f"ret_{window}"]
                    if abs(score) < min_score:
                        continue
                    tick_dir = "Bull" if score > 0 else "Bear"
                    # If we require endpoint confirmation
                    if min_ret > 0:
                        if abs(ret) < min_ret:
                            continue
                        ret_dir = "Bull" if ret > 0 else "Bear"
                        if ret_dir != tick_dir:
                            continue
                    direction = tick_dir
                    bets += 1
                    if direction == f["outcome"]:
                        wins += 1
                    pnl += payout(f["bull_wei"], f["bear_wei"], direction, f["outcome"])
                if bets >= 20:
                    results3.append({
                        "window": window, "min_score": min_score,
                        "min_ret": min_ret,
                        "bets": bets, "wins": wins,
                        "wr": wins / bets, "pnl": pnl, "ppb": pnl / bets,
                    })

    results3.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'win':>4} {'score':>6} {'min_ret':>8} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 65)
    for r in results3[:25]:
        print(f"{r['window']:>3}s {r['min_score']:>6.1f} {r['min_ret']:>8.4f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}")

    # =========================================================
    # 4. TICK-ONLY vs ENDPOINT-ONLY vs BOTH — head to head
    # =========================================================
    print("\n" + "=" * 85)
    print("4. HEAD-TO-HEAD: best of each approach at comparable bet counts")
    print("=" * 85)
    # Find results near 200-400 bets from each approach
    for label, res_list in [("tick", results), ("cascade", results2), ("hybrid", results3)]:
        close = [r for r in res_list if 150 <= r["bets"] <= 500]
        if close:
            best = max(close, key=lambda r: r["pnl"])
            print(f"\n  {label:>8}: {best}")


if __name__ == "__main__":
    main()
