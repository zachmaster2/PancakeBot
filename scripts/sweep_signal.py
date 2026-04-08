"""Sweep lookback / cutoff / threshold combos on cached 1s kline data.

Reads var/cutoff_spot_prices.jsonl (no API calls) and tests every
combination of parameters, printing a sorted table of results.
"""

from __future__ import annotations

import json
from pathlib import Path

ROUNDS_PATH = Path("var/closed_rounds.jsonl")
DATA_PATH = Path("var/cutoff_spot_prices.jsonl")

# Sweep ranges
CUTOFF_SECONDS_LIST = [5, 10, 17, 25, 30]
LOOKBACK_SECONDS_LIST = [10, 20, 30, 45, 60, 75, 90]
THRESHOLD_LIST = [0.0, 0.0001, 0.0003, 0.0005, 0.001, 0.002]


def load_data():
    records = []
    for line in DATA_PATH.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if not rec.get("error"):
                records.append(rec)

    rounds_by_epoch = {}
    for line in ROUNDS_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rounds_by_epoch[r["epoch"]] = r

    return records, rounds_by_epoch


def find_closest(klines_1s, target_ms):
    best = None
    best_dist = float("inf")
    for k in klines_1s:
        dist = abs(k[0] - target_ms)
        if dist < best_dist:
            best_dist = dist
            best = k
    if best is not None and best_dist <= 2000:
        return best[4]  # close price
    return None


def evaluate(records, rounds_by_epoch, cutoff_s, lookback_s, threshold):
    total = 0
    bets = 0
    wins = 0
    skips = 0

    for rec in records:
        rnd = rounds_by_epoch.get(rec["epoch"])
        if not rnd or rnd.get("failed") or rnd.get("position") not in ("Bull", "Bear"):
            continue

        lock_at_ms = rec["lock_at"] * 1000
        cutoff_ms = lock_at_ms - cutoff_s * 1000
        ago_ms = cutoff_ms - lookback_s * 1000

        spot_now = find_closest(rec["klines_1s"], cutoff_ms)
        spot_ago = find_closest(rec["klines_1s"], ago_ms)

        if spot_now is None or spot_ago is None or spot_ago == 0:
            continue

        total += 1
        ret = (spot_now / spot_ago) - 1

        if ret > threshold:
            direction = "Bull"
        elif ret < -threshold:
            direction = "Bear"
        else:
            skips += 1
            continue

        bets += 1
        if direction == rnd["position"]:
            wins += 1

    return total, bets, wins, skips


def main():
    records, rounds_by_epoch = load_data()
    print(f"Loaded {len(records)} rounds with 1s kline data\n")

    results = []
    for cutoff_s in CUTOFF_SECONDS_LIST:
        for lookback_s in LOOKBACK_SECONDS_LIST:
            for threshold in THRESHOLD_LIST:
                total, bets, wins, skips = evaluate(
                    records, rounds_by_epoch, cutoff_s, lookback_s, threshold
                )
                if bets >= 20:  # need minimum sample
                    wr = wins / bets
                    bet_rate = bets / total if total > 0 else 0
                    results.append({
                        "cutoff": cutoff_s,
                        "lookback": lookback_s,
                        "threshold": threshold,
                        "total": total,
                        "bets": bets,
                        "wins": wins,
                        "wr": wr,
                        "bet_rate": bet_rate,
                    })

    # Sort by win rate descending
    results.sort(key=lambda r: r["wr"], reverse=True)

    print(f"{'cutoff':>7} {'lookback':>8} {'threshold':>10} {'bets':>5} {'wins':>5} "
          f"{'win_rate':>8} {'bet_rate':>8} {'total':>5}")
    print("-" * 72)
    for r in results[:40]:
        print(f"{r['cutoff']:>5}s {r['lookback']:>6}s {r['threshold']:>10.4f} "
              f"{r['bets']:>5} {r['wins']:>5} "
              f"{r['wr']:>7.2%} {r['bet_rate']:>7.2%} {r['total']:>5}")


if __name__ == "__main__":
    main()
