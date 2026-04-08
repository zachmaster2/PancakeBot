"""Full grid sweep: cutoff x lookback x strength x hour filter.

Uses cached 1s kline data — no API calls.
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path

ROUNDS_PATH = Path("var/closed_rounds.jsonl")
DATA_PATH = Path("var/cutoff_spot_prices.jsonl")

BNB_WEI = 10**18
BET = 0.05
GAS_BET = 0.0002
GAS_CLAIM = 0.00025
FEE = 0.03

# Grid
CUTOFFS = [3, 5, 7, 10, 15]
LOOKBACKS = [5, 10, 15, 20, 30, 45, 60]
STRENGTHS = [0.0003, 0.0005, 0.0008, 0.001, 0.0015]
HOUR_FILTERS = {
    "all": range(24),
    "not_eu": list(range(0, 8)) + list(range(16, 24)),
    "asia": range(0, 8),
}


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
    return (best, best_d) if best else (None, float("inf"))


def payout(bull_bnb, bear_bnb, side, outcome):
    bw = int(bull_bnb * BNB_WEI)
    ew = int(bear_bnb * BNB_WEI)
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


def main():
    records, rounds_by_epoch = load_data()
    print(f"Loaded {len(records)} rounds\n")

    # Pre-compute per-round static data (pools, hour, outcome)
    round_meta = {}
    for rec in records:
        rnd = rounds_by_epoch.get(rec["epoch"])
        if not rnd or rnd.get("failed") or rnd["position"] not in ("Bull", "Bear"):
            continue
        bets = rnd.get("bets", [])
        bw, ew = 0, 0
        for b in bets:
            if b["createdAt"] > rnd["lockAt"]:
                continue
            if b["position"] == "Bull":
                bw += b["amountWei"]
            else:
                ew += b["amountWei"]
        dt = datetime.datetime.fromtimestamp(rnd["lockAt"], tz=datetime.timezone.utc)
        round_meta[rec["epoch"]] = {
            "outcome": rnd["position"],
            "hour": dt.hour,
            "bull_bnb": bw / BNB_WEI,
            "bear_bnb": ew / BNB_WEI,
            "lock_at_ms": rnd["lockAt"] * 1000,
            "klines_1s": rec["klines_1s"],
        }

    print(f"Usable rounds: {len(round_meta)}\n")

    # Pre-compute signals for all cutoff x lookback combos
    # signals[epoch][(cutoff, lookback)] = ret or None
    print("Pre-computing signals for all cutoff x lookback combos...")
    signals: dict[int, dict[tuple, float | None]] = {}
    for epoch, meta in round_meta.items():
        signals[epoch] = {}
        kl = meta["klines_1s"]
        lk = meta["lock_at_ms"]
        for cutoff in CUTOFFS:
            cn = lk - cutoff * 1000
            kn, dn = find_closest(kl, cn)
            if kn is None or dn > 2000:
                for lb in LOOKBACKS:
                    signals[epoch][(cutoff, lb)] = None
                continue
            sn = kn[4]
            for lb in LOOKBACKS:
                ca = cn - lb * 1000
                ka, da = find_closest(kl, ca)
                if ka is None or da > 2000 or ka[4] <= 0:
                    signals[epoch][(cutoff, lb)] = None
                else:
                    signals[epoch][(cutoff, lb)] = (sn / ka[4]) - 1

    # Sweep
    print("Running grid sweep...\n")
    results = []
    for cutoff in CUTOFFS:
        for lookback in LOOKBACKS:
            for strength in STRENGTHS:
                for hour_label, hours in HOUR_FILTERS.items():
                    bets = 0
                    wins = 0
                    total_pnl = 0.0

                    for epoch, meta in round_meta.items():
                        ret = signals[epoch].get((cutoff, lookback))
                        if ret is None:
                            continue
                        if abs(ret) < strength:
                            continue
                        if meta["hour"] not in hours:
                            continue

                        direction = "Bull" if ret > 0 else "Bear"
                        bets += 1
                        if direction == meta["outcome"]:
                            wins += 1
                        total_pnl += payout(
                            meta["bull_bnb"], meta["bear_bnb"],
                            direction, meta["outcome"],
                        )

                    if bets >= 15:
                        results.append({
                            "cutoff": cutoff,
                            "lookback": lookback,
                            "strength": strength,
                            "hours": hour_label,
                            "bets": bets,
                            "wins": wins,
                            "wr": wins / bets,
                            "pnl": total_pnl,
                            "ppb": total_pnl / bets,
                        })

    # Sort by PnL
    results.sort(key=lambda r: r["pnl"], reverse=True)

    print(f"{'cut':>4} {'look':>5} {'strength':>8} {'hours':>6} "
          f"{'bets':>5} {'wins':>5} {'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 75)
    for r in results[:50]:
        print(
            f"{r['cutoff']:>3}s {r['lookback']:>4}s {r['strength']:>8.4f} {r['hours']:>6} "
            f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
            f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}"
        )

    # Also print top by pnl/bet (edge quality)
    by_edge = sorted(results, key=lambda r: r["ppb"], reverse=True)
    print(f"\n{'--- TOP BY PNL/BET (edge quality) ---':^75}")
    print(f"{'cut':>4} {'look':>5} {'strength':>8} {'hours':>6} "
          f"{'bets':>5} {'wins':>5} {'wr':>7} {'pnl':>10} {'pnl/bet':>10}")
    print("-" * 75)
    for r in by_edge[:30]:
        print(
            f"{r['cutoff']:>3}s {r['lookback']:>4}s {r['strength']:>8.4f} {r['hours']:>6} "
            f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
            f"{r['pnl']:>+10.4f} {r['ppb']:>+10.6f}"
        )


if __name__ == "__main__":
    main()
