"""Sweep additive gates on cached 1s kline + round data.

Tests signal strength, pool imbalance, volatility, time-of-day,
and streak filters — all combinable — on the existing 5000-round dataset.
No API calls needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from itertools import product

ROUNDS_PATH = Path("var/closed_rounds.jsonl")
DATA_PATH = Path("var/cutoff_spot_prices.jsonl")

# Base signal params (best from sweep)
CUTOFF_SECONDS = 5
LOOKBACK_SECONDS = 20
THRESHOLD = 0.0005
TREASURY_FEE = 0.03
BET_SIZE_BNB = 0.05
GAS_BET = 0.0002
GAS_CLAIM = 0.00025
BNB_WEI = 10**18


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------

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
        return best
    return None


# ---------------------------------------------------------------
# Feature extraction per round
# ---------------------------------------------------------------

def extract_features(rec, rnd):
    """Return dict of features for a single round, or None if unusable."""
    if rnd.get("failed") or rnd.get("position") not in ("Bull", "Bear"):
        return None

    klines_1s = rec["klines_1s"]
    lock_at_ms = rec["lock_at"] * 1000
    cutoff_ms = lock_at_ms - CUTOFF_SECONDS * 1000
    ago_ms = cutoff_ms - LOOKBACK_SECONDS * 1000

    k_now = find_closest(klines_1s, cutoff_ms)
    k_ago = find_closest(klines_1s, ago_ms)
    if k_now is None or k_ago is None:
        return None

    spot_now = k_now[4]
    spot_ago = k_ago[4]
    if spot_ago <= 0:
        return None

    ret = (spot_now / spot_ago) - 1

    # Volatility: std of close prices across all 1s klines
    closes = [k[4] for k in klines_1s]
    mean_c = sum(closes) / len(closes)
    var_c = sum((c - mean_c) ** 2 for c in closes) / len(closes)
    vol = math.sqrt(var_c) / mean_c if mean_c > 0 else 0  # normalized vol

    # Pool imbalance from bets
    bets = rnd.get("bets", [])
    lock_at = rnd["lockAt"]
    bull_wei = 0
    bear_wei = 0
    for b in bets:
        if b["createdAt"] > lock_at:
            continue
        if b["position"] == "Bull":
            bull_wei += b["amountWei"]
        else:
            bear_wei += b["amountWei"]
    total_wei = bull_wei + bear_wei

    if total_wei > 0:
        bull_frac = bull_wei / total_wei
        bear_frac = bear_wei / total_wei
    else:
        bull_frac = 0.5
        bear_frac = 0.5

    # Time of day (UTC hour)
    import datetime
    dt = datetime.datetime.fromtimestamp(lock_at, tz=datetime.timezone.utc)
    hour = dt.hour

    return {
        "epoch": rec["epoch"],
        "ret": ret,
        "abs_ret": abs(ret),
        "direction": "Bull" if ret > 0 else "Bear",
        "outcome": rnd["position"],
        "vol": vol,
        "bull_frac": bull_frac,
        "bear_frac": bear_frac,
        "total_pool_bnb": total_wei / BNB_WEI,
        "bull_pool_bnb": bull_wei / BNB_WEI,
        "bear_pool_bnb": bear_wei / BNB_WEI,
        "hour": hour,
    }


# ---------------------------------------------------------------
# Payout calculation
# ---------------------------------------------------------------

def compute_payout(feat, bet_side):
    """Compute net profit for a bet, accounting for pool impact."""
    bet_wei = int(BET_SIZE_BNB * BNB_WEI)
    bull_pool = int(feat["bull_pool_bnb"] * BNB_WEI)
    bear_pool = int(feat["bear_pool_bnb"] * BNB_WEI)

    if bet_side == "Bull":
        bull_pool += bet_wei
    else:
        bear_pool += bet_wei
    total_pool = bull_pool + bear_pool

    if feat["outcome"] == bet_side:
        # Win
        my_pool = bull_pool if bet_side == "Bull" else bear_pool
        if my_pool <= 0:
            return -BET_SIZE_BNB - GAS_BET
        payout_mult = (total_pool * (1 - TREASURY_FEE)) / my_pool
        credit = BET_SIZE_BNB * payout_mult - GAS_CLAIM
        return credit - BET_SIZE_BNB - GAS_BET
    else:
        # Loss
        return -BET_SIZE_BNB - GAS_BET


# ---------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------

# Each gate is (name, param_values, filter_fn(feat, param) -> bool)
# filter_fn returns True to KEEP the bet

GATES = {
    "signal_strength": {
        "params": [0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003],
        "filter": lambda feat, min_ret: feat["abs_ret"] >= min_ret,
    },
    "pool_contrarian": {
        # Only bet when our side has < X fraction of pool (contrarian = better payout)
        "params": [1.0, 0.6, 0.5, 0.45, 0.4, 0.35],
        "filter": lambda feat, max_frac: (
            feat["bull_frac"] < max_frac if feat["direction"] == "Bull"
            else feat["bear_frac"] < max_frac
        ),
    },
    "min_pool_bnb": {
        # Only bet when total pool is above threshold (enough liquidity)
        "params": [0, 1, 3, 5, 10],
        "filter": lambda feat, min_pool: feat["total_pool_bnb"] >= min_pool,
    },
    "vol_regime": {
        # Only bet when vol is in a certain range
        # (low vol = mean-reversion? high vol = momentum?)
        "params": [
            (0, 0.001),      # very low vol
            (0.0001, 0.002), # low-mid
            (0.0005, 999),   # mid-high
            (0, 999),        # no filter
        ],
        "filter": lambda feat, bounds: bounds[0] <= feat["vol"] <= bounds[1],
    },
    "hour_bucket": {
        # UTC hour ranges
        "params": [
            range(0, 24),    # all hours (no filter)
            range(0, 8),     # Asia session
            range(8, 16),    # Europe session
            range(16, 24),   # US session
            range(12, 20),   # US overlap
        ],
        "filter": lambda feat, hours: feat["hour"] in hours,
    },
}


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def evaluate_gated(features, gate_configs):
    """Evaluate signal with a set of gate configs applied.

    gate_configs: dict of {gate_name: param_value}
    """
    bets = 0
    wins = 0
    total_pnl = 0.0

    for feat in features:
        # Base threshold filter
        if feat["abs_ret"] < THRESHOLD:
            continue

        # Apply all gates
        keep = True
        for gate_name, param in gate_configs.items():
            if not GATES[gate_name]["filter"](feat, param):
                keep = False
                break
        if not keep:
            continue

        bets += 1
        is_win = feat["direction"] == feat["outcome"]
        if is_win:
            wins += 1
        total_pnl += compute_payout(feat, feat["direction"])

    return bets, wins, total_pnl


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    records, rounds_by_epoch = load_data()
    print(f"Loaded {len(records)} rounds with 1s kline data\n")

    # Extract features
    features = []
    for rec in records:
        rnd = rounds_by_epoch.get(rec["epoch"])
        if rnd is None:
            continue
        feat = extract_features(rec, rnd)
        if feat is not None:
            features.append(feat)
    print(f"Extracted features for {len(features)} rounds\n")

    # ---- Individual gate sweeps ----
    print("=" * 80)
    print("INDIVIDUAL GATE SWEEPS (each gate tested alone)")
    print("=" * 80)

    for gate_name, gate_def in GATES.items():
        print(f"\n--- {gate_name} ---")
        print(f"{'param':>20} {'bets':>6} {'wins':>6} {'wr':>8} {'pnl_bnb':>10} {'pnl/bet':>10}")
        print("-" * 65)

        for param in gate_def["params"]:
            bets, wins, pnl = evaluate_gated(features, {gate_name: param})
            if bets >= 10:
                wr = wins / bets
                ppb = pnl / bets
                print(f"{str(param):>20} {bets:>6} {wins:>6} {wr:>7.1%} {pnl:>+10.4f} {ppb:>+10.6f}")

    # ---- Best combo sweep ----
    print("\n" + "=" * 80)
    print("COMBO SWEEP — signal_strength x pool_contrarian x min_pool")
    print("=" * 80)

    results = []
    for strength in [0.0005, 0.001, 0.0015, 0.002]:
        for contrarian in [1.0, 0.5, 0.45, 0.4]:
            for min_pool in [0, 3, 5]:
                gates = {
                    "signal_strength": strength,
                    "pool_contrarian": contrarian,
                    "min_pool_bnb": min_pool,
                }
                bets, wins, pnl = evaluate_gated(features, gates)
                if bets >= 10:
                    wr = wins / bets
                    results.append({
                        "strength": strength,
                        "contrarian": contrarian,
                        "min_pool": min_pool,
                        "bets": bets,
                        "wins": wins,
                        "wr": wr,
                        "pnl": pnl,
                        "pnl_per_bet": pnl / bets,
                    })

    results.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'strength':>8} {'contrar':>8} {'min_pl':>6} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl_bnb':>10} {'pnl/bet':>10}")
    print("-" * 70)
    for r in results[:25]:
        print(f"{r['strength']:>8.4f} {r['contrarian']:>8.2f} {r['min_pool']:>6} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['pnl_per_bet']:>+10.6f}")


if __name__ == "__main__":
    main()
