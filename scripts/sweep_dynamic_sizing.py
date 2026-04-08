"""Sweep dynamic impact-aware bet sizing based on payout asymmetry.

Instead of fixed 0.05 BNB per bet, scale bet size based on the
payout multiple — accounting for how our own bet degrades the multiple.

Approaches:
1. Fixed bet sizes: compare 0.01, 0.02, 0.05, 0.1, 0.2 BNB flat
2. Tiered sizing: bet more when payout mult is high, less when low
3. Kelly-inspired: f* = (p*mult - 1) / (mult - 1), capped
4. Impact-optimized: find bet size that maximizes expected profit per round

Signal: 7s+10s acceleration (thresh=0.0002)
Cutoff: 4s
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROUNDS_PATH = Path("var/closed_rounds.jsonl")
DATA_PATH = Path("var/cutoff_spot_prices.jsonl")

BNB_WEI = 10**18
GAS_BET = 0.0002
GAS_CLAIM = 0.00025
FEE = 0.03
CUTOFF_SECONDS = 4
MIN_BET = 0.001


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


def payout_multiple(bull_wei, bear_wei, side, bet_bnb):
    """Payout multiple accounting for our own bet's pool impact."""
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


def net_profit(bull_wei, bear_wei, side, outcome, bet_bnb):
    """Net profit/loss for a bet of variable size."""
    mult = payout_multiple(bull_wei, bear_wei, side, bet_bnb)
    if outcome == side:
        return bet_bnb * mult - GAS_CLAIM - bet_bnb - GAS_BET
    return -bet_bnb - GAS_BET


def expected_profit(bull_wei, bear_wei, side, bet_bnb, win_rate):
    """Expected profit per bet given assumed win rate."""
    mult = payout_multiple(bull_wei, bear_wei, side, bet_bnb)
    win_pnl = bet_bnb * mult - GAS_CLAIM - bet_bnb - GAS_BET
    lose_pnl = -bet_bnb - GAS_BET
    return win_rate * win_pnl + (1 - win_rate) * lose_pnl


def get_return(klines, cutoff_ms, lookback_s):
    kn = find_closest(klines, cutoff_ms)
    ka = find_closest(klines, cutoff_ms - lookback_s * 1000)
    if not kn or not ka or ka[4] <= 0:
        return None
    return (kn[4] / ka[4]) - 1


def accel_signal(feat):
    """7s+10s acceleration signal."""
    rs = feat.get("ret_7")
    rl = feat.get("ret_10")
    if rs is None or rl is None or rs == 0 or rl == 0:
        return None
    if (rs > 0) != (rl > 0):
        return None
    if max(abs(rs), abs(rl)) < 0.0002:
        return None
    return "Bull" if rs > 0 else "Bear"


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
        for lb in [7, 10]:
            feat[f"ret_{lb}"] = get_return(kl, cutoff_ms, lb)
        features.append(feat)

    # Filter to signal-firing rounds
    signals = []
    for f in features:
        direction = accel_signal(f)
        if direction is not None:
            f["direction"] = direction
            signals.append(f)

    print(f"Signal fires on {len(signals)} of {len(features)} rounds\n")

    # --- Pool size distribution ---
    pool_sizes = [(f["bull_wei"] + f["bear_wei"]) / BNB_WEI for f in signals]
    pool_sizes.sort()
    n = len(pool_sizes)
    print("Pool size distribution (BNB):")
    print(f"  min={pool_sizes[0]:.1f}  p10={pool_sizes[int(n*0.1)]:.1f}  "
          f"p25={pool_sizes[int(n*0.25)]:.1f}  median={pool_sizes[n//2]:.1f}  "
          f"p75={pool_sizes[int(n*0.75)]:.1f}  p90={pool_sizes[int(n*0.9)]:.1f}  "
          f"max={pool_sizes[-1]:.1f}")

    # --- Payout multiple distribution (at 0.05 BNB bet) ---
    mults = [payout_multiple(f["bull_wei"], f["bear_wei"], f["direction"], 0.05)
             for f in signals]
    mults.sort()
    print(f"\nPayout multiple distribution (at 0.05 BNB bet):")
    print(f"  min={mults[0]:.2f}  p10={mults[int(n*0.1)]:.2f}  "
          f"p25={mults[int(n*0.25)]:.2f}  median={mults[n//2]:.2f}  "
          f"p75={mults[int(n*0.75)]:.2f}  p90={mults[int(n*0.9)]:.2f}  "
          f"max={mults[-1]:.2f}")
    print()

    # =========================================================
    # 1. FIXED BET SIZES — impact comparison
    # =========================================================
    print("=" * 85)
    print("1. FIXED BET SIZE COMPARISON (impact-aware)")
    print("=" * 85)

    for bet_size in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        bets, wins, pnl = 0, 0, 0.0
        avg_mult = 0.0
        for f in signals:
            bets += 1
            mult = payout_multiple(f["bull_wei"], f["bear_wei"], f["direction"], bet_size)
            avg_mult += mult
            if f["direction"] == f["outcome"]:
                wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"],
                              f["direction"], f["outcome"], bet_size)
        avg_mult /= bets
        wr = wins / bets
        print(f"  bet={bet_size:.2f} BNB: bets={bets}  WR={wr:.1%}  "
              f"avg_mult={avg_mult:.3f}  PnL={pnl:+.4f}  PnL/bet={pnl/bets:+.6f}  "
              f"ROI={pnl/(bets*bet_size)*100:+.2f}%")

    # =========================================================
    # 2. TIERED SIZING: bet more when mult is high
    # =========================================================
    print("\n" + "=" * 85)
    print("2. TIERED SIZING: scale bet by payout multiple")
    print("=" * 85)

    # Pre-compute the "pre-bet" payout multiple (what we see before placing)
    # Use a tiny bet (0.001) to approximate what the mult looks like before our bet
    for f in signals:
        f["pre_mult"] = payout_multiple(f["bull_wei"], f["bear_wei"],
                                         f["direction"], 0.001)

    tier_configs = [
        ("flat_0.05", lambda m: 0.05),
        ("low0.02_high0.08", lambda m: 0.08 if m >= 2.0 else 0.02),
        ("low0.02_med0.05_high0.1", lambda m: 0.1 if m >= 2.5 else (0.05 if m >= 2.0 else 0.02)),
        ("skip_low_0.05", lambda m: 0.05 if m >= 1.8 else 0.0),
        ("skip_low_0.1", lambda m: 0.1 if m >= 1.8 else 0.0),
        ("linear_0.02_0.1", lambda m: min(0.1, max(0.02, 0.02 + 0.04 * (m - 1.5)))),
        ("linear_0.01_0.15", lambda m: min(0.15, max(0.01, 0.01 + 0.07 * (m - 1.5)))),
        ("aggressive_0.01_0.2", lambda m: min(0.2, max(0.01, 0.01 + 0.095 * (m - 1.5)))),
        ("log_scale", lambda m: min(0.15, max(0.01, 0.01 + 0.05 * math.log(max(m, 1.01))))),
        ("quadratic", lambda m: min(0.15, max(0.01, 0.005 * m * m))),
    ]

    print(f"\n{'config':>25} {'bets':>5} {'wins':>5} {'wr':>7} {'wagered':>10} "
          f"{'pnl':>10} {'roi':>8} {'pnl/bet':>10}")
    print("-" * 90)

    for label, size_fn in tier_configs:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for f in signals:
            bet_size = size_fn(f["pre_mult"])
            if bet_size < MIN_BET:
                continue
            bets += 1
            wagered += bet_size
            if f["direction"] == f["outcome"]:
                wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"],
                              f["direction"], f["outcome"], bet_size)
        if bets > 0:
            wr = wins / bets
            roi = pnl / wagered * 100
            print(f"{label:>25} {bets:>5} {wins:>5} {wr:>6.1%} {wagered:>10.2f} "
                  f"{pnl:>+10.4f} {roi:>+7.2f}% {pnl/bets:>+10.6f}")

    # =========================================================
    # 3. KELLY-INSPIRED SIZING
    # =========================================================
    print("\n" + "=" * 85)
    print("3. KELLY-INSPIRED SIZING: f* = (p*mult - 1) / (mult - 1)")
    print("   p = assumed win rate (from backtest)")
    print("   Bet = min(max_bet, fraction * kelly * bankroll)")
    print("=" * 85)

    # Use the observed WR as the assumed win probability
    overall_wr = sum(1 for f in signals if f["direction"] == f["outcome"]) / len(signals)
    print(f"\n  Observed WR = {overall_wr:.3f}")

    for assumed_wr in [0.58, 0.59, 0.60, 0.605]:
        for kelly_frac in [0.1, 0.25, 0.5]:
            for max_bet in [0.05, 0.1, 0.2, 0.5]:
                bankroll = 50.0
                bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                for f in signals:
                    mult = f["pre_mult"]
                    if mult <= 1.0:
                        continue
                    kelly = (assumed_wr * mult - 1.0) / (mult - 1.0)
                    if kelly <= 0:
                        continue
                    bet_size = min(max_bet, kelly_frac * kelly * bankroll)
                    bet_size = max(MIN_BET, bet_size)
                    bets += 1
                    wagered += bet_size
                    won = f["direction"] == f["outcome"]
                    if won:
                        wins += 1
                    p = net_profit(f["bull_wei"], f["bear_wei"],
                                   f["direction"], f["outcome"], bet_size)
                    pnl += p
                    bankroll += p
                    if bankroll <= 0:
                        break
                if bets > 0:
                    wr = wins / bets
                    roi = pnl / wagered * 100 if wagered > 0 else 0
                    print(f"  wr={assumed_wr:.2f} kelly_frac={kelly_frac:.2f} "
                          f"max={max_bet:.2f}: bets={bets:>4}  WR={wr:.1%}  "
                          f"wagered={wagered:.1f}  PnL={pnl:+.2f}  "
                          f"ROI={roi:+.2f}%  final_bank={bankroll:.2f}")

    # =========================================================
    # 4. OPTIMAL EXPECTED-PROFIT BET SIZE PER ROUND
    # =========================================================
    print("\n" + "=" * 85)
    print("4. IMPACT-OPTIMIZED: find per-round bet size maximizing E[profit]")
    print("   E[profit] = wr * win_pnl + (1-wr) * lose_pnl")
    print("   Searches 0.01 to 2.0 BNB in steps of 0.01")
    print("=" * 85)

    for assumed_wr in [0.58, 0.60, 0.605]:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        size_sum = 0.0
        for f in signals:
            # Find bet size that maximizes expected profit
            best_size = MIN_BET
            best_ep = expected_profit(f["bull_wei"], f["bear_wei"],
                                      f["direction"], MIN_BET, assumed_wr)
            for size_cents in range(1, 201):  # 0.01 to 2.00
                size = size_cents * 0.01
                ep = expected_profit(f["bull_wei"], f["bear_wei"],
                                     f["direction"], size, assumed_wr)
                if ep > best_ep:
                    best_ep = ep
                    best_size = size
            if best_ep <= 0:
                continue  # Skip if no profitable bet size exists
            bets += 1
            wagered += best_size
            size_sum += best_size
            if f["direction"] == f["outcome"]:
                wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"],
                              f["direction"], f["outcome"], best_size)
        if bets > 0:
            avg_size = size_sum / bets
            wr_actual = wins / bets
            roi = pnl / wagered * 100
            print(f"  assumed_wr={assumed_wr:.2f}: bets={bets}  actual_WR={wr_actual:.1%}  "
                  f"avg_bet={avg_size:.3f}  wagered={wagered:.1f}  "
                  f"PnL={pnl:+.4f}  ROI={roi:+.2f}%")

    # --- Distribution of optimal bet sizes ---
    print("\n  Optimal bet size distribution (at assumed_wr=0.605):")
    opt_sizes = []
    for f in signals:
        best_size = MIN_BET
        best_ep = expected_profit(f["bull_wei"], f["bear_wei"],
                                  f["direction"], MIN_BET, 0.605)
        for size_cents in range(1, 201):
            size = size_cents * 0.01
            ep = expected_profit(f["bull_wei"], f["bear_wei"],
                                 f["direction"], size, 0.605)
            if ep > best_ep:
                best_ep = ep
                best_size = size
        if best_ep > 0:
            opt_sizes.append(best_size)
    opt_sizes.sort()
    ns = len(opt_sizes)
    if ns > 0:
        print(f"    n={ns}  min={opt_sizes[0]:.2f}  p10={opt_sizes[int(ns*0.1)]:.2f}  "
              f"p25={opt_sizes[int(ns*0.25)]:.2f}  median={opt_sizes[ns//2]:.2f}  "
              f"p75={opt_sizes[int(ns*0.75)]:.2f}  p90={opt_sizes[int(ns*0.9)]:.2f}  "
              f"max={opt_sizes[-1]:.2f}")

    # =========================================================
    # 5. SIMPLE MULT-BASED SIZING (practical for live use)
    # =========================================================
    print("\n" + "=" * 85)
    print("5. PRACTICAL SIZING: simple rules for live deployment")
    print("   These are easy to implement and reason about")
    print("=" * 85)

    practical_configs = [
        ("flat_0.05", lambda m, d, bw, ew: 0.05),
        # Scale linearly with pool imbalance in our favor
        ("0.02_base+0.03*fav",
         lambda m, d, bw, ew: 0.02 + 0.03 * max(0, min(1, (m - 1.7) / 0.6))),
        # Bet proportional to how much our side is the minority
        ("0.03_base+0.07*fav",
         lambda m, d, bw, ew: 0.03 + 0.07 * max(0, min(1, (m - 1.7) / 0.6))),
        # Skip unfavorable, normal otherwise
        ("skip<1.7_else0.05",
         lambda m, d, bw, ew: 0.05 if m >= 1.7 else 0.0),
        # Two-tier: small bet normally, big bet when favorable
        ("0.03/<2.0_0.08/>=2.0",
         lambda m, d, bw, ew: 0.08 if m >= 2.0 else 0.03),
        # Three-tier
        ("0.02/<1.8_0.05/1.8-2.5_0.1/>2.5",
         lambda m, d, bw, ew: 0.1 if m >= 2.5 else (0.05 if m >= 1.8 else 0.02)),
        # Proportional to our side being the minority (as fraction of total)
        ("prop_minority_max0.15",
         lambda m, d, bw, ew: _prop_minority(m, d, bw, ew, 0.15)),
        ("prop_minority_max0.1",
         lambda m, d, bw, ew: _prop_minority(m, d, bw, ew, 0.1)),
    ]

    print(f"\n{'config':>30} {'bets':>5} {'wins':>5} {'wr':>7} {'wagered':>9} "
          f"{'avg_bet':>8} {'pnl':>10} {'roi':>8}")
    print("-" * 95)

    for label, size_fn in practical_configs:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for f in signals:
            mult = f["pre_mult"]
            bet_size = size_fn(mult, f["direction"], f["bull_wei"], f["bear_wei"])
            if bet_size < MIN_BET:
                continue
            bets += 1
            wagered += bet_size
            if f["direction"] == f["outcome"]:
                wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"],
                              f["direction"], f["outcome"], bet_size)
        if bets > 0:
            wr = wins / bets
            roi = pnl / wagered * 100
            avg_bet = wagered / bets
            print(f"{label:>30} {bets:>5} {wins:>5} {wr:>6.1%} {wagered:>9.2f} "
                  f"{avg_bet:>8.3f} {pnl:>+10.4f} {roi:>+7.2f}%")

    # =========================================================
    # 6. BANKROLL SIMULATION: tiered sizing with sequential bankroll
    # =========================================================
    print("\n" + "=" * 85)
    print("6. BANKROLL SIMULATION: realistic sequential with bankroll tracking")
    print("   Starting bankroll = 50 BNB, max bet = 2% of bankroll")
    print("=" * 85)

    for label, base_size_fn in [
        ("flat_2%_bankroll", lambda m, br: min(0.15, br * 0.02)),
        ("flat_1%_bankroll", lambda m, br: min(0.10, br * 0.01)),
        ("tiered_1-3%", lambda m, br: min(0.15, br * (0.03 if m >= 2.0 else 0.01))),
        ("tiered_0.5-2%", lambda m, br: min(0.15, br * (0.02 if m >= 2.0 else 0.005))),
        ("flat_0.05", lambda m, br: 0.05),
    ]:
        bankroll = 50.0
        peak = 50.0
        bets, wins, pnl = 0, 0, 0.0
        max_dd = 0.0
        for f in signals:
            mult = f["pre_mult"]
            bet_size = base_size_fn(mult, bankroll)
            if bet_size < MIN_BET:
                continue
            bet_size = min(bet_size, bankroll * 0.5)  # never bet more than 50%
            bets += 1
            won = f["direction"] == f["outcome"]
            if won:
                wins += 1
            p = net_profit(f["bull_wei"], f["bear_wei"],
                           f["direction"], f["outcome"], bet_size)
            pnl += p
            bankroll += p
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak
            if dd > max_dd:
                max_dd = dd
            if bankroll <= 0:
                print(f"  {label:>20}: BANKRUPT after {bets} bets")
                break
        else:
            wr = wins / bets if bets > 0 else 0
            print(f"  {label:>20}: bets={bets}  WR={wr:.1%}  "
                  f"PnL={pnl:+.4f}  final_bank={bankroll:.2f}  "
                  f"max_drawdown={max_dd:.1%}")


def _prop_minority(mult, direction, bull_wei, bear_wei, max_bet):
    """Bet proportional to how much our side is the minority."""
    total = bull_wei + bear_wei
    if total <= 0:
        return 0.0
    if direction == "Bull":
        our_frac = bull_wei / total
    else:
        our_frac = bear_wei / total
    # our_frac near 0.5 = balanced = low edge from payout
    # our_frac near 0 = highly skewed in our favor
    # Scale: when our_frac=0.3 → bet more, when our_frac=0.5 → bet less
    scale = max(0, min(1, (0.5 - our_frac) / 0.3))
    return max(MIN_BET, min(max_bet, 0.01 + scale * (max_bet - 0.01)))


if __name__ == "__main__":
    main()
