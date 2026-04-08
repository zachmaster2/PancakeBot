"""Combined sweep: chase +10 BNB per 5000 rounds.

Combines every profitable signal discovered so far:
- BNB acceleration (7+10, 5+10)
- BTC acceleration (5+30, 7+30)
- BTC standalone momentum
- BTC+BNB cross-asset agreement
- BTC fallback for BNB no-signal rounds
- Payout asymmetry gate & dynamic sizing
- Bet size optimization (0.10 BNB sweet spot)

Uses cutoff=4s. No API calls.
"""

from __future__ import annotations

import json
from pathlib import Path

BNB_DATA_PATH = Path("var/cutoff_spot_prices.jsonl")
BTC_DATA_PATH = Path("var/btc_spot_prices.jsonl")
ROUNDS_PATH = Path("var/closed_rounds.jsonl")

BNB_WEI = 10**18
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


def payout_multiple(bull_wei, bear_wei, side, bet_bnb):
    bet_wei = int(bet_bnb * BNB_WEI)
    bw = bull_wei + (bet_wei if side == "Bull" else 0)
    ew = bear_wei + (bet_wei if side == "Bear" else 0)
    tw = bw + ew
    my = bw if side == "Bull" else ew
    return (tw * (1 - FEE)) / my if my > 0 else 0


def net_profit(bw, ew, side, outcome, bet_bnb):
    m = payout_multiple(bw, ew, side, bet_bnb)
    return bet_bnb * m - GAS_CLAIM - bet_bnb - GAS_BET if outcome == side else -bet_bnb - GAS_BET


def get_return(klines, cutoff_ms, lookback_s):
    kn = find_closest(klines, cutoff_ms)
    ka = find_closest(klines, cutoff_ms - lookback_s * 1000)
    if not kn or not ka or ka[4] <= 0:
        return None
    return (kn[4] / ka[4]) - 1


def accel_signal(rets, short_lb, long_lb, thresh):
    rs = rets.get(short_lb)
    rl = rets.get(long_lb)
    if rs is None or rl is None or rs == 0 or rl == 0:
        return None
    if (rs > 0) != (rl > 0):
        return None
    if max(abs(rs), abs(rl)) < thresh:
        return None
    return "Bull" if rs > 0 else "Bear"


def simple_signal(rets, lb, thresh):
    ret = rets.get(lb)
    if ret is None or abs(ret) < thresh:
        return None
    return "Bull" if ret > 0 else "Bear"


def main():
    # Load data
    bnb_by_epoch = {}
    for line in BNB_DATA_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                bnb_by_epoch[r["epoch"]] = r

    btc_by_epoch = {}
    if BTC_DATA_PATH.exists():
        for line in BTC_DATA_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    btc_by_epoch[r["epoch"]] = r

    rounds_by_epoch = {}
    for line in ROUNDS_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rounds_by_epoch[r["epoch"]] = r

    # Build features
    features = []
    for epoch, bnb_rec in bnb_by_epoch.items():
        rnd = rounds_by_epoch.get(epoch)
        if not rnd or rnd.get("failed") or rnd["position"] not in ("Bull", "Bear"):
            continue
        lock_ms = bnb_rec["lock_at"] * 1000
        cutoff_ms = lock_ms - CUTOFF_SECONDS * 1000
        bull_wei, bear_wei = compute_pools(rnd)
        if bull_wei + bear_wei == 0:
            continue

        bnb_rets = {}
        for lb in [3, 5, 7, 10, 15, 20, 30]:
            bnb_rets[lb] = get_return(bnb_rec["klines_1s"], cutoff_ms, lb)

        btc_rets = {}
        btc_rec = btc_by_epoch.get(epoch)
        if btc_rec:
            for lb in [3, 5, 7, 10, 15, 20, 30, 45, 60]:
                btc_rets[lb] = get_return(btc_rec["klines_1s"], cutoff_ms, lb)

        # Pre-bet payout mult (use tiny bet to approximate)
        pre_mult_bull = payout_multiple(bull_wei, bear_wei, "Bull", 0.001)
        pre_mult_bear = payout_multiple(bull_wei, bear_wei, "Bear", 0.001)

        features.append({
            "epoch": epoch,
            "outcome": rnd["position"],
            "bull_wei": bull_wei,
            "bear_wei": bear_wei,
            "bnb_rets": bnb_rets,
            "btc_rets": btc_rets,
            "has_btc": bool(btc_rets),
            "pre_mult_bull": pre_mult_bull,
            "pre_mult_bear": pre_mult_bear,
        })

    n_with_btc = sum(1 for f in features if f["has_btc"])
    print(f"Total rounds: {len(features)}")
    print(f"Rounds with BTC data: {n_with_btc}")
    print()

    # =========================================================
    # Define composite strategies
    # =========================================================

    def strategy_bnb_accel_only(f, bet_bnb):
        """BNB 7+10 acceleration, fixed bet size."""
        d = accel_signal(f["bnb_rets"], 7, 10, 0.0002)
        return (d, bet_bnb) if d else (None, 0)

    def strategy_btc_accel_only(f, bet_bnb):
        """BTC 5+30 acceleration."""
        if not f["has_btc"]:
            return (None, 0)
        d = accel_signal(f["btc_rets"], 5, 30, 0.0003)
        return (d, bet_bnb) if d else (None, 0)

    def strategy_bnb_then_btc_accel(f, bet_bnb):
        """BNB accel primary, BTC accel fallback."""
        d = accel_signal(f["bnb_rets"], 7, 10, 0.0002)
        if d:
            return (d, bet_bnb)
        if not f["has_btc"]:
            return (None, 0)
        d = accel_signal(f["btc_rets"], 5, 30, 0.0003)
        return (d, bet_bnb) if d else (None, 0)

    def strategy_best_accel_any(f, bet_bnb):
        """Try BNB accel, BTC accel, BNB simple, BTC simple — first that fires."""
        # Tier 1: BNB acceleration (highest confidence)
        d = accel_signal(f["bnb_rets"], 7, 10, 0.0002)
        if d:
            return (d, bet_bnb)
        # Tier 2: BTC acceleration
        if f["has_btc"]:
            d = accel_signal(f["btc_rets"], 5, 30, 0.0003)
            if d:
                return (d, bet_bnb)
        # Tier 3: BNB simple with threshold
        d = simple_signal(f["bnb_rets"], 5, 0.0003)
        if d:
            return (d, bet_bnb)
        # Tier 4: BTC simple
        if f["has_btc"]:
            d = simple_signal(f["btc_rets"], 7, 0.0001)
            if d:
                return (d, bet_bnb)
        return (None, 0)

    def strategy_agreement_boost(f, bet_bnb):
        """Bet when BNB and BTC agree. Bet more when both strong."""
        if not f["has_btc"]:
            # No BTC data, fall back to BNB accel
            d = accel_signal(f["bnb_rets"], 7, 10, 0.0002)
            return (d, bet_bnb) if d else (None, 0)

        bnb_d = None
        btc_d = None

        # BNB direction (try multiple)
        for short, long, thresh in [(7, 10, 0.0002), (5, 10, 0.0002)]:
            d = accel_signal(f["bnb_rets"], short, long, thresh)
            if d:
                bnb_d = d
                break
        if bnb_d is None:
            d = simple_signal(f["bnb_rets"], 5, 0.0003)
            if d:
                bnb_d = d

        # BTC direction (try multiple)
        for short, long, thresh in [(5, 30, 0.0003), (7, 30, 0.0003)]:
            d = accel_signal(f["btc_rets"], short, long, thresh)
            if d:
                btc_d = d
                break
        if btc_d is None:
            d = simple_signal(f["btc_rets"], 7, 0.0001)
            if d:
                btc_d = d

        if bnb_d and btc_d:
            if bnb_d == btc_d:
                return (bnb_d, bet_bnb * 1.5)  # agreement boost
            else:
                return (None, 0)  # disagreement = skip
        elif bnb_d:
            return (bnb_d, bet_bnb)
        elif btc_d:
            return (btc_d, bet_bnb * 0.7)  # lower confidence
        return (None, 0)

    def strategy_tiered_confidence(f, base_bet):
        """Multi-tier confidence: scale bet by signal strength."""
        if not f["has_btc"]:
            d = accel_signal(f["bnb_rets"], 7, 10, 0.0002)
            return (d, base_bet) if d else (None, 0)

        # Count how many signals agree
        signals = []

        # BNB signals
        for short, long, thresh in [(7, 10, 0.0002), (5, 10, 0.0002), (5, 20, 0.0003)]:
            d = accel_signal(f["bnb_rets"], short, long, thresh)
            if d:
                signals.append(d)
        for lb, thresh in [(5, 0.0003), (10, 0.0003)]:
            d = simple_signal(f["bnb_rets"], lb, thresh)
            if d:
                signals.append(d)

        # BTC signals
        for short, long, thresh in [(5, 30, 0.0003), (7, 30, 0.0003), (5, 20, 0.0003)]:
            d = accel_signal(f["btc_rets"], short, long, thresh)
            if d:
                signals.append(d)
        for lb, thresh in [(7, 0.0001), (15, 0.0003)]:
            d = simple_signal(f["btc_rets"], lb, thresh)
            if d:
                signals.append(d)

        if not signals:
            return (None, 0)

        bull_count = sum(1 for s in signals if s == "Bull")
        bear_count = sum(1 for s in signals if s == "Bear")
        total = bull_count + bear_count

        if total < 2:
            # Only 1 signal — low confidence
            direction = signals[0]
            return (direction, base_bet * 0.5)

        majority = max(bull_count, bear_count)
        minority = min(bull_count, bear_count)

        if minority > 0 and majority <= minority + 1:
            return (None, 0)  # too close, skip

        direction = "Bull" if bull_count > bear_count else "Bear"
        confidence = majority / total

        if confidence >= 0.8 and total >= 4:
            return (direction, base_bet * 2.0)
        elif confidence >= 0.7 and total >= 3:
            return (direction, base_bet * 1.5)
        elif confidence >= 0.6:
            return (direction, base_bet)
        else:
            return (direction, base_bet * 0.5)

    # =========================================================
    # Run all strategies
    # =========================================================

    # Only test on rounds with BTC data
    btc_features = [f for f in features if f["has_btc"]]
    all_features = features

    strategies = [
        ("BNB 7+10 accel @0.05", strategy_bnb_accel_only, 0.05, all_features),
        ("BNB 7+10 accel @0.10", strategy_bnb_accel_only, 0.10, all_features),
        ("BTC 5+30 accel @0.05", strategy_btc_accel_only, 0.05, btc_features),
        ("BTC 5+30 accel @0.10", strategy_btc_accel_only, 0.10, btc_features),
        ("BNB+BTC accel @0.05", strategy_bnb_then_btc_accel, 0.05, btc_features),
        ("BNB+BTC accel @0.10", strategy_bnb_then_btc_accel, 0.10, btc_features),
        ("best_any @0.05", strategy_best_accel_any, 0.05, btc_features),
        ("best_any @0.10", strategy_best_accel_any, 0.10, btc_features),
        ("agreement_boost @0.05", strategy_agreement_boost, 0.05, btc_features),
        ("agreement_boost @0.10", strategy_agreement_boost, 0.10, btc_features),
        ("tiered_conf @0.05", strategy_tiered_confidence, 0.05, btc_features),
        ("tiered_conf @0.10", strategy_tiered_confidence, 0.10, btc_features),
    ]

    print("=" * 95)
    print("COMPOSITE STRATEGIES")
    print("=" * 95)
    print(f"\n{'strategy':>28} {'rounds':>6} {'bets':>5} {'wins':>5} {'wr':>7} "
          f"{'wagered':>8} {'pnl':>10} {'roi':>7} {'pnl/bet':>10}")
    print("-" * 95)

    for label, strat_fn, base_bet, feature_set in strategies:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for f in feature_set:
            direction, bet_size = strat_fn(f, base_bet)
            if direction is None or bet_size < 0.001:
                continue
            bets += 1
            wagered += bet_size
            if direction == f["outcome"]:
                wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"],
                              direction, f["outcome"], bet_size)
        if bets > 0:
            wr = wins / bets
            roi = pnl / wagered * 100 if wagered > 0 else 0
            # Extrapolate PnL to 5000 rounds
            pnl_per_round = pnl / len(feature_set)
            pnl_5k = pnl_per_round * 5000
            print(f"{label:>28} {len(feature_set):>6} {bets:>5} {wins:>5} {wr:>6.1%} "
                  f"{wagered:>8.1f} {pnl:>+10.4f} {roi:>+6.1f}% {pnl/bets:>+10.6f}"
                  f"  (proj 5k: {pnl_5k:>+.1f})")

    # =========================================================
    # Sweep: BTC accel params with bet sizes
    # =========================================================
    print("\n" + "=" * 95)
    print("BTC ACCELERATION SWEEP (on BTC-available rounds)")
    print("=" * 95)

    results = []
    for short_lb in [3, 5, 7, 10]:
        for long_lb in [15, 20, 30, 45, 60]:
            if long_lb <= short_lb:
                continue
            for thresh in [0.0, 0.0001, 0.0002, 0.0003, 0.0005]:
                for bet_bnb in [0.05, 0.10]:
                    bets, wins, pnl = 0, 0, 0.0
                    for f in btc_features:
                        d = accel_signal(f["btc_rets"], short_lb, long_lb, thresh)
                        if d is None:
                            continue
                        bets += 1
                        if d == f["outcome"]:
                            wins += 1
                        pnl += net_profit(f["bull_wei"], f["bear_wei"],
                                          d, f["outcome"], bet_bnb)
                    if bets >= 30:
                        pnl_5k = pnl / len(btc_features) * 5000
                        results.append({
                            "short": short_lb, "long": long_lb,
                            "thresh": thresh, "bet": bet_bnb,
                            "bets": bets, "wins": wins,
                            "wr": wins / bets, "pnl": pnl,
                            "pnl_5k": pnl_5k,
                        })

    results.sort(key=lambda r: r["pnl_5k"], reverse=True)
    print(f"\n{'short':>5} {'long':>5} {'thresh':>7} {'bet':>5} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'proj_5k':>10}")
    print("-" * 75)
    for r in results[:30]:
        print(f"{r['short']:>4}s {r['long']:>4}s {r['thresh']:>7.4f} {r['bet']:>5.2f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['pnl_5k']:>+10.1f}")

    # =========================================================
    # Combined BNB+BTC accel with payout gate & sizing
    # =========================================================
    print("\n" + "=" * 95)
    print("COMBINED BNB+BTC ACCEL + DYNAMIC SIZING + PAYOUT GATE")
    print("=" * 95)

    combo_results = []
    for bnb_config in [(7, 10, 0.0002), (5, 10, 0.0002)]:
        for btc_config in [(5, 30, 0.0003), (7, 30, 0.0003), (5, 20, 0.0003),
                           (7, 20, 0.0003), (5, 30, 0.0), (7, 30, 0.0)]:
            for base_bet in [0.05, 0.10]:
                for min_pay in [0.0, 1.8, 2.0]:
                    for use_dynamic in [False, True]:
                        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                        bnb_ct, btc_ct = 0, 0
                        for f in btc_features:
                            direction = None
                            source = None

                            # BNB accel
                            d = accel_signal(f["bnb_rets"], *bnb_config)
                            if d:
                                direction = d
                                source = "bnb"
                            else:
                                # BTC accel fallback
                                d = accel_signal(f["btc_rets"], *btc_config)
                                if d:
                                    direction = d
                                    source = "btc"

                            if direction is None:
                                continue

                            # Payout gate
                            mult = f["pre_mult_bull"] if direction == "Bull" else f["pre_mult_bear"]
                            if mult < min_pay:
                                continue

                            # Bet sizing
                            if use_dynamic:
                                # Scale by payout favorability
                                if mult >= 2.5:
                                    bet_size = base_bet * 2.0
                                elif mult >= 2.0:
                                    bet_size = base_bet * 1.5
                                elif mult >= 1.8:
                                    bet_size = base_bet
                                else:
                                    bet_size = base_bet * 0.6
                            else:
                                bet_size = base_bet

                            bets += 1
                            wagered += bet_size
                            if source == "bnb":
                                bnb_ct += 1
                            else:
                                btc_ct += 1
                            if direction == f["outcome"]:
                                wins += 1
                            pnl += net_profit(f["bull_wei"], f["bear_wei"],
                                              direction, f["outcome"], bet_size)

                        if bets >= 30:
                            pnl_5k = pnl / len(btc_features) * 5000
                            combo_results.append({
                                "bnb": f"{bnb_config[0]}+{bnb_config[1]}",
                                "btc": f"{btc_config[0]}+{btc_config[1]}@{btc_config[2]:.4f}",
                                "bet": base_bet, "min_pay": min_pay,
                                "dynamic": use_dynamic,
                                "bets": bets, "bnb_ct": bnb_ct, "btc_ct": btc_ct,
                                "wins": wins, "wr": wins / bets,
                                "pnl": pnl, "pnl_5k": pnl_5k,
                                "wagered": wagered,
                            })

    combo_results.sort(key=lambda r: r["pnl_5k"], reverse=True)
    print(f"\n{'bnb':>5} {'btc':>14} {'bet':>4} {'pay':>4} {'dyn':>3} "
          f"{'bets':>5} {'bnb':>4} {'btc':>4} {'wr':>7} {'pnl':>10} {'proj5k':>8}")
    print("-" * 85)
    for r in combo_results[:30]:
        print(f"{r['bnb']:>5} {r['btc']:>14} {r['bet']:>4.2f} {r['min_pay']:>4.1f} "
              f"{'Y' if r['dynamic'] else 'N':>3} "
              f"{r['bets']:>5} {r['bnb_ct']:>4} {r['btc_ct']:>4} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['pnl_5k']:>+8.1f}")


if __name__ == "__main__":
    main()
