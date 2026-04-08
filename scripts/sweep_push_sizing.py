"""Push sizing hard on proven signals to maximize PnL.

Key insight from sweep_10bnb: only BNB-derived signals work.
BTC is useful ONLY as a confidence booster for BNB signals.

Strategy:
  - Wider BNB accel (more lookback pairs) to find more rounds
  - Aggressive pool-proportional sizing on high-confidence rounds
  - BTC agreement as a confidence multiplier (bigger bet, not new bet)
  - Payout-extreme filter (avoid betting minority side on tiny pools)
  - Sweep cap, base_pct, boost multipliers to find max PnL
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


def main():
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

    # Build round features
    rounds = []
    for epoch, bnb_rec in bnb_by_epoch.items():
        rnd = rounds_by_epoch.get(epoch)
        if not rnd or rnd.get("failed") or rnd["position"] not in ("Bull", "Bear"):
            continue
        lock_ms = bnb_rec["lock_at"] * 1000
        cutoff_ms = lock_ms - CUTOFF_SECONDS * 1000
        bull_wei, bear_wei = compute_pools(rnd)
        if bull_wei + bear_wei == 0:
            continue

        pool_bnb = (bull_wei + bear_wei) / BNB_WEI

        bnb_rets = {}
        for lb in [3, 5, 7, 10, 15, 20, 30]:
            bnb_rets[lb] = get_return(bnb_rec["klines_1s"], cutoff_ms, lb)

        btc_rets = {}
        btc_rec = btc_by_epoch.get(epoch)
        if btc_rec:
            for lb in [5, 7, 10, 15, 20, 30]:
                btc_rets[lb] = get_return(btc_rec["klines_1s"], cutoff_ms, lb)

        pm_bull = payout_multiple(bull_wei, bear_wei, "Bull", 0.001)
        pm_bear = payout_multiple(bull_wei, bear_wei, "Bear", 0.001)

        rounds.append({
            "epoch": epoch,
            "outcome": rnd["position"],
            "bull_wei": bull_wei, "bear_wei": bear_wei,
            "pool_bnb": pool_bnb,
            "pm_bull": pm_bull, "pm_bear": pm_bear,
            "bnb_rets": bnb_rets,
            "btc_rets": btc_rets,
            "has_btc": bool(btc_rets),
        })

    print(f"Total rounds: {len(rounds)}")
    with_btc = sum(1 for r in rounds if r["has_btc"])
    print(f"With BTC: {with_btc}")
    print()

    # =========================================================
    # 1. WIDER BNB ACCEL: try more lookback pairs
    # =========================================================
    print("=" * 85)
    print("1. BNB ACCEL LOOKBACK PAIRS (all at thresh=0.0002)")
    print("=" * 85)

    pairs = [
        (3, 5), (3, 7), (3, 10), (3, 15), (3, 20),
        (5, 7), (5, 10), (5, 15), (5, 20),
        (7, 10), (7, 15), (7, 20), (7, 30),
        (10, 15), (10, 20), (10, 30),
    ]
    pair_results = {}
    for short, long in pairs:
        bets, wins = 0, 0
        epochs = set()
        for r in rounds:
            rs = r["bnb_rets"].get(short)
            rl = r["bnb_rets"].get(long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= 0.0002:
                    d = "Bull" if rs > 0 else "Bear"
                    bets += 1
                    epochs.add(r["epoch"])
                    if d == r["outcome"]:
                        wins += 1
        if bets >= 20:
            wr = wins / bets
            pair_results[(short, long)] = {"bets": bets, "wins": wins, "wr": wr, "epochs": epochs}
            print(f"  {short:>2}+{long:>2}: bets={bets:>5} WR={wr:.1%}")

    # Find union of high-WR pairs
    print("\n  Union analysis:")
    good_pairs = [(s, l) for (s, l), v in pair_results.items() if v["wr"] >= 0.59]
    print(f"  Pairs with WR >= 59%: {good_pairs}")

    # Cumulative union
    union = set()
    pair_order = sorted(good_pairs, key=lambda p: -pair_results[p]["wr"])
    for s, l in pair_order:
        new = pair_results[(s, l)]["epochs"] - union
        union |= pair_results[(s, l)]["epochs"]
        print(f"    +({s},{l}): {len(new)} new, total={len(union)}")

    # Test union as signal
    def wide_bnb_accel(r, pairs_list, thresh=0.0002):
        for short, long in pairs_list:
            rs = r["bnb_rets"].get(short)
            rl = r["bnb_rets"].get(long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= thresh:
                    return "Bull" if rs > 0 else "Bear"
        return None

    # Test different pair sets
    pair_sets = {
        "standard (7+10, 5+10)": [(7, 10), (5, 10)],
        "best3 by WR": pair_order[:3],
        "best5 by WR": pair_order[:5],
        "all good (WR>=59%)": pair_order,
        "wide (WR>=58%)": [(s, l) for (s, l), v in pair_results.items() if v["wr"] >= 0.58],
        "7+10 only": [(7, 10)],
        "5+7 only": [(5, 7)],
        "7+20 only": [(7, 20)],
    }

    print(f"\n{'pair set':>30} {'bets':>5} {'WR':>6} {'PnL@0.10':>10}")
    print("-" * 60)
    for label, ps in pair_sets.items():
        bets, wins, pnl = 0, 0, 0.0
        for r in rounds:
            d = wide_bnb_accel(r, ps)
            if d is None:
                continue
            bets += 1
            if d == r["outcome"]:
                wins += 1
            pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], 0.10)
        if bets > 0:
            print(f"  {label:>30}: {bets:>5} {wins/bets:>5.1%} {pnl:>+10.3f}")

    # =========================================================
    # 2. OPTIMAL STACKED STRATEGY with aggressive sizing
    # =========================================================
    print("\n" + "=" * 85)
    print("2. STACKED: BNB accel + BNB-any+BTC, varying sizing aggression")
    print("=" * 85)

    def stacked_signal(r, accel_pairs, btc_lb=30, btc_thresh=0.0003):
        """Returns (direction, tier, btc_agrees)."""
        # Tier 1: BNB accel
        d = wide_bnb_accel(r, accel_pairs)
        if d:
            # Check BTC
            btc_agrees = False
            if r["has_btc"]:
                btc_r = r["btc_rets"].get(btc_lb)
                if btc_r and abs(btc_r) >= btc_thresh:
                    btc_dir = "Bull" if btc_r > 0 else "Bear"
                    btc_agrees = (btc_dir == d)
            return (d, "accel", btc_agrees)

        # Tier 2: any BNB move + BTC confirmation
        if r["has_btc"]:
            bnb_r = r["bnb_rets"].get(7)
            if bnb_r is not None and bnb_r != 0:
                btc_r = r["btc_rets"].get(btc_lb)
                if btc_r and abs(btc_r) >= btc_thresh:
                    bnb_dir = "Bull" if bnb_r > 0 else "Bear"
                    btc_dir = "Bull" if btc_r > 0 else "Bear"
                    if bnb_dir == btc_dir:
                        return (bnb_dir, "any+btc", True)

        return (None, None, False)

    # Sweep sizing parameters
    best_pnl = -999
    best_config = None
    print(f"\n{'base':>5} {'cap':>5} {'btc_mult':>8} {'pay':>4} {'skip_pm':>7} "
          f"{'bets':>5} {'WR':>6} {'avg':>6} {'wagered':>8} {'PnL':>10} {'ROI':>6}")
    print("-" * 85)

    accel_pairs = [(7, 10), (5, 10)]  # Start with standard

    for base_pct in [0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        for cap in [0.30, 0.40, 0.50, 0.60, 0.70]:
            for btc_mult in [1.0, 1.3, 1.5, 2.0]:
                for use_pay in [True, False]:
                    for skip_pm in [0.0, 1.6, 1.7]:
                        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                        for r in rounds:
                            d, tier, btc_ag = stacked_signal(r, accel_pairs)
                            if d is None:
                                continue

                            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                            if pm < skip_pm:
                                continue

                            bs = max(0.05, r["pool_bnb"] * base_pct)
                            if use_pay:
                                if pm >= 2.0:
                                    bs *= 1.3
                                elif pm < 1.7:
                                    bs *= 0.6
                            if btc_ag:
                                bs *= btc_mult
                            bs = min(cap, bs)

                            bets += 1
                            wagered += bs
                            if d == r["outcome"]:
                                wins += 1
                            pnl += net_profit(r["bull_wei"], r["bear_wei"],
                                              d, r["outcome"], bs)

                        if bets >= 100 and pnl > best_pnl:
                            best_pnl = pnl
                            best_config = (base_pct, cap, btc_mult, use_pay, skip_pm)

    # Print top result
    if best_config:
        base_pct, cap, btc_mult, use_pay, skip_pm = best_config
        print(f"\n  BEST: base={base_pct} cap={cap} btc_mult={btc_mult} "
              f"pay={use_pay} skip_pm={skip_pm}")

    # Now print top 20 configs
    all_configs = []
    for base_pct in [0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        for cap in [0.30, 0.40, 0.50, 0.60, 0.70]:
            for btc_mult in [1.0, 1.3, 1.5, 2.0]:
                for use_pay in [True, False]:
                    for skip_pm in [0.0, 1.6, 1.7]:
                        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                        for r in rounds:
                            d, tier, btc_ag = stacked_signal(r, accel_pairs)
                            if d is None:
                                continue
                            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                            if pm < skip_pm:
                                continue
                            bs = max(0.05, r["pool_bnb"] * base_pct)
                            if use_pay:
                                if pm >= 2.0:
                                    bs *= 1.3
                                elif pm < 1.7:
                                    bs *= 0.6
                            if btc_ag:
                                bs *= btc_mult
                            bs = min(cap, bs)
                            bets += 1
                            wagered += bs
                            if d == r["outcome"]:
                                wins += 1
                            pnl += net_profit(r["bull_wei"], r["bear_wei"],
                                              d, r["outcome"], bs)
                        if bets >= 100:
                            all_configs.append({
                                "base": base_pct, "cap": cap,
                                "btc_mult": btc_mult, "pay": use_pay,
                                "skip_pm": skip_pm,
                                "bets": bets, "wins": wins,
                                "pnl": pnl, "wagered": wagered,
                            })

    all_configs.sort(key=lambda c: c["pnl"], reverse=True)
    print(f"\nTop 20 configs:")
    for c in all_configs[:20]:
        wr = c["wins"] / c["bets"]
        avg = c["wagered"] / c["bets"]
        roi = c["pnl"] / c["wagered"] * 100
        print(f"  base={c['base']:.2f} cap={c['cap']:.1f} btcx={c['btc_mult']:.1f} "
              f"pay={'Y' if c['pay'] else 'N'} skip={c['skip_pm']:.1f}: "
              f"bets={c['bets']:>5} WR={wr:.1%} avg={avg:.3f} "
              f"PnL={c['pnl']:+.3f} ROI={roi:+.1f}%")

    # =========================================================
    # 3. SEPARATE TIER SIZING: different sizing per tier
    # =========================================================
    print("\n" + "=" * 85)
    print("3. SEPARATE TIER SIZING")
    print("=" * 85)

    # Tier 1 (BNB accel) and Tier 2 (BNB any+BTC) get separate sizing
    for t1_base, t1_cap in [(0.06, 0.50), (0.07, 0.50), (0.08, 0.60),
                              (0.10, 0.60), (0.10, 0.70), (0.05, 0.40)]:
        for t2_base, t2_cap in [(0.04, 0.30), (0.05, 0.30), (0.05, 0.40),
                                  (0.06, 0.40)]:
            for btc_mult in [1.3, 1.5, 2.0]:
                bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                t1_bets, t1_pnl = 0, 0.0
                t2_bets, t2_pnl = 0, 0.0
                for r in rounds:
                    d, tier, btc_ag = stacked_signal(r, accel_pairs)
                    if d is None:
                        continue

                    pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]

                    if tier == "accel":
                        bs = max(0.05, r["pool_bnb"] * t1_base)
                        if pm >= 2.0:
                            bs *= 1.3
                        elif pm < 1.7:
                            bs *= 0.6
                        if btc_ag:
                            bs *= btc_mult
                        bs = min(t1_cap, bs)
                    else:  # any+btc
                        bs = max(0.05, r["pool_bnb"] * t2_base)
                        if pm >= 2.0:
                            bs *= 1.2
                        elif pm < 1.7:
                            bs *= 0.7
                        bs = min(t2_cap, bs)

                    bets += 1
                    wagered += bs
                    won = d == r["outcome"]
                    if won:
                        wins += 1
                    p = net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
                    pnl += p
                    if tier == "accel":
                        t1_bets += 1; t1_pnl += p
                    else:
                        t2_bets += 1; t2_pnl += p

                if bets >= 100 and pnl > 5.0:
                    wr = wins / bets
                    avg = wagered / bets
                    print(f"  T1({t1_base:.2f},{t1_cap:.1f}) T2({t2_base:.2f},{t2_cap:.1f}) "
                          f"btcx={btc_mult:.1f}: bets={bets} "
                          f"T1={t1_bets}/{t1_pnl:+.2f} T2={t2_bets}/{t2_pnl:+.2f} "
                          f"WR={wr:.1%} PnL={pnl:+.3f}")

    # =========================================================
    # 4. WIDER ACCEL PAIRS + aggressive sizing
    # =========================================================
    print("\n" + "=" * 85)
    print("4. WIDER ACCEL PAIRS (add 5+7, 7+20, 3+7)")
    print("=" * 85)

    wider_pairs_sets = [
        ("7+10,5+10", [(7, 10), (5, 10)]),
        ("7+10,5+10,5+7", [(7, 10), (5, 10), (5, 7)]),
        ("7+10,5+10,5+7,7+20", [(7, 10), (5, 10), (5, 7), (7, 20)]),
        ("7+10,5+10,5+7,7+20,3+7", [(7, 10), (5, 10), (5, 7), (7, 20), (3, 7)]),
        ("7+10,5+10,5+7,3+10", [(7, 10), (5, 10), (5, 7), (3, 10)]),
        ("7+10,5+10,7+20", [(7, 10), (5, 10), (7, 20)]),
        ("all WR>=59%", pair_order),
    ]

    for ps_label, ps in wider_pairs_sets:
        for base_pct, cap in [(0.05, 0.40), (0.06, 0.50), (0.07, 0.50),
                               (0.08, 0.60), (0.10, 0.60)]:
            bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
            for r in rounds:
                d, tier, btc_ag = stacked_signal(r, ps)
                if d is None:
                    continue
                pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                bs = max(0.05, r["pool_bnb"] * base_pct)
                if pm >= 2.0:
                    bs *= 1.3
                elif pm < 1.7:
                    bs *= 0.6
                if btc_ag:
                    bs *= 1.5
                bs = min(cap, bs)
                bets += 1
                wagered += bs
                if d == r["outcome"]:
                    wins += 1
                pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
            if bets >= 100:
                wr = wins / bets
                avg = wagered / bets
                print(f"  {ps_label:>30} base={base_pct:.2f} cap={cap:.1f}: "
                      f"bets={bets:>5} WR={wr:.1%} avg={avg:.3f} PnL={pnl:+.3f}")

    # =========================================================
    # 5. PER-ROUND OPTIMAL BET (theoretical max)
    # =========================================================
    print("\n" + "=" * 85)
    print("5. PER-ROUND OPTIMAL BET (brute force, assuming WR by tier)")
    print("=" * 85)

    for accel_wr in [0.600, 0.605, 0.610, 0.615]:
        for anybtc_wr in [0.63, 0.645, 0.660]:
            bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
            for r in rounds:
                d, tier, btc_ag = stacked_signal(r, accel_pairs)
                if d is None:
                    continue
                wr = accel_wr if tier == "accel" else anybtc_wr

                # Find optimal bet size
                best_ep = -999
                best_bs = 0
                for s100 in range(1, 101):
                    size = s100 * 0.01
                    pm = payout_multiple(r["bull_wei"], r["bear_wei"], d, size)
                    win_p = size * pm - GAS_CLAIM - size - GAS_BET
                    lose_p = -size - GAS_BET
                    ep = wr * win_p + (1 - wr) * lose_p
                    if ep > best_ep:
                        best_ep = ep
                        best_bs = size
                if best_ep <= 0:
                    continue

                bets += 1
                wagered += best_bs
                if d == r["outcome"]:
                    wins += 1
                pnl += net_profit(r["bull_wei"], r["bear_wei"],
                                  d, r["outcome"], best_bs)

            if bets > 0:
                wr_act = wins / bets
                avg = wagered / bets
                print(f"  accel_wr={accel_wr:.3f} anybtc_wr={anybtc_wr:.3f}: "
                      f"bets={bets} avg={avg:.3f} PnL={pnl:+.3f} actual_WR={wr_act:.1%}")

    # =========================================================
    # 6. BANKROLL SIMULATION of best stacked strategy
    # =========================================================
    print("\n" + "=" * 85)
    print("6. BANKROLL SIMULATION (50 BNB, best configs)")
    print("=" * 85)

    sim_configs = [
        ("L1+L2 flat 0.10", 0.10, 0.10, False, 0.10, 1.0),
        ("L1+L2 pool 5% cap 0.40", 0.05, 0.40, True, 0.05, 1.5),
        ("L1+L2 pool 6% cap 0.50", 0.06, 0.50, True, 0.06, 1.5),
        ("L1+L2 pool 7% cap 0.50", 0.07, 0.50, True, 0.07, 1.5),
        ("L1+L2 pool 8% cap 0.60", 0.08, 0.60, True, 0.08, 1.5),
        ("L1+L2 pool 10% cap 0.60", 0.10, 0.60, True, 0.10, 1.5),
    ]

    for label, base_pct, cap, use_pay, t2_base, btc_mult in sim_configs:
        bankroll = 50.0
        peak = 50.0
        max_dd = 0.0
        bets, wins, pnl = 0, 0, 0.0

        for r in rounds:
            d, tier, btc_ag = stacked_signal(r, accel_pairs)
            if d is None:
                continue

            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]

            if tier == "accel":
                bs = max(0.05, r["pool_bnb"] * base_pct)
            else:
                bs = max(0.05, r["pool_bnb"] * t2_base)

            if use_pay:
                if pm >= 2.0:
                    bs *= 1.3
                elif pm < 1.7:
                    bs *= 0.6
            if btc_ag:
                bs *= btc_mult
            bs = min(cap, bs)
            # Risk management: cap at 2% of bankroll
            bs = min(bs, bankroll * 0.02)
            if bs < 0.01 or bankroll < 5:
                continue

            bets += 1
            won = d == r["outcome"]
            if won:
                wins += 1
            p = net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
            pnl += p
            bankroll += p
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak
            if dd > max_dd:
                max_dd = dd

        if bets > 0:
            print(f"  {label:>35}: bets={bets} WR={wins/bets:.1%} "
                  f"PnL={pnl:+.3f} final={bankroll:.2f} maxDD={max_dd:.1%}")

    # =========================================================
    # 7. HOW FAR CAN SIZING ALONE GET US?
    # =========================================================
    print("\n" + "=" * 85)
    print("7. THEORETICAL MAX: if we sized perfectly per round")
    print("=" * 85)

    # Oracle sizing: use actual outcome to determine if we should bet large
    # This is cheating, but shows the ceiling
    bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
    for r in rounds:
        d, tier, btc_ag = stacked_signal(r, accel_pairs)
        if d is None:
            continue
        if d == r["outcome"]:
            # We win: bet max we can without cratering payout
            bs = min(0.50, r["pool_bnb"] * 0.10)
        else:
            # We lose: bet minimum
            bs = 0.05
        bets += 1
        wagered += bs
        if d == r["outcome"]:
            wins += 1
        pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
    print(f"  Oracle sizing (cheat): bets={bets} WR={wins/bets:.1%} "
          f"avg={wagered/bets:.3f} PnL={pnl:+.3f}")

    # Anti-oracle
    bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
    for r in rounds:
        d, tier, btc_ag = stacked_signal(r, accel_pairs)
        if d is None:
            continue
        bs = 0.10  # flat
        bets += 1
        wagered += bs
        if d == r["outcome"]:
            wins += 1
        pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
    print(f"  Flat 0.10 (baseline): bets={bets} WR={wins/bets:.1%} "
          f"avg={wagered/bets:.3f} PnL={pnl:+.3f}")

    # What WR would we need at 0.10 flat to hit +10 BNB?
    print(f"\n  At {bets} bets with 0.10 flat:")
    for target in [5, 7, 10, 15]:
        ev_per = target / bets
        # ev = wr * win - (1-wr) * loss
        # Approximate: avg win ~0.073, avg loss ~0.100
        wr_need = (ev_per + 0.1002) / (0.0731 + 0.1002)
        print(f"    +{target} BNB needs WR = {wr_need:.1%}")


if __name__ == "__main__":
    main()
