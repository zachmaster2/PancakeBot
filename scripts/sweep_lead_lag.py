"""Lead-lag sweep: BTC moves first, BNB catches up.

Core hypothesis: When BTC shows clear momentum but BNB's 1s klines
are still flat (no movement), BNB will eventually follow BTC's
direction. This specifically targets the 88% of rounds where BNB
shows zero movement — the exact rounds we currently skip.

Also tests:
- Pool-size-aware bet sizing (bet more on large pools where impact is lower)
- Combined confidence scoring from multiple assets
- Separate WR tracking for BNB-signal vs BTC-signal vs lead-lag rounds
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

    features = []
    for epoch, bnb_rec in bnb_by_epoch.items():
        btc_rec = btc_by_epoch.get(epoch)
        rnd = rounds_by_epoch.get(epoch)
        if btc_rec is None or rnd is None:
            continue
        if rnd.get("failed") or rnd["position"] not in ("Bull", "Bear"):
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
        for lb in [3, 5, 7, 10, 15, 20, 30, 45, 60]:
            btc_rets[lb] = get_return(btc_rec["klines_1s"], cutoff_ms, lb)

        # Check if BNB has moved at various lookbacks
        bnb_any_move = any(
            bnb_rets.get(lb) is not None and bnb_rets[lb] != 0
            for lb in [5, 7, 10]
        )

        features.append({
            "epoch": epoch, "outcome": rnd["position"],
            "bull_wei": bull_wei, "bear_wei": bear_wei,
            "pool_bnb": pool_bnb,
            "bnb_rets": bnb_rets, "btc_rets": btc_rets,
            "bnb_any_move": bnb_any_move,
        })

    print(f"Rounds with both BNB+BTC: {len(features)}")
    bnb_flat = sum(1 for f in features if not f["bnb_any_move"])
    print(f"BNB flat (no move at 5/7/10s): {bnb_flat} ({bnb_flat/len(features)*100:.1f}%)")
    print()

    # =========================================================
    # 1. LEAD-LAG: BTC moved, BNB flat → bet BTC's direction
    # =========================================================
    print("=" * 85)
    print("1. LEAD-LAG: BTC direction when BNB is flat")
    print("   Only fires on rounds where BNB shows NO movement at 5/7/10s")
    print("=" * 85)

    results1 = []
    for btc_lb in [5, 7, 10, 15, 20, 30]:
        for btc_thresh in [0.0, 0.0001, 0.0003, 0.0005]:
            bets, wins, pnl = 0, 0, 0.0
            for f in features:
                if f["bnb_any_move"]:
                    continue  # Only flat-BNB rounds
                ret = f["btc_rets"].get(btc_lb)
                if ret is None or ret == 0 or abs(ret) < btc_thresh:
                    continue
                d = "Bull" if ret > 0 else "Bear"
                bets += 1
                if d == f["outcome"]:
                    wins += 1
                pnl += net_profit(f["bull_wei"], f["bear_wei"], d, f["outcome"], 0.10)
            if bets >= 20:
                pnl_5k = pnl / len(features) * 5000
                results1.append({
                    "btc_lb": btc_lb, "thresh": btc_thresh,
                    "bets": bets, "wins": wins,
                    "wr": wins / bets, "pnl": pnl, "pnl_5k": pnl_5k,
                })

    results1.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'btc_lb':>6} {'thresh':>7} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'proj5k':>8}")
    print("-" * 55)
    for r in results1[:20]:
        print(f"{r['btc_lb']:>5}s {r['thresh']:>7.4f} {r['bets']:>5} {r['wins']:>5} "
              f"{r['wr']:>6.1%} {r['pnl']:>+10.4f} {r['pnl_5k']:>+8.1f}")

    # =========================================================
    # 2. LEAD-LAG with BTC acceleration on flat-BNB rounds
    # =========================================================
    print("\n" + "=" * 85)
    print("2. LEAD-LAG with BTC ACCELERATION on flat-BNB rounds")
    print("=" * 85)

    results2 = []
    for short_lb in [5, 7, 10]:
        for long_lb in [15, 20, 30, 45, 60]:
            if long_lb <= short_lb:
                continue
            for thresh in [0.0, 0.0003, 0.0005]:
                bets, wins, pnl = 0, 0, 0.0
                for f in features:
                    if f["bnb_any_move"]:
                        continue
                    rs = f["btc_rets"].get(short_lb)
                    rl = f["btc_rets"].get(long_lb)
                    if rs is None or rl is None or rs == 0 or rl == 0:
                        continue
                    if (rs > 0) != (rl > 0):
                        continue
                    if max(abs(rs), abs(rl)) < thresh:
                        continue
                    d = "Bull" if rs > 0 else "Bear"
                    bets += 1
                    if d == f["outcome"]:
                        wins += 1
                    pnl += net_profit(f["bull_wei"], f["bear_wei"], d, f["outcome"], 0.10)
                if bets >= 20:
                    pnl_5k = pnl / len(features) * 5000
                    results2.append({
                        "short": short_lb, "long": long_lb, "thresh": thresh,
                        "bets": bets, "wins": wins,
                        "wr": wins / bets, "pnl": pnl, "pnl_5k": pnl_5k,
                    })

    results2.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'short':>5} {'long':>5} {'thresh':>7} {'bets':>5} {'wins':>5} "
          f"{'wr':>7} {'pnl':>10} {'proj5k':>8}")
    print("-" * 60)
    for r in results2[:20]:
        print(f"{r['short']:>4}s {r['long']:>4}s {r['thresh']:>7.4f} "
              f"{r['bets']:>5} {r['wins']:>5} {r['wr']:>6.1%} "
              f"{r['pnl']:>+10.4f} {r['pnl_5k']:>+8.1f}")

    # =========================================================
    # 3. POOL-SIZE-AWARE SIZING: bet more on large pools
    # =========================================================
    print("\n" + "=" * 85)
    print("3. POOL-SIZE-AWARE BET SIZING")
    print("   Bet more on large pools (less impact), less on small pools")
    print("=" * 85)

    # Use the BNB accel signal with pool-aware sizing
    def bnb_accel(f):
        for short, long, thresh in [(7, 10, 0.0002), (5, 10, 0.0002)]:
            rs = f["bnb_rets"].get(short)
            rl = f["bnb_rets"].get(long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= thresh:
                    return "Bull" if rs > 0 else "Bear"
        return None

    sizing_configs = [
        ("flat 0.10", lambda p: 0.10),
        ("flat 0.05", lambda p: 0.05),
        # Pool-proportional: bet up to 3% of pool
        ("3% pool cap 0.20", lambda p: min(0.20, max(0.05, p * 0.03))),
        ("5% pool cap 0.30", lambda p: min(0.30, max(0.05, p * 0.05))),
        ("2% pool cap 0.15", lambda p: min(0.15, max(0.03, p * 0.02))),
        # Tiered by pool size
        ("tier: <2=0.05, 2-4=0.10, >4=0.15",
         lambda p: 0.15 if p >= 4 else (0.10 if p >= 2 else 0.05)),
        ("tier: <2=0.05, 2-4=0.10, >4=0.20",
         lambda p: 0.20 if p >= 4 else (0.10 if p >= 2 else 0.05)),
        ("tier: <3=0.05, 3-5=0.15, >5=0.25",
         lambda p: 0.25 if p >= 5 else (0.15 if p >= 3 else 0.05)),
    ]

    print(f"\n{'config':>38} {'bets':>5} {'wr':>7} {'wagered':>8} {'pnl':>10} {'roi':>7} {'proj5k':>8}")
    print("-" * 95)

    # Use ALL features (not just BTC-available) for BNB-only signal
    all_features = []
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
        for lb in [5, 7, 10]:
            bnb_rets[lb] = get_return(bnb_rec["klines_1s"], cutoff_ms, lb)
        all_features.append({
            "epoch": epoch, "outcome": rnd["position"],
            "bull_wei": bull_wei, "bear_wei": bear_wei,
            "pool_bnb": pool_bnb, "bnb_rets": bnb_rets,
        })

    for label, size_fn in sizing_configs:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for f in all_features:
            d = bnb_accel(f)
            if d is None:
                continue
            bet_size = size_fn(f["pool_bnb"])
            bets += 1
            wagered += bet_size
            if d == f["outcome"]:
                wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"], d, f["outcome"], bet_size)
        if bets > 0:
            wr = wins / bets
            roi = pnl / wagered * 100
            print(f"{label:>38} {bets:>5} {wr:>6.1%} {wagered:>8.1f} "
                  f"{pnl:>+10.4f} {roi:>+6.1f}% {pnl/len(all_features)*5000:>+8.1f}")

    # =========================================================
    # 4. STACKED STRATEGY: BNB accel + lead-lag on flat rounds
    #    Pool-size-aware sizing on both
    # =========================================================
    print("\n" + "=" * 85)
    print("4. STACKED: BNB accel + BTC lead-lag on flat rounds + pool sizing")
    print("=" * 85)

    # Find best lead-lag config from results2
    best_ll = results2[0] if results2 else None
    if best_ll:
        ll_short, ll_long, ll_thresh = best_ll["short"], best_ll["long"], best_ll["thresh"]
        print(f"  Best lead-lag: BTC {ll_short}+{ll_long}s thresh={ll_thresh}")

    for ll_short, ll_long, ll_thresh in [(5, 30, 0.0003), (7, 30, 0.0003),
                                          (5, 20, 0.0), (7, 20, 0.0003),
                                          (5, 30, 0.0), (7, 30, 0.0)]:
        for base_bet in [0.05, 0.10]:
            for use_pool_sizing in [False, True]:
                bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                bnb_ct, btc_ct = 0, 0
                for f in features:
                    d = bnb_accel(f)
                    source = "bnb"
                    if d is None and not f["bnb_any_move"]:
                        # Lead-lag: BTC accel on flat-BNB rounds
                        rs = f["btc_rets"].get(ll_short)
                        rl = f["btc_rets"].get(ll_long)
                        if (rs and rl and rs != 0 and rl != 0
                                and (rs > 0) == (rl > 0)
                                and max(abs(rs), abs(rl)) >= ll_thresh):
                            d = "Bull" if rs > 0 else "Bear"
                            source = "btc"
                    if d is None:
                        continue

                    if use_pool_sizing:
                        p = f["pool_bnb"]
                        bet_size = min(base_bet * 2, max(base_bet * 0.5, p * 0.03))
                    else:
                        bet_size = base_bet

                    bets += 1
                    wagered += bet_size
                    if source == "bnb":
                        bnb_ct += 1
                    else:
                        btc_ct += 1
                    if d == f["outcome"]:
                        wins += 1
                    pnl += net_profit(f["bull_wei"], f["bear_wei"],
                                      d, f["outcome"], bet_size)

                if bets >= 30:
                    wr = wins / bets
                    pnl_5k = pnl / len(features) * 5000
                    ps_label = "pool" if use_pool_sizing else "flat"
                    print(f"  BTC {ll_short}+{ll_long}@{ll_thresh:.4f} bet={base_bet:.2f} "
                          f"sizing={ps_label:>4}: bets={bets:>5} (bnb={bnb_ct} btc={btc_ct}) "
                          f"WR={wr:.1%}  PnL={pnl:+.4f}  proj5k={pnl_5k:+.1f}")

    # =========================================================
    # 5. WHAT DOES THE MATH NEED?
    # =========================================================
    print("\n" + "=" * 85)
    print("5. WHAT DOES +10 BNB REQUIRE?")
    print("=" * 85)
    print("  At 0.10 BNB bet, avg_mult=1.735:")
    print("  Win: +0.0731, Loss: -0.1002")
    for target_pnl in [5, 10, 15, 20]:
        for n_bets in [500, 1000, 1500, 2000, 2500, 3000]:
            ev_per_bet = target_pnl / n_bets
            # ev = wr * 0.0731 + (1-wr) * (-0.1002) = wr * 0.1733 - 0.1002
            wr_needed = (ev_per_bet + 0.1002) / 0.1733
            if 0.55 <= wr_needed <= 0.70:
                print(f"    +{target_pnl} BNB from {n_bets} bets: need WR = {wr_needed:.1%}")


if __name__ == "__main__":
    main()
