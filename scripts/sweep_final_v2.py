"""Final sweep with FULL BTC data (5000 rounds).

Focus: combine proven signals with optimal sizing.
No massive grids — targeted tests based on what we know works.

Best so far: wider pairs (7+10,5+10,5+7) + BNB-any+BTC layer = +7.66 at pool 5% cap 0.40
Now with full BTC data, the BNB-any+BTC layer should have more rounds.
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
            "epoch": epoch, "outcome": rnd["position"],
            "bull_wei": bull_wei, "bear_wei": bear_wei,
            "pool_bnb": pool_bnb,
            "pm_bull": pm_bull, "pm_bear": pm_bear,
            "bnb_rets": bnb_rets, "btc_rets": btc_rets,
            "has_btc": bool(btc_rets),
        })

    with_btc = sum(1 for r in rounds if r["has_btc"])
    print(f"Rounds: {len(rounds)}, with BTC: {with_btc}")

    # =========================================================
    # Signal definitions
    # =========================================================
    ACCEL_PAIRS = [(7, 10), (5, 10), (5, 7)]  # proven best set
    ACCEL_THRESH = 0.0002

    def get_signal(r, btc_lb=30, btc_thresh=0.0003):
        """Returns (direction, tier, btc_agrees, btc_disagrees)."""
        # Tier 1: BNB accel
        for short, long in ACCEL_PAIRS:
            rs = r["bnb_rets"].get(short)
            rl = r["bnb_rets"].get(long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= ACCEL_THRESH:
                    d = "Bull" if rs > 0 else "Bear"
                    btc_ag = btc_dis = False
                    if r["has_btc"]:
                        btc_r = r["btc_rets"].get(btc_lb)
                        if btc_r and abs(btc_r) >= btc_thresh:
                            btc_dir = "Bull" if btc_r > 0 else "Bear"
                            btc_ag = (btc_dir == d)
                            btc_dis = (btc_dir != d)
                    return (d, "accel", btc_ag, btc_dis)

        # Tier 2: any BNB move + BTC confirmation
        if r["has_btc"]:
            bnb_r = r["bnb_rets"].get(7)
            if bnb_r is not None and bnb_r != 0:
                btc_r = r["btc_rets"].get(btc_lb)
                if btc_r and abs(btc_r) >= btc_thresh:
                    bnb_dir = "Bull" if bnb_r > 0 else "Bear"
                    btc_dir = "Bull" if btc_r > 0 else "Bear"
                    if bnb_dir == btc_dir:
                        return (bnb_dir, "any+btc", True, False)

        return (None, None, False, False)

    # =========================================================
    # 1. BASELINE: flat 0.10, full BTC data
    # =========================================================
    print("\n" + "=" * 85)
    print("1. BASELINES (flat bet)")
    print("=" * 85)

    for flat_bet in [0.05, 0.10, 0.15, 0.20]:
        bets, wins, pnl = 0, 0, 0.0
        accel_n = anybtc_n = 0
        for r in rounds:
            d, tier, _, _ = get_signal(r)
            if d is None:
                continue
            bets += 1
            if tier == "accel": accel_n += 1
            else: anybtc_n += 1
            if d == r["outcome"]: wins += 1
            pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], flat_bet)
        print(f"  flat {flat_bet:.2f}: bets={bets} (accel={accel_n}, any+btc={anybtc_n}) "
              f"WR={wins/bets:.1%} PnL={pnl:+.3f}")

    # =========================================================
    # 2. TARGETED SIZING SWEEP (reduced grid)
    # =========================================================
    print("\n" + "=" * 85)
    print("2. SIZING SWEEP (pool-proportional + BTC boost + payout)")
    print("=" * 85)

    results = []
    for base_pct in [0.04, 0.05, 0.06, 0.07]:
        for cap in [0.30, 0.40, 0.50]:
            for btc_mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
                for btc_dis in [1.0, 0.5, 0.3, 0.0]:
                    for pay_hi, pay_lo in [(1.0, 1.0), (1.3, 0.6),
                                            (1.3, 0.5), (1.5, 0.5),
                                            (1.3, 0.3), (1.5, 0.3)]:
                        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                        for r in rounds:
                            d, tier, btc_ag, btc_dis_flag = get_signal(r)
                            if d is None:
                                continue
                            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]

                            bs = max(0.05, r["pool_bnb"] * base_pct)
                            if pm >= 2.0:
                                bs *= pay_hi
                            elif pm < 1.7:
                                bs *= pay_lo
                            if btc_ag:
                                bs *= btc_mult
                            elif btc_dis_flag:
                                bs *= btc_dis
                            bs = min(cap, bs)
                            if btc_dis_flag and btc_dis == 0:
                                continue  # Skip when BTC disagrees
                            if bs < 0.03:
                                continue

                            bets += 1
                            wagered += bs
                            if d == r["outcome"]:
                                wins += 1
                            pnl += net_profit(r["bull_wei"], r["bear_wei"],
                                              d, r["outcome"], bs)

                        if bets >= 50:
                            results.append({
                                "base": base_pct, "cap": cap,
                                "btc": btc_mult, "btc_dis": btc_dis,
                                "pay_hi": pay_hi, "pay_lo": pay_lo,
                                "bets": bets, "wins": wins,
                                "pnl": pnl, "wagered": wagered,
                            })

    results.sort(key=lambda c: c["pnl"], reverse=True)
    print(f"\nTop 30 of {len(results)} configs:")
    print(f"{'base':>5} {'cap':>4} {'btcA':>5} {'btcD':>5} "
          f"{'payH':>5} {'payL':>5} {'bets':>5} {'WR':>6} {'avg':>6} "
          f"{'PnL':>8} {'ROI':>6}")
    print("-" * 80)
    for c in results[:30]:
        wr = c["wins"] / c["bets"]
        avg = c["wagered"] / c["bets"]
        roi = c["pnl"] / c["wagered"] * 100
        print(f"{c['base']:>5.2f} {c['cap']:>4.1f} {c['btc']:>5.1f} "
              f"{c['btc_dis']:>5.1f} {c['pay_hi']:>5.1f} {c['pay_lo']:>5.1f} "
              f"{c['bets']:>5} {wr:>5.1%} {avg:>6.3f} "
              f"{c['pnl']:>+8.3f} {roi:>+5.1f}%")

    # =========================================================
    # 3. SKIP BTC-DISAGREE ROUNDS
    # =========================================================
    print("\n" + "=" * 85)
    print("3. SKIP BTC-DISAGREE (only bet when BTC agrees or no BTC signal)")
    print("=" * 85)

    for base_pct in [0.05, 0.06, 0.07]:
        for cap in [0.30, 0.40, 0.50]:
            for btc_mult in [1.5, 2.0, 2.5]:
                bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                for r in rounds:
                    d, tier, btc_ag, btc_dis = get_signal(r)
                    if d is None:
                        continue
                    if btc_dis:
                        continue  # Skip when BTC disagrees
                    pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                    bs = max(0.05, r["pool_bnb"] * base_pct)
                    if pm >= 2.0:
                        bs *= 1.3
                    elif pm < 1.7:
                        bs *= 0.5
                    if btc_ag:
                        bs *= btc_mult
                    bs = min(cap, bs)
                    bets += 1
                    wagered += bs
                    if d == r["outcome"]:
                        wins += 1
                    pnl += net_profit(r["bull_wei"], r["bear_wei"],
                                      d, r["outcome"], bs)
                if bets > 0:
                    wr = wins / bets
                    avg = wagered / bets
                    roi = pnl / wagered * 100
                    print(f"  base={base_pct:.2f} cap={cap:.1f} btcx={btc_mult:.1f}: "
                          f"bets={bets} WR={wr:.1%} avg={avg:.3f} "
                          f"PnL={pnl:+.3f} ROI={roi:+.1f}%")

    # =========================================================
    # 4. TIER-SPECIFIC WR and SIZING ANALYSIS
    # =========================================================
    print("\n" + "=" * 85)
    print("4. PER-TIER WR ANALYSIS")
    print("=" * 85)

    tier_stats = {}
    for r in rounds:
        d, tier, btc_ag, btc_dis = get_signal(r)
        if d is None:
            continue
        subtier = tier
        if tier == "accel":
            if btc_ag:
                subtier = "accel+btc_agree"
            elif btc_dis:
                subtier = "accel+btc_disagree"
            else:
                subtier = "accel+no_btc_sig"
        tier_stats.setdefault(subtier, {"n": 0, "w": 0})
        tier_stats[subtier]["n"] += 1
        if d == r["outcome"]:
            tier_stats[subtier]["w"] += 1

    for t, s in sorted(tier_stats.items(), key=lambda x: -x[1]["n"]):
        wr = s["w"] / s["n"]
        print(f"  {t:>25}: n={s['n']:>4} WR={wr:.1%}")

    # =========================================================
    # 5. SPLIT BNB ACCEL INTO BTC-AGREE vs NOT, SIZE DIFFERENTLY
    # =========================================================
    print("\n" + "=" * 85)
    print("5. BNB ACCEL: BTC-agree gets big bet, no-BTC gets moderate, BTC-disagree gets small/skip")
    print("=" * 85)

    for agree_mult in [1.5, 2.0, 2.5, 3.0]:
        for no_sig_mult in [0.8, 1.0, 1.2]:
            for dis_mult in [0.0, 0.3, 0.5]:
                for base_pct in [0.05, 0.06]:
                    for cap in [0.30, 0.40, 0.50]:
                        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                        for r in rounds:
                            d, tier, btc_ag, btc_dis = get_signal(r)
                            if d is None:
                                continue
                            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                            bs = max(0.05, r["pool_bnb"] * base_pct)
                            # Payout
                            if pm >= 2.0: bs *= 1.3
                            elif pm < 1.7: bs *= 0.5
                            # BTC tier
                            if btc_ag:
                                bs *= agree_mult
                            elif btc_dis:
                                if dis_mult == 0:
                                    continue
                                bs *= dis_mult
                            else:
                                bs *= no_sig_mult
                            bs = min(cap, bs)
                            if bs < 0.03:
                                continue
                            bets += 1
                            wagered += bs
                            if d == r["outcome"]: wins += 1
                            pnl += net_profit(r["bull_wei"], r["bear_wei"],
                                              d, r["outcome"], bs)
                        if bets >= 50 and pnl > 7.5:
                            wr = wins / bets
                            print(f"  ag={agree_mult:.1f} no={no_sig_mult:.1f} dis={dis_mult:.1f} "
                                  f"base={base_pct:.2f} cap={cap:.1f}: "
                                  f"bets={bets} WR={wr:.1%} PnL={pnl:+.3f}")

    # =========================================================
    # 6. BEST CONFIGS: stability check
    # =========================================================
    print("\n" + "=" * 85)
    print("6. STABILITY: best config on halves + quartiles")
    print("=" * 85)

    if results:
        best = results[0]
        half = len(rounds) // 2
        for label, subset in [("Full", rounds), ("1st half", rounds[:half]),
                                ("2nd half", rounds[half:])]:
            bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
            for r in subset:
                d, tier, btc_ag, btc_dis_flag = get_signal(r)
                if d is None:
                    continue
                pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                bs = max(0.05, r["pool_bnb"] * best["base"])
                if pm >= 2.0: bs *= best["pay_hi"]
                elif pm < 1.7: bs *= best["pay_lo"]
                if btc_ag: bs *= best["btc"]
                elif btc_dis_flag: bs *= best["btc_dis"]
                bs = min(best["cap"], bs)
                if best["btc_dis"] == 0 and btc_dis_flag:
                    continue
                if bs < 0.03:
                    continue
                bets += 1
                wagered += bs
                if d == r["outcome"]: wins += 1
                pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
            if bets > 0:
                print(f"  {label:>10}: bets={bets} WR={wins/bets:.1%} PnL={pnl:+.3f}")

        q = len(rounds) // 4
        for i in range(4):
            subset = rounds[i*q:(i+1)*q]
            bets, wins, pnl = 0, 0, 0.0
            for r in subset:
                d, tier, btc_ag, btc_dis_flag = get_signal(r)
                if d is None:
                    continue
                pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                bs = max(0.05, r["pool_bnb"] * best["base"])
                if pm >= 2.0: bs *= best["pay_hi"]
                elif pm < 1.7: bs *= best["pay_lo"]
                if btc_ag: bs *= best["btc"]
                elif btc_dis_flag: bs *= best["btc_dis"]
                bs = min(best["cap"], bs)
                if best["btc_dis"] == 0 and btc_dis_flag:
                    continue
                if bs < 0.03:
                    continue
                bets += 1
                if d == r["outcome"]: wins += 1
                pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
            if bets > 0:
                print(f"  Q{i+1}: bets={bets} WR={wins/bets:.1%} PnL={pnl:+.3f}")

    # =========================================================
    # 7. BANKROLL SIM with best
    # =========================================================
    print("\n" + "=" * 85)
    print("7. BANKROLL SIM (50 BNB)")
    print("=" * 85)

    if results:
        best = results[0]
        for risk_cap in [0.01, 0.015, 0.02, 0.03, 1.0]:
            bankroll = 50.0
            peak = 50.0
            max_dd = 0.0
            bets, wins, pnl = 0, 0, 0.0
            for r in rounds:
                d, tier, btc_ag, btc_dis_flag = get_signal(r)
                if d is None:
                    continue
                pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                bs = max(0.05, r["pool_bnb"] * best["base"])
                if pm >= 2.0: bs *= best["pay_hi"]
                elif pm < 1.7: bs *= best["pay_lo"]
                if btc_ag: bs *= best["btc"]
                elif btc_dis_flag: bs *= best["btc_dis"]
                bs = min(best["cap"], bs)
                if best["btc_dis"] == 0 and btc_dis_flag:
                    continue
                if risk_cap < 1.0:
                    bs = min(bs, bankroll * risk_cap)
                if bs < 0.01 or bankroll < 5:
                    continue
                bets += 1
                won = d == r["outcome"]
                if won: wins += 1
                p = net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
                pnl += p
                bankroll += p
                if bankroll > peak: peak = bankroll
                dd = (peak - bankroll) / peak
                if dd > max_dd: max_dd = dd
            label = f"risk={risk_cap:.0%}" if risk_cap < 1.0 else "uncapped"
            if bets > 0:
                print(f"  {label:>10}: bets={bets} WR={wins/bets:.1%} "
                      f"PnL={pnl:+.3f} final={bankroll:.2f} maxDD={max_dd:.1%}")

    # =========================================================
    # 8. WHAT IF: additional BTC lookbacks for tier 2
    # =========================================================
    print("\n" + "=" * 85)
    print("8. TIER 2 BTC LOOKBACK SWEEP")
    print("=" * 85)

    for btc_lb in [10, 15, 20, 30, 45]:
        for btc_thresh in [0.0001, 0.0002, 0.0003, 0.0004]:
            bets, wins, pnl = 0, 0, 0.0
            accel_n = anybtc_n = 0
            for r in rounds:
                d, tier, btc_ag, btc_dis = get_signal(r, btc_lb=btc_lb,
                                                        btc_thresh=btc_thresh)
                if d is None:
                    continue
                bets += 1
                if tier == "accel": accel_n += 1
                else: anybtc_n += 1
                if d == r["outcome"]: wins += 1
                pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], 0.10)
            if bets >= 100:
                wr = wins / bets
                print(f"  BTC lb={btc_lb:>2}s thresh={btc_thresh:.4f}: "
                      f"bets={bets:>5} (accel={accel_n} any+btc={anybtc_n}) "
                      f"WR={wr:.1%} PnL={pnl:+.3f}")


if __name__ == "__main__":
    main()
