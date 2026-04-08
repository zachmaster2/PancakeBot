"""Push from +9.39 to +10 BNB.

Current best: base=0.05, cap=0.30, btcA=2.0, payH=1.3, payL=0.6
841 bets, 62.1% WR, +9.39 BNB

Targeted tests:
1. Fine-tune cap (0.25-0.40)
2. Fine-tune BTC multipliers
3. Wider accel pairs to add rounds
4. Multiple BTC lookbacks for tier 2
5. Tier-specific sizing (accel vs any+btc get different base)
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
        for lb in [3, 5, 7, 10, 15, 20]:
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

    print(f"Rounds: {len(rounds)}, with BTC: {sum(1 for r in rounds if r['has_btc'])}")

    ACCEL_PAIRS = [(7, 10), (5, 10), (5, 7)]

    def get_signal(r, pairs=ACCEL_PAIRS, btc_lb=30, btc_thresh=0.0003):
        for short, long in pairs:
            rs = r["bnb_rets"].get(short)
            rl = r["bnb_rets"].get(long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= 0.0002:
                    d = "Bull" if rs > 0 else "Bear"
                    btc_ag = btc_dis = False
                    if r["has_btc"]:
                        btc_r = r["btc_rets"].get(btc_lb)
                        if btc_r and abs(btc_r) >= btc_thresh:
                            btc_dir = "Bull" if btc_r > 0 else "Bear"
                            btc_ag = (btc_dir == d)
                            btc_dis = (btc_dir != d)
                    return (d, "accel", btc_ag, btc_dis)
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

    def run_config(base, cap, btcA, btcD, payH, payL, rounds_list=rounds,
                   pairs=ACCEL_PAIRS, btc_lb=30, btc_thresh=0.0003,
                   t2_base=None):
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for r in rounds_list:
            d, tier, btc_ag, btc_dis = get_signal(r, pairs, btc_lb, btc_thresh)
            if d is None:
                continue
            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
            b = t2_base if (t2_base and tier == "any+btc") else base
            bs = max(0.05, r["pool_bnb"] * b)
            if pm >= 2.0: bs *= payH
            elif pm < 1.7: bs *= payL
            if btc_ag: bs *= btcA
            elif btc_dis: bs *= btcD
            bs = min(cap, bs)
            if bs < 0.03: continue
            bets += 1
            wagered += bs
            if d == r["outcome"]: wins += 1
            pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
        return bets, wins, pnl, wagered

    # =========================================================
    # 1. FINE-TUNE around best config
    # =========================================================
    print("\n" + "=" * 85)
    print("1. FINE-TUNE: cap, btcA, payH, payL")
    print("=" * 85)

    results = []
    for base in [0.040, 0.045, 0.050, 0.055, 0.060]:
        for cap in [0.25, 0.28, 0.30, 0.32, 0.35]:
            for btcA in [1.5, 1.7, 1.8, 2.0, 2.2, 2.5]:
                for btcD in [0.5, 0.7, 1.0]:
                    for payH in [1.0, 1.2, 1.3, 1.4]:
                        for payL in [0.4, 0.5, 0.6, 0.7]:
                            bets, wins, pnl, wag = run_config(
                                base, cap, btcA, btcD, payH, payL)
                            if bets >= 100:
                                results.append({
                                    "b": base, "c": cap, "bA": btcA,
                                    "bD": btcD, "pH": payH, "pL": payL,
                                    "bets": bets, "wins": wins,
                                    "pnl": pnl, "wag": wag,
                                })

    results.sort(key=lambda c: c["pnl"], reverse=True)
    print(f"\nTop 30 of {len(results)} configs:")
    for c in results[:30]:
        wr = c["wins"] / c["bets"]
        avg = c["wag"] / c["bets"]
        print(f"  b={c['b']:.3f} c={c['c']:.2f} bA={c['bA']:.1f} "
              f"bD={c['bD']:.1f} pH={c['pH']:.1f} pL={c['pL']:.1f}: "
              f"bets={c['bets']:>4} WR={wr:.1%} avg={avg:.3f} PnL={c['pnl']:+.3f}")

    # =========================================================
    # 2. WIDER PAIR SETS with best sizing
    # =========================================================
    print("\n" + "=" * 85)
    print("2. WIDER PAIR SETS with best sizing from Section 1")
    print("=" * 85)

    best = results[0]
    pair_options = {
        "7+10,5+10,5+7": [(7,10),(5,10),(5,7)],
        "+3+10": [(7,10),(5,10),(5,7),(3,10)],
        "+7+20": [(7,10),(5,10),(5,7),(7,20)],
        "+3+7": [(7,10),(5,10),(5,7),(3,7)],
        "+3+10,3+7": [(7,10),(5,10),(5,7),(3,10),(3,7)],
        "+7+20,3+10": [(7,10),(5,10),(5,7),(7,20),(3,10)],
        "+7+15": [(7,10),(5,10),(5,7),(7,15)],
    }

    for label, ps in pair_options.items():
        bets, wins, pnl, wag = run_config(
            best["b"], best["c"], best["bA"], best["bD"],
            best["pH"], best["pL"], pairs=ps)
        if bets > 0:
            wr = wins / bets
            print(f"  {label:>25}: bets={bets} WR={wr:.1%} PnL={pnl:+.3f}")

    # =========================================================
    # 3. TIER-SPECIFIC BASE (different base for accel vs any+btc)
    # =========================================================
    print("\n" + "=" * 85)
    print("3. TIER-SPECIFIC BASE (any+btc has 65% WR, deserves higher sizing)")
    print("=" * 85)

    for t1_base in [0.04, 0.05, 0.06]:
        for t2_base in [0.06, 0.07, 0.08, 0.10]:
            bets, wins, pnl, wag = run_config(
                t1_base, best["c"], best["bA"], best["bD"],
                best["pH"], best["pL"], t2_base=t2_base)
            if bets > 0:
                wr = wins / bets
                avg = wag / bets
                print(f"  T1 base={t1_base:.2f} T2 base={t2_base:.2f}: "
                      f"bets={bets} WR={wr:.1%} avg={avg:.3f} PnL={pnl:+.3f}")

    # =========================================================
    # 4. MULTI BTC LOOKBACK for tier 2
    # =========================================================
    print("\n" + "=" * 85)
    print("4. MULTI BTC LOOKBACK for tier 2 (use multiple BTC lookbacks)")
    print("=" * 85)

    # Instead of single btc_lb=30, try combining 20 and 30
    for btc_lb_t2 in [20, 30]:
        for btc_thresh_t2 in [0.0002, 0.0003]:
            bets, wins, pnl, wag = run_config(
                best["b"], best["c"], best["bA"], best["bD"],
                best["pH"], best["pL"],
                btc_lb=btc_lb_t2, btc_thresh=btc_thresh_t2)
            if bets > 0:
                wr = wins / bets
                print(f"  btc_lb={btc_lb_t2}s thresh={btc_thresh_t2:.4f}: "
                      f"bets={bets} WR={wr:.1%} PnL={pnl:+.3f}")

    # Custom: use btc 20+30 for tier 2 (either confirms)
    def get_signal_multi_btc(r, pairs=ACCEL_PAIRS):
        for short, long in pairs:
            rs = r["bnb_rets"].get(short)
            rl = r["bnb_rets"].get(long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= 0.0002:
                    d = "Bull" if rs > 0 else "Bear"
                    btc_ag = btc_dis = False
                    if r["has_btc"]:
                        btc_r = r["btc_rets"].get(30)
                        if btc_r and abs(btc_r) >= 0.0003:
                            btc_dir = "Bull" if btc_r > 0 else "Bear"
                            btc_ag = (btc_dir == d)
                            btc_dis = (btc_dir != d)
                    return (d, "accel", btc_ag, btc_dis)
        # Tier 2: try multiple BTC lookbacks
        if r["has_btc"]:
            bnb_r = r["bnb_rets"].get(7)
            if bnb_r is not None and bnb_r != 0:
                bnb_dir = "Bull" if bnb_r > 0 else "Bear"
                for btc_lb, btc_th in [(30, 0.0003), (20, 0.0002)]:
                    btc_r = r["btc_rets"].get(btc_lb)
                    if btc_r and abs(btc_r) >= btc_th:
                        btc_dir = "Bull" if btc_r > 0 else "Bear"
                        if bnb_dir == btc_dir:
                            return (bnb_dir, "any+btc", True, False)
        return (None, None, False, False)

    bets, wins, pnl, wag = 0, 0, 0.0, 0.0
    for r in rounds:
        d, tier, btc_ag, btc_dis = get_signal_multi_btc(r)
        if d is None: continue
        pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
        bs = max(0.05, r["pool_bnb"] * best["b"])
        if pm >= 2.0: bs *= best["pH"]
        elif pm < 1.7: bs *= best["pL"]
        if btc_ag: bs *= best["bA"]
        elif btc_dis: bs *= best["bD"]
        bs = min(best["c"], bs)
        if bs < 0.03: continue
        bets += 1
        wag += bs
        if d == r["outcome"]: wins += 1
        pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
    if bets > 0:
        print(f"\n  Multi-BTC (30@0.0003 OR 20@0.0002): "
              f"bets={bets} WR={wins/bets:.1%} PnL={pnl:+.3f}")

    # Also try: use BTC 20s@0.0002 as separate additional tier 2 round
    # (catches rounds where 30s didn't pass thresh but 20s did)
    def get_signal_wider_t2(r, pairs=ACCEL_PAIRS):
        for short, long in pairs:
            rs = r["bnb_rets"].get(short)
            rl = r["bnb_rets"].get(long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= 0.0002:
                    d = "Bull" if rs > 0 else "Bear"
                    btc_ag = btc_dis = False
                    if r["has_btc"]:
                        btc_r = r["btc_rets"].get(30)
                        if btc_r and abs(btc_r) >= 0.0003:
                            btc_dir = "Bull" if btc_r > 0 else "Bear"
                            btc_ag = (btc_dir == d)
                            btc_dis = (btc_dir != d)
                    return (d, "accel", btc_ag, btc_dis)
        # Tier 2: BNB any move + BTC 30s OR 20s confirms
        if r["has_btc"]:
            for bnb_lb in [7, 5, 10]:
                bnb_r = r["bnb_rets"].get(bnb_lb)
                if bnb_r is not None and bnb_r != 0:
                    bnb_dir = "Bull" if bnb_r > 0 else "Bear"
                    for btc_lb, btc_th in [(30, 0.0003), (20, 0.0002),
                                            (20, 0.0003), (15, 0.0003)]:
                        btc_r = r["btc_rets"].get(btc_lb)
                        if btc_r and abs(btc_r) >= btc_th:
                            btc_dir = "Bull" if btc_r > 0 else "Bear"
                            if bnb_dir == btc_dir:
                                return (bnb_dir, "any+btc", True, False)
                    break  # Only try once for tier 2
        return (None, None, False, False)

    bets, wins, pnl, wag = 0, 0, 0.0, 0.0
    t2_n = 0
    for r in rounds:
        d, tier, btc_ag, btc_dis = get_signal_wider_t2(r)
        if d is None: continue
        pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
        bs = max(0.05, r["pool_bnb"] * best["b"])
        if pm >= 2.0: bs *= best["pH"]
        elif pm < 1.7: bs *= best["pL"]
        if btc_ag: bs *= best["bA"]
        elif btc_dis: bs *= best["bD"]
        bs = min(best["c"], bs)
        if bs < 0.03: continue
        bets += 1
        if tier == "any+btc": t2_n += 1
        wag += bs
        if d == r["outcome"]: wins += 1
        pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
    if bets > 0:
        print(f"  Wider T2 (multi-BTC-lb, multi-BNB-lb): "
              f"bets={bets} (t2={t2_n}) WR={wins/bets:.1%} PnL={pnl:+.3f}")

    # =========================================================
    # 5. FINAL COMBO: wider pairs + wider T2 + tier-specific sizing
    # =========================================================
    print("\n" + "=" * 85)
    print("5. FINAL COMBOS")
    print("=" * 85)

    wider = [(7,10),(5,10),(5,7),(3,10)]

    for label, sig_fn in [
        ("std pairs, std T2", lambda r: get_signal(r)),
        ("std pairs, multi-BTC T2", lambda r: get_signal_multi_btc(r)),
        ("std pairs, wider T2", lambda r: get_signal_wider_t2(r)),
        ("wide4 pairs, std T2", lambda r: get_signal(r, wider)),
        ("wide4 pairs, multi-BTC T2", lambda r: get_signal_multi_btc(r, wider)),
        ("wide4 pairs, wider T2", lambda r: get_signal_wider_t2(r, wider)),
    ]:
        for t2_mult in [1.0, 1.2, 1.5]:
            bets, wins, pnl, wag = 0, 0, 0.0, 0.0
            for r in rounds:
                d, tier, btc_ag, btc_dis = sig_fn(r)
                if d is None: continue
                pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
                b = best["b"]
                if tier == "any+btc":
                    b = best["b"] * t2_mult
                bs = max(0.05, r["pool_bnb"] * b)
                if pm >= 2.0: bs *= best["pH"]
                elif pm < 1.7: bs *= best["pL"]
                if btc_ag: bs *= best["bA"]
                elif btc_dis: bs *= best["bD"]
                bs = min(best["c"], bs)
                if bs < 0.03: continue
                bets += 1
                wag += bs
                if d == r["outcome"]: wins += 1
                pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
            if bets > 0:
                wr = wins / bets
                print(f"  {label:>30} t2x={t2_mult:.1f}: "
                      f"bets={bets} WR={wr:.1%} PnL={pnl:+.3f}")

    # =========================================================
    # 6. STABILITY of best
    # =========================================================
    print("\n" + "=" * 85)
    print("6. STABILITY of overall best")
    print("=" * 85)

    overall_best = results[0]
    half = len(rounds) // 2
    for label, subset in [("Full", rounds), ("1st half", rounds[:half]),
                            ("2nd half", rounds[half:])]:
        b, w, p, wg = run_config(
            overall_best["b"], overall_best["c"], overall_best["bA"],
            overall_best["bD"], overall_best["pH"], overall_best["pL"],
            rounds_list=subset)
        if b > 0:
            print(f"  {label:>10}: bets={b} WR={w/b:.1%} PnL={p:+.3f}")

    q = len(rounds) // 4
    for i in range(4):
        subset = rounds[i*q:(i+1)*q]
        b, w, p, wg = run_config(
            overall_best["b"], overall_best["c"], overall_best["bA"],
            overall_best["bD"], overall_best["pH"], overall_best["pL"],
            rounds_list=subset)
        if b > 0:
            print(f"  Q{i+1}: bets={b} WR={w/b:.1%} PnL={p:+.3f}")

    # =========================================================
    # 7. Summary
    # =========================================================
    print("\n" + "=" * 85)
    print("SUMMARY")
    print("=" * 85)
    best = results[0]
    print(f"  Best config: base={best['b']:.3f} cap={best['c']:.2f} "
          f"btcA={best['bA']:.1f} btcD={best['bD']:.1f} "
          f"payH={best['pH']:.1f} payL={best['pL']:.1f}")
    print(f"  Bets: {best['bets']}, WR: {best['wins']/best['bets']:.1%}, "
          f"PnL: {best['pnl']:+.3f}")
    print(f"  Avg bet: {best['wag']/best['bets']:.3f}")
    print(f"  ROI: {best['pnl']/best['wag']*100:+.1f}%")


if __name__ == "__main__":
    main()
