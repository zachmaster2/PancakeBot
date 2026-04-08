"""Final push to +10 BNB: combine wider accel pairs + optimal sizing.

Best findings so far:
  - Wider pairs (7+10, 5+10, 5+7): 803 bets, 61.9% WR, +7.66 at pool 5% cap 0.40
  - BTC boost x2.0 helps on standard pairs
  - Need to combine both and test more aggressively

This script exhaustively combines pair set x sizing x BTC boost to find max.
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

    print(f"Rounds: {len(rounds)}, with BTC: {sum(1 for r in rounds if r['has_btc'])}")
    print()

    # Signal function
    def get_signal(r, accel_pairs, thresh=0.0002, btc_lb=30, btc_thresh=0.0003):
        # Tier 1: BNB accel
        for short, long in accel_pairs:
            rs = r["bnb_rets"].get(short)
            rl = r["bnb_rets"].get(long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= thresh:
                    d = "Bull" if rs > 0 else "Bear"
                    # BTC check
                    btc_ag = False
                    btc_dis = False
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
    # EXHAUSTIVE SWEEP: pair set x sizing x BTC boost
    # =========================================================
    pair_sets = {
        "std": [(7, 10), (5, 10)],
        "wide3": [(7, 10), (5, 10), (5, 7)],
        "wide4": [(7, 10), (5, 10), (5, 7), (3, 10)],
        "wide5": [(7, 10), (5, 10), (5, 7), (7, 20)],
    }

    all_results = []

    for ps_name, ps in pair_sets.items():
        for base_pct in [0.04, 0.05, 0.06, 0.07]:
            for cap in [0.30, 0.40, 0.50]:
                for btc_mult in [1.0, 1.3, 1.5, 2.0, 2.5]:
                    for btc_dis_mult in [1.0, 0.5, 0.3]:
                        for pay_hi in [1.0, 1.2, 1.3, 1.5]:
                            for pay_lo in [1.0, 0.7, 0.5, 0.3]:
                                bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                                for r in rounds:
                                    d, tier, btc_ag, btc_dis = get_signal(r, ps)
                                    if d is None:
                                        continue
                                    pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]

                                    bs = max(0.05, r["pool_bnb"] * base_pct)
                                    # Payout adjustment
                                    if pm >= 2.0:
                                        bs *= pay_hi
                                    elif pm < 1.7:
                                        bs *= pay_lo
                                    # BTC adjustment
                                    if btc_ag:
                                        bs *= btc_mult
                                    elif btc_dis:
                                        bs *= btc_dis_mult
                                    bs = min(cap, bs)
                                    if bs < 0.03:
                                        continue

                                    bets += 1
                                    wagered += bs
                                    if d == r["outcome"]:
                                        wins += 1
                                    pnl += net_profit(r["bull_wei"], r["bear_wei"],
                                                      d, r["outcome"], bs)

                                if bets >= 100:
                                    all_results.append({
                                        "ps": ps_name, "base": base_pct,
                                        "cap": cap, "btc": btc_mult,
                                        "btc_dis": btc_dis_mult,
                                        "pay_hi": pay_hi, "pay_lo": pay_lo,
                                        "bets": bets, "wins": wins,
                                        "pnl": pnl, "wagered": wagered,
                                    })

    all_results.sort(key=lambda c: c["pnl"], reverse=True)

    print("=" * 100)
    print("TOP 30 CONFIGS")
    print("=" * 100)
    print(f"{'ps':>6} {'base':>5} {'cap':>4} {'btcA':>5} {'btcD':>5} "
          f"{'payH':>5} {'payL':>5} {'bets':>5} {'WR':>6} {'avg':>6} "
          f"{'PnL':>8} {'ROI':>6}")
    print("-" * 95)
    for c in all_results[:30]:
        wr = c["wins"] / c["bets"]
        avg = c["wagered"] / c["bets"]
        roi = c["pnl"] / c["wagered"] * 100
        print(f"{c['ps']:>6} {c['base']:>5.2f} {c['cap']:>4.1f} {c['btc']:>5.1f} "
              f"{c['btc_dis']:>5.1f} {c['pay_hi']:>5.1f} {c['pay_lo']:>5.1f} "
              f"{c['bets']:>5} {wr:>5.1%} {avg:>6.3f} "
              f"{c['pnl']:>+8.3f} {roi:>+5.1f}%")

    # =========================================================
    # ZOOM IN: best pair set, fine-tune sizing
    # =========================================================
    best = all_results[0]
    print(f"\nBest pair set: {best['ps']}")
    print(f"Zooming in on fine-tuned sizing...")

    best_ps = pair_sets[best["ps"]]

    fine_results = []
    for base_pct in [x * 0.005 for x in range(6, 16)]:  # 0.030 to 0.075
        for cap in [x * 0.05 for x in range(4, 12)]:  # 0.20 to 0.55
            for btc_mult in [x * 0.25 for x in range(4, 13)]:  # 1.0 to 3.0
                for btc_dis in [x * 0.1 for x in range(2, 11)]:  # 0.2 to 1.0
                    for pay_hi in [x * 0.1 for x in range(10, 18)]:  # 1.0 to 1.7
                        for pay_lo in [x * 0.1 for x in range(2, 11)]:  # 0.2 to 1.0
                            bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
                            for r in rounds:
                                d, tier, btc_ag, btc_dis_flag = get_signal(r, best_ps)
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
                                if bs < 0.03:
                                    continue

                                bets += 1
                                wagered += bs
                                if d == r["outcome"]:
                                    wins += 1
                                pnl += net_profit(r["bull_wei"], r["bear_wei"],
                                                  d, r["outcome"], bs)

                            if bets >= 100:
                                fine_results.append({
                                    "base": base_pct, "cap": cap,
                                    "btc": btc_mult, "btc_dis": btc_dis,
                                    "pay_hi": pay_hi, "pay_lo": pay_lo,
                                    "bets": bets, "wins": wins,
                                    "pnl": pnl, "wagered": wagered,
                                })

    fine_results.sort(key=lambda c: c["pnl"], reverse=True)
    print(f"\nFine-tuned TOP 20:")
    for c in fine_results[:20]:
        wr = c["wins"] / c["bets"]
        avg = c["wagered"] / c["bets"]
        roi = c["pnl"] / c["wagered"] * 100
        print(f"  base={c['base']:.3f} cap={c['cap']:.2f} btcA={c['btc']:.2f} "
              f"btcD={c['btc_dis']:.1f} payH={c['pay_hi']:.1f} payL={c['pay_lo']:.1f}: "
              f"bets={c['bets']:>5} WR={wr:.1%} avg={avg:.3f} "
              f"PnL={c['pnl']:+.3f} ROI={roi:+.1f}%")

    # =========================================================
    # STABILITY CHECK: split in halves
    # =========================================================
    print("\n" + "=" * 85)
    print("STABILITY: best config on 1st half vs 2nd half")
    print("=" * 85)

    best_c = fine_results[0]
    half = len(rounds) // 2
    for label, subset in [("1st half", rounds[:half]), ("2nd half", rounds[half:])]:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for r in subset:
            d, tier, btc_ag, btc_dis_flag = get_signal(r, best_ps)
            if d is None:
                continue
            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
            bs = max(0.05, r["pool_bnb"] * best_c["base"])
            if pm >= 2.0:
                bs *= best_c["pay_hi"]
            elif pm < 1.7:
                bs *= best_c["pay_lo"]
            if btc_ag:
                bs *= best_c["btc"]
            elif btc_dis_flag:
                bs *= best_c["btc_dis"]
            bs = min(best_c["cap"], bs)
            if bs < 0.03:
                continue
            bets += 1
            wagered += bs
            if d == r["outcome"]:
                wins += 1
            pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
        if bets > 0:
            wr = wins / bets
            print(f"  {label}: bets={bets} WR={wr:.1%} PnL={pnl:+.3f}")

    # Quartile split
    q = len(rounds) // 4
    for i in range(4):
        subset = rounds[i*q:(i+1)*q]
        bets, wins, pnl = 0, 0, 0.0
        for r in subset:
            d, tier, btc_ag, btc_dis_flag = get_signal(r, best_ps)
            if d is None:
                continue
            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
            bs = max(0.05, r["pool_bnb"] * best_c["base"])
            if pm >= 2.0:
                bs *= best_c["pay_hi"]
            elif pm < 1.7:
                bs *= best_c["pay_lo"]
            if btc_ag:
                bs *= best_c["btc"]
            elif btc_dis_flag:
                bs *= best_c["btc_dis"]
            bs = min(best_c["cap"], bs)
            if bs < 0.03:
                continue
            bets += 1
            if d == r["outcome"]:
                wins += 1
            pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
        if bets > 0:
            print(f"  Q{i+1}: bets={bets} WR={wins/bets:.1%} PnL={pnl:+.3f}")

    # =========================================================
    # BANKROLL SIM with best config
    # =========================================================
    print("\n" + "=" * 85)
    print("BANKROLL SIM (50 BNB)")
    print("=" * 85)

    for risk_pct in [0.01, 0.015, 0.02, 0.03, 0.05, 1.0]:
        bankroll = 50.0
        peak = 50.0
        max_dd = 0.0
        bets, wins, pnl = 0, 0, 0.0
        for r in rounds:
            d, tier, btc_ag, btc_dis_flag = get_signal(r, best_ps)
            if d is None:
                continue
            pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
            bs = max(0.05, r["pool_bnb"] * best_c["base"])
            if pm >= 2.0:
                bs *= best_c["pay_hi"]
            elif pm < 1.7:
                bs *= best_c["pay_lo"]
            if btc_ag:
                bs *= best_c["btc"]
            elif btc_dis_flag:
                bs *= best_c["btc_dis"]
            bs = min(best_c["cap"], bs)
            # Risk cap
            if risk_pct < 1.0:
                bs = min(bs, bankroll * risk_pct)
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
        label = f"risk={risk_pct:.0%}" if risk_pct < 1.0 else "no risk cap"
        if bets > 0:
            print(f"  {label:>15}: bets={bets} WR={wins/bets:.1%} "
                  f"PnL={pnl:+.3f} final={bankroll:.2f} maxDD={max_dd:.1%}")

    # =========================================================
    # WHAT IF: extra BNB signal lookbacks add rounds?
    # =========================================================
    print("\n" + "=" * 85)
    print("EXTRA: testing threshold per pair (some pairs might tolerate lower thresh)")
    print("=" * 85)

    # Maybe some pairs work with lower threshold
    for pair_config in [
        ("7+10@0.0002, 5+10@0.0002, 5+7@0.0002",
         [(7, 10, 0.0002), (5, 10, 0.0002), (5, 7, 0.0002)]),
        ("7+10@0.0002, 5+10@0.0002, 5+7@0.00015",
         [(7, 10, 0.0002), (5, 10, 0.0002), (5, 7, 0.00015)]),
        ("7+10@0.00015, 5+10@0.0002, 5+7@0.0002",
         [(7, 10, 0.00015), (5, 10, 0.0002), (5, 7, 0.0002)]),
        ("7+10@0.0002, 5+7@0.0002, 7+20@0.00025",
         [(7, 10, 0.0002), (5, 7, 0.0002), (7, 20, 0.00025)]),
        ("7+10@0.0002, 5+7@0.0002, 3+10@0.0002",
         [(7, 10, 0.0002), (5, 7, 0.0002), (3, 10, 0.0002)]),
        ("7+10@0.0002, 5+7@0.0002, 3+7@0.00025",
         [(7, 10, 0.0002), (5, 7, 0.0002), (3, 7, 0.00025)]),
    ]:
        label, pairs = pair_config
        bets, wins, pnl = 0, 0, 0.0
        for r in rounds:
            d = None
            for short, long, thresh in pairs:
                rs = r["bnb_rets"].get(short)
                rl = r["bnb_rets"].get(long)
                if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                    if max(abs(rs), abs(rl)) >= thresh:
                        d = "Bull" if rs > 0 else "Bear"
                        break
            if d is None:
                # Try BNB any + BTC
                if r["has_btc"]:
                    bnb_r = r["bnb_rets"].get(7)
                    if bnb_r is not None and bnb_r != 0:
                        btc_r = r["btc_rets"].get(30)
                        if btc_r and abs(btc_r) >= 0.0003:
                            bnb_dir = "Bull" if bnb_r > 0 else "Bear"
                            btc_dir = "Bull" if btc_r > 0 else "Bear"
                            if bnb_dir == btc_dir:
                                d = bnb_dir
            if d is None:
                continue
            bets += 1
            if d == r["outcome"]:
                wins += 1
            pnl += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], 0.10)
        if bets > 0:
            print(f"  {label}: bets={bets} WR={wins/bets:.1%} PnL={pnl:+.3f}")


if __name__ == "__main__":
    main()
