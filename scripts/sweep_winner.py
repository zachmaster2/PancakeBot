"""Final validation of the +10 BNB strategy.

THE STRATEGY:
  Signal Layer 1 (BNB Acceleration):
    - Pairs: (7,10), (5,10), (5,7) at threshold >= 0.0002
    - Short + long must agree on direction, max(|ret|) >= thresh
  Signal Layer 2 (BNB Any Move + BTC Confirmation):
    - Any nonzero BNB 7s return + BTC 30s agrees, |BTC| >= 0.0003

  Sizing:
    - T1 base: 4% of pool (floor 0.05)
    - T2 base: 7% of pool (floor 0.05) -- higher because 65% WR
    - Cap: 0.35 BNB
    - BTC agree boost: x2.0 (on accel rounds)
    - BTC disagree: keep at 1.0x
    - Payout high (mult >= 2.0): x1.4
    - Payout low (mult < 1.7): x0.7

Tests: stability, bankroll, drawdown, per-tier breakdown, time-series PnL curve.
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

# ===== STRATEGY CONFIG =====
ACCEL_PAIRS = [(7, 10), (5, 10), (5, 7)]
ACCEL_THRESH = 0.0002
T1_BASE = 0.04
T2_BASE = 0.07
CAP = 0.35
BTC_AGREE_MULT = 2.0
BTC_DISAGREE_MULT = 1.0
PAYOUT_HI_MULT = 1.4  # when payout mult >= 2.0
PAYOUT_LO_MULT = 0.7  # when payout mult < 1.7
BTC_LB = 30
BTC_THRESH = 0.0003
FLOOR = 0.05


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


def get_signal(r):
    """Returns (direction, tier, btc_agrees, btc_disagrees)."""
    # Tier 1: BNB acceleration
    for short, long in ACCEL_PAIRS:
        rs = r["bnb_rets"].get(short)
        rl = r["bnb_rets"].get(long)
        if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
            if max(abs(rs), abs(rl)) >= ACCEL_THRESH:
                d = "Bull" if rs > 0 else "Bear"
                btc_ag = btc_dis = False
                if r["has_btc"]:
                    btc_r = r["btc_rets"].get(BTC_LB)
                    if btc_r and abs(btc_r) >= BTC_THRESH:
                        btc_dir = "Bull" if btc_r > 0 else "Bear"
                        btc_ag = (btc_dir == d)
                        btc_dis = (btc_dir != d)
                return (d, "accel", btc_ag, btc_dis)

    # Tier 2: any BNB move + BTC confirmation
    if r["has_btc"]:
        bnb_r = r["bnb_rets"].get(7)
        if bnb_r is not None and bnb_r != 0:
            btc_r = r["btc_rets"].get(BTC_LB)
            if btc_r and abs(btc_r) >= BTC_THRESH:
                bnb_dir = "Bull" if bnb_r > 0 else "Bear"
                btc_dir = "Bull" if btc_r > 0 else "Bear"
                if bnb_dir == btc_dir:
                    return (bnb_dir, "any+btc", True, False)

    return (None, None, False, False)


def compute_bet_size(r, d, tier, btc_ag, btc_dis):
    """Compute bet size for a round."""
    base = T2_BASE if tier == "any+btc" else T1_BASE
    bs = max(FLOOR, r["pool_bnb"] * base)

    pm = r["pm_bull"] if d == "Bull" else r["pm_bear"]
    if pm >= 2.0:
        bs *= PAYOUT_HI_MULT
    elif pm < 1.7:
        bs *= PAYOUT_LO_MULT

    if btc_ag:
        bs *= BTC_AGREE_MULT
    elif btc_dis:
        bs *= BTC_DISAGREE_MULT

    return min(CAP, bs)


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
        for lb in [3, 5, 7, 10]:
            bnb_rets[lb] = get_return(bnb_rec["klines_1s"], cutoff_ms, lb)
        btc_rets = {}
        btc_rec = btc_by_epoch.get(epoch)
        if btc_rec:
            for lb in [30]:
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

    print(f"Total rounds: {len(rounds)}")
    print(f"With BTC data: {sum(1 for r in rounds if r['has_btc'])}")
    print()

    # =========================================================
    # FULL RUN
    # =========================================================
    print("=" * 85)
    print("STRATEGY: BNB Accel + BNB-Any+BTC Confirmation")
    print(f"  Accel pairs: {ACCEL_PAIRS} @ thresh {ACCEL_THRESH}")
    print(f"  T1 base: {T1_BASE:.0%} pool | T2 base: {T2_BASE:.0%} pool | Cap: {CAP}")
    print(f"  BTC agree: x{BTC_AGREE_MULT} | BTC disagree: x{BTC_DISAGREE_MULT}")
    print(f"  Payout high (>=2.0): x{PAYOUT_HI_MULT} | low (<1.7): x{PAYOUT_LO_MULT}")
    print("=" * 85)

    bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
    tier_stats = {}
    bet_log = []

    for r in rounds:
        d, tier, btc_ag, btc_dis = get_signal(r)
        if d is None:
            continue
        bs = compute_bet_size(r, d, tier, btc_ag, btc_dis)
        if bs < 0.03:
            continue

        won = d == r["outcome"]
        p = net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)

        bets += 1
        wagered += bs
        if won:
            wins += 1
        pnl += p

        # Track sub-tier
        subtier = tier
        if tier == "accel":
            if btc_ag:
                subtier = "accel+btc_agree"
            elif btc_dis:
                subtier = "accel+btc_disagree"
            else:
                subtier = "accel+no_btc"
        tier_stats.setdefault(subtier, {"n": 0, "w": 0, "pnl": 0.0, "wag": 0.0})
        tier_stats[subtier]["n"] += 1
        if won:
            tier_stats[subtier]["w"] += 1
        tier_stats[subtier]["pnl"] += p
        tier_stats[subtier]["wag"] += bs

        bet_log.append({"pnl": p, "won": won, "bet": bs, "tier": tier})

    print(f"\n  Total bets: {bets}")
    print(f"  Win rate: {wins/bets:.1%}")
    print(f"  Total wagered: {wagered:.1f} BNB")
    print(f"  Average bet: {wagered/bets:.3f} BNB")
    print(f"  Total PnL: {pnl:+.3f} BNB")
    print(f"  ROI: {pnl/wagered*100:+.1f}%")
    print(f"  PnL per bet: {pnl/bets:+.4f} BNB")
    print(f"  Signal rate: {bets/len(rounds)*100:.1f}% of rounds")

    # =========================================================
    # PER-TIER BREAKDOWN
    # =========================================================
    print("\n" + "-" * 85)
    print("Per-tier breakdown:")
    print(f"  {'Tier':>25} {'N':>5} {'WR':>6} {'PnL':>8} {'Avg Bet':>8} {'ROI':>6}")
    for t, s in sorted(tier_stats.items(), key=lambda x: -x[1]["n"]):
        wr = s["w"] / s["n"]
        avg = s["wag"] / s["n"]
        roi = s["pnl"] / s["wag"] * 100 if s["wag"] > 0 else 0
        print(f"  {t:>25} {s['n']:>5} {wr:>5.1%} {s['pnl']:>+8.3f} {avg:>8.3f} {roi:>+5.1f}%")

    # =========================================================
    # STABILITY: halves + quartiles
    # =========================================================
    print("\n" + "-" * 85)
    print("Stability (halves and quartiles):")

    half = len(rounds) // 2
    for label, subset in [("1st half", rounds[:half]), ("2nd half", rounds[half:])]:
        b, w, p, wg = 0, 0, 0.0, 0.0
        for r in subset:
            d, tier, btc_ag, btc_dis = get_signal(r)
            if d is None: continue
            bs = compute_bet_size(r, d, tier, btc_ag, btc_dis)
            if bs < 0.03: continue
            b += 1; wg += bs
            if d == r["outcome"]: w += 1
            p += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
        if b > 0:
            print(f"  {label:>10}: bets={b:>4} WR={w/b:.1%} PnL={p:+.3f} avg={wg/b:.3f}")

    q = len(rounds) // 4
    for i in range(4):
        subset = rounds[i*q:(i+1)*q]
        b, w, p, wg = 0, 0, 0.0, 0.0
        for r in subset:
            d, tier, btc_ag, btc_dis = get_signal(r)
            if d is None: continue
            bs = compute_bet_size(r, d, tier, btc_ag, btc_dis)
            if bs < 0.03: continue
            b += 1; wg += bs
            if d == r["outcome"]: w += 1
            p += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
        if b > 0:
            print(f"  Q{i+1}: bets={b:>4} WR={w/b:.1%} PnL={p:+.3f}")

    # =========================================================
    # BANKROLL SIMULATION
    # =========================================================
    print("\n" + "-" * 85)
    print("Bankroll simulation (50 BNB start):")

    bankroll = 50.0
    peak = 50.0
    max_dd = 0.0
    min_br = 50.0
    cum_pnl = []
    streak = {"w": 0, "l": 0, "max_w": 0, "max_l": 0}

    for entry in bet_log:
        bankroll += entry["pnl"]
        cum_pnl.append(bankroll)
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak
        if dd > max_dd:
            max_dd = dd
        if bankroll < min_br:
            min_br = bankroll
        if entry["won"]:
            streak["w"] += 1
            streak["l"] = 0
            if streak["w"] > streak["max_w"]:
                streak["max_w"] = streak["w"]
        else:
            streak["l"] += 1
            streak["w"] = 0
            if streak["l"] > streak["max_l"]:
                streak["max_l"] = streak["l"]

    print(f"  Starting: 50.00 BNB")
    print(f"  Final:    {bankroll:.2f} BNB")
    print(f"  Peak:     {peak:.2f} BNB")
    print(f"  Min:      {min_br:.2f} BNB")
    print(f"  Max DD:   {max_dd:.1%}")
    print(f"  Max win streak:  {streak['max_w']}")
    print(f"  Max lose streak: {streak['max_l']}")

    # PnL at milestones
    milestones = [100, 200, 300, 400, 500, 600, 700, 800]
    print(f"\n  PnL at milestones:")
    for m in milestones:
        if m <= len(cum_pnl):
            print(f"    After {m} bets: {cum_pnl[m-1]-50:.3f} BNB")

    # =========================================================
    # WIN RATE BY POOL SIZE
    # =========================================================
    print("\n" + "-" * 85)
    print("Win rate by pool size:")
    pool_bins = [(0, 1.5, "tiny <1.5"), (1.5, 3, "small 1.5-3"),
                  (3, 5, "med 3-5"), (5, 10, "large 5-10"),
                  (10, 999, "huge >10")]
    for lo, hi, label in pool_bins:
        n, w, p, wg = 0, 0, 0.0, 0.0
        for r in rounds:
            if not (lo <= r["pool_bnb"] < hi):
                continue
            d, tier, btc_ag, btc_dis = get_signal(r)
            if d is None: continue
            bs = compute_bet_size(r, d, tier, btc_ag, btc_dis)
            if bs < 0.03: continue
            n += 1; wg += bs
            if d == r["outcome"]: w += 1
            p += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], bs)
        if n > 0:
            print(f"  {label:>15}: n={n:>4} WR={w/n:.1%} avg_bet={wg/n:.3f} PnL={p:+.3f}")

    # =========================================================
    # COMPARISON: flat bet alternatives
    # =========================================================
    print("\n" + "-" * 85)
    print("Comparison with flat bet sizing:")
    for flat in [0.05, 0.10, 0.15, 0.20]:
        b, w, p = 0, 0, 0.0
        for r in rounds:
            d, tier, btc_ag, btc_dis = get_signal(r)
            if d is None: continue
            b += 1
            if d == r["outcome"]: w += 1
            p += net_profit(r["bull_wei"], r["bear_wei"], d, r["outcome"], flat)
        if b > 0:
            print(f"  flat {flat:.2f}: bets={b} WR={w/b:.1%} PnL={p:+.3f}")

    print(f"\n  Smart sizing advantage over flat 0.10: "
          f"{pnl - sum(net_profit(r['bull_wei'], r['bear_wei'], d, r['outcome'], 0.10) for r in rounds if (d := get_signal(r)[0]) is not None):+.3f} BNB")


if __name__ == "__main__":
    main()
