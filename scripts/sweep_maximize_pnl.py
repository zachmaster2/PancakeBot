"""Maximize PnL on signal-firing rounds.

Since we can't increase bet count much (600 rounds fire, 88% are noise),
we must maximize profit per bet. Combines:
1. Pool-size-aware sizing (larger bets on bigger pools = less impact)
2. Payout-asymmetry sizing (larger bets when our side is the minority)
3. Combined: pool + payout jointly determine bet size
4. Signal confidence tiers (BNB+BTC agreement = bigger bet)
5. Optimal per-round bet size (brute force search)
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


def expected_profit(bw, ew, side, bet_bnb, wr):
    m = payout_multiple(bw, ew, side, bet_bnb)
    win_pnl = bet_bnb * m - GAS_CLAIM - bet_bnb - GAS_BET
    lose_pnl = -bet_bnb - GAS_BET
    return wr * win_pnl + (1 - wr) * lose_pnl


def get_return(klines, cutoff_ms, lookback_s):
    kn = find_closest(klines, cutoff_ms)
    ka = find_closest(klines, cutoff_ms - lookback_s * 1000)
    if not kn or not ka or ka[4] <= 0:
        return None
    return (kn[4] / ka[4]) - 1


def bnb_accel(bnb_rets):
    for short, long, thresh in [(7, 10, 0.0002), (5, 10, 0.0002)]:
        rs = bnb_rets.get(short)
        rl = bnb_rets.get(long)
        if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
            if max(abs(rs), abs(rl)) >= thresh:
                return "Bull" if rs > 0 else "Bear"
    return None


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

        pool_bnb = (bull_wei + bear_wei) / BNB_WEI
        bnb_rets = {}
        for lb in [3, 5, 7, 10, 15, 20]:
            bnb_rets[lb] = get_return(bnb_rec["klines_1s"], cutoff_ms, lb)

        btc_rets = {}
        btc_rec = btc_by_epoch.get(epoch)
        if btc_rec:
            for lb in [5, 7, 10, 15, 20, 30]:
                btc_rets[lb] = get_return(btc_rec["klines_1s"], cutoff_ms, lb)

        d = bnb_accel(bnb_rets)
        if d is None:
            continue  # Only signal-firing rounds

        # Pre-bet payout mult
        pre_mult = payout_multiple(bull_wei, bear_wei, d, 0.001)
        bull_frac = bull_wei / (bull_wei + bear_wei)

        # Check BTC agreement
        btc_agrees = False
        btc_signal = None
        if btc_rets:
            for short, long, thresh in [(5, 30, 0.0003), (7, 30, 0.0003)]:
                rs = btc_rets.get(short)
                rl = btc_rets.get(long)
                if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                    if max(abs(rs), abs(rl)) >= thresh:
                        btc_signal = "Bull" if rs > 0 else "Bear"
                        btc_agrees = (btc_signal == d)
                        break
            if btc_signal is None:
                # Simple BTC signal
                btc_ret = btc_rets.get(7)
                if btc_ret and btc_ret != 0:
                    btc_signal = "Bull" if btc_ret > 0 else "Bear"
                    btc_agrees = (btc_signal == d)

        # BNB signal strength
        bnb_strength = max(
            abs(bnb_rets.get(5, 0) or 0),
            abs(bnb_rets.get(7, 0) or 0),
            abs(bnb_rets.get(10, 0) or 0),
        )

        features.append({
            "epoch": epoch, "outcome": rnd["position"],
            "direction": d,
            "bull_wei": bull_wei, "bear_wei": bear_wei,
            "pool_bnb": pool_bnb,
            "pre_mult": pre_mult,
            "bull_frac": bull_frac,
            "btc_agrees": btc_agrees,
            "btc_signal": btc_signal,
            "has_btc": bool(btc_rets),
            "bnb_strength": bnb_strength,
        })

    print(f"Signal-firing rounds: {len(features)}")
    with_btc = sum(1 for f in features if f["has_btc"])
    print(f"  With BTC data: {with_btc}")
    agrees = sum(1 for f in features if f["btc_agrees"])
    print(f"  BTC agrees: {agrees}")
    print()

    # =========================================================
    # 1. BASELINES
    # =========================================================
    print("=" * 85)
    print("BASELINES")
    print("=" * 85)

    for label, bet_fn in [
        ("flat 0.05", lambda f: 0.05),
        ("flat 0.10", lambda f: 0.10),
        ("flat 0.15", lambda f: 0.15),
        ("flat 0.20", lambda f: 0.20),
    ]:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for f in features:
            bs = bet_fn(f)
            bets += 1; wagered += bs
            if f["direction"] == f["outcome"]: wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"], f["direction"], f["outcome"], bs)
        print(f"  {label:>30}: bets={bets} WR={wins/bets:.1%} wager={wagered:.0f} PnL={pnl:+.2f}")

    # =========================================================
    # 2. POOL-AWARE SIZING
    # =========================================================
    print("\n" + "=" * 85)
    print("POOL-AWARE SIZING")
    print("=" * 85)

    for label, bet_fn in [
        ("3% pool, floor 0.05, cap 0.25", lambda f: min(0.25, max(0.05, f["pool_bnb"] * 0.03))),
        ("4% pool, floor 0.05, cap 0.30", lambda f: min(0.30, max(0.05, f["pool_bnb"] * 0.04))),
        ("5% pool, floor 0.05, cap 0.35", lambda f: min(0.35, max(0.05, f["pool_bnb"] * 0.05))),
        ("5% pool, floor 0.05, cap 0.50", lambda f: min(0.50, max(0.05, f["pool_bnb"] * 0.05))),
        ("7% pool, floor 0.05, cap 0.50", lambda f: min(0.50, max(0.05, f["pool_bnb"] * 0.07))),
        ("10% pool, floor 0.05, cap 0.50", lambda f: min(0.50, max(0.05, f["pool_bnb"] * 0.10))),
    ]:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for f in features:
            bs = bet_fn(f)
            bets += 1; wagered += bs
            if f["direction"] == f["outcome"]: wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"], f["direction"], f["outcome"], bs)
        roi = pnl / wagered * 100
        avg_bet = wagered / bets
        print(f"  {label:>38}: avg={avg_bet:.3f} wager={wagered:.0f} PnL={pnl:+.2f} ROI={roi:+.1f}%")

    # =========================================================
    # 3. PAYOUT-AWARE SIZING
    # =========================================================
    print("\n" + "=" * 85)
    print("PAYOUT-AWARE SIZING")
    print("=" * 85)

    for label, bet_fn in [
        ("0.10 when mult>=1.8, else 0.05",
         lambda f: 0.10 if f["pre_mult"] >= 1.8 else 0.05),
        ("0.15 when mult>=2.0, 0.10 when >=1.8, else 0.05",
         lambda f: 0.15 if f["pre_mult"] >= 2.0 else (0.10 if f["pre_mult"] >= 1.8 else 0.05)),
        ("0.20 when mult>=2.0, 0.10 when >=1.8, else 0.05",
         lambda f: 0.20 if f["pre_mult"] >= 2.0 else (0.10 if f["pre_mult"] >= 1.8 else 0.05)),
        ("linear: 0.05 + 0.10 * (mult-1.5)",
         lambda f: min(0.30, max(0.05, 0.05 + 0.10 * (f["pre_mult"] - 1.5)))),
        ("skip mult<1.7, else 0.15",
         lambda f: 0.15 if f["pre_mult"] >= 1.7 else 0.0),
    ]:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for f in features:
            bs = bet_fn(f)
            if bs < 0.001: continue
            bets += 1; wagered += bs
            if f["direction"] == f["outcome"]: wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"], f["direction"], f["outcome"], bs)
        if bets > 0:
            roi = pnl / wagered * 100
            print(f"  {label:>50}: bets={bets} PnL={pnl:+.2f} ROI={roi:+.1f}%")

    # =========================================================
    # 4. COMBINED: pool + payout + BTC agreement
    # =========================================================
    print("\n" + "=" * 85)
    print("COMBINED: pool + payout + BTC confidence")
    print("=" * 85)

    def combined_sizing(f, base_pct=0.04, cap=0.40, btc_boost=1.5, payout_boost=True):
        """Combine pool-proportional + payout + BTC agreement."""
        pool = f["pool_bnb"]
        bet = max(0.05, pool * base_pct)

        # Payout boost
        if payout_boost and f["pre_mult"] >= 2.0:
            bet *= 1.3
        elif payout_boost and f["pre_mult"] < 1.7:
            bet *= 0.6

        # BTC agreement boost
        if f["btc_agrees"]:
            bet *= btc_boost
        elif f["btc_signal"] and not f["btc_agrees"]:
            bet *= 0.5  # BTC disagrees = reduce

        return min(cap, bet)

    configs = [
        ("base4% cap0.30 btcx1.5 +pay",
         lambda f: combined_sizing(f, 0.04, 0.30, 1.5, True)),
        ("base4% cap0.40 btcx1.5 +pay",
         lambda f: combined_sizing(f, 0.04, 0.40, 1.5, True)),
        ("base5% cap0.40 btcx1.5 +pay",
         lambda f: combined_sizing(f, 0.05, 0.40, 1.5, True)),
        ("base5% cap0.50 btcx2.0 +pay",
         lambda f: combined_sizing(f, 0.05, 0.50, 2.0, True)),
        ("base7% cap0.50 btcx1.5 +pay",
         lambda f: combined_sizing(f, 0.07, 0.50, 1.5, True)),
        ("base7% cap0.50 btcx2.0 +pay",
         lambda f: combined_sizing(f, 0.07, 0.50, 2.0, True)),
        ("base10% cap0.50 btcx1.5 +pay",
         lambda f: combined_sizing(f, 0.10, 0.50, 1.5, True)),
        ("base4% cap0.40 btcx1.5 nopay",
         lambda f: combined_sizing(f, 0.04, 0.40, 1.5, False)),
        ("base5% cap0.50 btcx1.0 +pay",
         lambda f: combined_sizing(f, 0.05, 0.50, 1.0, True)),
    ]

    for label, bet_fn in configs:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for f in features:
            bs = bet_fn(f)
            if bs < 0.001: continue
            bets += 1; wagered += bs
            if f["direction"] == f["outcome"]: wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"], f["direction"], f["outcome"], bs)
        if bets > 0:
            roi = pnl / wagered * 100
            avg_bet = wagered / bets
            print(f"  {label:>38}: bets={bets} avg={avg_bet:.3f} wager={wagered:.0f} "
                  f"PnL={pnl:+.2f} ROI={roi:+.1f}%")

    # =========================================================
    # 5. OPTIMAL PER-ROUND BET SIZE (brute force)
    # =========================================================
    print("\n" + "=" * 85)
    print("OPTIMAL PER-ROUND SIZING (wr=0.605, search 0.01-1.00)")
    print("=" * 85)

    for assumed_wr in [0.59, 0.60, 0.605, 0.61, 0.62]:
        bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
        for f in features:
            best_size = 0.01
            best_ep = expected_profit(f["bull_wei"], f["bear_wei"],
                                      f["direction"], 0.01, assumed_wr)
            for s100 in range(2, 101):
                size = s100 * 0.01
                ep = expected_profit(f["bull_wei"], f["bear_wei"],
                                     f["direction"], size, assumed_wr)
                if ep > best_ep:
                    best_ep = ep
                    best_size = size
            if best_ep <= 0:
                continue
            bets += 1
            wagered += best_size
            if f["direction"] == f["outcome"]:
                wins += 1
            pnl += net_profit(f["bull_wei"], f["bear_wei"],
                              f["direction"], f["outcome"], best_size)
        if bets > 0:
            avg = wagered / bets
            print(f"  wr={assumed_wr:.3f}: bets={bets} avg_bet={avg:.3f} "
                  f"wager={wagered:.0f} PnL={pnl:+.2f} actual_WR={wins/bets:.1%}")

    # =========================================================
    # 6. BANKROLL SIMULATION
    # =========================================================
    print("\n" + "=" * 85)
    print("BANKROLL SIMULATION (50 BNB start)")
    print("=" * 85)

    for label, bet_fn_factory in [
        ("flat 0.10", lambda br: lambda f: 0.10),
        ("1% bankroll cap 0.20",
         lambda br: lambda f: min(0.20, br[0] * 0.01)),
        ("2% bankroll cap 0.30",
         lambda br: lambda f: min(0.30, br[0] * 0.02)),
        ("4% pool cap 0.30",
         lambda br: lambda f: min(0.30, max(0.05, f["pool_bnb"] * 0.04))),
        ("5% pool cap 0.40",
         lambda br: lambda f: min(0.40, max(0.05, f["pool_bnb"] * 0.05))),
        ("5% pool cap 0.40 + 2% bank cap",
         lambda br: lambda f: min(0.40, max(0.05, f["pool_bnb"] * 0.05), br[0] * 0.02)),
        ("combined_best",
         lambda br: lambda f: combined_sizing(f, 0.05, 0.50, 1.5, True)),
    ]:
        bankroll = [50.0]  # mutable ref
        peak = 50.0
        bets, wins, pnl = 0, 0, 0.0
        max_dd = 0.0
        bet_fn = bet_fn_factory(bankroll)
        for f in features:
            bs = bet_fn(f)
            if bs < 0.001 or bs > bankroll[0] * 0.5:
                bs = min(bs, bankroll[0] * 0.5)
            if bs < 0.001:
                continue
            bets += 1
            won = f["direction"] == f["outcome"]
            if won:
                wins += 1
            p = net_profit(f["bull_wei"], f["bear_wei"],
                           f["direction"], f["outcome"], bs)
            pnl += p
            bankroll[0] += p
            if bankroll[0] > peak:
                peak = bankroll[0]
            dd = (peak - bankroll[0]) / peak
            if dd > max_dd:
                max_dd = dd
            if bankroll[0] <= 0:
                print(f"  {label:>35}: BANKRUPT after {bets} bets")
                break
        else:
            print(f"  {label:>35}: bets={bets} WR={wins/bets:.1%} "
                  f"PnL={pnl:+.2f} final={bankroll[0]:.2f} maxDD={max_dd:.1%}")


if __name__ == "__main__":
    main()
