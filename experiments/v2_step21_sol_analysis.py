"""Step 21: SOL cross-pair analysis + combined BTC+ETH+SOL.

Tests:
1. SOL multi-TF standalone
2. Overlap with BTC signal
3. SOL as sizing boost (like ETH)
4. Combined BTC + ETH + SOL sizing boost
5. 5-fold validation of best combo
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB, POOL_CUTOFF_SECONDS
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import _trim_to_window
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2
POOL_CUTOFF_S = POOL_CUTOFF_SECONDS
CC = 31
TF = 0.03


def load_kl(p):
    out = {}
    for line in Path(p).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("klines_1s") is not None:
            out[int(rec["epoch"])] = rec["klines_1s"]
    return out


def gc(raw, cms):
    t = _trim_to_window(raw, cms)
    return t[-CC:] if len(t) >= CC else None


def stl(rnd, bet, side):
    return settle_bet_against_closed_round(
        bet_bnb=bet, bet_side=side, round_closed=rnd,
        treasury_fee_fraction=TF,
    ).credit_bnb - bet - GAS_COST_BET_BNB


def ret(c, lb):
    if len(c) < lb + 1 or c[-(lb + 1)] == 0:
        return None
    return (c[-1] - c[-(lb + 1)]) / c[-(lb + 1)]


def gp(rnd, la):
    cu = la - POOL_CUTOFF_S
    b = s = 0
    for bet in rnd.bets:
        if int(bet.created_at) > cu:
            continue
        if bet.position == "Bull":
            b += int(bet.amount_wei)
        else:
            s += int(bet.amount_wei)
    return b / 1e18, s / 1e18


def mtf(closes, thresh=0.0001):
    rets = [ret(closes, lb) for lb in [3, 7, 15]]
    if any(r is None for r in rets):
        return None, 0
    if not (all(r > 0 for r in rets) or all(r < 0 for r in rets)):
        return None, 0
    m = min(abs(r) for r in rets)
    if m < thresh:
        return None, 0
    return ("Bull" if rets[0] > 0 else "Bear"), m


def confirms(closes, direction):
    """Check if closes multi-TF confirms the given direction."""
    rets = [ret(closes, lb) for lb in [3, 7, 15]]
    if any(r is None for r in rets):
        return 0.0
    is_bull = direction == "Bull"
    if all((r > 0) == is_bull for r in rets):
        return min(abs(r) for r in rets)
    return 0.0


def main():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    total = len(rounds)

    btc_kl = load_kl("var/btc_spot_prices.jsonl")
    eth_kl = load_kl("var/eth_spot_prices.jsonl")
    sol_kl = load_kl("var/sol_spot_prices.jsonl")
    print(f"BTC klines: {len(btc_kl)}")
    print(f"ETH klines: {len(eth_kl)}")
    print(f"SOL klines: {len(sol_kl)}")

    data = []
    for rnd in rounds:
        la = int(rnd.lock_at)
        ep = int(rnd.epoch)
        cms = (la - CUTOFF_S) * 1000
        br = btc_kl.get(ep)
        if not br:
            continue
        bc = gc(br, cms)
        if bc is None:
            continue
        btc_c = [k[4] for k in bc]

        eth_c = None
        er = eth_kl.get(ep)
        if er:
            ec = gc(er, cms)
            if ec:
                eth_c = [k[4] for k in ec]

        sol_c = None
        sr = sol_kl.get(ep)
        if sr:
            sc = gc(sr, cms)
            if sc:
                sol_c = [k[4] for k in sc]

        bp, sp = gp(rnd, la)
        pool = bp + sp
        data.append({
            "rnd": rnd, "btc_c": btc_c, "eth_c": eth_c, "sol_c": sol_c,
            "bp": bp, "sp": sp, "pool": pool,
        })

    print(f"Data: {len(data)}, with SOL: {sum(1 for d in data if d['sol_c'])}")
    fold_size = len(data) // 5

    # ================================================================
    print("\n" + "=" * 90)
    print("PART 1: SOL multi-TF standalone")
    print("=" * 90)
    for thresh in [0.0002, 0.0003, 0.0005]:
        results = []
        for d in data:
            if d["sol_c"] is None or d["pool"] < 2.0:
                continue
            sig, m = mtf(d["sol_c"], thresh)
            if sig is None:
                continue
            our = d["bp"] if sig == "Bull" else d["sp"]
            if our == 0:
                continue
            pm = d["pool"] * 0.97 / our
            if pm < 1.5:
                continue
            frac = min(0.03 + 100 * m, 0.30)
            frac = min(frac * max(0.5, 1.0 + 1.0 * (pm - 2.0)), 0.30)
            bet = max(0.01, min(2.0, d["pool"] * frac))
            results.append(stl(d["rnd"], bet, sig))
        n = len(results)
        if n < 20:
            print(f"  sol_mtf(t={thresh}): N={n} (too few)")
            continue
        wr = sum(1 for p in results if p > 0) / n * 100
        pnl = sum(results) / total * 2000
        print(f"  sol_mtf(t={thresh}): N={n} WR={wr:.1f}% /2k={pnl:+.3f}")

    # ================================================================
    print("\n" + "=" * 90)
    print("PART 2: Overlap BTC vs SOL vs ETH")
    print("=" * 90)
    btc_set = set()
    eth_set = set()
    sol_set = set()
    for d in data:
        if d["pool"] < 2.0:
            continue
        sig, _ = mtf(d["btc_c"])
        if sig:
            btc_set.add(d["rnd"].epoch)
        if d["eth_c"]:
            sig, _ = mtf(d["eth_c"])
            if sig:
                eth_set.add(d["rnd"].epoch)
        if d["sol_c"]:
            sig, _ = mtf(d["sol_c"])
            if sig:
                sol_set.add(d["rnd"].epoch)

    print(f"  BTC fires: {len(btc_set)}")
    print(f"  ETH fires: {len(eth_set)}")
    print(f"  SOL fires: {len(sol_set)}")
    print(f"  BTC & ETH: {len(btc_set & eth_set)}")
    print(f"  BTC & SOL: {len(btc_set & sol_set)}")
    print(f"  ETH & SOL: {len(eth_set & sol_set)}")
    print(f"  All three: {len(btc_set & eth_set & sol_set)}")
    print(f"  Union all: {len(btc_set | eth_set | sol_set)}")
    print(f"  SOL unique: {len(sol_set - btc_set - eth_set)}")

    # ================================================================
    print("\n" + "=" * 90)
    print("PART 3: SOL as sizing boost (like ETH)")
    print("=" * 90)
    for sol_w in [0.2, 0.3, 0.5]:
        fold_pnls = []
        for fold in range(5):
            s = fold * fold_size
            e = s + fold_size if fold < 4 else len(data)
            fd = data[s:e]
            pnl = 0
            for d in fd:
                if d["pool"] < 2.0:
                    continue
                sig, m = mtf(d["btc_c"])
                if sig is None:
                    continue
                thresh = 0.0001 if d["pool"] >= 3.0 else 0.0002
                if m < thresh:
                    continue
                our = d["bp"] if sig == "Bull" else d["sp"]
                if our == 0:
                    continue
                pm = d["pool"] * 0.97 / our
                if pm < 1.5:
                    continue

                # SOL confirmation boost
                sol_bonus = 0
                if d["sol_c"] is not None:
                    sol_bonus = confirms(d["sol_c"], sig) * sol_w

                strength = m + sol_bonus
                frac = min(0.03 + 100 * strength, 0.30)
                frac = min(frac * max(0.5, 1.0 + 1.0 * (pm - 2.0)), 0.30)
                bet = max(0.01, min(2.0, d["pool"] * frac))
                pnl += stl(d["rnd"], bet, sig)
            fold_pnls.append(pnl / len(fd) * 2000)
        avg = sum(fold_pnls) / 5
        pos = sum(1 for p in fold_pnls if p > 0)
        folds = "|".join(f"{p:+.2f}" for p in fold_pnls)
        print(f"  sol_w={sol_w}: avg={avg:+.3f} ({pos}/5) [{folds}]")

    # ================================================================
    print("\n" + "=" * 90)
    print("PART 4: Combined BTC + ETH + SOL sizing boost")
    print("=" * 90)
    for eth_w, sol_w in [(0.3, 0.0), (0.3, 0.2), (0.3, 0.3),
                          (0.2, 0.2), (0.5, 0.3)]:
        fold_pnls = []
        for fold in range(5):
            s = fold * fold_size
            e = s + fold_size if fold < 4 else len(data)
            fd = data[s:e]
            pnl = 0
            for d in fd:
                if d["pool"] < 2.0:
                    continue
                sig, m = mtf(d["btc_c"])
                if sig is None:
                    continue
                thresh = 0.0001 if d["pool"] >= 3.0 else 0.0002
                if m < thresh:
                    continue
                our = d["bp"] if sig == "Bull" else d["sp"]
                if our == 0:
                    continue
                pm = d["pool"] * 0.97 / our
                if pm < 1.5:
                    continue

                eth_bonus = 0
                if eth_w > 0 and d["eth_c"] is not None:
                    eth_bonus = confirms(d["eth_c"], sig) * eth_w
                sol_bonus = 0
                if sol_w > 0 and d["sol_c"] is not None:
                    sol_bonus = confirms(d["sol_c"], sig) * sol_w

                strength = m + eth_bonus + sol_bonus
                frac = min(0.03 + 100 * strength, 0.30)
                frac = min(frac * max(0.5, 1.0 + 1.0 * (pm - 2.0)), 0.30)
                bet = max(0.01, min(2.0, d["pool"] * frac))
                pnl += stl(d["rnd"], bet, sig)
            fold_pnls.append(pnl / len(fd) * 2000)
        avg = sum(fold_pnls) / 5
        pos = sum(1 for p in fold_pnls if p > 0)
        folds = "|".join(f"{p:+.2f}" for p in fold_pnls)
        print(f"  eth_w={eth_w} sol_w={sol_w}: avg={avg:+.3f} ({pos}/5) [{folds}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
