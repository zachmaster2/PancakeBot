"""Step 20: Novel BTC features using existing kline data.

1. BTC acceleration (2nd derivative) — is momentum building?
2. Last-candle volume spike + MTF confirmation
3. Max return in window (peak move, may have retraced)
4. Inter-round BTC momentum (full 30s window + short MTF)
5. BTC candle body/wick patterns on most recent candles
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
CANDLE_COUNT = 31
TREASURY_FEE = 0.03


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
    return t[-CANDLE_COUNT:] if len(t) >= CANDLE_COUNT else None


def stl(rnd, bet, side):
    o = settle_bet_against_closed_round(
        bet_bnb=bet, bet_side=side, round_closed=rnd,
        treasury_fee_fraction=TREASURY_FEE)
    return o.credit_bnb - bet - GAS_COST_BET_BNB


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


def run_with_sizing(d, sig, m):
    """Standard adaptive sizing + payout boost."""
    our = d["bp"] if sig == "Bull" else d["sp"]
    if our == 0:
        return None
    pm = d["pool"] * 0.97 / our
    if pm < 1.5:
        return None
    frac = min(0.03 + 100 * m, 0.30)
    frac = min(frac * max(0.5, 1.0 + 1.0 * (pm - 2.0)), 0.30)
    bet = max(0.01, min(2.0, d["pool"] * frac))
    return stl(d["rnd"], bet, sig)


def pr(label, results, total):
    n = len(results)
    if n < 20:
        print(f"  {label}: N={n} (too few)")
        return
    wr = sum(1 for p in results if p > 0) / n * 100
    pnl = sum(results)
    p2k = pnl / total * 2000
    flag = " ***" if pnl > 0 else ""
    print(f"  {label}: N={n} WR={wr:.1f}% /2k={p2k:+.3f}{flag}")


def main():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    total = len(rounds)
    btc_kl = load_kl("var/btc_spot_prices.jsonl")

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
        closes = [k[4] for k in bc]
        vols = [k[5] for k in bc]
        bp, sp = gp(rnd, la)
        pool = bp + sp
        data.append({
            "rnd": rnd, "c": closes, "v": vols,
            "bp": bp, "sp": sp, "pool": pool,
        })

    print(f"Data: {len(data)}")

    # =================================================================
    print("\n" + "=" * 90)
    print("PART 1: BTC ACCELERATION (2nd derivative)")
    print("=" * 90)

    for short_lb, long_lb in [(2, 5), (3, 7), (3, 10), (5, 15)]:
        for accel_thresh in [0.0001, 0.0002, 0.0003]:
            results = []
            for d in data:
                if d["pool"] < 2.0:
                    continue
                rs = ret(d["c"], short_lb)
                rl = ret(d["c"], long_lb)
                if rs is None or rl is None:
                    continue
                if not ((rs > 0 and rl > 0) or (rs < 0 and rl < 0)):
                    continue
                accel = abs(rs) - abs(rl) * (short_lb / long_lb)
                if accel < accel_thresh:
                    continue
                m = min(abs(rs), abs(rl))
                sig = "Bull" if rs > 0 else "Bear"
                p = run_with_sizing(d, sig, m)
                if p is not None:
                    results.append(p)
            pr(f"accel({short_lb},{long_lb},t={accel_thresh})", results, total)

    # =================================================================
    print("\n" + "=" * 90)
    print("PART 2: LAST-CANDLE VOLUME SPIKE + MTF")
    print("=" * 90)

    for vol_mult in [2.0, 3.0, 5.0]:
        results_with = []
        results_without = []
        for d in data:
            if d["pool"] < 2.0:
                continue
            rets = [ret(d["c"], lb) for lb in [3, 7, 15]]
            if any(r is None for r in rets):
                continue
            if not (all(r > 0 for r in rets) or all(r < 0 for r in rets)):
                continue
            m = min(abs(r) for r in rets)
            if m < 0.0001:
                continue
            sig = "Bull" if rets[0] > 0 else "Bear"
            p = run_with_sizing(d, sig, m)
            if p is None:
                continue

            last_vol = d["v"][-1]
            avg_vol = sum(d["v"][-6:-1]) / 5 if len(d["v"]) >= 6 else 0
            has_spike = avg_vol > 0 and last_vol >= avg_vol * vol_mult

            if has_spike:
                results_with.append(p)
            else:
                results_without.append(p)

        pr(f"MTF + vol_spike>={vol_mult}x", results_with, total)
        pr(f"MTF + vol_spike<{vol_mult}x", results_without, total)

    # =================================================================
    print("\n" + "=" * 90)
    print("PART 3: MAX RETURN IN WINDOW")
    print("=" * 90)

    for window in [5, 7, 10]:
        for thresh in [0.0005, 0.001, 0.0015]:
            follow = []
            revert = []
            for d in data:
                if d["pool"] < 2.0:
                    continue
                max_r = 0
                for i in range(1, min(window + 1, len(d["c"]))):
                    r = ret(d["c"], i)
                    if r is not None and abs(r) > abs(max_r):
                        max_r = r
                if abs(max_r) < thresh:
                    continue
                for sig, rl in [
                    ("Bull" if max_r > 0 else "Bear", follow),
                    ("Bear" if max_r > 0 else "Bull", revert),
                ]:
                    our = d["bp"] if sig == "Bull" else d["sp"]
                    if our == 0:
                        continue
                    pm = d["pool"] * 0.97 / our
                    if pm < 1.5:
                        continue
                    bet = max(0.01, min(2.0, d["pool"] * 0.05))
                    rl.append(stl(d["rnd"], bet, sig))

            pr(f"max_ret(w={window},t={thresh}) follow", follow, total)
            pr(f"max_ret(w={window},t={thresh}) revert", revert, total)

    # =================================================================
    print("\n" + "=" * 90)
    print("PART 4: FULL-WINDOW BTC MOMENTUM + short MTF")
    print("=" * 90)

    for full_thresh in [0.0003, 0.0005, 0.001, 0.002]:
        results = []
        for d in data:
            if d["pool"] < 2.0:
                continue
            r_full = ret(d["c"], 30)
            if r_full is None or abs(r_full) < full_thresh:
                continue
            rets = [ret(d["c"], lb) for lb in [3, 7, 15]]
            if any(r is None for r in rets):
                continue
            if not (all(r > 0 for r in rets) or all(r < 0 for r in rets)):
                continue
            if (r_full > 0) != (rets[0] > 0):
                continue
            m = min(abs(r) for r in rets)
            if m < 0.0001:
                continue
            sig = "Bull" if rets[0] > 0 else "Bear"
            p = run_with_sizing(d, sig, m)
            if p is not None:
                results.append(p)
        pr(f"full_30s>={full_thresh} + MTF", results, total)

    # =================================================================
    print("\n" + "=" * 90)
    print("PART 5: MONOTONIC CANDLES — all last N candles same direction")
    print("=" * 90)

    for n_candles in [3, 4, 5, 6, 7]:
        results = []
        for d in data:
            if d["pool"] < 2.0:
                continue
            if len(d["c"]) < n_candles + 1:
                continue
            # Check last N candles all same direction
            dirs = []
            for i in range(n_candles):
                idx = -(n_candles - i)
                diff = d["c"][idx] - d["c"][idx - 1]
                if diff > 0:
                    dirs.append(1)
                elif diff < 0:
                    dirs.append(-1)
                else:
                    dirs.append(0)
            if 0 in dirs or len(set(dirs)) > 1:
                continue
            m = abs(ret(d["c"], n_candles) or 0)
            if m < 0.0001:
                continue
            sig = "Bull" if dirs[0] > 0 else "Bear"
            p = run_with_sizing(d, sig, m)
            if p is not None:
                results.append(p)
        pr(f"monotonic_{n_candles}_candles", results, total)

    # Also: monotonic as STANDALONE (no MTF requirement)
    print("\n  --- Monotonic standalone (no MTF) ---")
    for n_candles in [5, 6, 7, 8]:
        results = []
        for d in data:
            if d["pool"] < 2.0:
                continue
            if len(d["c"]) < n_candles + 1:
                continue
            dirs = []
            for i in range(n_candles):
                idx = -(n_candles - i)
                diff = d["c"][idx] - d["c"][idx - 1]
                if diff > 0:
                    dirs.append(1)
                elif diff < 0:
                    dirs.append(-1)
                else:
                    dirs.append(0)
            if 0 in dirs or len(set(dirs)) > 1:
                continue
            m = abs(ret(d["c"], n_candles) or 0)
            if m < 0.0001:
                continue
            sig = "Bull" if dirs[0] > 0 else "Bear"
            our = d["bp"] if sig == "Bull" else d["sp"]
            if our == 0:
                continue
            pm = d["pool"] * 0.97 / our
            if pm < 1.5:
                continue
            bet = max(0.01, min(2.0, d["pool"] * 0.05))
            results.append(stl(d["rnd"], bet, sig))
        pr(f"monotonic_{n_candles}_standalone", results, total)

    print("\nDone.")


if __name__ == "__main__":
    main()
