"""Step 19: More unexplored angles using existing data.

1. Late bets — do bets placed after our cutoff predict outcomes?
2. Bet count ratio — number of bets (not amount) as signal
3. Pool growth velocity — fast-growing pools as filter
4. Wider BTC lookbacks (20s, 25s, 30s) in multi-TF
5. BTC-BNB price ratio momentum
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import (
    GAS_COST_BET_BNB, INTERVAL_SECONDS, POOL_CUTOFF_SECONDS,
)
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


def get_candles(raw, cutoff_ms):
    trimmed = _trim_to_window(raw, cutoff_ms)
    return trimmed[-CANDLE_COUNT:] if len(trimmed) >= CANDLE_COUNT else None


def settle(rnd, bet, side):
    out = settle_bet_against_closed_round(
        bet_bnb=bet, bet_side=side,
        round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
    )
    return out.credit_bnb - bet - GAS_COST_BET_BNB


def ret(closes, lb):
    if len(closes) < lb + 1 or closes[-(lb + 1)] == 0:
        return None
    return (closes[-1] - closes[-(lb + 1)]) / closes[-(lb + 1)]


def get_pools(rnd, lock_at):
    cutoff = lock_at - POOL_CUTOFF_S
    b = s = 0
    for bet in rnd.bets:
        if int(bet.created_at) > cutoff:
            continue
        if bet.position == "Bull":
            b += int(bet.amount_wei)
        else:
            s += int(bet.amount_wei)
    return b / 1e18, s / 1e18


def main():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    total = len(rounds)
    btc_kl = load_kl("var/btc_spot_prices.jsonl")
    bnb_kl = load_kl("var/cutoff_spot_prices.jsonl")
    print(f"Total rounds: {total}")

    # ==================================================================
    print("\n" + "=" * 100)
    print("PART 1: LATE BETS -- bets placed after our cutoff (lock-6s to lock)")
    print("=" * 100)

    for early_cut, late_cut in [(6, 3), (6, 0), (4, 0), (3, 0)]:
        follow_results = []
        fade_results = []
        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            bull_p, bear_p = get_pools(rnd, lock_at)
            pool = bull_p + bear_p
            if pool < 2.0:
                continue

            late_bull = late_bear = 0
            for bet in rnd.bets:
                ts = int(bet.created_at)
                if ts <= lock_at - early_cut or ts > lock_at - late_cut:
                    continue
                if bet.position == "Bull":
                    late_bull += int(bet.amount_wei)
                else:
                    late_bear += int(bet.amount_wei)

            late_total = late_bull + late_bear
            if late_total == 0:
                continue

            late_bull_frac = late_bull / late_total
            if late_bull_frac > 0.6:
                follow_results.append(settle(rnd, 0.10, "Bull"))
                fade_results.append(settle(rnd, 0.10, "Bear"))
            elif late_bull_frac < 0.4:
                follow_results.append(settle(rnd, 0.10, "Bear"))
                fade_results.append(settle(rnd, 0.10, "Bull"))

        n = len(follow_results)
        if n < 20:
            continue
        fw = sum(1 for p in follow_results if p > 0) / n * 100
        fa = sum(1 for p in fade_results if p > 0) / n * 100
        fp = sum(follow_results) / total * 2000
        fap = sum(fade_results) / total * 2000
        print(f"  lock-{early_cut}s..lock-{late_cut}s: N={n} "
              f"follow={fw:.1f}%({fp:+.3f}/2k) fade={fa:.1f}%({fap:+.3f}/2k)")

    # ==================================================================
    print("\n" + "=" * 100)
    print("PART 2: BET COUNT RATIO -- number of bets (not amount)")
    print("=" * 100)

    for min_bets in [5, 10, 15]:
        for count_ratio in [0.6, 0.7, 0.8]:
            follow = []
            fade = []
            for rnd in rounds:
                lock_at = int(rnd.lock_at)
                bull_p, bear_p = get_pools(rnd, lock_at)
                pool = bull_p + bear_p
                if pool < 2.0:
                    continue

                cutoff = lock_at - POOL_CUTOFF_S
                bc = sc = 0
                for bet in rnd.bets:
                    if int(bet.created_at) > cutoff:
                        continue
                    if bet.position == "Bull":
                        bc += 1
                    else:
                        sc += 1

                tb = bc + sc
                if tb < min_bets:
                    continue

                br = bc / tb
                if br > count_ratio:
                    follow.append(settle(rnd, 0.10, "Bull"))
                    fade.append(settle(rnd, 0.10, "Bear"))
                elif br < (1 - count_ratio):
                    follow.append(settle(rnd, 0.10, "Bear"))
                    fade.append(settle(rnd, 0.10, "Bull"))

            n = len(follow)
            if n < 30:
                continue
            fw = sum(1 for p in follow if p > 0) / n * 100
            fa = sum(1 for p in fade if p > 0) / n * 100
            print(f"  min_bets={min_bets} ratio>{count_ratio:.0%}: N={n} "
                  f"follow={fw:.1f}% fade={fa:.1f}%")

    # ==================================================================
    print("\n" + "=" * 100)
    print("PART 3: WIDER BTC LOOKBACKS in multi-TF")
    print("=" * 100)

    for tfs in [(3, 7, 15, 20), (3, 7, 15, 25), (3, 7, 15, 30),
                (3, 7, 20, 30), (3, 10, 20, 30), (5, 10, 20, 30)]:
        for thresh in [0.0001, 0.0002]:
            results = []
            for rnd in rounds:
                lock_at = int(rnd.lock_at)
                epoch = int(rnd.epoch)
                cutoff_ms = (lock_at - CUTOFF_S) * 1000
                btc_raw = btc_kl.get(epoch)
                if not btc_raw:
                    continue
                btc_c_raw = get_candles(btc_raw, cutoff_ms)
                if btc_c_raw is None:
                    continue
                btc_c = [k[4] for k in btc_c_raw]
                bull_p, bear_p = get_pools(rnd, lock_at)
                pool = bull_p + bear_p
                if pool < 2.0:
                    continue

                rets = [ret(btc_c, lb) for lb in tfs]
                if any(r is None for r in rets):
                    continue
                if not (all(r > 0 for r in rets) or all(r < 0 for r in rets)):
                    continue
                m = min(abs(r) for r in rets)
                if m < thresh:
                    continue
                sig = "Bull" if rets[0] > 0 else "Bear"
                our = bull_p if sig == "Bull" else bear_p
                if our == 0:
                    continue
                pm = pool * 0.97 / our
                if pm < 1.5:
                    continue
                frac = min(0.03 + 100 * m, 0.30)
                frac = min(frac * max(0.5, 1.0 + 1.0 * (pm - 2.0)), 0.30)
                bet = max(0.01, min(2.0, pool * frac))
                results.append(settle(rnd, bet, sig))

            n = len(results)
            if n < 30:
                continue
            wr = sum(1 for p in results if p > 0) / n * 100
            pnl = sum(results) / total * 2000
            label = "+".join(str(t) for t in tfs)
            flag = " ***" if pnl > 0 else ""
            print(f"  mtf({label},t={thresh}): N={n} WR={wr:.1f}% /2k={pnl:+.3f}{flag}")

    # ==================================================================
    print("\n" + "=" * 100)
    print("PART 4: BTC-BNB PRICE RATIO MOMENTUM")
    print("=" * 100)

    for lb in [5, 7, 10, 15]:
        for thresh in [0.0003, 0.0005, 0.001]:
            results = []
            for rnd in rounds:
                lock_at = int(rnd.lock_at)
                epoch = int(rnd.epoch)
                cutoff_ms = (lock_at - CUTOFF_S) * 1000
                btc_raw = btc_kl.get(epoch)
                bnb_raw = bnb_kl.get(epoch)
                if not btc_raw or not bnb_raw:
                    continue
                btc_c_raw = get_candles(btc_raw, cutoff_ms)
                bnb_c_raw = get_candles(bnb_raw, cutoff_ms)
                if btc_c_raw is None or bnb_c_raw is None:
                    continue
                btc_c = [k[4] for k in btc_c_raw]
                bnb_c = [k[4] for k in bnb_c_raw]
                bull_p, bear_p = get_pools(rnd, lock_at)
                pool = bull_p + bear_p
                if pool < 2.0:
                    continue

                if len(btc_c) < lb + 1 or len(bnb_c) < lb + 1:
                    continue
                if btc_c[-(lb + 1)] == 0 or bnb_c[-(lb + 1)] == 0:
                    continue
                ratio_now = btc_c[-1] / bnb_c[-1]
                ratio_ago = btc_c[-(lb + 1)] / bnb_c[-(lb + 1)]
                ratio_ret = (ratio_now - ratio_ago) / ratio_ago
                if abs(ratio_ret) < thresh:
                    continue

                # Rising ratio = BTC outperforming = BNB should catch up
                sig = "Bull" if ratio_ret > 0 else "Bear"
                our = bull_p if sig == "Bull" else bear_p
                if our == 0:
                    continue
                pm = pool * 0.97 / our
                if pm < 1.5:
                    continue
                frac = min(0.05, 0.30)
                bet = max(0.01, min(2.0, pool * frac))
                results.append(settle(rnd, bet, sig))

            n = len(results)
            if n < 30:
                continue
            wr = sum(1 for p in results if p > 0) / n * 100
            pnl = sum(results) / total * 2000
            print(f"  ratio_ret(lb={lb},t={thresh}): N={n} WR={wr:.1f}% "
                  f"/2k={pnl:+.3f}")

    # ==================================================================
    print("\n" + "=" * 100)
    print("PART 5: LATE BETS AS FILTER on MTF signal")
    print("  Does late-bet direction CONFIRMING our signal improve WR?")
    print("=" * 100)

    for name, filt_fn in [
        ("no filter (baseline)", lambda rnd, lock_at, sig: True),
        ("late bets confirm signal",
         lambda rnd, lock_at, sig: _late_confirms(rnd, lock_at, sig, True)),
        ("late bets oppose signal",
         lambda rnd, lock_at, sig: _late_confirms(rnd, lock_at, sig, False)),
        ("no late bets (quiet round)",
         lambda rnd, lock_at, sig: _no_late_bets(rnd, lock_at)),
    ]:
        results = []
        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            epoch = int(rnd.epoch)
            cutoff_ms = (lock_at - CUTOFF_S) * 1000
            btc_raw = btc_kl.get(epoch)
            if not btc_raw:
                continue
            btc_c_raw = get_candles(btc_raw, cutoff_ms)
            if btc_c_raw is None:
                continue
            btc_c = [k[4] for k in btc_c_raw]
            bull_p, bear_p = get_pools(rnd, lock_at)
            pool = bull_p + bear_p
            if pool < 2.0:
                continue
            rets = [ret(btc_c, lb) for lb in [3, 7, 15]]
            if any(r is None for r in rets):
                continue
            if not (all(r > 0 for r in rets) or all(r < 0 for r in rets)):
                continue
            m = min(abs(r) for r in rets)
            if m < 0.0001:
                continue
            sig = "Bull" if rets[0] > 0 else "Bear"
            our = bull_p if sig == "Bull" else bear_p
            if our == 0:
                continue
            pm = pool * 0.97 / our
            if pm < 1.5:
                continue

            if not filt_fn(rnd, lock_at, sig):
                continue

            frac = min(0.03 + 100 * m, 0.30)
            frac = min(frac * max(0.5, 1.0 + 1.0 * (pm - 2.0)), 0.30)
            bet = max(0.01, min(2.0, pool * frac))
            results.append(settle(rnd, bet, sig))

        n = len(results)
        if n < 20:
            print(f"  {name}: N={n} (too few)")
            continue
        wr = sum(1 for p in results if p > 0) / n * 100
        pnl = sum(results) / total * 2000
        print(f"  {name}: N={n} WR={wr:.1f}% /2k={pnl:+.3f}")

    print("\nDone.")


def _late_confirms(rnd, lock_at, sig, confirm):
    """Check if late bets (lock-6 to lock-3) confirm or oppose signal."""
    lb = ls = 0
    for bet in rnd.bets:
        ts = int(bet.created_at)
        if ts <= lock_at - 6 or ts > lock_at - 3:
            continue
        if bet.position == "Bull":
            lb += int(bet.amount_wei)
        else:
            ls += int(bet.amount_wei)
    total = lb + ls
    if total == 0:
        return False
    late_bull_frac = lb / total
    if confirm:
        return (sig == "Bull" and late_bull_frac > 0.6) or \
               (sig == "Bear" and late_bull_frac < 0.4)
    else:
        return (sig == "Bull" and late_bull_frac < 0.4) or \
               (sig == "Bear" and late_bull_frac > 0.6)


def _no_late_bets(rnd, lock_at):
    """Check if there are no late bets (quiet round)."""
    for bet in rnd.bets:
        ts = int(bet.created_at)
        if lock_at - 6 < ts <= lock_at - 3:
            return False
    return True


if __name__ == "__main__":
    main()
