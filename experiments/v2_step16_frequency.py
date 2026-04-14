"""Step 16: Attack the frequency constraint.

Current best: multi_tf(3+7+15, 0.0003) gives 46 bets/2k at 60.9% WR = +0.29/2k.
Target: 4.0/2k = 14x gap.

Two paths:
A) More signals per round — relax thresholds, use 2-of-3 agreement, shorter timeframes
B) Multiple independent signal families firing on different rounds

Also: test whether the 5-fold validated strategies actually compound when pool-sized.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from collections import defaultdict

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
SKIP_NIGHT = {0, 1, 2, 3, 4, 23}


def load_data():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())

    def load_kl(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    return rounds, load_kl("var/bnb_spot_prices.jsonl"), load_kl("var/btc_spot_prices.jsonl")


def get_candles(raw, cutoff_ms):
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None
    return trimmed[-CANDLE_COUNT:]


def settle(rnd, bet_bnb, side):
    out = settle_bet_against_closed_round(
        bet_bnb=bet_bnb, bet_side=side,
        round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
    )
    return out.credit_bnb - bet_bnb - GAS_COST_BET_BNB


def ret(closes, lb):
    if len(closes) < lb + 1 or closes[-(lb+1)] == 0:
        return None
    return (closes[-1] - closes[-(lb+1)]) / closes[-(lb+1)]


def get_pool(rnd, lock_at):
    pool_cutoff_ts = lock_at - POOL_CUTOFF_S
    bull_wei = bear_wei = 0
    for bet in rnd.bets:
        if int(bet.created_at) > pool_cutoff_ts:
            continue
        if bet.position == "Bull":
            bull_wei += int(bet.amount_wei)
        else:
            bear_wei += int(bet.amount_wei)
    return bull_wei / 1e18, bear_wei / 1e18


def print_result(label, results, total):
    n = len(results)
    if n < 15:
        print(f"  {label}: N={n} (too few)")
        return
    profits = [p for p, b in results]
    bets = [b for p, b in results]
    wins = sum(1 for p in profits if p > 0)
    wr = wins / n * 100
    pnl = sum(profits)
    pnl_2k = pnl / total * 2000
    avg_bet = sum(bets) / n
    bets_2k = n / total * 2000
    flag = " ***" if pnl > 0 else ""
    print(f"  {label}: WR={wr:5.1f}%({n:4d}) PnL={pnl:+8.3f} avg={avg_bet:.3f} "
          f"/2k={pnl_2k:+6.3f}({bets_2k:.0f}b){flag}")


def main():
    rounds, bnb_kl, btc_kl = load_data()
    total = len(rounds)
    print(f"Total rounds: {total}")

    # Pre-compute
    data = []
    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        hour = (lock_at % 86400) // 3600
        if hour in SKIP_NIGHT:
            continue
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
        btc_v = [k[5] for k in btc_c_raw]

        # Pre-compute returns
        btc_rets = {}
        bnb_rets = {}
        for lb in [1, 2, 3, 5, 7, 10, 15, 20]:
            btc_rets[lb] = ret(btc_c, lb)
            bnb_rets[lb] = ret(bnb_c, lb)

        # Volume ratio
        if len(btc_v) >= 10:
            rv = sum(btc_v[-5:]) / 5
            bv = sum(btc_v[-10:-5]) / 5
            vol_ratio = rv / bv if bv > 0 else 0
        else:
            vol_ratio = 0

        # VWAP returns
        vwap = {}
        for lb in [5, 7, 10]:
            tv = sum(btc_v[-lb:])
            if tv > 0 and len(btc_c) >= lb + 1:
                vw = 0
                for i in range(lb):
                    idx = -(lb - i)
                    if btc_c[idx - 1] != 0:
                        r_i = (btc_c[idx] - btc_c[idx - 1]) / btc_c[idx - 1]
                        vw += r_i * (btc_v[idx] / tv)
                vwap[lb] = vw
            else:
                vwap[lb] = None

        data.append({
            "rnd": rnd, "epoch": epoch, "hour": hour, "lock_at": lock_at,
            "btc_r": btc_rets, "bnb_r": bnb_rets, "vol_ratio": vol_ratio,
            "vwap": vwap, "btc_c": btc_c, "bnb_c": bnb_c,
        })

    print(f"Rounds with data: {len(data)}\n")

    # =====================================================================
    print("=" * 120)
    print("PART 1: Relax multi-tf threshold — more signals, lower WR?")
    print("=" * 120)

    for tfs in [(3,7,15), (3,5,7), (2,5,10), (3,7), (5,15), (3,10)]:
        for thresh in [0.0001, 0.0002, 0.0003, 0.0005]:
            results = []
            for d in data:
                sigs = []
                for lb in tfs:
                    r = d["btc_r"].get(lb)
                    if r is None or abs(r) < thresh:
                        sigs.append(0)
                    else:
                        sigs.append(1 if r > 0 else -1)
                non_zero = [s for s in sigs if s != 0]
                if len(non_zero) < len(tfs):
                    continue
                if len(set(non_zero)) > 1:
                    continue
                signal = "Bull" if non_zero[0] > 0 else "Bear"
                profit = settle(d["rnd"], 0.10, signal)
                results.append((profit, 0.10))
            label = "+".join(str(t) for t in tfs)
            print_result(f"mtf({label},t={thresh})", results, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 2: 2-of-3 agreement (relaxed from unanimous)")
    print("=" * 120)

    for tfs in [(3,7,15), (3,5,10), (5,10,20), (3,7,20)]:
        for thresh in [0.0002, 0.0003, 0.0005]:
            results = []
            for d in data:
                votes = []
                for lb in tfs:
                    r = d["btc_r"].get(lb)
                    if r is not None and abs(r) >= thresh:
                        votes.append(1 if r > 0 else -1)
                if len(votes) < 2:
                    continue
                bull = sum(1 for v in votes if v > 0)
                bear = sum(1 for v in votes if v < 0)
                if bull >= 2:
                    signal = "Bull"
                elif bear >= 2:
                    signal = "Bear"
                else:
                    continue
                profit = settle(d["rnd"], 0.10, signal)
                results.append((profit, 0.10))
            label = "+".join(str(t) for t in tfs)
            print_result(f"2of3({label},t={thresh})", results, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 3: Multiple signal families — how much overlap?")
    print("  Count rounds where different signal types fire")
    print("=" * 120)

    # Define signal families
    def sig_mtf(d, thresh=0.0003):
        for lb in [3, 7, 15]:
            r = d["btc_r"].get(lb)
            if r is None or abs(r) < thresh:
                return None
            if lb == 3:
                direction = 1 if r > 0 else -1
            elif (1 if r > 0 else -1) != direction:
                return None
        return "Bull" if direction > 0 else "Bear"

    def sig_vwap(d, lb=7, thresh=0.0007):
        v = d["vwap"].get(lb)
        if v is None or abs(v) < thresh:
            return None
        return "Bull" if v > 0 else "Bear"

    def sig_accel(d, lb=5, thresh=0.0007):
        r = d["btc_r"].get(lb)
        r2 = d["btc_r"].get(2)
        if r is None or r2 is None or abs(r) < thresh:
            return None
        if (r2 > 0) != (r > 0):
            return None
        return "Bull" if r > 0 else "Bear"

    def sig_bnb_catch(d, btc_lb=7, bnb_lb=5, thresh=0.001):
        br = d["btc_r"].get(btc_lb)
        nr = d["bnb_r"].get(bnb_lb)
        if br is None or nr is None or abs(br) < 0.0005:
            return None
        gap = br - nr
        if abs(gap) < thresh:
            return None
        return "Bull" if gap > 0 else "Bear"

    # Count overlaps
    mtf_set = set()
    vwap_set = set()
    accel_set = set()
    catch_set = set()

    for d in data:
        if sig_mtf(d) is not None:
            mtf_set.add(d["epoch"])
        if sig_vwap(d) is not None:
            vwap_set.add(d["epoch"])
        if sig_accel(d) is not None:
            accel_set.add(d["epoch"])
        if sig_bnb_catch(d) is not None:
            catch_set.add(d["epoch"])

    print(f"  MTF fires: {len(mtf_set)}")
    print(f"  VWAP fires: {len(vwap_set)}")
    print(f"  Accel fires: {len(accel_set)}")
    print(f"  BNB-catch fires: {len(catch_set)}")
    print(f"  MTF & VWAP: {len(mtf_set & vwap_set)}")
    print(f"  MTF & Accel: {len(mtf_set & accel_set)}")
    print(f"  VWAP unique (not in MTF): {len(vwap_set - mtf_set)}")
    print(f"  Accel unique (not in MTF): {len(accel_set - mtf_set)}")
    print(f"  Catch unique (not in MTF): {len(catch_set - mtf_set)}")
    print(f"  Union all: {len(mtf_set | vwap_set | accel_set | catch_set)}")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 4: Waterfall stacking — MTF first, then unique VWAP/accel/catch")
    print("=" * 120)

    for frac in [0.05, 0.10, 0.15]:
        results_by_source = defaultdict(list)
        all_results = []
        for d in data:
            # Priority: MTF > VWAP > Accel > BNB-catch
            sig = sig_mtf(d)
            source = "mtf"
            if sig is None:
                sig = sig_vwap(d)
                source = "vwap"
            if sig is None:
                sig = sig_accel(d)
                source = "accel"
            if sig is None:
                sig = sig_bnb_catch(d)
                source = "catch"
            if sig is None:
                continue

            pb, pe = get_pool(d["rnd"], d["lock_at"])
            bet = max(0.01, min(2.0, (pb + pe) * frac))
            profit = settle(d["rnd"], bet, sig)
            all_results.append((profit, bet))
            results_by_source[source].append((profit, bet))

        print_result(f"waterfall(mtf>vwap>accel>catch) frac={frac}", all_results, total)
        for src in ["mtf", "vwap", "accel", "catch"]:
            if src in results_by_source:
                print_result(f"  -> {src}", results_by_source[src], total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 5: 5-fold on best waterfall + pool sizing")
    print("=" * 120)

    fold_size = len(data) // 5
    for frac in [0.05, 0.10, 0.15]:
        print(f"\n  --- waterfall frac={frac} ---")
        fold_pnls = []
        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size
            fold_data = data[start:end]
            results = []
            for d in fold_data:
                sig = sig_mtf(d)
                if sig is None:
                    sig = sig_vwap(d)
                if sig is None:
                    sig = sig_accel(d)
                if sig is None:
                    sig = sig_bnb_catch(d)
                if sig is None:
                    continue
                pb, pe = get_pool(d["rnd"], d["lock_at"])
                bet = max(0.01, min(2.0, (pb + pe) * frac))
                profit = settle(d["rnd"], bet, sig)
                results.append(profit)

            n = len(results)
            wr = sum(1 for p in results if p > 0) / max(1, n) * 100
            pnl = sum(results)
            pnl_2k = pnl / len(fold_data) * 2000
            fold_pnls.append(pnl_2k)
            print(f"    Fold {fold+1}: WR={wr:5.1f}%({n:3d}) PnL={pnl:+7.3f} /2k={pnl_2k:+6.3f}")

        avg = sum(fold_pnls) / 5
        pos = sum(1 for p in fold_pnls if p > 0)
        print(f"    => avg /2k={avg:+.3f} ({pos}/5 positive folds)")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 6: 5-fold on MTF-only with pool sizing")
    print("=" * 120)

    for frac in [0.05, 0.10, 0.15]:
        print(f"\n  --- mtf(3+7+15,0.0003) frac={frac} ---")
        fold_pnls = []
        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size
            fold_data = data[start:end]
            results = []
            for d in fold_data:
                sig = sig_mtf(d)
                if sig is None:
                    continue
                pb, pe = get_pool(d["rnd"], d["lock_at"])
                bet = max(0.01, min(2.0, (pb + pe) * frac))
                profit = settle(d["rnd"], bet, sig)
                results.append(profit)

            n = len(results)
            wr = sum(1 for p in results if p > 0) / max(1, n) * 100
            pnl = sum(results)
            pnl_2k = pnl / len(fold_data) * 2000
            fold_pnls.append(pnl_2k)
            print(f"    Fold {fold+1}: WR={wr:5.1f}%({n:3d}) PnL={pnl:+7.3f} /2k={pnl_2k:+6.3f}")

        avg = sum(fold_pnls) / 5
        pos = sum(1 for p in fold_pnls if p > 0)
        print(f"    => avg /2k={avg:+.3f} ({pos}/5 positive folds)")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 7: Ultra-short timeframes — 1s, 2s, 3s agreement")
    print("=" * 120)

    for tfs in [(1,2,3), (1,3,5), (2,3,5), (1,2,3,5)]:
        for thresh in [0.0001, 0.0002, 0.0003, 0.0005]:
            results = []
            for d in data:
                sigs = []
                for lb in tfs:
                    r = d["btc_r"].get(lb)
                    if r is None or abs(r) < thresh:
                        sigs.append(0)
                    else:
                        sigs.append(1 if r > 0 else -1)
                non_zero = [s for s in sigs if s != 0]
                if len(non_zero) < len(tfs):
                    continue
                if len(set(non_zero)) > 1:
                    continue
                signal = "Bull" if non_zero[0] > 0 else "Bear"
                profit = settle(d["rnd"], 0.10, signal)
                results.append((profit, 0.10))
            label = "+".join(str(t) for t in tfs)
            print_result(f"ultra_short({label},t={thresh})", results, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 8: Ensemble scoring — weighted sum of multiple signals")
    print("=" * 120)

    for score_thresh in [2, 3, 4, 5]:
        results = []
        for d in data:
            score = 0

            # MTF component (weight 2)
            mtf_sig = sig_mtf(d)
            if mtf_sig is not None:
                score += 2 if mtf_sig == "Bull" else -2

            # Accel component (weight 1)
            accel_sig = sig_accel(d)
            if accel_sig is not None:
                score += 1 if accel_sig == "Bull" else -1

            # VWAP component (weight 1)
            vwap_sig = sig_vwap(d)
            if vwap_sig is not None:
                score += 1 if vwap_sig == "Bull" else -1

            # Volume (weight 1)
            if d["vol_ratio"] >= 1.5:
                # Volume confirms direction of BTC 5s
                r5 = d["btc_r"].get(5)
                if r5 is not None and abs(r5) > 0.0003:
                    score += 1 if r5 > 0 else -1

            if abs(score) < score_thresh:
                continue
            signal = "Bull" if score > 0 else "Bear"
            profit = settle(d["rnd"], 0.10, signal)
            results.append((profit, 0.10))
        print_result(f"ensemble(score>={score_thresh})", results, total)

    print("\nDone.")


if __name__ == "__main__":
    main()
