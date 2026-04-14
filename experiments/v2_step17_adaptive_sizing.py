"""Step 17: Adaptive bet sizing — scale bet with signal confidence.

Current best: MTF(3+7+15, 0.0003) frac=0.10 → +0.789/2k (5-fold avg)
Fold 3 hit +2.007/2k with many signals. The edge per bet needs to grow.

Hypothesis: larger BTC returns = more confident signal = should bet more.
Also: when all 3 timeframes show LARGE moves, the signal is stronger.

Approaches:
1. Continuous sizing: frac scales with min(|r_3|, |r_7|, |r_15|)
2. Tiered sizing: small/med/large tiers with different fracs
3. Confidence-weighted: weight by number of candles confirming direction
4. Variable threshold + sizing: lower thresh for small bets, higher for large
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
SKIP_NIGHT = set()  # include all hours


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
    return rounds, load_kl("var/cutoff_spot_prices.jsonl"), load_kl("var/btc_spot_prices.jsonl")


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

        r3 = ret(btc_c, 3)
        r7 = ret(btc_c, 7)
        r15 = ret(btc_c, 15)

        if r3 is None or r7 is None or r15 is None:
            continue

        data.append({
            "rnd": rnd, "lock_at": lock_at, "epoch": epoch,
            "r3": r3, "r7": r7, "r15": r15,
            "btc_c": btc_c, "bnb_c": bnb_c,
        })

    print(f"Rounds with data: {len(data)}\n")

    # =====================================================================
    print("=" * 120)
    print("PART 1: Tiered sizing — small/medium/large moves get different fracs")
    print("=" * 120)

    for base_thresh in [0.0002, 0.0003]:
        for (small_frac, med_frac, large_frac) in [(0.05, 0.10, 0.20),
                                                     (0.05, 0.15, 0.25),
                                                     (0.03, 0.10, 0.25),
                                                     (0.10, 0.15, 0.20)]:
            med_thresh = base_thresh * 2
            large_thresh = base_thresh * 4
            results = []
            for d in data:
                r3, r7, r15 = d["r3"], d["r7"], d["r15"]
                # All must agree in direction
                if not ((r3 > 0 and r7 > 0 and r15 > 0) or
                        (r3 < 0 and r7 < 0 and r15 < 0)):
                    continue
                # All must exceed base threshold
                min_abs = min(abs(r3), abs(r7), abs(r15))
                if min_abs < base_thresh:
                    continue

                # Tier by min absolute return
                if min_abs >= large_thresh:
                    frac = large_frac
                elif min_abs >= med_thresh:
                    frac = med_frac
                else:
                    frac = small_frac

                signal = "Bull" if r3 > 0 else "Bear"
                pb, pe = get_pool(d["rnd"], d["lock_at"])
                bet = max(0.01, min(2.0, (pb + pe) * frac))
                profit = settle(d["rnd"], bet, signal)
                results.append((profit, bet))

            print_result(
                f"tiered(t={base_thresh},f={small_frac}/{med_frac}/{large_frac})",
                results, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 2: Continuous scaling — frac = base + slope * min_return")
    print("=" * 120)

    for base_thresh in [0.0002, 0.0003]:
        for base_frac in [0.03, 0.05]:
            for slope in [50, 100, 200, 300]:
                results = []
                for d in data:
                    r3, r7, r15 = d["r3"], d["r7"], d["r15"]
                    if not ((r3 > 0 and r7 > 0 and r15 > 0) or
                            (r3 < 0 and r7 < 0 and r15 < 0)):
                        continue
                    min_abs = min(abs(r3), abs(r7), abs(r15))
                    if min_abs < base_thresh:
                        continue

                    frac = base_frac + slope * min_abs
                    frac = min(frac, 0.30)  # cap

                    signal = "Bull" if r3 > 0 else "Bear"
                    pb, pe = get_pool(d["rnd"], d["lock_at"])
                    bet = max(0.01, min(2.0, (pb + pe) * frac))
                    profit = settle(d["rnd"], bet, signal)
                    results.append((profit, bet))

                print_result(
                    f"continuous(t={base_thresh},base={base_frac},slope={slope})",
                    results, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 3: Variable threshold — lower thresh=more bets at small frac")
    print("=" * 120)

    # Idea: thresh=0.0002 bets at frac=0.03, thresh=0.0003 adds frac=0.07, etc.
    # Equivalent to tiered but with explicit threshold bands
    for config in [
        [(0.0002, 0.03), (0.0003, 0.05), (0.0005, 0.10)],
        [(0.0001, 0.02), (0.0003, 0.08), (0.0005, 0.15)],
        [(0.0002, 0.05), (0.0005, 0.15)],
        [(0.0003, 0.10)],  # baseline for comparison
    ]:
        results = []
        for d in data:
            r3, r7, r15 = d["r3"], d["r7"], d["r15"]
            if not ((r3 > 0 and r7 > 0 and r15 > 0) or
                    (r3 < 0 and r7 < 0 and r15 < 0)):
                continue
            min_abs = min(abs(r3), abs(r7), abs(r15))

            # Find the best matching threshold
            frac = 0
            for thresh, f in config:
                if min_abs >= thresh:
                    frac = f  # take the highest qualifying frac
            if frac == 0:
                continue

            signal = "Bull" if r3 > 0 else "Bear"
            pb, pe = get_pool(d["rnd"], d["lock_at"])
            bet = max(0.01, min(2.0, (pb + pe) * frac))
            profit = settle(d["rnd"], bet, signal)
            results.append((profit, bet))

        label = "+".join(f"t{t}@{f}" for t, f in config)
        print_result(f"var_thresh({label})", results, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 4: 5-fold validation of best adaptive strategies")
    print("=" * 120)

    def run_strategy(data_list, sizing_fn):
        results = []
        for d in data_list:
            r3, r7, r15 = d["r3"], d["r7"], d["r15"]
            if not ((r3 > 0 and r7 > 0 and r15 > 0) or
                    (r3 < 0 and r7 < 0 and r15 < 0)):
                continue
            min_abs = min(abs(r3), abs(r7), abs(r15))
            frac = sizing_fn(min_abs)
            if frac is None or frac <= 0:
                continue
            signal = "Bull" if r3 > 0 else "Bear"
            pb, pe = get_pool(d["rnd"], d["lock_at"])
            bet = max(0.01, min(2.0, (pb + pe) * frac))
            profit = settle(d["rnd"], bet, signal)
            results.append((profit, bet))
        return results

    strategies = {
        "flat_frac=0.10_t=0.0003": lambda m: 0.10 if m >= 0.0003 else None,
        "flat_frac=0.10_t=0.0002": lambda m: 0.10 if m >= 0.0002 else None,
        "tiered_0.0003_5/10/20": lambda m: (
            0.20 if m >= 0.0012 else
            0.10 if m >= 0.0006 else
            0.05 if m >= 0.0003 else None),
        "tiered_0.0002_3/10/20": lambda m: (
            0.20 if m >= 0.0008 else
            0.10 if m >= 0.0004 else
            0.03 if m >= 0.0002 else None),
        "continuous_t=0.0003_base=0.03_slope=100": lambda m: (
            min(0.03 + 100 * m, 0.30) if m >= 0.0003 else None),
        "continuous_t=0.0002_base=0.03_slope=100": lambda m: (
            min(0.03 + 100 * m, 0.30) if m >= 0.0002 else None),
        "var_t0002@0.05+t0005@0.15": lambda m: (
            0.15 if m >= 0.0005 else
            0.05 if m >= 0.0002 else None),
    }

    fold_size = len(data) // 5
    for name, sizing_fn in strategies.items():
        print(f"\n  --- {name} ---")
        fold_pnls = []
        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size
            fold_data = data[start:end]
            results = run_strategy(fold_data, sizing_fn)
            profits = [p for p, b in results]
            bets_list = [b for p, b in results]
            n = len(results)
            wr = sum(1 for p in profits if p > 0) / max(1, n) * 100
            pnl = sum(profits)
            pnl_2k = pnl / len(fold_data) * 2000
            avg_bet = sum(bets_list) / max(1, n)
            fold_pnls.append(pnl_2k)
            print(f"    Fold {fold+1}: WR={wr:5.1f}%({n:3d}) PnL={pnl:+7.3f} "
                  f"avg_bet={avg_bet:.3f} /2k={pnl_2k:+6.3f}")

        avg = sum(fold_pnls) / 5
        pos = sum(1 for p in fold_pnls if p > 0)
        mn = min(fold_pnls)
        mx = max(fold_pnls)
        print(f"    => avg /2k={avg:+.3f} (min={mn:+.3f} max={mx:+.3f}) "
              f"{pos}/5 positive folds")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 5: Per-tier WR analysis — do larger moves really predict better?")
    print("=" * 120)

    tiers = [
        ("tiny",  0.0002, 0.0003),
        ("small", 0.0003, 0.0005),
        ("med",   0.0005, 0.001),
        ("large", 0.001,  0.01),
    ]
    for tname, lo, hi in tiers:
        wins = 0
        count = 0
        for d in data:
            r3, r7, r15 = d["r3"], d["r7"], d["r15"]
            if not ((r3 > 0 and r7 > 0 and r15 > 0) or
                    (r3 < 0 and r7 < 0 and r15 < 0)):
                continue
            min_abs = min(abs(r3), abs(r7), abs(r15))
            if min_abs < lo or min_abs >= hi:
                continue
            signal = "Bull" if r3 > 0 else "Bear"
            profit = settle(d["rnd"], 0.10, signal)
            count += 1
            if profit > 0:
                wins += 1
        if count > 0:
            print(f"  {tname} ({lo}-{hi}): N={count} WR={wins/count*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
