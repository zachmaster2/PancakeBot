"""Scrutinize the best strategy for validity.

Tests:
1. RANDOMIZATION TEST: shuffle round outcomes → edge should vanish
2. NESTED CROSS-VALIDATION: select params on 4 folds, test on held-out fold
3. PARAMETER SENSITIVITY: how much does PnL change with small param changes?
4. TEMPORAL STABILITY: does edge decay over time? (rolling window)
5. DIRECTION BIAS: is there a bull/bear asymmetry?
6. POOL SIZE DEPENDENCE: does strategy work differently on small vs large pools?
"""
from __future__ import annotations

import json, sys, random
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


def mtf_signal_and_strength(d, thresh):
    """Returns (signal, min_abs) or (None, 0)."""
    r3, r7, r15 = d["r3"], d["r7"], d["r15"]
    if not ((r3 > 0 and r7 > 0 and r15 > 0) or
            (r3 < 0 and r7 < 0 and r15 < 0)):
        return None, 0
    min_abs = min(abs(r3), abs(r7), abs(r15))
    if min_abs < thresh:
        return None, 0
    return ("Bull" if r3 > 0 else "Bear"), min_abs


def run_strategy(data_list, thresh, base, slope, cap=0.30):
    """Run the continuous adaptive strategy. Returns list of (profit, bet, signal)."""
    results = []
    for d in data_list:
        signal, min_abs = mtf_signal_and_strength(d, thresh)
        if signal is None:
            continue
        frac = min(base + slope * min_abs, cap)
        pb, pe = get_pool(d["rnd"], d["lock_at"])
        bet = max(0.01, min(2.0, (pb + pe) * frac))
        profit = settle(d["rnd"], bet, signal)
        results.append((profit, bet, signal))
    return results


def summarize(results, total_rounds, label=""):
    n = len(results)
    if n < 10:
        return f"  {label}: N={n} (too few)"
    profits = [p for p, b, s in results]
    bets = [b for p, b, s in results]
    wins = sum(1 for p in profits if p > 0)
    wr = wins / n * 100
    pnl = sum(profits)
    pnl_2k = pnl / total_rounds * 2000
    avg_bet = sum(bets) / n
    return (f"  {label}: WR={wr:5.1f}%({n:4d}) PnL={pnl:+8.3f} avg_bet={avg_bet:.3f} "
            f"/2k={pnl_2k:+6.3f}")


def main():
    rounds, bnb_kl, btc_kl = load_data()
    total = len(rounds)
    print(f"Total rounds: {total}")

    data = []
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
        r3 = ret(btc_c, 3)
        r7 = ret(btc_c, 7)
        r15 = ret(btc_c, 15)
        if r3 is None or r7 is None or r15 is None:
            continue

        data.append({
            "rnd": rnd, "lock_at": lock_at, "epoch": epoch,
            "r3": r3, "r7": r7, "r15": r15,
        })

    print(f"Rounds with data: {len(data)}\n")

    # =====================================================================
    print("=" * 120)
    print("TEST 1: RANDOMIZATION — shuffle round outcomes, edge should vanish")
    print("=" * 120)

    # Real edge
    real = run_strategy(data, 0.0002, 0.03, 100)
    print(summarize(real, total, "REAL"))

    # Randomized: for each signal round, randomly assign Bull or Bear
    random.seed(42)
    rand_pnls = []
    for trial in range(20):
        trial_pnl = 0
        n = 0
        for d in data:
            signal, min_abs = mtf_signal_and_strength(d, 0.0002)
            if signal is None:
                continue
            # Random direction instead of signal
            rand_signal = random.choice(["Bull", "Bear"])
            frac = min(0.03 + 100 * min_abs, 0.30)
            pb, pe = get_pool(d["rnd"], d["lock_at"])
            bet = max(0.01, min(2.0, (pb + pe) * frac))
            profit = settle(d["rnd"], bet, rand_signal)
            trial_pnl += profit
            n += 1
        rand_pnls.append(trial_pnl)

    avg_rand = sum(rand_pnls) / len(rand_pnls)
    min_rand = min(rand_pnls)
    max_rand = max(rand_pnls)
    print(f"  RANDOM (20 trials): avg={avg_rand:+.3f} min={min_rand:+.3f} max={max_rand:+.3f}")
    real_pnl = sum(p for p, b, s in real)
    print(f"  Real PnL={real_pnl:+.3f} vs Random avg={avg_rand:+.3f}")
    print(f"  Edge over random: {real_pnl - avg_rand:+.3f}")

    # Also: always-Bull and always-Bear baselines
    bull_pnl = bear_pnl = 0
    bull_n = bear_n = 0
    for d in data:
        signal, min_abs = mtf_signal_and_strength(d, 0.0002)
        if signal is None:
            continue
        frac = min(0.03 + 100 * min_abs, 0.30)
        pb, pe = get_pool(d["rnd"], d["lock_at"])
        bet = max(0.01, min(2.0, (pb + pe) * frac))
        bull_pnl += settle(d["rnd"], bet, "Bull")
        bear_pnl += settle(d["rnd"], bet, "Bear")
        bull_n += 1
    print(f"  Always-Bull: PnL={bull_pnl:+.3f} ({bull_n} bets)")
    print(f"  Always-Bear: PnL={bear_pnl:+.3f} ({bull_n} bets)")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("TEST 2: NESTED CROSS-VALIDATION — select params on 4 folds, test on held-out")
    print("=" * 120)

    fold_size = len(data) // 5
    param_grid = [
        (0.0001, 0.03, 50), (0.0001, 0.03, 100), (0.0001, 0.05, 100),
        (0.0002, 0.03, 50), (0.0002, 0.03, 100), (0.0002, 0.05, 100),
        (0.0002, 0.05, 200),
        (0.0003, 0.03, 100), (0.0003, 0.05, 100), (0.0003, 0.10, 0),
        (0.0003, 0.05, 200),
        (0.0005, 0.10, 0), (0.0005, 0.15, 0),
    ]

    nested_fold_pnls = []
    for test_fold in range(5):
        # Train on other 4 folds
        train_data = []
        test_data = []
        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size if fold < 4 else len(data)
            if fold == test_fold:
                test_data = data[start:end]
            else:
                train_data.extend(data[start:end])

        train_total = len(train_data) + len(test_data)  # total rounds for normalization

        # Find best params on training folds
        best_pnl_2k = -999
        best_params = None
        for thresh, base, slope in param_grid:
            results = run_strategy(train_data, thresh, base, slope)
            if len(results) < 10:
                continue
            pnl = sum(p for p, b, s in results)
            pnl_2k = pnl / len(train_data) * 2000
            if pnl_2k > best_pnl_2k:
                best_pnl_2k = pnl_2k
                best_params = (thresh, base, slope)

        # Test on held-out fold
        results = run_strategy(test_data, *best_params)
        profits = [p for p, b, s in results]
        n = len(results)
        wr = sum(1 for p in profits if p > 0) / max(1, n) * 100
        pnl = sum(profits)
        pnl_2k = pnl / len(test_data) * 2000

        nested_fold_pnls.append(pnl_2k)
        print(f"  Fold {test_fold+1}: best_params={best_params} "
              f"WR={wr:.1f}%({n}) test_/2k={pnl_2k:+.3f}")

    avg_nested = sum(nested_fold_pnls) / 5
    pos_nested = sum(1 for p in nested_fold_pnls if p > 0)
    print(f"  => Nested CV avg /2k={avg_nested:+.3f} ({pos_nested}/5 positive)")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("TEST 3: PARAMETER SENSITIVITY — how robust to small changes?")
    print("=" * 120)

    for thresh in [0.00015, 0.0002, 0.00025, 0.0003]:
        for base in [0.02, 0.03, 0.04, 0.05]:
            for slope in [50, 75, 100, 125, 150]:
                results = run_strategy(data, thresh, base, slope)
                if len(results) < 50:
                    continue
                n = len(results)
                profits = [p for p, b, s in results]
                pnl = sum(profits)
                pnl_2k = pnl / total * 2000
                wr = sum(1 for p in profits if p > 0) / n * 100
                flag = " ***" if pnl > 0 else ""
                if abs(thresh - 0.0002) <= 0.0001 or abs(thresh - 0.0003) <= 0.00005:
                    print(f"  t={thresh:.5f} b={base:.2f} s={slope:3d}: "
                          f"WR={wr:5.1f}%({n:4d}) /2k={pnl_2k:+6.3f}{flag}")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("TEST 4: TEMPORAL STABILITY — rolling 5k-round windows")
    print("=" * 120)

    window = 5000
    for start in range(0, len(data) - window + 1, 2500):
        window_data = data[start:start + window]
        results = run_strategy(window_data, 0.0002, 0.03, 100)
        if len(results) < 10:
            continue
        profits = [p for p, b, s in results]
        n = len(results)
        wr = sum(1 for p in profits if p > 0) / n * 100
        pnl = sum(profits)
        pnl_2k = pnl / window * 2000
        epoch_start = window_data[0]["epoch"]
        epoch_end = window_data[-1]["epoch"]
        print(f"  rounds {start:5d}-{start+window:5d} (ep {epoch_start}-{epoch_end}): "
              f"WR={wr:5.1f}%({n:3d}) /2k={pnl_2k:+6.3f}")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("TEST 5: DIRECTION BIAS — Bull vs Bear signal performance")
    print("=" * 120)

    results = run_strategy(data, 0.0002, 0.03, 100)
    bull_results = [(p, b) for p, b, s in results if s == "Bull"]
    bear_results = [(p, b) for p, b, s in results if s == "Bear"]

    bull_n = len(bull_results)
    bear_n = len(bear_results)
    bull_wr = sum(1 for p, b in bull_results if p > 0) / bull_n * 100 if bull_n else 0
    bear_wr = sum(1 for p, b in bear_results if p > 0) / bear_n * 100 if bear_n else 0
    bull_pnl = sum(p for p, b in bull_results)
    bear_pnl_v = sum(p for p, b in bear_results)

    print(f"  Bull signals: N={bull_n} WR={bull_wr:.1f}% PnL={bull_pnl:+.3f}")
    print(f"  Bear signals: N={bear_n} WR={bear_wr:.1f}% PnL={bear_pnl_v:+.3f}")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("TEST 6: POOL SIZE DEPENDENCE")
    print("=" * 120)

    pool_buckets = {"tiny(<1)": [], "small(1-2)": [], "med(2-4)": [], "large(>4)": []}
    for d in data:
        signal, min_abs = mtf_signal_and_strength(d, 0.0002)
        if signal is None:
            continue
        pb, pe = get_pool(d["rnd"], d["lock_at"])
        vis_pool = pb + pe
        frac = min(0.03 + 100 * min_abs, 0.30)
        bet = max(0.01, min(2.0, vis_pool * frac))
        profit = settle(d["rnd"], bet, signal)

        if vis_pool < 1:
            pool_buckets["tiny(<1)"].append((profit, bet))
        elif vis_pool < 2:
            pool_buckets["small(1-2)"].append((profit, bet))
        elif vis_pool < 4:
            pool_buckets["med(2-4)"].append((profit, bet))
        else:
            pool_buckets["large(>4)"].append((profit, bet))

    for bname, trades in pool_buckets.items():
        if not trades:
            print(f"  {bname}: no trades")
            continue
        n = len(trades)
        wr = sum(1 for p, b in trades if p > 0) / n * 100
        pnl = sum(p for p, b in trades)
        avg_bet = sum(b for p, b in trades) / n
        print(f"  {bname}: N={n} WR={wr:.1f}% PnL={pnl:+.3f} avg_bet={avg_bet:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
