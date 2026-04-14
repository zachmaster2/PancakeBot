"""Step 12: Rerun core strategy with cutoff_seconds=2 data.

The resync produces klines trimmed to lock_at-2 instead of lock_at-4.
This gives us 2 extra seconds of BTC/BNB data -- the most recent, most
informative seconds closest to the lock event.

This script:
1. Check data availability and quality
2. Rerun the core strategy (BTC lead + spread + skip_night) at cutoff=2
3. Compare with cutoff=4 results
4. 5-fold validation with cutoff=2 to check regime robustness
5. Best strategy optimization with new data
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import (
    GAS_COST_BET_BNB, INTERVAL_SECONDS, BNB_WEI, POOL_CUTOFF_SECONDS,
)
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import _trim_to_window, _get_return
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2  # THE KEY CHANGE
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

    # BNB: resync replaced original file, so .jsonl is cutoff=2
    # BTC: check .new first (in-progress), then original (may be old cutoff=4)
    bnb_path = "var/cutoff_spot_prices.jsonl"
    btc_path = "var/btc_spot_prices.jsonl.new"
    if not Path(btc_path).exists():
        btc_path = "var/btc_spot_prices.jsonl"
        print(f"WARNING: BTC may be old cutoff=4 data from {btc_path}")

    print(f"BNB klines: {bnb_path}")
    print(f"BTC klines: {btc_path}")

    return rounds, load_kl(bnb_path), load_kl(btc_path)


def get_closes(raw, cutoff_ms):
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None
    return [k[4] for k in trimmed]


def get_pool_at_cutoff(rnd, lock_at):
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


def settle(rnd, bet_bnb, side):
    out = settle_bet_against_closed_round(
        bet_bnb=bet_bnb, bet_side=side,
        round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
    )
    return out.credit_bnb - bet_bnb - GAS_COST_BET_BNB


def run_stacked(rounds, btc_kl, spot_kl, pool_frac=0.25, skip_night=True):
    results = []
    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        hour = (lock_at % 86400) // 3600
        if skip_night and hour in SKIP_NIGHT:
            continue
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        btc_raw = btc_kl.get(epoch)
        bnb_raw = spot_kl.get(epoch)
        if not btc_raw or not bnb_raw:
            continue
        btc_closes = get_closes(btc_raw, cutoff_ms)
        bnb_closes = get_closes(bnb_raw, cutoff_ms)
        if btc_closes is None or bnb_closes is None:
            continue

        btc_r = _get_return(btc_closes, 7)
        if btc_r is None:
            continue

        signal = None
        source = None

        if abs(btc_r) >= 0.0007:
            btc_r_short = _get_return(btc_closes, 2)
            if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                signal = "Bull" if btc_r > 0 else "Bear"
                source = "btc"

        if signal is None:
            bnb_r = _get_return(bnb_closes, 5)
            if bnb_r is not None:
                spread = btc_r - bnb_r
                if abs(spread) >= 0.0007:
                    signal = "Bull" if spread > 0 else "Bear"
                    source = "spread"

        if signal is None:
            continue

        pb, pe = get_pool_at_cutoff(rnd, lock_at)
        vis_pool = pb + pe
        bet = max(0.01, min(2.0, vis_pool * pool_frac))
        profit = settle(rnd, bet, signal)
        results.append((profit, bet, source))

    return results


def main():
    rounds, spot_kl, btc_kl = load_data()
    total = len(rounds)
    print(f"Total rounds: {total}")

    # Check data availability
    btc_avail = sum(1 for rnd in rounds if btc_kl.get(int(rnd.epoch)) is not None)
    bnb_avail = sum(1 for rnd in rounds if spot_kl.get(int(rnd.epoch)) is not None)
    print(f"BTC klines available: {btc_avail}/{total} ({btc_avail/total*100:.1f}%)")
    print(f"BNB klines available: {bnb_avail}/{total} ({bnb_avail/total*100:.1f}%)")

    # Test how many rounds have valid closes at cutoff=2
    valid_count = 0
    for rnd in rounds[:1000]:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000
        btc_raw = btc_kl.get(epoch)
        if btc_raw:
            btc_closes = get_closes(btc_raw, cutoff_ms)
            if btc_closes is not None:
                valid_count += 1
    print(f"Valid BTC closes at cutoff={CUTOFF_S} (first 1000): {valid_count}/1000\n")

    # =====================================================================
    print("=" * 130)
    print(f"PART 1: Core strategy at cutoff={CUTOFF_S} -- BTC lead + accel")
    print("=" * 130)

    for btc_lb in [5, 7, 10]:
        for btc_thresh in [0.0005, 0.0007, 0.001]:
            trades = []
            bets = []

            for rnd in rounds:
                lock_at = int(rnd.lock_at)
                epoch = int(rnd.epoch)
                hour = (lock_at % 86400) // 3600
                if hour in SKIP_NIGHT:
                    continue
                cutoff_ms = (lock_at - CUTOFF_S) * 1000

                btc_raw = btc_kl.get(epoch)
                if not btc_raw:
                    continue
                btc_closes = get_closes(btc_raw, cutoff_ms)
                if btc_closes is None:
                    continue
                btc_r = _get_return(btc_closes, btc_lb)
                if btc_r is None or abs(btc_r) < btc_thresh:
                    continue
                signal = "Bull" if btc_r > 0 else "Bear"

                btc_r_short = _get_return(btc_closes, 2)
                if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                    continue

                pb, pe = get_pool_at_cutoff(rnd, lock_at)
                vis_pool = pb + pe
                bet = max(0.01, min(2.0, vis_pool * 0.25))
                profit = settle(rnd, bet, signal)
                trades.append(profit)
                bets.append(bet)

            n = len(trades)
            if n < 20:
                continue
            wr = sum(1 for p in trades if p > 0) / n * 100
            pnl = sum(trades)
            pnl_2k = pnl / total * 2000
            bets_2k = n / total * 2000
            avg_bet = sum(bets) / n
            flag = " ***" if pnl > 0 else ""
            print(f"  btc({btc_lb},{btc_thresh})+accel frac=0.25  "
                  f"WR={wr:5.1f}%({n:4d}) PnL={pnl:+7.2f} avg={avg_bet:.3f} "
                  f"/2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print(f"PART 2: Stacked strategy (BTC lead + spread) at cutoff={CUTOFF_S}")
    print("=" * 130)

    for pool_frac in [0.10, 0.15, 0.20, 0.25, 0.28]:
        results = run_stacked(rounds, btc_kl, spot_kl, pool_frac=pool_frac)
        profits = [p for p, b, s in results]
        bets_list = [b for p, b, s in results]
        btc_r = [(p, b) for p, b, s in results if s == "btc"]
        spr_r = [(p, b) for p, b, s in results if s == "spread"]

        n = len(profits)
        wr = sum(1 for p in profits if p > 0) / n * 100
        pnl = sum(profits)
        pnl_2k = pnl / total * 2000
        bets_2k = n / total * 2000
        avg_bet = sum(bets_list) / n

        wr_btc = sum(1 for p, b in btc_r if p > 0) / max(1, len(btc_r)) * 100
        wr_spr = sum(1 for p, b in spr_r if p > 0) / max(1, len(spr_r)) * 100

        flag = " ***" if pnl > 0 else ""
        print(f"  stacked frac={pool_frac:.2f}  WR={wr:5.1f}%({n:4d}) "
              f"PnL={pnl:+7.2f} avg={avg_bet:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b) "
              f"btc={len(btc_r)}@{wr_btc:.0f}% spr={len(spr_r)}@{wr_spr:.0f}%{flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 3: 5-fold validation at cutoff=2")
    print("=" * 130)

    fold_size = total // 5
    fold_pnls = []
    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size
        fold_rounds = rounds[start:end]

        results = run_stacked(fold_rounds, btc_kl, spot_kl, pool_frac=0.25)
        btc_results = [(p, b) for p, b, s in results if s == "btc"]
        spr_results = [(p, b) for p, b, s in results if s == "spread"]
        all_profits = [p for p, b, s in results]

        n = len(results)
        wr = sum(1 for p in all_profits if p > 0) / max(1, n) * 100
        pnl = sum(all_profits)
        pnl_2k = pnl / len(fold_rounds) * 2000
        fold_pnls.append(pnl_2k)

        wr_btc = sum(1 for p, b in btc_results if p > 0) / max(1, len(btc_results)) * 100
        wr_spr = sum(1 for p, b in spr_results if p > 0) / max(1, len(spr_results)) * 100

        print(f"  Fold {fold+1}: WR={wr:5.1f}%({n:3d}) PnL={pnl:+6.2f} /2k={pnl_2k:+5.2f} "
              f"btc={len(btc_results)}@{wr_btc:.0f}% spr={len(spr_results)}@{wr_spr:.0f}%")

    avg_pnl_2k = sum(fold_pnls) / len(fold_pnls)
    min_pnl_2k = min(fold_pnls)
    max_pnl_2k = max(fold_pnls)
    print(f"\n  Avg /2k across folds: {avg_pnl_2k:+.2f} (min={min_pnl_2k:+.2f}, max={max_pnl_2k:+.2f})")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 4: Parameter sweep at cutoff=2")
    print("  Try different lookbacks and thresholds to find cutoff=2 optimum")
    print("=" * 130)

    for btc_lb in [3, 5, 7, 10]:
        for btc_thresh in [0.0003, 0.0005, 0.0007, 0.001]:
            for use_accel in [True]:
                trades = []
                for rnd in rounds:
                    lock_at = int(rnd.lock_at)
                    epoch = int(rnd.epoch)
                    hour = (lock_at % 86400) // 3600
                    if hour in SKIP_NIGHT:
                        continue
                    cutoff_ms = (lock_at - CUTOFF_S) * 1000
                    btc_raw = btc_kl.get(epoch)
                    if not btc_raw:
                        continue
                    btc_closes = get_closes(btc_raw, cutoff_ms)
                    if btc_closes is None:
                        continue
                    btc_r = _get_return(btc_closes, btc_lb)
                    if btc_r is None or abs(btc_r) < btc_thresh:
                        continue
                    signal = "Bull" if btc_r > 0 else "Bear"
                    if use_accel:
                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                            continue
                    trades.append(settle(rnd, 0.10, signal))

                n = len(trades)
                if n < 30:
                    continue
                wr = sum(1 for p in trades if p > 0) / n * 100
                pnl = sum(trades)
                pnl_2k = pnl / total * 2000
                flag = " ***" if pnl > 0 else ""
                print(f"  btc({btc_lb},{btc_thresh})+accel  WR={wr:5.1f}%({n:4d}) "
                      f"PnL={pnl:+7.2f} /2k={pnl_2k:+5.2f}{flag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
