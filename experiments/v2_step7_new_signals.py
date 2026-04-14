"""Step 7: Explore new signal dimensions beyond BTC lead.

Step 6 best: btc(7,0.0007)+accel, frac=0.15, pm>=1.5 -> +0.60 BNB/2k.
That's 15% of the 4.0 target. Need fundamentally new signals or higher WR.

Key insight from step 6: WR by final pool size varies hugely:
  1-2 BNB: 52.5% (near random!)
  2-3 BNB: 63.2%
  5-8 BNB: 76.2%
  -> Signal works much better on larger pools.

This script explores:
1. WR by VISIBLE pool (lock-6) size -- can we filter on what we actually see?
2. BNB own momentum -- does BNB spot predict BNB outcome independently?
3. BNB-BTC spread -- if BTC moved but BNB hasn't, does BNB catch up?
4. Signal WR by BTC magnitude bins -- non-linear sweet spots?
5. WR by hour of day -- time-of-day interaction with signal
6. Cross-round momentum -- previous round same direction?
7. Opposite-side ratio sizing -- bet proportional to opposing pool
8. Stacking independent signals for more bet frequency
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

CUTOFF_S = 4          # TODO: change to 2 after resync
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
    return rounds, load_kl("var/cutoff_spot_prices.jsonl"), load_kl("var/btc_spot_prices.jsonl")


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


def get_final_pool(rnd):
    bull_wei = bear_wei = 0
    for bet in rnd.bets:
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


def main():
    rounds, spot_kl, btc_kl = load_data()
    split = int(len(rounds) * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    v_total = len(valid)
    print(f"Rounds: {len(rounds)} total, {len(train)} train, {len(valid)} validate\n")

    # Build epoch->round index for cross-round lookups
    epoch_to_idx = {}
    for i, rnd in enumerate(rounds):
        epoch_to_idx[int(rnd.epoch)] = i

    # =====================================================================
    print("=" * 130)
    print("PART 1: WR by VISIBLE pool size (lock-6) -- can we filter effectively?")
    print("=" * 130)

    for label, lb, thresh in [("btc(7,0.0007)+accel", 7, 0.0007)]:
        bucket_stats = defaultdict(lambda: [0, 0, 0.0])  # [total, wins, pnl]

        for rnd in valid:
            lock_at = int(rnd.lock_at)
            epoch = int(rnd.epoch)
            cutoff_ms = (lock_at - CUTOFF_S) * 1000

            btc_raw = btc_kl.get(epoch)
            if not btc_raw:
                continue
            btc_closes = get_closes(btc_raw, cutoff_ms)
            if btc_closes is None:
                continue
            btc_r = _get_return(btc_closes, lb)
            if btc_r is None or abs(btc_r) < thresh:
                continue
            signal = "Bull" if btc_r > 0 else "Bear"

            btc_r_short = _get_return(btc_closes, 2)
            if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                continue

            # Visible pool at lock-6
            pb, pe = get_pool_at_cutoff(rnd, lock_at)
            vis_pool = pb + pe

            profit = settle(rnd, 0.10, signal)
            won = profit > 0

            if vis_pool < 0.5:
                b = "<0.5"
            elif vis_pool < 1.0:
                b = "0.5-1"
            elif vis_pool < 1.5:
                b = "1-1.5"
            elif vis_pool < 2.0:
                b = "1.5-2"
            elif vis_pool < 3.0:
                b = "2-3"
            elif vis_pool < 5.0:
                b = "3-5"
            else:
                b = "5+"

            bucket_stats[b][0] += 1
            bucket_stats[b][1] += 1 if won else 0
            bucket_stats[b][2] += profit

        print(f"\n  {label} WR by VISIBLE pool (lock-6) [validation set]:")
        print(f"  {'Vis Pool':>10s} {'N':>5s} {'WR':>6s} {'PnL':>8s} {'PnL/bet':>8s}")
        print("  " + "-" * 42)
        for b in ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-5", "5+"]:
            if bucket_stats[b][0] == 0:
                continue
            n, w, pnl = bucket_stats[b]
            wr = w / n * 100
            print(f"  {b:>10s} {n:5d} {wr:5.1f}% {pnl:+7.3f} {pnl/n:+7.4f}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 2: BNB own momentum -- does BNB spot predict outcome?")
    print("  Independent signal, NOT combined with BTC")
    print("=" * 130)

    for lb in [2, 3, 5, 7, 10, 15, 20]:
        for thresh in [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002]:
            t_trades = []
            v_trades = []

            for rnd_set, trades_out in [(train, t_trades), (valid, v_trades)]:
                for rnd in rnd_set:
                    lock_at = int(rnd.lock_at)
                    epoch = int(rnd.epoch)
                    cutoff_ms = (lock_at - CUTOFF_S) * 1000

                    bnb_raw = spot_kl.get(epoch)
                    if not bnb_raw:
                        continue
                    bnb_closes = get_closes(bnb_raw, cutoff_ms)
                    if bnb_closes is None:
                        continue

                    bnb_r = _get_return(bnb_closes, lb)
                    if bnb_r is None or abs(bnb_r) < thresh:
                        continue
                    signal = "Bull" if bnb_r > 0 else "Bear"
                    trades_out.append(settle(rnd, 0.10, signal))

            nt, nv = len(t_trades), len(v_trades)
            if nt < 30 or nv < 15:
                continue
            wt = sum(1 for p in t_trades if p > 0) / nt * 100
            wv = sum(1 for p in v_trades if p > 0) / nv * 100
            pv = sum(v_trades)
            pnl_2k = pv / v_total * 2000
            flag = " ***" if pv > 0 else ""
            print(f"  bnb({lb},{thresh:6.4f})  T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                  f"PnL={pv:+7.2f} /2k={pnl_2k:+5.2f}{flag}")
        print()

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 3: BNB-BTC spread -- bet when BTC moved but BNB hasn't caught up")
    print("  spread = btc_return - bnb_return")
    print("=" * 130)

    for btc_lb in [5, 7, 10]:
        for bnb_lb in [5, 7, 10]:
            for spread_thresh in [0.0003, 0.0005, 0.0007, 0.001, 0.0015]:
                t_trades = []
                v_trades = []

                for rnd_set, trades_out in [(train, t_trades), (valid, v_trades)]:
                    for rnd in rnd_set:
                        lock_at = int(rnd.lock_at)
                        epoch = int(rnd.epoch)
                        cutoff_ms = (lock_at - CUTOFF_S) * 1000

                        btc_raw = btc_kl.get(epoch)
                        bnb_raw = spot_kl.get(epoch)
                        if not btc_raw or not bnb_raw:
                            continue
                        btc_closes = get_closes(btc_raw, cutoff_ms)
                        bnb_closes = get_closes(bnb_raw, cutoff_ms)
                        if btc_closes is None or bnb_closes is None:
                            continue

                        btc_r = _get_return(btc_closes, btc_lb)
                        bnb_r = _get_return(bnb_closes, bnb_lb)
                        if btc_r is None or bnb_r is None:
                            continue

                        # Spread: BTC moved more than BNB -> BNB will catch up
                        spread = btc_r - bnb_r
                        if abs(spread) < spread_thresh:
                            continue

                        # Direction: if spread > 0, BTC ahead -> BNB should go Bull
                        signal = "Bull" if spread > 0 else "Bear"
                        trades_out.append(settle(rnd, 0.10, signal))

                nt, nv = len(t_trades), len(v_trades)
                if nt < 30 or nv < 15:
                    continue
                wt = sum(1 for p in t_trades if p > 0) / nt * 100
                wv = sum(1 for p in v_trades if p > 0) / nv * 100
                pv = sum(v_trades)
                pnl_2k = pv / v_total * 2000
                flag = " ***" if pv > 0 else ""
                print(f"  spread(btc{btc_lb}-bnb{bnb_lb})>={spread_thresh:6.4f}  "
                      f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                      f"PnL={pv:+7.2f} /2k={pnl_2k:+5.2f}{flag}")
        print()

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 4: BTC signal WR by magnitude -- non-linear sweet spots?")
    print("  Does very strong BTC signal have different WR than moderate?")
    print("=" * 130)

    for lb, base_thresh in [(7, 0.0007)]:
        mag_buckets = defaultdict(lambda: [0, 0])  # [total, wins]

        for rnd in valid:
            lock_at = int(rnd.lock_at)
            epoch = int(rnd.epoch)
            cutoff_ms = (lock_at - CUTOFF_S) * 1000

            btc_raw = btc_kl.get(epoch)
            if not btc_raw:
                continue
            btc_closes = get_closes(btc_raw, cutoff_ms)
            if btc_closes is None:
                continue
            btc_r = _get_return(btc_closes, lb)
            if btc_r is None or abs(btc_r) < base_thresh:
                continue
            signal = "Bull" if btc_r > 0 else "Bear"

            # Accel filter
            btc_r_short = _get_return(btc_closes, 2)
            if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                continue

            mag = abs(btc_r)
            if mag < 0.001:
                b = "0.0007-0.001"
            elif mag < 0.0015:
                b = "0.001-0.0015"
            elif mag < 0.002:
                b = "0.0015-0.002"
            elif mag < 0.003:
                b = "0.002-0.003"
            else:
                b = "0.003+"

            profit = settle(rnd, 0.10, signal)
            mag_buckets[b][0] += 1
            mag_buckets[b][1] += 1 if profit > 0 else 0

        print(f"\n  btc(7,0.0007)+accel WR by BTC return magnitude [validation]:")
        print(f"  {'|btc_r|':>15s} {'N':>5s} {'WR':>6s}")
        print("  " + "-" * 30)
        for b in ["0.0007-0.001", "0.001-0.0015", "0.0015-0.002", "0.002-0.003", "0.003+"]:
            if mag_buckets[b][0] == 0:
                continue
            n, w = mag_buckets[b]
            print(f"  {b:>15s} {n:5d} {w/n*100:5.1f}%")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 5: WR by hour of day -- when does the signal work best?")
    print("=" * 130)

    for lb, thresh in [(7, 0.0007)]:
        hour_stats = defaultdict(lambda: [0, 0, 0.0])  # [total, wins, pnl]

        for rnd in valid:
            lock_at = int(rnd.lock_at)
            epoch = int(rnd.epoch)
            cutoff_ms = (lock_at - CUTOFF_S) * 1000

            btc_raw = btc_kl.get(epoch)
            if not btc_raw:
                continue
            btc_closes = get_closes(btc_raw, cutoff_ms)
            if btc_closes is None:
                continue
            btc_r = _get_return(btc_closes, lb)
            if btc_r is None or abs(btc_r) < thresh:
                continue
            signal = "Bull" if btc_r > 0 else "Bear"

            btc_r_short = _get_return(btc_closes, 2)
            if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                continue

            hour = (lock_at % 86400) // 3600
            profit = settle(rnd, 0.10, signal)
            hour_stats[hour][0] += 1
            hour_stats[hour][1] += 1 if profit > 0 else 0
            hour_stats[hour][2] += profit

        print(f"\n  btc(7,0.0007)+accel WR by hour [validation]:")
        print(f"  {'Hour':>4s} {'N':>5s} {'WR':>6s} {'PnL':>8s}")
        print("  " + "-" * 28)
        for h in range(24):
            if hour_stats[h][0] == 0:
                continue
            n, w, pnl = hour_stats[h]
            print(f"  {h:4d} {n:5d} {w/n*100:5.1f}% {pnl:+7.3f}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 6: Cross-round momentum -- does prev round same direction help?")
    print("=" * 130)

    for lb, thresh in [(7, 0.0007)]:
        # Build per-round signal direction
        round_signals = {}
        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            epoch = int(rnd.epoch)
            cutoff_ms = (lock_at - CUTOFF_S) * 1000
            btc_raw = btc_kl.get(epoch)
            if not btc_raw:
                continue
            btc_closes = get_closes(btc_raw, cutoff_ms)
            if btc_closes is None:
                continue
            btc_r = _get_return(btc_closes, lb)
            if btc_r is not None and abs(btc_r) >= thresh:
                round_signals[epoch] = "Bull" if btc_r > 0 else "Bear"

        # Test with cross-round filter
        for require_same_dir in [False, True]:
            t_trades = []
            v_trades = []
            for rnd_set, trades_out in [(train, t_trades), (valid, v_trades)]:
                for rnd in rnd_set:
                    lock_at = int(rnd.lock_at)
                    epoch = int(rnd.epoch)
                    cutoff_ms = (lock_at - CUTOFF_S) * 1000

                    btc_raw = btc_kl.get(epoch)
                    if not btc_raw:
                        continue
                    btc_closes = get_closes(btc_raw, cutoff_ms)
                    if btc_closes is None:
                        continue
                    btc_r = _get_return(btc_closes, lb)
                    if btc_r is None or abs(btc_r) < thresh:
                        continue
                    signal = "Bull" if btc_r > 0 else "Bear"

                    btc_r_short = _get_return(btc_closes, 2)
                    if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                        continue

                    if require_same_dir:
                        prev_epoch = epoch - 1
                        prev_sig = round_signals.get(prev_epoch)
                        if prev_sig is None or prev_sig != signal:
                            continue

                    trades_out.append(settle(rnd, 0.10, signal))

            nt, nv = len(t_trades), len(v_trades)
            dir_str = "+same_prev" if require_same_dir else "(baseline)"
            if nt < 20 or nv < 10:
                print(f"  btc(7,0.0007)+accel {dir_str:15s}  N too small (T={nt}, V={nv})")
                continue
            wt = sum(1 for p in t_trades if p > 0) / nt * 100
            wv = sum(1 for p in v_trades if p > 0) / nv * 100
            pv = sum(v_trades)
            pnl_2k = pv / v_total * 2000
            flag = " ***" if pv > 0 else ""
            print(f"  btc(7,0.0007)+accel {dir_str:15s}  T={wt:5.1f}%({nt:4d}) "
                  f"V={wv:5.1f}%({nv:4d}) PnL={pv:+7.2f} /2k={pnl_2k:+5.2f}{flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 7: Opposite-side ratio sizing -- bet based on payout source")
    print("  bet = opp_side_pool * frac (our payout comes from opposing side)")
    print("=" * 130)

    for lb, thresh in [(7, 0.0007)]:
        for opp_frac in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
            t_trades = []
            v_trades = []
            v_bets = []

            for rnd_set, trades_out, bets_out in [
                (train, t_trades, []), (valid, v_trades, v_bets)
            ]:
                for rnd in rnd_set:
                    lock_at = int(rnd.lock_at)
                    epoch = int(rnd.epoch)
                    cutoff_ms = (lock_at - CUTOFF_S) * 1000

                    btc_raw = btc_kl.get(epoch)
                    if not btc_raw:
                        continue
                    btc_closes = get_closes(btc_raw, cutoff_ms)
                    if btc_closes is None:
                        continue
                    btc_r = _get_return(btc_closes, lb)
                    if btc_r is None or abs(btc_r) < thresh:
                        continue
                    signal = "Bull" if btc_r > 0 else "Bear"

                    btc_r_short = _get_return(btc_closes, 2)
                    if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                        continue

                    pb, pe = get_pool_at_cutoff(rnd, lock_at)
                    opp_pool = pe if signal == "Bull" else pb  # opposing side is our payout source
                    if opp_pool < 0.01:
                        continue
                    bet = max(0.01, min(2.0, opp_pool * opp_frac))
                    profit = settle(rnd, bet, signal)
                    trades_out.append(profit)
                    bets_out.append(bet)

            nt, nv = len(t_trades), len(v_trades)
            if nt < 30 or nv < 10:
                continue
            wt = sum(1 for p in t_trades if p > 0) / nt * 100
            wv = sum(1 for p in v_trades if p > 0) / nv * 100
            pv = sum(v_trades)
            abv = sum(v_bets) / nv if nv else 0
            pnl_2k = pv / v_total * 2000
            flag = " ***" if pv > 0 else ""
            print(f"  btc(7,0.0007)+accel opp_frac={opp_frac}  "
                  f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                  f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}{flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 8: BTC+BNB combined signal -- both agreeing")
    print("  Only bet when BTC AND BNB momentum point same direction")
    print("=" * 130)

    for btc_lb, btc_thresh in [(7, 0.0007)]:
        for bnb_lb in [3, 5, 7, 10]:
            for bnb_thresh in [0.0001, 0.0003, 0.0005, 0.0007]:
                t_trades = []
                v_trades = []

                for rnd_set, trades_out in [(train, t_trades), (valid, v_trades)]:
                    for rnd in rnd_set:
                        lock_at = int(rnd.lock_at)
                        epoch = int(rnd.epoch)
                        cutoff_ms = (lock_at - CUTOFF_S) * 1000

                        btc_raw = btc_kl.get(epoch)
                        bnb_raw = spot_kl.get(epoch)
                        if not btc_raw or not bnb_raw:
                            continue
                        btc_closes = get_closes(btc_raw, cutoff_ms)
                        bnb_closes = get_closes(bnb_raw, cutoff_ms)
                        if btc_closes is None or bnb_closes is None:
                            continue

                        btc_r = _get_return(btc_closes, btc_lb)
                        if btc_r is None or abs(btc_r) < btc_thresh:
                            continue
                        signal = "Bull" if btc_r > 0 else "Bear"

                        # Accel
                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                            continue

                        # BNB confirmation
                        bnb_r = _get_return(bnb_closes, bnb_lb)
                        if bnb_r is None or abs(bnb_r) < bnb_thresh:
                            continue
                        # Must agree with BTC direction
                        if (bnb_r > 0) != (btc_r > 0):
                            continue

                        trades_out.append(settle(rnd, 0.10, signal))

                nt, nv = len(t_trades), len(v_trades)
                if nt < 20 or nv < 10:
                    continue
                wt = sum(1 for p in t_trades if p > 0) / nt * 100
                wv = sum(1 for p in v_trades if p > 0) / nv * 100
                pv = sum(v_trades)
                pnl_2k = pv / v_total * 2000
                flag = " ***" if pv > 0 else ""
                print(f"  btc(7,0.0007)+accel+bnb({bnb_lb},{bnb_thresh})  "
                      f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                      f"PnL={pv:+7.2f} /2k={pnl_2k:+5.2f}{flag}")
        print()

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 9: Best combinations with pool-proportional sizing")
    print("  Stack best filters found above")
    print("=" * 130)

    # Test combinations of: accel + pool filter + hour filter + pool sizing
    for min_vis_pool in [0.0, 1.0, 1.5]:
        for skip_low_wr_hours in [False, True]:
            # Hours with very few bets or consistently low WR
            # (will be filled in after Part 5 results, for now test concept)
            bad_hours = set()
            if skip_low_wr_hours:
                # Tentative: skip hours where we had 0 bets or expect low WR
                # This will be refined after seeing Part 5 results
                bad_hours = {0, 1, 2, 3, 4, 23}  # late night UTC

            for pool_frac in [0.08, 0.10, 0.12, 0.15]:
                t_trades = []
                v_trades = []
                v_bets = []

                for rnd_set, trades_out, bets_out in [
                    (train, t_trades, []), (valid, v_trades, v_bets)
                ]:
                    for rnd in rnd_set:
                        lock_at = int(rnd.lock_at)
                        epoch = int(rnd.epoch)
                        cutoff_ms = (lock_at - CUTOFF_S) * 1000

                        if skip_low_wr_hours:
                            hour = (lock_at % 86400) // 3600
                            if hour in bad_hours:
                                continue

                        btc_raw = btc_kl.get(epoch)
                        if not btc_raw:
                            continue
                        btc_closes = get_closes(btc_raw, cutoff_ms)
                        if btc_closes is None:
                            continue
                        btc_r = _get_return(btc_closes, 7)
                        if btc_r is None or abs(btc_r) < 0.0007:
                            continue
                        signal = "Bull" if btc_r > 0 else "Bear"

                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                            continue

                        pb, pe = get_pool_at_cutoff(rnd, lock_at)
                        vis_pool = pb + pe
                        if vis_pool < min_vis_pool:
                            continue

                        bet = max(0.01, min(2.0, vis_pool * pool_frac))
                        profit = settle(rnd, bet, signal)
                        trades_out.append(profit)
                        bets_out.append(bet)

                nt, nv = len(t_trades), len(v_trades)
                if nt < 20 or nv < 10:
                    continue
                wt = sum(1 for p in t_trades if p > 0) / nt * 100
                wv = sum(1 for p in v_trades if p > 0) / nv * 100
                pv = sum(v_trades)
                abv = sum(v_bets) / nv if nv else 0
                pnl_2k = pv / v_total * 2000
                bets_2k = nv / v_total * 2000
                hour_str = " skip_night" if skip_low_wr_hours else ""
                flag = " ***" if pv > 0 else ""
                print(f"  accel frac={pool_frac} minvis={min_vis_pool}{hour_str:12s}  "
                      f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                      f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
