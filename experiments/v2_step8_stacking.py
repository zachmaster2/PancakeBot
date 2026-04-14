"""Step 8: Signal stacking and aggressive sizing.

Step 7 best: accel+frac=0.15+skip_night -> +0.72 BNB/2k (68.0% WR, 19 bets/2k)
Target: 4.0 BNB/2k. Still 5.5x short.

Key leads:
1. Spread(btc7-bnb5)>=0.0007 has 69.3% WR on 75 v-bets. Independent signal?
2. Skip-night improves WR by ~2pp. What about more refined hour filtering?
3. Haven't tested frac > 0.15 with skip_night (higher WR may tolerate more sizing)
4. Pool-imbalance as signal direction (crowd is wrong?) - untested

This script:
1. Overlap analysis: BTC lead vs spread signal (how many unique rounds each?)
2. Higher pool fracs (0.18-0.30) with skip_night
3. Stacking: BTC lead signal + spread signal on non-overlapping rounds
4. Lower BTC threshold + spread filter as quality gate
5. Pool imbalance contrarian signal (bet against the crowd)
6. Adaptive: different frac for different pool sizes
7. Combined best: all viable improvements together
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

CUTOFF_S = 4
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


def main():
    rounds, spot_kl, btc_kl = load_data()
    split = int(len(rounds) * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    v_total = len(valid)
    print(f"Rounds: {len(rounds)} total, {len(train)} train, {len(valid)} validate\n")

    # =====================================================================
    print("=" * 130)
    print("PART 1: Overlap analysis -- BTC lead vs spread signal")
    print("=" * 130)

    btc_lead_epochs = set()
    spread_epochs = set()
    both_epochs = set()

    for rnd in valid:
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

        btc_r = _get_return(btc_closes, 7)
        bnb_r5 = _get_return(bnb_closes, 5)
        if btc_r is None:
            continue

        # BTC lead signal: btc(7,0.0007)+accel
        is_btc_lead = False
        if abs(btc_r) >= 0.0007:
            btc_r_short = _get_return(btc_closes, 2)
            if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                is_btc_lead = True
                btc_lead_epochs.add(epoch)

        # Spread signal: spread(btc7-bnb5) >= 0.0007
        is_spread = False
        if bnb_r5 is not None:
            spread = btc_r - bnb_r5
            if abs(spread) >= 0.0007:
                is_spread = True
                spread_epochs.add(epoch)

        if is_btc_lead and is_spread:
            both_epochs.add(epoch)

    only_btc = btc_lead_epochs - spread_epochs
    only_spread = spread_epochs - btc_lead_epochs
    print(f"  BTC lead signal rounds:  {len(btc_lead_epochs)}")
    print(f"  Spread signal rounds:    {len(spread_epochs)}")
    print(f"  Both:                    {len(both_epochs)}")
    print(f"  Only BTC lead:           {len(only_btc)}")
    print(f"  Only Spread:             {len(only_spread)}")
    print(f"  Union:                   {len(btc_lead_epochs | spread_epochs)}")

    # Check direction agreement on overlapping rounds
    agree = 0
    disagree = 0
    for rnd in valid:
        epoch = int(rnd.epoch)
        if epoch not in both_epochs:
            continue
        lock_at = int(rnd.lock_at)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000
        btc_closes = get_closes(btc_kl[epoch], cutoff_ms)
        bnb_closes = get_closes(spot_kl[epoch], cutoff_ms)
        btc_r = _get_return(btc_closes, 7)
        bnb_r5 = _get_return(bnb_closes, 5)
        btc_dir = "Bull" if btc_r > 0 else "Bear"
        spread_dir = "Bull" if (btc_r - bnb_r5) > 0 else "Bear"
        if btc_dir == spread_dir:
            agree += 1
        else:
            disagree += 1
    print(f"  Direction agreement on overlap: {agree} agree, {disagree} disagree")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 2: Higher pool fracs with skip_night")
    print("=" * 130)

    for pool_frac in [0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40]:
        for max_bet in [1.0, 2.0, 5.0]:
            t_trades, v_trades, v_bets = [], [], []
            for rnd_set, trades_out, bets_out in [
                (train, t_trades, []), (valid, v_trades, v_bets)
            ]:
                for rnd in rnd_set:
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
                    btc_r = _get_return(btc_closes, 7)
                    if btc_r is None or abs(btc_r) < 0.0007:
                        continue
                    signal = "Bull" if btc_r > 0 else "Bear"
                    btc_r_short = _get_return(btc_closes, 2)
                    if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                        continue

                    pb, pe = get_pool_at_cutoff(rnd, lock_at)
                    vis_pool = pb + pe
                    bet = max(0.01, min(max_bet, vis_pool * pool_frac))
                    profit = settle(rnd, bet, signal)
                    trades_out.append(profit)
                    bets_out.append(bet)

            nt, nv = len(t_trades), len(v_trades)
            if nt < 20 or nv < 10:
                continue
            wt = sum(1 for p in t_trades if p > 0) / nt * 100
            wv = sum(1 for p in v_trades if p > 0) / nv * 100
            pv = sum(v_trades)
            abv = sum(v_bets) / nv
            pnl_2k = pv / v_total * 2000
            bets_2k = nv / v_total * 2000
            flag = " ***" if pv > 0 else ""
            print(f"  frac={pool_frac} cap={max_bet}  "
                  f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                  f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 3: Stacking BTC lead + spread on non-overlapping rounds")
    print("  BTC lead fires first; spread fires ONLY if BTC lead didn't")
    print("=" * 130)

    for btc_thresh in [0.0007]:
        for spread_lb_bnb, spread_thresh in [(5, 0.0005), (5, 0.0007), (7, 0.0007)]:
            for pool_frac in [0.08, 0.10, 0.12, 0.15]:
                for skip_night in [False, True]:
                    t_trades, v_trades, v_bets = [], [], []
                    for rnd_set, trades_out, bets_out in [
                        (train, t_trades, []), (valid, v_trades, v_bets)
                    ]:
                        for rnd in rnd_set:
                            lock_at = int(rnd.lock_at)
                            epoch = int(rnd.epoch)
                            if skip_night:
                                hour = (lock_at % 86400) // 3600
                                if hour in SKIP_NIGHT:
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

                            # Signal 1: BTC lead + accel
                            if abs(btc_r) >= btc_thresh:
                                btc_r_short = _get_return(btc_closes, 2)
                                if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                                    signal = "Bull" if btc_r > 0 else "Bear"

                            # Signal 2: Spread (only if signal 1 didn't fire)
                            if signal is None:
                                bnb_r = _get_return(bnb_closes, spread_lb_bnb)
                                if bnb_r is not None:
                                    spread = btc_r - bnb_r
                                    if abs(spread) >= spread_thresh:
                                        signal = "Bull" if spread > 0 else "Bear"

                            if signal is None:
                                continue

                            pb, pe = get_pool_at_cutoff(rnd, lock_at)
                            vis_pool = pb + pe
                            bet = max(0.01, min(2.0, vis_pool * pool_frac))
                            profit = settle(rnd, bet, signal)
                            trades_out.append(profit)
                            bets_out.append(bet)

                    nt, nv = len(t_trades), len(v_trades)
                    if nt < 30 or nv < 15:
                        continue
                    wt = sum(1 for p in t_trades if p > 0) / nt * 100
                    wv = sum(1 for p in v_trades if p > 0) / nv * 100
                    pv = sum(v_trades)
                    abv = sum(v_bets) / nv
                    pnl_2k = pv / v_total * 2000
                    bets_2k = nv / v_total * 2000
                    night_str = "+skip_night" if skip_night else ""
                    flag = " ***" if pv > 0 else ""
                    print(f"  btc_lead+spread(bnb{spread_lb_bnb}>={spread_thresh}) "
                          f"frac={pool_frac}{night_str:12s}  "
                          f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                          f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 4: Lower BTC threshold + spread filter as quality gate")
    print("  Wider entry (btc >= 0.0005) but require spread confirmation")
    print("=" * 130)

    for btc_thresh in [0.0003, 0.0005, 0.0006]:
        for spread_thresh in [0.0003, 0.0005, 0.0007]:
            for pool_frac in [0.10, 0.15]:
                t_trades, v_trades, v_bets = [], [], []
                for rnd_set, trades_out, bets_out in [
                    (train, t_trades, []), (valid, v_trades, v_bets)
                ]:
                    for rnd in rnd_set:
                        lock_at = int(rnd.lock_at)
                        epoch = int(rnd.epoch)
                        hour = (lock_at % 86400) // 3600
                        if hour in SKIP_NIGHT:
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
                        if btc_r is None or abs(btc_r) < btc_thresh:
                            continue
                        signal = "Bull" if btc_r > 0 else "Bear"

                        # Accel
                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                            continue

                        # Spread confirmation
                        bnb_r = _get_return(bnb_closes, 5)
                        if bnb_r is None:
                            continue
                        spread = btc_r - bnb_r
                        # Spread must be in same direction as signal and above threshold
                        if signal == "Bull" and spread < spread_thresh:
                            continue
                        if signal == "Bear" and spread > -spread_thresh:
                            continue

                        pb, pe = get_pool_at_cutoff(rnd, lock_at)
                        vis_pool = pb + pe
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
                abv = sum(v_bets) / nv
                pnl_2k = pv / v_total * 2000
                bets_2k = nv / v_total * 2000
                flag = " ***" if pv > 0 else ""
                print(f"  btc>={btc_thresh}+accel+spread>={spread_thresh} frac={pool_frac}  "
                      f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                      f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 5: Pool imbalance contrarian -- bet against the crowd")
    print("  If the pool is heavily skewed, the crowd might be wrong")
    print("=" * 130)

    for imbalance_thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
        t_trades, v_trades = [], []
        for rnd_set, trades_out in [(train, t_trades), (valid, v_trades)]:
            for rnd in rnd_set:
                lock_at = int(rnd.lock_at)
                pb, pe = get_pool_at_cutoff(rnd, lock_at)
                pool_total = pb + pe
                if pool_total < 0.1:
                    continue

                bull_frac = pb / pool_total
                # If bull_frac > thresh -> crowd thinks Bull -> bet Bear (contrarian)
                # If (1-bull_frac) > thresh -> crowd thinks Bear -> bet Bull (contrarian)
                if bull_frac > imbalance_thresh:
                    signal = "Bear"
                elif (1 - bull_frac) > imbalance_thresh:
                    signal = "Bull"
                else:
                    continue

                trades_out.append(settle(rnd, 0.10, signal))

        nt, nv = len(t_trades), len(v_trades)
        if nt < 50 or nv < 20:
            print(f"  contrarian imb>{imbalance_thresh}  N too small (T={nt}, V={nv})")
            continue
        wt = sum(1 for p in t_trades if p > 0) / nt * 100
        wv = sum(1 for p in v_trades if p > 0) / nv * 100
        pv = sum(v_trades)
        pnl_2k = pv / v_total * 2000
        flag = " ***" if pv > 0 else ""
        print(f"  contrarian imb>{imbalance_thresh}  "
              f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
              f"PnL={pv:+7.2f} /2k={pnl_2k:+5.2f}{flag}")

    # And contrarian WITH BTC lead (only bet contrarian when BTC agrees)
    print("\n  Contrarian + BTC lead confirmation:")
    for imbalance_thresh in [0.55, 0.60, 0.65, 0.70]:
        t_trades, v_trades = [], []
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
                btc_r = _get_return(btc_closes, 7)
                if btc_r is None or abs(btc_r) < 0.0003:
                    continue
                btc_signal = "Bull" if btc_r > 0 else "Bear"

                pb, pe = get_pool_at_cutoff(rnd, lock_at)
                pool_total = pb + pe
                if pool_total < 0.1:
                    continue

                bull_frac = pb / pool_total
                if bull_frac > imbalance_thresh:
                    crowd_signal = "Bear"  # contrarian
                elif (1 - bull_frac) > imbalance_thresh:
                    crowd_signal = "Bull"  # contrarian
                else:
                    continue

                # Only bet if BTC and contrarian agree
                if btc_signal != crowd_signal:
                    continue

                trades_out.append(settle(rnd, 0.10, btc_signal))

        nt, nv = len(t_trades), len(v_trades)
        if nt < 20 or nv < 10:
            print(f"  btc+contrarian imb>{imbalance_thresh}  N too small (T={nt}, V={nv})")
            continue
        wt = sum(1 for p in t_trades if p > 0) / nt * 100
        wv = sum(1 for p in v_trades if p > 0) / nv * 100
        pv = sum(v_trades)
        pnl_2k = pv / v_total * 2000
        flag = " ***" if pv > 0 else ""
        print(f"  btc+contrarian imb>{imbalance_thresh}  "
              f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
              f"PnL={pv:+7.2f} /2k={pnl_2k:+5.2f}{flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 6: Adaptive sizing -- different frac for different pool sizes")
    print("  Small pool -> small frac (less dilution impact)")
    print("  Large pool -> large frac (can absorb our bet)")
    print("=" * 130)

    for small_frac, large_frac, pool_boundary in [
        (0.08, 0.20, 1.5),
        (0.08, 0.25, 1.5),
        (0.10, 0.20, 1.5),
        (0.10, 0.25, 1.5),
        (0.10, 0.30, 2.0),
        (0.12, 0.25, 2.0),
        (0.15, 0.30, 2.0),
        (0.08, 0.20, 1.0),
        (0.10, 0.25, 1.0),
    ]:
        t_trades, v_trades, v_bets = [], [], []
        for rnd_set, trades_out, bets_out in [
            (train, t_trades, []), (valid, v_trades, v_bets)
        ]:
            for rnd in rnd_set:
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
                btc_r = _get_return(btc_closes, 7)
                if btc_r is None or abs(btc_r) < 0.0007:
                    continue
                signal = "Bull" if btc_r > 0 else "Bear"
                btc_r_short = _get_return(btc_closes, 2)
                if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                    continue

                pb, pe = get_pool_at_cutoff(rnd, lock_at)
                vis_pool = pb + pe
                frac = large_frac if vis_pool >= pool_boundary else small_frac
                bet = max(0.01, min(2.0, vis_pool * frac))
                profit = settle(rnd, bet, signal)
                trades_out.append(profit)
                bets_out.append(bet)

        nt, nv = len(t_trades), len(v_trades)
        if nt < 20 or nv < 10:
            continue
        wt = sum(1 for p in t_trades if p > 0) / nt * 100
        wv = sum(1 for p in v_trades if p > 0) / nv * 100
        pv = sum(v_trades)
        abv = sum(v_bets) / nv
        pnl_2k = pv / v_total * 2000
        bets_2k = nv / v_total * 2000
        flag = " ***" if pv > 0 else ""
        print(f"  adaptive s={small_frac}/l={large_frac} @{pool_boundary}  "
              f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
              f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 7: Grand combined -- best of everything together")
    print("  BTC lead+accel + skip_night + pool-proportional + best add-ons")
    print("=" * 130)

    # Test: BTC lead as primary, spread as secondary, adaptive sizing
    for spread_as_secondary in [False, True]:
        for small_frac, large_frac, boundary in [
            (0.10, 0.20, 1.5),
            (0.12, 0.25, 1.5),
            (0.15, 0.30, 2.0),
            (0.15, 0.15, 0),   # flat frac for comparison
        ]:
            t_trades, v_trades, v_bets = [], [], []
            t_btc, t_spread = 0, 0
            v_btc, v_spread = 0, 0

            for rnd_set, trades_out, bets_out, is_valid in [
                (train, t_trades, [], False), (valid, v_trades, v_bets, True)
            ]:
                for rnd in rnd_set:
                    lock_at = int(rnd.lock_at)
                    epoch = int(rnd.epoch)
                    hour = (lock_at % 86400) // 3600
                    if hour in SKIP_NIGHT:
                        continue
                    cutoff_ms = (lock_at - CUTOFF_S) * 1000

                    btc_raw = btc_kl.get(epoch)
                    bnb_raw = spot_kl.get(epoch)
                    if not btc_raw:
                        continue
                    btc_closes = get_closes(btc_raw, cutoff_ms)
                    if btc_closes is None:
                        continue

                    btc_r = _get_return(btc_closes, 7)
                    if btc_r is None:
                        continue

                    signal = None
                    source = None

                    # Primary: BTC lead + accel
                    if abs(btc_r) >= 0.0007:
                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                            signal = "Bull" if btc_r > 0 else "Bear"
                            source = "btc"

                    # Secondary: spread (only if primary didn't fire)
                    if signal is None and spread_as_secondary and bnb_raw:
                        bnb_closes = get_closes(bnb_raw, cutoff_ms)
                        if bnb_closes is not None:
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
                    if boundary > 0:
                        frac = large_frac if vis_pool >= boundary else small_frac
                    else:
                        frac = small_frac
                    bet = max(0.01, min(2.0, vis_pool * frac))
                    profit = settle(rnd, bet, signal)
                    trades_out.append(profit)
                    bets_out.append(bet)

                    if is_valid:
                        if source == "btc":
                            v_btc += 1
                        else:
                            v_spread += 1
                    else:
                        if source == "btc":
                            t_btc += 1
                        else:
                            t_spread += 1

            nt, nv = len(t_trades), len(v_trades)
            if nt < 20 or nv < 10:
                continue
            wt = sum(1 for p in t_trades if p > 0) / nt * 100
            wv = sum(1 for p in v_trades if p > 0) / nv * 100
            pv = sum(v_trades)
            abv = sum(v_bets) / nv
            pnl_2k = pv / v_total * 2000
            bets_2k = nv / v_total * 2000
            spread_str = "+spread" if spread_as_secondary else ""
            sizing_str = f"adaptive({small_frac}/{large_frac}@{boundary})" if boundary > 0 else f"flat({small_frac})"
            flag = " ***" if pv > 0 else ""
            src = f"(btc={v_btc},spr={v_spread})" if spread_as_secondary else ""
            print(f"  {spread_str:8s} {sizing_str:25s}  "
                  f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                  f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b) "
                  f"{src}{flag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
