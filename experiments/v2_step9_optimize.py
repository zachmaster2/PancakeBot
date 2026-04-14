"""Step 9: Optimize stacked strategy and search for third signal.

Step 8 best: BTC lead+accel + spread(btc7-bnb5>=0.0007) + skip_night + frac=0.15
-> +1.25 BNB/2k (69.4% WR, 24 bets/2k)
Target: 4.0 BNB/2k. At 31%.

Remaining gap analysis:
- 24 bets/2k at avg 0.324 BNB = 7.8 BNB risked
- PnL 1.25 = 16% return on risked capital
- Need 4.0 = 51% return, or 3.2x more

This script:
1. WR breakdown: BTC lead vs spread signal (are both profitable?)
2. Fine-grained frac sweep on stacked strategy
3. Per-signal sizing (different frac for BTC lead vs spread)
4. Third signal: btc(15,0.001)+accel on non-overlapping rounds
5. More spread variants (different lookback combos) as extra signals
6. Kelly criterion sizing estimate
7. Best stacked strategy with all optimizations
"""
from __future__ import annotations

import json, math, sys
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
    print("PART 1: WR breakdown -- BTC lead vs spread-only rounds")
    print("=" * 130)

    for rnd_set, label in [(train, "TRAIN"), (valid, "VALID")]:
        btc_trades = []
        spread_trades = []

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
            if btc_r is None:
                continue

            # Primary: BTC lead + accel
            if abs(btc_r) >= 0.0007:
                btc_r_short = _get_return(btc_closes, 2)
                if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                    signal = "Bull" if btc_r > 0 else "Bear"
                    btc_trades.append(settle(rnd, 0.10, signal))
                    continue

            # Secondary: Spread
            bnb_r = _get_return(bnb_closes, 5)
            if bnb_r is not None:
                spread = btc_r - bnb_r
                if abs(spread) >= 0.0007:
                    signal = "Bull" if spread > 0 else "Bear"
                    spread_trades.append(settle(rnd, 0.10, signal))

        nb = len(btc_trades)
        ns = len(spread_trades)
        wb = sum(1 for p in btc_trades if p > 0) / max(1, nb) * 100
        ws = sum(1 for p in spread_trades if p > 0) / max(1, ns) * 100
        pb = sum(btc_trades)
        ps = sum(spread_trades)
        print(f"  {label:5s}: BTC lead = {wb:5.1f}% WR ({nb:4d} bets, PnL={pb:+7.2f})")
        print(f"         Spread   = {ws:5.1f}% WR ({ns:4d} bets, PnL={ps:+7.2f})")
        print(f"         Combined = {(sum(1 for p in btc_trades+spread_trades if p > 0))/(nb+ns)*100:5.1f}% WR ({nb+ns:4d} bets, PnL={pb+ps:+7.2f})")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 2: Fine-grained frac sweep on stacked strategy")
    print("=" * 130)

    for pool_frac in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
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
                if btc_r is None:
                    continue

                signal = None
                if abs(btc_r) >= 0.0007:
                    btc_r_short = _get_return(btc_closes, 2)
                    if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                        signal = "Bull" if btc_r > 0 else "Bear"

                if signal is None:
                    bnb_r = _get_return(bnb_closes, 5)
                    if bnb_r is not None:
                        spread = btc_r - bnb_r
                        if abs(spread) >= 0.0007:
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
        wt = sum(1 for p in t_trades if p > 0) / nt * 100
        wv = sum(1 for p in v_trades if p > 0) / nv * 100
        pv = sum(v_trades)
        abv = sum(v_bets) / nv
        pnl_2k = pv / v_total * 2000
        bets_2k = nv / v_total * 2000
        flag = " ***" if pv > 0 else ""
        print(f"  frac={pool_frac:.2f}  T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
              f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 3: Per-signal sizing (different frac for BTC lead vs spread)")
    print("=" * 130)

    for btc_frac in [0.12, 0.15, 0.18, 0.20, 0.22]:
        for spread_frac in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
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
                    if btc_r is None:
                        continue

                    signal = None
                    frac = 0

                    if abs(btc_r) >= 0.0007:
                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                            signal = "Bull" if btc_r > 0 else "Bear"
                            frac = btc_frac

                    if signal is None:
                        bnb_r = _get_return(bnb_closes, 5)
                        if bnb_r is not None:
                            spread = btc_r - bnb_r
                            if abs(spread) >= 0.0007:
                                signal = "Bull" if spread > 0 else "Bear"
                                frac = spread_frac

                    if signal is None:
                        continue

                    pb, pe = get_pool_at_cutoff(rnd, lock_at)
                    vis_pool = pb + pe
                    bet = max(0.01, min(2.0, vis_pool * frac))
                    profit = settle(rnd, bet, signal)
                    trades_out.append(profit)
                    bets_out.append(bet)

            nt, nv = len(t_trades), len(v_trades)
            wv = sum(1 for p in v_trades if p > 0) / nv * 100
            pv = sum(v_trades)
            abv = sum(v_bets) / nv
            pnl_2k = pv / v_total * 2000
            flag = " ***" if pv > 0 else ""
            print(f"  btc_f={btc_frac} spread_f={spread_frac}  "
                  f"V={wv:5.1f}%({nv:4d}) PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}{flag}")
        print()

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 4: Third signal -- btc(15,0.001)+accel on non-overlapping rounds")
    print("=" * 130)

    for third_lb, third_thresh in [(10, 0.0008), (10, 0.001), (15, 0.001), (15, 0.0012), (20, 0.0015)]:
        for pool_frac in [0.12, 0.15, 0.18]:
            t_trades, v_trades, v_bets = [], [], []
            v_sources = defaultdict(int)
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
                    if not btc_raw or not bnb_raw:
                        continue
                    btc_closes = get_closes(btc_raw, cutoff_ms)
                    bnb_closes = get_closes(bnb_raw, cutoff_ms)
                    if btc_closes is None or bnb_closes is None:
                        continue

                    btc_r7 = _get_return(btc_closes, 7)
                    if btc_r7 is None:
                        continue

                    signal = None
                    source = None

                    # Signal 1: BTC lead(7,0.0007) + accel
                    if abs(btc_r7) >= 0.0007:
                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is not None and (btc_r_short > 0) == (btc_r7 > 0):
                            signal = "Bull" if btc_r7 > 0 else "Bear"
                            source = "btc7"

                    # Signal 2: Spread(btc7-bnb5>=0.0007)
                    if signal is None:
                        bnb_r = _get_return(bnb_closes, 5)
                        if bnb_r is not None:
                            spread = btc_r7 - bnb_r
                            if abs(spread) >= 0.0007:
                                signal = "Bull" if spread > 0 else "Bear"
                                source = "spread"

                    # Signal 3: BTC lead(third_lb, third_thresh) + accel
                    if signal is None:
                        btc_r_third = _get_return(btc_closes, third_lb)
                        if btc_r_third is not None and abs(btc_r_third) >= third_thresh:
                            btc_r_short = _get_return(btc_closes, 2)
                            if btc_r_short is not None and (btc_r_short > 0) == (btc_r_third > 0):
                                signal = "Bull" if btc_r_third > 0 else "Bear"
                                source = "third"

                    if signal is None:
                        continue

                    pb, pe = get_pool_at_cutoff(rnd, lock_at)
                    vis_pool = pb + pe
                    bet = max(0.01, min(2.0, vis_pool * pool_frac))
                    profit = settle(rnd, bet, signal)
                    trades_out.append(profit)
                    bets_out.append(bet)
                    if is_valid:
                        v_sources[source] += 1

            nt, nv = len(t_trades), len(v_trades)
            if nt < 30 or nv < 15:
                continue
            wt = sum(1 for p in t_trades if p > 0) / nt * 100
            wv = sum(1 for p in v_trades if p > 0) / nv * 100
            pv = sum(v_trades)
            abv = sum(v_bets) / nv
            pnl_2k = pv / v_total * 2000
            bets_2k = nv / v_total * 2000
            flag = " ***" if pv > 0 else ""
            print(f"  +third({third_lb},{third_thresh}) frac={pool_frac}  "
                  f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                  f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b) "
                  f"src={dict(v_sources)}{flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 5: More spread variants as additional signals")
    print("  Try different lookback combos for spread to capture more rounds")
    print("=" * 130)

    # Baseline: btc7 + spread(btc7-bnb5>=0.0007)
    # Try adding spread variants with different lookbacks
    spread_variants = [
        (7, 5, 0.0007),   # baseline spread
        (7, 7, 0.0007),   # btc7-bnb7
        (10, 5, 0.0007),  # btc10-bnb5
        (10, 7, 0.0007),  # btc10-bnb7
        (10, 10, 0.001),  # btc10-bnb10 (higher thresh needed)
    ]

    for extra_spreads in [
        [(7, 7, 0.0007)],
        [(10, 5, 0.0007)],
        [(7, 7, 0.0007), (10, 5, 0.0007)],
        [(10, 7, 0.0007)],
        [(7, 7, 0.0007), (10, 7, 0.0007)],
    ]:
        for pool_frac in [0.12, 0.15]:
            t_trades, v_trades, v_bets = [], [], []
            v_sources = defaultdict(int)

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
                    if not btc_raw or not bnb_raw:
                        continue
                    btc_closes = get_closes(btc_raw, cutoff_ms)
                    bnb_closes = get_closes(bnb_raw, cutoff_ms)
                    if btc_closes is None or bnb_closes is None:
                        continue

                    btc_r7 = _get_return(btc_closes, 7)
                    if btc_r7 is None:
                        continue

                    signal = None
                    source = None

                    # Signal 1: BTC lead(7,0.0007) + accel
                    if abs(btc_r7) >= 0.0007:
                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is not None and (btc_r_short > 0) == (btc_r7 > 0):
                            signal = "Bull" if btc_r7 > 0 else "Bear"
                            source = "btc7"

                    # Signal 2: Primary spread (btc7-bnb5>=0.0007)
                    if signal is None:
                        bnb_r5 = _get_return(bnb_closes, 5)
                        if bnb_r5 is not None:
                            spread = btc_r7 - bnb_r5
                            if abs(spread) >= 0.0007:
                                signal = "Bull" if spread > 0 else "Bear"
                                source = "spr_75"

                    # Signal 3+: Extra spread variants
                    if signal is None:
                        for btc_lb_s, bnb_lb_s, s_thresh in extra_spreads:
                            btc_r_s = _get_return(btc_closes, btc_lb_s)
                            bnb_r_s = _get_return(bnb_closes, bnb_lb_s)
                            if btc_r_s is not None and bnb_r_s is not None:
                                s = btc_r_s - bnb_r_s
                                if abs(s) >= s_thresh:
                                    signal = "Bull" if s > 0 else "Bear"
                                    source = f"spr_{btc_lb_s}{bnb_lb_s}"
                                    break

                    if signal is None:
                        continue

                    pb, pe = get_pool_at_cutoff(rnd, lock_at)
                    vis_pool = pb + pe
                    bet = max(0.01, min(2.0, vis_pool * pool_frac))
                    profit = settle(rnd, bet, signal)
                    trades_out.append(profit)
                    bets_out.append(bet)
                    if is_valid:
                        v_sources[source] += 1

            nt, nv = len(t_trades), len(v_trades)
            if nt < 30 or nv < 15:
                continue
            wt = sum(1 for p in t_trades if p > 0) / nt * 100
            wv = sum(1 for p in v_trades if p > 0) / nv * 100
            pv = sum(v_trades)
            abv = sum(v_bets) / nv
            pnl_2k = pv / v_total * 2000
            bets_2k = nv / v_total * 2000
            extra_str = "+".join(f"spr({b},{n},{t})" for b, n, t in extra_spreads)
            flag = " ***" if pv > 0 else ""
            print(f"  +{extra_str:40s} f={pool_frac}  "
                  f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                  f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b) "
                  f"src={dict(v_sources)}{flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 6: Kelly criterion and risk-adjusted sizing")
    print("  Estimate optimal fraction from WR and payout ratio")
    print("=" * 130)

    # Run baseline stacked strategy, collect individual trade stats
    wins, losses = [], []
    for rnd in valid:
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
        if btc_r is None:
            continue

        signal = None
        if abs(btc_r) >= 0.0007:
            btc_r_short = _get_return(btc_closes, 2)
            if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                signal = "Bull" if btc_r > 0 else "Bear"

        if signal is None:
            bnb_r = _get_return(bnb_closes, 5)
            if bnb_r is not None:
                spread = btc_r - bnb_r
                if abs(spread) >= 0.0007:
                    signal = "Bull" if spread > 0 else "Bear"

        if signal is None:
            continue

        # Use fixed 0.10 bet to compute clean payout ratios
        out = settle_bet_against_closed_round(
            bet_bnb=0.10, bet_side=signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - 0.10 - GAS_COST_BET_BNB
        if profit > 0:
            wins.append(profit / 0.10)  # return ratio
        else:
            losses.append(abs(profit) / 0.10)

    w = len(wins) / (len(wins) + len(losses))
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    # Kelly: f* = (p*b - q) / b where p=win prob, q=loss prob, b=win/loss ratio
    b = avg_win / avg_loss
    kelly = (w * b - (1 - w)) / b
    half_kelly = kelly / 2

    print(f"  Win rate: {w*100:.1f}%")
    print(f"  Avg win return: {avg_win*100:.1f}%")
    print(f"  Avg loss return: {avg_loss*100:.1f}%")
    print(f"  Win/loss ratio (b): {b:.2f}")
    print(f"  Full Kelly: {kelly*100:.1f}% of bankroll")
    print(f"  Half Kelly: {half_kelly*100:.1f}% of bankroll")
    print(f"  Note: pool-frac sizing is relative to pool, not bankroll.")
    print(f"  But tells us: our edge supports aggressive sizing.")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 7: Best stacked strategy -- final optimization")
    print("  BTC lead + spread + skip_night + optimized sizing")
    print("=" * 130)

    # Also test: what if we ONLY bet on rounds where BOTH signals agree?
    # (i.e., BTC lead fires AND spread fires in same direction)
    print("\n  -- Both signals must agree (intersection) --")
    for pool_frac in [0.15, 0.20, 0.25, 0.30]:
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
                if btc_r is None:
                    continue

                # Must pass BTC lead + accel
                btc_signal = None
                if abs(btc_r) >= 0.0007:
                    btc_r_short = _get_return(btc_closes, 2)
                    if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                        btc_signal = "Bull" if btc_r > 0 else "Bear"

                if btc_signal is None:
                    continue

                # Must also pass spread
                bnb_r = _get_return(bnb_closes, 5)
                if bnb_r is None:
                    continue
                spread = btc_r - bnb_r
                if abs(spread) < 0.0005:  # lower spread thresh for intersection
                    continue
                spread_dir = "Bull" if spread > 0 else "Bear"
                if spread_dir != btc_signal:
                    continue

                pb, pe = get_pool_at_cutoff(rnd, lock_at)
                vis_pool = pb + pe
                bet = max(0.01, min(2.0, vis_pool * pool_frac))
                profit = settle(rnd, bet, btc_signal)
                trades_out.append(profit)
                bets_out.append(bet)

        nt, nv = len(t_trades), len(v_trades)
        if nt < 15 or nv < 5:
            print(f"  intersect frac={pool_frac}  N too small (T={nt}, V={nv})")
            continue
        wt = sum(1 for p in t_trades if p > 0) / nt * 100
        wv = sum(1 for p in v_trades if p > 0) / nv * 100
        pv = sum(v_trades)
        abv = sum(v_bets) / nv
        pnl_2k = pv / v_total * 2000
        bets_2k = nv / v_total * 2000
        flag = " ***" if pv > 0 else ""
        print(f"  intersect(spr>=0.0005) frac={pool_frac}  "
              f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
              f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # And intersection with higher spread threshold
    print("\n  -- Intersection with spread>=0.0003 --")
    for pool_frac in [0.15, 0.20, 0.25, 0.30]:
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
                if btc_r is None:
                    continue

                btc_signal = None
                if abs(btc_r) >= 0.0007:
                    btc_r_short = _get_return(btc_closes, 2)
                    if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                        btc_signal = "Bull" if btc_r > 0 else "Bear"

                if btc_signal is None:
                    continue

                bnb_r = _get_return(bnb_closes, 5)
                if bnb_r is None:
                    continue
                spread = btc_r - bnb_r
                if abs(spread) < 0.0003:
                    continue
                spread_dir = "Bull" if spread > 0 else "Bear"
                if spread_dir != btc_signal:
                    continue

                pb, pe = get_pool_at_cutoff(rnd, lock_at)
                vis_pool = pb + pe
                bet = max(0.01, min(2.0, vis_pool * pool_frac))
                profit = settle(rnd, bet, btc_signal)
                trades_out.append(profit)
                bets_out.append(bet)

        nt, nv = len(t_trades), len(v_trades)
        if nt < 15 or nv < 5:
            continue
        wt = sum(1 for p in t_trades if p > 0) / nt * 100
        wv = sum(1 for p in v_trades if p > 0) / nv * 100
        pv = sum(v_trades)
        abv = sum(v_bets) / nv
        pnl_2k = pv / v_total * 2000
        bets_2k = nv / v_total * 2000
        flag = " ***" if pv > 0 else ""
        print(f"  intersect(spr>=0.0003) frac={pool_frac}  "
              f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
              f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # Final: stacked with optimal per-signal sizing from Part 3
    print("\n  -- Stacked with higher btc_frac (BTC lead is more profitable) --")
    for btc_frac in [0.18, 0.20, 0.22]:
        for spread_frac in [0.10, 0.12, 0.15]:
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
                    if btc_r is None:
                        continue

                    signal = None
                    frac = 0

                    if abs(btc_r) >= 0.0007:
                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                            signal = "Bull" if btc_r > 0 else "Bear"
                            frac = btc_frac

                    if signal is None:
                        bnb_r = _get_return(bnb_closes, 5)
                        if bnb_r is not None:
                            spread = btc_r - bnb_r
                            if abs(spread) >= 0.0007:
                                signal = "Bull" if spread > 0 else "Bear"
                                frac = spread_frac

                    if signal is None:
                        continue

                    pb, pe = get_pool_at_cutoff(rnd, lock_at)
                    vis_pool = pb + pe
                    bet = max(0.01, min(2.0, vis_pool * frac))
                    profit = settle(rnd, bet, signal)
                    trades_out.append(profit)
                    bets_out.append(bet)

            nt, nv = len(t_trades), len(v_trades)
            wv = sum(1 for p in v_trades if p > 0) / nv * 100
            pv = sum(v_trades)
            abv = sum(v_bets) / nv
            pnl_2k = pv / v_total * 2000
            bets_2k = nv / v_total * 2000
            flag = " ***" if pv > 0 else ""
            print(f"  btc_f={btc_frac} spr_f={spread_frac}  "
                  f"V={wv:5.1f}%({nv:4d}) PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
