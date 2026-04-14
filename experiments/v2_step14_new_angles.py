"""Step 14: Explore fundamentally new signal dimensions.

Previous work exhausted BTC momentum, BNB momentum, spread, bet patterns.
This script explores dimensions we haven't touched:

1. VOLUME signals — volume spikes, trends, volume-weighted moves
2. Candle MICROSTRUCTURE — wick ratios, body sizes, consecutive patterns
3. MULTI-TIMEFRAME BTC — combine 3s + 7s + 15s into ensemble signal
4. BTC VOLATILITY filter — only trade in certain vol regimes
5. MEAN REVERSION — after big BTC moves, bet on reversion in BNB
6. TIME-OF-DAY interaction — which hours amplify BTC lead?
7. NON-LINEAR signal — weight by return magnitude, threshold curves
8. BTC-BNB CORRELATION filter — only trade when correlation is moderate
9. VOLUME DIVERGENCE — price up but volume down = reversal?
10. CONSECUTIVE CANDLE PATTERNS — 3+ candles same direction = momentum?
"""
from __future__ import annotations

import json, sys, math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import (
    GAS_COST_BET_BNB, INTERVAL_SECONDS, POOL_CUTOFF_SECONDS,
)
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import _trim_to_window, _get_return
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2
POOL_CUTOFF_S = POOL_CUTOFF_SECONDS
CANDLE_COUNT = 31
TREASURY_FEE = 0.03
SKIP_NIGHT = {0, 1, 2, 3, 4, 23}
BET_BNB = 0.10  # fixed small bet to avoid dilution effects


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


def get_closes(candles):
    return [k[4] for k in candles]


def get_volumes(candles):
    return [k[5] for k in candles]


def get_highs(candles):
    return [k[2] for k in candles]


def get_lows(candles):
    return [k[3] for k in candles]


def get_opens(candles):
    return [k[1] for k in candles]


def settle(rnd, side):
    out = settle_bet_against_closed_round(
        bet_bnb=BET_BNB, bet_side=side,
        round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
    )
    return out.credit_bnb - BET_BNB - GAS_COST_BET_BNB


def ret(closes, lb):
    if len(closes) < lb + 1 or closes[-(lb+1)] == 0:
        return None
    return (closes[-1] - closes[-(lb+1)]) / closes[-(lb+1)]


def print_result(label, trades, total_rounds):
    n = len(trades)
    if n < 20:
        print(f"  {label}: N={n} (too few)")
        return
    wins = sum(1 for p in trades if p > 0)
    wr = wins / n * 100
    pnl = sum(trades)
    pnl_2k = pnl / total_rounds * 2000
    flag = " ***" if pnl > 0 else ""
    print(f"  {label}: WR={wr:5.1f}%({n:4d}) PnL={pnl:+7.3f} /2k={pnl_2k:+6.3f}{flag}")


def main():
    rounds, bnb_kl, btc_kl = load_data()
    total = len(rounds)
    print(f"Total rounds: {total}\n")

    # Pre-compute candle data for all rounds
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
        btc_candles = get_candles(btc_raw, cutoff_ms)
        bnb_candles = get_candles(bnb_raw, cutoff_ms)
        if btc_candles is None or bnb_candles is None:
            continue

        data.append({
            "rnd": rnd,
            "epoch": epoch,
            "hour": hour,
            "btc_c": get_closes(btc_candles),
            "bnb_c": get_closes(bnb_candles),
            "btc_v": get_volumes(btc_candles),
            "bnb_v": get_volumes(bnb_candles),
            "btc_h": get_highs(btc_candles),
            "btc_l": get_lows(btc_candles),
            "btc_o": get_opens(btc_candles),
            "bnb_h": get_highs(bnb_candles),
            "bnb_l": get_lows(bnb_candles),
            "bnb_o": get_opens(bnb_candles),
        })

    print(f"Rounds with full data: {len(data)}\n")

    # =====================================================================
    print("=" * 120)
    print("PART 1: VOLUME SIGNALS — do volume spikes predict direction?")
    print("=" * 120)

    # 1a: Volume spike + direction agreement
    for vol_lb in [3, 5, 7]:
        for vol_mult in [1.5, 2.0, 3.0]:
            trades = []
            for d in data:
                vols = d["btc_v"]
                closes = d["btc_c"]
                if len(vols) < vol_lb + 1:
                    continue
                recent_vol = vols[-1]
                avg_vol = sum(vols[-(vol_lb+1):-1]) / vol_lb
                if avg_vol == 0:
                    continue
                if recent_vol < avg_vol * vol_mult:
                    continue
                # Volume spike detected — which direction did price move?
                r = ret(closes, 1)
                if r is None or r == 0:
                    continue
                signal = "Bull" if r > 0 else "Bear"
                trades.append(settle(d["rnd"], signal))
            print_result(f"vol_spike(lb={vol_lb},mult={vol_mult}x)+dir", trades, total)

    # 1b: Volume trend (increasing vol = momentum)
    for vol_lb in [5, 10]:
        trades = []
        for d in data:
            vols = d["btc_v"]
            closes = d["btc_c"]
            if len(vols) < vol_lb:
                continue
            recent_vols = vols[-vol_lb:]
            # Is volume increasing? Compare first half to second half
            half = vol_lb // 2
            first_avg = sum(recent_vols[:half]) / half
            second_avg = sum(recent_vols[half:]) / (vol_lb - half)
            if first_avg == 0:
                continue
            vol_trend = second_avg / first_avg
            if vol_trend < 1.5:  # volume not increasing enough
                continue
            btc_r = ret(closes, vol_lb)
            if btc_r is None or abs(btc_r) < 0.0003:
                continue
            signal = "Bull" if btc_r > 0 else "Bear"
            trades.append(settle(d["rnd"], signal))
        print_result(f"vol_trend(lb={vol_lb},trend>1.5x)+btc_dir", trades, total)

    # 1c: Volume-weighted return
    for lb in [5, 7, 10]:
        for thresh in [0.0005, 0.0007, 0.001]:
            trades = []
            for d in data:
                closes = d["btc_c"]
                vols = d["btc_v"]
                if len(closes) < lb + 1:
                    continue
                # Volume-weighted return
                total_vol = sum(vols[-lb:])
                if total_vol == 0:
                    continue
                vw_ret = 0
                for i in range(lb):
                    idx = -(lb - i)
                    if closes[idx - 1] == 0:
                        continue
                    r_i = (closes[idx] - closes[idx - 1]) / closes[idx - 1]
                    vw_ret += r_i * (vols[idx] / total_vol)
                if abs(vw_ret) < thresh:
                    continue
                signal = "Bull" if vw_ret > 0 else "Bear"
                trades.append(settle(d["rnd"], signal))
            print_result(f"vwap_ret(lb={lb},t={thresh})", trades, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 2: CANDLE MICROSTRUCTURE — wicks, body ratios, patterns")
    print("=" * 120)

    # 2a: Large wick ratio (rejection signal)
    for lb in [3, 5]:
        for wick_ratio in [0.6, 0.7, 0.8]:
            trades_follow = []
            trades_fade = []
            for d in data:
                highs = d["btc_h"][-lb:]
                lows = d["btc_l"][-lb:]
                opens = d["btc_o"][-lb:]
                closes = d["btc_c"][-lb:]

                # Average wick ratio over lookback
                total_wick_ratio = 0
                count = 0
                direction = 0
                for h, l, o, c in zip(highs, lows, opens, closes):
                    rng = h - l
                    if rng == 0:
                        continue
                    body = abs(c - o)
                    wick = rng - body
                    total_wick_ratio += wick / rng
                    direction += 1 if c > o else -1
                    count += 1
                if count == 0:
                    continue
                avg_wick = total_wick_ratio / count
                if avg_wick < wick_ratio:
                    continue
                if direction == 0:
                    continue
                # High wick = rejection. Follow or fade the direction?
                signal_follow = "Bull" if direction > 0 else "Bear"
                signal_fade = "Bear" if direction > 0 else "Bull"
                trades_follow.append(settle(d["rnd"], signal_follow))
                trades_fade.append(settle(d["rnd"], signal_fade))
            print_result(f"wick_ratio(lb={lb},wr>{wick_ratio}) follow", trades_follow, total)
            print_result(f"wick_ratio(lb={lb},wr>{wick_ratio}) fade", trades_fade, total)

    # 2b: Consecutive same-direction candles
    for consec in [3, 4, 5, 6, 7]:
        trades = []
        for d in data:
            closes = d["btc_c"]
            if len(closes) < consec + 1:
                continue
            # Check last N candles all same direction
            dirs = []
            for i in range(consec):
                idx = -(consec - i)
                if closes[idx] > closes[idx - 1]:
                    dirs.append(1)
                elif closes[idx] < closes[idx - 1]:
                    dirs.append(-1)
                else:
                    dirs.append(0)
            if 0 in dirs:
                continue
            if len(set(dirs)) > 1:
                continue
            signal = "Bull" if dirs[0] > 0 else "Bear"
            trades.append(settle(d["rnd"], signal))
        print_result(f"consec_{consec}_candles_btc", trades, total)

    # 2c: Body size (large body = strong momentum)
    for lb in [3, 5]:
        for body_mult in [1.5, 2.0, 3.0]:
            trades = []
            for d in data:
                opens = d["btc_o"]
                closes = d["btc_c"]
                if len(opens) < lb + 5:
                    continue
                # Recent body vs baseline body
                recent_bodies = [abs(closes[-(i+1)] - opens[-(i+1)]) for i in range(lb)]
                baseline_bodies = [abs(closes[-(i+lb+1)] - opens[-(i+lb+1)]) for i in range(5)]
                avg_recent = sum(recent_bodies) / lb
                avg_baseline = sum(baseline_bodies) / 5
                if avg_baseline == 0:
                    continue
                if avg_recent / avg_baseline < body_mult:
                    continue
                # Direction of recent candles
                r = ret(d["btc_c"], lb)
                if r is None or r == 0:
                    continue
                signal = "Bull" if r > 0 else "Bear"
                trades.append(settle(d["rnd"], signal))
            print_result(f"body_size(lb={lb},mult>{body_mult}x)", trades, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 3: MULTI-TIMEFRAME BTC — combine lookbacks for stronger signal")
    print("=" * 120)

    # 3a: Agreement across multiple timeframes
    for combo in [(3, 7), (3, 7, 15), (5, 10, 20), (3, 5, 7, 10)]:
        for thresh in [0.0003, 0.0005, 0.0007]:
            trades = []
            for d in data:
                closes = d["btc_c"]
                signals = []
                for lb in combo:
                    r = ret(closes, lb)
                    if r is None or abs(r) < thresh:
                        signals.append(0)
                    else:
                        signals.append(1 if r > 0 else -1)
                # All must agree and be non-zero
                non_zero = [s for s in signals if s != 0]
                if len(non_zero) < len(combo):
                    continue
                if len(set(non_zero)) > 1:
                    continue
                signal = "Bull" if non_zero[0] > 0 else "Bear"
                trades.append(settle(d["rnd"], signal))
            label = "+".join(str(lb) for lb in combo)
            print_result(f"multi_tf({label},t={thresh})", trades, total)

    # 3b: Weighted multi-timeframe (more weight to shorter)
    for combo, weights in [((3,7,15), (3,2,1)), ((5,10,20), (3,2,1))]:
        for thresh in [0.0005, 0.001, 0.002]:
            trades = []
            for d in data:
                closes = d["btc_c"]
                weighted_sum = 0
                total_weight = 0
                valid = True
                for lb, w in zip(combo, weights):
                    r = ret(closes, lb)
                    if r is None:
                        valid = False
                        break
                    weighted_sum += r * w
                    total_weight += w
                if not valid or total_weight == 0:
                    continue
                composite = weighted_sum / total_weight
                if abs(composite) < thresh:
                    continue
                signal = "Bull" if composite > 0 else "Bear"
                trades.append(settle(d["rnd"], signal))
            label = "+".join(f"{lb}x{w}" for lb, w in zip(combo, weights))
            print_result(f"weighted_tf({label},t={thresh})", trades, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 4: VOLATILITY FILTER — only trade in certain vol regimes")
    print("=" * 120)

    for vol_window in [15, 20, 30]:
        # Compute vol for each round
        vol_data = []
        for d in data:
            closes = d["btc_c"]
            if len(closes) < vol_window:
                continue
            rets = []
            for i in range(1, vol_window):
                if closes[-(i+1)] == 0:
                    continue
                rets.append((closes[-i] - closes[-(i+1)]) / closes[-(i+1)])
            if len(rets) < vol_window // 2:
                continue
            vol = (sum(r*r for r in rets) / len(rets)) ** 0.5
            vol_data.append((d, vol))

        if not vol_data:
            continue

        # Sort by vol to find percentiles
        vols_sorted = sorted(v for _, v in vol_data)
        p25 = vols_sorted[len(vols_sorted) // 4]
        p50 = vols_sorted[len(vols_sorted) // 2]
        p75 = vols_sorted[3 * len(vols_sorted) // 4]

        for vol_range, lo, hi in [("low_vol", 0, p25), ("med_vol", p25, p75),
                                   ("high_vol", p75, 1.0)]:
            trades = []
            for d, vol in vol_data:
                if vol < lo or vol >= hi:
                    continue
                btc_r = ret(d["btc_c"], 7)
                if btc_r is None or abs(btc_r) < 0.0007:
                    continue
                btc_r2 = ret(d["btc_c"], 2)
                if btc_r2 is None or (btc_r2 > 0) != (btc_r > 0):
                    continue
                signal = "Bull" if btc_r > 0 else "Bear"
                trades.append(settle(d["rnd"], signal))
            print_result(f"btc_lead+vol_filter(w={vol_window},{vol_range})", trades, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 5: MEAN REVERSION — after big BTC move, bet on BNB reversion")
    print("=" * 120)

    for lb in [10, 15, 20, 25]:
        for thresh in [0.001, 0.0015, 0.002, 0.003]:
            trades = []
            for d in data:
                btc_r = ret(d["btc_c"], lb)
                if btc_r is None or abs(btc_r) < thresh:
                    continue
                # Bet AGAINST the big BTC move (mean reversion)
                signal = "Bear" if btc_r > 0 else "Bull"
                trades.append(settle(d["rnd"], signal))
            print_result(f"mean_revert(btc_lb={lb},t={thresh})", trades, total)

    # Also: BNB has moved less than BTC — bet BNB catches up
    for btc_lb in [7, 10, 15]:
        for bnb_lb in [3, 5]:
            for thresh in [0.0005, 0.001]:
                trades = []
                for d in data:
                    btc_r = ret(d["btc_c"], btc_lb)
                    bnb_r = ret(d["bnb_c"], bnb_lb)
                    if btc_r is None or bnb_r is None:
                        continue
                    # BTC moved big, BNB hasn't caught up yet
                    gap = btc_r - bnb_r
                    if abs(gap) < thresh:
                        continue
                    if abs(btc_r) < 0.0005:
                        continue
                    # Bet BNB will catch up to BTC direction
                    signal = "Bull" if gap > 0 else "Bear"
                    trades.append(settle(d["rnd"], signal))
                print_result(f"catchup(btc{btc_lb}-bnb{bnb_lb},t={thresh})", trades, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 6: TIME-OF-DAY INTERACTION — which hours amplify BTC lead?")
    print("=" * 120)

    hour_trades = defaultdict(list)
    for d in data:
        btc_r = ret(d["btc_c"], 7)
        if btc_r is None or abs(btc_r) < 0.0007:
            continue
        btc_r2 = ret(d["btc_c"], 2)
        if btc_r2 is None or (btc_r2 > 0) != (btc_r > 0):
            continue
        signal = "Bull" if btc_r > 0 else "Bear"
        profit = settle(d["rnd"], signal)
        hour_trades[d["hour"]].append(profit)

    for h in sorted(hour_trades.keys()):
        trades = hour_trades[h]
        print_result(f"btc_lead@hour={h:02d}", trades, total)

    # Best/worst hour groups
    hour_wr = {}
    for h, trades in hour_trades.items():
        if len(trades) >= 10:
            hour_wr[h] = sum(1 for p in trades if p > 0) / len(trades)
    if hour_wr:
        best_hours = sorted(hour_wr, key=hour_wr.get, reverse=True)[:6]
        trades_best = []
        for d in data:
            if d["hour"] not in best_hours:
                continue
            btc_r = ret(d["btc_c"], 7)
            if btc_r is None or abs(btc_r) < 0.0007:
                continue
            btc_r2 = ret(d["btc_c"], 2)
            if btc_r2 is None or (btc_r2 > 0) != (btc_r > 0):
                continue
            signal = "Bull" if btc_r > 0 else "Bear"
            trades_best.append(settle(d["rnd"], signal))
        print_result(f"btc_lead@best_hours={best_hours}", trades_best, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 7: BTC-BNB CORRELATION FILTER — moderate correlation = signal works")
    print("=" * 120)

    # Rolling correlation over last N candles
    for corr_window in [20, 25, 30]:
        corr_data = []
        for d in data:
            btc_c = d["btc_c"]
            bnb_c = d["bnb_c"]
            if len(btc_c) < corr_window or len(bnb_c) < corr_window:
                continue
            # Compute returns
            btc_rets = [(btc_c[i] - btc_c[i-1]) / btc_c[i-1]
                        for i in range(-corr_window+1, 0) if btc_c[i-1] != 0]
            bnb_rets = [(bnb_c[i] - bnb_c[i-1]) / bnb_c[i-1]
                        for i in range(-corr_window+1, 0) if bnb_c[i-1] != 0]
            if len(btc_rets) != len(bnb_rets) or len(btc_rets) < corr_window // 2:
                continue
            n = len(btc_rets)
            mean_b = sum(btc_rets) / n
            mean_n = sum(bnb_rets) / n
            cov = sum((btc_rets[i] - mean_b) * (bnb_rets[i] - mean_n) for i in range(n)) / n
            std_b = (sum((r - mean_b)**2 for r in btc_rets) / n) ** 0.5
            std_n = (sum((r - mean_n)**2 for r in bnb_rets) / n) ** 0.5
            if std_b == 0 or std_n == 0:
                continue
            corr = cov / (std_b * std_n)
            corr_data.append((d, corr))

        if not corr_data:
            continue

        corrs_sorted = sorted(c for _, c in corr_data)
        p33 = corrs_sorted[len(corrs_sorted) // 3]
        p67 = corrs_sorted[2 * len(corrs_sorted) // 3]

        for label, lo, hi in [("low_corr", -1, p33), ("med_corr", p33, p67),
                               ("high_corr", p67, 1)]:
            trades = []
            for d, corr in corr_data:
                if corr < lo or corr >= hi:
                    continue
                btc_r = ret(d["btc_c"], 7)
                if btc_r is None or abs(btc_r) < 0.0007:
                    continue
                btc_r2 = ret(d["btc_c"], 2)
                if btc_r2 is None or (btc_r2 > 0) != (btc_r > 0):
                    continue
                signal = "Bull" if btc_r > 0 else "Bear"
                trades.append(settle(d["rnd"], signal))
            print_result(f"btc_lead+corr_filter(w={corr_window},{label})", trades, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 8: VOLUME DIVERGENCE — price up + volume down = weak, fade it?")
    print("=" * 120)

    for price_lb in [5, 7, 10]:
        for vol_lb in [5, 7]:
            trades_follow = []
            trades_fade = []
            for d in data:
                price_r = ret(d["btc_c"], price_lb)
                if price_r is None or abs(price_r) < 0.0005:
                    continue
                vols = d["btc_v"]
                if len(vols) < vol_lb + 1:
                    continue
                vol_now = sum(vols[-vol_lb:])
                vol_prev = sum(vols[-(vol_lb*2):-vol_lb])
                if vol_prev == 0:
                    continue
                vol_change = vol_now / vol_prev

                # Divergence: big price move but declining volume
                if vol_change >= 1.0:
                    continue  # volume confirming — skip
                # Volume declining while price moving — divergence
                signal_follow = "Bull" if price_r > 0 else "Bear"
                signal_fade = "Bear" if price_r > 0 else "Bull"
                trades_follow.append(settle(d["rnd"], signal_follow))
                trades_fade.append(settle(d["rnd"], signal_fade))
            print_result(f"vol_diverge(p={price_lb},v={vol_lb}) follow", trades_follow, total)
            print_result(f"vol_diverge(p={price_lb},v={vol_lb}) fade", trades_fade, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 9: BTC RANGE / ATR — use range instead of close-close return")
    print("=" * 120)

    for lb in [5, 7, 10]:
        for thresh in [0.0005, 0.001, 0.0015]:
            trades = []
            for d in data:
                highs = d["btc_h"][-lb:]
                lows = d["btc_l"][-lb:]
                closes = d["btc_c"]
                if len(closes) < lb + 1 or closes[-(lb+1)] == 0:
                    continue
                # Range-based: high of last N minus low of last N
                range_high = max(highs)
                range_low = min(lows)
                mid = closes[-(lb+1)]
                if mid == 0:
                    continue
                # Direction: is close near top or bottom of range?
                rng = range_high - range_low
                if rng / mid < thresh:
                    continue  # range too small
                position = (closes[-1] - range_low) / rng  # 0=bottom, 1=top
                if position > 0.7:
                    signal = "Bull"
                elif position < 0.3:
                    signal = "Bear"
                else:
                    continue
                trades.append(settle(d["rnd"], signal))
            print_result(f"range_position(lb={lb},t={thresh},>0.7/<0.3)", trades, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 10: ETH-PROXY via BTC — BTC return squared (non-linear)")
    print("=" * 120)

    # Non-linear: larger BTC moves might be proportionally more predictive
    for lb in [5, 7, 10]:
        for thresh in [0.0005, 0.0007, 0.001]:
            # Bin by magnitude
            bins = defaultdict(list)
            for d in data:
                btc_r = ret(d["btc_c"], lb)
                if btc_r is None:
                    continue
                abs_r = abs(btc_r)
                if abs_r < thresh:
                    continue
                # Bin: small, medium, large
                if abs_r < thresh * 2:
                    bname = "small"
                elif abs_r < thresh * 4:
                    bname = "medium"
                else:
                    bname = "large"
                signal = "Bull" if btc_r > 0 else "Bear"
                bins[bname].append(settle(d["rnd"], signal))
            for bname in ["small", "medium", "large"]:
                if bname in bins:
                    print_result(f"btc_mag(lb={lb},t={thresh},{bname})", bins[bname], total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 11: COMBINED VOLUME + MOMENTUM — volume confirms momentum?")
    print("=" * 120)

    for btc_lb in [5, 7]:
        for btc_thresh in [0.0005, 0.0007]:
            for vol_confirm in [True, False]:
                trades = []
                for d in data:
                    btc_r = ret(d["btc_c"], btc_lb)
                    if btc_r is None or abs(btc_r) < btc_thresh:
                        continue
                    btc_r2 = ret(d["btc_c"], 2)
                    if btc_r2 is None or (btc_r2 > 0) != (btc_r > 0):
                        continue

                    if vol_confirm:
                        # Volume should be above average
                        vols = d["btc_v"]
                        if len(vols) < btc_lb + 5:
                            continue
                        recent_vol = sum(vols[-btc_lb:]) / btc_lb
                        baseline_vol = sum(vols[-(btc_lb+5):-btc_lb]) / 5
                        if baseline_vol == 0:
                            continue
                        if recent_vol < baseline_vol * 1.2:
                            continue

                    signal = "Bull" if btc_r > 0 else "Bear"
                    trades.append(settle(d["rnd"], signal))
                vc = "+vol_confirm" if vol_confirm else ""
                print_result(f"btc({btc_lb},{btc_thresh})+accel{vc}", trades, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 12: BNB VOLUME SIGNAL — BNB volume spike independent of BTC")
    print("=" * 120)

    for vol_lb in [3, 5, 7]:
        for vol_mult in [1.5, 2.0, 3.0]:
            trades = []
            for d in data:
                vols = d["bnb_v"]
                closes = d["bnb_c"]
                if len(vols) < vol_lb + 3:
                    continue
                recent_vol = sum(vols[-vol_lb:]) / vol_lb
                baseline_vol = sum(vols[-(vol_lb+3):-vol_lb]) / 3
                if baseline_vol == 0:
                    continue
                if recent_vol < baseline_vol * vol_mult:
                    continue
                r = ret(closes, vol_lb)
                if r is None or r == 0:
                    continue
                signal = "Bull" if r > 0 else "Bear"
                trades.append(settle(d["rnd"], signal))
            print_result(f"bnb_vol(lb={vol_lb},mult>{vol_mult}x)+dir", trades, total)

    print("\nDone.")


if __name__ == "__main__":
    main()
