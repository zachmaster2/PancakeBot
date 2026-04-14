"""Step 11: Regime detection and adaptive filtering.

Full-dataset PnL is only +0.32/2k (8% of target). The strategy is regime-dependent:
fold 4 gives +3.05/2k, folds 1-3 are negative. Need to identify and exploit
favorable regimes while sitting out bad ones.

This script:
1. What makes fold 4 different? Market characteristics per fold.
2. BTC volatility as regime filter (only trade in volatile markets)
3. Rolling WR filter (sit out when recent signals are losing)
4. BTC-BNB correlation as regime filter
5. Combined regime filters
6. Alternative: higher BTC lookbacks that work in different regimes
7. Per-fold optimal parameters (is the problem param sensitivity or regime?)
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


def compute_btc_vol(btc_closes):
    """Compute BTC volatility from closes (stdev of log returns)."""
    if len(btc_closes) < 5:
        return None
    rets = []
    for i in range(1, len(btc_closes)):
        if btc_closes[i-1] > 0 and btc_closes[i] > 0:
            rets.append(math.log(btc_closes[i] / btc_closes[i-1]))
    if len(rets) < 3:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    return math.sqrt(var)


def compute_correlation(xs, ys):
    """Pearson correlation between two lists."""
    n = min(len(xs), len(ys))
    if n < 5:
        return None
    xs, ys = xs[:n], ys[:n]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx)**2 for x in xs) / n)
    sy = math.sqrt(sum((y - my)**2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def main():
    rounds, spot_kl, btc_kl = load_data()
    total = len(rounds)
    print(f"Total rounds: {total}\n")

    # =====================================================================
    print("=" * 130)
    print("PART 1: Market characteristics per fold -- what makes fold 4 special?")
    print("=" * 130)

    fold_size = total // 5
    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size
        fold_rounds = rounds[start:end]

        # Compute BTC volatility and BTC-BNB correlation for this fold
        btc_vols = []
        correlations = []

        for rnd in fold_rounds:
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

            vol = compute_btc_vol(btc_closes)
            if vol is not None:
                btc_vols.append(vol)

            # Compute correlation of returns
            btc_rets = [btc_closes[i]/btc_closes[i-1] - 1 for i in range(1, len(btc_closes))]
            bnb_rets = [bnb_closes[i]/bnb_closes[i-1] - 1 for i in range(1, len(bnb_closes))]
            corr = compute_correlation(btc_rets, bnb_rets)
            if corr is not None:
                correlations.append(corr)

        mean_vol = sum(btc_vols) / len(btc_vols) if btc_vols else 0
        mean_corr = sum(correlations) / len(correlations) if correlations else 0
        med_vol = sorted(btc_vols)[len(btc_vols)//2] if btc_vols else 0

        # Count signal fires
        n_signal = 0
        for rnd in fold_rounds:
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
            if btc_r is not None and abs(btc_r) >= 0.0007:
                n_signal += 1

        print(f"  Fold {fold+1}: mean_btc_vol={mean_vol:.6f} median_vol={med_vol:.6f} "
              f"mean_corr={mean_corr:.3f} signal_fires={n_signal}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 2: BTC volatility as regime filter")
    print("  Only trade when recent BTC vol is above threshold")
    print("=" * 130)

    # Compute per-round BTC vol for rolling window
    round_btc_vol = {}
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
        vol = compute_btc_vol(btc_closes)
        if vol is not None:
            round_btc_vol[epoch] = vol

    # Rolling vol: average of last N rounds' vol
    for vol_window in [20, 50, 100, 200]:
        for vol_thresh_pct in [40, 50, 60, 70]:
            # Compute percentile threshold from all vols
            all_vols = sorted(round_btc_vol.values())
            vol_thresh = all_vols[int(len(all_vols) * vol_thresh_pct / 100)]

            trades = []
            bets_used = []

            epoch_list = sorted(round_btc_vol.keys())
            epoch_set = set(epoch_list)
            # Build rolling vol buffer
            recent_vols = []

            for rnd in rounds:
                lock_at = int(rnd.lock_at)
                epoch = int(rnd.epoch)
                hour = (lock_at % 86400) // 3600
                if hour in SKIP_NIGHT:
                    continue

                if epoch in round_btc_vol:
                    recent_vols.append(round_btc_vol[epoch])
                    if len(recent_vols) > vol_window:
                        recent_vols.pop(0)

                if len(recent_vols) < vol_window:
                    continue

                # Check if current vol regime is active
                current_avg_vol = sum(recent_vols) / len(recent_vols)
                if current_avg_vol < vol_thresh:
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
                bet = max(0.01, min(2.0, vis_pool * 0.25))
                profit = settle(rnd, bet, signal)
                trades.append(profit)
                bets_used.append(bet)

            n = len(trades)
            if n < 20:
                continue
            wr = sum(1 for p in trades if p > 0) / n * 100
            pnl = sum(trades)
            pnl_2k = pnl / total * 2000
            bets_2k = n / total * 2000
            flag = " ***" if pnl > 0 else ""
            print(f"  vol_win={vol_window} vol_pct>={vol_thresh_pct}%  "
                  f"WR={wr:5.1f}%({n:4d}) PnL={pnl:+7.2f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 3: Rolling WR filter -- sit out when signal quality is low")
    print("=" * 130)

    for wr_window in [10, 15, 20, 30, 50]:
        for min_wr in [0.50, 0.55, 0.60, 0.65]:
            trades = []
            bets_used = []
            recent_outcomes = []  # 1=win, 0=loss
            skipped = 0

            for rnd in rounds:
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

                # Check rolling WR before deciding to bet
                if len(recent_outcomes) >= wr_window:
                    current_wr = sum(recent_outcomes[-wr_window:]) / wr_window
                    if current_wr < min_wr:
                        # Still track outcome for the rolling window
                        out = settle_bet_against_closed_round(
                            bet_bnb=0.01, bet_side=signal,
                            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
                        )
                        won = out.credit_bnb > 0.01
                        recent_outcomes.append(1 if won else 0)
                        skipped += 1
                        continue

                pb, pe = get_pool_at_cutoff(rnd, lock_at)
                vis_pool = pb + pe
                bet = max(0.01, min(2.0, vis_pool * 0.25))
                profit = settle(rnd, bet, signal)
                trades.append(profit)
                bets_used.append(bet)
                recent_outcomes.append(1 if profit > 0 else 0)

            n = len(trades)
            if n < 20:
                continue
            wr = sum(1 for p in trades if p > 0) / n * 100
            pnl = sum(trades)
            pnl_2k = pnl / total * 2000
            bets_2k = n / total * 2000
            flag = " ***" if pnl > 0 else ""
            print(f"  wr_win={wr_window:2d} min_wr={min_wr:.0%}  "
                  f"WR={wr:5.1f}%({n:4d}, skip={skipped:3d}) PnL={pnl:+7.2f} "
                  f"/2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 4: Per-round BTC vol as immediate filter")
    print("  Only trade when THIS round's BTC vol is high (not rolling)")
    print("=" * 130)

    for vol_thresh_pct in [30, 40, 50, 60, 70, 80]:
        all_vols = sorted(round_btc_vol.values())
        vol_thresh = all_vols[int(len(all_vols) * vol_thresh_pct / 100)]

        trades = []
        bets_used = []

        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            epoch = int(rnd.epoch)
            hour = (lock_at % 86400) // 3600
            if hour in SKIP_NIGHT:
                continue

            if epoch not in round_btc_vol or round_btc_vol[epoch] < vol_thresh:
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
            bet = max(0.01, min(2.0, vis_pool * 0.25))
            profit = settle(rnd, bet, signal)
            trades.append(profit)
            bets_used.append(bet)

        n = len(trades)
        if n < 20:
            continue
        wr = sum(1 for p in trades if p > 0) / n * 100
        pnl = sum(trades)
        pnl_2k = pnl / total * 2000
        bets_2k = n / total * 2000
        flag = " ***" if pnl > 0 else ""
        print(f"  vol_pct>={vol_thresh_pct}% (thresh={vol_thresh:.6f})  "
              f"WR={wr:5.1f}%({n:4d}) PnL={pnl:+7.2f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 5: Per-fold optimal BTC threshold sweep")
    print("  Is the problem param sensitivity or regime change?")
    print("=" * 130)

    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size
        fold_rounds = rounds[start:end]

        print(f"\n  Fold {fold+1}:")
        for btc_thresh in [0.0003, 0.0005, 0.0007, 0.001, 0.0015]:
            for use_accel in [True, False]:
                trades = []
                for rnd in fold_rounds:
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
                    if btc_r is None or abs(btc_r) < btc_thresh:
                        continue
                    signal = "Bull" if btc_r > 0 else "Bear"

                    if use_accel:
                        btc_r_short = _get_return(btc_closes, 2)
                        if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                            continue

                    trades.append(settle(rnd, 0.10, signal))

                n = len(trades)
                if n < 10:
                    continue
                wr = sum(1 for p in trades if p > 0) / n * 100
                pnl = sum(trades)
                pnl_2k = pnl / len(fold_rounds) * 2000
                accel_str = "+accel" if use_accel else "      "
                flag = " ***" if pnl > 0 else ""
                print(f"    btc(7,{btc_thresh}){accel_str}  WR={wr:5.1f}%({n:3d}) "
                      f"PnL={pnl:+6.2f} /2k={pnl_2k:+5.2f}{flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 6: Immediate BTC return magnitude as vol proxy")
    print("  Signal only when |btc_r| > higher threshold (more volatile moments)")
    print("=" * 130)

    for btc_thresh in [0.0007, 0.0010, 0.0012, 0.0015, 0.0020]:
        trades = []
        bets_used = []

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
            btc_r = _get_return(btc_closes, 7)
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
            bets_used.append(bet)

        n = len(trades)
        if n < 10:
            continue
        wr = sum(1 for p in trades if p > 0) / n * 100
        pnl = sum(trades)
        pnl_2k = pnl / total * 2000
        bets_2k = n / total * 2000
        flag = " ***" if pnl > 0 else ""
        print(f"  btc_thresh={btc_thresh} frac=0.25  "
              f"WR={wr:5.1f}%({n:4d}) PnL={pnl:+7.2f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
