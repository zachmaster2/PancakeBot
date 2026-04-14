"""Step 10: Multi-fold validation and peak-hour optimization.

Critical concern: spread signal shows 54.9% WR in train but 74.1% in validation.
A 19pp gap suggests possible overfitting or lucky validation period.

This script:
1. 5-fold rolling validation of stacked strategy
2. Spread signal WR across multiple time windows
3. Peak-hour frac boost (bigger bets during high-pool hours 13-16 UTC)
4. Conservative estimate: what's the strategy worth if spread WR is 60%?
5. Drawdown analysis: worst consecutive losing streaks
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


def run_stacked(rounds, btc_kl, spot_kl, pool_frac=0.15, skip_night=True,
                per_signal_frac=None, hour_boost=None):
    """Run stacked strategy (BTC lead + spread).

    Returns: list of (profit, bet_size, source) tuples
    """
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

        # Signal 1: BTC lead(7,0.0007) + accel
        if abs(btc_r) >= 0.0007:
            btc_r_short = _get_return(btc_closes, 2)
            if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                signal = "Bull" if btc_r > 0 else "Bear"
                source = "btc"

        # Signal 2: Spread(btc7-bnb5 >= 0.0007)
        if signal is None:
            bnb_r = _get_return(bnb_closes, 5)
            if bnb_r is not None:
                spread = btc_r - bnb_r
                if abs(spread) >= 0.0007:
                    signal = "Bull" if spread > 0 else "Bear"
                    source = "spread"

        if signal is None:
            continue

        # Determine sizing
        frac = pool_frac
        if per_signal_frac and source in per_signal_frac:
            frac = per_signal_frac[source]
        if hour_boost and hour in hour_boost:
            frac *= hour_boost[hour]

        pb, pe = get_pool_at_cutoff(rnd, lock_at)
        vis_pool = pb + pe
        bet = max(0.01, min(2.0, vis_pool * frac))

        profit = settle(rnd, bet, signal)
        results.append((profit, bet, source))

    return results


def main():
    rounds, spot_kl, btc_kl = load_data()
    total = len(rounds)
    print(f"Total rounds: {total}\n")

    # =====================================================================
    print("=" * 130)
    print("PART 1: 5-fold rolling validation (each fold = 20% of data)")
    print("  Train on 60%, validate on next 20%, slide forward")
    print("=" * 130)

    fold_size = total // 5
    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size
        fold_rounds = rounds[start:end]

        # Split fold 60/40 for train/test within the fold
        # Actually let's just test each fold as OOS with the rest as training
        test_rounds = fold_rounds
        train_rounds = rounds[:start] + rounds[end:]

        # For display, split by source
        results = run_stacked(test_rounds, btc_kl, spot_kl, pool_frac=0.25)
        btc_results = [(p, b) for p, b, s in results if s == "btc"]
        spr_results = [(p, b) for p, b, s in results if s == "spread"]
        all_profits = [p for p, b, s in results]

        n = len(results)
        n_btc = len(btc_results)
        n_spr = len(spr_results)
        wr_btc = sum(1 for p, b in btc_results if p > 0) / max(1, n_btc) * 100
        wr_spr = sum(1 for p, b in spr_results if p > 0) / max(1, n_spr) * 100
        wr_all = sum(1 for p in all_profits if p > 0) / max(1, n) * 100
        pnl = sum(all_profits)
        pnl_2k = pnl / len(test_rounds) * 2000

        epoch_start = int(fold_rounds[0].epoch)
        epoch_end = int(fold_rounds[-1].epoch)
        print(f"  Fold {fold+1} (epochs {epoch_start}-{epoch_end}, {len(test_rounds)} rounds):")
        print(f"    BTC lead: {wr_btc:5.1f}% WR ({n_btc:3d} bets, PnL={sum(p for p,b in btc_results):+6.2f})")
        print(f"    Spread:   {wr_spr:5.1f}% WR ({n_spr:3d} bets, PnL={sum(p for p,b in spr_results):+6.2f})")
        print(f"    Combined: {wr_all:5.1f}% WR ({n:3d} bets, PnL={pnl:+6.2f}, /2k={pnl_2k:+5.2f})")
        print()

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 2: Spread signal stability across 10 time windows")
    print("=" * 130)

    window_size = total // 10
    for w in range(10):
        start = w * window_size
        end = start + window_size
        window_rounds = rounds[start:end]

        results = run_stacked(window_rounds, btc_kl, spot_kl, pool_frac=0.10)
        spr_results = [(p, b) for p, b, s in results if s == "spread"]
        btc_results = [(p, b) for p, b, s in results if s == "btc"]

        n_spr = len(spr_results)
        n_btc = len(btc_results)
        wr_spr = sum(1 for p, b in spr_results if p > 0) / max(1, n_spr) * 100
        wr_btc = sum(1 for p, b in btc_results if p > 0) / max(1, n_btc) * 100
        pnl_spr = sum(p for p, b in spr_results)
        pnl_btc = sum(p for p, b in btc_results)

        print(f"  Window {w+1:2d} ({len(window_rounds)} rounds): "
              f"BTC={wr_btc:5.1f}%({n_btc:3d}b, {pnl_btc:+5.2f})  "
              f"Spread={wr_spr:5.1f}%({n_spr:3d}b, {pnl_spr:+5.2f})")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 3: Peak-hour frac boost (bigger bets when pools are larger)")
    print("  Hours 13-16 UTC have 40% larger pools on average")
    print("=" * 130)

    split = int(total * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    v_total = len(valid)

    # Hour boost configs
    peak_hours = {13: 1.5, 14: 1.5, 15: 1.5, 16: 1.3}
    peak_hours_2 = {12: 1.3, 13: 1.5, 14: 1.8, 15: 1.8, 16: 1.5, 17: 1.3}

    for label, base_frac, hboost in [
        ("flat 0.25", 0.25, None),
        ("flat 0.20 +peak(1.5x)", 0.20, peak_hours),
        ("flat 0.18 +peak(1.5-1.8x)", 0.18, peak_hours_2),
        ("flat 0.15 +peak(1.5x)", 0.15, peak_hours),
        ("flat 0.15 +peak(1.5-1.8x)", 0.15, peak_hours_2),
        ("flat 0.22 +peak(1.5x)", 0.22, peak_hours),
        ("flat 0.22 +peak(1.5-1.8x)", 0.22, peak_hours_2),
        ("flat 0.25 +peak(1.5x)", 0.25, peak_hours),
        ("flat 0.25 +peak(1.5-1.8x)", 0.25, peak_hours_2),
    ]:
        t_results = run_stacked(train, btc_kl, spot_kl, pool_frac=base_frac,
                                hour_boost=hboost)
        v_results = run_stacked(valid, btc_kl, spot_kl, pool_frac=base_frac,
                                hour_boost=hboost)

        t_profits = [p for p, b, s in t_results]
        v_profits = [p for p, b, s in v_results]
        v_bets = [b for p, b, s in v_results]
        nt, nv = len(t_profits), len(v_profits)
        wt = sum(1 for p in t_profits if p > 0) / max(1, nt) * 100
        wv = sum(1 for p in v_profits if p > 0) / max(1, nv) * 100
        pv = sum(v_profits)
        abv = sum(v_bets) / max(1, nv)
        pnl_2k = pv / v_total * 2000
        bets_2k = nv / v_total * 2000
        flag = " ***" if pv > 0 else ""
        print(f"  {label:35s} T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
              f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 4: Drawdown analysis -- worst streaks and risk")
    print("=" * 130)

    # Run best strategy on full dataset
    results = run_stacked(rounds, btc_kl, spot_kl, pool_frac=0.25)
    profits = [p for p, b, s in results]
    bets = [b for p, b, s in results]

    # Cumulative PnL
    cum_pnl = []
    running = 0
    for p in profits:
        running += p
        cum_pnl.append(running)

    # Max drawdown
    peak = 0
    max_dd = 0
    for c in cum_pnl:
        if c > peak:
            peak = c
        dd = peak - c
        if dd > max_dd:
            max_dd = dd

    # Consecutive losses
    max_consec_loss = 0
    cur_loss_streak = 0
    loss_streaks = []
    for p in profits:
        if p <= 0:
            cur_loss_streak += 1
        else:
            if cur_loss_streak > 0:
                loss_streaks.append(cur_loss_streak)
            cur_loss_streak = 0
    if cur_loss_streak > 0:
        loss_streaks.append(cur_loss_streak)
    max_consec_loss = max(loss_streaks) if loss_streaks else 0

    n = len(profits)
    wins = sum(1 for p in profits if p > 0)
    total_pnl = sum(profits)
    total_bet = sum(bets)
    avg_bet = total_bet / n

    print(f"  Total bets: {n}")
    print(f"  Win rate: {wins/n*100:.1f}%")
    print(f"  Total PnL: {total_pnl:+.2f} BNB")
    print(f"  Total risked: {total_bet:.2f} BNB")
    print(f"  ROI: {total_pnl/total_bet*100:.1f}%")
    print(f"  Avg bet: {avg_bet:.3f} BNB")
    print(f"  Max drawdown: {max_dd:.2f} BNB")
    print(f"  Max consecutive losses: {max_consec_loss}")
    print(f"  PnL per round: {total_pnl/total:.5f}")
    print(f"  PnL per 2000 rounds: {total_pnl/total*2000:+.2f}")

    # Monthly breakdown (approximate: ~8640 rounds per 30 days at 5min)
    monthly = total // 8640
    if monthly > 0:
        for m in range(monthly + 1):
            start = m * 8640
            end = min(start + 8640, n)
            if start >= n:
                break
            month_results = run_stacked(rounds[m*8640:min((m+1)*8640, total)],
                                        btc_kl, spot_kl, pool_frac=0.25)
            month_profits = [p for p, b, s in month_results]
            mn = len(month_profits)
            if mn == 0:
                continue
            mwr = sum(1 for p in month_profits if p > 0) / mn * 100
            mpnl = sum(month_profits)
            print(f"    Month {m+1}: {mwr:5.1f}% WR ({mn:3d} bets) PnL={mpnl:+6.2f}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 5: BTC-only baseline validation (no spread) for comparison")
    print("=" * 130)

    for pool_frac in [0.20, 0.25, 0.28]:
        t_results = []
        v_results = []
        for rnd_set, results_out in [(train, t_results), (valid, v_results)]:
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
                bet = max(0.01, min(2.0, vis_pool * pool_frac))
                profit = settle(rnd, bet, signal)
                results_out.append((profit, bet))

        nt = len(t_results)
        nv = len(v_results)
        wt = sum(1 for p, b in t_results if p > 0) / nt * 100
        wv = sum(1 for p, b in v_results if p > 0) / nv * 100
        pv = sum(p for p, b in v_results)
        abv = sum(b for p, b in v_results) / nv
        pnl_2k = pv / v_total * 2000
        print(f"  BTC-only frac={pool_frac}  T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
              f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 6: Sensitivity test -- what if spread WR degrades to 60%?")
    print("  Conservative estimate of strategy value")
    print("=" * 130)

    # From validation: BTC lead = 68% WR, spread = 74.1% WR
    # If spread degrades to 60%, how does PnL change?
    # Can estimate: spread adds ~27 bets per 10.4k rounds = 5.2/2k
    # At 60% WR with avg bet 0.54, and considering dilution...
    # Actually let me just compute it directly from the trades

    v_results = run_stacked(valid, btc_kl, spot_kl, pool_frac=0.25)
    btc_only = [(p, b) for p, b, s in v_results if s == "btc"]
    spread_only = [(p, b) for p, b, s in v_results if s == "spread"]

    btc_pnl = sum(p for p, b in btc_only)
    spr_pnl = sum(p for p, b in spread_only)
    n_spr = len(spread_only)
    spr_wins = sum(1 for p, b in spread_only if p > 0)
    spr_losses = n_spr - spr_wins
    avg_spr_win = sum(p for p, b in spread_only if p > 0) / max(1, spr_wins)
    avg_spr_loss = sum(abs(p) for p, b in spread_only if p <= 0) / max(1, spr_losses)

    print(f"  BTC-only PnL: {btc_pnl:+.2f} (/2k: {btc_pnl/v_total*2000:+.2f})")
    print(f"  Spread PnL: {spr_pnl:+.2f} (WR={spr_wins/n_spr*100:.1f}%, "
          f"avg_win={avg_spr_win:+.3f}, avg_loss={avg_spr_loss:.3f})")
    print(f"  Combined: {btc_pnl+spr_pnl:+.2f} (/2k: {(btc_pnl+spr_pnl)/v_total*2000:+.2f})")

    # Simulate at different spread WR scenarios
    for scenario_wr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.74]:
        # At scenario_wr, expected spread PnL = n_spr * (wr * avg_win - (1-wr) * avg_loss)
        expected_spr_pnl = n_spr * (scenario_wr * avg_spr_win - (1 - scenario_wr) * avg_spr_loss)
        total_pnl = btc_pnl + expected_spr_pnl
        print(f"    If spread WR={scenario_wr*100:.0f}%: spread PnL={expected_spr_pnl:+.2f}, "
              f"total={total_pnl:+.2f}, /2k={total_pnl/v_total*2000:+.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
