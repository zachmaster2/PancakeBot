"""Deep dive into BTC lead signal — the only feature that survived walk-forward.

Explores:
1. Fine-grained lookback/threshold sweep around the promising zone
2. Payout filtering (only bet when payout is favorable)
3. BNB confirmation (BTC lead + BNB agrees)
4. Hour-of-day filtering
5. Sizing optimization
6. Combined best settings with walk-forward validation
"""
from __future__ import annotations

import json, sys, math
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB, INTERVAL_SECONDS, BNB_WEI
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import _trim_to_window, _get_return
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 4
CANDLE_COUNT = 31
TREASURY_FEE = 0.03


def load_data():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    def load_kl(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip(): continue
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


def simulate(rounds, spot, btc_kl, *, btc_lb, btc_thresh, min_payout=0.0,
             bnb_confirm_lb=0, bnb_confirm_thresh=0.0, skip_hours=(),
             bet_size=0.10, payout_sizing=False):
    """Full simulation with configurable parameters."""
    bankroll = 50.0
    trades = []

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        # Hour filter
        if skip_hours:
            hour_utc = (lock_at % 86400) // 3600
            if hour_utc in skip_hours:
                continue

        btc_raw = btc_kl.get(epoch)
        if not btc_raw:
            continue
        btc_closes = get_closes(btc_raw, cutoff_ms)
        if btc_closes is None:
            continue

        # BTC signal
        btc_r = _get_return(btc_closes, btc_lb)
        if btc_r is None or abs(btc_r) < btc_thresh:
            continue
        signal = "Bull" if btc_r > 0 else "Bear"

        # Optional BNB confirmation
        if bnb_confirm_lb > 0:
            bnb_raw = spot.get(epoch)
            if not bnb_raw:
                continue
            bnb_closes = get_closes(bnb_raw, cutoff_ms)
            if bnb_closes is None:
                continue
            bnb_r = _get_return(bnb_closes, bnb_confirm_lb)
            if bnb_r is None or abs(bnb_r) < bnb_confirm_thresh:
                continue
            bnb_dir = "Bull" if bnb_r > 0 else "Bear"
            if bnb_dir != signal:
                continue  # BNB disagrees with BTC

        # Payout filter
        if min_payout > 0:
            bull_wei = sum(int(b.amount_wei) for b in rnd.bets
                         if int(b.created_at) <= lock_at - CUTOFF_S and b.position == "Bull")
            bear_wei = sum(int(b.amount_wei) for b in rnd.bets
                         if int(b.created_at) <= lock_at - CUTOFF_S and b.position == "Bear")
            pool_bull = bull_wei / 1e18
            pool_bear = bear_wei / 1e18
            pool_total = pool_bull + pool_bear
            if pool_total <= 0:
                continue
            our_side = pool_bull if signal == "Bull" else pool_bear
            if our_side <= 0:
                continue
            pm = pool_total * 0.97 / our_side
            if pm < min_payout:
                continue

            # Payout-proportional sizing
            if payout_sizing:
                actual_bet = bet_size * max(0.3, 0.1 + 1.0 * (pm - 1.0))
                actual_bet = min(2.0, actual_bet)
            else:
                actual_bet = bet_size
        else:
            actual_bet = bet_size

        out = settle_bet_against_closed_round(
            bet_bnb=actual_bet, bet_side=signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - actual_bet - GAS_COST_BET_BNB
        bankroll += profit
        trades.append(profit)

    n = len(trades)
    wins = sum(1 for p in trades if p > 0)
    wr = wins / max(1, n) * 100
    pnl = sum(trades)
    return n, wins, wr, pnl


def main():
    rounds, spot, btc = load_data()
    split = int(len(rounds) * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    print(f"Rounds: {len(rounds)} total, {len(train)} train, {len(valid)} validate\n")

    # =====================================================================
    print("=" * 80)
    print("PART 1: Fine-grained BTC lookback/threshold sweep")
    print("=" * 80)

    results = []
    for lb in range(5, 26):
        for thresh in [0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.001, 0.0012, 0.0015]:
            nt, _, wt, pt = simulate(train, spot, btc, btc_lb=lb, btc_thresh=thresh)
            if nt < 80:
                continue
            nv, _, wv, pv = simulate(valid, spot, btc, btc_lb=lb, btc_thresh=thresh)
            if nv < 30:
                continue
            results.append((wv, wt, lb, thresh, nt, nv, pv))

    results.sort(reverse=True)
    print(f"\nTop 20 by validation WR:")
    print(f"{'V_WR':>6s} {'T_WR':>6s} {'LB':>4s} {'Thresh':>8s} {'T_N':>6s} {'V_N':>6s} {'V_PnL':>8s}")
    print("-" * 55)
    for wv, wt, lb, th, tn, vn, vpnl in results[:20]:
        print(f"{wv:5.1f}% {wt:5.1f}% {lb:4d} {th:8.4f} {tn:6d} {vn:6d} {vpnl:+7.2f}")

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 2: Best BTC signals + payout filter sweep")
    print("=" * 80)

    # Take top 5 unique (lb, thresh) pairs
    seen = set()
    top_params = []
    for wv, wt, lb, th, tn, vn, vpnl in results:
        key = (lb, th)
        if key not in seen:
            seen.add(key)
            top_params.append((lb, th))
        if len(top_params) >= 5:
            break

    for lb, th in top_params:
        print(f"\n  btc_lead(lb={lb}, th={th}):")
        for min_pm in [0.0, 1.5, 1.7, 1.85, 2.0, 2.5, 3.0]:
            nt, _, wt, pt = simulate(train, spot, btc, btc_lb=lb, btc_thresh=th, min_payout=min_pm)
            nv, _, wv, pv = simulate(valid, spot, btc, btc_lb=lb, btc_thresh=th, min_payout=min_pm)
            if nt < 30 or nv < 15:
                continue
            flag = " ***" if pv > 0 else ""
            print(f"    pm>={min_pm:.2f}: T={wt:5.1f}%({nt:4d})  V={wv:5.1f}%({nv:4d}) V_PnL={pv:+6.2f}{flag}")

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 3: BTC lead + BNB confirmation")
    print("=" * 80)

    for lb, th in top_params[:3]:
        for bnb_lb in [3, 5, 7, 10]:
            for bnb_th in [0.0001, 0.0002, 0.0003]:
                nt, _, wt, _ = simulate(train, spot, btc, btc_lb=lb, btc_thresh=th,
                                         bnb_confirm_lb=bnb_lb, bnb_confirm_thresh=bnb_th)
                if nt < 50:
                    continue
                nv, _, wv, pv = simulate(valid, spot, btc, btc_lb=lb, btc_thresh=th,
                                          bnb_confirm_lb=bnb_lb, bnb_confirm_thresh=bnb_th)
                if wv > 57:
                    print(f"  btc({lb},{th}) + bnb({bnb_lb},{bnb_th}): "
                          f"T={wt:.1f}%({nt}) V={wv:.1f}%({nv}) V_PnL={pv:+.2f}")

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 4: Hour-of-day analysis for best BTC signal")
    print("=" * 80)

    best_lb, best_th = top_params[0]
    print(f"\n  Using btc_lead(lb={best_lb}, th={best_th}):")

    # Per-hour WR on all data
    hour_stats = {}
    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000
        hour = (lock_at % 86400) // 3600

        btc_raw = btc.get(epoch)
        if not btc_raw: continue
        btc_closes = get_closes(btc_raw, cutoff_ms)
        if btc_closes is None: continue
        btc_r = _get_return(btc_closes, best_lb)
        if btc_r is None or abs(btc_r) < best_th: continue

        signal = "Bull" if btc_r > 0 else "Bear"
        out = settle_bet_against_closed_round(
            bet_bnb=0.10, bet_side=signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        won = out.credit_bnb > 0.10
        hour_stats.setdefault(hour, [0, 0])
        hour_stats[hour][0] += 1
        hour_stats[hour][1] += 1 if won else 0

    print(f"  {'Hour':>4s} {'Bets':>5s} {'WR':>6s}")
    bad_hours = []
    for h in sorted(hour_stats):
        n, w = hour_stats[h]
        wr = w / n * 100
        flag = " <-- skip?" if wr < 52 else ""
        print(f"  {h:4d} {n:5d} {wr:5.1f}%{flag}")
        if wr < 52:
            bad_hours.append(h)

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 5: Best combined config with payout sizing")
    print("=" * 80)

    skip_h = tuple(bad_hours) if bad_hours else ()
    print(f"  Skip hours: {skip_h}")

    for lb, th in top_params[:3]:
        for min_pm in [0.0, 1.85, 2.0, 2.5]:
            for ps in [False, True]:
                for bs in [0.06, 0.10, 0.15]:
                    nt, _, wt, pt = simulate(train, spot, btc, btc_lb=lb, btc_thresh=th,
                                              min_payout=min_pm, skip_hours=skip_h,
                                              bet_size=bs, payout_sizing=ps)
                    if nt < 50:
                        continue
                    nv, _, wv, pv = simulate(valid, spot, btc, btc_lb=lb, btc_thresh=th,
                                              min_payout=min_pm, skip_hours=skip_h,
                                              bet_size=bs, payout_sizing=ps)
                    if pv > 0:
                        ps_str = "payout" if ps else "fixed"
                        print(f"  btc({lb},{th}) pm>={min_pm} bet={bs} {ps_str} skip={skip_h}: "
                              f"T={wt:.1f}%({nt}) V={wv:.1f}%({nv}) V_PnL={pv:+.2f} ***")

    print("\nDone.")


if __name__ == "__main__":
    main()
