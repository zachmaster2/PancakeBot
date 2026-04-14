"""Step 5: Realistic sizing and pure signal testing.

Key insight from pool analysis: payout data at lock-6 is unreliable.
pm>=2.50 estimated rounds actually have pm~2.08 final. Payout-proportional
sizing is betting big on wrong numbers.

This script tests:
1. Pure signal (no pool filter) at various fixed bet sizes
2. Dilution impact: does edge survive at realistic bet sizes?
3. Lower thresholds for more frequency with no payout filter
4. Signal-confidence sizing (BTC magnitude) instead of payout sizing
5. Multiple independent signal regimes for bet frequency
"""
from __future__ import annotations

import json, sys
from pathlib import Path

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

    return rounds, load_kl("var/bnb_spot_prices.jsonl"), load_kl("var/btc_spot_prices.jsonl")


def get_closes(raw, cutoff_ms):
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None
    return [k[4] for k in trimmed]


def get_pool(rnd, lock_at):
    pool_cutoff_ts = lock_at - POOL_CUTOFF_S
    bull_wei = 0
    bear_wei = 0
    for bet in rnd.bets:
        if int(bet.created_at) > pool_cutoff_ts:
            continue
        if bet.position == "Bull":
            bull_wei += int(bet.amount_wei)
        else:
            bear_wei += int(bet.amount_wei)
    return bull_wei / 1e18, bear_wei / 1e18


def simulate(rounds, btc_kl, spot_kl, *, config):
    btc_lb = config["btc_lb"]
    btc_thresh = config["btc_thresh"]
    bet_size = config.get("bet_size", 0.10)
    min_payout = config.get("min_payout", 0.0)
    require_accel = config.get("require_accel", False)
    accel_short = config.get("accel_short", 2)
    skip_hours = config.get("skip_hours", ())

    # Signal-confidence sizing: scale bet by BTC move magnitude
    confidence_sizing = config.get("confidence_sizing", False)
    confidence_base = config.get("confidence_base", 0.5)  # multiplier at threshold
    confidence_scale = config.get("confidence_scale", 500)  # how much more per unit of return

    # Max dilution: skip if our bet would be > X% of visible pool
    max_pool_fraction = config.get("max_pool_fraction", 1.0)  # 1.0 = no limit

    trades = []
    bet_sizes_used = []

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

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

        btc_r = _get_return(btc_closes, btc_lb)
        if btc_r is None or abs(btc_r) < btc_thresh:
            continue
        signal = "Bull" if btc_r > 0 else "Bear"

        # Acceleration filter
        if require_accel:
            btc_r_short = _get_return(btc_closes, accel_short)
            if btc_r_short is None:
                continue
            if (btc_r_short > 0) != (btc_r > 0):
                continue

        # Pool check (only for min_payout filter or dilution check)
        pool_bull, pool_bear = get_pool(rnd, lock_at)
        pool_total = pool_bull + pool_bear

        if min_payout > 0 and pool_total > 0:
            our_side = pool_bull if signal == "Bull" else pool_bear
            if our_side > 0:
                pm = pool_total * (1.0 - TREASURY_FEE) / our_side
                if pm < min_payout:
                    continue

        # Determine bet size
        actual_bet = bet_size
        if confidence_sizing:
            strength = abs(btc_r) / btc_thresh  # 1.0 at threshold, higher for stronger moves
            actual_bet = bet_size * max(0.3, confidence_base * strength)
            actual_bet = min(bet_size * 3.0, actual_bet)

        # Dilution check
        if max_pool_fraction < 1.0 and pool_total > 0:
            max_bet = pool_total * max_pool_fraction
            if actual_bet > max_bet:
                actual_bet = max_bet

        out = settle_bet_against_closed_round(
            bet_bnb=actual_bet, bet_side=signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - actual_bet - GAS_COST_BET_BNB
        trades.append(profit)
        bet_sizes_used.append(actual_bet)

    n = len(trades)
    wins = sum(1 for p in trades if p > 0)
    wr = wins / max(1, n) * 100
    pnl = sum(trades)
    avg_bet = sum(bet_sizes_used) / max(1, n)
    return n, wins, wr, pnl, avg_bet


def show(label, train, valid, btc, spot, config, *, min_train=30, min_valid=15):
    nt, _, wt, pt, abt = simulate(train, btc, spot, config=config)
    if nt < min_train:
        return
    nv, _, wv, pv, abv = simulate(valid, btc, spot, config=config)
    if nv < min_valid:
        return
    # Extrapolate to per-2000-rounds
    v_rounds = len(valid)
    bets_per_2k = nv / v_rounds * 2000
    pnl_per_2k = pv / v_rounds * 2000
    flag = " ***" if pv > 0 else ""
    print(f"  {label:55s} T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
          f"PnL={pv:+7.2f} avgBet={abv:.3f} /2k={pnl_per_2k:+5.2f}({bets_per_2k:.0f}b){flag}")


def main():
    rounds, spot, btc = load_data()
    split = int(len(rounds) * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    print(f"Rounds: {len(rounds)} total, {len(train)} train, {len(valid)} validate")
    print(f"Validation rounds: {len(valid)} (extrapolate /2k from this)\n")

    # =====================================================================
    print("=" * 120)
    print("PART 1: Pure signal at various bet sizes (NO pool filter)")
    print("  Does the edge survive at larger bets? (settlement includes dilution)")
    print("=" * 120)

    for btc_lb, btc_thresh in [(7, 0.0007), (10, 0.0005), (10, 0.0007), (15, 0.001)]:
        for bet_size in [0.05, 0.10, 0.25, 0.50, 1.0, 2.0]:
            cfg = {"btc_lb": btc_lb, "btc_thresh": btc_thresh, "bet_size": bet_size}
            show(f"btc({btc_lb},{btc_thresh}) bet={bet_size:.2f}",
                 train, valid, btc, spot, cfg)
        print()

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 2: Signal + acceleration at various bet sizes")
    print("=" * 120)

    for btc_lb, btc_thresh in [(7, 0.0007), (10, 0.0007)]:
        for bet_size in [0.05, 0.10, 0.25, 0.50, 1.0, 2.0]:
            cfg = {"btc_lb": btc_lb, "btc_thresh": btc_thresh,
                   "bet_size": bet_size, "require_accel": True, "accel_short": 2}
            show(f"btc({btc_lb},{btc_thresh})+accel bet={bet_size:.2f}",
                 train, valid, btc, spot, cfg)
        print()

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 3: Lower thresholds for MORE frequency (no pool filter)")
    print("  Can we get more bets while keeping positive PnL?")
    print("=" * 120)

    for btc_lb in [5, 7, 10, 15]:
        for btc_thresh in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0007, 0.001]:
            for bet_size in [0.10, 0.50]:
                cfg = {"btc_lb": btc_lb, "btc_thresh": btc_thresh, "bet_size": bet_size}
                show(f"btc({btc_lb},{btc_thresh}) bet={bet_size:.2f}",
                     train, valid, btc, spot, cfg, min_train=50)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 4: Signal-confidence sizing (BTC magnitude)")
    print("  Size bet based on how strong the BTC move is, not pool payout")
    print("=" * 120)

    for btc_lb, btc_thresh in [(7, 0.0007), (10, 0.0007)]:
        for base_bet in [0.10, 0.25, 0.50]:
            for conf_base, conf_scale in [(0.3, 1.0), (0.5, 1.0), (0.5, 1.5), (1.0, 0.5)]:
                # confidence_sizing: bet = base * max(0.3, conf_base * (|btc_r| / thresh))
                cfg = {"btc_lb": btc_lb, "btc_thresh": btc_thresh,
                       "bet_size": base_bet, "confidence_sizing": True,
                       "confidence_base": conf_base, "confidence_scale": conf_scale}
                show(f"btc({btc_lb},{btc_thresh}) conf(b={conf_base},s={conf_scale}) base={base_bet}",
                     train, valid, btc, spot, cfg)
        print()

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 5: With max pool fraction cap (dilution protection)")
    print("  Cap bet at X% of visible pool to avoid moving the payout too much")
    print("=" * 120)

    for btc_lb, btc_thresh in [(7, 0.0007)]:
        for bet_size in [0.50, 1.0, 2.0]:
            for max_frac in [0.02, 0.05, 0.10, 0.20]:
                cfg = {"btc_lb": btc_lb, "btc_thresh": btc_thresh,
                       "bet_size": bet_size, "require_accel": True, "accel_short": 2,
                       "max_pool_fraction": max_frac}
                show(f"btc({btc_lb},{btc_thresh})+accel bet={bet_size} maxpool={max_frac}",
                     train, valid, btc, spot, cfg)
        print()

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 6: Only coarse pool filter (pm >= 1.50, the only reliable threshold)")
    print("=" * 120)

    for btc_lb, btc_thresh in [(7, 0.0007), (10, 0.0007)]:
        for bet_size in [0.10, 0.25, 0.50, 1.0]:
            for accel in [False, True]:
                accel_str = "+accel" if accel else ""
                cfg = {"btc_lb": btc_lb, "btc_thresh": btc_thresh,
                       "bet_size": bet_size, "min_payout": 1.50,
                       "require_accel": accel, "accel_short": 2}
                show(f"btc({btc_lb},{btc_thresh}){accel_str} pm>=1.50 bet={bet_size}",
                     train, valid, btc, spot, cfg)
        print()

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 7: Multiple signal regimes (bet on MORE rounds)")
    print("  Combine different BTC lookbacks to cover more opportunities")
    print("=" * 120)

    for rnd_set_label, rnd_set_train, rnd_set_valid in [("all", train, valid)]:
        for bet_size in [0.10, 0.50]:
            # Try: bet if ANY of several BTC signals fire
            # Each signal independently profitable; different lookbacks fire on different rounds
            for signals in [
                [(7, 0.0007), (15, 0.001)],
                [(7, 0.0007), (10, 0.0007), (15, 0.001)],
                [(7, 0.0005), (15, 0.0007)],
            ]:
                label = "+".join(f"({lb},{th})" for lb, th in signals)
                # Simulate: first signal to fire wins
                for accel in [False, True]:
                    accel_str = "+accel" if accel else ""
                    trades_t = _multi_signal_sim(rnd_set_train, btc, signals, bet_size, accel)
                    trades_v = _multi_signal_sim(rnd_set_valid, btc, signals, bet_size, accel)
                    nt = len(trades_t)
                    nv = len(trades_v)
                    if nt < 50 or nv < 20:
                        continue
                    wt = sum(1 for p in trades_t if p > 0) / nt * 100
                    wv = sum(1 for p in trades_v if p > 0) / nv * 100
                    pv = sum(trades_v)
                    v_rounds = len(rnd_set_valid)
                    pnl_per_2k = pv / v_rounds * 2000
                    bets_per_2k = nv / v_rounds * 2000
                    flag = " ***" if pv > 0 else ""
                    print(f"  multi[{label}]{accel_str} bet={bet_size} "
                          f"T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                          f"PnL={pv:+7.2f} /2k={pnl_per_2k:+5.2f}({bets_per_2k:.0f}b){flag}")

    print("\nDone.")


def _multi_signal_sim(rounds, btc_kl, signals, bet_size, require_accel):
    """Simulate multiple BTC signal lookbacks: first to fire wins."""
    trades = []

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

        # Try each signal; first to fire determines direction
        fired_signal = None
        for lb, thresh in signals:
            btc_r = _get_return(btc_closes, lb)
            if btc_r is not None and abs(btc_r) >= thresh:
                if require_accel:
                    btc_r_short = _get_return(btc_closes, 2)
                    if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                        continue
                fired_signal = "Bull" if btc_r > 0 else "Bear"
                break

        if fired_signal is None:
            continue

        out = settle_bet_against_closed_round(
            bet_bnb=bet_size, bet_side=fired_signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - bet_size - GAS_COST_BET_BNB
        trades.append(profit)

    return trades


if __name__ == "__main__":
    main()
