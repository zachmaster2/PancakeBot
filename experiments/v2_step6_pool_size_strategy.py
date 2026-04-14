"""Step 6: Pool-size-aware strategy.

Key finding: dilution kills edge above 0.25 BNB on avg pools (~2.4 BNB).
But pool sizes vary hugely. On large-pool rounds, we can bet more.

This script:
1. Maps pool size distribution overall and for signal rounds
2. Tests pool-size-proportional sizing (bet X% of pool)
3. Tests minimum pool size filters (only bet when pool is large enough)
4. Analyzes: where do large pools happen? Time of day? Can we predict them?
5. Tests all viable combinations to maximize PnL/2000 rounds
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


def get_final_pool(rnd):
    bull_wei = bear_wei = 0
    for bet in rnd.bets:
        if bet.position == "Bull":
            bull_wei += int(bet.amount_wei)
        else:
            bear_wei += int(bet.amount_wei)
    return bull_wei / 1e18, bear_wei / 1e18


def simulate(rounds, btc_kl, *, config):
    btc_lb = config["btc_lb"]
    btc_thresh = config["btc_thresh"]
    require_accel = config.get("require_accel", False)
    accel_short = config.get("accel_short", 2)
    min_payout = config.get("min_payout", 0.0)

    # Pool-proportional sizing
    pool_frac = config.get("pool_frac", 0.0)  # bet = pool_total * pool_frac
    fixed_bet = config.get("fixed_bet", 0.10)  # fallback if pool_frac=0
    min_pool = config.get("min_pool", 0.0)     # skip if pool < this
    max_bet = config.get("max_bet", 2.0)
    min_bet = config.get("min_bet", 0.01)

    trades = []
    bets_used = []

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        btc_raw = btc_kl.get(epoch)
        if not btc_raw: continue
        btc_closes = get_closes(btc_raw, cutoff_ms)
        if btc_closes is None: continue

        btc_r = _get_return(btc_closes, btc_lb)
        if btc_r is None or abs(btc_r) < btc_thresh: continue
        signal = "Bull" if btc_r > 0 else "Bear"

        if require_accel:
            btc_r_short = _get_return(btc_closes, accel_short)
            if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                continue

        pool_bull, pool_bear = get_pool_at_cutoff(rnd, lock_at)
        pool_total = pool_bull + pool_bear

        if pool_total < min_pool:
            continue

        if min_payout > 0 and pool_total > 0:
            our_side = pool_bull if signal == "Bull" else pool_bear
            if our_side > 0:
                pm = pool_total * (1.0 - TREASURY_FEE) / our_side
                if pm < min_payout:
                    continue

        # Size bet
        if pool_frac > 0 and pool_total > 0:
            bet = pool_total * pool_frac
            bet = max(min_bet, min(max_bet, bet))
        else:
            bet = fixed_bet

        out = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - bet - GAS_COST_BET_BNB
        trades.append(profit)
        bets_used.append(bet)

    n = len(trades)
    wins = sum(1 for p in trades if p > 0)
    wr = wins / max(1, n) * 100
    pnl = sum(trades)
    avg_bet = sum(bets_used) / max(1, n)
    return n, wins, wr, pnl, avg_bet


def show(label, train, valid, btc, config, v_total):
    nt, _, wt, pt, abt = simulate(train, btc, config=config)
    if nt < 30: return
    nv, _, wv, pv, abv = simulate(valid, btc, config=config)
    if nv < 10: return
    pnl_2k = pv / v_total * 2000
    bets_2k = nv / v_total * 2000
    flag = " ***" if pv > 0 else ""
    print(f"  {label:65s} T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
          f"PnL={pv:+7.2f} avg={abv:.3f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")


def main():
    rounds, spot, btc = load_data()
    split = int(len(rounds) * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    v_total = len(valid)
    print(f"Rounds: {len(rounds)} total, {len(train)} train, {len(valid)} validate\n")

    # =====================================================================
    print("=" * 130)
    print("PART 1: Pool size distribution (at lock-6)")
    print("=" * 130)

    pool_sizes = []
    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        pb, pe = get_pool_at_cutoff(rnd, lock_at)
        pool_sizes.append(pb + pe)

    pool_sizes.sort()
    n = len(pool_sizes)
    print(f"  Pool sizes at lock-6 (N={n}):")
    print(f"    Mean:   {sum(pool_sizes)/n:.2f} BNB")
    print(f"    Median: {pool_sizes[n//2]:.2f} BNB")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = pool_sizes[int(n * pct / 100)]
        print(f"    P{pct:2d}:    {val:.2f} BNB")

    # Final pool sizes
    final_sizes = []
    for rnd in rounds:
        fb, fe = get_final_pool(rnd)
        final_sizes.append(fb + fe)
    final_sizes.sort()
    print(f"\n  Final pool sizes (N={n}):")
    print(f"    Mean:   {sum(final_sizes)/n:.2f} BNB")
    print(f"    Median: {final_sizes[n//2]:.2f} BNB")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = final_sizes[int(n * pct / 100)]
        print(f"    P{pct:2d}:    {val:.2f} BNB")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 2: Pool-proportional sizing sweeps")
    print("  bet = pool_total_at_lock6 * frac, capped")
    print("=" * 130)

    for btc_lb, btc_thresh in [(7, 0.0007)]:
        for accel in [False, True]:
            accel_str = "+accel" if accel else ""
            for pool_frac in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
                for max_b in [0.5, 1.0, 2.0, 5.0]:
                    cfg = {"btc_lb": btc_lb, "btc_thresh": btc_thresh,
                           "require_accel": accel,
                           "pool_frac": pool_frac, "max_bet": max_b}
                    show(f"btc({btc_lb},{btc_thresh}){accel_str} frac={pool_frac} cap={max_b}",
                         train, valid, btc, cfg, v_total)

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 3: Min pool size filter (only bet on larger pools)")
    print("=" * 130)

    for btc_lb, btc_thresh in [(7, 0.0007)]:
        for accel in [True]:
            for min_pool in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
                for pool_frac in [0.05, 0.10, 0.15]:
                    cfg = {"btc_lb": btc_lb, "btc_thresh": btc_thresh,
                           "require_accel": True,
                           "pool_frac": pool_frac, "max_bet": 5.0,
                           "min_pool": min_pool}
                    show(f"btc(7,0.0007)+accel minpool={min_pool} frac={pool_frac}",
                         train, valid, btc, cfg, v_total)

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 4: Pool size by hour of day")
    print("=" * 130)

    hour_pools = defaultdict(list)
    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        hour = (lock_at % 86400) // 3600
        pb, pe = get_final_pool(rnd)
        hour_pools[hour].append(pb + pe)

    print(f"\n  {'Hour':>4s} {'N':>6s} {'Mean':>8s} {'Median':>8s} {'P75':>8s} {'P90':>8s}")
    print("  " + "-" * 42)
    for h in range(24):
        pools = sorted(hour_pools[h])
        n = len(pools)
        if n == 0: continue
        mean = sum(pools) / n
        med = pools[n//2]
        p75 = pools[int(n*0.75)]
        p90 = pools[int(n*0.90)]
        print(f"  {h:4d} {n:6d} {mean:7.2f} {med:7.2f} {p75:7.2f} {p90:7.2f}")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 5: Signal WR by pool size bucket")
    print("=" * 130)

    for btc_lb, btc_thresh in [(7, 0.0007)]:
        pool_wr = defaultdict(lambda: [0, 0])  # bucket -> [total, wins]

        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            epoch = int(rnd.epoch)
            cutoff_ms = (lock_at - CUTOFF_S) * 1000

            btc_raw = btc.get(epoch)
            if not btc_raw: continue
            btc_closes = get_closes(btc_raw, cutoff_ms)
            if btc_closes is None: continue
            btc_r = _get_return(btc_closes, btc_lb)
            if btc_r is None or abs(btc_r) < btc_thresh: continue
            signal = "Bull" if btc_r > 0 else "Bear"

            # Check accel
            btc_r_short = _get_return(btc_closes, 2)
            if btc_r_short is None or (btc_r_short > 0) != (btc_r > 0):
                continue

            pb, pe = get_final_pool(rnd)
            pool = pb + pe

            out = settle_bet_against_closed_round(
                bet_bnb=0.10, bet_side=signal,
                round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
            )
            won = out.credit_bnb > 0.10

            if pool < 1.0: b = "<1"
            elif pool < 2.0: b = "1-2"
            elif pool < 3.0: b = "2-3"
            elif pool < 5.0: b = "3-5"
            elif pool < 8.0: b = "5-8"
            elif pool < 15.0: b = "8-15"
            else: b = "15+"

            pool_wr[b][0] += 1
            pool_wr[b][1] += 1 if won else 0

        print(f"\n  btc(7,0.0007)+accel WR by final pool size:")
        print(f"  {'Pool BNB':>10s} {'N':>5s} {'WR':>6s}")
        print("  " + "-" * 25)
        for b in ["<1", "1-2", "2-3", "3-5", "5-8", "8-15", "15+"]:
            if pool_wr[b][0] == 0: continue
            n, w = pool_wr[b]
            wr = w / n * 100
            print(f"  {b:>10s} {n:5d} {wr:5.1f}%")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 6: Best combined strategies targeting 4 BNB / 2000 rounds")
    print("=" * 130)

    # Multi-signal + pool-proportional + accel
    configs_to_try = []

    # Single signal, pool-proportional
    for pool_frac in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for min_pm in [0.0, 1.50]:
            configs_to_try.append((
                f"btc(7,0.0007)+accel frac={pool_frac} pm>={min_pm}",
                {"btc_lb": 7, "btc_thresh": 0.0007, "require_accel": True,
                 "pool_frac": pool_frac, "max_bet": 5.0, "min_payout": min_pm}
            ))

    # Add lower threshold variants for more bets
    for btc_thresh in [0.0005, 0.0006]:
        for pool_frac in [0.05, 0.08, 0.10]:
            configs_to_try.append((
                f"btc(7,{btc_thresh})+accel frac={pool_frac}",
                {"btc_lb": 7, "btc_thresh": btc_thresh, "require_accel": True,
                 "pool_frac": pool_frac, "max_bet": 5.0}
            ))

    # Multiple lookbacks
    for pool_frac in [0.05, 0.08, 0.10]:
        configs_to_try.append((
            f"btc(7,0.0007)+accel + btc(15,0.001)+accel frac={pool_frac}",
            {"btc_lb": 7, "btc_thresh": 0.0007, "require_accel": True,
             "pool_frac": pool_frac, "max_bet": 5.0,
             "_multi": [(7, 0.0007), (15, 0.001)]}
        ))

    for label, cfg in configs_to_try:
        if "_multi" in cfg:
            # Handle multi-signal separately
            signals = cfg.pop("_multi")
            pool_frac = cfg["pool_frac"]
            max_bet = cfg["max_bet"]
            trades_t = _multi_sim_pool(train, btc, signals, pool_frac, max_bet)
            trades_v = _multi_sim_pool(valid, btc, signals, pool_frac, max_bet)
            nt, nv = len(trades_t), len(trades_v)
            if nt < 30 or nv < 10:
                continue
            wt = sum(1 for p in trades_t if p > 0) / nt * 100
            wv = sum(1 for p in trades_v if p > 0) / nv * 100
            pv = sum(trades_v)
            pnl_2k = pv / v_total * 2000
            bets_2k = nv / v_total * 2000
            flag = " ***" if pv > 0 else ""
            print(f"  {label:65s} T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) "
                  f"PnL={pv:+7.2f} /2k={pnl_2k:+5.2f}({bets_2k:.0f}b){flag}")
        else:
            show(label, train, valid, btc, cfg, v_total)

    print("\nDone.")


def _multi_sim_pool(rounds, btc_kl, signals, pool_frac, max_bet):
    """Multi-signal simulation with pool-proportional sizing."""
    trades = []
    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        btc_raw = btc_kl.get(epoch)
        if not btc_raw: continue
        btc_closes = get_closes(btc_raw, cutoff_ms)
        if btc_closes is None: continue

        fired = None
        for lb, thresh in signals:
            btc_r = _get_return(btc_closes, lb)
            if btc_r is not None and abs(btc_r) >= thresh:
                btc_r_short = _get_return(btc_closes, 2)
                if btc_r_short is not None and (btc_r_short > 0) == (btc_r > 0):
                    fired = "Bull" if btc_r > 0 else "Bear"
                    break
        if fired is None:
            continue

        pool_bull, pool_bear = get_pool_at_cutoff(rnd, lock_at)
        pool_total = pool_bull + pool_bear
        bet = max(0.01, min(max_bet, pool_total * pool_frac))

        out = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=fired,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - bet - GAS_COST_BET_BNB
        trades.append(profit)
    return trades


if __name__ == "__main__":
    main()
