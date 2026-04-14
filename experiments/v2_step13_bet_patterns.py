"""Step 13: Bet pattern analysis -- is there signal in how bets arrive?

The current approach uses price momentum. But the pool (bets placed by other
participants) might also contain information:
- Smart money timing: do early/late bets predict outcome?
- Bet clustering: do bursts of bets in one direction predict?
- Whale bets: does a single large bet predict?
- Bet flow momentum: is the direction of recent bets predictive?
- Late reversal: if late bets flip the pool direction, is the new direction right?

These signals are independent of price data -- purely on-chain.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import (
    GAS_COST_BET_BNB, INTERVAL_SECONDS, POOL_CUTOFF_SECONDS,
)
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.runtime.settlement import settle_bet_against_closed_round

POOL_CUTOFF_S = POOL_CUTOFF_SECONDS
TREASURY_FEE = 0.03
SKIP_NIGHT = {0, 1, 2, 3, 4, 23}


def load_data():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    return rounds


def settle(rnd, bet_bnb, side):
    out = settle_bet_against_closed_round(
        bet_bnb=bet_bnb, bet_side=side,
        round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
    )
    return out.credit_bnb - bet_bnb - GAS_COST_BET_BNB


def main():
    rounds = load_data()
    total = len(rounds)
    print(f"Total rounds: {total}\n")

    # =====================================================================
    print("=" * 130)
    print("PART 1: Early vs late bets -- do early bettors know something?")
    print("  Split bets into early (before lock-30s) and late (lock-30 to lock-6)")
    print("=" * 130)

    for early_cutoff_before_lock in [60, 45, 30, 20]:
        bull_early = 0
        bear_early = 0
        wins_follow = 0
        wins_fade = 0
        total_trades = 0

        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            hour = (lock_at % 86400) // 3600
            if hour in SKIP_NIGHT:
                continue

            early_cutoff_ts = lock_at - early_cutoff_before_lock
            pool_cutoff_ts = lock_at - POOL_CUTOFF_S

            # Early bets: placed before early_cutoff
            e_bull = e_bear = 0
            for bet in rnd.bets:
                if int(bet.created_at) > early_cutoff_ts:
                    continue
                if bet.position == "Bull":
                    e_bull += int(bet.amount_wei)
                else:
                    e_bear += int(bet.amount_wei)

            e_total = e_bull + e_bear
            if e_total == 0:
                continue

            # Early direction
            early_dir = "Bull" if e_bull > e_bear else "Bear"

            # Check outcome
            profit_follow = settle(rnd, 0.10, early_dir)
            profit_fade = settle(rnd, 0.10, "Bear" if early_dir == "Bull" else "Bull")

            total_trades += 1
            if profit_follow > 0:
                wins_follow += 1
            if profit_fade > 0:
                wins_fade += 1

        wr_follow = wins_follow / max(1, total_trades) * 100
        wr_fade = wins_fade / max(1, total_trades) * 100
        print(f"  early_cutoff={early_cutoff_before_lock}s: N={total_trades} "
              f"follow_early={wr_follow:.1f}% fade_early={wr_fade:.1f}%")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 2: Late bet flow direction -- do last-minute bets predict?")
    print("  Look at bets between lock-20s and lock-6s")
    print("=" * 130)

    for late_window_start in [30, 20, 15, 10]:
        wins_follow = 0
        wins_fade = 0
        total_trades = 0

        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            hour = (lock_at % 86400) // 3600
            if hour in SKIP_NIGHT:
                continue

            late_start_ts = lock_at - late_window_start
            pool_cutoff_ts = lock_at - POOL_CUTOFF_S

            # Late bets: placed between late_start and pool_cutoff
            l_bull = l_bear = 0
            for bet in rnd.bets:
                ts = int(bet.created_at)
                if ts < late_start_ts or ts > pool_cutoff_ts:
                    continue
                if bet.position == "Bull":
                    l_bull += int(bet.amount_wei)
                else:
                    l_bear += int(bet.amount_wei)

            l_total = l_bull + l_bear
            if l_total == 0:
                continue

            late_dir = "Bull" if l_bull > l_bear else "Bear"

            profit_follow = settle(rnd, 0.10, late_dir)
            total_trades += 1
            if profit_follow > 0:
                wins_follow += 1

        wr_follow = wins_follow / max(1, total_trades) * 100
        print(f"  late_window={late_window_start}s-{POOL_CUTOFF_S}s: N={total_trades} "
              f"follow_late={wr_follow:.1f}%")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 3: Whale detection -- does a single large bet predict?")
    print("  Look for bets > X% of total pool")
    print("=" * 130)

    for whale_frac in [0.10, 0.15, 0.20, 0.30, 0.50]:
        wins_follow = 0
        total_trades = 0

        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            hour = (lock_at % 86400) // 3600
            if hour in SKIP_NIGHT:
                continue

            pool_cutoff_ts = lock_at - POOL_CUTOFF_S

            # All visible bets
            visible_bets = []
            for bet in rnd.bets:
                if int(bet.created_at) > pool_cutoff_ts:
                    continue
                visible_bets.append(bet)

            if not visible_bets:
                continue

            total_pool_wei = sum(int(b.amount_wei) for b in visible_bets)
            if total_pool_wei == 0:
                continue

            # Find largest single bet
            largest = max(visible_bets, key=lambda b: int(b.amount_wei))
            largest_frac = int(largest.amount_wei) / total_pool_wei

            if largest_frac < whale_frac:
                continue

            # Follow the whale
            whale_dir = largest.position
            profit = settle(rnd, 0.10, whale_dir)
            total_trades += 1
            if profit > 0:
                wins_follow += 1

        wr = wins_follow / max(1, total_trades) * 100
        print(f"  whale>{whale_frac*100:.0f}% of pool: N={total_trades} follow_whale={wr:.1f}%")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 4: Bet count momentum -- is number of bets predictive?")
    print("  If many more bets on one side, follow the crowd?")
    print("=" * 130)

    for min_bet_ratio in [1.5, 2.0, 3.0, 5.0]:
        wins_follow = 0
        wins_fade = 0
        total_trades = 0

        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            hour = (lock_at % 86400) // 3600
            if hour in SKIP_NIGHT:
                continue

            pool_cutoff_ts = lock_at - POOL_CUTOFF_S

            n_bull = n_bear = 0
            for bet in rnd.bets:
                if int(bet.created_at) > pool_cutoff_ts:
                    continue
                if bet.position == "Bull":
                    n_bull += 1
                else:
                    n_bear += 1

            if n_bull == 0 or n_bear == 0:
                continue

            ratio = max(n_bull / n_bear, n_bear / n_bull)
            if ratio < min_bet_ratio:
                continue

            crowd_dir = "Bull" if n_bull > n_bear else "Bear"
            profit_follow = settle(rnd, 0.10, crowd_dir)
            profit_fade = settle(rnd, 0.10, "Bear" if crowd_dir == "Bull" else "Bull")

            total_trades += 1
            if profit_follow > 0:
                wins_follow += 1
            if profit_fade > 0:
                wins_fade += 1

        wr_follow = wins_follow / max(1, total_trades) * 100
        wr_fade = wins_fade / max(1, total_trades) * 100
        print(f"  bet_ratio>={min_bet_ratio:.1f}x: N={total_trades} "
              f"follow={wr_follow:.1f}% fade={wr_fade:.1f}%")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 5: Pool flow reversal -- late bets flip early direction")
    print("  When late bets reverse the early pool direction, which is right?")
    print("=" * 130)

    wins_follow_early = 0
    wins_follow_late = 0
    total_reversals = 0

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        hour = (lock_at % 86400) // 3600
        if hour in SKIP_NIGHT:
            continue

        early_ts = lock_at - 30  # early = before 30s to lock
        pool_cutoff_ts = lock_at - POOL_CUTOFF_S

        e_bull = e_bear = l_bull = l_bear = 0
        for bet in rnd.bets:
            ts = int(bet.created_at)
            if ts <= early_ts:
                if bet.position == "Bull":
                    e_bull += int(bet.amount_wei)
                else:
                    e_bear += int(bet.amount_wei)
            elif ts <= pool_cutoff_ts:
                if bet.position == "Bull":
                    l_bull += int(bet.amount_wei)
                else:
                    l_bear += int(bet.amount_wei)

        e_total = e_bull + e_bear
        l_total = l_bull + l_bear
        if e_total == 0 or l_total == 0:
            continue

        early_dir = "Bull" if e_bull > e_bear else "Bear"

        # Check if late bets reversed the pool direction
        total_bull = e_bull + l_bull
        total_bear = e_bear + l_bear
        final_dir = "Bull" if total_bull > total_bear else "Bear"

        if early_dir == final_dir:
            continue  # no reversal

        total_reversals += 1
        profit_early = settle(rnd, 0.10, early_dir)
        profit_late = settle(rnd, 0.10, final_dir)

        if profit_early > 0:
            wins_follow_early += 1
        if profit_late > 0:
            wins_follow_late += 1

    if total_reversals > 0:
        print(f"  Reversals found: {total_reversals}")
        print(f"  Follow early (pre-reversal): {wins_follow_early/total_reversals*100:.1f}%")
        print(f"  Follow late (post-reversal): {wins_follow_late/total_reversals*100:.1f}%")
    else:
        print(f"  No reversals found")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 6: Previous round outcome as signal")
    print("  Does the previous round's direction predict this round?")
    print("=" * 130)

    for streak_len in [1, 2, 3]:
        wins_follow = 0
        wins_fade = 0
        total_trades = 0

        for i in range(streak_len, len(rounds)):
            rnd = rounds[i]
            lock_at = int(rnd.lock_at)
            hour = (lock_at % 86400) // 3600
            if hour in SKIP_NIGHT:
                continue

            # Check if previous N rounds all went same direction
            directions = []
            for j in range(1, streak_len + 1):
                prev = rounds[i - j]
                if hasattr(prev, 'lock_price') and hasattr(prev, 'close_price'):
                    lp = prev.lock_price
                    cp = prev.close_price
                    if lp is not None and cp is not None and lp > 0:
                        directions.append("Bull" if cp > lp else "Bear")

            if len(directions) != streak_len:
                continue
            if len(set(directions)) > 1:
                continue  # not all same direction

            streak_dir = directions[0]
            profit_follow = settle(rnd, 0.10, streak_dir)
            profit_fade = settle(rnd, 0.10, "Bear" if streak_dir == "Bull" else "Bull")

            total_trades += 1
            if profit_follow > 0:
                wins_follow += 1
            if profit_fade > 0:
                wins_fade += 1

        wr_follow = wins_follow / max(1, total_trades) * 100
        wr_fade = wins_fade / max(1, total_trades) * 100
        print(f"  streak={streak_len}: N={total_trades} "
              f"follow_streak={wr_follow:.1f}% fade_streak={wr_fade:.1f}%")

    # =====================================================================
    print(f"\n{'=' * 130}")
    print("PART 7: Payout ratio as signal")
    print("  When payout is very high on one side, is that side underbet?")
    print("=" * 130)

    for pm_threshold in [2.0, 2.5, 3.0, 4.0, 5.0]:
        wins_bet_underdog = 0
        total_trades = 0

        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            hour = (lock_at % 86400) // 3600
            if hour in SKIP_NIGHT:
                continue

            pool_cutoff_ts = lock_at - POOL_CUTOFF_S
            bull_wei = bear_wei = 0
            for bet in rnd.bets:
                if int(bet.created_at) > pool_cutoff_ts:
                    continue
                if bet.position == "Bull":
                    bull_wei += int(bet.amount_wei)
                else:
                    bear_wei += int(bet.amount_wei)

            total_wei = bull_wei + bear_wei
            if total_wei == 0 or bull_wei == 0 or bear_wei == 0:
                continue

            pm_bull = total_wei * (1 - TREASURY_FEE) / bull_wei
            pm_bear = total_wei * (1 - TREASURY_FEE) / bear_wei

            # Bet on the underdog (high payout side)
            if pm_bull >= pm_threshold:
                signal = "Bull"
            elif pm_bear >= pm_threshold:
                signal = "Bear"
            else:
                continue

            profit = settle(rnd, 0.10, signal)
            total_trades += 1
            if profit > 0:
                wins_bet_underdog += 1

        wr = wins_bet_underdog / max(1, total_trades) * 100
        print(f"  pm>={pm_threshold}: N={total_trades} bet_underdog={wr:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
