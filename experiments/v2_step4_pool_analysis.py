"""Step 4: Pool prediction quality analysis.

How well do pools at lock_at - 6 predict final pools?
This determines how much we can rely on pool-based filtering and sizing.

Analyzes:
1. Correlation between partial and final pools
2. Payout multiplier drift (estimated vs actual)
3. Pool imbalance stability (does direction hold?)
4. What fraction of final pool is present at various cutoffs
5. Distribution of payout estimation errors
"""
from __future__ import annotations

import json, sys, math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import INTERVAL_SECONDS, BNB_WEI, POOL_CUTOFF_SECONDS
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore

TREASURY_FEE = 0.03


def get_pools_at_cutoff(rnd, cutoff_ts):
    """Pool amounts from bets placed at or before cutoff_ts."""
    bull_wei = 0
    bear_wei = 0
    for bet in rnd.bets:
        if int(bet.created_at) > cutoff_ts:
            continue
        if bet.position == "Bull":
            bull_wei += int(bet.amount_wei)
        else:
            bear_wei += int(bet.amount_wei)
    return bull_wei / 1e18, bear_wei / 1e18


def get_final_pools(rnd):
    """Final pool amounts (all bets, no cutoff)."""
    bull_wei = 0
    bear_wei = 0
    for bet in rnd.bets:
        if bet.position == "Bull":
            bull_wei += int(bet.amount_wei)
        else:
            bear_wei += int(bet.amount_wei)
    return bull_wei / 1e18, bear_wei / 1e18


def payout_mult(pool_bull, pool_bear, side):
    total = pool_bull + pool_bear
    if total <= 0:
        return 0.0
    our = pool_bull if side == "Bull" else pool_bear
    if our <= 0:
        return 0.0
    return total * (1.0 - TREASURY_FEE) / our


def main():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    print(f"Loaded {len(rounds)} rounds\n")

    # =====================================================================
    print("=" * 80)
    print("PART 1: Pool fraction present at various cutoffs")
    print("=" * 80)

    cutoff_offsets = [2, 4, 6, 8, 10, 15, 20, 30]
    fraction_by_offset = {c: [] for c in cutoff_offsets}

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        final_bull, final_bear = get_final_pools(rnd)
        final_total = final_bull + final_bear
        if final_total <= 0:
            continue

        for offset in cutoff_offsets:
            cutoff_ts = lock_at - offset
            part_bull, part_bear = get_pools_at_cutoff(rnd, cutoff_ts)
            part_total = part_bull + part_bear
            fraction_by_offset[offset].append(part_total / final_total)

    print(f"\n  {'Cutoff':>10s} {'Median%':>8s} {'Mean%':>8s} {'P10':>8s} {'P25':>8s} {'P75':>8s} {'P90':>8s} {'N':>6s}")
    print("  " + "-" * 65)
    for offset in cutoff_offsets:
        fracs = sorted(fraction_by_offset[offset])
        n = len(fracs)
        if n == 0:
            continue
        mean = sum(fracs) / n
        median = fracs[n // 2]
        p10 = fracs[int(n * 0.10)]
        p25 = fracs[int(n * 0.25)]
        p75 = fracs[int(n * 0.75)]
        p90 = fracs[int(n * 0.90)]
        print(f"  lock-{offset:>2d}s {median*100:7.1f}% {mean*100:7.1f}% {p10*100:7.1f}% {p25*100:7.1f}% {p75*100:7.1f}% {p90*100:7.1f}% {n:6d}")

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 2: Payout multiplier drift (estimated at lock-6 vs actual)")
    print("=" * 80)

    pm_errors = []  # (estimated_pm, actual_pm, error)
    pm_direction_match = 0
    pm_direction_total = 0

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        cutoff_ts = lock_at - POOL_CUTOFF_SECONDS

        part_bull, part_bear = get_pools_at_cutoff(rnd, cutoff_ts)
        final_bull, final_bear = get_final_pools(rnd)

        if part_bull + part_bear <= 0 or final_bull + final_bear <= 0:
            continue

        for side in ["Bull", "Bear"]:
            est_pm = payout_mult(part_bull, part_bear, side)
            act_pm = payout_mult(final_bull, final_bear, side)
            if est_pm > 0 and act_pm > 0:
                pm_errors.append((est_pm, act_pm, est_pm - act_pm))

        # Direction: which side has higher payout?
        est_bull_pm = payout_mult(part_bull, part_bear, "Bull")
        est_bear_pm = payout_mult(part_bull, part_bear, "Bear")
        act_bull_pm = payout_mult(final_bull, final_bear, "Bull")
        act_bear_pm = payout_mult(final_bull, final_bear, "Bear")

        if est_bull_pm > 0 and est_bear_pm > 0 and act_bull_pm > 0 and act_bear_pm > 0:
            est_higher = "Bull" if est_bull_pm > est_bear_pm else "Bear"
            act_higher = "Bull" if act_bull_pm > act_bear_pm else "Bear"
            pm_direction_total += 1
            if est_higher == act_higher:
                pm_direction_match += 1

    errors_sorted = sorted(pm_errors, key=lambda x: x[2])
    abs_errors = sorted([abs(e[2]) for e in pm_errors])
    n = len(pm_errors)

    print(f"\n  Payout multiplier estimation error (estimated - actual):")
    print(f"  N = {n}")
    print(f"  Mean error:   {sum(e[2] for e in pm_errors) / n:+.4f}")
    print(f"  Median error: {errors_sorted[n//2][2]:+.4f}")
    print(f"  Mean |error|: {sum(abs_errors) / n:.4f}")
    print(f"  Median |error|: {abs_errors[n//2]:.4f}")
    print(f"  P90 |error|:  {abs_errors[int(n*0.90)]:.4f}")
    print(f"  P95 |error|:  {abs_errors[int(n*0.95)]:.4f}")
    print(f"  Max |error|:  {abs_errors[-1]:.4f}")

    print(f"\n  Higher-payout side matches at lock-6 vs final: "
          f"{pm_direction_match}/{pm_direction_total} = "
          f"{pm_direction_match/max(1,pm_direction_total)*100:.1f}%")

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 3: Pool imbalance stability")
    print("=" * 80)

    imb_bins = defaultdict(lambda: {"same": 0, "flip": 0, "total": 0})

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        cutoff_ts = lock_at - POOL_CUTOFF_SECONDS

        part_bull, part_bear = get_pools_at_cutoff(rnd, cutoff_ts)
        final_bull, final_bear = get_final_pools(rnd)
        part_total = part_bull + part_bear
        final_total = final_bull + final_bear

        if part_total <= 0 or final_total <= 0:
            continue

        part_imb = (part_bull - part_bear) / part_total  # positive = bull-heavy
        final_imb = (final_bull - final_bear) / final_total

        # Bin by absolute partial imbalance
        abs_imb = abs(part_imb)
        if abs_imb < 0.1:
            b = "<0.1"
        elif abs_imb < 0.2:
            b = "0.1-0.2"
        elif abs_imb < 0.3:
            b = "0.2-0.3"
        elif abs_imb < 0.4:
            b = "0.3-0.4"
        elif abs_imb < 0.5:
            b = "0.4-0.5"
        else:
            b = "0.5+"

        imb_bins[b]["total"] += 1
        if (part_imb > 0) == (final_imb > 0):
            imb_bins[b]["same"] += 1
        else:
            imb_bins[b]["flip"] += 1

    print(f"\n  Does imbalance DIRECTION hold from lock-6 to final?")
    print(f"  {'Imbalance':>10s} {'N':>6s} {'Same%':>7s} {'Flip%':>7s}")
    print("  " + "-" * 35)
    for b in ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5+"]:
        d = imb_bins[b]
        if d["total"] == 0:
            continue
        same_pct = d["same"] / d["total"] * 100
        flip_pct = d["flip"] / d["total"] * 100
        print(f"  {b:>10s} {d['total']:6d} {same_pct:6.1f}% {flip_pct:6.1f}%")

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 4: Payout filter reliability")
    print("  When we THINK pm >= X at lock-6, what is the ACTUAL pm?")
    print("=" * 80)

    # For each payout threshold, check what actual payout looks like
    thresholds = [1.5, 1.85, 2.0, 2.5, 3.0]

    for thresh in thresholds:
        actual_pms = []
        for rnd in rounds:
            lock_at = int(rnd.lock_at)
            cutoff_ts = lock_at - POOL_CUTOFF_SECONDS

            part_bull, part_bear = get_pools_at_cutoff(rnd, cutoff_ts)
            final_bull, final_bear = get_final_pools(rnd)

            if part_bull + part_bear <= 0 or final_bull + final_bear <= 0:
                continue

            for side in ["Bull", "Bear"]:
                est_pm = payout_mult(part_bull, part_bear, side)
                if est_pm >= thresh:
                    act_pm = payout_mult(final_bull, final_bear, side)
                    actual_pms.append(act_pm)

        if not actual_pms:
            continue
        actual_pms.sort()
        n = len(actual_pms)
        mean_act = sum(actual_pms) / n
        median_act = actual_pms[n // 2]
        still_above = sum(1 for p in actual_pms if p >= thresh)
        still_above_lower = sum(1 for p in actual_pms if p >= thresh * 0.8)
        print(f"\n  Estimated pm >= {thresh:.2f} (N={n}):")
        print(f"    Actual mean:   {mean_act:.3f}")
        print(f"    Actual median: {median_act:.3f}")
        print(f"    Actual P10:    {actual_pms[int(n*0.10)]:.3f}")
        print(f"    Actual P25:    {actual_pms[int(n*0.25)]:.3f}")
        print(f"    Still >= {thresh:.2f}: {still_above}/{n} ({still_above/n*100:.1f}%)")
        print(f"    Still >= {thresh*0.8:.2f}: {still_above_lower}/{n} ({still_above_lower/n*100:.1f}%)")

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 5: Late bet patterns - WHO bets late and HOW MUCH?")
    print("=" * 80)

    late_stats = {"count": 0, "late_bets": 0, "late_wei": 0, "total_wei": 0,
                  "whale_late": 0, "whale_total": 0}
    # Whale = single bet > 1 BNB
    WHALE_THRESHOLD = 1e18  # 1 BNB in wei

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        cutoff_ts = lock_at - POOL_CUTOFF_SECONDS
        if not rnd.bets:
            continue

        late_stats["count"] += 1
        for bet in rnd.bets:
            amt = int(bet.amount_wei)
            late_stats["total_wei"] += amt
            is_whale = amt >= WHALE_THRESHOLD

            if is_whale:
                late_stats["whale_total"] += 1

            if int(bet.created_at) > cutoff_ts:
                late_stats["late_bets"] += 1
                late_stats["late_wei"] += amt
                if is_whale:
                    late_stats["whale_late"] += 1

    total_bnb = late_stats["total_wei"] / 1e18
    late_bnb = late_stats["late_wei"] / 1e18
    print(f"\n  Rounds with bets: {late_stats['count']}")
    print(f"  Late bets (after lock-6): {late_stats['late_bets']}")
    print(f"  Late BNB: {late_bnb:.1f} / {total_bnb:.1f} ({late_bnb/max(1,total_bnb)*100:.1f}%)")
    print(f"  Whale bets (>= 1 BNB) late: {late_stats['whale_late']}/{late_stats['whale_total']} "
          f"({late_stats['whale_late']/max(1,late_stats['whale_total'])*100:.1f}%)")

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 6: Bet timing distribution (seconds before lock_at)")
    print("=" * 80)

    timing_buckets = defaultdict(lambda: {"count": 0, "wei": 0})

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        for bet in rnd.bets:
            secs_before = lock_at - int(bet.created_at)
            if secs_before < 0:
                secs_before = 0
            if secs_before > 300:
                continue  # skip bets from before round start

            if secs_before <= 1:
                b = "0-1s"
            elif secs_before <= 2:
                b = "1-2s"
            elif secs_before <= 3:
                b = "2-3s"
            elif secs_before <= 5:
                b = "3-5s"
            elif secs_before <= 10:
                b = "5-10s"
            elif secs_before <= 30:
                b = "10-30s"
            elif secs_before <= 60:
                b = "30-60s"
            elif secs_before <= 120:
                b = "60-120s"
            else:
                b = "120s+"

            timing_buckets[b]["count"] += 1
            timing_buckets[b]["wei"] += int(bet.amount_wei)

    total_count = sum(d["count"] for d in timing_buckets.values())
    total_wei = sum(d["wei"] for d in timing_buckets.values())

    print(f"\n  {'Window':>10s} {'Bets':>8s} {'%Bets':>7s} {'BNB':>10s} {'%BNB':>7s} {'Avg BNB':>8s}")
    print("  " + "-" * 55)
    for b in ["0-1s", "1-2s", "2-3s", "3-5s", "5-10s", "10-30s", "30-60s", "60-120s", "120s+"]:
        d = timing_buckets.get(b, {"count": 0, "wei": 0})
        if d["count"] == 0:
            continue
        bnb = d["wei"] / 1e18
        avg = bnb / d["count"]
        print(f"  {b:>10s} {d['count']:8d} {d['count']/total_count*100:6.1f}% {bnb:10.1f} {bnb/total_wei*1e18*100/max(1,total_wei):6.1f}% {avg:7.3f}")

    # Fix the percentage calculation
    print(f"\n  Recalculated with correct percentages:")
    print(f"  {'Window':>10s} {'Bets':>8s} {'%Bets':>7s} {'BNB':>10s} {'%BNB':>7s} {'Avg BNB':>8s}")
    print("  " + "-" * 55)
    total_bnb_all = total_wei / 1e18
    for b in ["0-1s", "1-2s", "2-3s", "3-5s", "5-10s", "10-30s", "30-60s", "60-120s", "120s+"]:
        d = timing_buckets.get(b, {"count": 0, "wei": 0})
        if d["count"] == 0:
            continue
        bnb = d["wei"] / 1e18
        avg = bnb / d["count"]
        print(f"  {b:>10s} {d['count']:8d} {d['count']/total_count*100:6.1f}% {bnb:10.1f} {bnb/total_bnb_all*100:6.1f}% {avg:7.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
