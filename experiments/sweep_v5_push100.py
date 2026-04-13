"""Sweep toward 100 BNB / 20k rounds.

Improvements tested (incrementally stacked):
  A. Baseline (current production)
  B. +S1: Kill BTC-disagrees bets
  C. +S3: Skip hours 6,7,10 when signal magnitude < 0.06%
  D. +Z1+Z2: Pool-relative sizing cap (10% of our_side) + raise floor to 0.25
  E. +Z3: High-conviction sizing (strong mag + BTC agrees → 20% of our_side)
  F. +C1: Add ret_10 auxiliary signal (no accel required, 56.2% WR)
  G. +S2: Skip 0.04-0.06% magnitude bucket

ALL use post-bet payout for decisions. ALL follow EXPERIMENT_RULES.md.
"""
from __future__ import annotations
import sys, json, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.fast_backtest import _load_data, _trim_klines, compute_signal, settle, _get_return
from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.domain.pool_amounts import compute_pool_amounts_wei_at_or_before


def _pool_info(rnd, lock_at):
    """Compute pool bull/bear/total at decision time (bets up to lock_at)."""
    bull_wei, bear_wei = 0, 0
    for b in rnd.bets:
        if int(b.created_at) > lock_at:
            continue
        if b.position == "Bull":
            bull_wei += int(b.amount_wei)
        else:
            bear_wei += int(b.amount_wei)
    bull = bull_wei / BNB_WEI
    bear = bear_wei / BNB_WEI
    return bull, bear, bull + bear


def _post_bet_payout(pool_total, our_side, bet_size, treasury_fee=0.03):
    """Compute payout multiplier AFTER our bet is added to the pool."""
    if our_side + bet_size <= 0:
        return 0.0
    return (pool_total + bet_size) * (1.0 - treasury_fee) / (our_side + bet_size)


def _signal_magnitude(bnb_closes):
    """Compute max |return| across acceleration pairs."""
    max_ret = 0.0
    for short, long in [(7, 10), (5, 10), (5, 7)]:
        for lb in (short, long):
            r = _get_return(bnb_closes, lb)
            if r is not None:
                max_ret = max(max_ret, abs(r))
    return max_ret


# ---- Feature flags for incremental stacking ----

FEATURES = {
    "A_baseline": {},
    "B_kill_btc_disagree": {"kill_btc_disagree": True},
    "C_skip_weak_bad_hours": {"kill_btc_disagree": True, "skip_weak_bad_hours": True},
    "D_pool_sizing": {"kill_btc_disagree": True, "skip_weak_bad_hours": True,
                      "pool_relative_sizing": True},
    "E_conviction_sizing": {"kill_btc_disagree": True, "skip_weak_bad_hours": True,
                            "pool_relative_sizing": True, "conviction_sizing": True},
    "F_ret10_signal": {"kill_btc_disagree": True, "skip_weak_bad_hours": True,
                       "pool_relative_sizing": True, "conviction_sizing": True,
                       "ret10_aux_signal": True},
    "G_skip_mid_mag": {"kill_btc_disagree": True, "skip_weak_bad_hours": True,
                       "pool_relative_sizing": True, "conviction_sizing": True,
                       "ret10_aux_signal": True, "skip_mid_magnitude": True},
}


def run_experiment(flags: dict, sim_size: int = 20000, verbose: bool = True):
    """Run backtest with given feature flags. Returns (net, segments, trades)."""
    rounds, spot, btc = _load_data()
    sim_rounds = rounds[-sim_size:]

    kill_btc_disagree = flags.get("kill_btc_disagree", False)
    skip_weak_bad_hours = flags.get("skip_weak_bad_hours", False)
    pool_relative_sizing = flags.get("pool_relative_sizing", False)
    conviction_sizing = flags.get("conviction_sizing", False)
    ret10_aux_signal = flags.get("ret10_aux_signal", False)
    skip_mid_magnitude = flags.get("skip_mid_magnitude", False)

    # Constants
    treasury_fee = 0.03
    skip_hours = {3, 4, 19}
    weak_bad_hours = {6, 7, 10}
    weak_mag_thresh = 0.0006  # below this = "weak"
    mid_mag_lo = 0.0004  # skip this bucket
    mid_mag_hi = 0.0006
    min_our_payout_pre = 1.85  # pre-bet payout floor (for initial filter)
    min_our_payout_post = 1.50  # post-bet payout floor (hard minimum)
    btc_contra_thresh = 0.0003
    btc_contra_min_payout = 3.0
    btc_contra_bet = 0.15

    # Sizing params
    base_frac = 0.06
    floor_bnb = 0.25 if pool_relative_sizing else 0.10
    pool_cap_frac = 0.10  # max bet = 10% of our_side
    conviction_cap_frac = 0.20  # high-conviction: up to 20% of our_side
    abs_cap_bnb = 2.0
    btc_agree_mult = 1.5
    btc_disagree_mult = 0.7  # only used if not killing btc disagree
    payout_linear_base = 0.1
    payout_linear_slope = 1.0

    # ret_10 aux signal params
    ret10_lookback = 10
    ret10_thresh = 0.0002
    ret10_min_payout_pre = 2.0  # need higher payout since weaker signal

    bankroll = 50.0
    trades = []

    for rnd in sim_rounds:
        epoch = int(rnd.epoch)
        lock_at = int(rnd.lock_at)
        cutoff_ms = (lock_at - 4) * 1000
        hour = (lock_at % 86400) // 3600

        # Hour skip
        if hour in skip_hours:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "hour_skip"))
            continue

        # Klines
        bnb_kl = spot.get(epoch)
        btc_kl = btc.get(epoch)
        if not bnb_kl:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "no_klines"))
            continue

        bnb_trimmed = _trim_klines(bnb_kl, cutoff_ms)
        btc_trimmed = _trim_klines(btc_kl, cutoff_ms) if btc_kl else None
        if len(bnb_trimmed) < 40:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "insufficient"))
            continue

        bnb_closes = [k[4] for k in bnb_trimmed]
        btc_closes = [k[4] for k in btc_trimmed] if btc_trimmed and len(btc_trimmed) >= 40 else None

        # Pool info at decision time
        pool_bull, pool_bear, pool_total = _pool_info(rnd, lock_at)
        if pool_total <= 0:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "no_pool"))
            continue

        # --- Signal computation ---
        signal, tier, btc_ag, btc_dis = compute_signal(bnb_closes, btc_closes, params={})
        mag = _signal_magnitude(bnb_closes) if signal is not None else 0.0

        # BTC contrarian path (when no main signal)
        if signal is None and btc_closes is not None:
            btc_r = _get_return(btc_closes, 30)
            if btc_r is not None and abs(btc_r) >= btc_contra_thresh:
                btc_dir = "Bull" if btc_r > 0 else "Bear"
                contra_dir = "Bear" if btc_dir == "Bull" else "Bull"
                our_side = pool_bull if contra_dir == "Bull" else pool_bear
                if our_side > 0:
                    pre_payout = pool_total * (1.0 - treasury_fee) / our_side
                    if pre_payout >= btc_contra_min_payout:
                        bet = btc_contra_bet
                        post_p = _post_bet_payout(pool_total, our_side, bet)
                        if post_p >= min_our_payout_post:
                            bankroll -= bet + GAS_COST_BET_BNB
                            credit, outcome = settle(bet, contra_dir, rnd, treasury_fee)
                            bankroll += credit
                            profit = credit - bet - GAS_COST_BET_BNB
                            trades.append((epoch, "BET", profit, bankroll, "btc_contra", contra_dir, outcome))
                            continue

        # ret_10 auxiliary signal (when no main signal)
        if signal is None and ret10_aux_signal:
            r10 = _get_return(bnb_closes, ret10_lookback)
            if r10 is not None and abs(r10) >= ret10_thresh:
                aux_dir = "Bull" if r10 > 0 else "Bear"
                our_side = pool_bull if aux_dir == "Bull" else pool_bear
                if our_side > 0:
                    pre_payout = pool_total * (1.0 - treasury_fee) / our_side
                    if pre_payout >= ret10_min_payout_pre:
                        # Size conservatively — weaker signal
                        bet = max(floor_bnb, pool_total * base_frac * 0.5)
                        if pool_relative_sizing:
                            bet = min(bet, our_side * pool_cap_frac * 0.5)
                        bet = min(abs_cap_bnb, bet)
                        post_p = _post_bet_payout(pool_total, our_side, bet)
                        if post_p >= min_our_payout_post:
                            bankroll -= bet + GAS_COST_BET_BNB
                            credit, outcome = settle(bet, aux_dir, rnd, treasury_fee)
                            bankroll += credit
                            profit = credit - bet - GAS_COST_BET_BNB
                            trades.append((epoch, "BET", profit, bankroll, "ret10_aux", aux_dir, outcome))
                            continue

        if signal is None:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "no_signal"))
            continue

        # --- Filters ---

        # S1: Kill BTC disagrees
        if kill_btc_disagree and btc_dis:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "btc_disagree"))
            continue

        # S3: Skip weak signal in bad hours
        if skip_weak_bad_hours and hour in weak_bad_hours and mag < weak_mag_thresh:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "weak_bad_hour"))
            continue

        # S2: Skip mid-magnitude bucket (0.04-0.06%)
        if skip_mid_magnitude and tier == "accel" and mid_mag_lo <= mag < mid_mag_hi:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "mid_magnitude"))
            continue

        # Pre-bet payout floor
        our_side = pool_bull if signal == "Bull" else pool_bear
        if our_side <= 0:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "no_our_side"))
            continue
        pre_payout = pool_total * (1.0 - treasury_fee) / our_side
        if pre_payout < min_our_payout_pre:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "low_payout"))
            continue

        # --- Sizing ---

        # Base size
        bet = max(floor_bnb, pool_total * base_frac)

        # Payout-proportional (using pre-bet payout for initial scaling)
        mult = max(0.3, payout_linear_base + payout_linear_slope * (pre_payout - 1.0))
        bet *= mult

        # BTC agreement
        if btc_ag:
            bet *= btc_agree_mult

        # Pool-relative cap (Z1)
        if pool_relative_sizing:
            max_pool_bet = our_side * pool_cap_frac
            # High-conviction gets a higher pool cap (Z3/E)
            if conviction_sizing and mag >= weak_mag_thresh and btc_ag:
                max_pool_bet = our_side * conviction_cap_frac
            bet = min(bet, max_pool_bet)

        # Absolute cap
        bet = min(abs_cap_bnb, bet)

        # Floor (after all adjustments)
        bet = max(floor_bnb, bet)

        # Post-bet payout check — if our bet crushes the payout, reduce or skip
        post_p = _post_bet_payout(pool_total, our_side, bet)
        if post_p < min_our_payout_post:
            # Try reducing bet to stay above post-payout floor
            # Solve: (pool_total + bet) * 0.97 / (our_side + bet) >= min_post
            # bet <= (pool_total * 0.97 - our_side * min_post) / (min_post - 0.97)
            if min_our_payout_post > 0.97:
                max_bet_for_floor = (pool_total * 0.97 - our_side * min_our_payout_post) / (min_our_payout_post - 0.97)
                if max_bet_for_floor >= floor_bnb:
                    bet = max_bet_for_floor
                    post_p = _post_bet_payout(pool_total, our_side, bet)
                else:
                    trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "post_payout_too_low"))
                    continue
            else:
                trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "post_payout_too_low"))
                continue

        # Execute
        bankroll -= bet + GAS_COST_BET_BNB
        credit, outcome = settle(bet, signal, rnd, treasury_fee)
        bankroll += credit
        profit = credit - bet - GAS_COST_BET_BNB
        trades.append((epoch, "BET", profit, bankroll, tier, signal, outcome))

    # Results
    net = bankroll - 50.0
    bet_trades = [t for t in trades if t[1] == "BET"]
    n_bets = len(bet_trades)
    n_wins = sum(1 for t in bet_trades if t[2] > 0)
    wr = n_wins / n_bets * 100 if n_bets else 0

    # Per-tier breakdown
    tier_stats = collections.defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
    for t in bet_trades:
        tier = t[4] or "unknown"
        tier_stats[tier]["bets"] += 1
        tier_stats[tier]["pnl"] += t[2]
        if t[2] > 0:
            tier_stats[tier]["wins"] += 1

    # Segments
    seg_size = sim_size // 8
    segments = []
    for s in range(8):
        chunk = trades[s * seg_size : (s + 1) * seg_size]
        bets_c = [t for t in chunk if t[1] == "BET"]
        wins_c = [t for t in bets_c if t[2] > 0]
        pnl_c = sum(t[2] for t in bets_c)
        wr_c = len(wins_c) / len(bets_c) * 100 if bets_c else 0
        segments.append((len(bets_c), wr_c, pnl_c))

    if verbose:
        print(f"NET: {net:+.2f} BNB | Bets: {n_bets} | WR: {wr:.1f}%")
        for tier, stats in sorted(tier_stats.items()):
            twr = stats["wins"] / stats["bets"] * 100 if stats["bets"] else 0
            print(f"  {tier:12s}: {stats['bets']:4d} bets, WR={twr:5.1f}%, PnL={stats['pnl']:+8.2f}")
        for i, (nb, wr_s, pnl_s) in enumerate(segments):
            print(f"  Seg{i + 1}: {nb:4d} bets, WR={wr_s:5.1f}%, PnL={pnl_s:+7.2f}")

    return net, segments, trades


if __name__ == "__main__":
    print("=" * 80)
    print("SWEEP V5: Push toward 100 BNB / 20k rounds")
    print("All experiments use post-bet payout, pool-relative sizing caps")
    print("=" * 80)

    for name, flags in FEATURES.items():
        print(f"\n--- {name} ---")
        net, segs, trades = run_experiment(flags, sim_size=20000, verbose=True)

    # Multi-window validation for the best config
    print("\n" + "=" * 80)
    print("MULTI-WINDOW VALIDATION (best config = G)")
    print("=" * 80)
    best_flags = FEATURES["G_skip_mid_mag"]
    print(f"\n{'Size':>7s}  {'NET':>8s}  {'PnL/1k':>7s}  {'Bets':>5s}  {'WR':>5s}  Pos Segs")
    print("-" * 60)
    for size in [10000, 15000, 20000, 30000, 40000, 49488]:
        net, segs, trades = run_experiment(best_flags, sim_size=size, verbose=False)
        n_bets = sum(1 for t in trades if t[1] == "BET")
        n_wins = sum(1 for t in trades if t[1] == "BET" and t[2] > 0)
        wr = n_wins / n_bets * 100 if n_bets else 0
        pnl_k = net / (size / 1000)
        pos_segs = sum(1 for _, _, p in segs if p > 0)
        print(f"{size:>7d}  {net:>+8.2f}  {pnl_k:>+7.2f}  {n_bets:>5d}  {wr:>5.1f}  {pos_segs}/8")
