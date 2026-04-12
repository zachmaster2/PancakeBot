"""Sweep experiments targeting +40 BNB over 20k rounds.

Tests:
  1. Payout-proportional sizing (bet scales linearly with payout)
  2. Relaxed payout floor (1.80-1.84) to capture more volume
  3. Continuous payout scaling vs threshold-based
  4. Hour-of-day analysis to find best hours
  5. Separate BTC-only contrarian signal on non-accel rounds
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.fast_backtest import _load_data, _trim_klines, compute_signal, settle, _get_return
from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB


def run_v2(params, sim_size=20000, verbose=True):
    """Enhanced backtest with payout-proportional sizing and BTC-only signal."""
    rounds, spot, btc = _load_data()
    sim_rounds = rounds[-sim_size:]

    cutoff_sec = params.get("cutoff_seconds", 4)
    base_frac = params.get("base_frac", 0.06)
    floor_bnb = params.get("floor_bnb", 0.10)
    cap_bnb = params.get("cap_bnb", 2.0)
    btc_agree_mult = params.get("btc_agree_mult", 1.5)
    btc_disagree_mult = params.get("btc_disagree_mult", 0.7)
    evening_skip = params.get("evening_skip", None)
    pool_confirm_thresh = params.get("pool_confirm_thresh", None)
    treasury_fee = params.get("treasury_fee", 0.03)
    min_our_payout = params.get("min_our_payout", 1.85)

    # Payout-proportional sizing params
    payout_sizing_mode = params.get("payout_sizing_mode", "threshold")  # "threshold" or "linear" or "quadratic"
    payout_hi_thresh = params.get("payout_hi_thresh", 2.5)
    payout_hi_mult = params.get("payout_hi_mult", 1.7)
    payout_lo_thresh = params.get("payout_lo_thresh", 1.7)
    payout_lo_mult = params.get("payout_lo_mult", 0.9)
    # For linear mode: bet_mult = payout_linear_base + payout_linear_slope * (payout - 1.0)
    payout_linear_base = params.get("payout_linear_base", 0.5)
    payout_linear_slope = params.get("payout_linear_slope", 0.5)

    # BTC-only signal on non-accel rounds
    btc_only_signal = params.get("btc_only_signal", False)
    btc_only_lookback = params.get("btc_only_lookback", 30)
    btc_only_thresh = params.get("btc_only_thresh", 0.0005)
    btc_only_min_payout = params.get("btc_only_min_payout", 2.2)
    btc_only_bet = params.get("btc_only_bet", 0.10)

    # Hour filter (whitelist)
    allowed_hours = params.get("allowed_hours", None)  # e.g. [0,1,2,...,17]

    bankroll = 50.0
    trades = []
    hour_stats = {}  # hour -> [wins, losses, pnl]

    for rnd in sim_rounds:
        epoch = int(rnd.epoch)
        lock_at = int(rnd.lock_at)
        cutoff_ms = (lock_at - cutoff_sec) * 1000
        hour = (lock_at % 86400) // 3600

        # Evening / hour filter
        if evening_skip:
            if evening_skip[0] <= hour < evening_skip[1]:
                trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "evening"))
                continue
        if allowed_hours is not None and hour not in allowed_hours:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "hour_skip"))
            continue

        # Get klines
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

        signal, tier, btc_ag, btc_dis = compute_signal(bnb_closes, btc_closes, params=params)

        # Pre-compute pools (we'll need them for multiple checks)
        bull_wei, bear_wei = 0, 0
        for b in rnd.bets:
            if int(b.created_at) > lock_at: continue
            if b.position == "Bull": bull_wei += int(b.amount_wei)
            else: bear_wei += int(b.amount_wei)
        pool_bull = bull_wei / BNB_WEI
        pool_bear = bear_wei / BNB_WEI
        pool_total = pool_bull + pool_bear

        # If no accel signal, try BTC-only contrarian signal
        if signal is None and btc_only_signal and btc_closes is not None and pool_total > 0:
            btc_r = _get_return(btc_closes, btc_only_lookback)
            if btc_r is not None and abs(btc_r) >= btc_only_thresh:
                btc_dir = "Bull" if btc_r > 0 else "Bear"
                # Contrarian: bet AGAINST BTC direction (crowd follows BTC, we fade)
                contra_dir = "Bear" if btc_dir == "Bull" else "Bull"
                our_side = pool_bull if contra_dir == "Bull" else pool_bear
                if our_side > 0:
                    pm = pool_total * (1.0 - treasury_fee) / our_side
                    if pm >= btc_only_min_payout:
                        bet = btc_only_bet
                        bankroll -= bet + GAS_COST_BET_BNB
                        credit, outcome = settle(bet, contra_dir, rnd, treasury_fee)
                        bankroll += credit
                        profit = credit - bet - GAS_COST_BET_BNB
                        trades.append((epoch, "BET", profit, bankroll, "btc_contra", contra_dir, outcome))
                        hour_stats.setdefault(hour, [0, 0, 0.0])
                        if profit > 0: hour_stats[hour][0] += 1
                        else: hour_stats[hour][1] += 1
                        hour_stats[hour][2] += profit
                        continue

        if signal is None:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "no_signal"))
            continue

        # Payout floor filter
        if min_our_payout is not None and pool_total > 0:
            our_side = pool_bull if signal == "Bull" else pool_bear
            if our_side > 0:
                pm = pool_total * (1.0 - treasury_fee) / our_side
                if pm < min_our_payout:
                    trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "low_payout"))
                    continue

        # Pool confirmation filter
        if pool_confirm_thresh is not None and pool_total > 0:
            imb = (pool_bull - pool_bear) / pool_total
            pool_dir = "Bull" if imb > 0 else "Bear"
            if abs(imb) >= pool_confirm_thresh and pool_dir != signal:
                trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "pool_disagrees"))
                continue

        # Sizing
        bet = max(floor_bnb, pool_total * base_frac) if pool_total > 0 else floor_bnb

        # Payout-based sizing
        our_side = pool_bull if signal == "Bull" else pool_bear
        if our_side > 0:
            pm = pool_total * (1.0 - treasury_fee) / our_side
            if payout_sizing_mode == "threshold":
                if pm >= payout_hi_thresh: bet *= payout_hi_mult
                elif pm < payout_lo_thresh: bet *= payout_lo_mult
            elif payout_sizing_mode == "linear":
                mult = max(0.3, payout_linear_base + payout_linear_slope * (pm - 1.0))
                bet *= mult
            elif payout_sizing_mode == "quadratic":
                mult = max(0.3, payout_linear_base + payout_linear_slope * ((pm - 1.0) ** 1.5))
                bet *= mult

        if btc_ag: bet *= btc_agree_mult
        elif btc_dis: bet *= btc_disagree_mult

        bet = min(cap_bnb, bet)

        # Execute
        bankroll -= bet + GAS_COST_BET_BNB
        credit, outcome = settle(bet, signal, rnd, treasury_fee)
        bankroll += credit
        profit = credit - bet - GAS_COST_BET_BNB
        trades.append((epoch, "BET", profit, bankroll, tier, signal, outcome))
        hour_stats.setdefault(hour, [0, 0, 0.0])
        if profit > 0: hour_stats[hour][0] += 1
        else: hour_stats[hour][1] += 1
        hour_stats[hour][2] += profit

    net = bankroll - 50.0

    # Segment analysis
    seg_size = sim_size // 4
    segments = []
    for s in range(4):
        chunk = trades[s*seg_size:(s+1)*seg_size]
        bets = [t for t in chunk if t[1] == "BET"]
        wins = [t for t in bets if t[2] > 0]
        pnl = sum(t[2] for t in bets)
        wr = len(wins)/len(bets)*100 if bets else 0
        segments.append((len(bets), wr, pnl))

    if verbose:
        total_bets = sum(1 for t in trades if t[1] == "BET")
        total_wins = sum(1 for t in trades if t[1] == "BET" and t[2] > 0)
        print(f"NET: {net:+.2f} BNB | Bets: {total_bets} | WR: {total_wins/max(1,total_bets)*100:.1f}%")
        # Tier breakdown
        tiers = {}
        for t in trades:
            if t[1] != "BET": continue
            tier_name = t[4] or "unknown"
            tiers.setdefault(tier_name, [0, 0, 0.0])
            tiers[tier_name][0] += 1
            if t[2] > 0: tiers[tier_name][1] += 1
            tiers[tier_name][2] += t[2]
        for tn in sorted(tiers):
            nb, nw, pnl = tiers[tn]
            print(f"  {tn:12s}: {nb:4d} bets, WR={nw/max(1,nb)*100:5.1f}%, PnL={pnl:+7.2f}")
        for i, (nb, wr, pnl) in enumerate(segments):
            print(f"  Seg{i+1}: {nb:4d} bets, WR={wr:5.1f}%, PnL={pnl:+7.2f}")

    return net, segments, trades, hour_stats


if __name__ == "__main__":
    import itertools

    # ---- Experiment 1: Baseline (current best) ----
    print("=" * 70)
    print("EXP 1: BASELINE (current best +33.49 config)")
    print("=" * 70)
    baseline = {
        "min_our_payout": 1.85,
        "base_frac": 0.06, "cap_bnb": 2.0, "floor_bnb": 0.10,
        "btc_agree_mult": 1.5, "btc_disagree_mult": 0.7,
        "payout_hi_thresh": 2.5, "payout_hi_mult": 1.7,
        "evening_skip": None, "pool_confirm_thresh": None,
    }
    net1, seg1, _, hrs1 = run_v2(baseline)

    # ---- Experiment 2: Linear payout-proportional sizing ----
    print("\n" + "=" * 70)
    print("EXP 2: LINEAR PAYOUT-PROPORTIONAL SIZING")
    print("=" * 70)
    best_linear = None
    for base, slope in [(0.3, 0.5), (0.4, 0.4), (0.2, 0.6), (0.5, 0.3), (0.3, 0.7), (0.1, 0.8)]:
        p = {**baseline, "payout_sizing_mode": "linear",
             "payout_linear_base": base, "payout_linear_slope": slope}
        net, segs, _, _ = run_v2(p, verbose=False)
        min_seg = min(s[2] for s in segs)
        print(f"  base={base}, slope={slope}: NET={net:+.2f}, minSeg={min_seg:+.2f}")
        if best_linear is None or net > best_linear[0]:
            best_linear = (net, base, slope, segs)

    print(f"  BEST: base={best_linear[1]}, slope={best_linear[2]}, NET={best_linear[0]:+.2f}")
    # Re-run best with verbose
    p = {**baseline, "payout_sizing_mode": "linear",
         "payout_linear_base": best_linear[1], "payout_linear_slope": best_linear[2]}
    run_v2(p)

    # ---- Experiment 3: Relaxed payout floor ----
    print("\n" + "=" * 70)
    print("EXP 3: PAYOUT FLOOR SWEEP (both threshold & linear sizing)")
    print("=" * 70)
    for floor in [1.75, 1.78, 1.80, 1.82, 1.84, 1.85, 1.88, 1.90, 1.95, 2.0]:
        p = {**baseline, "min_our_payout": floor}
        net, segs, _, _ = run_v2(p, verbose=False)
        min_seg = min(s[2] for s in segs)
        nbets = sum(s[0] for s in segs)
        print(f"  floor={floor:.2f} (threshold): NET={net:+.2f}, bets={nbets}, minSeg={min_seg:+.2f}")

    # With best linear sizing
    if best_linear:
        for floor in [1.75, 1.78, 1.80, 1.82, 1.84, 1.85, 1.88, 1.90, 1.95, 2.0]:
            p = {**baseline, "min_our_payout": floor, "payout_sizing_mode": "linear",
                 "payout_linear_base": best_linear[1], "payout_linear_slope": best_linear[2]}
            net, segs, _, _ = run_v2(p, verbose=False)
            min_seg = min(s[2] for s in segs)
            nbets = sum(s[0] for s in segs)
            print(f"  floor={floor:.2f} (linear):    NET={net:+.2f}, bets={nbets}, minSeg={min_seg:+.2f}")

    # ---- Experiment 4: BTC-only contrarian signal ----
    print("\n" + "=" * 70)
    print("EXP 4: BTC-ONLY CONTRARIAN SIGNAL (added to best accel config)")
    print("=" * 70)
    for btc_thresh in [0.0003, 0.0005, 0.0008, 0.001]:
        for btc_min_pay in [2.0, 2.2, 2.5, 3.0]:
            for btc_bet in [0.08, 0.10, 0.15]:
                p = {**baseline,
                     "btc_only_signal": True,
                     "btc_only_thresh": btc_thresh,
                     "btc_only_min_payout": btc_min_pay,
                     "btc_only_bet": btc_bet}
                net, segs, _, _ = run_v2(p, verbose=False)
                min_seg = min(s[2] for s in segs)
                nbets = sum(s[0] for s in segs)
                if net > 34:  # only show promising results
                    print(f"  btc_thresh={btc_thresh}, min_pay={btc_min_pay}, bet={btc_bet}: "
                          f"NET={net:+.2f}, bets={nbets}, minSeg={min_seg:+.2f}")

    # ---- Experiment 5: Hour-of-day analysis ----
    print("\n" + "=" * 70)
    print("EXP 5: HOUR-OF-DAY PnL BREAKDOWN")
    print("=" * 70)
    _, _, _, hrs = run_v2(baseline, verbose=False)
    print(f"  {'Hour':>4s}  {'Wins':>4s}  {'Losses':>6s}  {'WR':>5s}  {'PnL':>8s}")
    for h in range(24):
        if h in hrs:
            w, l, pnl = hrs[h]
            wr = w / max(1, w + l) * 100
            print(f"  {h:4d}  {w:4d}  {l:6d}  {wr:5.1f}%  {pnl:+8.2f}")

    # ---- Experiment 6: Combined best ideas ----
    print("\n" + "=" * 70)
    print("EXP 6: COMBINED OPTIMIZATIONS")
    print("=" * 70)
    # We'll combine the best results from above after seeing output
