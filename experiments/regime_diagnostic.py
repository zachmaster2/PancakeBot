"""Deep diagnostic: WHY is Nov-Dec 2025 (rounds 10k-20k) negative?

Computes per-segment metrics across 5x 10k windows to isolate structural
differences in the losing period. Follows EXPERIMENT_RULES.md.
"""
from __future__ import annotations
import sys, json, math, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.fast_backtest import _load_data, _trim_klines, compute_signal, settle, _get_return
from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB


def _pool_info_at_lock(rnd):
    lock_at = int(rnd.lock_at)
    bull_wei, bear_wei = 0, 0
    for b in rnd.bets:
        if int(b.created_at) > lock_at:
            continue
        if b.position == "Bull":
            bull_wei += int(b.amount_wei)
        else:
            bear_wei += int(b.amount_wei)
    return bull_wei / BNB_WEI, bear_wei / BNB_WEI


def _volatility_from_closes(closes):
    """Compute std dev of 1s log returns."""
    if len(closes) < 2:
        return 0.0
    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            rets.append(math.log(closes[i] / closes[i - 1]))
    if not rets:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(var)


def _magnitude(bnb_closes):
    max_ret = 0.0
    for short, long in [(7, 10), (5, 10), (5, 7)]:
        for lb in (short, long):
            r = _get_return(bnb_closes, lb)
            if r is not None:
                max_ret = max(max_ret, abs(r))
    return max_ret


def percentile(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def run_diagnostic():
    rounds, spot, btc = _load_data()
    total = len(rounds)
    seg_size = 10000
    n_segs = total // seg_size

    print(f"Total rounds: {total}, analyzing {n_segs} segments of {seg_size}")
    print(f"Epoch range: {rounds[0].epoch} - {rounds[-1].epoch}")
    print()

    # Collect per-round data for all rounds
    all_round_data = []

    for rnd in rounds:
        epoch = int(rnd.epoch)
        lock_at = int(rnd.lock_at)
        cutoff_ms = (lock_at - 4) * 1000
        hour = (lock_at % 86400) // 3600

        bnb_kl = spot.get(epoch)
        btc_kl = btc.get(epoch)

        rec = {
            "epoch": epoch,
            "lock_at": lock_at,
            "hour": hour,
            "has_klines": bool(bnb_kl),
        }

        pool_bull, pool_bear = _pool_info_at_lock(rnd)
        rec["pool_bull"] = pool_bull
        rec["pool_bear"] = pool_bear
        rec["pool_total"] = pool_bull + pool_bear

        if not bnb_kl:
            rec["signal"] = None
            rec["tier"] = None
            rec["btc_ag"] = False
            rec["btc_dis"] = False
            rec["vol"] = None
            rec["mag"] = 0.0
            rec["bnb_closes"] = None
            all_round_data.append(rec)
            continue

        bnb_trimmed = _trim_klines(bnb_kl, cutoff_ms)
        btc_trimmed = _trim_klines(btc_kl, cutoff_ms) if btc_kl else None

        if len(bnb_trimmed) < 40:
            rec["signal"] = None
            rec["tier"] = None
            rec["btc_ag"] = False
            rec["btc_dis"] = False
            rec["vol"] = None
            rec["mag"] = 0.0
            rec["bnb_closes"] = None
            all_round_data.append(rec)
            continue

        bnb_closes = [k[4] for k in bnb_trimmed]
        btc_closes = [k[4] for k in btc_trimmed] if btc_trimmed and len(btc_trimmed) >= 40 else None

        signal, tier, btc_ag, btc_dis = compute_signal(bnb_closes, btc_closes, params={})
        vol = _volatility_from_closes(bnb_closes)
        mag = _magnitude(bnb_closes) if signal else 0.0

        rec["signal"] = signal
        rec["tier"] = tier
        rec["btc_ag"] = btc_ag
        rec["btc_dis"] = btc_dis
        rec["vol"] = vol
        rec["mag"] = mag
        rec["bnb_closes"] = bnb_closes

        # Outcome truth (for WR analysis independent of sizing)
        rec["winner"] = rnd.position.upper() if rnd.position else None
        rec["failed"] = rnd.failed

        # Pre-bet payout on our side
        if signal and rec["pool_total"] > 0:
            our_side = pool_bull if signal == "Bull" else pool_bear
            if our_side > 0:
                rec["pre_payout"] = rec["pool_total"] * 0.97 / our_side
            else:
                rec["pre_payout"] = None
        else:
            rec["pre_payout"] = None

        # Does signal match crowd?
        if signal and rec["pool_total"] > 0:
            crowd_dir = "Bull" if pool_bull > pool_bear else "Bear"
            rec["signal_with_crowd"] = (signal == crowd_dir)
        else:
            rec["signal_with_crowd"] = None

        all_round_data.append(rec)

    # Now also run the actual backtest to get PnL per round
    from experiments.fast_backtest import run as run_backtest
    _, _, baseline_trades = run_backtest({}, sim_size=total, verbose=False)

    # Map epoch -> trade result
    trade_map = {}
    for t in baseline_trades:
        trade_map[t[0]] = t  # (epoch, action, profit, bankroll, tier, signal, outcome)

    # ================================================================
    # SEGMENT ANALYSIS
    # ================================================================
    for seg_idx in range(n_segs):
        seg_start = seg_idx * seg_size
        seg_end = min((seg_idx + 1) * seg_size, total)
        seg_data = all_round_data[seg_start:seg_end]
        seg_rounds = rounds[seg_start:seg_end]

        # Time range
        first_lock = seg_data[0]["lock_at"]
        last_lock = seg_data[-1]["lock_at"]
        from datetime import datetime, timezone
        t0 = datetime.fromtimestamp(first_lock, tz=timezone.utc).strftime("%Y-%m-%d")
        t1 = datetime.fromtimestamp(last_lock, tz=timezone.utc).strftime("%Y-%m-%d")

        print("=" * 80)
        print(f"SEGMENT {seg_idx + 1}: rounds {seg_start}-{seg_end} | {t0} to {t1}")
        print("=" * 80)

        # 1. VOLATILITY
        vols = [d["vol"] for d in seg_data if d["vol"] is not None]
        print(f"\n  VOLATILITY (std of 1s log returns, bps):")
        print(f"    P10={percentile(vols, 10)*10000:.2f}  P25={percentile(vols, 25)*10000:.2f}  "
              f"P50={percentile(vols, 50)*10000:.2f}  P75={percentile(vols, 75)*10000:.2f}  "
              f"P90={percentile(vols, 90)*10000:.2f}")

        # 2. SIGNAL RATES
        total_with_klines = sum(1 for d in seg_data if d["has_klines"])
        signals = [d for d in seg_data if d["signal"] is not None]
        tier1 = [d for d in signals if d["tier"] == "accel"]
        tier2 = [d for d in signals if d["tier"] == "any+btc"]
        print(f"\n  SIGNAL RATES:")
        print(f"    Total rounds: {len(seg_data)}, with klines: {total_with_klines}")
        print(f"    Signals: {len(signals)} ({len(signals)/total_with_klines*100:.1f}%)")
        print(f"    Tier1 (accel): {len(tier1)} ({len(tier1)/total_with_klines*100:.1f}%)")
        print(f"    Tier2 (any+btc): {len(tier2)} ({len(tier2)/total_with_klines*100:.1f}%)")

        # 3. WIN RATES (raw, no sizing — just signal accuracy)
        def wr_for(subset):
            wins = 0
            total_decided = 0
            for d in subset:
                if d.get("winner") is None or d.get("failed"):
                    continue
                total_decided += 1
                sig_u = d["signal"].upper() if d["signal"] else None
                if sig_u == d["winner"]:
                    wins += 1
            return wins, total_decided

        sig_wins, sig_total = wr_for(signals)
        t1_wins, t1_total = wr_for(tier1)
        t2_wins, t2_total = wr_for(tier2)
        btc_ag_sigs = [d for d in signals if d["btc_ag"]]
        btc_dis_sigs = [d for d in signals if d["btc_dis"]]
        btc_neut_sigs = [d for d in signals if not d["btc_ag"] and not d["btc_dis"]]
        bag_w, bag_t = wr_for(btc_ag_sigs)
        bdis_w, bdis_t = wr_for(btc_dis_sigs)
        bneut_w, bneut_t = wr_for(btc_neut_sigs)

        print(f"\n  RAW WIN RATES (signal accuracy, no sizing):")
        print(f"    All signals:  {sig_wins}/{sig_total} = {sig_wins/sig_total*100:.1f}%" if sig_total else "    All signals: n/a")
        print(f"    Tier1 (accel):{t1_wins}/{t1_total} = {t1_wins/t1_total*100:.1f}%" if t1_total else "    Tier1: n/a")
        print(f"    Tier2 (btc):  {t2_wins}/{t2_total} = {t2_wins/t2_total*100:.1f}%" if t2_total else "    Tier2: n/a")
        print(f"    BTC agrees:   {bag_w}/{bag_t} = {bag_w/bag_t*100:.1f}%" if bag_t else "    BTC agrees: n/a")
        print(f"    BTC disagrees:{bdis_w}/{bdis_t} = {bdis_w/bdis_t*100:.1f}%" if bdis_t else "    BTC disagrees: n/a")
        print(f"    BTC neutral:  {bneut_w}/{bneut_t} = {bneut_w/bneut_t*100:.1f}%" if bneut_t else "    BTC neutral: n/a")

        # 4. MAGNITUDE DISTRIBUTION (for tier1 only)
        if tier1:
            mags = [d["mag"] for d in tier1]
            print(f"\n  SIGNAL MAGNITUDE (tier1, bps):")
            print(f"    P10={percentile(mags, 10)*10000:.1f}  P25={percentile(mags, 25)*10000:.1f}  "
                  f"P50={percentile(mags, 50)*10000:.1f}  P75={percentile(mags, 75)*10000:.1f}  "
                  f"P90={percentile(mags, 90)*10000:.1f}")

            # WR by magnitude bucket
            mag_buckets = [
                ("< 3 bps", 0.0, 0.0003),
                ("3-6 bps", 0.0003, 0.0006),
                ("6-10 bps", 0.0006, 0.0010),
                (">= 10 bps", 0.0010, 1.0),
            ]
            print(f"    WR by magnitude:")
            for label, lo, hi in mag_buckets:
                bucket = [d for d in tier1 if lo <= d["mag"] < hi]
                bw, bt = wr_for(bucket)
                if bt >= 20:
                    print(f"      {label:>10s}: {bw}/{bt} = {bw/bt*100:.1f}% (n={bt})")
                else:
                    print(f"      {label:>10s}: n={bt} (too few)")

        # 5. POOL SIZE DISTRIBUTION
        pools = [d["pool_total"] for d in seg_data if d["pool_total"] > 0]
        print(f"\n  POOL SIZE (BNB):")
        print(f"    P10={percentile(pools, 10):.2f}  P25={percentile(pools, 25):.2f}  "
              f"P50={percentile(pools, 50):.2f}  P75={percentile(pools, 75):.2f}  "
              f"P90={percentile(pools, 90):.2f}")

        # 6. PAYOUT DISTRIBUTION (for signaled rounds)
        payouts = [d["pre_payout"] for d in signals if d["pre_payout"] is not None]
        if payouts:
            print(f"\n  PRE-BET PAYOUT (signaled rounds, our side):")
            print(f"    P10={percentile(payouts, 10):.2f}  P25={percentile(payouts, 25):.2f}  "
                  f"P50={percentile(payouts, 50):.2f}  P75={percentile(payouts, 75):.2f}  "
                  f"P90={percentile(payouts, 90):.2f}")
            low_payout = sum(1 for p in payouts if p < 1.85)
            print(f"    Below 1.85x: {low_payout}/{len(payouts)} ({low_payout/len(payouts)*100:.1f}%)")

        # 7. CROWD ALIGNMENT
        with_crowd = [d for d in signals if d["signal_with_crowd"] is True]
        against_crowd = [d for d in signals if d["signal_with_crowd"] is False]
        print(f"\n  CROWD ALIGNMENT:")
        print(f"    Signal WITH crowd: {len(with_crowd)} ({len(with_crowd)/len(signals)*100:.1f}%)")
        wc_w, wc_t = wr_for(with_crowd)
        ac_w, ac_t = wr_for(against_crowd)
        if wc_t:
            print(f"      WR with crowd: {wc_w}/{wc_t} = {wc_w/wc_t*100:.1f}%")
        if ac_t:
            print(f"      WR against crowd: {ac_w}/{ac_t} = {ac_w/ac_t*100:.1f}%")

        # 8. HOUR-OF-DAY BREAKDOWN
        print(f"\n  HOUR-OF-DAY (signaled rounds, WR):")
        hour_data = collections.defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0})
        for d in signals:
            if d.get("winner") is None or d.get("failed"):
                continue
            h = d["hour"]
            hour_data[h]["total"] += 1
            if d["signal"].upper() == d["winner"]:
                hour_data[h]["wins"] += 1
        print(f"    {'Hour':>4s} {'Sigs':>5s} {'WR':>6s}")
        for h in sorted(hour_data.keys()):
            hd = hour_data[h]
            wr = hd["wins"] / hd["total"] * 100 if hd["total"] else 0
            flag = " <<<" if hd["total"] >= 50 and wr < 52 else ""
            print(f"    {h:>4d} {hd['total']:>5d} {wr:>5.1f}%{flag}")

        # 9. BACKTEST PNL (actual, with sizing)
        seg_epochs = set(d["epoch"] for d in seg_data)
        seg_trades = [trade_map[e] for e in seg_epochs if e in trade_map]
        seg_bets = [t for t in seg_trades if t[1] == "BET"]
        seg_pnl = sum(t[2] for t in seg_bets)
        seg_wins_bt = sum(1 for t in seg_bets if t[2] > 0)
        seg_wr_bt = seg_wins_bt / len(seg_bets) * 100 if seg_bets else 0
        print(f"\n  BACKTEST PNL (baseline strategy):")
        print(f"    Bets: {len(seg_bets)}, Wins: {seg_wins_bt}, WR: {seg_wr_bt:.1f}%")
        print(f"    PnL: {seg_pnl:+.2f} BNB, PnL/1k: {seg_pnl/(seg_size/1000):+.2f}")

        # 10. LOSING STREAK ANALYSIS
        if seg_bets:
            max_streak = 0
            cur_streak = 0
            for t in seg_bets:
                if t[2] <= 0:
                    cur_streak += 1
                    max_streak = max(max_streak, cur_streak)
                else:
                    cur_streak = 0
            print(f"    Max losing streak: {max_streak}")

    # ================================================================
    # CROSS-SEGMENT COMPARISON: What makes Nov-Dec different?
    # ================================================================
    print("\n" + "=" * 80)
    print("CROSS-SEGMENT SUMMARY")
    print("=" * 80)

    headers = ["Seg", "Dates", "Vol_P50", "SigRate", "T1_WR", "T2_WR", "Mag_P50",
               "Pool_P50", "Pay_P50", "Crowd%", "PnL", "PnL/1k"]
    print(f"  {'  '.join(f'{h:>8s}' for h in headers)}")

    for seg_idx in range(n_segs):
        seg_start = seg_idx * seg_size
        seg_end = min((seg_idx + 1) * seg_size, total)
        seg_data = all_round_data[seg_start:seg_end]

        t0 = datetime.fromtimestamp(seg_data[0]["lock_at"], tz=timezone.utc).strftime("%m/%y")
        t1 = datetime.fromtimestamp(seg_data[-1]["lock_at"], tz=timezone.utc).strftime("%m/%y")

        vols = [d["vol"] for d in seg_data if d["vol"] is not None]
        vol_p50 = percentile(vols, 50) * 10000 if vols else 0

        total_kl = sum(1 for d in seg_data if d["has_klines"])
        signals = [d for d in seg_data if d["signal"] is not None]
        sig_rate = len(signals) / total_kl * 100 if total_kl else 0

        tier1 = [d for d in signals if d["tier"] == "accel"]
        tier2 = [d for d in signals if d["tier"] == "any+btc"]
        t1w, t1t = 0, 0
        for d in tier1:
            if d.get("winner") and not d.get("failed"):
                t1t += 1
                if d["signal"].upper() == d["winner"]:
                    t1w += 1
        t2w, t2t = 0, 0
        for d in tier2:
            if d.get("winner") and not d.get("failed"):
                t2t += 1
                if d["signal"].upper() == d["winner"]:
                    t2w += 1

        mags = [d["mag"] for d in tier1 if d["mag"] > 0]
        mag_p50 = percentile(mags, 50) * 10000 if mags else 0

        pools = [d["pool_total"] for d in seg_data if d["pool_total"] > 0]
        pool_p50 = percentile(pools, 50) if pools else 0

        payouts = [d["pre_payout"] for d in signals if d.get("pre_payout")]
        pay_p50 = percentile(payouts, 50) if payouts else 0

        with_crowd = sum(1 for d in signals if d.get("signal_with_crowd") is True)
        crowd_pct = with_crowd / len(signals) * 100 if signals else 0

        seg_epochs = set(d["epoch"] for d in seg_data)
        seg_bets = [trade_map[e] for e in seg_epochs if e in trade_map and trade_map[e][1] == "BET"]
        seg_pnl = sum(t[2] for t in seg_bets)
        pnl_k = seg_pnl / (seg_size / 1000)

        vals = [
            f"{seg_idx+1:>8d}",
            f"{t0}-{t1:>3s}",
            f"{vol_p50:>8.2f}",
            f"{sig_rate:>8.1f}",
            f"{t1w/t1t*100 if t1t else 0:>8.1f}",
            f"{t2w/t2t*100 if t2t else 0:>8.1f}",
            f"{mag_p50:>8.1f}",
            f"{pool_p50:>8.2f}",
            f"{pay_p50:>8.2f}",
            f"{crowd_pct:>8.1f}",
            f"{seg_pnl:>+8.2f}",
            f"{pnl_k:>+8.2f}",
        ]
        marker = " <<<" if seg_pnl < 0 else ""
        print(f"  {'  '.join(vals)}{marker}")

    # ================================================================
    # DEEP DIVE: What signal changes would fix the bad segment?
    # ================================================================
    print("\n" + "=" * 80)
    print("WHAT-IF ANALYSIS: Fixing the worst segment")
    print("=" * 80)

    # Find worst segment
    seg_pnls = []
    for seg_idx in range(n_segs):
        seg_start = seg_idx * seg_size
        seg_end = min((seg_idx + 1) * seg_size, total)
        seg_data = all_round_data[seg_start:seg_end]
        seg_epochs = set(d["epoch"] for d in seg_data)
        seg_bets = [trade_map[e] for e in seg_epochs if e in trade_map and trade_map[e][1] == "BET"]
        seg_pnls.append(sum(t[2] for t in seg_bets))

    worst_idx = seg_pnls.index(min(seg_pnls))
    print(f"\nWorst segment: {worst_idx + 1} (PnL: {seg_pnls[worst_idx]:+.2f})")

    worst_start = worst_idx * seg_size
    worst_end = min((worst_idx + 1) * seg_size, total)
    worst_data = all_round_data[worst_start:worst_end]
    worst_rounds = rounds[worst_start:worst_end]

    # Test various threshold adjustments on just the worst segment
    print("\n  Threshold sweep on worst segment (tier1 only):")
    print(f"  {'Thresh':>8s} {'Signals':>8s} {'WR':>6s} {'EV_est':>8s}")

    for thresh_bps in [2, 3, 4, 5, 6, 7, 8, 10]:
        thresh = thresh_bps / 10000.0
        wins, total_decided = 0, 0
        payouts_won = []
        for d in worst_data:
            if d["signal"] is None or d["tier"] != "accel":
                continue
            if d["mag"] < thresh:
                continue
            if d.get("winner") is None or d.get("failed"):
                continue
            total_decided += 1
            if d["signal"].upper() == d["winner"]:
                wins += 1
                if d.get("pre_payout"):
                    payouts_won.append(d["pre_payout"])

        if total_decided >= 20:
            wr = wins / total_decided * 100
            avg_payout = sum(payouts_won) / len(payouts_won) if payouts_won else 2.0
            ev = (wr / 100) * avg_payout - 1.0
            print(f"  {thresh_bps:>8d} {total_decided:>8d} {wr:>5.1f}% {ev:>+8.3f}")
        else:
            print(f"  {thresh_bps:>8d} {total_decided:>8d}   (too few)")

    # Test: what if we skip the worst segment entirely?
    print(f"\n  Skip worst segment entirely:")
    other_pnl = sum(p for i, p in enumerate(seg_pnls) if i != worst_idx)
    print(f"    Other segments PnL: {other_pnl:+.2f}")
    print(f"    PnL/1k (ex-worst):  {other_pnl / ((total - seg_size) / 1000):+.2f}")

    # Test: what if we use tighter threshold in worst segment, normal elsewhere?
    print(f"\n  Adaptive threshold (tight in worst, normal elsewhere):")
    # Re-run backtest with modified threshold for worst segment epochs
    worst_epochs = set(d["epoch"] for d in worst_data)

    for tight_bps in [4, 5, 6]:
        tight_thresh = tight_bps / 10000.0
        bankroll = 50.0
        n_bets = 0
        n_wins = 0
        pnl = 0.0

        for i, rnd in enumerate(rounds):
            epoch = int(rnd.epoch)
            lock_at = int(rnd.lock_at)
            cutoff_ms = (lock_at - 4) * 1000
            hour = (lock_at % 86400) // 3600

            if hour in (3, 4, 19):
                continue

            bnb_kl = spot.get(epoch)
            btc_kl = btc.get(epoch)
            if not bnb_kl:
                continue

            bnb_trimmed = _trim_klines(bnb_kl, cutoff_ms)
            btc_trimmed = _trim_klines(btc_kl, cutoff_ms) if btc_kl else None
            if len(bnb_trimmed) < 40:
                continue

            bnb_closes = [k[4] for k in bnb_trimmed]
            btc_closes = [k[4] for k in btc_trimmed] if btc_trimmed and len(btc_trimmed) >= 40 else None

            # Use tighter threshold in worst segment
            use_thresh = tight_thresh if epoch in worst_epochs else 0.0002
            signal, tier, btc_ag, btc_dis = compute_signal(
                bnb_closes, btc_closes, params={"accel_thresh": use_thresh}
            )

            if signal is None:
                continue

            # Pool confirmation
            bull_wei, bear_wei = 0, 0
            for b in rnd.bets:
                if int(b.created_at) > lock_at:
                    continue
                if b.position == "Bull":
                    bull_wei += int(b.amount_wei)
                else:
                    bear_wei += int(b.amount_wei)
            pool_bull = bull_wei / BNB_WEI
            pool_bear = bear_wei / BNB_WEI
            pool_total = pool_bull + pool_bear

            if pool_total <= 0:
                continue

            # Pool confirmation filter
            imb = (pool_bull - pool_bear) / pool_total
            pool_dir = "Bull" if imb > 0 else "Bear"
            if abs(imb) >= 0.10 and pool_dir != signal:
                continue

            our_side = pool_bull if signal == "Bull" else pool_bear
            if our_side <= 0:
                continue
            pre_payout = pool_total * 0.97 / our_side
            if pre_payout < 1.85:
                continue

            # Sizing (baseline)
            bet = max(0.05, pool_total * 0.06)
            mult_s = max(0.3, 0.1 + 1.0 * (pre_payout - 1.0))
            bet *= mult_s
            if btc_ag:
                bet *= 1.25
            elif btc_dis:
                bet *= 0.6
            bet = min(0.35, bet)

            bankroll -= bet + GAS_COST_BET_BNB
            credit, outcome = settle(bet, signal, rnd)
            bankroll += credit
            profit = credit - bet - GAS_COST_BET_BNB
            pnl += profit
            n_bets += 1
            if profit > 0:
                n_wins += 1

        wr = n_wins / n_bets * 100 if n_bets else 0
        print(f"    thresh={tight_bps}bps in worst: NET={bankroll-50:+.2f} BNB, "
              f"{n_bets} bets, WR={wr:.1f}%, PnL/1k={pnl/(total/1000):+.2f}")

    # ================================================================
    # REGIME DETECTION: Can we detect the bad regime in real-time?
    # ================================================================
    print("\n" + "=" * 80)
    print("REGIME DETECTION: Rolling metrics")
    print("=" * 80)

    # Compute rolling 500-round volatility and signal WR
    window = 500
    print(f"\n  Rolling {window}-round metrics (sampled every {window} rounds):")
    print(f"  {'RndIdx':>8s} {'Vol_P50':>8s} {'SigRate':>8s} {'RawWR':>6s} {'BT_PnL':>8s}")

    for start in range(0, total - window, window):
        end = start + window
        chunk = all_round_data[start:end]

        vols = [d["vol"] for d in chunk if d["vol"] is not None]
        vol_p50 = percentile(vols, 50) * 10000 if vols else 0

        total_kl = sum(1 for d in chunk if d["has_klines"])
        sigs = [d for d in chunk if d["signal"] is not None]
        sig_rate = len(sigs) / total_kl * 100 if total_kl else 0

        wins, total_d = 0, 0
        for d in sigs:
            if d.get("winner") and not d.get("failed"):
                total_d += 1
                if d["signal"].upper() == d["winner"]:
                    wins += 1
        raw_wr = wins / total_d * 100 if total_d else 0

        chunk_epochs = set(d["epoch"] for d in chunk)
        chunk_bets = [trade_map[e] for e in chunk_epochs if e in trade_map and trade_map[e][1] == "BET"]
        chunk_pnl = sum(t[2] for t in chunk_bets)

        flag = " <<<" if chunk_pnl < -1 else ""
        print(f"  {start:>8d} {vol_p50:>8.2f} {sig_rate:>8.1f} {raw_wr:>5.1f}% {chunk_pnl:>+8.2f}{flag}")


if __name__ == "__main__":
    from datetime import datetime, timezone
    run_diagnostic()
