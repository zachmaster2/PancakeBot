"""Regime fix: make strategy consistent across all 50k rounds.

Key insight from diagnostic:
  - High-vol periods (>0.7 bps) fire signals 59% of rounds but WR is only 54-56%
  - The <3 bps magnitude bucket has 53.8% WR in high-vol (noise)
  - Fix: require signal magnitude >= K * rolling_volatility (signal-to-noise filter)
  - This auto-adapts: high vol -> higher threshold -> fewer noise signals

Also tests:
  - Rolling vol gate: sit out entirely when recent vol too high
  - Minimum magnitude floor that scales with regime

ALL follow EXPERIMENT_RULES.md.
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


def _post_bet_payout(pool_total, our_side, bet_size, treasury_fee=0.03):
    if our_side + bet_size <= 0:
        return 0.0
    return (pool_total + bet_size) * (1.0 - treasury_fee) / (our_side + bet_size)


def _magnitude(bnb_closes):
    max_ret = 0.0
    for short, long in [(7, 10), (5, 10), (5, 7)]:
        for lb in (short, long):
            r = _get_return(bnb_closes, lb)
            if r is not None:
                max_ret = max(max_ret, abs(r))
    return max_ret


def _vol_from_closes(closes):
    """Std dev of 1s log returns."""
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


def run_config(config_name, params, sim_size=49488, verbose=True):
    """Run a single config. Returns (net, seg_results, trade_count, win_count)."""
    rounds, spot, btc = _load_data()
    sim_rounds = rounds[-sim_size:]

    # Config params with defaults matching production
    treasury_fee = 0.03
    skip_hours = params.get("skip_hours", {3, 4, 19})
    base_frac = params.get("base_frac", 0.06)
    floor_bnb = params.get("floor_bnb", 0.05)
    cap_bnb = params.get("cap_bnb", 0.35)
    btc_agree_mult = params.get("btc_agree_mult", 1.25)
    btc_disagree_mult = params.get("btc_disagree_mult", 0.6)
    pool_confirm_thresh = params.get("pool_confirm_thresh", 0.10)
    min_pre_payout = params.get("min_pre_payout", 1.85)

    # New regime-adaptive params
    snr_filter = params.get("snr_filter", None)  # signal-to-noise ratio minimum
    vol_gate = params.get("vol_gate", None)  # sit out if rolling vol > this (bps)
    vol_window = params.get("vol_window", 20)  # rounds to average vol over
    min_mag_bps = params.get("min_mag_bps", None)  # hard minimum magnitude floor
    skip_btc_disagree = params.get("skip_btc_disagree", False)
    adaptive_thresh = params.get("adaptive_thresh", None)  # (base_bps, ref_vol_bps)
    vol_size_scale = params.get("vol_size_scale", False)  # reduce bet size in high vol

    bankroll = 50.0
    trades = []
    recent_vols = []  # rolling window of per-round volatilities

    for rnd in sim_rounds:
        epoch = int(rnd.epoch)
        lock_at = int(rnd.lock_at)
        cutoff_ms = (lock_at - 4) * 1000
        hour = (lock_at % 86400) // 3600

        if hour in skip_hours:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "hour_skip"))
            continue

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

        # Track rolling volatility
        vol = _vol_from_closes(bnb_closes)
        recent_vols.append(vol)
        if len(recent_vols) > vol_window:
            recent_vols.pop(0)

        # Vol gate: sit out if recent volatility too high
        if vol_gate is not None and len(recent_vols) >= vol_window:
            avg_vol = sum(recent_vols) / len(recent_vols)
            if avg_vol * 10000 > vol_gate:
                trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "vol_gate"))
                continue

        # Signal computation - optionally with adaptive threshold
        sig_params = {}
        if adaptive_thresh is not None:
            base_bps, ref_vol_bps = adaptive_thresh
            if len(recent_vols) >= vol_window:
                avg_vol_bps = sum(recent_vols) / len(recent_vols) * 10000
                # Scale threshold: when vol is 2x ref, threshold is 2x base
                scale = avg_vol_bps / ref_vol_bps if ref_vol_bps > 0 else 1.0
                sig_params["accel_thresh"] = base_bps / 10000.0 * scale

        signal, tier, btc_ag, btc_dis = compute_signal(bnb_closes, btc_closes, params=sig_params)

        if signal is None:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "no_signal"))
            continue

        # SNR filter: require magnitude >= K * current volatility
        mag = _magnitude(bnb_closes)
        if snr_filter is not None and tier == "accel":
            if vol > 0:
                snr = mag / vol
                if snr < snr_filter:
                    trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "low_snr"))
                    continue

        # Hard magnitude floor
        if min_mag_bps is not None and tier == "accel":
            if mag < min_mag_bps / 10000.0:
                trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "low_mag"))
                continue

        # BTC disagree filter
        if skip_btc_disagree and btc_dis:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "btc_disagree"))
            continue

        # Pool info
        pool_bull, pool_bear = _pool_info_at_lock(rnd)
        pool_total = pool_bull + pool_bear
        if pool_total <= 0:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "no_pool"))
            continue

        # Pool confirmation
        if pool_confirm_thresh is not None:
            imb = (pool_bull - pool_bear) / pool_total
            pool_dir = "Bull" if imb > 0 else "Bear"
            if abs(imb) >= pool_confirm_thresh and pool_dir != signal:
                trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "pool_disagrees"))
                continue

        # Payout floor
        our_side = pool_bull if signal == "Bull" else pool_bear
        if our_side <= 0:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "no_our_side"))
            continue
        pre_payout = pool_total * 0.97 / our_side
        if pre_payout < min_pre_payout:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "low_payout"))
            continue

        # Sizing (production-equivalent)
        bet = max(floor_bnb, pool_total * base_frac)

        # Payout-proportional
        payout_mult = max(0.3, 0.1 + 1.0 * (pre_payout - 1.0))
        bet *= payout_mult

        # BTC agreement
        if btc_ag:
            bet *= btc_agree_mult
        elif btc_dis:
            bet *= btc_disagree_mult

        # Vol-scaled sizing: reduce in high vol to limit exposure
        if vol_size_scale and len(recent_vols) >= vol_window:
            avg_vol_bps = sum(recent_vols) / len(recent_vols) * 10000
            if avg_vol_bps > 0.5:  # reference vol
                vol_scale = 0.5 / avg_vol_bps
                vol_scale = max(0.3, min(1.0, vol_scale))  # clamp
                bet *= vol_scale

        bet = min(cap_bnb, bet)

        # Post-bet payout check
        post_p = _post_bet_payout(pool_total, our_side, bet)
        if post_p < 1.50:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "post_payout_low"))
            continue

        # Execute
        bankroll -= bet + GAS_COST_BET_BNB
        credit, outcome = settle(bet, signal, rnd, treasury_fee)
        bankroll += credit
        profit = credit - bet - GAS_COST_BET_BNB
        trades.append((epoch, "BET", profit, bankroll, tier, signal, outcome))

    net = bankroll - 50.0
    bet_trades = [t for t in trades if t[1] == "BET"]
    n_bets = len(bet_trades)
    n_wins = sum(1 for t in bet_trades if t[2] > 0)
    wr = n_wins / n_bets * 100 if n_bets else 0

    # 8-segment breakdown
    seg_size = sim_size // 8
    segments = []
    for s in range(8):
        chunk = trades[s * seg_size : (s + 1) * seg_size]
        bets_c = [t for t in chunk if t[1] == "BET"]
        wins_c = [t for t in bets_c if t[2] > 0]
        pnl_c = sum(t[2] for t in bets_c)
        wr_c = len(wins_c) / len(bets_c) * 100 if bets_c else 0
        segments.append((len(bets_c), wr_c, pnl_c))

    # 5-segment (10k each) breakdown
    seg5_size = 10000
    n_seg5 = sim_size // seg5_size
    seg5 = []
    for s in range(n_seg5):
        chunk = trades[s * seg5_size : (s + 1) * seg5_size]
        bets_c = [t for t in chunk if t[1] == "BET"]
        wins_c = [t for t in bets_c if t[2] > 0]
        pnl_c = sum(t[2] for t in bets_c)
        wr_c = len(wins_c) / len(bets_c) * 100 if bets_c else 0
        seg5.append((len(bets_c), wr_c, pnl_c))

    if verbose:
        print(f"\n  {config_name}")
        print(f"  NET: {net:+.2f} BNB | Bets: {n_bets} | WR: {wr:.1f}% | PnL/1k: {net/(sim_size/1000):+.2f}")

        # Skip reasons
        skip_reasons = collections.Counter()
        for t in trades:
            if t[1] == "SKIP":
                skip_reasons[t[6]] += 1
        relevant = {k: v for k, v in skip_reasons.items()
                    if k in ("low_snr", "vol_gate", "low_mag", "btc_disagree", "pool_disagrees", "low_payout")}
        if relevant:
            reasons_str = ", ".join(f"{k}={v}" for k, v in sorted(relevant.items(), key=lambda x: -x[1]))
            print(f"  Skips: {reasons_str}")

        # 10k segments
        print(f"  10k segments: ", end="")
        for i, (nb, wr_s, pnl_s) in enumerate(seg5):
            neg = " <<<" if pnl_s < 0 else ""
            print(f"[{nb:4d}b {wr_s:4.1f}% {pnl_s:+6.1f}]{neg}", end=" ")
        print()

        # Consistency metrics
        pnl_per_seg = [p for _, _, p in seg5]
        neg_segs = sum(1 for p in pnl_per_seg if p < 0)
        if pnl_per_seg:
            std_pnl = (sum((p - sum(pnl_per_seg)/len(pnl_per_seg))**2 for p in pnl_per_seg) / len(pnl_per_seg)) ** 0.5
            print(f"  Consistency: {n_seg5-neg_segs}/{n_seg5} positive, PnL std={std_pnl:.1f}")

    return net, segments, trades, seg5


def main():
    print("=" * 90)
    print("REGIME FIX: Making strategy consistent across 50k rounds")
    print("All configs use post-bet payout, follow EXPERIMENT_RULES.md")
    print("=" * 90)

    configs = collections.OrderedDict()

    # A. Production baseline (no evening skip, production sizing)
    configs["A_production_baseline"] = {}

    # B. Skip BTC disagrees
    configs["B_skip_btc_dis"] = {"skip_btc_disagree": True}

    # C. SNR filter: require mag >= 3x volatility
    configs["C_snr_3x"] = {"snr_filter": 3.0}

    # D. SNR filter: require mag >= 4x volatility
    configs["D_snr_4x"] = {"snr_filter": 4.0}

    # E. SNR filter: require mag >= 5x volatility
    configs["E_snr_5x"] = {"snr_filter": 5.0}

    # F. SNR 4x + skip BTC disagrees
    configs["F_snr4_btcdis"] = {"snr_filter": 4.0, "skip_btc_disagree": True}

    # G. Hard magnitude floor: 3 bps
    configs["G_mag_floor_3bps"] = {"min_mag_bps": 3.0}

    # H. Hard magnitude floor: 4 bps
    configs["H_mag_floor_4bps"] = {"min_mag_bps": 4.0}

    # I. Adaptive threshold (scale with rolling vol, ref=0.5 bps)
    configs["I_adaptive_thresh"] = {"adaptive_thresh": (2.0, 0.50)}

    # J. Adaptive threshold + skip BTC disagrees
    configs["J_adapt_btcdis"] = {"adaptive_thresh": (2.0, 0.50), "skip_btc_disagree": True}

    # K. Vol gate: sit out when rolling vol > 0.9 bps
    configs["K_vol_gate_0.9"] = {"vol_gate": 0.9}

    # L. Vol-scaled sizing: reduce bet size in high vol
    configs["L_vol_size_scale"] = {"vol_size_scale": True}

    # M. Vol-scaled sizing + SNR 4x
    configs["M_volsize_snr4"] = {"vol_size_scale": True, "snr_filter": 4.0}

    # N. SNR 4x + skip BTC dis + vol sizing
    configs["N_snr4_btcdis_volsize"] = {"snr_filter": 4.0, "skip_btc_disagree": True, "vol_size_scale": True}

    # O. Adaptive thresh (gentler: 2.0 base, 0.45 ref)
    configs["O_adapt_gentle"] = {"adaptive_thresh": (2.0, 0.45)}

    # P. SNR 3.5x + skip BTC dis
    configs["P_snr3.5_btcdis"] = {"snr_filter": 3.5, "skip_btc_disagree": True}

    results = {}
    for name, params in configs.items():
        net, segs, trades, seg5 = run_config(name, params)
        results[name] = {
            "net": net,
            "n_bets": sum(1 for t in trades if t[1] == "BET"),
            "seg5": seg5,
        }

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY (sorted by NET PnL)")
    print("=" * 90)
    print(f"  {'Config':30s} {'NET':>8s} {'PnL/1k':>7s} {'Bets':>6s} {'NegSeg':>7s} {'Worst10k':>9s} {'Best10k':>9s}")
    print("  " + "-" * 80)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]["net"])
    for name, r in sorted_results:
        pnl_k = r["net"] / (49488 / 1000)
        pnls = [p for _, _, p in r["seg5"]]
        neg = sum(1 for p in pnls if p < 0)
        worst = min(pnls) if pnls else 0
        best = max(pnls) if pnls else 0
        mark = " *" if neg <= 1 else ""
        print(f"  {name:30s} {r['net']:>+8.2f} {pnl_k:>+7.2f} {r['n_bets']:>6d} {neg:>4d}/4  {worst:>+9.2f} {best:>+9.2f}{mark}")


if __name__ == "__main__":
    main()
