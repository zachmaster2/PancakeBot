"""Regime fix v3: Targeted approaches to make Seg2 positive.

Finding from v2: vol_size_scale with production params gets +47.73 on 50k
but Seg2 (Nov-Dec) is still -3.5. Need to close that gap.

Approaches:
1. Aggressive vol scaling (lower reference point)
2. Drawdown throttle (reduce size after consecutive losses)
3. Rolling WR throttle (reduce size when recent WR drops below threshold)
4. Combined approaches

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


def run_v3(config_name, params, sim_size=49488, verbose=True):
    """Enhanced backtest with drawdown/WR throttle support."""
    rounds, spot, btc = _load_data()
    sim_rounds = rounds[-sim_size:]

    # Production params
    treasury_fee = 0.03
    skip_hours = params.get("skip_hours", {3, 4, 19})
    base_frac = params.get("base_frac", 0.06)
    floor_bnb = params.get("floor_bnb", 0.10)
    cap_bnb = params.get("cap_bnb", 2.0)
    btc_agree_mult = params.get("btc_agree_mult", 1.5)
    btc_disagree_mult = params.get("btc_disagree_mult", 0.7)
    pool_confirm_thresh = params.get("pool_confirm_thresh", None)
    min_pre_payout = params.get("min_pre_payout", 1.85)
    skip_btc_disagree = params.get("skip_btc_disagree", False)

    # Vol scaling params
    vol_ref_bps = params.get("vol_ref_bps", 0.5)  # reference vol for scaling
    vol_window = params.get("vol_window", 20)
    vol_scale_min = params.get("vol_scale_min", 0.3)  # minimum scale factor

    # Drawdown throttle: reduce bet after N consecutive losses
    dd_streak_thresh = params.get("dd_streak_thresh", None)  # trigger after N losses
    dd_scale = params.get("dd_scale", 0.5)  # scale factor when in drawdown

    # Rolling WR throttle: reduce bet when trailing WR is poor
    wr_window = params.get("wr_window", None)  # trailing window size
    wr_thresh = params.get("wr_thresh", 0.52)  # reduce when WR < this
    wr_scale = params.get("wr_scale", 0.5)  # scale factor

    # Trailing PnL throttle: reduce bet when trailing PnL is negative
    pnl_window = params.get("pnl_window", None)  # trailing window size
    pnl_scale = params.get("pnl_scale", 0.5)  # scale factor when trailing PnL < 0

    bankroll = 50.0
    trades = []
    recent_vols = []
    recent_outcomes = []  # 1 = win, 0 = loss
    recent_pnls = []  # per-trade PnL
    consecutive_losses = 0

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

        # Track rolling vol
        vol = _vol_from_closes(bnb_closes)
        recent_vols.append(vol)
        if len(recent_vols) > vol_window:
            recent_vols.pop(0)

        # Signal
        signal, tier, btc_ag, btc_dis = compute_signal(bnb_closes, btc_closes, params={})
        if signal is None:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "no_signal"))
            continue

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

        our_side = pool_bull if signal == "Bull" else pool_bear
        if our_side <= 0:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "no_our_side"))
            continue
        pre_payout = pool_total * 0.97 / our_side
        if pre_payout < min_pre_payout:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "low_payout"))
            continue

        # Base sizing (production-equivalent)
        bet = max(floor_bnb, pool_total * base_frac)
        payout_mult = max(0.3, 0.1 + 1.0 * (pre_payout - 1.0))
        bet *= payout_mult

        if btc_ag:
            bet *= btc_agree_mult
        elif btc_dis:
            bet *= btc_disagree_mult

        # Vol scaling
        if len(recent_vols) >= vol_window:
            avg_vol_bps = sum(recent_vols) / len(recent_vols) * 10000
            if avg_vol_bps > vol_ref_bps:
                vol_scale = vol_ref_bps / avg_vol_bps
                vol_scale = max(vol_scale_min, min(1.0, vol_scale))
                bet *= vol_scale

        # Drawdown throttle
        if dd_streak_thresh is not None and consecutive_losses >= dd_streak_thresh:
            bet *= dd_scale

        # Rolling WR throttle
        if wr_window is not None and len(recent_outcomes) >= wr_window:
            trailing_wr = sum(recent_outcomes[-wr_window:]) / wr_window
            if trailing_wr < wr_thresh:
                bet *= wr_scale

        # Rolling PnL throttle
        if pnl_window is not None and len(recent_pnls) >= pnl_window:
            trailing_pnl = sum(recent_pnls[-pnl_window:])
            if trailing_pnl < 0:
                bet *= pnl_scale

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

        # Update trailing stats
        is_win = 1 if profit > 0 else 0
        recent_outcomes.append(is_win)
        recent_pnls.append(profit)
        if profit > 0:
            consecutive_losses = 0
        else:
            consecutive_losses += 1

    net = bankroll - 50.0
    bet_trades = [t for t in trades if t[1] == "BET"]
    n_bets = len(bet_trades)
    n_wins = sum(1 for t in bet_trades if t[2] > 0)
    wr = n_wins / n_bets * 100 if n_bets else 0

    # 10k segments
    seg_size = 10000
    n_segs = sim_size // seg_size
    seg5 = []
    for s in range(n_segs):
        chunk = trades[s * seg_size : (s + 1) * seg_size]
        bets_c = [t for t in chunk if t[1] == "BET"]
        wins_c = [t for t in bets_c if t[2] > 0]
        pnl_c = sum(t[2] for t in bets_c)
        wr_c = len(wins_c) / len(bets_c) * 100 if bets_c else 0
        seg5.append((len(bets_c), wr_c, pnl_c))
    # Remainder
    remainder = trades[n_segs * seg_size:]
    if remainder:
        bets_r = [t for t in remainder if t[1] == "BET"]
        if bets_r:
            wins_r = [t for t in bets_r if t[2] > 0]
            pnl_r = sum(t[2] for t in bets_r)
            wr_r = len(wins_r) / len(bets_r) * 100
            seg5.append((len(bets_r), wr_r, pnl_r))

    if verbose:
        print(f"\n  {config_name}")
        print(f"  NET: {net:+.2f} BNB | Bets: {n_bets} | WR: {wr:.1f}% | PnL/1k: {net/(sim_size/1000):+.2f}")
        segs_str = " ".join(
            f"[{nb:4d}b {wr_s:4.1f}% {pnl_s:+6.1f}]{'<<<' if pnl_s < 0 else '   '}"
            for nb, wr_s, pnl_s in seg5
        )
        print(f"  Segments: {segs_str}")
        pnl_per_seg = [p for _, _, p in seg5]
        neg_segs = sum(1 for p in pnl_per_seg if p < 0)
        std = (sum((p - sum(pnl_per_seg)/len(pnl_per_seg))**2 for p in pnl_per_seg) / len(pnl_per_seg))**0.5 if pnl_per_seg else 0
        print(f"  {len(pnl_per_seg)-neg_segs}/{len(pnl_per_seg)} positive, Std={std:.1f}")

    return net, seg5, trades


def main():
    print("=" * 95)
    print("REGIME FIX V3: Targeted approaches to eliminate negative segments")
    print("=" * 95)

    configs = collections.OrderedDict()

    # Reference: production + vol_scale (best from v2)
    configs["REF_prod_vol"] = {}

    # 1. More aggressive vol scaling (lower ref point)
    configs["V1_vol_ref040"] = {"vol_ref_bps": 0.40}
    configs["V2_vol_ref035"] = {"vol_ref_bps": 0.35}
    configs["V3_vol_ref030"] = {"vol_ref_bps": 0.30}

    # 2. Drawdown throttle
    configs["DD1_streak3"] = {"dd_streak_thresh": 3, "dd_scale": 0.5}
    configs["DD2_streak4"] = {"dd_streak_thresh": 4, "dd_scale": 0.5}
    configs["DD3_streak3_s07"] = {"dd_streak_thresh": 3, "dd_scale": 0.7}

    # 3. Rolling WR throttle
    configs["WR1_w30_t52"] = {"wr_window": 30, "wr_thresh": 0.52, "wr_scale": 0.5}
    configs["WR2_w50_t52"] = {"wr_window": 50, "wr_thresh": 0.52, "wr_scale": 0.5}
    configs["WR3_w30_t53"] = {"wr_window": 30, "wr_thresh": 0.53, "wr_scale": 0.5}
    configs["WR4_w50_t53"] = {"wr_window": 50, "wr_thresh": 0.53, "wr_scale": 0.5}

    # 4. Rolling PnL throttle
    configs["PL1_w30"] = {"pnl_window": 30, "pnl_scale": 0.5}
    configs["PL2_w50"] = {"pnl_window": 50, "pnl_scale": 0.5}
    configs["PL3_w100"] = {"pnl_window": 100, "pnl_scale": 0.5}

    # 5. Combined: vol scale + drawdown throttle
    configs["C1_vol_dd3"] = {"dd_streak_thresh": 3, "dd_scale": 0.5}
    configs["C2_vol_ref040_dd3"] = {"vol_ref_bps": 0.40, "dd_streak_thresh": 3, "dd_scale": 0.5}

    # 6. Combined: vol scale + WR throttle
    configs["C3_vol_wr30"] = {"wr_window": 30, "wr_thresh": 0.52, "wr_scale": 0.5}
    configs["C4_vol_ref040_wr30"] = {"vol_ref_bps": 0.40, "wr_window": 30, "wr_thresh": 0.52, "wr_scale": 0.5}

    # 7. Combined: vol scale + PnL throttle
    configs["C5_vol_pnl50"] = {"pnl_window": 50, "pnl_scale": 0.5}
    configs["C6_vol_ref040_pnl50"] = {"vol_ref_bps": 0.40, "pnl_window": 50, "pnl_scale": 0.5}

    # 8. Combined: vol scale + btc_dis + best throttle
    configs["C7_vol_btcdis_dd3"] = {"skip_btc_disagree": True, "dd_streak_thresh": 3, "dd_scale": 0.5}
    configs["C8_vol_btcdis_wr30"] = {"skip_btc_disagree": True, "wr_window": 30, "wr_thresh": 0.52, "wr_scale": 0.5}

    results = {}
    for name, params in configs.items():
        net, seg5, trades = run_v3(name, params)
        results[name] = {
            "net": net,
            "n_bets": sum(1 for t in trades if t[1] == "BET"),
            "seg5": seg5,
        }

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(f"  {'Config':30s} {'NET':>8s} {'PnL/1k':>7s} {'Bets':>6s} {'Pos':>5s} {'Std':>6s} {'Worst':>8s} {'Best':>8s}")
    print("  " + "-" * 85)

    def sort_key(item):
        name, r = item
        pnls = [p for _, _, p in r["seg5"]]
        neg = sum(1 for p in pnls if p < 0)
        return (neg, -r["net"])

    for name, r in sorted(results.items(), key=sort_key):
        pnl_k = r["net"] / (49488 / 1000)
        pnls = [p for _, _, p in r["seg5"]]
        neg = sum(1 for p in pnls if p < 0)
        worst = min(pnls) if pnls else 0
        best = max(pnls) if pnls else 0
        std = (sum((p - sum(pnls)/len(pnls))**2 for p in pnls) / len(pnls))**0.5 if pnls else 0
        mark = " **" if neg == 0 else " *" if neg <= 1 else ""
        print(f"  {name:30s} {r['net']:>+8.2f} {pnl_k:>+7.2f} {r['n_bets']:>6d} {len(pnls)-neg:>2d}/{len(pnls):<2d} {std:>6.1f} {worst:>+8.2f} {best:>+8.2f}{mark}")


if __name__ == "__main__":
    main()
