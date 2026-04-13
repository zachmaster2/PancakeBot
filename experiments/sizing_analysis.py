"""Sizing analysis: bet size distribution, payout dilution, efficiency by quartile.

Uses MomentumOnlyPipeline exactly as production does.
Records per-bet details and reports sizing statistics.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB, BNB_WEI
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
from pancakebot.domain.strategy.momentum_pipeline import (
    MomentumOnlyPipeline,
    _compute_bet_size,
    _pools_from_bets,
    _BASE_FRAC, _FLOOR_BNB, _CAP_BNB,
    _BTC_AGREE_MULT, _BTC_DISAGREE_MULT,
    _PAYOUT_LINEAR_BASE, _PAYOUT_LINEAR_SLOPE,
    _MIN_OUR_PAYOUT,
    _BTC_CONTRA_BET_BNB,
)
from pancakebot.domain.strategy.momentum_gate import (
    compute_signal_from_klines,
    _trim_to_window,
    _CANDLE_COUNT,
    _get_return,
    _BTC_LOOKBACK,
    _BTC_THRESH,
)
from pancakebot.runtime.settlement import settle_bet_against_closed_round
from pancakebot.domain.pool_amounts import compute_pool_amounts_wei

# Production constants
_CUTOFF_SECONDS = 4
_MIN_BET_AMOUNT_BNB = 0.001
_TREASURY_FEE_FRACTION = 0.03


def percentile(sorted_vals, p):
    """Compute p-th percentile from a sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def load_data():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())

    def load_klines(path):
        out = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    spot = load_klines("var/cutoff_spot_prices.jsonl")
    btc = load_klines("var/btc_spot_prices.jsonl")
    return rounds, spot, btc


def run_analysis():
    rounds, spot, btc = load_data()
    print(f"Loaded {len(rounds)} rounds, {len(spot)} spot epochs, {len(btc)} btc epochs")

    # Build pipeline identical to production
    gate_config = MomentumGateConfig(
        enabled=True, symbol="BNB-USDT", btc_symbol="BTC-USDT",
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_config,
        gate=None,
        cutoff_seconds=_CUTOFF_SECONDS,
        min_bet_amount_bnb=_MIN_BET_AMOUNT_BNB,
        treasury_fee_fraction=_TREASURY_FEE_FRACTION,
    )
    pipeline.refresh_spot_klines(spot_klines_by_epoch=spot)
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc)

    bankroll = 50.0
    initial_bankroll = bankroll
    bet_records = []

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        cutoff_ts_ms = (lock_at - _CUTOFF_SECONDS) * 1000

        # Compute pools at decision time (bets at or before lock_at)
        pool_bull_bnb, pool_bear_bnb = _pools_from_bets(rnd, lock_at)
        pool_total = pool_bull_bnb + pool_bear_bnb

        decision = pipeline.decide_open_round(
            round_t=rnd,
            bankroll_bnb=bankroll,
            allow_oracle_mode=True,
        )

        if decision.action == "BET" and decision.bet_size_bnb > 0.0:
            bet_size = decision.bet_size_bnb
            bet_side = decision.bet_side

            # Our side pool and pre-bet payout
            our_side_pool = pool_bull_bnb if bet_side == "Bull" else pool_bear_bnb
            other_side_pool = pool_bear_bnb if bet_side == "Bull" else pool_bull_bnb

            if our_side_pool > 0 and pool_total > 0:
                pre_bet_payout = pool_total * (1.0 - _TREASURY_FEE_FRACTION) / our_side_pool
            else:
                pre_bet_payout = 0.0

            # Post-bet payout (after our bet dilutes our side)
            post_pool_total = pool_total + bet_size
            post_our_side = our_side_pool + bet_size
            if post_our_side > 0 and post_pool_total > 0:
                post_bet_payout = post_pool_total * (1.0 - _TREASURY_FEE_FRACTION) / post_our_side
            else:
                post_bet_payout = 0.0

            # Determine btc_agrees / btc_disagrees by re-evaluating BTC signal
            # We need to get these from the gate result
            # Re-evaluate from cache to get btc_agrees/btc_disagrees
            epoch = int(rnd.epoch)
            bnb_klines = spot.get(epoch)
            btc_klines_raw = btc.get(epoch)
            btc_agrees = False
            btc_disagrees = False
            is_contrarian = False

            if bnb_klines and len(bnb_klines) > 0:
                gate_result = compute_signal_from_klines(bnb_klines, btc_klines_raw, cutoff_ts_ms)
                if gate_result.signal is not None:
                    btc_agrees = gate_result.btc_agrees
                    btc_disagrees = gate_result.btc_disagrees
                else:
                    # This is a contrarian bet (main gate had no signal)
                    is_contrarian = True

            # Compute payout multiplier from the linear formula
            if our_side_pool > 0 and pool_total > 0:
                pm = pool_total * (1.0 - _TREASURY_FEE_FRACTION) / our_side_pool
                payout_mult = max(0.3, _PAYOUT_LINEAR_BASE + _PAYOUT_LINEAR_SLOPE * (pm - 1.0))
            else:
                pm = 0.0
                payout_mult = 0.3

            # Settlement
            bankroll -= bet_size + GAS_COST_BET_BNB
            outcome = settle_bet_against_closed_round(
                bet_bnb=bet_size,
                bet_side=bet_side,
                round_closed=rnd,
                treasury_fee_fraction=_TREASURY_FEE_FRACTION,
            )
            bankroll += outcome.credit_bnb
            profit = outcome.credit_bnb - bet_size - GAS_COST_BET_BNB

            bet_records.append({
                "epoch": epoch,
                "bet_size": bet_size,
                "pool_total": pool_total,
                "our_side_pool": our_side_pool,
                "pre_bet_payout": pre_bet_payout,
                "post_bet_payout": post_bet_payout,
                "profit": profit,
                "btc_agrees": btc_agrees,
                "btc_disagrees": btc_disagrees,
                "is_contrarian": is_contrarian,
                "won": outcome.outcome == "win",
                "payout_mult": payout_mult,
                "pm_raw": pm,
                "bet_side": bet_side,
            })
        else:
            pipeline.settle_closed_rounds(rounds=[rnd])
            continue

        pipeline.settle_closed_rounds(rounds=[rnd])

    net_pnl = bankroll - initial_bankroll
    total_bets = len(bet_records)
    total_wins = sum(1 for b in bet_records if b["won"])
    wr = total_wins / total_bets * 100 if total_bets > 0 else 0

    print(f"\n{'='*70}")
    print(f"BACKTEST SUMMARY: {total_bets} bets, {total_wins} wins, WR={wr:.1f}%, PnL={net_pnl:+.2f} BNB")
    print(f"{'='*70}")

    # Separate main signal bets from contrarian bets
    main_bets = [b for b in bet_records if not b["is_contrarian"]]
    contra_bets = [b for b in bet_records if b["is_contrarian"]]
    print(f"\nMain signal bets: {len(main_bets)}, Contrarian bets: {len(contra_bets)}")

    # =====================================================================
    # (a) Bet size distribution
    # =====================================================================
    sizes = sorted([b["bet_size"] for b in bet_records])
    print(f"\n--- (a) BET SIZE DISTRIBUTION (all {len(sizes)} bets) ---")
    print(f"  Min:    {min(sizes):.4f} BNB")
    print(f"  P25:    {percentile(sizes, 25):.4f} BNB")
    print(f"  Median: {percentile(sizes, 50):.4f} BNB")
    print(f"  P75:    {percentile(sizes, 75):.4f} BNB")
    print(f"  Max:    {max(sizes):.4f} BNB")
    print(f"  Mean:   {sum(sizes)/len(sizes):.4f} BNB")

    # Main signal only
    main_sizes = sorted([b["bet_size"] for b in main_bets])
    if main_sizes:
        print(f"\n  Main signal bets ({len(main_sizes)}):")
        print(f"    Min:    {min(main_sizes):.4f} BNB")
        print(f"    P25:    {percentile(main_sizes, 25):.4f} BNB")
        print(f"    Median: {percentile(main_sizes, 50):.4f} BNB")
        print(f"    P75:    {percentile(main_sizes, 75):.4f} BNB")
        print(f"    Max:    {max(main_sizes):.4f} BNB")
        print(f"    Mean:   {sum(main_sizes)/len(main_sizes):.4f} BNB")

    # =====================================================================
    # (b) Floor and cap hits
    # =====================================================================
    floor_hits = sum(1 for s in sizes if abs(s - _FLOOR_BNB) < 0.001)
    cap_hits = sum(1 for s in sizes if abs(s - _CAP_BNB) < 0.001)
    contra_fixed = sum(1 for b in bet_records if b["is_contrarian"])
    print(f"\n--- (b) FLOOR / CAP HITS ---")
    print(f"  Floor ({_FLOOR_BNB} BNB): {floor_hits} / {len(sizes)} ({floor_hits/len(sizes)*100:.1f}%)")
    print(f"  Cap ({_CAP_BNB} BNB):   {cap_hits} / {len(sizes)} ({cap_hits/len(sizes)*100:.1f}%)")
    print(f"  Contrarian fixed ({_BTC_CONTRA_BET_BNB} BNB): {contra_fixed} ({contra_fixed/len(sizes)*100:.1f}%)")
    mid_range = len(sizes) - floor_hits - cap_hits - contra_fixed
    print(f"  Mid-range (dynamic): {mid_range} ({mid_range/len(sizes)*100:.1f}%)")

    # =====================================================================
    # (c) Pre-bet vs post-bet payout comparison
    # =====================================================================
    print(f"\n--- (c) PRE-BET vs POST-BET PAYOUT (main signal only) ---")
    dilutions = []
    for b in main_bets:
        if b["pre_bet_payout"] > 0:
            drop_pct = (b["pre_bet_payout"] - b["post_bet_payout"]) / b["pre_bet_payout"] * 100
            dilutions.append(drop_pct)
    if dilutions:
        dilutions_sorted = sorted(dilutions)
        print(f"  Payout drop (our bet dilutes our side):")
        print(f"    Min drop:    {min(dilutions):.2f}%")
        print(f"    P25 drop:    {percentile(dilutions_sorted, 25):.2f}%")
        print(f"    Median drop: {percentile(dilutions_sorted, 50):.2f}%")
        print(f"    P75 drop:    {percentile(dilutions_sorted, 75):.2f}%")
        print(f"    Max drop:    {max(dilutions):.2f}%")
        print(f"    Mean drop:   {sum(dilutions)/len(dilutions):.2f}%")

        # Break down by pool size quartile
        pool_sizes = sorted([b["pool_total"] for b in main_bets])
        pool_p25 = percentile(pool_sizes, 25)
        pool_p50 = percentile(pool_sizes, 50)
        pool_p75 = percentile(pool_sizes, 75)
        print(f"\n  Payout drop by pool size quartile:")
        for label, lo, hi in [("Q1 (smallest pools)", 0, pool_p25),
                               ("Q2", pool_p25, pool_p50),
                               ("Q3", pool_p50, pool_p75),
                               ("Q4 (largest pools)", pool_p75, 1e9)]:
            q_dilutions = [d for b, d in zip(main_bets, dilutions)
                          if lo <= b["pool_total"] < hi or (hi == 1e9 and b["pool_total"] >= lo)]
            # Recompute for correct pairing
            q_recs = [(b, (b["pre_bet_payout"] - b["post_bet_payout"]) / b["pre_bet_payout"] * 100)
                      for b in main_bets if b["pre_bet_payout"] > 0
                      and ((lo <= b["pool_total"] < hi) or (hi == 1e9 and b["pool_total"] >= lo))]
            if q_recs:
                q_drops = [d for _, d in q_recs]
                q_pool_med = percentile(sorted([b["pool_total"] for b, _ in q_recs]), 50)
                q_bet_med = percentile(sorted([b["bet_size"] for b, _ in q_recs]), 50)
                print(f"    {label}: median pool={q_pool_med:.2f}, median bet={q_bet_med:.4f}, "
                      f"median payout drop={percentile(sorted(q_drops), 50):.2f}%, "
                      f"n={len(q_recs)}")

    # =====================================================================
    # (d) PnL per BNB wagered (efficiency) by bet size quartiles
    # =====================================================================
    print(f"\n--- (d) PNL EFFICIENCY BY BET SIZE QUARTILE ---")
    all_sorted_by_size = sorted(bet_records, key=lambda b: b["bet_size"])
    n = len(all_sorted_by_size)
    q_size = n // 4

    for qi in range(4):
        start = qi * q_size
        end = (qi + 1) * q_size if qi < 3 else n
        chunk = all_sorted_by_size[start:end]
        total_wagered = sum(b["bet_size"] for b in chunk)
        total_pnl = sum(b["profit"] for b in chunk)
        wins = sum(1 for b in chunk if b["won"])
        wr_q = wins / len(chunk) * 100 if chunk else 0
        efficiency = total_pnl / total_wagered if total_wagered > 0 else 0
        sizes_q = sorted([b["bet_size"] for b in chunk])

        print(f"  Q{qi+1} (size {sizes_q[0]:.4f} - {sizes_q[-1]:.4f} BNB):")
        print(f"    Bets: {len(chunk)}, WR: {wr_q:.1f}%, PnL: {total_pnl:+.2f} BNB")
        print(f"    Total wagered: {total_wagered:.2f} BNB")
        print(f"    PnL per BNB wagered: {efficiency:+.4f}")
        print(f"    Median bet: {percentile(sizes_q, 50):.4f} BNB")

    # =====================================================================
    # (e) Payout multiplier distribution (from the linear formula)
    # =====================================================================
    print(f"\n--- (e) PAYOUT MULTIPLIER (from linear formula, main signal only) ---")
    payout_mults = sorted([b["payout_mult"] for b in main_bets])
    if payout_mults:
        print(f"  Min:    {min(payout_mults):.4f}")
        print(f"  P10:    {percentile(payout_mults, 10):.4f}")
        print(f"  P25:    {percentile(payout_mults, 25):.4f}")
        print(f"  Median: {percentile(payout_mults, 50):.4f}")
        print(f"  P75:    {percentile(payout_mults, 75):.4f}")
        print(f"  P90:    {percentile(payout_mults, 90):.4f}")
        print(f"  Max:    {max(payout_mults):.4f}")

    # Also show the raw pre-bet payout multiplier distribution
    pm_raws = sorted([b["pm_raw"] for b in main_bets if b["pm_raw"] > 0])
    if pm_raws:
        print(f"\n  Raw pre-bet payout multiplier (pm = pool * 0.97 / our_side):")
        print(f"  Min:    {min(pm_raws):.4f}")
        print(f"  P10:    {percentile(pm_raws, 10):.4f}")
        print(f"  P25:    {percentile(pm_raws, 25):.4f}")
        print(f"  Median: {percentile(pm_raws, 50):.4f}")
        print(f"  P75:    {percentile(pm_raws, 75):.4f}")
        print(f"  P90:    {percentile(pm_raws, 90):.4f}")
        print(f"  Max:    {max(pm_raws):.4f}")

    # Show how the linear formula maps pm -> payout_mult
    print(f"\n  Linear formula: mult = max(0.3, {_PAYOUT_LINEAR_BASE} + {_PAYOUT_LINEAR_SLOPE} * (pm - 1.0))")
    print(f"  Example mappings: pm=1.85 -> {max(0.3, _PAYOUT_LINEAR_BASE + _PAYOUT_LINEAR_SLOPE * (1.85 - 1.0)):.2f}, "
          f"pm=2.0 -> {max(0.3, _PAYOUT_LINEAR_BASE + _PAYOUT_LINEAR_SLOPE * (2.0 - 1.0)):.2f}, "
          f"pm=3.0 -> {max(0.3, _PAYOUT_LINEAR_BASE + _PAYOUT_LINEAR_SLOPE * (3.0 - 1.0)):.2f}, "
          f"pm=5.0 -> {max(0.3, _PAYOUT_LINEAR_BASE + _PAYOUT_LINEAR_SLOPE * (5.0 - 1.0)):.2f}")

    # =====================================================================
    # (f) Correlation between bet size and win rate
    # =====================================================================
    print(f"\n--- (f) BET SIZE vs WIN RATE CORRELATION ---")

    # By deciles for finer resolution
    sorted_by_size = sorted(bet_records, key=lambda b: b["bet_size"])
    n_dec = len(sorted_by_size) // 10

    print(f"  {'Decile':<8} {'Size range':<24} {'Bets':>5} {'WR':>7} {'PnL':>10} {'Eff':>8}")
    for di in range(10):
        start = di * n_dec
        end = (di + 1) * n_dec if di < 9 else len(sorted_by_size)
        chunk = sorted_by_size[start:end]
        wins = sum(1 for b in chunk if b["won"])
        wr_d = wins / len(chunk) * 100 if chunk else 0
        pnl_d = sum(b["profit"] for b in chunk)
        wag_d = sum(b["bet_size"] for b in chunk)
        eff_d = pnl_d / wag_d if wag_d > 0 else 0
        s_min = min(b["bet_size"] for b in chunk)
        s_max = max(b["bet_size"] for b in chunk)
        print(f"  D{di+1:<6} {s_min:.4f} - {s_max:.4f} BNB   {len(chunk):>5} {wr_d:>6.1f}% {pnl_d:>+9.2f} {eff_d:>+7.4f}")

    # Also show btc_agrees vs btc_disagrees vs neither (main signal only)
    print(f"\n  BTC confirmation breakdown (main signal bets):")
    for label, filt in [("BTC agrees", lambda b: b["btc_agrees"]),
                         ("BTC disagrees", lambda b: b["btc_disagrees"]),
                         ("BTC neutral", lambda b: not b["btc_agrees"] and not b["btc_disagrees"])]:
        chunk = [b for b in main_bets if filt(b)]
        if chunk:
            wins = sum(1 for b in chunk if b["won"])
            wr_c = wins / len(chunk) * 100
            pnl_c = sum(b["profit"] for b in chunk)
            med_size = percentile(sorted([b["bet_size"] for b in chunk]), 50)
            print(f"    {label}: n={len(chunk)}, WR={wr_c:.1f}%, PnL={pnl_c:+.2f}, median size={med_size:.4f}")

    # =====================================================================
    # (g) Market impact analysis
    # =====================================================================
    print(f"\n--- (g) MARKET IMPACT ANALYSIS ---")

    # For various pool sizes, compute what bet size keeps payout drop < 10%
    pool_totals = sorted([b["pool_total"] for b in main_bets if b["pool_total"] > 0])
    med_pool = percentile(pool_totals, 50)
    p25_pool = percentile(pool_totals, 25)
    p75_pool = percentile(pool_totals, 75)

    print(f"  Pool size distribution: P25={p25_pool:.2f}, Median={med_pool:.2f}, P75={p75_pool:.2f} BNB")

    # For a given pool size and payout, find max bet that keeps drop < 10%
    # Pre-bet payout: pm = pool * 0.97 / our_side
    # Post-bet payout: pm' = (pool + bet) * 0.97 / (our_side + bet)
    # Drop = 1 - pm'/pm < 0.10
    # pm'/pm = [(pool + bet) / (our_side + bet)] / [pool / our_side]
    #        = [our_side * (pool + bet)] / [pool * (our_side + bet)]
    # Want: our_side * (pool + bet) / [pool * (our_side + bet)] >= 0.90
    # Solve for bet: bet <= our_side * pool * 0.10 / (0.90 * pool - our_side * 1.0)
    # Actually let's just simulate:

    print(f"\n  Max bet size to keep payout drop < 10%:")
    print(f"  {'Pool total':>12} {'Our side':>12} {'Pre-payout':>12} {'Max bet':>10} {'% of pool':>10}")

    for pool in [p25_pool, med_pool, p75_pool]:
        for our_frac in [0.30, 0.40, 0.50]:
            our_side = pool * our_frac
            pm_pre = pool * 0.97 / our_side
            # Binary search for max bet
            lo, hi = 0.0, 10.0
            for _ in range(100):
                mid = (lo + hi) / 2
                pm_post = (pool + mid) * 0.97 / (our_side + mid)
                drop = 1.0 - pm_post / pm_pre
                if drop < 0.10:
                    lo = mid
                else:
                    hi = mid
            max_bet = lo
            print(f"  {pool:>11.2f} {our_side:>11.2f} {pm_pre:>11.2f}x {max_bet:>9.3f} {max_bet/pool*100:>9.1f}%")

    # What % of our actual bets exceed the 10% payout-drop threshold?
    over_10pct_count = sum(1 for b in main_bets
                          if b["pre_bet_payout"] > 0
                          and (b["pre_bet_payout"] - b["post_bet_payout"]) / b["pre_bet_payout"] > 0.10)
    print(f"\n  Bets exceeding 10% payout drop: {over_10pct_count} / {len(main_bets)} "
          f"({over_10pct_count/len(main_bets)*100:.1f}%)")

    over_20pct_count = sum(1 for b in main_bets
                          if b["pre_bet_payout"] > 0
                          and (b["pre_bet_payout"] - b["post_bet_payout"]) / b["pre_bet_payout"] > 0.20)
    print(f"  Bets exceeding 20% payout drop: {over_20pct_count} / {len(main_bets)} "
          f"({over_20pct_count/len(main_bets)*100:.1f}%)")

    # Summary: bet/pool ratio distribution
    ratios = sorted([b["bet_size"] / b["pool_total"] for b in main_bets if b["pool_total"] > 0])
    if ratios:
        print(f"\n  Bet/pool ratio distribution (main signal):")
        print(f"    Min:    {min(ratios)*100:.2f}%")
        print(f"    P25:    {percentile(ratios, 25)*100:.2f}%")
        print(f"    Median: {percentile(ratios, 50)*100:.2f}%")
        print(f"    P75:    {percentile(ratios, 75)*100:.2f}%")
        print(f"    Max:    {max(ratios)*100:.2f}%")


if __name__ == "__main__":
    run_analysis()
