"""Sizing grid experiment: dilution cap x payout slope.

Tests every combination of:
  - dilution_cap_pct: [None (no cap), 15, 12, 10, 8]
  - payout_slope:     [0.5, 1.0 (baseline), 1.5, 2.0]

For each combo, replicates the full pipeline signal computation but overrides
_compute_bet_size with the experimental sizing logic.  Settlement uses the
production settle_bet_against_closed_round (which already accounts for market
impact in the pools).

Reports: total PnL, bet count, win rate, worst segment PnL.
"""
from __future__ import annotations

import json, sys, time, itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
from pancakebot.domain.strategy.momentum_pipeline import (
    MomentumOnlyPipeline,
    _BASE_FRAC,
    _FLOOR_BNB,
    _CAP_BNB,
    _BTC_AGREE_MULT,
    _BTC_DISAGREE_MULT,
    _PAYOUT_LINEAR_BASE,
)
from pancakebot.runtime.settlement import settle_bet_against_closed_round

# Production constants
_CUTOFF_SECONDS = 4
_MIN_BET_AMOUNT_BNB = 0.001
_TREASURY_FEE_FRACTION = 0.03
_SIM_SIZE = 34_000  # use all clean rounds
_INITIAL_BANKROLL = 50.0
_N_SEGMENTS = 5


# ---- Data loading ----

_cache: dict = {}

def _load_data():
    if _cache:
        return _cache["rounds"], _cache["spot"], _cache["btc"]
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

    spot = load_klines("var/bnb_spot_prices.jsonl")
    btc = load_klines("var/btc_spot_prices.jsonl")
    _cache.update(rounds=rounds, spot=spot, btc=btc)
    return rounds, spot, btc


# ---- Experimental sizing ----

def compute_bet_size_experimental(
    *,
    signal: str,
    btc_agrees: bool,
    btc_disagrees: bool,
    pool_bull_bnb: float,
    pool_bear_bnb: float,
    treasury_fee_fraction: float,
    payout_slope: float,
    dilution_cap_pct: float | None,
) -> float:
    """Replicated sizing with configurable payout_slope and dilution cap."""
    pool_bnb = pool_bull_bnb + pool_bear_bnb
    if pool_bnb <= 0:
        return _FLOOR_BNB

    bet = max(_FLOOR_BNB, pool_bnb * _BASE_FRAC)

    # Linear payout-proportional adjustment with experimental slope
    our_side = pool_bull_bnb if signal == "Bull" else pool_bear_bnb
    if our_side > 0:
        pm = pool_bnb * (1.0 - treasury_fee_fraction) / our_side
        mult = max(0.3, _PAYOUT_LINEAR_BASE + payout_slope * (pm - 1.0))
        bet *= mult

    # BTC confirmation boost
    if btc_agrees:
        bet *= _BTC_AGREE_MULT
    elif btc_disagrees:
        bet *= _BTC_DISAGREE_MULT

    bet = min(_CAP_BNB, bet)

    # Dilution cap: reduce bet so payout dilution stays under threshold
    if dilution_cap_pct is not None and our_side > 0 and pool_bnb > 0:
        bet = _apply_dilution_cap(bet, pool_bnb, our_side, treasury_fee_fraction, dilution_cap_pct)

    return bet


def _apply_dilution_cap(
    bet: float,
    pool_total: float,
    our_side: float,
    treasury_fee_fraction: float,
    cap_pct: float,
) -> float:
    """Reduce bet size if it would dilute payout by more than cap_pct%.

    dilution_pct = (pre_payout - post_payout) / pre_payout * 100
    where:
      pre_payout  = pool_total * 0.97 / our_side
      post_payout = (pool_total + bet) * 0.97 / (our_side + bet)
    """
    fee_mult = 1.0 - treasury_fee_fraction
    pre_payout = pool_total * fee_mult / our_side

    # Check if current bet exceeds cap
    post_payout = (pool_total + bet) * fee_mult / (our_side + bet)
    dilution = (pre_payout - post_payout) / pre_payout * 100.0

    if dilution <= cap_pct:
        return bet

    # Solve for max bet that keeps dilution at exactly cap_pct.
    # dilution = cap/100 => post_payout = pre_payout * (1 - cap/100)
    # (pool_total + b) * fee_mult / (our_side + b) = pre_payout * (1 - cap/100)
    # Let target = pre_payout * (1 - cap/100)
    # (pool_total + b) * fee_mult = target * (our_side + b)
    # pool_total * fee_mult + b * fee_mult = target * our_side + target * b
    # b * (fee_mult - target) = target * our_side - pool_total * fee_mult
    # b = (target * our_side - pool_total * fee_mult) / (fee_mult - target)

    target = pre_payout * (1.0 - cap_pct / 100.0)
    denom = fee_mult - target
    if abs(denom) < 1e-12:
        return bet  # degenerate case, keep original

    max_bet = (target * our_side - pool_total * fee_mult) / denom
    if max_bet < _MIN_BET_AMOUNT_BNB:
        return _MIN_BET_AMOUNT_BNB

    return min(bet, max_bet)


# ---- Single combo backtest ----

def run_combo(payout_slope: float, dilution_cap_pct: float | None) -> dict:
    """Run backtest for one (payout_slope, dilution_cap_pct) combination."""
    rounds, spot, btc = _load_data()
    sim_rounds = rounds[-_SIM_SIZE:]

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
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch=spot)
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc)

    bankroll = _INITIAL_BANKROLL
    trades = []

    for rnd in sim_rounds:
        decision = pipeline.decide_open_round(
            round_t=rnd,
            bankroll_bnb=bankroll,
            allow_oracle_mode=True,
        )

        if decision.action == "BET" and decision.bet_size_bnb > 0.0:
            # Re-derive pool info for this round to apply experimental sizing
            from pancakebot.core.constants import BNB_WEI
            from pancakebot.domain.strategy.momentum_pipeline import _pools_from_bets

            lock_at = int(rnd.lock_at)
            pool_bull_bnb, pool_bear_bnb = (0.0, 0.0)
            if rnd.bets:
                pool_bull_bnb, pool_bear_bnb = _pools_from_bets(rnd, lock_at)

            # For BTC contrarian bets (fixed size, no sizing override)
            if decision.selected_strategy == "momentum_gate" and decision.bet_side in ("Bull", "Bear"):
                # Detect if this was a BTC contrarian bet by checking if the
                # production size matches the fixed contrarian bet size
                from pancakebot.domain.strategy.momentum_pipeline import _BTC_CONTRA_BET_BNB
                is_contrarian = abs(decision.bet_size_bnb - _BTC_CONTRA_BET_BNB) < 0.001

                if is_contrarian:
                    # Contrarian bets use fixed size, no override
                    exp_bet = decision.bet_size_bnb
                else:
                    # Main signal: apply experimental sizing
                    # We need btc_agrees/btc_disagrees - re-derive from
                    # the production bet vs what we'd get without BTC modifiers
                    # Simpler: back out from the pipeline's decision by checking
                    # the bet against what _compute_bet_size would produce with
                    # btc_agrees=False and btc_disagrees=False

                    # Actually, we should just re-run the signal to get the
                    # gate result. But the pipeline already did that. The cleanest
                    # approach: replicate the gate evaluation to get btc_agrees/disagrees.
                    gate_result = pipeline._evaluate_from_cache(
                        epoch=int(rnd.epoch),
                        cutoff_ts_ms=(lock_at - _CUTOFF_SECONDS) * 1000,
                    )

                    exp_bet = compute_bet_size_experimental(
                        signal=decision.bet_side,
                        btc_agrees=gate_result.btc_agrees,
                        btc_disagrees=gate_result.btc_disagrees,
                        pool_bull_bnb=pool_bull_bnb,
                        pool_bear_bnb=pool_bear_bnb,
                        treasury_fee_fraction=_TREASURY_FEE_FRACTION,
                        payout_slope=payout_slope,
                        dilution_cap_pct=dilution_cap_pct,
                    )
            else:
                exp_bet = decision.bet_size_bnb

            if exp_bet < _MIN_BET_AMOUNT_BNB:
                trades.append((int(rnd.epoch), "SKIP", 0.0, bankroll))
                pipeline.settle_closed_rounds(rounds=[rnd])
                continue

            bankroll -= exp_bet + GAS_COST_BET_BNB
            outcome = settle_bet_against_closed_round(
                bet_bnb=exp_bet,
                bet_side=decision.bet_side,
                round_closed=rnd,
                treasury_fee_fraction=_TREASURY_FEE_FRACTION,
            )
            bankroll += outcome.credit_bnb
            profit = outcome.credit_bnb - exp_bet - GAS_COST_BET_BNB
            trades.append((int(rnd.epoch), "BET", profit, bankroll))
        else:
            trades.append((int(rnd.epoch), "SKIP", 0.0, bankroll))

        pipeline.settle_closed_rounds(rounds=[rnd])

    net = bankroll - _INITIAL_BANKROLL
    bets = [t for t in trades if t[1] == "BET"]
    wins = [t for t in bets if t[2] > 0]
    total_bets = len(bets)
    wr = len(wins) / max(1, total_bets) * 100

    # Segment analysis
    seg_size = max(1, len(trades) // _N_SEGMENTS)
    worst_seg_pnl = float("inf")
    for s in range(_N_SEGMENTS):
        start = s * seg_size
        end = (s + 1) * seg_size if s < _N_SEGMENTS - 1 else len(trades)
        chunk = trades[start:end]
        seg_bets = [t for t in chunk if t[1] == "BET"]
        seg_pnl = sum(t[2] for t in seg_bets)
        if seg_pnl < worst_seg_pnl:
            worst_seg_pnl = seg_pnl

    return {
        "payout_slope": payout_slope,
        "dilution_cap": dilution_cap_pct,
        "net_pnl": net,
        "bets": total_bets,
        "wr": wr,
        "worst_seg": worst_seg_pnl,
    }


# ---- Main ----

def main():
    dilution_caps = [None, 15.0, 12.0, 10.0, 8.0]
    payout_slopes = [0.5, 1.0, 1.5, 2.0]

    combos = list(itertools.product(dilution_caps, payout_slopes))
    results = []

    print(f"Running {len(combos)} combinations on {_SIM_SIZE} rounds...")
    print(f"{'Cap':>6s}  {'Slope':>5s}  {'PnL':>9s}  {'Bets':>5s}  {'WR%':>6s}  {'WorstSeg':>9s}")
    print("-" * 55)

    t0 = time.time()
    for i, (cap, slope) in enumerate(combos):
        cap_label = f"{cap:.0f}%" if cap is not None else "none"
        t1 = time.time()
        r = run_combo(payout_slope=slope, dilution_cap_pct=cap)
        elapsed = time.time() - t1
        results.append(r)
        print(
            f"{cap_label:>6s}  {slope:5.1f}  {r['net_pnl']:+9.2f}  {r['bets']:5d}  "
            f"{r['wr']:5.1f}%  {r['worst_seg']:+9.2f}  ({elapsed:.1f}s)"
        )

    total_elapsed = time.time() - t0
    print(f"\nTotal time: {total_elapsed:.0f}s")

    # Find best by PnL
    best = max(results, key=lambda r: r["net_pnl"])
    best_cap = f"{best['dilution_cap']:.0f}%" if best["dilution_cap"] is not None else "none"
    print(f"\n{'='*55}")
    print(f"BEST by PnL: cap={best_cap}, slope={best['payout_slope']:.1f}")
    print(f"  PnL: {best['net_pnl']:+.2f} BNB | Bets: {best['bets']} | WR: {best['wr']:.1f}% | Worst seg: {best['worst_seg']:+.2f}")

    # Also find best risk-adjusted (PnL with constraint: worst_seg > -5)
    safe = [r for r in results if r["worst_seg"] > -5.0]
    if safe:
        best_safe = max(safe, key=lambda r: r["net_pnl"])
        cap_s = f"{best_safe['dilution_cap']:.0f}%" if best_safe["dilution_cap"] is not None else "none"
        print(f"\nBEST safe (worst_seg > -5): cap={cap_s}, slope={best_safe['payout_slope']:.1f}")
        print(f"  PnL: {best_safe['net_pnl']:+.2f} BNB | Bets: {best_safe['bets']} | WR: {best_safe['wr']:.1f}% | Worst seg: {best_safe['worst_seg']:+.2f}")


if __name__ == "__main__":
    main()
