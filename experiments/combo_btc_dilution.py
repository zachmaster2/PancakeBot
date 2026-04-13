"""Combination test: BTC_AGREE_MULT x dilution_cap.

Tests 6 combinations on all available rounds:
  1. BTC_AGREE_MULT=1.8, no dilution cap
  2. BTC_AGREE_MULT=1.8, 12% dilution cap
  3. BTC_AGREE_MULT=1.8, 10% dilution cap
  4. BTC_AGREE_MULT=2.0, 12% dilution cap
  5. BTC_AGREE_MULT=2.0, 10% dilution cap
  6. BTC_AGREE_MULT=2.0, no dilution cap

Monkey-patches _BTC_AGREE_MULT and applies dilution cap as a post-sizing
step.  Does NOT modify any production files.

Reports: total PnL, bet count, WR, per-segment PnL (6 segments), worst segment.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
import pancakebot.domain.strategy.momentum_pipeline as mp_module
from pancakebot.domain.strategy.momentum_pipeline import (
    MomentumOnlyPipeline,
    _pools_from_bets,
    _BTC_CONTRA_BET_BNB,
)
from pancakebot.runtime.settlement import settle_bet_against_closed_round

# Production constants
_CUTOFF_SECONDS = 4
_MIN_BET_AMOUNT_BNB = 0.001
_TREASURY_FEE_FRACTION = 0.03
_INITIAL_BANKROLL = 50.0
_N_SEGMENTS = 6


# ---- Data loading (cached across runs in same process) ----
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

    spot = load_klines("var/cutoff_spot_prices.jsonl")
    btc = load_klines("var/btc_spot_prices.jsonl")
    _cache.update(rounds=rounds, spot=spot, btc=btc)
    return rounds, spot, btc


# ---- Dilution cap logic (from sizing_grid.py) ----

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
    # post_payout = pre_payout * (1 - cap/100)
    # (pool_total + b) * fee_mult / (our_side + b) = target
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

def run_combo(btc_agree_mult: float, dilution_cap_pct: float | None) -> dict:
    """Run backtest for one (btc_agree_mult, dilution_cap_pct) combination.

    Monkey-patches _BTC_AGREE_MULT, then applies dilution cap as a
    post-processing step after the pipeline returns a BET decision.
    """
    original_mult = mp_module._BTC_AGREE_MULT
    try:
        mp_module._BTC_AGREE_MULT = btc_agree_mult

        rounds, spot, btc = _load_data()
        # Use all available rounds
        sim_rounds = rounds

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

        bankroll = _INITIAL_BANKROLL
        trades = []

        for rnd in sim_rounds:
            decision = pipeline.decide_open_round(
                round_t=rnd,
                bankroll_bnb=bankroll,
                allow_oracle_mode=True,
            )

            if decision.action == "BET" and decision.bet_size_bnb > 0.0:
                bet_size = decision.bet_size_bnb

                # Apply dilution cap post-processing (only for non-contrarian bets)
                is_contrarian = abs(bet_size - _BTC_CONTRA_BET_BNB) < 0.001
                if dilution_cap_pct is not None and not is_contrarian:
                    lock_at = int(rnd.lock_at)
                    pool_bull_bnb, pool_bear_bnb = (0.0, 0.0)
                    if rnd.bets:
                        pool_bull_bnb, pool_bear_bnb = _pools_from_bets(rnd, lock_at)
                    pool_total = pool_bull_bnb + pool_bear_bnb
                    our_side = pool_bull_bnb if decision.bet_side == "Bull" else pool_bear_bnb

                    if our_side > 0 and pool_total > 0:
                        bet_size = _apply_dilution_cap(
                            bet_size, pool_total, our_side,
                            _TREASURY_FEE_FRACTION, dilution_cap_pct,
                        )

                if bet_size < _MIN_BET_AMOUNT_BNB:
                    trades.append({
                        "epoch": int(rnd.epoch),
                        "action": "SKIP",
                        "profit": 0.0,
                        "bankroll": bankroll,
                        "bet_size": 0.0,
                    })
                    pipeline.settle_closed_rounds(rounds=[rnd])
                    continue

                bankroll -= bet_size + GAS_COST_BET_BNB
                outcome = settle_bet_against_closed_round(
                    bet_bnb=bet_size,
                    bet_side=decision.bet_side,
                    round_closed=rnd,
                    treasury_fee_fraction=_TREASURY_FEE_FRACTION,
                )
                bankroll += outcome.credit_bnb
                profit = outcome.credit_bnb - bet_size - GAS_COST_BET_BNB

                trades.append({
                    "epoch": int(rnd.epoch),
                    "action": "BET",
                    "profit": profit,
                    "bankroll": bankroll,
                    "bet_size": bet_size,
                })
            else:
                trades.append({
                    "epoch": int(rnd.epoch),
                    "action": "SKIP",
                    "profit": 0.0,
                    "bankroll": bankroll,
                    "bet_size": 0.0,
                })

            pipeline.settle_closed_rounds(rounds=[rnd])

        net = bankroll - _INITIAL_BANKROLL

        # Stats
        bet_trades = [t for t in trades if t["action"] == "BET"]
        total_bets = len(bet_trades)
        total_wins = sum(1 for t in bet_trades if t["profit"] > 0)
        wr_pct = total_wins / max(1, total_bets) * 100
        avg_bet = sum(t["bet_size"] for t in bet_trades) / max(1, total_bets)

        # Fixed 6-segment breakdown over all trades
        seg_size = len(trades) // _N_SEGMENTS
        segments = []
        for s in range(_N_SEGMENTS):
            start = s * seg_size
            end = (s + 1) * seg_size if s < _N_SEGMENTS - 1 else len(trades)
            chunk = trades[start:end]
            bets = [t for t in chunk if t["action"] == "BET"]
            wins = [t for t in bets if t["profit"] > 0]
            pnl = sum(t["profit"] for t in bets)
            wr = len(wins) / len(bets) * 100 if bets else 0
            segments.append({"bets": len(bets), "wr": wr, "pnl": pnl})

        worst_seg_pnl = min(seg["pnl"] for seg in segments) if segments else 0.0

        return {
            "btc_agree_mult": btc_agree_mult,
            "dilution_cap": dilution_cap_pct,
            "net_pnl": net,
            "total_bets": total_bets,
            "wr_pct": wr_pct,
            "avg_bet": avg_bet,
            "segments": segments,
            "worst_seg_pnl": worst_seg_pnl,
        }

    finally:
        mp_module._BTC_AGREE_MULT = original_mult


# ---- Main ----

def main():
    combos = [
        (1.8, None),     # 1. BTC_AGREE_MULT=1.8, no dilution cap
        (1.8, 12.0),     # 2. BTC_AGREE_MULT=1.8, 12% dilution cap
        (1.8, 10.0),     # 3. BTC_AGREE_MULT=1.8, 10% dilution cap
        (2.0, 12.0),     # 4. BTC_AGREE_MULT=2.0, 12% dilution cap
        (2.0, 10.0),     # 5. BTC_AGREE_MULT=2.0, 10% dilution cap
        (2.0, None),     # 6. BTC_AGREE_MULT=2.0, no dilution cap
    ]

    print("Loading data...")
    rounds, _, _ = _load_data()
    print(f"Total rounds: {len(rounds)}")
    print(f"Segments: {_N_SEGMENTS}")
    print()

    results = []
    t0 = time.time()

    for i, (mult, cap) in enumerate(combos):
        cap_label = f"{cap:.0f}%" if cap is not None else "none"
        print(f"--- Combo {i+1}: BTC_AGREE_MULT={mult}, dilution_cap={cap_label} ---")
        t1 = time.time()
        r = run_combo(btc_agree_mult=mult, dilution_cap_pct=cap)
        elapsed = time.time() - t1
        results.append(r)

        print(f"  NET: {r['net_pnl']:+.2f} BNB | Bets: {r['total_bets']} | "
              f"WR: {r['wr_pct']:.1f}% | Avg bet: {r['avg_bet']:.3f} BNB ({elapsed:.1f}s)")
        for j, seg in enumerate(r["segments"]):
            print(f"    Seg{j+1}: {seg['bets']:4d} bets, "
                  f"WR={seg['wr']:5.1f}%, PnL={seg['pnl']:+7.2f}")
        print(f"  Worst segment PnL: {r['worst_seg_pnl']:+.2f}")
        print()

    total_elapsed = time.time() - t0
    print(f"Total time: {total_elapsed:.0f}s")

    # ---- Summary table ----
    print()
    print("=" * 95)
    print("SUMMARY")
    print("=" * 95)
    header = (f"{'Mult':>5} | {'Cap':>6} | {'Net PnL':>10} | {'Bets':>5} | "
              f"{'WR%':>6} | {'Avg Bet':>8} | {'Worst Seg':>10} | {'vs Baseline':>11}")
    print(header)
    print("-" * len(header))

    baseline_pnl = 49.81  # current production baseline
    for r in results:
        cap_label = f"{r['dilution_cap']:.0f}%" if r['dilution_cap'] is not None else "none"
        delta = r["net_pnl"] - baseline_pnl
        print(f"{r['btc_agree_mult']:5.1f} | {cap_label:>6} | {r['net_pnl']:+10.2f} | "
              f"{r['total_bets']:5d} | {r['wr_pct']:5.1f}% | {r['avg_bet']:8.3f} | "
              f"{r['worst_seg_pnl']:+10.2f} | {delta:+11.2f}")

    # ---- Segment comparison ----
    print()
    print("=" * 95)
    print("SEGMENT PnL COMPARISON")
    print("=" * 95)
    seg_header = f"{'Mult':>5} {'Cap':>6}" + "".join(f" | {'Seg'+str(i+1):>8}" for i in range(_N_SEGMENTS))
    print(seg_header)
    print("-" * len(seg_header))
    for r in results:
        cap_label = f"{r['dilution_cap']:.0f}%" if r['dilution_cap'] is not None else "none"
        row = f"{r['btc_agree_mult']:5.1f} {cap_label:>6}"
        for seg in r["segments"]:
            row += f" | {seg['pnl']:+8.2f}"
        print(row)

    # ---- Best combo ----
    print()
    best = max(results, key=lambda r: r["net_pnl"])
    best_cap = f"{best['dilution_cap']:.0f}%" if best['dilution_cap'] is not None else "none"
    print(f"BEST by PnL: mult={best['btc_agree_mult']}, cap={best_cap}")
    print(f"  PnL: {best['net_pnl']:+.2f} BNB | Bets: {best['total_bets']} | "
          f"WR: {best['wr_pct']:.1f}% | Worst seg: {best['worst_seg_pnl']:+.2f}")

    # Best risk-adjusted (worst_seg > -2.0)
    safe = [r for r in results if r["worst_seg_pnl"] > -2.0]
    if safe:
        best_safe = max(safe, key=lambda r: r["net_pnl"])
        cap_s = f"{best_safe['dilution_cap']:.0f}%" if best_safe['dilution_cap'] is not None else "none"
        print(f"\nBEST safe (worst_seg > -2.0): mult={best_safe['btc_agree_mult']}, cap={cap_s}")
        print(f"  PnL: {best_safe['net_pnl']:+.2f} BNB | Bets: {best_safe['total_bets']} | "
              f"WR: {best_safe['wr_pct']:.1f}% | Worst seg: {best_safe['worst_seg_pnl']:+.2f}")


if __name__ == "__main__":
    main()
