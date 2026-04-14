"""Sweep _BTC_AGREE_MULT to find optimal BTC-agree sizing multiplier.

Monkey-patches the module-level constant in momentum_pipeline before each
backtest run.  Does NOT modify any production files.

Tested values: [1.5 (baseline), 1.8, 2.0, 2.5, 3.0]
Reports: total PnL, bet count, WR, per-segment PnL, worst segment PnL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
import pancakebot.domain.strategy.momentum_pipeline as mp_module
from pancakebot.domain.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.runtime.settlement import settle_bet_against_closed_round

# Production constants
_CUTOFF_SECONDS = 4
_MIN_BET_AMOUNT_BNB = 0.001
_TREASURY_FEE_FRACTION = 0.03
_INITIAL_BANKROLL = 50.0

# Sweep values
_MULT_VALUES = [1.5, 1.8, 2.0, 2.5, 3.0]

# Number of segments for breakdown
_N_TARGET_SEG_SIZE = 5000
_N_SEGMENTS = 6


# ---- Data loading (cached) ----
_cache = {}


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


def run_backtest(mult_value: float) -> dict:
    """Run full backtest with a specific _BTC_AGREE_MULT value.

    Monkey-patches the module constant, runs the backtest, then restores it.
    """
    original = mp_module._BTC_AGREE_MULT
    try:
        # Monkey-patch the module-level constant
        mp_module._BTC_AGREE_MULT = mult_value

        rounds, spot, btc = _load_data()

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

        for rnd in rounds:
            decision = pipeline.decide_open_round(
                round_t=rnd,
                bankroll_bnb=bankroll,
                allow_oracle_mode=True,
            )

            if decision.action == "BET" and decision.bet_size_bnb > 0.0:
                bankroll -= decision.bet_size_bnb + GAS_COST_BET_BNB
                outcome = settle_bet_against_closed_round(
                    bet_bnb=decision.bet_size_bnb,
                    bet_side=decision.bet_side,
                    round_closed=rnd,
                    treasury_fee_fraction=_TREASURY_FEE_FRACTION,
                )
                bankroll += outcome.credit_bnb
                profit = outcome.credit_bnb - decision.bet_size_bnb - GAS_COST_BET_BNB

                trades.append({
                    "epoch": int(rnd.epoch),
                    "action": "BET",
                    "profit": profit,
                    "bankroll": bankroll,
                    "side": decision.bet_side,
                    "outcome": outcome.outcome,
                    "bet_size": decision.bet_size_bnb,
                })
            else:
                trades.append({
                    "epoch": int(rnd.epoch),
                    "action": "SKIP",
                    "profit": 0.0,
                    "bankroll": bankroll,
                    "side": None,
                    "outcome": decision.skip_reason or "skip",
                    "bet_size": 0.0,
                })

            pipeline.settle_closed_rounds(rounds=[rnd])

        net = bankroll - _INITIAL_BANKROLL

        # Fixed 6-segment breakdown
        bet_trades = [t for t in trades if t["action"] == "BET"]
        total_bets = len(bet_trades)
        total_wins = sum(1 for t in bet_trades if t["profit"] > 0)
        wr_pct = total_wins / max(1, total_bets) * 100
        avg_bet = sum(t["bet_size"] for t in bet_trades) / max(1, total_bets)

        # Segments over ALL trades (bets + skips), matching fast_backtest pattern
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
            "mult": mult_value,
            "net_pnl": net,
            "total_bets": total_bets,
            "wr_pct": wr_pct,
            "avg_bet": avg_bet,
            "segments": segments,
            "worst_seg_pnl": worst_seg_pnl,
        }

    finally:
        # Always restore original value
        mp_module._BTC_AGREE_MULT = original


def main():
    print("Loading data...")
    rounds, _, _ = _load_data()
    print(f"Total rounds: {len(rounds)}")
    print(f"Sweep values: {_MULT_VALUES}")
    print(f"Segments: {_N_SEGMENTS}")
    print()

    results = []
    for mult in _MULT_VALUES:
        tag = "BASELINE" if mult == 1.5 else ""
        print(f"--- BTC_AGREE_MULT = {mult} {tag} ---")
        r = run_backtest(mult)
        results.append(r)

        print(f"  NET: {r['net_pnl']:+.2f} BNB | Bets: {r['total_bets']} | "
              f"WR: {r['wr_pct']:.1f}% | Avg bet: {r['avg_bet']:.3f} BNB")
        for i, seg in enumerate(r["segments"]):
            print(f"    Seg{i+1}: {seg['bets']:4d} bets, "
                  f"WR={seg['wr']:5.1f}%, PnL={seg['pnl']:+7.2f}")
        print(f"  Worst segment PnL: {r['worst_seg_pnl']:+.2f}")
        print()

    # Summary comparison table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Mult':>5} | {'Net PnL':>10} | {'Bets':>5} | {'WR%':>6} | {'Avg Bet':>8} | {'Worst Seg':>10} | {'vs Base':>8}"
    print(header)
    print("-" * len(header))

    baseline_pnl = results[0]["net_pnl"]
    for r in results:
        delta = r["net_pnl"] - baseline_pnl
        delta_str = f"{delta:+.2f}" if r["mult"] != 1.5 else "---"
        print(f"{r['mult']:5.1f} | {r['net_pnl']:+10.2f} | {r['total_bets']:5d} | "
              f"{r['wr_pct']:5.1f}% | {r['avg_bet']:8.3f} | {r['worst_seg_pnl']:+10.2f} | {delta_str:>8}")

    # Segment-by-segment comparison
    print()
    print("=" * 80)
    print("SEGMENT PnL COMPARISON")
    print("=" * 80)
    seg_header = f"{'Mult':>5}" + "".join(f" | {'Seg'+str(i+1):>8}" for i in range(_N_SEGMENTS))
    print(seg_header)
    print("-" * len(seg_header))
    for r in results:
        row = f"{r['mult']:5.1f}"
        for seg in r["segments"]:
            row += f" | {seg['pnl']:+8.2f}"
        print(row)


if __name__ == "__main__":
    main()
