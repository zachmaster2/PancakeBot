"""Sweep _ACCEL_THRESH to find optimal acceleration threshold.

Monkey-patches pancakebot.domain.strategy.momentum_gate._ACCEL_THRESH
before each run.  All other pipeline parameters (sizing, filters, BTC
contrarian) remain at production values.

Tests: 1bp, 1.5bp, 2bp (baseline), 2.5bp, 3bp, 4bp.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pancakebot.domain.strategy.momentum_gate as mg
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
from pancakebot.domain.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.runtime.settlement import settle_bet_against_closed_round
import json

# Production constants
_CUTOFF_SECONDS = 4
_MIN_BET_AMOUNT_BNB = 0.001
_TREASURY_FEE_FRACTION = 0.03
_INITIAL_BANKROLL = 50.0

THRESHOLDS = [
    (0.00010, "1.0bp"),
    (0.00015, "1.5bp"),
    (0.00020, "2.0bp *"),   # baseline
    (0.00025, "2.5bp"),
    (0.00030, "3.0bp"),
    (0.00040, "4.0bp"),
]


def _load_data():
    """Load rounds + kline caches from disk (once)."""
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
    return rounds, spot, btc


def run_backtest(rounds, spot, btc, sim_size):
    """Run one backtest pass with whatever _ACCEL_THRESH is currently set."""
    sim_rounds = rounds[-sim_size:]

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
            bankroll -= decision.bet_size_bnb + GAS_COST_BET_BNB
            outcome = settle_bet_against_closed_round(
                bet_bnb=decision.bet_size_bnb,
                bet_side=decision.bet_side,
                round_closed=rnd,
                treasury_fee_fraction=_TREASURY_FEE_FRACTION,
            )
            bankroll += outcome.credit_bnb
            profit = outcome.credit_bnb - decision.bet_size_bnb - GAS_COST_BET_BNB
            trades.append(("BET", profit))
        else:
            trades.append(("SKIP", 0.0))

        pipeline.settle_closed_rounds(rounds=[rnd])

    net = bankroll - _INITIAL_BANKROLL
    bets = [t for t in trades if t[0] == "BET"]
    wins = [t for t in bets if t[1] > 0]
    total_bets = len(bets)
    wr = len(wins) / max(1, total_bets) * 100

    # Segment analysis (5k-round segments)
    n_segments = max(1, len(trades) // 5000)
    seg_size = len(trades) // n_segments
    segments = []
    for s in range(n_segments):
        start = s * seg_size
        end = (s + 1) * seg_size if s < n_segments - 1 else len(trades)
        chunk = trades[start:end]
        seg_bets = [t for t in chunk if t[0] == "BET"]
        seg_wins = [t for t in seg_bets if t[1] > 0]
        seg_pnl = sum(t[1] for t in seg_bets)
        seg_wr = len(seg_wins) / max(1, len(seg_bets)) * 100
        segments.append((len(seg_bets), seg_wr, seg_pnl))

    worst_seg_pnl = min(s[2] for s in segments) if segments else 0.0

    return {
        "net": net,
        "bets": total_bets,
        "wr": wr,
        "worst_seg": worst_seg_pnl,
        "segments": segments,
    }


def main():
    print("Loading data...")
    rounds, spot, btc = _load_data()
    total_rounds = len(rounds)
    print(f"Loaded {total_rounds} rounds, {len(spot)} spot epochs, {len(btc)} BTC epochs")
    print()

    # Save original threshold to restore later
    original_thresh = mg._ACCEL_THRESH

    results = []

    for thresh, label in THRESHOLDS:
        # Monkey-patch the module-level threshold
        mg._ACCEL_THRESH = thresh
        print(f"Running thresh={thresh:.5f} ({label})...", end=" ", flush=True)

        # Run on all available rounds
        r = run_backtest(rounds, spot, btc, sim_size=999999)
        results.append((thresh, label, r))
        print(f"PnL={r['net']:+.2f}, Bets={r['bets']}, WR={r['wr']:.1f}%")

    # Restore original
    mg._ACCEL_THRESH = original_thresh

    # Summary table
    print()
    print("=" * 90)
    print(f"{'Thresh':>10} {'Label':>7} {'PnL':>10} {'Bets':>6} {'WR%':>7} {'WorstSeg':>10} {'Bets/Seg':>10}")
    print("-" * 90)
    for thresh, label, r in results:
        avg_bets_seg = sum(s[0] for s in r["segments"]) / max(1, len(r["segments"]))
        print(
            f"{thresh:>10.5f} {label:>7} {r['net']:>+10.2f} {r['bets']:>6} "
            f"{r['wr']:>6.1f}% {r['worst_seg']:>+10.2f} {avg_bets_seg:>10.0f}"
        )
    print("=" * 90)

    # Per-segment detail for each threshold
    print()
    print("Per-segment breakdown:")
    for thresh, label, r in results:
        print(f"\n  {label} (thresh={thresh:.5f}):")
        for i, (nb, wr, pnl) in enumerate(r["segments"]):
            print(f"    Seg{i+1}: {nb:4d} bets, WR={wr:5.1f}%, PnL={pnl:+8.2f}")


if __name__ == "__main__":
    main()
