"""Fast standalone backtest harness for parameter experiments.

Uses MomentumOnlyPipeline directly — guaranteed exact match with
production backtest (runner.py).  Reports per-segment breakdown.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
from pancakebot.domain.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.runtime.settlement import settle_bet_against_closed_round

# Production constants (from config.toml / runtime_cfg)
_CUTOFF_SECONDS = 4
_MIN_BET_AMOUNT_BNB = 0.001
_TREASURY_FEE_FRACTION = 0.03

# ---- Data loading (cached across runs in same process) ----
_cache = {}

def _load_data():
    if _cache:
        return _cache["rounds"], _cache["spot"], _cache["btc"]
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())

    def load_klines(path):
        out = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    spot = load_klines("var/cutoff_spot_prices.jsonl")
    btc = load_klines("var/btc_spot_prices.jsonl")
    _cache.update(rounds=rounds, spot=spot, btc=btc)
    return rounds, spot, btc


# ---- Main backtest function ----

def run(sim_size=20000, verbose=True, initial_bankroll=50.0):
    """Run backtest using production pipeline exactly.

    Returns (net_pnl, segment_results, all_trades).
    """
    rounds, spot, btc = _load_data()
    sim_rounds = rounds[-sim_size:]

    # Build pipeline identical to production runner.py
    gate_config = MomentumGateConfig(
        enabled=True, symbol="BNB-USDT", btc_symbol="BTC-USDT",
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_config,
        gate=None,  # backtest mode — uses cached klines
        cutoff_seconds=_CUTOFF_SECONDS,
        min_bet_amount_bnb=_MIN_BET_AMOUNT_BNB,
        treasury_fee_fraction=_TREASURY_FEE_FRACTION,
    )
    pipeline.refresh_spot_klines(spot_klines_by_epoch=spot)
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc)

    bankroll = initial_bankroll
    trades = []  # (epoch, action, profit, bankroll, tier, side, outcome_or_skip)

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

            trades.append((
                int(rnd.epoch), "BET", profit, bankroll,
                None, decision.bet_side, outcome.outcome,
            ))
        else:
            trades.append((
                int(rnd.epoch), "SKIP", 0.0, bankroll,
                None, None, decision.skip_reason or "skip",
            ))

        pipeline.settle_closed_rounds(rounds=[rnd])

    net = bankroll - initial_bankroll

    # Segment analysis — fixed segments based on actual round count
    n_segments = max(1, len(trades) // 5000)
    seg_size = len(trades) // n_segments
    segments = []
    for s in range(n_segments):
        start = s * seg_size
        end = (s + 1) * seg_size if s < n_segments - 1 else len(trades)
        chunk = trades[start:end]
        bets = [t for t in chunk if t[1] == "BET"]
        wins = [t for t in bets if t[2] > 0]
        pnl = sum(t[2] for t in bets)
        wr = len(wins) / len(bets) * 100 if bets else 0
        segments.append((len(bets), wr, pnl))

    if verbose:
        total_bets = sum(1 for t in trades if t[1] == "BET")
        total_wins = sum(1 for t in trades if t[1] == "BET" and t[2] > 0)
        wr_pct = total_wins / max(1, total_bets) * 100
        print(f"NET: {net:+.2f} BNB | Bets: {total_bets} | WR: {wr_pct:.1f}%")
        for i, (nb, wr, pnl) in enumerate(segments):
            print(f"  Seg{i+1}: {nb:4d} bets, WR={wr:5.1f}%, PnL={pnl:+7.2f}")

        # Skip reason breakdown
        skip_reasons = {}
        for t in trades:
            if t[1] == "SKIP":
                reason = t[6]
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        if skip_reasons:
            print("  Skip reasons:")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")

    return net, segments, trades


if __name__ == "__main__":
    # Default: last 20k rounds
    print("=== Last 20k rounds ===")
    run(sim_size=20000)

    print()
    print("=== All rounds ===")
    run(sim_size=999999)  # all available (34k clean rounds)
