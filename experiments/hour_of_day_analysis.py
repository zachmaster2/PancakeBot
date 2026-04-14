"""Hour-of-day analysis on the FULL dataset using production MomentumOnlyPipeline.

Runs ALL available rounds (sim_size=999999) and breaks down performance
by UTC hour (0-23).  Flags any hour with n >= 50 AND WR < 52% AND PnL < 0.
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

# Production constants
_CUTOFF_SECONDS = 4
_MIN_BET_AMOUNT_BNB = 0.001
_TREASURY_FEE_FRACTION = 0.03


def _load_data():
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


def main():
    rounds, spot, btc = _load_data()
    sim_rounds = rounds[-999999:]  # all available
    print(f"Total rounds in dataset: {len(sim_rounds)}")

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
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch=spot)
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc)

    bankroll = 50.0
    initial_bankroll = bankroll

    # Per-hour accumulators: {hour: {"bets": 0, "wins": 0, "pnl": 0.0}}
    hour_stats = {h: {"bets": 0, "wins": 0, "pnl": 0.0} for h in range(24)}

    for rnd in sim_rounds:
        lock_at = int(rnd.lock_at)
        hour_utc = (lock_at % 86400) // 3600

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

            hour_stats[hour_utc]["bets"] += 1
            hour_stats[hour_utc]["pnl"] += profit
            if profit > 0:
                hour_stats[hour_utc]["wins"] += 1

        pipeline.settle_closed_rounds(rounds=[rnd])

    net = bankroll - initial_bankroll

    # Report
    print(f"\nOverall NET: {net:+.2f} BNB")
    print(f"{'Hour':>4}  {'Bets':>6}  {'WR%':>6}  {'PnL':>9}  {'Flag':>6}")
    print("-" * 42)

    flagged = []
    for h in range(24):
        s = hour_stats[h]
        n = s["bets"]
        wr = (s["wins"] / n * 100) if n > 0 else 0.0
        pnl = s["pnl"]
        flag = ""
        if n >= 50 and wr < 52.0 and pnl < 0:
            flag = "  <<<";
            flagged.append(h)
        print(f"{h:4d}  {n:6d}  {wr:5.1f}%  {pnl:+8.2f}  {flag}")

    print("-" * 42)
    total_bets = sum(s["bets"] for s in hour_stats.values())
    total_wins = sum(s["wins"] for s in hour_stats.values())
    total_pnl = sum(s["pnl"] for s in hour_stats.values())
    wr_all = (total_wins / total_bets * 100) if total_bets > 0 else 0.0
    print(f" ALL  {total_bets:6d}  {wr_all:5.1f}%  {total_pnl:+8.2f}")

    if flagged:
        print(f"\nFLAGGED hours (n>=50, WR<52%, PnL<0): {flagged}")
    else:
        print("\nNo hours flagged.")


if __name__ == "__main__":
    main()
