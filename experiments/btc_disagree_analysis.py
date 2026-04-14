"""Analyze BTC agree/disagree/neutral/contrarian bet performance.

Runs the full production backtest, independently computes BTC confirmation
status for every BET decision, and reports segmented breakdown.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import (
    MomentumGateConfig,
    MomentumGateResult,
    compute_signal_from_klines,
    _trim_to_window,
    _CANDLE_COUNT,
    _get_return,
    _BTC_LOOKBACK,
    _BTC_THRESH,
)
from pancakebot.domain.strategy.momentum_pipeline import (
    MomentumOnlyPipeline,
    _BTC_CONTRA_MIN_PAYOUT,
    _BTC_CONTRA_BET_BNB,
    _BTC_CONTRA_THRESH,
)
from pancakebot.runtime.settlement import settle_bet_against_closed_round

# Production constants
_CUTOFF_SECONDS = 4
_MIN_BET_AMOUNT_BNB = 0.001
_TREASURY_FEE_FRACTION = 0.03

# ---- Data loading ----

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


def classify_btc_status(
    epoch: int,
    cutoff_ts_ms: int,
    bet_side: str,
    bnb_klines_by_epoch: dict,
    btc_klines_by_epoch: dict,
) -> str:
    """Independently classify a bet as btc_agree, btc_disagree, btc_neutral, or contrarian.

    Uses compute_signal_from_klines to get the signal + btc_agrees/btc_disagrees.
    For contrarian bets (main gate has no signal), returns 'contrarian'.
    """
    bnb_klines = bnb_klines_by_epoch.get(epoch)
    btc_klines = btc_klines_by_epoch.get(epoch)

    if bnb_klines is None or len(bnb_klines) == 0:
        return "unknown"

    result = compute_signal_from_klines(bnb_klines, btc_klines, cutoff_ts_ms)

    # If the main gate has no signal, this must be a contrarian bet
    if result.signal is None:
        return "contrarian"

    if result.btc_agrees:
        return "btc_agree"
    elif result.btc_disagrees:
        return "btc_disagree"
    else:
        return "btc_neutral"


def run_analysis():
    rounds, spot, btc = _load_data()

    # Use all available rounds
    sim_rounds = rounds

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

    # Collect trades with BTC classification
    # (epoch, action, profit, bankroll, btc_status, bet_side, bet_size, outcome)
    trades = []

    for rnd in sim_rounds:
        epoch = int(rnd.epoch)
        lock_at = int(rnd.lock_at)
        cutoff_ts_ms = (lock_at - _CUTOFF_SECONDS) * 1000

        decision = pipeline.decide_open_round(
            round_t=rnd,
            bankroll_bnb=bankroll,
            allow_oracle_mode=True,
        )

        if decision.action == "BET" and decision.bet_size_bnb > 0.0:
            # Classify BTC status independently
            btc_status = classify_btc_status(
                epoch=epoch,
                cutoff_ts_ms=cutoff_ts_ms,
                bet_side=decision.bet_side,
                bnb_klines_by_epoch=spot,
                btc_klines_by_epoch=btc,
            )

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
                epoch, "BET", profit, bankroll,
                btc_status, decision.bet_side, decision.bet_size_bnb,
                outcome.outcome,
            ))
        else:
            trades.append((
                epoch, "SKIP", 0.0, bankroll,
                None, None, 0.0,
                decision.skip_reason or "skip",
            ))

        pipeline.settle_closed_rounds(rounds=[rnd])

    net = bankroll - initial_bankroll
    bet_trades = [t for t in trades if t[1] == "BET"]

    # ---- Overall summary ----
    print("=" * 80)
    print("BTC DISAGREE ANALYSIS - Full Backtest")
    print(f"Total rounds: {len(trades)} | Total bets: {len(bet_trades)}")
    total_wins = sum(1 for t in bet_trades if t[2] > 0)
    wr = total_wins / max(1, len(bet_trades)) * 100
    print(f"Overall NET: {net:+.2f} BNB | WR: {wr:.1f}%")
    print("=" * 80)

    # ---- Breakdown by BTC status ----
    categories = ["btc_agree", "btc_disagree", "btc_neutral", "contrarian", "unknown"]
    cat_labels = {
        "btc_agree": "BTC Agree",
        "btc_disagree": "BTC Disagree",
        "btc_neutral": "BTC Neutral",
        "contrarian": "BTC Contrarian (tier=None)",
        "unknown": "Unknown",
    }

    print()
    print("-" * 80)
    print(f"{'Category':<30} {'Count':>6} {'WR%':>7} {'PnL':>10} {'AvgBet':>8} {'AvgProfit':>10}")
    print("-" * 80)

    for cat in categories:
        cat_bets = [t for t in bet_trades if t[4] == cat]
        if not cat_bets:
            print(f"{cat_labels[cat]:<30} {'0':>6}")
            continue
        count = len(cat_bets)
        wins = sum(1 for t in cat_bets if t[2] > 0)
        pnl = sum(t[2] for t in cat_bets)
        avg_bet = sum(t[6] for t in cat_bets) / count
        avg_profit = pnl / count
        wr_cat = wins / count * 100
        print(f"{cat_labels[cat]:<30} {count:>6} {wr_cat:>6.1f}% {pnl:>+9.2f} {avg_bet:>8.4f} {avg_profit:>+9.4f}")

    print("-" * 80)

    # ---- BTC disagree segmented breakdown ----
    disagree_bets = [t for t in bet_trades if t[4] == "btc_disagree"]
    if disagree_bets:
        print()
        print("=" * 80)
        print("BTC DISAGREE - Segmented Breakdown (6 segments)")
        print("=" * 80)

        n_segments = 6
        seg_size = len(disagree_bets) // n_segments
        if seg_size == 0:
            seg_size = len(disagree_bets)
            n_segments = 1

        print(f"{'Segment':<10} {'Count':>6} {'WR%':>7} {'PnL':>10} {'AvgBet':>8} {'Wins':>6} {'Losses':>6}")
        print("-" * 60)

        for s in range(n_segments):
            start = s * seg_size
            end = (s + 1) * seg_size if s < n_segments - 1 else len(disagree_bets)
            chunk = disagree_bets[start:end]
            count = len(chunk)
            wins = sum(1 for t in chunk if t[2] > 0)
            losses = count - wins
            pnl = sum(t[2] for t in chunk)
            avg_bet = sum(t[6] for t in chunk) / count
            wr_seg = wins / count * 100
            print(f"  Seg{s+1:<5} {count:>6} {wr_seg:>6.1f}% {pnl:>+9.2f} {avg_bet:>8.4f} {wins:>6} {losses:>6}")

        print("-" * 60)

    # ---- BTC agree segmented breakdown (for comparison) ----
    agree_bets = [t for t in bet_trades if t[4] == "btc_agree"]
    if agree_bets:
        print()
        print("=" * 80)
        print("BTC AGREE - Segmented Breakdown (6 segments, for comparison)")
        print("=" * 80)

        n_segments = 6
        seg_size = len(agree_bets) // n_segments
        if seg_size == 0:
            seg_size = len(agree_bets)
            n_segments = 1

        print(f"{'Segment':<10} {'Count':>6} {'WR%':>7} {'PnL':>10} {'AvgBet':>8} {'Wins':>6} {'Losses':>6}")
        print("-" * 60)

        for s in range(n_segments):
            start = s * seg_size
            end = (s + 1) * seg_size if s < n_segments - 1 else len(agree_bets)
            chunk = agree_bets[start:end]
            count = len(chunk)
            wins = sum(1 for t in chunk if t[2] > 0)
            losses = count - wins
            pnl = sum(t[2] for t in chunk)
            avg_bet = sum(t[6] for t in chunk) / count
            wr_seg = wins / count * 100
            print(f"  Seg{s+1:<5} {count:>6} {wr_seg:>6.1f}% {pnl:>+9.2f} {avg_bet:>8.4f} {wins:>6} {losses:>6}")

        print("-" * 60)

    # ---- BTC neutral segmented breakdown ----
    neutral_bets = [t for t in bet_trades if t[4] == "btc_neutral"]
    if neutral_bets:
        print()
        print("=" * 80)
        print("BTC NEUTRAL - Segmented Breakdown (6 segments, for comparison)")
        print("=" * 80)

        n_segments = 6
        seg_size = len(neutral_bets) // n_segments
        if seg_size == 0:
            seg_size = len(neutral_bets)
            n_segments = 1

        print(f"{'Segment':<10} {'Count':>6} {'WR%':>7} {'PnL':>10} {'AvgBet':>8} {'Wins':>6} {'Losses':>6}")
        print("-" * 60)

        for s in range(n_segments):
            start = s * seg_size
            end = (s + 1) * seg_size if s < n_segments - 1 else len(neutral_bets)
            chunk = neutral_bets[start:end]
            count = len(chunk)
            wins = sum(1 for t in chunk if t[2] > 0)
            losses = count - wins
            pnl = sum(t[2] for t in chunk)
            avg_bet = sum(t[6] for t in chunk) / count
            wr_seg = wins / count * 100
            print(f"  Seg{s+1:<5} {count:>6} {wr_seg:>6.1f}% {pnl:>+9.2f} {avg_bet:>8.4f} {wins:>6} {losses:>6}")

        print("-" * 60)

    # ---- Counterfactual: skip all BTC-disagree bets ----
    print()
    print("=" * 80)
    print("COUNTERFACTUAL: What if we skip all BTC-disagree bets?")
    print("=" * 80)

    # Re-run the backtest, skipping BTC-disagree decisions
    bankroll_cf = 50.0
    trades_cf = []

    # Rebuild pipeline
    pipeline_cf = MomentumOnlyPipeline(
        config=gate_config,
        gate=None,
        cutoff_seconds=_CUTOFF_SECONDS,
        min_bet_amount_bnb=_MIN_BET_AMOUNT_BNB,
        treasury_fee_fraction=_TREASURY_FEE_FRACTION,
    )
    pipeline_cf.refresh_bnb_klines(bnb_klines_by_epoch=spot)
    pipeline_cf.refresh_btc_klines(btc_klines_by_epoch=btc)

    for rnd in sim_rounds:
        epoch = int(rnd.epoch)
        lock_at = int(rnd.lock_at)
        cutoff_ts_ms = (lock_at - _CUTOFF_SECONDS) * 1000

        decision = pipeline_cf.decide_open_round(
            round_t=rnd,
            bankroll_bnb=bankroll_cf,
            allow_oracle_mode=True,
        )

        if decision.action == "BET" and decision.bet_size_bnb > 0.0:
            btc_status = classify_btc_status(
                epoch=epoch,
                cutoff_ts_ms=cutoff_ts_ms,
                bet_side=decision.bet_side,
                bnb_klines_by_epoch=spot,
                btc_klines_by_epoch=btc,
            )

            if btc_status == "btc_disagree":
                # SKIP this bet in the counterfactual
                trades_cf.append((epoch, "SKIP_CF", 0.0, bankroll_cf, btc_status))
            else:
                bankroll_cf -= decision.bet_size_bnb + GAS_COST_BET_BNB
                outcome = settle_bet_against_closed_round(
                    bet_bnb=decision.bet_size_bnb,
                    bet_side=decision.bet_side,
                    round_closed=rnd,
                    treasury_fee_fraction=_TREASURY_FEE_FRACTION,
                )
                bankroll_cf += outcome.credit_bnb
                profit = outcome.credit_bnb - decision.bet_size_bnb - GAS_COST_BET_BNB
                trades_cf.append((epoch, "BET", profit, bankroll_cf, btc_status))
        else:
            trades_cf.append((epoch, "SKIP", 0.0, bankroll_cf, None))

        pipeline_cf.settle_closed_rounds(rounds=[rnd])

    net_cf = bankroll_cf - 50.0
    bets_cf = [t for t in trades_cf if t[1] == "BET"]
    wins_cf = sum(1 for t in bets_cf if t[2] > 0)
    wr_cf = wins_cf / max(1, len(bets_cf)) * 100

    print(f"  Original:       NET={net:+.2f} BNB, Bets={len(bet_trades)}, WR={wr:.1f}%")
    print(f"  Skip-disagree:  NET={net_cf:+.2f} BNB, Bets={len(bets_cf)}, WR={wr_cf:.1f}%")
    print(f"  Delta:          {net_cf - net:+.2f} BNB")

    disagree_pnl = sum(t[2] for t in bet_trades if t[4] == "btc_disagree")
    print(f"  Disagree PnL:   {disagree_pnl:+.2f} BNB (removed)")
    print()

    # ---- Win rate comparison by bet size buckets for disagree ----
    if disagree_bets:
        print("=" * 80)
        print("BTC DISAGREE - Win Rate by Bet Size Bucket")
        print("=" * 80)
        sizes = [t[6] for t in disagree_bets]
        min_s, max_s = min(sizes), max(sizes)
        buckets = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 2.5)]
        print(f"  Bet size range [{min_s:.4f}, {max_s:.4f}]")
        print(f"  {'Bucket':<15} {'Count':>6} {'WR%':>7} {'PnL':>10}")
        print("  " + "-" * 45)
        for lo, hi in buckets:
            bucket = [t for t in disagree_bets if lo <= t[6] < hi]
            if not bucket:
                continue
            c = len(bucket)
            w = sum(1 for t in bucket if t[2] > 0)
            p = sum(t[2] for t in bucket)
            print(f"  [{lo:.1f}, {hi:.1f}){'':<5} {c:>6} {w/c*100:>6.1f}% {p:>+9.2f}")

    print()
    print("DONE")


if __name__ == "__main__":
    run_analysis()
