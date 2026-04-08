"""Backtest runner — momentum-only offline replay.

Iterates closed rounds in chronological order, runs MomentumOnlyPipeline
(no live OKX gate; uses local klines cache instead), and settles each bet
against the closed-round pool data from closed_rounds.jsonl.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

from pancakebot.backtest.config import BacktestConfig
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info
from pancakebot.domain.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.domain.types import Kline, Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round


@dataclass(slots=True)
class _BacktestStats:
    num_bets: int = 0
    num_bets_bull: int = 0
    num_bets_bear: int = 0
    num_wins: int = 0
    num_wins_bull: int = 0
    num_wins_bear: int = 0
    gross_profit_bnb: float = 0.0
    gross_loss_bnb: float = 0.0
    skip_counts_by_reason: dict[str, int] = field(default_factory=dict)

    def count_skip(self, reason: str) -> None:
        key = str(reason).strip() or "unknown_skip_reason"
        self.skip_counts_by_reason[key] = self.skip_counts_by_reason.get(key, 0) + 1


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if int(den) > 0 else 0.0


def _load_all_rounds(runtime_cfg) -> list[Round]:
    """Load closed rounds from round_store (JSONL) or market_data_store (SQLite)."""
    market_data_store = getattr(runtime_cfg, "market_data_store", None)
    if market_data_store is not None and hasattr(market_data_store, "iter_closed_rounds"):
        return list(market_data_store.iter_closed_rounds())
    if runtime_cfg.round_store is not None:
        return list(runtime_cfg.round_store.iter_closed_rounds())
    raise InvariantError("backtest_no_round_store_available")


def _load_all_klines(runtime_cfg) -> list[Kline]:
    """Load all klines from klines_store."""
    if runtime_cfg.klines_store is None:
        raise InvariantError("backtest_klines_store_missing")
    klines = list(runtime_cfg.klines_store.iter_klines())
    if not klines:
        raise InvariantError("backtest_klines_store_empty")
    return klines


def run_backtest(*, runtime_cfg, backtest_cfg: BacktestConfig, out_dir: Path) -> None:
    backtest_cfg.validate()

    simulation_size = int(backtest_cfg.simulation_size)
    tail_offset_rounds = int(backtest_cfg.tail_offset_rounds)
    initial_bankroll_bnb = float(backtest_cfg.initial_bankroll_bnb)

    info("BACK", "SETUP", "START", msg="Loading closed rounds and klines")
    t0 = time.perf_counter()

    all_rounds = _load_all_rounds(runtime_cfg)
    if not all_rounds:
        raise InvariantError("backtest_no_closed_rounds")

    all_klines = _load_all_klines(runtime_cfg)

    # Select simulation window: epoch range takes priority over tail window.
    epoch_start = backtest_cfg.epoch_start
    epoch_end = backtest_cfg.epoch_end
    if epoch_start is not None or epoch_end is not None:
        sim_rounds = [
            r for r in all_rounds
            if (epoch_start is None or int(r.epoch) >= int(epoch_start))
            and (epoch_end is None or int(r.epoch) <= int(epoch_end))
        ]
        if not sim_rounds:
            raise InvariantError(
                f"backtest_no_rounds_in_epoch_range: start={epoch_start} end={epoch_end}"
            )
    else:
        effective_end = len(all_rounds) - int(tail_offset_rounds)
        if effective_end <= 0:
            raise InvariantError("backtest_tail_offset_exceeds_rounds")
        sim_rounds = all_rounds[max(0, effective_end - simulation_size): effective_end]
        if len(sim_rounds) < simulation_size:
            raise InvariantError(
                f"backtest_insufficient_rounds: need={simulation_size} have={len(sim_rounds)}"
            )

    first_epoch = int(sim_rounds[0].epoch)
    last_epoch = int(sim_rounds[-1].epoch)
    elapsed_load = float(time.perf_counter()) - float(t0)
    info(
        "BACK",
        "SETUP",
        "LOADED",
        msg=(
            f"rounds={len(all_rounds)} klines={len(all_klines)} "
            f"sim_rounds={len(sim_rounds)} epochs=[{first_epoch}..{last_epoch}] "
            f"load_elapsed={elapsed_load:.1f}s"
        ),
    )

    # Build momentum pipeline (no live gate — backtest uses klines cache).
    from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
    gate_config: MomentumGateConfig = runtime_cfg.momentum_gate_config  # type: ignore[assignment]
    pipeline = MomentumOnlyPipeline(
        config=gate_config,
        gate=None,
        cutoff_seconds=int(runtime_cfg.cutoff_seconds),
        min_bet_amount_bnb=float(runtime_cfg.min_bet_amount_bnb),
    )
    pipeline.refresh_klines(klines=list(all_klines))
    info("BACK", "SETUP", "PIPELINE", msg="MomentumOnlyPipeline ready (backtest/klines mode)")

    # Output files.
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "backtest_trades.csv"
    summary_path = out_dir / "backtest_summary.json"

    bankroll = float(initial_bankroll_bnb)
    stats = _BacktestStats()

    t_sim_start = time.perf_counter()
    with open(trades_path, "w", newline="", encoding="utf-8") as trades_f:
        trades_w = csv.writer(trades_f)
        trades_w.writerow(
            [
                "epoch",
                "action",
                "skip_reason",
                "direction",
                "bet_size_bnb",
                "ret_1m_signal",
                "profit_bnb",
                "bankroll_bnb",
            ]
        )

        for i, round_t in enumerate(sim_rounds):
            decision = pipeline.decide_open_round(
                round_t=round_t,
                bankroll_bnb=float(bankroll),
                allow_oracle_mode=True,
            )

            profit = 0.0
            if decision.action == "BET" and float(decision.bet_size_bnb) > 0.0:
                bet_side = str(decision.bet_side)
                if bet_side not in ("Bull", "Bear"):
                    raise InvariantError("backtest_bet_side_invalid")

                bankroll -= float(decision.bet_size_bnb) + float(GAS_COST_BET_BNB)
                outcome = settle_bet_against_closed_round(
                    bet_bnb=float(decision.bet_size_bnb),
                    bet_side=str(bet_side),
                    round_closed=round_t,
                    treasury_fee_fraction=float(runtime_cfg.treasury_fee_fraction),
                )
                credit_bnb = float(outcome.credit_bnb)
                bankroll += float(credit_bnb)
                profit = float(credit_bnb) - float(decision.bet_size_bnb) - float(GAS_COST_BET_BNB)

                stats.num_bets += 1
                if bet_side == "Bull":
                    stats.num_bets_bull += 1
                else:
                    stats.num_bets_bear += 1

                if str(outcome.outcome) == "win":
                    stats.num_wins += 1
                    if bet_side == "Bull":
                        stats.num_wins_bull += 1
                    else:
                        stats.num_wins_bear += 1

                if float(profit) > 0.0:
                    stats.gross_profit_bnb += float(profit)
                elif float(profit) < 0.0:
                    stats.gross_loss_bnb += -float(profit)
            else:
                stats.count_skip(str(decision.skip_reason or "unknown_skip_reason"))

            trades_w.writerow(
                [
                    int(round_t.epoch),
                    str(decision.action),
                    str(decision.skip_reason or ""),
                    str(decision.bet_side or ""),
                    float(decision.bet_size_bnb),
                    float(decision.p_bull) if decision.p_bull is not None else "",
                    float(profit),
                    float(bankroll),
                ]
            )

            pipeline.settle_closed_rounds(rounds=[round_t])

            if (i + 1) % 500 == 0:
                info(
                    "BACK",
                    "PROG",
                    "SIM",
                    msg=f"idx={i+1}/{len(sim_rounds)} bankroll={bankroll:.4f} BNB",
                )

    elapsed_sim = float(time.perf_counter()) - float(t_sim_start)

    # Summary statistics.
    total_rounds = len(sim_rounds)
    total_skips = int(total_rounds) - int(stats.num_bets)
    net_pnl = float(bankroll) - float(initial_bankroll_bnb)
    win_rate = _safe_rate(stats.num_wins, stats.num_bets)
    bet_rate = _safe_rate(stats.num_bets, total_rounds)

    info(
        "BACK",
        "RESULT",
        "SUMMARY",
        msg=(
            f"rounds={total_rounds} bets={stats.num_bets} "
            f"win_rate={win_rate:.1%} bet_rate={bet_rate:.1%} "
            f"net_pnl={net_pnl:+.4f} BNB "
            f"final_bankroll={bankroll:.4f} BNB "
            f"elapsed={elapsed_sim:.1f}s"
        ),
    )

    import json
    skip_detail = dict(sorted(stats.skip_counts_by_reason.items(), key=lambda x: -x[1]))
    summary = {
        "simulation_size": int(total_rounds),
        "first_epoch": int(first_epoch),
        "last_epoch": int(last_epoch),
        "initial_bankroll_bnb": float(initial_bankroll_bnb),
        "final_bankroll_bnb": float(bankroll),
        "net_pnl_bnb": float(net_pnl),
        "num_bets": int(stats.num_bets),
        "num_bets_bull": int(stats.num_bets_bull),
        "num_bets_bear": int(stats.num_bets_bear),
        "num_wins": int(stats.num_wins),
        "num_wins_bull": int(stats.num_wins_bull),
        "num_wins_bear": int(stats.num_wins_bear),
        "num_skips": int(total_skips),
        "win_rate": float(win_rate),
        "bet_rate": float(bet_rate),
        "gross_profit_bnb": float(stats.gross_profit_bnb),
        "gross_loss_bnb": float(stats.gross_loss_bnb),
        "skip_counts_by_reason": dict(skip_detail),
        "elapsed_sim_seconds": float(elapsed_sim),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    info("BACK", "RESULT", "FILES", msg=f"trades={trades_path} summary={summary_path}")
