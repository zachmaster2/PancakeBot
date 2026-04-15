"""Backtest runner — momentum-only offline replay.

Iterates closed rounds in chronological order, runs MomentumOnlyPipeline
(no live OKX gate; uses cached 1s klines instead), and settles each
bet against the closed-round pool data from closed_rounds.jsonl.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pancakebot.backtest.config import BacktestConfig
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info
from pancakebot.domain.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.domain.types import Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round

_BNB_KLINES_PATH = Path("var/bnb_spot_prices.jsonl")
_BTC_KLINES_PATH = Path("var/btc_spot_prices.jsonl")
_ETH_KLINES_PATH = Path("var/eth_spot_prices.jsonl")
_SOL_KLINES_PATH = Path("var/sol_spot_prices.jsonl")


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
        key = reason.strip() or "unknown_skip_reason"
        self.skip_counts_by_reason[key] = self.skip_counts_by_reason.get(key, 0) + 1


def _safe_rate(num: int, den: int) -> float:
    return num / den if den > 0 else 0.0


def _load_all_rounds(runtime_cfg) -> list[Round]:
    """Load closed rounds from round_store (JSONL) or market_data_store (SQLite)."""
    market_data_store = getattr(runtime_cfg, "market_data_store", None)
    if market_data_store is not None and hasattr(market_data_store, "iter_closed_rounds"):
        return list(market_data_store.iter_closed_rounds())
    if runtime_cfg.round_store is not None:
        return list(runtime_cfg.round_store.iter_closed_rounds())
    raise InvariantError("backtest_no_round_store_available")


def _load_klines_from(path: Path) -> dict[int, list[list]]:
    """Load pre-fetched 1s kline arrays from a JSONL file.

    Returns {epoch: [[ts_ms, o, h, l, c, vol], ...]} for epochs that
    have data.  Returns empty dict if the file doesn't exist.
    """
    if not path.exists():
        return {}
    result: dict[int, list[list]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error") or rec.get("klines_1s") is None:
            continue
        result[int(rec["epoch"])] = rec["klines_1s"]
    return result


def run_backtest(*, runtime_cfg, backtest_cfg: BacktestConfig, out_dir: Path) -> None:
    backtest_cfg.validate()

    simulation_size = backtest_cfg.simulation_size
    tail_offset_rounds = backtest_cfg.tail_offset_rounds
    initial_bankroll_bnb = backtest_cfg.initial_bankroll_bnb

    info("BACK", "SETUP", "START", msg="Loading closed rounds and klines")
    t0 = time.perf_counter()

    all_rounds = _load_all_rounds(runtime_cfg)
    if not all_rounds:
        raise InvariantError("backtest_no_closed_rounds")

    # Select simulation window: epoch range takes priority over tail window.
    epoch_start = backtest_cfg.epoch_start
    epoch_end = backtest_cfg.epoch_end
    if epoch_start is not None or epoch_end is not None:
        sim_rounds = [
            r for r in all_rounds
            if (epoch_start is None or r.epoch >= epoch_start)
            and (epoch_end is None or r.epoch <= epoch_end)
        ]
        if not sim_rounds:
            raise InvariantError(
                f"backtest_no_rounds_in_epoch_range: start={epoch_start} end={epoch_end}"
            )
    else:
        effective_end = len(all_rounds) - tail_offset_rounds
        if effective_end <= 0:
            raise InvariantError("backtest_tail_offset_exceeds_rounds")
        sim_rounds = all_rounds[max(0, effective_end - simulation_size): effective_end]
        if len(sim_rounds) < simulation_size:
            raise InvariantError(
                f"backtest_insufficient_rounds: need={simulation_size} have={len(sim_rounds)}"
            )

    first_epoch = sim_rounds[0].epoch
    last_epoch = sim_rounds[-1].epoch
    elapsed_load = time.perf_counter() - t0
    info(
        "BACK",
        "SETUP",
        "LOADED",
        msg=(
            f"rounds={len(all_rounds)} "
            f"sim_rounds={len(sim_rounds)} epochs=[{first_epoch}..{last_epoch}] "
            f"load_elapsed={elapsed_load:.1f}s"
        ),
    )

    # Load pre-fetched 1s klines for honest backtest signal.
    bnb_klines = _load_klines_from(_BNB_KLINES_PATH)
    if bnb_klines:
        info("BACK", "SETUP", "BNB_KL", msg=f"Loaded BNB 1s klines for {len(bnb_klines)} epochs")
    else:
        info("BACK", "SETUP", "BNB_KL", msg="No BNB klines found — backtest will skip all rounds")

    btc_klines = _load_klines_from(_BTC_KLINES_PATH)
    if btc_klines:
        info("BACK", "SETUP", "BTC_KL", msg=f"Loaded BTC 1s klines for {len(btc_klines)} epochs")

    eth_klines = _load_klines_from(_ETH_KLINES_PATH) if _ETH_KLINES_PATH.exists() else {}
    if eth_klines:
        info("BACK", "SETUP", "ETH_KL", msg=f"Loaded ETH 1s klines for {len(eth_klines)} epochs")

    sol_klines = _load_klines_from(_SOL_KLINES_PATH) if _SOL_KLINES_PATH.exists() else {}
    if sol_klines:
        info("BACK", "SETUP", "SOL_KL", msg=f"Loaded SOL 1s klines for {len(sol_klines)} epochs")

    # Build momentum pipeline (no live gate — backtest uses cached 1s klines).
    from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
    gate_config: MomentumGateConfig = runtime_cfg.momentum_gate_config  # type: ignore[assignment]
    pipeline = MomentumOnlyPipeline(
        config=gate_config,
        gate=None,
        cutoff_seconds=runtime_cfg.cutoff_seconds,
        min_bet_amount_bnb=runtime_cfg.min_bet_amount_bnb,
        treasury_fee_fraction=runtime_cfg.treasury_fee_fraction,
    )
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch=bnb_klines)
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    info("BACK", "SETUP", "PIPELINE", msg="MomentumOnlyPipeline ready (backtest/4-asset mode)")

    # Output files.
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "backtest_trades.csv"
    summary_path = out_dir / "backtest_summary.json"

    bankroll = initial_bankroll_bnb
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
                bankroll_bnb=bankroll,
                allow_oracle_mode=True,
            )

            profit = 0.0
            if decision.action == "BET" and decision.bet_size_bnb > 0.0:
                bet_side = decision.bet_side
                if bet_side not in ("Bull", "Bear"):
                    raise InvariantError("backtest_bet_side_invalid")

                bankroll -= decision.bet_size_bnb + GAS_COST_BET_BNB
                outcome = settle_bet_against_closed_round(
                    bet_bnb=decision.bet_size_bnb,
                    bet_side=bet_side,
                    round_closed=round_t,
                    treasury_fee_fraction=runtime_cfg.treasury_fee_fraction,
                )
                bankroll += outcome.credit_bnb
                profit = outcome.credit_bnb - decision.bet_size_bnb - GAS_COST_BET_BNB

                stats.num_bets += 1
                if bet_side == "Bull":
                    stats.num_bets_bull += 1
                else:
                    stats.num_bets_bear += 1

                if outcome.outcome == "win":
                    stats.num_wins += 1
                    if bet_side == "Bull":
                        stats.num_wins_bull += 1
                    else:
                        stats.num_wins_bear += 1

                if profit > 0.0:
                    stats.gross_profit_bnb += profit
                elif profit < 0.0:
                    stats.gross_loss_bnb += -profit

            else:
                stats.count_skip(decision.skip_reason or "unknown_skip_reason")

            trades_w.writerow(
                [
                    round_t.epoch,
                    decision.action,
                    decision.skip_reason or "",
                    decision.bet_side or "",
                    decision.bet_size_bnb,
                    decision.p_bull if decision.p_bull is not None else "",
                    profit,
                    bankroll,
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

    elapsed_sim = time.perf_counter() - t_sim_start

    # Summary statistics.
    total_rounds = len(sim_rounds)
    total_skips = total_rounds - stats.num_bets
    net_pnl = bankroll - initial_bankroll_bnb
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

    skip_detail = dict(sorted(stats.skip_counts_by_reason.items(), key=lambda x: -x[1]))
    summary = {
        "simulation_size": total_rounds,
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "initial_bankroll_bnb": initial_bankroll_bnb,
        "final_bankroll_bnb": bankroll,
        "net_pnl_bnb": net_pnl,
        "num_bets": stats.num_bets,
        "num_bets_bull": stats.num_bets_bull,
        "num_bets_bear": stats.num_bets_bear,
        "num_wins": stats.num_wins,
        "num_wins_bull": stats.num_wins_bull,
        "num_wins_bear": stats.num_wins_bear,
        "num_skips": total_skips,
        "win_rate": win_rate,
        "bet_rate": bet_rate,
        "gross_profit_bnb": stats.gross_profit_bnb,
        "gross_loss_bnb": stats.gross_loss_bnb,
        "skip_counts_by_reason": skip_detail,
        "elapsed_sim_seconds": elapsed_sim,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Generate equity curve plot.
    equity_path = out_dir / "equity_curves.png"
    _plot_equity_curve(trades_path, equity_path, initial_bankroll_bnb, total_rounds)
    info("BACK", "RESULT", "FILES", msg=f"trades={trades_path} summary={summary_path} equity={equity_path}")


def _plot_equity_curve(
    trades_path: Path,
    out_path: Path,
    initial_bankroll: float,
    total_rounds: int,
) -> None:
    """Generate equity curve PNG from backtest trades CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bankrolls = []
    with open(trades_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bankrolls.append(float(row["bankroll_bnb"]))

    pnl = [b - initial_bankroll for b in bankrolls]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: cumulative PnL
    axes[0].plot(range(len(pnl)), pnl, color="#E91E63", linewidth=1.5)
    axes[0].set_ylabel("Cumulative PnL (BNB)")
    axes[0].set_title(f"Backtest Equity Curve ({total_rounds} rounds)")
    axes[0].axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    axes[0].grid(True, alpha=0.3)

    # 5-fold boundaries
    fold_size = total_rounds // 5
    for i in range(1, 5):
        axes[0].axvline(x=i * fold_size, color="gray", linewidth=0.5, linestyle=":")
        axes[0].text(i * fold_size, axes[0].get_ylim()[1] * 0.95,
                     f"F{i+1}", fontsize=8, color="gray", ha="center")

    # Bottom: rolling PnL rate
    window = 500
    if len(bankrolls) > window:
        rolling = []
        for i in range(window, len(bankrolls)):
            rate = (bankrolls[i] - bankrolls[i - window]) / window * 2000
            rolling.append(rate)
        axes[1].plot(range(window, len(bankrolls)), rolling,
                     color="#E91E63", linewidth=1.0, alpha=0.8)

    axes[1].set_ylabel("Rolling PnL/2k (500-round window)")
    axes[1].set_xlabel("Round index")
    axes[1].axhline(y=0, color="red", linewidth=0.5, linestyle="--")
    axes[1].grid(True, alpha=0.3)
    for i in range(1, 5):
        axes[1].axvline(x=i * fold_size, color="gray", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
