"""Replay closed rounds through MomentumOnlyPipeline using cached 1s klines.

Settles each bet against historical pool data, writes per-trade CSV,
summary JSON, and an equity curve PNG.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pancakebot import paths as _paths
from pancakebot.config import BacktestConfig
from pancakebot.constants import BACKTEST_GAS_COST_BET_BNB
from pancakebot.util import InvariantError
from pancakebot.log import info
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.types import Round
from pancakebot.settlement import settle_bet_against_closed_round

_BNB_KLINES_PATH = Path(_paths.BNB_SPOT_PRICES_PATH)
_BTC_KLINES_PATH = Path(_paths.BTC_SPOT_PRICES_PATH)
_ETH_KLINES_PATH = Path(_paths.ETH_SPOT_PRICES_PATH)
_SOL_KLINES_PATH = Path(_paths.SOL_SPOT_PRICES_PATH)

# Extended-data paths (consumed when --use-extended-data is on; the loader
# accepts these as a sibling source of older epochs not in the canonical store.)
_EXT_CLOSED_ROUNDS_PATH = Path(_paths.EXTENDED_CLOSED_ROUNDS_PATH)
_EXT_BNB_KLINES_PATH = Path(_paths.EXTENDED_BNB_SPOT_PRICES_PATH)
_EXT_BTC_KLINES_PATH = Path(_paths.EXTENDED_BTC_SPOT_PRICES_PATH)
_EXT_ETH_KLINES_PATH = Path(_paths.EXTENDED_ETH_SPOT_PRICES_PATH)
_EXT_SOL_KLINES_PATH = Path(_paths.EXTENDED_SOL_SPOT_PRICES_PATH)


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


def _load_all_rounds(runtime_cfg, *, include_failed: bool = False) -> list[Round]:
    """Load closed rounds from round_store (JSONL) or market_data_store (SQLite).

    `include_failed=False` (default) excludes on-chain-failed rounds from the
    returned list. Backtests call with the default to match live betting
    economics: failed rounds refund all bets, so they contribute no signal
    information and would only inflate the iteration count.

    Callers needing the full raw set (e.g. sync-integrity checks, recovery
    tools) pass `include_failed=True`.
    """
    market_data_store = getattr(runtime_cfg, "market_data_store", None)
    if market_data_store is not None and hasattr(market_data_store, "iter_closed_rounds"):
        rounds = list(market_data_store.iter_closed_rounds())
    elif runtime_cfg.round_store is not None:
        rounds = list(runtime_cfg.round_store.iter_closed_rounds())
    else:
        raise InvariantError("backtest_no_round_store_available")

    if not include_failed:
        rounds = [r for r in rounds if not r.failed]
    return rounds


def _load_klines_from(
    path: Path,
    *,
    kline_cutoff_seconds: int,
    candle_count: int,
    extended_path: Path | None = None,
) -> dict[int, list[list]]:
    """Load pre-fetched 1s kline arrays from a JSONL file, pre-sliced to the
    exact window the strategy will read for the given (kline_cutoff_seconds, candle_count).

    Stored records hold 300 candles per round, oldest-first, with
    open_ts = lock_at - 301 .. lock_at - 2 seconds. The strategy at a given
    kline_cutoff_seconds reads the ``candle_count`` candles whose open_ts is in
    ``[lock_at - kline_cutoff_seconds - candle_count, lock_at - kline_cutoff_seconds - 1]``,
    i.e. the last ``candle_count`` candles ending ``(kline_cutoff_seconds - 1)``
    seconds before the most recent stored candle.

    Slice math (negative indexing into the stored list):
      start = -(kline_cutoff_seconds + candle_count - 1)
      end   = -(kline_cutoff_seconds - 1)   if kline_cutoff_seconds >= 2 else None

    Memory-bounded by construction: streams the file line-by-line (no
    ``path.read_text()`` materialising the whole 645 MB BTC file as a
    string) and keeps only ``candle_count`` candles per record. For the
    canonical (cutoff=2, lookbacks=(3,7,15)) baseline this is 16 candles
    per record × 38,306 rounds × 3 pairs ≈ 1 GB peak per process.

    Verified bit-identical on 2026-04-26 cutoff=2 baseline: per-fold
    summary hash ``aa39a3a73f4e4cb718beeffaa72a22ca`` reproduces.
    """
    if not path.exists() and (extended_path is None or not extended_path.exists()):
        return {}
    start_neg = -(kline_cutoff_seconds + candle_count - 1)
    end_neg: int | None = None if kline_cutoff_seconds == 1 else -(kline_cutoff_seconds - 1)
    result: dict[int, list[list]] = {}

    def _ingest(p: Path) -> None:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error") or rec.get("klines_1s") is None:
                    continue
                kl = rec["klines_1s"]
                if end_neg is None:
                    kl = kl[start_neg:]
                else:
                    kl = kl[start_neg:end_neg]
                ep = int(rec["epoch"])
                # Canonical wins on collision (extended ingested second; first writer keeps).
                if ep not in result:
                    result[ep] = kl

    # Canonical first so it wins any epoch collision (none expected).
    if path.exists():
        _ingest(path)
    if extended_path is not None and extended_path.exists():
        _ingest(extended_path)
    return result


def run_backtest(
    *,
    runtime_cfg,
    backtest_cfg: BacktestConfig,
    out_dir: Path,
    use_extended_data: bool = False,
) -> None:
    """Replay historical rounds + klines and produce trades + summary.

    BTC/ETH/SOL klines come from the OKX history-candles cache files
    (var/{btc,eth,sol}_spot_prices.jsonl, populated by `--sync`).

    When ``use_extended_data=True`` (default False), the loader also reads
    rounds + klines from ``var/extended/`` (older-than-canonical-floor epochs
    written by ``research/backfill_okx_extended.py``). Records flagged with
    ``data_status`` other than ``OK_FULL`` may have empty or partial
    ``klines_1s``; the strategy's existing ``_validate_klines_raw`` check
    naturally skips such rounds via ``gate_<sym>_insufficient`` skip reasons.
    Default OFF preserves canonical bit-identical behavior.
    """
    backtest_cfg.validate()

    backtest_round_count = backtest_cfg.backtest_round_count
    initial_bankroll_bnb = backtest_cfg.initial_bankroll_bnb

    info("START", "Loading closed rounds and klines")
    t0 = time.perf_counter()

    all_rounds = _load_all_rounds(runtime_cfg)
    if use_extended_data and _EXT_CLOSED_ROUNDS_PATH.exists():
        from pancakebot.market_data.round_store import ClosedRoundsStore
        ext_store = ClosedRoundsStore(str(_EXT_CLOSED_ROUNDS_PATH))
        ext_rounds = [r for r in ext_store.iter_closed_rounds() if not r.failed]
        existing = {int(r.epoch) for r in all_rounds}
        ext_only = [r for r in ext_rounds if int(r.epoch) not in existing]
        ext_only.sort(key=lambda r: int(r.epoch))
        all_rounds = ext_only + all_rounds
        info("START", f"loaded {len(ext_only)} extended rounds (older than canonical floor)")
    if not all_rounds:
        raise InvariantError("backtest_no_closed_rounds")

    # Select simulation window: epoch range takes priority over most-recent-N.
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
        sim_rounds = all_rounds[-backtest_round_count:]
        if len(sim_rounds) < backtest_round_count:
            raise InvariantError(
                f"backtest_insufficient_rounds: need={backtest_round_count} have={len(sim_rounds)}"
            )

    first_epoch = sim_rounds[0].epoch
    last_epoch = sim_rounds[-1].epoch
    elapsed_load = time.perf_counter() - t0
    info(
        "START",
        f"rounds={len(all_rounds)} "
        f"sim_rounds={len(sim_rounds)} epochs=[{first_epoch}..{last_epoch}] "
        f"load_elapsed={elapsed_load:.1f}s",
    )

    # BNB klines are stored on `_bnb_klines_by_epoch` but never read
    # by the strategy (verified by grep across pancakebot/: no
    # `_bnb_klines_by_epoch.get(...)` call site exists). Skip the
    # ~3 GB load entirely. The pipeline's `refresh_bnb_klines` still
    # accepts the empty dict so the wiring is preserved for any
    # future BNB-aware research that re-enables this path.
    bnb_klines: dict[int, list[list]] = {}
    info("START", "BNB load skipped (strategy does not read BNB klines; saves ~3 GB RAM)")

    # Compute the per-record kline window from the gate config.
    # candle_count = max(mtf_lookbacks) + 1 covers the deepest lookback
    # plus the anchor candle.
    _gc = runtime_cfg.strategy.gate
    _candle_count = max(_gc.mtf_lookbacks) + 1
    _cs = int(runtime_cfg.kline_cutoff_seconds)
    _ext_btc = _EXT_BTC_KLINES_PATH if use_extended_data else None
    _ext_eth = _EXT_ETH_KLINES_PATH if use_extended_data else None
    _ext_sol = _EXT_SOL_KLINES_PATH if use_extended_data else None
    btc_klines = _load_klines_from(
        _BTC_KLINES_PATH, kline_cutoff_seconds=_cs, candle_count=_candle_count,
        extended_path=_ext_btc,
    )
    if btc_klines:
        info("START", f"Loaded BTC 1s klines for {len(btc_klines)} epochs "
             f"(cutoff={_cs}, candle_count={_candle_count})")
    eth_klines = (
        _load_klines_from(
            _ETH_KLINES_PATH, kline_cutoff_seconds=_cs, candle_count=_candle_count,
            extended_path=_ext_eth,
        )
        if _ETH_KLINES_PATH.exists() or (_ext_eth is not None and _ext_eth.exists()) else {}
    )
    if eth_klines:
        info("START", f"Loaded ETH 1s klines for {len(eth_klines)} epochs")
    sol_klines = (
        _load_klines_from(
            _SOL_KLINES_PATH, kline_cutoff_seconds=_cs, candle_count=_candle_count,
            extended_path=_ext_sol,
        )
        if _SOL_KLINES_PATH.exists() or (_ext_sol is not None and _ext_sol.exists()) else {}
    )
    if sol_klines:
        info("START", f"Loaded SOL 1s klines for {len(sol_klines)} epochs")

    # Build momentum pipeline (no live gate -- backtest uses cached 1s klines).
    from pancakebot.bankroll_tracker import InMemoryBankrollTracker
    from pancakebot.strategy.momentum_gate import MomentumGateConfig
    gate_config: MomentumGateConfig = runtime_cfg.momentum_gate_config  # type: ignore[assignment]
    bankroll_tracker = InMemoryBankrollTracker(
        initial_bankroll=initial_bankroll_bnb,
        drawdown_peak_window_days=runtime_cfg.strategy.risk.drawdown_peak_window_days,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_config,
        strategy_config=runtime_cfg.strategy,
        gate=None,
        kline_cutoff_seconds=runtime_cfg.kline_cutoff_seconds,
        pool_cutoff_seconds=runtime_cfg.pool_cutoff_seconds,
        min_bet_amount_bnb=runtime_cfg.min_bet_amount_bnb,
        treasury_fee_fraction=runtime_cfg.treasury_fee_fraction,
        bankroll_tracker=bankroll_tracker,
    )
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch=bnb_klines)
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    info("READY", "MomentumOnlyPipeline ready (backtest/4-asset mode)")

    # Output files.
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "trades.csv"
    summary_path = out_dir / "summary.json"

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
                "profit_bnb",
                "bankroll_bnb",
            ]
        )

        for i, round_t in enumerate(sim_rounds):
            decision = pipeline.decide_open_round(round_t=round_t)

            profit = 0.0
            if decision.action == "BET" and decision.bet_size_bnb > 0.0:
                bet_side = decision.bet_side
                if bet_side not in ("Bull", "Bear"):
                    raise InvariantError("backtest_bet_side_invalid")

                bankroll -= decision.bet_size_bnb + BACKTEST_GAS_COST_BET_BNB
                outcome = settle_bet_against_closed_round(
                    bet_bnb=decision.bet_size_bnb,
                    bet_side=bet_side,
                    round_closed=round_t,
                    treasury_fee_fraction=runtime_cfg.treasury_fee_fraction,
                )
                bankroll += outcome.credit_bnb
                profit = outcome.credit_bnb - decision.bet_size_bnb - BACKTEST_GAS_COST_BET_BNB

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
                    profit,
                    bankroll,
                ]
            )

            # Forward the post-settlement bankroll snapshot to the tracker so
            # the risk checks see up-to-date current_bankroll / peak.
            pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))

            pipeline.settle_closed_rounds(rounds=[round_t])

            if (i + 1) % 500 == 0:
                info("PROGRESS", f"idx={i+1}/{len(sim_rounds)} bankroll={bankroll:.4f} BNB")

    elapsed_sim = time.perf_counter() - t_sim_start

    # Summary statistics.
    total_rounds = len(sim_rounds)
    total_skips = total_rounds - stats.num_bets
    net_pnl = bankroll - initial_bankroll_bnb
    win_rate = _safe_rate(stats.num_wins, stats.num_bets)
    bet_rate = _safe_rate(stats.num_bets, total_rounds)

    info(
        "SUMMARY",
        f"rounds={total_rounds} bets={stats.num_bets} "
        f"win_rate={win_rate:.1%} bet_rate={bet_rate:.1%} "
        f"net_pnl={net_pnl:+.4f} BNB "
        f"final_bankroll={bankroll:.4f} BNB "
        f"elapsed={elapsed_sim:.1f}s",
    )

    skip_detail = dict(sorted(stats.skip_counts_by_reason.items(), key=lambda x: -x[1]))
    summary = {
        "backtest_round_count": total_rounds,
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
    info("DONE", f"trades={trades_path} summary={summary_path} equity={equity_path}")


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
