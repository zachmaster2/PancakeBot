"""Shared helpers for block-level meta-strategy inspection experiments.

This module stays inspection-only. It consumes existing historical
`dislocation_trades.csv` artifacts and converts them into block-level summaries
that can be used by offline strategy-selection experiments.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from inspection.strategy_router_common import (
    BlockRoundSnapshot,
    StrategyTradeRow,
    load_block_round_snapshots,
    load_strategy_trade_rows,
    to_column_key_map,
)


@dataclass(frozen=True, slots=True)
class StrategyBlockMetrics:
    """One strategy's realized block metrics."""

    num_available_rounds: int
    num_bets: int
    num_wins: int
    bet_rate: float
    win_rate_on_bets: float
    positive_round_rate: float
    net_profit_bnb: float
    profit_per_500_rounds: float
    max_drawdown_bnb: float
    mean_expected_net_selected_bnb: float | None
    mean_abs_dislocation_bull: float | None
    mean_p_nowcast_bull: float | None
    mean_p_market_bull: float | None
    mean_bet_size_bnb: float | None


@dataclass(frozen=True, slots=True)
class BlockRegimeMetrics:
    """Consensus regime summary computed from aligned round artifacts."""

    market_bull_mean: float | None
    market_bull_std: float | None
    nowcast_bull_mean: float | None
    nowcast_bull_std: float | None
    abs_dislocation_mean: float | None
    abs_dislocation_std: float | None
    disagreement_rate: float | None


@dataclass(frozen=True, slots=True)
class MetaStrategyBlock:
    """One aligned historical block for all candidate strategies."""

    block_index: int
    sim_offset_rounds: int
    epoch_start: int
    epoch_end: int
    num_rounds: int
    regime: BlockRegimeMetrics
    strategies: dict[str, StrategyBlockMetrics]
    oracle_strategy_or_skip: str
    oracle_profit_bnb: float


def safe_rate(num: int, den: int) -> float:
    """Return `num / den`, or zero when the denominator is non-positive."""

    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def safe_mean(values: list[float]) -> float | None:
    """Return the mean of finite values, or `None` when empty."""

    if not values:
        return None
    return float(statistics.fmean(float(v) for v in values))


def safe_std(values: list[float]) -> float | None:
    """Return population stddev, or `None` when empty."""

    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(float(v) for v in values))


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except Exception:
        return None


def strategy_column_key_map(strategy_prefixes: list[str]) -> dict[str, str]:
    """Expose the same stable strategy->column key map used elsewhere."""

    return to_column_key_map(strategy_prefixes)


def parse_extra_series_specs(raw_specs: list[str]) -> dict[str, Path]:
    """Parse `name=csv_path` specs for extra aligned trade series."""

    out: dict[str, Path] = {}
    for raw_spec in raw_specs:
        spec = str(raw_spec).strip()
        if spec == "":
            continue
        if "=" not in spec:
            raise ValueError(f"meta_strategy_extra_series_spec_invalid: {spec}")
        name, raw_path = spec.split("=", 1)
        series_name = str(name).strip()
        path_text = str(raw_path).strip()
        if series_name == "" or path_text == "":
            raise ValueError(f"meta_strategy_extra_series_spec_invalid: {spec}")
        if series_name in out:
            raise ValueError(f"meta_strategy_extra_series_duplicate_name: {series_name}")
        out[str(series_name)] = Path(path_text)
    return out


def load_extra_trade_rows_by_name(
    specs: dict[str, Path],
) -> dict[str, dict[int, StrategyTradeRow]]:
    """Load additional aligned trade series keyed by synthetic strategy name."""

    return {
        str(name): load_generic_trade_rows(Path(path))
        for name, path in specs.items()
    }


def load_generic_trade_rows(trades_csv_path: Path) -> dict[int, StrategyTradeRow]:
    """Load either `dislocation_trades.csv` or `backtest_trades.csv` rows."""

    if not trades_csv_path.exists():
        raise FileNotFoundError(f"meta_strategy_generic_trades_missing: {trades_csv_path}")
    with trades_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(str(x) for x in (reader.fieldnames or []))
        if "expected_net_selected" in fieldnames:
            return load_strategy_trade_rows(trades_csv_path)
        if "ev_bnb" in fieldnames:
            rows: dict[int, StrategyTradeRow] = {}
            for raw in reader:
                epoch = int(raw["epoch"])
                action = str(raw.get("action", "")).strip().upper()
                direction = str(raw.get("direction", "")).strip().upper()
                if action not in ("BET", "SKIP"):
                    raise ValueError(f"meta_strategy_backtest_action_invalid: {action}")
                if action == "BET" and direction not in ("BULL", "BEAR"):
                    raise ValueError("meta_strategy_backtest_direction_missing_for_bet")
                if action == "SKIP":
                    direction = ""
                profit_raw = raw.get("profit_bnb")
                if profit_raw is None or str(profit_raw).strip() == "":
                    raise ValueError("meta_strategy_backtest_profit_missing")
                rows[int(epoch)] = StrategyTradeRow(
                    epoch=int(epoch),
                    action=str(action),
                    direction=str(direction),
                    expected_net_selected_bnb=_safe_float(raw.get("ev_bnb")),
                    dislocation_bull=None,
                    p_nowcast_bull=None,
                    p_market_bull=None,
                    bet_size_bnb=_safe_float(raw.get("bet_size_bnb")),
                    profit_bnb=float(profit_raw),
                )
            if not rows:
                raise ValueError(f"meta_strategy_generic_trades_empty: {trades_csv_path}")
            return rows
    raise ValueError(f"meta_strategy_generic_trades_schema_unknown: {trades_csv_path}")


def load_meta_strategy_blocks(
    *,
    strategy_prefixes: list[str],
    block_size: int,
    num_blocks: int,
    skip_most_recent_blocks: int,
    base_dir: Path,
    trades_filename: str,
    extra_series_rows_by_name: dict[str, dict[int, StrategyTradeRow]] | None = None,
) -> list[MetaStrategyBlock]:
    """Load aligned block artifacts and summarize them block-by-block."""

    snapshots = load_meta_strategy_round_snapshots(
        strategy_prefixes=strategy_prefixes,
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        skip_most_recent_blocks=int(skip_most_recent_blocks),
        base_dir=Path(base_dir),
        trades_filename=str(trades_filename),
        extra_series_rows_by_name=extra_series_rows_by_name,
    )
    if not snapshots:
        raise ValueError("meta_strategy_blocks_empty")

    all_series_names = [str(x) for x in strategy_prefixes] + [
        str(x) for x in dict(extra_series_rows_by_name or {})
    ]
    grouped = _group_snapshots_by_block(snapshots)
    return [
        summarize_meta_strategy_window(
            snapshots=block_rows,
            series_names=all_series_names,
            block_index=int(block_index),
            sim_offset_rounds=int(sim_offset_rounds),
        )
        for block_index, sim_offset_rounds, block_rows in grouped
    ]


def load_meta_strategy_round_snapshots(
    *,
    strategy_prefixes: list[str],
    block_size: int,
    num_blocks: int,
    skip_most_recent_blocks: int,
    base_dir: Path,
    trades_filename: str,
    extra_series_rows_by_name: dict[str, dict[int, StrategyTradeRow]] | None = None,
) -> list[BlockRoundSnapshot]:
    """Load the aligned round stream for all primary and extra series."""

    snapshots = load_block_round_snapshots(
        strategy_prefixes=strategy_prefixes,
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        skip_most_recent_blocks=int(skip_most_recent_blocks),
        base_dir=Path(base_dir),
        trades_filename=str(trades_filename),
    )
    if not snapshots:
        raise ValueError("meta_strategy_round_snapshots_empty")

    extra_rows = dict(extra_series_rows_by_name or {})
    if not extra_rows:
        return snapshots

    augmented: list[BlockRoundSnapshot] = []
    for snapshot in snapshots:
        rows_by_strategy = dict(snapshot.rows_by_strategy)
        for series_name, rows_by_epoch in extra_rows.items():
            rows_by_strategy[str(series_name)] = rows_by_epoch.get(int(snapshot.epoch))
        augmented.append(
            BlockRoundSnapshot(
                block_index=int(snapshot.block_index),
                sim_offset_rounds=int(snapshot.sim_offset_rounds),
                epoch=int(snapshot.epoch),
                rows_by_strategy=rows_by_strategy,
            )
        )
    return augmented


def summarize_meta_strategy_window(
    *,
    snapshots: list[BlockRoundSnapshot],
    series_names: list[str],
    block_index: int,
    sim_offset_rounds: int,
) -> MetaStrategyBlock:
    """Summarize one arbitrary contiguous window of aligned round snapshots."""

    if not snapshots:
        raise ValueError("meta_strategy_window_empty")

    strategies: dict[str, StrategyBlockMetrics] = {}
    for strategy in series_names:
        aligned_rows = [
            snapshot.rows_by_strategy.get(str(strategy))
            for snapshot in snapshots
        ]
        strategies[str(strategy)] = _summarize_trade_row_sequence(
            rows=aligned_rows,
            num_rounds=int(len(snapshots)),
        )

    oracle_strategy_or_skip = "SKIP"
    oracle_profit_bnb = 0.0
    for strategy in strategies:
        profit_bnb = float(strategies[str(strategy)].net_profit_bnb)
        if float(profit_bnb) > float(oracle_profit_bnb):
            oracle_profit_bnb = float(profit_bnb)
            oracle_strategy_or_skip = str(strategy)

    return MetaStrategyBlock(
        block_index=int(block_index),
        sim_offset_rounds=int(sim_offset_rounds),
        epoch_start=int(snapshots[0].epoch),
        epoch_end=int(snapshots[-1].epoch),
        num_rounds=int(len(snapshots)),
        regime=_summarize_block_regime(snapshots),
        strategies=strategies,
        oracle_strategy_or_skip=str(oracle_strategy_or_skip),
        oracle_profit_bnb=float(oracle_profit_bnb),
    )


def _group_snapshots_by_block(
    snapshots: list[BlockRoundSnapshot],
) -> list[tuple[int, int, list[BlockRoundSnapshot]]]:
    grouped: dict[int, list[BlockRoundSnapshot]] = {}
    offsets: dict[int, int] = {}
    for snapshot in snapshots:
        key = int(snapshot.block_index)
        grouped.setdefault(key, []).append(snapshot)
        offsets.setdefault(key, int(snapshot.sim_offset_rounds))
        if int(offsets[key]) != int(snapshot.sim_offset_rounds):
            raise ValueError("meta_strategy_block_offset_mismatch")

    out: list[tuple[int, int, list[BlockRoundSnapshot]]] = []
    for block_index in sorted(grouped):
        rows = sorted(grouped[block_index], key=lambda row: int(row.epoch))
        out.append((int(block_index), int(offsets[block_index]), rows))
    return out


def _summarize_trade_row_sequence(
    *,
    rows: list[StrategyTradeRow | None],
    num_rounds: int,
) -> StrategyBlockMetrics:
    num_available_rounds = 0
    num_bets = 0
    num_wins = 0
    positive_rounds = 0
    net_profit_bnb = 0.0
    cumulative_profit_bnb = 0.0
    peak_profit_bnb = 0.0
    max_drawdown_bnb = 0.0
    expected_values: list[float] = []
    abs_dislocation_values: list[float] = []
    nowcast_values: list[float] = []
    market_values: list[float] = []
    bet_sizes: list[float] = []

    for row in rows:
        if row is None:
            continue
        num_available_rounds += 1
        profit_bnb = float(row.profit_bnb)
        net_profit_bnb += float(profit_bnb)
        cumulative_profit_bnb += float(profit_bnb)
        if float(cumulative_profit_bnb) > float(peak_profit_bnb):
            peak_profit_bnb = float(cumulative_profit_bnb)
        drawdown_bnb = float(peak_profit_bnb) - float(cumulative_profit_bnb)
        if float(drawdown_bnb) > float(max_drawdown_bnb):
            max_drawdown_bnb = float(drawdown_bnb)
        if float(profit_bnb) > 0.0:
            positive_rounds += 1
        if str(row.action) == "BET":
            num_bets += 1
            if float(profit_bnb) > 0.0:
                num_wins += 1
        if row.expected_net_selected_bnb is not None:
            expected_values.append(float(row.expected_net_selected_bnb))
        if row.dislocation_bull is not None:
            abs_dislocation_values.append(abs(float(row.dislocation_bull)))
        if row.p_nowcast_bull is not None:
            nowcast_values.append(float(row.p_nowcast_bull))
        if row.p_market_bull is not None:
            market_values.append(float(row.p_market_bull))
        if row.bet_size_bnb is not None:
            bet_sizes.append(float(row.bet_size_bnb))

    profit_per_500_rounds = 0.0
    if int(num_rounds) > 0:
        profit_per_500_rounds = float(net_profit_bnb) / float(num_rounds) * 500.0

    return StrategyBlockMetrics(
        num_available_rounds=int(num_available_rounds),
        num_bets=int(num_bets),
        num_wins=int(num_wins),
        bet_rate=float(safe_rate(num_bets, num_rounds)),
        win_rate_on_bets=float(safe_rate(num_wins, num_bets)),
        positive_round_rate=float(safe_rate(positive_rounds, num_rounds)),
        net_profit_bnb=float(net_profit_bnb),
        profit_per_500_rounds=float(profit_per_500_rounds),
        max_drawdown_bnb=float(max_drawdown_bnb),
        mean_expected_net_selected_bnb=safe_mean(expected_values),
        mean_abs_dislocation_bull=safe_mean(abs_dislocation_values),
        mean_p_nowcast_bull=safe_mean(nowcast_values),
        mean_p_market_bull=safe_mean(market_values),
        mean_bet_size_bnb=safe_mean(bet_sizes),
    )


def _summarize_block_regime(block_rows: list[BlockRoundSnapshot]) -> BlockRegimeMetrics:
    median_market_values: list[float] = []
    median_nowcast_values: list[float] = []
    median_abs_dislocation_values: list[float] = []
    disagreement_count = 0
    disagreement_den = 0

    for snapshot in block_rows:
        market_value = _snapshot_consensus_value(
            snapshot=snapshot,
            accessor=lambda row: row.p_market_bull,
        )
        nowcast_value = _snapshot_consensus_value(
            snapshot=snapshot,
            accessor=lambda row: row.p_nowcast_bull,
        )
        abs_dislocation_value = _snapshot_consensus_value(
            snapshot=snapshot,
            accessor=lambda row: abs(float(row.dislocation_bull))
            if row.dislocation_bull is not None
            else None,
        )
        if market_value is not None:
            median_market_values.append(float(market_value))
        if nowcast_value is not None:
            median_nowcast_values.append(float(nowcast_value))
        if abs_dislocation_value is not None:
            median_abs_dislocation_values.append(float(abs_dislocation_value))
        if market_value is not None and nowcast_value is not None:
            disagreement_den += 1
            if float(market_value - 0.5) * float(nowcast_value - 0.5) < 0.0:
                disagreement_count += 1

    return BlockRegimeMetrics(
        market_bull_mean=safe_mean(median_market_values),
        market_bull_std=safe_std(median_market_values),
        nowcast_bull_mean=safe_mean(median_nowcast_values),
        nowcast_bull_std=safe_std(median_nowcast_values),
        abs_dislocation_mean=safe_mean(median_abs_dislocation_values),
        abs_dislocation_std=safe_std(median_abs_dislocation_values),
        disagreement_rate=(
            float(safe_rate(disagreement_count, disagreement_den))
            if int(disagreement_den) > 0
            else None
        ),
    )


def _snapshot_consensus_value(
    *,
    snapshot: BlockRoundSnapshot,
    accessor,
) -> float | None:
    values: list[float] = []
    row: StrategyTradeRow | None
    for row in snapshot.rows_by_strategy.values():
        if row is None:
            continue
        value = accessor(row)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return float(statistics.median(values))
