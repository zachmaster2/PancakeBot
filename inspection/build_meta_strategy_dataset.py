"""Build a block-level dataset for offline meta-strategy experiments.

This inspection tool consumes existing historical block artifacts and emits one
row per target block. Every feature is derived only from completed prior
blocks, and every label refers to realized performance on the next block.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from inspection.meta_strategy_common import (
    MetaStrategyBlock,
    load_extra_trade_rows_by_name,
    load_meta_strategy_blocks,
    load_meta_strategy_round_snapshots,
    parse_extra_series_specs,
    safe_mean,
    safe_std,
    summarize_meta_strategy_window,
    strategy_column_key_map,
)
from inspection.strategy_router_common import parse_strategy_prefixes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--strategy-prefixes", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--num-blocks", type=int, default=80)
    parser.add_argument("--skip-most-recent-blocks", type=int, default=0)
    parser.add_argument("--lookback-blocks", type=int, default=5)
    parser.add_argument("--decision-window-rounds", type=int, default=0)
    parser.add_argument("--history-block-rounds", type=int, default=0)
    parser.add_argument("--framing-mode", choices=("auto", "source_block_aligned", "rolling_rounds"), default="auto")
    parser.add_argument("--base-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--output-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--trades-filename", type=str, default="dislocation_trades.csv")
    parser.add_argument("--extra-series", action="append", default=[])
    parser.add_argument("--baseline-strategy-name", type=str, default="")
    return parser


def _build_columns(
    strategy_prefixes: list[str],
    key_map: dict[str, str],
    baseline_strategy_name: str,
) -> tuple[list[str], list[str], list[str]]:
    base_columns = [
        "target_block_index",
        "target_sim_offset_rounds",
        "target_epoch_start",
        "target_epoch_end",
        "target_num_rounds",
        "history_block_count",
    ]

    feature_columns: list[str] = []

    regime_metric_names = (
        "market_bull_mean",
        "market_bull_std",
        "nowcast_bull_mean",
        "nowcast_bull_std",
        "abs_dislocation_mean",
        "abs_dislocation_std",
        "disagreement_rate",
    )
    for metric_name in regime_metric_names:
        feature_columns.extend(
            [
                f"feat_regime_{metric_name}_last",
                f"feat_regime_{metric_name}_mean",
                f"feat_regime_{metric_name}_std",
                f"feat_regime_{metric_name}_recent3_mean",
                f"feat_regime_{metric_name}_recent3_minus_older_mean",
            ]
        )
    feature_columns.extend(
        [
            "feat_regime_nowcast_minus_market_last",
            "feat_regime_nowcast_minus_market_mean",
            "feat_regime_nowcast_minus_market_recent3_minus_older_mean",
        ]
    )

    for strategy_prefix in strategy_prefixes:
        key = key_map[str(strategy_prefix)]
        feature_columns.extend(
            [
                f"feat_{key}_profit_last_bnb",
                f"feat_{key}_profit_mean_bnb",
                f"feat_{key}_profit_std_bnb",
                f"feat_{key}_profit_recent3_mean_bnb",
                f"feat_{key}_profit_recent3_minus_older_mean_bnb",
                f"feat_{key}_profit_positive_streak_len",
                f"feat_{key}_profit_positive_streak_total_bnb",
                f"feat_{key}_profit_positive_transition_prob",
                f"feat_{key}_profit_positive_next_mean_bnb",
                f"feat_{key}_bet_rate_last",
                f"feat_{key}_bet_rate_mean",
                f"feat_{key}_win_rate_last",
                f"feat_{key}_win_rate_mean",
                f"feat_{key}_positive_round_rate_last",
                f"feat_{key}_positive_round_rate_mean",
                f"feat_{key}_max_drawdown_last_bnb",
                f"feat_{key}_max_drawdown_mean_bnb",
                f"feat_{key}_mean_expected_net_last_bnb",
                f"feat_{key}_mean_expected_net_mean_bnb",
                f"feat_{key}_mean_expected_net_recent3_mean_bnb",
                f"feat_{key}_mean_expected_net_recent3_minus_older_mean_bnb",
            ]
        )

    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name != "":
        baseline_key = key_map[str(baseline_name)]
        for strategy_prefix in strategy_prefixes:
            if str(strategy_prefix) == str(baseline_name):
                continue
            key = key_map[str(strategy_prefix)]
            feature_columns.extend(
                [
                    f"feat_{key}_delta_profit_last_vs_{baseline_key}_bnb",
                    f"feat_{key}_delta_profit_mean_vs_{baseline_key}_bnb",
                    f"feat_{key}_delta_profit_recent3_mean_vs_{baseline_key}_bnb",
                    f"feat_{key}_delta_profit_recent3_minus_older_vs_{baseline_key}_bnb",
                    f"feat_{key}_delta_profit_positive_streak_len_vs_{baseline_key}",
                    f"feat_{key}_delta_profit_positive_streak_total_vs_{baseline_key}_bnb",
                    f"feat_{key}_delta_profit_positive_transition_prob_vs_{baseline_key}",
                    f"feat_{key}_delta_profit_positive_next_mean_vs_{baseline_key}_bnb",
                    f"feat_{key}_delta_win_rate_mean_vs_{baseline_key}",
                    f"feat_{key}_delta_positive_round_rate_mean_vs_{baseline_key}",
                    f"feat_{key}_delta_expected_net_mean_vs_{baseline_key}_bnb",
                ]
            )

    label_columns: list[str] = []
    for strategy_prefix in strategy_prefixes:
        key = key_map[str(strategy_prefix)]
        label_columns.extend(
            [
                f"label_{key}_next_block_profit_bnb",
                f"label_{key}_next_block_profit_per_500_rounds",
                f"label_{key}_next_block_num_bets",
                f"label_{key}_next_block_bet_rate",
            ]
        )

    label_columns.extend(
        [
            "label_oracle_strategy_or_skip",
            "label_oracle_profit_bnb",
            "label_oracle_profit_per_500_rounds",
        ]
    )
    return base_columns + feature_columns + label_columns, feature_columns, label_columns


def _cell_or_blank(value: float | None) -> float | str:
    if value is None:
        return ""
    return float(value)


def _history_regime_mean(
    history: list[MetaStrategyBlock],
    accessor,
) -> float | None:
    values: list[float] = []
    for block in history:
        value = accessor(block)
        if value is None:
            continue
        values.append(float(value))
    return safe_mean(values)


def _history_values(history: list[Any], accessor) -> list[float]:
    values: list[float] = []
    for item in history:
        value = accessor(item)
        if value is None:
            continue
        values.append(float(value))
    return values


def _history_recent_mean(values: list[float], recent_count: int = 3) -> float | None:
    if not values:
        return None
    return safe_mean(values[-min(int(recent_count), len(values)) :])


def _history_recent_minus_older_mean(values: list[float], recent_count: int = 3) -> float | None:
    if not values:
        return None
    recent_n = min(int(recent_count), len(values))
    recent_values = values[-recent_n:]
    older_values = values[:-recent_n]
    if not recent_values or not older_values:
        return None
    recent_mean = safe_mean(recent_values)
    older_mean = safe_mean(older_values)
    if recent_mean is None or older_mean is None:
        return None
    return float(recent_mean) - float(older_mean)


def _tail_streak_len(values: list[float], predicate) -> int:
    count = 0
    for value in reversed(values):
        if not bool(predicate(float(value))):
            break
        count += 1
    return int(count)


def _tail_streak_total(values: list[float], predicate) -> float:
    total = 0.0
    for value in reversed(values):
        if not bool(predicate(float(value))):
            break
        total += float(value)
    return float(total)


def _transition_prob(values: list[float], predicate) -> float | None:
    if len(values) < 2:
        return None
    den = 0
    num = 0
    for idx in range(len(values) - 1):
        if bool(predicate(float(values[idx]))):
            den += 1
            if bool(predicate(float(values[idx + 1]))):
                num += 1
    if int(den) <= 0:
        return None
    return float(num) / float(den)


def _conditional_next_mean(values: list[float], predicate) -> float | None:
    if len(values) < 2:
        return None
    next_values: list[float] = []
    for idx in range(len(values) - 1):
        if bool(predicate(float(values[idx]))):
            next_values.append(float(values[idx + 1]))
    return safe_mean(next_values)


def _history_strategy_mean(
    history: list[MetaStrategyBlock],
    *,
    strategy_prefix: str,
    accessor,
) -> float | None:
    values: list[float] = []
    for block in history:
        value = accessor(block.strategies[str(strategy_prefix)])
        if value is None:
            continue
        values.append(float(value))
    return safe_mean(values)


def main() -> None:
    args = _build_parser().parse_args()

    strategy_prefixes = parse_strategy_prefixes(str(args.strategy_prefixes))
    if len(strategy_prefixes) < 2:
        raise ValueError("meta_strategy_dataset_requires_at_least_two_strategies")
    if int(args.lookback_blocks) <= 0:
        raise ValueError("meta_strategy_dataset_lookback_blocks_nonpositive")
    extra_specs = parse_extra_series_specs(list(args.extra_series))
    for extra_name in extra_specs:
        if str(extra_name) in strategy_prefixes:
            raise ValueError(f"meta_strategy_dataset_extra_series_name_conflicts: {extra_name}")
    extra_series_rows_by_name = load_extra_trade_rows_by_name(extra_specs)
    all_series_names = [str(x) for x in strategy_prefixes] + [str(x) for x in extra_specs]
    baseline_name = str(args.baseline_strategy_name).strip()
    if baseline_name != "" and baseline_name not in all_series_names:
        raise ValueError(f"meta_strategy_dataset_baseline_strategy_unknown: {baseline_name}")

    decision_window_rounds = (
        int(args.decision_window_rounds)
        if int(args.decision_window_rounds) > 0
        else int(args.block_size)
    )
    history_block_rounds = (
        int(args.history_block_rounds)
        if int(args.history_block_rounds) > 0
        else int(decision_window_rounds)
    )
    if int(decision_window_rounds) <= 0:
        raise ValueError("meta_strategy_dataset_decision_window_rounds_nonpositive")
    if int(history_block_rounds) <= 0:
        raise ValueError("meta_strategy_dataset_history_block_rounds_nonpositive")

    use_source_block_aligned = False
    framing_mode = str(args.framing_mode)
    if framing_mode == "source_block_aligned":
        use_source_block_aligned = True
    elif framing_mode == "rolling_rounds":
        use_source_block_aligned = False
    else:
        use_source_block_aligned = bool(
            int(decision_window_rounds) == int(args.block_size)
            and int(history_block_rounds) == int(args.block_size)
        )

    blocks: list[MetaStrategyBlock] = []
    round_snapshots = []
    if bool(use_source_block_aligned):
        blocks = load_meta_strategy_blocks(
            strategy_prefixes=strategy_prefixes,
            block_size=int(args.block_size),
            num_blocks=int(args.num_blocks),
            skip_most_recent_blocks=int(args.skip_most_recent_blocks),
            base_dir=Path(str(args.base_dir)),
            trades_filename=str(args.trades_filename),
            extra_series_rows_by_name=extra_series_rows_by_name,
        )
        if int(args.lookback_blocks) >= int(len(blocks)):
            raise ValueError("meta_strategy_dataset_lookback_blocks_too_large")
    else:
        round_snapshots = load_meta_strategy_round_snapshots(
            strategy_prefixes=strategy_prefixes,
            block_size=int(args.block_size),
            num_blocks=int(args.num_blocks),
            skip_most_recent_blocks=int(args.skip_most_recent_blocks),
            base_dir=Path(str(args.base_dir)),
            trades_filename=str(args.trades_filename),
            extra_series_rows_by_name=extra_series_rows_by_name,
        )
        history_round_count = int(args.lookback_blocks) * int(history_block_rounds)
        if int(history_round_count) <= 0:
            raise ValueError("meta_strategy_dataset_history_round_count_nonpositive")
        if int(history_round_count) + int(decision_window_rounds) > int(len(round_snapshots)):
            raise ValueError("meta_strategy_dataset_history_plus_target_exceeds_rounds")

    key_map = strategy_column_key_map(all_series_names)
    all_columns, feature_columns, label_columns = _build_columns(
        all_series_names,
        key_map,
        baseline_name,
    )

    rows_out: list[dict[str, Any]] = []
    strategy_label_net_bnb: dict[str, float] = {str(s): 0.0 for s in all_series_names}
    oracle_label_net_bnb = 0.0

    target_windows: list[tuple[list[MetaStrategyBlock], MetaStrategyBlock]] = []
    if bool(use_source_block_aligned):
        for idx in range(int(args.lookback_blocks), len(blocks)):
            target_windows.append(
                (
                    blocks[idx - int(args.lookback_blocks) : idx],
                    blocks[idx],
                )
            )
    else:
        history_round_count = int(args.lookback_blocks) * int(history_block_rounds)
        target_block_index = 0
        for target_start in range(
            int(history_round_count),
            int(len(round_snapshots)) - int(decision_window_rounds) + 1,
            int(decision_window_rounds),
        ):
            target_block_index += 1
            history: list[MetaStrategyBlock] = []
            for history_idx in range(int(args.lookback_blocks)):
                history_start = int(target_start) - int(history_round_count) + (
                    int(history_idx) * int(history_block_rounds)
                )
                history_end = int(history_start) + int(history_block_rounds)
                history.append(
                    summarize_meta_strategy_window(
                        snapshots=round_snapshots[int(history_start) : int(history_end)],
                        series_names=all_series_names,
                        block_index=int(history_idx) + 1,
                        sim_offset_rounds=int(len(round_snapshots)) - int(history_end),
                    )
                )
            target_end = int(target_start) + int(decision_window_rounds)
            target_windows.append(
                (
                    history,
                    summarize_meta_strategy_window(
                        snapshots=round_snapshots[int(target_start) : int(target_end)],
                        series_names=all_series_names,
                        block_index=int(target_block_index),
                        sim_offset_rounds=int(len(round_snapshots)) - int(target_end),
                    ),
                )
            )

    for history, target in target_windows:
        last_history = history[-1]

        regime_market_mean_values = _history_values(history, lambda block: block.regime.market_bull_mean)
        regime_market_std_values = _history_values(history, lambda block: block.regime.market_bull_std)
        regime_nowcast_mean_values = _history_values(history, lambda block: block.regime.nowcast_bull_mean)
        regime_nowcast_std_values = _history_values(history, lambda block: block.regime.nowcast_bull_std)
        regime_dislocation_mean_values = _history_values(history, lambda block: block.regime.abs_dislocation_mean)
        regime_dislocation_std_values = _history_values(history, lambda block: block.regime.abs_dislocation_std)
        regime_disagreement_values = _history_values(history, lambda block: block.regime.disagreement_rate)

        market_last = last_history.regime.market_bull_mean
        market_mean = safe_mean(regime_market_mean_values)
        nowcast_last = last_history.regime.nowcast_bull_mean
        nowcast_mean = safe_mean(regime_nowcast_mean_values)
        recent_market_mean = _history_recent_mean(regime_market_mean_values)
        recent_nowcast_mean = _history_recent_mean(regime_nowcast_mean_values)
        older_market_mean = (
            float(recent_market_mean) - float(_history_recent_minus_older_mean(regime_market_mean_values))
            if recent_market_mean is not None
            and _history_recent_minus_older_mean(regime_market_mean_values) is not None
            else None
        )
        older_nowcast_mean = (
            float(recent_nowcast_mean) - float(_history_recent_minus_older_mean(regime_nowcast_mean_values))
            if recent_nowcast_mean is not None
            and _history_recent_minus_older_mean(regime_nowcast_mean_values) is not None
            else None
        )

        row: dict[str, Any] = {
            "target_block_index": int(target.block_index),
            "target_sim_offset_rounds": int(target.sim_offset_rounds),
            "target_epoch_start": int(target.epoch_start),
            "target_epoch_end": int(target.epoch_end),
            "target_num_rounds": int(target.num_rounds),
            "history_block_count": int(len(history)),
            "feat_regime_market_bull_mean_last": _cell_or_blank(market_last),
            "feat_regime_market_bull_mean_mean": _cell_or_blank(market_mean),
            "feat_regime_market_bull_mean_std": _cell_or_blank(safe_std(regime_market_mean_values)),
            "feat_regime_market_bull_mean_recent3_mean": _cell_or_blank(
                _history_recent_mean(regime_market_mean_values)
            ),
            "feat_regime_market_bull_mean_recent3_minus_older_mean": _cell_or_blank(
                _history_recent_minus_older_mean(regime_market_mean_values)
            ),
            "feat_regime_market_bull_std_last": _cell_or_blank(last_history.regime.market_bull_std),
            "feat_regime_market_bull_std_mean": _cell_or_blank(safe_mean(regime_market_std_values)),
            "feat_regime_market_bull_std_std": _cell_or_blank(safe_std(regime_market_std_values)),
            "feat_regime_market_bull_std_recent3_mean": _cell_or_blank(
                _history_recent_mean(regime_market_std_values)
            ),
            "feat_regime_market_bull_std_recent3_minus_older_mean": _cell_or_blank(
                _history_recent_minus_older_mean(regime_market_std_values)
            ),
            "feat_regime_nowcast_bull_mean_last": _cell_or_blank(nowcast_last),
            "feat_regime_nowcast_bull_mean_mean": _cell_or_blank(nowcast_mean),
            "feat_regime_nowcast_bull_mean_std": _cell_or_blank(safe_std(regime_nowcast_mean_values)),
            "feat_regime_nowcast_bull_mean_recent3_mean": _cell_or_blank(
                _history_recent_mean(regime_nowcast_mean_values)
            ),
            "feat_regime_nowcast_bull_mean_recent3_minus_older_mean": _cell_or_blank(
                _history_recent_minus_older_mean(regime_nowcast_mean_values)
            ),
            "feat_regime_nowcast_bull_std_last": _cell_or_blank(last_history.regime.nowcast_bull_std),
            "feat_regime_nowcast_bull_std_mean": _cell_or_blank(safe_mean(regime_nowcast_std_values)),
            "feat_regime_nowcast_bull_std_std": _cell_or_blank(safe_std(regime_nowcast_std_values)),
            "feat_regime_nowcast_bull_std_recent3_mean": _cell_or_blank(
                _history_recent_mean(regime_nowcast_std_values)
            ),
            "feat_regime_nowcast_bull_std_recent3_minus_older_mean": _cell_or_blank(
                _history_recent_minus_older_mean(regime_nowcast_std_values)
            ),
            "feat_regime_abs_dislocation_mean_last": _cell_or_blank(last_history.regime.abs_dislocation_mean),
            "feat_regime_abs_dislocation_mean_mean": _cell_or_blank(safe_mean(regime_dislocation_mean_values)),
            "feat_regime_abs_dislocation_mean_std": _cell_or_blank(safe_std(regime_dislocation_mean_values)),
            "feat_regime_abs_dislocation_mean_recent3_mean": _cell_or_blank(
                _history_recent_mean(regime_dislocation_mean_values)
            ),
            "feat_regime_abs_dislocation_mean_recent3_minus_older_mean": _cell_or_blank(
                _history_recent_minus_older_mean(regime_dislocation_mean_values)
            ),
            "feat_regime_abs_dislocation_std_last": _cell_or_blank(last_history.regime.abs_dislocation_std),
            "feat_regime_abs_dislocation_std_mean": _cell_or_blank(safe_mean(regime_dislocation_std_values)),
            "feat_regime_abs_dislocation_std_std": _cell_or_blank(safe_std(regime_dislocation_std_values)),
            "feat_regime_abs_dislocation_std_recent3_mean": _cell_or_blank(
                _history_recent_mean(regime_dislocation_std_values)
            ),
            "feat_regime_abs_dislocation_std_recent3_minus_older_mean": _cell_or_blank(
                _history_recent_minus_older_mean(regime_dislocation_std_values)
            ),
            "feat_regime_disagreement_rate_last": _cell_or_blank(last_history.regime.disagreement_rate),
            "feat_regime_disagreement_rate_mean": _cell_or_blank(safe_mean(regime_disagreement_values)),
            "feat_regime_disagreement_rate_std": _cell_or_blank(safe_std(regime_disagreement_values)),
            "feat_regime_disagreement_rate_recent3_mean": _cell_or_blank(
                _history_recent_mean(regime_disagreement_values)
            ),
            "feat_regime_disagreement_rate_recent3_minus_older_mean": _cell_or_blank(
                _history_recent_minus_older_mean(regime_disagreement_values)
            ),
            "feat_regime_nowcast_minus_market_last": _cell_or_blank(
                (float(nowcast_last) - float(market_last))
                if nowcast_last is not None and market_last is not None
                else None
            ),
            "feat_regime_nowcast_minus_market_mean": _cell_or_blank(
                (float(nowcast_mean) - float(market_mean))
                if nowcast_mean is not None and market_mean is not None
                else None
            ),
            "feat_regime_nowcast_minus_market_recent3_minus_older_mean": _cell_or_blank(
                (
                    (float(recent_nowcast_mean) - float(recent_market_mean))
                    - (float(older_nowcast_mean) - float(older_market_mean))
                )
                if recent_nowcast_mean is not None
                and recent_market_mean is not None
                and older_nowcast_mean is not None
                and older_market_mean is not None
                else None
            ),
        }

        baseline_feature_values: dict[str, float | None] = {}
        for strategy_prefix in all_series_names:
            key = key_map[str(strategy_prefix)]
            last_metrics = last_history.strategies[str(strategy_prefix)]
            profit_values = _history_values(
                history,
                lambda block: block.strategies[str(strategy_prefix)].net_profit_bnb,
            )
            bet_rate_values = _history_values(
                history,
                lambda block: block.strategies[str(strategy_prefix)].bet_rate,
            )
            win_rate_values = _history_values(
                history,
                lambda block: block.strategies[str(strategy_prefix)].win_rate_on_bets,
            )
            positive_rate_values = _history_values(
                history,
                lambda block: block.strategies[str(strategy_prefix)].positive_round_rate,
            )
            drawdown_values = _history_values(
                history,
                lambda block: block.strategies[str(strategy_prefix)].max_drawdown_bnb,
            )
            expected_net_values = _history_values(
                history,
                lambda block: block.strategies[str(strategy_prefix)].mean_expected_net_selected_bnb,
            )
            profit_positive_streak_len = _tail_streak_len(
                profit_values,
                lambda value: float(value) > 0.0,
            )
            profit_positive_streak_total_bnb = _tail_streak_total(
                profit_values,
                lambda value: float(value) > 0.0,
            )

            row[f"feat_{key}_profit_last_bnb"] = float(last_metrics.net_profit_bnb)
            row[f"feat_{key}_profit_mean_bnb"] = float(safe_mean(profit_values) or 0.0)
            row[f"feat_{key}_profit_std_bnb"] = _cell_or_blank(safe_std(profit_values))
            row[f"feat_{key}_profit_recent3_mean_bnb"] = _cell_or_blank(_history_recent_mean(profit_values))
            row[f"feat_{key}_profit_recent3_minus_older_mean_bnb"] = _cell_or_blank(
                _history_recent_minus_older_mean(profit_values)
            )
            row[f"feat_{key}_profit_positive_streak_len"] = int(profit_positive_streak_len)
            row[f"feat_{key}_profit_positive_streak_total_bnb"] = float(profit_positive_streak_total_bnb)
            row[f"feat_{key}_profit_positive_transition_prob"] = _cell_or_blank(
                _transition_prob(profit_values, lambda value: float(value) > 0.0)
            )
            row[f"feat_{key}_profit_positive_next_mean_bnb"] = _cell_or_blank(
                _conditional_next_mean(profit_values, lambda value: float(value) > 0.0)
            )
            row[f"feat_{key}_bet_rate_last"] = float(last_metrics.bet_rate)
            row[f"feat_{key}_bet_rate_mean"] = float(safe_mean(bet_rate_values) or 0.0)
            row[f"feat_{key}_win_rate_last"] = float(last_metrics.win_rate_on_bets)
            row[f"feat_{key}_win_rate_mean"] = float(safe_mean(win_rate_values) or 0.0)
            row[f"feat_{key}_positive_round_rate_last"] = float(last_metrics.positive_round_rate)
            row[f"feat_{key}_positive_round_rate_mean"] = float(safe_mean(positive_rate_values) or 0.0)
            row[f"feat_{key}_max_drawdown_last_bnb"] = float(last_metrics.max_drawdown_bnb)
            row[f"feat_{key}_max_drawdown_mean_bnb"] = float(safe_mean(drawdown_values) or 0.0)
            row[f"feat_{key}_mean_expected_net_last_bnb"] = _cell_or_blank(
                last_metrics.mean_expected_net_selected_bnb
            )
            row[f"feat_{key}_mean_expected_net_mean_bnb"] = _cell_or_blank(safe_mean(expected_net_values))
            row[f"feat_{key}_mean_expected_net_recent3_mean_bnb"] = _cell_or_blank(
                _history_recent_mean(expected_net_values)
            )
            row[f"feat_{key}_mean_expected_net_recent3_minus_older_mean_bnb"] = _cell_or_blank(
                _history_recent_minus_older_mean(expected_net_values)
            )

            if str(strategy_prefix) == str(baseline_name):
                baseline_feature_values = {
                    "profit_mean_bnb": safe_mean(profit_values),
                    "profit_recent3_mean_bnb": _history_recent_mean(profit_values),
                    "profit_recent3_minus_older_mean_bnb": _history_recent_minus_older_mean(profit_values),
                    "win_rate_mean": safe_mean(win_rate_values),
                    "positive_round_rate_mean": safe_mean(positive_rate_values),
                    "expected_net_mean_bnb": safe_mean(expected_net_values),
                }

            target_metrics = target.strategies[str(strategy_prefix)]
            row[f"label_{key}_next_block_profit_bnb"] = float(target_metrics.net_profit_bnb)
            row[f"label_{key}_next_block_profit_per_500_rounds"] = float(
                target_metrics.profit_per_500_rounds
            )
            row[f"label_{key}_next_block_num_bets"] = int(target_metrics.num_bets)
            row[f"label_{key}_next_block_bet_rate"] = float(target_metrics.bet_rate)
            strategy_label_net_bnb[str(strategy_prefix)] += float(target_metrics.net_profit_bnb)

        if baseline_name != "":
            baseline_key = key_map[str(baseline_name)]
            for strategy_prefix in all_series_names:
                if str(strategy_prefix) == str(baseline_name):
                    continue
                key = key_map[str(strategy_prefix)]
                strategy_profit_values = _history_values(
                    history,
                    lambda block: block.strategies[str(strategy_prefix)].net_profit_bnb,
                )
                baseline_profit_values = _history_values(
                    history,
                    lambda block: block.strategies[str(baseline_name)].net_profit_bnb,
                )
                delta_profit_values = [
                    float(strategy_profit) - float(baseline_profit)
                    for strategy_profit, baseline_profit in zip(
                        strategy_profit_values,
                        baseline_profit_values,
                        strict=False,
                    )
                ]
                last_delta_profit = float(delta_profit_values[-1]) if delta_profit_values else None
                delta_positive_streak_len = _tail_streak_len(
                    delta_profit_values,
                    lambda value: float(value) > 0.0,
                )
                delta_positive_streak_total_bnb = _tail_streak_total(
                    delta_profit_values,
                    lambda value: float(value) > 0.0,
                )
                row[f"feat_{key}_delta_profit_last_vs_{baseline_key}_bnb"] = _cell_or_blank(
                    last_delta_profit
                )
                row[f"feat_{key}_delta_profit_mean_vs_{baseline_key}_bnb"] = _cell_or_blank(
                    (
                        float(row[f"feat_{key}_profit_mean_bnb"])
                        - float(baseline_feature_values["profit_mean_bnb"])
                    )
                    if baseline_feature_values.get("profit_mean_bnb") is not None
                    else None
                )
                row[f"feat_{key}_delta_profit_recent3_mean_vs_{baseline_key}_bnb"] = _cell_or_blank(
                    (
                        float(row[f"feat_{key}_profit_recent3_mean_bnb"])
                        - float(baseline_feature_values["profit_recent3_mean_bnb"])
                    )
                    if row[f"feat_{key}_profit_recent3_mean_bnb"] != ""
                    and baseline_feature_values.get("profit_recent3_mean_bnb") is not None
                    else None
                )
                row[f"feat_{key}_delta_profit_recent3_minus_older_vs_{baseline_key}_bnb"] = _cell_or_blank(
                    (
                        float(row[f"feat_{key}_profit_recent3_minus_older_mean_bnb"])
                        - float(baseline_feature_values["profit_recent3_minus_older_mean_bnb"])
                    )
                    if row[f"feat_{key}_profit_recent3_minus_older_mean_bnb"] != ""
                    and baseline_feature_values.get("profit_recent3_minus_older_mean_bnb") is not None
                    else None
                )
                row[f"feat_{key}_delta_profit_positive_streak_len_vs_{baseline_key}"] = int(
                    delta_positive_streak_len
                )
                row[f"feat_{key}_delta_profit_positive_streak_total_vs_{baseline_key}_bnb"] = float(
                    delta_positive_streak_total_bnb
                )
                row[f"feat_{key}_delta_profit_positive_transition_prob_vs_{baseline_key}"] = _cell_or_blank(
                    _transition_prob(delta_profit_values, lambda value: float(value) > 0.0)
                )
                row[f"feat_{key}_delta_profit_positive_next_mean_vs_{baseline_key}_bnb"] = _cell_or_blank(
                    _conditional_next_mean(delta_profit_values, lambda value: float(value) > 0.0)
                )
                row[f"feat_{key}_delta_win_rate_mean_vs_{baseline_key}"] = _cell_or_blank(
                    (
                        float(row[f"feat_{key}_win_rate_mean"])
                        - float(baseline_feature_values["win_rate_mean"])
                    )
                    if baseline_feature_values.get("win_rate_mean") is not None
                    else None
                )
                row[f"feat_{key}_delta_positive_round_rate_mean_vs_{baseline_key}"] = _cell_or_blank(
                    (
                        float(row[f"feat_{key}_positive_round_rate_mean"])
                        - float(baseline_feature_values["positive_round_rate_mean"])
                    )
                    if baseline_feature_values.get("positive_round_rate_mean") is not None
                    else None
                )
                row[f"feat_{key}_delta_expected_net_mean_vs_{baseline_key}_bnb"] = _cell_or_blank(
                    (
                        float(row[f"feat_{key}_mean_expected_net_mean_bnb"])
                        - float(baseline_feature_values["expected_net_mean_bnb"])
                    )
                    if row[f"feat_{key}_mean_expected_net_mean_bnb"] != ""
                    and baseline_feature_values.get("expected_net_mean_bnb") is not None
                    else None
                )

        row["label_oracle_strategy_or_skip"] = str(target.oracle_strategy_or_skip)
        row["label_oracle_profit_bnb"] = float(target.oracle_profit_bnb)
        row["label_oracle_profit_per_500_rounds"] = float(
            float(target.oracle_profit_bnb) / float(target.num_rounds) * 500.0
            if int(target.num_rounds) > 0
            else 0.0
        )
        oracle_label_net_bnb += float(target.oracle_profit_bnb)
        rows_out.append(row)

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.name_prefix}_meta_strategy_dataset.csv"
    meta_path = output_dir / f"{args.name_prefix}_meta_strategy_dataset_meta.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows_out)

    total_target_rounds = sum(int(row["target_num_rounds"]) for row in rows_out)
    strategy_summary: dict[str, Any] = {}
    for strategy_prefix in all_series_names:
        net_profit_bnb = float(strategy_label_net_bnb[str(strategy_prefix)])
        total_bets = int(
            sum(int(row[f"label_{key_map[str(strategy_prefix)]}_next_block_num_bets"]) for row in rows_out)
        )
        strategy_summary[str(strategy_prefix)] = {
            "column_key": str(key_map[str(strategy_prefix)]),
            "label_net_profit_bnb": float(net_profit_bnb),
            "label_net_profit_per_500_rounds": (
                float(net_profit_bnb) / float(total_target_rounds) * 500.0
                if int(total_target_rounds) > 0
                else 0.0
            ),
            "label_total_bets": int(total_bets),
            "label_bet_rate": (
                float(total_bets) / float(total_target_rounds)
                if int(total_target_rounds) > 0
                else 0.0
            ),
        }

    metadata = {
        "dataset": {
            "name_prefix": str(args.name_prefix),
            "base_dir": str(args.base_dir),
            "output_csv": str(csv_path),
            "output_meta_json": str(meta_path),
            "trades_filename": str(args.trades_filename),
            "extra_series": {str(k): str(v) for k, v in extra_specs.items()},
            "block_size": int(args.block_size),
            "decision_window_rounds": int(decision_window_rounds),
            "history_block_rounds": int(history_block_rounds),
            "framing_mode": (
                "source_block_aligned" if bool(use_source_block_aligned) else "rolling_rounds"
            ),
            "num_blocks": int(args.num_blocks),
            "skip_most_recent_blocks": int(args.skip_most_recent_blocks),
            "lookback_blocks": int(args.lookback_blocks),
            "baseline_strategy_name": str(baseline_name),
            "num_source_blocks": int(args.num_blocks),
            "num_source_rounds": (
                int(sum(int(block.num_rounds) for block in blocks))
                if bool(use_source_block_aligned)
                else int(len(round_snapshots))
            ),
            "num_rows": int(len(rows_out)),
        },
        "strategy_prefixes": [str(x) for x in all_series_names],
        "strategy_column_keys": {str(k): str(v) for k, v in key_map.items()},
        "feature_columns": [str(x) for x in feature_columns],
        "label_columns": [str(x) for x in label_columns],
        "summary": {
            "oracle_label_net_profit_bnb": float(oracle_label_net_bnb),
            "oracle_label_net_profit_per_500_rounds": (
                float(oracle_label_net_bnb) / float(total_target_rounds) * 500.0
                if int(total_target_rounds) > 0
                else 0.0
            ),
            "strategies": strategy_summary,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"DATASET_CSV={csv_path}")
    print(f"DATASET_META={meta_path}")
    print(f"ROWS={len(rows_out)}")
    print(f"ORACLE_NET_BNB={oracle_label_net_bnb}")


if __name__ == "__main__":
    main()
