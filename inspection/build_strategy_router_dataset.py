"""Build a router-training dataset from historical strategy block artifacts.

This tool is inspection-only. It does not run strategy logic; it only reads
existing `dislocation_trades.csv` artifacts and emits:
1) a wide CSV table for router model experiments
2) a JSON metadata/summary file

The dataset separates:
- router features (cutoff-time strategy signals)
- labels (realized profits and hindsight oracle choice)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from inspection.strategy_router_common import (
    direction_to_idx,
    load_block_round_snapshots,
    oracle_skip_pick,
    parse_strategy_prefixes,
    to_column_key_map,
)


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for router dataset construction."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--strategy-prefixes", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--num-blocks", type=int, default=80)
    parser.add_argument("--skip-most-recent-blocks", type=int, default=0)
    parser.add_argument("--base-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--output-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--trades-filename", type=str, default="dislocation_trades.csv")
    return parser


def _build_columns(strategy_prefixes: list[str], key_map: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """Return full CSV column list plus feature/label column subsets."""

    base_columns = [
        "block_index",
        "sim_offset_rounds",
        "epoch",
    ]

    feature_columns: list[str] = []
    label_columns: list[str] = []
    for strategy_prefix in strategy_prefixes:
        key = key_map[str(strategy_prefix)]
        feature_columns.extend(
            [
                f"feat_{key}_available",
                f"feat_{key}_bet_available",
                f"feat_{key}_direction_idx",
                f"feat_{key}_expected_net_selected_bnb",
                f"feat_{key}_abs_dislocation_bull",
                f"feat_{key}_p_nowcast_bull",
                f"feat_{key}_p_market_bull",
                f"feat_{key}_bet_size_bnb",
            ]
        )
        label_columns.extend(
            [
                f"label_{key}_profit_bnb",
            ]
        )

    label_columns.extend(
        [
            "label_best_strategy_or_skip",
            "label_best_action",
            "label_oracle_profit_bnb",
        ]
    )
    return base_columns + feature_columns + label_columns, feature_columns, label_columns


def _cell_or_blank(value: float | None) -> float | str:
    """Keep missing float features blank in CSV for readability."""

    if value is None:
        return ""
    return float(value)


def main() -> None:
    """Load block artifacts and emit router dataset CSV + metadata JSON."""

    args = _build_parser().parse_args()

    strategy_prefixes = parse_strategy_prefixes(str(args.strategy_prefixes))
    if len(strategy_prefixes) < 2:
        raise ValueError("router_dataset_requires_at_least_two_strategies")

    snapshots = load_block_round_snapshots(
        strategy_prefixes=strategy_prefixes,
        block_size=int(args.block_size),
        num_blocks=int(args.num_blocks),
        skip_most_recent_blocks=int(args.skip_most_recent_blocks),
        base_dir=Path(str(args.base_dir)),
        trades_filename=str(args.trades_filename),
    )
    if not snapshots:
        raise ValueError("router_dataset_no_snapshots")

    key_map = to_column_key_map(strategy_prefixes)
    all_columns, feature_columns, label_columns = _build_columns(strategy_prefixes, key_map)

    rows_out: list[dict[str, Any]] = []
    oracle_net_total_bnb = 0.0
    oracle_positive_rounds = 0
    rounds_total = 0

    strategy_net_total_bnb: dict[str, float] = {str(s): 0.0 for s in strategy_prefixes}
    strategy_bets: dict[str, int] = {str(s): 0 for s in strategy_prefixes}
    strategy_wins: dict[str, int] = {str(s): 0 for s in strategy_prefixes}

    for snapshot in snapshots:
        row: dict[str, Any] = {
            "block_index": int(snapshot.block_index),
            "sim_offset_rounds": int(snapshot.sim_offset_rounds),
            "epoch": int(snapshot.epoch),
        }

        best_strategy_or_skip, oracle_profit_bnb = oracle_skip_pick(snapshot.rows_by_strategy)
        row["label_best_strategy_or_skip"] = str(best_strategy_or_skip)
        row["label_best_action"] = "BET" if str(best_strategy_or_skip) != "SKIP" else "SKIP"
        row["label_oracle_profit_bnb"] = float(oracle_profit_bnb)

        oracle_net_total_bnb += float(oracle_profit_bnb)
        if float(oracle_profit_bnb) > 0.0:
            oracle_positive_rounds += 1
        rounds_total += 1

        for strategy_prefix in strategy_prefixes:
            key = key_map[str(strategy_prefix)]
            trade_row = snapshot.rows_by_strategy.get(str(strategy_prefix))
            if trade_row is None:
                row[f"feat_{key}_available"] = 0
                row[f"feat_{key}_bet_available"] = 0
                row[f"feat_{key}_direction_idx"] = -1
                row[f"feat_{key}_expected_net_selected_bnb"] = ""
                row[f"feat_{key}_abs_dislocation_bull"] = ""
                row[f"feat_{key}_p_nowcast_bull"] = ""
                row[f"feat_{key}_p_market_bull"] = ""
                row[f"feat_{key}_bet_size_bnb"] = ""
                row[f"label_{key}_profit_bnb"] = 0.0
                continue

            bet_available = 1 if str(trade_row.action) == "BET" else 0
            row[f"feat_{key}_available"] = 1
            row[f"feat_{key}_bet_available"] = int(bet_available)
            row[f"feat_{key}_direction_idx"] = int(direction_to_idx(trade_row.direction))
            row[f"feat_{key}_expected_net_selected_bnb"] = _cell_or_blank(
                trade_row.expected_net_selected_bnb
            )
            row[f"feat_{key}_abs_dislocation_bull"] = _cell_or_blank(
                abs(float(trade_row.dislocation_bull))
                if trade_row.dislocation_bull is not None
                else None
            )
            row[f"feat_{key}_p_nowcast_bull"] = _cell_or_blank(trade_row.p_nowcast_bull)
            row[f"feat_{key}_p_market_bull"] = _cell_or_blank(trade_row.p_market_bull)
            row[f"feat_{key}_bet_size_bnb"] = _cell_or_blank(trade_row.bet_size_bnb)
            row[f"label_{key}_profit_bnb"] = float(trade_row.profit_bnb)

            strategy_net_total_bnb[str(strategy_prefix)] += float(trade_row.profit_bnb)
            if int(bet_available) == 1:
                strategy_bets[str(strategy_prefix)] += 1
                if float(trade_row.profit_bnb) > 0.0:
                    strategy_wins[str(strategy_prefix)] += 1

        rows_out.append(row)

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.name_prefix}_router_dataset.csv"
    meta_path = output_dir / f"{args.name_prefix}_router_dataset_meta.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows_out)

    strategy_summary: dict[str, Any] = {}
    for strategy_prefix in strategy_prefixes:
        rounds_per_500 = float(rounds_total) / 500.0 if int(rounds_total) > 0 else 0.0
        net_total = float(strategy_net_total_bnb[str(strategy_prefix)])
        strategy_summary[str(strategy_prefix)] = {
            "column_key": str(key_map[str(strategy_prefix)]),
            "net_profit_bnb": float(net_total),
            "net_profit_per_500_rounds": float(net_total / rounds_per_500) if rounds_per_500 > 0 else 0.0,
            "num_bets": int(strategy_bets[str(strategy_prefix)]),
            "win_rate_on_bets": float(
                _safe_rate(
                    int(strategy_wins[str(strategy_prefix)]),
                    int(strategy_bets[str(strategy_prefix)]),
                )
            ),
        }

    rounds_per_500 = float(rounds_total) / 500.0 if int(rounds_total) > 0 else 0.0
    metadata = {
        "dataset": {
            "name_prefix": str(args.name_prefix),
            "base_dir": str(args.base_dir),
            "output_csv": str(csv_path),
            "output_meta_json": str(meta_path),
            "trades_filename": str(args.trades_filename),
            "block_size": int(args.block_size),
            "num_blocks": int(args.num_blocks),
            "skip_most_recent_blocks": int(args.skip_most_recent_blocks),
            "num_rows": int(len(rows_out)),
        },
        "strategy_prefixes": [str(x) for x in strategy_prefixes],
        "strategy_column_keys": {str(k): str(v) for k, v in key_map.items()},
        "feature_columns": [str(x) for x in feature_columns],
        "label_columns": [str(x) for x in label_columns],
        "summary": {
            "oracle_net_profit_bnb": float(oracle_net_total_bnb),
            "oracle_net_profit_per_500_rounds": (
                float(oracle_net_total_bnb / rounds_per_500)
                if rounds_per_500 > 0
                else 0.0
            ),
            "oracle_positive_round_fraction": float(_safe_rate(oracle_positive_rounds, rounds_total)),
            "strategies": strategy_summary,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"DATASET_CSV={csv_path}")
    print(f"DATASET_META={meta_path}")
    print(f"ROUNDS={rounds_total}")
    print(f"ORACLE_NET_BNB={oracle_net_total_bnb}")
    print(
        "ORACLE_PER_500="
        + str(metadata["summary"]["oracle_net_profit_per_500_rounds"])
    )


if __name__ == "__main__":
    main()
