from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from inspection.neural_direction_eval_common import (
    load_recent_direction_eval_slice,
    parse_nonnegative_int_list,
    parse_positive_int_list,
    rows_path,
    summary_path,
)
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_dataset import NeuralDirectionDataset

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class NeuralDirectionBaselineRow:
    baseline_name: str
    sim_size: int
    tail_offset_rounds: int
    num_examples: int
    win_rate: float
    fallback_count: int
    loaded_round_count: int
    total_rounds_available: int


@dataclass(frozen=True, slots=True)
class NeuralDirectionBaselineAggregateRow:
    baseline_name: str
    sim_size: int
    num_offsets: int
    mean_win_rate: float
    min_win_rate: float
    max_win_rate: float
    mean_num_examples: float
    total_fallback_count: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--sim-sizes", type=str, default="6480,8640,10800")
    parser.add_argument("--tail-offset-rounds", type=str, default="0,216,432,648,864")
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _win_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) <= 0:
        raise InvariantError("neural_direction_baseline_win_rate_empty")
    return float(np.mean(np.asarray(y_true, dtype=np.int64) == np.asarray(y_pred, dtype=np.int64)))


def _baseline_predictions(*, dataset: NeuralDirectionDataset, baseline_name: str) -> tuple[np.ndarray, int]:
    y_true = np.asarray(dataset.labels, dtype=np.int64)
    if str(baseline_name) == "always_bull":
        return np.ones_like(y_true, dtype=np.int64), 0
    if str(baseline_name) == "always_bear":
        return np.zeros_like(y_true, dtype=np.int64), 0
    if str(baseline_name) == "previous_settled_side":
        preds = np.asarray(dataset.previous_settled_labels, dtype=np.int64)
        fallback_count = int(np.sum(~np.asarray(dataset.previous_settled_available, dtype=bool)))
        return preds, fallback_count
    raise InvariantError("neural_direction_baseline_name_invalid")


def _aggregate_rows(rows: list[NeuralDirectionBaselineRow]) -> list[NeuralDirectionBaselineAggregateRow]:
    grouped: dict[tuple[str, int], list[NeuralDirectionBaselineRow]] = {}
    for row in rows:
        grouped.setdefault((str(row.baseline_name), int(row.sim_size)), []).append(row)

    out: list[NeuralDirectionBaselineAggregateRow] = []
    for key in sorted(grouped):
        group = grouped[key]
        win_rates = [float(row.win_rate) for row in group]
        out.append(
            NeuralDirectionBaselineAggregateRow(
                baseline_name=str(key[0]),
                sim_size=int(key[1]),
                num_offsets=int(len(group)),
                mean_win_rate=float(sum(win_rates) / float(len(win_rates))),
                min_win_rate=float(min(win_rates)),
                max_win_rate=float(max(win_rates)),
                mean_num_examples=float(sum(int(row.num_examples) for row in group) / float(len(group))),
                total_fallback_count=int(sum(int(row.fallback_count) for row in group)),
            )
        )
    return out


def main() -> None:
    args = _build_parser().parse_args()
    sim_sizes = parse_positive_int_list(args.sim_sizes)
    offsets = parse_nonnegative_int_list(args.tail_offset_rounds)
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_names = ("always_bull", "always_bear", "previous_settled_side")
    rows: list[NeuralDirectionBaselineRow] = []

    for sim_size in sim_sizes:
        for tail_offset_rounds in offsets:
            eval_slice = load_recent_direction_eval_slice(
                config_path=str(args.config),
                required_examples=int(sim_size),
                tail_offset_rounds=int(tail_offset_rounds),
            )
            dataset = eval_slice.dataset
            y_true = np.asarray(dataset.labels, dtype=np.int64)
            for baseline_name in baseline_names:
                y_pred, fallback_count = _baseline_predictions(
                    dataset=dataset,
                    baseline_name=str(baseline_name),
                )
                rows.append(
                    NeuralDirectionBaselineRow(
                        baseline_name=str(baseline_name),
                        sim_size=int(sim_size),
                        tail_offset_rounds=int(tail_offset_rounds),
                        num_examples=int(len(y_true)),
                        win_rate=float(_win_rate(y_true, y_pred)),
                        fallback_count=int(fallback_count),
                        loaded_round_count=int(eval_slice.loaded_round_count),
                        total_rounds_available=int(eval_slice.total_rounds_available),
                    )
                )

    aggregates = _aggregate_rows(rows)
    rows_out = rows_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_baselines_rows",
    )
    summary_out = summary_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_baselines_summary",
    )

    with rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary_payload = {
        "rows_csv_path": str(rows_out),
        "aggregates": [asdict(row) for row in aggregates],
        "row_count": int(len(rows)),
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
