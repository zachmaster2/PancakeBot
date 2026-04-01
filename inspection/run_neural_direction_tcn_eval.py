from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from inspection.neural_direction_eval_common import (
    feature_groups_help_text,
    load_recent_direction_eval_slice,
    parse_nonnegative_int_list,
    parse_optional_str_list,
    parse_positive_int_list,
    rows_path,
    summary_path,
)
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_tcn import (
    NeuralDirectionTcnConfig,
    build_sequence_examples_for_target_epochs,
    predict_neural_direction_tcn_probabilities,
    save_neural_direction_tcn_bundle,
    train_neural_direction_tcn,
)

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class NeuralDirectionTcnEvalRow:
    sim_size: int
    tail_offset_rounds: int
    train_size: int
    valid_size: int
    random_seed: int
    num_examples: int
    feature_dim: int
    seq_len: int
    loaded_round_count: int
    total_rounds_available: int
    bundle_path: str
    valid_win_rate: float
    test_win_rate: float


@dataclass(frozen=True, slots=True)
class NeuralDirectionTcnEvalAggregateRow:
    sim_size: int
    train_size: int
    valid_size: int
    random_seed: int
    num_offsets: int
    mean_test_win_rate: float
    min_test_win_rate: float
    max_test_win_rate: float
    mean_valid_win_rate: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--sim-sizes", type=str, default="6480,8640,10800")
    parser.add_argument("--tail-offset-rounds", type=str, default="0,216,432,648,864")
    parser.add_argument("--train-size", type=int, default=15000)
    parser.add_argument("--valid-size", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=20260401)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--channels", type=str, default="64,64")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience-epochs", type=int, default=5)
    parser.add_argument(
        "--include-feature-groups",
        type=str,
        default=None,
        help=f"Comma-separated feature groups to keep. Available: {feature_groups_help_text()}",
    )
    parser.add_argument(
        "--exclude-feature-groups",
        type=str,
        default=None,
        help=f"Comma-separated feature groups to drop. Available: {feature_groups_help_text()}",
    )
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _parse_channels(raw: str) -> tuple[int, ...]:
    return tuple(parse_positive_int_list(str(raw)))


def _split_target_epochs(
    *,
    target_epochs: tuple[int, ...],
    train_size: int,
    valid_size: int,
    sim_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    needed = int(train_size) + int(valid_size) + int(sim_size)
    if len(target_epochs) < int(needed):
        raise InvariantError("neural_direction_tcn_split_len_mismatch")
    target_tail = tuple(int(epoch) for epoch in target_epochs[-int(needed) :])
    train_epochs = tuple(int(epoch) for epoch in target_tail[: int(train_size)])
    valid_epochs = tuple(
        int(epoch)
        for epoch in target_tail[int(train_size) : int(train_size) + int(valid_size)]
    )
    test_epochs = tuple(int(epoch) for epoch in target_tail[-int(sim_size) :])
    return train_epochs, valid_epochs, test_epochs


def _win_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) <= 0:
        raise InvariantError("neural_direction_tcn_win_rate_empty")
    return float(np.mean(np.asarray(y_true, dtype=np.int64) == np.asarray(y_pred, dtype=np.int64)))


def _aggregate_rows(rows: list[NeuralDirectionTcnEvalRow]) -> list[NeuralDirectionTcnEvalAggregateRow]:
    grouped: dict[int, list[NeuralDirectionTcnEvalRow]] = {}
    for row in rows:
        grouped.setdefault(int(row.sim_size), []).append(row)
    out: list[NeuralDirectionTcnEvalAggregateRow] = []
    for sim_size in sorted(grouped):
        group = grouped[int(sim_size)]
        out.append(
            NeuralDirectionTcnEvalAggregateRow(
                sim_size=int(sim_size),
                train_size=int(group[0].train_size),
                valid_size=int(group[0].valid_size),
                random_seed=int(group[0].random_seed),
                num_offsets=int(len(group)),
                mean_test_win_rate=float(
                    sum(float(row.test_win_rate) for row in group) / float(len(group))
                ),
                min_test_win_rate=float(min(float(row.test_win_rate) for row in group)),
                max_test_win_rate=float(max(float(row.test_win_rate) for row in group)),
                mean_valid_win_rate=float(
                    sum(float(row.valid_win_rate) for row in group) / float(len(group))
                ),
            )
        )
    return out


def main() -> None:
    args = _build_parser().parse_args()
    sim_sizes = parse_positive_int_list(args.sim_sizes)
    offsets = parse_nonnegative_int_list(args.tail_offset_rounds)
    if int(args.train_size) <= 0:
        raise InvariantError("neural_direction_tcn_train_size_nonpositive")
    if int(args.valid_size) <= 0:
        raise InvariantError("neural_direction_tcn_valid_size_nonpositive")

    model_cfg = NeuralDirectionTcnConfig(
        seq_len=int(args.seq_len),
        channels=_parse_channels(args.channels),
        kernel_size=int(args.kernel_size),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        patience_epochs=int(args.patience_epochs),
    )
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[NeuralDirectionTcnEvalRow] = []
    include_feature_groups = parse_optional_str_list(args.include_feature_groups)
    exclude_feature_groups = parse_optional_str_list(args.exclude_feature_groups)

    for sim_size in sim_sizes:
        required_examples = (
            int(args.train_size)
            + int(args.valid_size)
            + int(sim_size)
            + int(model_cfg.seq_len)
            - 1
        )
        for tail_offset_rounds in offsets:
            eval_slice = load_recent_direction_eval_slice(
                config_path=str(args.config),
                required_examples=int(required_examples),
                tail_offset_rounds=int(tail_offset_rounds),
                include_feature_groups=include_feature_groups,
                exclude_feature_groups=exclude_feature_groups,
            )
            dataset = eval_slice.dataset
            train_epochs, valid_epochs, test_epochs = _split_target_epochs(
                target_epochs=dataset.target_epochs,
                train_size=int(args.train_size),
                valid_size=int(args.valid_size),
                sim_size=int(sim_size),
            )
            bundle = train_neural_direction_tcn(
                dataset=dataset,
                train_target_epochs=train_epochs,
                valid_target_epochs=valid_epochs,
                random_seed=int(args.random_seed),
                config=model_cfg,
            )
            bundle_path = (
                output_dir
                / f"{str(args.name_prefix)}_tail{int(sim_size)}_off{int(tail_offset_rounds):05d}_neural_direction_tcn.pt"
            ).resolve()
            save_neural_direction_tcn_bundle(bundle=bundle, path=str(bundle_path))
            test_x, test_y = build_sequence_examples_for_target_epochs(
                dataset=dataset,
                target_epochs=test_epochs,
                seq_len=int(model_cfg.seq_len),
            )
            probs = predict_neural_direction_tcn_probabilities(
                bundle=bundle,
                feature_sequences=test_x,
            )
            preds = (np.asarray(probs, dtype=np.float32) >= 0.5).astype(np.int64)
            rows.append(
                NeuralDirectionTcnEvalRow(
                    sim_size=int(sim_size),
                    tail_offset_rounds=int(tail_offset_rounds),
                    train_size=int(args.train_size),
                    valid_size=int(args.valid_size),
                    random_seed=int(args.random_seed),
                    num_examples=int(dataset.num_examples),
                    feature_dim=int(dataset.feature_matrix.shape[1]),
                    seq_len=int(model_cfg.seq_len),
                    loaded_round_count=int(eval_slice.loaded_round_count),
                    total_rounds_available=int(eval_slice.total_rounds_available),
                    bundle_path=str(bundle_path),
                    valid_win_rate=float(bundle.metadata["best_valid_win_rate"]),
                    test_win_rate=float(_win_rate(test_y, preds)),
                )
            )

    aggregates = _aggregate_rows(rows)
    rows_out = rows_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_tcn_eval_rows",
    )
    summary_out = summary_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_tcn_eval_summary",
    )

    with rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary_payload = {
        "rows_csv_path": str(rows_out),
        "aggregates": [asdict(row) for row in aggregates],
        "model_config": asdict(model_cfg),
        "include_feature_groups": None if include_feature_groups is None else list(include_feature_groups),
        "exclude_feature_groups": None if exclude_feature_groups is None else list(exclude_feature_groups),
        "row_count": int(len(rows)),
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
