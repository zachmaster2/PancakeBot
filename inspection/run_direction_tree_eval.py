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
from pancakebot.domain.models.direction_tree_model import (
    DirectionTreeConfig,
    default_direction_tree_config,
    predict_direction_tree_probabilities,
    save_direction_tree_bundle,
    train_direction_tree_classifier,
)

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class DirectionTreeEvalRow:
    model_type: str
    sim_size: int
    tail_offset_rounds: int
    train_size: int
    valid_size: int
    random_seed: int
    num_examples: int
    feature_dim: int
    loaded_round_count: int
    total_rounds_available: int
    bundle_path: str
    valid_win_rate: float
    test_win_rate: float


@dataclass(frozen=True, slots=True)
class DirectionTreeEvalAggregateRow:
    model_type: str
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
    parser.add_argument("--model-type", type=str, choices=("lightgbm", "catboost"), required=True)
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--sim-sizes", type=str, default="6480,8640,10800")
    parser.add_argument("--tail-offset-rounds", type=str, default="0,432,864")
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--valid-size", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=20260402)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--subsample", type=float, default=0.80)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
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


def _split_target_epochs(
    *,
    target_epochs: tuple[int, ...],
    train_size: int,
    valid_size: int,
    sim_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    needed = int(train_size) + int(valid_size) + int(sim_size)
    if len(target_epochs) != int(needed):
        raise InvariantError("direction_tree_split_len_mismatch")
    train_epochs = tuple(int(epoch) for epoch in target_epochs[: int(train_size)])
    valid_epochs = tuple(
        int(epoch)
        for epoch in target_epochs[int(train_size) : int(train_size) + int(valid_size)]
    )
    test_epochs = tuple(int(epoch) for epoch in target_epochs[-int(sim_size) :])
    return train_epochs, valid_epochs, test_epochs


def _win_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) <= 0:
        raise InvariantError("direction_tree_win_rate_empty")
    return float(np.mean(np.asarray(y_true, dtype=np.int64) == np.asarray(y_pred, dtype=np.int64)))


def _aggregate_rows(rows: list[DirectionTreeEvalRow]) -> list[DirectionTreeEvalAggregateRow]:
    grouped: dict[tuple[str, int], list[DirectionTreeEvalRow]] = {}
    for row in rows:
        grouped.setdefault((str(row.model_type), int(row.sim_size)), []).append(row)
    out: list[DirectionTreeEvalAggregateRow] = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), int(item[1]))):
        group = grouped[key]
        out.append(
            DirectionTreeEvalAggregateRow(
                model_type=str(group[0].model_type),
                sim_size=int(group[0].sim_size),
                train_size=int(group[0].train_size),
                valid_size=int(group[0].valid_size),
                random_seed=int(group[0].random_seed),
                num_offsets=int(len(group)),
                mean_test_win_rate=float(np.mean([row.test_win_rate for row in group])),
                min_test_win_rate=float(min(row.test_win_rate for row in group)),
                max_test_win_rate=float(max(row.test_win_rate for row in group)),
                mean_valid_win_rate=float(np.mean([row.valid_win_rate for row in group])),
            )
        )
    return out


def main() -> None:
    args = _build_parser().parse_args()
    sim_sizes = parse_positive_int_list(args.sim_sizes)
    offsets = parse_nonnegative_int_list(args.tail_offset_rounds)
    include_feature_groups = parse_optional_str_list(args.include_feature_groups)
    exclude_feature_groups = parse_optional_str_list(args.exclude_feature_groups)
    if int(args.train_size) <= 0:
        raise InvariantError("direction_tree_train_size_nonpositive")
    if int(args.valid_size) <= 0:
        raise InvariantError("direction_tree_valid_size_nonpositive")
    cfg = default_direction_tree_config(model_type=str(args.model_type))
    model_cfg = DirectionTreeConfig(
        model_type=str(args.model_type),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        early_stopping_rounds=int(args.early_stopping_rounds),
    )
    if cfg.model_type != model_cfg.model_type:
        raise InvariantError("direction_tree_model_type_cfg_mismatch")
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[DirectionTreeEvalRow] = []
    for sim_size in sim_sizes:
        required_examples = int(args.train_size) + int(args.valid_size) + int(sim_size)
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
            bundle = train_direction_tree_classifier(
                dataset=dataset,
                train_target_epochs=train_epochs,
                valid_target_epochs=valid_epochs,
                random_seed=int(args.random_seed),
                config=model_cfg,
            )
            suffix = "direction_tree.pkl"
            bundle_path = (
                output_dir
                / f"{str(args.name_prefix)}_tail{int(sim_size)}_off{int(tail_offset_rounds):05d}_{str(args.model_type)}_{suffix}"
            ).resolve()
            save_direction_tree_bundle(bundle=bundle, path=str(bundle_path))

            index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
            valid_idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in valid_epochs], dtype=np.int64)
            test_idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in test_epochs], dtype=np.int64)
            probs = predict_direction_tree_probabilities(
                bundle=bundle,
                feature_matrix=np.asarray(dataset.feature_matrix, dtype=np.float32),
            )
            valid_preds = (np.asarray(probs[valid_idx], dtype=np.float32) >= 0.5).astype(np.int64)
            test_preds = (np.asarray(probs[test_idx], dtype=np.float32) >= 0.5).astype(np.int64)
            rows.append(
                DirectionTreeEvalRow(
                    model_type=str(args.model_type),
                    sim_size=int(sim_size),
                    tail_offset_rounds=int(tail_offset_rounds),
                    train_size=int(args.train_size),
                    valid_size=int(args.valid_size),
                    random_seed=int(args.random_seed),
                    num_examples=int(dataset.num_examples),
                    feature_dim=int(dataset.feature_matrix.shape[1]),
                    loaded_round_count=int(eval_slice.loaded_round_count),
                    total_rounds_available=int(eval_slice.total_rounds_available),
                    bundle_path=str(bundle_path),
                    valid_win_rate=float(_win_rate(dataset.labels[valid_idx], valid_preds)),
                    test_win_rate=float(_win_rate(dataset.labels[test_idx], test_preds)),
                )
            )

    aggregates = _aggregate_rows(rows)
    rows_out = rows_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="direction_tree_eval_rows",
    )
    summary_out = summary_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="direction_tree_eval_summary",
    )
    with rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    summary_payload = {
        "model_type": str(args.model_type),
        "sim_sizes": [int(x) for x in sim_sizes],
        "tail_offset_rounds": [int(x) for x in offsets],
        "train_size": int(args.train_size),
        "valid_size": int(args.valid_size),
        "random_seed": int(args.random_seed),
        "rows_csv_path": str(rows_out),
        "aggregates": [asdict(row) for row in aggregates],
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
