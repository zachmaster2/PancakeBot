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
from pancakebot.domain.models.neural_direction_mlp import (
    NeuralDirectionMlpConfig,
    build_exponential_recency_weights,
    predict_neural_direction_probabilities,
    save_neural_direction_mlp_bundle,
    train_neural_direction_mlp,
)

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class NeuralDirectionMlpEvalRow:
    sim_size: int
    tail_offset_rounds: int
    training_policy: str
    train_size: int
    pretrain_size: int
    valid_size: int
    recency_half_life_examples: float | None
    random_seed: int
    num_examples: int
    feature_dim: int
    loaded_round_count: int
    total_rounds_available: int
    bundle_path: str
    valid_win_rate: float
    test_win_rate: float


@dataclass(frozen=True, slots=True)
class NeuralDirectionMlpEvalAggregateRow:
    sim_size: int
    training_policy: str
    train_size: int
    pretrain_size: int
    valid_size: int
    recency_half_life_examples: float | None
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
    parser.add_argument("--pretrain-size", type=int, default=0)
    parser.add_argument("--valid-size", type=int, default=3000)
    parser.add_argument("--recency-half-life-examples", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=20260401)
    parser.add_argument("--hidden-sizes", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience-epochs", type=int, default=6)
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


def _parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    values = parse_positive_int_list(str(raw))
    return tuple(int(value) for value in values)


def _split_target_epochs(
    *,
    target_epochs: tuple[int, ...],
    train_size: int,
    pretrain_size: int,
    valid_size: int,
    sim_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    needed = int(pretrain_size) + int(train_size) + int(valid_size) + int(sim_size)
    if len(target_epochs) != int(needed):
        raise InvariantError("neural_direction_mlp_split_len_mismatch")
    cursor = 0
    pretrain_train_epochs: tuple[int, ...] = ()
    pretrain_valid_epochs: tuple[int, ...] = ()
    if int(pretrain_size) > 0:
        if int(pretrain_size) <= int(valid_size):
            raise InvariantError("neural_direction_mlp_pretrain_size_too_small")
        pretrain_block = tuple(int(epoch) for epoch in target_epochs[: int(pretrain_size)])
        pretrain_train_epochs = tuple(int(epoch) for epoch in pretrain_block[: -int(valid_size)])
        pretrain_valid_epochs = tuple(int(epoch) for epoch in pretrain_block[-int(valid_size) :])
        cursor = int(pretrain_size)
    train_epochs = tuple(int(epoch) for epoch in target_epochs[cursor : cursor + int(train_size)])
    cursor += int(train_size)
    valid_epochs = tuple(int(epoch) for epoch in target_epochs[cursor : cursor + int(valid_size)])
    cursor += int(valid_size)
    test_epochs = tuple(int(epoch) for epoch in target_epochs[cursor : cursor + int(sim_size)])
    return pretrain_train_epochs, pretrain_valid_epochs, train_epochs, valid_epochs, test_epochs


def _win_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) <= 0:
        raise InvariantError("neural_direction_mlp_win_rate_empty")
    return float(np.mean(np.asarray(y_true, dtype=np.int64) == np.asarray(y_pred, dtype=np.int64)))


def _aggregate_rows(rows: list[NeuralDirectionMlpEvalRow]) -> list[NeuralDirectionMlpEvalAggregateRow]:
    grouped: dict[tuple[int, str, int, float | None], list[NeuralDirectionMlpEvalRow]] = {}
    for row in rows:
        key = (
            int(row.sim_size),
            str(row.training_policy),
            int(row.pretrain_size),
            None if row.recency_half_life_examples is None else float(row.recency_half_life_examples),
        )
        grouped.setdefault(key, []).append(row)
    out: list[NeuralDirectionMlpEvalAggregateRow] = []
    for key in sorted(grouped, key=lambda item: (int(item[0]), str(item[1]), int(item[2]), -1.0 if item[3] is None else float(item[3]))):
        group = grouped[key]
        out.append(
            NeuralDirectionMlpEvalAggregateRow(
                sim_size=int(group[0].sim_size),
                training_policy=str(group[0].training_policy),
                train_size=int(group[0].train_size),
                pretrain_size=int(group[0].pretrain_size),
                valid_size=int(group[0].valid_size),
                recency_half_life_examples=None
                if group[0].recency_half_life_examples is None
                else float(group[0].recency_half_life_examples),
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
        raise InvariantError("neural_direction_mlp_train_size_nonpositive")
    if int(args.pretrain_size) < 0:
        raise InvariantError("neural_direction_mlp_pretrain_size_negative")
    if int(args.valid_size) <= 0:
        raise InvariantError("neural_direction_mlp_valid_size_nonpositive")
    if float(args.recency_half_life_examples) < 0.0:
        raise InvariantError("neural_direction_mlp_recency_half_life_negative")

    model_cfg = NeuralDirectionMlpConfig(
        hidden_sizes=_parse_hidden_sizes(args.hidden_sizes),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        patience_epochs=int(args.patience_epochs),
    )

    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[NeuralDirectionMlpEvalRow] = []
    include_feature_groups = parse_optional_str_list(args.include_feature_groups)
    exclude_feature_groups = parse_optional_str_list(args.exclude_feature_groups)
    training_policy = _training_policy_name(
        pretrain_size=int(args.pretrain_size),
        recency_half_life_examples=float(args.recency_half_life_examples),
    )

    for sim_size in sim_sizes:
        required_examples = int(args.pretrain_size) + int(args.train_size) + int(args.valid_size) + int(sim_size)
        for tail_offset_rounds in offsets:
            eval_slice = load_recent_direction_eval_slice(
                config_path=str(args.config),
                required_examples=int(required_examples),
                tail_offset_rounds=int(tail_offset_rounds),
                include_feature_groups=include_feature_groups,
                exclude_feature_groups=exclude_feature_groups,
            )
            dataset = eval_slice.dataset
            pretrain_train_epochs, pretrain_valid_epochs, train_epochs, valid_epochs, test_epochs = _split_target_epochs(
                target_epochs=dataset.target_epochs,
                train_size=int(args.train_size),
                pretrain_size=int(args.pretrain_size),
                valid_size=int(args.valid_size),
                sim_size=int(sim_size),
            )
            initial_bundle = None
            if int(args.pretrain_size) > 0:
                initial_bundle = train_neural_direction_mlp(
                    dataset=dataset,
                    train_target_epochs=pretrain_train_epochs,
                    valid_target_epochs=pretrain_valid_epochs,
                    random_seed=int(args.random_seed),
                    config=model_cfg,
                )
            train_sample_weights = None
            if float(args.recency_half_life_examples) > 0.0:
                train_sample_weights = build_exponential_recency_weights(
                    num_examples=len(train_epochs),
                    half_life_examples=float(args.recency_half_life_examples),
                )
            bundle = train_neural_direction_mlp(
                dataset=dataset,
                train_target_epochs=train_epochs,
                valid_target_epochs=valid_epochs,
                random_seed=int(args.random_seed),
                config=model_cfg,
                train_sample_weights=train_sample_weights,
                initial_bundle=initial_bundle,
            )
            bundle_path = (
                output_dir
                / f"{str(args.name_prefix)}_tail{int(sim_size)}_off{int(tail_offset_rounds):05d}_neural_direction_mlp.pt"
            ).resolve()
            save_neural_direction_mlp_bundle(bundle=bundle, path=str(bundle_path))

            probs = predict_neural_direction_probabilities(
                bundle=bundle,
                feature_matrix=np.asarray(dataset.feature_matrix, dtype=np.float32),
            )
            preds = (np.asarray(probs, dtype=np.float32) >= 0.5).astype(np.int64)
            index_by_epoch = {
                int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)
            }
            test_idx = [int(index_by_epoch[int(epoch)]) for epoch in test_epochs]
            test_y = np.asarray(dataset.labels[test_idx], dtype=np.int64)
            test_pred = np.asarray(preds[test_idx], dtype=np.int64)
            rows.append(
                NeuralDirectionMlpEvalRow(
                    sim_size=int(sim_size),
                    tail_offset_rounds=int(tail_offset_rounds),
                    training_policy=str(training_policy),
                    train_size=int(args.train_size),
                    pretrain_size=int(args.pretrain_size),
                    valid_size=int(args.valid_size),
                    recency_half_life_examples=None
                    if float(args.recency_half_life_examples) <= 0.0
                    else float(args.recency_half_life_examples),
                    random_seed=int(args.random_seed),
                    num_examples=int(dataset.num_examples),
                    feature_dim=int(dataset.feature_matrix.shape[1]),
                    loaded_round_count=int(eval_slice.loaded_round_count),
                    total_rounds_available=int(eval_slice.total_rounds_available),
                    bundle_path=str(bundle_path),
                    valid_win_rate=float(bundle.metadata["best_valid_win_rate"]),
                    test_win_rate=float(_win_rate(test_y, test_pred)),
                )
            )

    aggregates = _aggregate_rows(rows)
    rows_out = rows_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_mlp_eval_rows",
    )
    summary_out = summary_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_mlp_eval_summary",
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
        "training_policy": str(training_policy),
        "train_size": int(args.train_size),
        "pretrain_size": int(args.pretrain_size),
        "valid_size": int(args.valid_size),
        "recency_half_life_examples": None
        if float(args.recency_half_life_examples) <= 0.0
        else float(args.recency_half_life_examples),
        "include_feature_groups": None if include_feature_groups is None else list(include_feature_groups),
        "exclude_feature_groups": None if exclude_feature_groups is None else list(exclude_feature_groups),
        "row_count": int(len(rows)),
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def _training_policy_name(*, pretrain_size: int, recency_half_life_examples: float) -> str:
    has_pretrain = int(pretrain_size) > 0
    has_recency = float(recency_half_life_examples) > 0.0
    if has_pretrain and has_recency:
        return "pretrain_finetune_recency_exp"
    if has_pretrain:
        return "pretrain_finetune"
    if has_recency:
        return "recency_exp"
    return "flat"


if __name__ == "__main__":
    main()
