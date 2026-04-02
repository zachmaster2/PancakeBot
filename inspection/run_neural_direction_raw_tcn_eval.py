from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from inspection.neural_direction_eval_common import (
    parse_nonnegative_int_list,
    parse_positive_int_list,
    rows_path,
    summary_path,
)
from inspection.neural_direction_raw_eval_common import load_recent_raw_direction_eval_slice
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_raw_sequence_dataset import (
    build_raw_sequence_examples_for_target_epochs,
    select_raw_sequence_lengths,
)
from pancakebot.domain.models.neural_direction_raw_tcn import (
    NeuralDirectionRawTcnConfig,
    predict_neural_direction_raw_tcn_probabilities,
    save_neural_direction_raw_tcn_bundle,
    train_neural_direction_raw_tcn,
)

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class NeuralDirectionRawTcnEvalRow:
    model_type: str
    training_policy: str
    sim_size: int
    tail_offset_rounds: int
    train_size: int
    valid_size: int
    random_seed: int
    settled_history_len: int
    round_seq_len: int
    round_flow_bins: int
    kline_seq_len: int
    num_examples: int
    round_feature_dim: int
    kline_feature_dim: int
    snapshot_dim: int
    loaded_round_count: int
    total_rounds_available: int
    bundle_path: str
    valid_win_rate: float
    test_win_rate: float


@dataclass(frozen=True, slots=True)
class NeuralDirectionRawTcnEvalAggregateRow:
    model_type: str
    training_policy: str
    sim_size: int
    train_size: int
    valid_size: int
    settled_history_len: int
    round_seq_len: int
    round_flow_bins: int
    kline_seq_len: int
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
    parser.add_argument("--sim-sizes", type=str, default="6480,10800")
    parser.add_argument("--tail-offset-rounds", type=str, default="0,432,864")
    parser.add_argument("--train-sizes", type=str, default="100000,200000")
    parser.add_argument("--valid-size", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=20260402)
    parser.add_argument("--settled-history-lens", type=str, default="8,16,32")
    parser.add_argument("--round-flow-bins", type=int, default=4)
    parser.add_argument("--kline-seq-lens", type=str, default="64")
    parser.add_argument("--round-channels", type=str, default="64,64")
    parser.add_argument("--kline-channels", type=str, default="64,64")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--snapshot-hidden-sizes", type=str, default="64")
    parser.add_argument("--fusion-hidden-sizes", type=str, default="64")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience-epochs", type=int, default=5)
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
        raise InvariantError("neural_direction_raw_tcn_split_len_mismatch")
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
        raise InvariantError("neural_direction_raw_tcn_win_rate_empty")
    return float(np.mean(np.asarray(y_true, dtype=np.int64) == np.asarray(y_pred, dtype=np.int64)))


def _aggregate_rows(rows: list[NeuralDirectionRawTcnEvalRow]) -> list[NeuralDirectionRawTcnEvalAggregateRow]:
    grouped: dict[tuple[int, int, int, int, int], list[NeuralDirectionRawTcnEvalRow]] = {}
    for row in rows:
        key = (
            int(row.sim_size),
            int(row.train_size),
            int(row.settled_history_len),
            int(row.round_flow_bins),
            int(row.kline_seq_len),
        )
        grouped.setdefault(key, []).append(row)
    out: list[NeuralDirectionRawTcnEvalAggregateRow] = []
    for key in sorted(grouped):
        group = grouped[key]
        out.append(
            NeuralDirectionRawTcnEvalAggregateRow(
                model_type="raw_tcn",
                training_policy="flat",
                sim_size=int(key[0]),
                train_size=int(key[1]),
                valid_size=int(group[0].valid_size),
                settled_history_len=int(key[2]),
                round_seq_len=int(group[0].round_seq_len),
                round_flow_bins=int(key[3]),
                kline_seq_len=int(key[4]),
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
    train_sizes = parse_positive_int_list(args.train_sizes)
    settled_history_lens = parse_positive_int_list(args.settled_history_lens)
    kline_seq_lens = parse_positive_int_list(args.kline_seq_lens)
    if int(args.valid_size) <= 0:
        raise InvariantError("neural_direction_raw_tcn_valid_size_nonpositive")
    if int(args.round_flow_bins) <= 0:
        raise InvariantError("neural_direction_raw_tcn_round_flow_bins_nonpositive")

    model_cfg = NeuralDirectionRawTcnConfig(
        round_channels=_parse_channels(args.round_channels),
        kline_channels=_parse_channels(args.kline_channels),
        kernel_size=int(args.kernel_size),
        snapshot_hidden_sizes=_parse_channels(args.snapshot_hidden_sizes),
        fusion_hidden_sizes=_parse_channels(args.fusion_hidden_sizes),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        patience_epochs=int(args.patience_epochs),
    )
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[NeuralDirectionRawTcnEvalRow] = []

    max_required_examples = int(max(train_sizes)) + int(args.valid_size) + int(max(sim_sizes))
    max_settled_history_len = int(max(settled_history_lens))
    max_kline_seq_len = int(max(kline_seq_lens))

    for tail_offset_rounds in offsets:
        eval_slice = load_recent_raw_direction_eval_slice(
            config_path=str(args.config),
            required_examples=int(max_required_examples),
            tail_offset_rounds=int(tail_offset_rounds),
            settled_history_len=int(max_settled_history_len),
            round_flow_bins=int(args.round_flow_bins),
            kline_seq_len=int(max_kline_seq_len),
        )
        for settled_history_len in settled_history_lens:
            for kline_seq_len in kline_seq_lens:
                dataset = select_raw_sequence_lengths(
                    dataset=eval_slice.dataset,
                    settled_history_len=int(settled_history_len),
                    kline_seq_len=int(kline_seq_len),
                )
                for train_size in train_sizes:
                    for sim_size in sim_sizes:
                        train_epochs, valid_epochs, test_epochs = _split_target_epochs(
                            target_epochs=dataset.target_epochs,
                            train_size=int(train_size),
                            valid_size=int(args.valid_size),
                            sim_size=int(sim_size),
                        )
                        bundle = train_neural_direction_raw_tcn(
                            dataset=dataset,
                            train_target_epochs=train_epochs,
                            valid_target_epochs=valid_epochs,
                            random_seed=int(args.random_seed),
                            config=model_cfg,
                        )
                        bundle_path = (
                            output_dir
                            / (
                                f"{str(args.name_prefix)}_t{int(train_size)}_rh{int(settled_history_len)}"
                                f"_k{int(kline_seq_len)}_tail{int(sim_size)}_off{int(tail_offset_rounds):05d}"
                                "_neural_direction_raw_tcn.pt"
                            )
                        ).resolve()
                        save_neural_direction_raw_tcn_bundle(bundle=bundle, path=str(bundle_path))
                        test_round_x, test_kline_x, test_snapshot_x, test_y = build_raw_sequence_examples_for_target_epochs(
                            dataset=dataset,
                            target_epochs=test_epochs,
                        )
                        probs = predict_neural_direction_raw_tcn_probabilities(
                            bundle=bundle,
                            round_sequence=test_round_x,
                            kline_sequence=test_kline_x,
                            snapshot_matrix=test_snapshot_x,
                        )
                        preds = (np.asarray(probs, dtype=np.float32) >= 0.5).astype(np.int64)
                        rows.append(
                            NeuralDirectionRawTcnEvalRow(
                                model_type="raw_tcn",
                                training_policy="flat",
                                sim_size=int(sim_size),
                                tail_offset_rounds=int(tail_offset_rounds),
                                train_size=int(train_size),
                                valid_size=int(args.valid_size),
                                random_seed=int(args.random_seed),
                                settled_history_len=int(settled_history_len),
                                round_seq_len=int(dataset.round_sequence.shape[1]),
                                round_flow_bins=int(args.round_flow_bins),
                                kline_seq_len=int(kline_seq_len),
                                num_examples=int(dataset.num_examples),
                                round_feature_dim=int(dataset.round_sequence.shape[2]),
                                kline_feature_dim=int(dataset.kline_sequence.shape[2]),
                                snapshot_dim=int(dataset.snapshot_matrix.shape[1]),
                                loaded_round_count=int(eval_slice.loaded_round_count),
                                total_rounds_available=int(eval_slice.total_rounds_available),
                                bundle_path=str(bundle_path),
                                valid_win_rate=float(bundle.metadata["best_valid_win_rate"]),
                                test_win_rate=float(_win_rate(test_y, preds)),
                            )
                        )

    if not rows:
        raise InvariantError("neural_direction_raw_tcn_rows_empty")
    aggregates = _aggregate_rows(rows)
    rows_out = rows_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_raw_tcn_eval_rows",
    )
    summary_out = summary_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_raw_tcn_eval_summary",
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
        "train_sizes": [int(value) for value in train_sizes],
        "settled_history_lens": [int(value) for value in settled_history_lens],
        "kline_seq_lens": [int(value) for value in kline_seq_lens],
        "round_flow_bins": int(args.round_flow_bins),
        "row_count": int(len(rows)),
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
