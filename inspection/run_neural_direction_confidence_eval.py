from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from inspection.neural_direction_eval_common import (
    load_recent_direction_eval_slice,
    rows_path,
    summary_path,
)
from inspection.neural_direction_raw_eval_common import load_recent_raw_direction_eval_slice
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_confidence import (
    apply_temperature_calibrator_to_probs,
    chosen_side_confidence,
    fit_temperature_calibrator_from_probs,
    summarize_confidence_buckets,
)
from pancakebot.domain.models.direction_tree_model import (
    load_direction_tree_bundle,
    predict_direction_tree_probabilities,
)
from pancakebot.domain.models.neural_direction_mlp import (
    load_neural_direction_mlp_bundle,
    predict_neural_direction_probabilities,
)
from pancakebot.domain.models.neural_direction_dataset import (
    select_feature_columns_exact,
)
from pancakebot.domain.models.neural_direction_raw_sequence_dataset import (
    build_raw_sequence_examples_for_target_epochs,
    select_raw_sequence_lengths,
)
from pancakebot.domain.models.neural_direction_raw_tcn import (
    load_neural_direction_raw_tcn_bundle,
    predict_neural_direction_raw_tcn_probabilities,
)
from pancakebot.domain.models.neural_direction_tcn import (
    build_sequence_examples_for_target_epochs,
    load_neural_direction_tcn_bundle,
    predict_neural_direction_tcn_probabilities,
)

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class NeuralDirectionConfidenceEvalRow:
    model_type: str
    source_rows_csv: str
    source_bundle_path: str
    sim_size: int
    tail_offset_rounds: int
    training_policy: str
    train_size: int
    pretrain_size: int
    valid_size: int
    recency_half_life_examples: float | None
    seq_len: int | None
    settled_history_len: int | None
    round_flow_bins: int | None
    kline_seq_len: int | None
    coverage_fraction_requested: float
    selected_count: int
    selected_fraction_actual: float
    selected_win_rate: float
    selected_mean_confidence: float
    selected_min_confidence: float
    selected_max_confidence: float
    overall_test_win_rate: float
    overall_test_mean_confidence: float
    calibration_temperature: float
    calibration_valid_loss_before: float
    calibration_valid_loss_after: float


@dataclass(frozen=True, slots=True)
class NeuralDirectionConfidenceAggregateRow:
    model_type: str
    sim_size: int
    training_policy: str
    train_size: int
    pretrain_size: int
    recency_half_life_examples: float | None
    seq_len: int | None
    settled_history_len: int | None
    round_flow_bins: int | None
    kline_seq_len: int | None
    coverage_fraction_requested: float
    num_offsets: int
    mean_selected_win_rate: float
    min_selected_win_rate: float
    max_selected_win_rate: float
    mean_selected_min_confidence: float
    mean_selected_mean_confidence: float
    mean_overall_test_win_rate: float


@dataclass(frozen=True, slots=True)
class _SourceEvalJob:
    model_type: str
    source_rows_csv: str
    source_bundle_path: str
    sim_size: int
    tail_offset_rounds: int
    training_policy: str
    train_size: int
    pretrain_size: int
    valid_size: int
    recency_half_life_examples: float | None
    seq_len: int | None
    settled_history_len: int | None
    round_flow_bins: int | None
    kline_seq_len: int | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--model-type", type=str, choices=("mlp", "tcn", "raw_tcn", "lightgbm", "catboost"), required=True)
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--rows-csvs", type=str, required=True)
    parser.add_argument("--coverage-fractions", type=str, default="1.0,0.75,0.5,0.25,0.10,0.05,0.02,0.01")
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _parse_rows_csvs(raw: str) -> tuple[Path, ...]:
    out: list[Path] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        out.append(Path(text).resolve())
    if not out:
        raise InvariantError("neural_direction_confidence_rows_csvs_empty")
    return tuple(out)


def _parse_coverage_fractions(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = float(text)
        if not (0.0 < float(value) <= 1.0):
            raise InvariantError("neural_direction_confidence_fraction_invalid")
        out.append(float(value))
    if not out:
        raise InvariantError("neural_direction_confidence_fraction_empty")
    return tuple(out)


def _load_source_jobs(*, rows_csvs: tuple[Path, ...], model_type: str) -> list[_SourceEvalJob]:
    out: list[_SourceEvalJob] = []
    for rows_csv in rows_csvs:
        with rows_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for source_row in reader:
                seq_len: int | None = None
                if str(model_type) == "tcn":
                    seq_len = int(source_row["seq_len"])
                settled_history_len = None
                round_flow_bins = None
                kline_seq_len = None
                if str(model_type) == "raw_tcn":
                    settled_history_len = int(source_row["settled_history_len"])
                    round_flow_bins = int(source_row["round_flow_bins"])
                    kline_seq_len = int(source_row["kline_seq_len"])
                out.append(
                    _SourceEvalJob(
                        model_type=str(model_type),
                        source_rows_csv=str(rows_csv),
                        source_bundle_path=str(source_row["bundle_path"]),
                        sim_size=int(source_row["sim_size"]),
                        tail_offset_rounds=int(source_row["tail_offset_rounds"]),
                        training_policy=str(source_row.get("training_policy", "flat")),
                        train_size=int(source_row["train_size"]),
                        pretrain_size=int(source_row.get("pretrain_size", 0)),
                        valid_size=int(source_row["valid_size"]),
                        recency_half_life_examples=(
                            None
                            if source_row.get("recency_half_life_examples") in (None, "", "None")
                            else float(source_row["recency_half_life_examples"])
                        ),
                        seq_len=None if seq_len is None else int(seq_len),
                        settled_history_len=None if settled_history_len is None else int(settled_history_len),
                        round_flow_bins=None if round_flow_bins is None else int(round_flow_bins),
                        kline_seq_len=None if kline_seq_len is None else int(kline_seq_len),
                    )
                )
    if not out:
        raise InvariantError("neural_direction_confidence_rows_empty")
    return out


def _split_target_epochs(
    *,
    target_epochs: tuple[int, ...],
    train_size: int,
    pretrain_size: int,
    valid_size: int,
    sim_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    needed = int(pretrain_size) + int(train_size) + int(valid_size) + int(sim_size)
    if len(target_epochs) < int(needed):
        raise InvariantError("neural_direction_confidence_split_len_mismatch")
    target_tail = tuple(int(epoch) for epoch in target_epochs[-int(needed) :])
    train_start = int(pretrain_size)
    train_end = int(pretrain_size) + int(train_size)
    train_epochs = tuple(int(epoch) for epoch in target_tail[train_start:train_end])
    valid_epochs = tuple(
        int(epoch)
        for epoch in target_tail[train_end : train_end + int(valid_size)]
    )
    test_epochs = tuple(int(epoch) for epoch in target_tail[-int(sim_size) :])
    return train_epochs, valid_epochs, test_epochs


def _mlp_probs_for_epochs(*, bundle_path: str, eval_slice, valid_epochs, test_epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle = load_neural_direction_mlp_bundle(str(bundle_path))
    dataset = select_feature_columns_exact(
        dataset=eval_slice.dataset,
        feature_columns=tuple(bundle.feature_columns),
    )
    probs_all = predict_neural_direction_probabilities(
        bundle=bundle,
        feature_matrix=np.asarray(dataset.feature_matrix, dtype=np.float32),
    )
    index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
    valid_idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in valid_epochs], dtype=np.int64)
    test_idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in test_epochs], dtype=np.int64)
    valid_probs = np.asarray(probs_all[valid_idx], dtype=np.float32)
    test_probs = np.asarray(probs_all[test_idx], dtype=np.float32)
    valid_y = np.asarray(dataset.labels[valid_idx], dtype=np.int64)
    test_y = np.asarray(dataset.labels[test_idx], dtype=np.int64)
    return valid_probs, valid_y, test_probs, test_y


def _tcn_probs_for_epochs(*, bundle_path: str, eval_slice, seq_len: int, valid_epochs, test_epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle = load_neural_direction_tcn_bundle(str(bundle_path))
    dataset = select_feature_columns_exact(
        dataset=eval_slice.dataset,
        feature_columns=tuple(bundle.feature_columns),
    )
    valid_x, valid_y = build_sequence_examples_for_target_epochs(
        dataset=dataset,
        target_epochs=valid_epochs,
        seq_len=int(seq_len),
    )
    test_x, test_y = build_sequence_examples_for_target_epochs(
        dataset=dataset,
        target_epochs=test_epochs,
        seq_len=int(seq_len),
    )
    valid_probs = predict_neural_direction_tcn_probabilities(
        bundle=bundle,
        feature_sequences=valid_x,
    )
    test_probs = predict_neural_direction_tcn_probabilities(
        bundle=bundle,
        feature_sequences=test_x,
    )
    return (
        np.asarray(valid_probs, dtype=np.float32),
        np.asarray(valid_y, dtype=np.int64),
        np.asarray(test_probs, dtype=np.float32),
        np.asarray(test_y, dtype=np.int64),
    )


def _tree_probs_for_epochs(*, bundle_path: str, eval_slice, valid_epochs, test_epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle = load_direction_tree_bundle(str(bundle_path))
    dataset = select_feature_columns_exact(
        dataset=eval_slice.dataset,
        feature_columns=tuple(bundle.feature_columns),
    )
    probs_all = predict_direction_tree_probabilities(
        bundle=bundle,
        feature_matrix=np.asarray(dataset.feature_matrix, dtype=np.float32),
    )
    index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
    valid_idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in valid_epochs], dtype=np.int64)
    test_idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in test_epochs], dtype=np.int64)
    valid_probs = np.asarray(probs_all[valid_idx], dtype=np.float32)
    test_probs = np.asarray(probs_all[test_idx], dtype=np.float32)
    valid_y = np.asarray(dataset.labels[valid_idx], dtype=np.int64)
    test_y = np.asarray(dataset.labels[test_idx], dtype=np.int64)
    return valid_probs, valid_y, test_probs, test_y


def _raw_tcn_probs_for_epochs(
    *,
    bundle_path: str,
    eval_slice,
    settled_history_len: int,
    kline_seq_len: int,
    valid_epochs,
    test_epochs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = select_raw_sequence_lengths(
        dataset=eval_slice.dataset,
        settled_history_len=int(settled_history_len),
        kline_seq_len=int(kline_seq_len),
    )
    bundle = load_neural_direction_raw_tcn_bundle(str(bundle_path))
    valid_round_x, valid_kline_x, valid_snapshot_x, valid_y = build_raw_sequence_examples_for_target_epochs(
        dataset=dataset,
        target_epochs=valid_epochs,
    )
    test_round_x, test_kline_x, test_snapshot_x, test_y = build_raw_sequence_examples_for_target_epochs(
        dataset=dataset,
        target_epochs=test_epochs,
    )
    valid_probs = predict_neural_direction_raw_tcn_probabilities(
        bundle=bundle,
        round_sequence=valid_round_x,
        kline_sequence=valid_kline_x,
        snapshot_matrix=valid_snapshot_x,
    )
    test_probs = predict_neural_direction_raw_tcn_probabilities(
        bundle=bundle,
        round_sequence=test_round_x,
        kline_sequence=test_kline_x,
        snapshot_matrix=test_snapshot_x,
    )
    return (
        np.asarray(valid_probs, dtype=np.float32),
        np.asarray(valid_y, dtype=np.int64),
        np.asarray(test_probs, dtype=np.float32),
        np.asarray(test_y, dtype=np.int64),
    )


def _aggregate_rows(rows: list[NeuralDirectionConfidenceEvalRow]) -> list[NeuralDirectionConfidenceAggregateRow]:
    grouped: dict[
        tuple[str, int, str, int, int, float | None, int | None, int | None, int | None, int | None, float],
        list[NeuralDirectionConfidenceEvalRow],
    ] = {}
    for row in rows:
        key = (
            str(row.model_type),
            int(row.sim_size),
            str(row.training_policy),
            int(row.train_size),
            int(row.pretrain_size),
            None
            if row.recency_half_life_examples is None
            else float(row.recency_half_life_examples),
            None if row.seq_len is None else int(row.seq_len),
            None if row.settled_history_len is None else int(row.settled_history_len),
            None if row.round_flow_bins is None else int(row.round_flow_bins),
            None if row.kline_seq_len is None else int(row.kline_seq_len),
            float(row.coverage_fraction_requested),
        )
        grouped.setdefault(key, []).append(row)
    out: list[NeuralDirectionConfidenceAggregateRow] = []
    for key in sorted(
        grouped,
        key=lambda item: (
            str(item[0]),
            int(item[1]),
            str(item[2]),
            int(item[3]),
            int(item[4]),
            -1.0 if item[5] is None else float(item[5]),
            -1 if item[6] is None else int(item[6]),
            -1 if item[7] is None else int(item[7]),
            -1 if item[8] is None else int(item[8]),
            -1 if item[9] is None else int(item[9]),
            -float(item[10]),
        ),
    ):
        group = grouped[key]
        out.append(
            NeuralDirectionConfidenceAggregateRow(
                model_type=str(key[0]),
                sim_size=int(key[1]),
                training_policy=str(key[2]),
                train_size=int(key[3]),
                pretrain_size=int(key[4]),
                recency_half_life_examples=None if key[5] is None else float(key[5]),
                seq_len=None if key[6] is None else int(key[6]),
                settled_history_len=None if key[7] is None else int(key[7]),
                round_flow_bins=None if key[8] is None else int(key[8]),
                kline_seq_len=None if key[9] is None else int(key[9]),
                coverage_fraction_requested=float(key[10]),
                num_offsets=int(len(group)),
                mean_selected_win_rate=float(np.mean([row.selected_win_rate for row in group])),
                min_selected_win_rate=float(min(row.selected_win_rate for row in group)),
                max_selected_win_rate=float(max(row.selected_win_rate for row in group)),
                mean_selected_min_confidence=float(np.mean([row.selected_min_confidence for row in group])),
                mean_selected_mean_confidence=float(np.mean([row.selected_mean_confidence for row in group])),
                mean_overall_test_win_rate=float(np.mean([row.overall_test_win_rate for row in group])),
            )
        )
    return out


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = str(args.model_type)
    rows_csvs = _parse_rows_csvs(args.rows_csvs)
    coverage_fractions = _parse_coverage_fractions(args.coverage_fractions)
    source_jobs = _load_source_jobs(rows_csvs=rows_csvs, model_type=str(model_type))

    rows_out: list[NeuralDirectionConfidenceEvalRow] = []
    max_required_examples_by_offset: dict[int, int] = {}
    max_raw_params_by_offset: dict[int, tuple[int, int, int]] = {}
    for job in source_jobs:
        required_examples = int(job.train_size) + int(job.valid_size) + int(job.sim_size)
        if job.seq_len is not None:
            required_examples += int(job.seq_len) - 1
        prev = max_required_examples_by_offset.get(int(job.tail_offset_rounds))
        if prev is None or int(required_examples) > int(prev):
            max_required_examples_by_offset[int(job.tail_offset_rounds)] = int(required_examples)
        if str(model_type) == "raw_tcn":
            prev_raw = max_raw_params_by_offset.get(int(job.tail_offset_rounds))
            candidate = (
                int(job.settled_history_len or 0),
                int(job.round_flow_bins or 0),
                int(job.kline_seq_len or 0),
            )
            if prev_raw is None or candidate > prev_raw:
                max_raw_params_by_offset[int(job.tail_offset_rounds)] = candidate

    if str(model_type) == "raw_tcn":
        eval_slices_by_offset = {
            int(tail_offset_rounds): load_recent_raw_direction_eval_slice(
                config_path=str(args.config),
                required_examples=int(required_examples),
                tail_offset_rounds=int(tail_offset_rounds),
                settled_history_len=int(max_raw_params_by_offset[int(tail_offset_rounds)][0]),
                round_flow_bins=int(max_raw_params_by_offset[int(tail_offset_rounds)][1]),
                kline_seq_len=int(max_raw_params_by_offset[int(tail_offset_rounds)][2]),
            )
            for tail_offset_rounds, required_examples in sorted(max_required_examples_by_offset.items())
        }
    else:
        eval_slices_by_offset = {
            int(tail_offset_rounds): load_recent_direction_eval_slice(
                config_path=str(args.config),
                required_examples=int(required_examples),
                tail_offset_rounds=int(tail_offset_rounds),
            )
            for tail_offset_rounds, required_examples in sorted(max_required_examples_by_offset.items())
        }

    for idx, job in enumerate(source_jobs, start=1):
        eval_slice = eval_slices_by_offset[int(job.tail_offset_rounds)]
        _, valid_epochs, test_epochs = _split_target_epochs(
            target_epochs=eval_slice.dataset.target_epochs,
            train_size=int(job.train_size),
            pretrain_size=int(job.pretrain_size),
            valid_size=int(job.valid_size),
            sim_size=int(job.sim_size),
        )
        if str(model_type) == "mlp":
            valid_probs, valid_y, test_probs, test_y = _mlp_probs_for_epochs(
                bundle_path=str(job.source_bundle_path),
                eval_slice=eval_slice,
                valid_epochs=valid_epochs,
                test_epochs=test_epochs,
            )
        elif str(model_type) == "tcn":
            valid_probs, valid_y, test_probs, test_y = _tcn_probs_for_epochs(
                bundle_path=str(job.source_bundle_path),
                eval_slice=eval_slice,
                seq_len=int(job.seq_len),
                valid_epochs=valid_epochs,
                test_epochs=test_epochs,
            )
        elif str(model_type) in ("lightgbm", "catboost"):
            valid_probs, valid_y, test_probs, test_y = _tree_probs_for_epochs(
                bundle_path=str(job.source_bundle_path),
                eval_slice=eval_slice,
                valid_epochs=valid_epochs,
                test_epochs=test_epochs,
            )
        else:
            valid_probs, valid_y, test_probs, test_y = _raw_tcn_probs_for_epochs(
                bundle_path=str(job.source_bundle_path),
                eval_slice=eval_slice,
                settled_history_len=int(job.settled_history_len),
                kline_seq_len=int(job.kline_seq_len),
                valid_epochs=valid_epochs,
                test_epochs=test_epochs,
            )

        calibrator = fit_temperature_calibrator_from_probs(
            bull_probs=valid_probs,
            labels=valid_y,
        )
        calibrated_test_probs = apply_temperature_calibrator_to_probs(
            bull_probs=test_probs,
            calibrator=calibrator,
        )
        test_pred = (np.asarray(test_probs, dtype=np.float32) >= 0.5).astype(np.int64)
        test_confidence = chosen_side_confidence(
            predicted_labels=test_pred,
            calibrated_bull_probs=calibrated_test_probs,
        )
        overall_test_win_rate = float(np.mean(test_pred == test_y))
        overall_test_mean_confidence = float(np.mean(test_confidence))
        buckets = summarize_confidence_buckets(
            labels=test_y,
            predicted_labels=test_pred,
            confidence=test_confidence,
            coverage_fractions=coverage_fractions,
        )
        for bucket in buckets:
            rows_out.append(
                NeuralDirectionConfidenceEvalRow(
                    model_type=str(model_type),
                    source_rows_csv=str(job.source_rows_csv),
                    source_bundle_path=str(job.source_bundle_path),
                    sim_size=int(job.sim_size),
                    tail_offset_rounds=int(job.tail_offset_rounds),
                    training_policy=str(job.training_policy),
                    train_size=int(job.train_size),
                    pretrain_size=int(job.pretrain_size),
                    valid_size=int(job.valid_size),
                    recency_half_life_examples=None
                    if job.recency_half_life_examples is None
                    else float(job.recency_half_life_examples),
                    seq_len=None if job.seq_len is None else int(job.seq_len),
                    settled_history_len=None
                    if job.settled_history_len is None
                    else int(job.settled_history_len),
                    round_flow_bins=None
                    if job.round_flow_bins is None
                    else int(job.round_flow_bins),
                    kline_seq_len=None
                    if job.kline_seq_len is None
                    else int(job.kline_seq_len),
                    coverage_fraction_requested=float(bucket.coverage_fraction_requested),
                    selected_count=int(bucket.selected_count),
                    selected_fraction_actual=float(bucket.selected_fraction_actual),
                    selected_win_rate=float(bucket.selected_win_rate),
                    selected_mean_confidence=float(bucket.selected_mean_confidence),
                    selected_min_confidence=float(bucket.selected_min_confidence),
                    selected_max_confidence=float(bucket.selected_max_confidence),
                    overall_test_win_rate=float(overall_test_win_rate),
                    overall_test_mean_confidence=float(overall_test_mean_confidence),
                    calibration_temperature=float(calibrator.temperature),
                    calibration_valid_loss_before=float(calibrator.valid_loss_before),
                    calibration_valid_loss_after=float(calibrator.valid_loss_after),
                )
            )
        print(
            {
                "phase": "job_done",
                "model_type": str(model_type),
                "job_index": int(idx),
                "job_count": int(len(source_jobs)),
                "training_policy": str(job.training_policy),
                "train_size": int(job.train_size),
                "pretrain_size": int(job.pretrain_size),
                "sim_size": int(job.sim_size),
                "tail_offset_rounds": int(job.tail_offset_rounds),
                "seq_len": None if job.seq_len is None else int(job.seq_len),
            },
            flush=True,
        )

    if not rows_out:
        raise InvariantError("neural_direction_confidence_rows_empty")

    aggregates = _aggregate_rows(rows_out)
    rows_out_path = rows_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_confidence_rows",
    )
    summary_out_path = summary_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_confidence_summary",
    )
    with rows_out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows_out[0]).keys()))
        writer.writeheader()
        for row in rows_out:
            writer.writerow(asdict(row))

    summary_payload = {
        "model_type": str(model_type),
        "rows_csvs": [str(path) for path in rows_csvs],
        "coverage_fractions": [float(v) for v in coverage_fractions],
        "rows_csv_path": str(rows_out_path),
        "aggregates": [asdict(row) for row in aggregates],
        "row_count": int(len(rows_out)),
    }
    summary_out_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
