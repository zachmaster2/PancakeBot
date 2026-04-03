from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from inspection.neural_direction_eval_common import (
    load_recent_direction_eval_slice,
    rows_path,
    summary_path,
)
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.direction_tree_model import (
    load_direction_tree_bundle,
    predict_direction_tree_probabilities,
)
from pancakebot.domain.models.neural_direction_confidence import (
    apply_temperature_calibrator_to_probs,
    chosen_side_confidence,
    fit_temperature_calibrator_from_probs,
)
from pancakebot.domain.models.neural_direction_dataset import (
    select_feature_columns_exact,
)
from pancakebot.domain.models.neural_direction_mlp import (
    load_neural_direction_mlp_bundle,
    predict_neural_direction_probabilities,
)
from pancakebot.domain.models.neural_direction_tcn import (
    build_sequence_examples_for_target_epochs,
    load_neural_direction_tcn_bundle,
    predict_neural_direction_tcn_probabilities,
)

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


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
    seq_len: int | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--rows-csvs", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _parse_rows_csvs(raw: str) -> tuple[Path, ...]:
    out: list[Path] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text != "":
            out.append(Path(text).resolve())
    if not out:
        raise InvariantError("direction_ensemble_rows_csvs_empty")
    return tuple(out)


def _load_source_jobs(*, rows_csvs: tuple[Path, ...]) -> list[_SourceEvalJob]:
    out: list[_SourceEvalJob] = []
    for rows_csv in rows_csvs:
        with rows_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for source_row in reader:
                model_type = str(source_row.get("model_type") or "")
                seq_len_raw = source_row.get("seq_len")
                seq_len = None if seq_len_raw in (None, "") else int(seq_len_raw)
                if str(model_type) == "":
                    model_type = "tcn" if seq_len is not None else "mlp"
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
                        seq_len=seq_len,
                    )
                )
    if not out:
        raise InvariantError("direction_ensemble_source_jobs_empty")
    return out


def _split_target_epochs(
    *,
    target_epochs: tuple[int, ...],
    train_size: int,
    pretrain_size: int,
    valid_size: int,
    sim_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    needed = int(pretrain_size) + int(train_size) + int(valid_size) + int(sim_size)
    if len(target_epochs) < int(needed):
        raise InvariantError("direction_ensemble_split_len_mismatch")
    target_tail = tuple(int(epoch) for epoch in target_epochs[-int(needed):])
    valid_start = int(pretrain_size) + int(train_size)
    valid_end = int(valid_start) + int(valid_size)
    valid_epochs = tuple(int(epoch) for epoch in target_tail[valid_start:valid_end])
    test_epochs = tuple(int(epoch) for epoch in target_tail[-int(sim_size):])
    return valid_epochs, test_epochs


def _mlp_probs_for_epochs(*, bundle_path: str, eval_slice, valid_epochs, test_epochs):
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
    return dataset, valid_probs, valid_y, test_probs, test_y


def _tcn_probs_for_epochs(*, bundle_path: str, eval_slice, seq_len: int, valid_epochs, test_epochs):
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
    valid_probs = predict_neural_direction_tcn_probabilities(bundle=bundle, feature_sequences=valid_x)
    test_probs = predict_neural_direction_tcn_probabilities(bundle=bundle, feature_sequences=test_x)
    return dataset, np.asarray(valid_probs, dtype=np.float32), np.asarray(valid_y, dtype=np.int64), np.asarray(test_probs, dtype=np.float32), np.asarray(test_y, dtype=np.int64)


def _tree_probs_for_epochs(*, bundle_path: str, eval_slice, valid_epochs, test_epochs):
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
    return dataset, valid_probs, valid_y, test_probs, test_y


def _meta_features(*, prob_by_model: dict[str, np.ndarray], model_order: list[str]) -> np.ndarray:
    mats = [np.asarray(prob_by_model[name], dtype=np.float32) for name in model_order]
    base = np.column_stack(mats)
    mean_col = np.mean(base, axis=1, keepdims=True)
    std_col = np.std(base, axis=1, keepdims=True)
    range_col = (np.max(base, axis=1, keepdims=True) - np.min(base, axis=1, keepdims=True)).astype(np.float32)
    return np.asarray(np.concatenate([base, mean_col, std_col, range_col], axis=1), dtype=np.float32)


def _win_rate(y_true: np.ndarray, bull_probs: np.ndarray) -> float:
    pred = (np.asarray(bull_probs, dtype=np.float32) >= 0.5).astype(np.int64)
    return float(np.mean(pred == np.asarray(y_true, dtype=np.int64)))


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_jobs = _load_source_jobs(rows_csvs=_parse_rows_csvs(args.rows_csvs))

    jobs_by_group: dict[tuple[int, int], list[_SourceEvalJob]] = {}
    for job in source_jobs:
        jobs_by_group.setdefault((int(job.sim_size), int(job.tail_offset_rounds)), []).append(job)

    round_rows: list[dict[str, object]] = []
    comparison_groups: list[dict[str, object]] = []

    for (sim_size, tail_offset_rounds), jobs in sorted(jobs_by_group.items()):
        required_examples = 0
        for job in jobs:
            needed = int(job.pretrain_size) + int(job.train_size) + int(job.valid_size) + int(job.sim_size)
            if job.seq_len is not None:
                needed += int(job.seq_len) - 1
            required_examples = max(int(required_examples), int(needed))
        eval_slice = load_recent_direction_eval_slice(
            config_path=str(args.config),
            required_examples=int(required_examples),
            tail_offset_rounds=int(tail_offset_rounds),
        )
        valid_epochs_ref: tuple[int, ...] | None = None
        test_epochs_ref: tuple[int, ...] | None = None
        valid_y_ref: np.ndarray | None = None
        test_y_ref: np.ndarray | None = None
        calibrated_valid_by_model: dict[str, np.ndarray] = {}
        calibrated_test_by_model: dict[str, np.ndarray] = {}
        summary_models: dict[str, dict[str, object]] = {}

        for job in sorted(jobs, key=lambda row: str(row.model_type)):
            valid_epochs, test_epochs = _split_target_epochs(
                target_epochs=eval_slice.dataset.target_epochs,
                train_size=int(job.train_size),
                pretrain_size=int(job.pretrain_size),
                valid_size=int(job.valid_size),
                sim_size=int(job.sim_size),
            )
            if str(job.model_type) == "mlp":
                _, valid_probs, valid_y, test_probs, test_y = _mlp_probs_for_epochs(
                    bundle_path=str(job.source_bundle_path),
                    eval_slice=eval_slice,
                    valid_epochs=valid_epochs,
                    test_epochs=test_epochs,
                )
            elif str(job.model_type) == "tcn":
                _, valid_probs, valid_y, test_probs, test_y = _tcn_probs_for_epochs(
                    bundle_path=str(job.source_bundle_path),
                    eval_slice=eval_slice,
                    seq_len=int(job.seq_len),
                    valid_epochs=valid_epochs,
                    test_epochs=test_epochs,
                )
            elif str(job.model_type) in ("lightgbm", "catboost"):
                _, valid_probs, valid_y, test_probs, test_y = _tree_probs_for_epochs(
                    bundle_path=str(job.source_bundle_path),
                    eval_slice=eval_slice,
                    valid_epochs=valid_epochs,
                    test_epochs=test_epochs,
                )
            else:
                raise InvariantError("direction_ensemble_model_type_unsupported")

            if valid_epochs_ref is None:
                valid_epochs_ref = tuple(int(epoch) for epoch in valid_epochs)
                test_epochs_ref = tuple(int(epoch) for epoch in test_epochs)
                valid_y_ref = np.asarray(valid_y, dtype=np.int64)
                test_y_ref = np.asarray(test_y, dtype=np.int64)
            else:
                if tuple(int(epoch) for epoch in valid_epochs) != valid_epochs_ref:
                    raise InvariantError("direction_ensemble_valid_epochs_misaligned")
                if tuple(int(epoch) for epoch in test_epochs) != test_epochs_ref:
                    raise InvariantError("direction_ensemble_test_epochs_misaligned")
                if not np.array_equal(np.asarray(valid_y, dtype=np.int64), valid_y_ref):
                    raise InvariantError("direction_ensemble_valid_labels_misaligned")
                if not np.array_equal(np.asarray(test_y, dtype=np.int64), test_y_ref):
                    raise InvariantError("direction_ensemble_test_labels_misaligned")

            calibrator = fit_temperature_calibrator_from_probs(
                bull_probs=valid_probs,
                labels=valid_y,
            )
            calibrated_valid = apply_temperature_calibrator_to_probs(
                bull_probs=valid_probs,
                calibrator=calibrator,
            )
            calibrated_test = apply_temperature_calibrator_to_probs(
                bull_probs=test_probs,
                calibrator=calibrator,
            )
            calibrated_valid_by_model[str(job.model_type)] = np.asarray(calibrated_valid, dtype=np.float32)
            calibrated_test_by_model[str(job.model_type)] = np.asarray(calibrated_test, dtype=np.float32)
            summary_models[str(job.model_type)] = {
                "source_rows_csv": str(job.source_rows_csv),
                "source_bundle_path": str(job.source_bundle_path),
                "train_size": int(job.train_size),
                "pretrain_size": int(job.pretrain_size),
                "valid_size": int(job.valid_size),
                "seq_len": None if job.seq_len is None else int(job.seq_len),
                "valid_win_rate": float(_win_rate(valid_y, calibrated_valid)),
                "test_win_rate": float(_win_rate(test_y, calibrated_test)),
                "mean_valid_confidence": float(np.mean(chosen_side_confidence(
                    predicted_labels=(np.asarray(calibrated_valid, dtype=np.float32) >= 0.5).astype(np.int64),
                    calibrated_bull_probs=np.asarray(calibrated_valid, dtype=np.float32),
                ))),
                "mean_test_confidence": float(np.mean(chosen_side_confidence(
                    predicted_labels=(np.asarray(calibrated_test, dtype=np.float32) >= 0.5).astype(np.int64),
                    calibrated_bull_probs=np.asarray(calibrated_test, dtype=np.float32),
                ))),
                "calibration_temperature": float(calibrator.temperature),
            }

        if valid_epochs_ref is None or test_epochs_ref is None or valid_y_ref is None or test_y_ref is None:
            raise InvariantError("direction_ensemble_group_empty")
        model_order = sorted(calibrated_valid_by_model)
        soft_valid = np.mean(np.column_stack([calibrated_valid_by_model[name] for name in model_order]), axis=1).astype(np.float32)
        soft_test = np.mean(np.column_stack([calibrated_test_by_model[name] for name in model_order]), axis=1).astype(np.float32)
        meta_valid_x = _meta_features(prob_by_model=calibrated_valid_by_model, model_order=model_order)
        meta_test_x = _meta_features(prob_by_model=calibrated_test_by_model, model_order=model_order)
        stacker = LogisticRegression(max_iter=2000, random_state=0)
        stacker.fit(meta_valid_x, valid_y_ref)
        stacked_valid = np.asarray(stacker.predict_proba(meta_valid_x)[:, 1], dtype=np.float32)
        stacked_test = np.asarray(stacker.predict_proba(meta_test_x)[:, 1], dtype=np.float32)

        for split_name, epochs, labels, prob_by_model, soft_probs, stacked_probs in (
            ("valid", valid_epochs_ref, valid_y_ref, calibrated_valid_by_model, soft_valid, stacked_valid),
            ("test", test_epochs_ref, test_y_ref, calibrated_test_by_model, soft_test, stacked_test),
        ):
            base_matrix = np.column_stack([prob_by_model[name] for name in model_order])
            for idx, epoch in enumerate(epochs):
                row: dict[str, object] = {
                    "sim_size": int(sim_size),
                    "tail_offset_rounds": int(tail_offset_rounds),
                    "split": str(split_name),
                    "target_epoch": int(epoch),
                    "label": int(labels[idx]),
                    "bull_vote_count": int(np.sum(base_matrix[idx] >= 0.5)),
                    "bear_vote_count": int(np.sum(base_matrix[idx] < 0.5)),
                    "bull_prob_mean": float(np.mean(base_matrix[idx])),
                    "bull_prob_std": float(np.std(base_matrix[idx])),
                    "bull_prob_min": float(np.min(base_matrix[idx])),
                    "bull_prob_max": float(np.max(base_matrix[idx])),
                    "p_bull_soft": float(soft_probs[idx]),
                    "pred_soft": int(float(soft_probs[idx]) >= 0.5),
                    "conf_soft": float(chosen_side_confidence(
                        predicted_labels=np.asarray([int(float(soft_probs[idx]) >= 0.5)], dtype=np.int64),
                        calibrated_bull_probs=np.asarray([float(soft_probs[idx])], dtype=np.float32),
                    )[0]),
                    "p_bull_stacked": float(stacked_probs[idx]),
                    "pred_stacked": int(float(stacked_probs[idx]) >= 0.5),
                    "conf_stacked": float(chosen_side_confidence(
                        predicted_labels=np.asarray([int(float(stacked_probs[idx]) >= 0.5)], dtype=np.int64),
                        calibrated_bull_probs=np.asarray([float(stacked_probs[idx])], dtype=np.float32),
                    )[0]),
                }
                for model_name in model_order:
                    p = float(prob_by_model[model_name][idx])
                    row[f"p_bull_{model_name}"] = p
                    row[f"pred_{model_name}"] = int(p >= 0.5)
                    row[f"conf_{model_name}"] = float(max(p, 1.0 - p))
                round_rows.append(row)

        comparison_groups.append(
            {
                "sim_size": int(sim_size),
                "tail_offset_rounds": int(tail_offset_rounds),
                "valid_size": int(len(valid_epochs_ref)),
                "test_size": int(len(test_epochs_ref)),
                "model_order": list(model_order),
                "models": summary_models,
                "soft_ensemble": {
                    "valid_win_rate": float(_win_rate(valid_y_ref, soft_valid)),
                    "test_win_rate": float(_win_rate(test_y_ref, soft_test)),
                },
                "stacked_ensemble": {
                    "valid_win_rate": float(_win_rate(valid_y_ref, stacked_valid)),
                    "test_win_rate": float(_win_rate(test_y_ref, stacked_test)),
                },
            }
        )

    grouped_aggregates: dict[int, dict[str, object]] = {}
    for group in comparison_groups:
        sim_size = int(group["sim_size"])
        agg = grouped_aggregates.setdefault(
            sim_size,
            {
                "sim_size": int(sim_size),
                "num_offsets": 0,
                "base_models": {},
                "soft_ensemble_test_win_rates": [],
                "stacked_ensemble_test_win_rates": [],
            },
        )
        agg["num_offsets"] = int(agg["num_offsets"]) + 1
        for model_name, stats in group["models"].items():
            model_list = agg["base_models"].setdefault(model_name, [])
            model_list.append(float(stats["test_win_rate"]))
        agg["soft_ensemble_test_win_rates"].append(float(group["soft_ensemble"]["test_win_rate"]))
        agg["stacked_ensemble_test_win_rates"].append(float(group["stacked_ensemble"]["test_win_rate"]))

    aggregates: list[dict[str, object]] = []
    for sim_size in sorted(grouped_aggregates):
        item = grouped_aggregates[int(sim_size)]
        base_models = {
            model_name: {
                "mean_test_win_rate": float(np.mean(rates)),
                "min_test_win_rate": float(np.min(rates)),
                "max_test_win_rate": float(np.max(rates)),
            }
            for model_name, rates in sorted(item["base_models"].items())
        }
        aggregates.append(
            {
                "sim_size": int(sim_size),
                "num_offsets": int(item["num_offsets"]),
                "base_models": base_models,
                "soft_ensemble": {
                    "mean_test_win_rate": float(np.mean(item["soft_ensemble_test_win_rates"])),
                    "min_test_win_rate": float(np.min(item["soft_ensemble_test_win_rates"])),
                    "max_test_win_rate": float(np.max(item["soft_ensemble_test_win_rates"])),
                },
                "stacked_ensemble": {
                    "mean_test_win_rate": float(np.mean(item["stacked_ensemble_test_win_rates"])),
                    "min_test_win_rate": float(np.min(item["stacked_ensemble_test_win_rates"])),
                    "max_test_win_rate": float(np.max(item["stacked_ensemble_test_win_rates"])),
                },
            }
        )

    fieldnames = list(round_rows[0].keys())
    rows_out = rows_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="direction_ensemble_rows")
    with rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in round_rows:
            writer.writerow(row)
    summary_out = summary_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="direction_ensemble_summary")
    summary_payload = {
        "rows_csv_path": str(rows_out),
        "comparison_groups": comparison_groups,
        "aggregates": aggregates,
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
