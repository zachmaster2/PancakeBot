from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path

import numpy as np

from inspection.neural_direction_eval_common import (
    load_recent_direction_eval_slice,
    rows_path,
    summary_path,
)
from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_confidence import (
    apply_temperature_calibrator_to_probs,
    chosen_side_confidence,
    fit_temperature_calibrator_from_probs,
)
from pancakebot.domain.models.neural_direction_mlp import (
    load_neural_direction_mlp_bundle,
    predict_neural_direction_probabilities,
)
from pancakebot.domain.models.neural_direction_policy import (
    confidence_threshold_for_target_coverage,
    simulate_confidence_threshold_policy,
)
from pancakebot.runtime.contract_constants_cache import load_contract_constants

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class _SourceEvalJob:
    source_rows_csv: str
    source_bundle_path: str
    training_policy: str
    sim_size: int
    tail_offset_rounds: int
    train_size: int
    pretrain_size: int
    valid_size: int


@dataclass(frozen=True, slots=True)
class NeuralDirectionPolicyEvalRow:
    source_rows_csv: str
    source_bundle_path: str
    training_policy: str
    sim_size: int
    tail_offset_rounds: int
    train_size: int
    pretrain_size: int
    valid_size: int
    target_coverage_fraction: float
    threshold_used: float
    bet_size_bnb: float
    num_rounds: int
    num_bets: int
    num_wins: int
    num_skips_below_threshold: int
    num_skips_insufficient_bankroll: int
    bet_rate: float
    win_rate: float
    net_profit_bnb: float
    profit_per_500_bnb: float
    max_drawdown_bnb: float
    final_bankroll_bnb: float
    selected_mean_confidence: float
    selected_min_confidence: float | None
    selected_max_confidence: float | None


@dataclass(frozen=True, slots=True)
class NeuralDirectionPolicyEvalAggregateRow:
    training_policy: str
    sim_size: int
    target_coverage_fraction: float
    bet_size_bnb: float
    num_offsets: int
    mean_threshold_used: float
    mean_bet_rate: float
    mean_win_rate: float
    mean_net_profit_bnb: float
    mean_profit_per_500_bnb: float
    mean_max_drawdown_bnb: float
    min_profit_per_500_bnb: float
    max_profit_per_500_bnb: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--rows-csvs", type=str, required=True)
    parser.add_argument("--coverage-fractions", type=str, default="0.10,0.05,0.02,0.01")
    parser.add_argument("--bet-sizes-bnb", type=str, default="0.10")
    parser.add_argument("--initial-bankroll-bnb", type=float, default=50.0)
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
        raise InvariantError("neural_direction_policy_rows_csvs_empty")
    return tuple(out)


def _parse_fraction_list(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = float(text)
        if not (0.0 < float(value) <= 1.0):
            raise InvariantError("neural_direction_policy_fraction_invalid")
        out.append(float(value))
    if not out:
        raise InvariantError("neural_direction_policy_fraction_empty")
    return tuple(out)


def _parse_bet_sizes(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = float(text)
        if float(value) <= 0.0:
            raise InvariantError("neural_direction_policy_bet_size_nonpositive")
        out.append(float(value))
    if not out:
        raise InvariantError("neural_direction_policy_bet_sizes_empty")
    return tuple(out)


def _load_source_jobs(*, rows_csvs: tuple[Path, ...]) -> list[_SourceEvalJob]:
    out: list[_SourceEvalJob] = []
    for rows_csv in rows_csvs:
        with rows_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for source_row in reader:
                out.append(
                    _SourceEvalJob(
                        source_rows_csv=str(rows_csv),
                        source_bundle_path=str(source_row["bundle_path"]),
                        training_policy=str(source_row.get("training_policy", "flat")),
                        sim_size=int(source_row["sim_size"]),
                        tail_offset_rounds=int(source_row["tail_offset_rounds"]),
                        train_size=int(source_row["train_size"]),
                        pretrain_size=int(source_row.get("pretrain_size", 0)),
                        valid_size=int(source_row["valid_size"]),
                    )
                )
    if not out:
        raise InvariantError("neural_direction_policy_source_jobs_empty")
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
        raise InvariantError("neural_direction_policy_split_len_mismatch")
    target_tail = tuple(int(epoch) for epoch in target_epochs[-int(needed) :])
    valid_start = int(pretrain_size) + int(train_size)
    valid_end = int(valid_start) + int(valid_size)
    valid_epochs = tuple(int(epoch) for epoch in target_tail[int(valid_start) : int(valid_end)])
    test_epochs = tuple(int(epoch) for epoch in target_tail[-int(sim_size) :])
    return valid_epochs, test_epochs


def _aggregate_rows(rows: list[NeuralDirectionPolicyEvalRow]) -> list[NeuralDirectionPolicyEvalAggregateRow]:
    grouped: dict[tuple[str, int, float, float], list[NeuralDirectionPolicyEvalRow]] = {}
    for row in rows:
        key = (
            str(row.training_policy),
            int(row.sim_size),
            float(row.target_coverage_fraction),
            float(row.bet_size_bnb),
        )
        grouped.setdefault(key, []).append(row)
    out: list[NeuralDirectionPolicyEvalAggregateRow] = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), int(item[1]), -float(item[2]), float(item[3]))):
        group = grouped[key]
        out.append(
            NeuralDirectionPolicyEvalAggregateRow(
                training_policy=str(key[0]),
                sim_size=int(key[1]),
                target_coverage_fraction=float(key[2]),
                bet_size_bnb=float(key[3]),
                num_offsets=int(len(group)),
                mean_threshold_used=float(np.mean([row.threshold_used for row in group])),
                mean_bet_rate=float(np.mean([row.bet_rate for row in group])),
                mean_win_rate=float(np.mean([row.win_rate for row in group])),
                mean_net_profit_bnb=float(np.mean([row.net_profit_bnb for row in group])),
                mean_profit_per_500_bnb=float(np.mean([row.profit_per_500_bnb for row in group])),
                mean_max_drawdown_bnb=float(np.mean([row.max_drawdown_bnb for row in group])),
                min_profit_per_500_bnb=float(min(row.profit_per_500_bnb for row in group)),
                max_profit_per_500_bnb=float(max(row.profit_per_500_bnb for row in group)),
            )
        )
    return out


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_app_config(str(args.config))
    constants = load_contract_constants()
    coverage_fractions = _parse_fraction_list(args.coverage_fractions)
    bet_sizes_bnb = _parse_bet_sizes(args.bet_sizes_bnb)
    if float(args.initial_bankroll_bnb) <= 0.0:
        raise InvariantError("neural_direction_policy_initial_bankroll_nonpositive")
    if any(float(bet_size) < float(constants.min_bet_amount_bnb) for bet_size in bet_sizes_bnb):
        raise InvariantError("neural_direction_policy_bet_size_below_min_bet")

    source_jobs = _load_source_jobs(rows_csvs=_parse_rows_csvs(args.rows_csvs))

    rows: list[NeuralDirectionPolicyEvalRow] = []
    for job_idx, job in enumerate(source_jobs, start=1):
        required_examples = int(job.pretrain_size) + int(job.train_size) + int(job.valid_size) + int(job.sim_size)
        eval_slice = load_recent_direction_eval_slice(
            config_path=str(args.config),
            required_examples=int(required_examples),
            tail_offset_rounds=int(job.tail_offset_rounds),
        )
        dataset = eval_slice.dataset
        valid_epochs, test_epochs = _split_target_epochs(
            target_epochs=dataset.target_epochs,
            train_size=int(job.train_size),
            pretrain_size=int(job.pretrain_size),
            valid_size=int(job.valid_size),
            sim_size=int(job.sim_size),
        )
        bundle = load_neural_direction_mlp_bundle(str(job.source_bundle_path))
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
        calibrator = fit_temperature_calibrator_from_probs(
            bull_probs=valid_probs,
            labels=valid_y,
        )
        calibrated_valid_probs = apply_temperature_calibrator_to_probs(
            bull_probs=valid_probs,
            calibrator=calibrator,
        )
        calibrated_test_probs = apply_temperature_calibrator_to_probs(
            bull_probs=test_probs,
            calibrator=calibrator,
        )
        valid_pred = (np.asarray(valid_probs, dtype=np.float32) >= 0.5).astype(np.int64)
        valid_conf = chosen_side_confidence(
            predicted_labels=valid_pred,
            calibrated_bull_probs=calibrated_valid_probs,
        )
        test_rounds = [
            eval_slice.target_rounds_by_epoch[int(epoch)]
            for epoch in test_epochs
        ]

        for coverage_fraction in coverage_fractions:
            threshold = confidence_threshold_for_target_coverage(
                chosen_side_confidence=valid_conf,
                target_coverage_fraction=float(coverage_fraction),
            )
            for bet_size_bnb in bet_sizes_bnb:
                result = simulate_confidence_threshold_policy(
                    rounds=test_rounds,
                    calibrated_bull_probs=calibrated_test_probs,
                    threshold=float(threshold),
                    bet_size_bnb=float(bet_size_bnb),
                    initial_bankroll_bnb=float(args.initial_bankroll_bnb),
                    treasury_fee_fraction=float(constants.treasury_fee_fraction),
                )
                rows.append(
                    NeuralDirectionPolicyEvalRow(
                        source_rows_csv=str(job.source_rows_csv),
                        source_bundle_path=str(job.source_bundle_path),
                        training_policy=str(job.training_policy),
                        sim_size=int(job.sim_size),
                        tail_offset_rounds=int(job.tail_offset_rounds),
                        train_size=int(job.train_size),
                        pretrain_size=int(job.pretrain_size),
                        valid_size=int(job.valid_size),
                        target_coverage_fraction=float(coverage_fraction),
                        threshold_used=float(threshold),
                        bet_size_bnb=float(bet_size_bnb),
                        num_rounds=int(result.num_rounds),
                        num_bets=int(result.num_bets),
                        num_wins=int(result.num_wins),
                        num_skips_below_threshold=int(result.num_skips_below_threshold),
                        num_skips_insufficient_bankroll=int(result.num_skips_insufficient_bankroll),
                        bet_rate=float(result.bet_rate),
                        win_rate=float(result.win_rate),
                        net_profit_bnb=float(result.net_profit_bnb),
                        profit_per_500_bnb=float(result.profit_per_500_bnb),
                        max_drawdown_bnb=float(result.max_drawdown_bnb),
                        final_bankroll_bnb=float(result.final_bankroll_bnb),
                        selected_mean_confidence=float(result.selected_mean_confidence),
                        selected_min_confidence=result.selected_min_confidence,
                        selected_max_confidence=result.selected_max_confidence,
                    )
                )
        print(
            {
                "phase": "job_done",
                "job_index": int(job_idx),
                "job_count": int(len(source_jobs)),
                "training_policy": str(job.training_policy),
                "sim_size": int(job.sim_size),
                "tail_offset_rounds": int(job.tail_offset_rounds),
            },
            flush=True,
        )
        del eval_slice
        del dataset
        gc.collect()

    if not rows:
        raise InvariantError("neural_direction_policy_rows_empty")
    aggregates = _aggregate_rows(rows)
    rows_out = rows_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_policy_rows",
    )
    summary_out = summary_path(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        suffix="neural_direction_policy_summary",
    )
    with rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary_payload = {
        "rows_csv_path": str(rows_out),
        "coverage_fractions": [float(value) for value in coverage_fractions],
        "bet_sizes_bnb": [float(value) for value in bet_sizes_bnb],
        "initial_bankroll_bnb": float(args.initial_bankroll_bnb),
        "aggregates": [asdict(row) for row in aggregates],
        "row_count": int(len(rows)),
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
