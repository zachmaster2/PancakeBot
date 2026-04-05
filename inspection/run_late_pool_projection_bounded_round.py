from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inspection.neural_direction_eval_common import (
    load_recent_direction_eval_slice,
    parse_positive_int_list,
    rows_path,
    summary_path,
)
from pancakebot.config.load_config import load_app_config
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.targets import compute_pool_forecast_targets
from pancakebot.domain.models.direction_tree_model import (
    load_direction_tree_bundle,
    predict_direction_tree_probabilities,
)
from pancakebot.domain.models.final_pool_model import FinalPoolModel
from pancakebot.domain.models.neural_direction_dataset import select_feature_columns_exact
from pancakebot.domain.models.neural_direction_mlp import (
    load_neural_direction_mlp_bundle,
    predict_neural_direction_probabilities,
)
from pancakebot.domain.models.neural_direction_tcn import (
    build_sequence_examples_for_target_epochs,
    load_neural_direction_tcn_bundle,
    predict_neural_direction_tcn_probabilities,
)
from pancakebot.domain.models.payout_aware_policy import (
    projected_profit_if_side_wins,
    simulate_payout_aware_policy,
    tune_side_thresholds,
)
from pancakebot.domain.types import Round
from pancakebot.runtime.contract_constants_cache import load_contract_constants

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class _BaseModelJob:
    model_name: str
    model_type: str
    bundle_path: str
    train_size: int
    valid_size: int
    seq_len: int | None


@dataclass(frozen=True, slots=True)
class LatePoolBoundedRow:
    direction_source: str
    train_size: int
    bet_size_bnb: float
    sim_size: int
    tail_offset_rounds: int
    bull_threshold: float
    bear_threshold: float
    num_rounds: int
    num_bets: int
    num_bull_bets: int
    num_bear_bets: int
    num_wins: int
    num_losses: int
    num_refunds: int
    num_skips_below_threshold: int
    num_skips_insufficient_bankroll: int
    bet_rate: float
    win_rate: float
    net_profit_bnb: float
    profit_per_500_bnb: float
    max_drawdown_bnb: float
    final_bankroll_bnb: float
    selected_mean_predicted_ev: float
    selected_min_predicted_ev: float | None
    selected_max_predicted_ev: float | None
    valid_num_bets: int
    valid_bet_rate: float
    valid_net_profit_bnb: float
    valid_profit_per_500_bnb: float
    valid_win_rate: float
    valid_late_total_mae: float
    valid_late_total_rmse: float
    valid_late_bull_frac_mae: float
    valid_late_bull_frac_rmse: float
    test_late_total_mae: float
    test_late_total_rmse: float
    test_late_bull_frac_mae: float
    test_late_bull_frac_rmse: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument(
        "--manifest-csv",
        type=str,
        default="../PancakeBot_var_exp/direction_ensemble_longstream_manifest_20260403.csv",
    )
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument(
        "--direction-sources",
        type=str,
        default="mlp,catboost,soft_mean_all,mean2_mlp_catboost",
    )
    parser.add_argument("--train-sizes", type=str, default="100000,200000,400000")
    parser.add_argument("--bet-sizes", type=str, default="0.05,0.10")
    parser.add_argument("--sim-size", type=int, default=50000)
    parser.add_argument("--valid-size", type=int, default=3000)
    parser.add_argument("--tail-offset-rounds", type=int, default=0)
    parser.add_argument("--robustness-offsets", type=str, default="0,5000,10000,15000")
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="-0.020,-0.010,-0.005,0.000,0.001,0.0025,0.005,0.010,0.020",
    )
    parser.add_argument("--valid-min-bet-rate", type=float, default=0.005)
    parser.add_argument("--pool-alpha-total", type=float, default=0.8)
    parser.add_argument("--pool-alpha-ratio", type=float, default=0.7)
    parser.add_argument("--random-seed", type=int, default=20260404)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument("--current-bar-net-bnb", type=float, default=3.7952054686396153)
    parser.add_argument("--current-bar-per500", type=float, default=0.037952054686396154)
    parser.add_argument("--current-bar-max-dd-bnb", type=float, default=3.216061216538158)
    return parser


def _parse_str_list(raw: str) -> list[str]:
    values = [str(token).strip() for token in str(raw).split(",") if str(token).strip() != ""]
    if not values:
        raise InvariantError("late_pool_bounded_round_str_list_empty")
    return values


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        out.append(float(text))
    if not out:
        raise InvariantError("late_pool_bounded_round_float_list_empty")
    return out


def _parse_nonnegative_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = int(text)
        if int(value) < 0:
            raise InvariantError("late_pool_bounded_round_negative_offset")
        out.append(int(value))
    if not out:
        raise InvariantError("late_pool_bounded_round_offset_list_empty")
    return out


def _load_base_model_jobs(*, manifest_csv: str) -> list[_BaseModelJob]:
    manifest_path = Path(str(manifest_csv)).resolve()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise InvariantError("late_pool_bounded_round_manifest_empty")
    jobs: list[_BaseModelJob] = []
    for row in rows:
        model_type = str(row.get("model_type") or "")
        seq_len_raw = row.get("seq_len")
        seq_len = None if seq_len_raw in (None, "") else int(seq_len_raw)
        if str(model_type) == "":
            model_type = "tcn" if seq_len is not None else "mlp"
        jobs.append(
            _BaseModelJob(
                model_name=str(model_type),
                model_type=str(model_type),
                bundle_path=str(row["bundle_path"]),
                train_size=int(row["train_size"]),
                valid_size=int(row["valid_size"]),
                seq_len=seq_len,
            )
        )
    return jobs


def _split_target_epochs(
    *,
    target_epochs: tuple[int, ...],
    train_size: int,
    valid_size: int,
    sim_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    needed = int(train_size) + int(valid_size) + int(sim_size)
    if len(target_epochs) < int(needed):
        raise InvariantError("late_pool_bounded_round_split_len_mismatch")
    tail = tuple(int(epoch) for epoch in target_epochs[-int(needed) :])
    train_epochs = tuple(int(epoch) for epoch in tail[: int(train_size)])
    valid_epochs = tuple(int(epoch) for epoch in tail[int(train_size) : int(train_size) + int(valid_size)])
    test_epochs = tuple(int(epoch) for epoch in tail[-int(sim_size) :])
    return train_epochs, valid_epochs, test_epochs


def _rows_for_target_epochs(
    *,
    feature_matrix: np.ndarray,
    target_epochs_source: tuple[int, ...],
    target_epochs: tuple[int, ...],
) -> np.ndarray:
    index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(target_epochs_source)}
    row_idx = [int(index_by_epoch[int(epoch)]) for epoch in target_epochs]
    return np.asarray(feature_matrix[np.asarray(row_idx, dtype=np.int64)], dtype=np.float32)


def _targets_for_target_epochs(
    *,
    target_values: np.ndarray,
    target_epochs_source: tuple[int, ...],
    target_epochs: tuple[int, ...],
) -> np.ndarray:
    index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(target_epochs_source)}
    row_idx = [int(index_by_epoch[int(epoch)]) for epoch in target_epochs]
    return np.asarray(target_values[np.asarray(row_idx, dtype=np.int64)], dtype=np.float32)


def _rounds_for_target_epochs(
    *,
    rounds_by_epoch: dict[int, Round],
    target_epochs: tuple[int, ...],
) -> list[Round]:
    return [rounds_by_epoch[int(epoch)] for epoch in target_epochs]


def _predict_probabilities_for_epochs(
    *,
    job: _BaseModelJob,
    eval_dataset,
    target_epochs: tuple[int, ...],
) -> np.ndarray:
    if str(job.model_type) == "mlp":
        bundle = load_neural_direction_mlp_bundle(str(job.bundle_path))
        dataset = select_feature_columns_exact(dataset=eval_dataset, feature_columns=tuple(bundle.feature_columns))
        probs_all = predict_neural_direction_probabilities(
            bundle=bundle,
            feature_matrix=np.asarray(dataset.feature_matrix, dtype=np.float32),
        )
        index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
        idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in target_epochs], dtype=np.int64)
        return np.asarray(probs_all[idx], dtype=np.float32)
    if str(job.model_type) == "tcn":
        if job.seq_len is None:
            raise InvariantError("late_pool_bounded_round_tcn_seq_len_missing")
        bundle = load_neural_direction_tcn_bundle(str(job.bundle_path))
        dataset = select_feature_columns_exact(dataset=eval_dataset, feature_columns=tuple(bundle.feature_columns))
        chunk_size = 8192
        probs_parts: list[np.ndarray] = []
        for start_idx in range(0, int(len(target_epochs)), int(chunk_size)):
            epoch_chunk = tuple(int(epoch) for epoch in target_epochs[int(start_idx) : int(start_idx) + int(chunk_size)])
            feature_sequences, _ = build_sequence_examples_for_target_epochs(
                dataset=dataset,
                target_epochs=epoch_chunk,
                seq_len=int(job.seq_len),
            )
            probs_parts.append(
                predict_neural_direction_tcn_probabilities(
                    bundle=bundle,
                    feature_sequences=feature_sequences,
                )
            )
        return np.asarray(np.concatenate(probs_parts, axis=0), dtype=np.float32)
    if str(job.model_type) in ("lightgbm", "catboost"):
        bundle = load_direction_tree_bundle(str(job.bundle_path))
        dataset = select_feature_columns_exact(dataset=eval_dataset, feature_columns=tuple(bundle.feature_columns))
        probs_all = predict_direction_tree_probabilities(
            bundle=bundle,
            feature_matrix=np.asarray(dataset.feature_matrix, dtype=np.float32),
        )
        index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
        idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in target_epochs], dtype=np.int64)
        return np.asarray(probs_all[idx], dtype=np.float32)
    raise InvariantError("late_pool_bounded_round_model_type_unknown")


def _build_feature_matrix(
    *,
    dataset,
    target_epochs: tuple[int, ...],
    base_jobs: list[_BaseModelJob],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
    idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in target_epochs], dtype=np.int64)
    base_x = np.asarray(dataset.feature_matrix[idx], dtype=np.float32)
    probs_by_model: dict[str, np.ndarray] = {}
    for job in base_jobs:
        probs_by_model[str(job.model_name)] = _predict_probabilities_for_epochs(
            job=job,
            eval_dataset=dataset,
            target_epochs=target_epochs,
        )
    model_order = sorted(probs_by_model)
    prob_matrix = np.column_stack([np.asarray(probs_by_model[name], dtype=np.float32) for name in model_order]).astype(np.float32)
    vote_bull = np.sum(prob_matrix >= 0.5, axis=1, keepdims=True).astype(np.float32)
    vote_bear = np.sum(prob_matrix < 0.5, axis=1, keepdims=True).astype(np.float32)
    mean_col = np.mean(prob_matrix, axis=1, keepdims=True).astype(np.float32)
    std_col = np.std(prob_matrix, axis=1, keepdims=True).astype(np.float32)
    min_col = np.min(prob_matrix, axis=1, keepdims=True).astype(np.float32)
    max_col = np.max(prob_matrix, axis=1, keepdims=True).astype(np.float32)
    range_col = (max_col - min_col).astype(np.float32)
    extra_matrix = np.concatenate(
        [
            prob_matrix,
            mean_col,
            std_col,
            min_col,
            max_col,
            range_col,
            vote_bull,
            vote_bear,
        ],
        axis=1,
    ).astype(np.float32)
    extra_columns = tuple(
        [f"p_bull_{name}" for name in model_order]
        + [
            "p_bull_mean",
            "p_bull_std",
            "p_bull_min",
            "p_bull_max",
            "p_bull_range",
            "bull_vote_count",
            "bear_vote_count",
        ]
    )
    combined_x = np.concatenate([base_x, extra_matrix], axis=1).astype(np.float32)
    combined_columns = tuple(str(col) for col in dataset.feature_columns) + extra_columns
    return combined_x, combined_columns, probs_by_model


def _direction_probabilities_for_source(
    *,
    probs_by_model: dict[str, np.ndarray],
    source: str,
) -> np.ndarray:
    if str(source) in probs_by_model:
        return np.asarray(probs_by_model[str(source)], dtype=np.float32)
    if str(source) == "soft_mean_all":
        ordered = [np.asarray(probs_by_model[name], dtype=np.float32) for name in sorted(probs_by_model)]
        return np.asarray(np.mean(np.column_stack(ordered), axis=1), dtype=np.float32)
    if str(source) == "mean2_mlp_catboost":
        if "mlp" not in probs_by_model or "catboost" not in probs_by_model:
            raise InvariantError("late_pool_bounded_round_direction_source_missing_models")
        return np.asarray(
            (
                np.asarray(probs_by_model["mlp"], dtype=np.float32)
                + np.asarray(probs_by_model["catboost"], dtype=np.float32)
            )
            / 2.0,
            dtype=np.float32,
        )
    raise InvariantError("late_pool_bounded_round_direction_source_unknown")


def _mae(*, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    if yt.shape != yp.shape or yt.ndim != 1 or int(len(yt)) <= 0:
        raise InvariantError("late_pool_bounded_round_mae_input_invalid")
    return float(np.mean(np.abs(yp - yt)))


def _rmse(*, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    if yt.shape != yp.shape or yt.ndim != 1 or int(len(yt)) <= 0:
        raise InvariantError("late_pool_bounded_round_rmse_input_invalid")
    return float(np.sqrt(np.mean(np.square(yp - yt))))


def _pool_target_arrays(
    *,
    rounds_by_epoch: dict[int, Round],
    target_epochs: tuple[int, ...],
    cutoff_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    late_total: list[float] = []
    late_bull_frac: list[float] = []
    for epoch in target_epochs:
        round_t = rounds_by_epoch[int(epoch)]
        targets = compute_pool_forecast_targets(round_t=round_t, cutoff_seconds=int(cutoff_seconds))
        late_total.append(float(targets.late_inflow_total_bnb))
        late_bull_frac.append(float(targets.late_inflow_bull_frac))
    return (
        np.asarray(late_total, dtype=np.float32),
        np.asarray(late_bull_frac, dtype=np.float32),
    )


def _expected_ev_arrays(
    *,
    rounds: list[Round],
    p_bull: np.ndarray,
    pred_late_total: np.ndarray,
    pred_late_bull_frac: np.ndarray,
    bet_size_bnb: float,
    cutoff_seconds: int,
    treasury_fee_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(p_bull, dtype=np.float32)
    late_total = np.asarray(pred_late_total, dtype=np.float32)
    late_frac = np.asarray(pred_late_bull_frac, dtype=np.float32)
    if probs.shape != late_total.shape or probs.shape != late_frac.shape:
        raise InvariantError("late_pool_bounded_round_ev_shape_mismatch")
    loss_const_bnb = -float(bet_size_bnb) - float(GAS_COST_BET_BNB)
    ev_bull: list[float] = []
    ev_bear: list[float] = []
    for round_t, prob_bull, pred_total, pred_frac in zip(rounds, probs, late_total, late_frac, strict=True):
        bull_profit = projected_profit_if_side_wins(
            round_closed=round_t,
            bet_size_bnb=float(bet_size_bnb),
            bet_side="Bull",
            treasury_fee_fraction=float(treasury_fee_fraction),
            cutoff_seconds=int(cutoff_seconds),
            pred_late_inflow_total_bnb=float(pred_total),
            pred_late_inflow_bull_frac=float(pred_frac),
        )
        bear_profit = projected_profit_if_side_wins(
            round_closed=round_t,
            bet_size_bnb=float(bet_size_bnb),
            bet_side="Bear",
            treasury_fee_fraction=float(treasury_fee_fraction),
            cutoff_seconds=int(cutoff_seconds),
            pred_late_inflow_total_bnb=float(pred_total),
            pred_late_inflow_bull_frac=float(pred_frac),
        )
        p_b = float(min(1.0, max(0.0, prob_bull)))
        ev_bull.append(float(p_b) * float(bull_profit) + (1.0 - float(p_b)) * float(loss_const_bnb))
        ev_bear.append((1.0 - float(p_b)) * float(bear_profit) + float(p_b) * float(loss_const_bnb))
    return np.asarray(ev_bull, dtype=np.float32), np.asarray(ev_bear, dtype=np.float32)


def _plot_cumulative_overlay(
    *,
    traces_by_label: dict[str, list[dict[str, object]]],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(14, 8))
    for label, rows in traces_by_label.items():
        ys = np.asarray([float(row["cumulative_profit_bnb"]) for row in rows], dtype=np.float32)
        xs = np.arange(1, int(len(ys)) + 1)
        plt.plot(xs, ys, label=label, linewidth=2)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Held-Out Round Index")
    plt.ylabel("Cumulative Profit (BNB)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_rolling_overlay(
    *,
    traces_by_label: dict[str, list[dict[str, object]]],
    output_path: Path,
    title: str,
    window_rounds: int = 2000,
) -> None:
    plt.figure(figsize=(14, 8))
    for label, rows in traces_by_label.items():
        realized = np.asarray([float(row["realized_profit_bnb"]) for row in rows], dtype=np.float32)
        if int(len(realized)) < int(window_rounds):
            continue
        rolled = np.convolve(realized, np.ones(int(window_rounds), dtype=np.float32), mode="valid")
        ys = rolled * 500.0 / float(window_rounds)
        xs = np.arange(int(window_rounds), int(window_rounds) + int(len(ys)))
        plt.plot(xs, ys, label=label, linewidth=1.6)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Held-Out Round Index")
    plt.ylabel(f"Rolling Net / 500 (window={int(window_rounds)})")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _decision(
    *,
    best_row: dict[str, object],
    robustness_rows: list[dict[str, object]],
    current_bar_net_bnb: float,
    current_bar_max_dd_bnb: float,
) -> tuple[str, str]:
    best_net = float(best_row["net_profit_bnb"])
    best_dd = float(best_row["max_drawdown_bnb"])
    robust_nets = [float(row["net_profit_bnb"]) for row in robustness_rows]
    robust_per500 = [float(row["profit_per_500_bnb"]) for row in robustness_rows]
    robust_dd = [float(row["max_drawdown_bnb"]) for row in robustness_rows]
    positive_count = sum(1 for value in robust_nets if float(value) > 0.0)
    mean_per500 = float(sum(robust_per500) / len(robust_per500))
    worst_per500 = float(min(robust_per500))
    best_beats_bar = float(best_net) > float(current_bar_net_bnb) or (
        float(best_net) >= float(current_bar_net_bnb) * 0.9 and float(best_dd) < float(current_bar_max_dd_bnb) * 0.75
    )
    robust_enough = (
        int(positive_count) >= max(3, len(robust_nets) - 1)
        and float(mean_per500) > 0.0
        and float(worst_per500) > -0.01
        and float(max(robust_dd)) <= 5.0
    )
    if bool(best_beats_bar) and bool(robust_enough):
        return "dry-run", "late-pool branch beat or materially stabilized the current bar and held up across adjacent latest-tail offsets"
    return "quit", "late-pool branch failed to produce a branch that both beats/stabilizes the current bar and holds up across adjacent latest-tail offsets"


def _evaluate_tail(
    *,
    config_path: str,
    manifest_csv: str,
    direction_sources: list[str],
    train_sizes: list[int],
    bet_sizes: list[float],
    sim_size: int,
    valid_size: int,
    tail_offset_rounds: int,
    threshold_grid: list[float],
    valid_min_bet_rate: float,
    pool_alpha_total: float,
    pool_alpha_ratio: float,
    random_seed: int,
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]], dict[str, dict[str, object]], dict[str, object]]:
    base_jobs = _load_base_model_jobs(manifest_csv=str(manifest_csv))
    max_train_size = max(int(value) for value in train_sizes)
    max_seq_warmup = max((int(job.seq_len) - 1) for job in base_jobs if job.seq_len is not None) if any(job.seq_len is not None for job in base_jobs) else 0
    required_examples = int(max_train_size) + int(valid_size) + int(sim_size) + int(max_seq_warmup)
    eval_slice = load_recent_direction_eval_slice(
        config_path=str(config_path),
        required_examples=int(required_examples),
        tail_offset_rounds=int(tail_offset_rounds),
    )
    runtime_cfg = load_app_config(str(config_path))
    constants = load_contract_constants()
    dataset = eval_slice.dataset
    max_target_epochs = tuple(int(epoch) for epoch in dataset.target_epochs)
    effective_target_epochs = (
        tuple(int(epoch) for epoch in max_target_epochs[int(max_seq_warmup) :])
        if int(max_seq_warmup) > 0
        else tuple(int(epoch) for epoch in max_target_epochs)
    )
    rounds_by_epoch = dict(eval_slice.target_rounds_by_epoch)
    combined_x, _combined_columns, probs_by_model = _build_feature_matrix(
        dataset=dataset,
        target_epochs=effective_target_epochs,
        base_jobs=base_jobs,
    )
    late_total_targets, late_frac_targets = _pool_target_arrays(
        rounds_by_epoch=rounds_by_epoch,
        target_epochs=effective_target_epochs,
        cutoff_seconds=int(runtime_cfg.cutoff_seconds),
    )

    final_rows: list[dict[str, object]] = []
    source_best_trace_by_name: dict[str, list[dict[str, object]]] = {}
    source_best_row_by_name: dict[str, dict[str, object]] = {}
    best_overall_row: dict[str, object] | None = None
    best_overall_trace: list[dict[str, object]] = []

    for train_size in train_sizes:
        train_epochs, valid_epochs, test_epochs = _split_target_epochs(
            target_epochs=effective_target_epochs,
            train_size=int(train_size),
            valid_size=int(valid_size),
            sim_size=int(sim_size),
        )
        train_x = _rows_for_target_epochs(feature_matrix=combined_x, target_epochs_source=effective_target_epochs, target_epochs=train_epochs)
        valid_x = _rows_for_target_epochs(feature_matrix=combined_x, target_epochs_source=effective_target_epochs, target_epochs=valid_epochs)
        test_x = _rows_for_target_epochs(feature_matrix=combined_x, target_epochs_source=effective_target_epochs, target_epochs=test_epochs)
        train_late_total = _targets_for_target_epochs(target_values=late_total_targets, target_epochs_source=effective_target_epochs, target_epochs=train_epochs)
        valid_late_total = _targets_for_target_epochs(target_values=late_total_targets, target_epochs_source=effective_target_epochs, target_epochs=valid_epochs)
        test_late_total = _targets_for_target_epochs(target_values=late_total_targets, target_epochs_source=effective_target_epochs, target_epochs=test_epochs)
        train_late_frac = _targets_for_target_epochs(target_values=late_frac_targets, target_epochs_source=effective_target_epochs, target_epochs=train_epochs)
        valid_late_frac = _targets_for_target_epochs(target_values=late_frac_targets, target_epochs_source=effective_target_epochs, target_epochs=valid_epochs)
        test_late_frac = _targets_for_target_epochs(target_values=late_frac_targets, target_epochs_source=effective_target_epochs, target_epochs=test_epochs)

        pool_model = FinalPoolModel(
            alpha_total=float(pool_alpha_total),
            alpha_ratio=float(pool_alpha_ratio),
            seed=int(random_seed) + int(train_size),
        )
        pool_model.fit(
            train_x,
            train_late_total,
            train_late_frac,
            x_eval=valid_x,
            y_total_eval=valid_late_total,
            y_frac_eval=valid_late_frac,
        )
        valid_pool_preds = list(pool_model.predict(valid_x))
        test_pool_preds = list(pool_model.predict(test_x))
        valid_pred_total = np.asarray([float(pair[0]) for pair in valid_pool_preds], dtype=np.float32)
        valid_pred_frac = np.asarray([float(pair[1]) for pair in valid_pool_preds], dtype=np.float32)
        test_pred_total = np.asarray([float(pair[0]) for pair in test_pool_preds], dtype=np.float32)
        test_pred_frac = np.asarray([float(pair[1]) for pair in test_pool_preds], dtype=np.float32)

        valid_rounds = _rounds_for_target_epochs(rounds_by_epoch=rounds_by_epoch, target_epochs=valid_epochs)
        test_rounds = _rounds_for_target_epochs(rounds_by_epoch=rounds_by_epoch, target_epochs=test_epochs)
        valid_total_mae = _mae(y_true=valid_late_total, y_pred=valid_pred_total)
        valid_total_rmse = _rmse(y_true=valid_late_total, y_pred=valid_pred_total)
        valid_frac_mae = _mae(y_true=valid_late_frac, y_pred=valid_pred_frac)
        valid_frac_rmse = _rmse(y_true=valid_late_frac, y_pred=valid_pred_frac)
        test_total_mae = _mae(y_true=test_late_total, y_pred=test_pred_total)
        test_total_rmse = _rmse(y_true=test_late_total, y_pred=test_pred_total)
        test_frac_mae = _mae(y_true=test_late_frac, y_pred=test_pred_frac)
        test_frac_rmse = _rmse(y_true=test_late_frac, y_pred=test_pred_frac)

        for direction_source in direction_sources:
            source_probs = _direction_probabilities_for_source(
                probs_by_model=probs_by_model,
                source=str(direction_source),
            )
            valid_p_bull = _targets_for_target_epochs(target_values=source_probs, target_epochs_source=effective_target_epochs, target_epochs=valid_epochs)
            test_p_bull = _targets_for_target_epochs(target_values=source_probs, target_epochs_source=effective_target_epochs, target_epochs=test_epochs)
            for bet_size in bet_sizes:
                valid_pred_bull, valid_pred_bear = _expected_ev_arrays(
                    rounds=valid_rounds,
                    p_bull=valid_p_bull,
                    pred_late_total=valid_pred_total,
                    pred_late_bull_frac=valid_pred_frac,
                    bet_size_bnb=float(bet_size),
                    cutoff_seconds=int(runtime_cfg.cutoff_seconds),
                    treasury_fee_fraction=float(constants.treasury_fee_fraction),
                )
                test_pred_bull, test_pred_bear = _expected_ev_arrays(
                    rounds=test_rounds,
                    p_bull=test_p_bull,
                    pred_late_total=test_pred_total,
                    pred_late_bull_frac=test_pred_frac,
                    bet_size_bnb=float(bet_size),
                    cutoff_seconds=int(runtime_cfg.cutoff_seconds),
                    treasury_fee_fraction=float(constants.treasury_fee_fraction),
                )
                threshold_choice = tune_side_thresholds(
                    rounds=valid_rounds,
                    predicted_ev_bull=valid_pred_bull,
                    predicted_ev_bear=valid_pred_bear,
                    threshold_grid=list(threshold_grid),
                    bet_size_bnb=float(bet_size),
                    initial_bankroll_bnb=float(runtime_cfg.dry_initial_bankroll_bnb),
                    treasury_fee_fraction=float(constants.treasury_fee_fraction),
                    min_bet_rate=float(valid_min_bet_rate),
                )
                valid_result, _ = simulate_payout_aware_policy(
                    rounds=valid_rounds,
                    predicted_ev_bull=valid_pred_bull,
                    predicted_ev_bear=valid_pred_bear,
                    bull_threshold=float(threshold_choice.bull_threshold),
                    bear_threshold=float(threshold_choice.bear_threshold),
                    bet_size_bnb=float(bet_size),
                    initial_bankroll_bnb=float(runtime_cfg.dry_initial_bankroll_bnb),
                    treasury_fee_fraction=float(constants.treasury_fee_fraction),
                )
                test_result, test_trace = simulate_payout_aware_policy(
                    rounds=test_rounds,
                    predicted_ev_bull=test_pred_bull,
                    predicted_ev_bear=test_pred_bear,
                    bull_threshold=float(threshold_choice.bull_threshold),
                    bear_threshold=float(threshold_choice.bear_threshold),
                    bet_size_bnb=float(bet_size),
                    initial_bankroll_bnb=float(runtime_cfg.dry_initial_bankroll_bnb),
                    treasury_fee_fraction=float(constants.treasury_fee_fraction),
                )
                row = asdict(
                    LatePoolBoundedRow(
                        direction_source=str(direction_source),
                        train_size=int(train_size),
                        bet_size_bnb=float(bet_size),
                        sim_size=int(sim_size),
                        tail_offset_rounds=int(tail_offset_rounds),
                        bull_threshold=float(threshold_choice.bull_threshold),
                        bear_threshold=float(threshold_choice.bear_threshold),
                        num_rounds=int(test_result.num_rounds),
                        num_bets=int(test_result.num_bets),
                        num_bull_bets=int(test_result.num_bull_bets),
                        num_bear_bets=int(test_result.num_bear_bets),
                        num_wins=int(test_result.num_wins),
                        num_losses=int(test_result.num_losses),
                        num_refunds=int(test_result.num_refunds),
                        num_skips_below_threshold=int(test_result.num_skips_below_threshold),
                        num_skips_insufficient_bankroll=int(test_result.num_skips_insufficient_bankroll),
                        bet_rate=float(test_result.bet_rate),
                        win_rate=float(test_result.win_rate),
                        net_profit_bnb=float(test_result.net_profit_bnb),
                        profit_per_500_bnb=float(test_result.profit_per_500_bnb),
                        max_drawdown_bnb=float(test_result.max_drawdown_bnb),
                        final_bankroll_bnb=float(test_result.final_bankroll_bnb),
                        selected_mean_predicted_ev=float(test_result.selected_mean_predicted_ev),
                        selected_min_predicted_ev=test_result.selected_min_predicted_ev,
                        selected_max_predicted_ev=test_result.selected_max_predicted_ev,
                        valid_num_bets=int(valid_result.num_bets),
                        valid_bet_rate=float(valid_result.bet_rate),
                        valid_net_profit_bnb=float(valid_result.net_profit_bnb),
                        valid_profit_per_500_bnb=float(valid_result.profit_per_500_bnb),
                        valid_win_rate=float(valid_result.win_rate),
                        valid_late_total_mae=float(valid_total_mae),
                        valid_late_total_rmse=float(valid_total_rmse),
                        valid_late_bull_frac_mae=float(valid_frac_mae),
                        valid_late_bull_frac_rmse=float(valid_frac_rmse),
                        test_late_total_mae=float(test_total_mae),
                        test_late_total_rmse=float(test_total_rmse),
                        test_late_bull_frac_mae=float(test_frac_mae),
                        test_late_bull_frac_rmse=float(test_frac_rmse),
                    )
                )
                final_rows.append(row)
                trace_rows = [
                    {
                        "direction_source": str(direction_source),
                        "train_size": int(train_size),
                        "bet_size_bnb": float(bet_size),
                        "sim_size": int(sim_size),
                        "tail_offset_rounds": int(tail_offset_rounds),
                        "target_epoch": int(trace.target_epoch),
                        "predicted_ev_bull": float(trace.predicted_ev_bull),
                        "predicted_ev_bear": float(trace.predicted_ev_bear),
                        "bull_threshold": float(trace.bull_threshold),
                        "bear_threshold": float(trace.bear_threshold),
                        "action": str(trace.action),
                        "selected_side": trace.selected_side,
                        "selected_predicted_ev": trace.selected_predicted_ev,
                        "realized_profit_bnb": float(trace.realized_profit_bnb),
                        "cumulative_profit_bnb": float(trace.cumulative_profit_bnb),
                        "bankroll_bnb": float(trace.bankroll_bnb),
                        "outcome": trace.outcome,
                        "pred_late_total_bnb": float(test_pred_total[idx]),
                        "pred_late_bull_frac": float(test_pred_frac[idx]),
                        "actual_late_total_bnb": float(test_late_total[idx]),
                        "actual_late_bull_frac": float(test_late_frac[idx]),
                    }
                    for idx, trace in enumerate(test_trace)
                ]
                source_key = str(direction_source)
                current_best_source = source_best_row_by_name.get(source_key)
                if current_best_source is None or (
                    float(row["net_profit_bnb"]),
                    -float(row["max_drawdown_bnb"]),
                    float(row["profit_per_500_bnb"]),
                ) > (
                    float(current_best_source["net_profit_bnb"]),
                    -float(current_best_source["max_drawdown_bnb"]),
                    float(current_best_source["profit_per_500_bnb"]),
                ):
                    source_best_row_by_name[source_key] = dict(row)
                    source_best_trace_by_name[source_key] = list(trace_rows)
                if best_overall_row is None or (
                    float(row["net_profit_bnb"]),
                    -float(row["max_drawdown_bnb"]),
                    float(row["profit_per_500_bnb"]),
                ) > (
                    float(best_overall_row["net_profit_bnb"]),
                    -float(best_overall_row["max_drawdown_bnb"]),
                    float(best_overall_row["profit_per_500_bnb"]),
                ):
                    best_overall_row = dict(row)
                    best_overall_trace = list(trace_rows)
                print(
                    {
                        "phase": "late_pool_config_done",
                        "direction_source": str(direction_source),
                        "train_size": int(train_size),
                        "bet_size_bnb": float(bet_size),
                        "test_profit_per_500_bnb": float(test_result.profit_per_500_bnb),
                        "test_net_profit_bnb": float(test_result.net_profit_bnb),
                        "test_bet_rate": float(test_result.bet_rate),
                        "test_late_total_mae": float(test_total_mae),
                        "test_late_bull_frac_mae": float(test_frac_mae),
                    },
                    flush=True,
                )

    if best_overall_row is None:
        raise InvariantError("late_pool_bounded_round_rows_empty")
    return final_rows, source_best_trace_by_name, source_best_row_by_name, {
        "best_row": dict(best_overall_row),
        "best_trace": list(best_overall_trace),
    }


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    direction_sources = _parse_str_list(args.direction_sources)
    train_sizes = [int(value) for value in parse_positive_int_list(args.train_sizes)]
    bet_sizes = _parse_float_list(args.bet_sizes)
    robustness_offsets = _parse_nonnegative_int_list(args.robustness_offsets)
    if 0 not in robustness_offsets:
        robustness_offsets = [0] + list(robustness_offsets)
    threshold_grid = _parse_float_list(args.threshold_grid)

    final_rows, source_best_trace_by_name, source_best_row_by_name, best_payload = _evaluate_tail(
        config_path=str(args.config),
        manifest_csv=str(args.manifest_csv),
        direction_sources=direction_sources,
        train_sizes=train_sizes,
        bet_sizes=bet_sizes,
        sim_size=int(args.sim_size),
        valid_size=int(args.valid_size),
        tail_offset_rounds=int(args.tail_offset_rounds),
        threshold_grid=threshold_grid,
        valid_min_bet_rate=float(args.valid_min_bet_rate),
        pool_alpha_total=float(args.pool_alpha_total),
        pool_alpha_ratio=float(args.pool_alpha_ratio),
        random_seed=int(args.random_seed),
    )
    best_row = dict(best_payload["best_row"])
    best_trace = list(best_payload["best_trace"])

    robustness_rows: list[dict[str, object]] = []
    robustness_traces: dict[str, list[dict[str, object]]] = {}
    best_source = str(best_row["direction_source"])
    best_train_size = int(best_row["train_size"])
    best_bet_size = float(best_row["bet_size_bnb"])
    for offset in robustness_offsets:
        rows, _, _, payload = _evaluate_tail(
            config_path=str(args.config),
            manifest_csv=str(args.manifest_csv),
            direction_sources=[str(best_source)],
            train_sizes=[int(best_train_size)],
            bet_sizes=[float(best_bet_size)],
            sim_size=int(args.sim_size),
            valid_size=int(args.valid_size),
            tail_offset_rounds=int(offset),
            threshold_grid=threshold_grid,
            valid_min_bet_rate=float(args.valid_min_bet_rate),
            pool_alpha_total=float(args.pool_alpha_total),
            pool_alpha_ratio=float(args.pool_alpha_ratio),
            random_seed=int(args.random_seed),
        )
        if len(rows) != 1:
            raise InvariantError("late_pool_bounded_round_robustness_rows_count_invalid")
        row = dict(rows[0])
        row["robustness_offset_rounds"] = int(offset)
        robustness_rows.append(row)
        robustness_traces[f"offset{int(offset)}"] = list(payload["best_trace"])

    decision, decision_reason = _decision(
        best_row=best_row,
        robustness_rows=robustness_rows,
        current_bar_net_bnb=float(args.current_bar_net_bnb),
        current_bar_max_dd_bnb=float(args.current_bar_max_dd_bnb),
    )

    rows_out = rows_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="late_pool_bounded_rows")
    with rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)

    best_trace_out = rows_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="late_pool_bounded_best_trace")
    with best_trace_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_trace[0].keys()))
        writer.writeheader()
        for row in best_trace:
            writer.writerow(row)

    robust_trace_out = rows_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="late_pool_bounded_robustness_traces")
    with robust_trace_out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["robustness_offset_rounds"] + list(robustness_traces[next(iter(robustness_traces))][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, rows in robustness_traces.items():
            offset = int(key.replace("offset", ""))
            for row in rows:
                writer.writerow({"robustness_offset_rounds": int(offset), **row})

    source_overlay_plot = output_dir / f"{args.name_prefix}_source_best_cumulative.png"
    source_roll_plot = output_dir / f"{args.name_prefix}_source_best_rolling.png"
    best_plot = output_dir / f"{args.name_prefix}_best_cumulative.png"
    robust_overlay_plot = output_dir / f"{args.name_prefix}_best_robustness_cumulative.png"
    _plot_cumulative_overlay(
        traces_by_label={key: value for key, value in source_best_trace_by_name.items()},
        output_path=source_overlay_plot,
        title="Late-Pool Projection Policy: Source Best Cumulative BNB",
    )
    _plot_rolling_overlay(
        traces_by_label={key: value for key, value in source_best_trace_by_name.items()},
        output_path=source_roll_plot,
        title="Late-Pool Projection Policy: Source Best Rolling Net / 500",
    )
    _plot_cumulative_overlay(
        traces_by_label={"best": best_trace},
        output_path=best_plot,
        title="Late-Pool Projection Policy: Best Latest-Tail Cumulative BNB",
    )
    _plot_cumulative_overlay(
        traces_by_label=robustness_traces,
        output_path=robust_overlay_plot,
        title="Late-Pool Projection Policy: Best Config Robustness Cumulative BNB",
    )

    summary_payload = {
        "decision": str(decision),
        "decision_reason": str(decision_reason),
        "current_bar": {
            "net_profit_bnb": float(args.current_bar_net_bnb),
            "profit_per_500_bnb": float(args.current_bar_per500),
            "max_drawdown_bnb": float(args.current_bar_max_dd_bnb),
        },
        "best_row": dict(best_row),
        "source_best_rows": {
            str(source): dict(source_best_row_by_name[str(source)])
            for source in direction_sources
            if str(source) in source_best_row_by_name
        },
        "final_rows": final_rows,
        "robustness_rows": robustness_rows,
        "artifacts": {
            "rows_csv_path": str(rows_out),
            "best_trace_csv_path": str(best_trace_out),
            "robustness_traces_csv_path": str(robust_trace_out),
            "source_best_cumulative": str(source_overlay_plot),
            "source_best_rolling": str(source_roll_plot),
            "best_cumulative": str(best_plot),
            "best_robustness_cumulative": str(robust_overlay_plot),
        },
    }
    summary_out = summary_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="late_pool_bounded_summary")
    summary_out.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    report_lines: list[str] = []
    report_lines.append("# Late-Pool Projection Bounded Round")
    report_lines.append("")
    report_lines.append("## Standard")
    report_lines.append("")
    report_lines.append(f"- latest contiguous held-out stream: `{int(args.sim_size)}` valid rounds")
    report_lines.append(f"- direction sources: `{', '.join(direction_sources)}`")
    report_lines.append(f"- train sizes: `{', '.join(str(int(v)) for v in train_sizes)}`")
    report_lines.append(f"- fixed stakes: `{', '.join(f'{float(v):.2f}' for v in bet_sizes)}`")
    report_lines.append(f"- pool model alphas: total `{float(args.pool_alpha_total):.3f}`, ratio `{float(args.pool_alpha_ratio):.3f}`")
    report_lines.append("")
    report_lines.append("## Final Decision")
    report_lines.append("")
    report_lines.append(f"- decision: `{decision}`")
    report_lines.append(f"- reason: {decision_reason}")
    report_lines.append("")
    report_lines.append("## Current Bar")
    report_lines.append("")
    report_lines.append(f"- net profit: `{float(args.current_bar_net_bnb):.6f}` BNB")
    report_lines.append(f"- profit per 500: `{float(args.current_bar_per500):.6f}`")
    report_lines.append(f"- max drawdown: `{float(args.current_bar_max_dd_bnb):.6f}` BNB")
    report_lines.append("")
    report_lines.append("## Best Late-Pool Row")
    report_lines.append("")
    report_lines.append(f"- direction source: `{best_row['direction_source']}`")
    report_lines.append(f"- train size: `{int(best_row['train_size'])}`")
    report_lines.append(f"- bet size: `{float(best_row['bet_size_bnb']):.2f}`")
    report_lines.append(f"- net profit: `{float(best_row['net_profit_bnb']):.6f}` BNB")
    report_lines.append(f"- profit per 500: `{float(best_row['profit_per_500_bnb']):.6f}`")
    report_lines.append(f"- bet rate: `{100.0 * float(best_row['bet_rate']):.3f}%`")
    report_lines.append(f"- win rate: `{100.0 * float(best_row['win_rate']):.3f}%`")
    report_lines.append(f"- max drawdown: `{float(best_row['max_drawdown_bnb']):.6f}` BNB")
    report_lines.append(f"- test late-total MAE: `{float(best_row['test_late_total_mae']):.6f}`")
    report_lines.append(f"- test late-frac MAE: `{float(best_row['test_late_bull_frac_mae']):.6f}`")
    report_lines.append("")
    report_lines.append("## Source Best Rows")
    report_lines.append("")
    report_lines.append("| Direction Source | Train | Stake | Net BNB | Net / 500 | Bet rate | Win rate | Max DD | LateTotal MAE | LateFrac MAE |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for source in direction_sources:
        row = source_best_row_by_name.get(str(source))
        if row is None:
            continue
        report_lines.append(
            f"| {row['direction_source']} | {int(row['train_size'])} | {float(row['bet_size_bnb']):.2f} | "
            f"{float(row['net_profit_bnb']):.6f} | {float(row['profit_per_500_bnb']):.6f} | "
            f"{100.0 * float(row['bet_rate']):.3f}% | {100.0 * float(row['win_rate']):.3f}% | "
            f"{float(row['max_drawdown_bnb']):.6f} | {float(row['test_late_total_mae']):.6f} | "
            f"{float(row['test_late_bull_frac_mae']):.6f} |"
        )
    report_lines.append("")
    report_lines.append("## All Final Rows")
    report_lines.append("")
    report_lines.append("| Direction Source | Train | Stake | Net BNB | Net / 500 | Bet rate | Win rate | Max DD |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(
        final_rows,
        key=lambda item: (
            float(item["net_profit_bnb"]),
            -float(item["max_drawdown_bnb"]),
            float(item["profit_per_500_bnb"]),
        ),
        reverse=True,
    ):
        report_lines.append(
            f"| {row['direction_source']} | {int(row['train_size'])} | {float(row['bet_size_bnb']):.2f} | "
            f"{float(row['net_profit_bnb']):.6f} | {float(row['profit_per_500_bnb']):.6f} | "
            f"{100.0 * float(row['bet_rate']):.3f}% | {100.0 * float(row['win_rate']):.3f}% | "
            f"{float(row['max_drawdown_bnb']):.6f} |"
        )
    report_lines.append("")
    report_lines.append("## Robustness Of Best Config")
    report_lines.append("")
    report_lines.append("| Offset | Net BNB | Net / 500 | Bet rate | Win rate | Max DD | LateTotal MAE | LateFrac MAE |")
    report_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(robustness_rows, key=lambda item: int(item["robustness_offset_rounds"])):
        report_lines.append(
            f"| {int(row['robustness_offset_rounds'])} | {float(row['net_profit_bnb']):.6f} | "
            f"{float(row['profit_per_500_bnb']):.6f} | {100.0 * float(row['bet_rate']):.3f}% | "
            f"{100.0 * float(row['win_rate']):.3f}% | {float(row['max_drawdown_bnb']):.6f} | "
            f"{float(row['test_late_total_mae']):.6f} | {float(row['test_late_bull_frac_mae']):.6f} |"
        )
    report_lines.append("")
    report_lines.append("## Plots")
    report_lines.append("")
    report_lines.append(f"- [source best cumulative]({source_overlay_plot})")
    report_lines.append(f"- [source best rolling]({source_roll_plot})")
    report_lines.append(f"- [best cumulative]({best_plot})")
    report_lines.append(f"- [best robustness cumulative]({robust_overlay_plot})")
    report_path = output_dir / f"{args.name_prefix}_late_pool_bounded_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")

    print(
        json.dumps(
            {
                "phase": "done",
                "decision": str(decision),
                "best_direction_source": str(best_row["direction_source"]),
                "best_train_size": int(best_row["train_size"]),
                "best_bet_size_bnb": float(best_row["bet_size_bnb"]),
                "best_net_profit_bnb": float(best_row["net_profit_bnb"]),
                "best_profit_per_500_bnb": float(best_row["profit_per_500_bnb"]),
                "report_path": str(report_path),
                "summary_path": str(summary_out),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
