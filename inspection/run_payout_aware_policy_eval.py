from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from pancakebot.config.load_config import load_app_config
from pancakebot.core.constants import GAS_COST_BET_BNB
from inspection.neural_direction_eval_common import (
    load_recent_direction_eval_slice,
    parse_positive_int_list,
    rows_path,
    summary_path,
)
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.direction_tree_model import (
    load_direction_tree_bundle,
    predict_direction_tree_probabilities,
)
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
    naive_cutoff_profit_if_side_wins,
    realized_profit_for_side,
    simulate_payout_aware_policy,
    tune_side_thresholds,
)
from pancakebot.domain.models.payout_aware_tree_model import (
    PayoutAwareTreeConfig,
    predict_payout_aware_tree_values,
    save_payout_aware_tree_bundle,
    train_payout_aware_tree_regressor,
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
class PayoutAwarePolicyEvalRow:
    payout_model_type: str
    target_mode: str
    direction_source: str
    sim_size: int
    tail_offset_rounds: int
    train_size: int
    valid_size: int
    bet_size_bnb: float
    valid_min_bet_rate: float
    bull_threshold: float
    bear_threshold: float
    bull_bundle_path: str
    bear_bundle_path: str
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--manifest-csv", type=str, required=True)
    parser.add_argument("--payout-model-types", type=str, default="catboost,lightgbm")
    parser.add_argument("--target-mode", type=str, choices=("direct_net", "win_profit_residual"), default="direct_net")
    parser.add_argument(
        "--direction-source",
        type=str,
        default="mlp",
        choices=("mlp", "catboost", "lightgbm", "tcn", "soft_mean_all", "mean2_mlp_catboost"),
    )
    parser.add_argument("--train-sizes", type=str, default="100000,200000,400000")
    parser.add_argument("--sim-size", type=int, default=50000)
    parser.add_argument("--valid-size", type=int, default=3000)
    parser.add_argument("--tail-offset-rounds", type=int, default=0)
    parser.add_argument("--bet-size-bnb", type=float, default=0.05)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=50.0)
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="-0.020,-0.010,-0.005,0.000,0.001,0.0025,0.005,0.0075,0.010,0.015,0.020",
    )
    parser.add_argument(
        "--threshold-quantiles",
        type=str,
        default="0.50,0.75,0.90,0.95,0.98,0.99",
    )
    parser.add_argument("--valid-min-bet-rate", type=float, default=0.005)
    parser.add_argument("--random-seed", type=int, default=20260403)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--subsample", type=float, default=0.80)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _parse_str_list(raw: str) -> list[str]:
    out = [str(token).strip() for token in str(raw).split(",") if str(token).strip() != ""]
    if not out:
        raise InvariantError("payout_aware_policy_str_list_empty")
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        out.append(float(text))
    if not out:
        raise InvariantError("payout_aware_policy_float_list_empty")
    return out


def _candidate_thresholds_from_predictions(
    *,
    predicted_values: np.ndarray,
    explicit_grid: list[float],
    quantiles: list[float],
) -> list[float]:
    preds = np.asarray(predicted_values, dtype=np.float32)
    if preds.ndim != 1:
        raise InvariantError("payout_aware_policy_candidate_threshold_rank_invalid")
    candidates = {float(value) for value in explicit_grid}
    for quantile in quantiles:
        q = float(quantile)
        if not (0.0 <= float(q) <= 1.0):
            raise InvariantError("payout_aware_policy_threshold_quantile_invalid")
        candidates.add(float(np.quantile(preds, q)))
    return sorted(float(value) for value in candidates)


def _load_base_model_jobs(*, manifest_csv: str) -> list[_BaseModelJob]:
    manifest_path = Path(str(manifest_csv)).resolve()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise InvariantError("payout_aware_policy_manifest_empty")
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
        raise InvariantError("payout_aware_policy_split_len_mismatch")
    tail = tuple(int(epoch) for epoch in target_epochs[-int(needed) :])
    train_epochs = tuple(int(epoch) for epoch in tail[: int(train_size)])
    valid_epochs = tuple(
        int(epoch)
        for epoch in tail[int(train_size) : int(train_size) + int(valid_size)]
    )
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
        dataset = select_feature_columns_exact(
            dataset=eval_dataset,
            feature_columns=tuple(bundle.feature_columns),
        )
        probs_all = predict_neural_direction_probabilities(
            bundle=bundle,
            feature_matrix=np.asarray(dataset.feature_matrix, dtype=np.float32),
        )
        index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
        idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in target_epochs], dtype=np.int64)
        return np.asarray(probs_all[idx], dtype=np.float32)
    if str(job.model_type) == "tcn":
        if job.seq_len is None:
            raise InvariantError("payout_aware_policy_tcn_seq_len_missing")
        bundle = load_neural_direction_tcn_bundle(str(job.bundle_path))
        dataset = select_feature_columns_exact(
            dataset=eval_dataset,
            feature_columns=tuple(bundle.feature_columns),
        )
        chunk_size = 8192
        probs_parts: list[np.ndarray] = []
        for start_idx in range(0, int(len(target_epochs)), int(chunk_size)):
            epoch_chunk = tuple(
                int(epoch)
                for epoch in target_epochs[int(start_idx) : int(start_idx) + int(chunk_size)]
            )
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
        dataset = select_feature_columns_exact(
            dataset=eval_dataset,
            feature_columns=tuple(bundle.feature_columns),
        )
        probs_all = predict_direction_tree_probabilities(
            bundle=bundle,
            feature_matrix=np.asarray(dataset.feature_matrix, dtype=np.float32),
        )
        index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
        idx = np.asarray([int(index_by_epoch[int(epoch)]) for epoch in target_epochs], dtype=np.int64)
        return np.asarray(probs_all[idx], dtype=np.float32)
    raise InvariantError("payout_aware_policy_model_type_unknown")


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
            raise InvariantError("payout_aware_policy_direction_source_missing_models")
        return np.asarray(
            (
                np.asarray(probs_by_model["mlp"], dtype=np.float32)
                + np.asarray(probs_by_model["catboost"], dtype=np.float32)
            )
            / 2.0,
            dtype=np.float32,
        )
    raise InvariantError("payout_aware_policy_direction_source_unknown")


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg = load_app_config(str(args.config))
    constants = load_contract_constants()
    if float(args.bet_size_bnb) < float(constants.min_bet_amount_bnb):
        raise InvariantError("payout_aware_policy_bet_size_below_min_bet")
    if float(args.initial_bankroll_bnb) <= 0.0:
        raise InvariantError("payout_aware_policy_initial_bankroll_nonpositive")
    payout_model_types = [str(value) for value in _parse_str_list(args.payout_model_types)]
    train_sizes = [int(value) for value in parse_positive_int_list(args.train_sizes)]
    threshold_grid = [float(value) for value in _parse_float_list(args.threshold_grid)]
    threshold_quantiles = [float(value) for value in _parse_float_list(args.threshold_quantiles)]
    base_jobs = _load_base_model_jobs(manifest_csv=str(args.manifest_csv))
    max_train_size = max(int(value) for value in train_sizes)
    max_seq_warmup = max((int(job.seq_len) - 1) for job in base_jobs if job.seq_len is not None) if any(job.seq_len is not None for job in base_jobs) else 0
    required_examples = int(max_train_size) + int(args.valid_size) + int(args.sim_size) + int(max_seq_warmup)
    eval_slice = load_recent_direction_eval_slice(
        config_path=str(args.config),
        required_examples=int(required_examples),
        tail_offset_rounds=int(args.tail_offset_rounds),
    )
    max_target_epochs = tuple(
        int(epoch)
        for epoch in eval_slice.dataset.target_epochs[
            -int(max_train_size + int(args.valid_size) + int(args.sim_size)) :
        ]
    )
    combined_x, combined_columns, probs_by_model = _build_feature_matrix(
        dataset=eval_slice.dataset,
        target_epochs=max_target_epochs,
        base_jobs=base_jobs,
    )
    rounds_by_epoch = {
        int(epoch): eval_slice.target_rounds_by_epoch[int(epoch)]
        for epoch in max_target_epochs
    }
    bull_targets_direct = np.asarray(
        [
            realized_profit_for_side(
                round_closed=rounds_by_epoch[int(epoch)],
                bet_size_bnb=float(args.bet_size_bnb),
                bet_side="Bull",
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
            )
            for epoch in max_target_epochs
        ],
        dtype=np.float32,
    )
    bear_targets_direct = np.asarray(
        [
            realized_profit_for_side(
                round_closed=rounds_by_epoch[int(epoch)],
                bet_size_bnb=float(args.bet_size_bnb),
                bet_side="Bear",
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
            )
            for epoch in max_target_epochs
        ],
        dtype=np.float32,
    )
    bull_naive_win_profit = np.asarray(
        [
            naive_cutoff_profit_if_side_wins(
                round_closed=rounds_by_epoch[int(epoch)],
                bet_size_bnb=float(args.bet_size_bnb),
                bet_side="Bull",
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
                cutoff_seconds=int(runtime_cfg.cutoff_seconds),
            )
            for epoch in max_target_epochs
        ],
        dtype=np.float32,
    )
    bear_naive_win_profit = np.asarray(
        [
            naive_cutoff_profit_if_side_wins(
                round_closed=rounds_by_epoch[int(epoch)],
                bet_size_bnb=float(args.bet_size_bnb),
                bet_side="Bear",
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
                cutoff_seconds=int(runtime_cfg.cutoff_seconds),
            )
            for epoch in max_target_epochs
        ],
        dtype=np.float32,
    )
    bull_target_residual = np.asarray(
        bull_targets_direct - bull_naive_win_profit,
        dtype=np.float32,
    )
    bear_target_residual = np.asarray(
        bear_targets_direct - bear_naive_win_profit,
        dtype=np.float32,
    )
    round_labels = np.asarray(
        [
            1 if str(rounds_by_epoch[int(epoch)].position) == "Bull" else 0
            for epoch in max_target_epochs
        ],
        dtype=np.int64,
    )
    direction_probs_all = _direction_probabilities_for_source(
        probs_by_model=probs_by_model,
        source=str(args.direction_source),
    )
    loss_const_bnb = -float(args.bet_size_bnb) - float(GAS_COST_BET_BNB)

    rows: list[PayoutAwarePolicyEvalRow] = []
    trace_rows: list[dict[str, object]] = []
    for payout_model_type in payout_model_types:
        model_cfg = PayoutAwareTreeConfig(
            model_type=str(payout_model_type),
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
        for train_size in train_sizes:
            train_epochs, valid_epochs, test_epochs = _split_target_epochs(
                target_epochs=max_target_epochs,
                train_size=int(train_size),
                valid_size=int(args.valid_size),
                sim_size=int(args.sim_size),
            )
            train_x = _rows_for_target_epochs(
                feature_matrix=combined_x,
                target_epochs_source=max_target_epochs,
                target_epochs=train_epochs,
            )
            valid_x = _rows_for_target_epochs(
                feature_matrix=combined_x,
                target_epochs_source=max_target_epochs,
                target_epochs=valid_epochs,
            )
            test_x = _rows_for_target_epochs(
                feature_matrix=combined_x,
                target_epochs_source=max_target_epochs,
                target_epochs=test_epochs,
            )
            train_bull_y_all = _targets_for_target_epochs(
                target_values=bull_targets_direct if str(args.target_mode) == "direct_net" else bull_target_residual,
                target_epochs_source=max_target_epochs,
                target_epochs=train_epochs,
            )
            valid_bull_y_all = _targets_for_target_epochs(
                target_values=bull_targets_direct if str(args.target_mode) == "direct_net" else bull_target_residual,
                target_epochs_source=max_target_epochs,
                target_epochs=valid_epochs,
            )
            train_bear_y_all = _targets_for_target_epochs(
                target_values=bear_targets_direct if str(args.target_mode) == "direct_net" else bear_target_residual,
                target_epochs_source=max_target_epochs,
                target_epochs=train_epochs,
            )
            valid_bear_y_all = _targets_for_target_epochs(
                target_values=bear_targets_direct if str(args.target_mode) == "direct_net" else bear_target_residual,
                target_epochs_source=max_target_epochs,
                target_epochs=valid_epochs,
            )
            train_labels = _targets_for_target_epochs(
                target_values=round_labels.astype(np.float32),
                target_epochs_source=max_target_epochs,
                target_epochs=train_epochs,
            ).astype(np.int64)
            valid_labels = _targets_for_target_epochs(
                target_values=round_labels.astype(np.float32),
                target_epochs_source=max_target_epochs,
                target_epochs=valid_epochs,
            ).astype(np.int64)
            if str(args.target_mode) == "direct_net":
                bull_train_x_fit = train_x
                bear_train_x_fit = train_x
                bull_valid_x_fit = valid_x
                bear_valid_x_fit = valid_x
                train_bull_y_fit = train_bull_y_all
                valid_bull_y_fit = valid_bull_y_all
                train_bear_y_fit = train_bear_y_all
                valid_bear_y_fit = valid_bear_y_all
            else:
                bull_train_mask = np.asarray(train_labels == 1, dtype=bool)
                bull_valid_mask = np.asarray(valid_labels == 1, dtype=bool)
                bear_train_mask = np.asarray(train_labels == 0, dtype=bool)
                bear_valid_mask = np.asarray(valid_labels == 0, dtype=bool)
                if int(np.sum(bull_train_mask)) <= 0 or int(np.sum(bear_train_mask)) <= 0:
                    raise InvariantError("payout_aware_policy_residual_train_mask_empty")
                if int(np.sum(bull_valid_mask)) <= 0 or int(np.sum(bear_valid_mask)) <= 0:
                    raise InvariantError("payout_aware_policy_residual_valid_mask_empty")
                bull_train_x_fit = np.asarray(train_x[bull_train_mask], dtype=np.float32)
                bull_valid_x_fit = np.asarray(valid_x[bull_valid_mask], dtype=np.float32)
                bear_train_x_fit = np.asarray(train_x[bear_train_mask], dtype=np.float32)
                bear_valid_x_fit = np.asarray(valid_x[bear_valid_mask], dtype=np.float32)
                train_bull_y_fit = np.asarray(train_bull_y_all[bull_train_mask], dtype=np.float32)
                valid_bull_y_fit = np.asarray(valid_bull_y_all[bull_valid_mask], dtype=np.float32)
                train_bear_y_fit = np.asarray(train_bear_y_all[bear_train_mask], dtype=np.float32)
                valid_bear_y_fit = np.asarray(valid_bear_y_all[bear_valid_mask], dtype=np.float32)
            bull_bundle = train_payout_aware_tree_regressor(
                feature_columns=combined_columns,
                train_x=bull_train_x_fit,
                train_y=train_bull_y_fit,
                valid_x=bull_valid_x_fit,
                valid_y=valid_bull_y_fit,
                random_seed=int(args.random_seed),
                config=model_cfg,
                metadata={
                    "target_name": "net_if_bull_bet"
                    if str(args.target_mode) == "direct_net"
                    else "bull_win_profit_residual",
                    "target_mode": str(args.target_mode),
                    "bet_size_bnb": float(args.bet_size_bnb),
                    "source_manifest_csv": str(Path(str(args.manifest_csv)).resolve()),
                    "longstream_sim_size": int(args.sim_size),
                },
            )
            bear_bundle = train_payout_aware_tree_regressor(
                feature_columns=combined_columns,
                train_x=bear_train_x_fit,
                train_y=train_bear_y_fit,
                valid_x=bear_valid_x_fit,
                valid_y=valid_bear_y_fit,
                random_seed=int(args.random_seed) + 1,
                config=model_cfg,
                metadata={
                    "target_name": "net_if_bear_bet"
                    if str(args.target_mode) == "direct_net"
                    else "bear_win_profit_residual",
                    "target_mode": str(args.target_mode),
                    "bet_size_bnb": float(args.bet_size_bnb),
                    "source_manifest_csv": str(Path(str(args.manifest_csv)).resolve()),
                    "longstream_sim_size": int(args.sim_size),
                },
            )
            bull_bundle_path = (
                output_dir
                / (
                    f"{str(args.name_prefix)}_tail{int(args.sim_size)}_off{int(args.tail_offset_rounds):05d}"
                    f"_train{int(train_size)}_{str(payout_model_type)}_bull_payout_tree.pkl"
                )
            ).resolve()
            bear_bundle_path = (
                output_dir
                / (
                    f"{str(args.name_prefix)}_tail{int(args.sim_size)}_off{int(args.tail_offset_rounds):05d}"
                    f"_train{int(train_size)}_{str(payout_model_type)}_bear_payout_tree.pkl"
                )
            ).resolve()
            save_payout_aware_tree_bundle(bundle=bull_bundle, path=str(bull_bundle_path))
            save_payout_aware_tree_bundle(bundle=bear_bundle, path=str(bear_bundle_path))

            valid_pred_bull = predict_payout_aware_tree_values(bundle=bull_bundle, feature_matrix=valid_x)
            valid_pred_bear = predict_payout_aware_tree_values(bundle=bear_bundle, feature_matrix=valid_x)
            test_pred_bull = predict_payout_aware_tree_values(bundle=bull_bundle, feature_matrix=test_x)
            test_pred_bear = predict_payout_aware_tree_values(bundle=bear_bundle, feature_matrix=test_x)
            if str(args.target_mode) == "win_profit_residual":
                valid_naive_bull = _targets_for_target_epochs(
                    target_values=bull_naive_win_profit,
                    target_epochs_source=max_target_epochs,
                    target_epochs=valid_epochs,
                )
                valid_naive_bear = _targets_for_target_epochs(
                    target_values=bear_naive_win_profit,
                    target_epochs_source=max_target_epochs,
                    target_epochs=valid_epochs,
                )
                test_naive_bull = _targets_for_target_epochs(
                    target_values=bull_naive_win_profit,
                    target_epochs_source=max_target_epochs,
                    target_epochs=test_epochs,
                )
                test_naive_bear = _targets_for_target_epochs(
                    target_values=bear_naive_win_profit,
                    target_epochs_source=max_target_epochs,
                    target_epochs=test_epochs,
                )
                valid_p_bull = _targets_for_target_epochs(
                    target_values=direction_probs_all,
                    target_epochs_source=max_target_epochs,
                    target_epochs=valid_epochs,
                )
                test_p_bull = _targets_for_target_epochs(
                    target_values=direction_probs_all,
                    target_epochs_source=max_target_epochs,
                    target_epochs=test_epochs,
                )
                valid_est_win_bull = np.asarray(valid_naive_bull + valid_pred_bull, dtype=np.float32)
                valid_est_win_bear = np.asarray(valid_naive_bear + valid_pred_bear, dtype=np.float32)
                test_est_win_bull = np.asarray(test_naive_bull + test_pred_bull, dtype=np.float32)
                test_est_win_bear = np.asarray(test_naive_bear + test_pred_bear, dtype=np.float32)
                valid_pred_bull = np.asarray(
                    valid_p_bull * valid_est_win_bull + (1.0 - valid_p_bull) * float(loss_const_bnb),
                    dtype=np.float32,
                )
                valid_pred_bear = np.asarray(
                    (1.0 - valid_p_bull) * valid_est_win_bear + valid_p_bull * float(loss_const_bnb),
                    dtype=np.float32,
                )
                test_pred_bull = np.asarray(
                    test_p_bull * test_est_win_bull + (1.0 - test_p_bull) * float(loss_const_bnb),
                    dtype=np.float32,
                )
                test_pred_bear = np.asarray(
                    (1.0 - test_p_bull) * test_est_win_bear + test_p_bull * float(loss_const_bnb),
                    dtype=np.float32,
                )
            valid_rounds = _rounds_for_target_epochs(rounds_by_epoch=rounds_by_epoch, target_epochs=valid_epochs)
            bull_threshold_candidates = _candidate_thresholds_from_predictions(
                predicted_values=valid_pred_bull,
                explicit_grid=threshold_grid,
                quantiles=threshold_quantiles,
            )
            bear_threshold_candidates = _candidate_thresholds_from_predictions(
                predicted_values=valid_pred_bear,
                explicit_grid=threshold_grid,
                quantiles=threshold_quantiles,
            )
            threshold_choice = tune_side_thresholds(
                rounds=valid_rounds,
                predicted_ev_bull=valid_pred_bull,
                predicted_ev_bear=valid_pred_bear,
                threshold_grid=sorted(set(bull_threshold_candidates + bear_threshold_candidates)),
                bet_size_bnb=float(args.bet_size_bnb),
                initial_bankroll_bnb=float(args.initial_bankroll_bnb),
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
                min_bet_rate=float(args.valid_min_bet_rate),
            )
            valid_result, _ = simulate_payout_aware_policy(
                rounds=valid_rounds,
                predicted_ev_bull=valid_pred_bull,
                predicted_ev_bear=valid_pred_bear,
                bull_threshold=float(threshold_choice.bull_threshold),
                bear_threshold=float(threshold_choice.bear_threshold),
                bet_size_bnb=float(args.bet_size_bnb),
                initial_bankroll_bnb=float(args.initial_bankroll_bnb),
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
            )
            test_rounds = _rounds_for_target_epochs(rounds_by_epoch=rounds_by_epoch, target_epochs=test_epochs)
            test_result, test_trace = simulate_payout_aware_policy(
                rounds=test_rounds,
                predicted_ev_bull=test_pred_bull,
                predicted_ev_bear=test_pred_bear,
                bull_threshold=float(threshold_choice.bull_threshold),
                bear_threshold=float(threshold_choice.bear_threshold),
                bet_size_bnb=float(args.bet_size_bnb),
                initial_bankroll_bnb=float(args.initial_bankroll_bnb),
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
            )
            rows.append(
                PayoutAwarePolicyEvalRow(
                    payout_model_type=str(payout_model_type),
                    target_mode=str(args.target_mode),
                    direction_source=str(args.direction_source),
                    sim_size=int(args.sim_size),
                    tail_offset_rounds=int(args.tail_offset_rounds),
                    train_size=int(train_size),
                    valid_size=int(args.valid_size),
                    bet_size_bnb=float(args.bet_size_bnb),
                    valid_min_bet_rate=float(args.valid_min_bet_rate),
                    bull_threshold=float(threshold_choice.bull_threshold),
                    bear_threshold=float(threshold_choice.bear_threshold),
                    bull_bundle_path=str(bull_bundle_path),
                    bear_bundle_path=str(bear_bundle_path),
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
                )
            )
            for trace in test_trace:
                idx_epoch = int(trace.target_epoch)
                trace_rows.append(
                    {
                        "payout_model_type": str(payout_model_type),
                        "target_mode": str(args.target_mode),
                        "direction_source": str(args.direction_source),
                        "train_size": int(train_size),
                        "sim_size": int(args.sim_size),
                        "tail_offset_rounds": int(args.tail_offset_rounds),
                        "target_epoch": int(trace.target_epoch),
                        "round_position": str(rounds_by_epoch[idx_epoch].position),
                        "predicted_ev_bull": float(trace.predicted_ev_bull),
                        "predicted_ev_bear": float(trace.predicted_ev_bear),
                        "bull_threshold": float(trace.bull_threshold),
                        "bear_threshold": float(trace.bear_threshold),
                        "action": str(trace.action),
                        "selected_side": trace.selected_side,
                        "selected_predicted_ev": trace.selected_predicted_ev,
                        "actual_net_bull": float(
                            realized_profit_for_side(
                                round_closed=rounds_by_epoch[idx_epoch],
                                bet_size_bnb=float(args.bet_size_bnb),
                                bet_side="Bull",
                                treasury_fee_fraction=float(constants.treasury_fee_fraction),
                            )
                        ),
                        "actual_net_bear": float(
                            realized_profit_for_side(
                                round_closed=rounds_by_epoch[idx_epoch],
                                bet_size_bnb=float(args.bet_size_bnb),
                                bet_side="Bear",
                                treasury_fee_fraction=float(constants.treasury_fee_fraction),
                            )
                        ),
                        "realized_profit_bnb": float(trace.realized_profit_bnb),
                        "cumulative_profit_bnb": float(trace.cumulative_profit_bnb),
                        "bankroll_bnb": float(trace.bankroll_bnb),
                        "outcome": trace.outcome,
                    }
                )
            print(
                {
                    "phase": "config_done",
                    "payout_model_type": str(payout_model_type),
                    "target_mode": str(args.target_mode),
                    "direction_source": str(args.direction_source),
                    "train_size": int(train_size),
                    "test_profit_per_500_bnb": float(test_result.profit_per_500_bnb),
                    "test_net_profit_bnb": float(test_result.net_profit_bnb),
                    "test_bet_rate": float(test_result.bet_rate),
                    "valid_profit_per_500_bnb": float(valid_result.profit_per_500_bnb),
                    "bull_threshold": float(threshold_choice.bull_threshold),
                    "bear_threshold": float(threshold_choice.bear_threshold),
                },
                flush=True,
            )

    if not rows:
        raise InvariantError("payout_aware_policy_rows_empty")
    rows_out = rows_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="payout_aware_policy_rows")
    with rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    trace_out = rows_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="payout_aware_policy_trace_rows")
    with trace_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        for row in trace_rows:
            writer.writerow(row)
    best_row = max(
        rows,
        key=lambda row: (
            float(row.net_profit_bnb),
            float(row.profit_per_500_bnb),
            float(row.bet_rate),
        ),
    )
    summary_payload = {
        "manifest_csv_path": str(Path(str(args.manifest_csv)).resolve()),
        "rows_csv_path": str(rows_out),
        "trace_csv_path": str(trace_out),
        "target_mode": str(args.target_mode),
        "direction_source": str(args.direction_source),
        "sim_size": int(args.sim_size),
        "tail_offset_rounds": int(args.tail_offset_rounds),
        "train_sizes": [int(value) for value in train_sizes],
        "payout_model_types": [str(value) for value in payout_model_types],
        "bet_size_bnb": float(args.bet_size_bnb),
        "initial_bankroll_bnb": float(args.initial_bankroll_bnb),
        "valid_min_bet_rate": float(args.valid_min_bet_rate),
        "threshold_grid": [float(value) for value in threshold_grid],
        "threshold_quantiles": [float(value) for value in threshold_quantiles],
        "best_row": asdict(best_row),
        "rows": [asdict(row) for row in rows],
    }
    summary_out = summary_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="payout_aware_policy_summary")
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
