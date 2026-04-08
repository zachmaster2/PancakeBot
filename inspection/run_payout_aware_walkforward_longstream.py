from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inspection.neural_direction_eval_common import load_recent_direction_eval_slice, rows_path, summary_path
from inspection.run_payout_aware_policy_eval import (
    _build_feature_matrix,
    _candidate_thresholds_from_predictions,
    _direction_probabilities_for_source,
    _load_base_model_jobs,
    _rounds_for_target_epochs,
    _rows_for_target_epochs,
    _targets_for_target_epochs,
)
from pancakebot.config.load_config import load_app_config
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.payout_aware_policy import (
    naive_cutoff_profit_if_side_wins,
    realized_profit_for_side,
    simulate_payout_aware_policy,
    tune_side_thresholds,
)
from pancakebot.domain.models.payout_aware_tree_model import (
    PayoutAwareTreeConfig,
    predict_payout_aware_tree_values,
    train_payout_aware_tree_regressor,
)
from pancakebot.runtime.contract_constants_cache import load_contract_constants

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class WalkforwardBlockRow:
    block_index: int
    retrain_train_start_epoch: int
    retrain_train_end_epoch: int
    retrain_valid_start_epoch: int
    retrain_valid_end_epoch: int
    test_start_epoch: int
    test_end_epoch: int
    train_size: int
    valid_size: int
    block_size: int
    bull_threshold: float
    bear_threshold: float
    block_num_rounds: int
    block_num_bets: int
    block_num_wins: int
    block_num_losses: int
    block_bet_rate: float
    block_win_rate: float
    block_net_profit_bnb: float
    block_profit_per_500_bnb: float
    cumulative_num_rounds: int
    cumulative_num_bets: int
    cumulative_num_wins: int
    cumulative_num_losses: int
    cumulative_bet_rate: float
    cumulative_win_rate: float
    cumulative_net_profit_bnb: float
    cumulative_profit_per_500_bnb: float
    cumulative_max_drawdown_bnb: float
    bankroll_bnb: float
    valid_num_bets: int
    valid_bet_rate: float
    valid_net_profit_bnb: float
    valid_profit_per_500_bnb: float
    valid_win_rate: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument(
        "--manifest-csv",
        type=str,
        default="../PancakeBot_var_exp/direction_ensemble_longstream_manifest_20260403.csv",
    )
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--payout-model-type", type=str, choices=("catboost", "lightgbm"), default="catboost")
    parser.add_argument("--target-mode", type=str, choices=("direct_net", "win_profit_residual"), default="win_profit_residual")
    parser.add_argument(
        "--direction-source",
        type=str,
        default="mlp",
        choices=("mlp", "catboost", "lightgbm", "tcn", "soft_mean_all", "mean2_mlp_catboost"),
    )
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--valid-size", type=int, default=3000)
    parser.add_argument("--sim-size", type=int, default=50000)
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--bet-size-bnb", type=float, default=0.05)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=50.0)
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="-0.020,-0.010,-0.005,0.000,0.001,0.0025,0.005,0.0075,0.010,0.015,0.020",
    )
    parser.add_argument("--threshold-quantiles", type=str, default="0.50,0.75,0.90,0.95,0.98,0.99")
    parser.add_argument("--valid-min-bet-rate", type=float, default=0.005)
    parser.add_argument("--tail-offset-rounds", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260404)
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


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        out.append(float(text))
    if not out:
        raise InvariantError("payout_walkforward_float_list_empty")
    return out


def _plot_cumulative(*, trace_rows: list[dict[str, object]], output_path: Path, title: str) -> None:
    xs = np.asarray([int(row["target_epoch"]) for row in trace_rows], dtype=np.int64)
    ys = np.asarray([float(row["global_cumulative_profit_bnb"]) for row in trace_rows], dtype=np.float32)
    plt.figure(figsize=(14, 8))
    plt.plot(xs, ys, linewidth=2)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Target Round Epoch")
    plt.ylabel("Cumulative Profit (BNB)")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_rolling(*, trace_rows: list[dict[str, object]], output_path: Path, title: str, window_rounds: int = 2000) -> None:
    realized = np.asarray([float(row["realized_profit_bnb"]) for row in trace_rows], dtype=np.float32)
    xs_full = np.asarray([int(row["target_epoch"]) for row in trace_rows], dtype=np.int64)
    if int(len(realized)) < int(window_rounds):
        return
    ys = np.convolve(realized, np.ones(int(window_rounds), dtype=np.float32), mode="valid") * 500.0 / float(window_rounds)
    xs = xs_full[int(window_rounds) - 1 :]
    plt.figure(figsize=(14, 8))
    plt.plot(xs, ys, linewidth=1.8)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Target Round Epoch")
    plt.ylabel(f"Rolling Net / 500 (window={int(window_rounds)})")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.train_size) <= 0 or int(args.valid_size) <= 0 or int(args.sim_size) <= 0 or int(args.block_size) <= 0:
        raise InvariantError("payout_walkforward_window_nonpositive")
    if int(args.sim_size) % int(args.block_size) != 0:
        raise InvariantError("payout_walkforward_sim_block_mismatch")
    if float(args.initial_bankroll_bnb) <= 0.0:
        raise InvariantError("payout_walkforward_initial_bankroll_nonpositive")

    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = load_app_config(str(args.config))
    constants = load_contract_constants()
    if float(args.bet_size_bnb) < float(constants.min_bet_amount_bnb):
        raise InvariantError("payout_walkforward_bet_size_below_min_bet")

    threshold_grid = _parse_float_list(args.threshold_grid)
    threshold_quantiles = _parse_float_list(args.threshold_quantiles)
    model_cfg = PayoutAwareTreeConfig(
        model_type=str(args.payout_model_type),
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

    base_jobs = _load_base_model_jobs(manifest_csv=str(args.manifest_csv))
    max_seq_warmup = (
        max((int(job.seq_len) - 1) for job in base_jobs if job.seq_len is not None)
        if any(job.seq_len is not None for job in base_jobs)
        else 0
    )
    required_examples = int(args.train_size) + int(args.valid_size) + int(args.sim_size) + int(max_seq_warmup)
    eval_slice = load_recent_direction_eval_slice(
        config_path=str(args.config),
        required_examples=int(required_examples),
        tail_offset_rounds=int(args.tail_offset_rounds),
    )
    raw_target_epochs = tuple(int(epoch) for epoch in eval_slice.dataset.target_epochs)
    target_epochs = (
        tuple(int(epoch) for epoch in raw_target_epochs[int(max_seq_warmup) :])
        if int(max_seq_warmup) > 0
        else raw_target_epochs
    )
    if len(target_epochs) != int(args.train_size) + int(args.valid_size) + int(args.sim_size):
        raise InvariantError("payout_walkforward_target_len_mismatch")

    combined_x, combined_columns, probs_by_model = _build_feature_matrix(
        dataset=eval_slice.dataset,
        target_epochs=target_epochs,
        base_jobs=base_jobs,
    )
    rounds_by_epoch = {int(epoch): eval_slice.target_rounds_by_epoch[int(epoch)] for epoch in target_epochs}
    direction_probs_all = _direction_probabilities_for_source(
        probs_by_model=probs_by_model,
        source=str(args.direction_source),
    )

    bull_targets_direct = np.asarray(
        [
            realized_profit_for_side(
                round_closed=rounds_by_epoch[int(epoch)],
                bet_size_bnb=float(args.bet_size_bnb),
                bet_side="Bull",
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
            )
            for epoch in target_epochs
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
            for epoch in target_epochs
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
            for epoch in target_epochs
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
            for epoch in target_epochs
        ],
        dtype=np.float32,
    )
    bull_target_residual = np.asarray(bull_targets_direct - bull_naive_win_profit, dtype=np.float32)
    bear_target_residual = np.asarray(bear_targets_direct - bear_naive_win_profit, dtype=np.float32)
    round_labels = np.asarray(
        [1 if str(rounds_by_epoch[int(epoch)].position) == "Bull" else 0 for epoch in target_epochs],
        dtype=np.int64,
    )

    sim_epochs = tuple(int(epoch) for epoch in target_epochs[-int(args.sim_size) :])
    index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(target_epochs)}
    initial_bankroll = float(args.initial_bankroll_bnb)
    bankroll = float(initial_bankroll)
    peak_bankroll = float(initial_bankroll)
    max_drawdown = 0.0
    cumulative_rounds = 0
    cumulative_bets = 0
    cumulative_wins = 0
    cumulative_losses = 0
    block_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []

    num_blocks = int(args.sim_size) // int(args.block_size)
    loss_const_bnb = -float(args.bet_size_bnb) - float(GAS_COST_BET_BNB)

    for block_index in range(int(num_blocks)):
        test_epochs = tuple(
            int(epoch)
            for epoch in sim_epochs[int(block_index) * int(args.block_size) : (int(block_index) + 1) * int(args.block_size)]
        )
        if len(test_epochs) != int(args.block_size):
            raise InvariantError("payout_walkforward_block_len_mismatch")
        test_start_idx = int(index_by_epoch[int(test_epochs[0])])
        train_start_idx = int(test_start_idx) - int(args.train_size)
        valid_start_idx = int(train_start_idx) - int(args.valid_size)
        if int(valid_start_idx) < 0:
            raise InvariantError("payout_walkforward_history_insufficient")
        valid_epochs = tuple(int(epoch) for epoch in target_epochs[int(valid_start_idx) : int(train_start_idx)])
        train_epochs = tuple(int(epoch) for epoch in target_epochs[int(train_start_idx) : int(test_start_idx)])

        train_x = _rows_for_target_epochs(feature_matrix=combined_x, target_epochs_source=target_epochs, target_epochs=train_epochs)
        valid_x = _rows_for_target_epochs(feature_matrix=combined_x, target_epochs_source=target_epochs, target_epochs=valid_epochs)
        test_x = _rows_for_target_epochs(feature_matrix=combined_x, target_epochs_source=target_epochs, target_epochs=test_epochs)
        train_labels = _targets_for_target_epochs(
            target_values=round_labels.astype(np.float32),
            target_epochs_source=target_epochs,
            target_epochs=train_epochs,
        ).astype(np.int64)
        valid_labels = _targets_for_target_epochs(
            target_values=round_labels.astype(np.float32),
            target_epochs_source=target_epochs,
            target_epochs=valid_epochs,
        ).astype(np.int64)

        if str(args.target_mode) == "direct_net":
            train_bull_y_all = _targets_for_target_epochs(target_values=bull_targets_direct, target_epochs_source=target_epochs, target_epochs=train_epochs)
            valid_bull_y_all = _targets_for_target_epochs(target_values=bull_targets_direct, target_epochs_source=target_epochs, target_epochs=valid_epochs)
            train_bear_y_all = _targets_for_target_epochs(target_values=bear_targets_direct, target_epochs_source=target_epochs, target_epochs=train_epochs)
            valid_bear_y_all = _targets_for_target_epochs(target_values=bear_targets_direct, target_epochs_source=target_epochs, target_epochs=valid_epochs)
            bull_train_x_fit = train_x
            bull_valid_x_fit = valid_x
            bear_train_x_fit = train_x
            bear_valid_x_fit = valid_x
            train_bull_y_fit = train_bull_y_all
            valid_bull_y_fit = valid_bull_y_all
            train_bear_y_fit = train_bear_y_all
            valid_bear_y_fit = valid_bear_y_all
        else:
            train_bull_y_all = _targets_for_target_epochs(target_values=bull_target_residual, target_epochs_source=target_epochs, target_epochs=train_epochs)
            valid_bull_y_all = _targets_for_target_epochs(target_values=bull_target_residual, target_epochs_source=target_epochs, target_epochs=valid_epochs)
            train_bear_y_all = _targets_for_target_epochs(target_values=bear_target_residual, target_epochs_source=target_epochs, target_epochs=train_epochs)
            valid_bear_y_all = _targets_for_target_epochs(target_values=bear_target_residual, target_epochs_source=target_epochs, target_epochs=valid_epochs)
            bull_train_mask = np.asarray(train_labels == 1, dtype=bool)
            bull_valid_mask = np.asarray(valid_labels == 1, dtype=bool)
            bear_train_mask = np.asarray(train_labels == 0, dtype=bool)
            bear_valid_mask = np.asarray(valid_labels == 0, dtype=bool)
            if int(np.sum(bull_train_mask)) <= 0 or int(np.sum(bear_train_mask)) <= 0:
                raise InvariantError("payout_walkforward_residual_train_mask_empty")
            if int(np.sum(bull_valid_mask)) <= 0 or int(np.sum(bear_valid_mask)) <= 0:
                raise InvariantError("payout_walkforward_residual_valid_mask_empty")
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
            random_seed=int(args.random_seed) + int(block_index) * 2,
            config=model_cfg,
            metadata={
                "walkforward_block_index": int(block_index),
                "target_name": "net_if_bull_bet" if str(args.target_mode) == "direct_net" else "bull_win_profit_residual",
            },
        )
        bear_bundle = train_payout_aware_tree_regressor(
            feature_columns=combined_columns,
            train_x=bear_train_x_fit,
            train_y=train_bear_y_fit,
            valid_x=bear_valid_x_fit,
            valid_y=valid_bear_y_fit,
            random_seed=int(args.random_seed) + int(block_index) * 2 + 1,
            config=model_cfg,
            metadata={
                "walkforward_block_index": int(block_index),
                "target_name": "net_if_bear_bet" if str(args.target_mode) == "direct_net" else "bear_win_profit_residual",
            },
        )
        valid_pred_bull = predict_payout_aware_tree_values(bundle=bull_bundle, feature_matrix=valid_x)
        valid_pred_bear = predict_payout_aware_tree_values(bundle=bear_bundle, feature_matrix=valid_x)
        test_pred_bull = predict_payout_aware_tree_values(bundle=bull_bundle, feature_matrix=test_x)
        test_pred_bear = predict_payout_aware_tree_values(bundle=bear_bundle, feature_matrix=test_x)

        valid_p_bull = _targets_for_target_epochs(target_values=direction_probs_all, target_epochs_source=target_epochs, target_epochs=valid_epochs)
        test_p_bull = _targets_for_target_epochs(target_values=direction_probs_all, target_epochs_source=target_epochs, target_epochs=test_epochs)

        if str(args.target_mode) == "win_profit_residual":
            valid_naive_bull = _targets_for_target_epochs(target_values=bull_naive_win_profit, target_epochs_source=target_epochs, target_epochs=valid_epochs)
            valid_naive_bear = _targets_for_target_epochs(target_values=bear_naive_win_profit, target_epochs_source=target_epochs, target_epochs=valid_epochs)
            test_naive_bull = _targets_for_target_epochs(target_values=bull_naive_win_profit, target_epochs_source=target_epochs, target_epochs=test_epochs)
            test_naive_bear = _targets_for_target_epochs(target_values=bear_naive_win_profit, target_epochs_source=target_epochs, target_epochs=test_epochs)
            valid_est_win_bull = np.asarray(valid_naive_bull + valid_pred_bull, dtype=np.float32)
            valid_est_win_bear = np.asarray(valid_naive_bear + valid_pred_bear, dtype=np.float32)
            test_est_win_bull = np.asarray(test_naive_bull + test_pred_bull, dtype=np.float32)
            test_est_win_bear = np.asarray(test_naive_bear + test_pred_bear, dtype=np.float32)
            valid_pred_bull = np.asarray(valid_p_bull * valid_est_win_bull + (1.0 - valid_p_bull) * float(loss_const_bnb), dtype=np.float32)
            valid_pred_bear = np.asarray((1.0 - valid_p_bull) * valid_est_win_bear + valid_p_bull * float(loss_const_bnb), dtype=np.float32)
            test_pred_bull = np.asarray(test_p_bull * test_est_win_bull + (1.0 - test_p_bull) * float(loss_const_bnb), dtype=np.float32)
            test_pred_bear = np.asarray((1.0 - test_p_bull) * test_est_win_bear + test_p_bull * float(loss_const_bnb), dtype=np.float32)

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
        valid_rounds = _rounds_for_target_epochs(rounds_by_epoch=rounds_by_epoch, target_epochs=valid_epochs)
        threshold_choice = tune_side_thresholds(
            rounds=valid_rounds,
            predicted_ev_bull=valid_pred_bull,
            predicted_ev_bear=valid_pred_bear,
            threshold_grid=sorted(set(bull_threshold_candidates + bear_threshold_candidates)),
            bet_size_bnb=float(args.bet_size_bnb),
            initial_bankroll_bnb=float(bankroll),
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
            initial_bankroll_bnb=float(bankroll),
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
            initial_bankroll_bnb=float(bankroll),
            treasury_fee_fraction=float(constants.treasury_fee_fraction),
        )

        cumulative_net_before_block = float(bankroll) - float(initial_bankroll)
        for trace in test_trace:
            global_cumulative = float(cumulative_net_before_block) + float(trace.cumulative_profit_bnb)
            global_bankroll = float(initial_bankroll) + float(global_cumulative)
            peak_bankroll = max(float(peak_bankroll), float(global_bankroll))
            max_drawdown = max(float(max_drawdown), float(peak_bankroll) - float(global_bankroll))
            idx_epoch = int(trace.target_epoch)
            trace_rows.append(
                {
                    "block_index": int(block_index),
                    "target_epoch": int(trace.target_epoch),
                    "round_position": str(rounds_by_epoch[idx_epoch].position),
                    "predicted_ev_bull": float(trace.predicted_ev_bull),
                    "predicted_ev_bear": float(trace.predicted_ev_bear),
                    "bull_threshold": float(trace.bull_threshold),
                    "bear_threshold": float(trace.bear_threshold),
                    "action": str(trace.action),
                    "selected_side": trace.selected_side,
                    "selected_predicted_ev": trace.selected_predicted_ev,
                    "realized_profit_bnb": float(trace.realized_profit_bnb),
                    "block_cumulative_profit_bnb": float(trace.cumulative_profit_bnb),
                    "global_cumulative_profit_bnb": float(global_cumulative),
                    "global_bankroll_bnb": float(global_bankroll),
                    "outcome": trace.outcome,
                }
            )

        bankroll = float(test_result.final_bankroll_bnb)
        cumulative_rounds += int(test_result.num_rounds)
        cumulative_bets += int(test_result.num_bets)
        cumulative_wins += int(test_result.num_wins)
        cumulative_losses += int(test_result.num_losses)
        cumulative_net = float(bankroll) - float(initial_bankroll)
        row = asdict(
            WalkforwardBlockRow(
                block_index=int(block_index),
                retrain_train_start_epoch=int(train_epochs[0]),
                retrain_train_end_epoch=int(train_epochs[-1]),
                retrain_valid_start_epoch=int(valid_epochs[0]),
                retrain_valid_end_epoch=int(valid_epochs[-1]),
                test_start_epoch=int(test_epochs[0]),
                test_end_epoch=int(test_epochs[-1]),
                train_size=int(args.train_size),
                valid_size=int(args.valid_size),
                block_size=int(args.block_size),
                bull_threshold=float(threshold_choice.bull_threshold),
                bear_threshold=float(threshold_choice.bear_threshold),
                block_num_rounds=int(test_result.num_rounds),
                block_num_bets=int(test_result.num_bets),
                block_num_wins=int(test_result.num_wins),
                block_num_losses=int(test_result.num_losses),
                block_bet_rate=float(test_result.bet_rate),
                block_win_rate=float(test_result.win_rate),
                block_net_profit_bnb=float(test_result.net_profit_bnb),
                block_profit_per_500_bnb=float(test_result.profit_per_500_bnb),
                cumulative_num_rounds=int(cumulative_rounds),
                cumulative_num_bets=int(cumulative_bets),
                cumulative_num_wins=int(cumulative_wins),
                cumulative_num_losses=int(cumulative_losses),
                cumulative_bet_rate=0.0 if int(cumulative_rounds) <= 0 else float(cumulative_bets / float(cumulative_rounds)),
                cumulative_win_rate=0.0 if int(cumulative_bets) <= 0 else float(cumulative_wins / float(cumulative_bets)),
                cumulative_net_profit_bnb=float(cumulative_net),
                cumulative_profit_per_500_bnb=float(cumulative_net) * 500.0 / float(cumulative_rounds),
                cumulative_max_drawdown_bnb=float(max_drawdown),
                bankroll_bnb=float(bankroll),
                valid_num_bets=int(valid_result.num_bets),
                valid_bet_rate=float(valid_result.bet_rate),
                valid_net_profit_bnb=float(valid_result.net_profit_bnb),
                valid_profit_per_500_bnb=float(valid_result.profit_per_500_bnb),
                valid_win_rate=float(valid_result.win_rate),
            )
        )
        block_rows.append(row)
        print(
            json.dumps(
                {
                    "phase": "block_done",
                    "block_index": int(block_index),
                    "test_epoch_range": [int(test_epochs[0]), int(test_epochs[-1])],
                    "block_net_profit_bnb": float(test_result.net_profit_bnb),
                    "block_profit_per_500_bnb": float(test_result.profit_per_500_bnb),
                    "block_bet_rate": float(test_result.bet_rate),
                    "block_win_rate": float(test_result.win_rate),
                    "cumulative_net_profit_bnb": float(cumulative_net),
                    "cumulative_profit_per_500_bnb": float(cumulative_net) * 500.0 / float(cumulative_rounds),
                    "cumulative_bet_rate": row["cumulative_bet_rate"],
                    "cumulative_win_rate": row["cumulative_win_rate"],
                    "cumulative_max_drawdown_bnb": float(max_drawdown),
                    "bankroll_bnb": float(bankroll),
                    "bull_threshold": float(threshold_choice.bull_threshold),
                    "bear_threshold": float(threshold_choice.bear_threshold),
                }
            ),
            flush=True,
        )

    if not block_rows:
        raise InvariantError("payout_walkforward_rows_empty")

    rows_out = rows_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="walkforward_blocks")
    with rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(block_rows[0].keys()))
        writer.writeheader()
        for row in block_rows:
            writer.writerow(row)

    trace_out = rows_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="walkforward_trace_rows")
    with trace_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        for row in trace_rows:
            writer.writerow(row)

    best_plot = output_dir / f"{args.name_prefix}_walkforward_cumulative.png"
    rolling_plot = output_dir / f"{args.name_prefix}_walkforward_rolling.png"
    _plot_cumulative(trace_rows=trace_rows, output_path=best_plot, title="Payout-Aware Walk-Forward Longstream: Cumulative BNB")
    _plot_rolling(trace_rows=trace_rows, output_path=rolling_plot, title="Payout-Aware Walk-Forward Longstream: Rolling Net / 500")

    final_row = dict(block_rows[-1])
    summary_payload = {
        "name_prefix": str(args.name_prefix),
        "manifest_csv_path": str(Path(str(args.manifest_csv)).resolve()),
        "direction_source": str(args.direction_source),
        "payout_model_type": str(args.payout_model_type),
        "target_mode": str(args.target_mode),
        "train_size": int(args.train_size),
        "valid_size": int(args.valid_size),
        "sim_size": int(args.sim_size),
        "block_size": int(args.block_size),
        "bet_size_bnb": float(args.bet_size_bnb),
        "initial_bankroll_bnb": float(args.initial_bankroll_bnb),
        "tail_offset_rounds": int(args.tail_offset_rounds),
        "final_block": final_row,
        "artifacts": {
            "blocks_csv_path": str(rows_out),
            "trace_csv_path": str(trace_out),
            "cumulative_plot_path": str(best_plot),
            "rolling_plot_path": str(rolling_plot),
        },
    }
    summary_out = summary_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="walkforward_summary")
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8", newline="\n")

    report_lines = [
        "# Payout-Aware Walk-Forward Longstream",
        "",
        "## Config",
        "",
        f"- direction source: `{args.direction_source}`",
        f"- payout model: `{args.payout_model_type}`",
        f"- target mode: `{args.target_mode}`",
        f"- train size: `{int(args.train_size)}`",
        f"- valid size: `{int(args.valid_size)}`",
        f"- retrain cadence: every `{int(args.block_size)}` rounds",
        f"- latest held-out stream: `{int(args.sim_size)}` rounds",
        f"- fixed stake: `{float(args.bet_size_bnb):.2f}` BNB",
        "",
        "## Final",
        "",
        f"- final net: `{float(final_row['cumulative_net_profit_bnb']):.6f}` BNB",
        f"- final net / 500: `{float(final_row['cumulative_profit_per_500_bnb']):.6f}`",
        f"- final bet rate: `{100.0 * float(final_row['cumulative_bet_rate']):.3f}%`",
        f"- final win rate: `{100.0 * float(final_row['cumulative_win_rate']):.3f}%`",
        f"- max drawdown: `{float(final_row['cumulative_max_drawdown_bnb']):.6f}` BNB",
        "",
        "## Plots",
        "",
        f"- [cumulative]({best_plot})",
        f"- [rolling]({rolling_plot})",
        "",
    ]
    report_lines.append("## Block Rows")
    report_lines.append("")
    report_lines.append("| Block | Test Epochs | Block Net BNB | Block / 500 | Cum Net BNB | Cum / 500 | Cum Bet rate | Cum Win rate | Cum Max DD |")
    report_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in block_rows:
        report_lines.append(
            f"| {int(row['block_index'])} | {int(row['test_start_epoch'])}..{int(row['test_end_epoch'])} | "
            f"{float(row['block_net_profit_bnb']):.6f} | {float(row['block_profit_per_500_bnb']):.6f} | "
            f"{float(row['cumulative_net_profit_bnb']):.6f} | {float(row['cumulative_profit_per_500_bnb']):.6f} | "
            f"{100.0 * float(row['cumulative_bet_rate']):.3f}% | {100.0 * float(row['cumulative_win_rate']):.3f}% | "
            f"{float(row['cumulative_max_drawdown_bnb']):.6f} |"
        )
    report_path = output_dir / f"{args.name_prefix}_walkforward_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")

    print(
        json.dumps(
            {
                "phase": "done",
                "final_net_profit_bnb": float(final_row["cumulative_net_profit_bnb"]),
                "final_profit_per_500_bnb": float(final_row["cumulative_profit_per_500_bnb"]),
                "final_bet_rate": float(final_row["cumulative_bet_rate"]),
                "final_win_rate": float(final_row["cumulative_win_rate"]),
                "final_max_drawdown_bnb": float(final_row["cumulative_max_drawdown_bnb"]),
                "report_path": str(report_path),
                "summary_path": str(summary_out),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
