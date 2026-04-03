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
from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_confidence import (
    chosen_side_confidence,
)
from pancakebot.domain.models.neural_direction_policy import (
    confidence_threshold_for_target_coverage,
    simulate_confidence_threshold_policy,
)
from pancakebot.runtime.contract_constants_cache import load_contract_constants

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class DirectionEnsemblePolicyEvalRow:
    model_name: str
    sim_size: int
    tail_offset_rounds: int
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
class DirectionEnsemblePolicyAggregateRow:
    model_name: str
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
    parser.add_argument("--rows-csv", type=str, required=True)
    parser.add_argument("--coverage-fractions", type=str, default="0.10,0.05,0.02,0.01")
    parser.add_argument("--bet-sizes-bnb", type=str, default="0.10")
    parser.add_argument("--initial-bankroll-bnb", type=float, default=50.0)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _parse_fraction_list(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = float(text)
        if not (0.0 < value <= 1.0):
            raise InvariantError("direction_ensemble_policy_fraction_invalid")
        out.append(float(value))
    if not out:
        raise InvariantError("direction_ensemble_policy_fraction_empty")
    return tuple(out)


def _parse_bet_sizes(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = float(text)
        if float(value) <= 0.0:
            raise InvariantError("direction_ensemble_policy_bet_size_nonpositive")
        out.append(float(value))
    if not out:
        raise InvariantError("direction_ensemble_policy_bet_sizes_empty")
    return tuple(out)


def _aggregate_rows(rows: list[DirectionEnsemblePolicyEvalRow]) -> list[DirectionEnsemblePolicyAggregateRow]:
    grouped: dict[tuple[str, int, float, float], list[DirectionEnsemblePolicyEvalRow]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row.model_name),
                int(row.sim_size),
                float(row.target_coverage_fraction),
                float(row.bet_size_bnb),
            ),
            [],
        ).append(row)
    out: list[DirectionEnsemblePolicyAggregateRow] = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), int(item[1]), -float(item[2]), float(item[3]))):
        group = grouped[key]
        out.append(
            DirectionEnsemblePolicyAggregateRow(
                model_name=str(key[0]),
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
    coverage_fractions = _parse_fraction_list(args.coverage_fractions)
    bet_sizes_bnb = _parse_bet_sizes(args.bet_sizes_bnb)
    cfg = load_app_config(str(args.config))
    constants = load_contract_constants()
    if any(float(bet_size) < float(constants.min_bet_amount_bnb) for bet_size in bet_sizes_bnb):
        raise InvariantError("direction_ensemble_policy_bet_size_below_min_bet")

    with Path(str(args.rows_csv)).resolve().open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        all_rows = [dict(row) for row in reader]
    if not all_rows:
        raise InvariantError("direction_ensemble_policy_rows_empty")

    prob_columns = sorted(
        col for col in all_rows[0].keys()
        if str(col).startswith("p_bull_")
    )
    if not prob_columns:
        raise InvariantError("direction_ensemble_policy_prob_columns_missing")
    model_names = [str(col)[len("p_bull_"):] for col in prob_columns]

    by_group: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in all_rows:
        key = (int(row["sim_size"]), int(row["tail_offset_rounds"]))
        by_group.setdefault(key, []).append(row)

    rows_out_list: list[DirectionEnsemblePolicyEvalRow] = []
    for (sim_size, tail_offset_rounds), rows_group in sorted(by_group.items()):
        valid_rows = sorted(
            [row for row in rows_group if str(row["split"]) == "valid"],
            key=lambda row: int(row["target_epoch"]),
        )
        test_rows = sorted(
            [row for row in rows_group if str(row["split"]) == "test"],
            key=lambda row: int(row["target_epoch"]),
        )
        if not valid_rows or not test_rows:
            raise InvariantError("direction_ensemble_policy_split_rows_missing")
        required_examples = int(len(valid_rows)) + int(len(test_rows))
        eval_slice = load_recent_direction_eval_slice(
            config_path=str(args.config),
            required_examples=int(required_examples),
            tail_offset_rounds=int(tail_offset_rounds),
        )
        target_rounds_by_epoch = eval_slice.target_rounds_by_epoch
        test_epochs = [int(row["target_epoch"]) for row in test_rows]
        test_rounds = [target_rounds_by_epoch[int(epoch)] for epoch in test_epochs]
        for model_name in model_names:
            valid_probs = np.asarray([float(row[f"p_bull_{model_name}"]) for row in valid_rows], dtype=np.float32)
            test_probs = np.asarray([float(row[f"p_bull_{model_name}"]) for row in test_rows], dtype=np.float32)
            valid_pred = (valid_probs >= 0.5).astype(np.int64)
            valid_conf = chosen_side_confidence(
                predicted_labels=valid_pred,
                calibrated_bull_probs=valid_probs,
            )
            for coverage_fraction in coverage_fractions:
                threshold = confidence_threshold_for_target_coverage(
                    chosen_side_confidence=valid_conf,
                    target_coverage_fraction=float(coverage_fraction),
                )
                for bet_size_bnb in bet_sizes_bnb:
                    result = simulate_confidence_threshold_policy(
                        rounds=test_rounds,
                        calibrated_bull_probs=test_probs,
                        threshold=float(threshold),
                        bet_size_bnb=float(bet_size_bnb),
                        initial_bankroll_bnb=float(args.initial_bankroll_bnb),
                        treasury_fee_fraction=float(constants.treasury_fee_fraction),
                    )
                    rows_out_list.append(
                        DirectionEnsemblePolicyEvalRow(
                            model_name=str(model_name),
                            sim_size=int(sim_size),
                            tail_offset_rounds=int(tail_offset_rounds),
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

    aggregates = _aggregate_rows(rows_out_list)
    rows_out = rows_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="direction_ensemble_policy_rows")
    with rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows_out_list[0]).keys()))
        writer.writeheader()
        for row in rows_out_list:
            writer.writerow(asdict(row))
    summary_out = summary_path(output_dir=output_dir, name_prefix=str(args.name_prefix), suffix="direction_ensemble_policy_summary")
    summary_payload = {
        "rows_csv_path": str(rows_out),
        "coverage_fractions": [float(x) for x in coverage_fractions],
        "bet_sizes_bnb": [float(x) for x in bet_sizes_bnb],
        "aggregates": [asdict(row) for row in aggregates],
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
