from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from pancakebot.core.errors import InvariantError
from inspection.run_profile_set_model_selector import CompareWindowRow, _load_compare_rows
from inspection.run_profile_set_penalty_selector import (
    _legacy_feature_dict,
    _legacy_feature_names,
    _predict_delta_ridge_with_penalties,
)
from inspection.run_profile_window_selector import _parse_positive_int_list

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class PenaltyWindowEvalRow:
    window_index: int
    tail_offset_rounds: int
    training_window_count: int
    cold_start_used: bool
    hold_forced: bool
    chosen_profile: str
    predicted_per_500: float
    realized_per_500: float
    prediction_error_per_500: float
    realized_selected_bet_rate: float
    estimated_selected_bet_rate: float
    baseline_profile: str
    baseline_realized_per_500: float
    baseline_realized_bet_rate: float
    oracle_profile: str
    oracle_realized_per_500: float
    oracle_realized_bet_rate: float
    gain_vs_baseline_per_500: float
    regret_vs_oracle_per_500: float
    predicted_baseline_per_500: float
    actual_per_500_json: str
    actual_bet_rate_json: str
    predicted_delta_json: str


@dataclass(frozen=True, slots=True)
class PenaltyWindowEvalSummary:
    baseline_profile_name: str
    feature_lookbacks_json: str
    min_train_windows: int
    min_hold_windows: int
    cold_start_mode: str
    cold_start_lookback: int
    margin_per_500: float
    skip_threshold_per_500: float
    ridge_alpha: float
    flow_penalty_per_500: float
    stageg2_penalty_per_500: float
    window_count: int
    mean_realized_per_500: float
    mean_predicted_per_500: float
    mean_prediction_error_per_500: float
    mean_selected_bet_rate: float
    mean_baseline_per_500: float
    mean_oracle_per_500: float
    mean_gain_vs_baseline_per_500: float
    mean_regret_vs_oracle_per_500: float
    positive_window_frac: float
    beat_baseline_window_frac: float
    match_oracle_window_frac: float
    skip_window_frac: float
    chosen_profile_counts_json: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-csv", type=str, required=True)
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--baseline-profile-name", type=str, default="stageb")
    parser.add_argument("--feature-lookbacks", type=str, default="1,3,5,8")
    parser.add_argument("--min-train-windows", type=int, required=True)
    parser.add_argument("--min-hold-windows", type=int, default=1)
    parser.add_argument("--cold-start-mode", type=str, default="trailing_best_vs_stageb_with_skip")
    parser.add_argument("--cold-start-lookback", type=int, default=5)
    parser.add_argument("--margin-per-500", type=float, default=-0.2)
    parser.add_argument("--skip-threshold-per-500", type=float, default=0.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--flow-penalty-per-500", type=float, default=0.2)
    parser.add_argument("--stageg2-penalty-per-500", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _best_profile(row: CompareWindowRow) -> tuple[str, float, float]:
    best_name = ""
    best_value = 0.0
    best_bet_rate = 0.0
    for name, value in row.per_500.items():
        if best_name == "" or float(value) > float(best_value):
            best_name = str(name)
            best_value = float(value)
            best_bet_rate = float(row.bet_rate[str(name)])
    if best_name == "":
        raise InvariantError("profile_set_penalty_window_eval_best_profile_missing")
    return str(best_name), float(best_value), float(best_bet_rate)


def _predicted_profile_mean_per_500(
    *,
    feature_row: dict[str, float],
    profile_name: str,
    feature_lookbacks: list[int],
) -> float:
    for lookback in sorted((int(x) for x in feature_lookbacks), reverse=True):
        key = f"feat_{str(profile_name)}_mean_per500_l{int(lookback)}"
        if str(key) in feature_row:
            return float(feature_row[str(key)])
    raise InvariantError("profile_set_penalty_window_eval_profile_mean_missing")


def _predicted_baseline_per_500(
    *,
    feature_row: dict[str, float],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
) -> float:
    return _predicted_profile_mean_per_500(
        feature_row=feature_row,
        profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
    )


def _estimated_profile_bet_rate_from_feature_row(
    *,
    feature_row: dict[str, float],
    profile_name: str,
    feature_lookbacks: list[int],
) -> float:
    for lookback in sorted((int(x) for x in feature_lookbacks), reverse=True):
        key = f"feat_{str(profile_name)}_mean_betrate_l{int(lookback)}"
        if str(key) in feature_row:
            return float(feature_row[str(key)])
    raise InvariantError("profile_set_penalty_window_eval_profile_betrate_missing")


def _cold_start_prediction(
    *,
    rows: list[CompareWindowRow],
    current_idx: int,
    feature_row: dict[str, float],
    profiles: list[str],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
    cold_start_mode: str,
    cold_start_lookback: int,
    margin_per_500: float,
    skip_threshold_per_500: float,
) -> tuple[str, float]:
    baseline_name = str(baseline_profile_name)
    predicted_baseline = (
        _predicted_baseline_per_500(
            feature_row=feature_row,
            baseline_profile_name=str(baseline_name),
            feature_lookbacks=feature_lookbacks,
        )
        if feature_row
        else 0.0
    )
    if str(cold_start_mode) == "baseline_or_skip":
        if float(predicted_baseline) <= float(skip_threshold_per_500):
            return "skip", 0.0
        return str(baseline_name), float(predicted_baseline)
    if str(cold_start_mode) == "prev_winner_with_skip":
        if int(current_idx) <= 0:
            if float(predicted_baseline) <= float(skip_threshold_per_500):
                return "skip", 0.0
            return str(baseline_name), float(predicted_baseline)
        prev_row = rows[int(current_idx) - 1]
        prev_winner, prev_value, _prev_bet_rate = _best_profile(prev_row)
        if float(prev_value) <= float(skip_threshold_per_500):
            return "skip", 0.0
        if feature_row:
            return str(prev_winner), _predicted_profile_mean_per_500(
                feature_row=feature_row,
                profile_name=str(prev_winner),
                feature_lookbacks=feature_lookbacks,
            )
        return "skip", 0.0
    if str(cold_start_mode) == "trailing_best_vs_stageb_with_skip":
        if int(current_idx) < int(cold_start_lookback):
            if float(predicted_baseline) <= float(skip_threshold_per_500):
                return "skip", 0.0
            return str(baseline_name), float(predicted_baseline)
        hist = rows[int(current_idx) - int(cold_start_lookback) : int(current_idx)]
        stageb_mean = float(sum(float(row.per_500[str(baseline_name)]) for row in hist) / len(hist))
        best_profile = str(baseline_name)
        best_mean = float(stageb_mean)
        for profile in profiles:
            if str(profile) == str(baseline_name):
                continue
            candidate_mean = float(sum(float(row.per_500[str(profile)]) for row in hist) / len(hist))
            if float(candidate_mean) > float(best_mean):
                best_profile = str(profile)
                best_mean = float(candidate_mean)
        if max(float(stageb_mean), float(best_mean)) <= float(skip_threshold_per_500):
            return "skip", 0.0
        if str(best_profile) != str(baseline_name) and float(best_mean - stageb_mean) > float(margin_per_500):
            if feature_row:
                return str(best_profile), _predicted_profile_mean_per_500(
                    feature_row=feature_row,
                    profile_name=str(best_profile),
                    feature_lookbacks=feature_lookbacks,
                )
            return "skip", 0.0
        return str(baseline_name), float(predicted_baseline)
    raise InvariantError("profile_set_penalty_window_eval_cold_start_mode_invalid")


def _eval_rows(
    *,
    rows: list[CompareWindowRow],
    profiles: list[str],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
    min_train_windows: int,
    min_hold_windows: int,
    cold_start_mode: str,
    cold_start_lookback: int,
    ridge_alpha: float,
    flow_penalty_per_500: float,
    stageg2_penalty_per_500: float,
    margin_per_500: float,
    skip_threshold_per_500: float,
) -> list[PenaltyWindowEvalRow]:
    feature_rows = [
        _legacy_feature_dict(
            rows=rows,
            idx=int(idx),
            profiles=profiles,
            baseline_profile_name=str(baseline_profile_name),
            feature_lookbacks=feature_lookbacks,
        )
        for idx in range(len(rows))
    ]
    feature_names = _legacy_feature_names(
        profiles=profiles,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
    )
    out: list[PenaltyWindowEvalRow] = []
    baseline_name = str(baseline_profile_name)
    held_profile = ""
    hold_remaining = 0
    for idx, row in enumerate(rows):
        current_feature_row = feature_rows[int(idx)]
        train_indices = [int(train_idx) for train_idx in range(int(idx)) if feature_rows[int(train_idx)] and int(train_idx) >= 1]
        predicted_baseline = _predicted_baseline_per_500(
            feature_row=current_feature_row,
            baseline_profile_name=str(baseline_name),
            feature_lookbacks=feature_lookbacks,
        ) if current_feature_row else 0.0
        cold_start_used = int(len(train_indices)) < int(min_train_windows)
        hold_forced = bool(int(hold_remaining) > 0 and str(held_profile) not in {"", "skip"})
        if hold_forced:
            chosen_profile = str(held_profile)
            predicted_value = (
                _predicted_profile_mean_per_500(
                    feature_row=current_feature_row,
                    profile_name=str(chosen_profile),
                    feature_lookbacks=feature_lookbacks,
                )
                if current_feature_row
                else 0.0
            )
            predicted_delta_json = {}
            hold_remaining -= 1
            if int(hold_remaining) <= 0:
                held_profile = ""
                hold_remaining = 0
        elif cold_start_used:
            chosen_profile, predicted_value = _cold_start_prediction(
                rows=rows,
                current_idx=int(idx),
                feature_row=current_feature_row,
                profiles=profiles,
                baseline_profile_name=str(baseline_name),
                feature_lookbacks=feature_lookbacks,
                cold_start_mode=str(cold_start_mode),
                cold_start_lookback=int(cold_start_lookback),
                margin_per_500=float(margin_per_500),
                skip_threshold_per_500=float(skip_threshold_per_500),
            )
            predicted_delta_json = {}
        else:
            predicted_delta_json = _predict_delta_ridge_with_penalties(
                rows=rows,
                feature_rows=feature_rows,
                feature_names=feature_names,
                train_indices=train_indices,
                current_idx=int(idx),
                baseline_profile_name=str(baseline_name),
                profiles=profiles,
                ridge_alpha=float(ridge_alpha),
                flow_penalty_per_500=float(flow_penalty_per_500),
                stageg2_penalty_per_500=float(stageg2_penalty_per_500),
            )
            chosen_profile = str(baseline_name)
            predicted_value = float(predicted_baseline)
            best_delta = 0.0
            for profile_name, predicted_delta in predicted_delta_json.items():
                if float(predicted_delta) > float(best_delta):
                    best_delta = float(predicted_delta)
                    chosen_profile = str(profile_name)
                    predicted_value = float(predicted_baseline + float(predicted_delta))
            if str(chosen_profile) != str(baseline_name) and float(best_delta) <= float(margin_per_500):
                chosen_profile = str(baseline_name)
                predicted_value = float(predicted_baseline)
            if float(predicted_value) <= float(skip_threshold_per_500):
                chosen_profile = "skip"
                predicted_value = 0.0
        if not hold_forced:
            if str(chosen_profile) != "skip" and int(min_hold_windows) > 1:
                held_profile = str(chosen_profile)
                hold_remaining = int(min_hold_windows) - 1
            else:
                held_profile = ""
                hold_remaining = 0

        realized_value = 0.0 if str(chosen_profile) == "skip" else float(row.per_500[str(chosen_profile)])
        realized_bet_rate = 0.0 if str(chosen_profile) == "skip" else float(row.bet_rate[str(chosen_profile)])
        estimated_bet_rate = 0.0
        if str(chosen_profile) != "skip":
            estimated_bet_rate = (
                _estimated_profile_bet_rate_from_feature_row(
                    feature_row=current_feature_row,
                    profile_name=str(chosen_profile),
                    feature_lookbacks=feature_lookbacks,
                )
                if current_feature_row
                else float(row.bet_rate[str(chosen_profile)])
            )
        oracle_profile, oracle_value, oracle_bet_rate = _best_profile(row)
        baseline_realized = float(row.per_500[str(baseline_name)])
        baseline_bet_rate = float(row.bet_rate[str(baseline_name)])
        out.append(
            PenaltyWindowEvalRow(
                window_index=int(idx),
                tail_offset_rounds=int(row.tail_offset_rounds),
                training_window_count=int(len(train_indices)),
                cold_start_used=bool(cold_start_used),
                hold_forced=bool(hold_forced),
                chosen_profile=str(chosen_profile),
                predicted_per_500=float(predicted_value),
                realized_per_500=float(realized_value),
                prediction_error_per_500=float(realized_value - float(predicted_value)),
                realized_selected_bet_rate=float(realized_bet_rate),
                estimated_selected_bet_rate=float(estimated_bet_rate),
                baseline_profile=str(baseline_name),
                baseline_realized_per_500=float(baseline_realized),
                baseline_realized_bet_rate=float(baseline_bet_rate),
                oracle_profile=str(oracle_profile),
                oracle_realized_per_500=float(oracle_value),
                oracle_realized_bet_rate=float(oracle_bet_rate),
                gain_vs_baseline_per_500=float(realized_value - float(baseline_realized)),
                regret_vs_oracle_per_500=float(float(oracle_value) - float(realized_value)),
                predicted_baseline_per_500=float(predicted_baseline),
                actual_per_500_json=json.dumps(
                    {str(k): float(v) for k, v in sorted(row.per_500.items())},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                actual_bet_rate_json=json.dumps(
                    {str(k): float(v) for k, v in sorted(row.bet_rate.items())},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                predicted_delta_json=json.dumps(
                    {str(k): float(v) for k, v in sorted(predicted_delta_json.items())},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
    return out


def _summary(rows: list[PenaltyWindowEvalRow], *, baseline_profile_name: str, feature_lookbacks: list[int], min_train_windows: int, min_hold_windows: int, cold_start_mode: str, cold_start_lookback: int, margin_per_500: float, skip_threshold_per_500: float, ridge_alpha: float, flow_penalty_per_500: float, stageg2_penalty_per_500: float) -> PenaltyWindowEvalSummary:
    if not rows:
        raise InvariantError("profile_set_penalty_window_eval_rows_empty")
    count = int(len(rows))
    chosen_counts = Counter(str(row.chosen_profile) for row in rows)
    positive_window_frac = float(sum(1 for row in rows if float(row.realized_per_500) > 0.0) / count)
    beat_baseline_window_frac = float(sum(1 for row in rows if float(row.realized_per_500) > float(row.baseline_realized_per_500)) / count)
    match_oracle_window_frac = float(sum(1 for row in rows if str(row.chosen_profile) == str(row.oracle_profile) or (str(row.chosen_profile) == "skip" and float(row.oracle_realized_per_500) <= 0.0)) / count)
    skip_window_frac = float(sum(1 for row in rows if str(row.chosen_profile) == "skip") / count)
    return PenaltyWindowEvalSummary(
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks_json=json.dumps(list(feature_lookbacks), separators=(",", ":")),
        min_train_windows=int(min_train_windows),
        min_hold_windows=int(min_hold_windows),
        cold_start_mode=str(cold_start_mode),
        cold_start_lookback=int(cold_start_lookback),
        margin_per_500=float(margin_per_500),
        skip_threshold_per_500=float(skip_threshold_per_500),
        ridge_alpha=float(ridge_alpha),
        flow_penalty_per_500=float(flow_penalty_per_500),
        stageg2_penalty_per_500=float(stageg2_penalty_per_500),
        window_count=int(count),
        mean_realized_per_500=float(sum(float(row.realized_per_500) for row in rows) / count),
        mean_predicted_per_500=float(sum(float(row.predicted_per_500) for row in rows) / count),
        mean_prediction_error_per_500=float(sum(float(row.prediction_error_per_500) for row in rows) / count),
        mean_selected_bet_rate=float(sum(float(row.realized_selected_bet_rate) for row in rows) / count),
        mean_baseline_per_500=float(sum(float(row.baseline_realized_per_500) for row in rows) / count),
        mean_oracle_per_500=float(sum(float(row.oracle_realized_per_500) for row in rows) / count),
        mean_gain_vs_baseline_per_500=float(sum(float(row.gain_vs_baseline_per_500) for row in rows) / count),
        mean_regret_vs_oracle_per_500=float(sum(float(row.regret_vs_oracle_per_500) for row in rows) / count),
        positive_window_frac=float(positive_window_frac),
        beat_baseline_window_frac=float(beat_baseline_window_frac),
        match_oracle_window_frac=float(match_oracle_window_frac),
        skip_window_frac=float(skip_window_frac),
        chosen_profile_counts_json=json.dumps(dict(sorted(chosen_counts.items())), sort_keys=True, separators=(",", ":")),
    )


def main() -> None:
    args = _build_parser().parse_args()
    profiles, rows = _load_compare_rows(Path(str(args.compare_csv)).resolve())
    baseline_profile_name = str(args.baseline_profile_name).strip()
    if baseline_profile_name == "" or baseline_profile_name not in profiles:
        raise InvariantError("profile_set_penalty_window_eval_baseline_invalid")
    feature_lookbacks = _parse_positive_int_list(str(args.feature_lookbacks))
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = _eval_rows(
        rows=rows,
        profiles=profiles,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
        min_train_windows=int(args.min_train_windows),
        min_hold_windows=int(args.min_hold_windows),
        cold_start_mode=str(args.cold_start_mode),
        cold_start_lookback=int(args.cold_start_lookback),
        ridge_alpha=float(args.ridge_alpha),
        flow_penalty_per_500=float(args.flow_penalty_per_500),
        stageg2_penalty_per_500=float(args.stageg2_penalty_per_500),
        margin_per_500=float(args.margin_per_500),
        skip_threshold_per_500=float(args.skip_threshold_per_500),
    )
    summary = _summary(
        eval_rows,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
        min_train_windows=int(args.min_train_windows),
        min_hold_windows=int(args.min_hold_windows),
        cold_start_mode=str(args.cold_start_mode),
        cold_start_lookback=int(args.cold_start_lookback),
        margin_per_500=float(args.margin_per_500),
        skip_threshold_per_500=float(args.skip_threshold_per_500),
        ridge_alpha=float(args.ridge_alpha),
        flow_penalty_per_500=float(args.flow_penalty_per_500),
        stageg2_penalty_per_500=float(args.stageg2_penalty_per_500),
    )
    csv_path = output_dir / f"{args.name_prefix}_profile_set_penalty_window_eval.csv"
    json_path = output_dir / f"{args.name_prefix}_profile_set_penalty_window_eval.json"
    summary_path = output_dir / f"{args.name_prefix}_profile_set_penalty_window_eval_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(eval_rows[0]).keys()))
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(asdict(row))
    json_path.write_text(
        json.dumps([asdict(row) for row in eval_rows], indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
