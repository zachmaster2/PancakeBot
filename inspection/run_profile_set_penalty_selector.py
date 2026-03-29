from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pancakebot.core.errors import InvariantError
from inspection.run_profile_set_model_selector import CompareWindowRow, _load_compare_rows
from inspection.run_profile_window_selector import _parse_float_list, _parse_positive_int_list

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_COLD_START_MODES = ("baseline_or_skip", "prev_winner_with_skip", "trailing_best_vs_stageb_with_skip")


@dataclass(frozen=True, slots=True)
class PenaltySelectorResult:
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
    mean_per_500: float
    mean_selected_bet_rate: float
    meets_min_selected_bet_rate: bool
    switch_count: int
    pick_counts_json: str


@dataclass(frozen=True, slots=True)
class PenaltyNextRecommendation:
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
    training_window_count: int
    chosen_profile: str
    chosen_predicted_per_500: float
    predicted_baseline_per_500: float
    estimated_selected_bet_rate: float
    predicted_delta_json: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-csv", type=str, required=True)
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--baseline-profile-name", type=str, default="stageb")
    parser.add_argument("--feature-lookbacks", type=str, default="1,3,5,8")
    parser.add_argument("--min-train-windows", type=str, default="8,10,12")
    parser.add_argument("--min-hold-windows", type=str, default="1,2")
    parser.add_argument("--cold-start-modes", type=str, default="baseline_or_skip,prev_winner_with_skip,trailing_best_vs_stageb_with_skip")
    parser.add_argument("--cold-start-lookbacks", type=str, default="1,3,5")
    parser.add_argument("--selector-margins-per-500", type=str, default="-0.2,0.0,0.2")
    parser.add_argument("--selector-skip-thresholds-per-500", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--ridge-alphas", type=str, default="0.25,0.5,1.0,2.0,5.0,10.0")
    parser.add_argument("--flow-penalties-per-500", type=str, default="0.0,0.1,0.2,0.3,0.5")
    parser.add_argument("--stageg2-penalties-per-500", type=str, default="0.0,0.1,0.2,0.3")
    parser.add_argument("--min-selected-bet-rate", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _parse_cold_start_mode_list(raw: str) -> list[str]:
    values = [str(token).strip() for token in str(raw).split(",") if str(token).strip() != ""]
    if not values:
        raise InvariantError("profile_set_penalty_cold_start_modes_empty")
    invalid = [str(value) for value in values if str(value) not in _COLD_START_MODES]
    if invalid:
        raise InvariantError(f"profile_set_penalty_cold_start_modes_invalid: {','.join(invalid)}")
    return values


def _legacy_feature_dict(
    *,
    rows: list[CompareWindowRow],
    idx: int,
    profiles: list[str],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
) -> dict[str, float]:
    if int(idx) <= 0:
        return {}
    out: dict[str, float] = {}
    baseline_name = str(baseline_profile_name)
    for lookback in feature_lookbacks:
        hist = rows[max(0, int(idx) - int(lookback)) : int(idx)]
        if not hist:
            continue
        for profile in profiles:
            values = [float(row.per_500[str(profile)]) for row in hist]
            bet_rates = [float(row.bet_rate[str(profile)]) for row in hist]
            mean_value = float(sum(values) / len(values))
            variance = float(sum((value - mean_value) ** 2 for value in values) / len(values))
            out[f"feat_{profile}_mean_per500_l{int(lookback)}"] = float(mean_value)
            out[f"feat_{profile}_std_per500_l{int(lookback)}"] = float(np.sqrt(variance))
            out[f"feat_{profile}_last_per500_l{int(lookback)}"] = float(values[-1])
            out[f"feat_{profile}_mean_betrate_l{int(lookback)}"] = float(sum(bet_rates) / len(bet_rates))
            out[f"feat_{profile}_last_betrate_l{int(lookback)}"] = float(bet_rates[-1])
            out[f"feat_{profile}_pos_frac_l{int(lookback)}"] = float(
                sum(1 for value in values if float(value) > 0.0) / len(values)
            )
        baseline_mean = float(out[f"feat_{baseline_name}_mean_per500_l{int(lookback)}"])
        baseline_last = float(out[f"feat_{baseline_name}_last_per500_l{int(lookback)}"])
        for profile in profiles:
            if str(profile) == str(baseline_name):
                continue
            out[f"feat_{profile}_delta_mean_vs_{baseline_name}_l{int(lookback)}"] = float(
                out[f"feat_{profile}_mean_per500_l{int(lookback)}"] - baseline_mean
            )
            out[f"feat_{profile}_delta_last_vs_{baseline_name}_l{int(lookback)}"] = float(
                out[f"feat_{profile}_last_per500_l{int(lookback)}"] - baseline_last
            )
    return out


def _legacy_feature_names(
    *,
    profiles: list[str],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
) -> list[str]:
    names: list[str] = []
    baseline_name = str(baseline_profile_name)
    for lookback in feature_lookbacks:
        for profile in profiles:
            names.extend(
                [
                    f"feat_{profile}_mean_per500_l{int(lookback)}",
                    f"feat_{profile}_std_per500_l{int(lookback)}",
                    f"feat_{profile}_last_per500_l{int(lookback)}",
                    f"feat_{profile}_mean_betrate_l{int(lookback)}",
                    f"feat_{profile}_last_betrate_l{int(lookback)}",
                    f"feat_{profile}_pos_frac_l{int(lookback)}",
                ]
            )
        for profile in profiles:
            if str(profile) == str(baseline_name):
                continue
            names.extend(
                [
                    f"feat_{profile}_delta_mean_vs_{baseline_name}_l{int(lookback)}",
                    f"feat_{profile}_delta_last_vs_{baseline_name}_l{int(lookback)}",
                ]
            )
    return names


def _feature_matrix(
    *,
    feature_rows: list[dict[str, float]],
    indices: list[int],
    current_idx: int,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray(
        [[float(feature_rows[int(idx)].get(str(name), np.nan)) for name in feature_names] for idx in indices],
        dtype=float,
    )
    x_current = np.asarray(
        [[float(feature_rows[int(current_idx)].get(str(name), np.nan)) for name in feature_names]],
        dtype=float,
    )
    if int(x_train.shape[1]) <= 0:
        return x_train, x_current
    usable_mask = ~np.all(np.isnan(x_train), axis=0)
    if not np.any(usable_mask):
        return np.empty((len(indices), 0), dtype=float), np.empty((1, 0), dtype=float)
    return x_train[:, usable_mask], x_current[:, usable_mask]


def _profile_penalty_per_500(
    *,
    profile_name: str,
    flow_penalty_per_500: float,
    stageg2_penalty_per_500: float,
) -> float:
    name = str(profile_name)
    if name.startswith("flow_"):
        return float(flow_penalty_per_500)
    if name.startswith("stageg2"):
        return float(stageg2_penalty_per_500)
    return 0.0


def _predicted_stageb_per_500(
    *,
    feature_row: dict[str, float],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
) -> float:
    longest = max(int(x) for x in feature_lookbacks)
    return float(feature_row[f"feat_{str(baseline_profile_name)}_mean_per500_l{int(longest)}"])


def _estimated_profile_bet_rate(
    *,
    feature_row: dict[str, float],
    profile_name: str,
    feature_lookbacks: list[int],
) -> float:
    longest = max(int(x) for x in feature_lookbacks)
    return float(feature_row[f"feat_{str(profile_name)}_mean_betrate_l{int(longest)}"])


def _cold_start_pick(
    *,
    rows: list[CompareWindowRow],
    current_idx: int,
    profiles: list[str],
    baseline_profile_name: str,
    cold_start_mode: str,
    cold_start_lookback: int,
    margin_per_500: float,
    skip_threshold_per_500: float,
) -> tuple[str, float, float]:
    row = rows[int(current_idx)]
    baseline_name = str(baseline_profile_name)
    baseline_value = float(row.per_500[str(baseline_name)])
    baseline_bet_rate = float(row.bet_rate[str(baseline_name)])
    if str(cold_start_mode) == "baseline_or_skip":
        if float(baseline_value) <= float(skip_threshold_per_500):
            return "skip", 0.0, 0.0
        return str(baseline_name), float(baseline_value), float(baseline_bet_rate)
    if str(cold_start_mode) == "prev_winner_with_skip":
        if int(current_idx) <= 0:
            if float(baseline_value) <= float(skip_threshold_per_500):
                return "skip", 0.0, 0.0
            return str(baseline_name), float(baseline_value), float(baseline_bet_rate)
        prev_row = rows[int(current_idx) - 1]
        prev_winner, prev_value = max(prev_row.per_500.items(), key=lambda item: float(item[1]))
        if float(prev_value) <= float(skip_threshold_per_500):
            return "skip", 0.0, 0.0
        return str(prev_winner), float(row.per_500[str(prev_winner)]), float(row.bet_rate[str(prev_winner)])
    if str(cold_start_mode) == "trailing_best_vs_stageb_with_skip":
        if int(current_idx) < int(cold_start_lookback):
            if float(baseline_value) <= float(skip_threshold_per_500):
                return "skip", 0.0, 0.0
            return str(baseline_name), float(baseline_value), float(baseline_bet_rate)
        hist = rows[int(current_idx) - int(cold_start_lookback) : int(current_idx)]
        stageb_mean = float(sum(float(x.per_500[str(baseline_name)]) for x in hist) / len(hist))
        best_profile_name = str(baseline_name)
        best_profile_mean = float(stageb_mean)
        for profile in profiles:
            if str(profile) == str(baseline_name):
                continue
            candidate_mean = float(sum(float(x.per_500[str(profile)]) for x in hist) / len(hist))
            if float(candidate_mean) > float(best_profile_mean):
                best_profile_name = str(profile)
                best_profile_mean = float(candidate_mean)
        if max(float(stageb_mean), float(best_profile_mean)) <= float(skip_threshold_per_500):
            return "skip", 0.0, 0.0
        if str(best_profile_name) != str(baseline_name) and float(best_profile_mean - stageb_mean) > float(margin_per_500):
            return str(best_profile_name), float(row.per_500[str(best_profile_name)]), float(row.bet_rate[str(best_profile_name)])
        return str(baseline_name), float(baseline_value), float(baseline_bet_rate)
    raise InvariantError("profile_set_penalty_cold_start_mode_invalid")


def _predict_delta_ridge_with_penalties(
    *,
    rows: list[CompareWindowRow],
    feature_rows: list[dict[str, float]],
    feature_names: list[str],
    train_indices: list[int],
    current_idx: int,
    baseline_profile_name: str,
    profiles: list[str],
    ridge_alpha: float,
    flow_penalty_per_500: float,
    stageg2_penalty_per_500: float,
) -> dict[str, float]:
    x_train, x_current = _feature_matrix(
        feature_rows=feature_rows,
        indices=train_indices,
        current_idx=int(current_idx),
        feature_names=feature_names,
    )
    if int(x_train.shape[1]) <= 0:
        return {}
    out: dict[str, float] = {}
    baseline_name = str(baseline_profile_name)
    for profile in profiles:
        if str(profile) == str(baseline_name):
            continue
        y_train = np.asarray(
            [
                float(rows[int(idx)].per_500[str(profile)]) - float(rows[int(idx)].per_500[str(baseline_name)])
                for idx in train_indices
            ],
            dtype=float,
        )
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            Ridge(alpha=float(ridge_alpha)),
        )
        model.fit(x_train, y_train)
        predicted_delta = float(model.predict(x_current)[0])
        predicted_delta -= _profile_penalty_per_500(
            profile_name=str(profile),
            flow_penalty_per_500=float(flow_penalty_per_500),
            stageg2_penalty_per_500=float(stageg2_penalty_per_500),
        )
        out[str(profile)] = float(predicted_delta)
    return out


def _pick_window(
    *,
    rows: list[CompareWindowRow],
    feature_rows: list[dict[str, float]],
    feature_names: list[str],
    current_idx: int,
    profiles: list[str],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
    min_train_windows: int,
    cold_start_mode: str,
    cold_start_lookback: int,
    ridge_alpha: float,
    flow_penalty_per_500: float,
    stageg2_penalty_per_500: float,
    margin_per_500: float,
    skip_threshold_per_500: float,
) -> tuple[str, float, float]:
    row = rows[int(current_idx)]
    train_indices = [int(idx) for idx in range(int(current_idx)) if feature_rows[int(idx)] and int(idx) >= 1]
    if len(train_indices) < int(min_train_windows):
        return _cold_start_pick(
            rows=rows,
            current_idx=int(current_idx),
            profiles=profiles,
            baseline_profile_name=str(baseline_profile_name),
            cold_start_mode=str(cold_start_mode),
            cold_start_lookback=int(cold_start_lookback),
            margin_per_500=float(margin_per_500),
            skip_threshold_per_500=float(skip_threshold_per_500),
        )
    predictions = _predict_delta_ridge_with_penalties(
        rows=rows,
        feature_rows=feature_rows,
        feature_names=feature_names,
        train_indices=train_indices,
        current_idx=int(current_idx),
        baseline_profile_name=str(baseline_profile_name),
        profiles=profiles,
        ridge_alpha=float(ridge_alpha),
        flow_penalty_per_500=float(flow_penalty_per_500),
        stageg2_penalty_per_500=float(stageg2_penalty_per_500),
    )
    predicted_baseline = _predicted_stageb_per_500(
        feature_row=feature_rows[int(current_idx)],
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
    )
    chosen_profile = str(baseline_profile_name)
    chosen_predicted = float(predicted_baseline)
    best_delta = 0.0
    for profile, predicted_delta in predictions.items():
        if float(predicted_delta) > float(best_delta):
            best_delta = float(predicted_delta)
            chosen_profile = str(profile)
            chosen_predicted = float(predicted_baseline + float(predicted_delta))
    if str(chosen_profile) != str(baseline_profile_name) and float(best_delta) <= float(margin_per_500):
        chosen_profile = str(baseline_profile_name)
        chosen_predicted = float(predicted_baseline)
    if float(chosen_predicted) <= float(skip_threshold_per_500):
        return "skip", 0.0, 0.0
    return str(chosen_profile), float(row.per_500[str(chosen_profile)]), float(row.bet_rate[str(chosen_profile)])


def _evaluate_penalty_selector(
    *,
    rows: list[CompareWindowRow],
    profiles: list[str],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
    min_train_windows_list: list[int],
    min_hold_windows_list: list[int],
    cold_start_modes: list[str],
    cold_start_lookbacks: list[int],
    margins_per_500: list[float],
    skip_thresholds_per_500: list[float],
    ridge_alphas: list[float],
    flow_penalties_per_500: list[float],
    stageg2_penalties_per_500: list[float],
    min_selected_bet_rate: float,
) -> list[PenaltySelectorResult]:
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
    results: list[PenaltySelectorResult] = []
    for min_train_windows in min_train_windows_list:
        for min_hold_windows in min_hold_windows_list:
            for cold_start_mode in cold_start_modes:
                active_cold_start_lookbacks = [0] if str(cold_start_mode) != "trailing_best_vs_stageb_with_skip" else cold_start_lookbacks
                for cold_start_lookback in active_cold_start_lookbacks:
                    for margin in margins_per_500:
                        for skip_threshold in skip_thresholds_per_500:
                            for ridge_alpha in ridge_alphas:
                                for flow_penalty in flow_penalties_per_500:
                                    for stageg2_penalty in stageg2_penalties_per_500:
                                        total = 0.0
                                        total_bet_rate = 0.0
                                        switch_count = 0
                                        previous_pick = ""
                                        held_profile = ""
                                        hold_remaining = 0
                                        picks = Counter()
                                        for idx in range(len(rows)):
                                            if int(hold_remaining) > 0 and str(held_profile) not in {"", "skip"}:
                                                pick = str(held_profile)
                                                value = float(rows[int(idx)].per_500[str(pick)])
                                                bet_rate = float(rows[int(idx)].bet_rate[str(pick)])
                                                hold_remaining -= 1
                                            else:
                                                pick, value, bet_rate = _pick_window(
                                                    rows=rows,
                                                    feature_rows=feature_rows,
                                                    feature_names=feature_names,
                                                    current_idx=int(idx),
                                                    profiles=profiles,
                                                    baseline_profile_name=str(baseline_profile_name),
                                                    feature_lookbacks=feature_lookbacks,
                                                    min_train_windows=int(min_train_windows),
                                                    cold_start_mode=str(cold_start_mode),
                                                    cold_start_lookback=int(cold_start_lookback),
                                                    ridge_alpha=float(ridge_alpha),
                                                    flow_penalty_per_500=float(flow_penalty),
                                                    stageg2_penalty_per_500=float(stageg2_penalty),
                                                    margin_per_500=float(margin),
                                                    skip_threshold_per_500=float(skip_threshold),
                                                )
                                                if str(pick) != "skip" and int(min_hold_windows) > 1:
                                                    held_profile = str(pick)
                                                    hold_remaining = int(min_hold_windows) - 1
                                                else:
                                                    held_profile = ""
                                                    hold_remaining = 0
                                            total += float(value)
                                            total_bet_rate += float(bet_rate)
                                            picks[str(pick)] += 1
                                            if previous_pick != "" and str(previous_pick) != str(pick):
                                                switch_count += 1
                                            previous_pick = str(pick)
                                        mean_bet_rate = float(total_bet_rate / len(rows))
                                        results.append(
                                            PenaltySelectorResult(
                                                feature_lookbacks_json=json.dumps(list(feature_lookbacks), separators=(",", ":")),
                                                min_train_windows=int(min_train_windows),
                                                min_hold_windows=int(min_hold_windows),
                                                cold_start_mode=str(cold_start_mode),
                                                cold_start_lookback=int(cold_start_lookback),
                                                margin_per_500=float(margin),
                                                skip_threshold_per_500=float(skip_threshold),
                                                ridge_alpha=float(ridge_alpha),
                                                flow_penalty_per_500=float(flow_penalty),
                                                stageg2_penalty_per_500=float(stageg2_penalty),
                                                mean_per_500=float(total / len(rows)),
                                                mean_selected_bet_rate=float(mean_bet_rate),
                                                meets_min_selected_bet_rate=float(mean_bet_rate) >= float(min_selected_bet_rate),
                                                switch_count=int(switch_count),
                                                pick_counts_json=json.dumps(dict(sorted(picks.items())), separators=(",", ":"), sort_keys=True),
                                            )
                                        )
    return sorted(
        results,
        key=lambda row: (
            1 if bool(row.meets_min_selected_bet_rate) else 0,
            float(row.mean_per_500),
            -int(row.switch_count),
            float(row.mean_selected_bet_rate),
        ),
        reverse=True,
    )


def predict_next_penalty_recommendation(
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
) -> PenaltyNextRecommendation:
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
    current_feature_row = _legacy_feature_dict(
        rows=rows,
        idx=len(rows),
        profiles=profiles,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
    )
    feature_rows_plus = list(feature_rows) + [current_feature_row]
    feature_names = _legacy_feature_names(
        profiles=profiles,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
    )
    train_indices = [int(idx) for idx in range(len(rows)) if feature_rows[int(idx)] and int(idx) >= 1]
    predicted_baseline = _predicted_stageb_per_500(
        feature_row=current_feature_row,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
    )
    if len(train_indices) < int(min_train_windows):
        chosen_profile = "skip" if float(predicted_baseline) <= float(skip_threshold_per_500) else str(baseline_profile_name)
        return PenaltyNextRecommendation(
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
            training_window_count=int(len(train_indices)),
            chosen_profile=str(chosen_profile),
            chosen_predicted_per_500=(0.0 if str(chosen_profile) == "skip" else float(predicted_baseline)),
            predicted_baseline_per_500=float(predicted_baseline),
            estimated_selected_bet_rate=(
                0.0
                if str(chosen_profile) == "skip"
                else _estimated_profile_bet_rate(
                    feature_row=current_feature_row,
                    profile_name=str(chosen_profile),
                    feature_lookbacks=feature_lookbacks,
                )
            ),
            predicted_delta_json=json.dumps({}, sort_keys=True, separators=(",", ":")),
        )
    predictions = _predict_delta_ridge_with_penalties(
        rows=rows,
        feature_rows=feature_rows_plus,
        feature_names=feature_names,
        train_indices=train_indices,
        current_idx=int(len(rows)),
        baseline_profile_name=str(baseline_profile_name),
        profiles=profiles,
        ridge_alpha=float(ridge_alpha),
        flow_penalty_per_500=float(flow_penalty_per_500),
        stageg2_penalty_per_500=float(stageg2_penalty_per_500),
    )
    chosen_profile = str(baseline_profile_name)
    chosen_predicted = float(predicted_baseline)
    best_delta = 0.0
    for profile, predicted_delta in predictions.items():
        if float(predicted_delta) > float(best_delta):
            best_delta = float(predicted_delta)
            chosen_profile = str(profile)
            chosen_predicted = float(predicted_baseline + float(predicted_delta))
    if str(chosen_profile) != str(baseline_profile_name) and float(best_delta) <= float(margin_per_500):
        chosen_profile = str(baseline_profile_name)
        chosen_predicted = float(predicted_baseline)
    if float(chosen_predicted) <= float(skip_threshold_per_500):
        chosen_profile = "skip"
        chosen_predicted = 0.0
    return PenaltyNextRecommendation(
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
        training_window_count=int(len(train_indices)),
        chosen_profile=str(chosen_profile),
        chosen_predicted_per_500=float(chosen_predicted),
        predicted_baseline_per_500=float(predicted_baseline),
        estimated_selected_bet_rate=(
            0.0
            if str(chosen_profile) == "skip"
            else _estimated_profile_bet_rate(
                feature_row=current_feature_row,
                profile_name=str(chosen_profile),
                feature_lookbacks=feature_lookbacks,
            )
        ),
        predicted_delta_json=json.dumps(
            {str(k): float(v) for k, v in sorted(predictions.items())},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def main() -> None:
    args = _build_parser().parse_args()
    if float(args.min_selected_bet_rate) < 0.0:
        raise InvariantError("profile_set_penalty_min_selected_bet_rate_negative")
    profiles, rows = _load_compare_rows(Path(str(args.compare_csv)).resolve())
    baseline_profile_name = str(args.baseline_profile_name).strip()
    if baseline_profile_name == "" or baseline_profile_name not in profiles:
        raise InvariantError("profile_set_penalty_baseline_profile_invalid")
    feature_lookbacks = _parse_positive_int_list(str(args.feature_lookbacks))
    min_train_windows_list = _parse_positive_int_list(str(args.min_train_windows))
    min_hold_windows_list = _parse_positive_int_list(str(args.min_hold_windows))
    cold_start_modes = _parse_cold_start_mode_list(str(args.cold_start_modes))
    cold_start_lookbacks = _parse_positive_int_list(str(args.cold_start_lookbacks))
    margins_per_500 = _parse_float_list(str(args.selector_margins_per_500))
    skip_thresholds_per_500 = _parse_float_list(str(args.selector_skip_thresholds_per_500))
    ridge_alphas = _parse_float_list(str(args.ridge_alphas))
    flow_penalties_per_500 = _parse_float_list(str(args.flow_penalties_per_500))
    stageg2_penalties_per_500 = _parse_float_list(str(args.stageg2_penalties_per_500))
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _evaluate_penalty_selector(
        rows=rows,
        profiles=profiles,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
        min_train_windows_list=min_train_windows_list,
        min_hold_windows_list=min_hold_windows_list,
        cold_start_modes=cold_start_modes,
        cold_start_lookbacks=cold_start_lookbacks,
        margins_per_500=margins_per_500,
        skip_thresholds_per_500=skip_thresholds_per_500,
        ridge_alphas=ridge_alphas,
        flow_penalties_per_500=flow_penalties_per_500,
        stageg2_penalties_per_500=stageg2_penalties_per_500,
        min_selected_bet_rate=float(args.min_selected_bet_rate),
    )

    csv_path = output_dir / f"{args.name_prefix}_profile_set_penalty_selectors.csv"
    json_path = output_dir / f"{args.name_prefix}_profile_set_penalty_selectors.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    json_path.write_text(
        json.dumps([asdict(row) for row in results], indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
