from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pancakebot.core.errors import InvariantError
from inspection.run_profile_window_selector import _parse_float_list, _parse_positive_int_list

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_MODEL_MODES = ("delta_ridge", "delta_logistic")


@dataclass(frozen=True, slots=True)
class CompareWindowRow:
    tail_offset_rounds: int
    per_500: dict[str, float]
    bet_rate: dict[str, float]


@dataclass(frozen=True, slots=True)
class ModelSelectorResult:
    mode: str
    feature_lookbacks_json: str
    min_train_windows: int
    min_hold_windows: int
    margin_per_500: float
    skip_threshold_per_500: float
    ridge_alpha: float
    logistic_c: float
    mean_per_500: float
    mean_selected_bet_rate: float
    meets_min_selected_bet_rate: bool
    switch_count: int
    pick_counts_json: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-csv", type=str, required=True)
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--baseline-profile-name", type=str, default="stageb")
    parser.add_argument("--feature-lookbacks", type=str, default="1,2,3,5")
    parser.add_argument("--min-train-windows", type=str, default="4,5,6,8")
    parser.add_argument("--min-hold-windows", type=str, default="1,2,3")
    parser.add_argument("--selector-margins-per-500", type=str, default="-0.2,0.0,0.2,0.5")
    parser.add_argument("--selector-skip-thresholds-per-500", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--ridge-alphas", type=str, default="0.5,1.0,5.0,10.0")
    parser.add_argument("--logistic-c-values", type=str, default="0.25,0.5,1.0,2.0")
    parser.add_argument("--min-selected-bet-rate", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    return parser


def _safe_float(raw: str | None) -> float:
    text = str(raw).strip() if raw is not None else ""
    if text == "":
        raise InvariantError("profile_set_model_float_missing")
    value = float(text)
    if not math.isfinite(value):
        raise InvariantError("profile_set_model_float_nonfinite")
    return float(value)


def _load_compare_rows(compare_csv: Path) -> tuple[list[str], list[CompareWindowRow]]:
    if not compare_csv.exists():
        raise FileNotFoundError(f"profile_set_model_compare_missing: {compare_csv}")
    rows: list[CompareWindowRow] = []
    with compare_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InvariantError("profile_set_model_compare_header_missing")
        fieldnames = [str(x) for x in reader.fieldnames]
        per_500_names = [
            str(name[: -len("_per_500")])
            for name in fieldnames
            if str(name).endswith("_per_500") and str(name) != "tail_offset_rounds"
        ]
        if not per_500_names:
            raise InvariantError("profile_set_model_profile_columns_missing")
        for raw in reader:
            per_500 = {str(name): _safe_float(raw.get(f"{name}_per_500")) for name in per_500_names}
            bet_rate = {str(name): _safe_float(raw.get(f"{name}_bet_rate")) for name in per_500_names}
            rows.append(
                CompareWindowRow(
                    tail_offset_rounds=int(raw["tail_offset_rounds"]),
                    per_500=per_500,
                    bet_rate=bet_rate,
                )
            )
    if not rows:
        raise InvariantError("profile_set_model_compare_rows_empty")
    ordered = sorted(rows, key=lambda row: int(row.tail_offset_rounds), reverse=True)
    return [str(name) for name in per_500_names], ordered


def _feature_dict(
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
            std_value = math.sqrt(variance)
            out[f"feat_{profile}_mean_per500_l{int(lookback)}"] = float(mean_value)
            out[f"feat_{profile}_std_per500_l{int(lookback)}"] = float(std_value)
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


def _feature_names(
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
        [
            [float(feature_rows[int(idx)].get(str(name), np.nan)) for name in feature_names]
            for idx in indices
        ],
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
        return (
            np.empty((len(indices), 0), dtype=float),
            np.empty((1, 0), dtype=float),
        )
    return x_train[:, usable_mask], x_current[:, usable_mask]


def _predict_delta_ridge(
    *,
    rows: list[CompareWindowRow],
    feature_rows: list[dict[str, float]],
    feature_names: list[str],
    train_indices: list[int],
    current_idx: int,
    baseline_profile_name: str,
    profiles: list[str],
    ridge_alpha: float,
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
        out[str(profile)] = float(model.predict(x_current)[0])
    return out


def _predict_delta_logistic(
    *,
    rows: list[CompareWindowRow],
    feature_rows: list[dict[str, float]],
    feature_names: list[str],
    train_indices: list[int],
    current_idx: int,
    baseline_profile_name: str,
    profiles: list[str],
    logistic_c: float,
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
        deltas = np.asarray(
            [
                float(rows[int(idx)].per_500[str(profile)]) - float(rows[int(idx)].per_500[str(baseline_name)])
                for idx in train_indices
            ],
            dtype=float,
        )
        positive_mask = deltas > 0.0
        if np.all(positive_mask) or np.all(~positive_mask):
            out[str(profile)] = float(np.mean(deltas))
            continue
        positive_mean = float(np.mean(deltas[positive_mask])) if np.any(positive_mask) else 0.0
        nonpositive_mean = float(np.mean(deltas[~positive_mask])) if np.any(~positive_mask) else 0.0
        y_train = np.asarray(positive_mask, dtype=int)
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(C=float(logistic_c), max_iter=2000, random_state=0),
        )
        model.fit(x_train, y_train)
        p_positive = float(model.predict_proba(x_current)[0][1])
        out[str(profile)] = float(p_positive * positive_mean + (1.0 - p_positive) * nonpositive_mean)
    return out


def _predicted_stageb_per_500(
    *,
    feature_row: dict[str, float],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
) -> float:
    longest = max(int(x) for x in feature_lookbacks)
    return float(feature_row[f"feat_{str(baseline_profile_name)}_mean_per500_l{int(longest)}"])


def _pick_model_window(
    *,
    mode: str,
    rows: list[CompareWindowRow],
    feature_rows: list[dict[str, float]],
    feature_names: list[str],
    current_idx: int,
    profiles: list[str],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
    min_train_windows: int,
    ridge_alpha: float,
    logistic_c: float,
    margin_per_500: float,
    skip_threshold_per_500: float,
) -> tuple[str, float, float]:
    row = rows[int(current_idx)]
    if str(mode) == "static_profile":
        metric = float(row.per_500[str(baseline_profile_name)])
        bet_rate = float(row.bet_rate[str(baseline_profile_name)])
        return str(baseline_profile_name), float(metric), float(bet_rate)
    if str(mode) == "oracle_with_skip":
        best_name = "skip"
        best_value = 0.0
        best_bet_rate = 0.0
        for profile in profiles:
            value = float(row.per_500[str(profile)])
            if float(value) > float(best_value):
                best_name = str(profile)
                best_value = float(value)
                best_bet_rate = float(row.bet_rate[str(profile)])
        if float(best_value) <= float(skip_threshold_per_500):
            return "skip", 0.0, 0.0
        return str(best_name), float(best_value), float(best_bet_rate)

    train_indices = [
        int(idx)
        for idx in range(int(current_idx))
        if feature_rows[int(idx)] and int(idx) >= 1
    ]
    if len(train_indices) < int(min_train_windows):
        metric = float(row.per_500[str(baseline_profile_name)])
        if float(metric) <= float(skip_threshold_per_500):
            return "skip", 0.0, 0.0
        return str(baseline_profile_name), float(metric), float(row.bet_rate[str(baseline_profile_name)])

    if str(mode) == "delta_ridge":
        predictions = _predict_delta_ridge(
            rows=rows,
            feature_rows=feature_rows,
            feature_names=feature_names,
            train_indices=train_indices,
            current_idx=int(current_idx),
            baseline_profile_name=str(baseline_profile_name),
            profiles=profiles,
            ridge_alpha=float(ridge_alpha),
        )
    elif str(mode) == "delta_logistic":
        predictions = _predict_delta_logistic(
            rows=rows,
            feature_rows=feature_rows,
            feature_names=feature_names,
            train_indices=train_indices,
            current_idx=int(current_idx),
            baseline_profile_name=str(baseline_profile_name),
            profiles=profiles,
            logistic_c=float(logistic_c),
        )
    else:
        raise InvariantError("profile_set_model_mode_invalid")

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
    return (
        str(chosen_profile),
        float(row.per_500[str(chosen_profile)]),
        float(row.bet_rate[str(chosen_profile)]),
    )


def _evaluate_model_selectors(
    *,
    rows: list[CompareWindowRow],
    profiles: list[str],
    baseline_profile_name: str,
    feature_lookbacks: list[int],
    min_train_windows_list: list[int],
    min_hold_windows_list: list[int],
    margins_per_500: list[float],
    skip_thresholds_per_500: list[float],
    ridge_alphas: list[float],
    logistic_c_values: list[float],
    min_selected_bet_rate: float,
) -> list[ModelSelectorResult]:
    feature_rows = [
        _feature_dict(
            rows=rows,
            idx=int(idx),
            profiles=profiles,
            baseline_profile_name=str(baseline_profile_name),
            feature_lookbacks=feature_lookbacks,
        )
        for idx in range(len(rows))
    ]
    names = _feature_names(
        profiles=profiles,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
    )
    results: list[ModelSelectorResult] = []

    def append_mode_result(
        *,
        mode: str,
        min_train_windows: int,
        min_hold_windows: int,
        margin_per_500: float,
        skip_threshold_per_500: float,
        ridge_alpha: float,
        logistic_c: float,
    ) -> None:
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
                pick, value, bet_rate = _pick_model_window(
                    mode=str(mode),
                    rows=rows,
                    feature_rows=feature_rows,
                    feature_names=names,
                    current_idx=int(idx),
                    profiles=profiles,
                    baseline_profile_name=str(baseline_profile_name),
                    feature_lookbacks=feature_lookbacks,
                    min_train_windows=int(min_train_windows),
                    ridge_alpha=float(ridge_alpha),
                    logistic_c=float(logistic_c),
                    margin_per_500=float(margin_per_500),
                    skip_threshold_per_500=float(skip_threshold_per_500),
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
            ModelSelectorResult(
                mode=str(mode),
                feature_lookbacks_json=json.dumps(list(feature_lookbacks), separators=(",", ":")),
                min_train_windows=int(min_train_windows),
                min_hold_windows=int(min_hold_windows),
                margin_per_500=float(margin_per_500),
                skip_threshold_per_500=float(skip_threshold_per_500),
                ridge_alpha=float(ridge_alpha),
                logistic_c=float(logistic_c),
                mean_per_500=float(total / len(rows)),
                mean_selected_bet_rate=float(mean_bet_rate),
                meets_min_selected_bet_rate=float(mean_bet_rate) >= float(min_selected_bet_rate),
                switch_count=int(switch_count),
                pick_counts_json=json.dumps(dict(sorted(picks.items())), separators=(",", ":"), sort_keys=True),
            )
        )

    for skip_threshold in skip_thresholds_per_500:
        append_mode_result(
            mode="oracle_with_skip",
            min_train_windows=0,
            min_hold_windows=1,
            margin_per_500=0.0,
            skip_threshold_per_500=float(skip_threshold),
            ridge_alpha=0.0,
            logistic_c=0.0,
        )
    append_mode_result(
        mode="static_profile",
        min_train_windows=0,
        min_hold_windows=1,
        margin_per_500=0.0,
        skip_threshold_per_500=0.0,
        ridge_alpha=0.0,
        logistic_c=0.0,
    )
    for min_train_windows in min_train_windows_list:
        for min_hold_windows in min_hold_windows_list:
            for margin in margins_per_500:
                for skip_threshold in skip_thresholds_per_500:
                    for ridge_alpha in ridge_alphas:
                        append_mode_result(
                            mode="delta_ridge",
                            min_train_windows=int(min_train_windows),
                            min_hold_windows=int(min_hold_windows),
                            margin_per_500=float(margin),
                            skip_threshold_per_500=float(skip_threshold),
                            ridge_alpha=float(ridge_alpha),
                            logistic_c=0.0,
                        )
                    for logistic_c in logistic_c_values:
                        append_mode_result(
                            mode="delta_logistic",
                            min_train_windows=int(min_train_windows),
                            min_hold_windows=int(min_hold_windows),
                            margin_per_500=float(margin),
                            skip_threshold_per_500=float(skip_threshold),
                            ridge_alpha=0.0,
                            logistic_c=float(logistic_c),
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


def main() -> None:
    args = _build_parser().parse_args()
    if float(args.min_selected_bet_rate) < 0.0:
        raise InvariantError("profile_set_model_min_selected_bet_rate_negative")
    profiles, rows = _load_compare_rows(Path(str(args.compare_csv)).resolve())
    baseline_profile_name = str(args.baseline_profile_name).strip()
    if baseline_profile_name == "" or baseline_profile_name not in profiles:
        raise InvariantError("profile_set_model_baseline_profile_invalid")
    feature_lookbacks = _parse_positive_int_list(str(args.feature_lookbacks))
    min_train_windows_list = _parse_positive_int_list(str(args.min_train_windows))
    min_hold_windows_list = _parse_positive_int_list(str(args.min_hold_windows))
    margins_per_500 = _parse_float_list(str(args.selector_margins_per_500))
    skip_thresholds_per_500 = _parse_float_list(str(args.selector_skip_thresholds_per_500))
    ridge_alphas = _parse_float_list(str(args.ridge_alphas))
    logistic_c_values = _parse_float_list(str(args.logistic_c_values))
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _evaluate_model_selectors(
        rows=rows,
        profiles=profiles,
        baseline_profile_name=str(baseline_profile_name),
        feature_lookbacks=feature_lookbacks,
        min_train_windows_list=min_train_windows_list,
        min_hold_windows_list=min_hold_windows_list,
        margins_per_500=margins_per_500,
        skip_thresholds_per_500=skip_thresholds_per_500,
        ridge_alphas=ridge_alphas,
        logistic_c_values=logistic_c_values,
        min_selected_bet_rate=float(args.min_selected_bet_rate),
    )

    csv_path = output_dir / f"{args.name_prefix}_profile_set_model_selectors.csv"
    json_path = output_dir / f"{args.name_prefix}_profile_set_model_selectors.json"
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
