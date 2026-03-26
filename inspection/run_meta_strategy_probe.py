"""Run walk-forward probes against a block-level meta-strategy dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inspection.strategy_router_common import parse_strategy_prefixes

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - import depends on local env
    LGBMRegressor = None


warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


_SELECTOR_MODES = (
    "skip_only",
    "fixed_strategy",
    "static_best_so_far",
    "trailing_mean",
    "online_knn_mean",
    "delta_trailing_mean",
    "delta_logistic",
    "delta_ridge",
    "delta_elastic_net",
    "delta_hgb",
    "delta_hgb_classifier",
    "delta_random_forest",
    "delta_extra_trees",
    "delta_gradient_boosting",
    "delta_lgbm",
)


@dataclass(frozen=True, slots=True)
class MetaDatasetRow:
    """One block-level decision row from the meta-strategy dataset."""

    target_block_index: int
    target_sim_offset_rounds: int
    target_epoch_start: int
    target_epoch_end: int
    target_num_rounds: int
    history_block_count: int
    features: dict[str, float | None]
    labels: dict[str, float]
    oracle_strategy_or_skip: str
    oracle_profit_bnb: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--dataset-csv", type=str, required=True)
    parser.add_argument("--dataset-meta", type=str, required=True)
    parser.add_argument("--selector-mode", type=str, choices=_SELECTOR_MODES, required=True)
    parser.add_argument("--active-strategy-names", type=str, default="")
    parser.add_argument("--fixed-strategy-name", type=str, default="")
    parser.add_argument("--baseline-strategy-name", type=str, default="")
    parser.add_argument("--trailing-history-blocks", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=7)
    parser.add_argument("--min-train-rows", type=int, default=8)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--elastic-net-alpha", type=float, default=0.05)
    parser.add_argument("--elastic-net-l1-ratio", type=float, default=0.5)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-depth", type=int, default=3)
    parser.add_argument("--hgb-max-iter", type=int, default=200)
    parser.add_argument("--hgb-l2-regularization", type=float, default=1.0)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=6)
    parser.add_argument("--tree-n-estimators", type=int, default=200)
    parser.add_argument("--tree-max-depth", type=int, default=3)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=6)
    parser.add_argument("--gbr-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-n-estimators", type=int, default=200)
    parser.add_argument("--lgbm-num-leaves", type=int, default=15)
    parser.add_argument("--lgbm-max-depth", type=int, default=3)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=10)
    parser.add_argument("--safety-margin-bnb", type=float, default=0.0)
    parser.add_argument("--min-hold-blocks", type=int, default=1)
    parser.add_argument("--baseline-mode", type=str, choices=("skip_all",), default="skip_all")
    parser.add_argument("--starting-bankroll-bnb", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--write-decisions", action="store_true", default=False)
    return parser


def _safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        value = float(text)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _load_meta(meta_path: Path) -> tuple[list[str], dict[str, str], list[str]]:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta_strategy_probe_meta_missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    strategies = [str(x) for x in meta.get("strategy_prefixes", [])]
    key_map = {str(k): str(v) for k, v in meta.get("strategy_column_keys", {}).items()}
    feature_columns = [str(x) for x in meta.get("feature_columns", [])]
    if not strategies:
        raise ValueError("meta_strategy_probe_meta_strategy_prefixes_missing")
    if not key_map:
        raise ValueError("meta_strategy_probe_meta_strategy_key_map_missing")
    if not feature_columns:
        raise ValueError("meta_strategy_probe_meta_feature_columns_missing")
    return strategies, key_map, feature_columns


def _load_rows(
    *,
    dataset_csv: Path,
    strategies: list[str],
    key_map: dict[str, str],
    feature_columns: list[str],
) -> list[MetaDatasetRow]:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"meta_strategy_probe_dataset_missing: {dataset_csv}")

    rows: list[MetaDatasetRow] = []
    with dataset_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            labels: dict[str, float] = {}
            for strategy in strategies:
                key = key_map[str(strategy)]
                label_profit = _safe_float(raw.get(f"label_{key}_next_block_profit_bnb"))
                if label_profit is None:
                    raise ValueError("meta_strategy_probe_label_profit_missing")
                labels[str(strategy)] = float(label_profit)
            rows.append(
                MetaDatasetRow(
                    target_block_index=int(raw["target_block_index"]),
                    target_sim_offset_rounds=int(raw["target_sim_offset_rounds"]),
                    target_epoch_start=int(raw["target_epoch_start"]),
                    target_epoch_end=int(raw["target_epoch_end"]),
                    target_num_rounds=int(raw["target_num_rounds"]),
                    history_block_count=int(raw["history_block_count"]),
                    features={
                        str(column): _safe_float(raw.get(str(column)))
                        for column in feature_columns
                    },
                    labels=labels,
                    oracle_strategy_or_skip=str(raw.get("label_oracle_strategy_or_skip", "SKIP")),
                    oracle_profit_bnb=float(_safe_float(raw.get("label_oracle_profit_bnb")) or 0.0),
                )
            )
    if not rows:
        raise ValueError("meta_strategy_probe_dataset_empty")
    return rows


def _resolve_active_strategies(
    *,
    all_strategies: list[str],
    raw_active_names: str,
) -> list[str]:
    raw = str(raw_active_names).strip()
    if raw == "":
        return [str(x) for x in all_strategies]
    names = parse_strategy_prefixes(raw)
    unknown = [name for name in names if str(name) not in all_strategies]
    if unknown:
        raise ValueError(f"meta_strategy_probe_active_strategy_unknown: {unknown[0]}")
    return [str(x) for x in names]


def _select_feature_columns(
    *,
    feature_columns: list[str],
    strategies: list[str],
    key_map: dict[str, str],
) -> list[str]:
    active_prefixes = {f"feat_{key_map[str(strategy)]}_" for strategy in strategies}
    out: list[str] = []
    for column in feature_columns:
        name = str(column)
        if name.startswith("feat_regime_"):
            out.append(str(name))
            continue
        if any(str(name).startswith(prefix) for prefix in active_prefixes):
            out.append(str(name))
    if not out:
        raise ValueError("meta_strategy_probe_feature_columns_empty_after_filter")
    return out


def _predict_static_best_so_far(
    *,
    train_rows: list[MetaDatasetRow],
    strategies: list[str],
) -> dict[str, float]:
    if not train_rows:
        return {}
    return {
        str(strategy): float(sum(row.labels[str(strategy)] for row in train_rows)) / float(len(train_rows))
        for strategy in strategies
    }


def _predict_trailing_mean(
    *,
    train_rows: list[MetaDatasetRow],
    strategies: list[str],
    trailing_history_blocks: int,
) -> dict[str, float]:
    if int(trailing_history_blocks) <= 0:
        raise ValueError("meta_strategy_probe_trailing_history_blocks_nonpositive")
    hist = train_rows[-int(trailing_history_blocks) :]
    if not hist:
        return {}
    return {
        str(strategy): float(sum(row.labels[str(strategy)] for row in hist)) / float(len(hist))
        for strategy in strategies
    }


def _predict_online_knn_mean(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    knn_k: int,
) -> dict[str, float]:
    if int(knn_k) <= 0:
        raise ValueError("meta_strategy_probe_knn_k_nonpositive")
    if len(train_rows) < int(knn_k):
        return {}

    feature_means: dict[str, float] = {}
    feature_stds: dict[str, float] = {}
    for column in feature_columns:
        values = [
            float(row.features[str(column)])
            for row in train_rows
            if row.features.get(str(column)) is not None
        ]
        if values:
            mean_value = float(sum(values) / len(values))
            variance = float(sum((value - mean_value) ** 2 for value in values) / len(values))
            std_value = math.sqrt(variance)
            feature_means[str(column)] = float(mean_value)
            feature_stds[str(column)] = float(std_value) if float(std_value) > 0.0 else 1.0
        else:
            feature_means[str(column)] = 0.0
            feature_stds[str(column)] = 1.0

    def encode(row: MetaDatasetRow) -> list[float]:
        out: list[float] = []
        for column in feature_columns:
            mean_value = float(feature_means[str(column)])
            raw_value = row.features.get(str(column))
            value = float(raw_value) if raw_value is not None else float(mean_value)
            out.append((float(value) - float(mean_value)) / float(feature_stds[str(column)]))
        return out

    current_vector = encode(current_row)
    distance_rows: list[tuple[float, MetaDatasetRow]] = []
    for row in train_rows:
        train_vector = encode(row)
        distance = math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(current_vector, train_vector, strict=False))
        )
        distance_rows.append((float(distance), row))
    distance_rows.sort(key=lambda item: (float(item[0]), int(item[1].target_block_index)))
    neighbors = [row for _, row in distance_rows[: int(knn_k)]]
    return {
        str(strategy): float(sum(row.labels[str(strategy)] for row in neighbors)) / float(len(neighbors))
        for strategy in strategies
    }


def _predict_delta_trailing_mean(
    *,
    train_rows: list[MetaDatasetRow],
    strategies: list[str],
    baseline_strategy_name: str,
    trailing_history_blocks: int,
) -> dict[str, float]:
    if int(trailing_history_blocks) <= 0:
        raise ValueError("meta_strategy_probe_trailing_history_blocks_nonpositive")
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    hist = train_rows[-int(trailing_history_blocks) :]
    if not hist:
        return {}

    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        deltas = [
            float(row.labels[str(strategy)]) - float(row.labels[str(baseline_name)])
            for row in hist
        ]
        predictions[str(strategy)] = float(sum(deltas) / len(deltas))
    return predictions


def _predict_delta_ridge(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    baseline_strategy_name: str,
    ridge_alpha: float,
) -> dict[str, float]:
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    if float(ridge_alpha) < 0.0:
        raise ValueError("meta_strategy_probe_ridge_alpha_negative")
    if not train_rows:
        return {}

    x_train, x_current = _feature_matrices(
        train_rows=train_rows,
        current_row=current_row,
        feature_columns=feature_columns,
    )
    if int(x_train.shape[1]) <= 0:
        return _predict_delta_train_mean(
            train_rows=train_rows,
            strategies=strategies,
            baseline_strategy_name=str(baseline_name),
        )
    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        y_train = np.asarray(
            [
                float(row.labels[str(strategy)]) - float(row.labels[str(baseline_name)])
                for row in train_rows
            ],
            dtype=float,
        )
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            Ridge(alpha=float(ridge_alpha)),
        )
        model.fit(x_train, y_train)
        predicted = float(model.predict(x_current)[0])
        predictions[str(strategy)] = float(predicted)
    return predictions


def _predict_delta_logistic(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    baseline_strategy_name: str,
    logistic_c: float,
) -> dict[str, float]:
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    if float(logistic_c) <= 0.0:
        raise ValueError("meta_strategy_probe_logistic_c_nonpositive")
    if not train_rows:
        return {}

    x_train, x_current = _feature_matrices(
        train_rows=train_rows,
        current_row=current_row,
        feature_columns=feature_columns,
    )
    if int(x_train.shape[1]) <= 0:
        return _predict_delta_train_mean(
            train_rows=train_rows,
            strategies=strategies,
            baseline_strategy_name=str(baseline_name),
        )
    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        delta_values = _delta_target_array(
            train_rows=train_rows,
            strategy_name=str(strategy),
            baseline_name=str(baseline_name),
        )
        positive_mask = delta_values > 0.0
        if np.all(positive_mask) or np.all(~positive_mask):
            predictions[str(strategy)] = float(np.mean(delta_values))
            continue
        positive_mean = float(np.mean(delta_values[positive_mask])) if np.any(positive_mask) else 0.0
        nonpositive_mean = float(np.mean(delta_values[~positive_mask])) if np.any(~positive_mask) else 0.0
        y_train = np.asarray(positive_mask, dtype=int)
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(
                C=float(logistic_c),
                max_iter=2000,
                random_state=0,
            ),
        )
        model.fit(x_train, y_train)
        p_positive = float(model.predict_proba(x_current)[0][1])
        predictions[str(strategy)] = float(
            float(p_positive) * float(positive_mean) + (1.0 - float(p_positive)) * float(nonpositive_mean)
        )
    return predictions


def _feature_matrices(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_train_full = np.asarray(
        [
            [
                (float(row.features[str(column)]) if row.features.get(str(column)) is not None else np.nan)
                for column in feature_columns
            ]
            for row in train_rows
        ],
        dtype=float,
    )
    x_current_full = np.asarray(
        [
            [
                (
                    float(current_row.features[str(column)])
                    if current_row.features.get(str(column)) is not None
                    else np.nan
                )
                for column in feature_columns
            ]
        ],
        dtype=float,
    )
    if int(x_train_full.shape[1]) <= 0:
        return x_train_full, x_current_full
    usable_mask = ~np.all(np.isnan(x_train_full), axis=0)
    if not np.any(usable_mask):
        return (
            np.empty((len(train_rows), 0), dtype=float),
            np.empty((1, 0), dtype=float),
        )
    return x_train_full[:, usable_mask], x_current_full[:, usable_mask]


def _delta_target_array(
    *,
    train_rows: list[MetaDatasetRow],
    strategy_name: str,
    baseline_name: str,
) -> np.ndarray:
    return np.asarray(
        [
            float(row.labels[str(strategy_name)]) - float(row.labels[str(baseline_name)])
            for row in train_rows
        ],
        dtype=float,
    )


def _predict_delta_train_mean(
    *,
    train_rows: list[MetaDatasetRow],
    strategies: list[str],
    baseline_strategy_name: str,
) -> dict[str, float]:
    baseline_name = str(baseline_strategy_name).strip()
    return {
        str(strategy): float(
            np.mean(
                _delta_target_array(
                    train_rows=train_rows,
                    strategy_name=str(strategy),
                    baseline_name=str(baseline_name),
                )
            )
        )
        for strategy in strategies
        if str(strategy) != str(baseline_name)
    }


def _predict_delta_elastic_net(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    baseline_strategy_name: str,
    elastic_net_alpha: float,
    elastic_net_l1_ratio: float,
) -> dict[str, float]:
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    if float(elastic_net_alpha) < 0.0:
        raise ValueError("meta_strategy_probe_elastic_net_alpha_negative")
    if float(elastic_net_l1_ratio) < 0.0 or float(elastic_net_l1_ratio) > 1.0:
        raise ValueError("meta_strategy_probe_elastic_net_l1_ratio_out_of_range")
    if not train_rows:
        return {}

    x_train, x_current = _feature_matrices(
        train_rows=train_rows,
        current_row=current_row,
        feature_columns=feature_columns,
    )
    if int(x_train.shape[1]) <= 0:
        return _predict_delta_train_mean(
            train_rows=train_rows,
            strategies=strategies,
            baseline_strategy_name=str(baseline_name),
        )
    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        y_train = _delta_target_array(
            train_rows=train_rows,
            strategy_name=str(strategy),
            baseline_name=str(baseline_name),
        )
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            ElasticNet(
                alpha=float(elastic_net_alpha),
                l1_ratio=float(elastic_net_l1_ratio),
                max_iter=10000,
                random_state=0,
            ),
        )
        model.fit(x_train, y_train)
        predicted = float(model.predict(x_current)[0])
        predictions[str(strategy)] = float(predicted)
    return predictions


def _predict_delta_hgb(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    baseline_strategy_name: str,
    hgb_learning_rate: float,
    hgb_max_depth: int,
    hgb_max_iter: int,
    hgb_l2_regularization: float,
    hgb_min_samples_leaf: int,
) -> dict[str, float]:
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    if float(hgb_learning_rate) <= 0.0:
        raise ValueError("meta_strategy_probe_hgb_learning_rate_nonpositive")
    if int(hgb_max_depth) <= 0:
        raise ValueError("meta_strategy_probe_hgb_max_depth_nonpositive")
    if int(hgb_max_iter) <= 0:
        raise ValueError("meta_strategy_probe_hgb_max_iter_nonpositive")
    if float(hgb_l2_regularization) < 0.0:
        raise ValueError("meta_strategy_probe_hgb_l2_negative")
    if int(hgb_min_samples_leaf) <= 0:
        raise ValueError("meta_strategy_probe_hgb_min_samples_leaf_nonpositive")
    if not train_rows:
        return {}

    x_train, x_current = _feature_matrices(
        train_rows=train_rows,
        current_row=current_row,
        feature_columns=feature_columns,
    )
    if int(x_train.shape[1]) <= 0:
        return _predict_delta_train_mean(
            train_rows=train_rows,
            strategies=strategies,
            baseline_strategy_name=str(baseline_name),
        )
    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        y_train = _delta_target_array(
            train_rows=train_rows,
            strategy_name=str(strategy),
            baseline_name=str(baseline_name),
        )
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            HistGradientBoostingRegressor(
                learning_rate=float(hgb_learning_rate),
                max_depth=int(hgb_max_depth),
                max_iter=int(hgb_max_iter),
                l2_regularization=float(hgb_l2_regularization),
                min_samples_leaf=int(hgb_min_samples_leaf),
                random_state=0,
            ),
        )
        model.fit(x_train, y_train)
        predicted = float(model.predict(x_current)[0])
        predictions[str(strategy)] = float(predicted)
    return predictions


def _predict_delta_hgb_classifier(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    baseline_strategy_name: str,
    hgb_learning_rate: float,
    hgb_max_depth: int,
    hgb_max_iter: int,
    hgb_l2_regularization: float,
    hgb_min_samples_leaf: int,
) -> dict[str, float]:
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    if float(hgb_learning_rate) <= 0.0:
        raise ValueError("meta_strategy_probe_hgb_learning_rate_nonpositive")
    if int(hgb_max_depth) <= 0:
        raise ValueError("meta_strategy_probe_hgb_max_depth_nonpositive")
    if int(hgb_max_iter) <= 0:
        raise ValueError("meta_strategy_probe_hgb_max_iter_nonpositive")
    if float(hgb_l2_regularization) < 0.0:
        raise ValueError("meta_strategy_probe_hgb_l2_negative")
    if int(hgb_min_samples_leaf) <= 0:
        raise ValueError("meta_strategy_probe_hgb_min_samples_leaf_nonpositive")
    if not train_rows:
        return {}

    x_train, x_current = _feature_matrices(
        train_rows=train_rows,
        current_row=current_row,
        feature_columns=feature_columns,
    )
    if int(x_train.shape[1]) <= 0:
        return _predict_delta_train_mean(
            train_rows=train_rows,
            strategies=strategies,
            baseline_strategy_name=str(baseline_name),
        )
    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        delta_values = _delta_target_array(
            train_rows=train_rows,
            strategy_name=str(strategy),
            baseline_name=str(baseline_name),
        )
        positive_mask = delta_values > 0.0
        if np.all(positive_mask) or np.all(~positive_mask):
            predictions[str(strategy)] = float(np.mean(delta_values))
            continue
        positive_mean = float(np.mean(delta_values[positive_mask])) if np.any(positive_mask) else 0.0
        nonpositive_mean = float(np.mean(delta_values[~positive_mask])) if np.any(~positive_mask) else 0.0
        y_train = np.asarray(positive_mask, dtype=int)
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            HistGradientBoostingClassifier(
                learning_rate=float(hgb_learning_rate),
                max_depth=int(hgb_max_depth),
                max_iter=int(hgb_max_iter),
                l2_regularization=float(hgb_l2_regularization),
                min_samples_leaf=int(hgb_min_samples_leaf),
                random_state=0,
            ),
        )
        model.fit(x_train, y_train)
        p_positive = float(model.predict_proba(x_current)[0][1])
        predictions[str(strategy)] = float(
            float(p_positive) * float(positive_mean) + (1.0 - float(p_positive)) * float(nonpositive_mean)
        )
    return predictions


def _predict_delta_random_forest(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    baseline_strategy_name: str,
    tree_n_estimators: int,
    tree_max_depth: int,
    tree_min_samples_leaf: int,
) -> dict[str, float]:
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    if int(tree_n_estimators) <= 0:
        raise ValueError("meta_strategy_probe_tree_n_estimators_nonpositive")
    if int(tree_max_depth) <= 0:
        raise ValueError("meta_strategy_probe_tree_max_depth_nonpositive")
    if int(tree_min_samples_leaf) <= 0:
        raise ValueError("meta_strategy_probe_tree_min_samples_leaf_nonpositive")
    if not train_rows:
        return {}

    x_train, x_current = _feature_matrices(
        train_rows=train_rows,
        current_row=current_row,
        feature_columns=feature_columns,
    )
    if int(x_train.shape[1]) <= 0:
        return _predict_delta_train_mean(
            train_rows=train_rows,
            strategies=strategies,
            baseline_strategy_name=str(baseline_name),
        )
    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        y_train = _delta_target_array(
            train_rows=train_rows,
            strategy_name=str(strategy),
            baseline_name=str(baseline_name),
        )
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            RandomForestRegressor(
                n_estimators=int(tree_n_estimators),
                max_depth=int(tree_max_depth),
                min_samples_leaf=int(tree_min_samples_leaf),
                random_state=0,
                n_jobs=1,
            ),
        )
        model.fit(x_train, y_train)
        predictions[str(strategy)] = float(model.predict(x_current)[0])
    return predictions


def _predict_delta_extra_trees(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    baseline_strategy_name: str,
    tree_n_estimators: int,
    tree_max_depth: int,
    tree_min_samples_leaf: int,
) -> dict[str, float]:
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    if int(tree_n_estimators) <= 0:
        raise ValueError("meta_strategy_probe_tree_n_estimators_nonpositive")
    if int(tree_max_depth) <= 0:
        raise ValueError("meta_strategy_probe_tree_max_depth_nonpositive")
    if int(tree_min_samples_leaf) <= 0:
        raise ValueError("meta_strategy_probe_tree_min_samples_leaf_nonpositive")
    if not train_rows:
        return {}

    x_train, x_current = _feature_matrices(
        train_rows=train_rows,
        current_row=current_row,
        feature_columns=feature_columns,
    )
    if int(x_train.shape[1]) <= 0:
        return _predict_delta_train_mean(
            train_rows=train_rows,
            strategies=strategies,
            baseline_strategy_name=str(baseline_name),
        )
    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        y_train = _delta_target_array(
            train_rows=train_rows,
            strategy_name=str(strategy),
            baseline_name=str(baseline_name),
        )
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            ExtraTreesRegressor(
                n_estimators=int(tree_n_estimators),
                max_depth=int(tree_max_depth),
                min_samples_leaf=int(tree_min_samples_leaf),
                random_state=0,
                n_jobs=1,
            ),
        )
        model.fit(x_train, y_train)
        predictions[str(strategy)] = float(model.predict(x_current)[0])
    return predictions


def _predict_delta_gradient_boosting(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    baseline_strategy_name: str,
    tree_n_estimators: int,
    tree_max_depth: int,
    tree_min_samples_leaf: int,
    gbr_learning_rate: float,
) -> dict[str, float]:
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    if int(tree_n_estimators) <= 0:
        raise ValueError("meta_strategy_probe_tree_n_estimators_nonpositive")
    if int(tree_max_depth) <= 0:
        raise ValueError("meta_strategy_probe_tree_max_depth_nonpositive")
    if int(tree_min_samples_leaf) <= 0:
        raise ValueError("meta_strategy_probe_tree_min_samples_leaf_nonpositive")
    if float(gbr_learning_rate) <= 0.0:
        raise ValueError("meta_strategy_probe_gbr_learning_rate_nonpositive")
    if not train_rows:
        return {}

    x_train, x_current = _feature_matrices(
        train_rows=train_rows,
        current_row=current_row,
        feature_columns=feature_columns,
    )
    if int(x_train.shape[1]) <= 0:
        return _predict_delta_train_mean(
            train_rows=train_rows,
            strategies=strategies,
            baseline_strategy_name=str(baseline_name),
        )
    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        y_train = _delta_target_array(
            train_rows=train_rows,
            strategy_name=str(strategy),
            baseline_name=str(baseline_name),
        )
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            GradientBoostingRegressor(
                n_estimators=int(tree_n_estimators),
                learning_rate=float(gbr_learning_rate),
                max_depth=int(tree_max_depth),
                min_samples_leaf=int(tree_min_samples_leaf),
                random_state=0,
            ),
        )
        model.fit(x_train, y_train)
        predictions[str(strategy)] = float(model.predict(x_current)[0])
    return predictions


def _predict_delta_lgbm(
    *,
    train_rows: list[MetaDatasetRow],
    current_row: MetaDatasetRow,
    strategies: list[str],
    feature_columns: list[str],
    baseline_strategy_name: str,
    lgbm_learning_rate: float,
    lgbm_n_estimators: int,
    lgbm_num_leaves: int,
    lgbm_max_depth: int,
    lgbm_min_child_samples: int,
) -> dict[str, float]:
    if LGBMRegressor is None:
        raise RuntimeError("meta_strategy_probe_lgbm_missing")
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if baseline_name not in strategies:
        raise ValueError(f"meta_strategy_probe_baseline_strategy_unknown: {baseline_name}")
    if float(lgbm_learning_rate) <= 0.0:
        raise ValueError("meta_strategy_probe_lgbm_learning_rate_nonpositive")
    if int(lgbm_n_estimators) <= 0:
        raise ValueError("meta_strategy_probe_lgbm_n_estimators_nonpositive")
    if int(lgbm_num_leaves) <= 1:
        raise ValueError("meta_strategy_probe_lgbm_num_leaves_too_small")
    if int(lgbm_max_depth) == 0 or int(lgbm_max_depth) < -1:
        raise ValueError("meta_strategy_probe_lgbm_max_depth_invalid")
    if int(lgbm_min_child_samples) <= 0:
        raise ValueError("meta_strategy_probe_lgbm_min_child_samples_nonpositive")
    if not train_rows:
        return {}

    x_train, x_current = _feature_matrices(
        train_rows=train_rows,
        current_row=current_row,
        feature_columns=feature_columns,
    )
    if int(x_train.shape[1]) <= 0:
        return _predict_delta_train_mean(
            train_rows=train_rows,
            strategies=strategies,
            baseline_strategy_name=str(baseline_name),
        )
    predictions: dict[str, float] = {}
    for strategy in strategies:
        if str(strategy) == str(baseline_name):
            continue
        y_train = _delta_target_array(
            train_rows=train_rows,
            strategy_name=str(strategy),
            baseline_name=str(baseline_name),
        )
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            LGBMRegressor(
                learning_rate=float(lgbm_learning_rate),
                n_estimators=int(lgbm_n_estimators),
                num_leaves=int(lgbm_num_leaves),
                max_depth=int(lgbm_max_depth),
                min_child_samples=int(lgbm_min_child_samples),
                objective="regression",
                random_state=0,
                n_jobs=1,
                verbose=-1,
            ),
        )
        model.fit(x_train, y_train)
        predictions[str(strategy)] = float(model.predict(x_current)[0])
    return predictions


def _pick_with_skip_safety(
    *,
    predictions: dict[str, float],
    safety_margin_bnb: float,
    baseline_mode: str,
) -> tuple[str, float, str]:
    if str(baseline_mode) != "skip_all":
        raise ValueError("meta_strategy_probe_baseline_mode_invalid")
    if not predictions:
        return "SKIP", 0.0, "no_prediction"

    best_strategy = "SKIP"
    best_predicted_bnb = float("-inf")
    for strategy, predicted_bnb in predictions.items():
        if float(predicted_bnb) > float(best_predicted_bnb):
            best_predicted_bnb = float(predicted_bnb)
            best_strategy = str(strategy)

    baseline_predicted_bnb = 0.0
    if float(best_predicted_bnb) < 0.0:
        return "SKIP", float(best_predicted_bnb), "predicted_negative"
    if float(best_predicted_bnb) < float(baseline_predicted_bnb) + float(safety_margin_bnb):
        return "SKIP", float(best_predicted_bnb), "below_safety_margin"
    return str(best_strategy), float(best_predicted_bnb), "predicted_best"


def _pick_against_baseline_delta(
    *,
    predictions: dict[str, float],
    baseline_strategy_name: str,
    safety_margin_bnb: float,
    incumbent_pick: str | None,
) -> tuple[str, float, str]:
    baseline_name = str(baseline_strategy_name).strip()
    if baseline_name == "":
        raise ValueError("meta_strategy_probe_baseline_strategy_name_missing")
    if not predictions:
        incumbent = str(incumbent_pick or baseline_name)
        if incumbent == "":
            incumbent = str(baseline_name)
        return str(incumbent), 0.0, "stay_on_baseline_no_prediction"

    scores: dict[str, float] = {str(baseline_name): 0.0}
    for strategy, predicted_delta_bnb in predictions.items():
        scores[str(strategy)] = float(predicted_delta_bnb)

    incumbent = str(incumbent_pick or baseline_name)
    if incumbent not in scores:
        incumbent = str(baseline_name)
    incumbent_score = float(scores[str(incumbent)])

    best_strategy = str(incumbent)
    best_delta_bnb = float(incumbent_score)
    for strategy, predicted_delta_bnb in scores.items():
        if float(predicted_delta_bnb) > float(best_delta_bnb):
            best_delta_bnb = float(predicted_delta_bnb)
            best_strategy = str(strategy)
    if str(best_strategy) != str(incumbent) and float(best_delta_bnb) < float(incumbent_score) + float(
        safety_margin_bnb
    ):
        if str(incumbent) == str(baseline_name):
            return str(incumbent), float(best_delta_bnb), "stay_on_baseline_margin"
        return str(incumbent), float(best_delta_bnb), "hold_incumbent_margin"
    if str(best_strategy) == str(baseline_name):
        return str(best_strategy), float(best_delta_bnb), "stay_on_baseline_margin"
    return str(best_strategy), float(best_delta_bnb), "switch_from_baseline"


def _active_oracle(
    *,
    row: MetaDatasetRow,
    strategies: list[str],
) -> tuple[str, float]:
    best_strategy = "SKIP"
    best_profit_bnb = 0.0
    for strategy in strategies:
        profit_bnb = float(row.labels[str(strategy)])
        if float(profit_bnb) > float(best_profit_bnb):
            best_profit_bnb = float(profit_bnb)
            best_strategy = str(strategy)
    return str(best_strategy), float(best_profit_bnb)


def _run_probe(
    *,
    rows: list[MetaDatasetRow],
    strategies: list[str],
    feature_columns: list[str],
    selector_mode: str,
    trailing_history_blocks: int,
    knn_k: int,
    min_train_rows: int,
    logistic_c: float,
    ridge_alpha: float,
    elastic_net_alpha: float,
    elastic_net_l1_ratio: float,
    hgb_learning_rate: float,
    hgb_max_depth: int,
    hgb_max_iter: int,
    hgb_l2_regularization: float,
    hgb_min_samples_leaf: int,
    tree_n_estimators: int = 200,
    tree_max_depth: int = 3,
    tree_min_samples_leaf: int = 6,
    gbr_learning_rate: float = 0.05,
    lgbm_learning_rate: float = 0.05,
    lgbm_n_estimators: int = 200,
    lgbm_num_leaves: int = 15,
    lgbm_max_depth: int = 3,
    lgbm_min_child_samples: int = 10,
    safety_margin_bnb: float,
    min_hold_blocks: int,
    baseline_mode: str,
    starting_bankroll_bnb: float,
    fixed_strategy_name: str,
    baseline_strategy_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if str(selector_mode) not in _SELECTOR_MODES:
        raise ValueError("meta_strategy_probe_selector_mode_invalid")
    if int(min_train_rows) < 0:
        raise ValueError("meta_strategy_probe_min_train_rows_negative")
    if int(min_hold_blocks) <= 0:
        raise ValueError("meta_strategy_probe_min_hold_blocks_nonpositive")

    cumulative_net_bnb = 0.0
    peak_net_bnb = 0.0
    max_drawdown_bnb = 0.0
    min_bankroll_bnb = float(starting_bankroll_bnb)
    total_rounds = 0
    num_active_blocks = 0
    num_skip_blocks = 0
    num_positive_active_blocks = 0
    num_switches = 0
    oracle_total_bnb = 0.0
    pick_reasons: dict[str, int] = {}
    picks_by_strategy: dict[str, int] = {str(strategy): 0 for strategy in strategies}
    picks_by_strategy["SKIP"] = 0
    decision_rows: list[dict[str, Any]] = []
    previous_pick: str | None = None
    hold_remaining = 0

    for idx, row in enumerate(rows):
        train_rows = rows[:idx]
        predictions: dict[str, float] = {}
        oracle_strategy_or_skip, oracle_profit_bnb = _active_oracle(row=row, strategies=strategies)

        if int(hold_remaining) > 0 and previous_pick is not None:
            pick = str(previous_pick)
            pick_score_bnb = 0.0
            pick_reason = "hold_interval"
            hold_remaining = int(hold_remaining - 1)
        elif str(selector_mode) == "skip_only":
            pick = "SKIP"
            pick_score_bnb = 0.0
            pick_reason = "skip_only_mode"
        elif str(selector_mode) == "fixed_strategy":
            strategy_name = str(fixed_strategy_name).strip()
            if strategy_name == "":
                raise ValueError("meta_strategy_probe_fixed_strategy_name_missing")
            if strategy_name not in strategies:
                raise ValueError(f"meta_strategy_probe_fixed_strategy_unknown: {strategy_name}")
            pick = str(strategy_name)
            pick_score_bnb = 0.0
            pick_reason = "fixed_strategy_mode"
        else:
            if str(selector_mode) == "static_best_so_far":
                predictions = _predict_static_best_so_far(
                    train_rows=train_rows,
                    strategies=strategies,
                )
            elif str(selector_mode) == "trailing_mean":
                predictions = _predict_trailing_mean(
                    train_rows=train_rows,
                    strategies=strategies,
                    trailing_history_blocks=int(trailing_history_blocks),
                )
            elif str(selector_mode) == "online_knn_mean":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_online_knn_mean(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        knn_k=int(knn_k),
                    )
            elif str(selector_mode) == "delta_trailing_mean":
                predictions = _predict_delta_trailing_mean(
                    train_rows=train_rows,
                    strategies=strategies,
                    baseline_strategy_name=str(baseline_strategy_name),
                    trailing_history_blocks=int(trailing_history_blocks),
                )
            elif str(selector_mode) == "delta_logistic":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_delta_logistic(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        baseline_strategy_name=str(baseline_strategy_name),
                        logistic_c=float(logistic_c),
                    )
            elif str(selector_mode) == "delta_ridge":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_delta_ridge(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        baseline_strategy_name=str(baseline_strategy_name),
                        ridge_alpha=float(ridge_alpha),
                    )
            elif str(selector_mode) == "delta_elastic_net":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_delta_elastic_net(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        baseline_strategy_name=str(baseline_strategy_name),
                        elastic_net_alpha=float(elastic_net_alpha),
                        elastic_net_l1_ratio=float(elastic_net_l1_ratio),
                    )
            elif str(selector_mode) == "delta_hgb":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_delta_hgb(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        baseline_strategy_name=str(baseline_strategy_name),
                        hgb_learning_rate=float(hgb_learning_rate),
                        hgb_max_depth=int(hgb_max_depth),
                        hgb_max_iter=int(hgb_max_iter),
                        hgb_l2_regularization=float(hgb_l2_regularization),
                        hgb_min_samples_leaf=int(hgb_min_samples_leaf),
                    )
            elif str(selector_mode) == "delta_hgb_classifier":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_delta_hgb_classifier(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        baseline_strategy_name=str(baseline_strategy_name),
                        hgb_learning_rate=float(hgb_learning_rate),
                        hgb_max_depth=int(hgb_max_depth),
                        hgb_max_iter=int(hgb_max_iter),
                        hgb_l2_regularization=float(hgb_l2_regularization),
                        hgb_min_samples_leaf=int(hgb_min_samples_leaf),
                    )
            elif str(selector_mode) == "delta_random_forest":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_delta_random_forest(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        baseline_strategy_name=str(baseline_strategy_name),
                        tree_n_estimators=int(tree_n_estimators),
                        tree_max_depth=int(tree_max_depth),
                        tree_min_samples_leaf=int(tree_min_samples_leaf),
                    )
            elif str(selector_mode) == "delta_extra_trees":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_delta_extra_trees(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        baseline_strategy_name=str(baseline_strategy_name),
                        tree_n_estimators=int(tree_n_estimators),
                        tree_max_depth=int(tree_max_depth),
                        tree_min_samples_leaf=int(tree_min_samples_leaf),
                    )
            elif str(selector_mode) == "delta_gradient_boosting":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_delta_gradient_boosting(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        baseline_strategy_name=str(baseline_strategy_name),
                        tree_n_estimators=int(tree_n_estimators),
                        tree_max_depth=int(tree_max_depth),
                        tree_min_samples_leaf=int(tree_min_samples_leaf),
                        gbr_learning_rate=float(gbr_learning_rate),
                    )
            elif str(selector_mode) == "delta_lgbm":
                if int(len(train_rows)) >= int(min_train_rows):
                    predictions = _predict_delta_lgbm(
                        train_rows=train_rows,
                        current_row=row,
                        strategies=strategies,
                        feature_columns=feature_columns,
                        baseline_strategy_name=str(baseline_strategy_name),
                        lgbm_learning_rate=float(lgbm_learning_rate),
                        lgbm_n_estimators=int(lgbm_n_estimators),
                        lgbm_num_leaves=int(lgbm_num_leaves),
                        lgbm_max_depth=int(lgbm_max_depth),
                        lgbm_min_child_samples=int(lgbm_min_child_samples),
                    )
            else:
                raise ValueError("meta_strategy_probe_selector_mode_unreachable")

            if str(selector_mode) in (
                "delta_trailing_mean",
                "delta_logistic",
                "delta_ridge",
                "delta_elastic_net",
                "delta_hgb",
                "delta_hgb_classifier",
                "delta_random_forest",
                "delta_extra_trees",
                "delta_gradient_boosting",
                "delta_lgbm",
            ):
                pick, pick_score_bnb, pick_reason = _pick_against_baseline_delta(
                    predictions=predictions,
                    baseline_strategy_name=str(baseline_strategy_name),
                    safety_margin_bnb=float(safety_margin_bnb),
                    incumbent_pick=previous_pick,
                )
            else:
                pick, pick_score_bnb, pick_reason = _pick_with_skip_safety(
                    predictions=predictions,
                    safety_margin_bnb=float(safety_margin_bnb),
                    baseline_mode=str(baseline_mode),
                )
            hold_remaining = int(max(0, int(min_hold_blocks) - 1))

        realized_profit_bnb = 0.0 if str(pick) == "SKIP" else float(row.labels[str(pick)])
        oracle_total_bnb += float(oracle_profit_bnb)
        total_rounds += int(row.target_num_rounds)

        switched = bool(previous_pick is not None and str(previous_pick) != str(pick))
        if bool(switched):
            num_switches += 1
        previous_pick = str(pick)

        if str(pick) == "SKIP":
            num_skip_blocks += 1
            picks_by_strategy["SKIP"] += 1
        else:
            num_active_blocks += 1
            picks_by_strategy[str(pick)] += 1
            if float(realized_profit_bnb) > 0.0:
                num_positive_active_blocks += 1

        pick_reasons[str(pick_reason)] = int(pick_reasons.get(str(pick_reason), 0) + 1)
        cumulative_net_bnb += float(realized_profit_bnb)
        if float(cumulative_net_bnb) > float(peak_net_bnb):
            peak_net_bnb = float(cumulative_net_bnb)
        drawdown_bnb = float(peak_net_bnb) - float(cumulative_net_bnb)
        if float(drawdown_bnb) > float(max_drawdown_bnb):
            max_drawdown_bnb = float(drawdown_bnb)
        bankroll_bnb = float(starting_bankroll_bnb) + float(cumulative_net_bnb)
        if float(bankroll_bnb) < float(min_bankroll_bnb):
            min_bankroll_bnb = float(bankroll_bnb)

        decision_rows.append(
            {
                "target_block_index": int(row.target_block_index),
                "target_sim_offset_rounds": int(row.target_sim_offset_rounds),
                "target_epoch_start": int(row.target_epoch_start),
                "target_epoch_end": int(row.target_epoch_end),
                "pick": str(pick),
                "pick_score_bnb": float(pick_score_bnb),
                "pick_reason": str(pick_reason),
                "realized_profit_bnb": float(realized_profit_bnb),
                "oracle_strategy_or_skip": str(oracle_strategy_or_skip),
                "oracle_profit_bnb": float(oracle_profit_bnb),
                "cum_net_bnb": float(cumulative_net_bnb),
                "regret_to_oracle_bnb": float(oracle_profit_bnb - realized_profit_bnb),
                "switched": int(switched),
            }
        )

    summary = {
        "num_blocks": int(len(rows)),
        "num_rounds": int(total_rounds),
        "num_active_blocks": int(num_active_blocks),
        "num_skip_blocks": int(num_skip_blocks),
        "num_positive_active_blocks": int(num_positive_active_blocks),
        "active_block_rate": float(_safe_rate(num_active_blocks, len(rows))),
        "positive_block_rate_on_active": float(_safe_rate(num_positive_active_blocks, num_active_blocks)),
        "net_profit_bnb": float(cumulative_net_bnb),
        "net_profit_per_500_rounds": (
            float(cumulative_net_bnb) / float(total_rounds) * 500.0
            if int(total_rounds) > 0
            else 0.0
        ),
        "oracle_profit_bnb": float(oracle_total_bnb),
        "oracle_profit_per_500_rounds": (
            float(oracle_total_bnb) / float(total_rounds) * 500.0
            if int(total_rounds) > 0
            else 0.0
        ),
        "capture_ratio_vs_oracle": (
            float(cumulative_net_bnb) / float(oracle_total_bnb)
            if float(oracle_total_bnb) > 0.0
            else 0.0
        ),
        "max_drawdown_bnb": float(max_drawdown_bnb),
        "starting_bankroll_bnb": float(starting_bankroll_bnb),
        "min_bankroll_bnb": float(min_bankroll_bnb),
        "loss_from_start_to_min_bnb": float(float(starting_bankroll_bnb) - float(min_bankroll_bnb)),
        "num_switches": int(num_switches),
        "pick_reasons": {str(k): int(v) for k, v in pick_reasons.items()},
        "picks_by_strategy": {str(k): int(v) for k, v in picks_by_strategy.items()},
    }
    return summary, decision_rows


def main() -> None:
    args = _build_parser().parse_args()
    all_strategies, key_map, feature_columns = _load_meta(Path(str(args.dataset_meta)))
    parse_strategy_prefixes(",".join(all_strategies))
    strategies = _resolve_active_strategies(
        all_strategies=all_strategies,
        raw_active_names=str(args.active_strategy_names),
    )
    active_feature_columns = _select_feature_columns(
        feature_columns=feature_columns,
        strategies=strategies,
        key_map=key_map,
    )

    rows = _load_rows(
        dataset_csv=Path(str(args.dataset_csv)),
        strategies=all_strategies,
        key_map=key_map,
        feature_columns=feature_columns,
    )
    summary, decision_rows = _run_probe(
        rows=rows,
        strategies=strategies,
        feature_columns=active_feature_columns,
        selector_mode=str(args.selector_mode),
        trailing_history_blocks=int(args.trailing_history_blocks),
        knn_k=int(args.knn_k),
        min_train_rows=int(args.min_train_rows),
        logistic_c=float(args.logistic_c),
        ridge_alpha=float(args.ridge_alpha),
        elastic_net_alpha=float(args.elastic_net_alpha),
        elastic_net_l1_ratio=float(args.elastic_net_l1_ratio),
        hgb_learning_rate=float(args.hgb_learning_rate),
        hgb_max_depth=int(args.hgb_max_depth),
        hgb_max_iter=int(args.hgb_max_iter),
        hgb_l2_regularization=float(args.hgb_l2_regularization),
        hgb_min_samples_leaf=int(args.hgb_min_samples_leaf),
        tree_n_estimators=int(args.tree_n_estimators),
        tree_max_depth=int(args.tree_max_depth),
        tree_min_samples_leaf=int(args.tree_min_samples_leaf),
        gbr_learning_rate=float(args.gbr_learning_rate),
        lgbm_learning_rate=float(args.lgbm_learning_rate),
        lgbm_n_estimators=int(args.lgbm_n_estimators),
        lgbm_num_leaves=int(args.lgbm_num_leaves),
        lgbm_max_depth=int(args.lgbm_max_depth),
        lgbm_min_child_samples=int(args.lgbm_min_child_samples),
        safety_margin_bnb=float(args.safety_margin_bnb),
        min_hold_blocks=int(args.min_hold_blocks),
        baseline_mode=str(args.baseline_mode),
        starting_bankroll_bnb=float(args.starting_bankroll_bnb),
        fixed_strategy_name=str(args.fixed_strategy_name),
        baseline_strategy_name=str(args.baseline_strategy_name),
    )

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.name_prefix}_meta_strategy_probe_summary.json"
    decisions_path = output_dir / f"{args.name_prefix}_meta_strategy_probe_decisions.csv"

    summary_payload = {
        "probe": {
            "name_prefix": str(args.name_prefix),
            "dataset_csv": str(args.dataset_csv),
            "dataset_meta": str(args.dataset_meta),
            "selector_mode": str(args.selector_mode),
            "active_strategy_names": [str(x) for x in strategies],
            "fixed_strategy_name": str(args.fixed_strategy_name),
            "baseline_strategy_name": str(args.baseline_strategy_name),
            "trailing_history_blocks": int(args.trailing_history_blocks),
            "knn_k": int(args.knn_k),
            "min_train_rows": int(args.min_train_rows),
            "logistic_c": float(args.logistic_c),
            "ridge_alpha": float(args.ridge_alpha),
            "elastic_net_alpha": float(args.elastic_net_alpha),
            "elastic_net_l1_ratio": float(args.elastic_net_l1_ratio),
            "hgb_learning_rate": float(args.hgb_learning_rate),
            "hgb_max_depth": int(args.hgb_max_depth),
            "hgb_max_iter": int(args.hgb_max_iter),
            "hgb_l2_regularization": float(args.hgb_l2_regularization),
            "hgb_min_samples_leaf": int(args.hgb_min_samples_leaf),
            "tree_n_estimators": int(args.tree_n_estimators),
            "tree_max_depth": int(args.tree_max_depth),
            "tree_min_samples_leaf": int(args.tree_min_samples_leaf),
            "gbr_learning_rate": float(args.gbr_learning_rate),
            "lgbm_learning_rate": float(args.lgbm_learning_rate),
            "lgbm_n_estimators": int(args.lgbm_n_estimators),
            "lgbm_num_leaves": int(args.lgbm_num_leaves),
            "lgbm_max_depth": int(args.lgbm_max_depth),
            "lgbm_min_child_samples": int(args.lgbm_min_child_samples),
            "safety_margin_bnb": float(args.safety_margin_bnb),
            "min_hold_blocks": int(args.min_hold_blocks),
            "baseline_mode": str(args.baseline_mode),
            "starting_bankroll_bnb": float(args.starting_bankroll_bnb),
        },
        "strategies": [str(x) for x in strategies],
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    if bool(args.write_decisions):
        with decisions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "target_block_index",
                    "target_sim_offset_rounds",
                    "target_epoch_start",
                    "target_epoch_end",
                    "pick",
                    "pick_score_bnb",
                    "pick_reason",
                    "realized_profit_bnb",
                    "oracle_strategy_or_skip",
                    "oracle_profit_bnb",
                    "cum_net_bnb",
                    "regret_to_oracle_bnb",
                    "switched",
                ],
            )
            writer.writeheader()
            writer.writerows(decision_rows)

    print(f"SUMMARY={summary_path}")
    if bool(args.write_decisions):
        print(f"DECISIONS={decisions_path}")
    print(f"MODE={args.selector_mode}")
    print(f"NET={summary['net_profit_bnb']}")
    print(f"NET_PER_500={summary['net_profit_per_500_rounds']}")
    print(f"CAPTURE={summary['capture_ratio_vs_oracle']}")


if __name__ == "__main__":
    main()
