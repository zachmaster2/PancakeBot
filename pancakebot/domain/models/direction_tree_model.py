from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Literal, Sequence

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping

from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.neural_direction_dataset import NeuralDirectionDataset

DirectionTreeModelType = Literal["lightgbm", "catboost"]


@dataclass(frozen=True, slots=True)
class DirectionTreeConfig:
    model_type: DirectionTreeModelType
    n_estimators: int = 600
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 50
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50


@dataclass(frozen=True, slots=True)
class DirectionTreeBundle:
    model_type: DirectionTreeModelType
    feature_columns: tuple[str, ...]
    config: DirectionTreeConfig
    model_payload: object
    metadata: dict[str, object]


def default_direction_tree_config(*, model_type: DirectionTreeModelType) -> DirectionTreeConfig:
    if str(model_type) not in ("lightgbm", "catboost"):
        raise InvariantError("direction_tree_model_type_unknown")
    return DirectionTreeConfig(model_type=str(model_type))


def train_direction_tree_classifier(
    *,
    dataset: NeuralDirectionDataset,
    train_target_epochs: Sequence[int],
    valid_target_epochs: Sequence[int],
    random_seed: int,
    config: DirectionTreeConfig | None = None,
) -> DirectionTreeBundle:
    _validate_dataset(dataset=dataset)
    train_x, train_y = _rows_for_target_epochs(dataset=dataset, target_epochs=train_target_epochs)
    valid_x, valid_y = _rows_for_target_epochs(dataset=dataset, target_epochs=valid_target_epochs)
    if len(train_x) <= 0:
        raise InvariantError("direction_tree_train_empty")
    if len(valid_x) <= 0:
        raise InvariantError("direction_tree_valid_empty")
    model_cfg = config or default_direction_tree_config(model_type="lightgbm")
    model = _fit_model(
        config=model_cfg,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        random_seed=int(random_seed),
    )
    valid_probs = predict_direction_tree_probabilities(
        bundle=DirectionTreeBundle(
            model_type=str(model_cfg.model_type),
            feature_columns=tuple(str(col) for col in dataset.feature_columns),
            config=model_cfg,
            model_payload=model,
            metadata={},
        ),
        feature_matrix=valid_x,
    )
    valid_pred = (np.asarray(valid_probs, dtype=np.float32) >= 0.5).astype(np.int64)
    return DirectionTreeBundle(
        model_type=str(model_cfg.model_type),
        feature_columns=tuple(str(col) for col in dataset.feature_columns),
        config=model_cfg,
        model_payload=model,
        metadata={
            "input_dim": int(train_x.shape[1]),
            "train_examples": int(len(train_x)),
            "valid_examples": int(len(valid_x)),
            "train_epoch_start": int(train_target_epochs[0]),
            "train_epoch_end": int(train_target_epochs[-1]),
            "valid_epoch_start": int(valid_target_epochs[0]),
            "valid_epoch_end": int(valid_target_epochs[-1]),
            "random_seed": int(random_seed),
            "best_valid_win_rate": float(np.mean(valid_pred == valid_y)),
            "best_iteration": _best_iteration(model=model, model_type=str(model_cfg.model_type)),
        },
    )


def predict_direction_tree_probabilities(
    *,
    bundle: DirectionTreeBundle,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    x = np.asarray(feature_matrix, dtype=np.float32)
    if x.ndim != 2:
        raise InvariantError("direction_tree_predict_x_rank_invalid")
    model = bundle.model_payload
    if str(bundle.model_type) == "lightgbm":
        probs = model.predict_proba(x)[:, 1]
    elif str(bundle.model_type) == "catboost":
        probs = model.predict_proba(x)[:, 1]
    else:
        raise InvariantError("direction_tree_model_type_unknown")
    return np.asarray(probs, dtype=np.float32)


def save_direction_tree_bundle(*, bundle: DirectionTreeBundle, path: str) -> None:
    out_path = Path(str(path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": str(bundle.model_type),
        "feature_columns": tuple(str(col) for col in bundle.feature_columns),
        "config": asdict(bundle.config),
        "model_payload": bundle.model_payload,
        "metadata": dict(bundle.metadata),
    }
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_direction_tree_bundle(path: str) -> DirectionTreeBundle:
    with Path(str(path)).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise InvariantError("direction_tree_bundle_payload_invalid")
    config_raw = payload.get("config")
    if not isinstance(config_raw, dict):
        raise InvariantError("direction_tree_bundle_config_missing")
    model_type = str(payload.get("model_type"))
    return DirectionTreeBundle(
        model_type=str(model_type),
        feature_columns=tuple(str(col) for col in payload["feature_columns"]),
        config=DirectionTreeConfig(
            model_type=str(config_raw["model_type"]),
            n_estimators=int(config_raw["n_estimators"]),
            learning_rate=float(config_raw["learning_rate"]),
            max_depth=int(config_raw["max_depth"]),
            num_leaves=int(config_raw["num_leaves"]),
            min_child_samples=int(config_raw["min_child_samples"]),
            subsample=float(config_raw["subsample"]),
            colsample_bytree=float(config_raw["colsample_bytree"]),
            reg_lambda=float(config_raw["reg_lambda"]),
            early_stopping_rounds=int(config_raw["early_stopping_rounds"]),
        ),
        model_payload=payload["model_payload"],
        metadata=dict(payload.get("metadata", {})),
    )


def _fit_model(
    *,
    config: DirectionTreeConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    random_seed: int,
):
    model_type = str(config.model_type)
    if model_type == "lightgbm":
        model = LGBMClassifier(
            n_estimators=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth),
            num_leaves=int(config.num_leaves),
            min_child_samples=int(config.min_child_samples),
            subsample=float(config.subsample),
            colsample_bytree=float(config.colsample_bytree),
            reg_lambda=float(config.reg_lambda),
            objective="binary",
            random_state=int(random_seed),
            verbosity=-1,
        )
        model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(stopping_rounds=int(config.early_stopping_rounds), verbose=False)],
        )
        return model
    if model_type == "catboost":
        model = CatBoostClassifier(
            iterations=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            depth=int(config.max_depth),
            l2_leaf_reg=float(config.reg_lambda),
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=int(random_seed),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_x,
            train_y,
            eval_set=(valid_x, valid_y),
            use_best_model=True,
            early_stopping_rounds=int(config.early_stopping_rounds),
            verbose=False,
        )
        return model
    raise InvariantError("direction_tree_model_type_unknown")


def _best_iteration(*, model, model_type: str) -> int:
    if str(model_type) == "lightgbm":
        best = getattr(model, "best_iteration_", None)
        return 0 if best is None else int(best)
    if str(model_type) == "catboost":
        best = model.get_best_iteration()
        return 0 if best is None else int(best)
    raise InvariantError("direction_tree_model_type_unknown")


def _validate_dataset(*, dataset: NeuralDirectionDataset) -> None:
    if int(dataset.feature_matrix.ndim) != 2:
        raise InvariantError("direction_tree_feature_rank_invalid")
    if int(dataset.feature_matrix.shape[0]) != int(dataset.num_examples):
        raise InvariantError("direction_tree_feature_rows_len_mismatch")
    if len(dataset.labels) != int(dataset.num_examples):
        raise InvariantError("direction_tree_labels_len_mismatch")


def _rows_for_target_epochs(
    *,
    dataset: NeuralDirectionDataset,
    target_epochs: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not target_epochs:
        raise InvariantError("direction_tree_target_epochs_empty")
    index_by_epoch = {int(epoch): idx for idx, epoch in enumerate(dataset.target_epochs)}
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    seen: set[int] = set()
    for epoch in target_epochs:
        key = int(epoch)
        if int(key) in seen:
            raise InvariantError("direction_tree_target_epochs_duplicate")
        seen.add(int(key))
        idx = index_by_epoch.get(int(key))
        if idx is None:
            raise InvariantError("direction_tree_target_epoch_missing")
        x_rows.append(np.asarray(dataset.feature_matrix[int(idx)], dtype=np.float32))
        y_rows.append(int(dataset.labels[int(idx)]))
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int64),
    )
