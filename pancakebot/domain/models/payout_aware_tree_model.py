from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Literal, Sequence

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping

from pancakebot.core.errors import InvariantError

PayoutAwareTreeModelType = Literal["lightgbm", "catboost"]


@dataclass(frozen=True, slots=True)
class PayoutAwareTreeConfig:
    model_type: PayoutAwareTreeModelType
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
class PayoutAwareTreeBundle:
    model_type: PayoutAwareTreeModelType
    feature_columns: tuple[str, ...]
    config: PayoutAwareTreeConfig
    model_payload: object
    metadata: dict[str, object]


def default_payout_aware_tree_config(
    *,
    model_type: PayoutAwareTreeModelType,
) -> PayoutAwareTreeConfig:
    if str(model_type) not in ("lightgbm", "catboost"):
        raise InvariantError("payout_aware_tree_model_type_unknown")
    return PayoutAwareTreeConfig(model_type=str(model_type))


def train_payout_aware_tree_regressor(
    *,
    feature_columns: Sequence[str],
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    random_seed: int,
    config: PayoutAwareTreeConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> PayoutAwareTreeBundle:
    x_train = np.asarray(train_x, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.float32)
    x_valid = np.asarray(valid_x, dtype=np.float32)
    y_valid = np.asarray(valid_y, dtype=np.float32)
    _validate_train_arrays(
        train_x=x_train,
        train_y=y_train,
        valid_x=x_valid,
        valid_y=y_valid,
    )
    model_cfg = config or default_payout_aware_tree_config(model_type="lightgbm")
    model = _fit_model(
        config=model_cfg,
        train_x=x_train,
        train_y=y_train,
        valid_x=x_valid,
        valid_y=y_valid,
        random_seed=int(random_seed),
    )
    valid_pred = predict_payout_aware_tree_values(
        bundle=PayoutAwareTreeBundle(
            model_type=str(model_cfg.model_type),
            feature_columns=tuple(str(col) for col in feature_columns),
            config=model_cfg,
            model_payload=model,
            metadata={},
        ),
        feature_matrix=x_valid,
    )
    residual = np.asarray(valid_pred, dtype=np.float32) - y_valid
    combined_metadata = dict(metadata or {})
    combined_metadata.update(
        {
            "input_dim": int(x_train.shape[1]),
            "train_examples": int(len(x_train)),
            "valid_examples": int(len(x_valid)),
            "random_seed": int(random_seed),
            "best_iteration": _best_iteration(model=model, model_type=str(model_cfg.model_type)),
            "valid_mae": float(np.mean(np.abs(residual))),
            "valid_rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "valid_target_mean": float(np.mean(y_valid)),
        }
    )
    return PayoutAwareTreeBundle(
        model_type=str(model_cfg.model_type),
        feature_columns=tuple(str(col) for col in feature_columns),
        config=model_cfg,
        model_payload=model,
        metadata=combined_metadata,
    )


def predict_payout_aware_tree_values(
    *,
    bundle: PayoutAwareTreeBundle,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    x = np.asarray(feature_matrix, dtype=np.float32)
    if x.ndim != 2:
        raise InvariantError("payout_aware_tree_predict_x_rank_invalid")
    model = bundle.model_payload
    pred = model.predict(x)
    return np.asarray(pred, dtype=np.float32)


def save_payout_aware_tree_bundle(*, bundle: PayoutAwareTreeBundle, path: str) -> None:
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


def load_payout_aware_tree_bundle(path: str) -> PayoutAwareTreeBundle:
    with Path(str(path)).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise InvariantError("payout_aware_tree_bundle_payload_invalid")
    config_raw = payload.get("config")
    if not isinstance(config_raw, dict):
        raise InvariantError("payout_aware_tree_bundle_config_missing")
    return PayoutAwareTreeBundle(
        model_type=str(payload["model_type"]),
        feature_columns=tuple(str(col) for col in payload["feature_columns"]),
        config=PayoutAwareTreeConfig(
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
    config: PayoutAwareTreeConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    random_seed: int,
):
    model_type = str(config.model_type)
    if model_type == "lightgbm":
        model = LGBMRegressor(
            n_estimators=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth),
            num_leaves=int(config.num_leaves),
            min_child_samples=int(config.min_child_samples),
            subsample=float(config.subsample),
            colsample_bytree=float(config.colsample_bytree),
            reg_lambda=float(config.reg_lambda),
            objective="regression",
            random_state=int(random_seed),
            verbosity=-1,
        )
        model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            eval_metric="l2",
            callbacks=[early_stopping(stopping_rounds=int(config.early_stopping_rounds), verbose=False)],
        )
        return model
    if model_type == "catboost":
        model = CatBoostRegressor(
            iterations=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            depth=int(config.max_depth),
            l2_leaf_reg=float(config.reg_lambda),
            loss_function="RMSE",
            eval_metric="RMSE",
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
    raise InvariantError("payout_aware_tree_model_type_unknown")


def _best_iteration(*, model, model_type: str) -> int:
    if str(model_type) == "lightgbm":
        best = getattr(model, "best_iteration_", None)
        return 0 if best is None else int(best)
    if str(model_type) == "catboost":
        best = model.get_best_iteration()
        return 0 if best is None else int(best)
    raise InvariantError("payout_aware_tree_model_type_unknown")


def _validate_train_arrays(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
) -> None:
    if train_x.ndim != 2 or valid_x.ndim != 2:
        raise InvariantError("payout_aware_tree_feature_rank_invalid")
    if train_y.ndim != 1 or valid_y.ndim != 1:
        raise InvariantError("payout_aware_tree_target_rank_invalid")
    if int(train_x.shape[0]) != int(len(train_y)):
        raise InvariantError("payout_aware_tree_train_len_mismatch")
    if int(valid_x.shape[0]) != int(len(valid_y)):
        raise InvariantError("payout_aware_tree_valid_len_mismatch")
    if int(train_x.shape[1]) != int(valid_x.shape[1]):
        raise InvariantError("payout_aware_tree_feature_dim_mismatch")
    if int(len(train_y)) <= 0:
        raise InvariantError("payout_aware_tree_train_empty")
    if int(len(valid_y)) <= 0:
        raise InvariantError("payout_aware_tree_valid_empty")
