"""Predictability gate model.

Outputs p_tradeable in [0, 1] for gating participation before direction/sizing.
"""

from __future__ import annotations

import warnings

import numpy as np
from lightgbm import LGBMClassifier, early_stopping

from pancakebot.core.errors import InvariantError


warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBM(Classifier|Regressor) was fitted with feature names",
    category=UserWarning,
)


_GATE_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss"],
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 150,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "max_bin": 255,
    "verbosity": -1,
    "num_threads": -1,
    "deterministic": True,
}

_TRAINING = {
    "num_boost_round": 1000,
    "early_stopping_rounds": 100,
}


class PredictabilityModel:
    """Binary classifier for tradeable vs non-tradeable rounds."""

    def __init__(self, *, seed: int, feature_indices: tuple[int, ...] | None = None):
        if int(seed) < 0:
            raise InvariantError("predictability_model_seed_negative")
        params = dict(_GATE_PARAMS)
        params["seed"] = int(seed)
        params["feature_fraction_seed"] = int(seed)
        params["bagging_seed"] = int(seed)
        params["data_random_seed"] = int(seed)
        self._m = LGBMClassifier(
            n_estimators=int(_TRAINING["num_boost_round"]),
            **params,
        )
        if feature_indices is None:
            self._feature_indices = None
        else:
            idxs = tuple(int(idx) for idx in feature_indices)
            if not idxs:
                raise InvariantError("predictability_feature_indices_empty")
            if any(int(idx) < 0 for idx in idxs):
                raise InvariantError("predictability_feature_indices_negative")
            if len(set(idxs)) != len(idxs):
                raise InvariantError("predictability_feature_indices_duplicate")
            self._feature_indices = idxs
        self._input_feature_count: int | None = None
        self._n_features: int | None = None
        self._constant_proba: float | None = None

    @staticmethod
    def _to_2d_array(x) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise InvariantError("predictability_x_not_2d")
        return arr

    def _select_features(self, arr: np.ndarray) -> np.ndarray:
        if self._feature_indices is None:
            return arr
        if arr.ndim != 2:
            raise InvariantError("predictability_x_not_2d")
        if arr.shape[1] <= int(max(self._feature_indices)):
            raise InvariantError("predictability_feature_indices_out_of_range")
        return arr[:, list(self._feature_indices)]

    def fit(self, x, y_tradeable, *, x_eval=None, y_eval=None, sample_weight=None) -> None:
        x_arr = self._to_2d_array(x)
        if x_arr.ndim != 2 or x_arr.shape[0] <= 1 or x_arr.shape[1] <= 0:
            raise InvariantError("predictability_fit_x_shape_invalid")
        x_fit_arr = self._select_features(x_arr)
        if x_fit_arr.ndim != 2 or x_fit_arr.shape[1] <= 0:
            raise InvariantError("predictability_fit_selected_x_shape_invalid")

        y_arr = np.asarray(list(y_tradeable), dtype=int)
        if y_arr.ndim != 1:
            raise InvariantError("predictability_fit_y_not_1d")
        if len(y_arr) < 2 or len(y_arr) != int(x_arr.shape[0]):
            raise InvariantError("predictability_fit_requires_at_least_2_rows")
        if not np.all((y_arr == 0) | (y_arr == 1)):
            raise InvariantError("predictability_fit_y_not_binary")

        sample_weight_arr = None
        if sample_weight is not None:
            sample_weight_arr = np.asarray(list(sample_weight), dtype=float)
            if sample_weight_arr.ndim != 1:
                raise InvariantError("predictability_fit_sample_weight_not_1d")
            if len(sample_weight_arr) != int(len(y_arr)):
                raise InvariantError("predictability_fit_sample_weight_len_mismatch")
            if not np.all(np.isfinite(sample_weight_arr)):
                raise InvariantError("predictability_fit_sample_weight_non_finite")
            if np.any(sample_weight_arr <= 0.0):
                raise InvariantError("predictability_fit_sample_weight_nonpositive")

        self._input_feature_count = int(x_arr.shape[1])
        self._n_features = int(x_fit_arr.shape[1])
        pos = int(np.sum(y_arr))
        if int(pos) == 0:
            self._constant_proba = 0.0
            return
        if int(pos) == int(len(y_arr)):
            self._constant_proba = 1.0
            return
        self._constant_proba = None

        callbacks = []
        fit_kwargs = {}
        if x_eval is not None and y_eval is not None:
            x_eval_arr = self._to_2d_array(x_eval)
            if x_eval_arr.ndim != 2 or x_eval_arr.shape[0] <= 0 or x_eval_arr.shape[1] != x_arr.shape[1]:
                raise InvariantError("predictability_eval_x_shape_invalid")
            x_eval_fit_arr = self._select_features(x_eval_arr)
            if x_eval_fit_arr.ndim != 2 or x_eval_fit_arr.shape[1] != x_fit_arr.shape[1]:
                raise InvariantError("predictability_eval_selected_x_shape_invalid")
            y_eval_arr = np.asarray(list(y_eval), dtype=int)
            if y_eval_arr.ndim != 1:
                raise InvariantError("predictability_eval_y_not_1d")
            if len(y_eval_arr) != int(x_eval_arr.shape[0]):
                raise InvariantError("predictability_eval_len_mismatch")
            if len(y_eval_arr) > 1 and np.any(y_eval_arr == 0) and np.any(y_eval_arr == 1):
                fit_kwargs["eval_set"] = [(x_eval_fit_arr, y_eval_arr)]
                fit_kwargs["eval_metric"] = "binary_logloss"
                callbacks.append(
                    early_stopping(
                        stopping_rounds=int(_TRAINING["early_stopping_rounds"]),
                        first_metric_only=True,
                        verbose=False,
                    )
                )
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        if sample_weight_arr is not None:
            fit_kwargs["sample_weight"] = sample_weight_arr

        self._m.fit(x_fit_arr, y_arr, **fit_kwargs)

    def predict_proba(self, x):
        if self._n_features is None or self._input_feature_count is None:
            raise InvariantError("predictability_predict_without_fit")
        x_arr = self._to_2d_array(x)
        if x_arr.ndim != 2 or x_arr.shape[0] <= 0:
            raise InvariantError("predictability_predict_x_shape_invalid")
        if int(x_arr.shape[1]) != int(self._input_feature_count):
            raise InvariantError("predictability_predict_input_feature_count_mismatch")
        x_pred_arr = self._select_features(x_arr)
        if int(x_pred_arr.shape[1]) != int(self._n_features):
            raise InvariantError("predictability_predict_feature_count_mismatch")

        if self._constant_proba is not None:
            return np.full(int(x_arr.shape[0]), float(self._constant_proba), dtype=float)

        proba = self._m.predict_proba(x_pred_arr)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise InvariantError("predictability_predict_proba_shape_invalid")
        out = arr[:, 1]
        if not np.all(np.isfinite(out)):
            raise InvariantError("predictability_predict_non_finite")
        return out
