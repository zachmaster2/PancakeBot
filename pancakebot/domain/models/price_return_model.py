"""Direction model for predicting Bull probability (pre-calibration score)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from pancakebot.core.errors import InvariantError


_DIRECTION_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss"],
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "min_gain_to_split": 0.0,
    "max_bin": 255,
    "verbosity": -1,
    "num_threads": -1,
    "deterministic": True,
}
_TRAINING = {
    "num_boost_round": 2000,
    "early_stopping_rounds": 100,
}


class PriceReturnModel:
    """Direction model producing an uncalibrated Bull score."""

    def __init__(self, *, alpha: float, seed: int):
        if float(alpha) <= 0.0:
            raise InvariantError("price_model_alpha_nonpositive")
        if int(seed) < 0:
            raise InvariantError("price_model_seed_negative")

        params = dict(_DIRECTION_PARAMS)
        params["lambda_l2"] = float(alpha)
        params["seed"] = int(seed)
        params["feature_fraction_seed"] = int(seed)
        params["bagging_seed"] = int(seed)
        params["data_random_seed"] = int(seed)
        self._m = LGBMClassifier(
            n_estimators=int(_TRAINING["num_boost_round"]),
            **params,
        )
        self._feature_names: list[str] | None = None

    @staticmethod
    def _to_2d_array(x) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise InvariantError("direction_x_not_2d")
        return arr

    @staticmethod
    def _make_feature_names(n_features: int) -> list[str]:
        if int(n_features) <= 0:
            raise InvariantError("direction_feature_count_nonpositive")
        return [f"f{idx}" for idx in range(int(n_features))]

    def fit(self, x, y_up, *, x_eval=None, y_eval=None, sample_weight=None) -> None:
        x_arr = self._to_2d_array(x)
        if x_arr.ndim != 2 or x_arr.shape[0] <= 1 or x_arr.shape[1] <= 0:
            raise InvariantError("direction_fit_x_shape_invalid")
        feature_names = self._make_feature_names(int(x_arr.shape[1]))
        x_df = pd.DataFrame(x_arr, columns=feature_names)

        y_arr = np.asarray(list(y_up), dtype=int)
        if y_arr.ndim != 1:
            raise InvariantError("direction_fit_y_not_1d")
        if len(y_arr) < 2 or len(y_arr) != int(x_arr.shape[0]):
            raise InvariantError("direction_fit_requires_at_least_2_rows")
        if not np.all((y_arr == 0) | (y_arr == 1)):
            raise InvariantError("direction_fit_y_not_binary")
        pos = int(np.sum(y_arr))
        if pos == 0 or pos == int(len(y_arr)):
            raise InvariantError("direction_fit_requires_both_classes")

        sample_weight_arr = None
        if sample_weight is not None:
            sample_weight_arr = np.asarray(list(sample_weight), dtype=float)
            if sample_weight_arr.ndim != 1:
                raise InvariantError("direction_fit_sample_weight_not_1d")
            if len(sample_weight_arr) != int(len(y_arr)):
                raise InvariantError("direction_fit_sample_weight_len_mismatch")
            if not np.all(np.isfinite(sample_weight_arr)):
                raise InvariantError("direction_fit_sample_weight_non_finite")
            if np.any(sample_weight_arr <= 0.0):
                raise InvariantError("direction_fit_sample_weight_nonpositive")

        callbacks = []
        fit_kwargs = {}
        if x_eval is not None and y_eval is not None:
            x_eval_arr = self._to_2d_array(x_eval)
            if x_eval_arr.ndim != 2 or x_eval_arr.shape[0] <= 0 or x_eval_arr.shape[1] != x_arr.shape[1]:
                raise InvariantError("direction_eval_x_shape_invalid")
            x_eval_df = pd.DataFrame(x_eval_arr, columns=feature_names)
            y_eval_arr = np.asarray(list(y_eval), dtype=int)
            if y_eval_arr.ndim != 1:
                raise InvariantError("direction_eval_y_not_1d")
            if len(y_eval_arr) != int(x_eval_arr.shape[0]):
                raise InvariantError("direction_eval_len_mismatch")
            if len(y_eval_arr) > 1 and np.all((y_eval_arr == 0) | (y_eval_arr == 1)):
                fit_kwargs["eval_set"] = [(x_eval_df, y_eval_arr)]
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

        self._m.fit(x_df, y_arr, **fit_kwargs)
        self._feature_names = feature_names

    def predict(self, x):
        if self._feature_names is None:
            raise InvariantError("direction_predict_without_fit")

        x_arr = self._to_2d_array(x)
        if x_arr.ndim != 2 or x_arr.shape[0] <= 0:
            raise InvariantError("direction_predict_x_shape_invalid")
        if int(x_arr.shape[1]) != int(len(self._feature_names)):
            raise InvariantError("direction_predict_feature_count_mismatch")

        x_df = pd.DataFrame(x_arr, columns=self._feature_names)
        proba = self._m.predict_proba(x_df)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise InvariantError("direction_predict_proba_shape_invalid")
        out = arr[:, 1]
        if not np.all(np.isfinite(out)):
            raise InvariantError("direction_predict_non_finite")
        return out
