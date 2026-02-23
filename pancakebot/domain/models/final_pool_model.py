"""Pool Forecast Model (cutoff->lock late inflow).

Predicts pool forecast primitives for a target_round:
- late_inflow_total_bnb (>= 0)
- late_inflow_bull_frac (in [0, 1])

Modeling details:
- late_inflow_total_bnb is modeled in log1p space and inverted with expm1.
- late_inflow_bull_frac is modeled in logit space and inverted with sigmoid,
  guaranteeing an in-range fraction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from pancakebot.core.errors import InvariantError


_LGBM_LATE_TOTAL = {
    "objective": "regression",
    "metric": ["l2"],
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "max_bin": 255,
    "verbosity": -1,
    "num_threads": -1,
    "deterministic": True,
}
_LGBM_LATE_FRAC = {
    "objective": "regression",
    "metric": ["l2"],
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 300,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "max_bin": 255,
    "verbosity": -1,
    "num_threads": -1,
    "deterministic": True,
}
_TRAINING = {
    "num_boost_round": 2000,
    "early_stopping_rounds": 100,
}
_LATE_TOTAL_WINSOR_Q = 0.995


class FinalPoolModel:
    """Predicts (late_inflow_total_bnb, late_inflow_bull_frac)."""

    _FRAC_EPS = 1e-6

    def __init__(self, *, alpha_total: float, alpha_ratio: float, seed: int):
        if float(alpha_total) <= 0.0:
            raise InvariantError("pool_model_alpha_total_nonpositive")
        if float(alpha_ratio) <= 0.0:
            raise InvariantError("pool_model_alpha_ratio_nonpositive")
        if int(seed) < 0:
            raise InvariantError("pool_model_seed_negative")

        total_params = dict(_LGBM_LATE_TOTAL)
        total_params["lambda_l2"] = float(alpha_total)
        total_params["seed"] = int(seed)
        total_params["feature_fraction_seed"] = int(seed)
        total_params["bagging_seed"] = int(seed)
        total_params["data_random_seed"] = int(seed)

        frac_params = dict(_LGBM_LATE_FRAC)
        frac_params["lambda_l2"] = float(alpha_ratio)
        frac_params["seed"] = int(seed)
        frac_params["feature_fraction_seed"] = int(seed)
        frac_params["bagging_seed"] = int(seed)
        frac_params["data_random_seed"] = int(seed)

        self._m_total_log1p = LGBMRegressor(
            n_estimators=int(_TRAINING["num_boost_round"]),
            **total_params,
        )
        self._m_frac_logit = LGBMRegressor(
            n_estimators=int(_TRAINING["num_boost_round"]),
            **frac_params,
        )
        self._feature_names: list[str] | None = None

    @staticmethod
    def _to_2d_array(x) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise InvariantError("pool_x_not_2d")
        return arr

    @staticmethod
    def _make_feature_names(n_features: int) -> list[str]:
        if int(n_features) <= 0:
            raise InvariantError("pool_feature_count_nonpositive")
        return [f"f{idx}" for idx in range(int(n_features))]

    @staticmethod
    def _logit(p: float) -> float:
        return math.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid_vec(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        pos = x >= 0.0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg])
        out[neg] = ex / (1.0 + ex)
        return out

    def _transform_total_targets(self, values, *, winsor_cap: float | None = None) -> np.ndarray:
        out: list[float] = []
        for v in values:
            fv = float(v)
            if not math.isfinite(fv):
                raise InvariantError("late_inflow_total_non_finite_in_training")
            if fv < 0.0:
                raise InvariantError("late_inflow_total_negative_in_training")
            if winsor_cap is not None:
                cap = float(winsor_cap)
                if not math.isfinite(cap) or cap < 0.0:
                    raise InvariantError("late_inflow_total_winsor_cap_invalid")
                fv = min(float(fv), float(cap))
            out.append(float(math.log1p(fv)))

        arr = np.asarray(out, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise InvariantError("late_inflow_total_log1p_non_finite")
        return arr

    def _transform_frac_targets(self, values) -> np.ndarray:
        eps = float(self._FRAC_EPS)
        out: list[float] = []
        for r in values:
            fr = float(r)
            if not math.isfinite(fr):
                raise InvariantError("late_inflow_bull_frac_non_finite_in_training")
            if not (0.0 <= fr <= 1.0):
                raise InvariantError("late_inflow_bull_frac_out_of_range_in_training")
            if fr <= 0.0:
                fr = eps
            elif fr >= 1.0:
                fr = 1.0 - eps
            out.append(float(self._logit(fr)))

        arr = np.asarray(out, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise InvariantError("late_inflow_bull_frac_logit_non_finite")
        return arr

    def fit(
        self,
        x,
        y_late_inflow_total_bnb,
        y_late_inflow_bull_frac,
        *,
        x_eval=None,
        y_total_eval=None,
        y_frac_eval=None,
        sample_weight=None,
    ) -> None:
        x_arr = self._to_2d_array(x)
        if x_arr.ndim != 2 or x_arr.shape[0] <= 1 or x_arr.shape[1] <= 0:
            raise InvariantError("pool_fit_x_shape_invalid")
        feature_names = self._make_feature_names(int(x_arr.shape[1]))
        x_df = pd.DataFrame(x_arr, columns=feature_names)

        raw_total = np.asarray(list(y_late_inflow_total_bnb), dtype=float)
        if raw_total.ndim != 1 or len(raw_total) <= 1:
            raise InvariantError("late_inflow_total_training_empty")
        if len(raw_total) != int(x_arr.shape[0]):
            raise InvariantError("late_inflow_total_training_len_mismatch")
        if not np.all(np.isfinite(raw_total)):
            raise InvariantError("late_inflow_total_non_finite_in_training")
        if np.any(raw_total < 0.0):
            raise InvariantError("late_inflow_total_negative_in_training")

        winsor_cap = float(np.quantile(raw_total, float(_LATE_TOTAL_WINSOR_Q)))
        if not math.isfinite(winsor_cap) or winsor_cap < 0.0:
            raise InvariantError("late_inflow_total_winsor_cap_invalid")

        y_total_log = self._transform_total_targets(raw_total, winsor_cap=float(winsor_cap))
        y_frac_logit = self._transform_frac_targets(y_late_inflow_bull_frac)
        if len(y_frac_logit) != int(x_arr.shape[0]):
            raise InvariantError("late_inflow_frac_training_len_mismatch")

        sample_weight_arr = None
        if sample_weight is not None:
            sample_weight_arr = np.asarray(list(sample_weight), dtype=float)
            if sample_weight_arr.ndim != 1:
                raise InvariantError("pool_fit_sample_weight_not_1d")
            if len(sample_weight_arr) != int(x_arr.shape[0]):
                raise InvariantError("pool_fit_sample_weight_len_mismatch")
            if not np.all(np.isfinite(sample_weight_arr)):
                raise InvariantError("pool_fit_sample_weight_non_finite")
            if np.any(sample_weight_arr <= 0.0):
                raise InvariantError("pool_fit_sample_weight_nonpositive")

        fit_total_kwargs = {}
        fit_frac_kwargs = {}

        if x_eval is not None and y_total_eval is not None and y_frac_eval is not None:
            x_eval_arr = self._to_2d_array(x_eval)
            if x_eval_arr.ndim != 2 or x_eval_arr.shape[0] <= 0 or x_eval_arr.shape[1] != x_arr.shape[1]:
                raise InvariantError("pool_eval_x_shape_invalid")
            x_eval_df = pd.DataFrame(x_eval_arr, columns=feature_names)
            y_total_eval_log = self._transform_total_targets(y_total_eval, winsor_cap=float(winsor_cap))
            y_frac_eval_logit = self._transform_frac_targets(y_frac_eval)
            if len(y_total_eval_log) != int(x_eval_arr.shape[0]):
                raise InvariantError("late_inflow_total_eval_len_mismatch")
            if len(y_frac_eval_logit) != int(x_eval_arr.shape[0]):
                raise InvariantError("late_inflow_frac_eval_len_mismatch")

            if len(y_total_eval_log) > 1:
                fit_total_kwargs["eval_set"] = [(x_eval_df, y_total_eval_log)]
                fit_total_kwargs["eval_metric"] = "l2"
                fit_total_kwargs["callbacks"] = [
                    early_stopping(
                        stopping_rounds=int(_TRAINING["early_stopping_rounds"]),
                        first_metric_only=True,
                        verbose=False,
                    )
                ]

            if len(y_frac_eval_logit) > 1:
                fit_frac_kwargs["eval_set"] = [(x_eval_df, y_frac_eval_logit)]
                fit_frac_kwargs["eval_metric"] = "l2"
                fit_frac_kwargs["callbacks"] = [
                    early_stopping(
                        stopping_rounds=int(_TRAINING["early_stopping_rounds"]),
                        first_metric_only=True,
                        verbose=False,
                    )
                ]

        if sample_weight_arr is not None:
            fit_total_kwargs["sample_weight"] = sample_weight_arr
            fit_frac_kwargs["sample_weight"] = sample_weight_arr

        self._m_total_log1p.fit(x_df, y_total_log, **fit_total_kwargs)
        self._m_frac_logit.fit(x_df, y_frac_logit, **fit_frac_kwargs)
        self._feature_names = feature_names

    def predict(self, x):
        if self._feature_names is None:
            raise InvariantError("pool_predict_without_fit")

        x_arr = self._to_2d_array(x)
        if x_arr.ndim != 2 or x_arr.shape[0] <= 0:
            raise InvariantError("pool_predict_x_shape_invalid")
        if int(x_arr.shape[1]) != int(len(self._feature_names)):
            raise InvariantError("pool_predict_feature_count_mismatch")

        x_df = pd.DataFrame(x_arr, columns=self._feature_names)

        total_log = np.asarray(self._m_total_log1p.predict(x_df), dtype=float)
        frac_logit = np.asarray(self._m_frac_logit.predict(x_df), dtype=float)
        if total_log.shape != frac_logit.shape:
            raise InvariantError("pool_predict_head_shape_mismatch")
        if not np.all(np.isfinite(total_log)):
            raise InvariantError("late_inflow_total_pred_non_finite")
        if not np.all(np.isfinite(frac_logit)):
            raise InvariantError("late_inflow_bull_frac_logit_pred_non_finite")

        late_total = np.expm1(total_log)
        late_total = np.maximum(late_total, 0.0)

        frac = self._sigmoid_vec(frac_logit)
        frac = np.clip(frac, 0.0, 1.0)
        if not np.all(np.isfinite(frac)):
            raise InvariantError("late_inflow_bull_frac_pred_not_finite")

        return list(zip(late_total.tolist(), frac.tolist()))
