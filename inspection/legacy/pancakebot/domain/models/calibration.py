from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression

from pancakebot.core.errors import InvariantError


@dataclass(slots=True)
class IsotonicCalibrator:
    """Monotonic calibration: mu -> p_up via isotonic regression."""

    _model: IsotonicRegression | None = None

    def fit(self, mu: Iterable[float], y_up: Iterable[int], *, sample_weight: Iterable[float] | None = None) -> None:
        mu_arr = np.asarray(list(mu), dtype=float)
        y_arr = np.asarray(list(y_up), dtype=float)
        if mu_arr.ndim != 1 or y_arr.ndim != 1:
            raise InvariantError("isotonic_fit_requires_1d")
        if len(mu_arr) != len(y_arr):
            raise InvariantError("isotonic_fit_len_mismatch")
        if len(mu_arr) < 2:
            raise InvariantError("isotonic_fit_requires_at_least_2")
        if not np.all(np.isfinite(mu_arr)):
            raise InvariantError("isotonic_fit_mu_non_finite")

        if np.any((y_arr != 0.0) & (y_arr != 1.0)):
            raise InvariantError("isotonic_fit_y_not_binary")

        n_pos = int(np.sum(y_arr))
        n = int(len(y_arr))
        if n_pos == 0 or n_pos == n:
            raise InvariantError("isotonic_fit_requires_both_classes")

        sample_weight_arr = None
        if sample_weight is not None:
            sample_weight_arr = np.asarray(list(sample_weight), dtype=float)
            if sample_weight_arr.ndim != 1:
                raise InvariantError("isotonic_fit_sample_weight_not_1d")
            if len(sample_weight_arr) != len(y_arr):
                raise InvariantError("isotonic_fit_sample_weight_len_mismatch")
            if not np.all(np.isfinite(sample_weight_arr)):
                raise InvariantError("isotonic_fit_sample_weight_non_finite")
            if np.any(sample_weight_arr <= 0.0):
                raise InvariantError("isotonic_fit_sample_weight_nonpositive")

        m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        m.fit(mu_arr, y_arr, sample_weight=sample_weight_arr)
        self._model = m

    def predict_proba_up(self, mu: Iterable[float] | float) -> np.ndarray | float:
        if self._model is None:
            raise InvariantError("isotonic_predict_without_fit")

        if isinstance(mu, (float, int)):
            val = float(mu)
            if not np.isfinite(val):
                raise InvariantError("isotonic_predict_mu_non_finite")
            return float(self._model.predict(np.asarray([val], dtype=float))[0])

        mu_arr = np.asarray(list(mu), dtype=float)
        if mu_arr.ndim != 1:
            raise InvariantError("isotonic_predict_requires_1d")
        if not np.all(np.isfinite(mu_arr)):
            raise InvariantError("isotonic_predict_mu_non_finite")
        out = self._model.predict(mu_arr)
        if not np.all(np.isfinite(out)):
            raise InvariantError("isotonic_predict_output_non_finite")
        return out
