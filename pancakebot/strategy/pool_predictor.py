"""PoolPredictor: linear regression for final pool state (bull / bear).

Replaces the gate's identity-based pool view (partial_total / partial_winner)
with a trained predictor that accounts for post-cutoff bet arrivals. Fitted
coefficients come from research/phase2_pool_predictor.py.

Integration is gated behind config.strategy.pool_predictor flags. Default
config ``model="none"`` keeps this module a no-op (``from_config`` returns
``None``) and the integration branches in momentum_pipeline stay dormant.

Coefficients JSON schema::

    {
      "model": "P2" | "P2_minimal" | "P3_lite",
      "bull": { "intercept": float, "coefs": { "feature_name": float, ... } },
      "bear": { "intercept": float, "coefs": { ... } }
    }

Expected feature names per model are defined below (``_FEATURES_BY_MODEL``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pancakebot.util import InvariantError


_P2_FEATURES: tuple[str, ...] = (
    "partial_bull",
    "partial_bear",
    "num_bets_pre_cutoff",
    "recent_bets_20s",
)
_P2_MINIMAL_FEATURES: tuple[str, ...] = (
    "partial_bull",
    "partial_bear",
)
_P3_LITE_EXTRA_KLINE_FEATURES: tuple[str, ...] = (
    "bnb_ret_20s",
    "bnb_range_25s",
    "bnb_microvol_5s",
)
_P3_LITE_FEATURES: tuple[str, ...] = _P2_FEATURES + _P3_LITE_EXTRA_KLINE_FEATURES

_FEATURES_BY_MODEL: dict[str, tuple[str, ...]] = {
    "P2": _P2_FEATURES,
    "P2_minimal": _P2_MINIMAL_FEATURES,
    "P3_lite": _P3_LITE_FEATURES,
}


def compute_bnb_kline_features(
    bnb_klines: list[list] | None,
) -> tuple[float, float, float] | None:
    """Compute ``(bnb_ret_20s, bnb_range_25s, bnb_microvol_5s)`` from a 31-candle window.

    Matches ``research/phase2_pool_predictor.py``. Returns ``None`` when the window
    has fewer than 26 candles (the widest lookback required) or any mean close
    is non-positive.

    Each kline row: ``[ts_ms, open, high, low, close, vol]``.
    """
    if not bnb_klines or len(bnb_klines) < 26:
        return None
    closes: list[float] = [float(k[4]) for k in bnb_klines]
    if closes[-21] <= 0.0:
        return None
    ret_20s: float = closes[-1] / closes[-21] - 1.0

    window_25 = closes[-26:]
    mean_25 = sum(window_25) / len(window_25)
    if mean_25 <= 0.0:
        return None
    range_25s: float = (max(window_25) - min(window_25)) / mean_25

    window_5 = closes[-6:]
    mean_5 = sum(window_5) / len(window_5)
    if mean_5 <= 0.0:
        return None
    microvol_5s: float = (max(window_5) - min(window_5)) / mean_5

    return ret_20s, range_25s, microvol_5s


@dataclass(frozen=True, slots=True)
class _SideCoefs:
    """Linear-regression coefficients for one side (bull or bear)."""
    intercept: float
    # Ordered (feature_name, weight) tuples in the model's feature order.
    weighted_features: tuple[tuple[str, float], ...]


class PoolPredictor:
    """Linear-regression predictor for ``(final_bull, final_bear)`` at cutoff.

    Instances are created via ``from_config`` and called via ``predict``. The
    prediction applies a sanity floor: the predicted final-pool value for each
    side is clamped to be at least the already-observed partial value (the
    final pool cannot be smaller than what has already been deposited).
    """

    __slots__ = ("_model", "_features", "_bull", "_bear")

    def __init__(self, *, model: str, bull: _SideCoefs, bear: _SideCoefs) -> None:
        if model not in _FEATURES_BY_MODEL:
            raise InvariantError(f"pool_predictor_invalid_model: {model}")
        self._model: str = model
        self._features: tuple[str, ...] = _FEATURES_BY_MODEL[model]
        # Sanity-check that both sides provide coefficients for every feature.
        for side_name, side in (("bull", bull), ("bear", bear)):
            have = {f for f, _ in side.weighted_features}
            missing = [f for f in self._features if f not in have]
            if missing:
                raise InvariantError(
                    f"pool_predictor_{side_name}_missing_coefs: {missing}"
                )
        self._bull: _SideCoefs = bull
        self._bear: _SideCoefs = bear

    @property
    def model(self) -> str:
        return self._model

    @property
    def requires_klines(self) -> bool:
        """True when the configured model consumes BNB kline features."""
        return self._model == "P3_lite"

    @staticmethod
    def from_config(cfg) -> "PoolPredictor | None":
        """Build a predictor from a ``PoolPredictorConfig``; returns ``None`` when
        ``model == 'none'``. Raises ``InvariantError`` on any schema/load error."""
        model = cfg.model
        if model == "none":
            return None
        # Config validation already ensured path is non-empty and exists; guard
        # again here for callers that bypass the config loader.
        path_str = cfg.coefficients_path
        if not path_str:
            raise InvariantError("pool_predictor_coefficients_path_empty")
        path = Path(path_str)
        if not path.exists():
            raise InvariantError(f"pool_predictor_coefficients_file_missing: {path}")
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise InvariantError(
                f"pool_predictor_coefficients_parse_failed: {e}"
            ) from e
        if not isinstance(doc, dict):
            raise InvariantError("pool_predictor_coefficients_not_dict")

        disk_model = doc.get("model")
        if disk_model != model:
            raise InvariantError(
                f"pool_predictor_model_mismatch: config={model!r} file={disk_model!r}"
            )
        if "bull" not in doc or "bear" not in doc:
            raise InvariantError("pool_predictor_coefficients_missing_side")

        features = _FEATURES_BY_MODEL[model]

        def _parse_side(sd: object, side_name: str) -> _SideCoefs:
            if not isinstance(sd, dict):
                raise InvariantError(f"pool_predictor_side_{side_name}_not_dict")
            intercept_raw = sd.get("intercept", 0.0)
            try:
                intercept = float(intercept_raw)
            except (TypeError, ValueError) as e:
                raise InvariantError(
                    f"pool_predictor_{side_name}_intercept_not_number"
                ) from e
            coef_dict = sd.get("coefs", {})
            if not isinstance(coef_dict, dict):
                raise InvariantError(f"pool_predictor_{side_name}_coefs_not_dict")
            ordered: list[tuple[str, float]] = []
            for f in features:
                if f not in coef_dict:
                    raise InvariantError(
                        f"pool_predictor_{side_name}_coef_missing: {f}"
                    )
                try:
                    ordered.append((f, float(coef_dict[f])))
                except (TypeError, ValueError) as e:
                    raise InvariantError(
                        f"pool_predictor_{side_name}_coef_not_number: {f}"
                    ) from e
            return _SideCoefs(intercept=intercept, weighted_features=tuple(ordered))

        return PoolPredictor(
            model=model,
            bull=_parse_side(doc["bull"], "bull"),
            bear=_parse_side(doc["bear"], "bear"),
        )

    def predict(
        self,
        *,
        partial_bull: float,
        partial_bear: float,
        num_bets_pre_cutoff: int,
        recent_bets_20s: int,
        bnb_klines: list[list] | None = None,
    ) -> tuple[float, float]:
        """Return ``(predicted_final_bull, predicted_final_bear)`` in BNB.

        The prediction is clamped to ``>= partial_{side}`` on each side: the
        final pool cannot shrink relative to what has already been deposited.
        """
        feature_values: dict[str, float] = {
            "partial_bull": float(partial_bull),
            "partial_bear": float(partial_bear),
            "num_bets_pre_cutoff": float(num_bets_pre_cutoff),
            "recent_bets_20s": float(recent_bets_20s),
        }
        if self.requires_klines:
            if bnb_klines is None:
                raise InvariantError("pool_predictor_p3_lite_requires_bnb_klines")
            feats = compute_bnb_kline_features(bnb_klines)
            if feats is None:
                raise InvariantError("pool_predictor_bnb_klines_insufficient")
            feature_values["bnb_ret_20s"] = feats[0]
            feature_values["bnb_range_25s"] = feats[1]
            feature_values["bnb_microvol_5s"] = feats[2]

        def _apply(side: _SideCoefs) -> float:
            v = side.intercept
            for name, weight in side.weighted_features:
                v += weight * feature_values[name]
            return v

        pred_bull = _apply(self._bull)
        pred_bear = _apply(self._bear)

        # Sanity floor: predicted final pool cannot be less than the partial
        # already observed at cutoff. Linear-regression predictions may
        # under-shoot on outliers; this bounds the downside cleanly.
        pred_bull = max(pred_bull, float(partial_bull))
        pred_bear = max(pred_bear, float(partial_bear))
        return pred_bull, pred_bear
