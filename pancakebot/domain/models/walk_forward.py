"""Walk-forward owner for training, calibration, and prediction (LOCKED).

This module is the single source of truth for:
- Train/calibration windowing over the rolling closed-round cache.
- Retrain/recalibrate cadence via *_interval config knobs.
- Isotonic calibration fit on an out-of-sample calibration window.

The caller owns:
- Maintaining the rolling closed cache (closed rounds only; epoch-ascending).
- Building target-round features at cutoff (planner.build_inputs).
- Strategy / EV / sizing decisions.

Terminology (approved):
- prior_context_rounds_required: number of prior rounds required as cross-round context.
- train_size: number of closed target rounds used to train the base models.
- calibrate_size: number of closed target rounds used to fit the calibrator.
- retrain_interval / recalibrate_interval: cadence in decision steps.
- steps_since_train / steps_since_calibrate: state counters.
- p_final: calibrated probability used for EV/sizing.

Pool forecasting (frozen contract):
- The pool model predicts primitives:
    pred_late_inflow_total_bnb (>= 0)
    pred_late_inflow_bull_frac (in [0,1])
  which the planner converts into canonical final_* pool forecasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.domain.types import Round
from pancakebot.domain.features.feature_builder import build_features, vectorize
from pancakebot.domain.features.schema import FEATURE_SCHEMA, max_required_context_klines_size, max_required_prior_context_rounds_size
from pancakebot.domain.features.targets import compute_pool_forecast_targets, compute_price_targets
from pancakebot.domain.models.calibration import IsotonicCalibrator
from pancakebot.domain.models.final_pool_model import FinalPoolModel
from pancakebot.domain.models.predictability_model import PredictabilityModel
from pancakebot.domain.models.predictability_modes import (
    DEFAULT_PREDICTABILITY_FEATURE_MODE,
    DEFAULT_PREDICTABILITY_LABEL_MODE,
    PREDICTABILITY_LABEL_MODE_BASELINE_LOG_IMBALANCE_SIDE,
    PREDICTABILITY_LABEL_MODE_CONTRARIAN_LOG_IMBALANCE_SIDE,
    PREDICTABILITY_LABEL_MODE_CONTRARIAN_PRICE_REGIME_VOTE_15_30_R20_R60_SIDE,
    PREDICTABILITY_LABEL_MODE_EITHER_SIDE_PROFITABLE,
    PREDICTABILITY_LABEL_MODE_PRICE_MOMENTUM_K15_SIDE,
    PREDICTABILITY_LABEL_MODE_REGIME_MAJORITY_R20_SIDE,
    predictability_feature_indices,
    validate_predictability_feature_mode,
    validate_predictability_label_mode,
)
from pancakebot.domain.models.price_return_model import PriceReturnModel
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info
from pancakebot.runtime.settlement import settle_bet_against_closed_round

_TRADEABLE_LOG_IMB_FEATURE = "log_imb_w_p_80_to_p_100"
_TRADEABLE_LOG_IMB_IDX = int(FEATURE_SCHEMA.columns.index(str(_TRADEABLE_LOG_IMB_FEATURE)))
_TRADEABLE_PRICE_MOMENTUM_FEATURE = "price_log_return_mean_k_15"
_TRADEABLE_PRICE_MOMENTUM_IDX = int(FEATURE_SCHEMA.columns.index(str(_TRADEABLE_PRICE_MOMENTUM_FEATURE)))
_TRADEABLE_PRICE_MOMENTUM_K30_FEATURE = "price_log_return_mean_k_30"
_TRADEABLE_PRICE_MOMENTUM_K30_IDX = int(FEATURE_SCHEMA.columns.index(str(_TRADEABLE_PRICE_MOMENTUM_K30_FEATURE)))
_TRADEABLE_REGIME_BULL_FRAC_FEATURE = "regime_bull_frac_r_20"
_TRADEABLE_REGIME_BULL_FRAC_IDX = int(FEATURE_SCHEMA.columns.index(str(_TRADEABLE_REGIME_BULL_FRAC_FEATURE)))
_TRADEABLE_REGIME_BULL_FRAC_R60_FEATURE = "regime_bull_frac_r_60"
_TRADEABLE_REGIME_BULL_FRAC_R60_IDX = int(FEATURE_SCHEMA.columns.index(str(_TRADEABLE_REGIME_BULL_FRAC_R60_FEATURE)))


@dataclass(frozen=True, slots=True)
class WalkForwardModels:
    price_model: PriceReturnModel
    pool_model: FinalPoolModel
    predictability_model: PredictabilityModel


@dataclass(slots=True)
class WalkForwardState:
    models: WalkForwardModels | None = None
    calibrator_final: IsotonicCalibrator | None = None
    steps_since_train: int = 0
    steps_since_calibrate: int = 0
    last_train_epoch: int | None = None
    last_calibrate_epoch: int | None = None


def _predictability_feature_mode(*, cfg: Any) -> str:
    return validate_predictability_feature_mode(
        str(getattr(cfg, "predictability_feature_mode", DEFAULT_PREDICTABILITY_FEATURE_MODE))
    )


def _predictability_label_mode(*, cfg: Any) -> str:
    return validate_predictability_label_mode(
        str(getattr(cfg, "predictability_label_mode", DEFAULT_PREDICTABILITY_LABEL_MODE))
    )


def ensure_state(
    *,
    cfg: Any,
    closed_rounds: list[Round],
    current_epoch: int,
    state: WalkForwardState | None,
) -> WalkForwardState:
    """Update (train/recalibrate) walk-forward state based on cadence and cache contents."""
    if current_epoch <= 0:
        raise InvariantError("current_epoch_invalid")

    if state is None:
        state = WalkForwardState()

    train_size = int(cfg.train_size)
    calibrate_size = int(cfg.calibrate_size)
    retrain_interval = int(cfg.retrain_interval)
    recalibrate_interval = int(cfg.recalibrate_interval)
    recency_weight_floor, recency_weight_power = _recency_weight_params(cfg=cfg)

    # Decide whether to retrain.
    should_train = False
    train_reason = ""

    if state.models is None:
        should_train = True
        train_reason = "initial"
    elif state.steps_since_train >= retrain_interval:
        should_train = True
        train_reason = "interval"

    # Decide whether to recalibrate (required at decision time).
    should_calibrate = False
    cal_reason = ""

    if state.calibrator_final is None:
        should_calibrate = True
        cal_reason = "initial"
    elif recalibrate_interval > 0:
        if state.steps_since_calibrate >= recalibrate_interval:
            should_calibrate = True
            cal_reason = "interval"
    if int(calibrate_size) <= 0:
        should_calibrate = False
        cal_reason = ""

    if should_train:
        state.models, state.calibrator_final = _train_and_maybe_calibrate(
            cfg=cfg,
            closed_rounds=closed_rounds,
            train_size=int(train_size),
            calibrate_size=int(calibrate_size),
        )
        state.steps_since_train = 0
        state.last_train_epoch = int(current_epoch)

        # Retrain forces immediate recalibration and resets cadence.
        state.steps_since_calibrate = 0
        state.last_calibrate_epoch = int(current_epoch)

        info(
            "MODEL",
            "TRAIN",
            "DONE",
            msg=(
                f"Walk-forward train ({train_reason}) epoch={int(current_epoch)} "
                f"train_size={int(train_size)} calibrate_size={int(calibrate_size)} "
                f"recency_weight_floor={float(recency_weight_floor):.4f} recency_weight_power={float(recency_weight_power):.4f} "
                f"predictability_feature_mode={str(_predictability_feature_mode(cfg=cfg))} "
                f"predictability_label_mode={str(_predictability_label_mode(cfg=cfg))} "
                f"prior_context_rounds_required={int(max_required_prior_context_rounds_size())}"
            ),
        )
        return state

    # No retrain; possibly recalibrate only.
    if should_calibrate:
        if state.models is None:
            raise InvariantError("calibrate_without_models")

        (
            x_price_cal,
            _y_ret_cal,
            y_up_cal,
            x_pool_cal,
            y_late_inflow_total_cal,
            y_late_inflow_bull_frac_cal,
        ) = _build_calibration_rows(
            cfg=cfg,
            closed_rounds=closed_rounds,
            calibrate_size=int(calibrate_size),
        )

        cal_sample_weight = _build_recency_weights_for_rows(
            cfg=cfg,
            n_rows=int(len(y_up_cal)),
        )
        mu_cal = list(state.models.price_model.predict(x_price_cal))
        state.calibrator_final = _fit_final_calibrator(
            mu_cal=mu_cal,
            y_up_cal=y_up_cal,
            sample_weight=cal_sample_weight,
        )

        pool_preds_cal = list(state.models.pool_model.predict(x_pool_cal))
        p_cal = state.calibrator_final.predict_proba_up(mu_cal)
        _log_model_diagnostics(
            y_up_cal=y_up_cal,
            p_up_cal=p_cal,
            y_late_inflow_total_cal=y_late_inflow_total_cal,
            y_late_inflow_bull_frac_cal=y_late_inflow_bull_frac_cal,
            pool_preds_cal=pool_preds_cal,
        )

        state.steps_since_calibrate = 0
        state.last_calibrate_epoch = int(current_epoch)

        info(
            "MODEL",
            "CAL",
            "DONE",
            msg=(
                f"Walk-forward calibrate ({cal_reason}) epoch={int(current_epoch)} "
                f"calibrate_size={int(calibrate_size)} "
                f"recency_weight_floor={float(recency_weight_floor):.4f} recency_weight_power={float(recency_weight_power):.4f} "
                f"prior_context_rounds_required={int(max_required_prior_context_rounds_size())}"
            ),
        )
    else:
        state.steps_since_train += 1
        state.steps_since_calibrate += 1

    return state


def predict_probabilities(*, state: WalkForwardState, mu: float) -> float:
    """Return p_final for the given mu using the current walk-forward state."""
    if state.models is None:
        raise InvariantError("predict_without_models")

    if state.calibrator_final is None:
        raise InvariantError("calibrated_probability_unavailable")

    p_final = float(state.calibrator_final.predict_proba_up(float(mu)))
    return float(p_final)


def predict_tradeable_probability(*, state: WalkForwardState, x_row: list[float]) -> float:
    if state.models is None:
        raise InvariantError("predict_tradeable_without_models")
    p = float(state.models.predictability_model.predict_proba([list(x_row)])[0])
    if not math.isfinite(p):
        raise InvariantError("predict_tradeable_non_finite")
    if p < 0.0 or p > 1.0:
        raise InvariantError("predict_tradeable_out_of_range")
    return float(p)


def _train_and_maybe_calibrate(
    *,
    cfg: Any,
    closed_rounds: list[Round],
    train_size: int,
    calibrate_size: int,
) -> tuple[WalkForwardModels, IsotonicCalibrator | None]:
    k = int(max_required_prior_context_rounds_size())
    gate_feature_indices = predictability_feature_indices(
        schema=FEATURE_SCHEMA,
        mode=str(_predictability_feature_mode(cfg=cfg)),
    )

    required = k + train_size + calibrate_size
    if len(closed_rounds) < required:
        raise InvariantError("insufficient_closed_rounds_for_walk_forward_train")

    tail = list(closed_rounds[-required:])

    # Use the older slice for calibration so the base models can train on the most-recent slice.
    cal_targets = tail[k : k + calibrate_size]
    train_targets = tail[k + calibrate_size :]

    if len(train_targets) != train_size:
        raise InvariantError("train_targets_size_mismatch")
    if len(cal_targets) != calibrate_size:
        raise InvariantError("cal_targets_size_mismatch")

    (
        x_price_train,
        _y_ret_train,
        y_up_train,
        x_pool_train,
        y_late_inflow_total,
        y_late_inflow_bull_frac,
        x_gate_train,
        y_tradeable_train,
    ) = _build_training_rows(
        cfg=cfg,
        rounds=tail,
        target_begin=int(k + calibrate_size),
        target_end=int(k + calibrate_size + train_size),
        prior_context_rounds_required=int(k),
    )

    if int(calibrate_size) <= 0:
        train_sample_weight = _build_recency_weights_for_rows(
            cfg=cfg,
            n_rows=int(len(y_up_train)),
        )
        price_model = PriceReturnModel(alpha=float(cfg.price_alpha), seed=int(cfg.random_seed))
        price_model.fit(
            x_price_train,
            y_up_train,
            sample_weight=train_sample_weight,
        )

        pool_model = FinalPoolModel(
            alpha_total=float(cfg.pool_alpha_total),
            alpha_ratio=float(cfg.pool_alpha_ratio),
            seed=int(cfg.random_seed),
        )
        pool_model.fit(
            x_pool_train,
            y_late_inflow_total,
            y_late_inflow_bull_frac,
            sample_weight=train_sample_weight,
        )
        predictability_model = PredictabilityModel(
            seed=int(cfg.random_seed),
            feature_indices=gate_feature_indices,
        )
        predictability_model.fit(
            x_gate_train,
            y_tradeable_train,
            sample_weight=train_sample_weight,
        )
        models = WalkForwardModels(
            price_model=price_model,
            pool_model=pool_model,
            predictability_model=predictability_model,
        )

        calibrator_final = _fit_final_calibrator(
            mu_cal=[0.0, 1.0],
            y_up_cal=[0, 1],
            sample_weight=[1.0, 1.0],
        )
        return models, calibrator_final

    (
        x_price_cal,
        _y_ret_cal,
        y_up_cal,
        x_pool_cal,
        y_late_inflow_total_cal,
        y_late_inflow_bull_frac_cal,
        x_gate_cal,
        y_tradeable_cal,
    ) = _build_training_rows(
        cfg=cfg,
        rounds=tail,
        target_begin=int(k),
        target_end=int(k + calibrate_size),
        prior_context_rounds_required=int(k),
    )

    train_sample_weight = _build_recency_weights_for_rows(
        cfg=cfg,
        n_rows=int(len(y_up_train)),
    )
    cal_sample_weight = _build_recency_weights_for_rows(
        cfg=cfg,
        n_rows=int(len(y_up_cal)),
    )

    price_model = PriceReturnModel(alpha=float(cfg.price_alpha), seed=int(cfg.random_seed))
    price_model.fit(
        x_price_train,
        y_up_train,
        x_eval=x_price_cal,
        y_eval=y_up_cal,
        sample_weight=train_sample_weight,
    )

    pool_model = FinalPoolModel(
        alpha_total=float(cfg.pool_alpha_total),
        alpha_ratio=float(cfg.pool_alpha_ratio),
        seed=int(cfg.random_seed),
    )
    pool_model.fit(
        x_pool_train,
        y_late_inflow_total,
        y_late_inflow_bull_frac,
        x_eval=x_pool_cal,
        y_total_eval=y_late_inflow_total_cal,
        y_frac_eval=y_late_inflow_bull_frac_cal,
        sample_weight=train_sample_weight,
    )
    predictability_model = PredictabilityModel(
        seed=int(cfg.random_seed),
        feature_indices=gate_feature_indices,
    )
    predictability_model.fit(
        x_gate_train,
        y_tradeable_train,
        x_eval=x_gate_cal,
        y_eval=y_tradeable_cal,
        sample_weight=train_sample_weight,
    )

    models = WalkForwardModels(
        price_model=price_model,
        pool_model=pool_model,
        predictability_model=predictability_model,
    )

    mu_cal = list(models.price_model.predict(x_price_cal))
    calibrator_final = _fit_final_calibrator(
        mu_cal=mu_cal,
        y_up_cal=y_up_cal,
        sample_weight=cal_sample_weight,
    )

    pool_preds_cal = list(models.pool_model.predict(x_pool_cal))
    p_cal = calibrator_final.predict_proba_up(mu_cal)
    _log_model_diagnostics(
        y_up_cal=y_up_cal,
        p_up_cal=p_cal,
        y_late_inflow_total_cal=y_late_inflow_total_cal,
        y_late_inflow_bull_frac_cal=y_late_inflow_bull_frac_cal,
        pool_preds_cal=pool_preds_cal,
    )

    return models, calibrator_final


def _build_calibration_rows(
    *,
    cfg: Any,
    closed_rounds: list[Round],
    calibrate_size: int,
):
    k = int(max_required_prior_context_rounds_size())
    required = k + int(cfg.train_size) + int(calibrate_size)
    if len(closed_rounds) < required:
        raise InvariantError("insufficient_closed_rounds_for_walk_forward_calibrate")

    tail = list(closed_rounds[-required:])
    cal_begin = k
    cal_end = cal_begin + int(calibrate_size)

    (
        x_price_cal,
        y_ret_cal,
        y_up_cal,
        x_pool_cal,
        y_late_inflow_total_cal,
        y_late_inflow_bull_frac_cal,
        _x_gate_cal,
        _y_tradeable_cal,
    ) = _build_training_rows(
        cfg=cfg,
        rounds=tail,
        target_begin=int(cal_begin),
        target_end=int(cal_end),
        prior_context_rounds_required=int(k),
    )
    return (
        x_price_cal,
        y_ret_cal,
        y_up_cal,
        x_pool_cal,
        y_late_inflow_total_cal,
        y_late_inflow_bull_frac_cal,
    )


def _fit_final_calibrator(
    *,
    mu_cal: list[float],
    y_up_cal: list[int],
    sample_weight: list[float] | None = None,
) -> IsotonicCalibrator:
    cal = IsotonicCalibrator()
    cal.fit(mu_cal, y_up_cal, sample_weight=sample_weight)
    return cal


def _recency_weight_params(*, cfg: Any) -> tuple[float, float]:
    floor = float(getattr(cfg, "recency_weight_floor", 1.0))
    power = float(getattr(cfg, "recency_weight_power", 1.0))
    if not math.isfinite(float(floor)) or not (0.0 < float(floor) <= 1.0):
        raise InvariantError("recency_weight_floor_out_of_range")
    if not math.isfinite(float(power)) or float(power) <= 0.0:
        raise InvariantError("recency_weight_power_must_be_positive")
    return float(floor), float(power)


def _build_recency_weights_for_rows(*, cfg: Any, n_rows: int) -> list[float]:
    if int(n_rows) <= 0:
        raise InvariantError("recency_weight_rows_invalid")
    floor, power = _recency_weight_params(cfg=cfg)
    if int(n_rows) == 1:
        return [1.0]

    out: list[float] = []
    denom = float(int(n_rows) - 1)
    for idx in range(int(n_rows)):
        t = float(idx) / float(denom)
        w = float(floor) + (1.0 - float(floor)) * float(math.pow(float(t), float(power)))
        if not math.isfinite(float(w)) or float(w) <= 0.0:
            raise InvariantError("recency_weight_non_finite_or_nonpositive")
        out.append(float(w))

    out[0] = float(floor)
    out[-1] = 1.0
    return out


def _log_model_diagnostics(
    *,
    y_up_cal: list[int],
    p_up_cal,
    y_late_inflow_total_cal: list[float],
    y_late_inflow_bull_frac_cal: list[float],
    pool_preds_cal: list[tuple[float, float]],
) -> None:
    if not y_up_cal:
        raise InvariantError("model_diag_empty_calibration_labels")

    probs = [float(v) for v in p_up_cal]
    if len(probs) != len(y_up_cal):
        raise InvariantError("model_diag_probability_len_mismatch")
    if len(pool_preds_cal) != len(y_late_inflow_total_cal) or len(pool_preds_cal) != len(y_late_inflow_bull_frac_cal):
        raise InvariantError("model_diag_pool_len_mismatch")

    logloss = _binary_logloss(y_true=y_up_cal, p_pred=probs)
    brier = _brier_score(y_true=y_up_cal, p_pred=probs)
    ece = _expected_calibration_error(y_true=y_up_cal, p_pred=probs, bins=10)

    total_log_true: list[float] = []
    total_log_pred: list[float] = []
    frac_true: list[float] = []
    frac_pred: list[float] = []

    for idx, (pred_total, pred_frac) in enumerate(pool_preds_cal):
        y_total = float(y_late_inflow_total_cal[idx])
        y_frac = float(y_late_inflow_bull_frac_cal[idx])
        if not math.isfinite(y_total) or y_total < 0.0:
            raise InvariantError("model_diag_total_label_invalid")
        if not math.isfinite(y_frac) or not (0.0 <= y_frac <= 1.0):
            raise InvariantError("model_diag_frac_label_invalid")

        pt = max(0.0, float(pred_total))
        pf = min(1.0, max(0.0, float(pred_frac)))
        if not math.isfinite(pt):
            raise InvariantError("model_diag_total_pred_invalid")
        if not math.isfinite(pf):
            raise InvariantError("model_diag_frac_pred_invalid")

        total_log_true.append(float(math.log1p(y_total)))
        total_log_pred.append(float(math.log1p(pt)))
        frac_true.append(float(y_frac))
        frac_pred.append(float(pf))

    total_log_mae = _mae(y_true=total_log_true, y_pred=total_log_pred)
    total_log_rmse = _rmse(y_true=total_log_true, y_pred=total_log_pred)
    frac_mae = _mae(y_true=frac_true, y_pred=frac_pred)

    info(
        "MODEL",
        "DIAG",
        "CAL",
        msg=(
            f"logloss={float(logloss):.6f} brier={float(brier):.6f} ece10={float(ece):.6f} "
            f"late_total_log1p_mae={float(total_log_mae):.6f} late_total_log1p_rmse={float(total_log_rmse):.6f} "
            f"late_bull_frac_mae={float(frac_mae):.6f} n={int(len(y_up_cal))}"
        ),
    )


def _binary_logloss(*, y_true: list[int], p_pred: list[float]) -> float:
    if len(y_true) != len(p_pred) or len(y_true) == 0:
        raise InvariantError("binary_logloss_input_invalid")
    eps = 1e-6
    total = 0.0
    for yt, pp in zip(y_true, p_pred):
        if yt not in (0, 1):
            raise InvariantError("binary_logloss_y_not_binary")
        p = min(1.0 - eps, max(eps, float(pp)))
        if not math.isfinite(p):
            raise InvariantError("binary_logloss_p_non_finite")
        total += -float(yt) * math.log(p) - (1.0 - float(yt)) * math.log(1.0 - p)
    return float(total) / float(len(y_true))


def _brier_score(*, y_true: list[int], p_pred: list[float]) -> float:
    if len(y_true) != len(p_pred) or len(y_true) == 0:
        raise InvariantError("brier_input_invalid")
    total = 0.0
    for yt, pp in zip(y_true, p_pred):
        if yt not in (0, 1):
            raise InvariantError("brier_y_not_binary")
        p = min(1.0, max(0.0, float(pp)))
        if not math.isfinite(p):
            raise InvariantError("brier_p_non_finite")
        d = p - float(yt)
        total += d * d
    return float(total) / float(len(y_true))


def _expected_calibration_error(*, y_true: list[int], p_pred: list[float], bins: int) -> float:
    if len(y_true) != len(p_pred) or len(y_true) == 0:
        raise InvariantError("ece_input_invalid")
    if int(bins) <= 0:
        raise InvariantError("ece_bins_invalid")

    n = int(len(y_true))
    acc_sum = [0.0] * int(bins)
    conf_sum = [0.0] * int(bins)
    count = [0] * int(bins)

    for yt, pp in zip(y_true, p_pred):
        if yt not in (0, 1):
            raise InvariantError("ece_y_not_binary")
        p = min(1.0, max(0.0, float(pp)))
        if not math.isfinite(p):
            raise InvariantError("ece_p_non_finite")
        b = int(min(int(bins) - 1, math.floor(p * float(bins))))
        acc_sum[b] += float(yt)
        conf_sum[b] += float(p)
        count[b] += 1

    ece = 0.0
    for b in range(int(bins)):
        c = int(count[b])
        if c <= 0:
            continue
        acc = float(acc_sum[b]) / float(c)
        conf = float(conf_sum[b]) / float(c)
        ece += (float(c) / float(n)) * abs(acc - conf)
    return float(ece)


def _mae(*, y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise InvariantError("mae_input_invalid")
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        if not math.isfinite(float(yt)) or not math.isfinite(float(yp)):
            raise InvariantError("mae_non_finite_input")
        total += abs(float(yp) - float(yt))
    return float(total) / float(len(y_true))


def _rmse(*, y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise InvariantError("rmse_input_invalid")
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        if not math.isfinite(float(yt)) or not math.isfinite(float(yp)):
            raise InvariantError("rmse_non_finite_input")
        d = float(yp) - float(yt)
        total += d * d
    return math.sqrt(float(total) / float(len(y_true)))


def _feature_anchor_close_time_ms_for_round(*, cfg, round_t: Round) -> int:
    if round_t.lock_at is None:
        raise InvariantError("round_lock_at_missing")
    lock_ts = int(round_t.lock_at)
    cutoff_ts = int(lock_ts) - int(cfg.cutoff_seconds)
    anchor_ms = int(cutoff_ts) * 1000
    latest_close_ms = cfg.klines_store.latest_close_time_ms()
    if latest_close_ms is None:
        raise InvariantError("klines_store_empty")
    if int(latest_close_ms) < int(anchor_ms):
        anchor_ms = int(latest_close_ms)
    return int(anchor_ms)


def _context_klines_for_round(*, cfg, round_t: Round) -> list:
    kk = int(max_required_context_klines_size())
    anchor_ms = _feature_anchor_close_time_ms_for_round(cfg=cfg, round_t=round_t)
    return cfg.klines_store.get_context_klines(anchor_close_time_ms=int(anchor_ms), size=int(kk))


def _tradeable_feature_value(*, x_row: list[float], idx: int, error_code: str) -> float:
    if int(idx) < 0 or int(idx) >= int(len(x_row)):
        raise InvariantError(str(error_code))
    return float(x_row[int(idx)])


def _bullish_vote(*, value: float, threshold: float) -> int:
    return 1 if (not math.isfinite(float(value)) or float(value) >= float(threshold)) else -1


def _invert_side(side: str) -> str:
    side_norm = str(side)
    if str(side_norm) == "Bull":
        return "Bear"
    if str(side_norm) == "Bear":
        return "Bull"
    raise InvariantError("tradeable_side_invalid")


def _tradeable_label_side(*, label_mode: str, x_row: list[float]) -> str:
    if str(label_mode) == PREDICTABILITY_LABEL_MODE_BASELINE_LOG_IMBALANCE_SIDE:
        log_imb = _tradeable_feature_value(
            x_row=x_row,
            idx=int(_TRADEABLE_LOG_IMB_IDX),
            error_code="tradeable_log_imb_feature_index_out_of_range",
        )
        return "Bull" if _bullish_vote(value=float(log_imb), threshold=0.0) > 0 else "Bear"

    if str(label_mode) == PREDICTABILITY_LABEL_MODE_CONTRARIAN_LOG_IMBALANCE_SIDE:
        return _invert_side(
            _tradeable_label_side(
                label_mode=PREDICTABILITY_LABEL_MODE_BASELINE_LOG_IMBALANCE_SIDE,
                x_row=list(x_row),
            )
        )

    if str(label_mode) == PREDICTABILITY_LABEL_MODE_PRICE_MOMENTUM_K15_SIDE:
        momentum = _tradeable_feature_value(
            x_row=x_row,
            idx=int(_TRADEABLE_PRICE_MOMENTUM_IDX),
            error_code="tradeable_price_momentum_feature_index_out_of_range",
        )
        return "Bull" if _bullish_vote(value=float(momentum), threshold=0.0) > 0 else "Bear"

    if str(label_mode) == PREDICTABILITY_LABEL_MODE_REGIME_MAJORITY_R20_SIDE:
        bull_frac = _tradeable_feature_value(
            x_row=x_row,
            idx=int(_TRADEABLE_REGIME_BULL_FRAC_IDX),
            error_code="tradeable_regime_bull_frac_feature_index_out_of_range",
        )
        return "Bull" if _bullish_vote(value=float(bull_frac), threshold=0.5) > 0 else "Bear"

    if str(label_mode) == PREDICTABILITY_LABEL_MODE_CONTRARIAN_PRICE_REGIME_VOTE_15_30_R20_R60_SIDE:
        vote_total = 0
        vote_total += _bullish_vote(
            value=_tradeable_feature_value(
                x_row=x_row,
                idx=int(_TRADEABLE_PRICE_MOMENTUM_IDX),
                error_code="tradeable_price_momentum_feature_index_out_of_range",
            ),
            threshold=0.0,
        )
        vote_total += _bullish_vote(
            value=_tradeable_feature_value(
                x_row=x_row,
                idx=int(_TRADEABLE_PRICE_MOMENTUM_K30_IDX),
                error_code="tradeable_price_momentum_k30_feature_index_out_of_range",
            ),
            threshold=0.0,
        )
        vote_total += _bullish_vote(
            value=_tradeable_feature_value(
                x_row=x_row,
                idx=int(_TRADEABLE_REGIME_BULL_FRAC_IDX),
                error_code="tradeable_regime_bull_frac_feature_index_out_of_range",
            ),
            threshold=0.5,
        )
        vote_total += _bullish_vote(
            value=_tradeable_feature_value(
                x_row=x_row,
                idx=int(_TRADEABLE_REGIME_BULL_FRAC_R60_IDX),
                error_code="tradeable_regime_bull_frac_r60_feature_index_out_of_range",
            ),
            threshold=0.5,
        )
        consensus_side = "Bull" if int(vote_total) >= 0 else "Bear"
        return _invert_side(str(consensus_side))

    raise InvariantError("predictability_label_mode_unknown")


def _tradeable_label(*, cfg: Any, round_t: Round, x_row: list[float]) -> int:
    baseline_bet = float(getattr(cfg, "predictability_baseline_bet_bnb", 0.05))
    if not math.isfinite(float(baseline_bet)) or float(baseline_bet) <= 0.0:
        raise InvariantError("predictability_baseline_bet_bnb_invalid")

    def _baseline_pnl_for_side(side: str) -> float:
        settled = settle_bet_against_closed_round(
            bet_bnb=float(baseline_bet),
            bet_side=str(side),
            round_closed=round_t,
            treasury_fee_fraction=float(cfg.treasury_fee_fraction),
        )
        return float(settled.credit_bnb) - float(baseline_bet) - float(GAS_COST_BET_BNB)

    label_mode = _predictability_label_mode(cfg=cfg)
    if str(label_mode) == PREDICTABILITY_LABEL_MODE_EITHER_SIDE_PROFITABLE:
        bull_pnl = _baseline_pnl_for_side("Bull")
        bear_pnl = _baseline_pnl_for_side("Bear")
        return 1 if max(float(bull_pnl), float(bear_pnl)) > 0.0 else 0

    side = _tradeable_label_side(label_mode=str(label_mode), x_row=list(x_row))
    pnl = _baseline_pnl_for_side(str(side))
    return 1 if float(pnl) > 0.0 else 0


def _feature_vector_for_round(
    *,
    cfg: Any,
    round_t: Round,
    prior_context_rounds: list[Round],
) -> list[float]:
    if round_t.lock_at is None:
        raise InvariantError("feature_vector_round_lock_at_missing")

    prior_last_epoch = int(prior_context_rounds[-1].epoch) if prior_context_rounds else 0
    anchor_close_time_ms = _feature_anchor_close_time_ms_for_round(cfg=cfg, round_t=round_t)
    lock_at = int(round_t.lock_at)
    feature_cache = getattr(cfg, "feature_cache_store", None)

    if feature_cache is not None:
        if not hasattr(feature_cache, "get_vector") or not hasattr(feature_cache, "put_vector"):
            raise InvariantError("feature_cache_store_invalid")
        cached = feature_cache.get_vector(
            epoch=int(round_t.epoch),
            cutoff_seconds=int(cfg.cutoff_seconds),
            schema_name=str(FEATURE_SCHEMA.name),
            start_at=int(round_t.start_at),
            lock_at=int(lock_at),
            prior_last_epoch=int(prior_last_epoch),
            anchor_close_time_ms=int(anchor_close_time_ms),
        )
        if cached is not None:
            return list(cached)

    context_klines = _context_klines_for_round(cfg=cfg, round_t=round_t)
    if not context_klines:
        raise InvariantError("feature_vector_context_klines_empty")

    feats = build_features(
        target_round=round_t,
        prior_context_rounds=prior_context_rounds,
        context_klines=context_klines,
        cutoff_seconds=int(cfg.cutoff_seconds),
    )
    x_row = vectorize(features=feats, schema=FEATURE_SCHEMA)

    if feature_cache is not None:
        feature_cache.put_vector(
            epoch=int(round_t.epoch),
            cutoff_seconds=int(cfg.cutoff_seconds),
            schema_name=str(FEATURE_SCHEMA.name),
            start_at=int(round_t.start_at),
            lock_at=int(lock_at),
            prior_last_epoch=int(prior_last_epoch),
            anchor_close_time_ms=int(anchor_close_time_ms),
            vector=list(x_row),
        )
    return list(x_row)


def _build_training_rows(
    *,
    cfg: Any,
    rounds: list[Round],
    target_begin: int,
    target_end: int,
    prior_context_rounds_required: int,
):
    """Build training rows for targets in rounds[target_begin:target_end].

    rounds must be epoch-ascending (oldest -> newest).
    For each target r_t at index i, prior context is rounds[:i].
    """
    if target_begin < 0 or target_end < 0 or target_end < target_begin:
        raise InvariantError("target_slice_invalid")
    if target_end > len(rounds):
        raise InvariantError("target_slice_out_of_bounds")

    x_price: list[list[float]] = []
    y_ret: list[float] = []
    y_up: list[int] = []

    x_pool: list[list[float]] = []
    y_late_inflow_total: list[float] = []
    y_late_inflow_bull_frac: list[float] = []
    x_gate: list[list[float]] = []
    y_tradeable: list[int] = []

    used = 0
    for i in range(int(target_begin), int(target_end)):
        r_t = rounds[i]
        if r_t.lock_at is None:
            raise InvariantError("train_round_missing_lock_at")

        prior_closed = list(rounds[:i])
        k = int(prior_context_rounds_required)
        prior_context_rounds = ([] if k <= 0 else list(prior_closed[-k:]))
        if len(prior_context_rounds) != int(k):
            raise InvariantError(
                f"prior_context_rounds_len_mismatch: got={len(prior_context_rounds)} expected={int(k)}"
            )
        if prior_context_rounds and int(prior_context_rounds[-1].epoch) >= int(r_t.epoch):
            raise InvariantError("train_prior_not_strictly_before_t")

        x_row = _feature_vector_for_round(
            cfg=cfg,
            round_t=r_t,
            prior_context_rounds=prior_context_rounds,
        )

        x_price.append(list(x_row))
        x_pool.append(list(x_row))
        x_gate.append(list(x_row))

        price_targets = compute_price_targets(round_t=r_t)
        ret_open = float(price_targets.ret_open)
        up = int(price_targets.up)
        if not math.isfinite(ret_open):
            raise InvariantError("price_target_ret_open_non_finite")
        if up not in (0, 1):
            raise InvariantError("price_target_up_not_binary")
        y_ret.append(float(ret_open))
        y_up.append(int(up))

        pool_targets = compute_pool_forecast_targets(round_t=r_t, cutoff_seconds=int(cfg.cutoff_seconds))
        late_total = float(pool_targets.late_inflow_total_bnb)
        late_frac = float(pool_targets.late_inflow_bull_frac)
        if not math.isfinite(late_total) or late_total < 0.0:
            raise InvariantError("pool_target_late_total_invalid")
        if not math.isfinite(late_frac) or not (0.0 <= late_frac <= 1.0):
            raise InvariantError("pool_target_late_frac_invalid")
        y_late_inflow_total.append(float(late_total))
        y_late_inflow_bull_frac.append(float(late_frac))

        y_tradeable.append(int(_tradeable_label(cfg=cfg, round_t=r_t, x_row=list(x_row))))

        used += 1

    if used <= 0:
        raise InvariantError("train_insufficient_history")

    return x_price, y_ret, y_up, x_pool, y_late_inflow_total, y_late_inflow_bull_frac, x_gate, y_tradeable

