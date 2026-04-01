from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from pancakebot.core.errors import InvariantError


_DEFAULT_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class NeuralDirectionTemperatureCalibrator:
    temperature: float
    valid_loss_before: float
    valid_loss_after: float


@dataclass(frozen=True, slots=True)
class NeuralDirectionConfidenceBucket:
    coverage_fraction_requested: float
    selected_count: int
    selected_fraction_actual: float
    selected_win_rate: float
    selected_mean_confidence: float
    selected_min_confidence: float
    selected_max_confidence: float


def fit_temperature_calibrator_from_probs(
    *,
    bull_probs: np.ndarray,
    labels: np.ndarray,
    max_steps: int = 400,
    learning_rate: float = 0.05,
) -> NeuralDirectionTemperatureCalibrator:
    probs = np.asarray(bull_probs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    if probs.ndim != 1:
        raise InvariantError("neural_direction_calibration_probs_rank_invalid")
    if y.ndim != 1:
        raise InvariantError("neural_direction_calibration_labels_rank_invalid")
    if len(probs) != len(y):
        raise InvariantError("neural_direction_calibration_len_mismatch")
    if len(probs) <= 0:
        raise InvariantError("neural_direction_calibration_empty")

    logits = _logit_from_probs(probs=probs)
    logits_tensor = torch.from_numpy(logits.astype(np.float32))
    labels_tensor = torch.from_numpy(y.astype(np.float32))
    loss_fn = nn.BCEWithLogitsLoss()
    valid_loss_before = float(loss_fn(logits_tensor, labels_tensor).item())

    log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.Adam([log_temperature], lr=float(learning_rate))

    for _ in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp_min(1e-3)
        scaled_logits = logits_tensor / temperature
        loss = loss_fn(scaled_logits, labels_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        temperature = float(torch.exp(log_temperature).clamp_min(1e-3).item())
        scaled_logits = logits_tensor / float(temperature)
        valid_loss_after = float(loss_fn(scaled_logits, labels_tensor).item())

    return NeuralDirectionTemperatureCalibrator(
        temperature=float(temperature),
        valid_loss_before=float(valid_loss_before),
        valid_loss_after=float(valid_loss_after),
    )


def apply_temperature_calibrator_to_probs(
    *,
    bull_probs: np.ndarray,
    calibrator: NeuralDirectionTemperatureCalibrator,
) -> np.ndarray:
    probs = np.asarray(bull_probs, dtype=np.float32)
    if probs.ndim != 1:
        raise InvariantError("neural_direction_apply_calibration_probs_rank_invalid")
    logits = _logit_from_probs(probs=probs)
    scaled_logits = logits / float(calibrator.temperature)
    out = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.asarray(out, dtype=np.float32)


def chosen_side_confidence(
    *,
    predicted_labels: np.ndarray,
    calibrated_bull_probs: np.ndarray,
) -> np.ndarray:
    preds = np.asarray(predicted_labels, dtype=np.int64)
    probs = np.asarray(calibrated_bull_probs, dtype=np.float32)
    if preds.ndim != 1 or probs.ndim != 1:
        raise InvariantError("neural_direction_confidence_rank_invalid")
    if len(preds) != len(probs):
        raise InvariantError("neural_direction_confidence_len_mismatch")
    conf = np.where(preds == 1, probs, 1.0 - probs)
    return np.asarray(conf, dtype=np.float32)


def summarize_confidence_buckets(
    *,
    labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidence: np.ndarray,
    coverage_fractions: Sequence[float],
) -> list[NeuralDirectionConfidenceBucket]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(predicted_labels, dtype=np.int64)
    conf = np.asarray(confidence, dtype=np.float32)
    if y_true.ndim != 1 or y_pred.ndim != 1 or conf.ndim != 1:
        raise InvariantError("neural_direction_bucket_rank_invalid")
    if len(y_true) != len(y_pred) or len(y_true) != len(conf):
        raise InvariantError("neural_direction_bucket_len_mismatch")
    if len(y_true) <= 0:
        raise InvariantError("neural_direction_bucket_empty")

    correct = (y_true == y_pred).astype(np.float32)
    order = np.argsort(-conf, kind="stable")
    buckets: list[NeuralDirectionConfidenceBucket] = []
    total = int(len(y_true))
    for raw_fraction in coverage_fractions:
        fraction = float(raw_fraction)
        if not (0.0 < float(fraction) <= 1.0):
            raise InvariantError("neural_direction_bucket_fraction_invalid")
        selected_count = max(1, int(np.ceil(float(total) * float(fraction))))
        selected_idx = order[:selected_count]
        selected_conf = conf[selected_idx]
        buckets.append(
            NeuralDirectionConfidenceBucket(
                coverage_fraction_requested=float(fraction),
                selected_count=int(selected_count),
                selected_fraction_actual=float(selected_count / float(total)),
                selected_win_rate=float(np.mean(correct[selected_idx])),
                selected_mean_confidence=float(np.mean(selected_conf)),
                selected_min_confidence=float(np.min(selected_conf)),
                selected_max_confidence=float(np.max(selected_conf)),
            )
        )
    return buckets


def _logit_from_probs(*, probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=np.float32), _DEFAULT_EPS, 1.0 - _DEFAULT_EPS)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float32)
