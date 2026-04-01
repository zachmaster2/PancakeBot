from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.feature_builder import build_features, vectorize
from pancakebot.domain.features.schema import (
    FEATURE_SCHEMA,
    max_required_context_klines_size,
    max_required_prior_context_rounds_size,
)
from pancakebot.domain.types import Round
from pancakebot.infra.feature_cache_store import FeatureCacheStore


@dataclass(frozen=True, slots=True)
class NeuralDirectionDataset:
    feature_columns: tuple[str, ...]
    target_epochs: tuple[int, ...]
    labels: np.ndarray
    previous_settled_labels: np.ndarray
    previous_settled_available: np.ndarray
    feature_matrix: np.ndarray
    metadata: dict[str, object]

    @property
    def num_examples(self) -> int:
        return int(len(self.target_epochs))


def available_feature_groups() -> tuple[str, ...]:
    seen: list[str] = []
    for feature_def in FEATURE_SCHEMA.features:
        group = str(feature_def.group)
        if group not in seen:
            seen.append(group)
    return tuple(seen)


def neural_direction_required_history_rounds() -> int:
    return int(max_required_prior_context_rounds_size())


def neural_direction_required_context_klines() -> int:
    return int(max_required_context_klines_size())


def tail_neural_direction_dataset(*, dataset: NeuralDirectionDataset, n: int) -> NeuralDirectionDataset:
    if int(n) <= 0:
        raise InvariantError("neural_direction_tail_n_nonpositive")
    if int(dataset.num_examples) < int(n):
        raise InvariantError("neural_direction_tail_n_exceeds_dataset")
    start = int(dataset.num_examples) - int(n)
    metadata = dict(dataset.metadata)
    metadata["tail_n"] = int(n)
    return NeuralDirectionDataset(
        feature_columns=tuple(dataset.feature_columns),
        target_epochs=tuple(int(epoch) for epoch in dataset.target_epochs[start:]),
        labels=np.asarray(dataset.labels[start:], dtype=np.int64),
        previous_settled_labels=np.asarray(dataset.previous_settled_labels[start:], dtype=np.int64),
        previous_settled_available=np.asarray(dataset.previous_settled_available[start:], dtype=bool),
        feature_matrix=np.asarray(dataset.feature_matrix[start:], dtype=np.float32),
        metadata=metadata,
    )


def select_feature_groups(
    *,
    dataset: NeuralDirectionDataset,
    include_groups: Sequence[str] | None = None,
    exclude_groups: Sequence[str] | None = None,
) -> NeuralDirectionDataset:
    include = None if include_groups is None else tuple(str(group) for group in include_groups)
    exclude = None if exclude_groups is None else tuple(str(group) for group in exclude_groups)
    available = set(available_feature_groups())
    for group in include or ():
        if str(group) not in available:
            raise InvariantError(f"neural_direction_include_group_unknown: {str(group)}")
    for group in exclude or ():
        if str(group) not in available:
            raise InvariantError(f"neural_direction_exclude_group_unknown: {str(group)}")

    selected_columns: list[str] = []
    selected_indices: list[int] = []
    for idx, feature_def in enumerate(FEATURE_SCHEMA.features):
        group = str(feature_def.group)
        if include is not None and str(group) not in include:
            continue
        if exclude is not None and str(group) in exclude:
            continue
        selected_indices.append(int(idx))
        selected_columns.append(str(feature_def.name))
    if not selected_indices:
        raise InvariantError("neural_direction_feature_selection_empty")

    metadata = dict(dataset.metadata)
    metadata["selected_feature_groups_include"] = None if include is None else tuple(include)
    metadata["selected_feature_groups_exclude"] = None if exclude is None else tuple(exclude)
    metadata["selected_feature_columns_count"] = int(len(selected_indices))
    return NeuralDirectionDataset(
        feature_columns=tuple(selected_columns),
        target_epochs=tuple(int(epoch) for epoch in dataset.target_epochs),
        labels=np.asarray(dataset.labels, dtype=np.int64),
        previous_settled_labels=np.asarray(dataset.previous_settled_labels, dtype=np.int64),
        previous_settled_available=np.asarray(dataset.previous_settled_available, dtype=bool),
        feature_matrix=np.asarray(dataset.feature_matrix[:, selected_indices], dtype=np.float32),
        metadata=metadata,
    )


def build_neural_direction_dataset(
    *,
    rounds: Sequence[Round],
    klines_store_like: Any,
    cutoff_seconds: int,
    feature_cache_store: FeatureCacheStore | None = None,
) -> NeuralDirectionDataset:
    history_n = int(neural_direction_required_history_rounds())
    if int(history_n) <= 0:
        raise InvariantError("neural_direction_history_n_invalid")
    if len(rounds) <= int(history_n):
        raise InvariantError("neural_direction_rounds_insufficient")
    if int(cutoff_seconds) <= 0:
        raise InvariantError("neural_direction_cutoff_seconds_nonpositive")

    target_epochs: list[int] = []
    labels: list[int] = []
    previous_settled_labels: list[int] = []
    previous_settled_available: list[bool] = []
    feature_rows: list[list[float]] = []

    skipped_house = 0
    skipped_failed = 0
    skipped_unusable = 0

    for idx in range(int(history_n), len(rounds)):
        target_round = rounds[int(idx)]
        position = str(target_round.position) if target_round.position is not None else ""

        if bool(target_round.failed):
            skipped_failed += 1
            continue
        if position == "House":
            skipped_house += 1
            continue
        if position not in ("Bull", "Bear"):
            skipped_unusable += 1
            continue
        if target_round.lock_at is None or target_round.close_at is None:
            skipped_unusable += 1
            continue
        if target_round.lock_price is None or target_round.close_price is None:
            skipped_unusable += 1
            continue
        if float(target_round.lock_price) <= 0.0 or float(target_round.close_price) <= 0.0:
            skipped_unusable += 1
            continue

        prior_context_rounds = list(rounds[int(idx) - int(history_n) : int(idx)])
        if len(prior_context_rounds) != int(history_n):
            raise InvariantError("neural_direction_prior_context_len_mismatch")
        prior_last_epoch = int(prior_context_rounds[-1].epoch)
        anchor_close_time_ms = int(int(target_round.lock_at) - int(cutoff_seconds)) * 1000

        cached_vector: list[float] | None = None
        if feature_cache_store is not None:
            cached_vector = feature_cache_store.get_vector(
                epoch=int(target_round.epoch),
                cutoff_seconds=int(cutoff_seconds),
                schema_name=str(FEATURE_SCHEMA.name),
                start_at=int(target_round.start_at),
                lock_at=int(target_round.lock_at),
                prior_last_epoch=int(prior_last_epoch),
                anchor_close_time_ms=int(anchor_close_time_ms),
            )

        if cached_vector is None:
            try:
                context_klines = list(
                    klines_store_like.get_context_klines(
                        anchor_close_time_ms=int(anchor_close_time_ms),
                        size=int(neural_direction_required_context_klines()),
                    )
                )
            except Exception as exc:
                raise InvariantError(
                    f"neural_direction_context_klines_unavailable: epoch={int(target_round.epoch)} err={exc}"
                ) from exc

            features = build_features(
                target_round=target_round,
                prior_context_rounds=prior_context_rounds,
                context_klines=context_klines,
                cutoff_seconds=int(cutoff_seconds),
            )
            cached_vector = vectorize(features=features, schema=FEATURE_SCHEMA)
            if feature_cache_store is not None:
                feature_cache_store.put_vector(
                    epoch=int(target_round.epoch),
                    cutoff_seconds=int(cutoff_seconds),
                    schema_name=str(FEATURE_SCHEMA.name),
                    start_at=int(target_round.start_at),
                    lock_at=int(target_round.lock_at),
                    prior_last_epoch=int(prior_last_epoch),
                    anchor_close_time_ms=int(anchor_close_time_ms),
                    vector=list(cached_vector),
                )

        prev_label, prev_available = previous_settled_direction_label(prior_context_rounds=prior_context_rounds)

        target_epochs.append(int(target_round.epoch))
        labels.append(direction_label_from_position(position))
        previous_settled_labels.append(int(prev_label))
        previous_settled_available.append(bool(prev_available))
        feature_rows.append(list(float(value) for value in cached_vector))

    if not target_epochs:
        raise InvariantError("neural_direction_dataset_empty")

    return NeuralDirectionDataset(
        feature_columns=tuple(str(col) for col in FEATURE_SCHEMA.columns),
        target_epochs=tuple(int(epoch) for epoch in target_epochs),
        labels=np.asarray(labels, dtype=np.int64),
        previous_settled_labels=np.asarray(previous_settled_labels, dtype=np.int64),
        previous_settled_available=np.asarray(previous_settled_available, dtype=bool),
        feature_matrix=np.asarray(feature_rows, dtype=np.float32),
        metadata={
            "schema_name": str(FEATURE_SCHEMA.name),
            "cutoff_seconds": int(cutoff_seconds),
            "required_history_rounds": int(history_n),
            "required_context_klines": int(neural_direction_required_context_klines()),
            "num_examples": int(len(target_epochs)),
            "skipped_house": int(skipped_house),
            "skipped_failed": int(skipped_failed),
            "skipped_unusable": int(skipped_unusable),
        },
    )


def direction_label_from_position(position: str) -> int:
    if str(position) == "Bull":
        return 1
    if str(position) == "Bear":
        return 0
    raise InvariantError("neural_direction_position_invalid")


def previous_settled_direction_label(*, prior_context_rounds: Sequence[Round]) -> tuple[int, bool]:
    if not prior_context_rounds:
        raise InvariantError("neural_direction_prior_context_empty")
    for round_t in reversed(list(prior_context_rounds[:-1])):
        if str(round_t.position) == "Bull":
            return 1, True
        if str(round_t.position) == "Bear":
            return 0, True
    return 1, False
