"""Direct action-value dataset/model utilities."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
from lightgbm import LGBMRegressor, early_stopping

from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.feature_builder import build_features, vectorize
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei_at_or_before
from pancakebot.domain.features.schema import FEATURE_SCHEMA, max_required_context_klines_size, max_required_prior_context_rounds_size
from pancakebot.domain.types import Kline, Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round


warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBM(Regressor|Classifier) was fitted with feature names",
    category=UserWarning,
)


_DIRECT_ACTION_BUNDLE_VERSION = "direct_action_bundle_v1"
_DIRECT_ACTION_SUMMARY_HORIZONS = (24, 72, 216)
_DEFAULT_ACTION_SIZES_BNB = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50)
_LGBM_Q10 = {
    "objective": "quantile",
    "alpha": 0.10,
    "metric": ["quantile"],
    "learning_rate": 0.05,
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
_LGBM_Q50 = {
    "objective": "quantile",
    "alpha": 0.50,
    "metric": ["quantile"],
    "learning_rate": 0.05,
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
_LGBM_TRAINING = {
    "num_boost_round": 800,
    "early_stopping_rounds": 50,
}


@dataclass(frozen=True, slots=True)
class DirectActionSpec:
    """One direct runtime action."""

    action_id: str
    label: str
    action: str
    bet_side: str | None
    bet_size_bnb: float


@dataclass(frozen=True, slots=True)
class DirectActionFeatureRow:
    """One `(round, action)` feature/label row."""

    target_epoch: int
    action_id: str
    feature_values: tuple[float, ...]
    realized_net_bnb: float


@dataclass(frozen=True, slots=True)
class DirectActionModelBundle:
    """Persisted direct-action model artifact."""

    feature_names: tuple[str, ...]
    action_specs: tuple[DirectActionSpec, ...]
    q10_model: object
    q50_model: object
    metadata: dict[str, object]

    def predict_quantiles(self, feature_rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
        x_arr = np.asarray(feature_rows, dtype=float)
        if x_arr.ndim != 2:
            raise InvariantError("direct_action_predict_x_not_2d")
        if int(x_arr.shape[0]) <= 0:
            return [], []
        if int(x_arr.shape[1]) != int(len(self.feature_names)):
            raise InvariantError("direct_action_predict_feature_count_mismatch")
        q10 = np.asarray(self.q10_model.predict(x_arr), dtype=float)
        q50 = np.asarray(self.q50_model.predict(x_arr), dtype=float)
        if q10.shape != q50.shape:
            raise InvariantError("direct_action_predict_quantile_shape_mismatch")
        if not np.all(np.isfinite(q10)) or not np.all(np.isfinite(q50)):
            raise InvariantError("direct_action_predict_non_finite")
        return q10.tolist(), q50.tolist()


@dataclass(frozen=True, slots=True)
class DirectActionDataset:
    """Contiguous direct-action dataset over a round tail."""

    feature_names: tuple[str, ...]
    action_specs: tuple[DirectActionSpec, ...]
    target_epochs: tuple[int, ...]
    rows: tuple[DirectActionFeatureRow, ...]
    rows_by_epoch: dict[int, dict[str, DirectActionFeatureRow]]


def default_direct_action_specs(
    action_sizes_bnb: Sequence[float] = _DEFAULT_ACTION_SIZES_BNB,
) -> tuple[DirectActionSpec, ...]:
    """Return the frozen first direct-action grid."""

    out = [
        DirectActionSpec(
            action_id="skip",
            label="Skip",
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
        )
    ]
    for size in action_sizes_bnb:
        size_bnb = float(size)
        if not math.isfinite(size_bnb) or size_bnb <= 0.0:
            raise InvariantError("direct_action_size_invalid")
        token = str(f"{size_bnb:.2f}").replace(".", "p")
        out.append(
            DirectActionSpec(
                action_id=f"bull_{token}",
                label=f"Bull @ {size_bnb:.2f}",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=float(size_bnb),
            )
        )
        out.append(
            DirectActionSpec(
                action_id=f"bear_{token}",
                label=f"Bear @ {size_bnb:.2f}",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=float(size_bnb),
            )
        )
    return tuple(out)


def direct_action_summary_horizons() -> tuple[int, ...]:
    return tuple(int(value) for value in _DIRECT_ACTION_SUMMARY_HORIZONS)


def direct_action_required_history_rounds() -> int:
    return int(max(int(max_required_prior_context_rounds_size()), int(max(_DIRECT_ACTION_SUMMARY_HORIZONS))))


def direct_action_feature_names() -> tuple[str, ...]:
    names = list(FEATURE_SCHEMA.columns)
    names.extend(
        [
            "action_is_skip",
            "action_is_bull",
            "action_is_bear",
            "action_bet_size_bnb",
            "action_log1p_bet_size_bnb",
            "action_cutoff_pool_share_total",
            "action_cutoff_pool_share_side",
        ]
    )
    for horizon in _DIRECT_ACTION_SUMMARY_HORIZONS:
        names.extend(
            [
                f"action_mean_net_h{int(horizon)}",
                f"action_positive_rate_h{int(horizon)}",
                f"action_std_net_h{int(horizon)}",
            ]
        )
    return tuple(str(name) for name in names)


def save_direct_action_bundle(*, bundle: DirectActionModelBundle, path: str) -> None:
    out = Path(str(path))
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": str(_DIRECT_ACTION_BUNDLE_VERSION),
        "feature_names": tuple(str(name) for name in bundle.feature_names),
        "action_specs": [
            {
                "action_id": str(spec.action_id),
                "label": str(spec.label),
                "action": str(spec.action),
                "bet_side": (None if spec.bet_side is None else str(spec.bet_side)),
                "bet_size_bnb": float(spec.bet_size_bnb),
            }
            for spec in bundle.action_specs
        ],
        "metadata": dict(bundle.metadata),
        "q10_model": bundle.q10_model,
        "q50_model": bundle.q50_model,
    }
    with gzip.open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_direct_action_bundle(path: str) -> DirectActionModelBundle:
    with gzip.open(Path(str(path)), "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise InvariantError("direct_action_bundle_payload_invalid")
    if str(payload.get("version", "")) != str(_DIRECT_ACTION_BUNDLE_VERSION):
        raise InvariantError("direct_action_bundle_version_invalid")
    feature_names_raw = tuple(str(name) for name in payload.get("feature_names", ()))
    if not feature_names_raw:
        raise InvariantError("direct_action_bundle_feature_names_empty")
    action_specs_raw = payload.get("action_specs", [])
    if not isinstance(action_specs_raw, list) or not action_specs_raw:
        raise InvariantError("direct_action_bundle_action_specs_empty")
    action_specs: list[DirectActionSpec] = []
    for row in action_specs_raw:
        if not isinstance(row, dict):
            raise InvariantError("direct_action_bundle_action_spec_invalid")
        action_specs.append(
            DirectActionSpec(
                action_id=str(row.get("action_id", "")),
                label=str(row.get("label", "")),
                action=str(row.get("action", "")),
                bet_side=(None if row.get("bet_side") is None else str(row.get("bet_side"))),
                bet_size_bnb=float(row.get("bet_size_bnb", 0.0)),
            )
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise InvariantError("direct_action_bundle_metadata_invalid")
    return DirectActionModelBundle(
        feature_names=tuple(feature_names_raw),
        action_specs=tuple(action_specs),
        q10_model=payload.get("q10_model"),
        q50_model=payload.get("q50_model"),
        metadata=dict(metadata),
    )


def build_direct_action_dataset(
    *,
    rounds: Sequence[Round],
    klines_store_like: object,
    cutoff_seconds: int,
    treasury_fee_fraction: float,
    feature_cache_store: object | None = None,
    action_specs: Sequence[DirectActionSpec] | None = None,
) -> DirectActionDataset:
    """Build one direct-action dataset for a contiguous closed-round slice."""

    feature_names = direct_action_feature_names()
    specs = tuple(default_direct_action_specs() if action_specs is None else action_specs)
    if not rounds:
        raise InvariantError("direct_action_dataset_rounds_empty")
    required_history = int(direct_action_required_history_rounds())
    base_prior_required = int(max_required_prior_context_rounds_size())
    if len(rounds) <= int(required_history):
        raise InvariantError("direct_action_dataset_rounds_insufficient")

    rows: list[DirectActionFeatureRow] = []
    rows_by_epoch: dict[int, dict[str, DirectActionFeatureRow]] = {}
    target_epochs: list[int] = []
    for idx in range(int(required_history), int(len(rounds))):
        round_t = rounds[int(idx)]
        target_epochs.append(int(round_t.epoch))
        prior_context_rounds = list(rounds[int(idx) - int(base_prior_required) : int(idx)])
        summary_rounds = list(rounds[max(0, int(idx) - int(required_history)) : int(idx)])
        if len(prior_context_rounds) != int(base_prior_required):
            raise InvariantError("direct_action_prior_context_rounds_insufficient")
        base_vector = _base_feature_vector_for_round(
            round_t=round_t,
            prior_context_rounds=prior_context_rounds,
            klines_store_like=klines_store_like,
            cutoff_seconds=int(cutoff_seconds),
            feature_cache_store=feature_cache_store,
        )
        row_map: dict[str, DirectActionFeatureRow] = {}
        for spec in specs:
            feature_values = _direct_action_feature_row_values(
                round_t=round_t,
                summary_rounds=summary_rounds,
                base_vector=base_vector,
                action_spec=spec,
                cutoff_seconds=int(cutoff_seconds),
                treasury_fee_fraction=float(treasury_fee_fraction),
            )
            row = DirectActionFeatureRow(
                target_epoch=int(round_t.epoch),
                action_id=str(spec.action_id),
                feature_values=tuple(float(v) for v in feature_values),
                realized_net_bnb=float(
                    realized_net_bnb_for_action(
                        action_spec=spec,
                        round_closed=round_t,
                        treasury_fee_fraction=float(treasury_fee_fraction),
                    )
                ),
            )
            row_map[str(spec.action_id)] = row
            rows.append(row)
        rows_by_epoch[int(round_t.epoch)] = row_map

    return DirectActionDataset(
        feature_names=tuple(feature_names),
        action_specs=tuple(specs),
        target_epochs=tuple(target_epochs),
        rows=tuple(rows),
        rows_by_epoch=rows_by_epoch,
    )


def train_direct_action_bundle(
    *,
    dataset: DirectActionDataset,
    train_target_epochs: Sequence[int],
    valid_target_epochs: Sequence[int],
    random_seed: int,
    recency_weight_floor: float = 0.5,
    recency_weight_power: float = 1.0,
    extra_metadata: dict[str, object] | None = None,
) -> DirectActionModelBundle:
    """Fit the first direct-action quantile bundle."""

    train_epochs = tuple(int(epoch) for epoch in train_target_epochs)
    valid_epochs = tuple(int(epoch) for epoch in valid_target_epochs)
    if not train_epochs:
        raise InvariantError("direct_action_train_epochs_empty")
    train_epoch_set = {int(epoch) for epoch in train_epochs}
    valid_epoch_set = {int(epoch) for epoch in valid_epochs}

    x_train: list[list[float]] = []
    y_train: list[float] = []
    sample_weight: list[float] = []
    weight_by_epoch = _build_epoch_recency_weights(
        target_epochs=train_epochs,
        floor=float(recency_weight_floor),
        power=float(recency_weight_power),
    )
    x_valid: list[list[float]] = []
    y_valid: list[float] = []
    for row in dataset.rows:
        epoch = int(row.target_epoch)
        if int(epoch) in train_epoch_set:
            x_train.append(list(row.feature_values))
            y_train.append(float(row.realized_net_bnb))
            sample_weight.append(float(weight_by_epoch[int(epoch)]))
        elif int(epoch) in valid_epoch_set:
            x_valid.append(list(row.feature_values))
            y_valid.append(float(row.realized_net_bnb))

    if len(x_train) <= 1:
        raise InvariantError("direct_action_train_rows_insufficient")

    x_train_arr = np.asarray(x_train, dtype=float)
    y_train_arr = np.asarray(y_train, dtype=float)
    sample_weight_arr = np.asarray(sample_weight, dtype=float)
    if x_train_arr.ndim != 2 or y_train_arr.ndim != 1:
        raise InvariantError("direct_action_train_array_shape_invalid")

    q10_model = _fit_quantile_model(
        x_train=x_train_arr,
        y_train=y_train_arr,
        sample_weight=sample_weight_arr,
        x_valid=(None if not x_valid else np.asarray(x_valid, dtype=float)),
        y_valid=(None if not y_valid else np.asarray(y_valid, dtype=float)),
        params=dict(_LGBM_Q10),
        random_seed=int(random_seed),
    )
    q50_model = _fit_quantile_model(
        x_train=x_train_arr,
        y_train=y_train_arr,
        sample_weight=sample_weight_arr,
        x_valid=(None if not x_valid else np.asarray(x_valid, dtype=float)),
        y_valid=(None if not y_valid else np.asarray(y_valid, dtype=float)),
        params=dict(_LGBM_Q50),
        random_seed=int(random_seed),
    )
    metadata = {
        "bundle_version": str(_DIRECT_ACTION_BUNDLE_VERSION),
        "summary_horizons": list(int(value) for value in _DIRECT_ACTION_SUMMARY_HORIZONS),
        "required_history_rounds": int(direct_action_required_history_rounds()),
        "score_mode": "q10",
        "train_target_epochs": [int(value) for value in train_epochs],
        "valid_target_epochs": [int(value) for value in valid_epochs],
        "recency_weight_floor": float(recency_weight_floor),
        "recency_weight_power": float(recency_weight_power),
        "random_seed": int(random_seed),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return DirectActionModelBundle(
        feature_names=tuple(dataset.feature_names),
        action_specs=tuple(dataset.action_specs),
        q10_model=q10_model,
        q50_model=q50_model,
        metadata=metadata,
    )


def realized_net_bnb_for_action(
    *,
    action_spec: DirectActionSpec,
    round_closed: Round,
    treasury_fee_fraction: float,
) -> float:
    """Return realized net `BNB` for one action on one closed round."""

    if str(action_spec.action) == "SKIP":
        return 0.0
    if str(action_spec.action) != "BET":
        raise InvariantError("direct_action_spec_action_invalid")
    if action_spec.bet_side not in ("Bull", "Bear"):
        raise InvariantError("direct_action_spec_bet_side_invalid")
    if float(action_spec.bet_size_bnb) <= 0.0:
        raise InvariantError("direct_action_spec_bet_size_nonpositive")
    outcome = settle_bet_against_closed_round(
        bet_bnb=float(action_spec.bet_size_bnb),
        bet_side=str(action_spec.bet_side),
        round_closed=round_closed,
        treasury_fee_fraction=float(treasury_fee_fraction),
    )
    credit_bnb = float(outcome.credit_bnb)
    return float(credit_bnb) - float(action_spec.bet_size_bnb) - float(GAS_COST_BET_BNB)


def action_spec_by_id(action_specs: Sequence[DirectActionSpec], action_id: str) -> DirectActionSpec:
    for spec in action_specs:
        if str(spec.action_id) == str(action_id):
            return spec
    raise InvariantError(f"direct_action_spec_missing: {action_id}")


def _fit_quantile_model(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    params: dict[str, object],
    random_seed: int,
) -> LGBMRegressor:
    model_params = dict(params)
    model_params["seed"] = int(random_seed)
    model_params["feature_fraction_seed"] = int(random_seed)
    model_params["bagging_seed"] = int(random_seed)
    model_params["data_random_seed"] = int(random_seed)
    model = LGBMRegressor(
        n_estimators=int(_LGBM_TRAINING["num_boost_round"]),
        **model_params,
    )
    fit_kwargs: dict[str, object] = {
        "sample_weight": sample_weight,
    }
    if x_valid is not None and y_valid is not None and int(len(y_valid)) > 1:
        fit_kwargs["eval_set"] = [(x_valid, y_valid)]
        fit_kwargs["eval_metric"] = "quantile"
        fit_kwargs["callbacks"] = [
            early_stopping(
                stopping_rounds=int(_LGBM_TRAINING["early_stopping_rounds"]),
                first_metric_only=True,
                verbose=False,
            )
        ]
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def _build_epoch_recency_weights(
    *,
    target_epochs: Sequence[int],
    floor: float,
    power: float,
) -> dict[int, float]:
    if not target_epochs:
        raise InvariantError("direct_action_recency_epochs_empty")
    if not math.isfinite(float(floor)) or not (0.0 < float(floor) <= 1.0):
        raise InvariantError("direct_action_recency_floor_out_of_range")
    if not math.isfinite(float(power)) or float(power) <= 0.0:
        raise InvariantError("direct_action_recency_power_nonpositive")
    if len(target_epochs) == 1:
        return {int(target_epochs[0]): 1.0}
    denom = float(len(target_epochs) - 1)
    out: dict[int, float] = {}
    for idx, epoch in enumerate(target_epochs):
        t = float(idx) / float(denom)
        weight = float(floor) + (1.0 - float(floor)) * float(math.pow(float(t), float(power)))
        out[int(epoch)] = float(weight)
    first_epoch = int(target_epochs[0])
    last_epoch = int(target_epochs[-1])
    out[int(first_epoch)] = float(floor)
    out[int(last_epoch)] = 1.0
    return out


def _base_feature_vector_for_round(
    *,
    round_t: Round,
    prior_context_rounds: Sequence[Round],
    klines_store_like: object,
    cutoff_seconds: int,
    feature_cache_store: object | None,
) -> list[float]:
    if round_t.lock_at is None:
        raise InvariantError("direct_action_round_lock_at_missing")
    prior_last_epoch = int(prior_context_rounds[-1].epoch) if prior_context_rounds else 0
    anchor_close_time_ms = _anchor_close_time_ms(
        round_t=round_t,
        klines_store_like=klines_store_like,
        cutoff_seconds=int(cutoff_seconds),
    )
    lock_at = int(round_t.lock_at)

    if feature_cache_store is not None:
        if not hasattr(feature_cache_store, "get_vector") or not hasattr(feature_cache_store, "put_vector"):
            raise InvariantError("direct_action_feature_cache_store_invalid")
        cached = feature_cache_store.get_vector(
            epoch=int(round_t.epoch),
            cutoff_seconds=int(cutoff_seconds),
            schema_name=str(FEATURE_SCHEMA.name),
            start_at=int(round_t.start_at),
            lock_at=int(lock_at),
            prior_last_epoch=int(prior_last_epoch),
            anchor_close_time_ms=int(anchor_close_time_ms),
        )
        if cached is not None:
            return list(cached)

    context_klines = _context_klines_for_round(
        round_t=round_t,
        klines_store_like=klines_store_like,
        cutoff_seconds=int(cutoff_seconds),
    )
    if not context_klines:
        raise InvariantError("direct_action_context_klines_empty")
    feats = build_features(
        target_round=round_t,
        prior_context_rounds=list(prior_context_rounds),
        context_klines=list(context_klines),
        cutoff_seconds=int(cutoff_seconds),
    )
    x_row = vectorize(features=feats, schema=FEATURE_SCHEMA)
    if feature_cache_store is not None:
        feature_cache_store.put_vector(
            epoch=int(round_t.epoch),
            cutoff_seconds=int(cutoff_seconds),
            schema_name=str(FEATURE_SCHEMA.name),
            start_at=int(round_t.start_at),
            lock_at=int(lock_at),
            prior_last_epoch=int(prior_last_epoch),
            anchor_close_time_ms=int(anchor_close_time_ms),
            vector=list(x_row),
        )
    return list(x_row)


def _context_klines_for_round(
    *,
    round_t: Round,
    klines_store_like: object,
    cutoff_seconds: int,
) -> list[Kline]:
    if not hasattr(klines_store_like, "get_context_klines") or not hasattr(klines_store_like, "latest_close_time_ms"):
        raise InvariantError("direct_action_klines_store_invalid")
    kk = int(max_required_context_klines_size())
    anchor_ms = _anchor_close_time_ms(
        round_t=round_t,
        klines_store_like=klines_store_like,
        cutoff_seconds=int(cutoff_seconds),
    )
    return list(
        klines_store_like.get_context_klines(
            anchor_close_time_ms=int(anchor_ms),
            size=int(kk),
        )
    )


def _anchor_close_time_ms(
    *,
    round_t: Round,
    klines_store_like: object,
    cutoff_seconds: int,
) -> int:
    if round_t.lock_at is None:
        raise InvariantError("direct_action_round_lock_at_missing")
    cutoff_ts = int(round_t.lock_at) - int(cutoff_seconds)
    anchor_ms = int(cutoff_ts) * 1000
    latest_close_ms = klines_store_like.latest_close_time_ms()
    if latest_close_ms is None:
        raise InvariantError("direct_action_klines_store_empty")
    if int(latest_close_ms) < int(anchor_ms):
        anchor_ms = int(latest_close_ms)
    return int(anchor_ms)


def _direct_action_feature_row_values(
    *,
    round_t: Round,
    summary_rounds: Sequence[Round],
    base_vector: Sequence[float],
    action_spec: DirectActionSpec,
    cutoff_seconds: int,
    treasury_fee_fraction: float,
) -> list[float]:
    if len(base_vector) != int(len(FEATURE_SCHEMA.columns)):
        raise InvariantError("direct_action_base_vector_len_mismatch")
    cutoff_snapshot = _cutoff_pool_snapshot_bnb(round_t=round_t, cutoff_seconds=int(cutoff_seconds))
    values = list(float(v) for v in base_vector)
    values.extend(
        [
            1.0 if str(action_spec.action) == "SKIP" else 0.0,
            1.0 if str(action_spec.bet_side or "") == "Bull" else 0.0,
            1.0 if str(action_spec.bet_side or "") == "Bear" else 0.0,
            float(action_spec.bet_size_bnb),
            float(math.log1p(float(action_spec.bet_size_bnb))),
            _action_pool_share_total(action_spec=action_spec, cutoff_snapshot=cutoff_snapshot),
            _action_pool_share_side(action_spec=action_spec, cutoff_snapshot=cutoff_snapshot),
        ]
    )
    for horizon in _DIRECT_ACTION_SUMMARY_HORIZONS:
        mean_net, positive_rate, std_net = _rolling_action_summary(
            prior_rounds=summary_rounds,
            action_spec=action_spec,
            horizon=int(horizon),
            treasury_fee_fraction=float(treasury_fee_fraction),
        )
        values.extend(
            [
                float(mean_net),
                float(positive_rate),
                float(std_net),
            ]
        )
    return values


def _cutoff_pool_snapshot_bnb(*, round_t: Round, cutoff_seconds: int) -> tuple[float, float, float]:
    if round_t.lock_at is None:
        raise InvariantError("direct_action_round_lock_at_missing")
    cutoff_ts = int(round_t.lock_at) - int(cutoff_seconds)
    pools = compute_pool_amounts_wei_at_or_before(
        bets=round_t.bets,
        cutoff_ts=int(cutoff_ts),
    )
    bull_pool_bnb = float(pools.bull_wei) / float(BNB_WEI)
    bear_pool_bnb = float(pools.bear_wei) / float(BNB_WEI)
    total_pool_bnb = float(pools.total_wei) / float(BNB_WEI)
    return float(total_pool_bnb), float(bull_pool_bnb), float(bear_pool_bnb)


def _action_pool_share_total(
    *,
    action_spec: DirectActionSpec,
    cutoff_snapshot: tuple[float, float, float],
) -> float:
    total_pool_bnb, _bull_pool_bnb, _bear_pool_bnb = cutoff_snapshot
    if float(total_pool_bnb) <= 0.0:
        return 0.0
    return float(action_spec.bet_size_bnb) / float(total_pool_bnb)


def _action_pool_share_side(
    *,
    action_spec: DirectActionSpec,
    cutoff_snapshot: tuple[float, float, float],
) -> float:
    _total_pool_bnb, bull_pool_bnb, bear_pool_bnb = cutoff_snapshot
    if str(action_spec.bet_side or "") == "Bull":
        if float(bull_pool_bnb) <= 0.0:
            return 0.0
        return float(action_spec.bet_size_bnb) / float(bull_pool_bnb)
    if str(action_spec.bet_side or "") == "Bear":
        if float(bear_pool_bnb) <= 0.0:
            return 0.0
        return float(action_spec.bet_size_bnb) / float(bear_pool_bnb)
    return 0.0


def _rolling_action_summary(
    *,
    prior_rounds: Sequence[Round],
    action_spec: DirectActionSpec,
    horizon: int,
    treasury_fee_fraction: float,
) -> tuple[float, float, float]:
    if int(horizon) <= 0:
        raise InvariantError("direct_action_summary_horizon_nonpositive")
    if str(action_spec.action) == "SKIP":
        return 0.0, 0.0, 0.0
    tail = list(prior_rounds[-int(horizon) :])
    if not tail:
        return 0.0, 0.0, 0.0
    profits = [
        float(
            realized_net_bnb_for_action(
                action_spec=action_spec,
                round_closed=round_t,
                treasury_fee_fraction=float(treasury_fee_fraction),
            )
        )
        for round_t in tail
    ]
    mean_net = float(sum(float(value) for value in profits) / float(len(profits)))
    positive_rate = float(sum(1 for value in profits if float(value) > 0.0) / float(len(profits)))
    if len(profits) == 1:
        std_net = 0.0
    else:
        variance = float(
            sum((float(value) - float(mean_net)) ** 2 for value in profits) / float(len(profits))
        )
        std_net = math.sqrt(float(variance))
    return float(mean_net), float(positive_rate), float(std_net)


def summarize_top_action_predictions(
    *,
    action_specs: Sequence[DirectActionSpec],
    q10_values: Sequence[float],
    q50_values: Sequence[float],
    top_k: int,
) -> str:
    """Return a compact JSON summary for logging/audit."""

    rows: list[dict[str, object]] = []
    for spec, q10_value, q50_value in zip(action_specs, q10_values, q50_values):
        rows.append(
            {
                "action_id": str(spec.action_id),
                "label": str(spec.label),
                "q10_net_bnb": float(q10_value),
                "q50_net_bnb": float(q50_value),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["q10_net_bnb"]),
            -float(row["q50_net_bnb"]),
            str(row["action_id"]),
        )
    )
    return json.dumps(rows[: max(0, int(top_k))], separators=(",", ":"), sort_keys=True)
