"""Feature building (canonical v8).

This module is the single owner of:
- reading the feature schema
- computing max_required_prior_context_rounds_size() and max_required_context_klines_size()
- enforcing context sizing + ordering invariants
- executing feature computations and returning a schema-ordered feature vector

Hard invariants
- max_required_* sizes are computed ONLY from explicit schema metadata.
- len(prior_context_rounds) == max_required_prior_context_rounds_size() (>= is forbidden).
- len(context_klines) == max_required_context_klines_size() (>= is forbidden).
- Feature computation is pure with respect to the provided contexts.
"""

from __future__ import annotations

import math
from typing import Sequence

from pancakebot.domain.types import Kline, Round
from pancakebot.domain.features.flow_features import compute_late_phase_features, compute_within_round_features
from pancakebot.domain.features.price_klines_features import compute_price_klines_features
from pancakebot.domain.features.schema import (
    FEATURE_SCHEMA,
    FeatureSchema,
    max_required_context_klines_size,
    max_required_prior_context_rounds_size,
)
from pancakebot.core.errors import InvariantError


def _require_strict_epoch_order(rounds: Sequence[Round]) -> None:
    prev = None
    for r in rounds:
        e = int(r.epoch)
        if prev is not None and e <= prev:
            raise InvariantError("prior_context_rounds_not_strictly_increasing")
        prev = e


def _require_strict_kline_order(klines: Sequence[Kline]) -> None:
    prev = None
    for k in klines:
        t = int(k.open_time_ms)
        if prev is not None and t <= prev:
            raise InvariantError("context_klines_not_strictly_increasing")
        prev = t


def _compute_regime_features(*, prior_context_rounds: Sequence[Round]) -> dict[str, float]:
    """Compute canonical regime features (v8).

    Outcome-eligible window:
      W := prior_context_rounds[:-1]
    House outcomes are ignored.
    """
    out: dict[str, float] = {}

    w = list(prior_context_rounds[:-1])

    def _outcomes_last_n(n: int) -> list[str]:
        if n <= 0:
            return []
        tail = w[-n:] if len(w) >= n else []
        return [str(r.position) for r in tail if r.position is not None]

    def _bull_bear_only(seq: list[str]) -> list[str]:
        return [s for s in seq if s in ("Bull", "Bear")]

    def _frac(seq: list[str], side: str) -> float:
        bb = _bull_bear_only(seq)
        denom = len(bb)
        if denom == 0:
            return 0.0
        num = sum(1 for s in bb if s == side)
        return float(num) / float(denom)

    def _flip_rate(seq: list[str]) -> float:
        bb = _bull_bear_only(seq)
        if len(bb) < 2:
            return 0.0
        flips = 0
        denom = 0
        for a, b in zip(bb[:-1], bb[1:]):
            denom += 1
            if a != b:
                flips += 1
        if denom == 0:
            return 0.0
        return float(flips) / float(denom)

    # r_20
    seq20 = _outcomes_last_n(20)
    out["regime_bull_frac_r_20"] = _frac(seq20, "Bull")
    out["regime_bear_frac_r_20"] = _frac(seq20, "Bear")
    out["regime_flip_rate_r_20"] = _flip_rate(seq20)

    # r_60
    seq60 = _outcomes_last_n(60)
    out["regime_bull_frac_r_60"] = _frac(seq60, "Bull")
    out["regime_bear_frac_r_60"] = _frac(seq60, "Bear")
    out["regime_flip_rate_r_60"] = _flip_rate(seq60)

    # streak_len (requires r_20 window by spec; computed on W, ignoring House)
    bb_rev = [s for s in reversed([str(r.position) for r in w if r.position is not None]) if s in ("Bull", "Bear")]
    if not bb_rev:
        out["regime_streak_len"] = 0.0
    else:
        direction = bb_rev[0]
        streak = 0
        for s in bb_rev:
            if s == direction:
                streak += 1
            else:
                break
        out["regime_streak_len"] = float(streak)

    return out


def validate_features_against_schema(*, features: dict[str, float], schema: FeatureSchema) -> None:
    """Validate that a feature dict matches the schema column set exactly."""
    cols = set(schema.columns)
    keys = set(features.keys())
    missing = sorted(cols - keys)
    extra = sorted(keys - cols)
    if missing:
        raise InvariantError(f"features_missing_columns: {missing[:10]}")
    if extra:
        raise InvariantError(f"features_extra_columns: {extra[:10]}")


def vectorize(*, features: dict[str, float], schema: FeatureSchema) -> list[float]:
    """Vectorize features in schema column order."""
    out: list[float] = []
    for col in schema.columns:
        out.append(float(features[col]))
    return out


def build_features(
    *,
    target_round: Round,
    prior_context_rounds: Sequence[Round],
    context_klines: Sequence[Kline],
    cutoff_seconds: int,
) -> dict[str, float]:
    """Build a full FEATURE_SCHEMA feature row for the target round."""

    k = int(max_required_prior_context_rounds_size())
    if len(prior_context_rounds) != k:
        raise InvariantError(f"prior_context_rounds_size_mismatch: got={len(prior_context_rounds)} expected={k}")
    _require_strict_epoch_order(prior_context_rounds)

    kk = int(max_required_context_klines_size())
    if len(context_klines) != kk:
        raise InvariantError(f"context_klines_size_mismatch: got={len(context_klines)} expected={kk}")
    _require_strict_kline_order(context_klines)

    if target_round.lock_at is None:
        raise InvariantError("target_round_lock_at_missing")
    lock_ts = int(target_round.lock_at)
    if int(lock_ts) <= 0:
        raise InvariantError("target_round_lock_at_invalid")
    if int(target_round.start_at) <= 0:
        raise InvariantError("target_round_start_at_missing")

    feats: dict[str, float] = {}

    # Within-round features (target-only; windows are percent of pre-cutoff interval).
    feats.update(
        compute_within_round_features(
            bets=target_round.bets,
            start_ts=int(target_round.start_at),
            lock_ts=int(lock_ts),
            cutoff_seconds=int(cutoff_seconds),
        )
    )

    # Late-phase features (lagged; derived from prior_context_rounds[-1]).
    prior_last = prior_context_rounds[-1]
    feats.update(
        compute_late_phase_features(
            bets=prior_last.bets,
            lock_ts=prior_last.lock_at,
            cutoff_seconds=int(cutoff_seconds),
        )
    )

    # External price (klines) features.
    feats.update(compute_price_klines_features(context_klines=list(context_klines)))

    # Regime features (outcome-dependent).
    feats.update(_compute_regime_features(prior_context_rounds=prior_context_rounds))

    # Fill missing schema columns with NaN (feature unavailability).
    full: dict[str, float] = {}
    for col in FEATURE_SCHEMA.columns:
        if col in feats:
            v = float(feats[col])
        else:
            v = float("nan")
        if math.isinf(v):
            raise InvariantError(f"feature_inf_developer_error: {col}")
        full[col] = v

    validate_features_against_schema(features=full, schema=FEATURE_SCHEMA)
    return full

