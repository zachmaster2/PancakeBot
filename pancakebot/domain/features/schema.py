"""Feature schema (canonical source of truth).

This module defines:
- The ordered feature column list used for training/prediction.
- The explicit context requirements per enabled feature.

Hard invariants
- max_required_prior_context_rounds_size() is computed ONLY from explicit schema metadata.
- max_required_context_klines_size() is computed ONLY from explicit schema metadata.
- No parsing, inference, or heuristics based on names are permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FeatureDef:
    """A single feature definition with explicit context requirements."""

    name: str
    group: str
    required_prior_context_rounds_size: int
    required_context_klines_size: int


@dataclass(frozen=True)
class FeatureSchema:
    """An ordered schema defined as a sequence of feature definitions."""

    name: str
    features: Tuple[FeatureDef, ...]

    @property
    def columns(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @property
    def required_prior_context_rounds_size(self) -> int:
        required = 0
        for f in self.features:
            required = max(required, int(f.required_prior_context_rounds_size))
        return int(required)

    @property
    def required_context_klines_size(self) -> int:
        required = 0
        for f in self.features:
            required = max(required, int(f.required_context_klines_size))
        return int(required)


def _assert_unique_columns(schema: FeatureSchema) -> None:
    seen: set[str] = set()
    dups: set[str] = set()
    for col in schema.columns:
        if col in seen:
            dups.add(col)
        seen.add(col)
    if dups:
        raise ValueError(f"{schema.name} schema has duplicate columns: {sorted(dups)}")


# ---------------------------
# Canonical v8 enabled schema
# ---------------------------

# Canonical group names (v8 frozen vocabulary).
_BET_AMOUNTS = "bet_amounts"
_BET_COUNTS = "bet_counts"
_IMBALANCE = "imbalance"
_DYNAMICS = "dynamics"
_CONCENTRATION = "concentration"
_FLAGS = "flags"
_LATE_PHASE = "late_phase"
_REGIME = "regime"
_PRICE = "price"

# Within-round windows (target-only).
_W0_50 = "w_p_0_to_p_50"
_W50_100 = "w_p_50_to_p_100"
_W0_100 = "w_p_0_to_p_100"


def _w(name: str, window: str) -> str:
    return f"{name}_{window}"


_FEATURES: list[FeatureDef] = []

# Bet amounts + counts + flags + imbalance + concentration for each within-round window.
for w in (_W0_50, _W50_100, _W0_100):
    # amounts
    _FEATURES.extend(
        [
            FeatureDef(name=_w("bull_sum", w), group=_BET_AMOUNTS, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("bear_sum", w), group=_BET_AMOUNTS, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("total_sum", w), group=_BET_AMOUNTS, required_prior_context_rounds_size=0, required_context_klines_size=0),
        ]
    )
    # counts
    _FEATURES.extend(
        [
            FeatureDef(name=_w("bull_n", w), group=_BET_COUNTS, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("bear_n", w), group=_BET_COUNTS, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("total_n", w), group=_BET_COUNTS, required_prior_context_rounds_size=0, required_context_klines_size=0),
        ]
    )
    # flags
    _FEATURES.extend(
        [
            FeatureDef(name=_w("has_any_bets", w), group=_FLAGS, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("has_bull_bets", w), group=_FLAGS, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("has_bear_bets", w), group=_FLAGS, required_prior_context_rounds_size=0, required_context_klines_size=0),
        ]
    )
    # imbalance
    _FEATURES.append(
        FeatureDef(name=_w("log_imb", w), group=_IMBALANCE, required_prior_context_rounds_size=0, required_context_klines_size=0)
    )

    # concentration (side-specific)
    _FEATURES.extend(
        [
            FeatureDef(name=_w("max_bet_bull", w), group=_CONCENTRATION, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("max_bet_bear", w), group=_CONCENTRATION, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("hhi_bull", w), group=_CONCENTRATION, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("hhi_bear", w), group=_CONCENTRATION, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("gini_bull", w), group=_CONCENTRATION, required_prior_context_rounds_size=0, required_context_klines_size=0),
            FeatureDef(name=_w("gini_bear", w), group=_CONCENTRATION, required_prior_context_rounds_size=0, required_context_klines_size=0),
        ]
    )

# Dynamics ratios (unitless comparisons between pre-cutoff windows).
_FEATURES.extend(
    [
        FeatureDef(
            name="bull_sum_ratio_w_p_50_to_p_100_over_w_p_0_to_p_50",
            group=_DYNAMICS,
            required_prior_context_rounds_size=0,
            required_context_klines_size=0,
        ),
        FeatureDef(
            name="bear_sum_ratio_w_p_50_to_p_100_over_w_p_0_to_p_50",
            group=_DYNAMICS,
            required_prior_context_rounds_size=0,
            required_context_klines_size=0,
        ),
        FeatureDef(
            name="total_sum_ratio_w_p_50_to_p_100_over_w_p_0_to_p_50",
            group=_DYNAMICS,
            required_prior_context_rounds_size=0,
            required_context_klines_size=0,
        ),
    ]
)

# Late phase (lagged to target; derived from prior_context_rounds[-1]).
_FEATURES.extend(
    [
        FeatureDef(name="late_bull_sum", group=_LATE_PHASE, required_prior_context_rounds_size=1, required_context_klines_size=0),
        FeatureDef(name="late_bear_sum", group=_LATE_PHASE, required_prior_context_rounds_size=1, required_context_klines_size=0),
        FeatureDef(name="late_total_sum", group=_LATE_PHASE, required_prior_context_rounds_size=1, required_context_klines_size=0),
        FeatureDef(name="late_bull_n", group=_LATE_PHASE, required_prior_context_rounds_size=1, required_context_klines_size=0),
        FeatureDef(name="late_bear_n", group=_LATE_PHASE, required_prior_context_rounds_size=1, required_context_klines_size=0),
        FeatureDef(name="late_total_n", group=_LATE_PHASE, required_prior_context_rounds_size=1, required_context_klines_size=0),
        FeatureDef(name="late_log_imb", group=_LATE_PHASE, required_prior_context_rounds_size=1, required_context_klines_size=0),
    ]
)

# External price features (1m klines context). For close-to-close returns over n minutes,
# required_context_klines_size = n + 1.
for n in (15, 30, 60, 120):
    req_k = int(n) + 1
    _FEATURES.extend(
        [
            FeatureDef(
                name=f"price_log_return_mean_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_log_return_std_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_log_return_abs_mean_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_log_return_abs_max_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_range_mean_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_range_max_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_volume_mean_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_volume_std_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_volume_max_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_trade_count_mean_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_trade_count_std_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
            FeatureDef(
                name=f"price_trade_count_max_k_{n}",
                group=_PRICE,
                required_prior_context_rounds_size=0,
                required_context_klines_size=req_k,
            ),
        ]
    )

# Regime features (outcome-dependent; operate on prior_context_rounds[:-1]).
# Window token r_<n> refers to n outcome-eligible rounds, so required_prior_context_rounds_size = n + 1.
_FEATURES.extend(
    [
        FeatureDef(name="regime_bull_frac_r_20", group=_REGIME, required_prior_context_rounds_size=21, required_context_klines_size=0),
        FeatureDef(name="regime_bear_frac_r_20", group=_REGIME, required_prior_context_rounds_size=21, required_context_klines_size=0),
        FeatureDef(name="regime_flip_rate_r_20", group=_REGIME, required_prior_context_rounds_size=21, required_context_klines_size=0),
        FeatureDef(name="regime_bull_frac_r_60", group=_REGIME, required_prior_context_rounds_size=61, required_context_klines_size=0),
        FeatureDef(name="regime_bear_frac_r_60", group=_REGIME, required_prior_context_rounds_size=61, required_context_klines_size=0),
        FeatureDef(name="regime_flip_rate_r_60", group=_REGIME, required_prior_context_rounds_size=61, required_context_klines_size=0),
        FeatureDef(name="regime_streak_len", group=_REGIME, required_prior_context_rounds_size=21, required_context_klines_size=0),
    ]
)

FEATURE_SCHEMA = FeatureSchema(name="canonical_v8", features=tuple(_FEATURES))
_assert_unique_columns(FEATURE_SCHEMA)


def max_required_prior_context_rounds_size() -> int:
    return int(FEATURE_SCHEMA.required_prior_context_rounds_size)


def max_required_context_klines_size() -> int:
    return int(FEATURE_SCHEMA.required_context_klines_size)

