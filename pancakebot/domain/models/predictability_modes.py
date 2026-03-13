from __future__ import annotations

from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.schema import FeatureSchema

PREDICTABILITY_FEATURE_MODE_ALL = "all_features"
PREDICTABILITY_FEATURE_MODE_REGIME_ONLY = "regime_only"
PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_ONLY = "arrival_microstructure_only"
PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_PLUS_REGIME = "arrival_microstructure_plus_regime"
PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_PLUS_REGIME_PLUS_PRICE = "arrival_microstructure_plus_regime_plus_price"

PREDICTABILITY_LABEL_MODE_BASELINE_LOG_IMBALANCE_SIDE = "baseline_log_imbalance_side"
PREDICTABILITY_LABEL_MODE_EITHER_SIDE_PROFITABLE = "either_side_profitable"

DEFAULT_PREDICTABILITY_FEATURE_MODE = PREDICTABILITY_FEATURE_MODE_ALL
DEFAULT_PREDICTABILITY_LABEL_MODE = PREDICTABILITY_LABEL_MODE_BASELINE_LOG_IMBALANCE_SIDE

ALLOWED_PREDICTABILITY_FEATURE_MODES = (
    PREDICTABILITY_FEATURE_MODE_ALL,
    PREDICTABILITY_FEATURE_MODE_REGIME_ONLY,
    PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_ONLY,
    PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_PLUS_REGIME,
    PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_PLUS_REGIME_PLUS_PRICE,
)
ALLOWED_PREDICTABILITY_LABEL_MODES = (
    PREDICTABILITY_LABEL_MODE_BASELINE_LOG_IMBALANCE_SIDE,
    PREDICTABILITY_LABEL_MODE_EITHER_SIDE_PROFITABLE,
)

_FEATURE_GROUPS_BY_MODE: dict[str, tuple[str, ...] | None] = {
    PREDICTABILITY_FEATURE_MODE_ALL: None,
    PREDICTABILITY_FEATURE_MODE_REGIME_ONLY: ("regime",),
    PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_ONLY: ("arrival_microstructure",),
    PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_PLUS_REGIME: ("arrival_microstructure", "regime"),
    PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_PLUS_REGIME_PLUS_PRICE: (
        "arrival_microstructure",
        "regime",
        "price",
    ),
}


def validate_predictability_feature_mode(mode: str) -> str:
    mode_norm = str(mode).strip()
    if mode_norm not in ALLOWED_PREDICTABILITY_FEATURE_MODES:
        raise InvariantError("predictability_feature_mode_invalid")
    return str(mode_norm)


def validate_predictability_label_mode(mode: str) -> str:
    mode_norm = str(mode).strip()
    if mode_norm not in ALLOWED_PREDICTABILITY_LABEL_MODES:
        raise InvariantError("predictability_label_mode_invalid")
    return str(mode_norm)


def predictability_feature_indices(*, schema: FeatureSchema, mode: str) -> tuple[int, ...]:
    mode_norm = validate_predictability_feature_mode(str(mode))
    groups = _FEATURE_GROUPS_BY_MODE[str(mode_norm)]
    if groups is None:
        return tuple(range(len(schema.features)))

    group_set = set(groups)
    out = tuple(idx for idx, feature in enumerate(schema.features) if str(feature.group) in group_set)
    if not out:
        raise InvariantError("predictability_feature_mode_empty")
    return tuple(int(idx) for idx in out)


def predictability_feature_columns(*, schema: FeatureSchema, mode: str) -> tuple[str, ...]:
    return tuple(str(schema.columns[idx]) for idx in predictability_feature_indices(schema=schema, mode=str(mode)))
