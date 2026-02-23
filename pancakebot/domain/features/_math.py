"""Shared math helpers for feature computation.

This module exists only to remove duplicated helper functions across feature
files. It must not introduce any smoothing or behavior changes.

Locked conventions:
- NaN is the marker for undefined numeric feature values.
- Presence is encoded via explicit mask features.
"""

from __future__ import annotations

import math
from typing import Any

NAN: float = float("nan")


def nan() -> float:
    """Return the NaN marker."""
    return NAN


def is_nan(x: float) -> bool:
    """True iff x is NaN."""
    return math.isnan(x)


def safe_is_nan(x: Any) -> bool:
    """Best-effort NaN check.

    This mirrors prior inline helpers that treated non-floatable values as NaN.
    """

    try:
        return is_nan(float(x))
    except (TypeError, ValueError):
        return True
