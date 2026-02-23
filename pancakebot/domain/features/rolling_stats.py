"""Shared rolling/tail-window statistics helpers for feature computation.

These helpers remove duplicated logic across:
- Group H rolling cutoff regime features
- Final Pool Model extension features

Locked conventions enforced here:
- NaN is the marker for undefined numeric values.
- Means require >= 1 defined value.
- Sample std requires >= 2 defined values.
- No smoothing; undefined cases stay NaN.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, TypeVar

from pancakebot.domain.features._math import NAN, is_nan

T = TypeVar("T")


def tail(seq: Sequence[T], n: int) -> list[T]:
    """Return the last n elements of seq as a new list."""
    if n <= 0:
        return []
    if len(seq) <= n:
        return list(seq)
    return list(seq[-n:])


def clean_floats(xs: Iterable[float]) -> list[float]:
    """Return a list of finite floats with NaNs removed."""
    out: list[float] = []
    for x in xs:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if is_nan(xf):
            continue
        out.append(xf)
    return out


def mean_defined(xs: Iterable[float]) -> tuple[float, int]:
    """Mean over defined values.

    Returns (mean, defined_mask) where defined_mask is 1 iff at least one value
    is defined. Undefined mean is NAN.
    """
    vals = clean_floats(xs)
    if not vals:
        return NAN, 0
    return float(sum(vals) / len(vals)), 1


def sample_std_defined(xs: Iterable[float]) -> tuple[float, int]:
    """Sample std over defined values.

    Returns (std, defined_mask) where defined_mask is 1 iff at least two values
    are defined. Undefined std is NAN.
    """
    vals = clean_floats(xs)
    if len(vals) < 2:
        return NAN, 0
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return float(math.sqrt(var)), 1
