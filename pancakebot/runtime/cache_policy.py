from __future__ import annotations

from pancakebot.domain.features.feature_builder import max_required_prior_context_rounds_size


def compute_required_cache_size(
    *,
    train_size: int,
    calibrate_size: int,
) -> int:
    """Return the required closed-round cache size.

    The cache stores CLOSED rounds only.

    Canonical sizing:
      required_total = max_required_prior_context_rounds_size() + train_size + calibrate_size

    """

    k = int(max_required_prior_context_rounds_size())
    return int(k) + int(train_size) + int(calibrate_size)

