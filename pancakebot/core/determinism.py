from __future__ import annotations

import random

import numpy as np


def set_global_determinism(*, seed: int) -> None:
    """Set global RNG seeds for deterministic behavior.

    This is best-effort determinism for model training and backtests.
    """
    random.seed(seed)
    np.random.seed(seed)
