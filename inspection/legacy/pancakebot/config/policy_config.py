from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Canonical v1.0 policy config.

    This section is intentionally small and strictly validated.
    Unknown keys must be rejected by the config loader.
    """

    kelly_multiplier: float = 0.5

    bankroll_cap_fraction: float = 0.10
    pool_cap_fraction: float = 0.10

    max_bet_bnb: float = 0.25

