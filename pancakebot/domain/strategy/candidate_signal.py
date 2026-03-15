"""Shared strategy-candidate signal contract.

This module defines the per-round candidate signal shape used by strategy
routers. The contract is strategy-family-agnostic so dislocation, ML, and
future candidates can be routed with one selector path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyCandidateSignal:
    """One candidate's actionable signal for a single round."""

    # Candidate identity.
    candidate_name: str

    # Candidate action proposal.
    action: str
    bet_side: str | None
    bet_size_bnb: float

    # Candidate economics and routing metadata.
    expected_profit_bnb: float | None
    selector_score_bnb: float | None

    # Candidate diagnostics.
    skip_reason: str | None
    p_bull: float | None
    dislocation_bull: float | None
    projected_late_ratio: float | None = None
    projected_late_imbalance: float | None = None
