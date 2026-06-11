"""StrategyPipeline: the protocol every strategy pipeline must satisfy.

This is the seam for adding a second strategy beside
``MomentumOnlyPipeline``. The runtime engine (``runtime/engine.py``), the
dry/live pipeline factory (``runtime/dry.py``) and the backtest runner
(``backtest/runner.py``) all drive their pipeline through exactly this
surface. The universal decision schema (``StrategyPipelineDecision`` and
its skip_context validation rules) lives here too — it is
strategy-invariant.

Beyond the required members, the engine probes two OPTIONAL private
attributes and degrades gracefully when they are absent (via
``getattr(..., None)`` / ``hasattr``):

- ``_gate``: when present (a ``MomentumGate``), the engine calls
  ``_gate.warmup_okx_session()`` at the OKX-warmup wake and reads the
  gate's per-symbol fetch timings into the cycle-audit columns. A strategy
  without ``_gate`` loses session warmup + fetch-timing telemetry, silently.
- ``_bankroll_tracker``: read for risk-state audit columns and the
  stale-bankroll fallback display.

A new strategy that wants those behaviors must expose the same attribute
names (or extend this protocol + the engine's probes together).

Backtest-pool note for new pipeline authors: cutoff-pool reconstruction
from a round's bet list (the ``block_ts < lock - pool_cutoff`` boundary
semantics) currently lives in ``momentum_pipeline._pools_from_bets`` —
reuse it rather than re-deriving the boundary rules.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pancakebot.bankroll_tracker import BankrollTracker
from pancakebot.types import Round
from pancakebot.util import InvariantError


# Mapping from skip_reason → keys that MUST be present in skip_context.
# Construction of a SKIP decision with one of these reasons but missing
# (or empty) skip_context raises InvariantError. The engine consumer
# trusts the data — no isinstance / None-fallback guards downstream.
_SKIP_CONTEXT_SCHEMA: Mapping[str, frozenset[str]] = {
    "risk_drawdown_breaker_fired": frozenset({"drawdown_pct", "threshold_pct"}),
    "risk_cooldown_active": frozenset({"rounds_remaining"}),
    "pool_below_minimum": frozenset({"pool_bnb", "min_pool_bnb_at_cutoff"}),
}


@dataclass(frozen=True, slots=True)
class StrategyPipelineDecision:
    """Normalized open-round strategy pipeline decision (every pipeline
    returns this type from ``decide_open_round``).

    ``skip_context`` carries per-reason structured payload that the engine
    consumes to compose operator-facing SKIP narratives (added 2026-05-18
    Phase B v2 T3-B). For SKIP decisions whose ``skip_reason`` is in
    ``_SKIP_CONTEXT_SCHEMA``, ``skip_context`` is REQUIRED and validated
    in ``__post_init__`` — the engine consumer reads keys directly and
    will raise loudly on any drift. For BET decisions and SKIPs whose
    wording doesn't need extra numbers (e.g. ``gate_no_signal``), it's
    None.

    Required-context reasons (and their keys):
      - ``risk_drawdown_breaker_fired`` → {"drawdown_pct", "threshold_pct"}
      - ``risk_cooldown_active`` → {"rounds_remaining"}
      - ``pool_below_minimum`` → {"pool_bnb", "min_pool_bnb_at_cutoff"}
    """

    action: str
    bet_side: str | None
    bet_size_bnb: float
    skip_reason: str | None
    skip_context: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        required_keys = _SKIP_CONTEXT_SCHEMA.get(self.skip_reason or "")
        if required_keys is None:
            return
        if self.skip_context is None:
            raise InvariantError(
                f"skip_reason={self.skip_reason!r} requires skip_context "
                f"with keys {sorted(required_keys)}; got None"
            )
        missing = required_keys - set(self.skip_context.keys())
        if missing:
            raise InvariantError(
                f"skip_reason={self.skip_reason!r} skip_context missing keys: "
                f"{sorted(missing)}"
            )


@runtime_checkable
class StrategyPipeline(Protocol):
    """Duck-typed pipeline contract consumed by engine.py + backtest/runner.py."""

    @property
    def last_settled_epoch(self) -> int | None: ...

    @property
    def router_mode(self) -> str:
        """Stable per-pipeline label recorded verbatim into the cycle_audit
        ``router_mode`` column. Each pipeline returns its OWN label
        (``MomentumOnlyPipeline`` returns ``"momentum_gate"``); "frozen"
        applies to the CSV column name, not the value."""
        ...

    # Backtest kline loading (live/dry pipelines fetch via their gate
    # instead; the BNB hook is retained for replay compatibility — research
    # drivers call it with an empty dict).
    def refresh_bnb_klines(self, *, bnb_klines_by_epoch: dict[int, list[list]]) -> None: ...
    def refresh_btc_klines(self, *, btc_klines_by_epoch: dict[int, list[list]]) -> None: ...
    def refresh_eth_klines(self, *, eth_klines_by_epoch: dict[int, list[list]]) -> None: ...
    def refresh_sol_klines(self, *, sol_klines_by_epoch: dict[int, list[list]]) -> None: ...

    # Settlement bookkeeping.
    def settle_closed_rounds(self, *, rounds: list[Round]) -> None: ...
    def bootstrap_from_closed_rounds(self, *, rounds: list[Round]) -> None: ...
    def record_settlement(self, *, bankroll: float, start_at: int) -> None: ...
    def set_bankroll_tracker(self, tracker: BankrollTracker | None) -> None: ...

    # The decision entry point, called once per open round.
    def decide_open_round(
        self,
        *,
        round_t: Round,
        pool_bull_bnb: float = 0.0,
        pool_bear_bnb: float = 0.0,
    ) -> StrategyPipelineDecision: ...
