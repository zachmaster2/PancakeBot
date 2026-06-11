"""StrategyPipeline: the protocol every strategy pipeline must satisfy.

This is the seam for adding a second strategy beside
``MomentumOnlyPipeline``. The runtime engine (``runtime/engine.py``), the
dry/live pipeline factory (``runtime/dry.py:_build_momentum_pipeline``) and
the backtest runner (``backtest/runner.py``) all drive their pipeline
through exactly this surface — typing-only; nothing here executes.

Beyond the required members, the engine probes two OPTIONAL private
attributes with ``hasattr`` and degrades gracefully when they are absent:

- ``_gate``: when present (a ``MomentumGate``), the engine calls
  ``_gate.warmup_okx_session()`` at the OKX-warmup wake and reads the
  gate's per-symbol fetch timings into the cycle-audit columns. A strategy
  without ``_gate`` loses session warmup + fetch-timing telemetry, silently.
- ``_bankroll_tracker``: read for risk-state audit columns.

A new strategy that wants those behaviors must expose the same attribute
names (or extend this protocol + the engine's probes together).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pancakebot.bankroll_tracker import BankrollTracker
from pancakebot.strategy.momentum_pipeline import StrategyPipelineDecision
from pancakebot.types import Round


@runtime_checkable
class StrategyPipeline(Protocol):
    """Duck-typed pipeline contract consumed by engine.py + backtest/runner.py."""

    @property
    def last_settled_epoch(self) -> int | None: ...

    @property
    def router_mode(self) -> str:
        """Frozen audit-CSV schema label (cycle_audit ``router_mode`` column)."""
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
