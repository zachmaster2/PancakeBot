"""Shared strategy pipeline for live, dry, and backtest modes.

The pipeline combines candidate providers (dislocation + optional ML adapter)
with the shared strategy router.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.strategy.dislocation_engine import DislocationEngine
from pancakebot.domain.strategy.ml_candidate_adapter import MlCandidateAdapter
from pancakebot.domain.strategy.router import StrategyRouter, StrategyRouterDecision
from pancakebot.domain.types import Kline, Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round


@dataclass(frozen=True, slots=True)
class StrategyPipelineDecision:
    """Normalized open-round strategy pipeline decision."""

    action: str
    selected_strategy: str | None
    bet_side: str | None
    bet_size_bnb: float
    expected_profit_bnb: float
    selector_score_bnb: float | None
    skip_reason: str | None
    p_bull: float | None


class StrategyPipeline:
    """Run candidate generation + routing as one shared pipeline."""

    def __init__(
        self,
        *,
        dislocation_engine: DislocationEngine,
        router: StrategyRouter,
        treasury_fee_fraction: float,
        ml_candidate_adapter: MlCandidateAdapter | None = None,
    ) -> None:
        if not (0.0 <= float(treasury_fee_fraction) < 1.0):
            raise InvariantError("strategy_pipeline_treasury_fee_fraction_out_of_range")
        self._dislocation_engine = dislocation_engine
        self._router = router
        self._treasury_fee_fraction = float(treasury_fee_fraction)
        self._ml_candidate_adapter = ml_candidate_adapter
        self._pending_candidate_signals_by_epoch: dict[int, dict[str, StrategyCandidateSignal]] = {}
        self._last_settled_epoch: int | None = None

    @property
    def router_mode(self) -> str:
        """Return current router mode."""

        return str(self._router.mode)

    def refresh_klines(self, *, klines: list[Kline]) -> None:
        """Refresh candidate providers with the latest kline context."""

        self._dislocation_engine.refresh_klines(list(klines))
        if self._ml_candidate_adapter is not None:
            self._ml_candidate_adapter.refresh_klines(klines=list(klines))

    def bootstrap_from_closed_rounds(self, *, rounds: list[Round]) -> None:
        """Bootstrap providers and router from historical closed rounds."""

        if not rounds:
            return
        for round_t in sorted(rounds, key=lambda x: int(x.epoch)):
            epoch = int(round_t.epoch)
            if self._last_settled_epoch is not None and int(epoch) <= int(self._last_settled_epoch):
                continue
            signals = self.candidate_signals_for_open_round(round_t=round_t)
            realized = self._realized_profit_by_candidate(
                candidate_signals=signals,
                round_closed=round_t,
            )
            self._router.observe_settlement(
                candidate_signals=signals,
                realized_profit_by_candidate=realized,
            )
            self._settle_providers(rounds=[round_t])
            self._pending_candidate_signals_by_epoch.pop(int(epoch), None)
            self._last_settled_epoch = int(epoch)

    def candidate_signals_for_open_round(self, *, round_t: Round) -> dict[str, StrategyCandidateSignal]:
        """Collect candidate signals for one target round."""

        signals = self._collect_candidate_signals(round_t=round_t)
        self._pending_candidate_signals_by_epoch[int(round_t.epoch)] = signals
        return dict(signals)

    def decide_open_round(
        self,
        *,
        round_t: Round,
        bankroll_bnb: float,
        allow_oracle_mode: bool,
    ) -> StrategyPipelineDecision:
        """Generate and route candidate signals for one open round."""

        signals = self.candidate_signals_for_open_round(round_t=round_t)
        realized: dict[str, float] | None = None
        if str(self._router.mode) == "oracle_skip":
            if not bool(allow_oracle_mode):
                raise InvariantError("oracle_router_mode_not_supported_live")
            realized = self._realized_profit_by_candidate(
                candidate_signals=signals,
                round_closed=round_t,
            )

        routed = self._router.route_round(
            candidate_signals=signals,
            bankroll_bnb=float(bankroll_bnb),
            bet_gas_cost_bnb=float(GAS_COST_BET_BNB),
            selector_ready=bool(self._dislocation_engine.selector_ready()),
            realized_profit_by_candidate=realized,
        )
        return self._to_pipeline_decision(routed=routed)

    def settle_closed_rounds(self, *, rounds: list[Round]) -> None:
        """Advance provider/router state from newly closed rounds."""

        if not rounds:
            return
        for round_t in sorted(rounds, key=lambda x: int(x.epoch)):
            epoch = int(round_t.epoch)
            if self._last_settled_epoch is not None and int(epoch) <= int(self._last_settled_epoch):
                continue
            signals = self._pending_candidate_signals_by_epoch.pop(int(epoch), None)
            if signals is None:
                signals = self._collect_candidate_signals(round_t=round_t)
                self._pending_candidate_signals_by_epoch.pop(int(epoch), None)
            realized = self._realized_profit_by_candidate(
                candidate_signals=signals,
                round_closed=round_t,
            )
            self._router.observe_settlement(
                candidate_signals=signals,
                realized_profit_by_candidate=realized,
            )
            self._settle_providers(rounds=[round_t])
            self._last_settled_epoch = int(epoch)

    def _collect_candidate_signals(self, *, round_t: Round) -> dict[str, StrategyCandidateSignal]:
        signals = self._dislocation_engine.candidate_signals_for_open_round(round_t=round_t)
        if self._ml_candidate_adapter is not None and bool(self._ml_candidate_adapter.enabled):
            ml_signal = self._ml_candidate_adapter.candidate_signal_for_open_round(round_t=round_t)
            ml_name = str(ml_signal.candidate_name)
            if ml_name in signals:
                raise InvariantError("strategy_pipeline_candidate_name_duplicate")
            signals[ml_name] = ml_signal
        return dict(signals)

    def _settle_providers(self, *, rounds: list[Round]) -> None:
        self._dislocation_engine.settle_closed_rounds(list(rounds))
        if self._ml_candidate_adapter is not None and bool(self._ml_candidate_adapter.enabled):
            self._ml_candidate_adapter.settle_closed_rounds(rounds=list(rounds))

    def _realized_profit_by_candidate(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        round_closed: Round,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for candidate_name, signal in candidate_signals.items():
            profit = 0.0
            if (
                str(signal.action) == "BET"
                and str(signal.bet_side) in ("Bull", "Bear")
                and float(signal.bet_size_bnb) > 0.0
            ):
                outcome = settle_bet_against_closed_round(
                    bet_bnb=float(signal.bet_size_bnb),
                    bet_side=str(signal.bet_side),
                    round_closed=round_closed,
                    treasury_fee_fraction=float(self._treasury_fee_fraction),
                )
                credit_bnb = float(outcome.credit_bnb)
                profit = (
                    float(credit_bnb)
                    - float(signal.bet_size_bnb)
                    - float(GAS_COST_BET_BNB)
                )
            out[str(candidate_name)] = float(profit)
        return out

    @staticmethod
    def _to_pipeline_decision(*, routed: StrategyRouterDecision) -> StrategyPipelineDecision:
        return StrategyPipelineDecision(
            action=str(routed.action),
            selected_strategy=(
                str(routed.selected_strategy) if routed.selected_strategy is not None else None
            ),
            bet_side=str(routed.bet_side) if routed.bet_side is not None else None,
            bet_size_bnb=float(routed.bet_size_bnb),
            expected_profit_bnb=float(routed.expected_profit_bnb),
            selector_score_bnb=(
                float(routed.selector_score_bnb) if routed.selector_score_bnb is not None else None
            ),
            skip_reason=str(routed.skip_reason) if routed.skip_reason is not None else None,
            p_bull=float(routed.p_bull) if routed.p_bull is not None else None,
        )
