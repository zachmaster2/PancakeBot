"""Shared strategy pipeline for live, dry, and backtest modes.

The pipeline combines candidate providers (dislocation + optional ML adapter)
with the shared strategy router.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.strategy.dislocation_engine import DislocationEngine
from pancakebot.domain.strategy.ml_candidate_adapter import MlCandidateAdapter
from pancakebot.domain.strategy.router import StrategyRouter, StrategyRouterDecision
from pancakebot.domain.types import Kline, Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round

_ML_UNTRADEABLE_SKIP_REASONS = frozenset(
    {
        "predictability_below_min",
        "p_bull_edge_below_min",
        "expected_net_below_min",
    }
)
_VALID_BET_SIDES = frozenset({"Bull", "Bear"})


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

    def export_kline_index_state(self) -> dict[str, object]:
        """Export dislocation kline index state for backtest caching."""

        return self._dislocation_engine.export_kline_index_state()

    def import_kline_index_state(self, *, state: dict[str, object]) -> None:
        """Restore dislocation kline index state from backtest cache."""

        self._dislocation_engine.import_kline_index_state(state=state)

    def export_bootstrap_state(self) -> dict[str, object]:
        """Export full pipeline warmup state for backtest snapshot caching."""

        return {
            "last_settled_epoch": self._last_settled_epoch,
            "dislocation_engine_state": self._dislocation_engine.export_bootstrap_state(),
            "router_state": self._router.export_bootstrap_state(),
            "ml_state": (
                self._ml_candidate_adapter.export_bootstrap_state()
                if self._ml_candidate_adapter is not None
                else None
            ),
        }

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        """Restore full pipeline warmup state from backtest snapshot cache."""

        dislocation_state = state.get("dislocation_engine_state")
        if not isinstance(dislocation_state, dict):
            raise InvariantError("pipeline_snapshot_dislocation_state_missing")
        self._dislocation_engine.import_bootstrap_state(state=dislocation_state)

        router_state = state.get("router_state")
        if not isinstance(router_state, dict):
            raise InvariantError("pipeline_snapshot_router_state_missing")
        self._router.import_bootstrap_state(state=router_state)

        ml_state = state.get("ml_state")
        if ml_state is not None:
            if self._ml_candidate_adapter is None:
                raise InvariantError("pipeline_snapshot_ml_state_without_adapter")
            if not isinstance(ml_state, dict):
                raise InvariantError("pipeline_snapshot_ml_state_invalid")
            self._ml_candidate_adapter.import_bootstrap_state(state=ml_state)

        last_settled_epoch = state.get("last_settled_epoch")
        if last_settled_epoch is None:
            self._last_settled_epoch = None
        else:
            self._last_settled_epoch = int(last_settled_epoch)
        self._pending_candidate_signals_by_epoch = {}

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
            signals = self._apply_ml_signal_coupling(
                round_t=round_t,
                candidate_signals=signals,
                ml_signal=ml_signal,
            )
            if bool(self._ml_candidate_adapter.emit_candidate):
                ml_name = str(ml_signal.candidate_name)
                if ml_name in signals:
                    raise InvariantError("strategy_pipeline_candidate_name_duplicate")
                signals[ml_name] = ml_signal
        return dict(signals)

    def _apply_ml_signal_coupling(
        self,
        *,
        round_t: Round,
        candidate_signals: dict[str, StrategyCandidateSignal],
        ml_signal: StrategyCandidateSignal,
    ) -> dict[str, StrategyCandidateSignal]:
        if self._ml_candidate_adapter is None:
            return dict(candidate_signals)

        out = dict(candidate_signals)
        if (
            bool(self._ml_candidate_adapter.veto_untradeable_candidates)
            and str(ml_signal.action) == "SKIP"
            and str(ml_signal.skip_reason or "") in _ML_UNTRADEABLE_SKIP_REASONS
        ):
            return {
                str(name): self._veto_candidate_signal(
                    signal=signal,
                    skip_reason="ml_veto_untradeable",
                )
                for name, signal in out.items()
            }

        if (
            bool(self._ml_candidate_adapter.veto_opposite_side_candidates)
            and str(ml_signal.action) == "BET"
            and str(ml_signal.bet_side or "") in _VALID_BET_SIDES
        ):
            ml_side = str(ml_signal.bet_side)
            out = {
                str(name): (
                    self._veto_candidate_signal(
                        signal=signal,
                        skip_reason="ml_veto_opposite_side",
                    )
                    if str(signal.action) == "BET"
                    and str(signal.bet_side or "") in _VALID_BET_SIDES
                    and str(signal.bet_side) != str(ml_side)
                    else signal
                )
                for name, signal in out.items()
            }

        if bool(self._ml_candidate_adapter.veto_candidate_expected_net_below_min):
            out = {
                str(name): self._veto_candidate_signal(
                    signal=signal,
                    skip_reason=str(skip_reason),
                )
                if (
                    (
                        skip_reason := self._ml_candidate_adapter.candidate_veto_skip_reason_for_open_round(
                            round_t=round_t,
                            candidate_signal=signal,
                        )
                    )
                    is not None
                )
                else signal
                for name, signal in out.items()
            }
        if bool(self._ml_candidate_adapter.rescore_baseline_candidates_with_expected_net):
            out = {
                str(name): (
                    self._rescore_candidate_signal(
                        signal=signal,
                        expected_profit_bnb=float(expected_net),
                    )
                    if expected_net is not None
                    else signal
                )
                for name, signal in out.items()
                for expected_net in [
                    self._ml_candidate_adapter.candidate_expected_net_for_open_round(
                        round_t=round_t,
                        candidate_signal=signal,
                    )
                ]
            }
        return out

    @staticmethod
    def _veto_candidate_signal(
        *,
        signal: StrategyCandidateSignal,
        skip_reason: str,
    ) -> StrategyCandidateSignal:
        if str(signal.action) != "BET":
            return signal
        return replace(
            signal,
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
            expected_profit_bnb=None,
            selector_score_bnb=None,
            skip_reason=str(skip_reason),
        )

    @staticmethod
    def _rescore_candidate_signal(
        *,
        signal: StrategyCandidateSignal,
        expected_profit_bnb: float,
    ) -> StrategyCandidateSignal:
        if str(signal.action) != "BET":
            return signal
        return replace(
            signal,
            expected_profit_bnb=float(expected_profit_bnb),
            selector_score_bnb=float(expected_profit_bnb),
        )

    def _settle_providers(self, *, rounds: list[Round]) -> None:
        self._dislocation_engine.settle_closed_rounds(list(rounds))
        if self._ml_candidate_adapter is not None:
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
