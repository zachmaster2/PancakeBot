"""Shared strategy-router contract for candidate selection.

The router consumes per-candidate signals and produces one normalized decision:
either place one bet or skip the round.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal

_ROUTER_MODES = ("selector_max_score", "skip_only", "oracle_skip")
_VALID_BET_SIDES = ("Bull", "Bear")


@dataclass(frozen=True, slots=True)
class StrategyRouterConfig:
    """Configuration for one router instance."""

    mode: str = "selector_max_score"
    score_threshold_bnb: float = -1e9

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate router configuration values."""

        if str(self.mode) not in _ROUTER_MODES:
            raise InvariantError("router_mode_invalid")
        if not isinstance(self.score_threshold_bnb, (int, float)):
            raise InvariantError("router_score_threshold_bnb_not_number")


@dataclass(frozen=True, slots=True)
class StrategyRouterDecision:
    """One routed strategy decision for an open round."""

    action: str
    selected_strategy: str | None
    bet_side: str | None
    bet_size_bnb: float
    expected_profit_bnb: float
    selector_score_bnb: float | None
    skip_reason: str | None
    p_bull: float | None


class StrategyRouter:
    """Select one strategy candidate signal for the current round."""

    def __init__(self, *, config: StrategyRouterConfig) -> None:
        self._config = config

    @property
    def mode(self) -> str:
        """Return router mode name."""

        return str(self._config.mode)

    def route_round(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        selector_ready: bool,
        realized_profit_by_candidate: dict[str, float] | None = None,
    ) -> StrategyRouterDecision:
        """Route one round from candidate signals to one action.

        Args:
            candidate_signals: Candidate signal table keyed by candidate name.
            bankroll_bnb: Current real bankroll used for affordability gates.
            bet_gas_cost_bnb: Gas cost added to bet-size affordability checks.
            selector_ready: Selector readiness signal from the strategy engine.
            realized_profit_by_candidate: Optional hindsight per-candidate PnL.
                Required by `oracle_skip` mode.
        """

        self._validate_candidate_signals(candidate_signals)
        if float(bankroll_bnb) < 0.0:
            raise InvariantError("router_bankroll_negative")
        if float(bet_gas_cost_bnb) < 0.0:
            raise InvariantError("router_bet_gas_cost_negative")

        mode = str(self._config.mode)
        if mode == "skip_only":
            return self._skip_decision(skip_reason="router_skip_only")
        if mode == "oracle_skip":
            return self._route_oracle_skip(
                candidate_signals=candidate_signals,
                bankroll_bnb=float(bankroll_bnb),
                bet_gas_cost_bnb=float(bet_gas_cost_bnb),
                realized_profit_by_candidate=realized_profit_by_candidate,
            )
        if mode == "selector_max_score":
            return self._route_selector_max_score(
                candidate_signals=candidate_signals,
                bankroll_bnb=float(bankroll_bnb),
                bet_gas_cost_bnb=float(bet_gas_cost_bnb),
                selector_ready=bool(selector_ready),
            )
        raise InvariantError("router_mode_unreachable")

    def _route_selector_max_score(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        selector_ready: bool,
    ) -> StrategyRouterDecision:
        if not bool(selector_ready):
            return self._skip_decision(skip_reason="selector_warmup")

        best_name: str | None = None
        best_score = float("-inf")
        for candidate_name, signal in candidate_signals.items():
            if str(signal.action) != "BET":
                continue
            if signal.selector_score_bnb is None:
                continue
            side = str(signal.bet_side or "")
            if side not in _VALID_BET_SIDES:
                raise InvariantError("router_candidate_bet_side_invalid")
            if float(signal.bet_size_bnb) <= 0.0:
                raise InvariantError("router_candidate_bet_size_nonpositive")
            score = float(signal.selector_score_bnb)
            if float(score) > float(best_score):
                best_score = float(score)
                best_name = str(candidate_name)

        if best_name is None:
            return self._skip_decision(skip_reason="selector_no_candidate")

        if float(best_score) < float(self._config.score_threshold_bnb):
            return self._skip_decision(skip_reason="router_score_below_threshold")

        signal = candidate_signals[str(best_name)]
        return self._to_affordability_checked_decision(
            candidate_name=str(best_name),
            signal=signal,
            bankroll_bnb=float(bankroll_bnb),
            bet_gas_cost_bnb=float(bet_gas_cost_bnb),
            selector_score_bnb=float(best_score),
        )

    def _route_oracle_skip(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        realized_profit_by_candidate: dict[str, float] | None,
    ) -> StrategyRouterDecision:
        if realized_profit_by_candidate is None:
            raise InvariantError("router_oracle_profit_table_missing")

        best_name: str | None = None
        best_profit = 0.0
        for candidate_name, signal in candidate_signals.items():
            if str(signal.action) != "BET":
                continue
            side = str(signal.bet_side or "")
            if side not in _VALID_BET_SIDES:
                raise InvariantError("router_candidate_bet_side_invalid")
            if float(signal.bet_size_bnb) <= 0.0:
                raise InvariantError("router_candidate_bet_size_nonpositive")
            if str(candidate_name) not in realized_profit_by_candidate:
                raise InvariantError("router_oracle_profit_candidate_missing")
            profit = float(realized_profit_by_candidate[str(candidate_name)])
            if float(profit) > float(best_profit):
                best_profit = float(profit)
                best_name = str(candidate_name)

        if best_name is None:
            return self._skip_decision(skip_reason="oracle_no_positive_profit")

        signal = candidate_signals[str(best_name)]
        return self._to_affordability_checked_decision(
            candidate_name=str(best_name),
            signal=signal,
            bankroll_bnb=float(bankroll_bnb),
            bet_gas_cost_bnb=float(bet_gas_cost_bnb),
            selector_score_bnb=None,
        )

    def _to_affordability_checked_decision(
        self,
        *,
        candidate_name: str,
        signal: StrategyCandidateSignal,
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        selector_score_bnb: float | None,
    ) -> StrategyRouterDecision:
        side = str(signal.bet_side or "")
        if side not in _VALID_BET_SIDES:
            raise InvariantError("router_candidate_bet_side_invalid")
        bet_size = float(signal.bet_size_bnb)
        total_cost = float(bet_size) + float(bet_gas_cost_bnb)
        if float(total_cost) > float(bankroll_bnb):
            return self._skip_decision(
                skip_reason="insufficient_bankroll_real",
                selected_strategy=str(candidate_name),
                selector_score_bnb=selector_score_bnb,
                p_bull=signal.p_bull,
            )
        return StrategyRouterDecision(
            action="BET",
            selected_strategy=str(candidate_name),
            bet_side=str(side),
            bet_size_bnb=float(bet_size),
            expected_profit_bnb=float(signal.expected_profit_bnb or 0.0),
            selector_score_bnb=(
                float(selector_score_bnb) if selector_score_bnb is not None else None
            ),
            skip_reason=None,
            p_bull=float(signal.p_bull) if signal.p_bull is not None else None,
        )

    @staticmethod
    def _validate_candidate_signals(candidate_signals: dict[str, StrategyCandidateSignal]) -> None:
        if not candidate_signals:
            raise InvariantError("router_candidate_signals_empty")
        for candidate_name, signal in candidate_signals.items():
            if str(candidate_name) != str(signal.candidate_name):
                raise InvariantError("router_candidate_signal_key_mismatch")

    @staticmethod
    def _skip_decision(
        *,
        skip_reason: str,
        selected_strategy: str | None = None,
        selector_score_bnb: float | None = None,
        p_bull: float | None = None,
    ) -> StrategyRouterDecision:
        return StrategyRouterDecision(
            action="SKIP",
            selected_strategy=str(selected_strategy) if selected_strategy is not None else None,
            bet_side=None,
            bet_size_bnb=0.0,
            expected_profit_bnb=0.0,
            selector_score_bnb=(
                float(selector_score_bnb) if selector_score_bnb is not None else None
            ),
            skip_reason=str(skip_reason),
            p_bull=float(p_bull) if p_bull is not None else None,
        )
