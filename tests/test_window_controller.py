from __future__ import annotations

import unittest

from pancakebot.config.strategy_config import WindowControllerConfig
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.strategy.pipeline import StrategyPipeline
from pancakebot.domain.strategy.router import StrategyRouterDecision
from pancakebot.domain.strategy.window_controller import WindowController
from pancakebot.domain.types import Round


def _signal(
    *,
    candidate_name: str,
    action: str,
    bet_side: str | None,
    bet_size_bnb: float,
    expected_profit_bnb: float | None,
    selector_score_bnb: float | None,
    skip_reason: str | None,
) -> StrategyCandidateSignal:
    return StrategyCandidateSignal(
        candidate_name=str(candidate_name),
        action=str(action),
        bet_side=bet_side,
        bet_size_bnb=float(bet_size_bnb),
        expected_profit_bnb=(
            None if expected_profit_bnb is None else float(expected_profit_bnb)
        ),
        selector_score_bnb=(
            None if selector_score_bnb is None else float(selector_score_bnb)
        ),
        skip_reason=skip_reason,
        p_bull=0.5,
        dislocation_bull=0.0,
    )


def _closed_round(*, epoch: int, position: str) -> Round:
    return Round(
        epoch=int(epoch),
        start_at=1_000 + int(epoch) * 300,
        lock_at=1_000 + int(epoch) * 300 + 300,
        close_at=1_000 + int(epoch) * 300 + 600,
        lock_price=600.0,
        close_price=601.0,
        position=str(position),
        failed=False,
        bets=(),
    )


def _open_round(*, epoch: int) -> Round:
    return Round(
        epoch=int(epoch),
        start_at=1_000 + int(epoch) * 300,
        lock_at=1_000 + int(epoch) * 300 + 300,
        close_at=None,
        lock_price=None,
        close_price=None,
        position=None,
        failed=None,
        bets=(),
    )


class _FakeDislocationEngine:
    def __init__(self, *, signals: dict[str, StrategyCandidateSignal]) -> None:
        self._signals = dict(signals)

    def candidate_signals_for_open_round(self, *, round_t: Round) -> dict[str, StrategyCandidateSignal]:
        _ = round_t
        return dict(self._signals)

    def refresh_klines(self, klines: list[object]) -> None:
        _ = klines

    def export_kline_index_state(self) -> dict[str, object]:
        return {}

    def import_kline_index_state(self, *, state: dict[str, object]) -> None:
        _ = state

    def export_bootstrap_state(self) -> dict[str, object]:
        return {}

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        _ = state

    def settle_closed_rounds(self, rounds: list[Round]) -> None:
        _ = rounds

    def selector_ready(self) -> bool:
        return True


class _CapturingRouter:
    mode = "selector_max_score"

    def __init__(self) -> None:
        self.last_candidate_signals: dict[str, StrategyCandidateSignal] | None = None

    def export_bootstrap_state(self) -> dict[str, object]:
        return {}

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        _ = state

    def observe_settlement(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        realized_profit_by_candidate: dict[str, float],
    ) -> None:
        _ = candidate_signals
        _ = realized_profit_by_candidate

    def route_round(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        selector_ready: bool,
        realized_profit_by_candidate: dict[str, float] | None = None,
    ) -> StrategyRouterDecision:
        _ = bankroll_bnb
        _ = bet_gas_cost_bnb
        _ = selector_ready
        _ = realized_profit_by_candidate
        self.last_candidate_signals = dict(candidate_signals)
        for name, signal in candidate_signals.items():
            if str(signal.action) == "BET":
                return StrategyRouterDecision(
                    action="BET",
                    selected_strategy=str(name),
                    bet_side=str(signal.bet_side),
                    bet_size_bnb=float(signal.bet_size_bnb),
                    expected_profit_bnb=float(signal.expected_profit_bnb or 0.0),
                    selector_score_bnb=float(signal.selector_score_bnb or 0.0),
                    skip_reason=None,
                    p_bull=float(signal.p_bull or 0.5),
                )
        return StrategyRouterDecision(
            action="SKIP",
            selected_strategy=None,
            bet_side=None,
            bet_size_bnb=0.0,
            expected_profit_bnb=0.0,
            selector_score_bnb=None,
            skip_reason="router_fallback_selector_no_candidate",
            p_bull=None,
        )


class WindowControllerTests(unittest.TestCase):
    def test_cold_start_defaults_to_baseline_profile(self) -> None:
        controller = WindowController(
            config=WindowControllerConfig(
                enabled=True,
                baseline_profile_name="stageB",
                alternate_profile_name="stageG2",
            )
        )
        signals = {
            "stageB": _signal(
                candidate_name="stageB",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.02,
                skip_reason=None,
            ),
            "stageG2": _signal(
                candidate_name="stageG2",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.03,
                selector_score_bnb=0.03,
                skip_reason=None,
            ),
        }

        decision = controller.decision_for_round(round_t=_open_round(epoch=1), candidate_signals=signals)

        self.assertEqual("profile", decision.selected_action)
        self.assertEqual("stageB", decision.selected_profile_name)
        self.assertEqual(0, int(decision.lookback_windows_used))

    def test_completed_window_can_flip_to_alternate_profile(self) -> None:
        controller = WindowController(
            config=WindowControllerConfig(
                enabled=True,
                baseline_profile_name="stageB",
                alternate_profile_name="stageG2",
                window_rounds=2,
                lookback_windows=1,
                margin_per_500=0.5,
            )
        )
        signals = {
            "stageB": _signal(
                candidate_name="stageB",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.02,
                skip_reason=None,
            ),
            "stageG2": _signal(
                candidate_name="stageG2",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.03,
                selector_score_bnb=0.03,
                skip_reason=None,
            ),
        }
        for epoch in (1, 2):
            round_t = _closed_round(epoch=epoch, position="Bull")
            realized = {
                "stageB": -0.1002,
                "stageG2": 0.0898,
            }
            controller.observe_round_settlement(
                round_t=round_t,
                candidate_signals=signals,
                realized_profit_by_candidate=realized,
            )

        decision = controller.decision_for_round(round_t=_open_round(epoch=3), candidate_signals=signals)

        self.assertEqual("stageG2", decision.selected_profile_name)
        self.assertGreater(float(decision.estimated_alternate_per_500 or 0.0), float(decision.estimated_baseline_per_500 or 0.0))

    def test_skip_mode_can_stand_down_when_both_profiles_are_nonpositive(self) -> None:
        controller = WindowController(
            config=WindowControllerConfig(
                enabled=True,
                mode="trailing_best_vs_baseline_with_skip",
                baseline_profile_name="stageB",
                alternate_profile_name="stageG2",
                window_rounds=2,
                lookback_windows=1,
                skip_threshold_per_500=0.0,
            )
        )
        signals = {
            "stageB": _signal(
                candidate_name="stageB",
                action="SKIP",
                bet_side=None,
                bet_size_bnb=0.0,
                expected_profit_bnb=None,
                selector_score_bnb=None,
                skip_reason="selector_no_candidate",
            ),
            "stageG2": _signal(
                candidate_name="stageG2",
                action="SKIP",
                bet_side=None,
                bet_size_bnb=0.0,
                expected_profit_bnb=None,
                selector_score_bnb=None,
                skip_reason="selector_no_candidate",
            ),
        }
        for epoch in (1, 2):
            controller.observe_round_settlement(
                round_t=_closed_round(epoch=epoch, position="Bull"),
                candidate_signals=signals,
                realized_profit_by_candidate={"stageB": 0.0, "stageG2": 0.0},
            )

        decision = controller.decision_for_round(round_t=_open_round(epoch=3), candidate_signals=signals)
        filtered = controller.apply_to_candidate_signals(candidate_signals=signals, decision=decision)

        self.assertEqual("skip", decision.selected_action)
        self.assertEqual("SKIP", filtered["stageB"].action)
        self.assertEqual("window_controller_skip", filtered["stageB"].skip_reason)

    def test_pipeline_routes_only_controller_selected_profile(self) -> None:
        controller = WindowController(
            config=WindowControllerConfig(
                enabled=True,
                baseline_profile_name="stageB",
                alternate_profile_name="stageG2",
                window_rounds=2,
                lookback_windows=1,
                margin_per_500=0.5,
            )
        )
        signals = {
            "stageB": _signal(
                candidate_name="stageB",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.02,
                skip_reason=None,
            ),
            "stageG2": _signal(
                candidate_name="stageG2",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.03,
                selector_score_bnb=0.03,
                skip_reason=None,
            ),
        }
        router = _CapturingRouter()
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(signals=signals),
            router=router,
            treasury_fee_fraction=0.03,
            window_controller=controller,
        )

        round_one = _closed_round(epoch=1, position="Bull")
        round_two = _closed_round(epoch=2, position="Bull")
        pipeline.candidate_signals_for_open_round(round_t=round_one)
        pipeline.settle_closed_rounds(rounds=[round_one])
        pipeline.candidate_signals_for_open_round(round_t=round_two)
        pipeline.settle_closed_rounds(rounds=[round_two])

        decision = pipeline.decide_open_round(
            round_t=_open_round(epoch=3),
            bankroll_bnb=50.0,
            allow_oracle_mode=False,
        )

        assert router.last_candidate_signals is not None
        self.assertEqual("SKIP", router.last_candidate_signals["stageB"].action)
        self.assertEqual("window_controller_profile_masked", router.last_candidate_signals["stageB"].skip_reason)
        self.assertEqual("BET", router.last_candidate_signals["stageG2"].action)
        self.assertEqual("stageG2", decision.controller_selected_profile)
        self.assertEqual("profile", decision.controller_selected_action)
        self.assertEqual("stageG2", decision.selected_strategy)

    def test_bootstrap_seeds_window_controller_history(self) -> None:
        controller = WindowController(
            config=WindowControllerConfig(
                enabled=True,
                baseline_profile_name="stageB",
                alternate_profile_name="stageG2",
                window_rounds=2,
                lookback_windows=1,
                margin_per_500=0.5,
            )
        )
        signals = {
            "stageB": _signal(
                candidate_name="stageB",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.02,
                skip_reason=None,
            ),
            "stageG2": _signal(
                candidate_name="stageG2",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.03,
                selector_score_bnb=0.03,
                skip_reason=None,
            ),
        }
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(signals=signals),
            router=_CapturingRouter(),
            treasury_fee_fraction=0.03,
            window_controller=controller,
        )

        pipeline.bootstrap_from_closed_rounds(
            rounds=[
                _closed_round(epoch=1, position="Bull"),
                _closed_round(epoch=2, position="Bull"),
            ]
        )
        decision = pipeline.decide_open_round(
            round_t=_open_round(epoch=3),
            bankroll_bnb=50.0,
            allow_oracle_mode=False,
        )

        self.assertEqual("stageG2", decision.controller_selected_profile)
        self.assertEqual(1, int(decision.controller_window_index or 0))

    def test_bootstrap_seeded_controller_does_not_crash_on_next_settlement(self) -> None:
        controller = WindowController(
            config=WindowControllerConfig(
                enabled=True,
                baseline_profile_name="stageB",
                alternate_profile_name="stageG2",
                window_rounds=2,
                lookback_windows=1,
                margin_per_500=0.5,
            )
        )
        signals = {
            "stageB": _signal(
                candidate_name="stageB",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.02,
                skip_reason=None,
            ),
            "stageG2": _signal(
                candidate_name="stageG2",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.1,
                expected_profit_bnb=0.03,
                selector_score_bnb=0.03,
                skip_reason=None,
            ),
        }
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(signals=signals),
            router=_CapturingRouter(),
            treasury_fee_fraction=0.03,
            window_controller=controller,
        )

        pipeline.bootstrap_from_closed_rounds(
            rounds=[
                _closed_round(epoch=1, position="Bull"),
                _closed_round(epoch=2, position="Bull"),
            ]
        )
        pipeline.decide_open_round(
            round_t=_open_round(epoch=4),
            bankroll_bnb=50.0,
            allow_oracle_mode=False,
        )
        pipeline.settle_closed_rounds(rounds=[_closed_round(epoch=3, position="Bull")])

        self.assertEqual(3, int(pipeline.last_settled_epoch or 0))


if __name__ == "__main__":
    unittest.main()
