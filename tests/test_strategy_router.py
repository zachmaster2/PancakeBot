"""Deterministic tests for the shared strategy router."""

from __future__ import annotations

import unittest

from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.strategy.router import StrategyRouter, StrategyRouterConfig


def _signal(
    *,
    candidate_name: str,
    action: str,
    bet_side: str | None,
    bet_size_bnb: float,
    expected_profit_bnb: float | None,
    selector_score_bnb: float | None,
    skip_reason: str | None,
    p_bull: float | None = 0.5,
) -> StrategyCandidateSignal:
    """Create one candidate signal fixture."""

    return StrategyCandidateSignal(
        candidate_name=str(candidate_name),
        action=str(action),
        bet_side=bet_side,
        bet_size_bnb=float(bet_size_bnb),
        expected_profit_bnb=(
            float(expected_profit_bnb) if expected_profit_bnb is not None else None
        ),
        selector_score_bnb=(
            float(selector_score_bnb) if selector_score_bnb is not None else None
        ),
        skip_reason=skip_reason,
        p_bull=float(p_bull) if p_bull is not None else None,
        dislocation_bull=0.0,
    )


class StrategyRouterTests(unittest.TestCase):
    """Test router selection behavior for core modes."""

    def test_skip_only_always_skips(self) -> None:
        router = StrategyRouter(config=StrategyRouterConfig(mode="skip_only"))
        candidate_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.01,
                selector_score_bnb=0.03,
                skip_reason=None,
            )
        }
        decision = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )

        self.assertEqual("SKIP", decision.action)
        self.assertEqual("router_skip_only", decision.skip_reason)
        self.assertIsNone(decision.selected_strategy)
        self.assertEqual(0.0, decision.bet_size_bnb)

    def test_oracle_skip_picks_best_positive_realized_profit(self) -> None:
        router = StrategyRouter(config=StrategyRouterConfig(mode="oracle_skip"))
        candidate_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.01,
                selector_score_bnb=None,
                skip_reason=None,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=None,
                skip_reason=None,
            ),
            "c": _signal(
                candidate_name="c",
                action="SKIP",
                bet_side=None,
                bet_size_bnb=0.0,
                expected_profit_bnb=None,
                selector_score_bnb=None,
                skip_reason="candidate_skip",
            ),
        }
        decision = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
            realized_profit_by_candidate={"a": -0.03, "b": 0.04, "c": 0.0},
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("b", decision.selected_strategy)
        self.assertEqual("Bear", decision.bet_side)
        self.assertIsNone(decision.skip_reason)

    def test_selector_max_score_respects_extra_score_threshold_gate(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="selector_max_score",
                score_threshold_bnb=0.25,
            )
        )
        candidate_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.01,
                selector_score_bnb=0.20,
                skip_reason=None,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.03,
                selector_score_bnb=0.11,
                skip_reason=None,
            ),
        }
        decision = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )

        self.assertEqual("SKIP", decision.action)
        self.assertEqual("router_score_below_threshold", decision.skip_reason)
        self.assertIsNone(decision.selected_strategy)

    def test_online_cellmean_routes_after_warmup(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_cellmean",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
            )
        )
        candidate_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.03,
                selector_score_bnb=None,
                skip_reason=None,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.01,
                selector_score_bnb=None,
                skip_reason=None,
            ),
        }

        warmup_decision = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("SKIP", warmup_decision.action)
        self.assertEqual("router_online_warmup", warmup_decision.skip_reason)

        router.observe_settlement(
            candidate_signals=candidate_signals,
            realized_profit_by_candidate={"a": 0.04, "b": -0.01},
        )

        decision = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("BET", decision.action)
        self.assertEqual("a", decision.selected_strategy)
        self.assertEqual("Bull", decision.bet_side)
        self.assertIsNone(decision.skip_reason)


if __name__ == "__main__":
    unittest.main()
