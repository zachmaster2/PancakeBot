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
    projected_late_imbalance: float | None = None,
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
        projected_late_imbalance=(
            float(projected_late_imbalance)
            if projected_late_imbalance is not None
            else None
        ),
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

    def test_online_cellmean_side_gap_routes_when_side_outperforms_opposite(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_cellmean_side_gap",
                online_warmup_rounds=2,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                online_use_direction_split=True,
            )
        )
        bull_signal = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=None,
                skip_reason=None,
            )
        }
        bear_signal = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=None,
                skip_reason=None,
            )
        }

        warmup_1 = router.route_round(
            candidate_signals=bull_signal,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("router_online_warmup", warmup_1.skip_reason)
        router.observe_settlement(
            candidate_signals=bull_signal,
            realized_profit_by_candidate={"a": 0.03},
        )

        warmup_2 = router.route_round(
            candidate_signals=bear_signal,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("router_online_warmup", warmup_2.skip_reason)
        router.observe_settlement(
            candidate_signals=bear_signal,
            realized_profit_by_candidate={"a": -0.01},
        )

        decision = router.route_round(
            candidate_signals=bull_signal,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("BET", decision.action)
        self.assertEqual("a", decision.selected_strategy)
        self.assertEqual("Bull", decision.bet_side)
        self.assertAlmostEqual(0.04, float(decision.selector_score_bnb))

    def test_online_cellmean_side_gap_skips_when_opposite_side_is_better(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_cellmean_side_gap",
                online_warmup_rounds=2,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                online_use_direction_split=True,
            )
        )
        bull_signal = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=None,
                skip_reason=None,
            )
        }
        bear_signal = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=None,
                skip_reason=None,
            )
        }

        _ = router.route_round(
            candidate_signals=bull_signal,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=bull_signal,
            realized_profit_by_candidate={"a": 0.01},
        )
        _ = router.route_round(
            candidate_signals=bear_signal,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=bear_signal,
            realized_profit_by_candidate={"a": 0.02},
        )

        decision = router.route_round(
            candidate_signals=bull_signal,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("SKIP", decision.action)
        self.assertEqual("router_online_no_candidate", decision.skip_reason)

    def test_online_cellmean_side_gap_requires_direction_split(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_cellmean_side_gap",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                online_use_direction_split=False,
            )
        )
        candidate_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=None,
                skip_reason=None,
            )
        }

        warmup = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("router_online_warmup", warmup.skip_reason)
        router.observe_settlement(
            candidate_signals=candidate_signals,
            realized_profit_by_candidate={"a": 0.01},
        )

        decision = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("SKIP", decision.action)
        self.assertEqual("router_online_side_gap_requires_direction_split", decision.skip_reason)

    def test_online_cellmean_selector_fallback_routes_when_online_has_no_candidate(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_cellmean_selector_fallback",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=2,
                online_score_threshold_bnb=-1.0,
                score_threshold_bnb=-1e9,
            )
        )
        candidate_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.01,
                selector_score_bnb=0.05,
                skip_reason=None,
            ),
        }
        warmup = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("router_online_warmup", warmup.skip_reason)

        router.observe_settlement(
            candidate_signals=candidate_signals,
            realized_profit_by_candidate={"a": 0.01, "b": 0.01},
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

    def test_online_cellmean_selector_gate_ranks_by_selector_score_after_online_gate(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_cellmean_selector_gate",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                score_threshold_bnb=-1e9,
            )
        )
        warmup_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.05,
                skip_reason=None,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
            ),
        }
        _ = router.route_round(
            candidate_signals=warmup_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=warmup_signals,
            realized_profit_by_candidate={"a": 0.05, "b": 0.02},
        )

        routed_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.04,
                skip_reason=None,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.12,
                skip_reason=None,
            ),
        }
        decision = router.route_round(
            candidate_signals=routed_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("b", decision.selected_strategy)
        self.assertEqual("Bear", decision.bet_side)
        self.assertAlmostEqual(0.12, float(decision.selector_score_bnb))

    def test_online_selector_score_fallback_uses_selector_score_cells(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_selector_score_fallback",
                online_warmup_rounds=3,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=0.02,
                score_threshold_bnb=-1e9,
            )
        )
        warmup_low = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
            )
        }
        warmup_high = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.30,
                skip_reason=None,
            )
        }
        warmup_mid = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.20,
                skip_reason=None,
            )
        }
        _ = router.route_round(
            candidate_signals=warmup_low,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=warmup_low,
            realized_profit_by_candidate={"a": 0.04},
        )
        _ = router.route_round(
            candidate_signals=warmup_high,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=warmup_high,
            realized_profit_by_candidate={"a": -0.02},
        )
        _ = router.route_round(
            candidate_signals=warmup_mid,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=warmup_mid,
            realized_profit_by_candidate={"a": 0.0},
        )

        decision = router.route_round(
            candidate_signals=warmup_low,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=False,
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("a", decision.selected_strategy)
        self.assertEqual("Bull", decision.bet_side)

    def test_online_selector_score_gate_ranks_by_selector_score_after_online_gate(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_selector_score_gate",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                score_threshold_bnb=-1e9,
            )
        )
        warmup_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.05,
                skip_reason=None,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
            ),
        }
        _ = router.route_round(
            candidate_signals=warmup_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=warmup_signals,
            realized_profit_by_candidate={"a": 0.05, "b": 0.02},
        )

        routed_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.04,
                skip_reason=None,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.12,
                skip_reason=None,
            ),
        }
        decision = router.route_round(
            candidate_signals=routed_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("b", decision.selected_strategy)
        self.assertEqual("Bear", decision.bet_side)
        self.assertAlmostEqual(0.12, float(decision.selector_score_bnb))

    def test_online_selector_score_side_gap_uses_selector_score_cells(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_selector_score_side_gap",
                online_warmup_rounds=2,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                online_use_direction_split=True,
                score_threshold_bnb=-1e9,
            )
        )
        bull_signal = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
            )
        }
        bear_signal = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
            )
        }

        _ = router.route_round(
            candidate_signals=bull_signal,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=bull_signal,
            realized_profit_by_candidate={"a": 0.03},
        )
        _ = router.route_round(
            candidate_signals=bear_signal,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=bear_signal,
            realized_profit_by_candidate={"a": -0.01},
        )

        decision = router.route_round(
            candidate_signals=bull_signal,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("a", decision.selected_strategy)
        self.assertEqual("Bull", decision.bet_side)
        self.assertAlmostEqual(0.04, float(decision.selector_score_bnb))

    def test_online_selector_score_late_imb_fallback_uses_late_context_cells(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_selector_score_late_imb_fallback",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                score_threshold_bnb=-1e9,
            )
        )
        warmup_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
                projected_late_imbalance=0.80,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
                projected_late_imbalance=-0.80,
            ),
        }
        _ = router.route_round(
            candidate_signals=warmup_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=warmup_signals,
            realized_profit_by_candidate={"a": 0.05, "b": -0.03},
        )

        routed_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.11,
                skip_reason=None,
                projected_late_imbalance=0.75,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.11,
                skip_reason=None,
                projected_late_imbalance=-0.75,
            ),
        }
        decision = router.route_round(
            candidate_signals=routed_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("a", decision.selected_strategy)
        self.assertEqual("Bull", decision.bet_side)

    def test_online_selector_score_late_imb_gate_prefers_selector_score_after_gate(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_selector_score_late_imb_gate",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                score_threshold_bnb=-1e9,
            )
        )
        warmup_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.05,
                skip_reason=None,
                projected_late_imbalance=0.60,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.08,
                skip_reason=None,
                projected_late_imbalance=0.65,
            ),
        }
        _ = router.route_round(
            candidate_signals=warmup_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=warmup_signals,
            realized_profit_by_candidate={"a": 0.04, "b": 0.03},
        )

        routed_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.04,
                skip_reason=None,
                projected_late_imbalance=0.62,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.12,
                skip_reason=None,
                projected_late_imbalance=0.67,
            ),
        }
        decision = router.route_round(
            candidate_signals=routed_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("b", decision.selected_strategy)
        self.assertEqual("Bull", decision.bet_side)
        self.assertAlmostEqual(0.12, float(decision.selector_score_bnb))

    def test_online_selector_score_side_late_fallback_uses_side_aligned_context(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_selector_score_side_late_fallback",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                score_threshold_bnb=-1e9,
            )
        )
        warmup_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
                projected_late_imbalance=0.80,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
                projected_late_imbalance=0.80,
            ),
        }
        _ = router.route_round(
            candidate_signals=warmup_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=warmup_signals,
            realized_profit_by_candidate={"a": 0.05, "b": -0.03},
        )

        routed_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.11,
                skip_reason=None,
                projected_late_imbalance=0.75,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.11,
                skip_reason=None,
                projected_late_imbalance=0.75,
            ),
        }
        decision = router.route_round(
            candidate_signals=routed_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("a", decision.selected_strategy)
        self.assertEqual("Bull", decision.bet_side)

    def test_online_selector_score_side_support_gate_prefers_selector_score_after_gate(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_selector_score_side_support_gate",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=1,
                online_score_threshold_bnb=-1.0,
                score_threshold_bnb=-1e9,
            )
        )
        warmup_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.05,
                skip_reason=None,
                p_bull=0.70,
                projected_late_imbalance=0.40,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.08,
                skip_reason=None,
                p_bull=0.20,
                projected_late_imbalance=-0.60,
            ),
        }
        _ = router.route_round(
            candidate_signals=warmup_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=warmup_signals,
            realized_profit_by_candidate={"a": 0.03, "b": 0.04},
        )

        routed_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.04,
                skip_reason=None,
                p_bull=0.72,
                projected_late_imbalance=0.38,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.12,
                skip_reason=None,
                p_bull=0.18,
                projected_late_imbalance=-0.58,
            ),
        }
        decision = router.route_round(
            candidate_signals=routed_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("b", decision.selected_strategy)
        self.assertEqual("Bear", decision.bet_side)
        self.assertAlmostEqual(0.12, float(decision.selector_score_bnb))

    def test_online_cellmean_backoff_routes_when_cell_is_sparse(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_cellmean_backoff",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=2,
                online_score_threshold_bnb=-1.0,
            )
        )
        candidate_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
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
        warmup = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("router_online_warmup", warmup.skip_reason)

        router.observe_settlement(
            candidate_signals=candidate_signals,
            realized_profit_by_candidate={"a": 0.03, "b": -0.02},
        )

        decision = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("BET", decision.action)
        self.assertEqual("a", decision.selected_strategy)
        self.assertIsNone(decision.skip_reason)

    def test_online_cellmean_selector_fallback_rejects_negative_selector_score(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_cellmean_selector_fallback",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=2,
                online_score_threshold_bnb=-1.0,
                score_threshold_bnb=-1e9,
            )
        )
        candidate_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=-0.01,
                skip_reason=None,
            ),
            "b": _signal(
                candidate_name="b",
                action="BET",
                bet_side="Bear",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.01,
                selector_score_bnb=-0.02,
                skip_reason=None,
            ),
        }
        _ = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=candidate_signals,
            realized_profit_by_candidate={"a": 0.01, "b": 0.01},
        )

        decision = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        self.assertEqual("SKIP", decision.action)
        self.assertEqual("router_fallback_selector_rejected", decision.skip_reason)
        self.assertEqual("a", decision.selected_strategy)

    def test_online_cellmean_selector_fallback_maps_selector_warmup_reason(self) -> None:
        router = StrategyRouter(
            config=StrategyRouterConfig(
                mode="online_cellmean_selector_fallback",
                online_warmup_rounds=1,
                online_num_quantile_bins=2,
                online_min_cell_obs=2,
                online_score_threshold_bnb=-1.0,
                score_threshold_bnb=-1e9,
            )
        )
        candidate_signals = {
            "a": _signal(
                candidate_name="a",
                action="BET",
                bet_side="Bull",
                bet_size_bnb=0.2,
                expected_profit_bnb=0.02,
                selector_score_bnb=0.10,
                skip_reason=None,
            )
        }
        _ = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=True,
        )
        router.observe_settlement(
            candidate_signals=candidate_signals,
            realized_profit_by_candidate={"a": 0.01},
        )

        decision = router.route_round(
            candidate_signals=candidate_signals,
            bankroll_bnb=1.0,
            bet_gas_cost_bnb=0.001,
            selector_ready=False,
        )
        self.assertEqual("SKIP", decision.action)
        self.assertEqual("router_fallback_selector_warmup", decision.skip_reason)


if __name__ == "__main__":
    unittest.main()
