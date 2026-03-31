from __future__ import annotations

import unittest
from unittest.mock import patch

from pancakebot.domain.strategy.direct_action_policy import DirectActionPolicy
from pancakebot.domain.strategy.direct_action_policy_model import DirectActionModelBundle, default_direct_action_specs
from pancakebot.domain.types import Round


class _ConstantModel:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def predict(self, x) -> list[float]:
        return list(self._values[: len(x)])


def _round(*, epoch: int) -> Round:
    return Round(
        epoch=int(epoch),
        start_at=1_000_000 + int(epoch) * 300,
        lock_at=1_000_000 + int(epoch) * 300 + 300,
        close_at=1_000_000 + int(epoch) * 300 + 600,
        lock_price=600.0,
        close_price=601.0,
        position="Bull",
        failed=False,
        bets=(),
    )


class DirectActionPolicyTests(unittest.TestCase):
    def _bundle(self, *, q10: list[float], q50: list[float]) -> DirectActionModelBundle:
        return DirectActionModelBundle(
            feature_names=("f1", "f2"),
            action_specs=default_direct_action_specs()[:3],
            q10_model=_ConstantModel(q10),
            q50_model=_ConstantModel(q50),
            metadata={"required_history_rounds": 5},
        )

    @patch("pancakebot.domain.strategy.direct_action_policy.direct_action_required_history_rounds", return_value=5)
    @patch("pancakebot.domain.strategy.direct_action_policy.max_required_prior_context_rounds_size", return_value=2)
    def test_history_insufficient_returns_skip(self, _prior_mock, _required_mock) -> None:
        policy = DirectActionPolicy(
            cutoff_seconds=10,
            treasury_fee_fraction=0.03,
            klines_store_like=object(),
            bundle=self._bundle(q10=[0.0, 0.0, 0.0], q50=[0.0, 0.0, 0.0]),
        )
        policy.observe_closed_rounds(rounds=[_round(epoch=1), _round(epoch=2), _round(epoch=3), _round(epoch=4)])

        decision = policy.decide_open_round(round_t=_round(epoch=5), bankroll_bnb=50.0)

        self.assertEqual("SKIP", decision.action)
        self.assertEqual("direct_action_history_insufficient", decision.skip_reason)

    @patch("pancakebot.domain.strategy.direct_action_policy._base_feature_vector_for_round", return_value=[1.0, 2.0])
    @patch(
        "pancakebot.domain.strategy.direct_action_policy._direct_action_feature_row_values",
        side_effect=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    )
    @patch("pancakebot.domain.strategy.direct_action_policy.direct_action_required_history_rounds", return_value=5)
    @patch("pancakebot.domain.strategy.direct_action_policy.max_required_prior_context_rounds_size", return_value=2)
    def test_positive_q10_picks_best_bet(
        self,
        _prior_mock,
        _required_mock,
        _feature_row_mock,
        _base_vector_mock,
    ) -> None:
        policy = DirectActionPolicy(
            cutoff_seconds=10,
            treasury_fee_fraction=0.03,
            klines_store_like=object(),
            bundle=self._bundle(q10=[0.0, 0.01, -0.02], q50=[0.0, 0.03, -0.01]),
        )
        policy.observe_closed_rounds(rounds=[_round(epoch=i) for i in range(1, 6)])

        decision = policy.decide_open_round(round_t=_round(epoch=6), bankroll_bnb=50.0)

        self.assertEqual("BET", decision.action)
        self.assertEqual("bull_0p05", decision.action_id)
        self.assertEqual("Bull", decision.bet_side)
        self.assertAlmostEqual(0.05, float(decision.bet_size_bnb))
        self.assertAlmostEqual(0.01, float(decision.score_bnb))
        self.assertAlmostEqual(0.03, float(decision.q50_net_bnb))

    @patch("pancakebot.domain.strategy.direct_action_policy._base_feature_vector_for_round", return_value=[1.0, 2.0])
    @patch(
        "pancakebot.domain.strategy.direct_action_policy._direct_action_feature_row_values",
        side_effect=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    )
    @patch("pancakebot.domain.strategy.direct_action_policy.direct_action_required_history_rounds", return_value=5)
    @patch("pancakebot.domain.strategy.direct_action_policy.max_required_prior_context_rounds_size", return_value=2)
    def test_nonpositive_best_q10_falls_back_to_skip_but_keeps_raw_scores(
        self,
        _prior_mock,
        _required_mock,
        _feature_row_mock,
        _base_vector_mock,
    ) -> None:
        policy = DirectActionPolicy(
            cutoff_seconds=10,
            treasury_fee_fraction=0.03,
            klines_store_like=object(),
            bundle=self._bundle(q10=[-0.02, -0.01, -0.03], q50=[-0.01, 0.02, -0.02]),
        )
        policy.observe_closed_rounds(rounds=[_round(epoch=i) for i in range(1, 6)])

        decision = policy.decide_open_round(round_t=_round(epoch=6), bankroll_bnb=50.0)

        self.assertEqual("SKIP", decision.action)
        self.assertEqual("skip", decision.action_id)
        self.assertEqual("direct_action_nonpositive_score", decision.skip_reason)
        self.assertAlmostEqual(-0.01, float(decision.score_bnb))
        self.assertAlmostEqual(0.02, float(decision.q50_net_bnb))


if __name__ == "__main__":
    unittest.main()
