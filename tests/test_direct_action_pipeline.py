from __future__ import annotations

import unittest

from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.direct_action_policy import DirectActionPolicyDecision
from pancakebot.domain.strategy.pipeline import StrategyPipeline
from pancakebot.domain.types import Round


class _FakeDislocationEngine:
    def __init__(self) -> None:
        self.requested_epochs: list[int] = []
        self.settled_epoch_batches: list[list[int]] = []

    def export_bootstrap_state(self) -> dict[str, object]:
        return {}

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        return None

    def export_kline_index_state(self) -> dict[str, object]:
        return {}

    def import_kline_index_state(self, *, state: dict[str, object]) -> None:
        return None

    def refresh_klines(self, klines) -> None:
        return None

    def candidate_signals_for_open_round(self, *, round_t: Round) -> dict[str, object]:
        self.requested_epochs.append(int(round_t.epoch))
        return {}

    def settle_closed_rounds(self, rounds) -> None:
        self.settled_epoch_batches.append([int(round_t.epoch) for round_t in rounds])


class _FakeRouter:
    mode = "selector_max_score"

    def export_bootstrap_state(self) -> dict[str, object]:
        return {}

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        return None


class _FakeDirectActionPolicy:
    enabled = True

    def __init__(self) -> None:
        self.observed_epoch_batches: list[list[int]] = []
        self.last_legacy_candidate_signals: dict[str, object] | None = None

    def decide_open_round(
        self,
        *,
        round_t: Round,
        bankroll_bnb: float,
        legacy_candidate_signals=None,
    ) -> DirectActionPolicyDecision:
        self.last_legacy_candidate_signals = dict(legacy_candidate_signals or {})
        return DirectActionPolicyDecision(
            enabled=True,
            action_id="bull_0p10",
            action_label="Bull @ 0.10",
            action="BET",
            bet_side="Bull",
            bet_size_bnb=0.10,
            score_bnb=0.012,
            q50_net_bnb=0.025,
            top_actions_json='[{"action_id":"bull_0p10"}]',
            skip_reason=None,
        )

    def observe_closed_rounds(self, *, rounds: list[Round]) -> None:
        self.observed_epoch_batches.append([int(round_t.epoch) for round_t in rounds])

    def export_bootstrap_state(self) -> dict[str, object]:
        return {"observed": list(self.observed_epoch_batches)}

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        self.observed_epoch_batches = [list(batch) for batch in state.get("observed", [])]


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


class DirectActionPipelineTests(unittest.TestCase):
    def _pipeline(self, policy: _FakeDirectActionPolicy) -> StrategyPipeline:
        return StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(),
            router=_FakeRouter(),
            treasury_fee_fraction=0.03,
            direct_action_policy=policy,
        )

    def test_decide_open_round_maps_direct_action_fields(self) -> None:
        policy = _FakeDirectActionPolicy()
        pipeline = self._pipeline(policy)

        decision = pipeline.decide_open_round(
            round_t=_round(epoch=10),
            bankroll_bnb=50.0,
            allow_oracle_mode=False,
        )

        self.assertEqual("BET", decision.action)
        self.assertEqual("direct_action_policy", decision.selected_strategy)
        self.assertEqual("Bull", decision.bet_side)
        self.assertAlmostEqual(0.10, float(decision.bet_size_bnb))
        self.assertEqual("direct_action_policy_v1", decision.direct_action_mode)
        self.assertEqual("bull_0p10", decision.direct_action_action_id)
        self.assertAlmostEqual(0.012, float(decision.direct_action_score_bnb or 0.0))
        self.assertAlmostEqual(0.025, float(decision.expected_profit_bnb))
        self.assertEqual({}, policy.last_legacy_candidate_signals)

    def test_bootstrap_and_settlement_route_only_to_direct_action_policy(self) -> None:
        policy = _FakeDirectActionPolicy()
        pipeline = self._pipeline(policy)

        pipeline.bootstrap_from_closed_rounds(rounds=[_round(epoch=1), _round(epoch=2)])
        pipeline.settle_closed_rounds(rounds=[_round(epoch=3)])

        self.assertEqual([[1], [2], [3]], policy.observed_epoch_batches)
        self.assertEqual(3, int(pipeline.last_settled_epoch or 0))

    def test_candidate_signals_are_invalid_when_direct_action_enabled(self) -> None:
        pipeline = self._pipeline(_FakeDirectActionPolicy())

        with self.assertRaises(InvariantError):
            pipeline.candidate_signals_for_open_round(round_t=_round(epoch=10))


if __name__ == "__main__":
    unittest.main()
