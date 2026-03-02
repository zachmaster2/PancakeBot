"""Deterministic tests for the ML candidate adapter."""

from __future__ import annotations

import unittest

from pancakebot.config.strategy_config import MlCandidateConfig
from pancakebot.domain.strategy.ml_candidate_adapter import MlCandidateAdapter
from pancakebot.domain.types import Round


class MlCandidateAdapterTests(unittest.TestCase):
    """Test minimal ML adapter behavior."""

    @staticmethod
    def _ml_cfg(*, enabled: bool, name: str) -> MlCandidateConfig:
        return MlCandidateConfig(
            enabled=bool(enabled),
            name=str(name),
            fixed_bet_bnb=0.2,
            min_tradeable_prob=0.51,
            min_prob_edge=0.0015,
            cutoff_pool_total_min_bnb=1.2,
            expected_net_min_bnb=0.0,
            train_size=8000,
            calibrate_size=4000,
            retrain_interval=500,
            recalibrate_interval=250,
            price_alpha=1.0,
            pool_alpha_total=1.0,
            pool_alpha_ratio=1.0,
            recency_weight_floor=0.1,
            recency_weight_power=2.0,
            predictability_baseline_bet_bnb=0.05,
            random_seed=1337,
        )

    def test_disabled_ml_candidate_emits_skip_signal(self) -> None:
        adapter = MlCandidateAdapter(
            config=self._ml_cfg(enabled=False, name="ml_test"),
            cutoff_seconds=17,
            treasury_fee_fraction=0.03,
            klines_store_like=object(),
        )
        round_t = Round(
            epoch=123,
            start_at=1_000_000,
            lock_at=1_000_300,
            close_at=1_000_600,
            lock_price=600.0,
            close_price=601.0,
            position="Bull",
            failed=False,
            bets=(),
        )

        signal = adapter.candidate_signal_for_open_round(round_t=round_t)
        self.assertEqual("ml_test", signal.candidate_name)
        self.assertEqual("SKIP", signal.action)
        self.assertEqual("ml_candidate_disabled", signal.skip_reason)
        self.assertEqual(0.0, signal.bet_size_bnb)


if __name__ == "__main__":
    unittest.main()
