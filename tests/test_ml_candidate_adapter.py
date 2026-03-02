"""Deterministic tests for the ML candidate adapter."""

from __future__ import annotations

import unittest

from pancakebot.config.strategy_config import MlCandidateConfig
from pancakebot.domain.strategy.ml_candidate_adapter import MlCandidateAdapter
from pancakebot.domain.types import Round


class MlCandidateAdapterTests(unittest.TestCase):
    """Test minimal ML adapter behavior."""

    def test_disabled_ml_candidate_emits_skip_signal(self) -> None:
        adapter = MlCandidateAdapter(
            config=MlCandidateConfig(enabled=False, name="ml_test"),
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
