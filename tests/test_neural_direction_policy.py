from __future__ import annotations

import unittest

import numpy as np

from pancakebot.domain.models.neural_direction_policy import (
    confidence_threshold_for_target_coverage,
    simulate_confidence_threshold_policy,
)
from pancakebot.domain.types import Bet, Round


class NeuralDirectionPolicyTests(unittest.TestCase):
    def test_confidence_threshold_for_target_coverage(self) -> None:
        confidence = np.asarray([0.51, 0.52, 0.60, 0.55, 0.80], dtype=np.float32)
        threshold = confidence_threshold_for_target_coverage(
            chosen_side_confidence=confidence,
            target_coverage_fraction=0.4,
        )
        self.assertAlmostEqual(0.60, float(threshold), places=5)

    def test_simulate_confidence_threshold_policy(self) -> None:
        rounds = [
            Round(
                epoch=1,
                start_at=1,
                lock_at=10,
                close_at=20,
                lock_price=100.0,
                close_price=101.0,
                position="Bull",
                failed=False,
                bets=(
                    Bet(wallet_address="a", amount_wei=int(1.0e18), position="Bull", created_at=5),
                    Bet(wallet_address="b", amount_wei=int(1.0e18), position="Bear", created_at=5),
                ),
            ),
            Round(
                epoch=2,
                start_at=21,
                lock_at=30,
                close_at=40,
                lock_price=100.0,
                close_price=99.0,
                position="Bear",
                failed=False,
                bets=(
                    Bet(wallet_address="a", amount_wei=int(1.0e18), position="Bull", created_at=25),
                    Bet(wallet_address="b", amount_wei=int(3.0e18), position="Bear", created_at=25),
                ),
            ),
            Round(
                epoch=3,
                start_at=41,
                lock_at=50,
                close_at=60,
                lock_price=100.0,
                close_price=101.0,
                position="Bull",
                failed=False,
                bets=(
                    Bet(wallet_address="a", amount_wei=int(2.0e18), position="Bull", created_at=45),
                    Bet(wallet_address="b", amount_wei=int(1.0e18), position="Bear", created_at=45),
                ),
            ),
        ]
        probs = np.asarray([0.70, 0.60, 0.51], dtype=np.float32)
        result = simulate_confidence_threshold_policy(
            rounds=rounds,
            calibrated_bull_probs=probs,
            threshold=0.55,
            bet_size_bnb=0.1,
            initial_bankroll_bnb=1.0,
            treasury_fee_fraction=0.03,
        )
        self.assertEqual(3, result.num_rounds)
        self.assertEqual(2, result.num_bets)
        self.assertEqual(1, result.num_skips_below_threshold)
        self.assertAlmostEqual(2.0 / 3.0, result.bet_rate)
        self.assertAlmostEqual(0.5, result.win_rate)
        self.assertIsNotNone(result.selected_min_confidence)


if __name__ == "__main__":
    unittest.main()
