from __future__ import annotations

import unittest

import numpy as np

from pancakebot.domain.models.payout_aware_policy import (
    naive_cutoff_profit_if_side_wins,
    projected_profit_if_side_wins,
    realized_profit_for_side,
    simulate_payout_aware_policy,
)
from pancakebot.domain.features.targets import compute_pool_forecast_targets
from pancakebot.domain.types import Bet, Round


def _closed_round(
    *,
    epoch: int,
    winner: str,
    bets: tuple[Bet, ...],
    lock_at: int = 200,
) -> Round:
    return Round(
        epoch=int(epoch),
        start_at=100,
        lock_at=int(lock_at),
        close_at=300,
        lock_price=1.0,
        close_price=1.1 if str(winner) == "Bull" else 0.9,
        position=str(winner),
        failed=False,
        bets=bets,
    )


class PayoutAwarePolicyTests(unittest.TestCase):
    def test_naive_cutoff_profit_matches_realized_when_cutoff_at_lock(self) -> None:
        round_closed = _closed_round(
            epoch=1,
            winner="Bull",
            bets=(
                Bet(wallet_address="a", amount_wei=int(1.2e18), position="Bull", created_at=120),
                Bet(wallet_address="b", amount_wei=int(0.8e18), position="Bear", created_at=150),
            ),
        )
        actual = realized_profit_for_side(
            round_closed=round_closed,
            bet_size_bnb=0.05,
            bet_side="Bull",
            treasury_fee_fraction=0.03,
        )
        naive = naive_cutoff_profit_if_side_wins(
            round_closed=round_closed,
            bet_size_bnb=0.05,
            bet_side="Bull",
            treasury_fee_fraction=0.03,
            cutoff_seconds=0,
        )
        self.assertAlmostEqual(actual, naive, places=9)

    def test_simulate_payout_aware_policy_chooses_higher_ev_side(self) -> None:
        rounds = [
            _closed_round(
                epoch=10,
                winner="Bull",
                bets=(
                    Bet(wallet_address="a", amount_wei=int(1.0e18), position="Bull", created_at=120),
                    Bet(wallet_address="b", amount_wei=int(0.9e18), position="Bear", created_at=140),
                ),
            ),
            _closed_round(
                epoch=11,
                winner="Bear",
                bets=(
                    Bet(wallet_address="a", amount_wei=int(0.7e18), position="Bull", created_at=120),
                    Bet(wallet_address="b", amount_wei=int(1.3e18), position="Bear", created_at=140),
                ),
            ),
            _closed_round(
                epoch=12,
                winner="Bull",
                bets=(
                    Bet(wallet_address="a", amount_wei=int(0.8e18), position="Bull", created_at=120),
                    Bet(wallet_address="b", amount_wei=int(0.8e18), position="Bear", created_at=140),
                ),
            ),
        ]
        result, traces = simulate_payout_aware_policy(
            rounds=rounds,
            predicted_ev_bull=np.asarray([0.01, -0.02, 0.03], dtype=np.float32),
            predicted_ev_bear=np.asarray([-0.01, 0.02, -0.01], dtype=np.float32),
            bull_threshold=0.0,
            bear_threshold=0.0,
            bet_size_bnb=0.05,
            initial_bankroll_bnb=5.0,
            treasury_fee_fraction=0.03,
        )
        self.assertEqual(result.num_rounds, 3)
        self.assertEqual(result.num_bets, 3)
        self.assertEqual(result.num_bull_bets, 2)
        self.assertEqual(result.num_bear_bets, 1)
        self.assertEqual([trace.selected_side for trace in traces], ["Bull", "Bear", "Bull"])
        self.assertGreater(result.net_profit_bnb, 0.0)

    def test_projected_profit_matches_realized_for_actual_late_inflow(self) -> None:
        round_closed = _closed_round(
            epoch=21,
            winner="Bull",
            bets=(
                Bet(wallet_address="a", amount_wei=int(1.0e18), position="Bull", created_at=120),
                Bet(wallet_address="b", amount_wei=int(0.8e18), position="Bear", created_at=150),
                Bet(wallet_address="c", amount_wei=int(0.6e18), position="Bull", created_at=190),
                Bet(wallet_address="d", amount_wei=int(0.4e18), position="Bear", created_at=198),
            ),
            lock_at=200,
        )
        targets = compute_pool_forecast_targets(round_t=round_closed, cutoff_seconds=17)
        actual = realized_profit_for_side(
            round_closed=round_closed,
            bet_size_bnb=0.05,
            bet_side="Bull",
            treasury_fee_fraction=0.03,
        )
        projected = projected_profit_if_side_wins(
            round_closed=round_closed,
            bet_size_bnb=0.05,
            bet_side="Bull",
            treasury_fee_fraction=0.03,
            cutoff_seconds=17,
            pred_late_inflow_total_bnb=float(targets.late_inflow_total_bnb),
            pred_late_inflow_bull_frac=float(targets.late_inflow_bull_frac),
        )
        self.assertAlmostEqual(actual, projected, places=9)


if __name__ == "__main__":
    unittest.main()
