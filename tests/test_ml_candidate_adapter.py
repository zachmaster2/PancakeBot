"""Deterministic tests for the ML candidate adapter."""

from __future__ import annotations

from dataclasses import replace
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pancakebot.core.constants import BNB_WEI
from pancakebot.config.strategy_config import MlCandidateConfig
from pancakebot.domain.strategy.ml_candidate_adapter import MlCandidateAdapter
from pancakebot.domain.types import Bet, Round


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
            expected_net_max_bnb=None,
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

    def test_predict_final_pools_uses_projection_cache(self) -> None:
        adapter = MlCandidateAdapter(
            config=self._ml_cfg(enabled=False, name="ml_test"),
            cutoff_seconds=17,
            treasury_fee_fraction=0.03,
            klines_store_like=object(),
        )

        history_round = Round(
            epoch=122,
            start_at=999_700,
            lock_at=1_000_000,
            close_at=1_000_300,
            lock_price=600.0,
            close_price=601.0,
            position="Bull",
            failed=False,
            bets=(),
        )
        adapter.settle_closed_rounds(rounds=[history_round])

        round_t = Round(
            epoch=123,
            start_at=1_000_000,
            lock_at=1_000_300,
            close_at=1_000_600,
            lock_price=600.0,
            close_price=601.0,
            position="Bull",
            failed=False,
            bets=(
                Bet(
                    wallet_address="0xabc",
                    amount_wei=int(BNB_WEI),
                    position="Bull",
                    created_at=1_000_250,
                ),
                Bet(
                    wallet_address="0xdef",
                    amount_wei=int(BNB_WEI),
                    position="Bear",
                    created_at=1_000_250,
                ),
            ),
        )

        pool_model = SimpleNamespace()
        pool_model.calls = 0

        def _predict(_rows):
            pool_model.calls = int(pool_model.calls) + 1
            return [(2.0, 0.6)]

        pool_model.predict = _predict
        fake_state = SimpleNamespace(models=SimpleNamespace(pool_model=pool_model))

        with (
            patch(
                "pancakebot.domain.strategy.ml_candidate_adapter.max_required_prior_context_rounds_size",
                return_value=1,
            ),
            patch(
                "pancakebot.domain.strategy.ml_candidate_adapter.ensure_state",
                return_value=fake_state,
            ) as ensure_state_mock,
            patch.object(
                MlCandidateAdapter,
                "_feature_vector_for_round",
                return_value=[0.1, 0.2],
            ) as feature_mock,
        ):
            first = adapter.predict_final_pools_for_round(round_t=round_t)
            second = adapter.predict_final_pools_for_round(round_t=round_t)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first, second)
        ensure_state_mock.assert_called_once()
        feature_mock.assert_called_once()
        self.assertEqual(1, int(pool_model.calls))

    def test_import_bootstrap_state_keeps_settled_projection_entries(self) -> None:
        adapter = MlCandidateAdapter(
            config=self._ml_cfg(enabled=False, name="ml_test"),
            cutoff_seconds=17,
            treasury_fee_fraction=0.03,
            klines_store_like=object(),
        )

        history_round = Round(
            epoch=122,
            start_at=999_700,
            lock_at=1_000_000,
            close_at=1_000_300,
            lock_price=600.0,
            close_price=601.0,
            position="Bull",
            failed=False,
            bets=(),
        )
        projection_key = (122, 1_000_000, 999_983, int(BNB_WEI), int(BNB_WEI))
        adapter.import_bootstrap_state(
            state={
                "history_rounds_json": [history_round.to_json()],
                "walk_forward_state": None,
                "final_pool_projection_cache": [
                    {
                        "k": [int(x) for x in projection_key],
                        "v": [3.0, 1.8, 1.2],
                    }
                ],
            }
        )

        cache = adapter._final_pool_projection_cache  # noqa: SLF001
        self.assertIn(projection_key, cache)
        self.assertEqual((3.0, 1.8, 1.2), cache[projection_key])

    def test_expected_net_above_max_skips_candidate(self) -> None:
        cfg = self._ml_cfg(enabled=True, name="ml_test")
        cfg = replace(cfg, expected_net_max_bnb=0.01)
        adapter = MlCandidateAdapter(
            config=cfg,
            cutoff_seconds=17,
            treasury_fee_fraction=0.03,
            klines_store_like=object(),
        )

        history_round = Round(
            epoch=122,
            start_at=999_700,
            lock_at=1_000_000,
            close_at=1_000_300,
            lock_price=600.0,
            close_price=601.0,
            position="Bull",
            failed=False,
            bets=(),
        )
        adapter.settle_closed_rounds(rounds=[history_round])

        round_t = Round(
            epoch=123,
            start_at=1_000_000,
            lock_at=1_000_300,
            close_at=1_000_600,
            lock_price=600.0,
            close_price=601.0,
            position="Bull",
            failed=False,
            bets=(
                Bet(
                    wallet_address="0xabc",
                    amount_wei=int(BNB_WEI),
                    position="Bull",
                    created_at=1_000_250,
                ),
                Bet(
                    wallet_address="0xdef",
                    amount_wei=int(BNB_WEI),
                    position="Bear",
                    created_at=1_000_250,
                ),
            ),
        )

        fake_price_model = SimpleNamespace(predict=lambda _rows: [0.2])
        fake_pool_model = SimpleNamespace(predict=lambda _rows: [(2.0, 0.6)])
        fake_state = SimpleNamespace(
            models=SimpleNamespace(price_model=fake_price_model, pool_model=fake_pool_model),
            calibrator_final=object(),
        )

        with (
            patch(
                "pancakebot.domain.strategy.ml_candidate_adapter.max_required_prior_context_rounds_size",
                return_value=1,
            ),
            patch(
                "pancakebot.domain.strategy.ml_candidate_adapter.ensure_state",
                return_value=fake_state,
            ),
            patch.object(
                MlCandidateAdapter,
                "_feature_vector_for_round",
                return_value=[0.1, 0.2],
            ),
            patch(
                "pancakebot.domain.strategy.ml_candidate_adapter.predict_probabilities",
                return_value=0.8,
            ),
            patch(
                "pancakebot.domain.strategy.ml_candidate_adapter.predict_tradeable_probability",
                return_value=0.9,
            ),
        ):
            signal = adapter.candidate_signal_for_open_round(round_t=round_t)

        self.assertEqual("SKIP", signal.action)
        self.assertEqual("expected_net_above_max", signal.skip_reason)


if __name__ == "__main__":
    unittest.main()
