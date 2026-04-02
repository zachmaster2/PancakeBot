from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pancakebot.domain.models.neural_direction_dataset import (
    available_feature_groups,
    build_neural_direction_dataset,
    direction_label_from_position,
    neural_direction_required_context_klines,
    neural_direction_required_history_rounds,
    previous_settled_direction_label,
    select_feature_columns_exact,
    select_feature_groups,
    tail_neural_direction_dataset,
)
from pancakebot.domain.types import Kline, Round
from pancakebot.infra.feature_cache_store import FeatureCacheStore


class _FakeKlineStore:
    def __init__(self, klines: list[Kline]) -> None:
        self._klines = list(klines)

    def get_context_klines(self, *, anchor_close_time_ms: int, size: int) -> list[Kline]:
        eligible = [k for k in self._klines if int(k.close_time_ms) <= int(anchor_close_time_ms)]
        if len(eligible) < int(size):
            raise RuntimeError("insufficient_klines")
        return list(eligible[-int(size) :])


def _closed_round(*, epoch: int, position: str, failed: bool = False) -> Round:
    start_at = 1_000_000 + int(epoch) * 300
    return Round(
        epoch=int(epoch),
        start_at=int(start_at),
        lock_at=int(start_at) + 300,
        close_at=int(start_at) + 600,
        lock_price=600.0,
        close_price=601.0 if str(position) == "Bull" else 599.0,
        position=str(position),
        failed=bool(failed),
        bets=(),
    )


def _kline(*, open_time_ms: int, close_price: float) -> Kline:
    return Kline(
        open_time_ms=int(open_time_ms),
        close_time_ms=int(open_time_ms) + 59_999,
        open_price=float(close_price),
        high_price=float(close_price) + 1.0,
        low_price=float(close_price) - 1.0,
        close_price=float(close_price),
        volume=10.0,
        quote_asset_volume=1000.0,
        number_of_trades=5,
        taker_buy_base_volume=4.0,
        taker_buy_quote_volume=400.0,
    )


class NeuralDirectionDatasetTests(unittest.TestCase):
    def test_direction_label_from_position_maps_bull_and_bear(self) -> None:
        self.assertEqual(1, direction_label_from_position("Bull"))
        self.assertEqual(0, direction_label_from_position("Bear"))

    def test_previous_settled_direction_label_ignores_locked_round_and_house(self) -> None:
        prior = [
            _closed_round(epoch=1, position="Bull"),
            _closed_round(epoch=2, position="House"),
            _closed_round(epoch=3, position="Bear"),
            _closed_round(epoch=4, position="Bull"),
        ]
        label, available = previous_settled_direction_label(prior_context_rounds=prior)
        self.assertEqual(0, label)
        self.assertTrue(available)

    def test_build_neural_direction_dataset_excludes_house_and_failed_targets(self) -> None:
        history_n = int(neural_direction_required_history_rounds())
        rounds = []
        for epoch in range(1, int(history_n) + 9):
            if epoch == int(history_n) + 2:
                rounds.append(_closed_round(epoch=epoch, position="House"))
            elif epoch == int(history_n) + 4:
                rounds.append(_closed_round(epoch=epoch, position="Bull", failed=True))
            else:
                rounds.append(_closed_round(epoch=epoch, position=("Bull" if epoch % 2 == 0 else "Bear")))

        earliest_ms = (int(rounds[0].start_at) - 10_000) * 1000
        klines = [
            _kline(open_time_ms=int(earliest_ms) + idx * 60_000, close_price=600.0 + idx * 0.1)
            for idx in range(int(neural_direction_required_context_klines()) + 400)
        ]
        dataset = build_neural_direction_dataset(
            rounds=rounds,
            klines_store_like=_FakeKlineStore(klines),
            cutoff_seconds=17,
        )

        self.assertEqual(6, dataset.num_examples)
        self.assertEqual(dataset.num_examples, len(dataset.labels))
        self.assertEqual(dataset.num_examples, len(dataset.target_epochs))
        self.assertEqual(2, dataset.metadata["skipped_house"] + dataset.metadata["skipped_failed"])
        self.assertEqual(len(dataset.feature_columns), dataset.feature_matrix.shape[1])
        self.assertTrue(bool(dataset.previous_settled_available.all()))

        tail = tail_neural_direction_dataset(dataset=dataset, n=3)
        self.assertEqual(3, tail.num_examples)

    def test_build_neural_direction_dataset_uses_feature_cache(self) -> None:
        history_n = int(neural_direction_required_history_rounds())
        rounds = [
            _closed_round(epoch=epoch, position=("Bull" if epoch % 2 == 0 else "Bear"))
            for epoch in range(1, int(history_n) + 5)
        ]
        earliest_ms = (int(rounds[0].start_at) - 10_000) * 1000
        klines = [
            _kline(open_time_ms=int(earliest_ms) + idx * 60_000, close_price=600.0 + idx * 0.1)
            for idx in range(int(neural_direction_required_context_klines()) + 400)
        ]
        with tempfile.TemporaryDirectory() as td:
            cache = FeatureCacheStore(str(Path(td) / "feature_cache.sqlite"))
            dataset1 = build_neural_direction_dataset(
                rounds=rounds,
                klines_store_like=_FakeKlineStore(klines),
                cutoff_seconds=17,
                feature_cache_store=cache,
            )
            dataset2 = build_neural_direction_dataset(
                rounds=rounds,
                klines_store_like=_FakeKlineStore(klines),
                cutoff_seconds=17,
                feature_cache_store=cache,
            )
            cache.close()

        self.assertEqual(dataset1.num_examples, dataset2.num_examples)
        self.assertEqual(tuple(dataset1.target_epochs), tuple(dataset2.target_epochs))

    def test_select_feature_groups_filters_matrix_columns(self) -> None:
        history_n = int(neural_direction_required_history_rounds())
        rounds = [
            _closed_round(epoch=epoch, position=("Bull" if epoch % 2 == 0 else "Bear"))
            for epoch in range(1, int(history_n) + 5)
        ]
        earliest_ms = (int(rounds[0].start_at) - 10_000) * 1000
        klines = [
            _kline(open_time_ms=int(earliest_ms) + idx * 60_000, close_price=600.0 + idx * 0.1)
            for idx in range(int(neural_direction_required_context_klines()) + 400)
        ]
        dataset = build_neural_direction_dataset(
            rounds=rounds,
            klines_store_like=_FakeKlineStore(klines),
            cutoff_seconds=17,
        )
        selected = select_feature_groups(
            dataset=dataset,
            include_groups=("price", "regime"),
        )
        self.assertLess(selected.feature_matrix.shape[1], dataset.feature_matrix.shape[1])
        self.assertTrue(all(col.startswith("price_") or col.startswith("regime_") for col in selected.feature_columns))
        self.assertIn("price", available_feature_groups())

    def test_select_feature_columns_exact_preserves_requested_order(self) -> None:
        history_n = int(neural_direction_required_history_rounds())
        rounds = [
            _closed_round(epoch=epoch, position=("Bull" if epoch % 2 == 0 else "Bear"))
            for epoch in range(1, int(history_n) + 5)
        ]
        earliest_ms = (int(rounds[0].start_at) - 10_000) * 1000
        klines = [
            _kline(open_time_ms=int(earliest_ms) + idx * 60_000, close_price=600.0 + idx * 0.1)
            for idx in range(int(neural_direction_required_context_klines()) + 400)
        ]
        dataset = build_neural_direction_dataset(
            rounds=rounds,
            klines_store_like=_FakeKlineStore(klines),
            cutoff_seconds=17,
        )
        requested = (
            str(dataset.feature_columns[5]),
            str(dataset.feature_columns[0]),
            str(dataset.feature_columns[3]),
        )
        selected = select_feature_columns_exact(
            dataset=dataset,
            feature_columns=requested,
        )
        self.assertEqual(requested, selected.feature_columns)
        self.assertEqual((dataset.num_examples, len(requested)), selected.feature_matrix.shape)
        self.assertEqual(
            float(dataset.feature_matrix[0, 5]),
            float(selected.feature_matrix[0, 0]),
        )


if __name__ == "__main__":
    unittest.main()
