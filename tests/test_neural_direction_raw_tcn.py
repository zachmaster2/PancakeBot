from __future__ import annotations

import unittest

import numpy as np

from pancakebot.domain.models.neural_direction_raw_sequence_dataset import (
    build_neural_direction_raw_sequence_dataset,
    build_raw_sequence_examples_for_target_epochs,
    select_raw_sequence_lengths,
)
from pancakebot.domain.models.neural_direction_raw_tcn import (
    NeuralDirectionRawTcnConfig,
    predict_neural_direction_raw_tcn_probabilities,
    train_neural_direction_raw_tcn,
)
from pancakebot.domain.types import Bet, Kline, Round


class _FakeKlinesStore:
    def __init__(self, klines: list[Kline]) -> None:
        self._klines = list(klines)

    def get_context_klines(self, *, anchor_close_time_ms: int, size: int) -> list[Kline]:
        eligible = [kline for kline in self._klines if int(kline.close_time_ms) <= int(anchor_close_time_ms)]
        if len(eligible) < int(size):
            raise RuntimeError("insufficient_klines")
        return list(eligible[-int(size) :])


def _make_round(*, epoch: int, start_at: int, bull_wei: int, bear_wei: int, position: str) -> Round:
    lock_at = int(start_at) + 240
    close_at = int(lock_at) + 60
    bull_close = 101.0 if str(position) == "Bull" else 99.0
    return Round(
        epoch=int(epoch),
        start_at=int(start_at),
        lock_at=int(lock_at),
        close_at=int(close_at),
        lock_price=100.0,
        close_price=float(bull_close),
        position=str(position),
        failed=False,
        bets=(
            Bet(wallet_address=f"bull_{epoch}", amount_wei=int(bull_wei), position="Bull", created_at=int(start_at) + 60),
            Bet(wallet_address=f"bear_{epoch}", amount_wei=int(bear_wei), position="Bear", created_at=int(start_at) + 120),
            Bet(wallet_address=f"bull2_{epoch}", amount_wei=int(bull_wei // 2), position="Bull", created_at=int(start_at) + 180),
        ),
    )


def _make_klines(*, count: int, start_open_ms: int) -> list[Kline]:
    out: list[Kline] = []
    close_price = 100.0
    for idx in range(int(count)):
        open_time_ms = int(start_open_ms) + int(idx) * 60_000
        open_price = float(close_price)
        close_price = float(open_price * (1.001 if int(idx) % 2 == 0 else 0.999))
        out.append(
            Kline(
                open_time_ms=int(open_time_ms),
                close_time_ms=int(open_time_ms) + 59_999,
                open_price=float(open_price),
                high_price=float(max(open_price, close_price) * 1.001),
                low_price=float(min(open_price, close_price) * 0.999),
                close_price=float(close_price),
                volume=float(10.0 + idx),
                quote_asset_volume=float(1000.0 + idx),
                number_of_trades=100 + int(idx),
                taker_buy_base_volume=float(5.0 + 0.1 * idx),
                taker_buy_quote_volume=float(500.0 + idx),
            )
        )
    return out


class NeuralDirectionRawTcnTests(unittest.TestCase):
    def test_build_dataset_and_select_shorter_lengths(self) -> None:
        rounds = [
            _make_round(epoch=epoch, start_at=1 + (epoch - 1) * 300, bull_wei=int(2e17 + epoch * 1e16), bear_wei=int(1e17), position="Bull" if epoch % 2 == 0 else "Bear")
            for epoch in range(1, 12)
        ]
        klines = _make_klines(count=200, start_open_ms=0)
        dataset = build_neural_direction_raw_sequence_dataset(
            rounds=rounds,
            klines_store_like=_FakeKlinesStore(klines),
            cutoff_seconds=20,
            settled_history_len=4,
            round_flow_bins=4,
            kline_seq_len=16,
        )
        self.assertGreaterEqual(dataset.num_examples, 1)
        self.assertEqual(6, dataset.round_sequence.shape[1])
        self.assertEqual(16, dataset.kline_sequence.shape[1])

        short_dataset = select_raw_sequence_lengths(
            dataset=dataset,
            settled_history_len=2,
            kline_seq_len=8,
        )
        self.assertEqual(4, short_dataset.round_sequence.shape[1])
        self.assertEqual(8, short_dataset.kline_sequence.shape[1])

    def test_train_and_predict_raw_tcn(self) -> None:
        rounds = [
            _make_round(
                epoch=epoch,
                start_at=1 + (epoch - 1) * 300,
                bull_wei=int(2e17 + (epoch % 3) * 2e16),
                bear_wei=int(1e17 + ((epoch + 1) % 3) * 1e16),
                position="Bull" if epoch % 2 == 0 else "Bear",
            )
            for epoch in range(1, 20)
        ]
        klines = _make_klines(count=400, start_open_ms=0)
        dataset = build_neural_direction_raw_sequence_dataset(
            rounds=rounds,
            klines_store_like=_FakeKlinesStore(klines),
            cutoff_seconds=20,
            settled_history_len=4,
            round_flow_bins=4,
            kline_seq_len=16,
        )
        train_epochs = tuple(int(epoch) for epoch in dataset.target_epochs[:8])
        valid_epochs = tuple(int(epoch) for epoch in dataset.target_epochs[8:12])
        test_epochs = tuple(int(epoch) for epoch in dataset.target_epochs[12:15])
        bundle = train_neural_direction_raw_tcn(
            dataset=dataset,
            train_target_epochs=train_epochs,
            valid_target_epochs=valid_epochs,
            random_seed=123,
            config=NeuralDirectionRawTcnConfig(
                round_channels=(16, 16),
                kline_channels=(16, 16),
                snapshot_hidden_sizes=(16,),
                fusion_hidden_sizes=(16,),
                batch_size=8,
                max_epochs=5,
                patience_epochs=2,
            ),
        )
        test_round_x, test_kline_x, test_snapshot_x, test_y = build_raw_sequence_examples_for_target_epochs(
            dataset=dataset,
            target_epochs=test_epochs,
        )
        probs = predict_neural_direction_raw_tcn_probabilities(
            bundle=bundle,
            round_sequence=test_round_x,
            kline_sequence=test_kline_x,
            snapshot_matrix=test_snapshot_x,
        )
        self.assertEqual(len(test_y), len(probs))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))


if __name__ == "__main__":
    unittest.main()
