from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pancakebot.domain.models.neural_direction_dataset import NeuralDirectionDataset
from pancakebot.domain.models.neural_direction_tcn import (
    NeuralDirectionTcnConfig,
    build_sequence_examples_for_target_epochs,
    load_neural_direction_tcn_bundle,
    predict_neural_direction_tcn_probabilities,
    save_neural_direction_tcn_bundle,
    train_neural_direction_tcn,
)


class NeuralDirectionTcnTests(unittest.TestCase):
    def _dataset(self) -> NeuralDirectionDataset:
        x = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.1],
                [0.3, 0.2],
                [0.4, 0.3],
                [0.5, 0.4],
                [1.0, 1.0],
                [1.1, 1.0],
                [1.2, 1.1],
                [1.3, 1.2],
                [1.4, 1.3],
                [1.5, 1.4],
            ],
            dtype=np.float32,
        )
        y = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
        return NeuralDirectionDataset(
            feature_columns=("f1", "f2"),
            target_epochs=tuple(range(100, 112)),
            labels=y,
            previous_settled_labels=np.asarray(y, dtype=np.int64),
            previous_settled_available=np.ones(len(y), dtype=bool),
            feature_matrix=x,
            metadata={"schema_name": "toy"},
        )

    def test_build_sequence_examples(self) -> None:
        dataset = self._dataset()
        x, y = build_sequence_examples_for_target_epochs(
            dataset=dataset,
            target_epochs=dataset.target_epochs[3:6],
            seq_len=4,
        )
        self.assertEqual((3, 4, 2), x.shape)
        self.assertEqual([0, 0, 0], list(y))

    def test_train_and_predict_tcn(self) -> None:
        dataset = self._dataset()
        bundle = train_neural_direction_tcn(
            dataset=dataset,
            train_target_epochs=dataset.target_epochs[3:10],
            valid_target_epochs=dataset.target_epochs[10:],
            random_seed=7,
            config=NeuralDirectionTcnConfig(
                seq_len=4,
                channels=(8, 8),
                kernel_size=3,
                dropout=0.0,
                learning_rate=0.01,
                weight_decay=0.0,
                batch_size=4,
                max_epochs=20,
                patience_epochs=4,
            ),
        )
        x, _ = build_sequence_examples_for_target_epochs(
            dataset=dataset,
            target_epochs=dataset.target_epochs[3:],
            seq_len=4,
        )
        probs = predict_neural_direction_tcn_probabilities(
            bundle=bundle,
            feature_sequences=x,
        )
        self.assertEqual((9,), probs.shape)
        self.assertTrue(float(np.min(probs)) >= 0.0)
        self.assertTrue(float(np.max(probs)) <= 1.0)

    def test_bundle_round_trip_preserves_probabilities(self) -> None:
        dataset = self._dataset()
        bundle = train_neural_direction_tcn(
            dataset=dataset,
            train_target_epochs=dataset.target_epochs[3:10],
            valid_target_epochs=dataset.target_epochs[10:],
            random_seed=9,
            config=NeuralDirectionTcnConfig(
                seq_len=4,
                channels=(8, 8),
                kernel_size=3,
                dropout=0.0,
                learning_rate=0.01,
                weight_decay=0.0,
                batch_size=4,
                max_epochs=12,
                patience_epochs=3,
            ),
        )
        x, _ = build_sequence_examples_for_target_epochs(
            dataset=dataset,
            target_epochs=dataset.target_epochs[3:],
            seq_len=4,
        )
        probs_before = predict_neural_direction_tcn_probabilities(
            bundle=bundle,
            feature_sequences=x,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "direction_tcn.pt"
            save_neural_direction_tcn_bundle(bundle=bundle, path=str(path))
            loaded = load_neural_direction_tcn_bundle(str(path))
        probs_after = predict_neural_direction_tcn_probabilities(
            bundle=loaded,
            feature_sequences=x,
        )
        self.assertTrue(np.allclose(probs_before, probs_after))


if __name__ == "__main__":
    unittest.main()
