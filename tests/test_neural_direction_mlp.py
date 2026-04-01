from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pancakebot.domain.models.neural_direction_dataset import NeuralDirectionDataset
from pancakebot.domain.models.neural_direction_mlp import (
    NeuralDirectionMlpConfig,
    load_neural_direction_mlp_bundle,
    predict_neural_direction_probabilities,
    save_neural_direction_mlp_bundle,
    train_neural_direction_mlp,
)


class NeuralDirectionMlpTests(unittest.TestCase):
    def _dataset(self) -> NeuralDirectionDataset:
        x = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.1],
                [0.3, 0.2],
                [1.0, 1.0],
                [1.1, 1.0],
                [1.2, 1.1],
                [1.3, 1.2],
                [0.15, 0.05],
                [1.15, 1.05],
            ],
            dtype=np.float32,
        )
        y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=np.int64)
        return NeuralDirectionDataset(
            feature_columns=("f1", "f2"),
            target_epochs=tuple(range(100, 110)),
            labels=y,
            previous_settled_labels=np.asarray(y, dtype=np.int64),
            previous_settled_available=np.ones(len(y), dtype=bool),
            feature_matrix=x,
            metadata={"schema_name": "toy"},
        )

    def test_train_and_predict_neural_direction_mlp(self) -> None:
        dataset = self._dataset()
        bundle = train_neural_direction_mlp(
            dataset=dataset,
            train_target_epochs=dataset.target_epochs[:8],
            valid_target_epochs=dataset.target_epochs[8:],
            random_seed=7,
            config=NeuralDirectionMlpConfig(
                hidden_sizes=(8,),
                dropout=0.0,
                learning_rate=0.01,
                weight_decay=0.0,
                batch_size=4,
                max_epochs=40,
                patience_epochs=6,
            ),
        )

        probs = predict_neural_direction_probabilities(
            bundle=bundle,
            feature_matrix=dataset.feature_matrix,
        )

        self.assertEqual((10,), probs.shape)
        self.assertTrue(float(np.min(probs)) >= 0.0)
        self.assertTrue(float(np.max(probs)) <= 1.0)
        self.assertIn("best_valid_win_rate", bundle.metadata)

    def test_bundle_round_trip_preserves_probabilities(self) -> None:
        dataset = self._dataset()
        bundle = train_neural_direction_mlp(
            dataset=dataset,
            train_target_epochs=dataset.target_epochs[:8],
            valid_target_epochs=dataset.target_epochs[8:],
            random_seed=11,
            config=NeuralDirectionMlpConfig(
                hidden_sizes=(8,),
                dropout=0.0,
                learning_rate=0.01,
                weight_decay=0.0,
                batch_size=4,
                max_epochs=30,
                patience_epochs=5,
            ),
        )
        probs_before = predict_neural_direction_probabilities(
            bundle=bundle,
            feature_matrix=dataset.feature_matrix,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "direction_mlp.pt"
            save_neural_direction_mlp_bundle(bundle=bundle, path=str(path))
            loaded = load_neural_direction_mlp_bundle(str(path))

        probs_after = predict_neural_direction_probabilities(
            bundle=loaded,
            feature_matrix=dataset.feature_matrix,
        )
        self.assertTrue(np.allclose(probs_before, probs_after))


if __name__ == "__main__":
    unittest.main()
