from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pancakebot.domain.models.direction_tree_model import (
    DirectionTreeConfig,
    load_direction_tree_bundle,
    predict_direction_tree_probabilities,
    save_direction_tree_bundle,
    train_direction_tree_classifier,
)
from pancakebot.domain.models.neural_direction_dataset import NeuralDirectionDataset


class DirectionTreeModelTests(unittest.TestCase):
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

    def _train_bundle(self, *, model_type: str):
        dataset = self._dataset()
        bundle = train_direction_tree_classifier(
            dataset=dataset,
            train_target_epochs=dataset.target_epochs[:10],
            valid_target_epochs=dataset.target_epochs[10:],
            random_seed=7,
            config=DirectionTreeConfig(
                model_type=str(model_type),
                n_estimators=40,
                learning_rate=0.1,
                max_depth=4,
                num_leaves=15,
                min_child_samples=1,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=1.0,
                early_stopping_rounds=10,
            ),
        )
        return dataset, bundle

    def test_train_and_predict_lightgbm(self) -> None:
        dataset, bundle = self._train_bundle(model_type="lightgbm")
        probs = predict_direction_tree_probabilities(
            bundle=bundle,
            feature_matrix=dataset.feature_matrix,
        )
        self.assertEqual((12,), probs.shape)
        self.assertTrue(float(np.min(probs)) >= 0.0)
        self.assertTrue(float(np.max(probs)) <= 1.0)

    def test_train_and_predict_catboost(self) -> None:
        dataset, bundle = self._train_bundle(model_type="catboost")
        probs = predict_direction_tree_probabilities(
            bundle=bundle,
            feature_matrix=dataset.feature_matrix,
        )
        self.assertEqual((12,), probs.shape)
        self.assertTrue(float(np.min(probs)) >= 0.0)
        self.assertTrue(float(np.max(probs)) <= 1.0)

    def test_bundle_round_trip_preserves_probabilities(self) -> None:
        dataset, bundle = self._train_bundle(model_type="lightgbm")
        probs_before = predict_direction_tree_probabilities(
            bundle=bundle,
            feature_matrix=dataset.feature_matrix,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "direction_tree.pkl"
            save_direction_tree_bundle(bundle=bundle, path=str(path))
            loaded = load_direction_tree_bundle(str(path))
        probs_after = predict_direction_tree_probabilities(
            bundle=loaded,
            feature_matrix=dataset.feature_matrix,
        )
        self.assertTrue(np.allclose(probs_before, probs_after))


if __name__ == "__main__":
    unittest.main()
