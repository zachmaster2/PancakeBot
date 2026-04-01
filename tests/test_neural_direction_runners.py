from __future__ import annotations

import unittest

from inspection.neural_direction_eval_common import (
    parse_nonnegative_int_list,
    parse_positive_int_list,
)
from inspection.run_neural_direction_baselines import (
    NeuralDirectionBaselineRow,
    _aggregate_rows as _aggregate_baseline_rows,
)
from inspection.run_neural_direction_mlp_eval import (
    NeuralDirectionMlpEvalRow,
    _aggregate_rows as _aggregate_mlp_rows,
    _parse_hidden_sizes,
    _training_policy_name,
)
from inspection.run_neural_direction_tcn_eval import (
    NeuralDirectionTcnEvalRow,
    _aggregate_rows as _aggregate_tcn_rows,
    _parse_channels,
)


class NeuralDirectionRunnerTests(unittest.TestCase):
    def test_parse_lists_validate_values(self) -> None:
        self.assertEqual([100, 200], parse_positive_int_list("100,200"))
        self.assertEqual([0, 250], parse_nonnegative_int_list("0,250"))
        with self.assertRaises(Exception):
            parse_positive_int_list("0")
        with self.assertRaises(Exception):
            parse_nonnegative_int_list("-1")

    def test_parse_hidden_sizes(self) -> None:
        self.assertEqual((128, 64), _parse_hidden_sizes("128,64"))
        self.assertEqual((64, 32), _parse_channels("64,32"))

    def test_baseline_aggregate_rows(self) -> None:
        rows = [
            NeuralDirectionBaselineRow(
                baseline_name="always_bull",
                sim_size=6480,
                tail_offset_rounds=0,
                num_examples=6400,
                win_rate=0.51,
                fallback_count=0,
                loaded_round_count=7000,
                total_rounds_available=9000,
            ),
            NeuralDirectionBaselineRow(
                baseline_name="always_bull",
                sim_size=6480,
                tail_offset_rounds=216,
                num_examples=6390,
                win_rate=0.49,
                fallback_count=0,
                loaded_round_count=7100,
                total_rounds_available=9000,
            ),
        ]
        aggregates = _aggregate_baseline_rows(rows)
        self.assertEqual(1, len(aggregates))
        self.assertAlmostEqual(0.50, aggregates[0].mean_win_rate)
        self.assertAlmostEqual(0.49, aggregates[0].min_win_rate)

    def test_mlp_aggregate_rows(self) -> None:
        rows = [
            NeuralDirectionMlpEvalRow(
                sim_size=6480,
                tail_offset_rounds=0,
                training_policy="flat",
                train_size=15000,
                pretrain_size=0,
                valid_size=3000,
                recency_half_life_examples=None,
                random_seed=1,
                num_examples=24480,
                feature_dim=10,
                loaded_round_count=26000,
                total_rounds_available=30000,
                bundle_path="a",
                valid_win_rate=0.55,
                test_win_rate=0.54,
            ),
            NeuralDirectionMlpEvalRow(
                sim_size=6480,
                tail_offset_rounds=216,
                training_policy="flat",
                train_size=15000,
                pretrain_size=0,
                valid_size=3000,
                recency_half_life_examples=None,
                random_seed=1,
                num_examples=24480,
                feature_dim=10,
                loaded_round_count=26000,
                total_rounds_available=30000,
                bundle_path="b",
                valid_win_rate=0.53,
                test_win_rate=0.52,
            ),
        ]
        aggregates = _aggregate_mlp_rows(rows)
        self.assertEqual(1, len(aggregates))
        self.assertAlmostEqual(0.53, aggregates[0].mean_test_win_rate)
        self.assertAlmostEqual(0.54, aggregates[0].mean_valid_win_rate)
        self.assertEqual("flat", aggregates[0].training_policy)

    def test_training_policy_name(self) -> None:
        self.assertEqual("flat", _training_policy_name(pretrain_size=0, recency_half_life_examples=0.0))
        self.assertEqual("recency_exp", _training_policy_name(pretrain_size=0, recency_half_life_examples=50_000.0))
        self.assertEqual("pretrain_finetune", _training_policy_name(pretrain_size=300_000, recency_half_life_examples=0.0))
        self.assertEqual(
            "pretrain_finetune_recency_exp",
            _training_policy_name(pretrain_size=300_000, recency_half_life_examples=50_000.0),
        )

    def test_tcn_aggregate_rows(self) -> None:
        rows = [
            NeuralDirectionTcnEvalRow(
                sim_size=6480,
                tail_offset_rounds=0,
                train_size=15000,
                valid_size=3000,
                random_seed=1,
                num_examples=24480,
                feature_dim=10,
                seq_len=16,
                loaded_round_count=26000,
                total_rounds_available=30000,
                bundle_path="a",
                valid_win_rate=0.57,
                test_win_rate=0.55,
            ),
            NeuralDirectionTcnEvalRow(
                sim_size=6480,
                tail_offset_rounds=216,
                train_size=15000,
                valid_size=3000,
                random_seed=1,
                num_examples=24480,
                feature_dim=10,
                seq_len=16,
                loaded_round_count=26000,
                total_rounds_available=30000,
                bundle_path="b",
                valid_win_rate=0.51,
                test_win_rate=0.53,
            ),
        ]
        aggregates = _aggregate_tcn_rows(rows)
        self.assertEqual(1, len(aggregates))
        self.assertAlmostEqual(0.54, aggregates[0].mean_test_win_rate)
        self.assertAlmostEqual(0.54, aggregates[0].mean_valid_win_rate)


if __name__ == "__main__":
    unittest.main()
