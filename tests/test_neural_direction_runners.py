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
from inspection.run_neural_direction_policy_eval import (
    NeuralDirectionPolicyEvalRow,
    _aggregate_rows as _aggregate_policy_rows,
)
from inspection.run_neural_direction_confidence_eval import (
    NeuralDirectionConfidenceEvalRow,
    _aggregate_rows as _aggregate_confidence_rows,
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

    def test_confidence_aggregate_rows_keep_training_policy(self) -> None:
        rows = [
            NeuralDirectionConfidenceEvalRow(
                model_type="mlp",
                source_rows_csv="rows_a.csv",
                source_bundle_path="bundle_a.pt",
                sim_size=6480,
                tail_offset_rounds=0,
                training_policy="pretrain_finetune",
                train_size=100000,
                pretrain_size=300000,
                valid_size=3000,
                recency_half_life_examples=None,
                seq_len=None,
                coverage_fraction_requested=0.1,
                selected_count=648,
                selected_fraction_actual=0.1,
                selected_win_rate=0.54,
                selected_mean_confidence=0.53,
                selected_min_confidence=0.52,
                selected_max_confidence=0.61,
                overall_test_win_rate=0.515,
                overall_test_mean_confidence=0.507,
                calibration_temperature=1.02,
                calibration_valid_loss_before=0.69,
                calibration_valid_loss_after=0.68,
            ),
            NeuralDirectionConfidenceEvalRow(
                model_type="mlp",
                source_rows_csv="rows_b.csv",
                source_bundle_path="bundle_b.pt",
                sim_size=6480,
                tail_offset_rounds=432,
                training_policy="pretrain_finetune",
                train_size=100000,
                pretrain_size=300000,
                valid_size=3000,
                recency_half_life_examples=None,
                seq_len=None,
                coverage_fraction_requested=0.1,
                selected_count=648,
                selected_fraction_actual=0.1,
                selected_win_rate=0.56,
                selected_mean_confidence=0.54,
                selected_min_confidence=0.53,
                selected_max_confidence=0.62,
                overall_test_win_rate=0.517,
                overall_test_mean_confidence=0.508,
                calibration_temperature=1.01,
                calibration_valid_loss_before=0.69,
                calibration_valid_loss_after=0.68,
            ),
        ]
        aggregates = _aggregate_confidence_rows(rows)
        self.assertEqual(1, len(aggregates))
        self.assertEqual("pretrain_finetune", aggregates[0].training_policy)
        self.assertEqual(300000, aggregates[0].pretrain_size)
        self.assertAlmostEqual(0.55, aggregates[0].mean_selected_win_rate)

    def test_policy_aggregate_rows(self) -> None:
        rows = [
            NeuralDirectionPolicyEvalRow(
                source_rows_csv="rows.csv",
                source_bundle_path="bundle.pt",
                training_policy="flat",
                sim_size=6480,
                tail_offset_rounds=0,
                train_size=400000,
                pretrain_size=0,
                valid_size=3000,
                target_coverage_fraction=0.05,
                threshold_used=0.537,
                bet_size_bnb=0.1,
                num_rounds=6480,
                num_bets=320,
                num_wins=180,
                num_skips_below_threshold=6160,
                num_skips_insufficient_bankroll=0,
                bet_rate=0.049,
                win_rate=0.5625,
                net_profit_bnb=1.2,
                profit_per_500_bnb=0.09259,
                max_drawdown_bnb=0.8,
                final_bankroll_bnb=51.2,
                selected_mean_confidence=0.55,
                selected_min_confidence=0.537,
                selected_max_confidence=0.65,
            ),
            NeuralDirectionPolicyEvalRow(
                source_rows_csv="rows.csv",
                source_bundle_path="bundle.pt",
                training_policy="flat",
                sim_size=6480,
                tail_offset_rounds=432,
                train_size=400000,
                pretrain_size=0,
                valid_size=3000,
                target_coverage_fraction=0.05,
                threshold_used=0.538,
                bet_size_bnb=0.1,
                num_rounds=6480,
                num_bets=330,
                num_wins=170,
                num_skips_below_threshold=6150,
                num_skips_insufficient_bankroll=0,
                bet_rate=0.051,
                win_rate=0.515,
                net_profit_bnb=0.6,
                profit_per_500_bnb=0.0463,
                max_drawdown_bnb=1.0,
                final_bankroll_bnb=50.6,
                selected_mean_confidence=0.551,
                selected_min_confidence=0.538,
                selected_max_confidence=0.66,
            ),
        ]
        aggregates = _aggregate_policy_rows(rows)
        self.assertEqual(1, len(aggregates))
        self.assertEqual("flat", aggregates[0].training_policy)
        self.assertAlmostEqual(0.5375, aggregates[0].mean_threshold_used)
        self.assertAlmostEqual((0.09259 + 0.0463) / 2.0, aggregates[0].mean_profit_per_500_bnb)


if __name__ == "__main__":
    unittest.main()
