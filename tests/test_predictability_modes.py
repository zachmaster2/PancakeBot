from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.schema import FEATURE_SCHEMA
from pancakebot.domain.models.predictability_modes import (
    PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_ONLY,
    PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_PLUS_REGIME,
    PREDICTABILITY_LABEL_MODE_CONTRARIAN_LOG_IMBALANCE_SIDE,
    PREDICTABILITY_LABEL_MODE_CONTRARIAN_PRICE_REGIME_VOTE_15_30_R20_R60_SIDE,
    PREDICTABILITY_LABEL_MODE_EITHER_SIDE_PROFITABLE,
    PREDICTABILITY_LABEL_MODE_PRICE_MOMENTUM_K15_SIDE,
    PREDICTABILITY_LABEL_MODE_REGIME_MAJORITY_R20_SIDE,
    predictability_feature_columns,
    validate_predictability_label_mode,
    validate_predictability_feature_mode,
)
from pancakebot.domain.models.walk_forward import _tradeable_label


class PredictabilityModeTests(unittest.TestCase):
    def test_arrival_microstructure_mode_selects_expected_columns(self) -> None:
        cols = predictability_feature_columns(
            schema=FEATURE_SCHEMA,
            mode=PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_ONLY,
        )
        self.assertEqual(
            (
                "bet_count_w_p_80_to_p_100",
                "bet_rate_w_p_80_to_p_100",
                "bet_sum_w_p_80_to_p_100",
                "bet_top1_share_w_p_80_to_p_100",
                "bet_hhi_w_p_80_to_p_100",
                "log_imb_w_p_80_to_p_100",
                "delta_bet_sum_w_p_80_to_p_100_minus_w_p_40_to_p_80",
            ),
            cols,
        )

    def test_combined_mode_is_limited_to_arrival_and_regime_groups(self) -> None:
        cols = predictability_feature_columns(
            schema=FEATURE_SCHEMA,
            mode=PREDICTABILITY_FEATURE_MODE_ARRIVAL_MICRO_PLUS_REGIME,
        )
        groups_by_name = {str(feature.name): str(feature.group) for feature in FEATURE_SCHEMA.features}
        self.assertIn("bet_count_w_p_80_to_p_100", cols)
        self.assertIn("regime_bull_frac_r_20", cols)
        self.assertTrue(all(groups_by_name[str(col)] in {"arrival_microstructure", "regime"} for col in cols))

    def test_invalid_feature_mode_is_rejected(self) -> None:
        with self.assertRaises(InvariantError):
            validate_predictability_feature_mode("bad_mode")

    def test_invalid_label_mode_is_rejected(self) -> None:
        with self.assertRaises(InvariantError):
            validate_predictability_label_mode("bad_label")

    def test_either_side_profitable_label_uses_both_sides(self) -> None:
        cfg = SimpleNamespace(
            predictability_baseline_bet_bnb=0.05,
            treasury_fee_fraction=0.03,
            predictability_label_mode=PREDICTABILITY_LABEL_MODE_EITHER_SIDE_PROFITABLE,
        )
        x_row = [0.0] * len(FEATURE_SCHEMA.columns)

        with patch("pancakebot.domain.models.walk_forward.settle_bet_against_closed_round") as settle_mock:
            settle_mock.side_effect = lambda **kwargs: SimpleNamespace(
                credit_bnb=(1.0 if str(kwargs["bet_side"]) == "Bear" else 0.0)
            )
            label = _tradeable_label(cfg=cfg, round_t=SimpleNamespace(), x_row=x_row)

        self.assertEqual(1, int(label))
        self.assertEqual(
            ["Bull", "Bear"],
            [str(call.kwargs["bet_side"]) for call in settle_mock.call_args_list],
        )

    def test_baseline_log_imbalance_label_uses_log_imb_sign(self) -> None:
        cfg = SimpleNamespace(
            predictability_baseline_bet_bnb=0.05,
            treasury_fee_fraction=0.03,
            predictability_label_mode="baseline_log_imbalance_side",
        )
        x_row = [0.0] * len(FEATURE_SCHEMA.columns)
        log_imb_idx = int(FEATURE_SCHEMA.columns.index("log_imb_w_p_80_to_p_100"))
        x_row[log_imb_idx] = -1.0

        with patch("pancakebot.domain.models.walk_forward.settle_bet_against_closed_round") as settle_mock:
            settle_mock.return_value = SimpleNamespace(credit_bnb=0.0)
            _tradeable_label(cfg=cfg, round_t=SimpleNamespace(), x_row=x_row)

        self.assertEqual("Bear", str(settle_mock.call_args.kwargs["bet_side"]))

    def test_price_momentum_label_uses_k15_sign(self) -> None:
        cfg = SimpleNamespace(
            predictability_baseline_bet_bnb=0.05,
            treasury_fee_fraction=0.03,
            predictability_label_mode=PREDICTABILITY_LABEL_MODE_PRICE_MOMENTUM_K15_SIDE,
        )
        x_row = [0.0] * len(FEATURE_SCHEMA.columns)
        idx = int(FEATURE_SCHEMA.columns.index("price_log_return_mean_k_15"))
        x_row[idx] = -0.01

        with patch("pancakebot.domain.models.walk_forward.settle_bet_against_closed_round") as settle_mock:
            settle_mock.return_value = SimpleNamespace(credit_bnb=0.0)
            _tradeable_label(cfg=cfg, round_t=SimpleNamespace(), x_row=x_row)

        self.assertEqual("Bear", str(settle_mock.call_args.kwargs["bet_side"]))

    def test_regime_majority_label_uses_r20_bull_frac(self) -> None:
        cfg = SimpleNamespace(
            predictability_baseline_bet_bnb=0.05,
            treasury_fee_fraction=0.03,
            predictability_label_mode=PREDICTABILITY_LABEL_MODE_REGIME_MAJORITY_R20_SIDE,
        )
        x_row = [0.0] * len(FEATURE_SCHEMA.columns)
        idx = int(FEATURE_SCHEMA.columns.index("regime_bull_frac_r_20"))
        x_row[idx] = 0.3

        with patch("pancakebot.domain.models.walk_forward.settle_bet_against_closed_round") as settle_mock:
            settle_mock.return_value = SimpleNamespace(credit_bnb=0.0)
            _tradeable_label(cfg=cfg, round_t=SimpleNamespace(), x_row=x_row)

        self.assertEqual("Bear", str(settle_mock.call_args.kwargs["bet_side"]))

    def test_contrarian_log_imbalance_label_inverts_log_imb_sign(self) -> None:
        cfg = SimpleNamespace(
            predictability_baseline_bet_bnb=0.05,
            treasury_fee_fraction=0.03,
            predictability_label_mode=PREDICTABILITY_LABEL_MODE_CONTRARIAN_LOG_IMBALANCE_SIDE,
        )
        x_row = [0.0] * len(FEATURE_SCHEMA.columns)
        log_imb_idx = int(FEATURE_SCHEMA.columns.index("log_imb_w_p_80_to_p_100"))
        x_row[log_imb_idx] = -1.0

        with patch("pancakebot.domain.models.walk_forward.settle_bet_against_closed_round") as settle_mock:
            settle_mock.return_value = SimpleNamespace(credit_bnb=0.0)
            _tradeable_label(cfg=cfg, round_t=SimpleNamespace(), x_row=x_row)

        self.assertEqual("Bull", str(settle_mock.call_args.kwargs["bet_side"]))

    def test_contrarian_price_regime_vote_label_inverts_consensus_side(self) -> None:
        cfg = SimpleNamespace(
            predictability_baseline_bet_bnb=0.05,
            treasury_fee_fraction=0.03,
            predictability_label_mode=PREDICTABILITY_LABEL_MODE_CONTRARIAN_PRICE_REGIME_VOTE_15_30_R20_R60_SIDE,
        )
        x_row = [0.0] * len(FEATURE_SCHEMA.columns)
        x_row[int(FEATURE_SCHEMA.columns.index("price_log_return_mean_k_15"))] = 0.01
        x_row[int(FEATURE_SCHEMA.columns.index("price_log_return_mean_k_30"))] = 0.01
        x_row[int(FEATURE_SCHEMA.columns.index("regime_bull_frac_r_20"))] = 0.8
        x_row[int(FEATURE_SCHEMA.columns.index("regime_bull_frac_r_60"))] = 0.7

        with patch("pancakebot.domain.models.walk_forward.settle_bet_against_closed_round") as settle_mock:
            settle_mock.return_value = SimpleNamespace(credit_bnb=0.0)
            _tradeable_label(cfg=cfg, round_t=SimpleNamespace(), x_row=x_row)

        self.assertEqual("Bear", str(settle_mock.call_args.kwargs["bet_side"]))


if __name__ == "__main__":
    unittest.main()
