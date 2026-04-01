from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pancakebot.domain.strategy.direct_action_policy_model import (
    DirectActionModelBundle,
    action_spec_by_id,
    direct_action_score_values,
    direct_action_feature_names,
    default_direct_action_specs,
    load_direct_action_bundle,
    realized_net_bnb_for_action,
    save_direct_action_bundle,
    summarize_top_action_predictions,
)
from pancakebot.domain.types import Round


class _ConstantModel:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def predict(self, x) -> list[float]:
        return list(self._values[: len(x)])


def _closed_round(*, epoch: int, position: str) -> Round:
    return Round(
        epoch=int(epoch),
        start_at=1_000_000 + int(epoch) * 300,
        lock_at=1_000_000 + int(epoch) * 300 + 300,
        close_at=1_000_000 + int(epoch) * 300 + 600,
        lock_price=600.0,
        close_price=601.0 if str(position) == "Bull" else 599.0,
        position=str(position),
        failed=False,
        bets=(),
    )


class DirectActionPolicyModelTests(unittest.TestCase):
    def test_default_direct_action_specs_include_skip_and_bidirectional_sizes(self) -> None:
        specs = default_direct_action_specs()

        self.assertEqual("skip", specs[0].action_id)
        self.assertEqual("SKIP", specs[0].action)
        self.assertEqual(13, len(specs))
        self.assertEqual("bull_0p05", specs[1].action_id)
        self.assertEqual("bear_0p50", specs[-1].action_id)

    def test_direct_action_feature_names_include_per_action_indicators(self) -> None:
        feature_names = direct_action_feature_names(action_specs=default_direct_action_specs()[:3])

        self.assertIn("action_id_is_skip", feature_names)
        self.assertIn("action_id_is_bull_0p05", feature_names)
        self.assertIn("action_id_is_bear_0p05", feature_names)

    def test_realized_net_bnb_for_action_skip_is_zero(self) -> None:
        spec = action_spec_by_id(default_direct_action_specs(), "skip")
        net = realized_net_bnb_for_action(
            action_spec=spec,
            round_closed=_closed_round(epoch=1, position="Bull"),
            treasury_fee_fraction=0.03,
        )
        self.assertAlmostEqual(0.0, net)

    def test_bundle_round_trip_preserves_predictions(self) -> None:
        specs = default_direct_action_specs()[:3]
        bundle = DirectActionModelBundle(
            feature_names=("f1", "f2"),
            action_specs=tuple(specs),
            q10_model=_ConstantModel([0.1, -0.2]),
            q50_model=_ConstantModel([0.3, 0.4]),
            metadata={"required_history_rounds": 15061},
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bundle.pkl.gz"
            save_direct_action_bundle(bundle=bundle, path=str(path))
            loaded = load_direct_action_bundle(str(path))

        q10, q50 = loaded.predict_quantiles([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual([0.1, -0.2], q10)
        self.assertEqual([0.3, 0.4], q50)

    def test_bundle_predict_quantiles_supports_per_action_models(self) -> None:
        specs = default_direct_action_specs()[:3]
        bundle = DirectActionModelBundle(
            feature_names=("f1", "f2"),
            action_specs=tuple(specs),
            q10_model={
                "skip": _ConstantModel([0.0]),
                "bull_0p05": _ConstantModel([0.1]),
                "bear_0p05": _ConstantModel([-0.2]),
            },
            q50_model={
                "skip": _ConstantModel([0.0]),
                "bull_0p05": _ConstantModel([0.3]),
                "bear_0p05": _ConstantModel([0.4]),
            },
            metadata={"required_history_rounds": 216, "model_layout": "per_action"},
        )

        q10, q50 = bundle.predict_quantiles(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            action_ids=["skip", "bull_0p05", "bear_0p05"],
        )

        self.assertEqual([0.0, 0.1, -0.2], q10)
        self.assertEqual([0.0, 0.3, 0.4], q50)

    def test_direct_action_score_values_supports_q50_minus_lambda_spread(self) -> None:
        scores = direct_action_score_values(
            q10_values=[-0.10, 0.00, 0.02],
            q50_values=[0.10, 0.03, 0.04],
            score_mode="q50_minus_lambda_spread",
            score_risk_lambda=0.25,
        )

        self.assertEqual(3, len(scores))
        self.assertAlmostEqual(0.05, scores[0])
        self.assertAlmostEqual(0.0225, scores[1])
        self.assertAlmostEqual(0.035, scores[2])

    def test_summarize_top_action_predictions_sorts_by_score_then_q10_then_q50(self) -> None:
        specs = default_direct_action_specs()[:3]
        summary = summarize_top_action_predictions(
            action_specs=specs,
            score_values=[0.0, 0.015, 0.015],
            q10_values=[0.0, 0.01, 0.01],
            q50_values=[0.0, 0.02, 0.03],
            top_k=2,
        )

        self.assertIn('"score_bnb":0.015', summary)
        self.assertIn('"action_id":"bear_0p05"', summary)
        self.assertIn('"action_id":"bull_0p05"', summary)
        self.assertNotIn('"action_id":"skip"', summary)


if __name__ == "__main__":
    unittest.main()
