from __future__ import annotations

import math
import unittest

from inspection.run_backtest_feature_attribution import _decile_summaries, _summarize_rows


class TestRunBacktestFeatureAttribution(unittest.TestCase):
    def test_summarize_rows_computes_profit_and_win_metrics(self) -> None:
        rows = [
            {"action": "BET", "profit_bnb": 1.2, "ev_bnb": 0.8},
            {"action": "BET", "profit_bnb": -0.4, "ev_bnb": 0.1},
            {"action": "SKIP", "profit_bnb": 0.0, "ev_bnb": 0.0},
        ]

        out = _summarize_rows(rows)

        self.assertEqual(3, out["num_rows"])
        self.assertEqual(2, out["num_bets"])
        self.assertAlmostEqual(2.0 / 3.0, out["bet_rate"])
        self.assertAlmostEqual(0.8, out["net_profit_bnb"])
        self.assertAlmostEqual(0.8 * 500.0 / 3.0, out["profit_per_500_rounds_bnb"])
        self.assertAlmostEqual(0.4, out["avg_profit_per_bet_bnb"])
        self.assertAlmostEqual(1.2, out["gross_profit_bnb"])
        self.assertAlmostEqual(0.4, out["gross_loss_bnb"])
        self.assertAlmostEqual(0.5, out["win_rate"])
        self.assertAlmostEqual(0.45, out["avg_ev_bnb_on_bets"])

    def test_decile_summaries_create_equal_sized_ordered_buckets(self) -> None:
        rows = [
            {"action": "BET", "profit_bnb": float(i), "ev_bnb": 0.0, "feature_x": float(i)}
            for i in range(6)
        ]

        out = _decile_summaries(rows=rows, feature="feature_x", buckets=3)

        self.assertEqual(3, len(out))
        self.assertEqual([1, 2, 3], [int(r["bucket_index"]) for r in out])
        self.assertEqual([2, 2, 2], [int(r["num_rows"]) for r in out])
        self.assertAlmostEqual(0.0, out[0]["feature_min"])
        self.assertAlmostEqual(1.0, out[0]["feature_max"])
        self.assertAlmostEqual(2.0, out[1]["feature_min"])
        self.assertAlmostEqual(3.0, out[1]["feature_max"])
        self.assertAlmostEqual(4.0, out[2]["feature_min"])
        self.assertAlmostEqual(5.0, out[2]["feature_max"])

    def test_decile_summaries_append_non_finite_bucket(self) -> None:
        rows = [
            {"action": "BET", "profit_bnb": 1.0, "ev_bnb": 0.0, "feature_x": 1.0},
            {"action": "SKIP", "profit_bnb": 0.0, "ev_bnb": 0.0, "feature_x": math.nan},
        ]

        out = _decile_summaries(rows=rows, feature="feature_x", buckets=2)

        self.assertEqual(2, len(out))
        self.assertEqual("decile", out[0]["bucket_kind"])
        self.assertEqual("non_finite", out[1]["bucket_kind"])
        self.assertEqual(1, out[1]["num_rows"])
        self.assertEqual(0, out[1]["num_bets"])


if __name__ == "__main__":
    unittest.main()
