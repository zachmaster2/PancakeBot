"""Tests for the promoted shared baseline and runtime state path config."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError

_ALT_A_NAME = "disloc_altA_20260227_x80"
_ALT_B_NAME = "disloc_altB_20260227_x80"


class LoadConfigBaselineDefaultTests(unittest.TestCase):
    def test_current_config_matches_promoted_online_selector_baseline(self) -> None:
        cfg = load_app_config("config.toml")

        self.assertEqual("online_selector_score_fallback", cfg.strategy.router.mode)
        self.assertAlmostEqual(0.008, float(cfg.strategy.router.online_score_threshold_bnb))
        self.assertEqual(
            "var/runtime/claim_scan_cursor.txt",
            cfg.runtime_state_paths.claim_scan_cursor_path,
        )
        self.assertEqual("var/runtime/dry_bets.jsonl", cfg.runtime_state_paths.dry_bets_path)
        self.assertEqual(
            "var/runtime/dry_settled_epochs.txt",
            cfg.runtime_state_paths.dry_settled_epochs_path,
        )
        self.assertEqual(
            "var/runtime/dry_audit_trades.csv",
            cfg.runtime_state_paths.dry_audit_trades_path,
        )
        self.assertEqual(
            "var/runtime/dry_bankroll_state.json",
            cfg.runtime_state_paths.dry_bankroll_state_path,
        )
        self.assertEqual(
            "var/runtime/dry_pipeline_bootstrap_state.pkl.gz",
            cfg.runtime_state_paths.dry_pipeline_bootstrap_state_path,
        )
        self.assertEqual(
            "var/runtime/live_pipeline_bootstrap_state.pkl.gz",
            cfg.runtime_state_paths.live_pipeline_bootstrap_state_path,
        )
        self.assertAlmostEqual(50.0, float(cfg.dry_initial_bankroll_bnb or 0.0))

        candidates = {str(c.name): c for c in cfg.strategy.dislocation.candidates}
        self.assertEqual([_ALT_A_NAME, _ALT_B_NAME], list(candidates.keys()))

        alt_a = candidates[_ALT_A_NAME]
        self.assertEqual("projected_final_model_only", alt_a.pool_total_gate_mode)
        self.assertAlmostEqual(0.5, float(alt_a.projected_final_pool_total_min_bnb))
        self.assertAlmostEqual(0.02, float(alt_a.market_extreme_min))
        self.assertTrue(bool(alt_a.late_model_veto_enabled))
        self.assertAlmostEqual(0.05, float(alt_a.late_model_veto_min_late_ratio))
        self.assertAlmostEqual(0.10, float(alt_a.late_model_veto_min_abs_imbalance))
        self.assertAlmostEqual(0.01, float(alt_a.bear_expected_net_extra_min_bnb))

        alt_b = candidates[_ALT_B_NAME]
        self.assertEqual("projected_final_model_only", alt_b.pool_total_gate_mode)
        self.assertAlmostEqual(0.5, float(alt_b.projected_final_pool_total_min_bnb))
        self.assertAlmostEqual(0.02, float(alt_b.market_extreme_min))
        self.assertTrue(bool(alt_b.late_model_veto_enabled))
        self.assertAlmostEqual(0.05, float(alt_b.late_model_veto_min_late_ratio))
        self.assertAlmostEqual(0.10, float(alt_b.late_model_veto_min_abs_imbalance))
        self.assertAlmostEqual(0.0, float(alt_b.bear_expected_net_extra_min_bnb))

    def test_runtime_state_paths_can_be_overridden(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = (
            str(base_text)
            .replace(
                'claim_scan_cursor_path = "var/runtime/claim_scan_cursor.txt"',
                'claim_scan_cursor_path = "var/custom/claim_cursor.txt"',
                1,
            )
            .replace(
                'dry_bets_path = "var/runtime/dry_bets.jsonl"',
                'dry_bets_path = "var/custom/dry_bets.jsonl"',
                1,
            )
            .replace(
                'dry_settled_epochs_path = "var/runtime/dry_settled_epochs.txt"',
                'dry_settled_epochs_path = "var/custom/dry_settled_epochs.txt"',
                1,
            )
            .replace(
                'dry_audit_trades_path = "var/runtime/dry_audit_trades.csv"',
                'dry_audit_trades_path = "var/custom/dry_audit_trades.csv"',
                1,
            )
            .replace(
                'dry_bankroll_state_path = "var/runtime/dry_bankroll_state.json"',
                'dry_bankroll_state_path = "var/custom/dry_bankroll_state.json"',
                1,
            )
            .replace(
                'dry_pipeline_bootstrap_state_path = "var/runtime/dry_pipeline_bootstrap_state.pkl.gz"',
                'dry_pipeline_bootstrap_state_path = "var/custom/dry_pipeline_bootstrap_state.pkl.gz"',
                1,
            )
            .replace(
                'live_pipeline_bootstrap_state_path = "var/runtime/live_pipeline_bootstrap_state.pkl.gz"',
                'live_pipeline_bootstrap_state_path = "var/custom/live_pipeline_bootstrap_state.pkl.gz"',
                1,
            )
            .replace(
                'dry_initial_bankroll_bnb = 50.0',
                'dry_initial_bankroll_bnb = 42.5',
                1,
            )
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_runtime_paths.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            cfg = load_app_config(str(cfg_path))

        self.assertEqual("var/custom/claim_cursor.txt", cfg.runtime_state_paths.claim_scan_cursor_path)
        self.assertEqual("var/custom/dry_bets.jsonl", cfg.runtime_state_paths.dry_bets_path)
        self.assertEqual(
            "var/custom/dry_settled_epochs.txt",
            cfg.runtime_state_paths.dry_settled_epochs_path,
        )
        self.assertEqual(
            "var/custom/dry_audit_trades.csv",
            cfg.runtime_state_paths.dry_audit_trades_path,
        )
        self.assertEqual(
            "var/custom/dry_bankroll_state.json",
            cfg.runtime_state_paths.dry_bankroll_state_path,
        )
        self.assertEqual(
            "var/custom/dry_pipeline_bootstrap_state.pkl.gz",
            cfg.runtime_state_paths.dry_pipeline_bootstrap_state_path,
        )
        self.assertEqual(
            "var/custom/live_pipeline_bootstrap_state.pkl.gz",
            cfg.runtime_state_paths.live_pipeline_bootstrap_state_path,
        )
        self.assertAlmostEqual(42.5, float(cfg.dry_initial_bankroll_bnb or 0.0))

    def test_unknown_paths_key_is_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace(
            'dry_audit_trades_path = "var/runtime/dry_audit_trades.csv"',
            (
                'dry_audit_trades_path = "var/runtime/dry_audit_trades.csv"\n'
                'unexpected_path_key = "var/runtime/unexpected.txt"'
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_bad_paths.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_runtime_state_paths_must_be_distinct(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace(
            'dry_settled_epochs_path = "var/runtime/dry_settled_epochs.txt"',
            'dry_settled_epochs_path = "var/runtime/dry_bets.jsonl"',
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_duplicate_runtime_paths.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_negative_dry_initial_bankroll_is_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = str(base_text).replace(
            'dry_initial_bankroll_bnb = 50.0',
            'dry_initial_bankroll_bnb = -1.0',
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_negative_dry_bankroll.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))


if __name__ == "__main__":
    unittest.main()
