"""Tests for the promoted shared baseline and runtime state path config."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pancakebot.config.load_config import load_app_config
from pancakebot.config.strategy_config import (
    DislocationSelectorConfig,
    FlowCandidateConfig,
    StrategyRouterConfig as StrategyConfigRouterConfig,
    WindowControllerConfig,
)
from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.pipeline import required_pipeline_warmup_rounds
from pancakebot.domain.strategy.router import StrategyRouterConfig as DomainRouterConfig

_STAGE_B_BULL_NAME = "disloc_stageB_bullonly_recent8pct_v1"
_CONS_NAME = "disloc_cons_20260227_x80"
_FLOW_NAME = "flow_lgbm_recent_t12k_r1k_regime40_v1"


class LoadConfigBaselineDefaultTests(unittest.TestCase):
    def test_current_config_matches_contained_selector_max_stageb_runtime(self) -> None:
        cfg = load_app_config("config.toml")

        self.assertEqual("selector_max_score", cfg.strategy.router.mode)
        self.assertEqual(10000, int(cfg.strategy.dislocation.selector.warmup_rounds))
        self.assertEqual(10000, int(cfg.strategy.router.online_warmup_rounds))
        self.assertEqual(10000, int(required_pipeline_warmup_rounds(strategy_cfg=cfg.strategy)))
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
            "var/runtime/dry_cycle_audit.csv",
            cfg.runtime_state_paths.dry_cycle_audit_path,
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
        self.assertEqual(False, bool(cfg.strategy.flow_candidate.enabled))
        self.assertEqual(_FLOW_NAME, cfg.strategy.flow_candidate.name)
        self.assertEqual("both", str(cfg.strategy.flow_candidate.allowed_sides))
        self.assertAlmostEqual(0.0, float(cfg.strategy.flow_candidate.selector_score_penalty_bnb))
        self.assertEqual(12000, int(cfg.strategy.flow_candidate.train_size))
        self.assertEqual(1000, int(cfg.strategy.flow_candidate.retrain_interval))
        self.assertAlmostEqual(0.0025, float(cfg.strategy.flow_candidate.ev_threshold))
        self.assertAlmostEqual(1.0, float(cfg.strategy.flow_candidate.min_total_pool_c))
        self.assertEqual(40, int(cfg.strategy.flow_candidate.roll_window))
        self.assertAlmostEqual(0.48, float(cfg.strategy.flow_candidate.roll_winrate_min))
        self.assertEqual(40, int(cfg.strategy.flow_candidate.cooldown_trades))
        self.assertEqual(False, bool(cfg.strategy.window_controller.enabled))
        self.assertEqual("trailing_best_vs_baseline", str(cfg.strategy.window_controller.mode))
        self.assertEqual(_STAGE_B_BULL_NAME, str(cfg.strategy.window_controller.baseline_profile_name))
        self.assertEqual(
            _CONS_NAME,
            str(cfg.strategy.window_controller.alternate_profile_name),
        )
        self.assertEqual(216, int(cfg.strategy.window_controller.window_rounds))
        self.assertEqual(3, int(cfg.strategy.window_controller.lookback_windows))
        self.assertAlmostEqual(1.0, float(cfg.strategy.window_controller.margin_per_500))
        self.assertAlmostEqual(0.0, float(cfg.strategy.window_controller.skip_threshold_per_500))

        candidates = {str(c.name): c for c in cfg.strategy.dislocation.candidates}
        self.assertEqual([_STAGE_B_BULL_NAME], list(candidates.keys()))

        stage_b_bull = candidates[_STAGE_B_BULL_NAME]
        self.assertEqual("cutoff_only", stage_b_bull.pool_total_gate_mode)
        self.assertAlmostEqual(0.02, float(stage_b_bull.expected_net_min_bnb))
        self.assertAlmostEqual(0.01, float(stage_b_bull.bull_expected_net_extra_min_bnb))
        self.assertEqual("adaptive_shadow", str(stage_b_bull.side_selection_mode))
        self.assertEqual("bull_only", str(stage_b_bull.allowed_sides))
        self.assertAlmostEqual(0.6, float(stage_b_bull.cutoff_pool_total_min_bnb))
        self.assertEqual(
            ("nowcast_when_market_disagree", "ev_max", "nowcast_contra"),
            tuple(str(item) for item in stage_b_bull.adaptive_candidate_modes),
        )
        self.assertEqual("off", str(stage_b_bull.perf_adapt_mode))
        self.assertAlmostEqual(0.1, float(stage_b_bull.fixed_bet_bnb))

    def test_code_defaults_match_contained_stageb_runtime(self) -> None:
        selector_defaults = DislocationSelectorConfig()
        config_router_defaults = StrategyConfigRouterConfig()
        domain_router_defaults = DomainRouterConfig()
        flow_defaults = FlowCandidateConfig()
        window_controller_defaults = WindowControllerConfig()

        self.assertEqual(10000, int(selector_defaults.warmup_rounds))

        self.assertEqual("selector_max_score", str(config_router_defaults.mode))
        self.assertEqual(10000, int(config_router_defaults.online_warmup_rounds))
        self.assertAlmostEqual(0.008, float(config_router_defaults.online_score_threshold_bnb))
        self.assertEqual(False, bool(config_router_defaults.online_use_direction_split))

        self.assertEqual("selector_max_score", str(domain_router_defaults.mode))
        self.assertEqual(10000, int(domain_router_defaults.online_warmup_rounds))
        self.assertAlmostEqual(0.008, float(domain_router_defaults.online_score_threshold_bnb))
        self.assertEqual(False, bool(domain_router_defaults.online_use_direction_split))

        self.assertEqual(False, bool(flow_defaults.enabled))
        self.assertEqual(_FLOW_NAME, str(flow_defaults.name))
        self.assertEqual("both", str(flow_defaults.allowed_sides))
        self.assertAlmostEqual(0.0, float(flow_defaults.selector_score_penalty_bnb))
        self.assertEqual(12000, int(flow_defaults.train_size))
        self.assertEqual(1000, int(flow_defaults.retrain_interval))
        self.assertAlmostEqual(0.0025, float(flow_defaults.ev_threshold))
        self.assertAlmostEqual(1.0, float(flow_defaults.min_total_pool_c))
        self.assertEqual(40, int(flow_defaults.roll_window))
        self.assertAlmostEqual(0.48, float(flow_defaults.roll_winrate_min))
        self.assertEqual(40, int(flow_defaults.cooldown_trades))
        self.assertEqual(False, bool(window_controller_defaults.enabled))
        self.assertEqual("trailing_best_vs_baseline", str(window_controller_defaults.mode))
        self.assertEqual(_STAGE_B_BULL_NAME, str(window_controller_defaults.baseline_profile_name))
        self.assertEqual(
            _CONS_NAME,
            str(window_controller_defaults.alternate_profile_name),
        )
        self.assertEqual(216, int(window_controller_defaults.window_rounds))
        self.assertEqual(3, int(window_controller_defaults.lookback_windows))
        self.assertAlmostEqual(1.0, float(window_controller_defaults.margin_per_500))
        self.assertAlmostEqual(0.0, float(window_controller_defaults.skip_threshold_per_500))

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
                'dry_cycle_audit_path = "var/runtime/dry_cycle_audit.csv"',
                'dry_cycle_audit_path = "var/custom/dry_cycle_audit.csv"',
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
            "var/custom/dry_cycle_audit.csv",
            cfg.runtime_state_paths.dry_cycle_audit_path,
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
