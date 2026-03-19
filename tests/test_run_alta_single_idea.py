from __future__ import annotations

import unittest

from inspection.run_alta_single_idea import (
    _apply_candidate_overrides,
    _apply_router_overrides,
    _candidate_pool,
    _parse_candidate_specific_overrides,
)
from inspection.backtest_harness_common import load_cfg
from pancakebot.core.errors import InvariantError


class TestRunAltaSingleIdea(unittest.TestCase):
    def test_candidate_pool_active_matches_loaded_strategy_candidates(self) -> None:
        cfg = load_cfg(config_path="config.toml")

        out = _candidate_pool(cfg=cfg, config_path="config.toml", candidate_source="active")

        self.assertEqual(
            [
                "disloc_altA_20260227_x80",
                "disloc_altB_20260227_x80",
                "disloc_altC_20260319_recent",
            ],
            [str(c.name) for c in out],
        )

    def test_candidate_pool_all_config_includes_inactive_candidates(self) -> None:
        cfg = load_cfg(config_path="config.toml")

        out = _candidate_pool(cfg=cfg, config_path="config.toml", candidate_source="all_config")
        names = {str(c.name) for c in out}

        self.assertIn("disloc_altA_20260227_x80", names)
        self.assertIn("disloc_altB_20260227_x80", names)
        self.assertIn("disloc_altC_20260319_recent", names)
        self.assertIn(
            "disloc_stageH_sidenowcast_when_market_disagree_perfflip_w80_h40_wr0p5_mnm0p001_x80",
            names,
        )
        self.assertIn("disloc_stageB_side_adaptive_shadow_ev0p146_skip_w80_h40_wr0p52_mn0p0_x80", names)
        self.assertGreater(len(out), len(cfg.strategy.dislocation.candidates))

    def test_router_overrides_apply_known_fields(self) -> None:
        cfg = load_cfg(config_path="config.toml")

        out = _apply_router_overrides(
            base_router=cfg.strategy.router,
            router_mode="online_cellmean_selector_fallback",
            router_score_threshold_bnb=0.01,
            router_overrides={
                "online_min_cell_obs": 7,
                "online_num_quantile_bins": 6,
            },
        )

        self.assertEqual("online_cellmean_selector_fallback", str(out.mode))
        self.assertEqual(0.01, float(out.score_threshold_bnb))
        self.assertEqual(7, int(out.online_min_cell_obs))
        self.assertEqual(6, int(out.online_num_quantile_bins))

    def test_router_overrides_reject_unknown_fields(self) -> None:
        cfg = load_cfg(config_path="config.toml")

        with self.assertRaises(InvariantError):
            _apply_router_overrides(
                base_router=cfg.strategy.router,
                router_mode="selector_max_score",
                router_score_threshold_bnb=None,
                router_overrides={"not_a_router_field": 1},
            )

    def test_candidate_specific_overrides_parse(self) -> None:
        out = _parse_candidate_specific_overrides(
            [
                "disloc_altA_20260227_x80:allowed_sides=bull_only",
                "disloc_altA_20260227_x80:market_extreme_min=0.02",
                "disloc_altB_20260227_x80:adaptive_candidate_modes=[\"market_follow\"]",
            ]
        )

        self.assertEqual("bull_only", out["disloc_altA_20260227_x80"]["allowed_sides"])
        self.assertEqual(0.02, out["disloc_altA_20260227_x80"]["market_extreme_min"])
        self.assertEqual(
            ["market_follow"],
            out["disloc_altB_20260227_x80"]["adaptive_candidate_modes"],
        )

    def test_candidate_specific_overrides_parse_bare_list_tokens(self) -> None:
        out = _parse_candidate_specific_overrides(
            [
                "disloc_altB_20260227_x80:adaptive_candidate_modes=[nowcast_when_market_disagree,market_follow,ev_max]",
            ]
        )

        self.assertEqual(
            ["nowcast_when_market_disagree", "market_follow", "ev_max"],
            out["disloc_altB_20260227_x80"]["adaptive_candidate_modes"],
        )

    def test_candidate_specific_overrides_reject_missing_candidate_separator(self) -> None:
        with self.assertRaises(InvariantError):
            _parse_candidate_specific_overrides(["allowed_sides=bull_only"])

    def test_candidate_specific_overrides_layer_on_top_of_global_apply_all(self) -> None:
        cfg = load_cfg(config_path="config.toml")
        by_name = {str(c.name): c for c in cfg.strategy.dislocation.candidates}

        global_overrides = {
            "market_extreme_min": 0.02,
            "late_model_veto_enabled": True,
            "late_model_veto_min_late_ratio": 0.05,
            "late_model_veto_min_abs_imbalance": 0.10,
        }
        alt_a = _apply_candidate_overrides(
            base_candidate=by_name["disloc_altA_20260227_x80"],
            overrides=global_overrides,
            stake_scale=1.0,
        )
        alt_a = _apply_candidate_overrides(
            base_candidate=alt_a,
            overrides={"allowed_sides": "bull_only"},
            stake_scale=1.0,
        )
        alt_b = _apply_candidate_overrides(
            base_candidate=by_name["disloc_altB_20260227_x80"],
            overrides=global_overrides,
            stake_scale=1.0,
        )

        self.assertEqual("bull_only", str(alt_a.allowed_sides))
        self.assertEqual(0.02, float(alt_a.market_extreme_min))
        self.assertTrue(bool(alt_a.late_model_veto_enabled))
        self.assertEqual("both", str(alt_b.allowed_sides))
        self.assertEqual(0.02, float(alt_b.market_extreme_min))
        self.assertTrue(bool(alt_b.late_model_veto_enabled))


if __name__ == "__main__":
    unittest.main()
