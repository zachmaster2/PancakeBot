from __future__ import annotations

import unittest

from inspection.run_alta_single_idea import _candidate_pool
from inspection.backtest_harness_common import load_cfg


class TestRunAltaSingleIdea(unittest.TestCase):
    def test_candidate_pool_active_matches_loaded_strategy_candidates(self) -> None:
        cfg = load_cfg(config_path="config.toml")

        out = _candidate_pool(cfg=cfg, config_path="config.toml", candidate_source="active")

        self.assertEqual(
            ["disloc_altA_20260227_x80", "disloc_altB_20260227_x80"],
            [str(c.name) for c in out],
        )

    def test_candidate_pool_all_config_includes_inactive_candidates(self) -> None:
        cfg = load_cfg(config_path="config.toml")

        out = _candidate_pool(cfg=cfg, config_path="config.toml", candidate_source="all_config")
        names = {str(c.name) for c in out}

        self.assertIn("disloc_altA_20260227_x80", names)
        self.assertIn("disloc_altB_20260227_x80", names)
        self.assertIn(
            "disloc_stageH_sidenowcast_when_market_disagree_perfflip_w80_h40_wr0p5_mnm0p001_x80",
            names,
        )
        self.assertIn("disloc_stageB_side_adaptive_shadow_ev0p146_skip_w80_h40_wr0p52_mn0p0_x80", names)
        self.assertGreater(len(out), len(cfg.strategy.dislocation.candidates))


if __name__ == "__main__":
    unittest.main()
