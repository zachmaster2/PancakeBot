from __future__ import annotations

import unittest

from inspection.run_backtest_scenario import _build_parser, _strategy_cfg_with_router_overrides
from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError


class RunBacktestScenarioDirectActionTests(unittest.TestCase):
    def test_direct_action_override_sets_bundle_and_disables_controller(self) -> None:
        cfg = load_app_config("config.toml")
        args = _build_parser().parse_args(
            [
                "--name",
                "demo",
                "--direct-action-enabled",
                "true",
                "--direct-action-model-bundle-path",
                "bundle.pkl.gz",
                "--window-controller-enabled",
                "false",
            ]
        )

        strategy_cfg = _strategy_cfg_with_router_overrides(strategy_cfg=cfg.strategy, args=args)

        self.assertEqual(True, bool(strategy_cfg.direct_action_policy.enabled))
        self.assertEqual("bundle.pkl.gz", str(strategy_cfg.direct_action_policy.model_bundle_path))
        self.assertEqual(False, bool(strategy_cfg.window_controller.enabled))

    def test_direct_action_conflict_with_controller_is_rejected(self) -> None:
        cfg = load_app_config("config.toml")
        args = _build_parser().parse_args(
            [
                "--name",
                "demo",
                "--direct-action-enabled",
                "true",
                "--direct-action-model-bundle-path",
                "bundle.pkl.gz",
            ]
        )

        with self.assertRaises(InvariantError):
            _strategy_cfg_with_router_overrides(strategy_cfg=cfg.strategy, args=args)


if __name__ == "__main__":
    unittest.main()
