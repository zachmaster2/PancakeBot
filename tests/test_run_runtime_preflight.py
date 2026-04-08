from __future__ import annotations

import unittest

from preflight import collect_preflight_checks


class RuntimePreflightTests(unittest.TestCase):
    def test_collect_preflight_checks_passes_for_current_config_without_env(self) -> None:
        cfg, checks = collect_preflight_checks(
            config_path="config.toml",
            check_env=False,
            env={},
        )

        self.assertTrue(bool(cfg.momentum_gate.enabled))
        self.assertTrue(all(bool(check.passed) for check in checks))
        check_names = {check.name for check in checks}
        self.assertIn("dry_cycle_audit_parent", check_names)

    def test_collect_preflight_checks_reports_missing_env_when_requested(self) -> None:
        _cfg, checks = collect_preflight_checks(
            config_path="config.toml",
            check_env=True,
            env={},
        )

        env_checks = {check.name: check for check in checks if check.name.startswith("env:")}
        self.assertNotIn("env:THE_GRAPH_API_KEY", env_checks)
        self.assertEqual(False, bool(env_checks["env:BSC_WALLET_PRIVATE_KEY"].passed))


if __name__ == "__main__":
    unittest.main()
