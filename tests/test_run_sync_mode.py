from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import run


class RunSyncModeTests(unittest.TestCase):
    def test_parse_args_accepts_sync(self) -> None:
        args = run._parse_args(["--sync"])
        self.assertTrue(bool(args.sync))
        self.assertFalse(bool(args.dry))
        self.assertFalse(bool(args.backtest))

    def test_parse_args_rejects_multiple_modes(self) -> None:
        with self.assertRaises(SystemExit):
            run._parse_args(["--dry", "--sync"])

    def test_run_from_config_sync_dispatches_without_private_key(self) -> None:
        fake_cfg = SimpleNamespace(
            closed_rounds_path="var/closed_rounds.jsonl",
            klines_path="var/klines.jsonl",
            market_data_db_path="var/market_data.sqlite",
            abi_json_path="abi/predictionv2.json",
            cutoff_seconds=12,
            latency_log_path="var/latency.log",
            dry_initial_bankroll_bnb=50.0,
            wait_for_bet_receipt=False,
            bet_receipt_timeout_seconds=30,
            runtime_state_paths=SimpleNamespace(
                claim_scan_cursor_path="var/runtime/claim_scan_cursor.txt",
                dry_bets_path="var/runtime/dry_bets.jsonl",
                dry_settled_epochs_path="var/runtime/dry_settled_epochs.txt",
                dry_audit_trades_path="var/runtime/dry_audit_trades.csv",
                dry_cycle_audit_path="var/runtime/dry_cycle_audit.csv",
                dry_bankroll_state_path="var/runtime/dry_bankroll_state.json",
                dry_pipeline_bootstrap_state_path="var/runtime/dry_pipeline.pkl.gz",
                live_pipeline_bootstrap_state_path="var/runtime/live_pipeline.pkl.gz",
            ),
            momentum_gate=MagicMock(),
            backtest=MagicMock(),
        )
        with (
            patch("pancakebot.integration.app.load_app_config", return_value=fake_cfg),
            patch("pancakebot.integration.app.ClosedRoundsStore"),
            patch("pancakebot.integration.app.load_env"),
            patch("pancakebot.integration.app.require_env", return_value="graph-key") as require_env_mock,
            patch("pancakebot.integration.app.GraphClient"),
            patch("pancakebot.integration.app.sync_runtime_market_data") as sync_mock,
        ):
            from pancakebot.integration.app import run_from_config

            sync_mock.return_value = MagicMock(
                stored_closed_round_count=100,
                earliest_closed_epoch=1,
                latest_closed_epoch=100,
                klines_total=288000,
                klines_appended=10,
            )
            run_from_config(config_path="config.toml", dry=False, backtest=False, sync=True)

        require_env_mock.assert_called_once_with("THE_GRAPH_API_KEY")
        sync_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
