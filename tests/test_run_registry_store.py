from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from pancakebot.infra.run_registry_store import RunRegistryStore


class RunRegistryStoreTests(unittest.TestCase):
    def test_start_complete_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "run_registry.sqlite"
            store = RunRegistryStore(str(db_path))
            try:
                store.start_run(
                    run_name="run_a",
                    config_path="config.toml",
                    metadata={"sim_size": 100},
                )
                store.complete_run(
                    run_name="run_a",
                    summary_path="s.json",
                    trades_path="t.csv",
                    summary={"net_profit_bnb": 1.25, "num_bets": 7},
                    max_drawdown_bnb=0.9,
                    profit_per_500_bnb=6.25,
                )

                store.start_run(
                    run_name="run_b",
                    config_path="config.toml",
                    metadata={"sim_size": 200},
                )
                store.fail_run(run_name="run_b", error_text="boom")
            finally:
                store.close()

            conn = sqlite3.connect(str(db_path))
            try:
                rows = conn.execute(
                    "SELECT run_name, status, net_profit_bnb, num_bets, error_text FROM runs ORDER BY run_name ASC"
                ).fetchall()
            finally:
                conn.close()

            self.assertEqual(2, len(rows))
            self.assertEqual("run_a", rows[0][0])
            self.assertEqual("completed", rows[0][1])
            self.assertAlmostEqual(1.25, float(rows[0][2]), places=9)
            self.assertEqual(7, int(rows[0][3]))
            self.assertEqual("run_b", rows[1][0])
            self.assertEqual("failed", rows[1][1])
            self.assertEqual("boom", rows[1][4])


if __name__ == "__main__":
    unittest.main()
