from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any

from pancakebot.core.errors import InvariantError


class RunRegistryStore:
    """SQLite registry for experiment/backtest runs and outcomes."""

    def __init__(self, path_sqlite: str) -> None:
        if str(path_sqlite).strip() == "":
            raise InvariantError("run_registry_path_empty")
        self._path = str(path_sqlite)
        p = Path(self._path)
        parent = p.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=15000;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_name TEXT PRIMARY KEY,
                started_at_ts INTEGER NOT NULL,
                updated_at_ts INTEGER NOT NULL,
                finished_at_ts INTEGER NULL,
                status TEXT NOT NULL,
                config_path TEXT NOT NULL,
                summary_path TEXT NULL,
                trades_path TEXT NULL,
                net_profit_bnb REAL NULL,
                profit_per_500_bnb REAL NULL,
                num_bets INTEGER NULL,
                max_drawdown_bnb REAL NULL,
                metadata_json TEXT NOT NULL,
                error_text TEXT NULL
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, updated_at_ts DESC)"
        )
        self._conn.commit()

    @property
    def path(self) -> str:
        return self._path

    def start_run(
        self,
        *,
        run_name: str,
        config_path: str,
        metadata: dict[str, Any],
    ) -> None:
        if str(run_name).strip() == "":
            raise InvariantError("run_registry_run_name_empty")
        now_ts = int(time.time())
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_name,
                    started_at_ts,
                    updated_at_ts,
                    finished_at_ts,
                    status,
                    config_path,
                    summary_path,
                    trades_path,
                    net_profit_bnb,
                    profit_per_500_bnb,
                    num_bets,
                    max_drawdown_bnb,
                    metadata_json,
                    error_text
                ) VALUES (?, ?, ?, NULL, 'running', ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, NULL)
                ON CONFLICT(run_name) DO UPDATE SET
                    updated_at_ts=excluded.updated_at_ts,
                    finished_at_ts=NULL,
                    status='running',
                    config_path=excluded.config_path,
                    summary_path=NULL,
                    trades_path=NULL,
                    net_profit_bnb=NULL,
                    profit_per_500_bnb=NULL,
                    num_bets=NULL,
                    max_drawdown_bnb=NULL,
                    metadata_json=excluded.metadata_json,
                    error_text=NULL
                """,
                (
                    str(run_name),
                    int(now_ts),
                    int(now_ts),
                    str(config_path),
                    str(payload),
                ),
            )
            self._conn.commit()

    def complete_run(
        self,
        *,
        run_name: str,
        summary_path: str,
        trades_path: str,
        summary: dict[str, Any],
        max_drawdown_bnb: float | None = None,
        profit_per_500_bnb: float | None = None,
    ) -> None:
        now_ts = int(time.time())
        with self._lock:
            self._conn.execute(
                """
                UPDATE runs
                SET
                    updated_at_ts = ?,
                    finished_at_ts = ?,
                    status = 'completed',
                    summary_path = ?,
                    trades_path = ?,
                    net_profit_bnb = ?,
                    profit_per_500_bnb = ?,
                    num_bets = ?,
                    max_drawdown_bnb = ?,
                    error_text = NULL
                WHERE run_name = ?
                """,
                (
                    int(now_ts),
                    int(now_ts),
                    str(summary_path),
                    str(trades_path),
                    float(summary.get("net_profit_bnb", 0.0)),
                    (None if profit_per_500_bnb is None else float(profit_per_500_bnb)),
                    int(summary.get("num_bets", 0)),
                    (None if max_drawdown_bnb is None else float(max_drawdown_bnb)),
                    str(run_name),
                ),
            )
            self._conn.commit()

    def fail_run(self, *, run_name: str, error_text: str) -> None:
        now_ts = int(time.time())
        with self._lock:
            self._conn.execute(
                """
                UPDATE runs
                SET
                    updated_at_ts = ?,
                    finished_at_ts = ?,
                    status = 'failed',
                    error_text = ?
                WHERE run_name = ?
                """,
                (
                    int(now_ts),
                    int(now_ts),
                    str(error_text),
                    str(run_name),
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return
