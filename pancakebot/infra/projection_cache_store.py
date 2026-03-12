from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from threading import RLock

from pancakebot.core.errors import InvariantError


class ProjectionCacheStore:
    """Persistent cache for ML final-pool projections at round cutoff."""

    def __init__(self, path_sqlite: str, *, commit_every_writes: int = 500) -> None:
        if str(path_sqlite).strip() == "":
            raise InvariantError("projection_cache_path_empty")
        if int(commit_every_writes) <= 0:
            raise InvariantError("projection_cache_commit_every_writes_nonpositive")
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
            CREATE TABLE IF NOT EXISTS final_pool_projection (
                epoch INTEGER NOT NULL,
                lock_at INTEGER NOT NULL,
                cutoff_ts INTEGER NOT NULL,
                bull_wei TEXT NOT NULL,
                bear_wei TEXT NOT NULL,
                is_available INTEGER NOT NULL,
                final_total_bnb REAL NULL,
                final_bull_bnb REAL NULL,
                final_bear_bnb REAL NULL,
                updated_at_ts INTEGER NOT NULL,
                PRIMARY KEY (epoch, lock_at, cutoff_ts, bull_wei, bear_wei)
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_meta (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            """
        )
        self._conn.commit()

        self._pending_writes = 0
        self._commit_every_writes = int(commit_every_writes)
        self._closed = False

    @property
    def path(self) -> str:
        return self._path

    def lookup_projection(
        self,
        *,
        epoch: int,
        lock_at: int,
        cutoff_ts: int,
        bull_wei: int,
        bear_wei: int,
    ) -> tuple[bool, tuple[float, float, float] | None]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    is_available,
                    final_total_bnb,
                    final_bull_bnb,
                    final_bear_bnb
                FROM final_pool_projection
                WHERE epoch = ?
                  AND lock_at = ?
                  AND cutoff_ts = ?
                  AND bull_wei = ?
                  AND bear_wei = ?
                """,
                (
                    int(epoch),
                    int(lock_at),
                    int(cutoff_ts),
                    str(int(bull_wei)),
                    str(int(bear_wei)),
                ),
            ).fetchone()
        if row is None:
            return False, None
        if int(row[0]) == 0:
            return True, None
        return (
            True,
            (
                float(row[1]),
                float(row[2]),
                float(row[3]),
            ),
        )

    def put_projection(
        self,
        *,
        epoch: int,
        lock_at: int,
        cutoff_ts: int,
        bull_wei: int,
        bear_wei: int,
        projection: tuple[float, float, float] | None,
    ) -> None:
        if projection is None:
            is_available = 0
            total = None
            bull = None
            bear = None
        else:
            is_available = 1
            total = float(projection[0])
            bull = float(projection[1])
            bear = float(projection[2])
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO final_pool_projection (
                    epoch,
                    lock_at,
                    cutoff_ts,
                    bull_wei,
                    bear_wei,
                    is_available,
                    final_total_bnb,
                    final_bull_bnb,
                    final_bear_bnb,
                    updated_at_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(epoch),
                    int(lock_at),
                    int(cutoff_ts),
                    str(int(bull_wei)),
                    str(int(bear_wei)),
                    int(is_available),
                    total,
                    bull,
                    bear,
                    int(time.time()),
                ),
            )
            self._pending_writes += 1
            if int(self._pending_writes) >= int(self._commit_every_writes):
                self._conn.commit()
                self._pending_writes = 0

    def prune_before_or_equal_epoch(self, *, epoch: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM final_pool_projection WHERE epoch <= ?",
                (int(epoch),),
            )
            deleted = int(cur.rowcount if cur.rowcount is not None else 0)
            self._pending_writes += 1
            if int(self._pending_writes) >= int(self._commit_every_writes):
                self._conn.commit()
                self._pending_writes = 0
        return int(deleted)

    def flush(self) -> None:
        with self._lock:
            if bool(self._closed):
                return
            if int(self._pending_writes) > 0:
                self._conn.commit()
                self._pending_writes = 0

    def close(self) -> None:
        with self._lock:
            if bool(self._closed):
                return
            if int(self._pending_writes) > 0:
                self._conn.commit()
                self._pending_writes = 0
            self._conn.close()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return
