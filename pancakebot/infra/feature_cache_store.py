from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import RLock

from pancakebot.core.errors import InvariantError


class FeatureCacheStore:
    """Persistent feature-vector cache keyed by epoch + context signature.

    This cache is shared by backtest/inspection runs to avoid recomputing
    canonical feature vectors for the same epoch/context tuple.
    """

    def __init__(self, path_sqlite: str, *, commit_every_writes: int = 500) -> None:
        if str(path_sqlite).strip() == "":
            raise InvariantError("feature_cache_path_empty")
        if int(commit_every_writes) <= 0:
            raise InvariantError("feature_cache_commit_every_writes_nonpositive")
        self._path = str(path_sqlite)
        p = Path(self._path)
        parent = p.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._commit_every_writes = int(commit_every_writes)
        self._pending_writes = 0
        self._closed = False
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_vectors (
                epoch INTEGER NOT NULL,
                cutoff_seconds INTEGER NOT NULL,
                schema_name TEXT NOT NULL,
                start_at INTEGER NOT NULL,
                lock_at INTEGER NOT NULL,
                prior_last_epoch INTEGER NOT NULL,
                anchor_close_time_ms INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                updated_at_ts INTEGER NOT NULL,
                PRIMARY KEY (
                    epoch,
                    cutoff_seconds,
                    schema_name,
                    start_at,
                    lock_at,
                    prior_last_epoch,
                    anchor_close_time_ms
                )
            );
            """
        )
        self._conn.commit()
        self._mem: dict[tuple[int, int, str, int, int, int, int], tuple[float, ...]] = {}

    def get_vector(
        self,
        *,
        epoch: int,
        cutoff_seconds: int,
        schema_name: str,
        start_at: int,
        lock_at: int,
        prior_last_epoch: int,
        anchor_close_time_ms: int,
    ) -> list[float] | None:
        key = self._key(
            epoch=epoch,
            cutoff_seconds=cutoff_seconds,
            schema_name=schema_name,
            start_at=start_at,
            lock_at=lock_at,
            prior_last_epoch=prior_last_epoch,
            anchor_close_time_ms=anchor_close_time_ms,
        )
        cached = self._mem.get(key)
        if cached is not None:
            return list(cached)

        with self._lock:
            row = self._conn.execute(
                """
                SELECT vector_json
                FROM feature_vectors
                WHERE epoch = ?
                  AND cutoff_seconds = ?
                  AND schema_name = ?
                  AND start_at = ?
                  AND lock_at = ?
                  AND prior_last_epoch = ?
                  AND anchor_close_time_ms = ?
                """,
                (
                    int(epoch),
                    int(cutoff_seconds),
                    str(schema_name),
                    int(start_at),
                    int(lock_at),
                    int(prior_last_epoch),
                    int(anchor_close_time_ms),
                ),
            ).fetchone()
        if row is None:
            return None
        try:
            raw = json.loads(str(row[0]))
        except (TypeError, ValueError) as e:
            raise InvariantError("feature_cache_vector_json_invalid") from e
        if not isinstance(raw, list):
            raise InvariantError("feature_cache_vector_not_list")
        out = tuple(float(x) for x in raw)
        self._mem[key] = out
        return list(out)

    def put_vector(
        self,
        *,
        epoch: int,
        cutoff_seconds: int,
        schema_name: str,
        start_at: int,
        lock_at: int,
        prior_last_epoch: int,
        anchor_close_time_ms: int,
        vector: list[float],
    ) -> None:
        if not vector:
            raise InvariantError("feature_cache_vector_empty")
        key = self._key(
            epoch=epoch,
            cutoff_seconds=cutoff_seconds,
            schema_name=schema_name,
            start_at=start_at,
            lock_at=lock_at,
            prior_last_epoch=prior_last_epoch,
            anchor_close_time_ms=anchor_close_time_ms,
        )
        vec = tuple(float(x) for x in vector)
        self._mem[key] = vec
        payload = json.dumps(list(vec), separators=(",", ":"), allow_nan=True)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO feature_vectors (
                    epoch,
                    cutoff_seconds,
                    schema_name,
                    start_at,
                    lock_at,
                    prior_last_epoch,
                    anchor_close_time_ms,
                    vector_json,
                    updated_at_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(epoch),
                    int(cutoff_seconds),
                    str(schema_name),
                    int(start_at),
                    int(lock_at),
                    int(prior_last_epoch),
                    int(anchor_close_time_ms),
                    str(payload),
                    int(time.time()),
                ),
            )
            self._pending_writes += 1
            if int(self._pending_writes) >= int(self._commit_every_writes):
                self._conn.commit()
                self._pending_writes = 0

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

    @staticmethod
    def _key(
        *,
        epoch: int,
        cutoff_seconds: int,
        schema_name: str,
        start_at: int,
        lock_at: int,
        prior_last_epoch: int,
        anchor_close_time_ms: int,
    ) -> tuple[int, int, str, int, int, int, int]:
        return (
            int(epoch),
            int(cutoff_seconds),
            str(schema_name),
            int(start_at),
            int(lock_at),
            int(prior_last_epoch),
            int(anchor_close_time_ms),
        )
