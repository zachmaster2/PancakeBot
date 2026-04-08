from __future__ import annotations

import json
import os
import sqlite3
from bisect import bisect_left, bisect_right
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import RLock

from pancakebot.core.errors import InvariantError
from pancakebot.domain.types import Bet, Kline, Round


class MarketDataDb:
    """SQLite mirror for canonical market data (closed rounds + bets + klines)."""

    _ROUND_BATCH_ROWS = 1000
    _BET_BATCH_ROWS = 5000
    _KLINE_BATCH_ROWS = 2000
    _ROUND_INGEST_VERSION = "rounds_v2_text_wei"
    _KLINE_INGEST_VERSION = "klines_v1"

    def __init__(self, path_sqlite: str) -> None:
        if str(path_sqlite).strip() == "":
            raise InvariantError("market_data_db_path_empty")
        self._path = str(path_sqlite)
        p = Path(self._path)
        parent = p.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=15000;")
        self._init_schema()

    @property
    def path(self) -> str:
        return self._path

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rounds (
                    epoch INTEGER PRIMARY KEY,
                    start_at INTEGER NOT NULL,
                    lock_at INTEGER NULL,
                    close_at INTEGER NULL,
                    lock_price REAL NULL,
                    close_price REAL NULL,
                    position TEXT NULL,
                    failed INTEGER NULL,
                    bet_count INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS round_bets (
                    round_epoch INTEGER NOT NULL,
                    bet_index INTEGER NOT NULL,
                    wallet_address TEXT NOT NULL,
                    amount_wei TEXT NOT NULL,
                    position TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (round_epoch, bet_index)
                );
                CREATE INDEX IF NOT EXISTS idx_round_bets_epoch
                  ON round_bets(round_epoch, bet_index);
                CREATE INDEX IF NOT EXISTS idx_round_bets_created
                  ON round_bets(created_at);

                CREATE TABLE IF NOT EXISTS klines (
                    open_time_ms INTEGER PRIMARY KEY,
                    close_time_ms INTEGER NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    quote_asset_volume REAL NOT NULL
                );
                """
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

    def ensure_sources_synced(
        self,
        *,
        rounds_jsonl_path: str,
        klines_jsonl_path: str,
    ) -> dict[str, bool]:
        rounds_changed = self._ensure_rounds_synced(rounds_jsonl_path=str(rounds_jsonl_path))
        klines_changed = self._ensure_klines_synced(klines_jsonl_path=str(klines_jsonl_path))
        return {
            "rounds_changed": bool(rounds_changed),
            "klines_changed": bool(klines_changed),
        }

    def load_tail_rounds(self, *, n: int) -> list[Round]:
        if int(n) <= 0:
            raise InvariantError("market_data_db_tail_n_invalid")
        with self._lock:
            round_rows = self._conn.execute(
                """
                SELECT
                    epoch,
                    start_at,
                    lock_at,
                    close_at,
                    lock_price,
                    close_price,
                    position,
                    failed
                FROM rounds
                ORDER BY epoch DESC
                LIMIT ?
                """,
                (int(n),),
            ).fetchall()

            if not round_rows:
                return []

            rows_asc = list(reversed(round_rows))
            min_epoch = int(rows_asc[0]["epoch"])
            max_epoch = int(rows_asc[-1]["epoch"])
            bet_rows = self._conn.execute(
                """
                SELECT
                    round_epoch,
                    wallet_address,
                    amount_wei,
                    position,
                    created_at
                FROM round_bets
                WHERE round_epoch >= ? AND round_epoch <= ?
                ORDER BY round_epoch ASC, bet_index ASC
                """,
                (int(min_epoch), int(max_epoch)),
            ).fetchall()

        by_epoch: dict[int, list[Bet]] = {}
        for row in bet_rows:
            epoch = int(row["round_epoch"])
            by_epoch.setdefault(epoch, []).append(
                Bet(
                    wallet_address=str(row["wallet_address"]),
                    amount_wei=_parse_wei(row["amount_wei"]),
                    position=str(row["position"]),
                    created_at=int(row["created_at"]),
                )
            )

        out: list[Round] = []
        for row in rows_asc:
            epoch = int(row["epoch"])
            failed_raw = row["failed"]
            failed = None if failed_raw is None else bool(int(failed_raw))
            out.append(
                Round(
                    epoch=epoch,
                    start_at=int(row["start_at"]),
                    lock_at=(None if row["lock_at"] is None else int(row["lock_at"])),
                    close_at=(None if row["close_at"] is None else int(row["close_at"])),
                    lock_price=(None if row["lock_price"] is None else float(row["lock_price"])),
                    close_price=(None if row["close_price"] is None else float(row["close_price"])),
                    position=(None if row["position"] is None else str(row["position"])),
                    failed=failed,
                    bets=tuple(by_epoch.get(epoch, [])),
                )
            )
        return out

    def load_all_klines(self) -> list[Kline]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    open_time_ms,
                    close_time_ms,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    quote_asset_volume
                FROM klines
                ORDER BY open_time_ms ASC
                """
            ).fetchall()
        return [self._kline_from_row(row) for row in rows]

    def count_rounds(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(1) FROM rounds").fetchone()
        if row is None:
            return 0
        return int(row[0])

    def count_klines(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(1) FROM klines").fetchone()
        if row is None:
            return 0
        return int(row[0])

    def rounds_source_signature(self) -> dict[str, object]:
        raw = self._meta_get("rounds_source_signature")
        if raw is None:
            return {}
        try:
            out = json.loads(str(raw))
        except (TypeError, ValueError) as e:
            raise InvariantError("market_data_db_rounds_signature_invalid") from e
        if not isinstance(out, dict):
            raise InvariantError("market_data_db_rounds_signature_not_dict")
        return out

    def klines_source_signature(self) -> dict[str, object]:
        raw = self._meta_get("klines_source_signature")
        if raw is None:
            return {}
        try:
            out = json.loads(str(raw))
        except (TypeError, ValueError) as e:
            raise InvariantError("market_data_db_klines_signature_invalid") from e
        if not isinstance(out, dict):
            raise InvariantError("market_data_db_klines_signature_not_dict")
        return out

    def _ensure_rounds_synced(self, *, rounds_jsonl_path: str) -> bool:
        source_path = Path(str(rounds_jsonl_path))
        if not source_path.exists():
            raise InvariantError("market_data_db_rounds_source_missing")
        source_sig = {
            "ingest_version": str(self._ROUND_INGEST_VERSION),
            "source": _file_signature(path=str(source_path)),
        }
        source_sig_text = json.dumps(source_sig, sort_keys=True, separators=(",", ":"))
        current_sig_text = self._meta_get("rounds_source_signature")

        if current_sig_text == source_sig_text and int(self.count_rounds()) > 0:
            return False

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM round_bets")
                self._conn.execute("DELETE FROM rounds")

                prev_epoch: int | None = None
                round_rows: list[tuple[object, ...]] = []
                bet_rows: list[tuple[object, ...]] = []
                round_count = 0

                with source_path.open("r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if line == "":
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise InvariantError(
                                f"market_data_db_rounds_json_parse_failed: line={line_no} err={e}"
                            ) from e

                        round_t = Round.from_json(obj)
                        epoch = int(round_t.epoch)
                        if prev_epoch is not None and int(epoch) <= int(prev_epoch):
                            raise InvariantError("market_data_db_rounds_not_strictly_increasing")
                        prev_epoch = int(epoch)
                        round_count += 1

                        round_rows.append(
                            (
                                int(round_t.epoch),
                                int(round_t.start_at),
                                (None if round_t.lock_at is None else int(round_t.lock_at)),
                                (None if round_t.close_at is None else int(round_t.close_at)),
                                (None if round_t.lock_price is None else float(round_t.lock_price)),
                                (None if round_t.close_price is None else float(round_t.close_price)),
                                (None if round_t.position is None else str(round_t.position)),
                                (None if round_t.failed is None else (1 if bool(round_t.failed) else 0)),
                                int(len(round_t.bets)),
                            )
                        )
                        for bet_index, b in enumerate(round_t.bets):
                            bet_rows.append(
                                (
                                    int(round_t.epoch),
                                    int(bet_index),
                                    str(b.wallet_address),
                                    str(int(b.amount_wei)),
                                    str(b.position),
                                    int(b.created_at),
                                )
                            )

                        if len(round_rows) >= int(self._ROUND_BATCH_ROWS):
                            self._conn.executemany(
                                """
                                INSERT INTO rounds (
                                    epoch,
                                    start_at,
                                    lock_at,
                                    close_at,
                                    lock_price,
                                    close_price,
                                    position,
                                    failed,
                                    bet_count
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                round_rows,
                            )
                            round_rows = []
                        if len(bet_rows) >= int(self._BET_BATCH_ROWS):
                            self._conn.executemany(
                                """
                                INSERT INTO round_bets (
                                    round_epoch,
                                    bet_index,
                                    wallet_address,
                                    amount_wei,
                                    position,
                                    created_at
                                ) VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                bet_rows,
                            )
                            bet_rows = []

                if round_rows:
                    self._conn.executemany(
                        """
                        INSERT INTO rounds (
                            epoch,
                            start_at,
                            lock_at,
                            close_at,
                            lock_price,
                            close_price,
                            position,
                            failed,
                            bet_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        round_rows,
                    )
                if bet_rows:
                    self._conn.executemany(
                        """
                        INSERT INTO round_bets (
                            round_epoch,
                            bet_index,
                            wallet_address,
                            amount_wei,
                            position,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        bet_rows,
                    )

                if int(round_count) <= 0:
                    raise InvariantError("market_data_db_rounds_empty_source")
                self._meta_set("rounds_source_signature", str(source_sig_text))
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def _ensure_klines_synced(self, *, klines_jsonl_path: str) -> bool:
        source_path = Path(str(klines_jsonl_path))
        if not source_path.exists():
            raise InvariantError("market_data_db_klines_source_missing")
        source_sig = {
            "ingest_version": str(self._KLINE_INGEST_VERSION),
            "source": _file_signature(path=str(source_path)),
        }
        source_sig_text = json.dumps(source_sig, sort_keys=True, separators=(",", ":"))
        current_sig_text = self._meta_get("klines_source_signature")

        if current_sig_text == source_sig_text and int(self.count_klines()) > 0:
            return False

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM klines")
                prev_open: int | None = None
                kline_rows: list[tuple[object, ...]] = []
                kline_count = 0
                with source_path.open("r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if line == "":
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise InvariantError(
                                f"market_data_db_klines_json_parse_failed: line={line_no} err={e}"
                            ) from e
                        k = Kline.from_json(obj)
                        if prev_open is not None and int(k.open_time_ms) <= int(prev_open):
                            raise InvariantError("market_data_db_klines_not_strictly_increasing")
                        prev_open = int(k.open_time_ms)
                        kline_count += 1
                        kline_rows.append(
                            (
                                int(k.open_time_ms),
                                int(k.close_time_ms),
                                float(k.open_price),
                                float(k.high_price),
                                float(k.low_price),
                                float(k.close_price),
                                float(k.volume),
                                float(k.quote_asset_volume),
                            )
                        )
                        if len(kline_rows) >= int(self._KLINE_BATCH_ROWS):
                            self._conn.executemany(
                                """
                                INSERT INTO klines (
                                    open_time_ms,
                                    close_time_ms,
                                    open_price,
                                    high_price,
                                    low_price,
                                    close_price,
                                    volume,
                                    quote_asset_volume
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                kline_rows,
                            )
                            kline_rows = []

                if kline_rows:
                    self._conn.executemany(
                        """
                        INSERT INTO klines (
                            open_time_ms,
                            close_time_ms,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                            volume,
                            quote_asset_volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        kline_rows,
                    )

                if int(kline_count) <= 0:
                    raise InvariantError("market_data_db_klines_empty_source")
                self._meta_set("klines_source_signature", str(source_sig_text))
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def _meta_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                (str(key),),
            ).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def _meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO meta (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (str(key), str(value)),
        )

    @staticmethod
    def _kline_from_row(row: sqlite3.Row) -> Kline:
        return Kline(
            open_time_ms=int(row["open_time_ms"]),
            close_time_ms=int(row["close_time_ms"]),
            open_price=float(row["open_price"]),
            high_price=float(row["high_price"]),
            low_price=float(row["low_price"]),
            close_price=float(row["close_price"]),
            volume=float(row["volume"]),
            quote_asset_volume=float(row["quote_asset_volume"]),
        )


class SqliteKlinesStore:
    """Read-only kline store backed by MarketDataDb (in-memory indexed)."""

    def __init__(self, *, market_data_db: MarketDataDb) -> None:
        self._market_data_db = market_data_db
        self._path = str(market_data_db.path)
        self._klines = list(market_data_db.load_all_klines())
        self._open_times = [int(k.open_time_ms) for k in self._klines]
        if self._open_times:
            prev = self._open_times[0]
            for idx in range(1, len(self._open_times)):
                cur = int(self._open_times[idx])
                if int(cur) <= int(prev):
                    raise InvariantError("sqlite_klines_store_non_monotonic")
                prev = int(cur)

    @property
    def path(self) -> str:
        return self._path

    def latest_open_time_ms(self) -> int | None:
        return int(self._open_times[-1]) if self._open_times else None

    def earliest_open_time_ms(self) -> int | None:
        return int(self._open_times[0]) if self._open_times else None

    def latest_close_time_ms(self) -> int | None:
        if not self._klines:
            return None
        return int(self._klines[-1].close_time_ms)

    def get_klines_between(self, *, start_open_time_ms: int, end_open_time_ms: int) -> list[Kline]:
        if int(end_open_time_ms) <= int(start_open_time_ms):
            return []
        if not self._klines:
            return []
        start_idx = bisect_left(self._open_times, int(start_open_time_ms))
        end_idx = bisect_left(self._open_times, int(end_open_time_ms))
        return list(self._klines[start_idx:end_idx])

    def get_context_klines(self, *, anchor_close_time_ms: int, size: int) -> list[Kline]:
        if int(size) <= 0:
            raise InvariantError("sqlite_klines_context_size_invalid")
        if not self._klines:
            raise InvariantError("sqlite_klines_store_empty")
        idx = bisect_right(self._open_times, int(anchor_close_time_ms)) - 1
        if int(idx) < 0:
            raise InvariantError("sqlite_klines_anchor_before_first")
        while idx >= 0 and int(self._klines[idx].close_time_ms) > int(anchor_close_time_ms):
            idx -= 1
        if idx < 0:
            raise InvariantError("sqlite_klines_anchor_before_first")
        start = int(idx) - int(size) + 1
        if int(start) < 0:
            raise InvariantError("sqlite_klines_insufficient_coverage")
        out = self._klines[int(start): int(idx) + 1]
        if len(out) != int(size):
            raise InvariantError("sqlite_klines_context_len_mismatch")
        return list(out)

    def append_many(self, klines: list[Kline]) -> int:
        _ = klines
        raise InvariantError("sqlite_klines_store_read_only")

    def prepend_many(self, klines: list[Kline]) -> int:
        _ = klines
        raise InvariantError("sqlite_klines_store_read_only")


def _file_signature(*, path: str) -> dict[str, object]:
    p = Path(str(path))
    if not p.exists():
        return {"path": str(path), "exists": False}
    st = os.stat(str(p))
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _parse_wei(raw: object) -> int:
    text = str(raw).strip()
    try:
        return int(text)
    except ValueError:
        try:
            return int(Decimal(text))
        except (InvalidOperation, ValueError) as e:
            raise InvariantError("market_data_db_bet_amount_wei_parse_failed") from e
