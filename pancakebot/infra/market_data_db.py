from __future__ import annotations

import json
import os
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import RLock

from pancakebot.core.errors import InvariantError
from pancakebot.domain.types import Bet, Round


class MarketDataDb:
    """SQLite mirror for canonical market data (closed rounds + bets + klines)."""

    _ROUND_BATCH_ROWS = 1000
    _BET_BATCH_ROWS = 5000
    _ROUND_INGEST_VERSION = "rounds_v2_text_wei"

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
    ) -> dict[str, bool]:
        rounds_changed = self._ensure_rounds_synced(rounds_jsonl_path=str(rounds_jsonl_path))
        return {
            "rounds_changed": bool(rounds_changed),
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
            from pancakebot.core.constants import INTERVAL_SECONDS
            start_at = int(row["start_at"])
            out.append(
                Round(
                    epoch=epoch,
                    start_at=start_at,
                    lock_at=start_at + INTERVAL_SECONDS,
                    lock_price=(None if row["lock_price"] is None else float(row["lock_price"])),
                    close_price=(None if row["close_price"] is None else float(row["close_price"])),
                    position=(None if row["position"] is None else str(row["position"])),
                    failed=failed,
                    bets=tuple(by_epoch.get(epoch, [])),
                )
            )
        return out

    def count_rounds(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(1) FROM rounds").fetchone()
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
                                None,  # lock_at: computed at runtime from start_at + INTERVAL_SECONDS
                                None,  # close_at: removed
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
