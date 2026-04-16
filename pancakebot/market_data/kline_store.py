"""Epoch-ascending JSONL store for per-round 1s kline arrays."""
from __future__ import annotations

import json
import os
from typing import Iterator

from pancakebot.util import InvariantError


class KlineStore:
    """Epoch-ascending JSONL store for 1s kline arrays."""

    def __init__(self, path_jsonl: str) -> None:
        if not path_jsonl:
            raise InvariantError("KlineStore_requires_path")
        self._path = path_jsonl

    @property
    def path_jsonl(self) -> str:
        return self._path

    def exists(self) -> bool:
        return os.path.exists(self._path)

    # --------------------------
    # Reads
    # --------------------------

    def iter_records(self) -> Iterator[dict]:
        if not self.exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def load_done_epochs(self) -> set[int]:
        return {int(rec["epoch"]) for rec in self.iter_records()}

    def load_earliest_epoch(self) -> int | None:
        for rec in self.iter_records():
            return int(rec["epoch"])
        return None

    def load_latest_epoch(self) -> int | None:
        if not self.exists():
            return None
        latest: int | None = None
        for rec in self.iter_records():
            latest = int(rec["epoch"])
        return latest

    def count(self) -> int:
        if not self.exists():
            return 0
        n = 0
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    # --------------------------
    # Writes
    # --------------------------

    def write_new(self, records_asc: list[dict]) -> None:
        """Create a new store from epoch-ascending records."""
        if self.exists():
            raise InvariantError("kline_store_write_new_already_exists")
        if not records_asc:
            raise InvariantError("kline_store_write_new_empty")
        self._validate_ascending(records_asc)

        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            for rec in records_asc:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def append_after(self, prev_epoch: int, records_asc: list[dict]) -> int:
        """Append epoch-ascending records strictly after prev_epoch.

        Returns the latest epoch after appending.
        """
        if not records_asc:
            return prev_epoch

        first_epoch = int(records_asc[0]["epoch"])
        if first_epoch <= prev_epoch:
            raise InvariantError(
                f"kline_store_append_not_after: first={first_epoch} prev={prev_epoch}"
            )
        self._validate_ascending(records_asc)

        with open(self._path, "a", encoding="utf-8") as f:
            for rec in records_asc:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                f.flush()
        return int(records_asc[-1]["epoch"])

    def rewrite(self, records_asc: list[dict]) -> None:
        """Rewrite the entire store with epoch-ascending records.

        Used when gap-filling requires merging new records into the middle
        of the existing sequence.
        """
        if not records_asc:
            raise InvariantError("kline_store_rewrite_empty")
        self._validate_ascending(records_asc)

        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records_asc:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        os.replace(tmp_path, self._path)

    def purge_and_rewrite(self) -> int:
        """Remove any error records and re-validate ordering.

        Returns the number of records purged.
        """
        if not self.exists():
            return 0

        kept: list[str] = []
        purged = 0
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error"):
                    purged += 1
                    continue
                kept.append(line)

        if purged > 0:
            with open(self._path, "w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")

        return purged

    @staticmethod
    def _validate_ascending(records: list[dict]) -> None:
        prev: int | None = None
        for idx, rec in enumerate(records):
            epoch = int(rec["epoch"])
            if prev is not None and epoch <= prev:
                raise InvariantError(
                    f"kline_store_not_ascending: idx={idx} got={epoch} prev={prev}"
                )
            prev = epoch
