from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from pathlib import Path

from pancakebot.domain.types import Kline
from pancakebot.core.errors import InvariantError


class KlinesStore:
    """Append-only JSONL store for fully CLOSED Binance US 1m klines.

    Disk format (var/klines.jsonl):
      - one JSON object per line
      - strictly increasing open_time_ms
      - no duplicates

    This store maintains an in-memory ordered list for fast lookups.
    """

    def __init__(self, path: str) -> None:
        self._path = str(path)
        self._klines: list[Kline] = []
        self._open_times: list[int] = []
        self._load_existing()

    @property
    def path(self) -> str:
        return self._path

    def _load_existing(self) -> None:
        p = Path(self._path)
        if not p.exists():
            return
        lines = p.read_text().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            k = Kline.from_json(rec)
            self._append_in_memory(k)

    def _append_in_memory(self, k: Kline) -> None:
        if self._open_times:
            if int(k.open_time_ms) <= int(self._open_times[-1]):
                raise InvariantError("klines_store_non_monotonic")
        self._klines.append(k)
        self._open_times.append(int(k.open_time_ms))

    def append_many(self, klines: list[Kline]) -> int:
        """Append new klines (must be strictly after current tail). Returns appended count."""

        if not klines:
            return 0

        # Validate strict ordering + no duplicates within batch.
        prev = None
        for k in klines:
            if prev is not None and int(k.open_time_ms) <= int(prev.open_time_ms):
                raise InvariantError("klines_batch_not_strictly_increasing")
            prev = k

        # Validate against store tail.
        if self._open_times and int(klines[0].open_time_ms) <= int(self._open_times[-1]):
            raise InvariantError("klines_batch_overlaps_store")

        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            for k in klines:
                rec = k.to_json()
                f.write(json.dumps(rec, separators=(",", ":"), sort_keys=True))
                f.write("\n")
                self._append_in_memory(k)

        return int(len(klines))

    def prepend_many(self, klines: list[Kline]) -> int:
        """Prepend older klines (must be strictly before current head). Returns prepended count."""

        if not klines:
            return 0

        # Validate strict ordering + no duplicates within batch.
        prev = None
        for k in klines:
            if prev is not None and int(k.open_time_ms) <= int(prev.open_time_ms):
                raise InvariantError("klines_batch_not_strictly_increasing")
            prev = k

        # Empty store: prepend behaves like append.
        if not self._open_times:
            return self.append_many(klines)

        head = int(self._open_times[0])
        if int(klines[-1].open_time_ms) >= int(head):
            raise InvariantError("klines_batch_overlaps_store_head")

        path = Path(self._path)
        if not path.exists():
            raise InvariantError("klines_prepend_requires_existing_store_file")

        replace_path = path.with_suffix(path.suffix + ".prepend.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(replace_path, "w") as out:
            for k in klines:
                rec = k.to_json()
                out.write(json.dumps(rec, separators=(",", ":"), sort_keys=True))
                out.write("\n")
            with open(path, "r") as existing:
                for line in existing:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")

        replace_path.replace(path)

        self._klines = list(klines) + self._klines
        self._open_times = [int(k.open_time_ms) for k in klines] + self._open_times
        return int(len(klines))

    def latest_open_time_ms(self) -> int | None:
        return int(self._open_times[-1]) if self._open_times else None

    def earliest_open_time_ms(self) -> int | None:
        return int(self._open_times[0]) if self._open_times else None

    def latest_close_time_ms(self) -> int | None:
        if not self._klines:
            return None
        return int(self._klines[-1].close_time_ms)

    def get_klines_between(self, *, start_open_time_ms: int, end_open_time_ms: int) -> list[Kline]:
        """Return klines with start_open_time_ms <= open_time_ms < end_open_time_ms."""
        if int(end_open_time_ms) <= int(start_open_time_ms):
            return []
        if not self._klines:
            return []

        start_idx = bisect_left(self._open_times, int(start_open_time_ms))
        end_idx = bisect_left(self._open_times, int(end_open_time_ms))
        return list(self._klines[start_idx:end_idx])

    def get_context_klines(self, *, anchor_close_time_ms: int, size: int) -> list[Kline]:
        """Return the last `size` klines with close_time_ms <= anchor_close_time_ms."""

        if size <= 0:
            raise InvariantError("context_klines_size_invalid")

        if not self._klines:
            raise InvariantError("klines_store_empty")

        # Find rightmost kline whose close_time_ms <= anchor.
        # We binary-search over open_time_ms, then adjust by checking close_time.
        # For 1m klines, open_time order equals close_time order.
        idx = bisect_right(self._open_times, int(anchor_close_time_ms)) - 1
        if idx < 0:
            raise InvariantError("klines_anchor_before_first")

        # Walk backward until close_time_ms <= anchor is satisfied.
        while idx >= 0 and int(self._klines[idx].close_time_ms) > int(anchor_close_time_ms):
            idx -= 1
        if idx < 0:
            raise InvariantError("klines_anchor_before_first")

        start = idx - int(size) + 1
        if start < 0:
            raise InvariantError("klines_insufficient_coverage")

        out = self._klines[start : idx + 1]
        if len(out) != int(size):
            raise InvariantError("context_klines_len_mismatch")

        return list(out)
