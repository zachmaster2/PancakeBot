from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from pancakebot.domain.types import Kline
from pancakebot.core.errors import InvariantError


@dataclass(slots=True)
class RollingKlinesCache:
    """In-memory rolling cache of fully closed 1m klines.

    Invariants:
      - `klines` is strictly open_time_ms-ascending.
      - `capacity` is a hard max. If len(klines) > capacity,
        the earliest klines are dropped from memory only.
    """

    klines: list[Kline]
    capacity: int
    _open_times: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.capacity) <= 0:
            raise InvariantError("klines_cache_capacity_nonpositive")
        self._assert_strictly_increasing(self.klines)
        self._open_times = [int(k.open_time_ms) for k in self.klines]
        self._trim_in_place()

    def latest_open_time_ms(self) -> int | None:
        return int(self._open_times[-1]) if self._open_times else None

    def latest_close_time_ms(self) -> int | None:
        if not self.klines:
            return None
        return int(self.klines[-1].close_time_ms)

    def extend(self, new_klines_asc: list[Kline]) -> None:
        if not new_klines_asc:
            return

        if not self.klines:
            self.klines.extend(new_klines_asc)
            self._open_times.extend([int(k.open_time_ms) for k in new_klines_asc])
            self._assert_strictly_increasing(self.klines)
            self._trim_in_place()
            return

        prev = int(self._open_times[-1])
        for idx, k in enumerate(new_klines_asc):
            ot = int(k.open_time_ms)
            if ot <= prev:
                raise InvariantError(f"klines_cache_extend_not_strictly_increasing: idx={idx} got={ot} prev={prev}")
            prev = ot

        self.klines.extend(new_klines_asc)
        self._open_times.extend([int(k.open_time_ms) for k in new_klines_asc])
        self._trim_in_place()

    def get_context_klines(self, *, anchor_close_time_ms: int, size: int) -> list[Kline]:
        """Return the last `size` klines with close_time_ms <= anchor_close_time_ms."""
        if int(size) <= 0:
            raise InvariantError("context_klines_size_invalid")
        if not self.klines:
            raise InvariantError("klines_cache_empty")

        idx = bisect_right(self._open_times, int(anchor_close_time_ms)) - 1
        if idx < 0:
            raise InvariantError("klines_anchor_before_first")

        while idx >= 0 and int(self.klines[idx].close_time_ms) > int(anchor_close_time_ms):
            idx -= 1
        if idx < 0:
            raise InvariantError("klines_anchor_before_first")

        start = idx - int(size) + 1
        if start < 0:
            raise InvariantError("klines_insufficient_coverage")

        out = self.klines[start : idx + 1]
        if len(out) != int(size):
            raise InvariantError("context_klines_len_mismatch")
        return list(out)

    def _trim_in_place(self) -> None:
        if len(self.klines) <= int(self.capacity):
            return
        drop = len(self.klines) - int(self.capacity)
        if drop <= 0:
            return
        self.klines = self.klines[drop:]
        self._open_times = self._open_times[drop:]

    @staticmethod
    def _assert_strictly_increasing(klines_asc: list[Kline]) -> None:
        prev: int | None = None
        for idx, k in enumerate(klines_asc):
            ot = int(k.open_time_ms)
            if prev is not None and ot <= prev:
                raise InvariantError(f"klines_cache_not_strictly_increasing: idx={idx} got={ot} prev={prev}")
            prev = ot
