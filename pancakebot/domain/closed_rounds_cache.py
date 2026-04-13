from __future__ import annotations

from dataclasses import dataclass

from pancakebot.domain.types import Round
from pancakebot.core.errors import InvariantError


@dataclass(slots=True)
class RollingClosedRoundsCache:
    """In-memory rolling cache of usable closed rounds.

    Invariants:
      - `rounds` is strictly epoch-ascending.
      - `capacity` is a hard max. If len(rounds) > capacity,
        the earliest rounds are dropped from memory only.

    Notes:
      - Disk is never trimmed.
      - No dedupe logic is performed.
    """

    rounds: list[Round]
    capacity: int

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise InvariantError("closed_round_cache_capacity_nonpositive")
        self._assert_strictly_increasing(self.rounds)
        self._trim_in_place()

    @property
    def earliest_epoch(self) -> int:
        if not self.rounds:
            raise InvariantError("closed_round_cache_empty")
        return self.rounds[0].epoch

    @property
    def latest_epoch(self) -> int:
        if not self.rounds:
            raise InvariantError("closed_round_cache_empty")
        return self.rounds[-1].epoch

    def tail(self, n: int) -> list[Round]:
        """Return the last n rounds in epoch-ascending order (oldest -> newest)."""
        if n <= 0:
            return []
        return list(self.rounds[-n:])

    def extend(self, new_rounds_asc: list[Round]) -> None:
        if not new_rounds_asc:
            return

        if not self.rounds:
            self.rounds.extend(new_rounds_asc)
            self._assert_strictly_increasing(self.rounds)
            self._trim_in_place()
            return

        prev = self.rounds[-1].epoch
        for idx, r in enumerate(new_rounds_asc):
            if r.epoch <= prev:
                raise InvariantError(f"cache_extend_not_strictly_increasing: idx={idx} got={r.epoch} prev={prev}")
            prev = r.epoch

        self.rounds.extend(new_rounds_asc)
        self._trim_in_place()

    def get_round(self, epoch: int) -> Round | None:
        """Return the cached Round for epoch, or None if not present."""
        lo = 0
        hi = len(self.rounds) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            me = self.rounds[mid].epoch
            if me == epoch:
                return self.rounds[mid]
            if me < epoch:
                lo = mid + 1
            else:
                hi = mid - 1
        return None



    def _trim_in_place(self) -> None:
        if len(self.rounds) <= self.capacity:
            return
        drop = len(self.rounds) - self.capacity
        self.rounds = self.rounds[drop:]

    @staticmethod
    def _assert_strictly_increasing(rounds_asc: list[Round]) -> None:
        prev: int | None = None
        for idx, r in enumerate(rounds_asc):
            if prev is not None and r.epoch <= prev:
                raise InvariantError(f"cache_not_strictly_increasing: idx={idx} got={r.epoch} prev={prev}")
            prev = r.epoch
