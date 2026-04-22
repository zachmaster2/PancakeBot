"""Bankroll tracker: rolling-window peak tracking + cooldown state.

Consumed by MomentumOnlyPipeline's risk checks (bankroll-minimum, drawdown-from-peak
circuit breaker, fractional bet cap). Two implementations:

- ``InMemoryBankrollTracker``: used by the backtest loop. No disk I/O.
- ``PersistedBankrollTracker``: used by dry/live. Appends each settlement change
  to a JSONL file and persists pause state to a sibling JSON file.

Window semantics: ``peak_bankroll(as_of_start_at)`` returns the max bankroll over
entries with ``start_at >= as_of_start_at - window_days * 86400``. The most-recent
entry with ``start_at < window_start`` is preserved as a boundary (so the
"entering-window" bankroll is always part of the peak candidates).

Cooldown semantics: ``cooldown_rounds`` is a count of ROUNDS. Each round where the
pipeline observes ``is_paused() == True`` calls ``tick_cooldown()`` which
decrements the counter; at 0, ``is_paused()`` returns False and betting resumes.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from pancakebot.util import InvariantError


_SECONDS_PER_DAY = 86400


@dataclass(slots=True)
class _Entry:
    start_at: int
    bankroll: float
    event: str   # "init" | "settlement"


class BankrollTracker(ABC):
    """Abstract tracker. Concrete classes: InMemoryBankrollTracker, PersistedBankrollTracker."""

    @abstractmethod
    def record_settlement(self, bankroll: float, start_at: int) -> None:
        """Record a bankroll snapshot tied to a round's ``start_at``.

        If the new value equals the last recorded value, this is a no-op (dedup).
        """

    @abstractmethod
    def current_bankroll(self) -> float:
        """Return the most-recent recorded bankroll value."""

    @abstractmethod
    def peak_bankroll(self, as_of_start_at: int) -> float:
        """Return max bankroll over the rolling window ending at ``as_of_start_at``.

        Includes the boundary entry (most-recent entry before the window) so the
        peak reflects the "entering-window" bankroll.
        """

    @abstractmethod
    def is_paused(self, as_of_start_at: int) -> bool:
        """Return True iff cooldown_remaining > 0."""

    @abstractmethod
    def set_paused(self, cooldown_rounds: int, triggered_at: int) -> None:
        """Enter cooldown for ``cooldown_rounds`` rounds starting now."""

    @abstractmethod
    def tick_cooldown(self) -> None:
        """Decrement the cooldown counter by one; floor at 0."""

    @abstractmethod
    def cooldown_remaining(self) -> int:
        """Return the number of cooldown rounds remaining (0 when not paused)."""


class InMemoryBankrollTracker(BankrollTracker):
    """Backtest-mode tracker. No disk I/O. Bootstraps with ``initial_bankroll``."""

    __slots__ = ("_entries", "_window_days", "_cooldown", "_triggered_at", "_seeded")

    def __init__(self, *, initial_bankroll: float, window_days: int) -> None:
        if initial_bankroll <= 0.0:
            raise InvariantError("bankroll_tracker_initial_bankroll_not_positive")
        if window_days <= 0:
            raise InvariantError("bankroll_tracker_window_days_not_positive")
        self._entries: deque[_Entry] = deque()
        self._window_days = int(window_days)
        self._cooldown: int = 0
        self._triggered_at: int | None = None
        # Seeded lazily on first record_settlement when we know a real start_at;
        # prior to that, current_bankroll returns the initial value.
        self._seeded = False
        self._initial = float(initial_bankroll)

    def record_settlement(self, bankroll: float, start_at: int) -> None:
        if not self._seeded:
            self._entries.append(_Entry(
                start_at=int(start_at), bankroll=float(self._initial), event="init",
            ))
            self._seeded = True
        # Dedup: if the new value matches the most-recent recorded, skip.
        if self._entries and abs(self._entries[-1].bankroll - float(bankroll)) < 1e-12:
            return
        self._entries.append(_Entry(
            start_at=int(start_at), bankroll=float(bankroll), event="settlement",
        ))
        self._prune()

    def _prune(self) -> None:
        """Keep entries in window + one boundary entry (most-recent-before-window)."""
        if not self._entries:
            return
        latest_start = self._entries[-1].start_at
        window_start = latest_start - self._window_days * _SECONDS_PER_DAY
        # Find the oldest entry we want to KEEP:
        # - all entries with start_at >= window_start
        # - PLUS the most-recent entry with start_at < window_start (boundary)
        first_in_window_idx: int | None = None
        for i, e in enumerate(self._entries):
            if e.start_at >= window_start:
                first_in_window_idx = i
                break
        if first_in_window_idx is None:
            # No entries in window -- keep just the latest as boundary.
            latest = self._entries[-1]
            self._entries.clear()
            self._entries.append(latest)
            return
        boundary_idx = max(0, first_in_window_idx - 1)
        if boundary_idx == 0:
            return
        # Drop everything before boundary_idx.
        kept = list(self._entries)[boundary_idx:]
        self._entries.clear()
        self._entries.extend(kept)

    def current_bankroll(self) -> float:
        if not self._seeded:
            return self._initial
        return self._entries[-1].bankroll

    def peak_bankroll(self, as_of_start_at: int) -> float:
        if not self._entries:
            return self._initial if not self._seeded else 0.0
        window_start = int(as_of_start_at) - self._window_days * _SECONDS_PER_DAY
        # Entries in window:
        in_window = [e.bankroll for e in self._entries if e.start_at >= window_start]
        # Boundary (most-recent entry before window):
        before = [e for e in self._entries if e.start_at < window_start]
        if not in_window and not before:
            return 0.0
        candidates: list[float] = list(in_window)
        if before:
            candidates.append(before[-1].bankroll)
        if not candidates:
            return 0.0
        return max(candidates)

    def is_paused(self, as_of_start_at: int) -> bool:
        return self._cooldown > 0

    def set_paused(self, cooldown_rounds: int, triggered_at: int) -> None:
        if cooldown_rounds < 0:
            raise InvariantError("bankroll_tracker_cooldown_negative")
        self._cooldown = int(cooldown_rounds)
        self._triggered_at = int(triggered_at)

    def tick_cooldown(self) -> None:
        if self._cooldown > 0:
            self._cooldown -= 1
        if self._cooldown == 0:
            self._triggered_at = None

    def cooldown_remaining(self) -> int:
        return self._cooldown


class PersistedBankrollTracker(InMemoryBankrollTracker):
    """Dry/live-mode tracker. Persists change-events to JSONL + pause state to JSON.

    On init:
      - If ``path`` exists: read all entries into memory, prune, seed flag set.
      - If not: bootstrap lazily on first record_settlement with ``initial_bankroll``.
    """

    __slots__ = ("_history_path", "_pause_path")

    def __init__(self, *, path: Path, initial_bankroll: float, window_days: int) -> None:
        super().__init__(initial_bankroll=initial_bankroll, window_days=window_days)
        self._history_path = Path(path)
        self._pause_path = self._history_path.parent / "pause_state.json"
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        # History.
        if self._history_path.exists():
            try:
                for line in self._history_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self._entries.append(_Entry(
                        start_at=int(obj["start_at"]),
                        bankroll=float(obj["bankroll"]),
                        event=str(obj.get("event", "settlement")),
                    ))
                if self._entries:
                    self._seeded = True
                    self._prune()
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                raise InvariantError(
                    f"bankroll_tracker_history_parse_failed: {self._history_path} err={e}"
                ) from e
        # Pause state.
        if self._pause_path.exists():
            try:
                doc = json.loads(self._pause_path.read_text(encoding="utf-8"))
                if bool(doc.get("paused", False)):
                    self._cooldown = int(doc.get("cooldown_remaining", 0))
                    ta = doc.get("triggered_at")
                    self._triggered_at = int(ta) if ta is not None else None
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                raise InvariantError(
                    f"bankroll_tracker_pause_state_parse_failed: {self._pause_path} err={e}"
                ) from e

    def _append_history(self, entry: _Entry) -> None:
        line = json.dumps({
            "start_at": entry.start_at,
            "bankroll": entry.bankroll,
            "event": entry.event,
        }, separators=(",", ":"))
        with self._history_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _persist_pause_state(self) -> None:
        doc = {
            "paused": self._cooldown > 0,
            "cooldown_remaining": self._cooldown,
            "triggered_at": self._triggered_at,
        }
        tmp = self._pause_path.with_suffix(self._pause_path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(self._pause_path)

    # Override mutating methods to persist changes.

    def record_settlement(self, bankroll: float, start_at: int) -> None:
        prev_len = len(self._entries)
        super().record_settlement(bankroll, start_at)
        if len(self._entries) > prev_len:
            # Append only the newest entry.
            self._append_history(self._entries[-1])
        elif prev_len == 0 and len(self._entries) == 1:
            # Seeded with init entry on this call.
            self._append_history(self._entries[-1])

    def set_paused(self, cooldown_rounds: int, triggered_at: int) -> None:
        super().set_paused(cooldown_rounds, triggered_at)
        self._persist_pause_state()

    def tick_cooldown(self) -> None:
        was_paused = self._cooldown > 0
        super().tick_cooldown()
        # Persist on every tick while paused; on unpause the file reflects paused=false.
        if was_paused:
            self._persist_pause_state()
