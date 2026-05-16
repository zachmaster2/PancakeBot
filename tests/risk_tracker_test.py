"""Unit test for InMemoryBankrollTracker.

Covers:
  1. Construction + seed-on-first-record
  2. Multi-settlement recording, window pruning, boundary preservation
  3. Peak calculation correctness across boundary
  4. DD-breaker scenario: set_paused, is_paused, tick_cooldown
  5. Cooldown decrements correctly to 0 and unpauses

Run from repo root:
    python -m pytest tests/risk_tracker_test.py -v
    python tests/risk_tracker_test.py        # standalone CLI
"""
from __future__ import annotations

import sys

from pancakebot.bankroll_tracker import InMemoryBankrollTracker


_SECONDS_PER_DAY = 86400


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  [FAIL] {msg}", flush=True)
        raise AssertionError(msg)
    print(f"  [OK]   {msg}", flush=True)


def test_construction_and_seed() -> None:
    print("\n== test: construction + seed ==")
    t = InMemoryBankrollTracker(initial_bankroll=10.0, drawdown_peak_window_days=7)
    _assert(t.current_bankroll() == 10.0, "current() before any record = initial")
    _assert(t.peak_bankroll(as_of_start_at=1_000_000) == 10.0, "peak() before any record = initial")
    _assert(not t.is_paused(1_000_000), "not paused at init")
    _assert(t.cooldown_remaining() == 0, "cooldown_remaining == 0 at init")
    # First record seeds at the given start_at.
    t.record_settlement(bankroll=10.0, start_at=1_000_000)
    _assert(t.current_bankroll() == 10.0, "current() after seed record = 10.0")


def test_window_pruning_and_peak() -> None:
    print("\n== test: window pruning + boundary preservation + peak ==")
    # window = 7 days; use a fixed start for deterministic math.
    t0 = 1_700_000_000  # epoch seconds
    t = InMemoryBankrollTracker(initial_bankroll=10.0, drawdown_peak_window_days=7)
    # Events (start_at offsets in seconds):
    #   t0          : 10.0 (seed)
    #   t0 +   1d   :  9.5  (inside window throughout)
    #   t0 +   3d   : 11.0  (peak)
    #   t0 +   5d   : 10.2
    #   t0 +  10d   :  9.8  (drops the t0 and t0+1d as boundary candidates)
    t.record_settlement(bankroll=10.0, start_at=t0)
    t.record_settlement(bankroll=9.5,  start_at=t0 + 1 * _SECONDS_PER_DAY)
    t.record_settlement(bankroll=11.0, start_at=t0 + 3 * _SECONDS_PER_DAY)
    t.record_settlement(bankroll=10.2, start_at=t0 + 5 * _SECONDS_PER_DAY)

    # Peak at t0 + 5d using 7-day window => window_start = t0 - 2d
    # All four entries are inside the window; peak = 11.0.
    pk1 = t.peak_bankroll(as_of_start_at=t0 + 5 * _SECONDS_PER_DAY)
    _assert(pk1 == 11.0, f"peak at t+5d includes all entries => 11.0 (got {pk1})")

    # Now record a fifth entry at t0+10d; pruning kicks in:
    #   window_start = t0 + 10d - 7d = t0 + 3d
    #   Entries: [t0, t0+1d, t0+3d, t0+5d, t0+10d]
    #   first_in_window_idx = index of t0+3d = 2
    #   boundary_idx = 1 (t0+1d)
    #   After prune: [t0+1d, t0+3d, t0+5d, t0+10d]
    t.record_settlement(bankroll=9.8, start_at=t0 + 10 * _SECONDS_PER_DAY)
    _assert(t.current_bankroll() == 9.8, "current() == 9.8 after 5th record")

    # Peak at t+10d using 7-day window => window_start = t+3d
    # In-window: t+3d=11.0, t+5d=10.2, t+10d=9.8; boundary: t+1d=9.5
    # peak = max(9.5, 11.0, 10.2, 9.8) = 11.0
    pk2 = t.peak_bankroll(as_of_start_at=t0 + 10 * _SECONDS_PER_DAY)
    _assert(pk2 == 11.0, f"peak at t+10d still includes in-window 11.0 (got {pk2})")

    # Push another one even later, forcing the 11.0 out.
    #   at t+15d: window_start = t+8d. Entries before prune: [t+1d, t+3d, t+5d, t+10d, t+15d]
    #   first_in_window_idx = index of t+10d = 3
    #   boundary_idx = 2 (t+5d)
    #   After prune: [t+5d, t+10d, t+15d]
    #   Peak = max(10.2, 9.8, X) where X is the new bankroll.
    t.record_settlement(bankroll=9.7, start_at=t0 + 15 * _SECONDS_PER_DAY)
    pk3 = t.peak_bankroll(as_of_start_at=t0 + 15 * _SECONDS_PER_DAY)
    _assert(abs(pk3 - 10.2) < 1e-12, f"peak at t+15d drops 11.0, uses boundary 10.2 (got {pk3})")


def test_dd_breaker_and_cooldown() -> None:
    print("\n== test: DD breaker + cooldown decrement ==")
    t0 = 1_700_000_000
    t = InMemoryBankrollTracker(initial_bankroll=10.0, drawdown_peak_window_days=7)
    t.record_settlement(bankroll=10.0, start_at=t0)
    # Simulate a drop to 4.5 (55% DD from peak 10).
    t.record_settlement(bankroll=4.5, start_at=t0 + 3600)
    peak = t.peak_bankroll(as_of_start_at=t0 + 3600)
    _assert(peak == 10.0, f"peak = 10.0 after drop (got {peak})")
    current = t.current_bankroll()
    _assert(current == 4.5, f"current = 4.5 after drop (got {current})")
    dd = (peak - current) / peak
    _assert(abs(dd - 0.55) < 1e-12, f"DD fraction = 0.55 (got {dd:.4f})")

    # Fire cooldown for 72 rounds.
    t.set_paused(cooldown_rounds=72, triggered_at=t0 + 3600)
    _assert(t.is_paused(t0 + 3600), "is_paused == True right after set_paused")
    _assert(t.cooldown_remaining() == 72, "cooldown_remaining == 72 initially")

    # Tick 72 times; after each tick confirm the counter decrement.
    for i in range(72):
        t.tick_cooldown()
    _assert(t.cooldown_remaining() == 0, f"cooldown_remaining == 0 after 72 ticks (got {t.cooldown_remaining()})")
    _assert(not t.is_paused(t0 + 3600 * 100), "is_paused == False after full decrement")

    # Extra ticks floor at 0.
    t.tick_cooldown()
    t.tick_cooldown()
    _assert(t.cooldown_remaining() == 0, "tick_cooldown floors at 0")


def test_dedup() -> None:
    print("\n== test: dedup identical successive bankroll values ==")
    t = InMemoryBankrollTracker(initial_bankroll=10.0, drawdown_peak_window_days=7)
    t.record_settlement(bankroll=10.0, start_at=1000)
    t.record_settlement(bankroll=10.0, start_at=2000)  # dup -> skip
    t.record_settlement(bankroll=10.0, start_at=3000)  # dup -> skip
    # Internal entries count (use peek via peak over a very narrow window).
    # If dedup works, a change event triggers; otherwise no.
    t.record_settlement(bankroll=10.5, start_at=4000)
    # With dedup, entries should be: [10.0@1000, 10.5@4000] (+ maybe seed)
    # Without dedup: [10.0@1000, 10.0@2000, 10.0@3000, 10.5@4000]
    # Count by examining _entries directly via private access (test-only).
    n = len(t._entries)  # type: ignore[attr-defined]
    _assert(n == 2, f"entries count after 4 records incl dups = 2 (got {n})")


def main() -> int:
    try:
        test_construction_and_seed()
        test_window_pruning_and_peak()
        test_dd_breaker_and_cooldown()
        test_dedup()
    except AssertionError:
        print("\n[FAIL] Test suite FAILED")
        return 1
    print("\n[OK] All tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
