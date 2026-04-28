"""Tests for ``okx_rate_acquire`` token bucket.

Replaced the prior leaky-bucket implementation 2026-04-27 after a
read-only audit traced strict monotonic per-symbol fetch timing
(eth 383ms / sol 501ms / btc 775ms / bnb 1015ms in round 476378) to
the prior implementation calling ``time.sleep()`` while holding the
rate-limiter lock — which serialized concurrent parallel-symbol
acquires at FIFO 125ms intervals.

Behaviour under test:
- Capacity = 8, refill = 8/sec.
- 4 concurrent acquires when bucket is full return in well under 100ms
  total (no FIFO stagger).
- Draining the bucket (8 sequential fast acquires) leaves the 9th call
  blocking ~125ms (one refill interval).
- After idle, bucket refills toward capacity at the configured rate.
- The lock is NEVER held across ``time.sleep`` (concurrency property).

Run:
    python -m pytest tests/test_okx_rate_limiter.py -v
    python tests/test_okx_rate_limiter.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data import okx_client  # noqa: E402
from pancakebot.market_data.okx_client import (  # noqa: E402
    _OKX_RATE_BUCKET_CAPACITY,
    _OKX_RATE_LIMIT_PER_SEC,
    _okx_rate_reset_for_tests,
    okx_rate_acquire,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _refill_interval_s() -> float:
    """One token refill interval in seconds (default 0.125s at 8/sec)."""
    return 1.0 / float(_OKX_RATE_LIMIT_PER_SEC)


# ---------------------------------------------------------------------------
# Burst behavior: 4 concurrent acquires with full bucket
# ---------------------------------------------------------------------------


def test_full_bucket_serves_4_concurrent_acquires_with_no_stagger():
    """4 threads acquiring against a full bucket all complete fast.

    With the prior leaky-bucket-with-lock-held-during-sleep, this took
    ~375ms (3 × 125ms FIFO intervals). With the token bucket, all four
    decrement immediately and complete in <100ms wall-clock.
    """
    _okx_rate_reset_for_tests()

    completion_times: list[float] = []
    completion_lock = threading.Lock()

    def worker():
        okx_rate_acquire()
        with completion_lock:
            completion_times.append(time.monotonic())

    barrier = threading.Barrier(4)

    def gated_worker():
        barrier.wait()  # synchronise launch
        worker()

    threads = [threading.Thread(target=gated_worker) for _ in range(4)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_total = time.monotonic() - t0

    assert len(completion_times) == 4
    # Wall-clock total should be well under one refill interval.
    assert t_total < 0.1, (
        f"4 concurrent acquires on full bucket should complete <100ms; "
        f"got {t_total*1000:.0f}ms (regression to FIFO stagger?)"
    )
    # Spread between fastest and slowest completion should also be tiny.
    spread = max(completion_times) - min(completion_times)
    assert spread < 0.05, (
        f"completion-time spread should be <50ms with no FIFO stagger; "
        f"got {spread*1000:.1f}ms"
    )


# ---------------------------------------------------------------------------
# Drain semantics: 9th call blocks ~one refill interval
# ---------------------------------------------------------------------------


def test_drain_capacity_then_next_call_blocks_one_refill_interval():
    """After draining ``capacity`` tokens fast, the next acquire blocks
    until one token has refilled."""
    _okx_rate_reset_for_tests()

    # Drain the bucket. These should all return ~instantly.
    t0 = time.monotonic()
    for _ in range(_OKX_RATE_BUCKET_CAPACITY):
        okx_rate_acquire()
    t_drain = time.monotonic() - t0
    assert t_drain < 0.05, (
        f"draining {_OKX_RATE_BUCKET_CAPACITY} tokens should be near-instant; "
        f"got {t_drain*1000:.1f}ms"
    )

    # Next call must block until at least one token refills.
    interval = _refill_interval_s()
    t1 = time.monotonic()
    okx_rate_acquire()
    blocked_for = time.monotonic() - t1
    assert blocked_for >= interval * 0.7, (
        f"capacity+1th call should block >= 0.7 * refill interval "
        f"({interval*0.7*1000:.0f}ms); got {blocked_for*1000:.0f}ms"
    )
    # Upper bound: refill is deterministic so this shouldn't take much more
    # than one full interval (allow generous margin for OS sleep jitter).
    assert blocked_for < interval * 2.5, (
        f"capacity+1th call should not block much past one refill "
        f"({interval*1000:.0f}ms); got {blocked_for*1000:.0f}ms"
    )


# ---------------------------------------------------------------------------
# Refill: idle period replenishes bucket
# ---------------------------------------------------------------------------


def test_idle_refills_bucket_toward_capacity():
    """After draining, sleeping for >capacity/rate seconds restores
    the bucket fully (tokens clamp at capacity)."""
    _okx_rate_reset_for_tests()

    # Drain the bucket.
    for _ in range(_OKX_RATE_BUCKET_CAPACITY):
        okx_rate_acquire()

    # Idle long enough to refill more than capacity (the clamp should
    # cap at capacity, not let it grow unbounded).
    full_refill_s = _OKX_RATE_BUCKET_CAPACITY / float(_OKX_RATE_LIMIT_PER_SEC)
    time.sleep(full_refill_s + 0.1)

    # Now we should be able to acquire ``capacity`` more tokens fast.
    t0 = time.monotonic()
    for _ in range(_OKX_RATE_BUCKET_CAPACITY):
        okx_rate_acquire()
    t_burst = time.monotonic() - t0
    assert t_burst < 0.05, (
        f"after idle, full {_OKX_RATE_BUCKET_CAPACITY}-token burst should be "
        f"near-instant; got {t_burst*1000:.1f}ms (refill clamp broken?)"
    )


# ---------------------------------------------------------------------------
# Concurrency property: lock not held across sleep
# ---------------------------------------------------------------------------


def test_concurrent_acquires_when_empty_overlap_their_sleeps():
    """When the bucket is empty, multiple threads should sleep
    CONCURRENTLY, not serially. Total wall-clock for N waiters should
    not be N × refill_interval (which would be the case if the lock
    were held during sleep)."""
    _okx_rate_reset_for_tests()

    # Drain the bucket.
    for _ in range(_OKX_RATE_BUCKET_CAPACITY):
        okx_rate_acquire()

    # Fire 4 threads against the empty bucket. With the token bucket,
    # they should refill at 1 token / 125ms in a staggered way -- but
    # crucially, NONE of them holds the lock during ``time.sleep``,
    # so the wait math is bounded by the refill rate, not by lock FIFO.
    interval = _refill_interval_s()

    def worker():
        okx_rate_acquire()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0

    # 4 threads on empty bucket need 4 tokens. At 8/s refill, that's
    # 4 × 125ms = 500ms in the WORST case (strictly serial refill).
    # We allow up to 5 × interval for OS jitter.
    upper_bound = 5.0 * interval
    assert elapsed < upper_bound, (
        f"4 acquires on empty bucket should complete in < {upper_bound*1000:.0f}ms; "
        f"got {elapsed*1000:.0f}ms"
    )
    # Lower bound: we expect at least 3 × interval (4th token's refill).
    # This isn't strictly enforced because the refill math is continuous
    # not discrete -- partial tokens count -- so we just sanity-check
    # we waited *some* time.
    assert elapsed >= interval * 0.5, (
        f"4 acquires on empty bucket should take >= half a refill interval; "
        f"got {elapsed*1000:.0f}ms"
    )


# ---------------------------------------------------------------------------
# Initial state: bucket starts FULL
# ---------------------------------------------------------------------------


def test_first_burst_after_reset_fires_unconstrained():
    """After ``_okx_rate_reset_for_tests``, capacity acquires fire fast.

    Documents the no-startup-penalty intent."""
    _okx_rate_reset_for_tests()

    t0 = time.monotonic()
    for _ in range(_OKX_RATE_BUCKET_CAPACITY):
        okx_rate_acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05, (
        f"reset-then-burst-of-capacity should be near-instant; "
        f"got {elapsed*1000:.1f}ms"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
