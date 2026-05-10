"""Unit tests for RPC endpoint hedging.

Covers the fan-out behaviour, EndpointHealthTracker classification +
retest cadence, and integration at the ``_do_hedged_post`` level.

The strategy hash is preserved by these tests because the hedging
layer is below the result-parsing logic — the canonical 5-fold +
holdout assertions in ``tests/test_in_process_runner.py`` are the
end-to-end check that's still bit-identical.

Tests use ``unittest.mock`` to patch ``_rpc_post`` (the lowest-level
HTTP call) with controllable timing and outcomes.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain.rpc_poller import (  # noqa: E402
    DEFAULT_HEDGED_ENDPOINTS,
    EndpointHealthTracker,
    HedgedAllFailed,
    RpcPoller,
)
from pancakebot.util import InvariantError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_poller(
    *,
    endpoint_pool: list[str] | None = None,
    hedge_fan_out: int = 1,
) -> RpcPoller:
    return RpcPoller(
        interval_seconds=300,
        endpoint_pool=endpoint_pool or ["https://test.example.com"],
        hedge_fan_out=hedge_fan_out,
    )


def _make_responder(*, response: bytes, sleep_s: float = 0.0,
                    raises: BaseException | None = None):
    """Build a fake ``_rpc_post`` callable with controllable timing."""
    def fn(url, body, *, timeout_seconds):
        if sleep_s > 0:
            time.sleep(sleep_s)
        if raises is not None:
            raise raises
        return response
    return fn


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_invalid_hedge_fan_out_zero_raises():
    """fan_out < 1 is invalid."""
    with pytest.raises(InvariantError, match="hedge_fan_out_below_one"):
        RpcPoller(
            interval_seconds=300,
            endpoint_pool=["https://a"],
            hedge_fan_out=0,
        )


def test_invalid_hedge_fan_out_exceeds_pool_size_raises():
    """fan_out > len(pool) is a misconfig — must crash loudly."""
    with pytest.raises(InvariantError, match="hedge_fan_out_exceeds_pool_size"):
        RpcPoller(
            interval_seconds=300,
            endpoint_pool=["https://a"],
            hedge_fan_out=3,
        )


def test_invalid_hedge_fan_out_three_with_two_endpoints_raises():
    """Same misconfig — slightly different sizes for confidence."""
    with pytest.raises(InvariantError, match="hedge_fan_out_exceeds_pool_size"):
        RpcPoller(
            interval_seconds=300,
            endpoint_pool=["https://a", "https://b"],
            hedge_fan_out=3,
        )


def test_default_hedged_endpoints_constant_is_three():
    """Spec contract: top-3 endpoints from Track H. If this changes,
    the memo (and any operator config relying on it) needs updating."""
    assert len(DEFAULT_HEDGED_ENDPOINTS) == 3
    # Specific endpoints from memo. If you tune the pool, update both.
    assert DEFAULT_HEDGED_ENDPOINTS[0].endswith("defibit.io")
    assert DEFAULT_HEDGED_ENDPOINTS[1].endswith("ninicoin.io")
    assert DEFAULT_HEDGED_ENDPOINTS[2].endswith("binance.org")


# ---------------------------------------------------------------------------
# Fan-out behaviour
# ---------------------------------------------------------------------------

def test_fan_out_1_uses_single_endpoint():
    """At fan_out=1, _do_hedged_post takes the fast path: a single
    direct urlopen with the picked endpoint, NO threadpool, no
    fan-out overhead. External behaviour matches the pre-hedging
    code exactly."""
    p = _make_poller(endpoint_pool=["https://only.example.com"])
    body = b'{"jsonrpc":"2.0","id":1,"method":"x","params":[]}'

    calls = []

    def fake_post(url, b, *, timeout_seconds):
        calls.append((url, b, timeout_seconds))
        return b'{"jsonrpc":"2.0","id":1,"result":"ok"}'

    p._rpc_post = fake_post  # type: ignore[assignment]

    ep, resp = p._do_hedged_post(body, timeout_seconds=5)
    assert ep == "https://only.example.com"
    assert resp == b'{"jsonrpc":"2.0","id":1,"result":"ok"}'
    assert calls == [("https://only.example.com", body, 5)]
    assert p._executor is None, "fan_out=1 must not allocate executor"


def test_fan_out_2_picks_min_latency():
    """Fastest endpoint's response wins; slow endpoint's response is
    discarded. The total wallclock is bounded by the FAST endpoint's
    RTT, not the slow one."""
    pool = ["https://slow.example.com", "https://fast.example.com"]
    p = _make_poller(endpoint_pool=pool, hedge_fan_out=2)

    fast_body = b'{"jsonrpc":"2.0","id":1,"result":"fast"}'
    slow_body = b'{"jsonrpc":"2.0","id":1,"result":"slow"}'

    def fake_post(url, b, *, timeout_seconds):
        if "fast" in url:
            time.sleep(0.05)
            return fast_body
        else:
            time.sleep(0.50)
            return slow_body

    p._rpc_post = fake_post  # type: ignore[assignment]

    t0 = time.monotonic()
    ep, resp = p._do_hedged_post(b"x", timeout_seconds=5)
    elapsed = time.monotonic() - t0

    assert ep == "https://fast.example.com"
    assert resp == fast_body
    # Bounded by fast RTT (0.05s) + small overhead, well below slow RTT.
    assert elapsed < 0.40, f"hedging took {elapsed:.3f}s; expected ~0.05s"


def test_fan_out_2_handles_one_endpoint_failure():
    """One endpoint raises; the other succeeds. Result returned;
    failing endpoint marked unhealthy."""
    pool = ["https://broken.example.com", "https://good.example.com"]
    p = _make_poller(endpoint_pool=pool, hedge_fan_out=2)

    good_body = b'{"jsonrpc":"2.0","id":1,"result":"good"}'

    def fake_post(url, b, *, timeout_seconds):
        if "broken" in url:
            raise ConnectionError("simulated_outage")
        time.sleep(0.05)  # ensure broken's failure registers first
        return good_body

    p._rpc_post = fake_post  # type: ignore[assignment]

    ep, resp = p._do_hedged_post(b"x", timeout_seconds=5)
    assert ep == "https://good.example.com"
    assert resp == good_body

    # Health tracker: broken=failure, good=success.
    stats = p._health.stats()
    assert stats["https://broken.example.com"]["consecutive_failures"] == 1
    assert stats["https://good.example.com"]["consecutive_failures"] == 0


def test_fan_out_3_handles_two_endpoint_failures():
    """Two endpoints raise; one succeeds. Result returned."""
    pool = ["https://bad1.example.com", "https://bad2.example.com",
            "https://good.example.com"]
    p = _make_poller(endpoint_pool=pool, hedge_fan_out=3)

    good_body = b'{"jsonrpc":"2.0","id":1,"result":"good"}'

    def fake_post(url, b, *, timeout_seconds):
        if "bad" in url:
            raise ConnectionError(f"outage_{url}")
        time.sleep(0.10)  # ensure bad's failures register first
        return good_body

    p._rpc_post = fake_post  # type: ignore[assignment]

    ep, resp = p._do_hedged_post(b"x", timeout_seconds=5)
    assert ep == "https://good.example.com"
    assert resp == good_body

    stats = p._health.stats()
    assert stats["https://bad1.example.com"]["consecutive_failures"] == 1
    assert stats["https://bad2.example.com"]["consecutive_failures"] == 1
    assert stats["https://good.example.com"]["consecutive_failures"] == 0


def test_all_endpoints_fail_raises_composite_error():
    """Every endpoint raises -> HedgedAllFailed with all errors
    surfaced in the message."""
    pool = ["https://a.example.com", "https://b.example.com",
            "https://c.example.com"]
    p = _make_poller(endpoint_pool=pool, hedge_fan_out=3)

    def fake_post(url, b, *, timeout_seconds):
        raise ConnectionError(f"down_{url}")

    p._rpc_post = fake_post  # type: ignore[assignment]

    with pytest.raises(HedgedAllFailed) as exc_info:
        p._do_hedged_post(b"x", timeout_seconds=5)

    msg = str(exc_info.value)
    assert "all_hedged_endpoints_failed (3)" in msg
    # All three endpoints listed.
    assert "a.example.com" in msg
    assert "b.example.com" in msg
    assert "c.example.com" in msg
    assert len(exc_info.value.errors) == 3


# ---------------------------------------------------------------------------
# EndpointHealthTracker — classification + retest
# ---------------------------------------------------------------------------

def test_health_tracker_warmup_is_healthy_by_default():
    """Brand-new endpoint with no recorded outcomes: healthy."""
    h = EndpointHealthTracker(["https://a"])
    assert h.is_healthy("https://a") is True


def test_health_tracker_marks_unhealthy_on_consecutive_failures():
    """5 consecutive failures triggers fast-trip (consecutive_failures
    gate), even before the rolling window fills."""
    h = EndpointHealthTracker(["https://a"])
    for _ in range(4):
        h.record("https://a", success=False, rtt_ms=100)
    assert h.is_healthy("https://a") is True, "4 fails: still healthy"
    h.record("https://a", success=False, rtt_ms=100)
    assert h.is_healthy("https://a") is False, "5th fail: tripped"


def test_health_tracker_rolling_window_success_rate_gate():
    """Once the window fills (100 outcomes), success_rate <= 0.90 -> unhealthy."""
    h = EndpointHealthTracker(["https://a"])
    # 89 successes + 11 failures interleaved so consecutive_failures resets.
    # We want consecutive_failures < 5 so only the success_rate gate fires.
    for i in range(100):
        # fail every 10th so consec_failures stays at 1.
        success = (i % 10 != 0)
        h.record("https://a", success=success, rtt_ms=100)
    # 90 successes / 100: success_rate = 0.90, threshold is "> 0.90" so
    # exactly 0.90 should NOT pass. Spec: success_rate > 0.90.
    # Implementation rejects when (successes/n) <= 0.90. So 0.90 -> unhealthy.
    assert h.is_healthy("https://a") is False


def test_health_tracker_rolling_window_recovers():
    """Window full of mostly successes (95%) -> healthy."""
    h = EndpointHealthTracker(["https://a"])
    for i in range(100):
        success = (i % 20 != 0)  # 95 successes / 100
        h.record("https://a", success=success, rtt_ms=100)
    assert h.is_healthy("https://a") is True


def test_health_tracker_excludes_unhealthy_from_pick_n():
    """When healthy endpoints are available, pick_n returns only the
    healthy ones (no fallback to unhealthy needed)."""
    h = EndpointHealthTracker(["https://good", "https://broken"])
    # broken: 5 consecutive failures -> unhealthy.
    for _ in range(5):
        h.record("https://broken", success=False, rtt_ms=100)
    # good: 1 success -> healthy.
    h.record("https://good", success=True, rtt_ms=100)

    # Reset the global pick counter to bypass periodic-retest pressure
    # for THIS particular pick. Use a small N.
    picks = h.pick_n(1)
    assert picks == ["https://good"]


def test_health_tracker_picks_fastest_healthy_first():
    """pick_n sorts healthy endpoints by p50 RTT ascending."""
    h = EndpointHealthTracker(["https://slow", "https://medium", "https://fast"])
    # 5 outcomes per endpoint with distinct p50.
    for _ in range(5):
        h.record("https://slow", success=True, rtt_ms=900)
        h.record("https://medium", success=True, rtt_ms=500)
        h.record("https://fast", success=True, rtt_ms=100)
    picks = h.pick_n(3)
    assert picks == ["https://fast", "https://medium", "https://slow"]


def test_health_tracker_falls_back_to_unhealthy_when_short():
    """If only 1 healthy endpoint exists but pick_n(3) requested,
    fall back to unhealthy ones to fill the count (degraded mode
    beats no-RPC mode)."""
    h = EndpointHealthTracker(["https://good", "https://bad1", "https://bad2"])
    h.record("https://good", success=True, rtt_ms=100)
    for _ in range(5):
        h.record("https://bad1", success=False, rtt_ms=100)
        h.record("https://bad2", success=False, rtt_ms=100)
    picks = h.pick_n(3)
    assert len(picks) == 3
    assert "https://good" in picks
    assert "https://bad1" in picks
    assert "https://bad2" in picks


def test_health_tracker_periodic_retest():
    """Even when healthy alternatives exist, an unhealthy endpoint
    gets re-tested every Nth pick (1-in-10 retry pressure) AND when
    it hasn't been probed in 60+ seconds. Verify the 60s-stall path."""
    h = EndpointHealthTracker(["https://good1", "https://good2", "https://broken"])
    # broken: 5 fails ago = unhealthy.
    for _ in range(5):
        h.record("https://broken", success=False, rtt_ms=100)
    h.record("https://good1", success=True, rtt_ms=200)
    h.record("https://good2", success=True, rtt_ms=300)

    # Force broken's last_request_at to be >60s ago to trigger retest.
    with h._lock:
        h._health["https://broken"].last_request_at = time.time() - 120

    picks = h.pick_n(2)
    # broken should be included as the retest candidate; one healthy joins.
    assert "https://broken" in picks
    assert len(picks) == 2


def test_health_tracker_records_under_pick_n_counter_for_one_in_ten():
    """1-in-10 retest cadence kicks in even without 60s stall. After
    10 picks, broken should be included at least once."""
    h = EndpointHealthTracker(["https://good", "https://broken"])
    for _ in range(5):
        h.record("https://broken", success=False, rtt_ms=100)
    h.record("https://good", success=True, rtt_ms=200)

    # Reset broken's last_request_at to "now" so the 60s-forced retest
    # does NOT kick in. Only the 10th-pick counter should trigger inclusion.
    with h._lock:
        h._health["https://broken"].last_request_at = time.time()

    saw_broken = False
    for _ in range(15):
        picks = h.pick_n(1)
        if "https://broken" in picks:
            saw_broken = True
            break
    assert saw_broken, "broken should be retested within 15 picks"


def test_health_tracker_stats_shape():
    """stats() returns one entry per endpoint with documented fields."""
    h = EndpointHealthTracker(["https://a", "https://b"])
    h.record("https://a", success=True, rtt_ms=300)
    h.record("https://b", success=False, rtt_ms=5000)
    s = h.stats()
    assert set(s.keys()) == {"https://a", "https://b"}
    for url, fields in s.items():
        assert "healthy" in fields
        assert "success_rate" in fields
        assert "p50_rtt_ms" in fields
        assert "p99_rtt_ms" in fields
        assert "consecutive_failures" in fields
        assert "total_requests" in fields


# ---------------------------------------------------------------------------
# Integration: RpcPoller stats expose health
# ---------------------------------------------------------------------------

def test_rpc_poller_stats_expose_endpoint_health():
    """RpcPoller.stats includes per-endpoint health breakdown."""
    p = _make_poller(
        endpoint_pool=["https://a", "https://b"],
        hedge_fan_out=2,
    )
    s = p.stats
    assert "endpoint_health" in s
    assert "https://a" in s["endpoint_health"]
    assert "https://b" in s["endpoint_health"]
    assert s["hedge_fan_out"] == 2


def test_rpc_call_single_uses_hedging_at_fan_out_2():
    """End-to-end at the _rpc_call_single boundary: parses JSON-RPC
    envelope, returns result, distributes calls across endpoints."""
    pool = ["https://a", "https://b"]
    p = _make_poller(endpoint_pool=pool, hedge_fan_out=2)

    a_body = b'{"jsonrpc":"2.0","id":1,"result":"0x42"}'
    b_body = b'{"jsonrpc":"2.0","id":1,"result":"0x99"}'

    def fake_post(url, b, *, timeout_seconds):
        if "a" in url and "/" not in url[8:]:
            time.sleep(0.02)
            return a_body
        time.sleep(0.30)
        return b_body

    p._rpc_post = fake_post  # type: ignore[assignment]
    result = p._rpc_call_single("eth_blockNumber", [])
    # 'a' wins (faster); result is the 'a' body.
    assert result == "0x42"


def test_rpc_batch_uses_hedging_at_fan_out_2():
    """End-to-end at the _rpc_batch boundary: parses JSON-RPC list,
    returns aligned results, distributes batched calls across endpoints."""
    pool = ["https://a", "https://b"]
    p = _make_poller(endpoint_pool=pool, hedge_fan_out=2)

    fast_body = (
        b'[{"jsonrpc":"2.0","id":0,"result":"r0"},'
        b'{"jsonrpc":"2.0","id":1,"result":"r1"}]'
    )

    def fake_post(url, b, *, timeout_seconds):
        if "a" in url and "/" not in url[8:]:
            return fast_body
        time.sleep(0.30)
        return b'[]'  # malformed for b — but a wins, so b's body is discarded

    p._rpc_post = fake_post  # type: ignore[assignment]
    results = p._rpc_batch([("eth_blockNumber", []), ("eth_blockNumber", [])])
    assert len(results) == 2
    assert results[0] == ("r0", None)
    assert results[1] == ("r1", None)


# ---------------------------------------------------------------------------
# Health-driven endpoint rotation across many calls
# ---------------------------------------------------------------------------

def test_health_driven_deprioritisation_includes_recovery_path():
    """Endpoint that fails 5 times becomes unhealthy AND is excluded
    from pick_n until it recovers (or the periodic retest brings it back)."""
    pool = ["https://a", "https://b"]
    h = EndpointHealthTracker(pool)
    # a fails 5 times.
    for _ in range(5):
        h.record("https://a", success=False, rtt_ms=100)
    # b succeeds.
    h.record("https://b", success=True, rtt_ms=200)

    # pick_n(1): without retest pressure, only b should be picked.
    # We disable retest by setting a's last_request_at to NOW.
    with h._lock:
        h._health["https://a"].last_request_at = time.time()
    # Force pick_counter to a non-multiple of 10.
    with h._lock:
        h._pick_counter = 5

    picks = h.pick_n(1)
    assert picks == ["https://b"]

    # Now: simulate recovery — record many successes for a.
    for _ in range(20):
        h.record("https://a", success=True, rtt_ms=100)
    # a's consecutive_failures reset on first success; with <100
    # window outcomes, warmup rule says healthy. 5+20=25 outcomes, all
    # post-recovery 20 successes -> healthy.
    assert h.is_healthy("https://a") is True
