"""Unit tests for the fire-to-all-pool RPC hedging transport.

Every JSON-RPC call fires in parallel to every endpoint in the pool
via a shared ThreadPoolExecutor; first 200 response wins, the rest
are abandoned. There is no endpoint selection logic — if an endpoint
misbehaves, the operator removes it from the constant.

Tests use ``unittest.mock`` to patch ``_rpc_post`` (the lowest-level
HTTP call) with controllable timing and outcomes.

See var/incident_reports/2026_05_11_parallel_request_transport_bottleneck.md
for the design rationale (replaces the prior pick_n + per-endpoint
health-tracker model after measured 4-way parallel failures).
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
    READ_PATH_HEDGED_ENDPOINTS,
    HedgedAllFailed,
    RpcPoller,
)
from pancakebot.util import InvariantError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_poller(*, endpoint_pool: list[str] | None = None) -> RpcPoller:
    return RpcPoller(
        interval_seconds=300,
        endpoint_pool=endpoint_pool or ["https://test.example.com"],
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

def test_empty_endpoint_pool_raises():
    with pytest.raises(InvariantError, match="endpoint_pool_empty"):
        RpcPoller(interval_seconds=300, endpoint_pool=[])


def test_default_hedged_endpoints_constant_is_three():
    """Spec contract: 3 endpoints, one per fault-domain family
    (Bundle 6 trim, 2026-05-15):
      - AWS EC2 us-east (AS14618):  bsc-dataseed1.binance.org
      - AWS Global Accelerator (AS16509): bsc-dataseed1.defibit.io
      - Cloudflare (AS13335): bsc-rpc.publicnode.com

    Dropped from the prior 6-endpoint pool:
      - bsc-dataseed3.binance.org  (identical IP pool to dataseed1)
      - bsc-dataseed1.ninicoin.io  (same AWS GA family as defibit,
                                    worst anchor p50 in that family)
      - bsc.rpc.blxrbdn.com        (same AWS GA family, middling)

    Per-endpoint probe stats (n=15, 2026-05-15) showed
    bsc-dataseed1.binance.org is the only endpoint with batch p95
    under the bot's 5s timeout — it's the load-bearing endpoint
    in production. Family diversity preserved: AWS EC2 + AWS GA +
    Cloudflare → cold-start burst load (~20 batches × N endpoints)
    drops from ~120 to ~60 concurrent in-flight requests.
    """
    assert len(READ_PATH_HEDGED_ENDPOINTS) == 3
    assert "bsc-dataseed1.binance.org" in READ_PATH_HEDGED_ENDPOINTS[0]
    assert "bsc-dataseed1.defibit.io" in READ_PATH_HEDGED_ENDPOINTS[1]
    assert "bsc-rpc.publicnode.com" in READ_PATH_HEDGED_ENDPOINTS[2]


# ---------------------------------------------------------------------------
# Fire-to-all-pool transport behaviour
# ---------------------------------------------------------------------------

def test_single_endpoint_pool_returns_response():
    """At pool-size 1, _do_hedged_post takes the single-endpoint
    fast path (no executor) and returns the response."""
    body = b'{"jsonrpc":"2.0","method":"x","params":[],"id":1}'
    p = _make_poller(endpoint_pool=["https://only.example.com"])
    with mock.patch.object(
        p, "_rpc_post",
        side_effect=_make_responder(response=b'{"result":42}'),
    ) as m:
        ep, resp = p._do_hedged_post(body, timeout_seconds=5)
    assert ep == "https://only.example.com"
    assert resp == b'{"result":42}'
    assert m.call_count == 1


def test_first_response_wins_three_endpoints():
    """Three endpoints; whichever returns first wins. Use distinct
    sleep times to make the winner deterministic."""
    pool = ["https://slow.example.com", "https://medium.example.com",
            "https://fast.example.com"]
    p = _make_poller(endpoint_pool=pool)

    def per_url(url, body, *, timeout_seconds):
        if "fast" in url:
            return b'{"result":"fast"}'
        if "medium" in url:
            time.sleep(0.2)
            return b'{"result":"medium"}'
        time.sleep(2.0)
        return b'{"result":"slow"}'

    with mock.patch.object(p, "_rpc_post", side_effect=per_url):
        ep, resp = p._do_hedged_post(b"x", timeout_seconds=5)
    assert ep == "https://fast.example.com"
    assert resp == b'{"result":"fast"}'


def test_first_success_among_mixed_outcomes_wins():
    """One endpoint succeeds fast, others fail. The success wins
    even though there are failed siblings."""
    pool = ["https://broken.example.com", "https://good.example.com"]
    p = _make_poller(endpoint_pool=pool)

    def per_url(url, body, *, timeout_seconds):
        if "broken" in url:
            raise ConnectionError("oops")
        return b'{"result":"ok"}'

    with mock.patch.object(p, "_rpc_post", side_effect=per_url):
        ep, resp = p._do_hedged_post(b"x", timeout_seconds=5)
    assert ep == "https://good.example.com"
    assert resp == b'{"result":"ok"}'


def test_all_fail_raises_hedged_all_failed():
    """Every endpoint raises -> HedgedAllFailed with per-endpoint
    errors attached."""
    pool = ["https://a.example.com", "https://b.example.com",
            "https://c.example.com"]
    p = _make_poller(endpoint_pool=pool)

    def per_url(url, body, *, timeout_seconds):
        raise ConnectionError(f"down: {url}")

    with mock.patch.object(p, "_rpc_post", side_effect=per_url):
        with pytest.raises(HedgedAllFailed) as exc_info:
            p._do_hedged_post(b"x", timeout_seconds=5)
    assert len(exc_info.value.errors) == 3
    endpoints_in_errors = {ep for ep, _ in exc_info.value.errors}
    assert endpoints_in_errors == set(pool)


def test_all_timeout_raises_hedged_all_failed():
    """Every endpoint exceeds the deadline -> HedgedAllFailed with
    TimeoutError entries."""
    pool = ["https://a.example.com", "https://b.example.com"]
    p = _make_poller(endpoint_pool=pool)

    def slow(url, body, *, timeout_seconds):
        time.sleep(2.0)
        return b'{"result":"too late"}'

    with mock.patch.object(p, "_rpc_post", side_effect=slow):
        with pytest.raises(HedgedAllFailed) as exc_info:
            # 0.3s timeout < 2s sleep on every endpoint
            p._do_hedged_post(b"x", timeout_seconds=0.3)
    assert len(exc_info.value.errors) >= 1
    assert all(
        isinstance(e, TimeoutError) for _, e in exc_info.value.errors
    )


def test_single_endpoint_failure_raises_hedged_all_failed():
    """At pool-size 1, transport-level failure still produces a
    HedgedAllFailed (preserves caller's error contract)."""
    p = _make_poller(endpoint_pool=["https://only.example.com"])
    with mock.patch.object(
        p, "_rpc_post", side_effect=ConnectionError("nope"),
    ):
        with pytest.raises(HedgedAllFailed) as exc_info:
            p._do_hedged_post(b"x", timeout_seconds=5)
    assert len(exc_info.value.errors) == 1
    assert exc_info.value.errors[0][0] == "https://only.example.com"


def test_winner_updates_current_endpoint():
    pool = ["https://broken.example.com", "https://winner.example.com"]
    p = _make_poller(endpoint_pool=pool)

    def per_url(url, body, *, timeout_seconds):
        if "broken" in url:
            raise ConnectionError("broken is down")
        return b'{"result":"winner"}'

    with mock.patch.object(p, "_rpc_post", side_effect=per_url):
        ep, _ = p._do_hedged_post(b"x", timeout_seconds=5)
    assert ep == "https://winner.example.com"
    assert p.current_endpoint == "https://winner.example.com"


def test_stragglers_do_not_block_winner_return():
    """Critical latency property: when ONE endpoint returns fast and the
    rest sleep long enough to outlive any plausible bot deadline,
    _do_hedged_post must still return promptly (driven by the
    FIRST_COMPLETED winner — abandoned futures keep running in the
    executor but don't gate the caller).
    """
    pool = ["https://slow1.example.com", "https://fast.example.com",
            "https://slow2.example.com"]
    p = _make_poller(endpoint_pool=pool)

    def per_url(url, body, *, timeout_seconds):
        if "fast" in url:
            return b'{"result":"fast"}'
        # Sleep way longer than any test should tolerate.
        time.sleep(5.0)
        return b'{"result":"too late"}'

    t0 = time.monotonic()
    with mock.patch.object(p, "_rpc_post", side_effect=per_url):
        ep, _ = p._do_hedged_post(b"x", timeout_seconds=5)
    elapsed = time.monotonic() - t0
    assert ep == "https://fast.example.com"
    # Winner returns immediately; abandoned futures must not block.
    # Allow 1s for thread submit/wait overhead under load.
    assert elapsed < 1.0, f"winner-return took {elapsed:.2f}s; stragglers blocked"


def test_fires_one_request_per_pool_endpoint():
    """Verifies every endpoint in the pool receives the request
    (no selection, no fan-out N: it's len(pool))."""
    pool = ["https://a.example.com", "https://b.example.com",
            "https://c.example.com", "https://d.example.com"]
    p = _make_poller(endpoint_pool=pool)
    seen_urls: set[str] = set()
    lock = threading.Lock()

    def record(url, body, *, timeout_seconds):
        with lock:
            seen_urls.add(url)
        # Stagger so the test isn't a race for `seen` to grow
        time.sleep(0.05)
        return b'{"result":"ok"}'

    with mock.patch.object(p, "_rpc_post", side_effect=record):
        p._do_hedged_post(b"x", timeout_seconds=5)
    assert seen_urls == set(pool)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_exposes_pool_size_no_health_tracker():
    """The stats surface no longer carries pick_n / health metrics —
    just the pool size and per-call status."""
    pool = ["https://a.example.com", "https://b.example.com"]
    p = _make_poller(endpoint_pool=pool)
    s = p.stats
    assert s["endpoint_pool_size"] == 2
    assert "endpoint_health" not in s
    assert "hedge_fan_out" not in s


# ---------------------------------------------------------------------------
# Integration: _rpc_call_single hits the hedged transport
# ---------------------------------------------------------------------------

def test_rpc_call_single_uses_hedged_transport():
    pool = ["https://a.example.com", "https://b.example.com"]
    p = _make_poller(endpoint_pool=pool)
    with mock.patch.object(
        p, "_rpc_post",
        side_effect=_make_responder(response=b'{"jsonrpc":"2.0","id":1,"result":"0x10"}'),
    ):
        out = p._rpc_call_single("eth_blockNumber", [])
    assert out == "0x10"
