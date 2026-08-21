"""Per-method read routing + censored getLogs latency observability.

Era 13 split: eth_getLogs goes to RPC_GETLOGS_ENDPOINT, every other read
stays on bloXroute. The latency metric must count TIMEOUTS as censored
observations at the bound — dropping them would make it report a healthy
p99 during an outage where 100% of calls time out, which is exactly the
2026-08-17 failure mode it exists to surface.
"""
import time

import pytest

from pancakebot.chain.rpc_poller import (
    RPC_BLOXROUTE_ENDPOINT,
    RPC_GETLOGS_ENDPOINT,
    _GETLOGS_TIMEOUT_MS,
    RpcPoller,
    _endpoint_for,
)


def test_only_getlogs_is_routed_away_from_bloxroute():
    assert _endpoint_for("eth_getLogs") == RPC_GETLOGS_ENDPOINT
    assert RPC_GETLOGS_ENDPOINT != RPC_BLOXROUTE_ENDPOINT
    for method in ("eth_blockNumber", "eth_getBlockByNumber", "eth_call",
                   "eth_getBlockReceipts", "anything_else"):
        assert _endpoint_for(method) == RPC_BLOXROUTE_ENDPOINT


def test_timeout_fits_two_attempts_inside_the_single_poll_cap():
    """The old 600ms silently disabled the in-call retry: a second attempt
    could never start inside the 950ms cap. The re-derived value must."""
    from pancakebot import timing_constants as _tc
    from pancakebot.chain.rpc_poller import _BLX_RETRY_BACKOFF_MS
    two_attempts = 2 * _GETLOGS_TIMEOUT_MS + _BLX_RETRY_BACKOFF_MS
    assert two_attempts < _tc.RPC_POLL_WALL_CAP_SINGLE_MS


def test_stats_surface_names_both_endpoints():
    p = RpcPoller(interval_seconds=300)
    st = p.stats
    assert st["current_endpoint"] == RPC_BLOXROUTE_ENDPOINT
    assert st["getlogs_endpoint"] == RPC_GETLOGS_ENDPOINT
    assert st["getlogs_censored_samples"] == 0


def test_p99_none_until_enough_samples():
    p = RpcPoller(interval_seconds=300)
    assert p.getlogs_p99_ms is None
    for _ in range(19):
        p._record_getlogs_latency(20.0, censored=False)
    assert p.getlogs_p99_ms is None
    p._record_getlogs_latency(20.0, censored=False)
    assert p.getlogs_p99_ms == 20


def test_timeouts_are_censored_observations_not_dropped():
    """The regression guard: an all-timeout window must report >= the
    timeout bound, never a healthy number and never None."""
    p = RpcPoller(interval_seconds=300)
    for _ in range(50):
        p._record_getlogs_latency(float(_GETLOGS_TIMEOUT_MS), censored=True)
    assert p.getlogs_p99_ms is not None
    assert p.getlogs_p99_ms >= _GETLOGS_TIMEOUT_MS
    assert p.stats["getlogs_censored_samples"] == 50


def test_healthy_samples_do_not_mask_a_degrading_tail():
    p = RpcPoller(interval_seconds=300)
    for _ in range(95):
        p._record_getlogs_latency(15.0, censored=False)
    for _ in range(5):
        p._record_getlogs_latency(float(_GETLOGS_TIMEOUT_MS), censored=True)
    assert p.getlogs_p99_ms >= _GETLOGS_TIMEOUT_MS


def test_transport_records_timeout_but_not_fast_rpc_errors(monkeypatch):
    p = RpcPoller(interval_seconds=300)

    def slow_boom(url, body, *, timeout_seconds):
        time.sleep(timeout_seconds)
        raise TimeoutError("read timed out")

    monkeypatch.setattr(p, "_rpc_post", slow_boom)
    with pytest.raises(TimeoutError):
        p._bloxroute_call("eth_getLogs", [{}], timeout_ms=20, attempts=1)
    assert p.stats["getlogs_censored_samples"] == 1

    def fast_boom(url, body, *, timeout_seconds):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(p, "_rpc_post", fast_boom)
    with pytest.raises(ConnectionRefusedError):
        p._bloxroute_call("eth_getLogs", [{}], timeout_ms=20, attempts=1)
    # a fast rejection is not a latency observation
    assert p.stats["getlogs_censored_samples"] == 1

    # and a non-getLogs method never feeds the getLogs metric
    monkeypatch.setattr(p, "_rpc_post", slow_boom)
    with pytest.raises(TimeoutError):
        p._bloxroute_call("eth_blockNumber", [], timeout_ms=20, attempts=1)
    assert p.stats["getlogs_censored_samples"] == 1


def test_transport_posts_getlogs_to_the_getlogs_host(monkeypatch):
    p = RpcPoller(interval_seconds=300)
    seen = []

    def capture(url, body, *, timeout_seconds):
        seen.append(url)
        return b'{"jsonrpc":"2.0","id":1,"result":[]}'

    monkeypatch.setattr(p, "_rpc_post", capture)
    assert p._bloxroute_call("eth_getLogs", [{}], timeout_ms=50, attempts=1) == []
    p._bloxroute_call("eth_blockNumber", [], timeout_ms=50, attempts=1)
    assert seen == [RPC_GETLOGS_ENDPOINT, RPC_BLOXROUTE_ENDPOINT]
    # a successful call is recorded as an uncensored sample
    assert p.stats["getlogs_censored_samples"] == 0


def test_alarm_payload_carries_the_censored_p99():
    from pancakebot.runtime.pool_gate_alarm import PoolGateAlarm
    a = PoolGateAlarm(threshold=2)
    a.record(ready=False, reason="pool_uncovered", epoch=1, now=0.0,
             blocks_short=655, getlogs_p99_ms=250)
    ev = a.record(ready=False, reason="pool_uncovered", epoch=2, now=300.0,
                  blocks_short=655, getlogs_p99_ms=250)
    assert ev.fields["getlogs_p99_ms"] == ">=250"
    assert "getlogs_p99_ms=>=250" in ev.detail
