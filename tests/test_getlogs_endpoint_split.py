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

# ---- periodic health line (observability) ---------------------------------

def test_stats_exposes_error_counter():
    p = RpcPoller(interval_seconds=300)
    assert p.stats["getlogs_errors"] == 0


def test_health_line_reports_host_and_censored_p99(monkeypatch):
    """Emitted once per round even when healthy: silence was the failure
    mode of the 2026-08-17 outage."""
    import pancakebot.chain.rpc_poller as mod
    p = RpcPoller(interval_seconds=300)
    lines = []
    monkeypatch.setattr(mod, "info", lambda *a, **k: lines.append(a))

    p._log_getlogs_health()
    assert lines and lines[-1][0] == "POLL"
    assert "p99=n/a" in lines[-1][1]              # below the sample floor
    assert RPC_GETLOGS_ENDPOINT in lines[-1][1]

    # an UNcensored p99 is an exact measurement: no ">=" prefix, or a bound
    # becomes indistinguishable from a measurement
    for _ in range(25):
        p._record_getlogs_latency(20.0, censored=False)
    p._log_getlogs_health()
    assert "p99=20ms" in lines[-1][1]
    assert ">=" not in lines[-1][1]
    assert "censored=0" in lines[-1][1]

    for _ in range(30):
        p._record_getlogs_latency(float(_GETLOGS_TIMEOUT_MS), censored=True)
    p._log_getlogs_health()
    # censored samples must surface as a lower bound, never as a flat number
    assert ">=%dms" % _GETLOGS_TIMEOUT_MS in lines[-1][1]
    assert "censored=30" in lines[-1][1]


def test_transport_counts_every_getlogs_error_not_just_timeouts(monkeypatch):
    p = RpcPoller(interval_seconds=300)

    def fast_boom(url, body, *, timeout_seconds):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(p, "_rpc_post", fast_boom)
    with pytest.raises(ConnectionRefusedError):
        p._bloxroute_call("eth_getLogs", [{}], timeout_ms=20, attempts=1)
    # a fast rejection is not a LATENCY sample, but it IS an error
    assert p.stats["getlogs_censored_samples"] == 0
    assert p.stats["getlogs_errors"] == 1


# ---- diagnosability: swallowed causes must survive ------------------------

def test_round_start_block_failure_carries_its_cause(monkeypatch):
    """`except Exception: return None` discarded the cause, so a timeout, an
    HTTP error and a malformed result all read as "RPC failed" -- which is
    why the 2026-08 header-path degradation went a week uncharacterised.
    Fail-safe behaviour is unchanged; only the cause is added."""
    p = RpcPoller(interval_seconds=300)

    def boom(**kwargs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(p, "_bloxroute_latest_header", boom)
    assert p._compute_round_start_block(1_787_000_000) is None   # still None
    assert "TimeoutError" in p._last_rs_block_error
    assert "read timed out" in p._last_rs_block_error


def test_round_start_block_success_clears_the_stale_cause(monkeypatch):
    p = RpcPoller(interval_seconds=300)
    p._last_rs_block_error = "TimeoutError: stale"
    monkeypatch.setattr(p, "_bloxroute_latest_header",
                        lambda **kw: (1_000_000, 1_787_000_000, None))
    p._compute_round_start_block(1_787_000_000)
    assert p._last_rs_block_error is None


def test_anchor_poll_fallback_names_the_exception_type(monkeypatch):
    """"timeout_or_transport" alone cannot separate a read timeout from a
    connection reset from a malformed-JSON parse."""
    p = RpcPoller(interval_seconds=300)
    seen = []
    monkeypatch.setattr(p, "_record_anchor_outcome",
                        lambda **kw: seen.append(kw))

    def boom(*a, **kw):
        raise ConnectionResetError("peer reset")

    monkeypatch.setattr(p, "_bloxroute_call", boom)
    assert p.fire_anchor_poll(timeout_s=0.2) is None            # still None
    assert seen and seen[-1]["fell_back"] is True
    assert "ConnectionResetError" in seen[-1]["reason"]


def test_health_line_reports_head_fetch_ok_by_default(monkeypatch):
    import pancakebot.chain.rpc_poller as mod
    p = RpcPoller(interval_seconds=300)
    lines = []
    monkeypatch.setattr(mod, "info", lambda *a, **k: lines.append(a))
    p._log_getlogs_health()
    assert "head_fetch=ok" in lines[-1][1]


def test_health_line_carries_the_feasibility_head_fetch_cause(monkeypatch):
    """The INFEAS gate fails OPEN when this fetch dies. Behaviour is
    unchanged, but the cause must not be invisible -- and it rides the
    existing once-per-round line, so no new log line is added."""
    import pancakebot.chain.rpc_poller as mod
    p = RpcPoller(interval_seconds=300)
    lines = []
    monkeypatch.setattr(mod, "info", lambda *a, **k: lines.append(a))
    p._last_head_fetch_error = "ReadTimeoutError: timed out"
    p._log_getlogs_health()
    assert "head_fetch=ReadTimeoutError: timed out" in lines[-1][1]


def test_health_line_surfaces_window_fill_and_warming(monkeypatch):
    """An operator must be able to tell 'still warming after a restart'
    from 'quiet because things are fine'."""
    import pancakebot.chain.rpc_poller as mod
    p = RpcPoller(interval_seconds=300)
    lines = []
    monkeypatch.setattr(mod, "info", lambda *a, **k: lines.append(a))
    p.set_health_extra(window_rounds="12/120(warming)")
    p._log_getlogs_health()
    assert "window_rounds=12/120(warming)" in lines[-1][1]
    p.set_health_extra(window_rounds="120/120")
    p._log_getlogs_health()
    assert "window_rounds=120/120" in lines[-1][1]
    assert "warming" not in lines[-1][1]
