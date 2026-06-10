"""Transport tests for the single-source bloXroute read path (Era 12b).

Covers ``_bloxroute_call`` — THE transport every read RPC goes through:
- tight per-attempt timeout passed to ``_rpc_post``
- bounded retries with backoff (momentary-blip recovery)
- raises the LAST error on exhaustion
- JSON-RPC error envelopes raise (and are retried like transport errors)
- wall-clock cap: an attempt whose timeout cannot complete before
  ``abort_at`` is NOT started (the cap is a hard bound by construction)
- ``_bloxroute_block_number`` parsing on top of it

All tests stub ``_rpc_post`` — no network.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain.rpc_poller import (  # noqa: E402
    RpcPoller,
    RPC_BLOXROUTE_ENDPOINT,
)
from pancakebot.util import InvariantError  # noqa: E402


def _make_poller() -> RpcPoller:
    return RpcPoller(interval_seconds=300)


def test_endpoint_constant_is_bloxroute():
    assert RPC_BLOXROUTE_ENDPOINT == "https://bsc.rpc.blxrbdn.com"


def test_call_posts_to_bloxroute_with_timeout(monkeypatch):
    """The call POSTs the JSON-RPC envelope to RPC_BLOXROUTE_ENDPOINT with
    the per-attempt timeout (ms -> seconds) and returns the result field."""
    import json as _json
    p = _make_poller()
    captured: dict = {}

    def _capture(url, body, *, timeout_seconds):
        captured["url"] = url
        captured["req"] = _json.loads(body)
        captured["timeout_seconds"] = timeout_seconds
        return b'{"jsonrpc":"2.0","id":1,"result":"0x64"}'

    monkeypatch.setattr(p, "_rpc_post", _capture)
    out = p._bloxroute_call("eth_blockNumber", [], timeout_ms=250, attempts=1)
    assert out == "0x64"
    assert captured["url"] == RPC_BLOXROUTE_ENDPOINT
    assert captured["req"]["method"] == "eth_blockNumber"
    assert captured["timeout_seconds"] == pytest.approx(0.250)


def test_call_retries_transient_failure_then_succeeds(monkeypatch):
    p = _make_poller()
    calls = {"n": 0}

    def _flaky(url, body, *, timeout_seconds):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("blip")
        return b'{"jsonrpc":"2.0","id":1,"result":"0x1"}'

    monkeypatch.setattr(p, "_rpc_post", _flaky)
    out = p._bloxroute_call("eth_blockNumber", [], timeout_ms=100, attempts=3)
    assert out == "0x1"
    assert calls["n"] == 3


def test_call_raises_last_error_on_exhaustion(monkeypatch):
    p = _make_poller()
    calls = {"n": 0}

    def _always_fail(url, body, *, timeout_seconds):
        calls["n"] += 1
        raise TimeoutError(f"fail_{calls['n']}")

    monkeypatch.setattr(p, "_rpc_post", _always_fail)
    with pytest.raises(TimeoutError, match="fail_2"):
        p._bloxroute_call("eth_blockNumber", [], timeout_ms=100, attempts=2)
    assert calls["n"] == 2


def test_call_rpc_error_envelope_raises_and_is_retried(monkeypatch):
    """A 200-with-error-envelope counts as a failed attempt (retried), and
    the raised error carries the method + envelope for the operator log."""
    p = _make_poller()
    calls = {"n": 0}

    def _err(url, body, *, timeout_seconds):
        calls["n"] += 1
        return b'{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"x"}}'

    monkeypatch.setattr(p, "_rpc_post", _err)
    with pytest.raises(InvariantError, match="rpc_error:eth_getLogs"):
        p._bloxroute_call("eth_getLogs", [{}], timeout_ms=100, attempts=2)
    assert calls["n"] == 2


def test_call_wall_cap_blocks_attempt_that_cannot_finish(monkeypatch):
    """An attempt is not even STARTED if its timeout would overrun abort_at —
    this is what makes the poll wall cap a hard bound, not advisory."""
    p = _make_poller()
    monkeypatch.setattr(
        p, "_rpc_post",
        lambda url, body, *, timeout_seconds: pytest.fail(
            "attempt must not start past the wall cap"
        ),
    )
    with pytest.raises(InvariantError, match="poll_wall_cap_exceeded:eth_blockNumber"):
        p._bloxroute_call(
            "eth_blockNumber", [], timeout_ms=250, attempts=2,
            abort_at=time.monotonic() + 0.050,  # 50ms left < 250ms timeout
        )


def test_call_wall_cap_allows_attempt_that_fits(monkeypatch):
    p = _make_poller()
    monkeypatch.setattr(
        p, "_rpc_post",
        lambda url, body, *, timeout_seconds: b'{"jsonrpc":"2.0","id":1,"result":"0x1"}',
    )
    out = p._bloxroute_call(
        "eth_blockNumber", [], timeout_ms=100, attempts=1,
        abort_at=time.monotonic() + 60.0,
    )
    assert out == "0x1"


def test_block_number_parses_and_propagates_attempts(monkeypatch):
    p = _make_poller()
    captured: dict = {}

    def _capture_call(method, params, *, timeout_ms, attempts, abort_at=None):
        captured["method"] = method
        captured["attempts"] = attempts
        return hex(103_000_000)

    monkeypatch.setattr(p, "_bloxroute_call", _capture_call)
    assert p._bloxroute_block_number(attempts=3) == 103_000_000
    assert captured["method"] == "eth_blockNumber"
    assert captured["attempts"] == 3


def test_block_number_raises_on_unexpected_result(monkeypatch):
    p = _make_poller()
    monkeypatch.setattr(
        p, "_bloxroute_call",
        lambda method, params, **kw: {"not": "a_string"},
    )
    with pytest.raises(InvariantError, match="eth_blockNumber_unexpected_result"):
        p._bloxroute_block_number(attempts=1)
