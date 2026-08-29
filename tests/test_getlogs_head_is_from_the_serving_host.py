"""A getLogs range must be bounded by the head of the host that serves it.

THE INCIDENT. The Era 13 split (2026-08-21) routed eth_getLogs to
RPC_GETLOGS_ENDPOINT and left every other read on bloXroute. The poll loop
kept taking its head from bloXroute and clamping with
``min(chunk_start + CHUNK - 1, head)``. That clamp is correct -- against
bloXroute. Served by another host it can overshoot by a block whenever
ordinary propagation leaves the getLogs host behind:

    -32602 "block range extends beyond current head block:
            requested 118612457, head 118612456"

Twice in five days (2026-08-26 periodic, 2026-08-28 single poll). The
second cost a round: the chunk raised, the chunk loop broke with the
cursor UN-ADVANCED, and the next round skipped on POOL UNCOVERED.

WHY IT WAS INVISIBLE. Every existing poller test stubs the head with a
single scalar, so both hosts implicitly agreed and the bug could not be
expressed. These tests give the two hosts DIFFERENT heads -- which is the
whole content of the defect -- and then assert on what the poller
actually did, not on what it should have computed.

ARTIFACTS, NOT RECOMPUTATIONS (the standard set on 2026-08-28 after a
reconstruction-based check validated a broken breaker for four days):
  * the exact toBlock the transport was ASKED for, captured at the
    _rpc_post seam; and
  * the cursor the poller actually LEFT BEHIND.
Both are produced by the code under test. Neither is re-derived here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain import rpc_poller as rp  # noqa: E402
from pancakebot.chain.rpc_poller import (  # noqa: E402
    RPC_BLOXROUTE_ENDPOINT,
    RPC_GETLOGS_ENDPOINT,
    RpcPoller,
)

# The getLogs host is ONE BLOCK BEHIND -- the exact incident condition.
BLOXROUTE_HEAD = 118_612_457
GETLOGS_HEAD = 118_612_456
CURSOR = 118_612_449


class _Chain:
    """Two hosts at different heights, wired at the transport seam.

    eth_getLogs rejects a toBlock beyond GETLOGS_HEAD exactly as BSC does,
    so a cross-node bound fails here for the same reason it failed live.
    """

    def __init__(self) -> None:
        self.getlogs_ranges: list[tuple[int, int]] = []
        self.head_calls: list[str] = []

    def post(self, url, body, *, timeout_seconds):
        import json
        req = json.loads(body)
        method = req["method"]
        if method == "eth_blockNumber":
            self.head_calls.append(url)
            head = (GETLOGS_HEAD if url == RPC_GETLOGS_ENDPOINT
                    else BLOXROUTE_HEAD)
            return json.dumps({"result": hex(head)}).encode()
        if method == "eth_getLogs":
            flt = req["params"][0]
            lo, hi = int(flt["fromBlock"], 16), int(flt["toBlock"], 16)
            self.getlogs_ranges.append((lo, hi))
            if hi > GETLOGS_HEAD:
                return json.dumps({"error": {
                    "code": -32602,
                    "message": ("block range extends beyond current head "
                                f"block: requested {hi}, head {GETLOGS_HEAD}"),
                }}).encode()
            return json.dumps({"result": []}).encode()
        raise AssertionError(f"unexpected method {method}")


def _poller(monkeypatch) -> tuple[RpcPoller, _Chain]:
    p = RpcPoller(interval_seconds=300)
    chain = _Chain()
    monkeypatch.setattr(p, "_rpc_post", chain.post)
    p._last_polled_block_number = CURSOR
    p._connected = True
    return p, chain


# ---- the head is sourced from the serving host ---------------------------

def test_the_poll_head_comes_from_the_getlogs_host(monkeypatch):
    """THE regression, at its shortest."""
    p, chain = _poller(monkeypatch)
    assert p._poll_head_block_number(attempts=1) == GETLOGS_HEAD
    assert chain.head_calls == [RPC_GETLOGS_ENDPOINT], (
        "the poll head was read from the wrong host; a head is node-relative "
        f"and must come from whoever serves the range (got {chain.head_calls})")


def test_the_two_hosts_really_do_disagree_in_this_fixture():
    """Guard the guard: if both heads were equal the tests above would pass
    vacuously and prove nothing, which is exactly how the live bug hid
    inside a suite that stubbed one scalar head."""
    assert BLOXROUTE_HEAD != GETLOGS_HEAD
    assert BLOXROUTE_HEAD == GETLOGS_HEAD + 1


def test_the_bloxroute_head_is_still_reachable_explicitly(monkeypatch):
    """The fix removes an implicit default, not the capability. Chain-global
    reads (block timestamps) legitimately stay on the fast host."""
    p, chain = _poller(monkeypatch)
    got = p._head_block_number(endpoint=RPC_BLOXROUTE_ENDPOINT, attempts=1)
    assert got == BLOXROUTE_HEAD
    assert chain.head_calls == [RPC_BLOXROUTE_ENDPOINT]


# ---- ARTIFACT 1: the range actually requested ----------------------------

def test_no_range_is_ever_requested_beyond_the_serving_host_head(monkeypatch):
    """Asserted on the toBlock the transport was ASKED for -- captured at
    the seam, not recomputed from the inputs."""
    p, chain = _poller(monkeypatch)
    p._poll_now(deadline_ms=0, label="single")

    assert chain.getlogs_ranges, "no getLogs range was issued at all"
    worst = max(hi for _, hi in chain.getlogs_ranges)
    assert worst <= GETLOGS_HEAD, (
        f"requested toBlock {worst} exceeds the serving host head "
        f"{GETLOGS_HEAD} — this is the -32602 head-race, reintroduced")


def test_the_fixture_would_have_caught_the_old_bound(monkeypatch):
    """Proof the fixture has teeth: bound the range by the OTHER host, as
    the pre-fix code did, and the same chain rejects it."""
    p, chain = _poller(monkeypatch)
    bad_head = p._head_block_number(
        endpoint=RPC_BLOXROUTE_ENDPOINT, attempts=1)
    with pytest.raises(Exception) as exc:
        p._fetch_and_process_logs(CURSOR + 1, bad_head, attempts=1)
    assert "-32602" in str(exc.value) or "32602" in str(exc.value)


# ---- ARTIFACT 2: the cursor the poller left behind -----------------------

def test_the_cursor_advances_rather_than_stalling(monkeypatch):
    """The second-order damage was not the error, it was the `break` that
    left the cursor un-advanced, starving pool coverage and skipping the
    NEXT round on POOL UNCOVERED. Assert the cursor the poller actually
    left, which is engine-written state."""
    p, chain = _poller(monkeypatch)
    p._poll_now(deadline_ms=0, label="single")
    assert p._last_polled_block_number == GETLOGS_HEAD, (
        f"cursor left at {p._last_polled_block_number}, expected "
        f"{GETLOGS_HEAD}; an un-advanced cursor is what cost round 510959")


def test_the_cursor_never_moves_backwards(monkeypatch):
    """Forward-only. A getLogs host that is momentarily BEHIND the cursor
    must produce a no-op poll, never a rewind -- otherwise blocks already
    counted into the pool would be re-counted."""
    p, chain = _poller(monkeypatch)
    p._last_polled_block_number = GETLOGS_HEAD + 5   # cursor ahead of host
    p._poll_now(deadline_ms=0, label="single")
    assert p._last_polled_block_number == GETLOGS_HEAD + 5
    assert chain.getlogs_ranges == [], (
        "a host behind the cursor must issue no range at all")


def test_a_lagging_host_cannot_stall_the_cursor_permanently(monkeypatch):
    """Item 3. The tighter bound slows the cursor only while the host lags;
    once it catches up the cursor follows. A permanent stall would need the
    getLogs host to stop advancing entirely, which is an endpoint outage --
    the condition ENDPOINT_MOVE_TRIGGER exists to detect, not something the
    bound should paper over."""
    global GETLOGS_HEAD
    p, chain = _poller(monkeypatch)
    p._poll_now(deadline_ms=0, label="single")
    first = p._last_polled_block_number
    original = GETLOGS_HEAD
    try:
        GETLOGS_HEAD = original + 10          # host catches up
        p._poll_now(deadline_ms=0, label="single")
    finally:
        GETLOGS_HEAD = original
    assert p._last_polled_block_number > first, "cursor failed to resume"
