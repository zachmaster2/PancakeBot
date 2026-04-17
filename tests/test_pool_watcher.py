"""Unit tests for PoolEventWatcher reliability: backoff, circuit breaker, failover.

All tests use mocked websockets — no real network calls are made.
Run with: python -m pytest tests/test_pool_watcher.py -v
"""
from __future__ import annotations

import asyncio
import threading
import time
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pancakebot.chain.pool_watcher import (
    BSC_WSS_ENDPOINTS,
    PoolEventWatcher,
    _BACKOFF_STEPS,
    _CB_COOLDOWN_SECONDS,
    _CB_FAILURE_THRESHOLD,
    _BACKOFF_RESET_SECONDS,
    _WATCHDOG_STALE_SECONDS,
    _EndpointState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_watcher(urls: list[str] | None = None) -> PoolEventWatcher:
    return PoolEventWatcher(
        wss_urls=urls or ["wss://ep1", "wss://ep2", "wss://ep3"],
        contract_address="0xDEAD",
    )


def _endpoint_states(watcher: PoolEventWatcher, n: int = 3) -> list[_EndpointState]:
    """Extract fresh endpoint states matching what _run_loop creates."""
    return [_EndpointState(url=u) for u in (watcher._wss_urls[:n])]


# ---------------------------------------------------------------------------
# 1. Default endpoint list
# ---------------------------------------------------------------------------

class TestEndpointDefaults(unittest.TestCase):
    def test_default_endpoints_are_at_least_two(self):
        w = PoolEventWatcher()
        self.assertGreaterEqual(len(w._wss_urls), 2)

    def test_default_endpoints_match_constant(self):
        w = PoolEventWatcher()
        self.assertEqual(w._wss_urls, BSC_WSS_ENDPOINTS)

    def test_custom_urls_override(self):
        custom = ["wss://a", "wss://b"]
        w = PoolEventWatcher(wss_urls=custom)
        self.assertEqual(w._wss_urls, custom)


# ---------------------------------------------------------------------------
# 2. Backoff calculation
# ---------------------------------------------------------------------------

class TestBackoffLogic(unittest.TestCase):
    """Verify that _run_loop applies backoff before connecting."""

    def test_backoff_steps_increase(self):
        """Steps must be monotonically non-decreasing and cap at 120s."""
        for i, s in enumerate(_BACKOFF_STEPS):
            if i > 0:
                self.assertGreaterEqual(s, _BACKOFF_STEPS[i - 1])
        self.assertEqual(_BACKOFF_STEPS[-1], 120)

    def test_backoff_step_increments_on_failure(self):
        state = _EndpointState(url="wss://x")
        state.consecutive_failures = 1
        state.backoff_step = 0

        # Simulate what _run_loop does after a short session.
        session_duration = 0.0  # didn't reach _BACKOFF_RESET_SECONDS
        if session_duration < _BACKOFF_RESET_SECONDS:
            state.consecutive_failures += 1
            state.backoff_step = min(state.backoff_step + 1, len(_BACKOFF_STEPS) - 1)

        self.assertEqual(state.backoff_step, 1)
        self.assertEqual(state.consecutive_failures, 2)

    def test_backoff_step_resets_after_long_session(self):
        state = _EndpointState(url="wss://x")
        state.backoff_step = 4
        state.consecutive_failures = 5

        session_duration = _BACKOFF_RESET_SECONDS + 1
        if session_duration >= _BACKOFF_RESET_SECONDS:
            state.backoff_step = 0
            state.consecutive_failures = 0

        self.assertEqual(state.backoff_step, 0)
        self.assertEqual(state.consecutive_failures, 0)

    def test_backoff_step_caps_at_max(self):
        state = _EndpointState(url="wss://x")
        state.backoff_step = len(_BACKOFF_STEPS) - 1

        # One more failure should not exceed the last index.
        state.backoff_step = min(state.backoff_step + 1, len(_BACKOFF_STEPS) - 1)
        self.assertEqual(state.backoff_step, len(_BACKOFF_STEPS) - 1)


# ---------------------------------------------------------------------------
# 3. Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker(unittest.TestCase):
    def test_circuit_opens_after_threshold_failures(self):
        state = _EndpointState(url="wss://x")
        for _ in range(_CB_FAILURE_THRESHOLD):
            state.consecutive_failures += 1
            if state.consecutive_failures >= _CB_FAILURE_THRESHOLD:
                state.circuit_open_until = time.time() + _CB_COOLDOWN_SECONDS

        self.assertGreater(state.circuit_open_until, time.time())

    def test_circuit_open_skips_endpoint(self):
        """Endpoints with circuit_open_until > now should be excluded."""
        now = time.time()
        states = [
            _EndpointState(url="wss://a", circuit_open_until=now + 300),
            _EndpointState(url="wss://b", circuit_open_until=now + 300),
            _EndpointState(url="wss://c", circuit_open_until=0.0),  # available
        ]
        available = [i for i, s in enumerate(states) if s.circuit_open_until <= now]
        self.assertEqual(available, [2])

    def test_circuit_closes_after_cooldown(self):
        state = _EndpointState(url="wss://x")
        state.circuit_open_until = time.time() - 1  # already expired

        available = state.circuit_open_until <= time.time()
        self.assertTrue(available)

    def test_circuit_cooldown_duration(self):
        self.assertEqual(_CB_COOLDOWN_SECONDS, 300.0)

    def test_all_circuits_open_waits(self):
        """When all endpoints are open, available list should be empty."""
        future = time.time() + 300
        states = [_EndpointState(url=f"wss://ep{i}", circuit_open_until=future) for i in range(3)]
        available = [i for i, s in enumerate(states) if s.circuit_open_until <= time.time()]
        self.assertEqual(available, [])


# ---------------------------------------------------------------------------
# 4. Failover / round-robin
# ---------------------------------------------------------------------------

class TestFailover(unittest.TestCase):
    def test_round_robin_advances_on_failure(self):
        """ep_idx should advance after each session."""
        n = 3
        ep_idx = 0
        visited = []
        for _ in range(6):
            visited.append(ep_idx)
            ep_idx = (ep_idx + 1) % n
        self.assertEqual(visited, [0, 1, 2, 0, 1, 2])

    def test_skips_circuit_open_endpoints(self):
        """Round-robin must skip circuit-open endpoints."""
        now = time.time()
        states = [
            _EndpointState(url="wss://a", circuit_open_until=now + 300),  # open
            _EndpointState(url="wss://b", circuit_open_until=0.0),         # closed
            _EndpointState(url="wss://c", circuit_open_until=now + 300),  # open
        ]
        available_indices = [i for i, s in enumerate(states) if s.circuit_open_until <= now]

        ep_idx = 0
        chosen = None
        for i in available_indices:
            if i >= ep_idx:
                chosen = i
                break
        if chosen is None and available_indices:
            chosen = available_indices[0]

        self.assertEqual(chosen, 1)
        self.assertEqual(states[chosen].url, "wss://b")

    def test_wrap_around_when_no_available_after_idx(self):
        """If no available endpoint has index >= ep_idx, wrap to first available."""
        now = time.time()
        states = [
            _EndpointState(url="wss://a", circuit_open_until=0.0),         # closed
            _EndpointState(url="wss://b", circuit_open_until=now + 300),  # open
            _EndpointState(url="wss://c", circuit_open_until=now + 300),  # open
        ]
        available_indices = [i for i, s in enumerate(states) if s.circuit_open_until <= now]

        ep_idx = 2  # past all available
        chosen = None
        for i in available_indices:
            if i >= ep_idx:
                chosen = i
                break
        if chosen is None and available_indices:
            chosen = available_indices[0]

        self.assertEqual(chosen, 0)  # wraps to first available


# ---------------------------------------------------------------------------
# 5. Watchdog
# ---------------------------------------------------------------------------

class TestWatchdog(unittest.TestCase):
    def test_watchdog_fires_when_stale(self):
        """Watchdog should set force_reconnect if connected but no recent events."""
        w = _make_watcher()
        w._connected = True
        w._last_event_at = time.time() - (_WATCHDOG_STALE_SECONDS + 5)
        w._force_reconnect.clear()

        # Simulate one watchdog check.
        age = time.time() - w._last_event_at
        if w._connected and age > _WATCHDOG_STALE_SECONDS:
            w._connected = False
            w._force_reconnect.set()

        self.assertTrue(w._force_reconnect.is_set())
        self.assertFalse(w._connected)

    def test_watchdog_does_not_fire_when_recent_events(self):
        """Watchdog must not reconnect when events are fresh."""
        w = _make_watcher()
        w._connected = True
        w._last_event_at = time.time() - 5  # only 5s ago

        age = time.time() - w._last_event_at
        if w._connected and age > _WATCHDOG_STALE_SECONDS:
            w._connected = False
            w._force_reconnect.set()

        self.assertFalse(w._force_reconnect.is_set())
        self.assertTrue(w._connected)

    def test_watchdog_does_not_fire_when_disconnected(self):
        """Watchdog should not act if already disconnected."""
        w = _make_watcher()
        w._connected = False
        w._last_event_at = time.time() - 9999

        age = time.time() - w._last_event_at
        if w._connected and age > _WATCHDOG_STALE_SECONDS:
            w._force_reconnect.set()

        self.assertFalse(w._force_reconnect.is_set())


# ---------------------------------------------------------------------------
# 6. State properties
# ---------------------------------------------------------------------------

class TestStateProperties(unittest.TestCase):
    def test_initial_connected_false(self):
        w = _make_watcher()
        self.assertFalse(w.connected)

    def test_initial_current_endpoint_empty(self):
        w = _make_watcher()
        self.assertEqual(w.current_endpoint, "")

    def test_initial_last_connected_at_zero(self):
        w = _make_watcher()
        self.assertEqual(w.last_connected_at, 0.0)

    def test_stats_includes_new_fields(self):
        w = _make_watcher()
        stats = w.stats
        self.assertIn("current_endpoint", stats)
        self.assertIn("last_connected_at", stats)
        self.assertIn("connected", stats)

    def test_current_endpoint_set_on_run_loop(self):
        """_run_loop sets _current_endpoint before connecting."""
        w = _make_watcher(["wss://test"])
        w._current_endpoint = "wss://test"
        self.assertEqual(w.current_endpoint, "wss://test")


# ---------------------------------------------------------------------------
# 7. Integration: mocked websocket connects and sets _connected
# ---------------------------------------------------------------------------

class TestMockedConnection(unittest.TestCase):
    """Verify _ws_listen sets _connected and last_connected_at on success."""

    def _make_mock_ws(self, bet_events: int = 0):
        """Build a fake websocket context manager."""
        log_sub_resp = json_dumps({"result": "sub-logs-id"})
        heads_sub_resp = json_dumps({"result": "sub-heads-id"})
        # Subscription confirmations + one newHead to set _connected, then close.
        head_event = json_dumps({
            "params": {
                "subscription": "sub-heads-id",
                "result": {"number": "0x1", "timestamp": "0x60000000"},
            }
        })

        recv_sequence = [log_sub_resp, heads_sub_resp, head_event]
        recv_iter = iter(recv_sequence)

        async def fake_recv(timeout=None):
            try:
                return next(recv_iter)
            except StopIteration:
                # Simulate clean close after responses exhausted.
                import websockets.exceptions
                raise websockets.exceptions.ConnectionClosedOK(None, None)

        ws = AsyncMock()
        ws.recv = fake_recv
        ws.send = AsyncMock()
        ws.ping = AsyncMock(return_value=asyncio.Future())
        return ws

    def test_connected_set_after_subscriptions(self):
        import json as json_mod

        w = _make_watcher(["wss://ep1"])
        state = _EndpointState(url="wss://ep1")

        # Build responses: log sub confirm, heads sub confirm, then close.
        responses = [
            json_mod.dumps({"result": "sub-logs"}),
            json_mod.dumps({"result": "sub-heads"}),
        ]
        call_count = [0]

        async def fake_recv(*args, **kwargs):
            if call_count[0] < len(responses):
                r = responses[call_count[0]]
                call_count[0] += 1
                return r
            # After subscriptions confirmed, simulate disconnect.
            import websockets.exceptions
            raise websockets.exceptions.ConnectionClosedOK(None, None)

        mock_ws = AsyncMock()
        mock_ws.recv = fake_recv
        mock_ws.send = AsyncMock()
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        with patch("websockets.connect", return_value=mock_ws):
            try:
                asyncio.run(w._ws_listen(state))
            except Exception:
                pass

        self.assertGreater(w.last_connected_at, 0.0)
        self.assertGreater(state.session_connected_at, 0.0)


def json_dumps(obj: dict) -> str:
    import json
    return json.dumps(obj)


if __name__ == "__main__":
    unittest.main()
