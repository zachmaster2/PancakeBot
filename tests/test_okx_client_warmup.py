"""Tests for OkxClient.warmup() session-reset behaviour.

Verifies the per-round connection-affinity-break fix for OKX kline lag.

Run:
    python -m pytest tests/test_okx_client_warmup.py -v
    python tests/test_okx_client_warmup.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.okx_client import OkxClient  # noqa: E402


def test_warmup_default_connections_matches_gate_symbol_count():
    """Default ``connections`` matches the live decision-path gate's
    concurrent-fetch symbol count, so every parallel request finds a
    pre-established TLS connection. With BNB currently disabled in
    ``MomentumGate._SYMBOLS_FETCHED``, the gate fetches 3 symbols
    (BTC/ETH/SOL) and the warmup default is 3."""
    from pancakebot.strategy.momentum_gate import _SYMBOLS_FETCHED
    sig = inspect.signature(OkxClient.warmup)
    default = sig.parameters["connections"].default
    expected = len(_SYMBOLS_FETCHED)
    assert default == expected, (
        f"OkxClient.warmup() default connections must equal the gate's "
        f"_SYMBOLS_FETCHED count ({expected}); got {default}"
    )


def test_warmup_replaces_session_with_fresh_instance():
    """Each warmup() call must give us a brand-new requests.Session."""
    c = OkxClient(timeout_seconds=5.0)
    sess_before = c._session
    # Mock the network calls so we don't actually hit OKX in unit tests.
    with mock.patch.object(sess_before, "get") as _mock_old_get:
        # Patch the new session's .get too (any new session created during
        # warmup) -- intercept at the module level.
        import requests
        original_session = requests.Session
        # Track what new session(s) get instantiated
        instantiated_sessions: list = []

        def tracking_session():
            s = original_session()
            instantiated_sessions.append(s)
            # Patch its .get to be a no-op so we don't hit network
            s.get = mock.MagicMock(return_value=mock.MagicMock(status_code=200))
            return s

        with mock.patch("pancakebot.market_data.okx_client.requests.Session",
                        side_effect=tracking_session):
            c.warmup(connections=2)

    sess_after = c._session
    assert sess_after is not sess_before, (
        "warmup() must replace self._session with a fresh instance "
        "(connection-affinity break)"
    )
    assert sess_after in instantiated_sessions, (
        "the new session must be one of the freshly-instantiated ones"
    )


def test_warmup_closes_old_session():
    """Old session.close() is called as part of warmup."""
    c = OkxClient(timeout_seconds=5.0)
    sess_old = c._session
    with mock.patch.object(sess_old, "close") as mock_close, \
         mock.patch("pancakebot.market_data.okx_client.requests.Session") as mock_session_ctor:
        new_sess = mock.MagicMock()
        new_sess.get.return_value = mock.MagicMock(status_code=200)
        mock_session_ctor.return_value = new_sess
        c.warmup(connections=1)
    mock_close.assert_called_once()


def test_warmup_swallows_close_exception():
    """If old session.close() raises, warmup must continue without crashing."""
    c = OkxClient(timeout_seconds=5.0)
    sess_old = c._session
    with mock.patch.object(sess_old, "close",
                           side_effect=RuntimeError("simulated close failure")), \
         mock.patch("pancakebot.market_data.okx_client.requests.Session") as mock_session_ctor:
        new_sess = mock.MagicMock()
        new_sess.get.return_value = mock.MagicMock(status_code=200)
        mock_session_ctor.return_value = new_sess
        # Must NOT raise.
        c.warmup(connections=1)
    # Confirm we still got the new session despite the close failure.
    assert c._session is new_sess


def test_warmup_handles_request_exception_gracefully():
    """If the warmup GET raises, warmup must not propagate the exception."""
    import requests
    c = OkxClient(timeout_seconds=5.0)
    with mock.patch("pancakebot.market_data.okx_client.requests.Session") as mock_session_ctor:
        new_sess = mock.MagicMock()
        new_sess.get.side_effect = requests.ConnectionError("simulated network failure")
        mock_session_ctor.return_value = new_sess
        # Must NOT raise -- the existing best-effort warmup behaviour
        # must still work after the fix.
        c.warmup(connections=2)


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
