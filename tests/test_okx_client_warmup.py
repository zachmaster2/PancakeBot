"""Tests for OkxClient.warmup() session-reset behaviour.

Verifies the per-round connection-affinity-break fix for OKX kline lag
(see research/okx_kline_freshness_fix_design.md).

Run:
    python -m pytest tests/test_okx_client_warmup.py -v
    python tests/test_okx_client_warmup.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.okx_client import OkxClient  # noqa: E402


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


def test_warmup_preserves_response_headers_dict():
    """The OkxClient._last_response_headers dict (per-symbol cache for
    diagnostic header capture) must survive the session swap.

    It's an OkxClient attribute, not on Session, so this is a sanity
    check that the fix doesn't inadvertently clear it.
    """
    c = OkxClient(timeout_seconds=5.0)
    c._last_response_headers["BTC-USDT"] = {"cf-cache-status": "DYNAMIC", "age": "0"}

    with mock.patch("pancakebot.market_data.okx_client.requests.Session") as mock_session_ctor:
        new_sess = mock.MagicMock()
        new_sess.get.return_value = mock.MagicMock(status_code=200)
        mock_session_ctor.return_value = new_sess
        c.warmup(connections=1)

    # Headers from prior rounds should still be readable post-warmup.
    assert c._last_response_headers["BTC-USDT"] == {
        "cf-cache-status": "DYNAMIC", "age": "0",
    }


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


def test_fetch_1s_klines_uses_fresh_session_per_call():
    """fetch_1s_klines must use a brand-new Session per call (not self._session).

    This is the within-round affinity break (added 2026-04-25 after n=17
    measurement showed warmup-only reset wasn't enough). Each kline fetch
    gets its own TCP/TLS, immune to any pool reuse from warmup or other
    fetches.
    """
    c = OkxClient(timeout_seconds=5.0)
    # Setup: existing self._session must NOT be used.
    c._session = mock.MagicMock(name="should_not_be_used")
    c._session.get = mock.MagicMock(side_effect=AssertionError(
        "fetch_1s_klines used self._session instead of a fresh one"
    ))

    # Track Session() instantiations.
    instantiated: list = []
    real_session_ctor = __import__("requests").Session

    def tracking_ctor():
        s = mock.MagicMock(name="fresh_session")
        # Mock a healthy response with 1 candle. body.code="0" required
        # by _classify_response for SUCCESS path.
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "code": "0",
            "msg": "",
            "data": [["1777000000000", "100.0", "100.5", "99.5", "100.2", "1.0"]],
        }
        s.get = mock.MagicMock(return_value=mock_resp)
        s.close = mock.MagicMock()
        instantiated.append(s)
        return s

    with mock.patch("pancakebot.market_data.okx_client.requests.Session",
                    side_effect=tracking_ctor):
        result = c.fetch_1s_klines(symbol="BTC-USDT", count=1)

    assert len(instantiated) == 1, (
        f"fetch_1s_klines must create exactly one fresh Session per call, "
        f"got {len(instantiated)}"
    )
    fresh = instantiated[0]
    fresh.get.assert_called_once()
    fresh.close.assert_called_once(), "fresh session must be closed after fetch"
    # Result should be parsed correctly.
    assert len(result) == 1
    assert result[0]["close_price"] == 100.2


def test_fetch_1s_klines_closes_session_even_on_exception():
    """If the fetch GET raises, the fresh session must STILL be closed."""
    import requests
    c = OkxClient(timeout_seconds=5.0)
    fresh = mock.MagicMock()
    fresh.get = mock.MagicMock(
        side_effect=requests.ConnectionError("simulated net failure")
    )
    fresh.close = mock.MagicMock()
    with mock.patch("pancakebot.market_data.okx_client.requests.Session",
                    return_value=fresh):
        # ConnectionError is classified as RETRYABLE -> raises OkxTransientError
        from pancakebot.market_data.okx_client import OkxTransientError
        try:
            c.fetch_1s_klines(symbol="BTC-USDT", count=1)
        except OkxTransientError:
            pass
    fresh.close.assert_called_once(), "session must be closed even when fetch raises"


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
