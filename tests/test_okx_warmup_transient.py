"""Regression test for the OKX warmup ConnectionError fix.

Before this fix, ``OkxClient.warmup()`` re-raised ``requests.exceptions.
ConnectionError`` (wrapping a ``ConnectionResetError`` from a socket reset),
which propagated up through ``momentum_gate.warmup_session`` -> engine ->
run.py and killed the dry bot. It happened twice in 24h
(2026-04-22 22:58 EDT, 2026-04-22 23:59 EDT).

The fix: ``warmup()`` is now best-effort. It logs WARN and swallows
``requests.RequestException`` + raw ``ConnectionResetError``/``OSError``;
non-network exceptions still propagate so real bugs aren't masked.

Run:
    python -m pytest tests/test_okx_warmup_transient.py -v
    # or standalone:
    python tests/test_okx_warmup_transient.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.okx_client import OkxClient, OkxTransientError  # noqa: E402


def _make_client_with_session_that_raises(exc: BaseException) -> OkxClient:
    """Build a client whose underlying session.get always raises *exc*."""
    c = OkxClient(timeout_seconds=5.0)
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=exc)
    c._session = mock_session
    return c


def test_warmup_swallows_connection_error():
    """requests.ConnectionError from every pool request should NOT propagate."""
    exc = requests.exceptions.ConnectionError(
        "Connection aborted.",
        ConnectionResetError(10054, "An existing connection was forcibly closed"),
    )
    c = _make_client_with_session_that_raises(exc)
    # Before fix: this raised. After fix: returns silently (3 WARN logs emitted).
    c.warmup(connections=3)


def test_warmup_swallows_timeout():
    """requests.Timeout should also be swallowed (same class of transient error)."""
    c = _make_client_with_session_that_raises(requests.exceptions.Timeout("read timeout"))
    c.warmup(connections=2)


def test_warmup_swallows_raw_connection_reset():
    """Raw ConnectionResetError (Python 3.13 sometimes surfaces this directly)."""
    c = _make_client_with_session_that_raises(ConnectionResetError(10054, "reset"))
    c.warmup(connections=2)


def test_warmup_swallows_raw_oserror():
    """Generic OSError (socket-level) also swallowed."""
    c = _make_client_with_session_that_raises(OSError("network unreachable"))
    c.warmup(connections=2)


def test_warmup_does_not_mask_non_network_errors():
    """A genuine bug (e.g. TypeError) MUST still propagate -- fix must not
    turn into a blanket ``except Exception`` swallow."""
    c = _make_client_with_session_that_raises(TypeError("something is wrong"))
    raised = False
    try:
        c.warmup(connections=1)
    except TypeError:
        raised = True
    assert raised, "warmup swallowed a non-network exception; fix is too broad"


def test_warmup_does_not_swallow_keyboard_interrupt():
    """KeyboardInterrupt (subclass of BaseException, NOT Exception) must
    still propagate. Regression guard against an over-broad future rewrite
    that uses ``except BaseException`` by mistake."""
    c = _make_client_with_session_that_raises(KeyboardInterrupt("ctrl-c"))
    raised = False
    try:
        c.warmup(connections=1)
    except KeyboardInterrupt:
        raised = True
    assert raised, "warmup swallowed KeyboardInterrupt; that breaks clean shutdown"


def test_warmup_partial_failure_succeeds():
    """Only SOME pool connections fail. Warmup should log those and return
    normally -- the successful requests still warmed the pool."""
    c = OkxClient(timeout_seconds=5.0)
    mock_session = MagicMock()
    # 3 submissions: two succeed (return a mock response), one raises.
    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_session.get = MagicMock(side_effect=[
        mock_response_ok,
        requests.exceptions.ConnectionError("socket reset"),
        mock_response_ok,
    ])
    c._session = mock_session
    # Must not raise even though one of the three futures errored.
    c.warmup(connections=3)
    # Confirm all 3 submissions were attempted (future consumed).
    assert mock_session.get.call_count == 3, (
        f"expected 3 get calls (one per connection), got {mock_session.get.call_count}"
    )


def test_fetch_1s_klines_still_wraps_connection_error():
    """Ensure the pre-existing ``fetch_1s_klines`` path still converts
    ConnectionError -> OkxTransientError. (This is pre-fix behaviour; we
    verify the fix didn't regress it.)"""
    c = _make_client_with_session_that_raises(
        requests.exceptions.ConnectionError("Connection aborted.")
    )
    raised = False
    try:
        c.fetch_1s_klines(symbol="BTC-USDT", count=25)
    except OkxTransientError:
        raised = True
    assert raised, "fetch_1s_klines should wrap ConnectionError as OkxTransientError"


def main() -> int:
    tests = [
        test_warmup_swallows_connection_error,
        test_warmup_swallows_timeout,
        test_warmup_swallows_raw_connection_reset,
        test_warmup_swallows_raw_oserror,
        test_warmup_does_not_mask_non_network_errors,
        test_warmup_does_not_swallow_keyboard_interrupt,
        test_warmup_partial_failure_succeeds,
        test_fetch_1s_klines_still_wraps_connection_error,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  [OK] {t.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failures.append(t.__name__)
    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
