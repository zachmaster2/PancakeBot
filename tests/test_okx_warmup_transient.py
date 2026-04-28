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

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.okx_client import OkxClient, RETRY_GATE  # noqa: E402
from pancakebot.util import InvariantError, TransientOkxError  # noqa: E402


def _make_client_with_session_that_raises(exc: BaseException) -> OkxClient:
    """Build a client whose underlying session.get always raises *exc*.

    Sets ``_session`` directly. Note: ``warmup()`` replaces ``_session``
    on each call (per-round connection-affinity break, design doc
    research/okx_kline_freshness_fix_design.md), so callers exercising
    warmup() should use ``_patch_session_factory`` instead -- this
    helper is only useful for tests that exercise ``kline_fetch_window``
    (which does NOT replace the session).
    """
    c = OkxClient(timeout_seconds=5.0)
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=exc)
    c._session = mock_session
    return c


@contextlib.contextmanager
def _patch_session_factory(get_side_effect):
    """Patch ``requests.Session`` in the okx_client module so any session
    created (including by warmup()) has ``.get`` driven by *get_side_effect*
    and a ``.close()`` mock that doesn't error.

    Yields the MagicMock used as the new session so tests can inspect
    ``mock_session.get.call_count`` etc.
    """
    new_sess = MagicMock()
    new_sess.get = MagicMock(side_effect=get_side_effect)
    new_sess.close = MagicMock()
    with patch("pancakebot.market_data.okx_client.requests.Session",
               return_value=new_sess):
        yield new_sess


def test_warmup_swallows_connection_error():
    """requests.ConnectionError from every pool request should NOT propagate."""
    exc = requests.exceptions.ConnectionError(
        "Connection aborted.",
        ConnectionResetError(10054, "An existing connection was forcibly closed"),
    )
    c = OkxClient(timeout_seconds=5.0)
    with _patch_session_factory(get_side_effect=exc):
        # Before fix: this raised. After fix: returns silently (3 WARN logs emitted).
        c.warmup(connections=3)


def test_warmup_swallows_timeout():
    """requests.Timeout should also be swallowed (same class of transient error)."""
    c = OkxClient(timeout_seconds=5.0)
    with _patch_session_factory(get_side_effect=requests.exceptions.Timeout("read timeout")):
        c.warmup(connections=2)


def test_warmup_swallows_raw_connection_reset():
    """Raw ConnectionResetError (Python 3.13 sometimes surfaces this directly)."""
    c = OkxClient(timeout_seconds=5.0)
    with _patch_session_factory(get_side_effect=ConnectionResetError(10054, "reset")):
        c.warmup(connections=2)


def test_warmup_swallows_raw_oserror():
    """Generic OSError (socket-level) also swallowed."""
    c = OkxClient(timeout_seconds=5.0)
    with _patch_session_factory(get_side_effect=OSError("network unreachable")):
        c.warmup(connections=2)


def test_warmup_does_not_mask_non_network_errors():
    """A genuine bug (e.g. TypeError) MUST still propagate -- fix must not
    turn into a blanket ``except Exception`` swallow."""
    c = OkxClient(timeout_seconds=5.0)
    raised = False
    with _patch_session_factory(get_side_effect=TypeError("something is wrong")):
        try:
            c.warmup(connections=1)
        except TypeError:
            raised = True
    assert raised, "warmup swallowed a non-network exception; fix is too broad"


def test_warmup_does_not_swallow_keyboard_interrupt():
    """KeyboardInterrupt (subclass of BaseException, NOT Exception) must
    still propagate. Regression guard against an over-broad future rewrite
    that uses ``except BaseException`` by mistake."""
    c = OkxClient(timeout_seconds=5.0)
    raised = False
    with _patch_session_factory(get_side_effect=KeyboardInterrupt("ctrl-c")):
        try:
            c.warmup(connections=1)
        except KeyboardInterrupt:
            raised = True
    assert raised, "warmup swallowed KeyboardInterrupt; that breaks clean shutdown"


def test_warmup_partial_failure_succeeds():
    """Only SOME pool connections fail. Warmup should log those and return
    normally -- the successful requests still warmed the pool."""
    c = OkxClient(timeout_seconds=5.0)
    # 3 submissions: two succeed (return a mock response), one raises.
    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    side_effect = [
        mock_response_ok,
        requests.exceptions.ConnectionError("socket reset"),
        mock_response_ok,
    ]
    with _patch_session_factory(get_side_effect=side_effect) as mock_sess:
        # Must not raise even though one of the three futures errored.
        c.warmup(connections=3)
        # Confirm all 3 submissions were attempted (future consumed).
        assert mock_sess.get.call_count == 3, (
            f"expected 3 get calls (one per connection), got {mock_sess.get.call_count}"
        )


def test_kline_fetch_window_wraps_connection_error_into_transient():
    """``kline_fetch_window`` (the canonical primitive) classifies
    ConnectionError as RETRYABLE and exhausts the policy → TransientOkxError.
    The error type matters: callers on the live decision path catch
    TransientOkxError to skip the round; InvariantError would crash the
    bot (reserved for shape-violation conditions)."""
    c = _make_client_with_session_that_raises(
        requests.exceptions.ConnectionError("Connection aborted.")
    )
    # Use RETRY_GATE (2 attempts) so the test runs in <3s instead of
    # RETRY_SYNC's 30+ seconds of cumulative backoff.
    raised = False
    try:
        c.kline_fetch_window(
            symbol="BTC-USDT",
            oldest_open_ms=1_000_000_000_000,
            newest_open_ms_inclusive=1_000_000_001_000,
            retry_policy=RETRY_GATE,
        )
    except TransientOkxError:
        raised = True
    assert raised, (
        "kline_fetch_window should raise TransientOkxError after retry exhaustion"
    )


# ---------------------------------------------------------------------------
# kline_fetch_window contiguity validation (Item A, 2026-04-27)
# ---------------------------------------------------------------------------

def _mock_okx_response(rows_newest_first):
    """Build a MagicMock requests.Response with OKX /history-candles shape.
    *rows_newest_first* is an OKX-format row list (newest first), each row
    is the 9-field OKX wire format ``[ts, o, h, l, c, vol, volCcy,
    volCcyQuote, confirm]``.
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.json = MagicMock(return_value={"code": "0", "data": rows_newest_first})
    return resp


def _okx_row(ts_ms, close=100.0):
    """One OKX wire-format row at ts_ms (string-typed per OKX convention)."""
    s = str(close)
    return [str(ts_ms), s, s, s, s, "1.0", "100", "100", "1"]


def _client_with_canned_response(resp):
    """OkxClient whose session.get returns a fixed response."""
    c = OkxClient(timeout_seconds=5.0)
    c._session = MagicMock()
    c._session.get = MagicMock(return_value=resp)
    return c


def test_kline_fetch_window_accepts_contiguous_sequence():
    """Happy path: 3 strictly-contiguous candles pass.

    Return shape is ``(rows, rtt_ms)`` since the 2026-04-27 timing fix.
    The mock response returns instantly so rtt_ms should be ~0 ms.
    """
    # OKX returns newest-first; we want internal oldest-first
    # [1000_000, 1001_000, 1002_000].
    rows = [_okx_row(1002_000), _okx_row(1001_000), _okx_row(1000_000)]
    resp = _mock_okx_response(rows)
    c = _client_with_canned_response(resp)
    arrays, rtt_ms = c.kline_fetch_window(
        symbol="BTC-USDT",
        oldest_open_ms=1000_000,
        newest_open_ms_inclusive=1002_000,
        retry_policy=RETRY_GATE,
    )
    assert [a[0] for a in arrays] == [1000_000, 1001_000, 1002_000]
    # Mocked .get() returns instantly; rtt_ms should be a small non-negative int.
    assert isinstance(rtt_ms, int)
    assert rtt_ms >= 0
    assert rtt_ms < 100, f"mocked .get() should be near-instant, got {rtt_ms}ms"


def test_kline_fetch_window_rejects_gap_in_middle():
    """3 rows, correct length, correct first/last, but a 2s gap between
    rows 1 and 2 (1000_000 -> 1002_000 with 1001_000 missing). The
    boundary check passes; only the contiguity check catches it."""
    # Synthetic shape: 1000_000, 1002_000, 1003_000. First=1000, last=1003,
    # length=3 — boundary check happy. But idx=1 has delta=2000.
    # Need to pass expected_count for the request. We tell OKX we want
    # [1000..1003] (4 candles), but OKX returns 3 rows. We need len match.
    # Use [1000..1002] as request (3 candles), have rows[1] = 1002 (skip 1001).
    # First check: oldest=1000 ✓, newest=1002 ✓. Then contiguity fires.
    rows_newest_first = [
        _okx_row(1002_000),  # newest
        _okx_row(1002_000),  # NOT — these aren't gappy enough
        _okx_row(1000_000),  # oldest
    ]
    # Actually, to exercise the gap check, we need rows where adjacent
    # internal-order timestamps differ by != 1000ms. With the boundary
    # check forcing first=oldest_open_ms and last=newest_open_ms_inclusive,
    # any non-contiguity has to be in the MIDDLE.
    # Set up: request [1000, 1003] -> expected_count=4. OKX returns 4 rows
    # but the internal #2 is 1002 (skipping 1001) -- duplicate of #3 maybe.
    # Cleaner: request [1000, 1002] -> expected_count=3, OKX returns
    # [1002, 1500, 1000] (oldest-first: [1000, 1500, 1002]).
    # That breaks contiguity AND order. Boundary fails (last=1002 != 1500).
    # So I need the boundary to match. Let me use an explicit middle gap:
    # request [1000, 1003] -> expected_count=4. OKX returns 4 rows:
    # newest-first [1003, 1002, 1000, 999] -> oldest-first [999, 1000,
    # 1002, 1003]. Boundary: first=999 != 1000 -> boundary fails first.
    #
    # The only way to trip JUST contiguity is: first matches oldest, last
    # matches newest, length matches expected_count, but internal delta
    # isn't 1000. That requires a row at oldest, a row at newest, plus
    # internal rows whose timestamps don't fit the 1000ms grid.
    # Example: request [1000, 1003] (count=4). OKX returns
    # newest-first [1003, 1002, 1002, 1000] -> oldest-first
    # [1000, 1002, 1002, 1003]. Length=4 ✓, first=1000 ✓, last=1003 ✓.
    # Contiguity: idx=1 delta=1002-1000=2 (in seconds; 2000ms != 1000) -> fails.
    rows_newest_first = [
        _okx_row(1003_000),
        _okx_row(1002_000),
        _okx_row(1002_000),  # duplicate to keep length at 4
        _okx_row(1000_000),
    ]
    resp = _mock_okx_response(rows_newest_first)
    c = _client_with_canned_response(resp)
    raised = False
    try:
        c.kline_fetch_window(
            symbol="BTC-USDT",
            oldest_open_ms=1000_000,
            newest_open_ms_inclusive=1003_000,
            retry_policy=RETRY_GATE,
        )
    except InvariantError as e:
        assert "kline_fetch_integrity_violation" in str(e), str(e)
        assert "noncontiguous" in str(e), str(e)
        raised = True
    assert raised, "expected InvariantError on mid-window non-contiguity"


def test_kline_fetch_window_rejects_duplicate_timestamp():
    """Two adjacent rows at the same timestamp (duplicate) → contiguity
    delta == 0, fails."""
    # request [1000, 1003] count=4. Returned oldest-first: [1000, 1001, 1001, 1003]
    rows_newest_first = [
        _okx_row(1003_000),
        _okx_row(1001_000),
        _okx_row(1001_000),  # duplicate
        _okx_row(1000_000),
    ]
    resp = _mock_okx_response(rows_newest_first)
    c = _client_with_canned_response(resp)
    raised = False
    try:
        c.kline_fetch_window(
            symbol="BTC-USDT",
            oldest_open_ms=1000_000,
            newest_open_ms_inclusive=1003_000,
            retry_policy=RETRY_GATE,
        )
    except InvariantError as e:
        assert "kline_fetch_integrity_violation" in str(e), str(e)
        assert "noncontiguous" in str(e), str(e)
        raised = True
    assert raised, "expected InvariantError on duplicate timestamp"


def test_kline_fetch_window_rejects_out_of_order_rows():
    """Internal rows out of strictly-increasing order → contiguity delta
    is negative, fails."""
    # request [1000, 1003] count=4. Returned oldest-first: [1000, 1002, 1001, 1003]
    rows_newest_first = [
        _okx_row(1003_000),
        _okx_row(1001_000),  # internal #3 oldest-first = 1001
        _okx_row(1002_000),  # internal #2 oldest-first = 1002 (out of order!)
        _okx_row(1000_000),
    ]
    resp = _mock_okx_response(rows_newest_first)
    c = _client_with_canned_response(resp)
    raised = False
    try:
        c.kline_fetch_window(
            symbol="BTC-USDT",
            oldest_open_ms=1000_000,
            newest_open_ms_inclusive=1003_000,
            retry_policy=RETRY_GATE,
        )
    except InvariantError as e:
        assert "kline_fetch_integrity_violation" in str(e), str(e)
        assert "noncontiguous" in str(e), str(e)
        raised = True
    assert raised, "expected InvariantError on out-of-order rows"


def main() -> int:
    tests = [
        test_warmup_swallows_connection_error,
        test_warmup_swallows_timeout,
        test_warmup_swallows_raw_connection_reset,
        test_warmup_swallows_raw_oserror,
        test_warmup_does_not_mask_non_network_errors,
        test_warmup_does_not_swallow_keyboard_interrupt,
        test_warmup_partial_failure_succeeds,
        test_kline_fetch_window_wraps_connection_error_into_transient,
        test_kline_fetch_window_accepts_contiguous_sequence,
        test_kline_fetch_window_rejects_gap_in_middle,
        test_kline_fetch_window_rejects_duplicate_timestamp,
        test_kline_fetch_window_rejects_out_of_order_rows,
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
