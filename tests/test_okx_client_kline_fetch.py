"""Tests for ``OkxClient.kline_fetch_window`` query construction and error
classification.

Covers the 2026-04-27 fix that adds an opt-in ``send_before_bound``
parameter to the canonical kline-fetch primitive. The motivation: the
live-decision gate's per-round REST fetch crashed ~67% of rounds with
``kline_fetch_integrity_violation`` because OKX's ``after``-only filter
silently slid the window when the newest requested candle hadn't yet been
published. Sending ``before`` as well pins both bounds; an underfilled
window then surfaces as ``TransientOkxError`` (skip the round) instead
of ``InvariantError`` (crash the bot).

Run:
    python -m pytest tests/test_okx_client_kline_fetch.py -v
    python tests/test_okx_client_kline_fetch.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.okx_client import (  # noqa: E402
    OkxClient,
    RetryPolicy,
)
from pancakebot.util import InvariantError, TransientOkxError  # noqa: E402


# Tight retry policy for tests: 2 attempts with a near-zero backoff so the
# retry path is exercised without slowing the suite.
_FAST_RETRY = RetryPolicy(max_attempts=2, backoff_seconds=(0.001,))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kline_response(rows: list[list]) -> mock.MagicMock:
    """Build a MagicMock Response wrapping an OKX-shaped JSON body.

    OKX returns rows newest-first as strings; ``kline_fetch_window``
    reverses them to oldest-first ``[ts_ms, o, h, l, c, v]`` floats.
    """
    resp = mock.MagicMock()
    resp.status_code = 200
    okx_rows = list(reversed([
        [str(int(r[0])), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5])]
        for r in rows
    ]))
    resp.json.return_value = {"code": "0", "data": okx_rows}
    return resp


def _flat_rows(*, oldest_open_ms: int, count: int) -> list[list]:
    """Build a contiguous oldest-first 1s-window of constant-price rows."""
    return [
        [oldest_open_ms + i * 1000, 100.0, 100.0, 100.0, 100.0, 1.0]
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Query construction: send_before_bound semantics
# ---------------------------------------------------------------------------


def test_send_before_bound_true_sends_both_after_and_before():
    """Live-gate path: both bounds pinned with correct off-by-one.

    after = newest_open_ms_inclusive + 1000 (OKX after is exclusive)
    before = oldest_open_ms - 1000 (OKX before is exclusive)
    """
    client = OkxClient(timeout_seconds=5.0)
    oldest = 1_777_007_690_000
    newest = oldest + 15 * 1000
    rows = _flat_rows(oldest_open_ms=oldest, count=16)
    with mock.patch.object(
        client._session, "get", return_value=_make_kline_response(rows),
    ) as mock_get:
        client.kline_fetch_window(
            symbol="BTC-USDT",
            oldest_open_ms=oldest,
            newest_open_ms_inclusive=newest,
            retry_policy=_FAST_RETRY,
            send_before_bound=True,
        )
    assert mock_get.call_count == 1
    params = mock_get.call_args.kwargs["params"]
    assert params["after"] == str(newest + 1000), (
        f"after must be newest+1000; got {params['after']}"
    )
    assert params["before"] == str(oldest - 1000), (
        f"before must be oldest-1000; got {params.get('before')}"
    )
    assert params["limit"] == "16"
    assert params["bar"] == "1s"
    assert params["instId"] == "BTC-USDT"


def test_send_before_bound_false_omits_before_param():
    """Sync (default) path: ``before`` is NOT sent. Preserves byte-identity
    with the canonical baseline at the OKX-request layer."""
    client = OkxClient(timeout_seconds=5.0)
    oldest = 1_777_007_690_000
    newest = oldest + 299 * 1000  # 300-candle sync window
    rows = _flat_rows(oldest_open_ms=oldest, count=300)
    with mock.patch.object(
        client._session, "get", return_value=_make_kline_response(rows),
    ) as mock_get:
        client.kline_fetch_window(
            symbol="BTC-USDT",
            oldest_open_ms=oldest,
            newest_open_ms_inclusive=newest,
            retry_policy=_FAST_RETRY,
            # send_before_bound omitted → defaults to False
        )
    params = mock_get.call_args.kwargs["params"]
    assert params["after"] == str(newest + 1000)
    assert "before" not in params, (
        f"sync default must not send `before`; got params={params}"
    )


# ---------------------------------------------------------------------------
# Error classification: short response with both bounds → TransientOkxError
# ---------------------------------------------------------------------------


def test_short_response_with_both_bounds_raises_transient_after_retries():
    """OKX returns FEWER than expected_count rows (e.g. newest candle not
    yet published). Classified ``INSUFFICIENT`` → retry → on exhaustion,
    raises ``TransientOkxError`` (NOT ``InvariantError``).

    This is the canonical failure mode the ``before`` bound enables: the
    live decision path can skip the round on a single-candle delay
    instead of crashing on ``boundary_mismatch``.
    """
    client = OkxClient(timeout_seconds=5.0)
    oldest = 1_777_007_690_000
    newest = oldest + 15 * 1000  # expecting 16 candles
    short_rows = _flat_rows(oldest_open_ms=oldest, count=15)  # one short
    with mock.patch.object(
        client._session, "get", return_value=_make_kline_response(short_rows),
    ) as mock_get:
        raised: Exception | None = None
        try:
            client.kline_fetch_window(
                symbol="BTC-USDT",
                oldest_open_ms=oldest,
                newest_open_ms_inclusive=newest,
                retry_policy=_FAST_RETRY,
                send_before_bound=True,
            )
        except TransientOkxError as e:
            raised = e
        except InvariantError as e:
            raised = e
    assert isinstance(raised, TransientOkxError), (
        f"short response must raise TransientOkxError, got "
        f"{type(raised).__name__}: {raised}"
    )
    assert "kline_fetch_exhausted" in str(raised)
    # Both attempts hit OKX (retry is the whole point of the policy).
    assert mock_get.call_count == _FAST_RETRY.max_attempts


# ---------------------------------------------------------------------------
# Error classification: shape violations stay InvariantError
# ---------------------------------------------------------------------------


def test_boundary_mismatch_raises_invariant_error():
    """Returned rows match expected_count but oldest/newest open_times
    differ from requested. Still ``InvariantError`` (data shape wrong)."""
    client = OkxClient(timeout_seconds=5.0)
    oldest = 1_777_007_690_000
    newest = oldest + 15 * 1000
    # Wrong window: shifted 5 seconds older. Length matches, but boundaries
    # don't equal what we asked for.
    wrong_rows = _flat_rows(oldest_open_ms=oldest - 5_000, count=16)
    with mock.patch.object(
        client._session, "get", return_value=_make_kline_response(wrong_rows),
    ):
        raised: Exception | None = None
        try:
            client.kline_fetch_window(
                symbol="BTC-USDT",
                oldest_open_ms=oldest,
                newest_open_ms_inclusive=newest,
                retry_policy=_FAST_RETRY,
                send_before_bound=True,
            )
        except (InvariantError, TransientOkxError) as e:
            raised = e
    assert isinstance(raised, InvariantError), (
        f"boundary mismatch must raise InvariantError; got "
        f"{type(raised).__name__}: {raised}"
    )
    assert "boundary_mismatch" in str(raised)


def test_noncontiguous_rows_raise_invariant_error():
    """Returned rows match expected_count and boundaries, but a middle
    candle is missing (replaced with a duplicate). Contiguity check
    catches this -- still ``InvariantError``."""
    client = OkxClient(timeout_seconds=5.0)
    oldest = 1_777_007_690_000
    newest = oldest + 15 * 1000
    rows = _flat_rows(oldest_open_ms=oldest, count=16)
    # Inject a hole: replace row index 7 timestamp with a duplicate of
    # row 6 → adjacent delta becomes 0 (or row 8 jumps 2000ms from 6).
    rows[7] = list(rows[6])  # duplicate ts at idx 7
    with mock.patch.object(
        client._session, "get", return_value=_make_kline_response(rows),
    ):
        raised: Exception | None = None
        try:
            client.kline_fetch_window(
                symbol="BTC-USDT",
                oldest_open_ms=oldest,
                newest_open_ms_inclusive=newest,
                retry_policy=_FAST_RETRY,
                send_before_bound=True,
            )
        except (InvariantError, TransientOkxError) as e:
            raised = e
    assert isinstance(raised, InvariantError), (
        f"noncontiguous rows must raise InvariantError; got "
        f"{type(raised).__name__}: {raised}"
    )
    assert "noncontiguous" in str(raised)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


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
