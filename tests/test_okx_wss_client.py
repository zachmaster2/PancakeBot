"""Unit tests for OkxWssClient -- ring buffer + push handling + bootstrap.

Network-bound tests (real WSS connect) are out of scope here -- they'd be
flaky and slow. Coverage focuses on the in-memory state machine:
  - _InstrumentRing enforces deque maxlen
  - get_window: stale, insufficient, normal paths
  - _handle_candle_push: confirm semantics, dedupe by ts
  - _handle_sub_ack: bootstrap state transition
  - is_ready: requires all 3 conditions per ring
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.okx_wss_client import (  # noqa: E402
    OkxWssClient,
    _InstrumentRing,
    _DEFAULT_RING_MAX,
    _MIN_CANDLES_FOR_READY,
)


def _make_client_no_bootstrap(instruments=("BTC-USDT",), ring_max=10) -> OkxWssClient:
    """Construct a client without invoking start() / network."""
    fake_okx = mock.MagicMock()
    return OkxWssClient(
        okx_client=fake_okx,
        instruments=instruments,
        ring_max=ring_max,
    )


# ---------------------------------------------------------------------------
# _InstrumentRing
# ---------------------------------------------------------------------------

def test_ring_deque_respects_maxlen():
    """Ring discards oldest when at maxlen."""
    c = _make_client_no_bootstrap(ring_max=3)
    ring = c._rings["BTC-USDT"]
    for ts in (1000, 2000, 3000, 4000):
        ring.klines.append([ts, 0.0, 0.0, 0.0, 100.0, 0.0])
    assert len(ring.klines) == 3
    assert ring.klines[0][0] == 2000  # 1000 evicted
    assert ring.klines[-1][0] == 4000


# ---------------------------------------------------------------------------
# get_window
# ---------------------------------------------------------------------------

def test_get_window_stale_when_no_recent_push():
    c = _make_client_no_bootstrap()
    ring = c._rings["BTC-USDT"]
    ring.last_received_ms = int(time.time() * 1000) - 10_000  # 10s old
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=int(time.time() * 1000),
                                  expected_count=31, stale_threshold_ms=5000)
    assert klines is None
    assert reason == "wss_stale"


def test_get_window_insufficient_when_ring_too_small():
    c = _make_client_no_bootstrap(ring_max=50)
    ring = c._rings["BTC-USDT"]
    now_ms = int(time.time() * 1000)
    ring.last_received_ms = now_ms
    # Only put 5 candles
    for i in range(5):
        ring.klines.append([now_ms - (5 - i) * 1000, 0.0, 0.0, 0.0, 100.0, 0.0])
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=now_ms,
                                  expected_count=31, stale_threshold_ms=5000)
    assert klines is None
    assert reason == "wss_insufficient"


def test_get_window_returns_filtered_window_in_normal_path():
    c = _make_client_no_bootstrap(ring_max=50)
    ring = c._rings["BTC-USDT"]
    now_ms = int(time.time() * 1000)
    ring.last_received_ms = now_ms
    # 35 candles, oldest first, ts increments by 1000ms.
    base_ts = now_ms - 35_000
    for i in range(35):
        ring.klines.append([base_ts + i * 1000, 0.0, 0.0, 0.0, 100.0 + i, 0.0])
    cutoff_ms = base_ts + 31_000  # filter to 31 candles
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=cutoff_ms,
                                  expected_count=31, stale_threshold_ms=5000)
    assert reason is None
    assert klines is not None
    assert len(klines) == 31
    # Newest in returned should be cutoff - 1000
    assert klines[-1][0] == cutoff_ms - 1000


def test_get_window_unknown_symbol():
    c = _make_client_no_bootstrap()
    klines, reason = c.get_window("DOGE-USDT", cutoff_ms=0)
    assert klines is None
    assert reason == "wss_unknown_symbol"


def test_get_window_threshold_local_frame_no_skew_correction():
    """Critical: last_received_ms is LOCAL clock; threshold cancels skew.

    Set last_received to 4s old (LOCAL). Threshold is 5s. Should NOT be
    stale even if module-level skew is huge -- the comparison is
    LOCAL-vs-LOCAL.
    """
    c = _make_client_no_bootstrap()
    ring = c._rings["BTC-USDT"]
    now_ms = int(time.time() * 1000)
    ring.last_received_ms = now_ms - 4_000
    # Add a candle so insufficient doesn't trigger first
    for i in range(31):
        ring.klines.append([now_ms - (31 - i) * 1000, 0.0, 0.0, 0.0, 100.0, 0.0])
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=now_ms,
                                  expected_count=31, stale_threshold_ms=5000)
    # Not stale (4s < 5s threshold). Must return klines OR
    # "wss_insufficient" -- never "wss_stale".
    assert reason != "wss_stale"


# ---------------------------------------------------------------------------
# _handle_candle_push
# ---------------------------------------------------------------------------

def _push_row(ts_ms, close, confirm):
    """OKX candle1s row: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]."""
    return [str(ts_ms), str(close), str(close), str(close), str(close),
            "1.0", "100", "100", confirm]


def test_handle_candle_push_confirm_1_appends_to_ring():
    c = _make_client_no_bootstrap()
    ring = c._rings["BTC-USDT"]
    # Bootstrap conditions: pretend REST done, sub-ack done. We're testing
    # whether a confirm=1 push moves the bootstrap-first-push-done flag.
    ring.bootstrap_rest_done = True
    ring.bootstrap_sub_ack = True
    ring.last_candle_ts_ms = 1000_000
    c._handle_candle_push("BTC-USDT", [_push_row(1001_000, 100.0, "1")])
    assert len(ring.klines) == 1
    assert ring.klines[0][0] == 1001_000
    assert ring.klines[0][4] == 100.0
    assert ring.last_candle_ts_ms == 1001_000
    assert ring.bootstrap_first_push_done is True


def test_handle_candle_push_confirm_0_stays_pending():
    c = _make_client_no_bootstrap()
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    ring.bootstrap_sub_ack = True
    c._handle_candle_push("BTC-USDT", [_push_row(2000_000, 100.0, "0")])
    assert len(ring.klines) == 0  # no append on mid-bar
    assert ring.pending_open_time_ms == 2000_000
    assert ring.bootstrap_first_push_done is False  # not yet


def test_handle_candle_push_dedupe_on_overlap():
    """If a confirm=1 push arrives with ts <= last_candle_ts_ms, skip
    (handles REST/WSS bootstrap overlap)."""
    c = _make_client_no_bootstrap()
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    ring.bootstrap_sub_ack = True
    ring.last_candle_ts_ms = 5000_000  # REST already filled past this
    # Push an "earlier" or equal ts -- must NOT append.
    c._handle_candle_push("BTC-USDT", [_push_row(5000_000, 100.0, "1")])
    c._handle_candle_push("BTC-USDT", [_push_row(4999_000, 100.0, "1")])
    assert len(ring.klines) == 0
    # A strictly-newer push must succeed.
    c._handle_candle_push("BTC-USDT", [_push_row(5001_000, 100.0, "1")])
    assert len(ring.klines) == 1
    assert ring.klines[0][0] == 5001_000


def test_handle_candle_push_unknown_symbol_ignored():
    c = _make_client_no_bootstrap()
    # Should not raise; just returns
    c._handle_candle_push("DOGE-USDT", [_push_row(1000, 100.0, "1")])
    # No state changed
    assert len(c._rings["BTC-USDT"].klines) == 0


def test_handle_candle_push_malformed_row_skipped():
    c = _make_client_no_bootstrap()
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    ring.bootstrap_sub_ack = True
    # Too few fields
    c._handle_candle_push("BTC-USDT", [["1000", "100"]])
    # Non-numeric
    c._handle_candle_push("BTC-USDT", [_push_row("abc", "xyz", "1")])
    assert len(ring.klines) == 0


def test_handle_candle_push_updates_last_received_ms():
    c = _make_client_no_bootstrap()
    ring = c._rings["BTC-USDT"]
    before = ring.last_received_ms
    c._handle_candle_push("BTC-USDT", [_push_row(1000, 100.0, "0")])
    assert ring.last_received_ms > before


# ---------------------------------------------------------------------------
# _handle_sub_ack
# ---------------------------------------------------------------------------

def test_handle_sub_ack_marks_bootstrap_state():
    c = _make_client_no_bootstrap()
    assert c._rings["BTC-USDT"].bootstrap_sub_ack is False
    c._handle_sub_ack({"event": "subscribe", "arg": {"channel": "candle1s", "instId": "BTC-USDT"}})
    assert c._rings["BTC-USDT"].bootstrap_sub_ack is True


def test_handle_sub_ack_unknown_symbol_silently_ignored():
    c = _make_client_no_bootstrap()
    c._handle_sub_ack({"event": "subscribe", "arg": {"channel": "candle1s", "instId": "DOGE-USDT"}})
    assert c._rings["BTC-USDT"].bootstrap_sub_ack is False


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------

def test_is_ready_requires_all_3_conditions_per_ring():
    c = _make_client_no_bootstrap(instruments=("BTC-USDT", "ETH-USDT"))
    assert not c.is_ready()  # nothing started
    btc = c._rings["BTC-USDT"]
    eth = c._rings["ETH-USDT"]
    btc.bootstrap_rest_done = True
    btc.bootstrap_sub_ack = True
    btc.bootstrap_first_push_done = True
    assert not c.is_ready()  # ETH still pending
    eth.bootstrap_rest_done = True
    eth.bootstrap_sub_ack = True
    eth.bootstrap_first_push_done = True
    assert c.is_ready()


def test_is_ready_partial_per_ring_blocks():
    c = _make_client_no_bootstrap()
    btc = c._rings["BTC-USDT"]
    btc.bootstrap_rest_done = True
    btc.bootstrap_sub_ack = True
    # First push not yet
    assert not c.is_ready()
    btc.bootstrap_first_push_done = True
    assert c.is_ready()


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def test_stats_returns_per_symbol_dict():
    c = _make_client_no_bootstrap(instruments=("BTC-USDT", "ETH-USDT"))
    s = c.stats()
    assert "BTC-USDT" in s and "ETH-USDT" in s
    assert s["BTC-USDT"]["ring_size"] == 0
    assert s["BTC-USDT"]["rest_done"] is False


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_constructor_rejects_empty_instruments():
    fake = mock.MagicMock()
    try:
        OkxWssClient(okx_client=fake, instruments=())
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty instruments")


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
