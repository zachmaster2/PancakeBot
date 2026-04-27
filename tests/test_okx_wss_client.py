"""Unit tests for OkxWssClient -- ring buffer + push state machine + REST-fill.

Focuses on the in-memory state machine. Network-bound tests (real WSS connect)
are out of scope. Coverage:
  - get_window: stale, insufficient, normal, gap-fill-in-progress paths
  - is_ready: requires sub_ack + first_push_done + rest_done + !gap_fill
  - _handle_candle_push state machine:
      State 1 (pre-bootstrap)        -> buffer; first confirm=1 signals action
      State 2 (REST-repair pending)  -> buffer everything
      State 3 (steady state)         -> gap detection (any gap triggers action)
  - _apply_steady_state_row: append continuous, detect gap, handle dups
  - _rest_fill_to_T: REST fetch + boundary verify + atomic replace + drain
      Boundary OK   -> ring replaced with REST [oldest..T-1]; buffer drained
      REST[T] != WSS[T] -> _fatal_error set, _stop_event signalled
      REST returns no T -> _fatal_error set after retry
  - bootstrap_first_push_done resets on reconnect (state-reset block)
  - is_ready() False while gap_fill_in_progress
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.okx_wss_client import (  # noqa: E402
    OkxWssClient,
    _DEFAULT_RING_MAX,
    _HISTORY_OLDEST_OFFSET_MS,
    _rows_equal,
)
from pancakebot.util import InvariantError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(
    instruments=("BTC-USDT",),
    ring_max=10,
    next_lock_at_ms: int = 1_700_000_000_000,
) -> OkxWssClient:
    """Construct a client with a stub OKX client and fixed next_lock_at."""
    fake_okx = mock.MagicMock()
    return OkxWssClient(
        okx_client=fake_okx,
        instruments=instruments,
        next_lock_at_ms_provider=lambda: next_lock_at_ms,
        ring_max=ring_max,
    )


def _push_row(ts_ms, close, confirm, *, openp=None, high=None, low=None, vol=1.0):
    """OKX candle1s row: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]."""
    o = close if openp is None else openp
    h = close if high is None else high
    l_ = close if low is None else low
    return [str(ts_ms), str(o), str(h), str(l_), str(close), str(vol),
            "100", "100", confirm]


def _set_steady_state(ring, last_ts_ms: int, *, seed_ring_tail: bool = True,
                      tail_close: float = 100.0) -> None:
    """Mark the ring as fully bootstrapped (state 3) for tests that exercise
    the steady-state path directly. By default also appends a matching tail
    entry at last_ts_ms with OHLCV [tail_close]*5 so the duplicate / OOO
    anomaly checks (Phase 2 spec item 12) can compare against a real row."""
    ring.bootstrap_sub_ack = True
    ring.bootstrap_first_push_done = True
    ring.bootstrap_rest_done = True
    ring.gap_fill_in_progress = False
    ring.last_candle_ts_ms = last_ts_ms
    if seed_ring_tail:
        ring.klines.append([last_ts_ms, tail_close, tail_close, tail_close,
                            tail_close, 1.0])


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_constructor_rejects_empty_instruments():
    fake = mock.MagicMock()
    raised = False
    try:
        OkxWssClient(
            okx_client=fake, instruments=(),
            next_lock_at_ms_provider=lambda: 0,
        )
    except ValueError:
        raised = True
    assert raised, "expected ValueError on empty instruments"


# ---------------------------------------------------------------------------
# Ring deque maxlen
# ---------------------------------------------------------------------------

def test_ring_deque_respects_maxlen():
    c = _make_client(ring_max=3)
    ring = c._rings["BTC-USDT"]
    for ts in (1000, 2000, 3000, 4000):
        ring.klines.append([ts, 0.0, 0.0, 0.0, 100.0, 0.0])
    assert len(ring.klines) == 3
    assert ring.klines[0][0] == 2000  # 1000 evicted
    assert ring.klines[-1][0] == 4000


# ---------------------------------------------------------------------------
# get_window
# ---------------------------------------------------------------------------

def test_get_window_unknown_symbol():
    c = _make_client()
    klines, reason = c.get_window("DOGE-USDT", cutoff_ms=0, expected_count=16)
    assert klines is None
    assert reason == "wss_unknown_symbol"


def test_get_window_returns_bootstrap_pending_until_rest_done():
    """Until ``bootstrap_rest_done`` flips True (post-boundary-verify drain),
    every read returns ``wss_bootstrap_pending`` -- regardless of the ring's
    last_received_ms or how many candles are present. This is the WSS-layer
    self-enforcing gate that lets the strategy gate drop its BTC-specific
    cutoff defensive check."""
    c = _make_client(ring_max=50)
    ring = c._rings["BTC-USDT"]
    now_ms = (int(time.time() * 1000) // 1000) * 1000  # align to second boundary
    # Populate a "looks healthy" ring but DON'T flip rest_done.
    ring.last_received_ms = now_ms
    for i in range(20):
        ring.klines.append([now_ms - (20 - i) * 1000, 0.0, 0.0, 0.0, 100.0, 0.0])
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=now_ms, expected_count=16)
    assert klines is None
    assert reason == "wss_bootstrap_pending"
    # Flip rest_done -> normal read works (newest in valid is now-1000 == cutoff-1000).
    ring.bootstrap_rest_done = True
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=now_ms, expected_count=16)
    assert reason is None, f"unexpected skip: {reason}"
    assert klines is not None


def test_get_window_returns_gap_fill_in_progress_when_mid_repair():
    """A ring that completed bootstrap (rest_done=True) but is currently
    mid-gap-fill returns ``wss_gap_fill_in_progress`` -- regardless of
    last_received_ms or ring contents."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    ring.gap_fill_in_progress = True
    klines, reason = c.get_window(
        "BTC-USDT", cutoff_ms=int(time.time() * 1000), expected_count=16,
    )
    assert klines is None
    assert reason == "wss_gap_fill_in_progress"


def test_get_window_insufficient_when_ring_too_small():
    c = _make_client(ring_max=50)
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    now_ms = (int(time.time() * 1000) // 1000) * 1000
    ring.last_received_ms = now_ms
    for i in range(5):
        ring.klines.append([now_ms - (5 - i) * 1000, 0.0, 0.0, 0.0, 100.0, 0.0])
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=now_ms, expected_count=16)
    assert klines is None
    assert reason == "wss_insufficient"


# --- Item 9 wss_newest_lagging + Item 13 needs_reconnect signal ---

def test_get_window_newest_lagging_when_ring_behind_expected():
    """Ring's newest is older than ``cutoff_ms - 1000`` (push hasn't
    arrived yet, or WSS feed silent). Returns ``wss_newest_lagging`` and
    flags the ring for daemon reconnect (silent-WSS-death recovery)."""
    c = _make_client(ring_max=50)
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    cutoff_ms = 2_000_000_000_000  # arbitrary large round-aligned value
    # Populate enough candles ending at cutoff-2000 (one second BEHIND expected).
    for i in range(20):
        ring.klines.append(
            [cutoff_ms - 21_000 + i * 1000, 0.0, 0.0, 0.0, 100.0, 0.0]
        )
    # Newest in valid window = cutoff-2000, expected = cutoff-1000.
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=cutoff_ms, expected_count=16)
    assert klines is None
    assert reason == "wss_newest_lagging"
    assert ring.needs_reconnect is True
    assert ring.newest_lagging_streak == 1


def test_get_window_newest_lagging_idempotent_until_reconnect_clears():
    """Multiple ``get_window`` calls between reconnects don't re-increment
    the streak (counter only ticks when needs_reconnect transitions
    False -> True). This avoids bogus escalation if the gate calls
    get_window every second for 10s before the daemon picks up the signal."""
    c = _make_client(ring_max=50)
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    cutoff_ms = 2_000_000_000_000
    for i in range(20):
        ring.klines.append(
            [cutoff_ms - 21_000 + i * 1000, 0.0, 0.0, 0.0, 100.0, 0.0]
        )
    # 5 successive calls -- streak still 1 after.
    for _ in range(5):
        klines, reason = c.get_window("BTC-USDT", cutoff_ms=cutoff_ms, expected_count=16)
        assert reason == "wss_newest_lagging"
    assert ring.newest_lagging_streak == 1
    assert c._fatal_error is None


def test_get_window_success_clears_needs_reconnect_and_streak():
    """On a successful read (post-recovery), the per-ring needs_reconnect
    flag and newest_lagging_streak both reset."""
    c = _make_client(ring_max=50)
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    ring.needs_reconnect = True
    ring.newest_lagging_streak = 2
    cutoff_ms = 2_000_000_000_000
    for i in range(20):
        ring.klines.append(
            [cutoff_ms - 20_000 + i * 1000, 0.0, 0.0, 0.0, 100.0, 0.0]
        )
    # Newest = cutoff - 1000 -- fresh.
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=cutoff_ms, expected_count=16)
    assert reason is None
    assert klines is not None
    assert ring.needs_reconnect is False
    assert ring.newest_lagging_streak == 0


def test_get_window_newest_lagging_escalates_after_3_unrecovered_cycles():
    """After ``_NEWEST_LAGGING_MAX_RECONNECTS`` consecutive transitions,
    the client sets ``_fatal_error`` and signals the stop event."""
    c = _make_client(ring_max=50)
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    cutoff_ms = 2_000_000_000_000
    for i in range(20):
        ring.klines.append(
            [cutoff_ms - 21_000 + i * 1000, 0.0, 0.0, 0.0, 100.0, 0.0]
        )
    # Cycle 1
    c.get_window("BTC-USDT", cutoff_ms=cutoff_ms, expected_count=16)
    assert c._fatal_error is None
    # Simulate daemon reconnect (clear flag, leave streak).
    ring.needs_reconnect = False
    # Cycle 2
    c.get_window("BTC-USDT", cutoff_ms=cutoff_ms, expected_count=16)
    assert c._fatal_error is None
    ring.needs_reconnect = False
    # Cycle 3 -- escalation
    c.get_window("BTC-USDT", cutoff_ms=cutoff_ms, expected_count=16)
    assert c._fatal_error is not None
    assert "okx_wss_newest_lagging_unrecoverable" in c._fatal_error
    assert c._stop_event.is_set()


def test_any_needs_reconnect_helper():
    c = _make_client(instruments=("BTC-USDT", "ETH-USDT"))
    assert c._any_needs_reconnect() is False
    c._rings["ETH-USDT"].needs_reconnect = True
    assert c._any_needs_reconnect() is True
    c._rings["ETH-USDT"].needs_reconnect = False
    assert c._any_needs_reconnect() is False


def test_get_window_returns_filtered_window_in_normal_path():
    c = _make_client(ring_max=50)
    ring = c._rings["BTC-USDT"]
    ring.bootstrap_rest_done = True
    now_ms = (int(time.time() * 1000) // 1000) * 1000
    ring.last_received_ms = now_ms
    base_ts = now_ms - 35_000
    for i in range(35):
        ring.klines.append([base_ts + i * 1000, 0.0, 0.0, 0.0, 100.0 + i, 0.0])
    cutoff_ms = base_ts + 31_000
    klines, reason = c.get_window("BTC-USDT", cutoff_ms=cutoff_ms, expected_count=16)
    assert reason is None
    assert klines is not None
    assert len(klines) == 16
    assert klines[-1][0] == cutoff_ms - 1000


# ---------------------------------------------------------------------------
# _handle_sub_ack
# ---------------------------------------------------------------------------

def test_handle_sub_ack_marks_bootstrap_state():
    c = _make_client()
    assert c._rings["BTC-USDT"].bootstrap_sub_ack is False
    c._handle_sub_ack({"event": "subscribe",
                       "arg": {"channel": "candle1s", "instId": "BTC-USDT"}})
    assert c._rings["BTC-USDT"].bootstrap_sub_ack is True


def test_handle_sub_ack_unknown_symbol_silently_ignored():
    c = _make_client()
    c._handle_sub_ack({"event": "subscribe",
                       "arg": {"channel": "candle1s", "instId": "DOGE-USDT"}})
    assert c._rings["BTC-USDT"].bootstrap_sub_ack is False


# ---------------------------------------------------------------------------
# _handle_candle_push -- State 1 (pre-bootstrap)
# ---------------------------------------------------------------------------

def test_state1_mid_bar_is_discarded_entirely():
    """Per Phase 2 spec item 14, confirm=0 (mid-bar) is discarded across all
    states: no buffer, no first_push_done flip, no last_received_ms update."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    actions = c._handle_candle_push("BTC-USDT", [_push_row(2000_000, 100.0, "0")])
    assert actions == []
    assert ring.bootstrap_first_push_done is False
    assert ring.first_push_open_ts_ms == 0
    assert len(ring.gap_buffer) == 0
    assert ring.last_received_ms == 0  # confirm=0 must NOT update this


def test_state1_first_confirm1_records_T_and_signals_action():
    """The first confirm=1 push records first_push_open_ts_ms = T and emits
    an (symbol, T) action for the listener to spawn the REST-fill task."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    actions = c._handle_candle_push("BTC-USDT", [_push_row(2000_000, 100.0, "1")])
    assert actions == [("BTC-USDT", 2000_000)]
    assert ring.bootstrap_first_push_done is True
    assert ring.first_push_open_ts_ms == 2000_000
    assert ring.bootstrap_rest_done is False  # still pending REST
    assert len(ring.gap_buffer) == 1
    # Ring NOT yet appended to (REST will replace it).
    assert len(ring.klines) == 0


def test_state1_subsequent_pushes_buffer_only_confirm1():
    """After the first confirm=1, subsequent confirm=1 rows (state 2) buffer
    without additional actions; confirm=0 is discarded entirely."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    actions1 = c._handle_candle_push("BTC-USDT", [_push_row(2000_000, 100.0, "1")])
    assert len(actions1) == 1
    actions2 = c._handle_candle_push("BTC-USDT", [_push_row(2001_000, 101.0, "1")])
    assert actions2 == []
    actions3 = c._handle_candle_push("BTC-USDT", [_push_row(2002_000, 102.0, "0")])
    assert actions3 == []
    # Only the two confirm=1 rows are buffered (the confirm=0 was dropped).
    assert len(ring.gap_buffer) == 2
    assert ring.gap_buffer[0][1][0] == 2000_000
    assert ring.gap_buffer[1][1][0] == 2001_000


# ---------------------------------------------------------------------------
# _handle_candle_push -- State 3 (steady state) gap detection
# ---------------------------------------------------------------------------

def test_state3_continuous_confirm1_appends():
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000)
    # Ring starts with seeded tail at 1000_000.
    assert len(ring.klines) == 1
    actions = c._handle_candle_push("BTC-USDT", [_push_row(1001_000, 101.0, "1")])
    assert actions == []
    assert len(ring.klines) == 2
    assert ring.klines[-1][0] == 1001_000
    assert ring.last_candle_ts_ms == 1001_000


def test_state3_gap_of_one_triggers_gap_fill():
    """ANY gap > 1s (per spec item 12) triggers gap-fill. ts = last + 2000
    is one missed candle."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000)
    pre_size = len(ring.klines)
    actions = c._handle_candle_push("BTC-USDT", [_push_row(1002_000, 100.0, "1")])
    assert actions == [("BTC-USDT", 1002_000)]
    assert ring.gap_fill_in_progress is True
    # Gap-detect does NOT append the trigger row to klines (it goes to buffer).
    assert len(ring.klines) == pre_size
    assert len(ring.gap_buffer) == 1
    assert ring.gap_buffer[0][1][0] == 1002_000


def test_state3_large_gap_triggers_gap_fill():
    """100s gap -- same gap-fill flow."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000)
    actions = c._handle_candle_push("BTC-USDT", [_push_row(1100_000, 100.0, "1")])
    assert actions == [("BTC-USDT", 1100_000)]
    assert ring.gap_fill_in_progress is True


# --- Item 12 differentiated anomaly handling ---

def test_state3_duplicate_matching_values_silent():
    """ts == last with OHLCV matching the ring tail -> silent discard
    (no action, no buffer, no warning, ring unchanged)."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000, tail_close=100.0)
    pre_size = len(ring.klines)
    actions = c._handle_candle_push("BTC-USDT", [_push_row(1000_000, 100.0, "1")])
    assert actions == []
    assert len(ring.klines) == pre_size  # unchanged
    assert ring.gap_fill_in_progress is False
    assert len(ring.gap_buffer) == 0


def test_state3_duplicate_differing_values_logs_warning_and_drops():
    """ts == last but OHLCV differs from ring tail -> warning logged,
    discarded, ring unchanged. Could indicate OKX backend serving
    inconsistent data for the same closed candle."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000, tail_close=100.0)
    pre_size = len(ring.klines)
    # close differs (100.0 vs 999.0)
    actions = c._handle_candle_push("BTC-USDT", [_push_row(1000_000, 999.0, "1")])
    assert actions == []
    assert len(ring.klines) == pre_size
    assert ring.gap_fill_in_progress is False


def test_state3_out_of_order_matching_existing_silent():
    """ts < last and matches an existing ring entry -> silent discard.
    Late-arriving retransmission of a candle we already have."""
    c = _make_client(ring_max=10)
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000, tail_close=100.0)
    # Append 999 second to make it part of ring
    earlier = [999_000, 99.0, 99.0, 99.0, 99.0, 1.0]
    ring.klines.appendleft(earlier)
    pre_size = len(ring.klines)
    # Push the same earlier candle again (matching values).
    actions = c._handle_candle_push("BTC-USDT", [_push_row(999_000, 99.0, "1")])
    assert actions == []
    assert len(ring.klines) == pre_size
    assert ring.gap_fill_in_progress is False


def test_state3_out_of_order_differing_logs_and_drops():
    """ts < last and matches an existing ring entry but OHLCV differs ->
    warning logged, discarded."""
    c = _make_client(ring_max=10)
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000, tail_close=100.0)
    earlier = [999_000, 99.0, 99.0, 99.0, 99.0, 1.0]
    ring.klines.appendleft(earlier)
    pre_size = len(ring.klines)
    # Push earlier candle with DIFFERENT close.
    actions = c._handle_candle_push("BTC-USDT", [_push_row(999_000, 12345.0, "1")])
    assert actions == []
    assert len(ring.klines) == pre_size  # unchanged
    assert ring.gap_fill_in_progress is False


def test_state3_out_of_order_older_than_oldest_silent():
    """ts < last AND older than ring's oldest -> silent discard
    (irrelevant; rolled out of the window already)."""
    c = _make_client(ring_max=5)
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000, tail_close=100.0)
    # Ring tail at 1000_000; oldest at 1000_000 (since we only seeded one).
    # Push something WAY older.
    actions = c._handle_candle_push("BTC-USDT", [_push_row(500_000, 50.0, "1")])
    assert actions == []
    assert ring.gap_fill_in_progress is False


def test_state3_confirm0_is_discarded_silently():
    """Per Phase 2 spec item 14: confirm=0 push in state 3 is fully
    discarded -- no ring mutation, no last_received_ms update, no buffer."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000)
    pre_received = ring.last_received_ms
    pre_size = len(ring.klines)
    actions = c._handle_candle_push("BTC-USDT", [_push_row(1001_000, 100.0, "0")])
    assert actions == []
    assert len(ring.klines) == pre_size  # unchanged
    assert len(ring.gap_buffer) == 0
    assert ring.last_received_ms == pre_received  # NOT updated by confirm=0


def test_state2_during_gap_fill_buffers_only_confirm1():
    """gap_fill_in_progress=True suppresses steady-state logic; confirm=1
    rows buffer for drain. confirm=0 is discarded (item 14)."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000)
    ring.gap_fill_in_progress = True
    pre_size = len(ring.klines)
    actions = c._handle_candle_push("BTC-USDT", [
        _push_row(1001_000, 100.0, "1"),
        _push_row(1002_000, 101.0, "1"),
        _push_row(1003_000, 102.0, "0"),
    ])
    assert actions == []
    assert len(ring.klines) == pre_size  # gap-fill will atomically replace
    # Only the two confirm=1 rows are buffered.
    assert len(ring.gap_buffer) == 2
    assert ring.gap_buffer[0][1][0] == 1001_000
    assert ring.gap_buffer[1][1][0] == 1002_000


def test_handle_push_unknown_symbol_ignored():
    c = _make_client()
    c._handle_candle_push("DOGE-USDT", [_push_row(1000, 100.0, "1")])
    assert len(c._rings["BTC-USDT"].klines) == 0
    assert len(c._rings["BTC-USDT"].gap_buffer) == 0


def test_handle_push_malformed_row_skipped():
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000)
    pre_size = len(ring.klines)
    # Too few fields
    c._handle_candle_push("BTC-USDT", [["1000", "100"]])
    # Non-numeric ts
    c._handle_candle_push("BTC-USDT", [_push_row("abc", "xyz", "1")])
    # None field (TypeError catch)
    c._handle_candle_push("BTC-USDT", [[1000, None, None, None, None, None,
                                       "100", "100", "1"]])
    assert len(ring.klines) == pre_size  # unchanged


def test_confirm1_push_updates_last_received_ms():
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    before = ring.last_received_ms
    c._handle_candle_push("BTC-USDT", [_push_row(1000, 100.0, "1")])
    assert ring.last_received_ms > before


def test_confirm0_push_does_not_update_last_received_ms():
    """Per Phase 2 spec item 14: confirm=0 (mid-bar) MUST NOT update
    last_received_ms. Otherwise WSS that emits only mid-bars (no closes)
    would mask the silent-death failure mode that newest_lagging is
    designed to catch."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    before = ring.last_received_ms
    c._handle_candle_push("BTC-USDT", [_push_row(1000, 100.0, "0")])
    assert ring.last_received_ms == before, (
        "confirm=0 push must NOT touch last_received_ms"
    )


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------

def test_is_ready_requires_all_four_conditions():
    c = _make_client(instruments=("BTC-USDT", "ETH-USDT"))
    assert not c.is_ready()
    btc = c._rings["BTC-USDT"]
    eth = c._rings["ETH-USDT"]
    for r in (btc, eth):
        r.bootstrap_sub_ack = True
        r.bootstrap_first_push_done = True
        r.bootstrap_rest_done = True
    assert c.is_ready()
    # gap_fill_in_progress on either suppresses readiness.
    btc.gap_fill_in_progress = True
    assert not c.is_ready()
    btc.gap_fill_in_progress = False
    assert c.is_ready()


def test_is_ready_requires_all_four_instruments_including_bnb():
    """BNB-USDT is a first-class instrument: is_ready() blocks until BNB's
    bootstrap completes, identical to BTC/ETH/SOL. (Bot bets on BNB/USD.)"""
    c = _make_client(instruments=("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"))
    rings = list(c._rings.values())
    # All except BNB ready
    for r in rings:
        if r.symbol == "BNB-USDT":
            continue
        r.bootstrap_sub_ack = True
        r.bootstrap_first_push_done = True
        r.bootstrap_rest_done = True
    assert not c.is_ready(), "is_ready() must block while BNB bootstrap pending"
    # Complete BNB
    bnb = c._rings["BNB-USDT"]
    bnb.bootstrap_sub_ack = True
    bnb.bootstrap_first_push_done = True
    bnb.bootstrap_rest_done = True
    assert c.is_ready()
    # BNB gap-fill suppresses readiness too
    bnb.gap_fill_in_progress = True
    assert not c.is_ready()


def test_is_ready_partial_conditions_block():
    c = _make_client()
    btc = c._rings["BTC-USDT"]
    btc.bootstrap_sub_ack = True
    assert not c.is_ready()
    btc.bootstrap_first_push_done = True
    assert not c.is_ready()
    btc.bootstrap_rest_done = True
    assert c.is_ready()


# ---------------------------------------------------------------------------
# _rest_fill_to_T  (the unified bootstrap + gap-fill path)
# ---------------------------------------------------------------------------

def _T_row_arr(ts_ms, close, vol=1.0):
    """Plain [ts, o, h, l, c, v] array as ring stores."""
    return [ts_ms, float(close), float(close), float(close), float(close), float(vol)]


def test_rest_fill_atomic_replace_and_drain():
    """Successful flow: REST returns [oldest..T] matching the WSS first-push
    T row; ring is replaced with [oldest..T-1]; buffered post-T pushes drain."""
    next_lock_at_ms = 1_700_000_300_000   # T at lock-50000 (any value > oldest)
    T_ms = next_lock_at_ms - 50_000        # 1_700_000_250_000
    oldest_ms = next_lock_at_ms - _HISTORY_OLDEST_OFFSET_MS

    c = _make_client(next_lock_at_ms=next_lock_at_ms)
    ring = c._rings["BTC-USDT"]
    # State 1 -> first-push T arrives
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms, 100.0, "1")])
    # Then a couple of post-T pushes arrive while REST is "in flight" (state 2)
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms + 1000, 101.0, "1")])
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms + 2000, 102.0, "1")])
    assert len(ring.gap_buffer) == 3

    # Stub REST to return oldest..T (3 entries here for compactness; in real
    # operation this could be up to 300). The T entry MUST equal the WSS
    # first-push T row exactly.
    rest_arrays = [
        _T_row_arr(T_ms - 2000, 98.0),
        _T_row_arr(T_ms - 1000, 99.0),
        _T_row_arr(T_ms, 100.0),  # boundary — must match WSS [T_ms, 100, 100, 100, 100, 1.0]
    ]
    c._client.fetch_kline_window = mock.MagicMock(return_value=rest_arrays)

    # Skip the 2s sleep for fast tests by patching asyncio.sleep.
    async def _no_sleep(_seconds):
        return
    with mock.patch("pancakebot.market_data.okx_wss_client.asyncio.sleep", _no_sleep):
        asyncio.run(c._rest_fill_to_T("BTC-USDT", T_ms))

    assert c._fatal_error is None
    assert ring.bootstrap_rest_done is True
    assert ring.gap_fill_in_progress is False
    # Ring contents = REST[oldest..T-1] (drop REST's T) + drained T, T+1, T+2
    expected_ts = [T_ms - 2000, T_ms - 1000, T_ms, T_ms + 1000, T_ms + 2000]
    assert [k[0] for k in ring.klines] == expected_ts
    assert ring.last_candle_ts_ms == T_ms + 2000
    assert ring.gap_buffer == []
    # Verify fetch was called with the right window
    call_kwargs = c._client.fetch_kline_window.call_args.kwargs
    assert call_kwargs["symbol"] == "BTC-USDT"
    assert call_kwargs["oldest_open_ms"] == oldest_ms
    assert call_kwargs["newest_open_ms_inclusive"] == T_ms


def test_rest_fill_boundary_mismatch_sets_fatal():
    """REST[T] OHLCV != WSS first-push T → InvariantError trail (fatal_error
    set, stop_event signalled). No silent recovery."""
    next_lock_at_ms = 1_700_000_300_000
    T_ms = next_lock_at_ms - 50_000

    c = _make_client(next_lock_at_ms=next_lock_at_ms)
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms, 100.0, "1")])
    # REST returns a T entry with DIFFERENT close (99.0 vs WSS's 100.0).
    rest_arrays = [
        _T_row_arr(T_ms - 1000, 99.0),
        _T_row_arr(T_ms, 99.0),  # WSS has 100.0 here -- divergence
    ]
    c._client.fetch_kline_window = mock.MagicMock(return_value=rest_arrays)

    async def _no_sleep(_):
        return
    with mock.patch("pancakebot.market_data.okx_wss_client.asyncio.sleep", _no_sleep):
        asyncio.run(c._rest_fill_to_T("BTC-USDT", T_ms))

    assert c._fatal_error is not None
    assert "okx_rest_wss_boundary_mismatch" in c._fatal_error
    assert c._stop_event.is_set()


def test_rest_fill_boundary_unavailable_after_retry_sets_fatal():
    """Both REST attempts raise InvariantError (e.g. boundary unavailable);
    second failure sets _fatal_error. No third attempt."""
    next_lock_at_ms = 1_700_000_300_000
    T_ms = next_lock_at_ms - 50_000

    c = _make_client(next_lock_at_ms=next_lock_at_ms)
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms, 100.0, "1")])
    c._client.fetch_kline_window = mock.MagicMock(
        side_effect=InvariantError("okx_kline_window_boundary_mismatch: simulated"),
    )

    async def _no_sleep(_):
        return
    with mock.patch("pancakebot.market_data.okx_wss_client.asyncio.sleep", _no_sleep):
        asyncio.run(c._rest_fill_to_T("BTC-USDT", T_ms))

    assert c._fatal_error is not None
    assert "okx_rest_boundary_unavailable" in c._fatal_error
    assert c._stop_event.is_set()
    # Exactly 2 attempts (RETRY_WSS shape: try, retry once, then fatal).
    assert c._client.fetch_kline_window.call_count == 2


def test_rest_fill_start_of_round_only_T_returned():
    """Start-of-round edge case: oldest_needed == T (the very first second
    of the round), so REST returns exactly one row [T]. After bootstrap
    the ring must contain [T] (not empty), last_candle_ts_ms == T, and
    a subsequent confirm=1 push at T+1000 must append continuously
    through normal gap detection -- no special empty-ring branch."""
    # Choose T = next_lock - 301_000 (i.e. exactly oldest_needed).
    next_lock_at_ms = 1_700_000_300_000
    T_ms = next_lock_at_ms - _HISTORY_OLDEST_OFFSET_MS  # == oldest_needed
    c = _make_client(next_lock_at_ms=next_lock_at_ms)
    ring = c._rings["BTC-USDT"]
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms, 100.0, "1")])
    # REST returns just [T] (oldest == newest_inclusive == T).
    rest_arrays = [_T_row_arr(T_ms, 100.0)]
    c._client.fetch_kline_window = mock.MagicMock(return_value=rest_arrays)

    async def _no_sleep(_):
        return
    with mock.patch("pancakebot.market_data.okx_wss_client.asyncio.sleep", _no_sleep):
        asyncio.run(c._rest_fill_to_T("BTC-USDT", T_ms))

    # Ring must hold exactly [T] -- the boundary T row from WSS, NOT empty.
    assert c._fatal_error is None
    assert ring.bootstrap_rest_done is True
    assert ring.gap_fill_in_progress is False
    assert [k[0] for k in ring.klines] == [T_ms]
    assert ring.last_candle_ts_ms == T_ms

    # A subsequent confirm=1 push at T+1000 must append continuously via
    # normal gap detection (no special-case bootstrap path).
    actions = c._handle_candle_push("BTC-USDT", [_push_row(T_ms + 1000, 101.0, "1")])
    assert actions == []
    assert [k[0] for k in ring.klines] == [T_ms, T_ms + 1000]
    assert ring.last_candle_ts_ms == T_ms + 1000

    # And a gap (T+3000 with last==T+1000) still triggers gap-fill.
    actions = c._handle_candle_push("BTC-USDT", [_push_row(T_ms + 3000, 103.0, "1")])
    assert actions == [("BTC-USDT", T_ms + 3000)]
    assert ring.gap_fill_in_progress is True


def test_rest_fill_snapshots_oldest_needed_once_per_flow():
    """Item 11: ``next_lock_at_ms_provider`` is called exactly ONCE at the
    start of ``_rest_fill_to_T``. Even if the round transitions during the
    flow (provider would now return a different value), retries inside
    ``_fetch_rest_with_retry`` reuse the snapshotted oldest_needed."""
    next_lock_at_ms = 1_700_000_300_000
    T_ms = next_lock_at_ms - 50_000

    # Provider returns the snapshotted lock first, then a SHIFTED lock on
    # subsequent calls (simulating round transition).
    call_log: list[int] = []
    next_round_lock_ms = next_lock_at_ms + 300_000  # next round (5min later)

    def _provider() -> int:
        call_log.append(next_lock_at_ms if not call_log else next_round_lock_ms)
        return call_log[-1]

    fake_okx = mock.MagicMock()
    c = OkxWssClient(
        okx_client=fake_okx,
        instruments=("BTC-USDT",),
        next_lock_at_ms_provider=_provider,
        ring_max=10,
    )
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms, 100.0, "1")])

    rest_arrays = [
        _T_row_arr(T_ms - 1000, 99.0),
        _T_row_arr(T_ms, 100.0),
    ]
    c._client.fetch_kline_window = mock.MagicMock(return_value=rest_arrays)

    async def _no_sleep(_):
        return
    with mock.patch("pancakebot.market_data.okx_wss_client.asyncio.sleep", _no_sleep):
        asyncio.run(c._rest_fill_to_T("BTC-USDT", T_ms))

    # Provider called exactly once -- the snapshot.
    assert len(call_log) == 1, (
        f"expected 1 provider call (snapshot at flow start); got {len(call_log)}"
    )
    # Verify the fetch used the snapshotted oldest_needed (NOT the
    # round-transitioned value).
    expected_oldest = next_lock_at_ms - _HISTORY_OLDEST_OFFSET_MS
    call_kwargs = c._client.fetch_kline_window.call_args.kwargs
    assert call_kwargs["oldest_open_ms"] == expected_oldest


def test_rest_fill_drain_skips_supersede_t_and_earlier():
    """The drain MUST skip the buffered WSS T row (already appended by the
    explicit step) AND any rows at ts <= T (superseded by the appended T).
    Only ts > T entries flow through gap detection. confirm=0 mid-bars at
    ANY ts are dropped at the push handler entry per spec item 14, so the
    drain never sees them."""
    next_lock_at_ms = 1_700_000_300_000
    T_ms = next_lock_at_ms - 50_000
    c = _make_client(next_lock_at_ms=next_lock_at_ms)
    ring = c._rings["BTC-USDT"]
    # Buffer-eligible: confirm=1 at T, confirm=1 at T+1000.
    # confirm=0 mid-bars at T-1000 and T are silently discarded.
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms - 1000, 99.5, "0")])
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms, 99.9, "0")])
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms, 100.0, "1")])
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms + 1000, 101.0, "1")])
    # Verify the mid-bars never made it into the buffer.
    assert len(ring.gap_buffer) == 2
    assert ring.gap_buffer[0][1][0] == T_ms
    assert ring.gap_buffer[1][1][0] == T_ms + 1000

    rest_arrays = [
        _T_row_arr(T_ms - 1000, 99.0),
        _T_row_arr(T_ms, 100.0),
    ]
    c._client.fetch_kline_window = mock.MagicMock(return_value=rest_arrays)

    async def _no_sleep(_):
        return
    with mock.patch("pancakebot.market_data.okx_wss_client.asyncio.sleep", _no_sleep):
        asyncio.run(c._rest_fill_to_T("BTC-USDT", T_ms))

    # Expected ring: REST[T-1000] + WSS[T] + drained[T+1000].
    assert c._fatal_error is None
    assert [k[0] for k in ring.klines] == [T_ms - 1000, T_ms, T_ms + 1000]
    assert ring.last_candle_ts_ms == T_ms + 1000


def test_rest_fill_drain_discovers_secondary_gap():
    """Edge case: while bootstrap REST was in flight, the WSS feed dropped
    a candle (T+1 missing). The drain appends T (continuous), then detects
    the gap at T+2 -- sets gap_fill_in_progress, buffers T+2 and T+3 (the
    second gap-detect short-circuits to buffer only), and the listener
    will spawn a follow-up REST-fill task (we capture create_task to
    verify the spawn intent without actually running the nested coroutine)."""
    next_lock_at_ms = 1_700_000_300_000
    T_ms = next_lock_at_ms - 50_000

    c = _make_client(next_lock_at_ms=next_lock_at_ms)
    ring = c._rings["BTC-USDT"]
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms, 100.0, "1")])
    # T+1 missing from buffer -- only T, T+2, T+3 buffered.
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms + 2000, 102.0, "1")])
    c._handle_candle_push("BTC-USDT", [_push_row(T_ms + 3000, 103.0, "1")])

    rest_arrays = [
        _T_row_arr(T_ms - 1000, 99.0),
        _T_row_arr(T_ms, 100.0),  # boundary OK
    ]
    c._client.fetch_kline_window = mock.MagicMock(return_value=rest_arrays)

    spawned_targets: list = []

    def _capture_create_task(coro):
        # Close the coro immediately so it doesn't leak; we only care that
        # _rest_fill_to_T tried to spawn a nested fill.
        spawned_targets.append("nested_fill_spawned")
        coro.close()
        return mock.MagicMock()

    async def _no_sleep(_):
        return

    with mock.patch("pancakebot.market_data.okx_wss_client.asyncio.sleep", _no_sleep), \
         mock.patch("pancakebot.market_data.okx_wss_client.asyncio.create_task",
                    side_effect=_capture_create_task):
        asyncio.run(c._rest_fill_to_T("BTC-USDT", T_ms))

    # No fatal -- the first REST-fill succeeded, drain then detected the
    # secondary gap.
    assert c._fatal_error is None
    # T was appended successfully via the drain.
    assert ring.last_candle_ts_ms == T_ms
    # Secondary gap detected: gap_fill_in_progress, T+2 + T+3 in buffer.
    assert ring.gap_fill_in_progress is True
    buffered_ts = sorted(arr[0] for _, arr in ring.gap_buffer)
    assert buffered_ts == [T_ms + 2000, T_ms + 3000]
    # And the listener would have been asked to spawn ONE nested task
    # (the second buffered row short-circuits via the gap_fill_in_progress
    # guard, so only one action emits).
    assert spawned_targets == ["nested_fill_spawned"]


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def test_stats_returns_per_symbol_dict_all_four_instruments():
    """Stats must include all 4 first-class instruments (BNB included)."""
    c = _make_client(instruments=("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"))
    s = c.stats()
    for sym in ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"):
        assert sym in s, f"stats missing {sym}"
        assert s[sym]["ring_size"] == 0
        assert s[sym]["sub_ack"] is False
        assert s[sym]["first_push_done"] is False
        assert s[sym]["rest_done"] is False
        assert s[sym]["gap_fill_in_progress"] is False
        assert s[sym]["gap_buffer_size"] == 0
        assert s[sym]["first_push_open_ts_ms"] == 0


# ---------------------------------------------------------------------------
# _rows_equal -- math.isclose tolerance (Item 5, 2026-04-27)
# ---------------------------------------------------------------------------

def test_rows_equal_strict_equality():
    """Bit-identical rows match (the common case where REST and WSS parse
    the same OKX wire string)."""
    a = [1000_000, 100.0, 100.0, 100.0, 100.0, 1.0]
    b = [1000_000, 100.0, 100.0, 100.0, 100.0, 1.0]
    assert _rows_equal(a, b) is True


def test_rows_equal_sub_ulp_difference_passes():
    """Two rows whose OHLCV floats differ by sub-ULP noise (e.g. one parsed
    via slightly different code path that adds floating-point error well
    below the 1e-12 tolerance) still match. Regression guard against any
    future return to strict ``==`` that would spuriously flag this as
    boundary mismatch."""
    a = [1000_000, 100.0, 100.0, 100.0, 100.0, 1.0]
    # Add ~1e-14 error to one field -- below abs_tol of 1e-12.
    b = [1000_000, 100.0 + 1e-14, 100.0, 100.0, 100.0, 1.0]
    assert _rows_equal(a, b) is True


def test_rows_equal_just_above_tolerance_fails():
    """Differences larger than the 1e-12 tolerance still fail loud
    (real divergence, not representation noise)."""
    a = [1000_000, 100.0, 100.0, 100.0, 100.0, 1.0]
    b = [1000_000, 100.0 + 1e-9, 100.0, 100.0, 100.0, 1.0]
    assert _rows_equal(a, b) is False


def test_rows_equal_real_price_divergence_fails():
    """A penny-scale price disagreement (well above tolerance) fails as
    expected -- this is what the boundary check is supposed to catch."""
    a = [1000_000, 100.0, 100.0, 100.0, 100.0, 1.0]
    b = [1000_000, 100.01, 100.0, 100.0, 100.0, 1.0]
    assert _rows_equal(a, b) is False


def test_rows_equal_timestamp_strict_equality():
    """Timestamp field is integer; even a 1ms difference fails (no float
    tolerance applied to ts)."""
    a = [1000_000, 100.0, 100.0, 100.0, 100.0, 1.0]
    b = [1000_001, 100.0, 100.0, 100.0, 100.0, 1.0]
    assert _rows_equal(a, b) is False


# ---------------------------------------------------------------------------
# Multi-row push atomicity (Item 6, 2026-04-27)
# ---------------------------------------------------------------------------

def test_multi_row_push_holds_single_lock():
    """A multi-row push acquires ``self._lock`` exactly ONCE and applies all
    rows under that single lock. Without this, a main-thread reader could
    observe a partially-applied push (some rows in ring, others not) --
    after Item 1B removed the BTC defensive cutoff check, that partial
    state would slip through to the strategy gate undetected.

    We verify by counting acquires/releases on a wrapper around the lock
    that records every entry/exit. A single 3-row push must produce
    exactly 1 acquire + 1 release."""
    c = _make_client()
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=1000_000)

    # Wrap the lock to count acquire/release calls. ``threading.Lock`` is
    # a class-instance-level primitive in CPython; simplest to swap with a
    # MagicMock that delegates to a real lock.
    import threading
    real_lock = threading.Lock()
    acquire_count = [0]
    release_count = [0]

    class _CountingLock:
        def __enter__(self):
            acquire_count[0] += 1
            real_lock.acquire()
            return self
        def __exit__(self, exc_type, exc, tb):
            release_count[0] += 1
            real_lock.release()
            return False
    c._lock = _CountingLock()

    # 3-row push: continuous T+1, T+2, T+3 (all confirm=1).
    actions = c._handle_candle_push("BTC-USDT", [
        _push_row(1001_000, 101.0, "1"),
        _push_row(1002_000, 102.0, "1"),
        _push_row(1003_000, 103.0, "1"),
    ])
    assert actions == []
    assert acquire_count[0] == 1, (
        f"expected exactly 1 lock acquire for the 3-row push, got {acquire_count[0]}"
    )
    assert release_count[0] == 1, (
        f"expected exactly 1 lock release for the 3-row push, got {release_count[0]}"
    )
    # Ring tail is the seeded 1000_000 + 3 appended.
    assert [k[0] for k in ring.klines][-3:] == [1001_000, 1002_000, 1003_000]
    assert ring.last_candle_ts_ms == 1003_000


def test_multi_row_push_atomically_applies_all_or_none():
    """A multi-row push under a single lock means a concurrent reader sees
    EITHER all 3 rows or none of the 3, never just 1 or 2. We can't easily
    test true threading interleaving deterministically; instead we verify
    the post-push ring state -- which combined with the lock-count test
    above guarantees atomicity from the reader's perspective."""
    c = _make_client(ring_max=10)
    ring = c._rings["BTC-USDT"]
    _set_steady_state(ring, last_ts_ms=2000_000)
    pre_size = len(ring.klines)  # seeded tail
    # 3-row push.
    c._handle_candle_push("BTC-USDT", [
        _push_row(2001_000, 200.0, "1"),
        _push_row(2002_000, 201.0, "1"),
        _push_row(2003_000, 202.0, "1"),
    ])
    # Post-state: ring grew by exactly 3 (atomic apply).
    assert len(ring.klines) == pre_size + 3
    assert [k[0] for k in ring.klines][-3:] == [2001_000, 2002_000, 2003_000]
    assert ring.last_candle_ts_ms == 2003_000


def test_bnb_state1_first_push_signals_action_like_other_instruments():
    """BNB push handling is IDENTICAL to BTC/ETH/SOL: state-1 first-push
    records T and emits a (BNB, T) action for the listener to spawn the
    REST-fill task. Regression guard against any future code that
    silently treats BNB as second-class."""
    c = _make_client(instruments=("BTC-USDT", "BNB-USDT"))
    actions = c._handle_candle_push("BNB-USDT", [_push_row(2000_000, 600.0, "1")])
    assert actions == [("BNB-USDT", 2000_000)]
    bnb = c._rings["BNB-USDT"]
    assert bnb.bootstrap_first_push_done is True
    assert bnb.first_push_open_ts_ms == 2000_000


# ---------------------------------------------------------------------------
# Standalone runner (also pytest-compatible)
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
