"""Tests for ``MomentumGate.evaluate`` per-round REST fetch path.

The gate fires 3 parallel ``OkxClient.kline_fetch_window`` calls
(BTC/ETH/SOL) and aggregates results by exception class. BNB fetch
is currently disabled (see ``MomentumGate._OKX_SYMBOLS_FETCHED`` for
re-enable steps).

Behavior under test:
- Healthy path: all 3 symbols fetched, signal computed off the trimmed window.
- ``InvariantError`` from any symbol → reraised (bot crashes).
- ``TransientOkxError`` (any subset) → per-symbol warn + skip with
  ``kline_fetch_transient_failure``, increments streak counter.
- 3 consecutive transient rounds → escalates to ``InvariantError``.
- Streak resets to 0 on any successful 3-symbol fetch (regardless of signal).
- Fetch window is cutoff- and lookback-aware: newest = lock_at -
  cutoff*1000 - 1000, oldest = newest - max(mtf_lookbacks)*1000, expected
  count = max(mtf_lookbacks) + 1. Confirms the 2026-04-27 fix for
  ``kline_fetch_integrity_violation`` crashes that used a hardcoded
  300-candle, cutoff-blind window.
- ``send_before_bound=True`` is set so OKX cannot slide the window.
- ``last_fetch_timing`` populated with 3 entries (btc/eth/sol).

Run:
    python -m pytest tests/test_momentum_gate_kline_fetch.py -v
    python tests/test_momentum_gate_kline_fetch.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.strategy.momentum_gate import (  # noqa: E402
    MomentumGate,
    MomentumGateConfig,
    _MAX_CONSECUTIVE_FETCH_FAILURES,
)
from pancakebot.util import InvariantError, TransientOkxError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOCK_AT_MS = 1_777_007_708_000  # arbitrary recent lock_at, in ms
_CUTOFF_SECONDS = 2


def _make_gate(*, kline_cutoff_seconds: int = _CUTOFF_SECONDS, mtf_lookbacks=(3, 7, 15)):
    """Build a gate whose okx client is a MagicMock. Returns (gate, fake_client)."""
    cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=kline_cutoff_seconds,
        mtf_lookbacks=mtf_lookbacks,
        mtf_min_return_threshold=0.0001,
    )
    fake_client = mock.MagicMock()
    gate = MomentumGate(config=cfg, okx_client=fake_client)
    return gate, fake_client


def _flat_klines(
    *,
    lock_at_ms: int,
    kline_cutoff_seconds: int = _CUTOFF_SECONDS,
    candle_count: int = 16,
    base_close: float = 100.0,
) -> list[list]:
    """Build an exact-sized window with constant price (no signal).

    Mimics what ``OkxClient.kline_fetch_window`` returns under the
    cutoff-/lookback-aware live request:
        newest = lock_at - cutoff*1000 - 1000
        oldest = newest - (candle_count - 1) * 1000
    """
    newest = lock_at_ms - kline_cutoff_seconds * 1000 - 1000
    oldest = newest - (candle_count - 1) * 1000
    return [
        [oldest + i * 1000, base_close, base_close, base_close, base_close, 1.0]
        for i in range(candle_count)
    ]


def _trending_klines(
    *,
    lock_at_ms: int,
    kline_cutoff_seconds: int = _CUTOFF_SECONDS,
    candle_count: int = 16,
    slope: float = 0.0001,
) -> list[list]:
    """Build an exact-sized window with monotonic upward price drift to fire BTC Bull."""
    newest = lock_at_ms - kline_cutoff_seconds * 1000 - 1000
    oldest = newest - (candle_count - 1) * 1000
    base = 100.0
    return [
        [oldest + i * 1000,
         base + i * slope, base + i * slope, base + i * slope,
         base + i * slope, 1.0]
        for i in range(candle_count)
    ]


def _make_router(
    *,
    ok_klines: list[list],
    errors: dict[str, Exception] | None = None,
    rtt_ms: int = 100,
):
    """side_effect router for fake_client.kline_fetch_window. Symbols in
    ``errors`` map to a raise; everything else returns the new
    ``(rows, rtt_ms)`` tuple shape introduced 2026-04-27 alongside the
    true-RTT-timing fix.

    Accepts arbitrary kwargs so the router doesn't have to track every
    optional ``kline_fetch_window`` parameter (``send_before_bound`` etc.)."""
    errors = errors or {}

    def _fetch(*, symbol, **_kwargs):
        if symbol in errors:
            raise errors[symbol]
        return ok_klines, rtt_ms

    return _fetch


def _capture_warns():
    """Patch ``pancakebot.strategy.momentum_gate.warn`` and return
    (patcher_context, calls_list)."""
    calls: list[tuple] = []

    def _capture(*args, **kwargs):
        calls.append((args, kwargs))

    patcher = mock.patch(
        "pancakebot.strategy.momentum_gate.warn", side_effect=_capture,
    )
    return patcher, calls


# ---------------------------------------------------------------------------
# Healthy path
# ---------------------------------------------------------------------------

def test_evaluate_fetches_three_symbols_in_parallel():
    """All 3 symbols (BTC/ETH/SOL) hit OKX in a single round. BNB is
    currently disabled in MomentumGate._OKX_SYMBOLS_FETCHED."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
    )
    gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    called_symbols = sorted(
        c.kwargs["symbol"] for c in fake_client.kline_fetch_window.call_args_list
    )
    assert called_symbols == ["BTC-USDT", "ETH-USDT", "SOL-USDT"]


def test_evaluate_window_is_cutoff_and_lookback_aware():
    """Window math: newest = lock - cutoff*1000 - 1000, oldest = newest -
    max(lookbacks)*1000, count = max(lookbacks) + 1. Verifies the 2026-04-27
    fix that prevents requesting candles OKX hasn't published yet."""
    for cs in (1, 2, 5, 30):
        gate, fake_client = _make_gate(kline_cutoff_seconds=cs)
        fake_client.kline_fetch_window.side_effect = _make_router(
            ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS, kline_cutoff_seconds=cs),
        )
        gate.evaluate(lock_at_ms=_LOCK_AT_MS)
        expected_newest = _LOCK_AT_MS - cs * 1000 - 1000
        expected_oldest = expected_newest - 15 * 1000  # max((3,7,15)) = 15
        for c in fake_client.kline_fetch_window.call_args_list:
            assert c.kwargs["oldest_open_ms"] == expected_oldest, (
                f"cutoff={cs}: oldest got {c.kwargs['oldest_open_ms']}, "
                f"expected {expected_oldest}"
            )
            assert c.kwargs["newest_open_ms_inclusive"] == expected_newest, (
                f"cutoff={cs}: newest got {c.kwargs['newest_open_ms_inclusive']}, "
                f"expected {expected_newest}"
            )
            # Confirm the gate opts in to the both-bounds query so OKX
            # cannot slide the window.
            assert c.kwargs.get("send_before_bound") is True, (
                f"cutoff={cs}: gate must send_before_bound=True; "
                f"got {c.kwargs.get('send_before_bound')!r}"
            )


def test_evaluate_window_respects_non_default_lookbacks():
    """Non-default mtf_lookbacks=(5, 20, 60) → expected_count = 61, oldest
    bound = newest - 60_000. Confirms window math is keyed off the
    configured lookbacks rather than hardcoded constants."""
    lb = (5, 20, 60)
    cs = 2
    gate, fake_client = _make_gate(kline_cutoff_seconds=cs, mtf_lookbacks=lb)
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(
            lock_at_ms=_LOCK_AT_MS, kline_cutoff_seconds=cs, candle_count=61,
        ),
    )
    gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    expected_newest = _LOCK_AT_MS - cs * 1000 - 1000
    expected_oldest = expected_newest - 60 * 1000
    for c in fake_client.kline_fetch_window.call_args_list:
        assert c.kwargs["oldest_open_ms"] == expected_oldest, (
            f"non-default lookbacks: oldest got {c.kwargs['oldest_open_ms']}, "
            f"expected {expected_oldest}"
        )
        assert c.kwargs["newest_open_ms_inclusive"] == expected_newest, (
            f"non-default lookbacks: newest got "
            f"{c.kwargs['newest_open_ms_inclusive']}, expected {expected_newest}"
        )
    # candle_count derives from max(lookbacks)+1.
    assert gate._candle_count == 61


def test_evaluate_populates_last_fetch_timing_with_three_entries():
    """``last_fetch_timing`` has one entry per fetched symbol. With BNB
    disabled, that's btc/eth/sol — three entries."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
    )
    gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert gate.last_fetch_timing is not None
    assert set(gate.last_fetch_timing.keys()) == {"btc_ms", "eth_ms", "sol_ms"}


def test_evaluate_records_per_symbol_rtt_from_fetch_return():
    """``last_fetch_timing`` records the rtt_ms returned by each
    ``kline_fetch_window`` call (true per-symbol HTTP RTT), NOT
    wall-clock since submit-loop start. This guarantees rate-limiter
    wait does not bleed into the timing log."""
    gate, fake_client = _make_gate()
    # Router returns rtt_ms=42 for every symbol regardless of symbol or
    # window args. The gate's reap loop should record this verbatim.
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
        rtt_ms=42,
    )
    gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert gate.last_fetch_timing == {
        "btc_ms": 42, "eth_ms": 42, "sol_ms": 42,
    }, (
        f"last_fetch_timing must reflect per-symbol rtt_ms returned by "
        f"kline_fetch_window, not wall-clock-since-submit; "
        f"got {gate.last_fetch_timing}"
    )


def test_evaluate_returns_no_signal_on_flat_market():
    """Flat prices → multi-TF returns are all 0 → gate_no_signal."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
    )
    result = gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert result.signal is None
    assert result.skip_reason == "gate_no_signal"


def test_evaluate_returns_signal_on_trending_market():
    """Monotonic uptrend → multi-TF agree → Bull signal fires."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_trending_klines(lock_at_ms=_LOCK_AT_MS, slope=0.05),
    )
    result = gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert result.signal == "Bull", f"expected Bull, got {result.signal!r} skip={result.skip_reason!r}"
    assert result.tier == "multi_tf"
    assert result.skip_reason is None


# ---------------------------------------------------------------------------
# TransientOkxError handling
# ---------------------------------------------------------------------------

def test_evaluate_skips_with_kline_fetch_transient_failure_on_any_transient():
    """A single TransientOkxError on any symbol skips with the canonical
    ``kline_fetch_transient_failure`` reason."""
    for failing_symbol in ("BTC-USDT", "ETH-USDT", "SOL-USDT"):
        gate, fake_client = _make_gate()
        fake_client.kline_fetch_window.side_effect = _make_router(
            ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
            errors={failing_symbol: TransientOkxError(
                "kline_fetch_exhausted: simulated"
            )},
        )
        result = gate.evaluate(lock_at_ms=_LOCK_AT_MS)
        assert result.skip_reason == "kline_fetch_transient_failure", (
            f"{failing_symbol} transient: expected kline_fetch_transient_failure, "
            f"got {result.skip_reason!r}"
        )
        assert result.signal is None


def test_evaluate_emits_one_warn_per_failed_symbol_on_transient():
    """Per-symbol detail goes into one warn() per failure."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
        errors={
            "ETH-USDT": TransientOkxError("kline_fetch_exhausted: net_error"),
            "SOL-USDT": TransientOkxError("kline_fetch_exhausted: net_error"),
        },
    )
    patcher, warn_calls = _capture_warns()
    with patcher:
        gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert len(warn_calls) == 2, (
        f"expected 2 warns (ETH+SOL), got {len(warn_calls)}: {warn_calls}"
    )
    sub_tags = sorted(args[1] for args, _ in warn_calls)
    assert sub_tags == ["ETH", "SOL"]
    for args, kwargs in warn_calls:
        assert args[0] == "GATE"
        assert args[2] == "FETCH_FAIL"
        assert "kline_fetch_exhausted" in kwargs.get("msg", "")


def test_evaluate_all_three_transient_produces_three_warns_and_one_skip():
    """All-3-down round: 3 warns + 1 generic skip + streak += 1."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=[],
        errors={
            "BTC-USDT": TransientOkxError("kline_fetch_exhausted: net_down"),
            "ETH-USDT": TransientOkxError("kline_fetch_exhausted: net_down"),
            "SOL-USDT": TransientOkxError("kline_fetch_exhausted: net_down"),
        },
    )
    patcher, warn_calls = _capture_warns()
    with patcher:
        result = gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert result.skip_reason == "kline_fetch_transient_failure"
    assert len(warn_calls) == 3
    assert sorted(args[1] for args, _ in warn_calls) == ["BTC", "ETH", "SOL"]
    assert gate._consecutive_fetch_failures == 1


# ---------------------------------------------------------------------------
# Streak counter behavior
# ---------------------------------------------------------------------------

def test_streak_resets_on_successful_fetch_regardless_of_signal():
    """Streak counter resets on a clean 3-symbol fetch even when the
    downstream signal is gate_no_signal. The streak measures fetch
    health, not signal availability -- a quiet market shouldn't escalate."""
    gate, fake_client = _make_gate()
    # Round 1: all 3 transient.
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=[],
        errors={s: TransientOkxError("net_down") for s in
                ("BTC-USDT", "ETH-USDT", "SOL-USDT")},
    )
    patcher, _calls = _capture_warns()
    with patcher:
        gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert gate._consecutive_fetch_failures == 1

    # Round 2: clean fetch, but flat market → gate_no_signal.
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS + 300_000),
    )
    result = gate.evaluate(lock_at_ms=_LOCK_AT_MS + 300_000)
    assert result.skip_reason == "gate_no_signal"
    assert gate._consecutive_fetch_failures == 0, (
        "streak must reset to 0 after a clean fetch even when signal misses"
    )


def test_streak_escalates_to_invariant_error_at_three_consecutive_failures():
    """Three back-to-back transient rounds → InvariantError on the 3rd."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=[],
        errors={s: TransientOkxError("net_down") for s in
                ("BTC-USDT", "ETH-USDT", "SOL-USDT")},
    )
    patcher, _calls = _capture_warns()
    raised = None
    with patcher:
        # First _MAX_CONSECUTIVE_FETCH_FAILURES - 1 rounds skip; the Nth
        # fires InvariantError.
        for i in range(_MAX_CONSECUTIVE_FETCH_FAILURES - 1):
            result = gate.evaluate(lock_at_ms=_LOCK_AT_MS + i * 300_000)
            assert result.skip_reason == "kline_fetch_transient_failure"
        try:
            gate.evaluate(
                lock_at_ms=_LOCK_AT_MS + _MAX_CONSECUTIVE_FETCH_FAILURES * 300_000,
            )
        except InvariantError as e:
            raised = e
    assert raised is not None, (
        f"expected InvariantError after {_MAX_CONSECUTIVE_FETCH_FAILURES} consecutive transients"
    )
    assert "kline_fetch_failure_streak_max_reached" in str(raised)
    assert f"streak={_MAX_CONSECUTIVE_FETCH_FAILURES}" in str(raised)


def test_streak_does_not_escalate_when_a_clean_round_intervenes():
    """N-1 transient rounds, then a clean round, then N-1 more transient
    rounds → no escalation (streak was reset between clusters)."""
    gate, fake_client = _make_gate()
    patcher, _calls = _capture_warns()

    # First (N-1) transient rounds.
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=[],
        errors={s: TransientOkxError("net_down") for s in
                ("BTC-USDT", "ETH-USDT", "SOL-USDT")},
    )
    with patcher:
        for i in range(_MAX_CONSECUTIVE_FETCH_FAILURES - 1):
            gate.evaluate(lock_at_ms=_LOCK_AT_MS + i * 300_000)
    assert gate._consecutive_fetch_failures == _MAX_CONSECUTIVE_FETCH_FAILURES - 1

    # Clean round resets the streak.
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS + 9_000_000),
    )
    gate.evaluate(lock_at_ms=_LOCK_AT_MS + 9_000_000)
    assert gate._consecutive_fetch_failures == 0

    # Second (N-1) transient rounds -- still no escalation because reset.
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=[],
        errors={s: TransientOkxError("net_down") for s in
                ("BTC-USDT", "ETH-USDT", "SOL-USDT")},
    )
    with patcher:
        for i in range(_MAX_CONSECUTIVE_FETCH_FAILURES - 1):
            result = gate.evaluate(lock_at_ms=_LOCK_AT_MS + 10_000_000 + i * 300_000)
            assert result.skip_reason == "kline_fetch_transient_failure"
    # Should be at N-1, not escalated.
    assert gate._consecutive_fetch_failures == _MAX_CONSECUTIVE_FETCH_FAILURES - 1


# ---------------------------------------------------------------------------
# InvariantError propagation
# ---------------------------------------------------------------------------

def test_evaluate_reraises_invariant_error_from_any_symbol():
    """Any InvariantError from any symbol surfaces as InvariantError --
    the bot crashes. These indicate OKX returned a malformed window;
    silently skipping would mask shape violations."""
    for failing_symbol in ("BTC-USDT", "ETH-USDT", "SOL-USDT"):
        gate, fake_client = _make_gate()
        fake_client.kline_fetch_window.side_effect = _make_router(
            ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
            errors={failing_symbol: InvariantError(
                "kline_fetch_integrity_violation: simulated"
            )},
        )
        raised = None
        try:
            gate.evaluate(lock_at_ms=_LOCK_AT_MS)
        except InvariantError as e:
            raised = e
        assert raised is not None, (
            f"{failing_symbol} InvariantError must reraise (bot crashes)"
        )
        assert "kline_fetch_integrity_violation" in str(raised)
        assert failing_symbol[:3].lower() in str(raised), (
            f"raised message should name the failing symbol: {raised}"
        )


def test_evaluate_invariant_error_does_not_increment_streak():
    """InvariantError is fail-loud, not a transient skip -- it shouldn't
    contribute to the streak counter (the bot crashes anyway, but if
    something catches and retries the gate the counter shouldn't drift)."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
        errors={"BTC-USDT": InvariantError("kline_fetch_integrity_violation: x")},
    )
    try:
        gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    except InvariantError:
        pass
    assert gate._consecutive_fetch_failures == 0


# ---------------------------------------------------------------------------
# Disabled gate
# ---------------------------------------------------------------------------

def test_evaluate_returns_no_signal_when_disabled():
    """An ``enabled=False`` gate skips everything and never hits OKX."""
    cfg = MomentumGateConfig(
        enabled=False,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        kline_cutoff_seconds=2,
        mtf_lookbacks=(3, 7, 15),
        mtf_min_return_threshold=0.0001,
    )
    fake_client = mock.MagicMock()
    gate = MomentumGate(config=cfg, okx_client=fake_client)
    result = gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert result.signal is None
    assert result.skip_reason is None
    assert fake_client.kline_fetch_window.call_count == 0


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
