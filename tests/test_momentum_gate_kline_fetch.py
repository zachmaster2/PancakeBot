"""Tests for ``MomentumGate.evaluate`` per-round REST fetch path.

Covers the post-WSS-revert data plane (2026-04-27): the gate fires 4 parallel
``OkxClient.kline_fetch_window`` calls (BTC/ETH/SOL/BNB) anchored at
``lock_at_ms`` and aggregates results by exception class.

Behavior under test:
- Healthy path: all 4 symbols fetched, signal computed off the trimmed window.
- ``InvariantError`` from any symbol → reraised (bot crashes).
- ``TransientOkxError`` (any subset) → per-symbol warn + skip with
  ``kline_fetch_transient_failure``, increments streak counter.
- 3 consecutive transient rounds → escalates to ``InvariantError``.
- Streak resets to 0 on any successful 4-symbol fetch (regardless of signal).
- Fetch window is ``[lock_at_ms - 301_000, lock_at_ms - 2_000]`` for every
  symbol, independent of ``cutoff_seconds``.
- ``last_fetch_timing`` populated with 4 entries (btc/eth/sol/bnb).

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


def _make_gate(*, cutoff_seconds: int = _CUTOFF_SECONDS, mtf_lookbacks=(3, 7, 15)):
    """Build a gate whose okx client is a MagicMock. Returns (gate, fake_client)."""
    cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        cutoff_seconds=cutoff_seconds,
        mtf_lookbacks=mtf_lookbacks,
        mtf_threshold=0.0001,
    )
    fake_client = mock.MagicMock()
    gate = MomentumGate(config=cfg, okx_client=fake_client)
    return gate, fake_client


def _flat_klines(*, lock_at_ms: int, count: int = 300, base_close: float = 100.0) -> list[list]:
    """Build a 300-candle window with constant price (no signal)."""
    oldest = lock_at_ms - 301_000
    return [
        [oldest + i * 1000, base_close, base_close, base_close, base_close, 1.0]
        for i in range(count)
    ]


def _trending_klines(*, lock_at_ms: int, slope: float = 0.0001) -> list[list]:
    """Build a 300-candle window with monotonic upward price drift to fire BTC Bull."""
    oldest = lock_at_ms - 301_000
    base = 100.0
    return [
        [oldest + i * 1000,
         base + i * slope, base + i * slope, base + i * slope,
         base + i * slope, 1.0]
        for i in range(300)
    ]


def _make_router(*, ok_klines: list[list], errors: dict[str, Exception] | None = None):
    """side_effect router for fake_client.kline_fetch_window. Symbols in
    ``errors`` map to a raise; everything else returns ``ok_klines``."""
    errors = errors or {}

    def _fetch(*, symbol, oldest_open_ms, newest_open_ms_inclusive,
               retry_policy, rate_acquire_fn=None):
        if symbol in errors:
            raise errors[symbol]
        return ok_klines

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

def test_evaluate_fetches_all_four_symbols_in_parallel():
    """All 4 symbols (BTC/ETH/SOL/BNB) hit OKX in a single round."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
    )
    gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    called_symbols = sorted(
        c.kwargs["symbol"] for c in fake_client.kline_fetch_window.call_args_list
    )
    assert called_symbols == ["BNB-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT"]


def test_evaluate_uses_lock_at_anchored_window_independent_of_cutoff():
    """Window is ``[lock_at-301s, lock_at-2s]`` regardless of cutoff_seconds.
    The 300-candle window matches sync.py and the on-disk rebuild."""
    for cs in (1, 2, 5, 30):
        gate, fake_client = _make_gate(cutoff_seconds=cs)
        fake_client.kline_fetch_window.side_effect = _make_router(
            ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
        )
        gate.evaluate(lock_at_ms=_LOCK_AT_MS)
        # Every call should have the same window regardless of cutoff_seconds.
        for c in fake_client.kline_fetch_window.call_args_list:
            assert c.kwargs["oldest_open_ms"] == _LOCK_AT_MS - 301_000, (
                f"cutoff_seconds={cs}: oldest must be lock-301s"
            )
            assert c.kwargs["newest_open_ms_inclusive"] == _LOCK_AT_MS - 2_000, (
                f"cutoff_seconds={cs}: newest must be lock-2s"
            )


def test_evaluate_populates_last_fetch_timing_with_four_entries():
    """``last_fetch_timing`` has bnb_ms entry post-revert (matching the
    actual fetched symbols, not the 3-entry WSS-era shape)."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
    )
    gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert gate.last_fetch_timing is not None
    assert set(gate.last_fetch_timing.keys()) == {"btc_ms", "eth_ms", "sol_ms", "bnb_ms"}


def test_evaluate_populates_capture_fields_on_healthy_path():
    """``last_*_klines_raw`` and ``last_returns`` populated for capture."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=_flat_klines(lock_at_ms=_LOCK_AT_MS),
    )
    gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    # Each capture field is a candle_count-length list of dicts.
    cc = gate._candle_count
    assert gate.last_btc_klines_raw is not None
    assert len(gate.last_btc_klines_raw) == cc
    assert gate.last_eth_klines_raw is not None
    assert len(gate.last_eth_klines_raw) == cc
    assert gate.last_sol_klines_raw is not None
    assert len(gate.last_sol_klines_raw) == cc
    assert gate.last_returns is not None
    # returns dict has btc/eth/sol × each lookback.
    assert "btc_r3" in gate.last_returns
    assert "btc_r7" in gate.last_returns
    assert "btc_r15" in gate.last_returns


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
    for failing_symbol in ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"):
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
            "BNB-USDT": TransientOkxError("kline_fetch_exhausted: net_error"),
        },
    )
    patcher, warn_calls = _capture_warns()
    with patcher:
        gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert len(warn_calls) == 2, (
        f"expected 2 warns (ETH+BNB), got {len(warn_calls)}: {warn_calls}"
    )
    sub_tags = sorted(args[1] for args, _ in warn_calls)
    assert sub_tags == ["BNB", "ETH"]
    for args, kwargs in warn_calls:
        assert args[0] == "GATE"
        assert args[2] == "FETCH_FAIL"
        assert "kline_fetch_exhausted" in kwargs.get("msg", "")


def test_evaluate_all_four_transient_produces_four_warns_and_one_skip():
    """All-4-down round: 4 warns + 1 generic skip + streak += 1."""
    gate, fake_client = _make_gate()
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=[],
        errors={
            "BTC-USDT": TransientOkxError("kline_fetch_exhausted: net_down"),
            "ETH-USDT": TransientOkxError("kline_fetch_exhausted: net_down"),
            "SOL-USDT": TransientOkxError("kline_fetch_exhausted: net_down"),
            "BNB-USDT": TransientOkxError("kline_fetch_exhausted: net_down"),
        },
    )
    patcher, warn_calls = _capture_warns()
    with patcher:
        result = gate.evaluate(lock_at_ms=_LOCK_AT_MS)
    assert result.skip_reason == "kline_fetch_transient_failure"
    assert len(warn_calls) == 4
    assert sorted(args[1] for args, _ in warn_calls) == ["BNB", "BTC", "ETH", "SOL"]
    assert gate._consecutive_fetch_failures == 1


# ---------------------------------------------------------------------------
# Streak counter behavior
# ---------------------------------------------------------------------------

def test_streak_resets_on_successful_fetch_regardless_of_signal():
    """Streak counter resets on a clean 4-symbol fetch even when the
    downstream signal is gate_no_signal. The streak measures fetch
    health, not signal availability -- a quiet market shouldn't escalate."""
    gate, fake_client = _make_gate()
    # Round 1: all 4 transient.
    fake_client.kline_fetch_window.side_effect = _make_router(
        ok_klines=[],
        errors={s: TransientOkxError("net_down") for s in
                ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT")},
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
                ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT")},
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
                ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT")},
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
                ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT")},
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
    for failing_symbol in ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"):
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
        cutoff_seconds=2,
        mtf_lookbacks=(3, 7, 15),
        mtf_threshold=0.0001,
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
