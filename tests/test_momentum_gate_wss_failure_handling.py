"""WSS-failure handling tests for ``MomentumGate.evaluate`` (Phase 2 spec
item 16, 2026-04-27).

Architecture:

  - Single generic skip reason ``risk_kline_wss_failure`` for ANY WSS
    failure across BTC/ETH/SOL/BNB. Histogram cardinality stays low.
  - One ``warn()`` per failed symbol naming the specific reason. Operator
    reconciles the generic skip with the warn-log cluster around it.
  - All four ``get_window`` calls fire upfront so per-call side effects
    (notably ``needs_reconnect`` propagation from item 13) never get
    short-circuited by an earlier failure.

This file supersedes the prior per-reason mapping test
(``test_momentum_gate_wss_skip_mapping.py``) and the per-symbol BNB-suffix
test (``test_momentum_gate_bnb_parity.py``); both were deleted when item
16 collapsed the skip-reason surface to a single variant.
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
    _WSS_FAILURE_SKIP_REASON,
)


_GENERIC_SKIP = "risk_kline_wss_failure"
assert _WSS_FAILURE_SKIP_REASON == _GENERIC_SKIP, (
    "module constant drifted from the test's expectation"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gate_with_mocked_wss():
    cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        mtf_lookbacks=(3, 7, 15),
        mtf_threshold=0.0001,
    )
    fake_okx = mock.MagicMock()
    fake_wss = mock.MagicMock()
    gate = MomentumGate(config=cfg, okx_client=fake_okx, wss_client=fake_wss)
    return gate, fake_wss, cfg


def _healthy_klines(cutoff_ts_ms: int, candle_count: int) -> list[list]:
    """Build a candle_count-element kline list ending at cutoff - 1000 with
    a benign monotonic price (no signal fires)."""
    base_ts = cutoff_ts_ms - candle_count * 1000
    return [
        [base_ts + i * 1000, 100.0, 100.0, 100.0, 100.0, 1.0]
        for i in range(candle_count)
    ]


def _make_router(*, healthy_klines, fail_map: dict[str, str]):
    """Build a get_window side_effect router. ``fail_map`` maps symbol ->
    skip_reason for symbols that should fail; everything else returns
    healthy_klines + None."""
    def _router(symbol, cutoff_ms, expected_count):
        if symbol in fail_map:
            return (None, fail_map[symbol])
        return (healthy_klines, None)
    return _router


def _capture_warns():
    """Patch ``pancakebot.strategy.momentum_gate.warn`` and return
    (patcher_context, calls_list)."""
    calls: list[tuple] = []

    def _capture(*args, **kwargs):
        calls.append((args, kwargs))

    patcher = mock.patch(
        "pancakebot.strategy.momentum_gate.warn", side_effect=_capture
    )
    return patcher, calls


# ---------------------------------------------------------------------------
# Generic skip reason
# ---------------------------------------------------------------------------

def test_evaluate_emits_generic_skip_when_any_symbol_fails():
    """Any single-symbol WSS failure -> the gate skips with the SINGLE
    generic ``risk_kline_wss_failure`` reason. No per-symbol or per-reason
    variants in the histogram."""
    for sym in ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"):
        gate, fake_wss, _cfg = _make_gate_with_mocked_wss()
        cutoff_ts_ms = 2_000_000_000_000
        klines = _healthy_klines(cutoff_ts_ms, gate._candle_count)
        fake_wss.get_window.side_effect = _make_router(
            healthy_klines=klines,
            fail_map={sym: "wss_newest_lagging"},
        )
        result = gate.evaluate(cutoff_ts_ms=cutoff_ts_ms)
        assert result.skip_reason == _GENERIC_SKIP, (
            f"{sym} failure -- expected generic skip, got {result.skip_reason!r}"
        )


def test_evaluate_generic_skip_independent_of_specific_reason():
    """All five known WSS skip reasons collapse to the same generic skip.
    Per-reason histogram differentiation lives in warn logs, NOT skip
    reasons."""
    for reason in (
        "wss_bootstrap_pending",
        "wss_gap_fill_in_progress",
        "wss_insufficient",
        "wss_newest_lagging",
        "wss_unknown_symbol",
    ):
        gate, fake_wss, _cfg = _make_gate_with_mocked_wss()
        cutoff_ts_ms = 2_000_000_000_000
        klines = _healthy_klines(cutoff_ts_ms, gate._candle_count)
        fake_wss.get_window.side_effect = _make_router(
            healthy_klines=klines,
            fail_map={"BTC-USDT": reason},
        )
        result = gate.evaluate(cutoff_ts_ms=cutoff_ts_ms)
        assert result.skip_reason == _GENERIC_SKIP, (
            f"reason={reason} -- expected generic skip, got {result.skip_reason!r}"
        )


# ---------------------------------------------------------------------------
# Per-failure warn log
# ---------------------------------------------------------------------------

def test_evaluate_emits_warn_log_per_failed_symbol_with_specific_reason():
    """One ``warn()`` per failed symbol. Subsystem = WSS_GATE,
    sub = symbol-uppercase, event = FAIL, msg includes the specific reason."""
    gate, fake_wss, _cfg = _make_gate_with_mocked_wss()
    cutoff_ts_ms = 2_000_000_000_000
    klines = _healthy_klines(cutoff_ts_ms, gate._candle_count)
    fake_wss.get_window.side_effect = _make_router(
        healthy_klines=klines,
        fail_map={"ETH-USDT": "wss_newest_lagging"},
    )
    patcher, warn_calls = _capture_warns()
    with patcher:
        gate.evaluate(cutoff_ts_ms=cutoff_ts_ms)
    assert len(warn_calls) == 1, (
        f"expected exactly 1 warn for ETH failure, got {len(warn_calls)}: {warn_calls}"
    )
    args, kwargs = warn_calls[0]
    assert args[0] == "WSS_GATE"
    assert args[1] == "ETH"
    assert args[2] == "FAIL"
    assert "wss_newest_lagging" in kwargs.get("msg", ""), kwargs


def test_evaluate_warn_log_does_not_use_skip_tag():
    """Regression guard: the per-symbol warn MUST NOT use the 'SKIP' event
    tag. The generic skip reason itself already conveys the round was
    skipped; the warn is per-symbol detail, NOT a skip event."""
    gate, fake_wss, _cfg = _make_gate_with_mocked_wss()
    cutoff_ts_ms = 2_000_000_000_000
    klines = _healthy_klines(cutoff_ts_ms, gate._candle_count)
    fake_wss.get_window.side_effect = _make_router(
        healthy_klines=klines,
        fail_map={"BTC-USDT": "wss_newest_lagging"},
    )
    patcher, warn_calls = _capture_warns()
    with patcher:
        gate.evaluate(cutoff_ts_ms=cutoff_ts_ms)
    for args, kwargs in warn_calls:
        assert "SKIP" not in args, (
            f"warn() must not use SKIP tag; got args={args}"
        )


def test_evaluate_no_warns_on_healthy_path():
    """All 4 symbols healthy -> no warn logs emitted (no failure to triage)."""
    gate, fake_wss, _cfg = _make_gate_with_mocked_wss()
    cutoff_ts_ms = 2_000_000_000_000
    klines = _healthy_klines(cutoff_ts_ms, gate._candle_count)
    fake_wss.get_window.side_effect = _make_router(
        healthy_klines=klines, fail_map={},
    )
    patcher, warn_calls = _capture_warns()
    with patcher:
        result = gate.evaluate(cutoff_ts_ms=cutoff_ts_ms)
    assert warn_calls == [], (
        f"healthy path must not emit warns; got {warn_calls}"
    )
    assert result.skip_reason != _GENERIC_SKIP, result.skip_reason


# ---------------------------------------------------------------------------
# All-4-fail multi-warn cluster
# ---------------------------------------------------------------------------

def test_evaluate_all_four_failures_produces_four_warns_and_one_skip():
    """When all 4 symbols fail (e.g. WSS connection dead), the round
    produces 4 warn logs (one per symbol) clustered around 1 generic skip.
    This is the ALL_DEAD signal the operator visually scans for."""
    gate, fake_wss, _cfg = _make_gate_with_mocked_wss()
    cutoff_ts_ms = 2_000_000_000_000
    fake_wss.get_window.side_effect = _make_router(
        healthy_klines=[],
        fail_map={
            "BTC-USDT": "wss_newest_lagging",
            "ETH-USDT": "wss_newest_lagging",
            "SOL-USDT": "wss_newest_lagging",
            "BNB-USDT": "wss_newest_lagging",
        },
    )
    patcher, warn_calls = _capture_warns()
    with patcher:
        result = gate.evaluate(cutoff_ts_ms=cutoff_ts_ms)
    assert result.skip_reason == _GENERIC_SKIP
    assert len(warn_calls) == 4, (
        f"expected 4 warns (1 per failed symbol); got {len(warn_calls)}: {warn_calls}"
    )
    # All 4 symbol tags present (BTC, ETH, SOL, BNB).
    sub_tags = sorted(args[1] for args, _ in warn_calls)
    assert sub_tags == ["BNB", "BTC", "ETH", "SOL"]


def test_evaluate_partial_failures_produce_matching_warn_count():
    """ETH + BNB fail; BTC + SOL healthy -> exactly 2 warns (one per
    failed symbol) + 1 generic skip. No noise for healthy symbols."""
    gate, fake_wss, _cfg = _make_gate_with_mocked_wss()
    cutoff_ts_ms = 2_000_000_000_000
    klines = _healthy_klines(cutoff_ts_ms, gate._candle_count)
    fake_wss.get_window.side_effect = _make_router(
        healthy_klines=klines,
        fail_map={
            "ETH-USDT": "wss_gap_fill_in_progress",
            "BNB-USDT": "wss_bootstrap_pending",
        },
    )
    patcher, warn_calls = _capture_warns()
    with patcher:
        result = gate.evaluate(cutoff_ts_ms=cutoff_ts_ms)
    assert result.skip_reason == _GENERIC_SKIP
    assert len(warn_calls) == 2
    sub_tags = sorted(args[1] for args, _ in warn_calls)
    assert sub_tags == ["BNB", "ETH"]
    # Reasons get into the msg per-symbol.
    msgs_by_sub = {args[1]: kwargs.get("msg", "") for args, kwargs in warn_calls}
    assert "wss_gap_fill_in_progress" in msgs_by_sub["ETH"]
    assert "wss_bootstrap_pending" in msgs_by_sub["BNB"]


# ---------------------------------------------------------------------------
# All-four-calls invariant (needs_reconnect side effect must propagate)
# ---------------------------------------------------------------------------

def test_evaluate_calls_get_window_for_all_four_symbols_on_healthy_path():
    """All 4 calls fire on the healthy path."""
    gate, fake_wss, cfg = _make_gate_with_mocked_wss()
    cutoff_ts_ms = 2_000_000_000_000
    klines = _healthy_klines(cutoff_ts_ms, gate._candle_count)
    fake_wss.get_window.side_effect = _make_router(
        healthy_klines=klines, fail_map={},
    )
    gate.evaluate(cutoff_ts_ms=cutoff_ts_ms)
    called_symbols = sorted(
        call.args[0] if call.args else call.kwargs.get("symbol")
        for call in fake_wss.get_window.call_args_list
    )
    assert called_symbols == sorted([
        cfg.btc_symbol, cfg.eth_symbol, cfg.sol_symbol, cfg.bnb_symbol,
    ])


def test_evaluate_calls_get_window_for_all_four_symbols_even_on_first_failure():
    """Even when BTC fails (the canonical "first" symbol checked), every
    other symbol's get_window MUST also have been invoked. This is the
    non-short-circuit invariant that lets ``needs_reconnect`` propagate
    from any silently-dead symbol regardless of which one happens to be
    listed first."""
    gate, fake_wss, cfg = _make_gate_with_mocked_wss()
    cutoff_ts_ms = 2_000_000_000_000
    klines = _healthy_klines(cutoff_ts_ms, gate._candle_count)
    # BTC fails; every other symbol healthy.
    fake_wss.get_window.side_effect = _make_router(
        healthy_klines=klines,
        fail_map={"BTC-USDT": "wss_newest_lagging"},
    )
    gate.evaluate(cutoff_ts_ms=cutoff_ts_ms)
    called_symbols = sorted(
        call.args[0] if call.args else call.kwargs.get("symbol")
        for call in fake_wss.get_window.call_args_list
    )
    assert called_symbols == sorted([
        cfg.btc_symbol, cfg.eth_symbol, cfg.sol_symbol, cfg.bnb_symbol,
    ]), (
        "all 4 get_window calls MUST fire even when an earlier one returned None; "
        "needs_reconnect side effect must propagate from every symbol"
    )


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
