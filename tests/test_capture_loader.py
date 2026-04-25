"""Tests for backtest replay using captured klines.

Covers ``load_klines_from_capture`` (the loader that reshapes capture
records into the per-epoch kline dict the backtest pipeline expects)
and ``_merge_captured_with_history`` (capture + history fallback).

Run:
    python -m pytest tests/test_capture_loader.py -v
    python tests/test_capture_loader.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime.kline_capture import (  # noqa: E402
    CAPTURE_SCHEMA_VERSION,
    load_klines_from_capture,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def _ohlcv(ts_ms: int, c: float = 100.0) -> list:
    return [ts_ms, c, c, c, c, 0.0]


def test_load_klines_from_capture_btc():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        _write_jsonl(path, [
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "epoch": 100,
                "klines_btc": [_ohlcv(1000, 100.0), _ohlcv(2000, 100.5)],
                "klines_eth": [_ohlcv(1000, 50.0)],
                "klines_sol": None,
            },
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "epoch": 101,
                "klines_btc": [_ohlcv(3000, 101.0)],
                "klines_eth": None,
                "klines_sol": None,
            },
        ])
        btc = load_klines_from_capture(path, "btc")
        assert set(btc.keys()) == {100, 101}
        assert btc[100][0] == [1000, 100.0, 100.0, 100.0, 100.0, 0.0]
        assert btc[101][0] == [3000, 101.0, 101.0, 101.0, 101.0, 0.0]


def test_load_klines_from_capture_skips_none_assets():
    """ETH/SOL == None means the gate didn't fetch -- skip the epoch for that asset."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        _write_jsonl(path, [
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "epoch": 100,
                "klines_btc": [_ohlcv(1000)],
                "klines_eth": None,
                "klines_sol": None,
            },
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "epoch": 101,
                "klines_btc": [_ohlcv(2000)],
                "klines_eth": [_ohlcv(2000)],
                "klines_sol": None,
            },
        ])
        eth = load_klines_from_capture(path, "eth")
        assert list(eth.keys()) == [101]
        sol = load_klines_from_capture(path, "sol")
        assert sol == {}


def test_load_klines_from_capture_skips_unknown_schema():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        _write_jsonl(path, [
            {
                "schema_version": 999,  # future version this build doesn't know
                "epoch": 100,
                "klines_btc": [_ohlcv(1000)],
            },
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "epoch": 101,
                "klines_btc": [_ohlcv(2000)],
            },
        ])
        btc = load_klines_from_capture(path, "btc")
        assert list(btc.keys()) == [101], f"expected only epoch 101, got {list(btc.keys())}"


def test_load_klines_from_capture_invalid_asset_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        path.touch()
        try:
            load_klines_from_capture(path, "bnb")
        except ValueError as e:
            assert "btc/eth/sol" in str(e).lower()
            return
        raise AssertionError("expected ValueError for asset='bnb'")


def test_load_klines_from_capture_missing_file_returns_empty():
    result = load_klines_from_capture(Path("/nonexistent/cap.jsonl"), "btc")
    assert result == {}


def test_merge_captured_with_history_capture_wins_overlap():
    """Capture wins where both have an epoch (capture is the more recent fetch)."""
    from pancakebot.backtest.runner import _merge_captured_with_history

    with tempfile.TemporaryDirectory() as tmp:
        cap_path = Path(tmp) / "cap.jsonl"
        hist_path = Path(tmp) / "history.jsonl"
        # Capture has epochs 100, 101 with one set of values
        _write_jsonl(cap_path, [
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "epoch": 100,
                "klines_btc": [_ohlcv(1000, 100.0)],
                "klines_eth": None, "klines_sol": None,
            },
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "epoch": 101,
                "klines_btc": [_ohlcv(2000, 101.0)],
                "klines_eth": None, "klines_sol": None,
            },
        ])
        # History has epochs 99, 100, 102 with different values for 100
        with hist_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": 99, "klines_1s": [_ohlcv(500, 99.0)]}) + "\n")
            f.write(json.dumps({"epoch": 100, "klines_1s": [_ohlcv(1000, 999.99)]}) + "\n")
            f.write(json.dumps({"epoch": 102, "klines_1s": [_ohlcv(4000, 102.0)]}) + "\n")

        merged, n_cap, n_hist = _merge_captured_with_history(
            asset="btc", captured_path=cap_path, history_path=hist_path,
        )
        # Merged should have epochs 99, 100, 101, 102
        assert sorted(merged.keys()) == [99, 100, 101, 102]
        # 100 came from BOTH; capture must win (close=100.0, not 999.99)
        assert merged[100][0][4] == 100.0, f"capture should win on overlap, got {merged[100][0][4]}"
        # 99 and 102 came from history only
        assert merged[99][0][4] == 99.0
        assert merged[102][0][4] == 102.0
        # 101 came from capture only
        assert merged[101][0][4] == 101.0
        # n_cap = epochs from capture, n_hist = epochs from history NOT overridden by capture
        assert n_cap == 2
        # n_hist = total - captured = 4 - 2 = 2
        assert n_hist == 2


def test_merge_captured_with_history_no_capture_falls_back():
    """When capture is empty, all epochs come from history."""
    from pancakebot.backtest.runner import _merge_captured_with_history

    with tempfile.TemporaryDirectory() as tmp:
        cap_path = Path(tmp) / "cap.jsonl"
        cap_path.touch()  # exists but empty
        hist_path = Path(tmp) / "history.jsonl"
        with hist_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": 100, "klines_1s": [_ohlcv(1000)]}) + "\n")
            f.write(json.dumps({"epoch": 101, "klines_1s": [_ohlcv(2000)]}) + "\n")

        merged, n_cap, n_hist = _merge_captured_with_history(
            asset="btc", captured_path=cap_path, history_path=hist_path,
        )
        assert sorted(merged.keys()) == [100, 101]
        assert n_cap == 0
        assert n_hist == 2


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
