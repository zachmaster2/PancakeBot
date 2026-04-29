"""Tests for the --use-extended-data path: extended-first loader fallback,
partial-data skip behavior, default-OFF canonical preservation.

The extended-data feature reads ``var/extended/<symbol>.jsonl`` (older
epochs not in canonical) plus the canonical ``var/<symbol>.jsonl``. Records
in the extended store may carry a ``data_status`` field of ``OK_FULL``,
``OK_PARTIAL``, ``MISSING``, ``MISSING_VERIFIED``, etc. — partial/missing
data is loaded as-is and the strategy's existing
``_validate_klines_raw`` check naturally skips rounds via
``gate_<sym>_insufficient`` skip reasons.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.in_process_runner import (  # noqa: E402
    _load_klines_unified,
    _load_all_rounds,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_kline_record(epoch: int, lock_at: int, n_candles: int = 300,
                       data_status: str | None = None) -> dict:
    """Build a synthetic kline record with `n_candles` 1s candles."""
    klines: list[list] = []
    # Mimic the canonical window: open_ts in [lock_at - 301, lock_at - 2].
    # When n_candles < 300 we just write n_candles candles ending at lock_at - 2.
    last_open_s = lock_at - 2
    for i in range(n_candles):
        ts_s = last_open_s - (n_candles - 1 - i)
        klines.append([ts_s * 1000, 100.0, 100.0, 100.0, 100.0, 0.001])
    rec: dict = {"epoch": epoch, "lock_at": lock_at, "klines_1s": klines}
    if data_status is not None:
        rec["data_status"] = data_status
    return rec


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------------------
# Loader tests
# ----------------------------------------------------------------------------


def test_load_klines_unified_canonical_only(tmp_path: Path) -> None:
    """Loader returns canonical records when extended_path is None."""
    canon = tmp_path / "btc.jsonl"
    _write_jsonl(canon, [
        _make_kline_record(epoch=437562, lock_at=1765444970),
        _make_kline_record(epoch=437563, lock_at=1765445270),
    ])

    result = _load_klines_unified(canon, earliest_offset=18, latest_offset=3)
    assert set(result.keys()) == {437562, 437563}
    # Window [3..18] from end should yield exactly 16 candles
    for ep, kl in result.items():
        assert len(kl) == 16


def test_load_klines_unified_extended_extends_canonical(tmp_path: Path) -> None:
    """Extended records appear in the loader output when extended_path is provided."""
    canon = tmp_path / "btc.jsonl"
    ext = tmp_path / "extended" / "btc.jsonl"
    _write_jsonl(canon, [
        _make_kline_record(epoch=437562, lock_at=1765444970),
        _make_kline_record(epoch=437563, lock_at=1765445270),
    ])
    _write_jsonl(ext, [
        _make_kline_record(epoch=430000, lock_at=1765000000, data_status="OK_FULL"),
        _make_kline_record(epoch=430001, lock_at=1765000300, data_status="OK_FULL"),
    ])

    result = _load_klines_unified(
        canon, earliest_offset=18, latest_offset=3, extended_path=ext,
    )
    assert set(result.keys()) == {430000, 430001, 437562, 437563}


def test_load_klines_unified_canonical_wins_on_collision(tmp_path: Path) -> None:
    """When the same epoch appears in both files, canonical wins.

    No overlap is expected by construction (extended is older-only), but the
    loader's canonical-first ingestion guards against any future drift.
    """
    canon = tmp_path / "btc.jsonl"
    ext = tmp_path / "extended" / "btc.jsonl"

    # Canonical record has 300 candles (full).
    canon_rec = _make_kline_record(epoch=437562, lock_at=1765444970, n_candles=300)
    # Extended record has 0 candles (would normally indicate MISSING).
    ext_rec = _make_kline_record(epoch=437562, lock_at=1765444970, n_candles=0,
                                  data_status="MISSING")
    _write_jsonl(canon, [canon_rec])
    _write_jsonl(ext, [ext_rec])

    result = _load_klines_unified(
        canon, earliest_offset=18, latest_offset=3, extended_path=ext,
    )
    assert 437562 in result
    # Should be canonical's window (16 candles), not extended's empty list.
    assert len(result[437562]) == 16


def test_load_klines_unified_handles_empty_extended(tmp_path: Path) -> None:
    """Records with empty klines_1s in the extended file are still indexed,
    yielding empty arrays. The strategy's _validate_klines_raw will catch this."""
    canon = tmp_path / "btc.jsonl"
    ext = tmp_path / "extended" / "btc.jsonl"
    _write_jsonl(canon, [_make_kline_record(epoch=437562, lock_at=1765444970)])
    # MISSING record: data_status set, klines_1s = []
    _write_jsonl(ext, [
        {"epoch": 430000, "lock_at": 1765000000, "klines_1s": [],
         "data_status": "MISSING_VERIFIED", "detail": "5x_retry_confirmed"},
    ])

    result = _load_klines_unified(
        canon, earliest_offset=18, latest_offset=3, extended_path=ext,
    )
    assert set(result.keys()) == {430000, 437562}
    assert result[430000] == []  # empty slice of empty list
    assert len(result[437562]) == 16


def test_load_klines_unified_skips_klines_1s_none(tmp_path: Path) -> None:
    """Records with klines_1s field explicitly set to None are skipped."""
    canon = tmp_path / "btc.jsonl"
    _write_jsonl(canon, [
        _make_kline_record(epoch=437562, lock_at=1765444970),
        {"epoch": 437563, "lock_at": 1765445270, "klines_1s": None,
         "error": "fetch_failed"},
    ])

    result = _load_klines_unified(canon, earliest_offset=18, latest_offset=3)
    assert set(result.keys()) == {437562}


def test_load_klines_unified_no_files_returns_empty(tmp_path: Path) -> None:
    """Both paths missing -> empty dict."""
    canon = tmp_path / "missing.jsonl"
    ext = tmp_path / "also_missing.jsonl"
    result = _load_klines_unified(canon, earliest_offset=18, latest_offset=3,
                                   extended_path=ext)
    assert result == {}


# ----------------------------------------------------------------------------
# CLI flag validation
# ----------------------------------------------------------------------------


def test_use_extended_data_only_with_backtest() -> None:
    """run.py rejects --use-extended-data without --backtest."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "run.py"),
         "--dry", "--use-extended-data"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "use-extended-data" in (result.stderr + result.stdout).lower()


# ----------------------------------------------------------------------------
# Partial-data semantics for the strategy gate
# ----------------------------------------------------------------------------


def test_validate_klines_raw_flags_partial_as_insufficient() -> None:
    """Strategy validation (`_validate_klines_raw`) returns
    ``gate_<sym>_insufficient`` for arrays shorter than candle_count, the
    natural skip path for extended PARTIAL/MISSING records.
    """
    from pancakebot.strategy.momentum_gate import _validate_klines_raw

    cutoff_ms = 1765444968 * 1000  # cutoff = lock_at - 2
    candle_count = 16

    # Empty list (MISSING)
    reason = _validate_klines_raw([], cutoff_ms, "btc", candle_count=candle_count)
    assert reason is not None
    assert "insufficient" in reason
    assert "got=0" in reason

    # Short list (PARTIAL, e.g. 14 candles)
    short_kl = [[(cutoff_ms - i * 1000), 100.0, 100.0, 100.0, 100.0, 0.0]
                for i in range(13, -1, -1)]  # 14 candles
    reason = _validate_klines_raw(short_kl, cutoff_ms, "btc", candle_count=candle_count)
    assert reason is not None
    assert "insufficient" in reason


def test_validate_klines_raw_passes_full_window() -> None:
    """Full 16-candle aligned window passes validation cleanly."""
    from pancakebot.strategy.momentum_gate import _validate_klines_raw

    cutoff_ms = 1765444968 * 1000
    candle_count = 16
    # Build 16 candles ending exactly at cutoff_ms - 1000
    last_ts = cutoff_ms - 1000
    kl = [
        [last_ts - (15 - i) * 1000, 100.0, 100.0, 100.0, 100.0, 0.0]
        for i in range(16)
    ]
    reason = _validate_klines_raw(kl, cutoff_ms, "btc", candle_count=candle_count)
    assert reason is None
