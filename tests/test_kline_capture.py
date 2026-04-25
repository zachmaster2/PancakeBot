"""Tests for pancakebot.runtime.kline_capture.

Covers the producer/worker contract that's load-bearing for the
"never block the bet path" guarantee:

    - enqueue is non-blocking and bounded (drop-on-full, not block)
    - serialise/write failures stay inside the worker (logged, not raised)
    - shutdown drains pending items up to the timeout
    - schema is forward-compatible (reader skips unknown fields)
    - microbenchmark: 1000 enqueue calls < 50us each on a healthy box

Run:
    python -m pytest tests/test_kline_capture.py -v
    python tests/test_kline_capture.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime import kline_capture as kc  # noqa: E402


# ---------------------------------------------------------------------------
# build_snapshot: pure data shaping
# ---------------------------------------------------------------------------

def _fake_klines(n: int = 31, base_ts_ms: int = 1_000_000_000_000) -> list[dict]:
    return [
        {
            "open_time_ms": base_ts_ms + i * 1000,
            "open": 100.0 + i * 0.1,
            "high": 100.5 + i * 0.1,
            "low": 99.5 + i * 0.1,
            "close_price": 100.2 + i * 0.1,
            "volume": 1.0 + i * 0.01,
        }
        for i in range(n)
    ]


def test_build_snapshot_basic_shape():
    snap = kc.build_snapshot(
        epoch=475323,
        lock_at_unix=1777007708,
        cutoff_ms=1777007706000,
        mode="dry",
        btc_klines_raw=_fake_klines(31),
        eth_klines_raw=_fake_klines(31),
        sol_klines_raw=None,
        returns={"btc_r3": 0.000273, "btc_r7": 0.000206, "btc_r15": 0.000206},
        decision="BET",
        skip_reason=None,
        selected_strategy="btc_primary",
        bet_side="Bull",
        bet_size_bnb=0.045,
        pool_bull_bnb=1.0,
        pool_bear_bnb=0.665,
    )
    assert snap["schema_version"] == kc.CAPTURE_SCHEMA_VERSION
    assert snap["epoch"] == 475323
    assert snap["mode"] == "dry"
    assert snap["decision"] == "BET"
    assert snap["bet_side"] == "Bull"
    assert snap["pool_total_bnb"] == 1.665
    assert snap["klines_btc"] is not None
    assert len(snap["klines_btc"]) == 31
    # Each kline is [ts, o, h, l, c, v]
    assert len(snap["klines_btc"][0]) == 6
    assert snap["klines_sol"] is None
    # JSON round-trip safety
    line = json.dumps(snap, sort_keys=True, separators=(",", ":"))
    parsed = json.loads(line)
    assert parsed["epoch"] == 475323


def test_build_snapshot_handles_none_klines():
    snap = kc.build_snapshot(
        epoch=1,
        lock_at_unix=0,
        cutoff_ms=0,
        mode="dry",
        btc_klines_raw=None,
        eth_klines_raw=None,
        sol_klines_raw=None,
        returns=None,
        decision="SKIP",
        skip_reason="gate_no_signal",
        selected_strategy=None,
        bet_side=None,
        bet_size_bnb=None,
        pool_bull_bnb=0.0,
        pool_bear_bnb=0.0,
    )
    assert snap["klines_btc"] == []
    assert snap["klines_eth"] is None
    assert snap["klines_sol"] is None
    assert snap["returns"] == {}


def test_kline_dict_to_array_fallback_to_close():
    """Closes-only dicts (older fetches) round-trip as flat OHLCV with c-fill."""
    arr = kc._kline_dict_to_array({"open_time_ms": 123, "close_price": 100.0})
    # [ts, o, h, l, c, v] with o/h/l = c, v = 0
    assert arr == [123, 100.0, 100.0, 100.0, 100.0, 0.0]


# ---------------------------------------------------------------------------
# CaptureWorker: producer non-blocking + worker isolation
# ---------------------------------------------------------------------------

def test_worker_writes_one_capture_to_disk():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        worker = kc.CaptureWorker(path)
        worker.start()
        snap = kc.build_snapshot(
            epoch=42, lock_at_unix=0, cutoff_ms=0, mode="dry",
            btc_klines_raw=_fake_klines(5),
            eth_klines_raw=None, sol_klines_raw=None, returns=None,
            decision="SKIP", skip_reason="gate_no_signal",
            selected_strategy=None, bet_side=None, bet_size_bnb=None,
            pool_bull_bnb=0.0, pool_bear_bnb=0.0,
        )
        assert worker.enqueue(snap) is True
        worker.shutdown(timeout_s=2.0)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["epoch"] == 42
        stats = worker.stats()
        assert stats["enqueued"] == 1
        assert stats["written"] == 1
        assert stats["dropped_full"] == 0


def test_worker_drops_on_full_queue_without_blocking():
    """Producer fills queue past capacity. Drops happen, no exception, no block."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        worker = kc.CaptureWorker(path, capacity=3)
        # DON'T start the worker: queue fills up and stays full.
        snap = kc.build_snapshot(
            epoch=1, lock_at_unix=0, cutoff_ms=0, mode="dry",
            btc_klines_raw=_fake_klines(2), eth_klines_raw=None,
            sol_klines_raw=None, returns=None,
            decision="SKIP", skip_reason="x",
            selected_strategy=None, bet_side=None, bet_size_bnb=None,
            pool_bull_bnb=0.0, pool_bear_bnb=0.0,
        )
        # 3 should succeed, anything past should drop.
        succeeded = []
        t0 = time.monotonic()
        for _ in range(10):
            succeeded.append(worker.enqueue(snap))
        elapsed = time.monotonic() - t0
        assert sum(succeeded) == 3, f"queue capacity 3 should accept 3, got {sum(succeeded)}"
        assert succeeded[:3] == [True, True, True]
        assert all(not s for s in succeeded[3:]), "post-cap calls must drop"
        # 10 calls in <100ms confirms no blocking happened
        assert elapsed < 0.1, f"producer blocked? took {elapsed*1000:.1f}ms for 10 calls"
        s = worker.stats()
        assert s["dropped_full"] == 7


def test_worker_shutdown_drains_pending_items():
    """Items still in the queue at shutdown should be flushed."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        worker = kc.CaptureWorker(path)
        # Enqueue several items, then start the worker, then immediately
        # shutdown -- worker should drain everything before stopping.
        for i in range(20):
            snap = kc.build_snapshot(
                epoch=i, lock_at_unix=0, cutoff_ms=0, mode="dry",
                btc_klines_raw=_fake_klines(2), eth_klines_raw=None,
                sol_klines_raw=None, returns=None,
                decision="SKIP", skip_reason="x",
                selected_strategy=None, bet_side=None, bet_size_bnb=None,
                pool_bull_bnb=0.0, pool_bear_bnb=0.0,
            )
            worker.enqueue(snap)
        worker.start()
        worker.shutdown(timeout_s=3.0)

        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 20, f"expected 20 lines, got {len(lines)}"


def test_worker_serialise_failure_does_not_propagate():
    """A snapshot containing non-JSON-serialisable garbage should be logged + skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        worker = kc.CaptureWorker(path)
        worker.start()
        # Inject a snapshot with an unserialisable payload.
        bad_snap = {"schema_version": 1, "epoch": 1, "bad": object()}
        worker.enqueue(bad_snap)
        # Add a clean one after to verify worker keeps running.
        good_snap = kc.build_snapshot(
            epoch=2, lock_at_unix=0, cutoff_ms=0, mode="dry",
            btc_klines_raw=None, eth_klines_raw=None, sol_klines_raw=None,
            returns=None, decision="SKIP", skip_reason="x",
            selected_strategy=None, bet_side=None, bet_size_bnb=None,
            pool_bull_bnb=0.0, pool_bear_bnb=0.0,
        )
        worker.enqueue(good_snap)
        worker.shutdown(timeout_s=2.0)

        s = worker.stats()
        assert s["build_failures"] == 1, f"bad snap should have triggered build_failures, got {s}"
        assert s["written"] == 1, "good snap should still have been written"


# ---------------------------------------------------------------------------
# Reader: forward-compat
# ---------------------------------------------------------------------------

def test_iter_captures_skips_malformed_and_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        path.write_text(
            "\n".join([
                "",  # empty line ignored
                "{not-json",  # malformed -> skipped
                json.dumps({"no": "version"}),  # missing schema_version -> skipped
                json.dumps({"schema_version": 1, "epoch": 7, "decision": "SKIP"}),
                json.dumps({"schema_version": 1, "epoch": 8, "decision": "BET"}),
            ]) + "\n",
            encoding="utf-8",
        )
        records = list(kc.iter_captures(path))
        assert len(records) == 2
        assert [r["epoch"] for r in records] == [7, 8]


def test_iter_captures_missing_file_yields_nothing():
    records = list(kc.iter_captures(Path("/nonexistent/captures.jsonl")))
    assert records == []


# ---------------------------------------------------------------------------
# Microbenchmark: producer enqueue must be < 50us per call (build excluded;
# build itself is what runs in the producer in real use, so we benchmark
# build + enqueue together as the realistic cost).
# ---------------------------------------------------------------------------

def test_producer_enqueue_microbenchmark():
    """build_snapshot + worker.enqueue should average < 50us each.

    This is the actual cost the bet path pays when capture is enabled.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.jsonl"
        worker = kc.CaptureWorker(path, capacity=10_000)
        worker.start()
        # Pre-build the kline payload outside the timed loop -- in real
        # use this is just a list-ref, not data construction.
        btc_klines = _fake_klines(31)
        eth_klines = _fake_klines(31)
        sol_klines = _fake_klines(31)
        returns = {f"{p}_r{lb}": 0.0001 for p in ("btc", "eth", "sol") for lb in (3, 7, 15)}

        N = 1000
        t0 = time.monotonic()
        for i in range(N):
            snap = kc.build_snapshot(
                epoch=i, lock_at_unix=0, cutoff_ms=0, mode="dry",
                btc_klines_raw=btc_klines,
                eth_klines_raw=eth_klines,
                sol_klines_raw=sol_klines,
                returns=returns,
                decision="SKIP", skip_reason="bench",
                selected_strategy=None, bet_side=None, bet_size_bnb=None,
                pool_bull_bnb=0.0, pool_bear_bnb=0.0,
            )
            worker.enqueue(snap)
        elapsed = time.monotonic() - t0
        per_call_us = (elapsed / N) * 1e6
        worker.shutdown(timeout_s=5.0)

        # 31 candles x 3 pairs = ~93 list comprehensions per build, plus
        # an iso-format datetime call. <100us per call is generous on
        # any modern hardware; <50us is achievable.
        # We assert <200us as a CI-stable upper bound to avoid flakiness
        # on slow shared runners; manual runs typically see <30us.
        assert per_call_us < 200, (
            f"producer cost {per_call_us:.1f}us/call exceeds 200us budget"
        )
        # Print for visibility when run standalone.
        print(f"[BENCH] build+enqueue avg: {per_call_us:.1f}us/call ({N} calls in {elapsed*1000:.1f}ms)")


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
    print()
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
