"""Memory-efficient numpy kline loader for research backtests.

Replacement for `research.in_process_runner._load_klines_unified` that stores
each round's kline window as a numpy float64 array instead of nested Python
lists. ~5.7x memory reduction (272 bytes/kline Python -> 48 bytes/kline numpy).

API-compatible with the existing loader:
  - Same signature, same return shape: dict[int, ndarray]
  - Per-entry shape: (N, 6) where N = max_lookback+1+cutoff window, columns
    are [ts_ms, o, h, l, c, v] (float64)
  - Slicing via _slice_per_entry_numpy returns a numpy VIEW (no copy)

Downstream pipeline (`compute_signal_from_klines`) reads:
  - klines[-1][0] -> numpy float64 scalar; int(.) works
  - [k[4] for k in klines] -> list of numpy float64 scalars; arithmetic works

NOT for production live bot — that fetches klines fresh per round via OKX REST.
This is research-side only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_klines_unified_numpy(
    path: Path,
    *,
    earliest_offset: int,
    latest_offset: int,
    extended_path: Path | None = None,
) -> dict[int, np.ndarray]:
    """Load + slice each per-round record to the unified window as numpy arrays.

    Same slicing semantics as `in_process_runner._load_klines_unified`:
      start_neg = -(earliest_offset - 1)
      end_neg   = -(latest_offset - 2) if latest_offset >= 3 else None

    Returns dict[epoch -> (N, 6) float64 ndarray].
    """
    if not path.exists() and (extended_path is None or not extended_path.exists()):
        return {}
    if latest_offset < 2:
        raise ValueError(f"latest_offset_must_be_ge_2: {latest_offset}")

    start_neg = -(earliest_offset - 1)
    end_neg: int | None = None if latest_offset == 2 else -(latest_offset - 2)
    result: dict[int, np.ndarray] = {}

    def _ingest(p: Path) -> None:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error") or rec.get("klines_1s") is None:
                    continue
                kl = rec["klines_1s"]
                if end_neg is None:
                    kl_slice = kl[start_neg:]
                else:
                    kl_slice = kl[start_neg:end_neg]
                if not kl_slice:
                    continue
                ep = int(rec["epoch"])
                if ep in result:
                    continue  # canonical wins
                # Convert nested list to (N, 6) numpy array. Each kline is
                # [ts_ms, o, h, l, c, v] — first is int, rest may be str or
                # number. np.asarray with dtype=float64 coerces uniformly.
                arr = np.asarray(kl_slice, dtype=np.float64)
                # Defensive: only store if shape matches expectations
                if arr.ndim == 2 and arr.shape[1] == 6:
                    result[ep] = arr

    if path.exists():
        _ingest(path)
    if extended_path is not None and extended_path.exists():
        _ingest(extended_path)
    return result


def slice_per_entry_numpy(
    kl_unified: np.ndarray,
    *,
    kline_cutoff_seconds: int,
    max_lookback: int,
    earliest_offset: int,
) -> np.ndarray:
    """Per-entry slice matching in_process_runner._slice_per_entry semantics.

    Returns a numpy VIEW (no copy) of shape (max_lookback+1, 6).
    """
    end_idx = earliest_offset - kline_cutoff_seconds
    start_idx = end_idx - (max_lookback + 1)
    return kl_unified[start_idx:end_idx]
