"""Sync BNB/USDT 1m klines from OKX public API into a local JSONL file.

Resumable: if the output file already exists, only fetches candles newer than
the last stored one (plus backwards-fills any gap up to tail_days).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from pancakebot.core.logging import info

_OKX_BASE = "https://www.okx.com"
_SYMBOL = "BNB-USDT"
_BAR_MS = 60_000
_MAX_PER_REQUEST = 100
_RATE_LIMIT_SLEEP = 0.25


def sync_okx_klines(*, out_path: str, tail_days: float = 200) -> dict[str, object]:
    """Sync OKX 1m klines to out_path. Returns a summary dict."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    now_ms = int(time.time() * 1000)
    target_start_ms = int(now_ms - tail_days * 86400 * 1000)

    tail_ms = _read_tail(path)
    head_ms = _read_head(path)

    info("SYNC", "KLINES", "START",
         msg=f"OKX klines sync: tail_days={tail_days} out={out_path} "
             f"existing={'none' if tail_ms is None else f'{_fmt(head_ms)}..{_fmt(tail_ms)}'}")

    # Phase 1: Backfill history backwards if needed
    if head_ms is None or head_ms > target_start_ms + _BAR_MS:
        info("SYNC", "KLINES", "BKFILL", msg=f"Backfilling from {_fmt(target_start_ms)}")
        collected: list[dict] = []
        after_ms = int(head_ms) if head_ms is not None else None
        batch_count = 0
        while True:
            batch = _fetch_candles(after_ms=after_ms, use_history=True)
            batch_count += 1
            if not batch:
                break
            oldest = batch[0]["open_time_ms"]
            collected.extend(c for c in batch if c["open_time_ms"] >= target_start_ms)
            if batch_count % 50 == 0:
                info("SYNC", "KLINES", "BKFILL_PROG",
                     msg=f"batch={batch_count} at={_fmt(oldest)} collected={len(collected):,}")
            if oldest <= target_start_ms:
                break
            after_ms = int(oldest)
            time.sleep(_RATE_LIMIT_SLEEP)

        collected.sort(key=lambda d: d["open_time_ms"])
        if collected:
            if path.exists() and head_ms is not None:
                existing = path.read_text(encoding="utf-8")
                with path.open("w", encoding="utf-8") as fh:
                    for c in collected:
                        fh.write(json.dumps(c, separators=(",", ":"), sort_keys=True) + "\n")
                    fh.write(existing)
            else:
                with path.open("w", encoding="utf-8") as fh:
                    for c in collected:
                        fh.write(json.dumps(c, separators=(",", ":"), sort_keys=True) + "\n")
            tail_ms = _read_tail(path)
            info("SYNC", "KLINES", "BKFILL_DONE", msg=f"wrote {len(collected):,} historical candles")
    else:
        info("SYNC", "KLINES", "BKFILL_SKIP", msg="History already covers target start")

    # Phase 2: Forward-fill from tail to now
    tail_ms = _read_tail(path)
    if tail_ms is None:
        return {"status": "no_data"}

    gap_bars = (now_ms - tail_ms) // _BAR_MS
    info("SYNC", "KLINES", "FORWARD", msg=f"Forward-filling gap={gap_bars:,} bars from {_fmt(tail_ms)}")

    appended = 0
    if gap_bars <= 300:
        batch = _fetch_candles(after_ms=None, use_history=False)
        new_bars = sorted((c for c in batch if c["open_time_ms"] > tail_ms),
                          key=lambda d: d["open_time_ms"])
        if new_bars:
            with path.open("a", encoding="utf-8") as fh:
                for c in new_bars:
                    fh.write(json.dumps(c, separators=(",", ":"), sort_keys=True) + "\n")
            appended = len(new_bars)
    else:
        new_bars_all: list[dict] = []
        after_ms_fwd = None
        batch_count = 0
        while True:
            batch = _fetch_candles(after_ms=after_ms_fwd, use_history=True)
            batch_count += 1
            if not batch:
                break
            oldest = batch[0]["open_time_ms"]
            new_bars_all.extend(c for c in batch if c["open_time_ms"] > tail_ms)
            if oldest <= tail_ms:
                break
            after_ms_fwd = int(oldest)
            time.sleep(_RATE_LIMIT_SLEEP)

        new_bars_all.sort(key=lambda d: d["open_time_ms"])
        if new_bars_all:
            with path.open("a", encoding="utf-8") as fh:
                for c in new_bars_all:
                    fh.write(json.dumps(c, separators=(",", ":"), sort_keys=True) + "\n")
            appended = len(new_bars_all)

    final_head = _read_head(path)
    final_tail = _read_tail(path)
    total_lines = _count_lines(path)

    info("SYNC", "KLINES", "DONE",
         msg=f"total={total_lines:,} appended={appended} "
             f"coverage={_fmt(final_head)}..{_fmt(final_tail)}")

    return {
        "total_candles": int(total_lines),
        "appended": int(appended),
        "head_ms": int(final_head) if final_head else None,
        "tail_ms": int(final_tail) if final_tail else None,
    }


def _fetch_candles(*, after_ms: int | None, use_history: bool) -> list[dict]:
    endpoint = "/api/v5/market/history-candles" if use_history else "/api/v5/market/candles"
    params: dict = {"instId": _SYMBOL, "bar": "1m", "limit": str(_MAX_PER_REQUEST)}
    if after_ms is not None:
        params["after"] = str(int(after_ms))

    for attempt in range(5):
        try:
            r = requests.get(f"{_OKX_BASE}{endpoint}", params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(5)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"OKX HTTP {r.status_code}: {r.text[:200]}")
            payload = r.json()
            if payload.get("code") != "0":
                raise RuntimeError(f"OKX API error: {payload}")
            rows = payload.get("data", [])
            break
        except requests.RequestException as e:
            if attempt == 4:
                raise
            time.sleep(3)
    else:
        raise RuntimeError("OKX klines fetch failed after retries")

    out: list[dict] = []
    for row in rows:
        if int(row[8]) != 1:
            continue  # skip unconfirmed candle
        ts_ms = int(row[0])
        out.append({
            "open_time_ms": ts_ms,
            "close_time_ms": ts_ms + _BAR_MS - 1,
            "open_price": float(row[1]),
            "high_price": float(row[2]),
            "low_price": float(row[3]),
            "close_price": float(row[4]),
            "volume": float(row[5]),
            "quote_asset_volume": float(row[6]),
        })
    out.sort(key=lambda d: d["open_time_ms"])
    return out


def _read_tail(path: Path) -> int | None:
    if not path.exists():
        return None
    last = ""
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                last = line.strip()
    return int(json.loads(last)["open_time_ms"]) if last else None


def _read_head(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                return int(json.loads(line.strip())["open_time_ms"])
    return None


def _count_lines(path: Path) -> int:
    with path.open("r") as fh:
        return sum(1 for line in fh if line.strip())


def _fmt(ms: int | None) -> str:
    if ms is None:
        return "none"
    import datetime
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.UTC).strftime("%Y-%m-%d %H:%M")
