"""Backfill BNB/USDT 1m klines from OKX public API into a local JSONL file.

OKX is accessible from US IPs for unauthenticated market data.
OKX provides near-100% 1m bar coverage for BNB/USDT (vs ~20% on Binance US).

Output format matches the existing klines.jsonl schema used by the project,
with two differences:
  - taker_buy_base_volume and taker_buy_quote_volume are absent (OKX does not
    provide taker-side breakdown in the standard candles endpoint)
  - number_of_trades is absent for the same reason

The correlation script handles missing fields gracefully (skips those features).

Usage:
  python inspection/run_okx_klines_backfill.py --out-path ../PancakeBot_var_data/klines_okx.jsonl --tail-days 200

Resumable: if the output file already exists, the script starts from the most
recent stored candle and only fetches newer data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


_OKX_BASE = "https://www.okx.com"
_SYMBOL = "BNB-USDT"
_BAR = "1m"
_BAR_MS = 60_000  # milliseconds per bar
_MAX_PER_REQUEST = 100
_RATE_LIMIT_SLEEP = 0.25  # seconds between requests (4 req/s, well inside 20/2s limit)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OKX BNB/USDT 1m kline backfill")
    p.add_argument("--out-path", type=str,
                   default="../PancakeBot_var_data/klines_okx.jsonl",
                   help="Output JSONL path")
    p.add_argument("--tail-days", type=float, default=200,
                   help="How many days of history to fetch (from now backwards)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be fetched without writing")
    return p


# ---------------------------------------------------------------------------
# OKX API
# ---------------------------------------------------------------------------

def _fetch_candles(
    *,
    after_ms: int | None = None,
    use_history: bool = True,
) -> list[dict]:
    """Fetch up to _MAX_PER_REQUEST closed 1m candles from OKX.

    OKX returns rows in descending order (newest first).
    'after' means: return rows with ts < after_ms (exclusive upper bound).

    Returns list of dicts in ascending order (oldest first), confirmed closed only.
    """
    endpoint = "/api/v5/market/history-candles" if use_history else "/api/v5/market/candles"
    params: dict = {
        "instId": _SYMBOL,
        "bar": _BAR,
        "limit": str(_MAX_PER_REQUEST),
    }
    if after_ms is not None:
        params["after"] = str(int(after_ms))

    for attempt in range(5):
        try:
            r = requests.get(f"{_OKX_BASE}{endpoint}", params=params, timeout=15)
            if r.status_code == 429:
                print("  Rate limited, sleeping 5s ...")
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
            print(f"  Request error ({e}), retrying in 3s ...")
            time.sleep(3)
    else:
        raise RuntimeError("OKX fetch failed after retries")

    out: list[dict] = []
    for row in rows:
        confirm = int(row[8])
        if confirm != 1:
            continue  # skip open/unconfirmed candle
        ts_ms = int(row[0])
        out.append({
            "open_time_ms": ts_ms,
            "close_time_ms": ts_ms + _BAR_MS - 1,
            "open_price": float(row[1]),
            "high_price": float(row[2]),
            "low_price": float(row[3]),
            "close_price": float(row[4]),
            "volume": float(row[5]),           # base currency (BNB)
            "quote_asset_volume": float(row[6]),  # quote currency (USDT)
        })

    # Sort ascending (oldest first)
    out.sort(key=lambda d: d["open_time_ms"])
    return out


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _read_tail(path: Path) -> int | None:
    """Return open_time_ms of the last stored candle, or None if file is empty."""
    if not path.exists():
        return None
    with path.open("r") as fh:
        last_line = ""
        for line in fh:
            if line.strip():
                last_line = line.strip()
    if not last_line:
        return None
    return int(json.loads(last_line)["open_time_ms"])


def _read_head(path: Path) -> int | None:
    """Return open_time_ms of the first stored candle, or None if file is empty."""
    if not path.exists():
        return None
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                return int(json.loads(line.strip())["open_time_ms"])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now_ms = int(time.time() * 1000)
    target_start_ms = int(now_ms - args.tail_days * 86400 * 1000)

    tail_ms = _read_tail(out_path)
    head_ms = _read_head(out_path)

    import datetime

    def fmt(ms: int) -> str:
        return datetime.datetime.fromtimestamp(ms / 1000, datetime.UTC).strftime("%Y-%m-%d %H:%M")

    print(f"OKX BNB/USDT 1m kline backfill")
    print(f"  Target start : {fmt(target_start_ms)}")
    print(f"  Target end   : {fmt(now_ms)}")
    print(f"  Out path     : {out_path}")

    if tail_ms is not None:
        print(f"  Existing data: {fmt(head_ms)} -> {fmt(tail_ms)} ({tail_ms=})")
    else:
        print(f"  No existing data, starting fresh")

    if args.dry_run:
        print("Dry run — no writes.")
        return

    # ------------------------------------------------------------------
    # Phase 1: Backfill history backwards from target_start_ms to wherever
    # the existing file starts (or all the way back if no file).
    # ------------------------------------------------------------------

    existing_head = head_ms  # oldest candle currently in file (None if empty)
    fetch_backwards_until = int(target_start_ms)

    if existing_head is None or existing_head > fetch_backwards_until + _BAR_MS:
        print(f"\nPhase 1: Backfilling backwards to {fmt(fetch_backwards_until)} ...")

        # We'll collect batches in memory then prepend to file
        # For simplicity: collect ALL backwards batches first, then write.
        # (File is small enough — 200 days * 1440 = 288k lines ≈ 50MB)
        collected: list[dict] = []

        # Start pagination just before existing head (or from now)
        after_ms = int(existing_head) if existing_head is not None else None

        batch_count = 0
        while True:
            batch = _fetch_candles(after_ms=after_ms, use_history=True)
            batch_count += 1

            if not batch:
                print(f"  Batch {batch_count}: empty response, stopping backwards fetch")
                break

            oldest_in_batch = batch[0]["open_time_ms"]
            newest_in_batch = batch[-1]["open_time_ms"]

            # Only keep candles we actually need
            batch = [c for c in batch if c["open_time_ms"] >= fetch_backwards_until]
            collected.extend(batch)

            if batch_count % 50 == 0:
                print(f"  Batch {batch_count}: fetched to {fmt(oldest_in_batch)}, "
                      f"collected {len(collected):,} candles so far ...")

            if oldest_in_batch <= fetch_backwards_until:
                print(f"  Reached target start ({fmt(fetch_backwards_until)}), done backwards")
                break

            after_ms = int(oldest_in_batch)  # next page: fetch before oldest we have
            time.sleep(_RATE_LIMIT_SLEEP)

        # Sort ascending
        collected.sort(key=lambda d: d["open_time_ms"])

        if collected:
            if out_path.exists() and existing_head is not None:
                # Prepend: write collected + existing
                print(f"  Prepending {len(collected):,} historical candles ...")
                existing_lines = out_path.read_text(encoding="utf-8")
                with out_path.open("w", encoding="utf-8") as fh:
                    for c in collected:
                        fh.write(json.dumps(c, separators=(",", ":"), sort_keys=True))
                        fh.write("\n")
                    fh.write(existing_lines)
            else:
                print(f"  Writing {len(collected):,} candles to new file ...")
                with out_path.open("w", encoding="utf-8") as fh:
                    for c in collected:
                        fh.write(json.dumps(c, separators=(",", ":"), sort_keys=True))
                        fh.write("\n")

            # Update tail_ms for phase 2
            tail_ms = _read_tail(out_path)
        else:
            print("  No historical candles collected.")
    else:
        print(f"\nPhase 1: History already covers target start, skipping backwards fetch.")

    # ------------------------------------------------------------------
    # Phase 2: Forward-fill from tail to now.
    # ------------------------------------------------------------------
    tail_ms = _read_tail(out_path)
    if tail_ms is None:
        print("No data in file after phase 1. Exiting.")
        return

    print(f"\nPhase 2: Forward-filling from {fmt(tail_ms)} to now ...")

    appended = 0
    # For recent candles, use the regular candles endpoint (higher limit, more up to date)
    # Paginate forward: fetch batches where ts > tail_ms
    current_after = None  # no 'after' = get most recent
    forward_batches: list[list[dict]] = []

    # We need to collect from tail_ms forward.
    # OKX doesn't have a "since" parameter easily, but we can use 'before' (exclusive lower)
    # Actually OKX 'before' means: return rows with ts > before_ms
    # So we use before=tail_ms to get candles newer than our tail.

    # Strategy: fetch recent candles using 'before' pagination going forward.
    # But OKX sorts newest first, so:
    #   - First request: no 'before', returns latest 100
    #   - We only keep what's newer than tail_ms
    #   - If oldest in batch > tail_ms, we need to paginate backwards with 'after'
    #     from oldest_in_batch until we reach tail_ms
    # Actually this gets complex. Simpler:
    # Use candles endpoint without pagination, get last 300, filter to > tail_ms.
    # If gap is > 300 bars, use history-candles with after to walk forward.

    gap_bars = (now_ms - tail_ms) // _BAR_MS
    print(f"  Gap: ~{gap_bars:,} bars")

    if gap_bars <= 300:
        # Single request covers it
        batch = _fetch_candles(after_ms=None, use_history=False)
        new_bars = [c for c in batch if c["open_time_ms"] > tail_ms]
        new_bars.sort(key=lambda d: d["open_time_ms"])
        if new_bars:
            with out_path.open("a", encoding="utf-8") as fh:
                for c in new_bars:
                    fh.write(json.dumps(c, separators=(",", ":"), sort_keys=True))
                    fh.write("\n")
            appended = len(new_bars)
    else:
        # Need to paginate: walk backwards from now until we reach tail_ms,
        # collect everything newer than tail_ms, then append in order.
        new_bars_all: list[dict] = []
        after_ms_fwd = None
        batch_count = 0
        while True:
            batch = _fetch_candles(after_ms=after_ms_fwd, use_history=True)
            batch_count += 1
            if not batch:
                break
            oldest = batch[0]["open_time_ms"]
            relevant = [c for c in batch if c["open_time_ms"] > tail_ms]
            new_bars_all.extend(relevant)

            if batch_count % 50 == 0:
                print(f"  Forward batch {batch_count}: at {fmt(oldest)}, "
                      f"collected {len(new_bars_all):,} so far ...")

            if oldest <= tail_ms:
                break
            after_ms_fwd = int(oldest)
            time.sleep(_RATE_LIMIT_SLEEP)

        new_bars_all.sort(key=lambda d: d["open_time_ms"])
        if new_bars_all:
            with out_path.open("a", encoding="utf-8") as fh:
                for c in new_bars_all:
                    fh.write(json.dumps(c, separators=(",", ":"), sort_keys=True))
                    fh.write("\n")
            appended = len(new_bars_all)

    print(f"  Appended {appended:,} new candles")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    final_head = _read_head(out_path)
    final_tail = _read_tail(out_path)
    # Count lines
    with out_path.open("r") as fh:
        total_lines = sum(1 for l in fh if l.strip())

    print(f"\nDone.")
    print(f"  File: {out_path}")
    print(f"  Total candles: {total_lines:,}")
    if final_head and final_tail:
        print(f"  Coverage: {fmt(final_head)} -> {fmt(final_tail)}")

    # Quick quality check
    print("\nQuality check (last 1440 candles = 1 day):")
    with out_path.open("r") as fh:
        all_lines = [l.strip() for l in fh if l.strip()]
    recent = [json.loads(l) for l in all_lines[-1440:]]
    zero_vol = sum(1 for c in recent if float(c["volume"]) == 0)
    print(f"  Zero-volume bars: {zero_vol}/{len(recent)} = {zero_vol/len(recent):.1%}")


if __name__ == "__main__":
    main()
