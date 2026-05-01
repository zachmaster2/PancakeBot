"""p3b OKX trade-tape fetcher with caching + checkpoint.

Walks `/api/v5/market/history-trades?instId=BNB-USDT` backward from current
tradeId until trade ts <= target start_at. Writes raw trades to JSONL cache
and a checkpoint JSON for resumability.

Usage:
    python p3b_okx_trade_fetcher.py --slice post_v1
    python p3b_okx_trade_fetcher.py --slice f5
    python p3b_okx_trade_fetcher.py --slice f4
    python p3b_okx_trade_fetcher.py --all  # all three slices in one walk

Per orchestrator v1.1 (locked):
- Single continuous walk-back from "now" to f4 start_at = most efficient
- Cache: var/extended/okx_trades_BNB-USDT.jsonl (single file; epoch-binned analysis post-hoc)
- Rate limit: 4 req/s (well under OKX 8 req/s budget; leaves dry-bot headroom)
- Trades stored as raw OKX records: {instId, side, sz, px, source, tradeId, ts}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

OKX_BASE = "https://www.okx.com"
INST = "BNB-USDT"
RATE_SLEEP_S = 0.25  # 4 req/s
USER_AGENT = "p3b-fetcher/1.0"

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")

FLOOR_START_AT = 1765444670
FLOOR_EPOCH = 437562
EPOCH_DURATION = 300

# Slice boundaries (epoch range)
SLICES = {
    "post_v1": (475312, 477254),
    "f5":      (466782, 474086),
    "f4":      (459477, 466781),
}

CACHE_PATH = REPO / "var" / "extended" / "okx_trades_BNB-USDT.jsonl"
CHECKPOINT_PATH = REPO / "var" / "extended" / "okx_trades_BNB-USDT_checkpoint.json"

MAX_RETRIES = 5
BACKOFF_BASE_S = 2.0


def epoch_start_at(ep: int) -> int:
    return FLOOR_START_AT + (ep - FLOOR_EPOCH) * EPOCH_DURATION


def http_get(url: str, timeout: float = 15.0) -> tuple[int, dict, float]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - t0
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body), elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        body = e.read().decode("utf-8") if hasattr(e, "read") else ""
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            obj = {"raw": body}
        return e.code, obj, elapsed
    except urllib.error.URLError as e:
        return -1, {"error": str(e)}, time.perf_counter() - t0


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def write_checkpoint(state: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, CHECKPOINT_PATH)


def fetch_one_page(after_id: str | None) -> tuple[list[dict], int, dict]:
    """Fetch a single page. Returns (data, status, raw_obj). Retries on 429
    with exponential backoff."""
    if after_id is None:
        url = f"{OKX_BASE}/api/v5/market/history-trades?instId={INST}&limit=100"
    else:
        url = f"{OKX_BASE}/api/v5/market/history-trades?instId={INST}&limit=100&after={after_id}"
    for attempt in range(MAX_RETRIES):
        status, obj, _ = http_get(url)
        if status == 429:
            backoff = BACKOFF_BASE_S * (2 ** attempt)
            print(f"    HTTP 429 attempt {attempt+1}/{MAX_RETRIES}, sleeping {backoff:.1f}s",
                  flush=True)
            time.sleep(backoff)
            continue
        if status != 200 or not isinstance(obj, dict):
            return [], status, obj if isinstance(obj, dict) else {}
        if obj.get("code") != "0":
            print(f"    OKX api code {obj.get('code')!r} msg {obj.get('msg', '')!r}", flush=True)
            return [], status, obj
        return obj.get("data") or [], status, obj
    return [], -1, {"error": "max retries exhausted"}


def fetch_walk_back(target_start_at_s: int, *, resume: bool = True) -> dict:
    """Walk back from current (or checkpoint-resume tradeId) until trade.ts
    <= target_start_at_s * 1000. Append all trades to CACHE_PATH (JSONL,
    one trade per line). Update CHECKPOINT_PATH after each page.

    Returns summary dict.
    """
    target_ts_ms = target_start_at_s * 1000
    state = load_checkpoint() if resume else {}
    after_id = state.get("oldest_tradeId_fetched") if resume else None
    n_pages_prev = state.get("n_pages", 0)
    n_trades_prev = state.get("n_trades", 0)

    if after_id:
        print(f"  Resuming from oldest_tradeId={after_id} "
              f"(prev pages: {n_pages_prev}, prev trades: {n_trades_prev})",
              flush=True)
    else:
        print("  Starting fresh from current tradeId (no resume)", flush=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_pages = 0
    n_trades = 0
    n_429 = 0
    t0 = time.time()

    with open(CACHE_PATH, "a", encoding="utf-8") as f_out:
        while True:
            data, status, raw_obj = fetch_one_page(after_id)
            n_pages += 1
            if status == 429:
                n_429 += 1
            if not data:
                if status == 200 and isinstance(raw_obj, dict) and raw_obj.get("code") == "0":
                    print(f"  [page {n_pages}] empty data — out of retention or no more pages",
                          flush=True)
                else:
                    print(f"  [page {n_pages}] FAIL status={status} obj={raw_obj}", flush=True)
                break
            for t in data:
                f_out.write(json.dumps(t) + "\n")
                n_trades += 1
            f_out.flush()
            oldest = data[-1]
            after_id = oldest["tradeId"]
            oldest_ts_ms = int(oldest["ts"])

            # Periodic progress + checkpoint
            if n_pages % 50 == 0 or n_pages == 1:
                dt = datetime.fromtimestamp(oldest_ts_ms / 1000, tz=timezone.utc)
                rate = n_pages / max(0.001, time.time() - t0)
                pct_done = max(0, min(100, 100 * (target_ts_ms - oldest_ts_ms) /
                                                 max(1, target_ts_ms - oldest_ts_ms - 1)))
                # Better progress: how far back have we walked vs how far we need
                remaining_s = max(0, (oldest_ts_ms - target_ts_ms) / 1000)
                print(f"  [page {n_pages:>5}] +{len(data)} trades, "
                      f"oldest={dt.isoformat()}, total={n_trades}, "
                      f"rate={rate:.2f} req/s, remaining={remaining_s/3600:.1f}h",
                      flush=True)

            if n_pages % 100 == 0:
                write_checkpoint({
                    "oldest_tradeId_fetched": after_id,
                    "oldest_ts_ms": oldest_ts_ms,
                    "n_pages": n_pages_prev + n_pages,
                    "n_trades": n_trades_prev + n_trades,
                    "target_start_at_s": target_start_at_s,
                    "last_update": datetime.now(timezone.utc).isoformat(),
                })

            if oldest_ts_ms <= target_ts_ms:
                print(f"  [page {n_pages}] reached target boundary; stopping", flush=True)
                break

            time.sleep(RATE_SLEEP_S)

    elapsed = time.time() - t0
    final_state = {
        "oldest_tradeId_fetched": after_id,
        "n_pages": n_pages_prev + n_pages,
        "n_trades": n_trades_prev + n_trades,
        "target_start_at_s": target_start_at_s,
        "last_update": datetime.now(timezone.utc).isoformat(),
        "completed": True,
    }
    write_checkpoint(final_state)
    print(f"\n  WALK COMPLETE: {n_pages} new pages, {n_trades} new trades, "
          f"elapsed {elapsed:.1f}s ({elapsed/60:.1f}m), 429 errors {n_429}", flush=True)
    return final_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=list(SLICES.keys()) + ["all"], default="all")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--target-epoch", type=int, default=None,
                        help="custom target lower-bound epoch")
    args = parser.parse_args()

    if args.target_epoch is not None:
        target_ep = args.target_epoch
    elif args.slice == "all":
        # f4 has the oldest start_at among the three slices
        target_ep = SLICES["f4"][0]
    else:
        target_ep = SLICES[args.slice][0]
    target_start_at_s = epoch_start_at(target_ep)
    print(f"Target slice lower-bound epoch: {target_ep}")
    print(f"Target start_at: {target_start_at_s} = "
          f"{datetime.fromtimestamp(target_start_at_s, tz=timezone.utc).isoformat()}")
    print(f"Cache: {CACHE_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    fetch_walk_back(target_start_at_s, resume=not args.no_resume)


if __name__ == "__main__":
    main()
