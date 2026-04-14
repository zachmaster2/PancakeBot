"""Re-sync klines at cutoff_seconds=2.

Writes to a new file (.new), one record per fetch, so progress is
visible and interruptions are resumable. Atomically replaces the
original when done.
"""
from __future__ import annotations

import json, sys, time, os, threading, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import INTERVAL_SECONDS
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore

CUTOFF_S = 2
CANDLE_COUNT = 31
WORKERS = 8
BATCH_SIZE = 80
RATE_PER_SEC = 9  # OKX allows 20/2s=10/s per endpoint


class _RateLimiter:
    """Global rate limiter: ensures at most max_per_sec requests across all threads."""

    def __init__(self, max_per_sec: float) -> None:
        self._min_interval = 1.0 / max_per_sec
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


_rate_candles = _RateLimiter(RATE_PER_SEC)
_rate_history = _RateLimiter(RATE_PER_SEC)

_EP_LIMITERS = {
    "candles": _rate_candles,
    "history-candles": _rate_history,
}


def fetch(inst_id, cutoff_ms):
    # Try "candles" first (recent data), fall back to "history-candles" (older)
    for ep in ("candles", "history-candles"):
        url = (f"https://www.okx.com/api/v5/market/{ep}"
               f"?instId={inst_id}&bar=1s&limit={CANDLE_COUNT}&after={cutoff_ms}")
        limiter = _EP_LIMITERS[ep]
        for attempt in range(2):
            limiter.acquire()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PancakeBot/1.0"})
                resp = urllib.request.urlopen(req, timeout=5)
                body = json.loads(resp.read())
                if body.get("code") == "0":
                    if body.get("data") and len(body["data"]) >= CANDLE_COUNT * 0.9:
                        rows = body["data"]
                        return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                                 float(r[4]), float(r[5])] for r in reversed(rows)][-CANDLE_COUNT:]
                    # code=0 but insufficient data: endpoint doesn't have it, skip retries
                    break
            except Exception:
                continue
    return None


def main():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds_map = {int(r.epoch): r for r in store.iter_closed_rounds()}
    print(f"Loaded {len(rounds_map)} rounds", flush=True)

    for label, path, inst_id in [
        ("BNB", "var/bnb_spot_prices.jsonl", "BNB-USDT"),
        ("BTC", "var/btc_spot_prices.jsonl", "BTC-USDT"),
    ]:
        print(f"\n=== {label} ===", flush=True)
        new_path = path + ".new"

        # Load existing records into a dict for lookup
        existing = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            existing[int(rec["epoch"])] = rec

        # Check what's already done in the .new file (resume support)
        done_epochs = set()
        if os.path.exists(new_path):
            for line in Path(new_path).read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                done_epochs.add(int(rec["epoch"]))
            print(f"  Resuming: {len(done_epochs)} already in .new file", flush=True)

        # Build work list: epochs sorted, skip already done
        all_epochs = sorted(existing.keys())
        to_process = [ep for ep in all_epochs if ep not in done_epochs]
        print(f"  Total: {len(all_epochs)}, To process: {len(to_process)}", flush=True)

        updated = 0
        reused = 0
        failed = 0

        with open(new_path, "a", encoding="utf-8") as out_f:
            for batch_start in range(0, len(to_process), BATCH_SIZE):
                batch_epochs = to_process[batch_start:batch_start + BATCH_SIZE]

                # Determine which need fetching vs which are already correct
                fetch_list = []  # (epoch, cutoff_ms)
                for ep in batch_epochs:
                    rnd = rounds_map.get(ep)
                    if not rnd:
                        continue
                    expected = (rnd.start_at + INTERVAL_SECONDS - CUTOFF_S) * 1000 - 1000
                    rec = existing.get(ep, {})
                    kl = rec.get("klines_1s", [])
                    if kl and len(kl) == CANDLE_COUNT and int(kl[-1][0]) == expected:
                        # Already correct — write as-is
                        out_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                        reused += 1
                    else:
                        cutoff_ms = (rnd.start_at + INTERVAL_SECONDS - CUTOFF_S) * 1000
                        fetch_list.append((ep, cutoff_ms))

                # Fetch in parallel
                if fetch_list:
                    results = {}
                    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                        futures = {pool.submit(fetch, inst_id, cm): ep for ep, cm in fetch_list}
                        for fut in as_completed(futures):
                            ep = futures[fut]
                            klines = fut.result()
                            if klines and len(klines) == CANDLE_COUNT:
                                results[ep] = klines
                            else:
                                failed += 1

                    # Write fetched results in epoch order
                    for ep, _ in fetch_list:
                        if ep in results:
                            rnd = rounds_map[ep]
                            rec = {
                                "epoch": ep,
                                "lock_at": rnd.start_at + INTERVAL_SECONDS,
                                "klines_1s": results[ep],
                            }
                            out_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                            out_f.flush()
                            updated += 1

                done = batch_start + len(batch_epochs)
                if done % 500 < BATCH_SIZE or done >= len(to_process):
                    print(f"  {done + len(done_epochs)}/{len(all_epochs)}: "
                          f"{updated} fetched, {reused} reused, {failed} failed",
                          flush=True)

        # Atomic replace
        total_new = len(done_epochs) + reused + updated
        print(f"  Final: {total_new} records in .new, replacing original", flush=True)
        os.replace(new_path, path)
        print(f"  Done: {path}", flush=True)


if __name__ == "__main__":
    main()
