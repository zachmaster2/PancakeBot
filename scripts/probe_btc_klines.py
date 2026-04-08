"""Probe: fetch BTC/USDT 1s klines from OKX for each round.

Same timestamps as BNB data in cutoff_spot_prices.jsonl.
Fetches 100 1s candles anchored at lockAt for each round.

Output: var/btc_spot_prices.jsonl
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import urllib.request
import urllib.error

NUM_ROUNDS = 5000
OKX_BASE = "https://www.okx.com"
INST_ID = "BTC-USDT"
KLINES_PER_ROUND = 100
REQUEST_DELAY = 0.13

ROUNDS_PATH = Path("var/closed_rounds.jsonl")
OUT_PATH = Path("var/btc_spot_prices.jsonl")


def fetch_1s_klines(anchor_ms: int) -> list[list] | None:
    after_ms = anchor_ms + 1000
    for endpoint in ("history-candles", "candles"):
        url = (
            f"{OKX_BASE}/api/v5/market/{endpoint}"
            f"?instId={INST_ID}&bar=1s&limit={KLINES_PER_ROUND}"
            f"&after={after_ms}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PancakeBot/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError):
            continue

        if body.get("code") != "0" or not body.get("data"):
            continue

        rows = body["data"]
        if len(rows) < KLINES_PER_ROUND * 0.9:
            continue

        out = []
        for row in reversed(rows):
            out.append([
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ])
        return out
    return None


def main() -> None:
    lines = ROUNDS_PATH.read_text().splitlines()
    rounds = [json.loads(l) for l in lines[-NUM_ROUNDS:]]
    print(f"Loaded {len(rounds)} rounds  (epochs {rounds[0]['epoch']}..{rounds[-1]['epoch']})")

    done_epochs: set[int] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if line.strip():
                done_epochs.add(json.loads(line)["epoch"])
        print(f"Resuming: {len(done_epochs)} rounds already fetched")

    remaining = [r for r in rounds if r["epoch"] not in done_epochs]
    print(f"Fetching {len(remaining)} remaining rounds...")

    fetched = 0
    errors = 0
    with open(OUT_PATH, "a") as f:
        for i, rnd in enumerate(remaining):
            epoch = rnd["epoch"]
            lock_at_ms = rnd["lockAt"] * 1000

            klines = fetch_1s_klines(lock_at_ms)

            if klines is None:
                errors += 1
                rec = {
                    "epoch": epoch,
                    "lock_at": rnd["lockAt"],
                    "klines_1s": None,
                    "error": True,
                }
            else:
                rec = {
                    "epoch": epoch,
                    "lock_at": rnd["lockAt"],
                    "klines_1s": klines,
                    "error": False,
                }

            f.write(json.dumps(rec) + "\n")
            fetched += 1

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(remaining)}  (errors={errors})")

            time.sleep(REQUEST_DELAY)

    print(f"\nDone: {fetched} fetched, {errors} errors")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
