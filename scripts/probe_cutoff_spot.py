"""Probe: fetch 100 x 1s OKX klines per round ending at closeAt.

For each round, fetches 100 1s candles ending at closeAt and saves the
full array.  This covers ~100s before close — spanning cutoff, lock, and
close — so we can experiment offline with different lookback windows and
cutoff offsets without re-fetching.

Output: var/cutoff_spot_prices.jsonl — one JSON line per round:
  { epoch, lock_at, close_at, klines_1s: [[ts_ms, o, h, l, c, vol], ...] }

The 100 klines are anchored at lockAt (not closeAt), covering roughly
[lockAt - 100s, lockAt].  This spans the decision window: cutoff at
lockAt - 17s and lookback origins up to ~80s before cutoff.

After fetching, runs signal-quality analysis with configurable params.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Config — fetching
# ---------------------------------------------------------------------------
NUM_ROUNDS = 5000
OKX_BASE = "https://www.okx.com"
INST_ID = "BNB-USDT"
KLINES_PER_ROUND = 100
REQUEST_DELAY = 0.13

ROUNDS_PATH = Path("var/closed_rounds.jsonl")
OUT_PATH = Path("var/cutoff_spot_prices.jsonl")

# ---------------------------------------------------------------------------
# Config — analysis (change these to experiment without refetching)
# ---------------------------------------------------------------------------
CUTOFF_SECONDS = 17       # decision time = lockAt - this
LOOKBACK_SECONDS = 60     # momentum window
THRESHOLD = 0.0001

# ---------------------------------------------------------------------------
# OKX fetch
# ---------------------------------------------------------------------------

def fetch_1s_klines(close_at_ms: int) -> list[list] | None:
    """Fetch KLINES_PER_ROUND 1s klines ending at close_at_ms.

    Returns list of [ts_ms, open, high, low, close, volume] sorted
    oldest-first, or None on failure.
    """
    after_ms = close_at_ms + 1000
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

        rows = body["data"]  # newest first
        if len(rows) < KLINES_PER_ROUND * 0.9:
            continue

        # Convert to compact format, oldest first
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    lines = ROUNDS_PATH.read_text().splitlines()
    rounds = [json.loads(l) for l in lines[-NUM_ROUNDS:]]
    print(f"Loaded {len(rounds)} rounds  (epochs {rounds[0]['epoch']}..{rounds[-1]['epoch']})")

    # Resume support
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
                    "close_at": rnd["closeAt"],
                    "klines_1s": None,
                    "error": True,
                }
            else:
                rec = {
                    "epoch": epoch,
                    "lock_at": rnd["lockAt"],
                    "close_at": rnd["closeAt"],
                    "klines_1s": klines,
                    "error": False,
                }

            f.write(json.dumps(rec) + "\n")
            fetched += 1

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(remaining)}  (errors={errors})")

            time.sleep(REQUEST_DELAY)

    print(f"\nDone: {fetched} fetched, {errors} errors")
    print(f"Output: {OUT_PATH}")
    analyse()


# ---------------------------------------------------------------------------
# Analysis — rerun with different constants without refetching
# ---------------------------------------------------------------------------

def _find_closest_kline(klines_1s: list[list], target_ms: int) -> list | None:
    """Return the 1s kline whose ts is closest to target_ms."""
    best = None
    best_dist = float("inf")
    for k in klines_1s:
        dist = abs(k[0] - target_ms)
        if dist < best_dist:
            best_dist = dist
            best = k
    if best is not None and best_dist <= 2000:  # within 2s tolerance
        return best
    return None


def analyse() -> None:
    records: list[dict] = []
    for line in OUT_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("error"):
            records.append(rec)

    rounds_by_epoch: dict[int, dict] = {}
    for line in ROUNDS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rounds_by_epoch[r["epoch"]] = r

    total = 0
    bets = 0
    wins = 0
    wins_bull = 0
    wins_bear = 0
    bets_bull = 0
    bets_bear = 0
    skips = 0

    for rec in records:
        rnd = rounds_by_epoch.get(rec["epoch"])
        if not rnd or rnd.get("failed") or rnd.get("position") not in ("Bull", "Bear"):
            continue

        klines_1s = rec["klines_1s"]
        lock_at_ms = rec["lock_at"] * 1000
        cutoff_ms = lock_at_ms - CUTOFF_SECONDS * 1000
        ago_ms = cutoff_ms - LOOKBACK_SECONDS * 1000

        k_now = _find_closest_kline(klines_1s, cutoff_ms)
        k_ago = _find_closest_kline(klines_1s, ago_ms)

        if k_now is None or k_ago is None:
            continue

        total += 1
        spot_now = k_now[4]   # close price
        spot_ago = k_ago[4]   # close price

        if spot_ago == 0:
            skips += 1
            continue

        ret = (spot_now / spot_ago) - 1

        if ret > THRESHOLD:
            direction = "Bull"
        elif ret < -THRESHOLD:
            direction = "Bear"
        else:
            skips += 1
            continue

        bets += 1
        outcome = rnd["position"]
        if direction == "Bull":
            bets_bull += 1
        else:
            bets_bear += 1
        if direction == outcome:
            wins += 1
            if direction == "Bull":
                wins_bull += 1
            else:
                wins_bear += 1

    print("\n" + "=" * 50)
    print("SIGNAL QUALITY — 1s klines")
    print("=" * 50)
    print(f"  ret = spot(cutoff) / spot(cutoff - {LOOKBACK_SECONDS}s) - 1")
    print(f"  cutoff = lockAt - {CUTOFF_SECONDS}s")
    print(f"  threshold = {THRESHOLD}")
    print("=" * 50)
    print(f"Rounds analysed:  {total}")
    print(f"Skips (|ret| <= threshold): {skips}")
    print(f"Bets:             {bets}  ({bets_bull} Bull, {bets_bear} Bear)")
    print(f"Wins:             {wins}  ({wins_bull} Bull, {wins_bear} Bear)")
    if bets > 0:
        wr = wins / bets
        print(f"Win rate:         {wr:.4f}  ({wr*100:.2f}%)")
    if bets_bull > 0:
        print(f"  Bull win rate:  {wins_bull/bets_bull:.4f}  ({wins_bull/bets_bull*100:.2f}%)")
    if bets_bear > 0:
        print(f"  Bear win rate:  {wins_bear/bets_bear:.4f}  ({wins_bear/bets_bear*100:.2f}%)")
    if total > 0:
        print(f"Bet rate:         {bets/total:.4f}  ({bets/total*100:.2f}%)")


if __name__ == "__main__":
    main()
