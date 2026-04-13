"""Trim stored klines to the corrected cutoff (start_at + 296) and 31 candles.

The old klines were fetched at Graph's lockAt - 4 = start_at + 302.
The correct cutoff is start_at + 300 - 4 = start_at + 296.
This script trims each kline record to 31 candles ending at the correct cutoff.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import INTERVAL_SECONDS
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore

CUTOFF_SECONDS = 4
CANDLE_COUNT = 31


def main():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = {int(r.epoch): r for r in store.iter_closed_rounds()}
    print(f"Loaded {len(rounds)} rounds")

    for label, path in [("BNB spot", "var/cutoff_spot_prices.jsonl"),
                         ("BTC spot", "var/btc_spot_prices.jsonl")]:
        print(f"\n=== {label}: {path} ===")
        records = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))

        print(f"  Records: {len(records)}")

        trimmed_count = 0
        out_records = []

        for rec in records:
            epoch = int(rec["epoch"])
            klines = rec.get("klines_1s")
            if klines is None:
                out_records.append(rec)
                continue

            rnd = rounds.get(epoch)
            if rnd is None:
                out_records.append(rec)
                continue

            # Correct cutoff: start_at + INTERVAL_SECONDS - CUTOFF_SECONDS
            correct_cutoff_ms = (rnd.start_at + INTERVAL_SECONDS - CUTOFF_SECONDS) * 1000

            # Filter: keep candles with open_time < correct_cutoff_ms
            before = [k for k in klines if int(k[0]) < correct_cutoff_ms]
            trimmed = before[-CANDLE_COUNT:] if len(before) > CANDLE_COUNT else before

            if len(trimmed) != len(klines):
                trimmed_count += 1

            rec["klines_1s"] = trimmed
            # Update lock_at in the kline record to match the correct value
            rec["lock_at"] = rnd.start_at + INTERVAL_SECONDS
            out_records.append(rec)

        print(f"  Trimmed: {trimmed_count}/{len(records)} records")

        # Check candle counts
        counts = {}
        for rec in out_records:
            kl = rec.get("klines_1s")
            n = len(kl) if kl else 0
            counts[n] = counts.get(n, 0) + 1
        print(f"  Candle count distribution: {dict(sorted(counts.items()))}")

        # Write back
        with open(path, "w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        print(f"  Written: {len(out_records)} records to {path}")


if __name__ == "__main__":
    main()
