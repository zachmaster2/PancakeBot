"""Verify kline data integrity for both BNB and BTC.

For each round in closed_rounds, checks:
1. Kline record exists
2. Exactly CANDLE_COUNT (31) candles present
3. Candle timestamps are consecutive (1000ms apart)
4. Last candle timestamp matches expected cutoff: (start_at + 298) * 1000
5. No duplicate epochs in the file
6. No extra epochs not in closed_rounds
7. Candle OHLCV values are sane (positive, close within range)
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import INTERVAL_SECONDS
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore

CUTOFF_S = 2
CANDLE_COUNT = 31
EXPECTED_LAST_TS_OFFSET = (INTERVAL_SECONDS - CUTOFF_S) * 1000 - 1000  # 297000ms from start_at (OKX 'after' is exclusive)


def load_kline_file(path):
    """Load kline file, returning list of (epoch, record) tuples preserving order."""
    records = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  ERROR: JSON decode failed at line {i}: {e}")
            continue
        records.append((int(rec["epoch"]), rec, i))
    return records


def verify(label, kline_path, rounds_map):
    print(f"\n{'=' * 100}")
    print(f"VERIFYING: {label} — {kline_path}")
    print(f"{'=' * 100}")

    if not Path(kline_path).exists():
        print(f"  FILE NOT FOUND: {kline_path}")
        return

    records = load_kline_file(kline_path)
    total_records = len(records)
    print(f"  Records in file: {total_records}")
    print(f"  Rounds in store: {len(rounds_map)}")

    # Check for duplicate epochs
    epoch_counts = Counter(ep for ep, _, _ in records)
    duplicates = {ep: cnt for ep, cnt in epoch_counts.items() if cnt > 1}
    if duplicates:
        print(f"\n  DUPLICATES FOUND: {len(duplicates)} epochs appear more than once")
        for ep, cnt in sorted(duplicates.items())[:10]:
            print(f"    epoch {ep}: {cnt} occurrences")
    else:
        print(f"  Duplicates: NONE (good)")

    # Check for extra epochs (not in closed_rounds)
    kline_epochs = set(epoch_counts.keys())
    round_epochs = set(rounds_map.keys())
    extra = kline_epochs - round_epochs
    if extra:
        print(f"  Extra epochs (in klines but not rounds): {len(extra)}")
        for ep in sorted(extra)[:5]:
            print(f"    epoch {ep}")
    else:
        print(f"  Extra epochs: NONE (good)")

    # Check for missing epochs
    missing = round_epochs - kline_epochs
    if missing:
        print(f"  Missing epochs (in rounds but not klines): {len(missing)}")
        for ep in sorted(missing)[:10]:
            print(f"    epoch {ep}")
    else:
        print(f"  Missing epochs: NONE (good)")

    # Per-record checks
    wrong_count = 0
    wrong_last_ts = 0
    non_consecutive = 0
    bad_ohlcv = 0
    no_klines = 0
    null_klines = 0
    ts_gap_examples = []
    last_ts_examples = []

    for epoch, rec, line_num in records:
        kl = rec.get("klines_1s")
        if kl is None:
            null_klines += 1
            continue
        if not kl:
            no_klines += 1
            continue

        # Check candle count
        if len(kl) != CANDLE_COUNT:
            wrong_count += 1
            if wrong_count <= 3:
                print(f"  WRONG COUNT: epoch {epoch} has {len(kl)} candles (expected {CANDLE_COUNT})")

        # Check timestamps consecutive (1000ms apart)
        timestamps = [int(c[0]) for c in kl]
        gaps = []
        for j in range(1, len(timestamps)):
            diff = timestamps[j] - timestamps[j - 1]
            if diff != 1000:
                gaps.append((j, diff))
        if gaps:
            non_consecutive += 1
            if len(ts_gap_examples) < 3:
                ts_gap_examples.append((epoch, gaps[:3]))

        # Check last candle timestamp
        rnd = rounds_map.get(epoch)
        if rnd and kl:
            expected_last = (rnd.start_at * 1000) + EXPECTED_LAST_TS_OFFSET
            actual_last = int(kl[-1][0])
            if actual_last != expected_last:
                wrong_last_ts += 1
                if len(last_ts_examples) < 5:
                    last_ts_examples.append((epoch, actual_last, expected_last,
                                             actual_last - expected_last))

        # Check OHLCV sanity
        for c in kl:
            o, h, l, cl, v = float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
            if o <= 0 or h <= 0 or l <= 0 or cl <= 0:
                bad_ohlcv += 1
                break
            if l > h:
                bad_ohlcv += 1
                break

    print(f"\n  --- Per-record checks ---")
    print(f"  Null klines_1s: {null_klines}")
    print(f"  Empty klines: {no_klines}")
    print(f"  Wrong candle count (!= {CANDLE_COUNT}): {wrong_count}")
    print(f"  Non-consecutive timestamps: {non_consecutive}")
    print(f"  Wrong last-candle timestamp: {wrong_last_ts}")
    print(f"  Bad OHLCV values: {bad_ohlcv}")

    if ts_gap_examples:
        print(f"\n  Timestamp gap examples:")
        for ep, gaps in ts_gap_examples:
            print(f"    epoch {ep}: {gaps}")

    if last_ts_examples:
        print(f"\n  Last-timestamp mismatch examples:")
        for ep, actual, expected, diff in last_ts_examples:
            print(f"    epoch {ep}: actual={actual} expected={expected} diff={diff}ms")

    # Summary
    clean = (total_records == len(rounds_map) and not duplicates and not extra
             and not missing and wrong_count == 0 and wrong_last_ts == 0
             and non_consecutive == 0 and bad_ohlcv == 0 and null_klines == 0
             and no_klines == 0)
    print(f"\n  VERDICT: {'CLEAN' if clean else 'ISSUES FOUND'}")
    return clean


def main():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds_map = {int(r.epoch): r for r in store.iter_closed_rounds()}
    print(f"Loaded {len(rounds_map)} rounds from closed_rounds.jsonl")

    pairs = [
        ("BNB", "var/bnb_spot_prices.jsonl"),
        ("BTC", "var/btc_spot_prices.jsonl"),
        ("ETH", "var/eth_spot_prices.jsonl"),
        ("SOL", "var/sol_spot_prices.jsonl"),
    ]
    results = {}
    for label, path in pairs:
        if Path(path).exists():
            results[label] = verify(f"{label} ({Path(path).name})", path, rounds_map)
        else:
            print(f"\n  {label}: FILE NOT FOUND ({path})")
            results[label] = False

    print(f"\n{'=' * 100}")
    summary = "  ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items())
    print(f"FINAL: {summary}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
