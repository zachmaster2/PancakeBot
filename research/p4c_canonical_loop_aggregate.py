"""Aggregate var/extended/okx_staleness_probe.jsonl into per-symbol + pooled stats.

Reads the JSONL produced by p4c_canonical_loop_probe.py and computes:
  - n, mean, median, p90, p95, p99, max
  - per-symbol RTT (btc/eth/sol/bnb)
  - pooled all-symbols RTT
  - round-level gate_elapsed_ms
  - first-try-round rate
  - skip_reason distribution
  - skew distribution (and drift)

Usage::
    py research/p4c_canonical_loop_aggregate.py [path/to/jsonl]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT = _REPO_ROOT / "var" / "extended" / "okx_staleness_probe.jsonl"


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = min(int(pct * n / 100.0), n - 1)
    return sorted_values[idx]


def _summarize(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: (no data)"
    s = sorted(values)
    return (
        f"{label:<32} n={len(values):>4} "
        f"mean={statistics.mean(values):>7.0f}ms "
        f"p50={_percentile(s, 50):>5.0f}ms "
        f"p90={_percentile(s, 90):>5.0f}ms "
        f"p95={_percentile(s, 95):>5.0f}ms "
        f"p99={_percentile(s, 99):>5.0f}ms "
        f"max={max(values):>5.0f}ms"
    )


def main(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: not found: {path}")
        return 1

    rows: list[dict] = []
    errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                errors += 1
                continue
            rows.append(obj)

    if not rows:
        print(f"# {path}: no valid rows ({errors} errors)")
        return 1

    print(f"# {path}")
    print(f"# valid rows: {len(rows)}; errors: {errors}")
    print()

    print("=== Round-level gate.evaluate elapsed (ms) ===")
    gate_elapsed = [r["gate_elapsed_ms"] for r in rows if r.get("gate_elapsed_ms") is not None]
    print(_summarize("gate_elapsed_ms (round-level)", gate_elapsed))
    print()

    print("=== Per-symbol RTT (ms) ===")
    for sym in ("btc", "eth", "sol", "bnb"):
        col = f"{sym}_ms"
        vals = [r[col] for r in rows if r.get(col) is not None]
        print(_summarize(f"{sym.upper():<4} ({col})", vals))
    print()

    print("=== Pooled all-symbols RTT (ms) ===")
    pooled = []
    for r in rows:
        for sym in ("btc", "eth", "sol", "bnb"):
            v = r.get(f"{sym}_ms")
            if v is not None:
                pooled.append(v)
    print(_summarize("pooled (all-symbols)", pooled))
    print()

    print("=== First-try round rate ===")
    n = len(rows)
    first_try = sum(1 for r in rows if r.get("first_try_round"))
    print(f"first_try_round: {first_try}/{n} = {100*first_try/n:.1f}%")
    print(f"retry-needed:    {n-first_try}/{n} = {100*(n-first_try)/n:.1f}%")
    print()

    print("=== Skip-reason distribution ===")
    skips = Counter(r.get("skip_reason", "(none)") for r in rows)
    for reason, count in skips.most_common():
        print(f"  {str(reason):<40} {count:>4} ({100*count/n:.1f}%)")
    print()

    print("=== Skew distribution (s) ===")
    skews = [r.get("skew_s", 0.0) * 1000 for r in rows]  # convert to ms
    if skews:
        s = sorted(skews)
        print(f"skew_ms: n={len(skews)} "
              f"min={min(skews):.0f} p50={_percentile(s, 50):.0f} "
              f"p95={_percentile(s, 95):.0f} max={max(skews):.0f}")
        # Drift: max - min over the run
        drift = max(skews) - min(skews)
        print(f"skew drift over run: {drift:.0f} ms (max-min)")
    print()

    # Wake-time vs candle-close: how much post-close do we wake?
    if rows:
        post_close_at_wake_ms = [
            r["wake_okx_ms"] - r["candle_close_okx_ms"]
            for r in rows
            if r.get("wake_okx_ms") and r.get("candle_close_okx_ms")
        ]
        print("=== wake-time post candle-close (ms) ===")
        print(_summarize("wake - candle_close", post_close_at_wake_ms))

    return 0


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
    sys.exit(main(p))
