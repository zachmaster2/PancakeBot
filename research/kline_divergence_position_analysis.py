"""Position-wise comparison: live OKX /candles vs history /history-candles.

For each of the 134 captured rounds, align the live and history kline
windows by timestamp and compute per-candle, per-OHLC field deltas.
Aggregate by window position (newest = -1, oldest = -31) to test the
hypothesis: "only the newest candle differs" (publish-lag) vs broader
systematic divergence.

Usage:
    python research/kline_divergence_position_analysis.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pancakebot.market_data.kline_store import KlineStore  # noqa: E402

CAPTURE_PATH = REPO_ROOT / "var" / "dry" / "captured_klines.jsonl"
HIST_PATH = REPO_ROOT / "var" / "btc_spot_prices.jsonl"

DIVERGE_EPOCHS = [475653, 475660, 475711]


def main() -> int:
    # Load captures (epoch -> {klines_btc, cutoff_ms, ...})
    captures: dict[int, dict] = {}
    for line in CAPTURE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ep = int(rec["epoch"])
        if rec.get("klines_btc"):
            captures[ep] = rec

    # Load history (epoch -> klines_1s arrays [[ts,o,h,l,c,v],...])
    hist_store = KlineStore(str(HIST_PATH))
    history: dict[int, list[list]] = {
        r["epoch"]: r["klines_1s"] for r in hist_store.iter_records()
    }

    # Per-position deltas. Key = position from newest (e.g. -1 = newest, -2 = second-newest).
    # Value = list of (ts, c_live, c_hist, o_live, o_hist, h_live, h_hist, l_live, l_hist).
    pos_deltas: dict[int, list[dict]] = defaultdict(list)

    aligned = 0
    skipped = 0
    for ep, cap in sorted(captures.items()):
        cap_klines = cap["klines_btc"]
        hist_klines = history.get(ep)
        if not hist_klines:
            skipped += 1
            continue
        # Build timestamp -> kline maps
        cap_by_ts = {int(k[0]): k for k in cap_klines}
        hist_by_ts = {int(k[0]): k for k in hist_klines}
        # Window cutoff
        cutoff_ms = int(cap["cutoff_ms"])
        # Use the captured window's timestamps (they're the live fetch's view).
        # Sort by ts ascending; newest is the last element.
        ts_sorted = sorted(t for t in cap_by_ts.keys() if t < cutoff_ms)
        ts_sorted = ts_sorted[-31:]  # last 31 (the gate's window)
        if len(ts_sorted) < 31:
            skipped += 1
            continue
        # For each ts in the window, position from newest (-1 = newest, -2 = 2nd newest, etc.)
        for offset, ts in enumerate(ts_sorted, start=1):  # offset=1 means oldest in our 31-window
            pos = -(31 - offset + 1)  # pos -31 (oldest) ... -1 (newest)
            cap_k = cap_by_ts[ts]
            hist_k = hist_by_ts.get(ts)
            if hist_k is None:
                continue
            # [ts, o, h, l, c, v]
            pos_deltas[pos].append({
                "epoch": ep, "ts": ts,
                "o_live": cap_k[1], "o_hist": hist_k[1],
                "h_live": cap_k[2], "h_hist": hist_k[2],
                "l_live": cap_k[3], "l_hist": hist_k[3],
                "c_live": cap_k[4], "c_hist": hist_k[4],
            })
        aligned += 1

    print(f"aligned: {aligned} rounds")
    print(f"skipped: {skipped} rounds (missing history or fewer than 31 candles)")
    print()

    # Aggregate by position
    print(f"{'pos':<5} {'n':>5} {'%diff':>8} {'mean$':>10} {'p50$':>10} {'p95$':>10} {'max$':>10}")
    print("-" * 60)
    for pos in sorted(pos_deltas.keys()):
        rows = pos_deltas[pos]
        diffs = [abs(r["c_live"] - r["c_hist"]) for r in rows]
        nonzero = [d for d in diffs if d > 0.0001]  # ignore floating-point noise
        n = len(diffs)
        pct_diff = 100 * len(nonzero) / n if n else 0
        mean_d = statistics.mean(diffs) if diffs else 0
        p50 = statistics.median(diffs) if diffs else 0
        p95 = sorted(diffs)[int(0.95 * n) - 1] if n >= 20 else max(diffs) if diffs else 0
        mx = max(diffs) if diffs else 0
        print(f"{pos:<5} {n:>5} {pct_diff:>7.2f}% {mean_d:>10.4f} {p50:>10.4f} {p95:>10.4f} {mx:>10.4f}")

    # Same for OHL fields, just at the newest position to check whether it's only close
    print("\n--- newest-candle (pos=-1) field-by-field divergence ---")
    rows = pos_deltas[-1]
    for field in ("o", "h", "l", "c"):
        diffs = [abs(r[f"{field}_live"] - r[f"{field}_hist"]) for r in rows]
        nonzero = [d for d in diffs if d > 0.0001]
        n = len(diffs)
        pct = 100 * len(nonzero) / n if n else 0
        mx = max(diffs) if diffs else 0
        mean_d = statistics.mean(diffs) if diffs else 0
        print(f"  {field}: nonzero {len(nonzero)}/{n} ({pct:.2f}%)  mean=${mean_d:.4f}  max=${mx:.4f}")

    # The 3 known-divergent rounds, full position-wise dump
    print(f"\n--- known-divergent rounds, position-wise ---")
    for ep in DIVERGE_EPOCHS:
        cap = captures.get(ep)
        if not cap: continue
        hist_klines = history.get(ep, [])
        cap_by_ts = {int(k[0]): k for k in cap["klines_btc"]}
        hist_by_ts = {int(k[0]): k for k in hist_klines}
        cutoff_ms = int(cap["cutoff_ms"])
        ts_sorted = sorted(t for t in cap_by_ts.keys() if t < cutoff_ms)[-31:]
        print(f"\nepoch {ep} cutoff_ms={cutoff_ms}")
        print(f"  {'pos':>4} {'ts':>14} {'live_c':>10} {'hist_c':>10} {'diff$':>9}")
        for offset, ts in enumerate(ts_sorted, start=1):
            pos = -(31 - offset + 1)
            c_live = cap_by_ts[ts][4]
            c_hist = hist_by_ts.get(ts, [None]*6)[4] if hist_by_ts.get(ts) else None
            diff = (c_live - c_hist) if c_hist is not None else None
            mark = "  <-- DIVERGE" if diff is not None and abs(diff) > 0.5 else ""
            diff_str = f"{diff:+.4f}" if diff is not None else "n/a"
            hist_str = f"{c_hist:.4f}" if c_hist is not None else "missing"
            print(f"  {pos:>4} {ts:>14} {c_live:>10.4f} {hist_str:>10} {diff_str:>9}{mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
