"""Kline reliability deep-dive analysis (Bundle 3 research).

Snapshots cycle_audit.csv, filters to rows since bc4b4ee (2026-05-12 15:41 UTC),
and produces:
  - per-symbol latency percentiles (p50/p90/p95/p99/max) via numpy linear interp
  - failure pattern statistics (per-symbol dominance, hour-of-day, clustering,
    pre-failure slow-tail buildup)
  - TLS / connection-error correlation with stdout logs
"""
from __future__ import annotations

import csv
import datetime as dt
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/Users/zking/Documents/GitHub/PancakeBot")
CSV_PATH = ROOT / "var/dry/cycle_audit.csv"
LOG_DIR = ROOT / "var/dry/logs"
SNAPSHOT = ROOT / "var/strategy_review/.kline_reliability_snapshot.csv"

# bc4b4ee landed 2026-05-12 15:41 UTC (per task notes). The audit columns
# became non-empty from that point.
CUTOFF_TS = int(dt.datetime(2026, 5, 12, 15, 41, tzinfo=dt.timezone.utc).timestamp())


def snapshot_csv() -> None:
    shutil.copy2(CSV_PATH, SNAPSHOT)


def load_rows() -> list[dict]:
    with open(SNAPSHOT, newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    out = []
    for r in rows:
        try:
            lock_ts = int(r["lock_ts"])
        except (TypeError, ValueError):
            continue
        if lock_ts < CUTOFF_TS:
            continue
        out.append(r)
    return out


def parse_ms(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pctile(arr: list[float], q: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(np.asarray(arr, dtype=float), q, method="linear"))


def stats_block(name: str, vals: list[float]) -> dict:
    if not vals:
        return {"name": name, "n": 0}
    arr = np.asarray(vals)
    return {
        "name": name,
        "n": len(vals),
        "p50": pctile(vals, 50),
        "p90": pctile(vals, 90),
        "p95": pctile(vals, 95),
        "p99": pctile(vals, 99),
        "max": float(arr.max()),
        "slow_gt_500": int((arr > 500).sum()),
        "slow_gt_1000": int((arr > 1000).sum()),
    }


def main() -> None:
    snapshot_csv()
    rows = load_rows()
    print(f"# rows since bc4b4ee cutoff ({CUTOFF_TS}): {len(rows)}")
    if rows:
        first = int(rows[0]["lock_ts"])
        last = int(rows[-1]["lock_ts"])
        print(f"# period: {dt.datetime.fromtimestamp(first, dt.timezone.utc)} -> "
              f"{dt.datetime.fromtimestamp(last, dt.timezone.utc)}")
        print(f"# duration: {(last - first) / 3600:.1f} hours")

    btc, eth, sol = [], [], []
    btc_fail_partial = eth_fail_partial = sol_fail_partial = 0

    fails = []
    skips_breakdown: dict[str, int] = {}
    action_breakdown: dict[str, int] = {}

    for r in rows:
        action_breakdown[r["action"]] = action_breakdown.get(r["action"], 0) + 1
        sr = r["skip_reason"]
        if sr:
            skips_breakdown[sr] = skips_breakdown.get(sr, 0) + 1
        b = parse_ms(r["btc_fetch_ms"])
        e = parse_ms(r["eth_fetch_ms"])
        s = parse_ms(r["sol_fetch_ms"])
        if b is not None:
            btc.append(b)
        if e is not None:
            eth.append(e)
        if s is not None:
            sol.append(s)
        if "kline_fetch_transient_failure" in sr:
            fails.append(r)
            if b is None:
                btc_fail_partial += 1
            if e is None:
                eth_fail_partial += 1
            if s is None:
                sol_fail_partial += 1

    print("\n## action breakdown")
    for k, v in sorted(action_breakdown.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    print("\n## skip_reason breakdown")
    for k, v in sorted(skips_breakdown.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    print("\n## per-symbol latency (non-null fetch_ms)")
    for blk in (stats_block("BTC", btc), stats_block("ETH", eth), stats_block("SOL", sol)):
        if blk["n"] == 0:
            print(f"  {blk['name']}: n=0")
            continue
        print(f"  {blk['name']}: n={blk['n']:4d}  p50={blk['p50']:7.1f}  "
              f"p90={blk['p90']:7.1f}  p95={blk['p95']:7.1f}  "
              f"p99={blk['p99']:7.1f}  max={blk['max']:7.1f}  "
              f">500ms={blk['slow_gt_500']}  >1000ms={blk['slow_gt_1000']}")

    print("\n## failure analysis")
    print(f"  total kline_fetch_transient_failure rows: {len(fails)}")
    print(f"    rows w/ empty btc_fetch_ms: {btc_fail_partial}")
    print(f"    rows w/ empty eth_fetch_ms: {eth_fail_partial}")
    print(f"    rows w/ empty sol_fetch_ms: {sol_fail_partial}")
    # Symbol-dominance: which symbol failed (empty fetch_ms) MOST often?
    # If a fetch raises, all three may be empty. Useful to enumerate exact patterns.
    patterns: dict[tuple, int] = {}
    for r in fails:
        empty = (
            r["btc_fetch_ms"] == "",
            r["eth_fetch_ms"] == "",
            r["sol_fetch_ms"] == "",
        )
        patterns[empty] = patterns.get(empty, 0) + 1
    print("  empty-pattern counts (btc_empty, eth_empty, sol_empty):")
    for k, v in sorted(patterns.items()):
        print(f"    {k}: {v}")

    # Hour-of-day distribution
    if fails:
        hours: dict[int, int] = {}
        for r in fails:
            ts = int(r["lock_ts"])
            h = dt.datetime.fromtimestamp(ts, dt.timezone.utc).hour
            hours[h] = hours.get(h, 0) + 1
        print("  hour-of-day (UTC) histogram of failures:")
        for h in sorted(hours):
            print(f"    {h:02d}h: {'#' * hours[h]} ({hours[h]})")

    # Clustering: chain-length distribution
    print("\n## failure clustering")
    fail_set = {int(r["lock_ts"]) for r in fails}
    lock_seq = [int(r["lock_ts"]) for r in rows]
    # walk through rows in order, count consecutive failure runs
    runs: list[int] = []
    cur = 0
    for ts in lock_seq:
        if ts in fail_set:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    print(f"  chains: {sorted(runs)}")
    if runs:
        print(f"  chain count: {len(runs)}, longest chain: {max(runs)}, "
              f"mean: {sum(runs)/len(runs):.2f}")

    # Pre-failure slow-tail check: median fetch_ms in 3 rounds preceding each fail
    print("\n## pre-failure latency (3 rounds before each failure)")
    by_ts = sorted(rows, key=lambda r: int(r["lock_ts"]))
    ts_to_idx = {int(r["lock_ts"]): i for i, r in enumerate(by_ts)}
    pre_max_ms = []
    pre_p50_ms = []
    for r in fails:
        idx = ts_to_idx[int(r["lock_ts"])]
        window = by_ts[max(0, idx - 3):idx]
        vals = []
        for w in window:
            for col in ("btc_fetch_ms", "eth_fetch_ms", "sol_fetch_ms"):
                v = parse_ms(w[col])
                if v is not None:
                    vals.append(v)
        if vals:
            pre_max_ms.append(max(vals))
            pre_p50_ms.append(float(np.median(vals)))
    if pre_max_ms:
        print(f"  failure events with preceding-3-round data: {len(pre_max_ms)}")
        print(f"  median of pre-window MAX fetch_ms: {float(np.median(pre_max_ms)):.1f}")
        print(f"  median of pre-window MEDIAN fetch_ms: {float(np.median(pre_p50_ms)):.1f}")
        print(f"  per-fail (max, p50): " +
              ", ".join(f"({m:.0f},{p:.0f})" for m, p in zip(pre_max_ms, pre_p50_ms)))

    # Failure timestamps
    print("\n## failure timestamps (lock_ts UTC, hh:mm:ss)")
    for r in fails:
        ts = int(r["lock_ts"])
        print(f"  {dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()}  "
              f"epoch={r['current_epoch']}  decision_latency_ms={r['decision_latency_ms']}")

    # ---------------- LOG SCAN ----------------
    print("\n## log scan for OKX / TLS / connection signatures")
    # Scan recent logs covering the audit window.
    pattern_warn = re.compile(
        r"(WARN GATE|WARN NET|ERROR NET|RequestException|ConnectionResetError|"
        r"ReadTimeout|Connect.*Timeout|SSLError|EOF occurred|RemoteDisconnected|"
        r"BadStatusLine|okx|OKX)",
        re.IGNORECASE,
    )
    log_files = sorted(LOG_DIR.glob("dry-auto-2026051[23]-*.log"))
    print(f"  scanning {len(log_files)} dry-auto logs from 2026-05-12 / 13")
    hits = []
    for lf in log_files:
        try:
            with open(lf, encoding="utf-8", errors="ignore") as fh:
                for ln in fh:
                    if pattern_warn.search(ln):
                        hits.append((lf.name, ln.rstrip()))
        except OSError:
            continue
    print(f"  candidate hits: {len(hits)}")
    # Categorize the hits
    cats: dict[str, int] = {}
    for _, ln in hits:
        if "WARN GATE" in ln:
            cats["WARN GATE"] = cats.get("WARN GATE", 0) + 1
        elif "WARN NET" in ln:
            cats["WARN NET"] = cats.get("WARN NET", 0) + 1
        elif "ERROR NET" in ln:
            cats["ERROR NET"] = cats.get("ERROR NET", 0) + 1
        elif "RequestException" in ln:
            cats["RequestException"] = cats.get("RequestException", 0) + 1
        elif "ConnectionResetError" in ln:
            cats["ConnectionResetError"] = cats.get("ConnectionResetError", 0) + 1
        elif "ReadTimeout" in ln:
            cats["ReadTimeout"] = cats.get("ReadTimeout", 0) + 1
        elif "SSLError" in ln:
            cats["SSLError"] = cats.get("SSLError", 0) + 1
        elif "RemoteDisconnected" in ln:
            cats["RemoteDisconnected"] = cats.get("RemoteDisconnected", 0) + 1
        else:
            cats["other"] = cats.get("other", 0) + 1
    print("  category counts:")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v}")
    # Print up to 30 sample hits
    print("  sample hits (up to 30):")
    for fname, ln in hits[:30]:
        print(f"    [{fname}] {ln[:200]}")


if __name__ == "__main__":
    main()
