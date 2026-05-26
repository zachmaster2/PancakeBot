"""Step 20 — external signal sources, free options.

20A: Google Trends daily search volume for "bitcoin" and "crypto" via pytrends.
20B: Reddit historical — NOT VIABLE for our backtest range (documented).
20C: OKX BNB-USDT-SWAP perp funding rate via public REST (substitutes for
     geo-blocked binance.com fapi).

Per-cohort means + Cohen's d (extension vs CV5).
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore
import requests  # type: ignore

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
import research.in_process_runner as ipr  # noqa: E402
ipr._EXT_CLOSED_ROUNDS_PATH = EXT_DIR / "closed_rounds.jsonl"

# Backtest range corresponds to extension+cv5+...+post_fresh
BACKTEST_START = datetime(2025, 10, 1, tzinfo=timezone.utc)
BACKTEST_END = datetime(2026, 5, 26, tzinfo=timezone.utc)

COHORT_DEFS = [
    ("extension", 422298, 437561),
    ("cv5", 437562, 474086),
    ("gap_post_cv5_pre_holdout", 474087, 474879),
    ("holdout", 474880, 475311),
    ("ext_v2", 475312, 479952),
    ("fresh_oos", 479953, 483191),
    ("post_fresh", 483192, 999999),
]
COHORT_ORDER = [c[0] for c in COHORT_DEFS]


def cohort_of(epoch: int) -> str:
    for name, lo, hi in COHORT_DEFS:
        if lo <= epoch <= hi:
            return name
    return "unknown"


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    am, bm = float(np.mean(a_arr)), float(np.mean(b_arr))
    av, bv = float(np.var(a_arr, ddof=1)), float(np.var(b_arr, ddof=1))
    pooled = math.sqrt(((len(a) - 1) * av + (len(b) - 1) * bv) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (am - bm) / pooled


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "p10": None, "p50": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(arr)),
        "stdev": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


# ============================================================
# 20A: Google Trends
# ============================================================

def fetch_google_trends() -> dict[str, list[tuple[str, int]]]:
    """Daily search volume per keyword. Each entry: (date_iso, value 0-100)."""
    from pytrends.request import TrendReq  # lazy import in case of network
    timeframe = f"{BACKTEST_START.strftime('%Y-%m-%d')} {BACKTEST_END.strftime('%Y-%m-%d')}"
    print(f"  fetching Google Trends for {timeframe}", flush=True)
    series = {}
    pytrends = TrendReq(hl='en-US', tz=0, timeout=(10, 25),
                         retries=2, backoff_factor=1.0)
    for kw in ["bitcoin", "crypto"]:
        t = time.time()
        pytrends.build_payload([kw], timeframe=timeframe, geo='', gprop='')
        df = pytrends.interest_over_time()
        if df.empty:
            print(f"    {kw}: EMPTY (rate-limited or no data)", flush=True)
            series[kw] = []
            continue
        rows = [(idx.date().isoformat(), int(row[kw]))
                for idx, row in df.iterrows() if kw in df.columns]
        series[kw] = rows
        print(f"    {kw}: {len(rows)} daily points ({time.time()-t:.1f}s)", flush=True)
        time.sleep(2.0)  # gentle rate-limit pacing
    return series


# ============================================================
# 20C: OKX BNB-USDT-SWAP funding
# ============================================================

def fetch_okx_funding() -> list[dict[str, Any]]:
    """Paginated fetch of BNB-USDT-SWAP funding history covering backtest range."""
    start_ms = int(BACKTEST_START.timestamp() * 1000)
    end_ms = int(BACKTEST_END.timestamp() * 1000)
    base = "https://www.okx.com/api/v5/public/funding-rate-history"
    inst = "BNB-USDT-SWAP"
    all_rows: list[dict[str, Any]] = []
    last_oldest_ts = None
    pages = 0
    while True:
        params = {"instId": inst, "limit": "100"}
        if last_oldest_ts is not None:
            # OKX semantics for funding-rate-history:
            #   `before` returns NEWER records than fundingTime
            #   `after`  returns OLDER records than fundingTime
            # We iterate newest -> oldest, so use `after=oldest_so_far`.
            params["after"] = str(last_oldest_ts)
        r = requests.get(base, params=params, timeout=10)
        if r.status_code != 200:
            print(f"    OKX HTTP {r.status_code}: {r.text[:200]}", flush=True)
            break
        payload = r.json()
        data = payload.get("data", [])
        if not data:
            break
        pages += 1
        oldest_in_batch = int(data[-1]["fundingTime"])
        for d in data:
            ts = int(d["fundingTime"])
            if ts < start_ms - 86400_000:
                continue  # margin of 1 day for window edge
            all_rows.append({
                "ts_ms": ts,
                "ts_iso": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                "rate": float(d["fundingRate"]),
                "realized_rate": float(d.get("realizedRate", d["fundingRate"])),
            })
        # Stop if we've gone before backtest range
        if oldest_in_batch < start_ms:
            break
        if last_oldest_ts is not None and oldest_in_batch >= last_oldest_ts:
            break  # no progress, avoid infinite loop
        last_oldest_ts = oldest_in_batch
        time.sleep(0.25)  # OKX rate-limit friendly
    all_rows.sort(key=lambda x: x["ts_ms"])
    # Dedupe by ts
    seen = set()
    dedup = []
    for r in all_rows:
        if r["ts_ms"] in seen:
            continue
        seen.add(r["ts_ms"])
        dedup.append(r)
    print(f"    OKX returned {len(dedup)} unique funding intervals across {pages} pages", flush=True)
    return dedup


# ============================================================
# Per-cohort analysis
# ============================================================

def assign_trends_to_rounds(trends_series, all_rounds):
    """Map round → (btc_z, crypto_z) Z-scores based on round's date."""
    out_by_kw = {}
    for kw, rows in trends_series.items():
        if not rows:
            out_by_kw[kw] = {}
            continue
        vals = np.array([v for _, v in rows], dtype=float)
        mean = float(vals.mean()); stdev = float(vals.std(ddof=1)) if len(vals) > 1 else 1.0
        date_to_z = {}
        for d_iso, v in rows:
            z = (v - mean) / (stdev if stdev > 0 else 1.0)
            date_to_z[d_iso] = z
        out_by_kw[kw] = date_to_z

    # For each round, look up its date's Z-score per keyword
    round_to_z = {}
    for r in all_rounds:
        ep = int(r.epoch)
        if ep < COHORT_DEFS[0][1] or ep > COHORT_DEFS[-1][2]:
            continue
        date_iso = datetime.fromtimestamp(int(r.start_at), tz=timezone.utc).date().isoformat()
        z_btc = out_by_kw.get("bitcoin", {}).get(date_iso)
        z_crypto = out_by_kw.get("crypto", {}).get(date_iso)
        round_to_z[ep] = {"btc_z": z_btc, "crypto_z": z_crypto, "date": date_iso}
    return round_to_z


def assign_funding_to_rounds(funding_rows, all_rounds):
    """Map round → most-recent funding rate as of round's start_at."""
    if not funding_rows:
        return {}
    ts_arr = np.array([r["ts_ms"] for r in funding_rows], dtype=np.int64)
    rate_arr = np.array([r["rate"] for r in funding_rows], dtype=float)
    out = {}
    for r in all_rounds:
        ep = int(r.epoch)
        if ep < COHORT_DEFS[0][1] or ep > COHORT_DEFS[-1][2]:
            continue
        round_ts_ms = int(r.start_at) * 1000
        idx = int(np.searchsorted(ts_arr, round_ts_ms, side="right")) - 1
        if idx < 0:
            continue
        out[ep] = {
            "funding_rate": float(rate_arr[idx]),
            "abs_funding_rate": float(abs(rate_arr[idx])),
            "funding_ts_ms": int(ts_arr[idx]),
        }
    return out


def per_cohort_summary(round_to_value, key) -> dict[str, dict[str, Any]]:
    per_cohort = {c: [] for c in COHORT_ORDER}
    for ep, d in round_to_value.items():
        coh = cohort_of(ep)
        v = d.get(key)
        if v is not None:
            per_cohort[coh].append(float(v))
    return {c: summarize(vals) for c, vals in per_cohort.items()}, per_cohort


# ============================================================
# Main
# ============================================================

def main():
    t_all = time.time()
    print("--- loading rounds (cohort assignment only, no klines needed) ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  {len(all_rounds)} rounds", flush=True)

    # Filter to backtest range
    sim_rounds = [r for r in all_rounds
                   if COHORT_DEFS[0][1] <= int(r.epoch) <= COHORT_DEFS[-1][2]]
    print(f"  {len(sim_rounds)} rounds in backtest range", flush=True)

    # Cohort populations
    cohort_round_count = {c: 0 for c in COHORT_ORDER}
    for r in sim_rounds:
        cohort_round_count[cohort_of(int(r.epoch))] += 1
    print(f"  cohort populations: {cohort_round_count}", flush=True)

    # === 20A: Google Trends ===
    print("\n=== 20A: Google Trends (daily search volume) ===", flush=True)
    try:
        trends_series = fetch_google_trends()
    except Exception as e:
        print(f"  PYTRENDS FAILURE: {e!r}", flush=True)
        trends_series = {"bitcoin": [], "crypto": []}

    round_to_trends = assign_trends_to_rounds(trends_series, sim_rounds)
    btc_z_summary, btc_z_raw = per_cohort_summary(round_to_trends, "btc_z")
    crypto_z_summary, crypto_z_raw = per_cohort_summary(round_to_trends, "crypto_z")

    print("  btc_z per cohort (mean, stdev):", flush=True)
    for c in COHORT_ORDER:
        s = btc_z_summary[c]
        if s["n"] > 0:
            print(f"    {c:>28s}: n={s['n']:>5d}  mean={s['mean']:+.4f}  stdev={s['stdev']:.4f}", flush=True)
    print("  crypto_z per cohort (mean, stdev):", flush=True)
    for c in COHORT_ORDER:
        s = crypto_z_summary[c]
        if s["n"] > 0:
            print(f"    {c:>28s}: n={s['n']:>5d}  mean={s['mean']:+.4f}  stdev={s['stdev']:.4f}", flush=True)

    # === 20C: OKX BNB-USDT-SWAP funding ===
    print("\n=== 20C: OKX BNB-USDT-SWAP funding rate ===", flush=True)
    try:
        funding_rows = fetch_okx_funding()
    except Exception as e:
        print(f"  OKX FAILURE: {e!r}", flush=True)
        funding_rows = []

    round_to_funding = assign_funding_to_rounds(funding_rows, sim_rounds)
    funding_summary, funding_raw = per_cohort_summary(round_to_funding, "funding_rate")
    abs_funding_summary, abs_funding_raw = per_cohort_summary(round_to_funding, "abs_funding_rate")

    print("  funding_rate per cohort (mean, stdev):", flush=True)
    for c in COHORT_ORDER:
        s = funding_summary[c]
        if s["n"] > 0:
            print(f"    {c:>28s}: n={s['n']:>5d}  mean={s['mean']:+.6e}  stdev={s['stdev']:.4e}", flush=True)
    print("  abs(funding_rate) per cohort (mean, stdev):", flush=True)
    for c in COHORT_ORDER:
        s = abs_funding_summary[c]
        if s["n"] > 0:
            print(f"    {c:>28s}: n={s['n']:>5d}  mean={s['mean']:+.6e}  stdev={s['stdev']:.4e}", flush=True)

    # === Cohen's d extension vs CV5 ===
    print("\n=== Effect sizes (extension vs CV5) ===", flush=True)
    test_results: list[dict[str, Any]] = []

    for label, raw in [
        ("20A_btc_z", btc_z_raw), ("20A_crypto_z", crypto_z_raw),
        ("20C_funding_rate", funding_raw), ("20C_abs_funding_rate", abs_funding_raw),
    ]:
        ext = raw.get("extension", [])
        cv5 = raw.get("cv5", [])
        if len(ext) < 2 or len(cv5) < 2:
            test_results.append({"feature": label, "ext_mean": None, "cv5_mean": None,
                                  "cohens_d": None, "n_ext": len(ext), "n_cv5": len(cv5)})
            continue
        d = cohens_d(ext, cv5)
        test_results.append({
            "feature": label, "ext_mean": float(np.mean(ext)), "cv5_mean": float(np.mean(cv5)),
            "cohens_d": d, "n_ext": len(ext), "n_cv5": len(cv5),
        })

    print(f"  {'feature':>22s}  {'ext_mean':>12s}  {'cv5_mean':>12s}  {'cohens_d':>10s}", flush=True)
    test_results_sorted = sorted(test_results, key=lambda x: -abs(x["cohens_d"] or 0))
    for tr in test_results_sorted:
        em = f"{tr['ext_mean']:+12.4e}" if tr["ext_mean"] is not None else "         n/a"
        cm = f"{tr['cv5_mean']:+12.4e}" if tr["cv5_mean"] is not None else "         n/a"
        d_str = f"{tr['cohens_d']:+10.4f}" if tr["cohens_d"] is not None else "       n/a"
        print(f"  {tr['feature']:>22s}  {em}  {cm}  {d_str}", flush=True)

    # Save
    out_path = REPO / "var" / "strategy_review" / "step20_external_signals_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "backtest_start": BACKTEST_START.isoformat(),
                "backtest_end": BACKTEST_END.isoformat(),
                "cohort_defs": [list(c) for c in COHORT_DEFS],
            },
            "20A_google_trends": {
                "raw_series": trends_series,
                "btc_z_per_cohort": btc_z_summary,
                "crypto_z_per_cohort": crypto_z_summary,
                "n_rounds_mapped": len(round_to_trends),
            },
            "20C_okx_funding": {
                "raw_funding": funding_rows,
                "funding_rate_per_cohort": funding_summary,
                "abs_funding_rate_per_cohort": abs_funding_summary,
                "n_rounds_mapped": len(round_to_funding),
            },
            "test_results_sorted": test_results_sorted,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
