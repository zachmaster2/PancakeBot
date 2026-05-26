"""Step 18a — F2 timestamp alignment sanity check.

20 random epochs from extension range + 5 from CV5 (control).
For each: per-asset 24h sample counts, first/last ts offset, mean/stdev
inter-sample interval. Plus a strict-alignment (+/-30s) re-computation
of the cross-asset Pearson correlation to compare against Step 17's
hourly-rolling-average finding.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np  # type: ignore

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
import research.in_process_runner as ipr  # noqa: E402
ipr._EXT_CLOSED_ROUNDS_PATH = EXT_DIR / "closed_rounds.jsonl"
ipr._EXT_BTC_KLINES_PATH = EXT_DIR / "btc_spot_prices.jsonl"
ipr._EXT_ETH_KLINES_PATH = EXT_DIR / "eth_spot_prices.jsonl"
ipr._EXT_SOL_KLINES_PATH = EXT_DIR / "sol_spot_prices.jsonl"

# Reuse the single-pass loader from Step 17
sys.path.insert(0, str(REPO / "research"))
from feature_characterization_step17_2026_05_26 import load_klines_for_step17  # noqa: E402

ASSETS = ("BTC", "ETH", "SOL", "BNB")
EXTENSION_RANGE = (422298, 437561)
CV5_RANGE = (437562, 474086)
N_EXT_SAMPLES = 20
N_CV5_SAMPLES = 5
WINDOW_24H_S = 24 * 3600
ALIGN_TOL_S = 30
ROUND_LOCK_OFFSET_S = 300  # standard PancakeBot round duration
SEED = 42


def main():
    t_all = time.time()
    print("--- Loading rounds + 1-min samples per asset ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    epoch_to_ts = {int(r.epoch): int(r.start_at) for r in all_rounds}
    print(f"  {len(all_rounds)} rounds", flush=True)

    t = time.time()
    bnb_path = REPO / "var" / "bnb_spot_prices.jsonl"
    bnb_ext_path = EXT_DIR / "bnb_spot_prices.jsonl"

    btc_samples, _ = load_klines_for_step17(
        ipr._BTC_KLINES_PATH, ipr._EXT_BTC_KLINES_PATH,
        want_pipeline_slice=False, pipeline_earliest_offset=18, pipeline_latest_offset=3)
    eth_samples, _ = load_klines_for_step17(
        ipr._ETH_KLINES_PATH, ipr._EXT_ETH_KLINES_PATH,
        want_pipeline_slice=False, pipeline_earliest_offset=18, pipeline_latest_offset=3)
    sol_samples, _ = load_klines_for_step17(
        ipr._SOL_KLINES_PATH, ipr._EXT_SOL_KLINES_PATH,
        want_pipeline_slice=False, pipeline_earliest_offset=18, pipeline_latest_offset=3)
    bnb_samples, _ = load_klines_for_step17(
        bnb_path, bnb_ext_path,
        want_pipeline_slice=False, pipeline_earliest_offset=18, pipeline_latest_offset=3)
    print(f"  klines: BTC={len(btc_samples)} ETH={len(eth_samples)} "
          f"SOL={len(sol_samples)} BNB={len(bnb_samples)} ({time.time()-t:.1f}s)", flush=True)

    samples_by_asset = {"BTC": btc_samples, "ETH": eth_samples,
                         "SOL": sol_samples, "BNB": bnb_samples}

    # Build global timeline per asset
    t = time.time()
    timelines = {a: [] for a in ASSETS}
    for asset in ASSETS:
        for _, samples in samples_by_asset[asset].items():
            for ts_sec, price in samples:
                timelines[asset].append((ts_sec, price))
        timelines[asset].sort()
    ts_arr = {a: np.array([ts for ts, _ in timelines[a]], dtype=np.int64) for a in ASSETS}
    px_arr = {a: np.array([px for _, px in timelines[a]], dtype=np.float64) for a in ASSETS}
    for a in ASSETS:
        print(f"  {a} timeline: {len(ts_arr[a])} points", flush=True)
    print(f"  timelines built ({time.time()-t:.1f}s)", flush=True)

    # Pick random epochs
    rng = random.Random(SEED)
    ext_epochs = [ep for ep in range(EXTENSION_RANGE[0], EXTENSION_RANGE[1] + 1)
                   if ep in epoch_to_ts]
    cv5_epochs = [ep for ep in range(CV5_RANGE[0], CV5_RANGE[1] + 1)
                   if ep in epoch_to_ts]
    sample_ext = sorted(rng.sample(ext_epochs, N_EXT_SAMPLES))
    sample_cv5 = sorted(rng.sample(cv5_epochs, N_CV5_SAMPLES))

    def analyze_epoch(ep: int) -> dict:
        start_at = epoch_to_ts[ep]
        lock_at_s = start_at + ROUND_LOCK_OFFSET_S
        window_lo = lock_at_s - WINDOW_24H_S
        window_hi = lock_at_s

        per_asset_arrays = {}
        per_asset_stats = {}
        for a in ASSETS:
            ts = ts_arr[a]
            px = px_arr[a]
            idx_lo = int(np.searchsorted(ts, window_lo, side="left"))
            idx_hi = int(np.searchsorted(ts, window_hi, side="right"))
            sub_ts = ts[idx_lo:idx_hi]
            sub_px = px[idx_lo:idx_hi]
            per_asset_arrays[a] = (sub_ts, sub_px)
            n = int(len(sub_ts))
            if n < 2:
                per_asset_stats[a] = {"n": n, "first_off_s": None, "last_off_s": None,
                                       "mean_int": None, "stdev_int": None}
                continue
            first_off = int(sub_ts[0] - lock_at_s)
            last_off = int(sub_ts[-1] - lock_at_s)
            intervals = np.diff(sub_ts.astype(np.float64))
            mean_int = float(np.mean(intervals))
            stdev_int = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0
            per_asset_stats[a] = {"n": n, "first_off_s": first_off, "last_off_s": last_off,
                                   "mean_int": mean_int, "stdev_int": stdev_int}

        # Strict-alignment correlation
        btc_ts, btc_px = per_asset_arrays["BTC"]
        if len(btc_ts) < 50:
            return {"epoch": ep, "lock_at_s": lock_at_s,
                    "per_asset": per_asset_stats,
                    "n_aligned": 0, "strict_corr": None, "pair_corrs": None}

        aligned_prices = {a: [] for a in ASSETS}
        for i in range(len(btc_ts)):
            ts_b = int(btc_ts[i])
            ats = {"BTC": float(btc_px[i])}
            include = True
            for a in ("ETH", "SOL", "BNB"):
                a_ts, a_px = per_asset_arrays[a]
                if len(a_ts) == 0:
                    include = False
                    break
                j = int(np.searchsorted(a_ts, ts_b))
                candidates = []
                if j > 0:
                    candidates.append(j - 1)
                if j < len(a_ts):
                    candidates.append(j)
                best_j = None
                best_dt = ALIGN_TOL_S + 1
                for c in candidates:
                    dt = abs(int(a_ts[c]) - ts_b)
                    if dt < best_dt:
                        best_dt = dt
                        best_j = c
                if best_j is None or best_dt > ALIGN_TOL_S:
                    include = False
                    break
                ats[a] = float(a_px[best_j])
            if include:
                for a in ASSETS:
                    aligned_prices[a].append(ats[a])

        n_aligned = len(aligned_prices["BTC"])
        if n_aligned < 50:
            return {"epoch": ep, "lock_at_s": lock_at_s,
                    "per_asset": per_asset_stats,
                    "n_aligned": n_aligned, "strict_corr": None, "pair_corrs": None}

        # Log returns per asset from aligned prices
        returns = {a: [] for a in ASSETS}
        for a in ASSETS:
            prices = aligned_prices[a]
            for k in range(1, len(prices)):
                if prices[k - 1] > 0 and prices[k] > 0:
                    returns[a].append(math.log(prices[k] / prices[k - 1]))

        min_n = min(len(returns[a]) for a in ASSETS)
        if min_n < 50:
            return {"epoch": ep, "lock_at_s": lock_at_s,
                    "per_asset": per_asset_stats,
                    "n_aligned": n_aligned, "strict_corr": None, "pair_corrs": None}

        pairs = [("BTC", "ETH"), ("BTC", "SOL"), ("BTC", "BNB"),
                 ("ETH", "SOL"), ("ETH", "BNB"), ("SOL", "BNB")]
        pair_corrs = {}
        valid_corrs = []
        for a1, a2 in pairs:
            r1 = np.array(returns[a1][:min_n])
            r2 = np.array(returns[a2][:min_n])
            if r1.std() == 0 or r2.std() == 0:
                pair_corrs[f"{a1}-{a2}"] = None
                continue
            c = float(np.corrcoef(r1, r2)[0, 1])
            pair_corrs[f"{a1}-{a2}"] = c
            valid_corrs.append(c)

        mean_corr = float(np.mean(valid_corrs)) if valid_corrs else None
        return {"epoch": ep, "lock_at_s": lock_at_s,
                "per_asset": per_asset_stats,
                "n_aligned": n_aligned, "strict_corr": mean_corr,
                "pair_corrs": pair_corrs}

    def print_epoch_block(res: dict, label: str):
        ep = res["epoch"]
        print(f"\nEpoch {ep} ({label}, lock_at_s={res['lock_at_s']}):", flush=True)
        for a in ASSETS:
            d = res["per_asset"][a]
            if d["n"] < 2:
                print(f"  {a:>4s}: n={d['n']:>5d}  (insufficient data)", flush=True)
                continue
            print(f"  {a:>4s}: n={d['n']:>5d}  first={d['first_off_s']:>+7d}s  "
                  f"last={d['last_off_s']:>+5d}s  mean_int={d['mean_int']:>6.2f}s  "
                  f"stdev={d['stdev_int']:>6.2f}s", flush=True)
        if res["strict_corr"] is not None:
            pair_str = ", ".join(f"{k}={v:+.3f}" if v is not None else f"{k}=n/a"
                                  for k, v in res["pair_corrs"].items())
            print(f"  strict-aligned (+/-{ALIGN_TOL_S}s): n_aln={res['n_aligned']}  "
                  f"mean_corr={res['strict_corr']:+.4f}", flush=True)
            print(f"    pairs: {pair_str}", flush=True)
        else:
            print(f"  strict-aligned: insufficient (n_aln={res['n_aligned']})", flush=True)

    print(f"\n--- Extension cohort: {N_EXT_SAMPLES} random epochs (seed={SEED}) ---", flush=True)
    results_ext = []
    for ep in sample_ext:
        res = analyze_epoch(ep)
        results_ext.append(res)
        print_epoch_block(res, "extension")

    print(f"\n--- CV5 cohort: {N_CV5_SAMPLES} random epochs (control) ---", flush=True)
    results_cv5 = []
    for ep in sample_cv5:
        res = analyze_epoch(ep)
        results_cv5.append(res)
        print_epoch_block(res, "cv5")

    # Summary
    print("\n--- Summary ---", flush=True)
    strict_corrs_ext = [r["strict_corr"] for r in results_ext if r["strict_corr"] is not None]
    strict_corrs_cv5 = [r["strict_corr"] for r in results_cv5 if r["strict_corr"] is not None]

    ext_mean_strict = float(np.mean(strict_corrs_ext)) if strict_corrs_ext else None
    cv5_mean_strict = float(np.mean(strict_corrs_cv5)) if strict_corrs_cv5 else None

    print(f"  Extension strict-aligned correlation: n={len(strict_corrs_ext)}/{N_EXT_SAMPLES} "
          f"mean={ext_mean_strict:.4f}" if ext_mean_strict is not None else
          f"  Extension strict-aligned correlation: n=0/{N_EXT_SAMPLES} (no valid)", flush=True)
    print(f"  CV5 strict-aligned correlation:       n={len(strict_corrs_cv5)}/{N_CV5_SAMPLES} "
          f"mean={cv5_mean_strict:.4f}" if cv5_mean_strict is not None else
          f"  CV5 strict-aligned correlation:       n=0/{N_CV5_SAMPLES} (no valid)", flush=True)
    print(f"  Step 17 reference (hourly-avg):       extension=+0.0225  cv5=+0.7833", flush=True)

    # Verdict
    if ext_mean_strict is not None and cv5_mean_strict is not None:
        if ext_mean_strict < 0.4 and cv5_mean_strict > 0.6:
            print(f"  VERDICT: F2 regime collapse appears REAL "
                  f"(ext={ext_mean_strict:+.3f} < 0.4 < {cv5_mean_strict:+.3f}=cv5)", flush=True)
        elif ext_mean_strict > 0.5:
            print(f"  VERDICT: F2 appears to be TIMESTAMP ARTIFACT "
                  f"(ext jumped from +0.022 to {ext_mean_strict:+.3f} with strict alignment)", flush=True)
        else:
            print(f"  VERDICT: Inconclusive "
                  f"(ext={ext_mean_strict:+.3f}, cv5={cv5_mean_strict:+.3f})", flush=True)

    # Save
    out_path = REPO / "var" / "strategy_review" / "step18a_alignment_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "n_ext": N_EXT_SAMPLES, "n_cv5": N_CV5_SAMPLES,
                "window_24h_s": WINDOW_24H_S, "align_tol_s": ALIGN_TOL_S,
                "seed": SEED,
            },
            "extension_results": results_ext,
            "cv5_results": results_cv5,
            "summary": {
                "extension_strict_corr_mean": ext_mean_strict,
                "extension_strict_corr_n": len(strict_corrs_ext),
                "cv5_strict_corr_mean": cv5_mean_strict,
                "cv5_strict_corr_n": len(strict_corrs_cv5),
                "step17_reference_extension": 0.0225,
                "step17_reference_cv5": 0.7833,
            },
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
