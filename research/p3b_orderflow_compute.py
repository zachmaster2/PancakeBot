"""p3b orderflow features main compute.

Per orchestrator v1.1 (locked, ratified):
  - Run feature compute on post-v1 (gate, n=1,943) + f5 (confirmation, n=7,305)
    + f4 (confirmation, n=7,305).
  - Per feature × slice: Pearson r, Fisher-z 95% CI, p-value.
  - Bonferroni: per-feature significance threshold = 0.05/12 ≈ 0.004.
  - R3 conditional: if |r(trade_volume_total, post-v1 outcome)| >= 0.05,
    reclassify as Tier 0 confounder + add to partial-r control list.
  - For Tier 1 features passing |r|≥0.066 AND p<0.004: partial-r vs
    bull_pool_ratio (and trade_volume_total if R3 triggered) on post-v1.
  - PASS criterion: |r|>=0.066 AND p<0.004 on post-v1 AND same sign on
    f5 AND f4 AND |r_f5|>=0.5*|r_post-v1| AND |r_f4|>=0.5*|r_post-v1|
    AND if Tier 1: |partial_r| >= 0.05.

Output: var/extended/p3b_orderflow_results.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")
WORKTREE = REPO / ".claude" / "worktrees" / "stupefied-bell-4d955c"
sys.path.insert(0, str(WORKTREE))

from research.p3b_orderflow_features import (
    FEATURE_NAMES, TIER0_FEATURES, TIER1_FEATURES, compute_features, round_outcome,
    EPOCH_DURATION, DATA_HORIZON_OFFSET_S,
)

CACHE_PATH = REPO / "var" / "extended" / "okx_trades_BNB-USDT.jsonl"
CANONICAL_ROUNDS_PATH = REPO / "var" / "closed_rounds.jsonl"

FLOOR_START_AT = 1765444670
FLOOR_EPOCH = 437562

POSTV1_LO, POSTV1_HI = 475312, 477254
F5_LO, F5_HI = 466782, 474086
F4_LO, F4_HI = 459477, 466781

PASS_R_MIN = 0.066
PASS_P_MAX = 0.004
PARTIAL_R_MIN = 0.05
R3_TIER0_EXPANSION_THRESHOLD = 0.05

OUT = REPO / "var" / "extended" / "p3b_orderflow_results.json"


def epoch_for_ts(ts_s: int) -> int:
    return (ts_s - FLOOR_START_AT) // EPOCH_DURATION + FLOOR_EPOCH


def epoch_start_at(ep: int) -> int:
    return FLOOR_START_AT + (ep - FLOOR_EPOCH) * EPOCH_DURATION


def load_trades_binned(slice_lo: int, slice_hi: int) -> dict[int, list[dict]]:
    """Load trades from cache, filter to [slice_lo, slice_hi] epochs, bin by epoch."""
    by_epoch: dict[int, list[dict]] = defaultdict(list)
    n_total = 0
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            ts_s = int(t["ts"]) // 1000
            if ts_s < FLOOR_START_AT:
                continue
            ep = epoch_for_ts(ts_s)
            if not (slice_lo <= ep <= slice_hi):
                continue
            sa = epoch_start_at(int(ep))
            cutoff = sa + EPOCH_DURATION - DATA_HORIZON_OFFSET_S
            if not (sa <= ts_s < cutoff):
                continue  # post-cutoff trade
            by_epoch[int(ep)].append(t)
            n_total += 1
    print(f"  loaded {n_total} trades across {len(by_epoch)} rounds in [{slice_lo}..{slice_hi}]",
          flush=True)
    return by_epoch


def load_round_outcomes(slice_lo: int, slice_hi: int) -> dict[int, int]:
    """Return {epoch -> 1 (Bull) | 0 (Bear)}."""
    out = {}
    with open(CANONICAL_ROUNDS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec["epoch"])
            if not (slice_lo <= ep <= slice_hi):
                continue
            o = round_outcome(rec)
            if o is not None:
                out[ep] = o
    return out


def load_round_pool_data(slice_lo: int, slice_hi: int) -> dict[int, dict]:
    """For partial-r control vs bull_pool_ratio: load per-round bull/bear pool."""
    BNB_WEI = 1_000_000_000_000_000_000
    out = {}
    with open(CANONICAL_ROUNDS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec["epoch"])
            if not (slice_lo <= ep <= slice_hi):
                continue
            sa = int(rec["startAt"])
            cutoff = sa + EPOCH_DURATION - DATA_HORIZON_OFFSET_S
            bull = bear = 0
            for b in rec.get("bets") or []:
                if int(b["createdAt"]) >= cutoff:
                    continue
                amt = int(b["amountWei"])
                if b["position"] == "Bull":
                    bull += amt
                else:
                    bear += amt
            total = bull + bear
            out[ep] = {
                "bull_pool_ratio": bull / total if total > 0 else float("nan"),
            }
    return out


def pearson_r_p(x: np.ndarray, y: np.ndarray):
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    mx, my = xv.mean(), yv.mean()
    dx, dy = xv - mx, yv - my
    num = (dx * dy).sum()
    den = math.sqrt((dx * dx).sum() * (dy * dy).sum())
    if den == 0:
        return 0.0, 1.0, 0.0, 0.0, n
    r = num / den
    if abs(r) >= 1.0:
        return float(r), 0.0, float(r), float(r), n
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r * r)
    p = math.erfc(abs(t) / math.sqrt(2))
    z = math.atanh(r)
    se_z = 1.0 / math.sqrt(n - 3) if n > 3 else float("nan")
    if math.isnan(se_z):
        return float(r), float(p), float("nan"), float("nan"), n
    return float(r), float(p), float(math.tanh(z - 1.96 * se_z)), float(math.tanh(z + 1.96 * se_z)), n


def partial_r(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    xv, yv, zv = x[mask], y[mask], z[mask]
    if len(xv) < 4:
        return float("nan")
    rxy, *_ = pearson_r_p(xv, yv)
    rxz, *_ = pearson_r_p(xv, zv)
    ryz, *_ = pearson_r_p(yv, zv)
    den = math.sqrt((1 - rxz * rxz) * (1 - ryz * ryz))
    if den == 0:
        return float("nan")
    return float((rxy - rxz * ryz) / den)


def build_features_for_slice(
    slice_lo: int, slice_hi: int, *, large_size_threshold_usd: float,
):
    """Returns (feature_arrays, outcomes, pool_data, n_rounds_with_outcome)."""
    by_epoch_trades = load_trades_binned(slice_lo, slice_hi)
    outcomes = load_round_outcomes(slice_lo, slice_hi)
    pools = load_round_pool_data(slice_lo, slice_hi)

    # Compute features for every round with outcome (regardless of trade availability)
    epochs = sorted(outcomes.keys())
    feats_list = []
    out_list = []
    bp_list = []
    for ep in epochs:
        sa = epoch_start_at(ep)
        trades = by_epoch_trades.get(ep, [])
        f = compute_features(sa, trades, large_size_threshold_usd=large_size_threshold_usd)
        feats_list.append(f)
        out_list.append(outcomes[ep])
        bp_list.append(pools.get(ep, {}).get("bull_pool_ratio", float("nan")))

    n = len(feats_list)
    feats_arrays = {
        name: np.array([f.get(name, float("nan")) for f in feats_list], dtype=np.float64)
        for name in FEATURE_NAMES
    }
    return feats_arrays, np.array(out_list, dtype=np.float64), np.array(bp_list, dtype=np.float64), n


def compute_global_large_size_threshold(slice_lo: int, slice_hi: int, percentile: float = 90.0) -> float:
    """Compute the percentile-th USD trade size across the slice's trades."""
    sizes = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            ts_s = int(t["ts"]) // 1000
            if ts_s < FLOOR_START_AT:
                continue
            ep = epoch_for_ts(ts_s)
            if not (slice_lo <= ep <= slice_hi):
                continue
            sa = epoch_start_at(int(ep))
            cutoff = sa + EPOCH_DURATION - DATA_HORIZON_OFFSET_S
            if not (sa <= ts_s < cutoff):
                continue
            sizes.append(float(t["sz"]) * float(t["px"]))
    if not sizes:
        return 0.0
    return float(np.percentile(sizes, percentile))


def main():
    print("=" * 100)
    print("p3b orderflow features — main compute")
    print(f"  thresholds: |r|>={PASS_R_MIN}, p<{PASS_P_MAX} (Bonferroni 0.05/12)")
    print(f"  partial-r vs bull_pool_ratio: >={PARTIAL_R_MIN}")
    print(f"  R3 conditional Tier 0 expansion threshold: |r(volume,outcome)|>={R3_TIER0_EXPANSION_THRESHOLD}")
    print("=" * 100)
    t_start = time.time()

    # Compute per-slice large-trade threshold (90th-percentile USD size)
    print("\n[1/4] Computing large-trade thresholds per slice...", flush=True)
    thresholds = {}
    for name, (lo, hi) in [("post_v1", (POSTV1_LO, POSTV1_HI)),
                              ("f5", (F5_LO, F5_HI)),
                              ("f4", (F4_LO, F4_HI))]:
        thr = compute_global_large_size_threshold(lo, hi)
        thresholds[name] = thr
        print(f"  {name}: 90th-pctile USD trade size = ${thr:,.2f}", flush=True)

    # Build features per slice
    print("\n[2/4] Building features per slice...", flush=True)
    slice_data = {}
    for name, (lo, hi) in [("post_v1", (POSTV1_LO, POSTV1_HI)),
                              ("f5", (F5_LO, F5_HI)),
                              ("f4", (F4_LO, F4_HI))]:
        t0 = time.time()
        feats, outs, bp, n = build_features_for_slice(
            lo, hi, large_size_threshold_usd=thresholds[name],
        )
        print(f"  {name}: {n} rounds with outcome. elapsed={time.time()-t0:.1f}s", flush=True)
        slice_data[name] = {
            "features": feats, "outcomes": outs, "bull_pool_ratio": bp, "n": n,
            "epoch_range": (lo, hi),
        }

    # R3 conditional check
    print("\n[3/4] R3 conditional check (trade_volume_total × outcome on post-v1)...",
          flush=True)
    pv1 = slice_data["post_v1"]
    vol_arr = pv1["features"]["trade_volume_total"]
    out_arr = pv1["outcomes"]
    r_vol, p_vol, ci_lo_v, ci_hi_v, n_v = pearson_r_p(vol_arr, out_arr)
    print(f"  trade_volume_total × outcome (post-v1, n={n_v}): r={r_vol:+.5f} p={p_vol:.5f} "
          f"CI=[{ci_lo_v:+.4f}, {ci_hi_v:+.4f}]", flush=True)
    r3_triggered = abs(r_vol) >= R3_TIER0_EXPANSION_THRESHOLD
    extra_partial_controls = ["bull_pool_ratio"]
    if r3_triggered:
        print(f"  R3 TRIGGERED: |r|={abs(r_vol):.4f} >= {R3_TIER0_EXPANSION_THRESHOLD}. "
              "trade_volume_total reclassified Tier 0; added to partial-r control list.",
              flush=True)
        extra_partial_controls.append("trade_volume_total")
        # Reclassify trade_volume_total in the feature classification
        effective_tier0 = TIER0_FEATURES | {"trade_volume_total"}
        effective_tier1 = TIER1_FEATURES - {"trade_volume_total"}
    else:
        print(f"  R3 NOT TRIGGERED: |r|={abs(r_vol):.4f} < {R3_TIER0_EXPANSION_THRESHOLD}. "
              "trade_volume_total stays Tier 1.", flush=True)
        effective_tier0 = set(TIER0_FEATURES)
        effective_tier1 = set(TIER1_FEATURES)

    # Per-feature × per-slice correlations
    print("\n[4/4] Per-feature correlations and partial-r controls...", flush=True)
    per_feature: dict[str, dict] = {}
    for fname in FEATURE_NAMES:
        tier = "tier0" if fname in effective_tier0 else "tier1"
        per_slice = {}
        for sname, sd in slice_data.items():
            x = sd["features"][fname]
            y = sd["outcomes"]
            r, p, ci_lo, ci_hi, n = pearson_r_p(x, y)
            per_slice[sname] = {
                "r": r, "p": p, "ci95_lo": ci_lo, "ci95_hi": ci_hi, "n": n,
            }
        per_feature[fname] = {"tier": tier, "per_slice": per_slice}

    # Sign-and-magnitude consistency + partial-r
    pass_candidates = []
    for fname in FEATURE_NAMES:
        info = per_feature[fname]
        post = info["per_slice"]["post_v1"]
        f5 = info["per_slice"]["f5"]
        f4 = info["per_slice"]["f4"]
        # Gate
        if math.isnan(post["r"]) or math.isnan(post["p"]):
            continue
        gate_ok = abs(post["r"]) >= PASS_R_MIN and post["p"] < PASS_P_MAX
        if not gate_ok:
            continue
        # Same sign on all three slices
        sign_ok = (
            (post["r"] > 0 and f5["r"] > 0 and f4["r"] > 0) or
            (post["r"] < 0 and f5["r"] < 0 and f4["r"] < 0)
        )
        # Magnitude consistency (50% of post-v1 |r| on each confirmation)
        mag_ok = (
            abs(f5["r"]) >= 0.5 * abs(post["r"]) and
            abs(f4["r"]) >= 0.5 * abs(post["r"])
        )
        cand = {
            "feature": fname, "tier": info["tier"], "post_v1": post, "f5": f5, "f4": f4,
            "sign_consistent": sign_ok, "magnitude_consistent": mag_ok,
        }
        # Partial-r vs bull_pool_ratio (and trade_volume_total if R3 triggered)
        if info["tier"] == "tier1":
            x = slice_data["post_v1"]["features"][fname]
            y = slice_data["post_v1"]["outcomes"]
            partial_results = {}
            for ctrl in extra_partial_controls:
                if ctrl == "bull_pool_ratio":
                    z = slice_data["post_v1"]["bull_pool_ratio"]
                else:
                    z = slice_data["post_v1"]["features"][ctrl]
                pr = partial_r(x, y, z)
                partial_results[ctrl] = pr
            cand["partial_r"] = partial_results
            cand["partial_r_pass"] = all(
                not math.isnan(v) and abs(v) >= PARTIAL_R_MIN
                for v in partial_results.values()
            )
        else:
            cand["partial_r"] = None
            cand["partial_r_pass"] = None
        # Final PASS
        if cand["tier"] == "tier1":
            cand["overall_pass"] = (
                cand["sign_consistent"] and cand["magnitude_consistent"]
                and cand["partial_r_pass"]
            )
        else:
            cand["overall_pass"] = (
                cand["sign_consistent"] and cand["magnitude_consistent"]
            )
        pass_candidates.append(cand)

    # Print summary table
    print("\n" + "=" * 130)
    print("CORRELATION TABLE — per feature × per slice")
    print("=" * 130)
    hdr = (f"{'feature':<28}{'tier':<6}"
           f"{'post_v1_r':>11}{'p':>9}{'CI':>22} | "
           f"{'f5_r':>10}{'p':>9}{'CI':>22} | "
           f"{'f4_r':>10}{'p':>9}{'CI':>22}")
    print(hdr)
    print("-" * len(hdr))
    for fname in FEATURE_NAMES:
        info = per_feature[fname]
        post = info["per_slice"]["post_v1"]
        f5 = info["per_slice"]["f5"]
        f4 = info["per_slice"]["f4"]
        marker = ""
        if abs(post["r"]) >= PASS_R_MIN and post["p"] < PASS_P_MAX:
            marker = " <-- post-v1 GATE"
        ci_post = f"[{post['ci95_lo']:+.3f},{post['ci95_hi']:+.3f}]"
        ci_f5 = f"[{f5['ci95_lo']:+.3f},{f5['ci95_hi']:+.3f}]"
        ci_f4 = f"[{f4['ci95_lo']:+.3f},{f4['ci95_hi']:+.3f}]"
        print(f"{fname:<28}{info['tier']:<6}"
              f"{post['r']:>+11.4f}{post['p']:>9.4f}{ci_post:>22} | "
              f"{f5['r']:>+10.4f}{f5['p']:>9.4f}{ci_f5:>22} | "
              f"{f4['r']:>+10.4f}{f4['p']:>9.4f}{ci_f4:>22}{marker}")

    print("\n" + "=" * 100)
    print("PASS CANDIDATES")
    print("=" * 100)
    if not pass_candidates:
        print("  NONE — no feature passed post-v1 gate.")
    else:
        for c in pass_candidates:
            print(f"  {c['feature']} ({c['tier']}): "
                  f"sign_consistent={c['sign_consistent']} "
                  f"magnitude_consistent={c['magnitude_consistent']}")
            if c["partial_r"] is not None:
                for ctrl, val in c["partial_r"].items():
                    print(f"    partial_r vs {ctrl}: {val:+.4f}")
                print(f"    partial_r_pass (>= {PARTIAL_R_MIN}): {c['partial_r_pass']}")
            print(f"    OVERALL PASS: {c['overall_pass']}")

    # Save
    out = {
        "spec": {
            "thresholds": {"pass_r_min": PASS_R_MIN, "pass_p_max": PASS_P_MAX,
                              "partial_r_min": PARTIAL_R_MIN,
                              "r3_tier0_expansion_threshold": R3_TIER0_EXPANSION_THRESHOLD},
            "slices": {
                "post_v1": [POSTV1_LO, POSTV1_HI],
                "f5": [F5_LO, F5_HI],
                "f4": [F4_LO, F4_HI],
            },
            "tier0_initial": sorted(TIER0_FEATURES),
            "tier1_initial": sorted(TIER1_FEATURES),
            "tier0_effective_after_r3": sorted(effective_tier0),
            "tier1_effective_after_r3": sorted(effective_tier1),
            "large_trade_thresholds_usd": thresholds,
        },
        "r3_check": {
            "trade_volume_total_r_postv1": r_vol,
            "trade_volume_total_p_postv1": p_vol,
            "r3_triggered": r3_triggered,
            "partial_r_controls": extra_partial_controls,
        },
        "per_feature": per_feature,
        "pass_candidates": pass_candidates,
        "elapsed_seconds": time.time() - t_start,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResult JSON: {OUT}", flush=True)
    print(f"Total elapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
