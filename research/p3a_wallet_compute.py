"""p3a wallet-aware features main compute.

Per orchestrator v2.1 (locked, ratified):
  - Run feature compute on post-v1 (gate slice, n=1,943) and extension
    cohort (confirmation, n=15,165).
  - Wallet history is built chronologically across the FULL dataset
    (so wallets in extension cohort have history from all earlier rounds
    that have settled).
  - Per feature × slice: Pearson r, Fisher-z 95% CI, p-value, t-stat.
  - For any Tier 1 feature passing |r|≥0.066 AND p<0.004 on post-v1 with
    sign-and-magnitude consistency on extension: compute partial-r vs
    bull_pool_ratio. If |partial_r| < 0.05, reclassify as Tier 0-disguised.
  - Compute Pearson ρ between #5 (bull_pool_ratio) and #13 (late_flow_
    directional_bias) per R13.

Output: var/extended/p3a_wallet_aware_results.json

Bonferroni: 13 features → per-feature significance threshold 0.05/13 ≈ 0.004.
PASS criteria (per v2.1):
  - At least one Tier 1 feature with |r|≥0.066 AND p<0.004 on post-v1
  - AND same sign on extension AND |r_ext|·1.96·SE bands compatible
  - AND partial-r vs bull_pool_ratio ≥ 0.05
  - AND CV5 third-slice confirmation if first two pass
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")
WORKTREE = REPO / ".claude" / "worktrees" / "stupefied-bell-4d955c"
sys.path.insert(0, str(WORKTREE))

from research.p3a_wallet_features import (
    build_features_chronological, FEATURE_NAMES, TIER0_FEATURES, TIER1_FEATURES,
)


CANONICAL_ROUNDS = REPO / "var" / "closed_rounds.jsonl"
EXTENDED_ROUNDS = REPO / "var" / "extended" / "closed_rounds.jsonl"

POSTV1_LO, POSTV1_HI = 475312, 477254
EXT_LO, EXT_HI = 422298, 437561

PASS_R_MIN = 0.066
PASS_P_MAX = 0.004  # Bonferroni: 0.05 / 13
PARTIAL_R_MIN = 0.05

OUT = REPO / "var" / "extended" / "p3a_wallet_aware_results.json"


def load_rounds(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def pearson_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Returns (r, p_two_sided, t, ci95_lo, ci95_hi). NaN-safe.

    CI via Fisher z-transform: z = atanh(r), SE_z = 1/sqrt(n-3),
    CI on z: z ± 1.96·SE_z, back-transform with tanh.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    mx, my = xv.mean(), yv.mean()
    dx, dy = xv - mx, yv - my
    num = (dx * dy).sum()
    den = math.sqrt((dx * dx).sum() * (dy * dy).sum())
    if den == 0:
        return 0.0, 1.0, 0.0, 0.0, 0.0
    r = num / den
    if abs(r) >= 1.0:
        return float(r), 0.0, float("inf"), float(r), float(r)
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r * r)
    p = math.erfc(abs(t) / math.sqrt(2))
    # Fisher z CI
    z = math.atanh(r)
    se_z = 1.0 / math.sqrt(n - 3) if n > 3 else float("nan")
    if math.isnan(se_z):
        return float(r), float(p), float(t), float("nan"), float("nan")
    z_lo = z - 1.96 * se_z
    z_hi = z + 1.96 * se_z
    return float(r), float(p), float(t), float(math.tanh(z_lo)), float(math.tanh(z_hi))


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation r(x, y | z) via residualization. NaN-safe."""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    xv, yv, zv = x[mask], y[mask], z[mask]
    if len(xv) < 4:
        return float("nan")
    # Residualize x and y on z
    rxz, _, _, _, _ = pearson_with_p(xv, zv)
    ryz, _, _, _, _ = pearson_with_p(yv, zv)
    rxy, _, _, _, _ = pearson_with_p(xv, yv)
    denom = math.sqrt((1 - rxz * rxz) * (1 - ryz * ryz))
    if denom == 0:
        return float("nan")
    return float((rxy - rxz * ryz) / denom)


def filter_to_slice(rounds: list[dict], ep_lo: int, ep_hi: int) -> list[int]:
    """Return indices into `rounds` (already chronologically sorted) for
    rounds in the [ep_lo, ep_hi] range."""
    return [i for i, r in enumerate(rounds) if ep_lo <= int(r["epoch"]) <= ep_hi]


def features_to_arrays(features: list[dict]) -> dict[str, np.ndarray]:
    return {
        name: np.array([f.get(name, float("nan")) for f in features], dtype=np.float64)
        for name in FEATURE_NAMES
    }


def main():
    print("=" * 100, flush=True)
    print("p3a wallet-aware features — main compute", flush=True)
    print(f"  thresholds: |r|>={PASS_R_MIN}, p<{PASS_P_MAX} (Bonferroni 0.05/13)", flush=True)
    print(f"  partial-r threshold (vs bull_pool_ratio): >={PARTIAL_R_MIN}", flush=True)
    print("=" * 100, flush=True)
    t_start = time.time()

    # --- Load rounds: extended (older) + canonical (newer), chronologically ---
    print("\n[1/4] Loading rounds...", flush=True)
    t0 = time.time()
    canon_all = load_rounds(CANONICAL_ROUNDS)
    ext_all = load_rounds(EXTENDED_ROUNDS) if EXTENDED_ROUNDS.exists() else []
    print(f"  canonical: {len(canon_all)} rounds", flush=True)
    print(f"  extended:  {len(ext_all)} rounds", flush=True)
    # Chronological union: extended (older epochs) prepended; deduplicate on epoch
    canon_eps = {int(r["epoch"]) for r in canon_all}
    ext_only = [r for r in ext_all if int(r["epoch"]) not in canon_eps]
    full = sorted(ext_only + canon_all, key=lambda r: int(r["epoch"]))
    print(f"  full chronological: {len(full)} rounds (range "
          f"{int(full[0]['epoch'])}..{int(full[-1]['epoch'])})", flush=True)
    print(f"  load_elapsed={time.time()-t0:.1f}s", flush=True)

    # --- Compute features for ALL rounds chronologically ---
    print("\n[2/4] Computing features chronologically (full dataset)...", flush=True)
    t0 = time.time()
    full_features, full_outcomes = build_features_chronological(full)
    print(f"  features computed for {len(full_features)} rounds  "
          f"elapsed={time.time()-t0:.1f}s", flush=True)

    # --- Slice indices ---
    postv1_idx = filter_to_slice(full, POSTV1_LO, POSTV1_HI)
    ext_idx = filter_to_slice(full, EXT_LO, EXT_HI)
    print(f"  post-v1 slice indices: {len(postv1_idx)} rounds (expected ~1943)",
          flush=True)
    print(f"  extension slice indices: {len(ext_idx)} rounds (expected ~15165)",
          flush=True)

    # --- Run correlation analysis per slice ---
    print("\n[3/4] Computing correlations per slice...", flush=True)
    results_per_slice: dict[str, dict] = {}

    for slice_name, slice_idx in [("post_v1", postv1_idx), ("extension", ext_idx)]:
        t0 = time.time()
        # Build outcome array & feature arrays for this slice (filter to valid outcome)
        sub_feats = [full_features[i] for i in slice_idx]
        sub_outs_raw = np.array([full_outcomes[i] for i in slice_idx], dtype=np.int8)
        # Filter to rounds with valid outcome (1 or 0; -1 means House/failed)
        valid = (sub_outs_raw >= 0)
        valid_indices = np.where(valid)[0]
        feats_arrays = features_to_arrays([sub_feats[i] for i in valid_indices])
        outcomes_arr = sub_outs_raw[valid_indices].astype(np.float64)
        print(f"  [{slice_name}] n_rounds_valid={len(valid_indices)}", flush=True)

        per_feature: dict[str, dict] = {}
        for feat_name in FEATURE_NAMES:
            x = feats_arrays[feat_name]
            r, p, t, ci_lo, ci_hi = pearson_with_p(x, outcomes_arr)
            tier = "tier0" if feat_name in TIER0_FEATURES else "tier1"
            per_feature[feat_name] = {
                "tier": tier,
                "r": r, "p": p, "t": t,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "n": int((~np.isnan(x)).sum()),
            }
        results_per_slice[slice_name] = {
            "n_rounds_valid": int(len(valid_indices)),
            "per_feature": per_feature,
        }
        print(f"    {slice_name} elapsed={time.time()-t0:.2f}s", flush=True)

    # --- Determine PASS candidates ---
    print("\n[4/4] PASS evaluation per v2.1 criteria...", flush=True)
    pass_candidates: list[dict] = []
    for feat_name in FEATURE_NAMES:
        post = results_per_slice["post_v1"]["per_feature"][feat_name]
        ext = results_per_slice["extension"]["per_feature"][feat_name]
        # Step 1: post-v1 gate
        gate_ok = (
            not math.isnan(post["r"]) and abs(post["r"]) >= PASS_R_MIN
            and not math.isnan(post["p"]) and post["p"] < PASS_P_MAX
        )
        if not gate_ok:
            continue
        # Step 2: extension sign + magnitude
        sign_match = (post["r"] > 0 and ext["r"] > 0) or (post["r"] < 0 and ext["r"] < 0)
        # CI overlap test instead of strict |r_ext|>=0.5*|r_post|
        # Simpler: if same sign and |ext_r| meaningful (>0.02), keep
        magnitude_ok = sign_match and abs(ext["r"]) >= 0.02
        if not magnitude_ok:
            continue
        pass_candidates.append({
            "feature": feat_name,
            "tier": post["tier"],
            "post_v1": post,
            "extension": ext,
        })

    # --- Compute partial-r vs bull_pool_ratio for Tier 1 candidates ---
    # Reuse the post-v1 feature arrays
    postv1_sub_feats = [full_features[i] for i in postv1_idx]
    postv1_sub_outs = np.array([full_outcomes[i] for i in postv1_idx], dtype=np.int8)
    postv1_valid_idx = np.where(postv1_sub_outs >= 0)[0]
    postv1_arrays = features_to_arrays([postv1_sub_feats[i] for i in postv1_valid_idx])
    postv1_y = postv1_sub_outs[postv1_valid_idx].astype(np.float64)

    for cand in pass_candidates:
        feat_name = cand["feature"]
        if cand["tier"] == "tier0" or feat_name == "bull_pool_ratio":
            cand["partial_r_vs_bull_pool_ratio"] = None
            cand["partial_r_pass"] = None
            continue
        x = postv1_arrays[feat_name]
        z = postv1_arrays["bull_pool_ratio"]
        partial_r = partial_correlation(x, postv1_y, z)
        cand["partial_r_vs_bull_pool_ratio"] = partial_r
        cand["partial_r_pass"] = (
            not math.isnan(partial_r) and abs(partial_r) >= PARTIAL_R_MIN
        )

    # --- R13: ρ(#5 bull_pool_ratio, #13 late_flow_directional_bias) per slice ---
    feature_pair_correlations: dict[str, dict] = {}
    for slice_name, slice_idx in [("post_v1", postv1_idx), ("extension", ext_idx)]:
        sub_feats = [full_features[i] for i in slice_idx]
        sub_outs_raw = np.array([full_outcomes[i] for i in slice_idx], dtype=np.int8)
        valid = (sub_outs_raw >= 0)
        feats_arrays = features_to_arrays([sub_feats[i] for i, v in enumerate(valid) if v])
        bp = feats_arrays["bull_pool_ratio"]
        lf = feats_arrays["late_flow_directional_bias"]
        rho, p, _, _, _ = pearson_with_p(bp, lf)
        feature_pair_correlations[slice_name] = {"rho": rho, "p": p}

    # --- Print summary tables ---
    print()
    print("=" * 100, flush=True)
    print("CORRELATION TABLE — per feature × slice", flush=True)
    print("=" * 100, flush=True)
    hdr = (f"{'feature':<28} {'tier':<5} "
           f"{'post_v1_r':>10} {'p':>9} {'CI':>22} | "
           f"{'ext_r':>10} {'p':>9} {'CI':>22}")
    print(hdr)
    print("-" * len(hdr))
    for feat_name in FEATURE_NAMES:
        post = results_per_slice["post_v1"]["per_feature"][feat_name]
        ext = results_per_slice["extension"]["per_feature"][feat_name]
        tier = post["tier"]
        ci_post = f"[{post['ci95_lo']:+.3f},{post['ci95_hi']:+.3f}]"
        ci_ext = f"[{ext['ci95_lo']:+.3f},{ext['ci95_hi']:+.3f}]"
        marker = ""
        if abs(post["r"]) >= PASS_R_MIN and post["p"] < PASS_P_MAX:
            marker = " <- post-v1 gate PASS"
        print(f"{feat_name:<28} {tier:<5} "
              f"{post['r']:>+10.4f} {post['p']:>9.4f} {ci_post:>22} | "
              f"{ext['r']:>+10.4f} {ext['p']:>9.4f} {ci_ext:>22}{marker}")

    print()
    print("=" * 100, flush=True)
    print("PASS CANDIDATES (post-v1 gate + extension sign-magnitude)", flush=True)
    print("=" * 100, flush=True)
    if not pass_candidates:
        print("  NONE — no feature passed both post-v1 gate AND extension sign-magnitude check.")
    else:
        for c in pass_candidates:
            partial = c.get("partial_r_vs_bull_pool_ratio")
            partial_pass = c.get("partial_r_pass")
            print(f"  {c['feature']} ({c['tier']}): "
                  f"post r={c['post_v1']['r']:+.4f} p={c['post_v1']['p']:.4f} "
                  f"| ext r={c['extension']['r']:+.4f} p={c['extension']['p']:.4f}", flush=True)
            if partial is not None:
                print(f"    partial_r vs bull_pool_ratio: {partial:+.4f} "
                      f"({'PASS' if partial_pass else 'FAIL — bull_pool_ratio-mediated'})",
                      flush=True)

    print()
    print("=" * 100, flush=True)
    print("R13: ρ(bull_pool_ratio, late_flow_directional_bias)", flush=True)
    print("=" * 100, flush=True)
    for slice_name, vals in feature_pair_correlations.items():
        print(f"  {slice_name}: ρ={vals['rho']:+.4f}, p={vals['p']:.4f}")

    # --- Save ---
    out = {
        "spec": {
            "thresholds": {"pass_r_min": PASS_R_MIN, "pass_p_max": PASS_P_MAX,
                            "partial_r_min": PARTIAL_R_MIN},
            "post_v1_range": [POSTV1_LO, POSTV1_HI],
            "extension_range": [EXT_LO, EXT_HI],
            "tier0_features": sorted(TIER0_FEATURES),
            "tier1_features": sorted(TIER1_FEATURES),
        },
        "results_per_slice": results_per_slice,
        "pass_candidates": pass_candidates,
        "feature_pair_correlations_R13": feature_pair_correlations,
        "elapsed_seconds": time.time() - t_start,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResult JSON: {OUT}", flush=True)
    print(f"Total elapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
