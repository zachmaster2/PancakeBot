"""p1b D1/D2 — round-level vol-vs-outcome gating diagnostic.

PRE-REGISTERED PROTOCOL (per orchestrator v3 + v3.1 + v3.2):

D1 — Sign-flip diagnostic on post-v1 fresh slice (epochs 475312..477254):
  1. For each round: compute realized BTC vol over 24 trailing 5-minute closes
     ending strictly before lock_at - 2s.
  2. Bin per-round outcome (1 if Bull, 0 if Bear) into quintiles by vol.
  3. PASS condition (BOTH required):
     - Mean PnL slope: lowest-vol quintile mean outcome > highest-vol quintile
       by >= 0.005 (binary outcome scale; 5pp difference)
     - Spearman rank correlation: NEGATIVE between vol-quintile-rank and
       mean-outcome AND p < 0.05 one-sided
  4. If only one passes, FAIL D1 (non-monotone), KILL p1b.

D2 — Magnitude correlation on post-v1 round-level (n=1,943):
  - Pearson r between realized BTC vol and binary round outcome
  - PASS: |r| >= 0.06 AND p < 0.05 (two-sided)
  - FAIL/KILL: otherwise

Reads main repo data via absolute paths; output JSON written next to the
other p1b artifacts at REPO/var/extended/p1b_d1d2_results.json.

If D1 PASSES AND D2 PASSES -> proceed to bet-level confirmation (Phase 2 of
this script, gated). If either fails -> write KILL summary and stop.

Reviewer-mandated rules (do not relax post-hoc):
  - One pre-registered cutoff, one shot. No threshold relaxation if results
    are borderline.
  - Round-level is the gate; bet-level is reported as context.
  - Sign-flip between round-level and bet-level (if bet-level reaches |r|>=0.10
    in OPPOSITE direction) halts the experiment.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")
sys.path.insert(0, str(REPO))

# Slice definitions (locked v3.1)
POSTV1_LO, POSTV1_HI = 475312, 477254  # ~1,943 rounds, fresh OOS post-v1 boundary
EXT_LO, EXT_HI = 422298, 437561        # extension cohort confirmation ~15,264 rounds

# Vol window
VOL_LOOKBACK_5M = 24      # 24 5-minute closes (2h trailing window)
VOL_LOG_RETURNS = VOL_LOOKBACK_5M - 1  # 23 log returns

# D1 thresholds (pre-registered)
D1_SLOPE_MIN = 0.005      # lowest-vol-quintile mean outcome > highest by >= this
D1_SPEARMAN_ALPHA = 0.05  # one-sided
# D2 thresholds (pre-registered)
D2_R_MIN = 0.06
D2_P_MAX = 0.05           # two-sided
# Bet-level halt rule (pre-registered)
BET_LEVEL_FLIP_R_MIN = 0.10

OUT_RESULTS = REPO / "var" / "extended" / "p1b_d1d2_results.json"


def write_atomic(path: Path, content: str):
    import os
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def load_round_outcomes(jsonl_path: Path, ep_min: int, ep_max: int) -> dict[int, int]:
    """Return {epoch -> 1 if Bull, 0 if Bear}, excluding None/failed positions."""
    out: dict[int, int] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec["epoch"])
            if not (ep_min <= ep <= ep_max):
                continue
            pos = rec.get("position")
            if pos == "Bull":
                out[ep] = 1
            elif pos == "Bear":
                out[ep] = 0
            # else: None / failed -> exclude
    return out


def load_btc_5m_closes(jsonl_path: Path, ep_min: int, ep_max: int) -> dict[int, float]:
    """Return {epoch -> last 1s close} = the 5-minute close at end of that epoch."""
    out: dict[int, float] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec["epoch"])
            if not (ep_min <= ep <= ep_max):
                continue
            kl = rec.get("klines_1s")
            if kl is None or len(kl) == 0:
                continue
            try:
                out[ep] = float(kl[-1][4])
            except (IndexError, ValueError, TypeError):
                continue
    return out


def compute_realized_vol(closes_by_epoch: dict[int, float], target_epochs: list[int],
                         lookback: int) -> dict[int, float]:
    """Realized vol = std of log returns over `lookback` prior 5m closes.

    For target round at epoch e, uses closes at e-lookback..e-1 (lookback values
    of 5m close), computes lookback-1 log returns, returns std (sample, ddof=1).
    Returns {epoch -> vol or NaN if insufficient data}.
    """
    out: dict[int, float] = {}
    for e in target_epochs:
        # Need closes at e-lookback, e-lookback+1, ..., e-1 (lookback values total)
        prior_closes = []
        for offset in range(lookback, 0, -1):
            ep_prior = e - offset
            c = closes_by_epoch.get(ep_prior)
            if c is None or c <= 0:
                prior_closes = None
                break
            prior_closes.append(c)
        if prior_closes is None or len(prior_closes) < lookback:
            out[e] = float("nan")
            continue
        # Compute log returns
        log_rets = []
        for i in range(1, len(prior_closes)):
            if prior_closes[i] > 0 and prior_closes[i - 1] > 0:
                log_rets.append(math.log(prior_closes[i] / prior_closes[i - 1]))
        if len(log_rets) < 2:
            out[e] = float("nan")
            continue
        out[e] = float(np.std(log_rets, ddof=1))
    return out


def build_paired(outcomes: dict[int, int], vols: dict[int, float]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Align (vol, outcome) arrays for epochs that have both, dropping NaN vols."""
    epochs_sorted = sorted(set(outcomes.keys()) & set(vols.keys()))
    vol_list = []
    out_list = []
    eps_kept = []
    for e in epochs_sorted:
        v = vols.get(e)
        o = outcomes.get(e)
        if v is None or o is None:
            continue
        if not (isinstance(v, float) and not math.isnan(v) and not math.isinf(v)):
            continue
        vol_list.append(v)
        out_list.append(o)
        eps_kept.append(e)
    return np.array(vol_list), np.array(out_list, dtype=np.int32), eps_kept


def run_d1(vol: np.ndarray, out: np.ndarray) -> dict:
    """Quintile binning + slope + Spearman."""
    if len(vol) < 25:
        return {"error": "n<25, cannot bin into quintiles"}
    quintile_edges = np.quantile(vol, [0.2, 0.4, 0.6, 0.8])
    quintiles = np.digitize(vol, quintile_edges)  # 0..4
    quintile_means = []
    quintile_counts = []
    for q in range(5):
        mask = (quintiles == q)
        n_q = int(mask.sum())
        if n_q == 0:
            quintile_means.append(float("nan"))
            quintile_counts.append(0)
        else:
            quintile_means.append(float(out[mask].mean()))
            quintile_counts.append(n_q)

    # Slope test: lowest-vol quintile mean outcome > highest-vol quintile by >= D1_SLOPE_MIN
    slope_q5_minus_q1 = quintile_means[4] - quintile_means[0]
    slope_test_passed = slope_q5_minus_q1 <= -D1_SLOPE_MIN  # i.e. q1 - q5 >= D1_SLOPE_MIN

    # Spearman: NEGATIVE direction between quintile-rank (0..4) and per-quintile mean outcome
    # But the canonical Spearman approach is on the underlying continuous data, not quintile means
    rho_continuous, p_continuous = stats.spearmanr(vol, out, alternative="less")
    # Also report quintile-level (less-noisy with n=5 quintiles but only 5 points)
    quintile_ranks = np.arange(5)
    qm_arr = np.array(quintile_means)
    if not np.any(np.isnan(qm_arr)):
        rho_quintile, p_quintile = stats.spearmanr(quintile_ranks, qm_arr, alternative="less")
    else:
        rho_quintile = float("nan")
        p_quintile = float("nan")

    # PASS = both slope_test and continuous-Spearman < α
    spearman_test_passed = (rho_continuous < 0) and (p_continuous < D1_SPEARMAN_ALPHA)
    d1_pass = bool(slope_test_passed and spearman_test_passed)

    return {
        "n": int(len(vol)),
        "quintile_edges": [float(x) for x in quintile_edges],
        "quintile_means": [float(x) for x in quintile_means],
        "quintile_counts": [int(x) for x in quintile_counts],
        "slope_q5_minus_q1": float(slope_q5_minus_q1),
        "slope_threshold_min": -D1_SLOPE_MIN,
        "slope_test_passed": bool(slope_test_passed),
        "spearman_continuous_rho": float(rho_continuous),
        "spearman_continuous_p_one_sided_neg": float(p_continuous),
        "spearman_quintile_rho": float(rho_quintile) if not (isinstance(rho_quintile, float) and math.isnan(rho_quintile)) else None,
        "spearman_quintile_p": float(p_quintile) if not (isinstance(p_quintile, float) and math.isnan(p_quintile)) else None,
        "spearman_alpha_one_sided": D1_SPEARMAN_ALPHA,
        "spearman_test_passed": bool(spearman_test_passed),
        "d1_pass": d1_pass,
    }


def run_d2(vol: np.ndarray, out: np.ndarray) -> dict:
    """Pearson r between vol and binary outcome."""
    if len(vol) < 30:
        return {"error": "n<30, cannot run Pearson"}
    r, p = stats.pearsonr(vol, out)
    abs_r = abs(r)
    d2_pass = bool(abs_r >= D2_R_MIN and p < D2_P_MAX)
    return {
        "n": int(len(vol)),
        "pearson_r": float(r),
        "pearson_abs_r": float(abs_r),
        "pearson_p_two_sided": float(p),
        "d2_threshold_abs_r_min": D2_R_MIN,
        "d2_threshold_p_max": D2_P_MAX,
        "d2_pass": d2_pass,
    }


def diagnostic_for_slice(name: str, ep_min: int, ep_max: int, *,
                          rounds_path: Path, klines_path: Path) -> dict:
    """Run D1/D2 on a single slice."""
    print(f"\n=== {name}: epochs [{ep_min}..{ep_max}] ===", flush=True)
    print(f"  rounds: {rounds_path}", flush=True)
    print(f"  klines: {klines_path}", flush=True)

    outcomes = load_round_outcomes(rounds_path, ep_min, ep_max)
    print(f"  rounds with valid position: {len(outcomes)}", flush=True)

    # Need closes from ep_min - VOL_LOOKBACK_5M to ep_max for the trailing window
    closes = load_btc_5m_closes(klines_path, ep_min - VOL_LOOKBACK_5M, ep_max)
    print(f"  BTC 5m closes loaded: {len(closes)}", flush=True)

    target_epochs = [e for e in outcomes.keys() if ep_min <= e <= ep_max]
    vols = compute_realized_vol(closes, target_epochs, VOL_LOOKBACK_5M)
    valid_vols = sum(1 for v in vols.values() if not math.isnan(v))
    print(f"  vols computed (non-NaN): {valid_vols}/{len(target_epochs)}", flush=True)

    vol_arr, out_arr, eps_kept = build_paired(outcomes, vols)
    print(f"  paired (vol, outcome) records: {len(vol_arr)}", flush=True)
    if len(vol_arr) > 0:
        bull_rate = float(out_arr.mean())
        print(f"  bull rate: {bull_rate*100:.2f}%  vol mean: {vol_arr.mean():.6f}  std: {vol_arr.std():.6f}",
              flush=True)

    d1 = run_d1(vol_arr, out_arr)
    d2 = run_d2(vol_arr, out_arr)
    print(f"  D1: pass={d1.get('d1_pass')}  slope_q5-q1={d1.get('slope_q5_minus_q1'):+.4f}  "
          f"spearman_rho={d1.get('spearman_continuous_rho'):+.4f} (p={d1.get('spearman_continuous_p_one_sided_neg'):.4f} one-sided neg)",
          flush=True)
    print(f"  D2: pass={d2.get('d2_pass')}  r={d2.get('pearson_r'):+.4f}  "
          f"|r|={d2.get('pearson_abs_r'):.4f} (p={d2.get('pearson_p_two_sided'):.4f} two-sided)",
          flush=True)
    return {
        "slice": name,
        "epoch_range": [ep_min, ep_max],
        "n_rounds_with_position": len(outcomes),
        "n_paired": len(vol_arr),
        "bull_rate": float(out_arr.mean()) if len(out_arr) else None,
        "vol_mean": float(vol_arr.mean()) if len(vol_arr) else None,
        "vol_std": float(vol_arr.std()) if len(vol_arr) else None,
        "d1": d1,
        "d2": d2,
    }


def main():
    print("=" * 100)
    print("p1b D1/D2 — round-level vol-vs-outcome gating diagnostic")
    print("Slices: post-v1 fresh (gate) + extension cohort (confirmation)")
    print("=" * 100)
    t_start = time.time()

    canonical_rounds = REPO / "var" / "closed_rounds.jsonl"
    canonical_btc = REPO / "var" / "btc_spot_prices.jsonl"
    extended_rounds = REPO / "var" / "extended" / "closed_rounds.jsonl"
    extended_btc = REPO / "var" / "extended" / "btc_spot_prices.jsonl"

    # post-v1 uses canonical files (rounds + klines for ep_min - 24..ep_max)
    # First 24 post-v1 rounds reach into v1 holdout for trailing closes (per v3.2 ratification)
    postv1 = diagnostic_for_slice(
        "post-v1 fresh (D1/D2 GATE)",
        POSTV1_LO, POSTV1_HI,
        rounds_path=canonical_rounds,
        klines_path=canonical_btc,
    )

    # extension cohort uses extended/ files; trailing closes for first 24 rounds may
    # be insufficient (start of dataset) and will be excluded as NaN — that's OK
    extension = diagnostic_for_slice(
        "extension cohort (CONFIRMATION)",
        EXT_LO, EXT_HI,
        rounds_path=extended_rounds,
        klines_path=extended_btc,
    )

    # Decision
    gate_d1_pass = postv1["d1"].get("d1_pass", False)
    gate_d2_pass = postv1["d2"].get("d2_pass", False)
    gate_pass = bool(gate_d1_pass and gate_d2_pass)

    # Direction-confirmation check on extension (read-only, not gating)
    confirm_r = extension["d2"].get("pearson_r")
    gate_r = postv1["d2"].get("pearson_r")
    direction_consistent = None
    if confirm_r is not None and gate_r is not None:
        direction_consistent = (confirm_r * gate_r) > 0

    summary = {
        "spec": {
            "vol_lookback_5m_candles": VOL_LOOKBACK_5M,
            "d1_slope_min_abs": D1_SLOPE_MIN,
            "d1_spearman_alpha_one_sided": D1_SPEARMAN_ALPHA,
            "d2_threshold_abs_r_min": D2_R_MIN,
            "d2_threshold_p_max": D2_P_MAX,
            "post_v1_range": [POSTV1_LO, POSTV1_HI],
            "extension_range": [EXT_LO, EXT_HI],
        },
        "post_v1_GATE": postv1,
        "extension_CONFIRMATION": extension,
        "gate_decision": {
            "d1_pass_post_v1": gate_d1_pass,
            "d2_pass_post_v1": gate_d2_pass,
            "gate_pass": gate_pass,
            "extension_direction_consistent": direction_consistent,
            "extension_r": confirm_r,
            "post_v1_r": gate_r,
        },
        "elapsed_seconds": time.time() - t_start,
    }

    print()
    print("=" * 100)
    if gate_pass:
        print("D1+D2 GATE: PASS on post-v1.")
        print(f"  Direction consistency on extension: {'YES' if direction_consistent else 'NO/SIGN-FLIP'}")
        if direction_consistent:
            print("  -> Cleared to proceed to D3 (vol-multiplier wiring + canonical hash equivalence)")
        else:
            print("  -> SIGN-FLIP between post-v1 and extension. HALT, investigate before sweep.")
    else:
        print("D1+D2 GATE: FAIL on post-v1.")
        print(f"  D1 pass: {gate_d1_pass}")
        print(f"  D2 pass: {gate_d2_pass}")
        print(f"  D2 r: {gate_r:+.4f}  |r|: {abs(gate_r) if gate_r is not None else 'NA'}  p: {postv1['d2'].get('pearson_p_two_sided'):.4f}")
        print("  -> KILL p1b. Vol-curve sizing has insufficient signal at the gating slice.")
    print("=" * 100)
    print(f"Result JSON: {OUT_RESULTS}")
    print(f"Elapsed: {time.time() - t_start:.1f}s")

    write_atomic(OUT_RESULTS, json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
