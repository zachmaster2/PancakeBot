"""Offline correlation analysis: CEX order flow and late bet flow vs round outcomes.

Hypothesis 1: Binance 1m kline features at cutoff time (price momentum,
taker-buy ratio, volume spike) predict round outcome.

Hypothesis 2: Post-cutoff bet flow (bets placed after lock_at - cutoff_seconds)
predicts round outcome better than pre-cutoff pool state alone.

Both analyses use data already on disk -- no new fetching required.

Outputs (all under ../PancakeBot_var_exp/):
  <prefix>_row_data.csv            per-round feature table
  <prefix>_correlations.csv        Pearson r of each feature vs outcome
  <prefix>_quintile_winrates.csv   win rate by feature quintile
  <prefix>_report.md               narrative summary
  <prefix>_cumulative_bnb.png      cumulative profit for top kline signals
  <prefix>_rolling_winrate.png     rolling win rate for top signals vs baseline
  <prefix>_late_flow_vs_outcome.png  late-flow direction vs outcome
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left, bisect_right
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CEX flow + late-bet correlation analysis")
    p.add_argument("--closed-rounds-path", type=str,
                   default="../PancakeBot_var_data/closed_rounds.jsonl")
    p.add_argument("--klines-path", type=str,
                   default="../PancakeBot_var_data/klines.jsonl")
    p.add_argument("--out-dir", type=str, default="../PancakeBot_var_exp")
    p.add_argument("--prefix", type=str, required=True,
                   help="Output file prefix, e.g. flow_corr_20260406")
    p.add_argument("--tail-rounds", type=int, default=50000,
                   help="Number of most recent rounds to analyse")
    p.add_argument("--cutoff-seconds", type=int, default=17,
                   help="Seconds before lock_at that constitute decision cutoff")
    p.add_argument("--rolling-window", type=int, default=2000,
                   help="Rolling window size for win-rate plots")
    p.add_argument("--min-pool-bnb", type=float, default=0.1,
                   help="Minimum total pool BNB to include a round")
    p.add_argument("--fixed-bet-bnb", type=float, default=0.05,
                   help="Simulated fixed stake for EV curves")
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_klines(path: str) -> tuple[list[int], list[dict]]:
    """Return (open_time_ms list, kline dict list) sorted ascending."""
    klines: list[dict] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            klines.append(json.loads(line))
    open_times = [int(k["open_time_ms"]) for k in klines]
    return open_times, klines


def _load_rounds(path: str, tail: int) -> list[dict]:
    with open(path, "r") as fh:
        lines = [l for l in fh if l.strip()]
    lines = lines[-tail:]
    return [json.loads(l) for l in lines]


# ---------------------------------------------------------------------------
# Kline feature helpers
# ---------------------------------------------------------------------------

def _kline_at_or_before(open_times: list[int], ts_ms: int) -> int:
    """Index of the last kline whose open_time_ms <= ts_ms, or -1."""
    idx = bisect_right(open_times, ts_ms) - 1
    return idx


def _kline_features(
    open_times: list[int],
    klines: list[dict],
    cutoff_ts_ms: int,
    windows: tuple[int, ...] = (1, 5, 15, 30),
) -> dict[str, float]:
    """Compute kline-derived features at cutoff time.

    All features are causal: they only use klines whose open_time_ms < cutoff_ts_ms
    (i.e., fully closed klines completed before our decision moment).
    """
    feats: dict[str, float] = {}
    idx = _kline_at_or_before(open_times, cutoff_ts_ms - 1)  # strictly before cutoff
    if idx < 0:
        return feats  # no klines available

    max_w = max(windows)
    if idx < max_w - 1:
        return feats  # not enough history

    for w in windows:
        slice_k = klines[idx - w + 1: idx + 1]  # w klines ending at idx
        if len(slice_k) < w:
            continue

        closes = [float(k["close_price"]) for k in slice_k]
        opens = [float(k["open_price"]) for k in slice_k]
        vols = [float(k["volume"]) for k in slice_k]
        # taker_buy and number_of_trades are Binance-specific; absent in OKX klines.
        tb_base = [float(k["taker_buy_base_volume"]) for k in slice_k
                   if k.get("taker_buy_base_volume") is not None]
        n_trades = [float(k["number_of_trades"]) for k in slice_k
                    if k.get("number_of_trades") is not None]

        # Price return over window
        ret = (closes[-1] / opens[0]) - 1.0
        feats[f"ret_{w}m"] = float(ret)

        # Taker-buy ratio (CEX order flow imbalance).
        # Only computed when the kline source provides taker_buy data (Binance).
        # OKX klines omit this field; tb_base will be empty in that case.
        total_vol = sum(vols)
        total_tb = sum(tb_base)
        if total_vol > 0 and tb_base:
            feats[f"taker_buy_ratio_{w}m"] = float(total_tb / total_vol)

        # Volume relative to 60m baseline (only for short windows)
        if w <= 5:
            baseline_idx_start = max(0, idx - 59)
            baseline_k = klines[baseline_idx_start: idx + 1]
            if len(baseline_k) >= 10:
                baseline_vol = sum(float(k["volume"]) for k in baseline_k) / len(baseline_k)
                if baseline_vol > 0:
                    feats[f"vol_spike_{w}m"] = float(sum(vols) / w / baseline_vol)

        # Intra-window high-low range (normalised volatility)
        highs = [float(k["high_price"]) for k in slice_k]
        lows = [float(k["low_price"]) for k in slice_k]
        mid = (closes[-1] + opens[0]) / 2.0
        if mid > 0:
            feats[f"hl_range_{w}m"] = float((max(highs) - min(lows)) / mid)

        # Average trades-per-minute (Binance only)
        if n_trades:
            feats[f"trades_{w}m"] = float(sum(n_trades) / w)

    return feats


# ---------------------------------------------------------------------------
# Bet pool feature helpers
# ---------------------------------------------------------------------------

def _bet_pool_features(bets: list[dict], lock_at_s: int, cutoff_seconds: int) -> dict[str, float]:
    """Compute pool-composition features from bet records.

    cutoff_ts_s = lock_at_s - cutoff_seconds
    Pre-cutoff bets: createdAt <= cutoff_ts_s
    Post-cutoff bets: createdAt > cutoff_ts_s
    """
    feats: dict[str, float] = {}
    if not bets:
        return feats

    cutoff_ts_s = int(lock_at_s) - int(cutoff_seconds)

    pre_bull = 0.0
    pre_bear = 0.0
    post_bull = 0.0
    post_bear = 0.0

    for b in bets:
        amt = float(b["amountWei"]) / 1e18
        ts = int(b["createdAt"])
        is_bull = b["position"] == "Bull"
        if ts <= cutoff_ts_s:
            if is_bull:
                pre_bull += amt
            else:
                pre_bear += amt
        else:
            if is_bull:
                post_bull += amt
            else:
                post_bear += amt

    pre_total = pre_bull + pre_bear
    post_total = post_bull + post_bear
    final_bull = pre_bull + post_bull
    final_bear = pre_bear + post_bear
    final_total = final_bull + final_bear

    if pre_total > 0:
        feats["pre_bull_share"] = float(pre_bull / pre_total)
        feats["pre_total_bnb"] = float(pre_total)

    if post_total > 0:
        feats["post_bull_share"] = float(post_bull / post_total)
        feats["post_total_bnb"] = float(post_total)
        feats["post_frac_of_final"] = float(post_total / max(final_total, 1e-9))

    if final_total > 0:
        feats["final_bull_share"] = float(final_bull / final_total)
        feats["final_total_bnb"] = float(final_total)

    if pre_total > 0 and post_total > 0:
        # Direction agreement: both sides biased the same way?
        pre_side = 1 if pre_bull > pre_bear else -1
        post_side = 1 if post_bull > post_bear else -1
        feats["late_flow_agrees_with_early"] = float(pre_side == post_side)
        feats["late_flow_direction"] = float(post_side)  # +1=Bull, -1=Bear

        # Magnitude of shift in bull_share from pre to final
        feats["bull_share_shift"] = float(feats.get("final_bull_share", 0.5) - feats.get("pre_bull_share", 0.5))

    return feats


# ---------------------------------------------------------------------------
# Outcome label
# ---------------------------------------------------------------------------

def _outcome(r: dict) -> int | None:
    """1 = Bull wins, 0 = Bear wins, None = skip."""
    if r.get("failed"):
        return None
    pos = r.get("position")
    if pos == "Bull":
        return 1
    if pos == "Bear":
        return 0
    return None


# ---------------------------------------------------------------------------
# EV simulation helpers
# ---------------------------------------------------------------------------

def _payout_multiplier(winner_pool: float, loser_pool: float, fee: float = 0.03) -> float:
    """Net payout multiplier on a winning bet (gross - 1, so 0 = break-even)."""
    if winner_pool <= 0:
        return 0.0
    gross = (winner_pool + loser_pool) * (1.0 - fee) / winner_pool
    return float(gross - 1.0)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)

    print(f"Loading klines from {args.klines_path} ...")
    open_times, klines = _load_klines(args.klines_path)
    print(f"  {len(klines):,} klines loaded")

    print(f"Loading closed rounds from {args.closed_rounds_path} (tail={args.tail_rounds:,}) ...")
    rounds = _load_rounds(args.closed_rounds_path, args.tail_rounds)
    print(f"  {len(rounds):,} rounds loaded")

    # ------------------------------------------------------------------
    # Build per-round feature rows
    # ------------------------------------------------------------------
    print("Computing features ...")
    rows: list[dict] = []
    skipped_no_outcome = 0
    skipped_no_klines = 0
    skipped_low_pool = 0

    for r in rounds:
        outcome = _outcome(r)
        if outcome is None:
            skipped_no_outcome += 1
            continue

        lock_at_s = int(r["lockAt"])
        cutoff_ts_ms = (lock_at_s - args.cutoff_seconds) * 1000

        kf = _kline_features(open_times, klines, cutoff_ts_ms)
        if not kf:
            skipped_no_klines += 1
            continue

        bets = r.get("bets", [])
        bf = _bet_pool_features(bets, lock_at_s, args.cutoff_seconds)

        final_total = bf.get("final_total_bnb", 0.0)
        if final_total < args.min_pool_bnb:
            skipped_low_pool += 1
            continue

        row: dict = {
            "epoch": int(r["epoch"]),
            "lock_at": int(lock_at_s),
            "outcome": int(outcome),
            "n_bets": int(len(bets)),
        }
        row.update(kf)
        row.update(bf)

        # Simple EV if we bet the kline-momentum direction
        final_bull_bnb = bf.get("final_bull_share", 0.5) * final_total
        final_bear_bnb = (1.0 - bf.get("final_bull_share", 0.5)) * final_total
        if final_bull_bnb > 0 and final_bear_bnb > 0:
            bull_mult = _payout_multiplier(final_bull_bnb, final_bear_bnb)
            bear_mult = _payout_multiplier(final_bear_bnb, final_bull_bnb)
            row["bull_net_mult"] = float(bull_mult)
            row["bear_net_mult"] = float(bear_mult)

        rows.append(row)

    print(f"  Valid rows: {len(rows):,}")
    print(f"  Skipped no outcome: {skipped_no_outcome:,}, no klines: {skipped_no_klines:,}, low pool: {skipped_low_pool:,}")

    if len(rows) < 500:
        print("ERROR: Too few valid rows. Aborting.")
        return

    # ------------------------------------------------------------------
    # Write row CSV
    # ------------------------------------------------------------------
    row_csv_path = out_dir / f"{prefix}_row_data.csv"
    all_keys = list(rows[0].keys())
    for row in rows:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)

    with row_csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in all_keys})
    print(f"Row data written to {row_csv_path}")

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------
    outcomes = np.array([r["outcome"] for r in rows], dtype=np.float64)

    feature_cols = [k for k in all_keys if k not in {"epoch", "lock_at", "outcome", "n_bets"}]

    corr_rows: list[dict] = []
    for feat in feature_cols:
        vals = []
        idxs = []
        for i, row in enumerate(rows):
            v = row.get(feat)
            if v != "" and v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(v))
                idxs.append(i)
        if len(vals) < 100:
            continue
        x = np.array(vals, dtype=np.float64)
        y = outcomes[idxs]
        # Pearson r
        if x.std() < 1e-12:
            r_val = 0.0
        else:
            r_val = float(np.corrcoef(x, y)[0, 1])
        corr_rows.append({
            "feature": feat,
            "n": len(vals),
            "pearson_r": round(r_val, 6),
            "abs_r": round(abs(r_val), 6),
            "mean": round(float(x.mean()), 6),
            "std": round(float(x.std()), 6),
        })

    corr_rows.sort(key=lambda d: d["abs_r"], reverse=True)

    corr_csv_path = out_dir / f"{prefix}_correlations.csv"
    with corr_csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature", "n", "pearson_r", "abs_r", "mean", "std"])
        w.writeheader()
        w.writerows(corr_rows)
    print(f"Correlations written to {corr_csv_path}")

    # ------------------------------------------------------------------
    # Quintile win rates
    # ------------------------------------------------------------------
    quintile_rows: list[dict] = []
    baseline_wr = float(outcomes.mean())

    for feat in feature_cols:
        vals_idx = [(float(row[feat]), i) for i, row in enumerate(rows)
                    if row.get(feat) != "" and row.get(feat) is not None]
        if len(vals_idx) < 200:
            continue
        vals_idx.sort(key=lambda t: t[0])
        n = len(vals_idx)
        q_size = n // 5
        for q in range(5):
            start = q * q_size
            end = start + q_size if q < 4 else n
            chunk = vals_idx[start:end]
            chunk_outcomes = [outcomes[i] for _, i in chunk]
            wr = float(np.mean(chunk_outcomes))
            lo = float(chunk[0][0])
            hi = float(chunk[-1][0])
            quintile_rows.append({
                "feature": feat,
                "quintile": q + 1,
                "n": len(chunk),
                "val_lo": round(lo, 6),
                "val_hi": round(hi, 6),
                "win_rate": round(wr, 6),
                "vs_baseline": round(wr - baseline_wr, 6),
            })

    quintile_csv_path = out_dir / f"{prefix}_quintile_winrates.csv"
    with quintile_csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature", "quintile", "n", "val_lo", "val_hi", "win_rate", "vs_baseline"])
        w.writeheader()
        w.writerows(quintile_rows)
    print(f"Quintile win rates written to {quintile_csv_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_rolling_winrate(
        rows=rows,
        outcomes=outcomes,
        top_corr_rows=corr_rows[:6],
        rolling_window=args.rolling_window,
        baseline_wr=baseline_wr,
        out_path=out_dir / f"{prefix}_rolling_winrate.png",
    )

    _plot_late_flow_vs_outcome(
        rows=rows,
        outcomes=outcomes,
        rolling_window=args.rolling_window,
        out_path=out_dir / f"{prefix}_late_flow_vs_outcome.png",
    )

    _plot_ev_simulation(
        rows=rows,
        outcomes=outcomes,
        corr_rows=corr_rows,
        fixed_bet=args.fixed_bet_bnb,
        out_path=out_dir / f"{prefix}_cumulative_bnb.png",
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report_path = out_dir / f"{prefix}_report.md"
    _write_report(
        path=report_path,
        args=args,
        n_rows=len(rows),
        baseline_wr=baseline_wr,
        corr_rows=corr_rows,
        quintile_rows=quintile_rows,
        prefix=prefix,
    )
    print(f"Report written to {report_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _rolling_winrate_for_mask(outcomes: np.ndarray, mask: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling win rate over rounds where mask is True."""
    ys = np.where(mask, outcomes, np.nan)
    # use pandas-style rolling via convolution on non-nan
    result = []
    xs = []
    buf: list[float] = []
    for i, (y, m) in enumerate(zip(ys, mask)):
        if m:
            buf.append(float(outcomes[i]))
            if len(buf) >= window:
                result.append(float(np.mean(buf[-window:])))
                xs.append(i)
    return np.array(xs), np.array(result)


def _plot_rolling_winrate(
    *,
    rows: list[dict],
    outcomes: np.ndarray,
    top_corr_rows: list[dict],
    rolling_window: int,
    baseline_wr: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for ax_idx, cr in enumerate(top_corr_rows[:6]):
        ax = axes_flat[ax_idx]
        feat = cr["feature"]
        r_val = cr["pearson_r"]

        # Get per-row feature value and split into top/bottom quintile
        feat_vals = [(float(row[feat]) if row.get(feat) not in (None, "") else None)
                     for row in rows]
        valid = [(v, i) for i, v in enumerate(feat_vals) if v is not None]
        if not valid:
            continue
        valid_sorted = sorted(valid, key=lambda t: t[0])
        n = len(valid_sorted)
        bottom_20_idx = set(i for _, i in valid_sorted[:n // 5])
        top_20_idx = set(i for _, i in valid_sorted[-(n // 5):])

        mask_top = np.array([i in top_20_idx for i in range(len(rows))])
        mask_bot = np.array([i in bottom_20_idx for i in range(len(rows))])
        mask_all = np.ones(len(rows), dtype=bool)

        xs_all, ys_all = _rolling_winrate_for_mask(outcomes, mask_all, rolling_window)
        xs_top, ys_top = _rolling_winrate_for_mask(outcomes, mask_top, rolling_window)
        xs_bot, ys_bot = _rolling_winrate_for_mask(outcomes, mask_bot, rolling_window)

        ax.plot(xs_all, ys_all, color="gray", linewidth=1.0, alpha=0.6, label="baseline")
        ax.plot(xs_top, ys_top, color="#e6454a", linewidth=1.6, label=f"top 20% {feat}")
        ax.plot(xs_bot, ys_bot, color="#1f77b4", linewidth=1.6, label=f"bot 20% {feat}")
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(baseline_wr, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.set_title(f"{feat}  (r={r_val:+.4f})", fontsize=9)
        ax.set_xlabel("Round index")
        ax.set_ylabel(f"Rolling win rate ({rolling_window}r)")
        ax.legend(fontsize=7)
        ax.set_ylim(0.40, 0.60)

    plt.suptitle("Rolling Win Rate by Feature Quintile", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Rolling win rate plot saved to {out_path}")


def _plot_late_flow_vs_outcome(
    *,
    rows: list[dict],
    outcomes: np.ndarray,
    rolling_window: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: pre_bull_share vs outcome (what you can observe at cutoff)
    ax = axes[0, 0]
    feat = "pre_bull_share"
    vals = [(float(r[feat]), i) for i, r in enumerate(rows) if r.get(feat) not in (None, "")]
    if vals:
        vals_sorted = sorted(vals, key=lambda t: t[0])
        n = len(vals_sorted)
        bins = 20
        bin_size = n // bins
        xs, ys, ns = [], [], []
        for b in range(bins):
            chunk = vals_sorted[b * bin_size: (b + 1) * bin_size]
            mean_val = float(np.mean([v for v, _ in chunk]))
            mean_out = float(np.mean([outcomes[i] for _, i in chunk]))
            xs.append(mean_val)
            ys.append(mean_out)
            ns.append(len(chunk))
        ax.scatter(xs, ys, s=[n / 3 for n in ns], alpha=0.7, color="#1f77b4")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Pre-cutoff Bull share (observable at decision)")
        ax.set_ylabel("Win rate (Bull)")
        ax.set_title("Pre-cutoff pool state vs outcome")

    # Panel 2: final_bull_share vs outcome (NOT observable at decision time)
    ax = axes[0, 1]
    feat = "final_bull_share"
    vals = [(float(r[feat]), i) for i, r in enumerate(rows) if r.get(feat) not in (None, "")]
    if vals:
        vals_sorted = sorted(vals, key=lambda t: t[0])
        n = len(vals_sorted)
        bin_size = n // bins
        xs, ys, ns = [], [], []
        for b in range(bins):
            chunk = vals_sorted[b * bin_size: (b + 1) * bin_size]
            mean_val = float(np.mean([v for v, _ in chunk]))
            mean_out = float(np.mean([outcomes[i] for _, i in chunk]))
            xs.append(mean_val)
            ys.append(mean_out)
            ns.append(len(chunk))
        ax.scatter(xs, ys, s=[n / 3 for n in ns], alpha=0.7, color="#e6454a")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Final pool Bull share (NOT observable at decision)")
        ax.set_ylabel("Win rate (Bull)")
        ax.set_title("Final pool state vs outcome (retrospective only)")

    # Panel 3: post_bull_share vs outcome (late flow direction)
    ax = axes[1, 0]
    feat = "post_bull_share"
    vals = [(float(r[feat]), i) for i, r in enumerate(rows) if r.get(feat) not in (None, "")]
    if vals:
        vals_sorted = sorted(vals, key=lambda t: t[0])
        n = len(vals_sorted)
        bin_size = n // bins
        xs, ys, ns = [], [], []
        for b in range(bins):
            chunk = vals_sorted[b * bin_size: (b + 1) * bin_size]
            mean_val = float(np.mean([v for v, _ in chunk]))
            mean_out = float(np.mean([outcomes[i] for _, i in chunk]))
            xs.append(mean_val)
            ys.append(mean_out)
            ns.append(len(chunk))
        ax.scatter(xs, ys, s=[n / 3 for n in ns], alpha=0.7, color="#2ca02c")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Post-cutoff Bull share (last 17s bets)")
        ax.set_ylabel("Win rate (Bull)")
        ax.set_title("Post-cutoff (late) bet flow vs outcome (retrospective)")

    # Panel 4: bull_share_shift rolling win rate (late flow contrarian/follower)
    ax = axes[1, 1]
    feat = "bull_share_shift"
    vals = [(float(r[feat]), i) for i, r in enumerate(rows) if r.get(feat) not in (None, "")]
    if vals:
        vals_sorted = sorted(vals, key=lambda t: t[0])
        n = len(vals_sorted)
        top20 = set(i for _, i in vals_sorted[-(n // 5):])  # late flow strongly Bull
        bot20 = set(i for _, i in vals_sorted[:n // 5])     # late flow strongly Bear

        mask_top = np.array([i in top20 for i in range(len(rows))])
        mask_bot = np.array([i in bot20 for i in range(len(rows))])
        mask_all = np.ones(len(rows), dtype=bool)

        xs_all, ys_all = _rolling_winrate_for_mask(outcomes, mask_all, rolling_window)
        xs_top, ys_top = _rolling_winrate_for_mask(outcomes, mask_top, rolling_window)
        xs_bot, ys_bot = _rolling_winrate_for_mask(outcomes, mask_bot, rolling_window)

        ax.plot(xs_all, ys_all, color="gray", linewidth=1.0, alpha=0.6, label="baseline")
        ax.plot(xs_top, ys_top, color="#e6454a", linewidth=1.6, label="late flow +Bull (top 20%)")
        ax.plot(xs_bot, ys_bot, color="#1f77b4", linewidth=1.6, label="late flow +Bear (bot 20%)")
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Round index")
        ax.set_ylabel(f"Rolling win rate ({rolling_window}r)")
        ax.set_title("Bull share shift (pre→final) vs outcome — is late flow smart?")
        ax.legend(fontsize=8)
        ax.set_ylim(0.40, 0.60)

    plt.suptitle("Late Bet Flow vs Round Outcome", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Late flow plot saved to {out_path}")


def _plot_ev_simulation(
    *,
    rows: list[dict],
    outcomes: np.ndarray,
    corr_rows: list[dict],
    fixed_bet: float,
    out_path: Path,
) -> None:
    """Simulate fixed-stake bets using top kline signals as directional trigger."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    # Pick top 4 kline features (exclude bet-pool features for live-actionable test)
    kline_feats = [cr for cr in corr_rows if not any(
        cr["feature"].startswith(p) for p in ("pre_", "post_", "final_", "late_", "bull_share_shift")
    )][:4]

    for ax_idx, cr in enumerate(kline_feats):
        ax = axes_flat[ax_idx]
        feat = cr["feature"]
        r_val = cr["pearson_r"]

        feat_vals = [(float(row[feat]) if row.get(feat) not in (None, "") else None)
                     for row in rows]
        valid_vals = [v for v in feat_vals if v is not None]
        if not valid_vals:
            continue
        p80 = float(np.percentile(valid_vals, 80))
        p20 = float(np.percentile(valid_vals, 20))

        # Strategy: if r > 0, bet Bull when feature > p80; if r < 0, bet Bull when < p20
        # (i.e., always bet in the direction the feature implies)
        cum_bnb = []
        total = 0.0
        round_idx = 0
        for i, row in enumerate(rows):
            v = feat_vals[i]
            if v is None:
                continue
            outcome = int(outcomes[i])

            # Decide direction
            if r_val > 0:
                bet_bull = v > p80
                bet_bear = v < p20
            else:
                bet_bull = v < p20
                bet_bear = v > p80

            if not bet_bull and not bet_bear:
                round_idx += 1
                continue

            # Compute payout multipliers
            bull_mult = row.get("bull_net_mult", "")
            bear_mult = row.get("bear_net_mult", "")
            if bull_mult == "" or bear_mult == "":
                round_idx += 1
                continue
            bull_mult = float(bull_mult)
            bear_mult = float(bear_mult)

            if bet_bull:
                profit = fixed_bet * bull_mult if outcome == 1 else -fixed_bet
            else:
                profit = fixed_bet * bear_mult if outcome == 0 else -fixed_bet

            total += profit
            cum_bnb.append(total)
            round_idx += 1

        if not cum_bnb:
            continue

        xs = np.arange(1, len(cum_bnb) + 1)
        ys = np.array(cum_bnb)
        color = "#e6454a" if ys[-1] > 0 else "#1f77b4"
        ax.plot(xs, ys, color=color, linewidth=1.6)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.fill_between(xs, ys, 0, where=(ys > 0), alpha=0.15, color="#2ca02c")
        ax.fill_between(xs, ys, 0, where=(ys < 0), alpha=0.15, color="#e6454a")
        n_bets = len(cum_bnb)
        final = ys[-1]
        ax.set_title(f"{feat}  r={r_val:+.4f}\n{n_bets} bets, final={final:+.4f} BNB", fontsize=9)
        ax.set_xlabel("Bet index")
        ax.set_ylabel("Cumulative BNB")

    plt.suptitle(f"EV Simulation — Top Kline Signals (fixed stake {fixed_bet} BNB, top/bot 20%)", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Cumulative BNB plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    *,
    path: Path,
    args: argparse.Namespace,
    n_rows: int,
    baseline_wr: float,
    corr_rows: list[dict],
    quintile_rows: list[dict],
    prefix: str,
) -> None:
    lines: list[str] = []

    lines.append(f"# Binance Flow Correlation Report — {prefix}\n")
    lines.append(f"- Tail rounds analysed: {args.tail_rounds:,}\n")
    lines.append(f"- Valid rows after filtering: {n_rows:,}\n")
    lines.append(f"- Cutoff seconds: {args.cutoff_seconds}\n")
    lines.append(f"- Baseline win rate (Bull): {baseline_wr:.4f}\n")
    lines.append(f"- Min pool BNB: {args.min_pool_bnb}\n")
    lines.append(f"- Fixed bet (EV sim): {args.fixed_bet_bnb} BNB\n")
    lines.append("\n")

    lines.append("## Top Feature Correlations (|r| ranked)\n\n")
    lines.append("| Feature | N | Pearson r | |r| | Mean | Std |\n")
    lines.append("|---------|---|-----------|-----|------|-----|\n")
    for cr in corr_rows[:20]:
        lines.append(
            f"| {cr['feature']} | {cr['n']:,} | {cr['pearson_r']:+.5f} | "
            f"{cr['abs_r']:.5f} | {cr['mean']:.4f} | {cr['std']:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## Key Win Rates by Quintile (top features)\n\n")
    top_feats = [cr["feature"] for cr in corr_rows[:6]]
    q_by_feat: dict[str, list[dict]] = {}
    for qr in quintile_rows:
        f = qr["feature"]
        if f in top_feats:
            q_by_feat.setdefault(f, []).append(qr)

    for feat in top_feats:
        if feat not in q_by_feat:
            continue
        lines.append(f"### {feat}\n\n")
        lines.append("| Q | N | Val range | Win rate | vs baseline |\n")
        lines.append("|---|---|-----------|----------|-------------|\n")
        for qr in q_by_feat[feat]:
            lines.append(
                f"| {qr['quintile']} | {qr['n']:,} | "
                f"[{qr['val_lo']:.4f}, {qr['val_hi']:.4f}] | "
                f"{qr['win_rate']:.4f} | {qr['vs_baseline']:+.4f} |\n"
            )
        lines.append("\n")

    lines.append("## Interpretation Notes\n\n")
    lines.append(
        "- `pre_bull_share` and `final_bull_share` measure pool composition at cutoff vs final.\n"
        "  A contrarian pattern (high bull_share -> Bear wins) confirms the market self-corrects.\n"
    )
    lines.append(
        "- `post_bull_share` is the direction of late bets (after cutoff). "
        "  If this is predictive, late bettors have an edge you currently can't observe.\n"
    )
    lines.append(
        "- `bull_share_shift` = final_bull_share - pre_bull_share. "
        "  Positive = late money piled Bull. If correlated with Bull wins, late bettors are informed.\n"
    )
    lines.append(
        "- Kline features (ret_Nm, taker_buy_ratio_Nm) test whether CEX price action predicts outcome.\n"
        "  Any stable non-zero r here is the key finding for Option B.\n"
    )
    lines.append(
        "- Rolling win rate plots show whether these signals are regime-stable or episodic.\n"
    )
    lines.append("\n")
    lines.append("## Output Files\n\n")
    lines.append(f"- Row data: `{prefix}_row_data.csv`\n")
    lines.append(f"- Correlations: `{prefix}_correlations.csv`\n")
    lines.append(f"- Quintile win rates: `{prefix}_quintile_winrates.csv`\n")
    lines.append(f"- Rolling win rate plot: `{prefix}_rolling_winrate.png`\n")
    lines.append(f"- Late flow plot: `{prefix}_late_flow_vs_outcome.png`\n")
    lines.append(f"- Cumulative BNB plot: `{prefix}_cumulative_bnb.png`\n")

    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
