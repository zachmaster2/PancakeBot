"""Phase 1 — Partial-vs-final pool gap analysis across all closed rounds.

Read-only. Reads var/closed_rounds.jsonl, computes per-round gap stats using
production _pools_from_bets, and writes a text summary plus PNG plots to
var/research/final_pool_analysis/.

Usage (from repo root):
    python research/phase1_pool_gap_analysis.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pancakebot.constants import BNB_WEI, POOL_CUTOFF_SECONDS
from pancakebot.strategy.momentum_pipeline import _pools_from_bets
from pancakebot.types import Round

REPO_ROOT = Path(__file__).resolve().parent.parent
CLOSED_ROUNDS_PATH = REPO_ROOT / "var" / "closed_rounds.jsonl"
OUT_DIR = REPO_ROOT / "var" / "research" / "final_pool_analysis"


def _load_rounds() -> list[Round]:
    rounds: list[Round] = []
    with CLOSED_ROUNDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rounds.append(Round.from_json(obj))
    return rounds


def _per_round_stats(round_t: Round) -> dict | None:
    """Compute all derived quantities for one round. Returns None if we must
    skip (failed round, missing price, or pathological zero pools)."""
    lock_at = round_t.lock_at
    if lock_at is None:
        return None
    cutoff_ts = lock_at - POOL_CUTOFF_SECONDS

    # Partial pool at cutoff (reuse production function).
    partial_bull, partial_bear = _pools_from_bets(round_t, cutoff_ts)

    # Final pool.
    final_bull_wei = sum(b.amount_wei for b in round_t.bets if b.position == "Bull")
    final_bear_wei = sum(b.amount_wei for b in round_t.bets if b.position == "Bear")
    final_bull = float(final_bull_wei) / float(BNB_WEI)
    final_bear = float(final_bear_wei) / float(BNB_WEI)

    partial_total = partial_bull + partial_bear
    final_total = final_bull + final_bear

    late_bull = final_bull - partial_bull
    late_bear = final_bear - partial_bear
    late_total = late_bull + late_bear

    # Skip rounds whose FINAL pool is zero — degenerate.
    if final_total <= 0:
        return None

    late_share_total = late_total / final_total
    late_share_bull = (late_bull / final_bull) if final_bull > 0 else np.nan
    late_share_bear = (late_bear / final_bear) if final_bear > 0 else np.nan

    # Winner side from round.position ("Bull" / "Bear" / None). Failed rounds
    # have position=None → skip payout-error calc.
    winner = round_t.position  # "Bull" | "Bear" | None
    winner_valid = winner in ("Bull", "Bear") and not bool(round_t.failed)

    partial_winner = np.nan
    final_winner = np.nan
    gate_payout = np.nan      # partial_total / partial_winner (bot's real view)
    mixed_payout = np.nan     # final_total / partial_winner (user's literal)
    final_payout = np.nan
    gate_error = np.nan
    mixed_error = np.nan

    if winner_valid:
        if winner == "Bull":
            partial_winner = partial_bull
            final_winner = final_bull
        else:
            partial_winner = partial_bear
            final_winner = final_bear

        if partial_winner > 0 and final_winner > 0:
            gate_payout = partial_total / partial_winner
            mixed_payout = final_total / partial_winner
            final_payout = final_total / final_winner
            gate_error = gate_payout - final_payout
            mixed_error = mixed_payout - final_payout

    # Recent-bet-arrival-rate: number of bets with
    # cutoff - 20 <= created_at < cutoff. Measures pre-cutoff momentum.
    recent_start = cutoff_ts - 20
    recent_bets = sum(
        1 for b in round_t.bets
        if recent_start <= b.created_at < cutoff_ts
    )

    # Partial side ratio at cutoff (bull share of partial pool).
    bull_share_partial = (partial_bull / partial_total) if partial_total > 0 else np.nan

    # Late side skew: (late_bull - late_bear) / (late_bull + late_bear).
    if late_total > 1e-12 or late_total < -1e-12:
        late_side_skew = (late_bull - late_bear) / late_total
    else:
        late_side_skew = np.nan

    # Time bins from start_at.
    dt = datetime.fromtimestamp(round_t.start_at, tz=timezone.utc)
    hour_of_day = dt.hour
    day_of_week = dt.weekday()

    return {
        "epoch": round_t.epoch,
        "start_at": round_t.start_at,
        "lock_at": lock_at,
        "position": winner,
        "failed": bool(round_t.failed),
        "num_bets": len(round_t.bets),
        "partial_bull": partial_bull,
        "partial_bear": partial_bear,
        "partial_total": partial_total,
        "final_bull": final_bull,
        "final_bear": final_bear,
        "final_total": final_total,
        "late_bull": late_bull,
        "late_bear": late_bear,
        "late_total": late_total,
        "late_share_total": late_share_total,
        "late_share_bull": late_share_bull,
        "late_share_bear": late_share_bear,
        "late_side_skew": late_side_skew,
        "partial_winner": partial_winner,
        "final_winner": final_winner,
        "gate_payout": gate_payout,
        "mixed_payout": mixed_payout,
        "final_payout": final_payout,
        "gate_error": gate_error,
        "mixed_error": mixed_error,
        "recent_bets_20s": recent_bets,
        "bull_share_partial": bull_share_partial,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
    }


def _dist_describe(s: pd.Series) -> dict[str, float]:
    """Common distribution summary for a numeric series (NaNs dropped)."""
    x = s.dropna()
    if x.empty:
        return {k: float("nan") for k in ("n", "mean", "median", "std", "p10", "p25", "p75", "p90", "p99", "max")}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std()),
        "p10": float(x.quantile(0.10)),
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
        "p90": float(x.quantile(0.90)),
        "p99": float(x.quantile(0.99)),
        "max": float(x.max()),
    }


def _fmt_desc(name: str, d: dict[str, float]) -> str:
    return (
        f"  {name:<22}  n={d['n']:>6}  mean={d['mean']:+.4f}  median={d['median']:+.4f}  "
        f"std={d['std']:.4f}  p10={d['p10']:+.4f}  p25={d['p25']:+.4f}  "
        f"p75={d['p75']:+.4f}  p90={d['p90']:+.4f}  p99={d['p99']:+.4f}  max={d['max']:+.4f}"
    )


def main() -> int:
    print(f"Loading {CLOSED_ROUNDS_PATH} ...", flush=True)
    rounds = _load_rounds()
    print(f"  loaded {len(rounds)} rounds", flush=True)

    print("Computing per-round stats ...", flush=True)
    rows = []
    skipped = 0
    for r in rounds:
        rec = _per_round_stats(r)
        if rec is None:
            skipped += 1
            continue
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"  kept {len(df)} rounds; skipped {skipped} (failed/degenerate)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "phase1_report.txt"

    lines: list[str] = []
    lines.append("=" * 110)
    lines.append("PHASE 1 — Partial-vs-Final Pool Gap Analysis")
    lines.append(f"Source: {CLOSED_ROUNDS_PATH}")
    lines.append(f"Rounds loaded: {len(rounds)}  kept: {len(df)}  skipped: {skipped}")
    lines.append(f"Epoch range: {int(df['epoch'].min())} -> {int(df['epoch'].max())}")
    lines.append("=" * 110)

    # 1. Late-share distributions.
    lines.append("\n[1] LATE-ARRIVAL POOL SHARE")
    lines.append("(fraction of final pool that arrives AFTER cutoff = lock_at - 6s)")
    for col in ("late_share_total", "late_share_bull", "late_share_bear"):
        lines.append(_fmt_desc(col, _dist_describe(df[col])))

    # Also express as BNB absolute values for context.
    lines.append("\n    absolute late BNB (how much pool the gate misses):")
    for col in ("late_total", "late_bull", "late_bear"):
        lines.append(_fmt_desc(col, _dist_describe(df[col])))

    # 2. Payout error distributions.
    lines.append("\n[2] PAYOUT-RATIO ERROR (gate view vs actual settlement)")
    lines.append("gate_payout  = partial_total / partial_winner_side  (what the bot actually computes at cutoff)")
    lines.append("mixed_payout = final_total   / partial_winner_side  (literal per-spec — isolates winner-side shift)")
    lines.append("final_payout = final_total   / final_winner_side    (truth at settlement)")

    df_valid = df[df["final_payout"].notna()].copy()
    for col in ("gate_payout", "mixed_payout", "final_payout"):
        lines.append(_fmt_desc(col, _dist_describe(df_valid[col])))
    lines.append("")
    for col in ("gate_error", "mixed_error"):
        lines.append(_fmt_desc(col, _dist_describe(df_valid[col])))

    # Sign split on gate_error and mixed_error.
    for col in ("gate_error", "mixed_error"):
        x = df_valid[col].dropna()
        pct_over = float((x > 0).sum() / x.size * 100)
        pct_under = float((x < 0).sum() / x.size * 100)
        pct_zero = float((x == 0).sum() / x.size * 100)
        lines.append(f"    {col:<13}  positive (gate overestimated payout): {pct_over:.1f}%"
                     f"   negative (gate underestimated): {pct_under:.1f}%"
                     f"   exactly zero: {pct_zero:.1f}%")

    # 3. Momentum asymmetry: late bets vs winner side.
    lines.append("\n[3] LATE-SIDE SKEW vs WINNER SIDE")
    lines.append("late_side_skew = (late_bull - late_bear) / (late_bull + late_bear)")
    lines.append("  +1.0 = all late money Bull; -1.0 = all late money Bear; 0 = symmetric")
    for winner_label in ("Bull", "Bear"):
        grp = df_valid[df_valid["position"] == winner_label]["late_side_skew"].dropna()
        lines.append(f"  winner={winner_label:<4}  n={len(grp):>5}  mean_skew={grp.mean():+.4f}  "
                     f"median_skew={grp.median():+.4f}")
    # Interpretation per winner-side.
    bull_wins = df_valid[df_valid["position"] == "Bull"]
    bear_wins = df_valid[df_valid["position"] == "Bear"]
    # A positive mean for Bull-wins and negative mean for Bear-wins means late flow
    # tilts TOWARD the winner (dumb-money-reacts-to-move pattern). Opposite sign =
    # late flow is contrarian to eventual outcome.

    # 4. Cross-sections of late_share_total.
    lines.append("\n[4] LATE-SHARE CROSS-SECTIONS")

    # 4a. By hour-of-day.
    lines.append("  4a. By hour-of-day (UTC)")
    hour_grp = df.groupby("hour_of_day")["late_share_total"].agg(["mean", "median", "count"])
    lines.append(f"  {'hour':>4}  {'mean':>8}  {'median':>8}  {'n':>6}")
    for h, row in hour_grp.iterrows():
        lines.append(f"  {int(h):>4}  {row['mean']:>8.4f}  {row['median']:>8.4f}  {int(row['count']):>6}")

    # 4b. By day-of-week (0=Mon ... 6=Sun).
    lines.append("\n  4b. By day-of-week (0=Mon, 6=Sun)")
    dow_grp = df.groupby("day_of_week")["late_share_total"].agg(["mean", "median", "count"])
    lines.append(f"  {'dow':>4}  {'mean':>8}  {'median':>8}  {'n':>6}")
    for d, row in dow_grp.iterrows():
        lines.append(f"  {int(d):>4}  {row['mean']:>8.4f}  {row['median']:>8.4f}  {int(row['count']):>6}")

    # 4c. By partial_total quintile.
    lines.append("\n  4c. By partial_total size quintile (Q1=smallest pool)")
    df["pool_q"] = pd.qcut(df["partial_total"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    pool_grp = df.groupby("pool_q", observed=True)["late_share_total"].agg(["mean", "median", "count"])
    bounds = df["partial_total"].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    lines.append(f"  {'quintile':>8}  {'partial_range':<22}  {'mean':>8}  {'median':>8}  {'n':>6}")
    for i, q in enumerate(q_labels):
        row = pool_grp.loc[q]
        rng = f"[{bounds[i]:.3f}, {bounds[i+1]:.3f}]"
        lines.append(f"  {q:>8}  {rng:<22}  {row['mean']:>8.4f}  {row['median']:>8.4f}  {int(row['count']):>6}")

    # 4d. Volatility cross-section — skipped (not wired up per user).
    lines.append("\n  4d. BNB volatility cross-section: SKIPPED (per user spec — skip if not wired up)")

    # 5. Feature correlations.
    lines.append("\n[5] FEATURE CORRELATIONS with late_share_total (Pearson)")
    feats = [
        "partial_total",
        "recent_bets_20s",
        "bull_share_partial",
        "hour_of_day",
        "day_of_week",
        "num_bets",
    ]
    corrs = []
    for f in feats:
        x = df[[f, "late_share_total"]].dropna()
        if x.empty:
            corrs.append((f, float("nan"), 0))
            continue
        c = float(x[f].corr(x["late_share_total"]))
        corrs.append((f, c, len(x)))
    corrs.sort(key=lambda t: abs(t[1]), reverse=True)
    lines.append(f"  {'feature':<22}  {'pearson':>10}  {'|r|':>8}  {'n':>7}")
    for f, c, n in corrs:
        lines.append(f"  {f:<22}  {c:>+10.4f}  {abs(c):>8.4f}  {n:>7}")

    # Plots.
    lines.append("\n[6] PLOTS")
    plt.rcParams["figure.dpi"] = 120

    # 6a. Histogram of late_share_total.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df["late_share_total"].dropna().clip(lower=-0.1, upper=1.0)
    ax.hist(x, bins=80, color="#1f77b4", alpha=0.8)
    ax.axvline(x.mean(), color="red", linestyle="--", label=f"mean={x.mean():.4f}")
    ax.axvline(x.median(), color="orange", linestyle="--", label=f"median={x.median():.4f}")
    ax.set_xlabel("late_share_total = (final_total - partial_total) / final_total")
    ax.set_ylabel("rounds")
    ax.set_title("Distribution of late-arrival pool share")
    ax.legend()
    p = OUT_DIR / "hist_late_share_total.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    lines.append(f"  saved {p.name}")

    # 6b. Scatter: partial_total vs final_total.
    fig, ax = plt.subplots(figsize=(8, 8))
    xs = df["partial_total"].values
    ys = df["final_total"].values
    ax.scatter(xs, ys, s=2, alpha=0.2, color="#1f77b4")
    lim = max(xs.max(), ys.max()) * 1.02
    ax.plot([0, lim], [0, lim], "r--", label="y = x (no-late-arrivals baseline)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("partial_total BNB (at cutoff)")
    ax.set_ylabel("final_total BNB (at settlement)")
    ax.set_title("Partial vs final pool size")
    ax.legend()
    p = OUT_DIR / "scatter_partial_vs_final_total.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    lines.append(f"  saved {p.name}")

    # 6c. Scatter: gate_payout vs final_payout (the gate-relevance picture).
    fig, ax = plt.subplots(figsize=(8, 8))
    dv = df_valid[["gate_payout", "final_payout"]].dropna()
    xs = dv["gate_payout"].values.clip(max=10)
    ys = dv["final_payout"].values.clip(max=10)
    ax.scatter(xs, ys, s=2, alpha=0.2, color="#2ca02c")
    lim = max(float(xs.max()), float(ys.max())) * 1.05
    ax.plot([0, lim], [0, lim], "r--", label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("gate_payout = partial_total / partial_winner  (bot's cutoff view)")
    ax.set_ylabel("final_payout = final_total / final_winner  (truth)")
    ax.set_title("Gate vs actual payout ratio (winner side)")
    ax.legend()
    p = OUT_DIR / "scatter_gate_vs_final_payout.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    lines.append(f"  saved {p.name}")

    # 6d. Heatmap: hour-of-day × day-of-week mean late_share_total.
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot_table(
        values="late_share_total",
        index="day_of_week",
        columns="hour_of_day",
        aggfunc="mean",
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("hour-of-day (UTC)")
    ax.set_ylabel("day-of-week")
    ax.set_title("Mean late_share_total by (dow × hour)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean late_share_total")
    p = OUT_DIR / "heatmap_dow_hour_late_share.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    lines.append(f"  saved {p.name}")

    # Write report.
    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Report written: {report_path}")
    print(f"Plots in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
