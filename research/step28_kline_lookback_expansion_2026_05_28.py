"""Step 28 — Kline lookback expansion (FULL 13-variant run, numpy loader).

Successor to step28_kline_lookback_expansion_2026_05_27.py which ran 4
variants under the Python-list loader (capped at max_lookback=60 due to
~3.1 GB free RAM at start). This run uses the verified-equivalent numpy
kline loader (research/numpy_kline_loader.py) which reduces per-kline
memory from 272 bytes (Python list) -> 48 bytes (numpy float64), letting
us run the full 13-variant matrix including max_lookback=290.

Variants tested (13 total = canonical + 12 candidates):
  Reference (1):
    (3, 7, 15)
  4-tuple anchored at canonical (3,7,15) (4):
    (3, 7, 15, 30), (3, 7, 15, 60), (3, 7, 15, 120), (3, 7, 15, 290)
  4-tuple non-anchored (3):
    (3, 7, 30, 90), (5, 15, 45, 120), (3, 15, 60, 290)
  5-tuple anchored (3):
    (3, 7, 15, 30, 60), (3, 7, 15, 60, 290), (3, 7, 15, 30, 120)
  5-tuple non-anchored (2):
    (3, 7, 15, 60, 240), (3, 7, 15, 45, 180)

Walk-forward CV: TRAIN=15000, TEST=3000, STEP=3000.
All variants share cs=2 (kline_cutoff_seconds) — frozen invariant.
5 BNB initial bankroll (deployable scale).
Permutation null on the best 4-tuple AND best 5-tuple vs canonical.

Outputs:
  - var/strategy_review/step28_kline_lookback_expansion_data_2026_05_28.json
  - var/strategy_review/2026_05_28_step28_kline_lookback_expansion.md
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

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

from research.numpy_kline_loader import (  # noqa: E402
    load_klines_unified_numpy, slice_per_entry_numpy,
)

from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.constants import MAX_GAS_COST_BET_BNB  # noqa: E402
from pancakebot.settlement import settle_bet_against_closed_round  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402
from pancakebot.bankroll_tracker import InMemoryBankrollTracker  # noqa: E402


# Walk-forward design (matches Step 3)
EPOCH_MIN = 422298
EPOCH_MAX = 484999
TRAIN_ROUNDS = 15000
TEST_ROUNDS = 3000
STEP_ROUNDS = 3000

# Hard invariant
CANONICAL_CUTOFF = 2
POOL_CUTOFF = 6

# Strategy / risk constants
TREASURY_FEE = 0.03
MIN_BET = 0.001
DRAWDOWN_PEAK_WINDOW_DAYS = 7
COOLDOWN_ROUNDS = 72
ABS_DD_FRAC = 0.15
INITIAL_BANKROLL = 5.0
SETTLEMENT_VISIBILITY_DELAY = 2  # timing-fixed pattern (Step 24/25/26/27)
PERMUTATION_SEEDS = 1000

# Variants — full 13-variant matrix
CANONICAL = (3, 7, 15)
VARIANTS = [
    ("canonical_3_7_15", (3, 7, 15)),
    # 4-tuple anchored
    ("anchored_3_7_15_30", (3, 7, 15, 30)),
    ("anchored_3_7_15_60", (3, 7, 15, 60)),
    ("anchored_3_7_15_120", (3, 7, 15, 120)),
    ("anchored_3_7_15_290", (3, 7, 15, 290)),
    # 4-tuple non-anchored
    ("free_3_7_30_90", (3, 7, 30, 90)),
    ("free_5_15_45_120", (5, 15, 45, 120)),
    ("free_3_15_60_290", (3, 15, 60, 290)),
    # 5-tuple anchored
    ("anchored_3_7_15_30_60", (3, 7, 15, 30, 60)),
    ("anchored_3_7_15_60_290", (3, 7, 15, 60, 290)),
    ("anchored_3_7_15_30_120", (3, 7, 15, 30, 120)),
    # 5-tuple non-anchored
    ("free_3_7_15_60_240", (3, 7, 15, 60, 240)),
    ("free_3_7_15_45_180", (3, 7, 15, 45, 180)),
]
MAX_LOOKBACK = max(max(lb) for _, lb in VARIANTS)  # = 290


class Step28Tracker(InMemoryBankrollTracker):
    """Timing-fixed tracker matching Step 24/25/26 pattern."""

    def __init__(self, *, initial_bankroll, drawdown_peak_window_days, peak_mode,
                  cooldown_rounds, abs_dd_frac):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cd_total = int(cooldown_rounds)
        self._abs_dd_frac = float(abs_dd_frac)
        self.n_pauses_fired = 0

    def is_paused(self, as_of_start_at):
        if self._cooldown > 0:
            return True
        current = self.current_bankroll()
        peak = self.peak_bankroll(as_of_start_at)
        if peak > 0:
            dd = (peak - current) / peak
            if dd >= self._abs_dd_frac:
                if self._cd_total > 0:
                    self.set_paused(self._cd_total + 1, as_of_start_at)
                self.n_pauses_fired += 1
                return self._cd_total > 0
        return False


def build_windows() -> list[dict[str, int]]:
    windows: list[dict[str, int]] = []
    train_start = EPOCH_MIN
    while True:
        train_end = train_start + TRAIN_ROUNDS - 1
        test_start = train_end + 1
        test_end = test_start + TEST_ROUNDS - 1
        if test_end > EPOCH_MAX:
            break
        windows.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
        })
        train_start += STEP_ROUNDS
    return windows


def run_window(*, lookbacks: tuple[int, ...],
                window: dict[str, int],
                all_rounds: list,
                btc_klines: dict, eth_klines: dict, sol_klines: dict) -> dict[str, Any]:
    """Walk-forward window: train phase warms up tracker/bankroll, test phase
    records bets. Returns dict with test-phase PnL, bets, WR, bet_records."""
    overrides = {
        "gate": {"mtf_lookbacks": list(lookbacks)},
        "risk": {"max_drawdown_fraction_from_peak": 1.0},
    }
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=lookbacks,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    tracker = Step28Tracker(
        initial_bankroll=INITIAL_BANKROLL,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        cooldown_rounds=COOLDOWN_ROUNDS,
        abs_dd_frac=ABS_DD_FRAC,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_cfg, strategy_config=sc, gate=None,
        kline_cutoff_seconds=CANONICAL_CUTOFF, pool_cutoff_seconds=POOL_CUTOFF,
        min_bet_amount_bnb=MIN_BET, treasury_fee_fraction=TREASURY_FEE,
        bankroll_tracker=tracker,
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    train_start, train_end = window["train_start"], window["train_end"]
    test_start, test_end = window["test_start"], window["test_end"]

    sim_rounds = [r for r in all_rounds if train_start <= r.epoch <= test_end]
    sim_rounds.sort(key=lambda r: int(r.epoch))

    bankroll = float(INITIAL_BANKROLL); peak = bankroll; max_dd = 0.0
    test_pnl = 0.0
    test_bets = 0
    test_wins = 0
    test_bet_records: list[dict[str, Any]] = []

    pending_settlements: deque[dict[str, Any]] = deque()

    for round_t in sim_rounds:
        ep = int(round_t.epoch)

        while pending_settlements and pending_settlements[0]["delivery_round_epoch"] <= ep:
            d = pending_settlements.popleft()
            pipeline.record_settlement(bankroll=d["bankroll"], start_at=d["start_at"])

        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        bet_size = float(decision.bet_size_bnb)
        side = str(decision.bet_side)
        bankroll -= bet_size + MAX_GAS_COST_BET_BNB
        outcome = settle_bet_against_closed_round(
            bet_bnb=bet_size, bet_side=side, round_closed=round_t,
            treasury_fee_fraction=TREASURY_FEE,
        )
        bankroll += outcome.credit_bnb
        profit = outcome.credit_bnb - bet_size - MAX_GAS_COST_BET_BNB
        won = outcome.outcome == "win"

        if test_start <= ep <= test_end:
            test_pnl += profit
            test_bets += 1
            if won:
                test_wins += 1
            test_bet_records.append({
                "epoch": ep, "profit": profit, "won": won, "side": side,
            })

        if bankroll > peak: peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd: max_dd = dd

        pending_settlements.append({
            "start_at": int(round_t.start_at),
            "bankroll": bankroll,
            "delivery_round_epoch": ep + SETTLEMENT_VISIBILITY_DELAY,
        })
        pipeline.settle_closed_rounds(rounds=[round_t])

    return {
        "test_pnl": test_pnl,
        "test_bets": test_bets,
        "test_wins": test_wins,
        "test_wr": test_wins / test_bets if test_bets else 0.0,
        "test_bet_records": test_bet_records,
        "final_bankroll": bankroll,
        "max_dd": max_dd,
    }


def permutation_null_test(*, bets_a, bets_b, n_seeds=PERMUTATION_SEEDS, base_seed=42):
    profits_a = [b["profit"] for b in bets_a]
    profits_b = [b["profit"] for b in bets_b]
    obs_D = sum(profits_a) - sum(profits_b)
    pool = profits_a + profits_b
    n_a = len(profits_a)
    rng = random.Random(base_seed)
    perm_Ds = []
    for _ in range(n_seeds):
        rng.shuffle(pool)
        perm_D = sum(pool[:n_a]) - sum(pool[n_a:n_a + len(profits_b)])
        perm_Ds.append(perm_D)
    perm_Ds_sorted = sorted(perm_Ds)
    n_geq = sum(1 for d in perm_Ds if d >= obs_D)
    return {
        "observed_D": obs_D, "n_seeds": n_seeds, "p_value": n_geq / n_seeds,
        "perm_D_mean": statistics.mean(perm_Ds),
        "perm_D_stdev": statistics.stdev(perm_Ds) if len(perm_Ds) > 1 else 0.0,
        "perm_D_p05": perm_Ds_sorted[int(0.05 * n_seeds)],
        "perm_D_p95": perm_Ds_sorted[int(0.95 * n_seeds)],
    }


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a_arr = np.asarray(a, dtype=float); b_arr = np.asarray(b, dtype=float)
    am, bm = float(a_arr.mean()), float(b_arr.mean())
    av, bv = float(a_arr.var(ddof=1)), float(b_arr.var(ddof=1))
    pooled = math.sqrt(((len(a) - 1) * av + (len(b) - 1) * bv) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (am - bm) / pooled


def write_markdown_report(*, results, best_4, best_5, null_4, null_5, canonical_name,
                           windows, elapsed_s, out_path):
    canonical = results[canonical_name]

    def fmt_lookbacks(lb): return f"({', '.join(str(x) for x in lb)})"

    lines = []
    lines.append("# Step 28 — Kline Lookback Expansion (full 13-variant run)")
    lines.append("")
    lines.append(f"**Run date**: 2026-05-28")
    lines.append(f"**Elapsed**: {elapsed_s:.1f}s")
    lines.append(f"**Walk-forward folds**: {len(windows)}  "
                 f"(TRAIN={TRAIN_ROUNDS}, TEST={TEST_ROUNDS}, STEP={STEP_ROUNDS})")
    lines.append(f"**Cutoff**: cs={CANONICAL_CUTOFF} (HARD invariant — unchanged)  ")
    lines.append(f"**Scale**: {INITIAL_BANKROLL} BNB initial bankroll  ")
    lines.append(f"**Max lookback**: {MAX_LOOKBACK}s  ")
    lines.append(f"**Loader**: numpy (research/numpy_kline_loader.py — 5.7× memory reduction vs Python lists)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Headline matrix (sorted by total OOS PnL desc)")
    lines.append("")
    lines.append("| Variant | Lookbacks | OOS PnL (BNB) | Δ vs canon | Bets | WR | $/bet | Fold σ |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    canon_pnl = canonical["total_oos_pnl"]
    rows = sorted(results.items(), key=lambda kv: kv[1]["total_oos_pnl"], reverse=True)
    for name, r in rows:
        delta = r["total_oos_pnl"] - canon_pnl
        delta_str = f"{delta:+.4f}" if name != canonical_name else "—"
        lines.append(
            f"| `{name}` | {fmt_lookbacks(r['lookbacks'])} | "
            f"{r['total_oos_pnl']:+.4f} | {delta_str} | "
            f"{r['total_bets']} | {r['avg_wr']*100:.2f}% | "
            f"{r['avg_pnl_per_bet']:+.5f} | {r['per_fold_pnl_stdev']:.4f} |"
        )
    lines.append("")

    lines.append("## Per-fold breakdown")
    lines.append("")
    lines.append("Each cell is per-fold test PnL (BNB). Folds are ordered chronologically.")
    lines.append("")
    n_folds = len(windows)
    fold_header = " | ".join(f"F{i+1}" for i in range(n_folds))
    lines.append(f"| Variant | {fold_header} | Σ |")
    lines.append("|---" + "|---:" * (n_folds + 1) + "|")
    for name, r in rows:
        pnl_cells = " | ".join(f"{p:+.4f}" for p in r["per_fold_pnl"])
        lines.append(f"| `{name}` | {pnl_cells} | {r['total_oos_pnl']:+.4f} |")
    lines.append("")

    lines.append("## Cohen's d (per-bet profit vs canonical)")
    lines.append("")
    lines.append("| Variant | Cohen's d |")
    lines.append("|---|---:|")
    canonical_bets = [b for f in canonical["fold_results"] for b in f["test_bet_records"]]
    canonical_profits = [b["profit"] for b in canonical_bets]
    for name, r in rows:
        if name == canonical_name:
            continue
        variant_bets = [b for f in r["fold_results"] for b in f["test_bet_records"]]
        d = cohens_d([b["profit"] for b in variant_bets], canonical_profits)
        lines.append(f"| `{name}` | {d:+.4f} |")
    lines.append("")

    lines.append("## Permutation null vs canonical (1000 seeds, right-tail)")
    lines.append("")
    lines.append(f"### Best 4-tuple: `{best_4[0]}` {fmt_lookbacks(best_4[1]['lookbacks'])}")
    lines.append("")
    lines.append(f"- Observed D (variant − canonical): **{null_4['observed_D']:+.4f}** BNB")
    lines.append(f"- Null mean: {null_4['perm_D_mean']:+.4f}  σ: {null_4['perm_D_stdev']:.4f}")
    lines.append(f"- Null p05/p95: {null_4['perm_D_p05']:+.4f} / {null_4['perm_D_p95']:+.4f}")
    lines.append(f"- **p-value (right-tail): {null_4['p_value']:.4f}**")
    lines.append("")
    if best_5:
        lines.append(f"### Best 5-tuple: `{best_5[0]}` {fmt_lookbacks(best_5[1]['lookbacks'])}")
        lines.append("")
        lines.append(f"- Observed D (variant − canonical): **{null_5['observed_D']:+.4f}** BNB")
        lines.append(f"- Null mean: {null_5['perm_D_mean']:+.4f}  σ: {null_5['perm_D_stdev']:.4f}")
        lines.append(f"- Null p05/p95: {null_5['perm_D_p05']:+.4f} / {null_5['perm_D_p95']:+.4f}")
        lines.append(f"- **p-value (right-tail): {null_5['p_value']:.4f}**")
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    deployable_4 = null_4["p_value"] < 0.05 and null_4["observed_D"] > 0
    deployable_5 = bool(best_5) and null_5["p_value"] < 0.05 and null_5["observed_D"] > 0
    if deployable_4 or deployable_5:
        lines.append("**At least one variant clears p<0.05 right-tail with positive observed D:**")
        if deployable_4:
            lines.append(f"- `{best_4[0]}`: D={null_4['observed_D']:+.4f}, p={null_4['p_value']:.4f}")
        if deployable_5:
            lines.append(f"- `{best_5[0]}`: D={null_5['observed_D']:+.4f}, p={null_5['p_value']:.4f}")
        lines.append("")
        lines.append("**Candidates for promotion** — but the promotion rule requires "
                     "passing the frozen holdout + extension_v2 OOS slices independently "
                     "before deployment. This run shows walk-forward CV evidence only.")
    else:
        lines.append(f"**No variant clears p<0.05 right-tail.**")
        lines.append("")
        lines.append(f"- Best 4-tuple `{best_4[0]}`: observed D={null_4['observed_D']:+.4f}, "
                     f"p={null_4['p_value']:.4f}")
        if best_5:
            lines.append(f"- Best 5-tuple `{best_5[0]}`: observed D={null_5['observed_D']:+.4f}, "
                         f"p={null_5['p_value']:.4f}")
        lines.append("")
        lines.append("**Recommendation**: keep canonical (3, 7, 15). The expanded lookbacks "
                     "(including 290s) do not produce a statistically distinguishable improvement "
                     "on this walk-forward CV. The kline_cutoff_seconds=2 invariant remains "
                     "the right anchor; additional lookbacks add noise without signal.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    t_all = time.time()
    print("--- loading rounds ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  {len(all_rounds)} rounds", flush=True)

    earliest_offset = CANONICAL_CUTOFF + MAX_LOOKBACK + 1  # = 293
    latest_offset = CANONICAL_CUTOFF + 1
    print(f"--- loading klines at max_lookback={MAX_LOOKBACK} "
          f"(earliest_offset={earliest_offset}) using numpy loader ---", flush=True)
    t_kl = time.time()
    btc = load_klines_unified_numpy(
        ipr._BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth = load_klines_unified_numpy(
        ipr._ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol = load_klines_unified_numpy(
        ipr._SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  klines loaded ({time.time()-t_kl:.1f}s): "
          f"btc={len(btc)}, eth={len(eth)}, sol={len(sol)}", flush=True)

    # Slice per-entry (numpy view, no copy)
    btc_klines = {ep: slice_per_entry_numpy(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=MAX_LOOKBACK,
                                            earliest_offset=earliest_offset)
                  for ep, kl in btc.items()}
    eth_klines = {ep: slice_per_entry_numpy(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=MAX_LOOKBACK,
                                            earliest_offset=earliest_offset)
                  for ep, kl in eth.items()}
    sol_klines = {ep: slice_per_entry_numpy(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=MAX_LOOKBACK,
                                            earliest_offset=earliest_offset)
                  for ep, kl in sol.items()}

    windows = build_windows()
    print(f"--- walk-forward windows: {len(windows)} ---", flush=True)

    print("\n=== Walk-forward results per variant ===", flush=True)
    results: dict[str, dict[str, Any]] = {}
    for name, lookbacks in VARIANTS:
        t_v = time.time()
        fold_results = []
        for i, w in enumerate(windows):
            r = run_window(
                lookbacks=lookbacks, window=w,
                all_rounds=all_rounds,
                btc_klines=btc_klines, eth_klines=eth_klines, sol_klines=sol_klines,
            )
            fold_results.append(r)

        total_oos_pnl = sum(f["test_pnl"] for f in fold_results)
        total_bets = sum(f["test_bets"] for f in fold_results)
        total_wins = sum(f["test_wins"] for f in fold_results)
        avg_wr = total_wins / total_bets if total_bets else 0.0
        avg_pnl_per_bet = total_oos_pnl / total_bets if total_bets else 0.0
        per_fold_pnl = [f["test_pnl"] for f in fold_results]
        per_fold_pnl_stdev = statistics.stdev(per_fold_pnl) if len(per_fold_pnl) > 1 else 0.0

        results[name] = {
            "lookbacks": lookbacks,
            "total_oos_pnl": total_oos_pnl,
            "total_bets": total_bets,
            "total_wins": total_wins,
            "avg_wr": avg_wr,
            "avg_pnl_per_bet": avg_pnl_per_bet,
            "per_fold_pnl": per_fold_pnl,
            "per_fold_pnl_stdev": per_fold_pnl_stdev,
            "fold_results": fold_results,
        }
        delta = total_oos_pnl - results.get("canonical_3_7_15", results[name])["total_oos_pnl"]
        print(f"  {name:>30s} {str(lookbacks):>30s}: "
              f"OOS_pnl={total_oos_pnl:+.4f} delta={delta:+.4f} "
              f"bets={total_bets} WR={avg_wr*100:.2f}% "
              f"avg/bet={avg_pnl_per_bet:+.5f} fold_stdev={per_fold_pnl_stdev:.4f} "
              f"({time.time()-t_v:.1f}s)", flush=True)

    four_tuple_variants = {k: v for k, v in results.items()
                           if k != "canonical_3_7_15" and len(v["lookbacks"]) == 4}
    five_tuple_variants = {k: v for k, v in results.items()
                            if len(v["lookbacks"]) == 5}
    best_4 = max(four_tuple_variants.items(), key=lambda kv: kv[1]["total_oos_pnl"])
    best_5 = max(five_tuple_variants.items(), key=lambda kv: kv[1]["total_oos_pnl"]) if five_tuple_variants else None

    canonical = results["canonical_3_7_15"]
    canonical_bets = [b for f in canonical["fold_results"] for b in f["test_bet_records"]]

    print(f"\n  best 4-tuple: {best_4[0]} OOS_pnl={best_4[1]['total_oos_pnl']:+.4f} "
          f"(canonical={canonical['total_oos_pnl']:+.4f})", flush=True)
    if best_5:
        print(f"  best 5-tuple: {best_5[0]} OOS_pnl={best_5[1]['total_oos_pnl']:+.4f}", flush=True)

    print("\n=== Cohen's d (per-bet profit vs canonical) ===", flush=True)
    for name, r in results.items():
        if name == "canonical_3_7_15":
            continue
        variant_bets = [b for f in r["fold_results"] for b in f["test_bet_records"]]
        d = cohens_d([b["profit"] for b in variant_bets], [b["profit"] for b in canonical_bets])
        print(f"  {name:>30s}: d={d:+.4f}", flush=True)

    print("\n=== Permutation null vs canonical (1000 seeds) ===", flush=True)
    best_4_bets = [b for f in best_4[1]["fold_results"] for b in f["test_bet_records"]]
    null_4 = permutation_null_test(bets_a=best_4_bets, bets_b=canonical_bets)
    print(f"  Best 4-tuple ({best_4[0]}):", flush=True)
    print(f"    Observed D: {null_4['observed_D']:+.4f}", flush=True)
    print(f"    p-value: {null_4['p_value']:.4f}", flush=True)

    null_5 = None
    if best_5:
        best_5_bets = [b for f in best_5[1]["fold_results"] for b in f["test_bet_records"]]
        null_5 = permutation_null_test(bets_a=best_5_bets, bets_b=canonical_bets)
        print(f"\n  Best 5-tuple ({best_5[0]}):", flush=True)
        print(f"    Observed D: {null_5['observed_D']:+.4f}", flush=True)
        print(f"    p-value: {null_5['p_value']:.4f}", flush=True)

    elapsed_s = time.time() - t_all

    # JSON output (strip fold_results to keep size sane)
    def strip(r):
        return {k: v for k, v in r.items() if k != "fold_results"}

    out_json = REPO / "var" / "strategy_review" / "step28_kline_lookback_expansion_data_2026_05_28.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "train_rounds": TRAIN_ROUNDS, "test_rounds": TEST_ROUNDS,
                "step_rounds": STEP_ROUNDS, "max_lookback": MAX_LOOKBACK,
                "initial_bankroll": INITIAL_BANKROLL,
                "settlement_visibility_delay": SETTLEMENT_VISIBILITY_DELAY,
                "permutation_seeds": PERMUTATION_SEEDS,
                "loader": "numpy",
            },
            "results": {k: strip(v) for k, v in results.items()},
            "best_4_tuple_name": best_4[0],
            "best_5_tuple_name": best_5[0] if best_5 else None,
            "permutation_null_best_4": null_4,
            "permutation_null_best_5": null_5,
            "elapsed_seconds": elapsed_s,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_json}", flush=True)

    # Markdown report
    out_md = REPO / "var" / "strategy_review" / "2026_05_28_step28_kline_lookback_expansion.md"
    write_markdown_report(
        results=results, best_4=best_4, best_5=best_5,
        null_4=null_4, null_5=null_5,
        canonical_name="canonical_3_7_15",
        windows=windows, elapsed_s=elapsed_s,
        out_path=out_md,
    )
    print(f"wrote {out_md}", flush=True)
    print(f"total elapsed: {elapsed_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
