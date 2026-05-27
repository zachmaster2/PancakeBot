"""Step 25 — diagnose what loss_cd_1 actually catches.

User insight: with production-faithful timing, loss_cd_1 skips the round
at observation-time offset +2 from a losing bet (round N's loss is first
visible at decision N+2). So the "danger zone" is offset +2 in real time.

Five tests, all using Step 24's timing-fixed tracker:
  T1: Per-offset bet EV after a loss (offsets {+2,+3,+4,+5,+6,+8,+12})
  T2: Per-offset bet EV after a WIN (control)
  T3: Cumulative-loss conditional (last K=5 bets: 0..5 losses)
  T4: Direction-specific (BULL-loss vs BEAR-loss; BULL-bet vs BEAR-bet next)
  T5: Pool-state delta at offset +2 (bull_ratio at L, L+1, L+2)
"""
from __future__ import annotations

import json
import math
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

from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.constants import MAX_GAS_COST_BET_BNB, BNB_WEI  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402
from pancakebot.settlement import settle_bet_against_closed_round  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402
from pancakebot.bankroll_tracker import InMemoryBankrollTracker  # noqa: E402


EPOCH_MIN = 422298
EPOCH_MAX = 484999
CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2
POOL_CUTOFF = 6
TREASURY_FEE = 0.03
MIN_BET = 0.001
DRAWDOWN_PEAK_WINDOW_DAYS = 7
ABS_DD_FRAC = 0.15
INITIAL_BANKROLL = 5.0
COOLDOWN_ROUNDS = 72
SETTLEMENT_VISIBILITY_DELAY = 2  # matches Step 24

# Observation-time offsets (in epochs since the loss round)
OFFSETS = (2, 3, 4, 5, 6, 8, 12)
LOSS_LOOKBACK_K = 5
BOOTSTRAP_SEEDS = 1000

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


# ============================================================
# Timing-fixed tracker (same pattern as Step 24)
# ============================================================

class Step25Tracker(InMemoryBankrollTracker):
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


# ============================================================
# Timing-fixed backtest with per-bet logging + pool capture
# ============================================================

def run_baseline(all_rounds, btc_klines, eth_klines, sol_klines):
    overrides = {
        "gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)},
        "risk": {"max_drawdown_fraction_from_peak": 1.0},
    }
    sc = load_strategy_config_from_dict(overrides)
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    tracker = Step25Tracker(
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

    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX]
    sim_rounds.sort(key=lambda r: int(r.epoch))

    # Pre-compute pool ratios for ALL rounds (regardless of bot bet) — needed
    # for Test 5 which needs pool state at L+1 and L+2 even if the bot didn't
    # bet on those rounds.
    pool_ratio_by_epoch: dict[int, float] = {}
    for r in sim_rounds:
        ep = int(r.epoch)
        if not r.bets:
            continue
        try:
            pa = compute_pool_amounts_wei(bets=r.bets)
            bull = pa.bull_wei / BNB_WEI
            bear = pa.bear_wei / BNB_WEI
            total = bull + bear
            if total > 0:
                pool_ratio_by_epoch[ep] = bull / total
        except Exception:
            pass

    bankroll = float(INITIAL_BANKROLL); peak = bankroll; max_dd_frac = 0.0
    bet_records: list[dict[str, Any]] = []
    pending_settlements: deque[dict[str, Any]] = deque()

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)

        # Deliver pending settlements at this round's decision
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

        pool_ratio = pool_ratio_by_epoch.get(ep)
        bet_records.append({
            "epoch": ep, "start_at": int(round_t.start_at), "cohort": coh,
            "side": side, "won": won, "profit": profit, "bet_size": bet_size,
            "pool_ratio": pool_ratio,
        })

        if bankroll > peak: peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac: max_dd_frac = dd

        pending_settlements.append({
            "start_at": int(round_t.start_at),
            "bankroll": bankroll,
            "delivery_round_epoch": ep + SETTLEMENT_VISIBILITY_DELAY,
        })
        pipeline.settle_closed_rounds(rounds=[round_t])

    return bet_records, bankroll - INITIAL_BANKROLL, pool_ratio_by_epoch


# ============================================================
# Bootstrap CI
# ============================================================

def bootstrap_ci_mean(values, n_boot=BOOTSTRAP_SEEDS, alpha=0.05, seed=42):
    if not values:
        return (None, None)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    means.sort()
    lo = float(means[int(alpha / 2 * n_boot)])
    hi = float(means[int((1 - alpha / 2) * n_boot)])
    return (lo, hi)


# ============================================================
# Analyses
# ============================================================

def t1_per_offset_post_loss(bet_records, offsets):
    """For each bet B, find the most recent PRIOR loss L within max(offsets)
    rounds. If exists, record (offset = B.epoch - L.epoch, B.profit)."""
    by_offset: dict[int, list[float]] = {o: [] for o in offsets}
    bet_epochs = [b["epoch"] for b in bet_records]
    # Build "loss" set (epoch -> True if that bet lost)
    losses = {b["epoch"] for b in bet_records if not b["won"]}
    loss_epochs_sorted = sorted(losses)
    max_offset = max(offsets)

    for B in bet_records:
        ep_b = B["epoch"]
        # Find prior loss within max_offset epochs
        idx = np.searchsorted(loss_epochs_sorted, ep_b) - 1
        if idx < 0:
            continue
        L_ep = loss_epochs_sorted[idx]
        if L_ep == ep_b:
            # Same round (shouldn't happen — loss is a closed bet)
            idx -= 1
            if idx < 0:
                continue
            L_ep = loss_epochs_sorted[idx]
        offset = ep_b - L_ep
        if offset in by_offset:
            by_offset[offset].append(B["profit"])
    return by_offset


def t2_per_offset_post_win(bet_records, offsets):
    """Same as T1 but conditioning on PRIOR WIN."""
    by_offset: dict[int, list[float]] = {o: [] for o in offsets}
    wins = {b["epoch"] for b in bet_records if b["won"]}
    win_epochs_sorted = sorted(wins)
    max_offset = max(offsets)

    for B in bet_records:
        ep_b = B["epoch"]
        idx = np.searchsorted(win_epochs_sorted, ep_b) - 1
        if idx < 0:
            continue
        W_ep = win_epochs_sorted[idx]
        if W_ep == ep_b:
            idx -= 1
            if idx < 0:
                continue
            W_ep = win_epochs_sorted[idx]
        offset = ep_b - W_ep
        if offset in by_offset:
            by_offset[offset].append(B["profit"])
    return by_offset


def t3_cumulative_loss_buckets(bet_records, k=LOSS_LOOKBACK_K):
    """For each bet B (starting from index k onward), count losses in the
    previous k bets. Bucket B's profit by loss count {0,1,2,3,4,5}."""
    by_count: dict[int, list[float]] = {i: [] for i in range(k + 1)}
    for i in range(k, len(bet_records)):
        prev_k = bet_records[i - k:i]
        loss_count = sum(1 for b in prev_k if not b["won"])
        by_count[loss_count].append(bet_records[i]["profit"])
    return by_count


def t4_direction_specific(bet_records, offsets):
    """For each bet B, find prior loss L. Bucket B's profit by (L.side, B.side).
    L_BULL = loss when bet was BULL (so market closed BEAR).
    L_BEAR = loss when bet was BEAR (so market closed BULL).
    Returns {(L_side, B_side, offset): [profits]}."""
    bucket: dict[tuple, list[float]] = {}
    losses_by_epoch = {b["epoch"]: b["side"] for b in bet_records if not b["won"]}
    loss_epochs_sorted = sorted(losses_by_epoch.keys())

    for B in bet_records:
        ep_b = B["epoch"]
        idx = np.searchsorted(loss_epochs_sorted, ep_b) - 1
        if idx < 0:
            continue
        L_ep = loss_epochs_sorted[idx]
        if L_ep == ep_b:
            idx -= 1
            if idx < 0:
                continue
            L_ep = loss_epochs_sorted[idx]
        offset = ep_b - L_ep
        if offset not in offsets:
            continue
        L_side = losses_by_epoch[L_ep]
        key = (L_side, B["side"], offset)
        bucket.setdefault(key, []).append(B["profit"])
    return bucket


def t5_pool_delta(bet_records, pool_ratio_by_epoch):
    """For each loss L, capture pool ratios at L, L+1, L+2.
    Returns aligned arrays."""
    pool_at_L: list[float] = []
    pool_at_L1: list[float] = []
    pool_at_L2: list[float] = []
    for B in bet_records:
        if B["won"]:
            continue
        L_ep = B["epoch"]
        r0 = pool_ratio_by_epoch.get(L_ep)
        r1 = pool_ratio_by_epoch.get(L_ep + 1)
        r2 = pool_ratio_by_epoch.get(L_ep + 2)
        if r0 is not None and r1 is not None and r2 is not None:
            pool_at_L.append(r0)
            pool_at_L1.append(r1)
            pool_at_L2.append(r2)
    return pool_at_L, pool_at_L1, pool_at_L2


# ============================================================
# Main
# ============================================================

def main():
    t_all = time.time()
    print("--- loading rounds + klines ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  {len(all_rounds)} rounds", flush=True)

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1
    t = time.time()
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  klines loaded ({time.time()-t:.1f}s)", flush=True)
    btc_kl = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                        max_lookback=max_lookback,
                                        earliest_offset=earliest_offset)
              for ep, kl in btc.items()}
    eth_kl = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                        max_lookback=max_lookback,
                                        earliest_offset=earliest_offset)
              for ep, kl in eth.items()}
    sol_kl = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                        max_lookback=max_lookback,
                                        earliest_offset=earliest_offset)
              for ep, kl in sol.items()}

    # ----- Run timing-fixed baseline -----
    print("\n--- timing-fixed baseline (cd=72, dd=0.15, 5 BNB) ---", flush=True)
    t = time.time()
    bet_records, total_pnl, pool_ratios = run_baseline(all_rounds, btc_kl, eth_kl, sol_kl)
    print(f"  bets={len(bet_records)} pnl={total_pnl:+.4f} ({time.time()-t:.1f}s)", flush=True)
    n_losses = sum(1 for b in bet_records if not b["won"])
    n_wins = len(bet_records) - n_losses
    baseline_edge = sum(b["profit"] for b in bet_records) / len(bet_records)
    print(f"  losses={n_losses}  wins={n_wins}  baseline edge={baseline_edge:+.5f} BNB/bet", flush=True)

    # ----- Test 1: per-offset post-loss EV -----
    print("\n=== T1: Per-offset bet EV AFTER A LOSS ===", flush=True)
    t1 = t1_per_offset_post_loss(bet_records, OFFSETS)
    print(f"  {'offset':>7s}  {'n':>5s}  {'mean PnL':>10s}  {'95% CI':>22s}  {'vs edge':>9s}", flush=True)
    for o in OFFSETS:
        vals = t1[o]
        if not vals:
            print(f"  +{o:>5d}  n=0  (no bets at this exact offset)", flush=True)
            continue
        m = float(np.mean(vals))
        lo, hi = bootstrap_ci_mean(vals)
        diff = m - baseline_edge
        flag = "  LO" if m < baseline_edge - 0.01 else ("  HI" if m > baseline_edge + 0.01 else "  ok")
        print(f"  +{o:>5d}  {len(vals):>5d}  {m:>+10.5f}  [{lo:>+.5f}, {hi:>+.5f}]  {diff:>+9.5f}{flag}", flush=True)

    # ----- Test 2: per-offset post-win EV (control) -----
    print("\n=== T2: Per-offset bet EV AFTER A WIN (control) ===", flush=True)
    t2 = t2_per_offset_post_win(bet_records, OFFSETS)
    print(f"  {'offset':>7s}  {'n':>5s}  {'mean PnL':>10s}  {'95% CI':>22s}  {'vs edge':>9s}", flush=True)
    for o in OFFSETS:
        vals = t2[o]
        if not vals:
            print(f"  +{o:>5d}  n=0", flush=True)
            continue
        m = float(np.mean(vals))
        lo, hi = bootstrap_ci_mean(vals)
        diff = m - baseline_edge
        flag = "  LO" if m < baseline_edge - 0.01 else ("  HI" if m > baseline_edge + 0.01 else "  ok")
        print(f"  +{o:>5d}  {len(vals):>5d}  {m:>+10.5f}  [{lo:>+.5f}, {hi:>+.5f}]  {diff:>+9.5f}{flag}", flush=True)

    # ----- Test 3: cumulative loss conditional -----
    print(f"\n=== T3: Cumulative loss in last {LOSS_LOOKBACK_K} bets ===", flush=True)
    t3 = t3_cumulative_loss_buckets(bet_records, k=LOSS_LOOKBACK_K)
    print(f"  {'losses':>7s}  {'n':>5s}  {'mean PnL':>10s}  {'95% CI':>22s}  {'vs edge':>9s}", flush=True)
    for k in range(LOSS_LOOKBACK_K + 1):
        vals = t3[k]
        if not vals:
            continue
        m = float(np.mean(vals))
        lo, hi = bootstrap_ci_mean(vals)
        diff = m - baseline_edge
        flag = "  LO" if m < baseline_edge - 0.01 else ("  HI" if m > baseline_edge + 0.01 else "  ok")
        print(f"  {k:>7d}  {len(vals):>5d}  {m:>+10.5f}  [{lo:>+.5f}, {hi:>+.5f}]  {diff:>+9.5f}{flag}", flush=True)

    # ----- Test 4: direction-specific -----
    print("\n=== T4: Direction-specific (L_side, B_side, offset) ===", flush=True)
    t4 = t4_direction_specific(bet_records, OFFSETS)
    print(f"  {'L_side':>6s} {'B_side':>6s} {'offset':>7s}  {'n':>5s}  {'mean PnL':>10s}  {'95% CI':>22s}", flush=True)
    for o in OFFSETS:
        for L_side in ("bull", "bear"):
            for B_side in ("bull", "bear"):
                key = (L_side, B_side, o)
                vals = t4.get(key, [])
                if not vals:
                    continue
                m = float(np.mean(vals))
                lo, hi = bootstrap_ci_mean(vals)
                print(f"  {L_side:>6s} {B_side:>6s} {o:>+7d}  {len(vals):>5d}  {m:>+10.5f}  [{lo:>+.5f}, {hi:>+.5f}]", flush=True)

    # ----- Test 5: pool-state delta at L, L+1, L+2 -----
    print("\n=== T5: Pool ratio at loss L, L+1, L+2 ===", flush=True)
    p0, p1, p2 = t5_pool_delta(bet_records, pool_ratios)
    if not p0:
        print("  no valid loss triples", flush=True)
    else:
        print(f"  n={len(p0)} losses with complete L, L+1, L+2 pool data", flush=True)
        for name, arr in [("L pool ratio", p0), ("L+1 pool ratio", p1), ("L+2 pool ratio", p2)]:
            a = np.asarray(arr)
            print(f"  {name:>16s}: mean={a.mean():+.4f} stdev={a.std(ddof=1):.4f} "
                  f"p10={np.quantile(a, 0.10):+.4f} p50={np.quantile(a, 0.50):+.4f} "
                  f"p90={np.quantile(a, 0.90):+.4f}", flush=True)
        # Paired deltas
        delta01 = np.asarray(p1) - np.asarray(p0)
        delta12 = np.asarray(p2) - np.asarray(p1)
        delta02 = np.asarray(p2) - np.asarray(p0)
        for name, arr in [("L+1 - L", delta01), ("L+2 - L+1", delta12), ("L+2 - L", delta02)]:
            lo, hi = bootstrap_ci_mean(list(arr))
            print(f"  {name:>16s}: mean delta={arr.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]", flush=True)

    # ----- Save -----
    out_path = REPO / "var" / "strategy_review" / "step25_loss_offset_diagnostics_data.json"

    def serialize_buckets(d):
        return {str(k): {"n": len(v), "mean": float(np.mean(v)) if v else None,
                          "ci_lo": bootstrap_ci_mean(v)[0] if v else None,
                          "ci_hi": bootstrap_ci_mean(v)[1] if v else None,
                          "values_sample": v[:50] if len(v) > 50 else v}
                for k, v in d.items()}

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "initial_bankroll": INITIAL_BANKROLL,
                "abs_dd_frac": ABS_DD_FRAC, "cooldown_rounds": COOLDOWN_ROUNDS,
                "settlement_visibility_delay": SETTLEMENT_VISIBILITY_DELAY,
                "offsets": list(OFFSETS), "loss_lookback_k": LOSS_LOOKBACK_K,
                "bootstrap_seeds": BOOTSTRAP_SEEDS,
            },
            "baseline": {"total_pnl": total_pnl, "n_bets": len(bet_records),
                          "n_losses": n_losses, "n_wins": n_wins,
                          "baseline_edge_per_bet": baseline_edge},
            "T1_per_offset_post_loss": serialize_buckets(t1),
            "T2_per_offset_post_win": serialize_buckets(t2),
            "T3_cumulative_loss": serialize_buckets(t3),
            "T4_direction_specific": {str(k): {"n": len(v), "mean": float(np.mean(v))}
                                        for k, v in t4.items()},
            "T5_pool_delta": {
                "n": len(p0),
                "L_mean": float(np.mean(p0)) if p0 else None,
                "L1_mean": float(np.mean(p1)) if p1 else None,
                "L2_mean": float(np.mean(p2)) if p2 else None,
                "delta_L1_L_mean": float(np.mean(np.asarray(p1) - np.asarray(p0))) if p0 else None,
                "delta_L2_L1_mean": float(np.mean(np.asarray(p2) - np.asarray(p1))) if p0 else None,
                "delta_L2_L_mean": float(np.mean(np.asarray(p2) - np.asarray(p0))) if p0 else None,
            },
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
