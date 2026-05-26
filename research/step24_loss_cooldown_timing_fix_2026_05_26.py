"""Step 24 — loss-conditional cooldown (Part B) + timing-fix sizing (Part A).

Part A finding: backtest fires cooldown 1 round earlier than production.
  - Backtest: loss in round N → tracker reflects N at decision N+1 → fires at N+1
  - Production: wallet shows N's claim only at decision N+2 → fires at N+2
  Fix: delay settlement visibility by 2 rounds (so round N's outcome enters
  tracker at decision N+2). Implemented via pending-settlements queue.

Part B: loss-conditional cooldown — after each LOSS bet, skip next K rounds
(with production-realistic delay: K-round skip starts at N+2 after a loss
in N). Independent of drawdown-based fires.

Variants:
  1. drawdown_only (Step 15 cd=3 with timing fix, for sizing the Part A effect)
  2. loss_cd_1     — drawdown_cd_72 + loss_cd_1
  3. loss_cd_2     — drawdown_cd_72 + loss_cd_2
  4. loss_cd_3     — drawdown_cd_72 + loss_cd_3
  5. loss_cd_5     — drawdown_cd_72 + loss_cd_5
  6. loss_cd_8     — drawdown_cd_72 + loss_cd_8
  7. loss_cd_3_only         — drawdown disabled, loss_cd=3
  8. loss_cd_3 + dd_cd_3    — both at 3 rounds
  9. baseline cd=72 (timing-fixed for reference)

Permutation null on best variant. Reference: prod cd=72 = +44.87 (timing-bugged).
"""
from __future__ import annotations

import json
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

from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.constants import MAX_GAS_COST_BET_BNB  # noqa: E402
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
PERMUTATION_SEEDS = 1000
SETTLEMENT_VISIBILITY_DELAY = 2  # rounds; matches production claim_scan timing

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
# Tracker with drawdown + optional loss-conditional cooldown
# ============================================================

class Step24Tracker(InMemoryBankrollTracker):
    """Drawdown breaker (cd=drawdown_cd) + optional loss-conditional cooldown.

    Both cooldowns share the single _cooldown counter via set_paused.
    Loss-conditional cooldown is set externally (when a bet outcome is known)
    via mark_loss_cooldown.
    """

    def __init__(self, *, initial_bankroll, drawdown_peak_window_days, peak_mode,
                  drawdown_cd_rounds, abs_dd_frac, drawdown_enabled=True):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._drawdown_cd = int(drawdown_cd_rounds)
        self._abs_dd_frac = float(abs_dd_frac)
        self._drawdown_enabled = bool(drawdown_enabled)
        self.n_drawdown_fires = 0
        self.n_loss_cd_fires = 0

    def is_paused(self, as_of_start_at):
        if self._cooldown > 0:
            return True
        if not self._drawdown_enabled:
            return False
        current = self.current_bankroll()
        peak = self.peak_bankroll(as_of_start_at)
        if peak > 0:
            dd = (peak - current) / peak
            if dd >= self._abs_dd_frac:
                if self._drawdown_cd > 0:
                    self.set_paused(self._drawdown_cd + 1, as_of_start_at)
                self.n_drawdown_fires += 1
                return self._drawdown_cd > 0
        return False

    def mark_loss_cooldown(self, *, k_rounds: int, as_of_start_at: int):
        """Called externally when a settled bet was a LOSS, to start a
        k_rounds cooldown beginning at the NEXT round the bot observes."""
        if k_rounds <= 0:
            return
        # +1 to compensate for pipeline's tick_cooldown decrement on observation
        existing = self._cooldown
        new_cd = max(existing, k_rounds + 1)
        self.set_paused(new_cd, as_of_start_at)
        self.n_loss_cd_fires += 1


# ============================================================
# Backtest runner with timing-correct settlement delay
# ============================================================

def run_backtest(*, all_rounds, btc_klines, eth_klines, sol_klines,
                  drawdown_cd_rounds: int, loss_cd_rounds: int,
                  drawdown_enabled: bool = True,
                  initial_bankroll: float = INITIAL_BANKROLL,
                  label: str = "baseline"):
    """Backtest with:
      - Production-correct settlement visibility delay (2 rounds)
      - Optional drawdown breaker (cd = drawdown_cd_rounds; disabled if
        drawdown_enabled=False)
      - Optional loss-conditional cooldown (loss_cd_rounds > 0)
    """
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
    tracker = Step24Tracker(
        initial_bankroll=initial_bankroll,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        drawdown_cd_rounds=drawdown_cd_rounds,
        abs_dd_frac=ABS_DD_FRAC,
        drawdown_enabled=drawdown_enabled,
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

    # Per-cohort tracking
    per_cohort = {c: {"n_rounds": 0, "n_bets": 0, "n_wins": 0, "pnl_bnb": 0.0,
                       "n_drawdown_skip": 0, "n_loss_cd_skip": 0} for c in COHORT_ORDER}
    bankroll = float(initial_bankroll); peak = bankroll; max_dd_frac = 0.0
    bet_records: list[dict[str, Any]] = []

    # Pending settlements queue: each entry is dict with start_at, bankroll,
    # delivery_round_epoch (the round at whose decision this settlement
    # becomes visible).
    pending_settlements: deque[dict[str, Any]] = deque()
    # Pending loss cooldowns: queue of (delivery_round_epoch, start_at) that
    # should fire mark_loss_cooldown when the corresponding round is being
    # decided.
    pending_loss_cooldowns: deque[tuple[int, int]] = deque()

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1

        # Apply settlement visibility (matches production claim_scan timing)
        while pending_settlements and pending_settlements[0]["delivery_round_epoch"] <= ep:
            delivered = pending_settlements.popleft()
            pipeline.record_settlement(
                bankroll=delivered["bankroll"],
                start_at=delivered["start_at"],
            )

        # Apply pending loss cooldowns (also delayed by 2 rounds)
        while pending_loss_cooldowns and pending_loss_cooldowns[0][0] <= ep:
            _, fire_start_at = pending_loss_cooldowns.popleft()
            if loss_cd_rounds > 0:
                tracker.mark_loss_cooldown(
                    k_rounds=loss_cd_rounds,
                    as_of_start_at=int(round_t.start_at),
                )

        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            sr = decision.skip_reason or ""
            if sr == "risk_drawdown_breaker_fired":
                per_cohort[coh]["n_drawdown_skip"] += 1
            elif sr == "risk_cooldown_active":
                # Could be drawdown or loss-cooldown — bucket by tracker counters
                # (approximation: we can't distinguish per-skip cleanly, but the
                # counters give totals)
                per_cohort[coh]["n_loss_cd_skip"] += 1
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

        per_cohort[coh]["n_bets"] += 1
        per_cohort[coh]["pnl_bnb"] += profit
        if won:
            per_cohort[coh]["n_wins"] += 1
        bet_records.append({"epoch": ep, "cohort": coh, "profit": profit,
                             "won": won, "side": side})

        if bankroll > peak: peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac: max_dd_frac = dd

        # Queue settlement for SETTLEMENT_VISIBILITY_DELAY rounds later
        pending_settlements.append({
            "start_at": int(round_t.start_at),
            "bankroll": bankroll,
            "delivery_round_epoch": ep + SETTLEMENT_VISIBILITY_DELAY,
        })

        # If this bet was a LOSS and loss-conditional cooldown is enabled,
        # queue a loss-CD trigger to fire at the next visible decision
        # (also delayed by 2 rounds).
        if not won and loss_cd_rounds > 0:
            pending_loss_cooldowns.append(
                (ep + SETTLEMENT_VISIBILITY_DELAY, int(round_t.start_at))
            )

        pipeline.settle_closed_rounds(rounds=[round_t])

    for cd in per_cohort.values():
        cd["win_rate"] = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0.0

    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())
    return {
        "label": label,
        "drawdown_cd_rounds": drawdown_cd_rounds,
        "loss_cd_rounds": loss_cd_rounds,
        "drawdown_enabled": drawdown_enabled,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - initial_bankroll,
            "final_bankroll": bankroll,
        },
        "max_drawdown_frac": max_dd_frac,
        "n_drawdown_fires": tracker.n_drawdown_fires,
        "n_loss_cd_fires": tracker.n_loss_cd_fires,
        "per_cohort": per_cohort,
        "bet_records": bet_records,
    }


# ============================================================
# Permutation null (same as Step 15/16/19/22a)
# ============================================================

def permutation_null(*, bets_candidate, bets_baseline, n_seeds, base_seed=42):
    cohort_bets = {}
    for b in bets_candidate:
        c = b["cohort"]
        cohort_bets.setdefault(c, {"cand": [], "base": []})["cand"].append(b["profit"])
    for b in bets_baseline:
        c = b["cohort"]
        cohort_bets.setdefault(c, {"cand": [], "base": []})["base"].append(b["profit"])

    obs_D = sum(b["profit"] for b in bets_candidate) - sum(b["profit"] for b in bets_baseline)
    rng = random.Random(base_seed)
    perm_Ds = []
    for _ in range(n_seeds):
        perm_D = 0.0
        for coh, d in cohort_bets.items():
            pool = d["cand"] + d["base"]
            n_cand = len(d["cand"]); n_base = len(d["base"])
            if not pool: continue
            rng.shuffle(pool)
            perm_D += sum(pool[:n_cand]) - sum(pool[n_cand:n_cand + n_base])
        perm_Ds.append(perm_D)
    perm_Ds_sorted = sorted(perm_Ds)
    n_geq = sum(1 for d in perm_Ds if d >= obs_D)
    return {
        "observed_D": obs_D, "n_seeds": n_seeds, "p_value": n_geq / n_seeds,
        "perm_D_mean": statistics.mean(perm_Ds),
        "perm_D_stdev": statistics.stdev(perm_Ds) if len(perm_Ds) > 1 else 0.0,
        "perm_D_p05": perm_Ds_sorted[int(0.05 * n_seeds)],
        "perm_D_p50": perm_Ds_sorted[int(0.50 * n_seeds)],
        "perm_D_p95": perm_Ds_sorted[int(0.95 * n_seeds)],
        "perm_D_p99": perm_Ds_sorted[int(0.99 * n_seeds)],
    }


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
    t_kl = time.time()
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
    print(f"  klines loaded ({time.time()-t_kl:.1f}s)", flush=True)
    btc_klines = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                  for ep, kl in btc.items()}
    eth_klines = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                  for ep, kl in eth.items()}
    sol_klines = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                  for ep, kl in sol.items()}

    common = dict(
        all_rounds=all_rounds, btc_klines=btc_klines,
        eth_klines=eth_klines, sol_klines=sol_klines,
    )

    print("\n=== Part A sizing: cd=72 + cd=3 with timing-fix ===", flush=True)

    t = time.time()
    baseline_cd72_fixed = run_backtest(
        drawdown_cd_rounds=72, loss_cd_rounds=0, drawdown_enabled=True,
        label="cd72_timing_fixed", **common,
    )
    s = baseline_cd72_fixed["summary"]
    print(f"  cd=72 (timing fixed): pnl={s['net_pnl_bnb']:+.4f} bets={s['num_bets']} "
          f"fires={baseline_cd72_fixed['n_drawdown_fires']} ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    cd3_fixed = run_backtest(
        drawdown_cd_rounds=3, loss_cd_rounds=0, drawdown_enabled=True,
        label="cd3_timing_fixed", **common,
    )
    s = cd3_fixed["summary"]
    delta = s["net_pnl_bnb"] - baseline_cd72_fixed["summary"]["net_pnl_bnb"]
    print(f"  cd=3 (timing fixed): pnl={s['net_pnl_bnb']:+.4f} delta vs cd72 fixed={delta:+.4f} "
          f"bets={s['num_bets']} fires={cd3_fixed['n_drawdown_fires']} ({time.time()-t:.1f}s)", flush=True)
    print(f"  (Step 15 cd=3 with bug: +47.03; cd=72 with bug: +44.87; bugged delta +2.16)", flush=True)

    print("\n=== Part B: loss-conditional cooldown variants ===", flush=True)
    print(f"  reference: cd72_timing_fixed pnl={baseline_cd72_fixed['summary']['net_pnl_bnb']:+.4f}", flush=True)

    variants_specs = [
        ("loss_cd_1",   dict(drawdown_cd_rounds=72, loss_cd_rounds=1, drawdown_enabled=True)),
        ("loss_cd_2",   dict(drawdown_cd_rounds=72, loss_cd_rounds=2, drawdown_enabled=True)),
        ("loss_cd_3",   dict(drawdown_cd_rounds=72, loss_cd_rounds=3, drawdown_enabled=True)),
        ("loss_cd_5",   dict(drawdown_cd_rounds=72, loss_cd_rounds=5, drawdown_enabled=True)),
        ("loss_cd_8",   dict(drawdown_cd_rounds=72, loss_cd_rounds=8, drawdown_enabled=True)),
        ("loss_cd_3_only",        dict(drawdown_cd_rounds=0,  loss_cd_rounds=3, drawdown_enabled=False)),
        ("loss_cd_3_plus_dd_cd3", dict(drawdown_cd_rounds=3,  loss_cd_rounds=3, drawdown_enabled=True)),
    ]

    results: dict[str, Any] = {"cd72_timing_fixed": baseline_cd72_fixed,
                                "cd3_timing_fixed": cd3_fixed}
    for name, kw in variants_specs:
        t = time.time()
        r = run_backtest(label=name, **kw, **common)
        s = r["summary"]
        delta = s["net_pnl_bnb"] - baseline_cd72_fixed["summary"]["net_pnl_bnb"]
        print(f"  {name:>26s}: pnl={s['net_pnl_bnb']:+.4f} delta={delta:+.4f} "
              f"bets={s['num_bets']} dd_fires={r['n_drawdown_fires']} "
              f"loss_fires={r['n_loss_cd_fires']} max_dd={r['max_drawdown_frac']*100:.2f}% "
              f"({time.time()-t:.1f}s)", flush=True)
        results[name] = r

    # Best variant by PnL
    all_variants = {k: v for k, v in results.items() if k.startswith("loss_cd")}
    best_name = max(all_variants.keys(),
                     key=lambda k: all_variants[k]["summary"]["net_pnl_bnb"])
    best = all_variants[best_name]
    print(f"\n  best variant by PnL: {best_name} (pnl={best['summary']['net_pnl_bnb']:+.4f})", flush=True)

    # Per-cohort breakdown of best
    print(f"\n--- Per-cohort: {best_name} vs cd72_timing_fixed ---", flush=True)
    print(f"  {'cohort':>30s} {'base_bets':>10s} {'base_PnL':>10s} "
          f"{'best_bets':>10s} {'best_PnL':>10s} {'dPnL':>9s}", flush=True)
    for c in COHORT_ORDER:
        bc = best["per_cohort"][c]
        bs = baseline_cd72_fixed["per_cohort"][c]
        dp = bc["pnl_bnb"] - bs["pnl_bnb"]
        print(f"  {c:>30s} {bs['n_bets']:>10d} {bs['pnl_bnb']:>+10.4f} "
              f"{bc['n_bets']:>10d} {bc['pnl_bnb']:>+10.4f} {dp:>+9.4f}", flush=True)

    # Permutation null on best
    print(f"\n--- Permutation null on {best_name} ({PERMUTATION_SEEDS} seeds) ---", flush=True)
    t = time.time()
    null = permutation_null(
        bets_candidate=best["bet_records"],
        bets_baseline=baseline_cd72_fixed["bet_records"],
        n_seeds=PERMUTATION_SEEDS,
    )
    print(f"  Observed D: {null['observed_D']:+.4f}", flush=True)
    print(f"  Null mean: {null['perm_D_mean']:+.4f}  stdev: {null['perm_D_stdev']:.4f}", flush=True)
    print(f"  Null p05/p50/p95/p99: {null['perm_D_p05']:+.4f} / {null['perm_D_p50']:+.4f} / "
          f"{null['perm_D_p95']:+.4f} / {null['perm_D_p99']:+.4f}", flush=True)
    print(f"  p-value: {null['p_value']:.4f}", flush=True)
    print(f"  ({time.time()-t:.1f}s)", flush=True)

    # Save
    def strip(r):
        return {k: v for k, v in r.items() if k != "bet_records"}
    out_path = REPO / "var" / "strategy_review" / "step24_loss_cooldown_timing_fix_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "initial_bankroll": INITIAL_BANKROLL,
                "abs_dd_frac": ABS_DD_FRAC,
                "settlement_visibility_delay": SETTLEMENT_VISIBILITY_DELAY,
                "permutation_seeds": PERMUTATION_SEEDS,
            },
            "results": {k: strip(v) for k, v in results.items()},
            "best_variant_name": best_name,
            "permutation_null_best": null,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
