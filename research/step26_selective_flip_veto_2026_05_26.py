"""Step 26 — Selective flip-veto backtest.

Based on Step 25's finding: at observation-time offset +2 from a loss
(real-time round T-2 = the first round to know about the loss), bets in
the OPPOSITE direction lose ~-0.12 to -0.15 BNB/bet, while same-direction
bets are baseline (+0.03).

Selective flip-veto = skip the bet if the most-recent-OBSERVED bet was a
LOSS and the would-be direction is OPPOSITE to that losing bet.

All backtests use the timing-fixed pattern (Step 24): settlement
visibility delayed by 2 rounds (production-faithful).

Variants (5 BNB scale):
  1. baseline               cd=72, dd=0.15, no flip-veto
  2. flip_veto_pure         cd=72, dd=0.15, +flip-veto (standard)
  3. flip_veto_only         cd=72, dd=disabled, +flip-veto
  4. flip_veto_plus_cd3     cd=3, dd=0.15, +flip-veto
  5. flip_veto_plus_loss_cd1  cd=72, dd=0.15, +loss_cd_1, +flip-veto
  6. flip_veto_extended_p2_p3  cd=72, dd=0.15, flip-veto fires at real-time
                                 offsets 2 OR 3 from the most recent OBSERVED loss

Permutation null on best variant (1000 seeds).
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
SETTLEMENT_VISIBILITY_DELAY = 2

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


class Step26Tracker(InMemoryBankrollTracker):
    def __init__(self, *, initial_bankroll, drawdown_peak_window_days, peak_mode,
                  cooldown_rounds, abs_dd_frac, drawdown_enabled=True):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cd_total = int(cooldown_rounds)
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
                if self._cd_total > 0:
                    self.set_paused(self._cd_total + 1, as_of_start_at)
                self.n_drawdown_fires += 1
                return self._cd_total > 0
        return False

    def mark_loss_cooldown(self, *, k_rounds: int, as_of_start_at: int):
        if k_rounds <= 0:
            return
        existing = self._cooldown
        new_cd = max(existing, k_rounds + 1)
        self.set_paused(new_cd, as_of_start_at)
        self.n_loss_cd_fires += 1


def run_backtest(*,
                  all_rounds, btc_klines, eth_klines, sol_klines,
                  drawdown_cd_rounds: int = 72,
                  loss_cd_rounds: int = 0,
                  drawdown_enabled: bool = True,
                  flip_veto_mode: str = "none",  # 'none' | 'standard' | 'extended'
                  initial_bankroll: float = INITIAL_BANKROLL,
                  label: str = "baseline"):
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
    tracker = Step26Tracker(
        initial_bankroll=initial_bankroll,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        cooldown_rounds=drawdown_cd_rounds,
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

    per_cohort = {c: {"n_rounds": 0, "n_bets": 0, "n_wins": 0, "pnl_bnb": 0.0,
                       "n_other_skip": 0, "n_flip_vetoed": 0} for c in COHORT_ORDER}
    bankroll = float(initial_bankroll); peak = bankroll; max_dd_frac = 0.0
    bet_records: list[dict[str, Any]] = []

    pending_settlements: deque[dict[str, Any]] = deque()
    pending_loss_cooldowns: deque[tuple[int, int]] = deque()

    # Per-bet history for flip-veto lookup. Each entry: (epoch, side, won)
    settled_history: list[dict[str, Any]] = []

    # Flip-veto diagnostic counters
    flip_veto_total = 0
    flip_by_offset: dict[int, int] = {}  # real-time offset -> count
    flip_by_key: dict[tuple, int] = {}   # (loss_side, would_be_side, offset) -> count

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        coh = cohort_of(ep)
        per_cohort[coh]["n_rounds"] += 1

        # Deliver pending settlements (Step 24 pattern)
        while pending_settlements and pending_settlements[0]["delivery_round_epoch"] <= ep:
            d = pending_settlements.popleft()
            pipeline.record_settlement(bankroll=d["bankroll"], start_at=d["start_at"])

        # Deliver pending loss cooldowns
        while pending_loss_cooldowns and pending_loss_cooldowns[0][0] <= ep:
            _, fire_start_at = pending_loss_cooldowns.popleft()
            if loss_cd_rounds > 0:
                tracker.mark_loss_cooldown(
                    k_rounds=loss_cd_rounds,
                    as_of_start_at=int(round_t.start_at),
                )

        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            per_cohort[coh]["n_other_skip"] += 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Flip-veto check: based on most-recent OBSERVED bet (with visibility delay)
        would_be_side = str(decision.bet_side)
        observed_threshold_epoch = ep - SETTLEMENT_VISIBILITY_DELAY
        skip_flip = False
        flip_diag = None

        if flip_veto_mode == "standard":
            # Most recent observed bet (any outcome)
            for b in reversed(settled_history):
                if b["epoch"] <= observed_threshold_epoch:
                    if not b["won"]:
                        # Most recent observed bet was a LOSS
                        if would_be_side != b["side"]:
                            skip_flip = True
                            flip_diag = (b["side"], would_be_side, ep - b["epoch"])
                    # else: WIN — no veto
                    break
        elif flip_veto_mode == "extended":
            # Most recent observed LOSS, then check offset is 2 or 3
            for b in reversed(settled_history):
                if b["epoch"] <= observed_threshold_epoch and not b["won"]:
                    offset = ep - b["epoch"]
                    if offset in (2, 3) and would_be_side != b["side"]:
                        skip_flip = True
                        flip_diag = (b["side"], would_be_side, offset)
                    break

        if skip_flip:
            flip_veto_total += 1
            per_cohort[coh]["n_flip_vetoed"] += 1
            if flip_diag is not None:
                flip_by_offset[flip_diag[2]] = flip_by_offset.get(flip_diag[2], 0) + 1
                flip_by_key[flip_diag] = flip_by_key.get(flip_diag, 0) + 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Place bet
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
        bet_records.append({"epoch": ep, "cohort": coh, "side": side,
                             "won": won, "profit": profit})
        settled_history.append({"epoch": ep, "side": side, "won": won})

        if bankroll > peak: peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac: max_dd_frac = dd

        pending_settlements.append({
            "start_at": int(round_t.start_at),
            "bankroll": bankroll,
            "delivery_round_epoch": ep + SETTLEMENT_VISIBILITY_DELAY,
        })
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
        "flip_veto_mode": flip_veto_mode,
        "summary": {
            "num_bets": total_bets, "num_wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets else 0.0,
            "net_pnl_bnb": bankroll - initial_bankroll,
            "final_bankroll": bankroll,
        },
        "max_drawdown_frac": max_dd_frac,
        "n_drawdown_fires": tracker.n_drawdown_fires,
        "n_loss_cd_fires": tracker.n_loss_cd_fires,
        "n_flip_vetoes": flip_veto_total,
        "flip_by_offset": flip_by_offset,
        "flip_by_key": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in flip_by_key.items()},
        "per_cohort": per_cohort,
        "bet_records": bet_records,
    }


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

    common = dict(all_rounds=all_rounds, btc_klines=btc_kl,
                   eth_klines=eth_kl, sol_klines=sol_kl)

    variants = [
        ("baseline",                    dict(drawdown_cd_rounds=72, loss_cd_rounds=0,
                                              drawdown_enabled=True, flip_veto_mode="none")),
        ("flip_veto_pure",              dict(drawdown_cd_rounds=72, loss_cd_rounds=0,
                                              drawdown_enabled=True, flip_veto_mode="standard")),
        ("flip_veto_only",              dict(drawdown_cd_rounds=72, loss_cd_rounds=0,
                                              drawdown_enabled=False, flip_veto_mode="standard")),
        ("flip_veto_plus_cd3",          dict(drawdown_cd_rounds=3, loss_cd_rounds=0,
                                              drawdown_enabled=True, flip_veto_mode="standard")),
        ("flip_veto_plus_loss_cd1",     dict(drawdown_cd_rounds=72, loss_cd_rounds=1,
                                              drawdown_enabled=True, flip_veto_mode="standard")),
        ("flip_veto_extended_p2_p3",    dict(drawdown_cd_rounds=72, loss_cd_rounds=0,
                                              drawdown_enabled=True, flip_veto_mode="extended")),
    ]

    print("\n=== 6 backtests at 5 BNB (all timing-fixed) ===", flush=True)
    results = {}
    for name, kw in variants:
        t = time.time()
        r = run_backtest(label=name, **kw, **common)
        s = r["summary"]
        delta = 0.0 if name == "baseline" else s["net_pnl_bnb"] - results["baseline"]["summary"]["net_pnl_bnb"]
        print(f"  {name:>30s}: pnl={s['net_pnl_bnb']:+.4f} delta={delta:+.4f} "
              f"bets={s['num_bets']} dd_fires={r['n_drawdown_fires']} "
              f"loss_fires={r['n_loss_cd_fires']} flip_vetoes={r['n_flip_vetoes']} "
              f"max_dd={r['max_drawdown_frac']*100:.2f}% ({time.time()-t:.1f}s)", flush=True)
        results[name] = r

    # Identify best variant by PnL
    variant_names = [n for n, _ in variants if n != "baseline"]
    best_name = max(variant_names, key=lambda n: results[n]["summary"]["net_pnl_bnb"])
    best = results[best_name]
    print(f"\n  best variant by PnL: {best_name} (pnl={best['summary']['net_pnl_bnb']:+.4f})", flush=True)

    # Per-cohort breakdown of best
    print(f"\n--- Per-cohort: {best_name} vs baseline ---", flush=True)
    print(f"  {'cohort':>30s} {'base_bets':>10s} {'base_PnL':>10s} "
          f"{'best_bets':>10s} {'best_PnL':>10s} {'flips':>6s} {'dPnL':>9s}", flush=True)
    for c in COHORT_ORDER:
        bc = best["per_cohort"][c]
        bs = results["baseline"]["per_cohort"][c]
        dp = bc["pnl_bnb"] - bs["pnl_bnb"]
        print(f"  {c:>30s} {bs['n_bets']:>10d} {bs['pnl_bnb']:>+10.4f} "
              f"{bc['n_bets']:>10d} {bc['pnl_bnb']:>+10.4f} {bc['n_flip_vetoed']:>6d} {dp:>+9.4f}", flush=True)

    # Diagnostic: flip-veto offset distribution + key breakdown
    print(f"\n--- Diagnostic: flip-veto offset distribution for {best_name} ---", flush=True)
    print(f"  flip-vetoes by real-time offset (T - L.epoch):", flush=True)
    for off in sorted(best["flip_by_offset"].keys()):
        n = best["flip_by_offset"][off]
        frac = n / best["n_flip_vetoes"] if best["n_flip_vetoes"] else 0
        print(f"    offset +{off}: {n}  ({frac*100:.1f}%)", flush=True)

    print(f"\n  Top 8 flip-veto (loss_side, would_be_side, offset) keys:", flush=True)
    sorted_keys = sorted(best["flip_by_key"].items(), key=lambda kv: -kv[1])[:8]
    for key, n in sorted_keys:
        print(f"    {key}: {n}", flush=True)

    # Permutation null on best
    print(f"\n--- Permutation null on {best_name} ({PERMUTATION_SEEDS} seeds) ---", flush=True)
    t = time.time()
    null = permutation_null(
        bets_candidate=best["bet_records"],
        bets_baseline=results["baseline"]["bet_records"],
        n_seeds=PERMUTATION_SEEDS,
    )
    print(f"  Observed D: {null['observed_D']:+.4f}", flush=True)
    print(f"  Null mean: {null['perm_D_mean']:+.4f}  stdev: {null['perm_D_stdev']:.4f}", flush=True)
    print(f"  Null p05/p50/p95/p99: {null['perm_D_p05']:+.4f} / {null['perm_D_p50']:+.4f} / "
          f"{null['perm_D_p95']:+.4f} / {null['perm_D_p99']:+.4f}", flush=True)
    print(f"  p-value: {null['p_value']:.4f}", flush=True)
    print(f"  ({time.time()-t:.1f}s)", flush=True)

    def strip(r):
        return {k: v for k, v in r.items() if k != "bet_records"}
    out_path = REPO / "var" / "strategy_review" / "step26_selective_flip_veto_data.json"
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
