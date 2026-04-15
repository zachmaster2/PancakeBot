"""Step 22: Filter + sizing sweep for higher bet rate and PnL.

Live dry mode showed 4/4 signal fires skipped by pool_below_minimum
(pools 1.27-1.66 BNB, threshold 2.0). This experiment sweeps pool min,
signal threshold, payout floor, and sizing parameters.

Phase 1: Filter sweep (36 configs x 5-fold)
Phase 2: Sizing sweep on best Phase 1 config (27 configs x 5-fold)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pancakebot.domain.strategy.momentum_gate as _gate_mod
from pancakebot.core.constants import (
    BNB_WEI, GAS_COST_BET_BNB, POOL_CUTOFF_SECONDS, TREASURY_FEE_FRACTION,
)
from pancakebot.domain.strategy.momentum_gate import compute_signal_from_klines
from pancakebot.domain.strategy.momentum_pipeline import _pools_from_bets
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2
N_FOLDS = 5
INITIAL_BANKROLL = 50.0
MIN_BET_AMOUNT = 0.001


# ---- Data structures ----

@dataclass
class PrecomputedRound:
    rnd: object  # Round
    epoch: int
    lock_at: int
    signal: str | None
    signal_strength: float
    eth_confirm: float
    sol_confirm: float
    pool_bull: float
    pool_bear: float
    pool_total: float


@dataclass
class SweepConfig:
    # Filters
    min_pool_bnb: float = 2.0
    thresh_mode: str = "adaptive"  # "uniform" or "adaptive"
    uniform_thresh: float = 0.0001
    small_thresh: float = 0.0002
    large_thresh: float = 0.0001
    thresh_boundary: float = 3.0
    min_payout: float = 1.5
    # Sizing
    base_frac: float = 0.03
    sizing_slope: float = 100.0
    payout_slope: float = 1.0
    eth_weight: float = 0.3
    sol_weight: float = 0.3
    max_frac: float = 0.30
    floor_bnb: float = 0.01
    cap_bnb: float = 2.0


@dataclass
class FoldResult:
    n_bets: int
    n_wins: int
    pnl: float
    n_rounds: int

    @property
    def pnl_2k(self) -> float:
        return self.pnl / self.n_rounds * 2000 if self.n_rounds > 0 else 0.0

    @property
    def wr(self) -> float:
        return self.n_wins / self.n_bets * 100 if self.n_bets > 0 else 0.0


# ---- Data loading ----

def load_data():
    print("Loading data...", end=" ", flush=True)
    t0 = time.time()
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())

    def load_kl(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    bnb = load_kl("var/bnb_spot_prices.jsonl")
    btc = load_kl("var/btc_spot_prices.jsonl")
    eth = load_kl("var/eth_spot_prices.jsonl")
    sol = load_kl("var/sol_spot_prices.jsonl")
    print(f"{len(rounds)} rounds, {time.time()-t0:.1f}s")
    return rounds, bnb, btc, eth, sol


# ---- Signal pre-computation ----

def precompute_signals(rounds, bnb_kl, btc_kl, eth_kl, sol_kl) -> list[PrecomputedRound]:
    """Pre-compute signal and pool data for all rounds.

    Monkey-patches _MTF_THRESH to 0.00005 so signals with strength down
    to that level are captured. The sweep applies its own thresholds later.
    """
    # Temporarily lower the gate threshold to capture weak signals
    orig_thresh = _gate_mod._MTF_THRESH
    _gate_mod._MTF_THRESH = 0.00005
    try:
        return _precompute_inner(rounds, bnb_kl, btc_kl, eth_kl, sol_kl)
    finally:
        _gate_mod._MTF_THRESH = orig_thresh


def _precompute_inner(rounds, bnb_kl, btc_kl, eth_kl, sol_kl) -> list[PrecomputedRound]:
    print("Pre-computing signals...", end=" ", flush=True)
    t0 = time.time()
    result = []
    for rnd in rounds:
        epoch = int(rnd.epoch)
        lock_at = int(rnd.lock_at)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        bnb_raw = bnb_kl.get(epoch)
        btc_raw = btc_kl.get(epoch)
        eth_raw = eth_kl.get(epoch)
        sol_raw = sol_kl.get(epoch)

        if bnb_raw is None or btc_raw is None:
            result.append(PrecomputedRound(
                rnd=rnd, epoch=epoch, lock_at=lock_at,
                signal=None, signal_strength=0.0,
                eth_confirm=0.0, sol_confirm=0.0,
                pool_bull=0.0, pool_bear=0.0, pool_total=0.0,
            ))
            continue

        sig = compute_signal_from_klines(
            bnb_raw, btc_raw, cutoff_ms,
            eth_klines=eth_raw, sol_klines=sol_raw,
        )

        pool_bull, pool_bear = _pools_from_bets(rnd, lock_at - POOL_CUTOFF_SECONDS)

        result.append(PrecomputedRound(
            rnd=rnd, epoch=epoch, lock_at=lock_at,
            signal=sig.signal,
            signal_strength=sig.signal_strength,
            eth_confirm=sig.eth_confirmation_strength,
            sol_confirm=sig.sol_confirmation_strength,
            pool_bull=pool_bull, pool_bear=pool_bear,
            pool_total=pool_bull + pool_bear,
        ))

    n_signals = sum(1 for p in result if p.signal is not None)
    print(f"{len(result)} rounds, {n_signals} signals ({n_signals/len(result)*100:.1f}%), {time.time()-t0:.1f}s")
    return result


# ---- Simulation ----

def simulate_fold(fold: list[PrecomputedRound], cfg: SweepConfig) -> FoldResult:
    """Run one fold with the given config. Returns metrics."""
    bankroll = INITIAL_BANKROLL
    n_bets = 0
    n_wins = 0
    pnl = 0.0

    for pr in fold:
        if pr.signal is None:
            continue

        # Signal threshold
        if cfg.thresh_mode == "uniform":
            if pr.signal_strength < cfg.uniform_thresh:
                continue
        else:  # adaptive
            thresh = cfg.large_thresh if pr.pool_total >= cfg.thresh_boundary else cfg.small_thresh
            if pr.signal_strength < thresh:
                continue

        # Pool minimum
        if pr.pool_total < cfg.min_pool_bnb:
            continue

        # Payout floor
        our_side = pr.pool_bull if pr.signal == "Bull" else pr.pool_bear
        if our_side > 0 and pr.pool_total > 0:
            payout = pr.pool_total * (1.0 - TREASURY_FEE_FRACTION) / our_side
            if payout < cfg.min_payout:
                continue
        elif our_side <= 0:
            payout = 99.0  # empty side = huge payout
        else:
            continue

        # Sizing
        effective = pr.signal_strength
        if pr.eth_confirm > 0:
            effective += pr.eth_confirm * cfg.eth_weight
        if pr.sol_confirm > 0:
            effective += pr.sol_confirm * cfg.sol_weight

        frac = min(cfg.base_frac + cfg.sizing_slope * effective, cfg.max_frac)
        if our_side > 0:
            payout_mult = max(0.5, 1.0 + cfg.payout_slope * (payout - 2.0))
            frac = min(frac * payout_mult, cfg.max_frac)
        bet = max(cfg.floor_bnb, min(cfg.cap_bnb, pr.pool_total * frac))

        if bet < MIN_BET_AMOUNT:
            continue

        # Settle
        bankroll -= bet + GAS_COST_BET_BNB
        outcome = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=pr.signal,
            round_closed=pr.rnd, treasury_fee_fraction=TREASURY_FEE_FRACTION,
        )
        bankroll += outcome.credit_bnb
        profit = outcome.credit_bnb - bet - GAS_COST_BET_BNB
        pnl += profit
        n_bets += 1
        if profit > 0:
            n_wins += 1

    return FoldResult(n_bets=n_bets, n_wins=n_wins, pnl=pnl, n_rounds=len(fold))


def run_5fold(precomputed: list[PrecomputedRound], cfg: SweepConfig):
    """Run 5-fold validation. Returns (fold_results, summary)."""
    fold_size = len(precomputed) // N_FOLDS
    folds = [precomputed[i * fold_size:(i + 1) * fold_size] for i in range(N_FOLDS)]

    fold_results = [simulate_fold(f, cfg) for f in folds]

    total_bets = sum(fr.n_bets for fr in fold_results)
    total_wins = sum(fr.n_wins for fr in fold_results)
    total_pnl = sum(fr.pnl for fr in fold_results)
    total_rounds = sum(fr.n_rounds for fr in fold_results)

    pnl_2ks = [fr.pnl_2k for fr in fold_results]
    avg_pnl_2k = sum(pnl_2ks) / len(pnl_2ks)
    fold_std = (sum((p - avg_pnl_2k) ** 2 for p in pnl_2ks) / len(pnl_2ks)) ** 0.5
    n_positive = sum(1 for p in pnl_2ks if p > 0)

    return fold_results, {
        "total_bets": total_bets,
        "bets_2k": total_bets / total_rounds * 2000 if total_rounds > 0 else 0,
        "wr": total_wins / total_bets * 100 if total_bets > 0 else 0,
        "pnl_2k": avg_pnl_2k,
        "fold_std": fold_std,
        "n_positive": n_positive,
        "pnl_2ks": pnl_2ks,
    }


# ---- Phase 1: Filter sweep ----

def phase1_filter_sweep(precomputed):
    print("\n" + "=" * 90)
    print("PHASE 1: Filter Sweep (sizing held at production defaults)")
    print("=" * 90)

    min_pool_values = [1.0, 1.25, 1.5, 2.0]
    thresh_configs = [
        ("uniform", 0.00008),
        ("uniform", 0.0001),
        ("adaptive", None),
    ]
    min_payout_values = [1.2, 1.3, 1.5]

    results = []
    for mp in min_pool_values:
        for tm, tv in thresh_configs:
            for pay in min_payout_values:
                cfg = SweepConfig(
                    min_pool_bnb=mp,
                    thresh_mode=tm,
                    uniform_thresh=tv if tv else 0.0001,
                    min_payout=pay,
                )
                folds, summary = run_5fold(precomputed, cfg)
                label = f"pool>={mp:.2f} thresh={tm}{'='+str(tv) if tv else '(adaptive)'} pay>={pay}"
                results.append((label, cfg, folds, summary))

    # Sort by PnL/2k descending
    results.sort(key=lambda x: x[3]["pnl_2k"], reverse=True)

    # Print header
    print(f"\n{'Config':<52} {'bets/2k':>7} {'WR%':>6} {'PnL/2k':>8} {'std':>6} {'pos':>4}  {'f1':>7} {'f2':>7} {'f3':>7} {'f4':>7} {'f5':>7}")
    print("-" * 130)

    for label, cfg, folds, s in results:
        pnls = s["pnl_2ks"]
        marker = " ***" if s["n_positive"] >= 5 else " **" if s["n_positive"] >= 4 else ""
        print(f"{label:<52} {s['bets_2k']:7.1f} {s['wr']:5.1f}% {s['pnl_2k']:+7.2f} {s['fold_std']:6.2f} {s['n_positive']:>3}/5 "
              f"{pnls[0]:+7.2f} {pnls[1]:+7.2f} {pnls[2]:+7.2f} {pnls[3]:+7.2f} {pnls[4]:+7.2f}{marker}")

    # Find best
    best_5_5 = [r for r in results if r[3]["n_positive"] == 5]
    if best_5_5:
        best = best_5_5[0]
    else:
        best_4_5 = [r for r in results if r[3]["n_positive"] >= 4]
        best = best_4_5[0] if best_4_5 else results[0]

    print(f"\n>>> BEST PHASE 1: {best[0]} -> PnL/2k={best[3]['pnl_2k']:+.2f} ({best[3]['n_positive']}/5)")
    return best[1]  # return SweepConfig


# ---- Phase 2: Sizing sweep ----

def phase2_sizing_sweep(precomputed, base_cfg: SweepConfig):
    print("\n" + "=" * 90)
    print(f"PHASE 2: Sizing Sweep (filters from Phase 1: pool>={base_cfg.min_pool_bnb}, "
          f"thresh={base_cfg.thresh_mode}, pay>={base_cfg.min_payout})")
    print("=" * 90)

    slopes = [50, 75, 100]
    base_fracs = [0.03, 0.04, 0.05]
    caps = [1.0, 1.5, 2.0]

    results = []
    for slope in slopes:
        for bf in base_fracs:
            for cap in caps:
                cfg = SweepConfig(
                    min_pool_bnb=base_cfg.min_pool_bnb,
                    thresh_mode=base_cfg.thresh_mode,
                    uniform_thresh=base_cfg.uniform_thresh,
                    small_thresh=base_cfg.small_thresh,
                    large_thresh=base_cfg.large_thresh,
                    thresh_boundary=base_cfg.thresh_boundary,
                    min_payout=base_cfg.min_payout,
                    base_frac=bf,
                    sizing_slope=slope,
                    cap_bnb=cap,
                )
                folds, summary = run_5fold(precomputed, cfg)
                label = f"slope={slope} base={bf:.2f} cap={cap:.1f}"
                results.append((label, cfg, folds, summary))

    results.sort(key=lambda x: x[3]["pnl_2k"], reverse=True)

    print(f"\n{'Config':<28} {'bets/2k':>7} {'WR%':>6} {'PnL/2k':>8} {'std':>6} {'pos':>4}  {'f1':>7} {'f2':>7} {'f3':>7} {'f4':>7} {'f5':>7}")
    print("-" * 110)

    for label, cfg, folds, s in results:
        pnls = s["pnl_2ks"]
        marker = " ***" if s["n_positive"] >= 5 else " **" if s["n_positive"] >= 4 else ""
        print(f"{label:<28} {s['bets_2k']:7.1f} {s['wr']:5.1f}% {s['pnl_2k']:+7.2f} {s['fold_std']:6.2f} {s['n_positive']:>3}/5 "
              f"{pnls[0]:+7.2f} {pnls[1]:+7.2f} {pnls[2]:+7.2f} {pnls[3]:+7.2f} {pnls[4]:+7.2f}{marker}")

    best_5_5 = [r for r in results if r[3]["n_positive"] == 5]
    if best_5_5:
        best = best_5_5[0]
    else:
        best_4_5 = [r for r in results if r[3]["n_positive"] >= 4]
        best = best_4_5[0] if best_4_5 else results[0]

    print(f"\n>>> BEST PHASE 2: {best[0]} -> PnL/2k={best[3]['pnl_2k']:+.2f} ({best[3]['n_positive']}/5), fold_std={best[3]['fold_std']:.2f}")
    return best[1]


# ---- Main ----

def main():
    t_start = time.time()

    rounds, bnb_kl, btc_kl, eth_kl, sol_kl = load_data()
    precomputed = precompute_signals(rounds, bnb_kl, btc_kl, eth_kl, sol_kl)

    # Verify baseline: production config should give ~+2.73/2k
    print("\nVerifying baseline (production config)...")
    baseline_cfg = SweepConfig()  # all defaults = production
    _, baseline = run_5fold(precomputed, baseline_cfg)
    print(f"Baseline: bets/2k={baseline['bets_2k']:.1f}, WR={baseline['wr']:.1f}%, "
          f"PnL/2k={baseline['pnl_2k']:+.2f}, fold_std={baseline['fold_std']:.2f}, "
          f"pos={baseline['n_positive']}/5")
    print(f"Per-fold: {' '.join(f'{p:+.2f}' for p in baseline['pnl_2ks'])}")

    best_filter = phase1_filter_sweep(precomputed)
    best_overall = phase2_sizing_sweep(precomputed, best_filter)

    # Final comparison
    _, final = run_5fold(precomputed, best_overall)
    print("\n" + "=" * 90)
    print("FINAL COMPARISON")
    print("=" * 90)
    print(f"Baseline:  bets/2k={baseline['bets_2k']:5.1f}, WR={baseline['wr']:5.1f}%, "
          f"PnL/2k={baseline['pnl_2k']:+6.2f}, fold_std={baseline['fold_std']:.2f}")
    print(f"Best:      bets/2k={final['bets_2k']:5.1f}, WR={final['wr']:5.1f}%, "
          f"PnL/2k={final['pnl_2k']:+6.2f}, fold_std={final['fold_std']:.2f}")
    delta_pnl = final["pnl_2k"] - baseline["pnl_2k"]
    print(f"Delta:     bets/2k {final['bets_2k']-baseline['bets_2k']:+5.1f}, "
          f"PnL/2k {delta_pnl:+6.2f}")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
