"""Step 8 — static-bankroll backtest at 5 BNB and 50 BNB scales.

Goal: measure the strategy's "pure edge" per cohort with no risk-gate interference.

Methodology:
  - Run canonical (3, 7, 15) cs=2 over full 422298..484000 range.
  - At each decision, the BankrollTracker reports a STATIC bankroll
    regardless of running PnL. This:
      * Freezes the bankroll cap in `_compute_bet_size` to STATIC × 0.05.
      * Makes drawdown_from_peak = 0% always (peak == current == STATIC).
      * Makes min_bankroll_bnb_to_bet check always pass (STATIC > 0.20).
      * Makes cooldown never fire (because drawdown breaker doesn't fire).
  - PnL accumulates separately as a running side-record; never feeds back
    into bet sizing or risk gates.

Two variants:
  A: STATIC = 5.0 BNB
  B: STATIC = 50.0 BNB

Sanity check: with no risk gates, static-5 PnL should be ~10% of static-50
PnL per cohort (linear scaling via the 0.05 bankroll-cap fraction, since
both scales bind the cap below the absolute max_bet_bnb_btc_primary=2.0).
A nonlinearity would expose a sizing artifact worth investigating.

Output:
  - var/strategy_review/static_bankroll_step8_data.json
  - var/strategy_review/2026_05_26_static_bankroll_step8.md (written separately)
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPOCH_MIN = 422298
EPOCH_MAX = 484000

CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2
POOL_CUTOFF = 6

TREASURY_FEE = 0.03
MIN_BET = 0.001

COHORT_ORDER = ("extension", "cv5", "holdout", "ext_v2", "fresh_oos", "post_fresh")


def cohort_of(epoch: int) -> str:
    if 422298 <= epoch <= 437561: return "extension"
    if 437562 <= epoch <= 474086: return "cv5"
    if 474880 <= epoch <= 475311: return "holdout"
    if 475312 <= epoch <= 479952: return "ext_v2"
    if 479953 <= epoch <= 483191: return "fresh_oos"
    return "post_fresh"


# ---------------------------------------------------------------------------
# StaticBankrollTracker — minimal protocol implementation
# ---------------------------------------------------------------------------

class StaticBankrollTracker:
    """Freezes the reported bankroll at a static value. All update calls
    are no-ops. Drawdown is always 0%; cooldown never fires.

    Implements the subset of the BankrollTracker protocol that
    `MomentumOnlyPipeline.decide_open_round` calls.
    """

    def __init__(self, static_bankroll: float) -> None:
        self._static = float(static_bankroll)

    def current_bankroll(self) -> float:
        return self._static

    def peak_bankroll(self, start_at: int) -> float:  # noqa: ARG002
        return self._static

    def is_paused(self, start_at: int) -> bool:  # noqa: ARG002
        return False

    def tick_cooldown(self) -> None:
        pass

    def cooldown_remaining(self) -> int:
        return 0

    def set_paused(self, rounds: int, start_at: int) -> None:  # noqa: ARG002
        pass

    def record_settlement(self, bankroll: float, start_at: int) -> None:  # noqa: ARG002
        pass


# ---------------------------------------------------------------------------
# Custom runner — applies sizing decisions but accumulates PnL side-record
# ---------------------------------------------------------------------------

def run_static_backtest(*, static_bankroll: float, label: str,
                         all_rounds: list, btc: dict, eth: dict, sol: dict,
                         earliest_offset: int) -> dict[str, Any]:
    sc = load_strategy_config_from_dict(
        {"gate": {"mtf_lookbacks": list(CANONICAL_LOOKBACKS)}}
    )
    max_lookback = max(CANONICAL_LOOKBACKS)

    # Slice klines per-entry (canonical max_lookback=15)
    btc_klines = {
        ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                  max_lookback=max_lookback,
                                  earliest_offset=earliest_offset)
        for ep, kl in btc.items()
    }
    eth_klines = {
        ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                  max_lookback=max_lookback,
                                  earliest_offset=earliest_offset)
        for ep, kl in eth.items()
    }
    sol_klines = {
        ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                  max_lookback=max_lookback,
                                  earliest_offset=earliest_offset)
        for ep, kl in sol.items()
    }

    gate_cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold,
    )
    tracker = StaticBankrollTracker(static_bankroll=static_bankroll)
    pipeline = MomentumOnlyPipeline(
        config=gate_cfg,
        strategy_config=sc,
        gate=None,
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        pool_cutoff_seconds=POOL_CUTOFF,
        min_bet_amount_bnb=MIN_BET,
        treasury_fee_fraction=TREASURY_FEE,
        bankroll_tracker=tracker,
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    sim_rounds = [r for r in all_rounds if EPOCH_MIN <= r.epoch <= EPOCH_MAX]
    print(f"\n--- static-bankroll {static_bankroll} BNB ({label}): "
          f"{len(sim_rounds)} rounds ---")
    t0 = time.time()

    per_cohort: dict[str, dict[str, Any]] = {
        c: {
            "n_rounds": 0, "n_bets": 0, "n_wins": 0,
            "pnl_bnb": 0.0, "total_bet_size_bnb": 0.0,
            "per_bet_profits": [],
        } for c in COHORT_ORDER
    }
    skip_counts: dict[str, int] = {}
    running_pnl = 0.0

    for round_t in sim_rounds:
        coh = cohort_of(int(round_t.epoch))
        per_cohort[coh]["n_rounds"] += 1

        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            sr = decision.skip_reason or "unknown"
            skip_counts[sr] = skip_counts.get(sr, 0) + 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        bet_size = float(decision.bet_size_bnb)
        side = str(decision.bet_side)

        outcome = settle_bet_against_closed_round(
            bet_bnb=bet_size,
            bet_side=side,
            round_closed=round_t,
            treasury_fee_fraction=TREASURY_FEE,
        )
        profit = outcome.credit_bnb - bet_size - MAX_GAS_COST_BET_BNB
        running_pnl += profit

        per_cohort[coh]["n_bets"] += 1
        per_cohort[coh]["pnl_bnb"] += profit
        per_cohort[coh]["total_bet_size_bnb"] += bet_size
        if outcome.outcome == "win":
            per_cohort[coh]["n_wins"] += 1
        per_cohort[coh]["per_bet_profits"].append(profit)

        # Bankroll is STATIC — do NOT call tracker.record_settlement(real_bankroll).
        pipeline.settle_closed_rounds(rounds=[round_t])

    elapsed = time.time() - t0
    total_bets = sum(c["n_bets"] for c in per_cohort.values())
    total_wins = sum(c["n_wins"] for c in per_cohort.values())
    print(f"  bets={total_bets} wins={total_wins} "
          f"WR={total_wins/total_bets if total_bets else 0:.4f} "
          f"pnl={running_pnl:+.4f} BNB  ({elapsed:.1f}s)")

    # Per-cohort: compute mean profit-per-bet + WR + mean bet size
    for c in COHORT_ORDER:
        cd = per_cohort[c]
        if cd["n_bets"] > 0:
            cd["win_rate"] = cd["n_wins"] / cd["n_bets"]
            cd["mean_profit_per_bet"] = statistics.mean(cd["per_bet_profits"])
            cd["mean_bet_size_bnb"] = cd["total_bet_size_bnb"] / cd["n_bets"]
            cd["stdev_profit_per_bet"] = (statistics.stdev(cd["per_bet_profits"])
                                          if cd["n_bets"] > 1 else 0.0)
        else:
            cd["win_rate"] = 0.0
            cd["mean_profit_per_bet"] = 0.0
            cd["mean_bet_size_bnb"] = 0.0
            cd["stdev_profit_per_bet"] = 0.0
        # Strip per-bet list before persist (large)
        del cd["per_bet_profits"]

    return {
        "static_bankroll": static_bankroll,
        "label": label,
        "total_bets": total_bets,
        "total_wins": total_wins,
        "total_pnl_bnb": running_pnl,
        "total_win_rate": total_wins / total_bets if total_bets else 0.0,
        "per_cohort": per_cohort,
        "skip_counts_top10": dict(sorted(skip_counts.items(), key=lambda x: -x[1])[:10]),
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    t_all = time.time()

    print("--- loading rounds (canonical + extended) ---")
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  loaded {len(all_rounds)} rounds; range "
          f"[{all_rounds[0].epoch}..{all_rounds[-1].epoch}]")

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1
    print(f"  earliest_offset={earliest_offset} latest_offset={latest_offset}")

    print("--- loading klines unified ---")
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  BTC={len(btc)} ETH={len(eth)} SOL={len(sol)}")

    run_5 = run_static_backtest(
        static_bankroll=5.0, label="static_5bnb",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset,
    )
    run_50 = run_static_backtest(
        static_bankroll=50.0, label="static_50bnb",
        all_rounds=all_rounds, btc=btc, eth=eth, sol=sol,
        earliest_offset=earliest_offset,
    )

    # --- Per-cohort comparison ---
    print(f"\n=== Per-cohort: STATIC bankroll ===")
    print(f"{'cohort':>12s} {'rounds':>7s}  "
          f"{'S5:bets':>8s} {'WR':>7s} {'PnL':>10s} {'mean_bet':>9s} {'pnl/bet':>10s}  "
          f"{'S50:bets':>9s} {'WR':>7s} {'PnL':>10s} {'mean_bet':>9s} {'pnl/bet':>10s}  "
          f"{'ratio':>7s}")
    for coh in COHORT_ORDER:
        c5 = run_5["per_cohort"][coh]
        c50 = run_50["per_cohort"][coh]
        ratio = c50["pnl_bnb"] / c5["pnl_bnb"] if abs(c5["pnl_bnb"]) > 1e-9 else float("nan")
        print(f"{coh:>12s} {c5['n_rounds']:>7d}  "
              f"{c5['n_bets']:>8d} {c5['win_rate']:>7.4f} {c5['pnl_bnb']:>+10.4f} "
              f"{c5['mean_bet_size_bnb']:>9.4f} {c5['mean_profit_per_bet']:>+10.5f}  "
              f"{c50['n_bets']:>9d} {c50['win_rate']:>7.4f} {c50['pnl_bnb']:>+10.4f} "
              f"{c50['mean_bet_size_bnb']:>9.4f} {c50['mean_profit_per_bet']:>+10.5f}  "
              f"{ratio:>7.4f}")

    # --- Compare to Step 7 dynamic results ---
    step7_path = REPO / "var" / "strategy_review" / "bankroll_scale_rerun_step7_data.json"
    step7_data: dict[str, Any] | None = None
    if step7_path.exists():
        step7_data = json.loads(step7_path.read_text(encoding="utf-8"))
        print(f"\n=== Dynamic vs Static comparison (Step 7 vs Step 8) ===")
        print(f"{'cohort':>12s}  {'dyn5':>10s} {'stat5':>10s} {'d-s':>8s}  "
              f"{'dyn50':>10s} {'stat50':>10s} {'d-s':>8s}")
        d5 = step7_data["run_5bnb"]["per_cohort"]
        d50 = step7_data["run_50bnb"]["per_cohort"]
        s5 = run_5["per_cohort"]
        s50 = run_50["per_cohort"]
        for coh in COHORT_ORDER:
            print(f"{coh:>12s}  "
                  f"{d5[coh]['pnl_bnb']:>+10.4f} {s5[coh]['pnl_bnb']:>+10.4f} "
                  f"{d5[coh]['pnl_bnb'] - s5[coh]['pnl_bnb']:>+8.4f}  "
                  f"{d50[coh]['pnl_bnb']:>+10.4f} {s50[coh]['pnl_bnb']:>+10.4f} "
                  f"{d50[coh]['pnl_bnb'] - s50[coh]['pnl_bnb']:>+8.4f}")
        d5_total = step7_data["run_5bnb"]["summary"]["net_pnl_bnb"]
        d50_total = step7_data["run_50bnb"]["summary"]["net_pnl_bnb"]
        s5_total = run_5["total_pnl_bnb"]
        s50_total = run_50["total_pnl_bnb"]
        print(f"{'TOTAL':>12s}  "
              f"{d5_total:>+10.4f} {s5_total:>+10.4f} {d5_total - s5_total:>+8.4f}  "
              f"{d50_total:>+10.4f} {s50_total:>+10.4f} {d50_total - s50_total:>+8.4f}")

    # --- Persist ---
    out_path = REPO / "var" / "strategy_review" / "static_bankroll_step8_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "treasury_fee_fraction": TREASURY_FEE,
                "min_bet_bnb": MIN_BET,
            },
            "run_static_5bnb": run_5,
            "run_static_50bnb": run_50,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
