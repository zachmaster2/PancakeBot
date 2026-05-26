"""Step 23 — intra-cooldown bet density analysis (no new sweep).

Key insight: gate decisions are bankroll-INDEPENDENT (canonical multi-TF
momentum gate uses only klines). So cd=0's BET stream is the complete
"what the gate would have BET" reference; cd=72's BETs are a subset
(those not in any cooldown window).

Two backtests:
  1. baseline_cd72 — actual production (cd=72, dd=0.15, 5 BNB). Log
     fire epochs + BET epochs.
  2. baseline_cd0  — same dd, cooldown disabled. Log BET epochs (= gate
     BET decisions when bankroll-pause not active).

Analyses:
  1. Inter-bet interval distribution (from cd=72 BET stream).
  2. Counterfactual blocked bets per cd in {3,5,8,12,24,48,72}:
     for each baseline fire epoch F, count cd=0 BETs in [F+1, F+cd].
  3. Post-fire bet density: cd=0 BETs in [F+1, F+8] across all fires.
  4. Cooldown saved-PnL accounting: pull from Step 15 JSON.
"""
from __future__ import annotations

import json
import sys
import time
import statistics
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
CD_VALUES_TO_TEST = (3, 5, 8, 12, 24, 48, 72)


class LoggingTracker(InMemoryBankrollTracker):
    """Logs fire epochs (start_at) when breaker triggers."""

    def __init__(self, *, initial_bankroll, drawdown_peak_window_days, peak_mode,
                  cooldown_rounds, abs_dd_frac):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cd_total = int(cooldown_rounds)
        self._abs_dd_frac = float(abs_dd_frac)
        self.fire_start_ats: list[int] = []  # start_at timestamps when breaker fired

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
                self.fire_start_ats.append(int(as_of_start_at))
                return self._cd_total > 0
        return False


def run_logging_backtest(*, all_rounds, btc_klines, eth_klines, sol_klines,
                          cooldown_rounds, abs_dd_frac):
    """Returns dict with bet_epochs, fire_epochs, total_pnl, n_bets, max_dd."""
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
    tracker = LoggingTracker(
        initial_bankroll=INITIAL_BANKROLL,
        drawdown_peak_window_days=DRAWDOWN_PEAK_WINDOW_DAYS,
        peak_mode="rolling_7d",
        cooldown_rounds=cooldown_rounds,
        abs_dd_frac=abs_dd_frac,
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

    bankroll = float(INITIAL_BANKROLL); peak = bankroll; max_dd = 0.0
    bet_epochs: list[int] = []
    for round_t in sim_rounds:
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
        bet_epochs.append(int(round_t.epoch))
        if bankroll > peak: peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd: max_dd = dd
        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])

    # Map fire start_ats to epochs
    start_at_to_epoch = {int(r.start_at): int(r.epoch) for r in sim_rounds}
    fire_epochs = sorted({start_at_to_epoch[s] for s in tracker.fire_start_ats
                           if s in start_at_to_epoch})

    return {
        "bet_epochs": bet_epochs,
        "fire_epochs": fire_epochs,
        "total_pnl": bankroll - INITIAL_BANKROLL,
        "n_bets": len(bet_epochs),
        "max_dd_frac": max_dd,
    }


def interval_distribution(bet_epochs: list[int]) -> dict[str, Any]:
    if len(bet_epochs) < 2:
        return {}
    sorted_bets = sorted(bet_epochs)
    gaps = [sorted_bets[i + 1] - sorted_bets[i] for i in range(len(sorted_bets) - 1)]
    arr = np.asarray(gaps)
    histogram = {
        "le_3": int(np.sum(arr <= 3)),
        "le_5": int(np.sum(arr <= 5)),
        "le_8": int(np.sum(arr <= 8)),
        "le_12": int(np.sum(arr <= 12)),
        "le_24": int(np.sum(arr <= 24)),
        "le_48": int(np.sum(arr <= 48)),
        "le_72": int(np.sum(arr <= 72)),
        "le_200": int(np.sum(arr <= 200)),
        "gt_200": int(np.sum(arr > 200)),
        "total": int(len(arr)),
    }
    summary = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
    }
    return {"histogram": histogram, "summary": summary, "n_intervals": int(len(arr))}


def counterfactual_blocked(cd0_bet_epochs: list[int], fire_epochs: list[int],
                             cd_values: tuple[int, ...]) -> dict[int, int]:
    """For each cd value, count cd0 bets within [F+1, F+cd] across all fires."""
    cd0_bets_set = set(cd0_bet_epochs)
    out = {}
    for cd in cd_values:
        count = 0
        for f in fire_epochs:
            for e in range(f + 1, f + cd + 1):
                if e in cd0_bets_set:
                    count += 1
        out[cd] = count
    return out


def post_fire_density(cd0_bet_epochs: list[int], fire_epochs: list[int],
                       n: int) -> dict[str, Any]:
    """Fraction of rounds in [F+1, F+n] that have a cd0 BET, across all fires."""
    cd0_bets_set = set(cd0_bet_epochs)
    total_window_rounds = n * len(fire_epochs)
    bets_in_window = 0
    for f in fire_epochs:
        for e in range(f + 1, f + n + 1):
            if e in cd0_bets_set:
                bets_in_window += 1
    return {
        "n_rounds_per_fire": n,
        "n_fires": len(fire_epochs),
        "total_window_rounds": total_window_rounds,
        "bets_in_window": bets_in_window,
        "bet_rate": bets_in_window / total_window_rounds if total_window_rounds else 0.0,
    }


def step15_cooldown_table():
    """Pull cd vs (bets, PnL) from Step 15 JSON if available."""
    p = REPO / "var" / "strategy_review" / "cooldown_sweep_step15_data.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        d = json.load(f)
    # Step 15 format: per_scale[5.0][cd_str] = {n_bets, pnl_bnb, ...}
    out = {}
    if "per_scale" in d and "5.0" in d["per_scale"]:
        for cd_str, vals in d["per_scale"]["5.0"].items():
            out[int(cd_str)] = vals
    return out


def main():
    t_all = time.time()

    # ----- Load rounds + klines -----
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

    # ----- Baseline cd=72 -----
    print("\n--- baseline (cd=72, dd=0.15, 5 BNB) ---", flush=True)
    t = time.time()
    cd72 = run_logging_backtest(
        all_rounds=all_rounds, btc_klines=btc_kl, eth_klines=eth_kl, sol_klines=sol_kl,
        cooldown_rounds=72, abs_dd_frac=ABS_DD_FRAC,
    )
    print(f"  bets={cd72['n_bets']} fires={len(cd72['fire_epochs'])} "
          f"pnl={cd72['total_pnl']:+.4f} max_dd={cd72['max_dd_frac']*100:.2f}% "
          f"({time.time()-t:.1f}s)", flush=True)

    # ----- cd=0 (no cooldown) -----
    print("\n--- cd=0 (cooldown disabled, dd=0.15, 5 BNB) ---", flush=True)
    t = time.time()
    cd0 = run_logging_backtest(
        all_rounds=all_rounds, btc_klines=btc_kl, eth_klines=eth_kl, sol_klines=sol_kl,
        cooldown_rounds=0, abs_dd_frac=ABS_DD_FRAC,
    )
    print(f"  bets={cd0['n_bets']} fires={len(cd0['fire_epochs'])} "
          f"pnl={cd0['total_pnl']:+.4f} max_dd={cd0['max_dd_frac']*100:.2f}% "
          f"({time.time()-t:.1f}s)", flush=True)

    # ----- Analysis -----
    print("\n=== 1. Inter-bet interval distribution (cd=72 BET stream) ===", flush=True)
    dist = interval_distribution(cd72["bet_epochs"])
    print(f"  n_intervals: {dist['n_intervals']}", flush=True)
    print(f"  summary (rounds between consecutive bets):", flush=True)
    for k, v in dist["summary"].items():
        print(f"    {k:>6s}: {v:>8.2f}", flush=True)
    print(f"  histogram (cumulative counts):", flush=True)
    total = dist["histogram"]["total"]
    for k in ("le_3", "le_5", "le_8", "le_12", "le_24", "le_48", "le_72", "le_200", "gt_200"):
        c = dist["histogram"][k]
        pct = c / total * 100 if total else 0
        print(f"    {k:>8s}: {c:>5d} ({pct:>5.2f}%)", flush=True)

    print("\n=== 2. Counterfactual blocked bets per cd value ===", flush=True)
    print(f"  baseline cd=72 fire count: {len(cd72['fire_epochs'])}", flush=True)
    print(f"  cd=0 BET count: {cd0['n_bets']}", flush=True)
    print(f"  cd=72 BET count: {cd72['n_bets']}  (blocked = {cd0['n_bets'] - cd72['n_bets']})", flush=True)
    blocked = counterfactual_blocked(cd0["bet_epochs"], cd72["fire_epochs"], CD_VALUES_TO_TEST)
    print(f"  {'cd':>5s}  {'blocked_in_windows':>20s}", flush=True)
    for cd, count in blocked.items():
        print(f"  {cd:>5d}  {count:>20d}", flush=True)

    print("\n=== 3. Post-fire bet density (cd=0 BETs in [F+1, F+8]) ===", flush=True)
    density = post_fire_density(cd0["bet_epochs"], cd72["fire_epochs"], 8)
    baseline_bet_rate = cd0["n_bets"] / 62084
    print(f"  baseline cd=0 bet rate (across all 62084 rounds): {baseline_bet_rate*100:.3f}%", flush=True)
    print(f"  post-fire window: {density['n_rounds_per_fire']} rounds × {density['n_fires']} fires = "
          f"{density['total_window_rounds']} round-slots", flush=True)
    print(f"  cd=0 bets in window: {density['bets_in_window']}", flush=True)
    print(f"  post-fire bet rate: {density['bet_rate']*100:.3f}%", flush=True)
    ratio = density['bet_rate'] / baseline_bet_rate if baseline_bet_rate else 0
    print(f"  ratio to baseline: {ratio:.2f}x", flush=True)

    print("\n=== 4. Step 15 cooldown saved-PnL accounting (5 BNB) ===", flush=True)
    s15 = step15_cooldown_table()
    if s15 is None:
        print("  Step 15 data not found", flush=True)
    else:
        production_bets = s15.get(72, {}).get("n_bets") or cd72["n_bets"]
        production_pnl = s15.get(72, {}).get("pnl_bnb") or cd72["total_pnl"]
        print(f"  cd=72 production: bets={production_bets} pnl={production_pnl:+.4f}", flush=True)
        print(f"  {'cd':>4s}  {'bets':>6s}  {'PnL':>10s}  {'d_bets vs72':>11s}  {'d_PnL vs72':>10s}  {'PnL/extra_bet':>13s}", flush=True)
        for cd in (0, 3, 5, 8, 12, 24, 48, 72):
            row = s15.get(cd)
            if not row:
                continue
            bets = row["n_bets"]
            pnl = row["pnl_bnb"]
            d_bets = bets - production_bets
            d_pnl = pnl - production_pnl
            per_extra = d_pnl / d_bets if d_bets != 0 else 0
            print(f"  {cd:>4d}  {bets:>6d}  {pnl:>+10.4f}  {d_bets:>+11d}  {d_pnl:>+10.4f}  "
                  f"{per_extra:>+13.4f}", flush=True)

    # ----- Save -----
    out_path = REPO / "var" / "strategy_review" / "step23_intra_cooldown_bet_density_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {"abs_dd_frac": ABS_DD_FRAC, "initial_bankroll": INITIAL_BANKROLL,
                        "cd_values_to_test": list(CD_VALUES_TO_TEST)},
            "cd72": {"n_bets": cd72["n_bets"], "n_fires": len(cd72["fire_epochs"]),
                      "total_pnl": cd72["total_pnl"], "max_dd_frac": cd72["max_dd_frac"],
                      "fire_epochs": cd72["fire_epochs"]},
            "cd0": {"n_bets": cd0["n_bets"], "n_fires": len(cd0["fire_epochs"]),
                     "total_pnl": cd0["total_pnl"], "max_dd_frac": cd0["max_dd_frac"]},
            "interval_distribution": dist,
            "counterfactual_blocked": blocked,
            "post_fire_density_8": density,
            "step15_data": s15,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
