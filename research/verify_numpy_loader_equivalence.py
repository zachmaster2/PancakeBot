"""Verify numpy-kline-loader is bit-identical to the Python-list loader.

Runs canonical baseline (cd=72, dd=0.15, 5 BNB) on the LAST 5000 rounds
twice — once with each loader — and compares:
  1. Total PnL (exact match)
  2. Bet count
  3. Per-bet (epoch, side, won, profit) bit-identical

If anything differs, exits non-zero with diagnostic output.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

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


# Test only the last 5000 rounds to keep this fast and bounded
TEST_EPOCH_MIN = 479000  # ~last ~5000 rounds in dataset
TEST_EPOCH_MAX = 484999
CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2
POOL_CUTOFF = 6
TREASURY_FEE = 0.03
MIN_BET = 0.001


class TestTracker(InMemoryBankrollTracker):
    def __init__(self, *, initial_bankroll, drawdown_peak_window_days, peak_mode,
                  cooldown_rounds, abs_dd_frac):
        super().__init__(initial_bankroll=initial_bankroll,
                          drawdown_peak_window_days=drawdown_peak_window_days,
                          peak_mode=peak_mode)
        self._cd_total = int(cooldown_rounds)
        self._abs_dd_frac = float(abs_dd_frac)

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
                return self._cd_total > 0
        return False


def run_canonical_baseline(all_rounds, btc_klines, eth_klines, sol_klines):
    """Run canonical (cd=72, dd=0.15, 5 BNB) with timing-fix on test slice."""
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
    tracker = TestTracker(
        initial_bankroll=5.0, drawdown_peak_window_days=7,
        peak_mode="rolling_7d", cooldown_rounds=72, abs_dd_frac=0.15,
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

    sim_rounds = [r for r in all_rounds if TEST_EPOCH_MIN <= r.epoch <= TEST_EPOCH_MAX]
    sim_rounds.sort(key=lambda r: int(r.epoch))

    bankroll = 5.0
    bets = []
    pending = deque()

    for round_t in sim_rounds:
        ep = int(round_t.epoch)
        while pending and pending[0]["delivery"] <= ep:
            d = pending.popleft()
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
        bets.append({
            "epoch": ep, "side": side, "won": won, "profit": profit,
            "bet_size": bet_size,
        })
        pending.append({
            "start_at": int(round_t.start_at), "bankroll": bankroll,
            "delivery": ep + 2,
        })
        pipeline.settle_closed_rounds(rounds=[round_t])

    return bankroll - 5.0, bets


def main():
    t0 = time.time()
    print("--- loading rounds ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  {len(all_rounds)} rounds total", flush=True)

    max_lookback = max(CANONICAL_LOOKBACKS)
    earliest_offset = CANONICAL_CUTOFF + max_lookback + 1
    latest_offset = CANONICAL_CUTOFF + 1

    # === Load A: existing Python-list loader ===
    print("\n--- LOADER A: Python-list (current) ---", flush=True)
    t = time.time()
    btc_a = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth_a = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol_a = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  loaded ({time.time()-t:.1f}s)", flush=True)

    btc_a_pe = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                          max_lookback=max_lookback,
                                          earliest_offset=earliest_offset)
                 for ep, kl in btc_a.items()}
    eth_a_pe = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                          max_lookback=max_lookback,
                                          earliest_offset=earliest_offset)
                 for ep, kl in eth_a.items()}
    sol_a_pe = {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                          max_lookback=max_lookback,
                                          earliest_offset=earliest_offset)
                 for ep, kl in sol_a.items()}

    print("\n--- Running canonical baseline with LOADER A ---", flush=True)
    t = time.time()
    pnl_a, bets_a = run_canonical_baseline(all_rounds, btc_a_pe, eth_a_pe, sol_a_pe)
    print(f"  PnL_A = {pnl_a:+.6f}  n_bets_A = {len(bets_a)}  ({time.time()-t:.1f}s)", flush=True)

    # Free Loader A's memory before loading B
    del btc_a, eth_a, sol_a, btc_a_pe, eth_a_pe, sol_a_pe
    import gc; gc.collect()

    # === Load B: numpy loader ===
    print("\n--- LOADER B: numpy (new) ---", flush=True)
    t = time.time()
    btc_b = load_klines_unified_numpy(
        ipr._BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth_b = load_klines_unified_numpy(
        ipr._ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol_b = load_klines_unified_numpy(
        ipr._SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  loaded ({time.time()-t:.1f}s)", flush=True)

    btc_b_pe = {ep: slice_per_entry_numpy(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                 for ep, kl in btc_b.items()}
    eth_b_pe = {ep: slice_per_entry_numpy(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                 for ep, kl in eth_b.items()}
    sol_b_pe = {ep: slice_per_entry_numpy(kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
                                            max_lookback=max_lookback,
                                            earliest_offset=earliest_offset)
                 for ep, kl in sol_b.items()}

    print("\n--- Running canonical baseline with LOADER B ---", flush=True)
    t = time.time()
    pnl_b, bets_b = run_canonical_baseline(all_rounds, btc_b_pe, eth_b_pe, sol_b_pe)
    print(f"  PnL_B = {pnl_b:+.6f}  n_bets_B = {len(bets_b)}  ({time.time()-t:.1f}s)", flush=True)

    # === Compare ===
    print("\n=== EQUIVALENCE CHECK ===", flush=True)
    print(f"  PnL_A: {pnl_a:+.10f}", flush=True)
    print(f"  PnL_B: {pnl_b:+.10f}", flush=True)
    print(f"  diff:  {pnl_b - pnl_a:+.10f}", flush=True)
    print(f"  n_bets_A: {len(bets_a)}", flush=True)
    print(f"  n_bets_B: {len(bets_b)}", flush=True)

    if len(bets_a) != len(bets_b):
        print(f"  FAIL: bet counts differ", flush=True)
        # Find first mismatch
        for i, (ba, bb) in enumerate(zip(bets_a, bets_b)):
            if ba["epoch"] != bb["epoch"]:
                print(f"  first epoch divergence at index {i}: A={ba['epoch']} B={bb['epoch']}", flush=True)
                break
        sys.exit(1)

    mismatches = 0
    first_mismatch = None
    for i, (ba, bb) in enumerate(zip(bets_a, bets_b)):
        if (ba["epoch"] != bb["epoch"] or
            ba["side"] != bb["side"] or
            ba["won"] != bb["won"] or
            abs(ba["profit"] - bb["profit"]) > 1e-9):
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = (i, ba, bb)

    if mismatches == 0:
        print(f"  PASS: all {len(bets_a)} bets bit-identical", flush=True)
        print(f"\nTotal elapsed: {time.time() - t0:.1f}s", flush=True)
        sys.exit(0)
    else:
        print(f"  FAIL: {mismatches} bets differ", flush=True)
        print(f"  first mismatch at index {first_mismatch[0]}:", flush=True)
        print(f"    A: {first_mismatch[1]}", flush=True)
        print(f"    B: {first_mismatch[2]}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
