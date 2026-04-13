"""Gradual expansion robustness test.

Tests the final production strategy at increasing window sizes (20k → 22k)
to check whether PnL/1k stays stable and no segment collapses.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.sweep_v2 import run_v2


PRODUCTION_CONFIG = {
    "min_our_payout": 1.85,
    "base_frac": 0.06, "cap_bnb": 2.0, "floor_bnb": 0.10,
    "btc_agree_mult": 1.5, "btc_disagree_mult": 0.7,
    "evening_skip": None, "pool_confirm_thresh": None,
    "payout_sizing_mode": "linear",
    "payout_linear_base": 0.1, "payout_linear_slope": 1.0,
    "btc_only_signal": True,
    "btc_only_thresh": 0.0003, "btc_only_min_payout": 3.0, "btc_only_bet": 0.15,
    "allowed_hours": [h for h in range(24) if h not in [3, 4, 19]],
}


def test_window(sim_size, n_segs=4):
    """Run backtest at given window size and report."""
    try:
        net, segs, trades, hrs = run_v2(PRODUCTION_CONFIG, sim_size=sim_size, verbose=False)
    except Exception as e:
        print(f"  {sim_size:6d}: ERROR — {e}")
        return None

    total_bets = sum(s[0] for s in segs)
    total_wins = sum(1 for t in trades if t[1] == "BET" and t[2] > 0)
    wr = total_wins / max(1, total_bets) * 100
    pnl_per_k = net / sim_size * 1000
    min_seg = min(s[2] for s in segs)
    max_seg = max(s[2] for s in segs)
    neg_segs = sum(1 for s in segs if s[2] < 0)

    seg_str = " | ".join(f"{s[2]:+.1f}" for s in segs)
    flag = " ⚠️" if neg_segs > 0 else " ✓"

    print(f"  {sim_size:6d}: NET={net:+7.2f}  bets={total_bets:5d}  WR={wr:5.1f}%  "
          f"PnL/1k={pnl_per_k:+5.2f}  minSeg={min_seg:+6.2f}  segs=[{seg_str}]{flag}")

    return net, segs, pnl_per_k


if __name__ == "__main__":
    from experiments.fast_backtest import _load_data
    rounds, spot, btc = _load_data()
    max_with_klines = len(rounds)
    # Count how many of the last N rounds have klines
    for trial_sz in range(len(rounds), 0, -1000):
        window = rounds[-trial_sz:]
        covered = sum(1 for r in window if int(r.epoch) in spot)
        if covered >= trial_sz * 0.95:
            max_with_klines = trial_sz
            break

    print(f"Total rounds: {len(rounds)}")
    print(f"BNB kline coverage: {len(spot)} epochs")
    print(f"Max usable window (≥95% kline coverage): ~{max_with_klines}")
    print()
    print("GRADUAL EXPANSION TEST (production config)")
    print("=" * 90)

    sizes = list(range(20000, min(max_with_klines + 1, len(rounds) + 1), 2000))
    if sizes and sizes[-1] != min(max_with_klines, len(rounds)):
        sizes.append(min(max_with_klines, len(rounds)))

    for sz in sizes:
        test_window(sz)

    # Also do 8-segment analysis on the largest window
    if sizes:
        largest = sizes[-1]
        print(f"\n8-SEGMENT DETAIL for last {largest} rounds:")
        print("-" * 90)
        try:
            net, _, trades, _ = run_v2(PRODUCTION_CONFIG, sim_size=largest, verbose=False)
            seg_size = largest // 8
            for s in range(8):
                chunk = trades[s*seg_size:(s+1)*seg_size]
                bets = [t for t in chunk if t[1] == "BET"]
                wins = [t for t in bets if t[2] > 0]
                pnl = sum(t[2] for t in bets)
                wr = len(wins)/len(bets)*100 if bets else 0
                flag = " ⚠️" if pnl < 0 else ""
                print(f"  Seg{s+1}: {len(bets):4d} bets, WR={wr:5.1f}%, PnL={pnl:+7.2f}{flag}")
        except Exception as e:
            print(f"  ERROR: {e}")
