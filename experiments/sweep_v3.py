"""Combined optimization sweep - targeting +40 BNB."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.sweep_v2 import run_v2


def quick(params, label=""):
    net, segs, trades, hrs = run_v2(params, verbose=False)
    min_seg = min(s[2] for s in segs)
    nbets = sum(s[0] for s in segs)
    # Count tier breakdown
    tier_pnl = {}
    for t in trades:
        if t[1] != "BET": continue
        tier_pnl.setdefault(t[4], 0.0)
        tier_pnl[t[4]] += t[2]
    tier_str = " | ".join(f"{k}={v:+.1f}" for k, v in sorted(tier_pnl.items()))
    print(f"  {label:40s} NET={net:+7.2f} bets={nbets:4d} minSeg={min_seg:+6.2f} | {tier_str}")
    return net, segs, trades, hrs


if __name__ == "__main__":
    baseline = {
        "min_our_payout": 1.85,
        "base_frac": 0.06, "cap_bnb": 2.0, "floor_bnb": 0.10,
        "btc_agree_mult": 1.5, "btc_disagree_mult": 0.7,
        "payout_hi_thresh": 2.5, "payout_hi_mult": 1.7,
        "evening_skip": None, "pool_confirm_thresh": None,
    }

    # ---- 1. Combine linear sizing + BTC contrarian ----
    print("=" * 90)
    print("COMBO: Linear sizing + BTC contrarian")
    print("=" * 90)
    for base, slope in [(0.1, 0.8), (0.1, 0.9), (0.1, 1.0), (0.05, 1.0), (0.15, 0.7)]:
        for btc_thresh in [0.0003, 0.0004]:
            for btc_min_pay in [2.5, 3.0, 3.5]:
                for btc_bet in [0.10, 0.15, 0.20]:
                    p = {**baseline,
                         "payout_sizing_mode": "linear",
                         "payout_linear_base": base, "payout_linear_slope": slope,
                         "btc_only_signal": True,
                         "btc_only_thresh": btc_thresh,
                         "btc_only_min_payout": btc_min_pay,
                         "btc_only_bet": btc_bet}
                    net, segs, _, _ = run_v2(p, verbose=False)
                    min_seg = min(s[2] for s in segs)
                    if net > 36:
                        quick(p, f"b={base} s={slope} bthr={btc_thresh} bmp={btc_min_pay} bb={btc_bet}")

    # ---- 2. Add hour filtering on top ----
    print("\n" + "=" * 90)
    print("COMBO: Above + skip worst hours (3,4,6,7)")
    print("=" * 90)
    bad_hours_sets = [
        [3, 4],
        [3, 4, 6, 7],
        [3, 4, 6, 7, 10],
        [3, 4, 6, 7, 10, 19],
    ]
    for bad_hours in bad_hours_sets:
        good_hours = [h for h in range(24) if h not in bad_hours]
        for base, slope in [(0.1, 0.8), (0.1, 1.0)]:
            for btc_thresh in [0.0003]:
                for btc_min_pay in [2.5, 3.0]:
                    for btc_bet in [0.15, 0.20]:
                        p = {**baseline,
                             "payout_sizing_mode": "linear",
                             "payout_linear_base": base, "payout_linear_slope": slope,
                             "btc_only_signal": True,
                             "btc_only_thresh": btc_thresh,
                             "btc_only_min_payout": btc_min_pay,
                             "btc_only_bet": btc_bet,
                             "allowed_hours": good_hours}
                        net, segs, _, _ = run_v2(p, verbose=False)
                        min_seg = min(s[2] for s in segs)
                        if net > 36:
                            label = f"skip{bad_hours} b={base} s={slope} bmp={btc_min_pay} bb={btc_bet}"
                            quick(p, label)

    # ---- 3. Aggressive cap/sizing with contrarian ----
    print("\n" + "=" * 90)
    print("COMBO: Cap/sizing sweep with linear + BTC contrarian")
    print("=" * 90)
    for cap in [1.5, 2.0, 3.0, 5.0]:
        for floor in [0.08, 0.10, 0.12, 0.15]:
            for base_frac in [0.04, 0.06, 0.08]:
                p = {**baseline,
                     "cap_bnb": cap, "floor_bnb": floor, "base_frac": base_frac,
                     "payout_sizing_mode": "linear",
                     "payout_linear_base": 0.1, "payout_linear_slope": 0.8,
                     "btc_only_signal": True,
                     "btc_only_thresh": 0.0003,
                     "btc_only_min_payout": 3.0,
                     "btc_only_bet": 0.15}
                net, segs, _, _ = run_v2(p, verbose=False)
                min_seg = min(s[2] for s in segs)
                if net > 36:
                    quick(p, f"cap={cap} floor={floor} bf={base_frac}")

    # ---- 4. BTC agree/disagree multiplier sweep ----
    print("\n" + "=" * 90)
    print("BTC AGREE/DISAGREE MULT SWEEP")
    print("=" * 90)
    best_combo = {**baseline,
                  "payout_sizing_mode": "linear",
                  "payout_linear_base": 0.1, "payout_linear_slope": 0.8,
                  "btc_only_signal": True,
                  "btc_only_thresh": 0.0003,
                  "btc_only_min_payout": 3.0,
                  "btc_only_bet": 0.15}
    for ag in [1.0, 1.3, 1.5, 1.8, 2.0]:
        for dis in [0.5, 0.6, 0.7, 0.8, 1.0]:
            p = {**best_combo, "btc_agree_mult": ag, "btc_disagree_mult": dis}
            net, segs, _, _ = run_v2(p, verbose=False)
            min_seg = min(s[2] for s in segs)
            if net > 36:
                quick(p, f"ag={ag} dis={dis}")

    # ---- 5. Payout floor fine-tuning around 1.85 ----
    print("\n" + "=" * 90)
    print("PAYOUT FLOOR FINE-TUNE with best combo")
    print("=" * 90)
    for floor in [1.83, 1.84, 1.85, 1.86, 1.87, 1.88]:
        p = {**best_combo, "min_our_payout": floor}
        quick(p, f"floor={floor}")

    # ---- 6. Overall best ----
    print("\n" + "=" * 90)
    print("FULL VERBOSE: Best combo")
    print("=" * 90)
    run_v2(best_combo)
