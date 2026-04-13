"""Regime fix v2: Stack winning features from v1.

Key finding from v1:
  - vol_size_scale is the ONLY 4/4 positive config (reduces bet in high vol)
  - skip_btc_disagree has highest total PnL (+35.21)
  - mag_floor_3bps is simple and effective (+31.99)

This sweep stacks them and tests combinations.
ALL follow EXPERIMENT_RULES.md.
"""
from __future__ import annotations
import sys, json, math, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.regime_fix import run_config


def main():
    print("=" * 90)
    print("REGIME FIX V2: Stacking winning features")
    print("=" * 90)

    configs = collections.OrderedDict()

    # Baselines for reference
    configs["REF_baseline"] = {}
    configs["REF_vol_scale"] = {"vol_size_scale": True}

    # Stack: vol_size_scale + skip_btc_disagree
    configs["S1_volscale_btcdis"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
    }

    # Stack: vol_size_scale + mag_floor_3bps
    configs["S2_volscale_mag3"] = {
        "vol_size_scale": True,
        "min_mag_bps": 3.0,
    }

    # Stack: vol_size_scale + btc_dis + mag3
    configs["S3_volscale_btcdis_mag3"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
        "min_mag_bps": 3.0,
    }

    # Stack: vol_size_scale + SNR 3x
    configs["S4_volscale_snr3"] = {
        "vol_size_scale": True,
        "snr_filter": 3.0,
    }

    # Stack: vol_size_scale + SNR 3x + btc_dis
    configs["S5_volscale_snr3_btcdis"] = {
        "vol_size_scale": True,
        "snr_filter": 3.0,
        "skip_btc_disagree": True,
    }

    # Test different vol_size reference points
    # (the vol_size_scale in regime_fix.py uses ref=0.5 bps)
    # Let's try with different effective scales by using different cap floors

    # Stack: vol_size + btc_dis + lower payout floor (let in more rounds)
    configs["S6_volscale_btcdis_pay175"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
        "min_pre_payout": 1.75,
    }

    # Stack: vol_size + btc_dis + higher payout floor (be more selective)
    configs["S7_volscale_btcdis_pay190"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
        "min_pre_payout": 1.90,
    }

    # Stack: vol_size + btc_dis + larger base_frac (more aggressive when conditions good)
    configs["S8_volscale_btcdis_frac8"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
        "base_frac": 0.08,
    }

    # Stack: vol_size + btc_dis + bigger btc_agree multiplier
    configs["S9_volscale_btcdis_btcag175"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
        "btc_agree_mult": 1.75,
    }

    # Stack: vol_size + btc_dis + floor 0.08 + cap 0.45
    configs["S10_volscale_btcdis_wider"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
        "floor_bnb": 0.08,
        "cap_bnb": 0.45,
    }

    # Stack: vol_size + btc_dis + btc_agree 1.75 + base 0.08
    configs["S11_volscale_btcdis_aggressive"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
        "btc_agree_mult": 1.75,
        "base_frac": 0.08,
    }

    # Stack: vol_size + btc_dis + snr 3x (triple stack)
    configs["S12_triple_stack"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
        "snr_filter": 3.0,
    }

    # Stack: all winners at once
    configs["S13_all_winners"] = {
        "vol_size_scale": True,
        "skip_btc_disagree": True,
        "snr_filter": 3.0,
        "min_mag_bps": 3.0,
    }

    results = {}
    for name, params in configs.items():
        net, segs, trades, seg5 = run_config(name, params)
        results[name] = {
            "net": net,
            "n_bets": sum(1 for t in trades if t[1] == "BET"),
            "seg5": seg5,
        }

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY (sorted by consistency then NET)")
    print("=" * 90)
    print(f"  {'Config':35s} {'NET':>8s} {'PnL/1k':>7s} {'Bets':>6s} {'Pos':>4s} {'Std':>6s} {'Worst':>8s} {'Best':>8s}")
    print("  " + "-" * 85)

    # Sort by: fewest negative segments first, then by net PnL
    def sort_key(item):
        name, r = item
        pnls = [p for _, _, p in r["seg5"]]
        neg = sum(1 for p in pnls if p < 0)
        return (neg, -r["net"])

    for name, r in sorted(results.items(), key=sort_key):
        pnl_k = r["net"] / (49488 / 1000)
        pnls = [p for _, _, p in r["seg5"]]
        neg = sum(1 for p in pnls if p < 0)
        worst = min(pnls) if pnls else 0
        best = max(pnls) if pnls else 0
        std = (sum((p - sum(pnls) / len(pnls)) ** 2 for p in pnls) / len(pnls)) ** 0.5 if pnls else 0
        mark = " **" if neg == 0 else " *" if neg <= 1 else ""
        print(f"  {name:35s} {r['net']:>+8.2f} {pnl_k:>+7.2f} {r['n_bets']:>6d} {len(pnls)-neg}/{len(pnls)} {std:>6.1f} {worst:>+8.2f} {best:>+8.2f}{mark}")

    # Detailed view of top configs
    print("\n" + "=" * 90)
    print("TOP CONFIGS: Detailed 10k segment view")
    print("=" * 90)

    # Get top 5 by sort_key
    top5 = sorted(results.items(), key=sort_key)[:5]
    for name, r in top5:
        pnls = [p for _, _, p in r["seg5"]]
        neg = sum(1 for p in pnls if p < 0)
        print(f"\n  {name}: NET={r['net']:+.2f}, {len(r['seg5'])-neg}/{len(r['seg5'])} positive")
        for i, (nb, wr, pnl) in enumerate(r["seg5"]):
            bar = "+" * int(max(0, pnl) * 2) + "-" * int(max(0, -pnl) * 2)
            print(f"    10k-{i+1}: {nb:4d} bets, WR={wr:5.1f}%, PnL={pnl:+7.2f} |{bar}")


if __name__ == "__main__":
    main()
