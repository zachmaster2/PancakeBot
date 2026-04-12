"""Robustness verification for best configs found in sweep_v3.

Does 8-segment and 10-segment analysis to check consistency.
Also tests sensitivity to parameter perturbations (overfitting check).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.sweep_v2 import run_v2


def detailed_report(params, label, n_segs=8, sim_size=20000):
    """Run with N segments and detailed tier+skip breakdown."""
    net, _, trades, hrs = run_v2(params, sim_size=sim_size, verbose=False)
    seg_size = sim_size // n_segs
    total_bets = sum(1 for t in trades if t[1] == "BET")
    total_wins = sum(1 for t in trades if t[1] == "BET" and t[2] > 0)

    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"NET: {net:+.2f} BNB | Bets: {total_bets} | WR: {total_wins/max(1,total_bets)*100:.1f}%")

    # Tier breakdown
    tiers = {}
    for t in trades:
        if t[1] != "BET": continue
        tn = t[4] or "unknown"
        tiers.setdefault(tn, [0, 0, 0.0])
        tiers[tn][0] += 1
        if t[2] > 0: tiers[tn][1] += 1
        tiers[tn][2] += t[2]
    for tn in sorted(tiers):
        nb, nw, pnl = tiers[tn]
        print(f"  {tn:12s}: {nb:4d} bets, WR={nw/max(1,nb)*100:5.1f}%, PnL={pnl:+7.2f}")

    # Segment analysis
    segs = []
    for s in range(n_segs):
        chunk = trades[s*seg_size:(s+1)*seg_size]
        bets = [t for t in chunk if t[1] == "BET"]
        wins = [t for t in bets if t[2] > 0]
        pnl = sum(t[2] for t in bets)
        wr = len(wins)/len(bets)*100 if bets else 0
        segs.append((len(bets), wr, pnl))
        print(f"  Seg{s+1:2d}: {len(bets):4d} bets, WR={wr:5.1f}%, PnL={pnl:+7.2f}")

    neg_segs = sum(1 for s in segs if s[2] < 0)
    min_seg = min(s[2] for s in segs)
    max_seg = max(s[2] for s in segs)
    print(f"  Neg segments: {neg_segs}/{n_segs} | MinSeg={min_seg:+.2f} | MaxSeg={max_seg:+.2f}")

    # Skip reason breakdown
    skips = {}
    for t in trades:
        if t[1] == "SKIP":
            reason = t[6]
            skips[reason] = skips.get(reason, 0) + 1
    top_skips = sorted(skips.items(), key=lambda x: -x[1])[:8]
    print(f"  Top skips: {', '.join(f'{r}={c}' for r, c in top_skips)}")

    return net, segs


if __name__ == "__main__":
    # ---- Best configs to verify ----

    # Config A: Conservative skip [3,4] + linear sizing + BTC contrarian
    config_a = {
        "min_our_payout": 1.85,
        "base_frac": 0.06, "cap_bnb": 2.0, "floor_bnb": 0.10,
        "btc_agree_mult": 1.5, "btc_disagree_mult": 0.7,
        "evening_skip": None, "pool_confirm_thresh": None,
        "payout_sizing_mode": "linear",
        "payout_linear_base": 0.1, "payout_linear_slope": 1.0,
        "btc_only_signal": True,
        "btc_only_thresh": 0.0003, "btc_only_min_payout": 3.0, "btc_only_bet": 0.15,
        "allowed_hours": [h for h in range(24) if h not in [3, 4]],
    }

    # Config B: More aggressive skip [3,4,6,7] + same
    config_b = {**config_a,
        "allowed_hours": [h for h in range(24) if h not in [3, 4, 6, 7]],
    }

    # Config C: Skip [3,4,6,7,10,19] + same
    config_c = {**config_a,
        "allowed_hours": [h for h in range(24) if h not in [3, 4, 6, 7, 10, 19]],
    }

    # Config D: No hour skip, payout floor 1.86 + linear + BTC contrarian
    config_d = {
        "min_our_payout": 1.86,
        "base_frac": 0.06, "cap_bnb": 2.0, "floor_bnb": 0.10,
        "btc_agree_mult": 1.8, "btc_disagree_mult": 0.7,
        "evening_skip": None, "pool_confirm_thresh": None,
        "payout_sizing_mode": "linear",
        "payout_linear_base": 0.1, "payout_linear_slope": 1.0,
        "btc_only_signal": True,
        "btc_only_thresh": 0.0003, "btc_only_min_payout": 3.0, "btc_only_bet": 0.15,
    }

    # Config E: Skip [3,4] only, payout floor 1.86
    config_e = {**config_d,
        "allowed_hours": [h for h in range(24) if h not in [3, 4]],
    }

    print("8-SEGMENT ANALYSIS (robust consistency check)")
    print("=" * 80)
    for name, cfg in [("A: skip[3,4]+linear+btc_contra pf=1.85", config_a),
                       ("B: skip[3,4,6,7]+linear+btc_contra pf=1.85", config_b),
                       ("C: skip[3,4,6,7,10,19]+linear+btc_contra pf=1.85", config_c),
                       ("D: no_skip+linear+btc_contra pf=1.86 ag=1.8", config_d),
                       ("E: skip[3,4]+linear+btc_contra pf=1.86 ag=1.8", config_e)]:
        detailed_report(cfg, name, n_segs=8)

    # ---- Sensitivity analysis (overfitting check) ----
    print("\n\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS: Perturbing Config A parameters ±10-20%")
    print("=" * 80)

    base_cfg = config_a.copy()
    perturbations = [
        ("base (original)", {}),
        ("payout_floor 1.83", {"min_our_payout": 1.83}),
        ("payout_floor 1.87", {"min_our_payout": 1.87}),
        ("linear_slope 0.8", {"payout_linear_slope": 0.8}),
        ("linear_slope 1.2", {"payout_linear_slope": 1.2}),
        ("floor_bnb 0.08", {"floor_bnb": 0.08}),
        ("floor_bnb 0.12", {"floor_bnb": 0.12}),
        ("btc_agree 1.3", {"btc_agree_mult": 1.3}),
        ("btc_agree 1.8", {"btc_agree_mult": 1.8}),
        ("btc_disagree 0.5", {"btc_disagree_mult": 0.5}),
        ("btc_disagree 0.9", {"btc_disagree_mult": 0.9}),
        ("base_frac 0.04", {"base_frac": 0.04}),
        ("base_frac 0.08", {"base_frac": 0.08}),
        ("btc_contra_bet 0.10", {"btc_only_bet": 0.10}),
        ("btc_contra_bet 0.20", {"btc_only_bet": 0.20}),
        ("btc_contra_min_pay 2.5", {"btc_only_min_payout": 2.5}),
        ("btc_contra_min_pay 3.5", {"btc_only_min_payout": 3.5}),
        ("skip only h3", {"allowed_hours": [h for h in range(24) if h not in [3]]}),
        ("skip h3,h4,h6", {"allowed_hours": [h for h in range(24) if h not in [3, 4, 6]]}),
        ("no hour skip", {"allowed_hours": None}),
    ]

    print(f"  {'Perturbation':40s} {'NET':>8s}  {'minSeg':>8s}  {'Bets':>5s}")
    print(f"  {'-'*40} {'-'*8}  {'-'*8}  {'-'*5}")
    for label, delta in perturbations:
        p = {**base_cfg, **delta}
        net, segs, _, _ = run_v2(p, verbose=False)
        min_seg = min(s[2] for s in segs)
        nbets = sum(s[0] for s in segs)
        flag = " <<" if net < 35 else (" **" if net > 45 else "")
        print(f"  {label:40s} {net:+8.2f}  {min_seg:+8.2f}  {nbets:5d}{flag}")

    # ---- Check if Config A would work with a different 20k window ----
    print("\n\n" + "=" * 80)
    print("WINDOW ROBUSTNESS: Config A on different round ranges")
    print("=" * 80)
    from experiments.fast_backtest import _load_data
    rounds, _, _ = _load_data()
    total = len(rounds)
    print(f"  Total available rounds: {total}")

    for start_frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        start_idx = int(total * start_frac)
        end_idx = min(start_idx + 20000, total)
        actual_size = end_idx - start_idx
        if actual_size < 10000: continue
        # We can't easily pass start offset to run_v2, so we'll use sim_size trick
        # Actually run_v2 always takes the LAST sim_size rounds. Let's compute equivalent.
        # To get rounds[start_idx:end_idx], we need sim_size = total - start_idx,
        # then only process first actual_size. This is hacky but let's just pass actual_size for last-N.
        # Better: let's just test different sim_size values (which change which rounds we see)
        pass

    # More direct: test on last 10k, last 15k, last 20k, last 25k (if available)
    for sz in [10000, 12000, 15000, 18000, 20000]:
        if sz > total: continue
        net, segs, _, _ = run_v2(config_a, sim_size=sz, verbose=False)
        min_seg = min(s[2] for s in segs)
        nbets = sum(s[0] for s in segs)
        pnl_per_k = net / sz * 1000
        print(f"  Last {sz:6d}: NET={net:+8.2f} BNB, bets={nbets:5d}, minSeg={min_seg:+6.2f}, PnL/1k={pnl_per_k:+.2f}")
