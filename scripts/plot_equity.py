"""Plot equity curves: current config vs alternatives for linearity comparison."""
from __future__ import annotations

import json
import sys
import time
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
MIN_BET = 0.001

def load_and_precompute():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    def lk(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out
    bnb = lk("var/bnb_spot_prices.jsonl")
    btc = lk("var/btc_spot_prices.jsonl")
    eth = lk("var/eth_spot_prices.jsonl")
    sol = lk("var/sol_spot_prices.jsonl")

    orig = _gate_mod._MTF_THRESH
    _gate_mod._MTF_THRESH = 0.00005
    try:
        result = []
        for rnd in rounds:
            ep = int(rnd.epoch)
            la = int(rnd.lock_at)
            cms = (la - CUTOFF_S) * 1000
            b_raw, t_raw = bnb.get(ep), btc.get(ep)
            if b_raw is None or t_raw is None:
                result.append((rnd, None, 0, 0, 0, 0, 0, 0))
                continue
            sig = compute_signal_from_klines(b_raw, t_raw, cms, eth_klines=eth.get(ep), sol_klines=sol.get(ep))
            pb, pe = _pools_from_bets(rnd, la - POOL_CUTOFF_SECONDS)
            result.append((rnd, sig.signal, sig.signal_strength, sig.eth_confirmation_strength,
                           sig.sol_confirmation_strength, pb, pe, pb + pe))
    finally:
        _gate_mod._MTF_THRESH = orig
    return result


def simulate(data, min_pool, thresh_mode, uniform_thresh, small_thresh, large_thresh,
             thresh_boundary, min_payout, base_frac, slope, cap_bnb, payout_slope=1.0,
             eth_w=0.3, sol_w=0.3, max_frac=0.30):
    """Returns list of (round_index, bankroll) for equity curve."""
    bankroll = 50.0
    curve = [(0, bankroll)]
    for i, (rnd, signal, strength, eth_c, sol_c, pb, pe, pt) in enumerate(data):
        if signal is None:
            curve.append((i + 1, bankroll))
            continue
        if thresh_mode == "uniform":
            if strength < uniform_thresh: curve.append((i+1, bankroll)); continue
        else:
            t = large_thresh if pt >= thresh_boundary else small_thresh
            if strength < t: curve.append((i+1, bankroll)); continue
        if pt < min_pool: curve.append((i+1, bankroll)); continue
        our = pb if signal == "Bull" else pe
        if our > 0 and pt > 0:
            pay = pt * 0.97 / our
            if pay < min_payout: curve.append((i+1, bankroll)); continue
        elif our <= 0:
            pay = 99.0
        else:
            curve.append((i+1, bankroll)); continue
        eff = strength + (eth_c * eth_w if eth_c > 0 else 0) + (sol_c * sol_w if sol_c > 0 else 0)
        frac = min(base_frac + slope * eff, max_frac)
        if our > 0:
            pm = max(0.5, 1.0 + payout_slope * (pay - 2.0))
            frac = min(frac * pm, max_frac)
        bet = max(0.01, min(cap_bnb, pt * frac))
        if bet < MIN_BET: curve.append((i+1, bankroll)); continue
        bankroll -= bet + GAS_COST_BET_BNB
        out = settle_bet_against_closed_round(bet_bnb=bet, bet_side=signal, round_closed=rnd, treasury_fee_fraction=0.03)
        bankroll += out.credit_bnb
        curve.append((i + 1, bankroll))
    return curve


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Loading and pre-computing...", flush=True)
    data = load_and_precompute()
    print(f"  {len(data)} rounds loaded")

    configs = {
        "BTC only (pool>=1.5, b=.04)": dict(
            min_pool=1.5, thresh_mode="adaptive", uniform_thresh=0.0001,
            small_thresh=0.0002, large_thresh=0.0001, thresh_boundary=3.0,
            min_payout=1.5, base_frac=0.04, slope=100, cap_bnb=2.0,
        ),
        "Prior (pool>=2.0, b=.03)": dict(
            min_pool=2.0, thresh_mode="adaptive", uniform_thresh=0.0001,
            small_thresh=0.0002, large_thresh=0.0001, thresh_boundary=3.0,
            min_payout=1.5, base_frac=0.03, slope=100, cap_bnb=2.0,
        ),
    }

    # Load production backtest with regime-2 from trades CSV
    import csv
    regime2_curve = []
    trades_path = Path("var/backtest_trades.csv")
    if trades_path.exists():
        with open(trades_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                regime2_curve.append((i + 1, float(row["bankroll_bnb"]) - 50.0))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    colors = ["#E91E63", "#2196F3", "#9E9E9E"]
    # Plot regime-2 curve first (from production backtest)
    if regime2_curve:
        xs = [c[0] for c in regime2_curve]
        ys = [c[1] for c in regime2_curve]
        axes[0].plot(xs, ys, label="Current: BTC + ETH/SOL regime-2", color=colors[0], linewidth=2.0, alpha=0.9)

    for (label, cfg), color in zip(configs.items(), colors[1:]):
        print(f"  Simulating: {label}...", flush=True)
        curve = simulate(data, **cfg)
        xs = [c[0] for c in curve]
        ys = [c[1] - 50.0 for c in curve]  # PnL relative to start
        lw = 1.5 if "BTC only" in label else 1.0
        axes[0].plot(xs, ys, label=label, color=color, linewidth=lw, alpha=0.7)

    axes[0].set_ylabel("Cumulative PnL (BNB)")
    axes[0].set_title("Equity Curves: Strategy Variants Over 35k Rounds")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    axes[0].grid(True, alpha=0.3)

    # 5-fold boundaries
    fold_size = len(data) // 5
    for i in range(1, 5):
        axes[0].axvline(x=i * fold_size, color="gray", linewidth=0.5, linestyle=":")
        axes[0].text(i * fold_size, axes[0].get_ylim()[1] * 0.95, f"F{i+1}", fontsize=8, color="gray", ha="center")

    # Bottom panel: rolling 500-round PnL rate
    window = 500
    if regime2_curve:
        ys_r2 = [c[1] + 50.0 for c in regime2_curve]  # absolute bankroll
        rolling = []
        for i in range(window, len(ys_r2)):
            rate = (ys_r2[i] - ys_r2[i - window]) / window * 2000
            rolling.append((i, rate))
        if rolling:
            axes[1].plot([r[0] for r in rolling], [r[1] for r in rolling],
                        label="Current: BTC + ETH/SOL regime-2", color=colors[0], linewidth=1.5, alpha=0.8)

    for (label, cfg), color in zip(configs.items(), colors[1:]):
        curve = simulate(data, **cfg)
        ys = [c[1] for c in curve]
        rolling = []
        for i in range(window, len(ys)):
            rate = (ys[i] - ys[i - window]) / window * 2000
            rolling.append((i, rate))
        if rolling:
            lw = 1.2 if "BTC only" in label else 0.8
            axes[1].plot([r[0] for r in rolling], [r[1] for r in rolling],
                        label=label, color=color, linewidth=lw, alpha=0.6)

    axes[1].set_ylabel("Rolling PnL/2k (500-round window)")
    axes[1].set_xlabel("Round index")
    axes[1].axhline(y=0, color="red", linewidth=0.5, linestyle="--")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    for i in range(1, 5):
        axes[1].axvline(x=i * fold_size, color="gray", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    out_path = "var/equity_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
