"""Generate final performance plots for the optimized strategy."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def main():
    trades_path = Path("var/backtest_output/backtest_trades.csv")
    summary_path = Path("var/backtest_output/backtest_summary.json")

    with open(summary_path) as f:
        summary = json.load(f)

    # Read trades CSV
    epochs = []
    bankrolls = []
    profits = []
    actions = []
    directions = []

    with open(trades_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            bankrolls.append(float(row["bankroll_bnb"]))
            profits.append(float(row["profit_bnb"]))
            actions.append(row["action"])
            directions.append(row["direction"])

    # Filter to bets only for cumulative PnL
    bet_epochs = []
    cum_pnl = []
    running = 0.0
    bet_count = 0
    for i in range(len(epochs)):
        if actions[i] == "BET":
            running += profits[i]
            bet_count += 1
            bet_epochs.append(bet_count)
            cum_pnl.append(running)

    # 1. Equity curve (cumulative PnL by bet number)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    ax1.plot(bet_epochs, cum_pnl, color="#2196F3", linewidth=1.2)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.fill_between(bet_epochs, 0, cum_pnl, alpha=0.15, color="#2196F3")
    ax1.set_xlabel("Bet Number")
    ax1.set_ylabel("Cumulative PnL (BNB)")
    ax1.set_title(f"Equity Curve — {summary['net_pnl_bnb']:+.2f} BNB over {summary['num_bets']} bets")
    ax1.grid(True, alpha=0.3)

    # 2. Segment breakdown
    n_segs = 6
    seg_size = len(bet_epochs) // n_segs
    seg_labels = []
    seg_pnls = []
    seg_wrs = []
    seg_counts = []

    bet_profits = [p for i, p in enumerate(profits) if actions[i] == "BET"]
    for s in range(n_segs):
        start = s * seg_size
        end = (s + 1) * seg_size if s < n_segs - 1 else len(bet_profits)
        chunk = bet_profits[start:end]
        pnl = sum(chunk)
        wins = sum(1 for p in chunk if p > 0)
        wr = wins / len(chunk) * 100 if chunk else 0
        seg_labels.append(f"Seg{s+1}\n({len(chunk)} bets)")
        seg_pnls.append(pnl)
        seg_wrs.append(wr)
        seg_counts.append(len(chunk))

    ax2 = axes[0, 1]
    colors = ["#4CAF50" if p >= 0 else "#F44336" for p in seg_pnls]
    bars = ax2.bar(seg_labels, seg_pnls, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, wr in zip(bars, seg_wrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"WR={wr:.0f}%", ha="center", va="bottom", fontsize=9)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("PnL (BNB)")
    ax2.set_title("Per-Segment Breakdown (~350 bets each)")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Rolling win rate (50-bet window)
    window = 50
    rolling_wr = []
    for i in range(len(bet_profits)):
        start = max(0, i - window + 1)
        chunk = bet_profits[start:i+1]
        wins = sum(1 for p in chunk if p > 0)
        rolling_wr.append(wins / len(chunk) * 100)

    ax3 = axes[1, 0]
    ax3.plot(bet_epochs, rolling_wr, color="#FF9800", linewidth=0.8, alpha=0.7)
    ax3.axhline(50, color="red", linewidth=0.8, linestyle="--", label="Breakeven")
    ax3.axhline(summary["win_rate"] * 100, color="#2196F3", linewidth=0.8,
                linestyle="--", label=f"Avg WR={summary['win_rate']*100:.1f}%")
    ax3.set_xlabel("Bet Number")
    ax3.set_ylabel("Win Rate (%)")
    ax3.set_title(f"Rolling {window}-Bet Win Rate")
    ax3.legend(loc="lower right")
    ax3.set_ylim(25, 85)
    ax3.grid(True, alpha=0.3)

    # 4. Profit distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(bet_profits, bins=80, color="#9C27B0", alpha=0.7, edgecolor="black", linewidth=0.3)
    avg_profit = sum(bet_profits) / len(bet_profits)
    ax4.axvline(0, color="red", linewidth=1, linestyle="--")
    ax4.axvline(avg_profit, color="#2196F3", linewidth=1, linestyle="--",
                label=f"Avg={avg_profit:+.4f}")
    ax4.set_xlabel("Per-Bet Profit (BNB)")
    ax4.set_ylabel("Count")
    ax4.set_title("Profit Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle("PancakeBot — Optimized Strategy Performance\n"
                 f"34k rounds | {summary['num_bets']} bets | {summary['win_rate']*100:.1f}% WR | "
                 f"{summary['net_pnl_bnb']:+.2f} BNB net",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path("var/backtest_output/final_performance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
