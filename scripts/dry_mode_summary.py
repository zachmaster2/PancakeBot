"""Generate a concise summary of dry mode performance from the audit CSV.

Reads var/runtime/dry_cycle_audit.csv and outputs key stats:
- Total rounds processed, bets placed, win rate
- PnL (from bankroll changes)
- Skip reason breakdown
- Recent bet history
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

_GAS_COST = 0.0002  # approximate gas per bet


def summarize(csv_path: str = "var/runtime/dry_cycle_audit.csv") -> str:
    path = Path(csv_path)
    if not path.exists():
        return "No dry mode audit file found. Dry mode hasn't been run yet."

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return "Dry mode audit file is empty. No rounds processed yet."

    # Basic counts
    total_rounds = len(rows)
    n_bets = sum(1 for r in rows if r["action"] == "BET")
    skips = [r for r in rows if r["action"] == "SKIP"]

    # Time range
    first_ts = int(rows[0]["cycle_ts"])
    last_ts = int(rows[-1]["cycle_ts"])
    first_dt = datetime.fromtimestamp(first_ts, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc)
    duration_hrs = (last_ts - first_ts) / 3600

    # --- Bet settlement tracking ---
    # Uses pipeline_last_settled_epoch to detect when a bet's round is settled,
    # then checks whether bankroll increased (win) or stayed flat (loss).
    # This correctly handles losses which cause no bankroll change.
    pending_bets: list[dict] = []  # FIFO queue
    settled_bets: list[dict] = []

    prev_bankroll_after = float(rows[0]["bankroll_before_action_bnb"])

    for row in rows:
        bankroll_before = float(row["bankroll_before_action_bnb"])
        bankroll_after = float(row["bankroll_after_action_bnb"])
        settled_epoch = int(row.get("pipeline_last_settled_epoch") or 0)

        # Detect bankroll jump from previous row (win payout received)
        credit = bankroll_before - prev_bankroll_after
        win_credit = credit if credit > 0.0001 else 0.0

        # Resolve any pending bets whose epoch has been settled
        still_pending = []
        for bet in pending_bets:
            if settled_epoch >= bet["epoch"]:
                if win_credit > 0.0001:
                    # This bet won -- the bankroll jump is the credit
                    settled_bets.append({
                        **bet, "result": "WIN",
                        "pnl": win_credit - bet["size"] - _GAS_COST,
                    })
                    win_credit = 0.0  # consume the credit
                else:
                    # This bet lost -- no bankroll change
                    settled_bets.append({
                        **bet, "result": "LOSS",
                        "pnl": -(bet["size"] + _GAS_COST),
                    })
            else:
                still_pending.append(bet)
        pending_bets = still_pending

        # Track new bets
        if row["action"] == "BET":
            pending_bets.append({
                "epoch": int(row["current_epoch"]),
                "side": row["bet_side"],
                "size": float(row["bet_size_bnb"]),
            })

        prev_bankroll_after = bankroll_after

    wins = sum(1 for b in settled_bets if b["result"] == "WIN")
    losses = sum(1 for b in settled_bets if b["result"] == "LOSS")
    settled_count = len(settled_bets)
    unsettled = len(pending_bets)

    # Use actual bankroll change for net PnL (always correct)
    initial_bankroll = float(rows[0]["bankroll_before_action_bnb"])
    last_bankroll = float(rows[-1]["bankroll_after_action_bnb"])
    net_pnl = last_bankroll - initial_bankroll

    # Skip reasons
    skip_reasons: dict[str, int] = {}
    for r in skips:
        reason = r.get("skip_reason", "unknown")
        # Normalize alignment errors to a single key
        if reason.startswith("gate_bnb_unexpected_newest"):
            reason = "gate_bnb_stale_candle"
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    # Build summary
    lines = []
    lines.append("=" * 50)
    lines.append("  DRY MODE PERFORMANCE SUMMARY")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"  Period: {first_dt.strftime('%Y-%m-%d %H:%M')} to {last_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append(f"  Duration: {duration_hrs:.1f} hours ({duration_hrs/24:.1f} days)")
    lines.append("")
    lines.append(f"  Rounds processed: {total_rounds}")
    lines.append(f"  Bets placed:      {n_bets} ({n_bets/max(1,total_rounds)*100:.1f}% bet rate)")
    lines.append(f"  Settled:           {settled_count}")
    if unsettled:
        lines.append(f"  Unsettled:         {unsettled}")
    lines.append("")

    if settled_count > 0:
        wr = wins / settled_count * 100
        lines.append(f"  Wins:     {wins}")
        lines.append(f"  Losses:   {losses}")
        lines.append(f"  Win Rate: {wr:.1f}%")
        lines.append("")
        lines.append(f"  Net PnL:      {net_pnl:+.4f} BNB")
        lines.append(f"  Bankroll:     {last_bankroll:.4f} BNB (started {initial_bankroll:.4f})")
        if settled_count >= 5:
            lines.append(f"  Avg per bet:  {net_pnl/settled_count:+.4f} BNB")
    else:
        lines.append(f"  No settled bets yet.")
        lines.append(f"  Bankroll:  {last_bankroll:.4f} BNB")

    # Confidence note
    lines.append("")
    if settled_count < 50:
        lines.append(f"  [EARLY] Need ~{180 - settled_count} more bets for statistical significance")
    elif settled_count < 180:
        lines.append(f"  [BUILDING] Directional read -- need ~{180 - settled_count} more for 95% confidence")
    else:
        lines.append(f"  [SIGNIFICANT] Statistically significant sample ({settled_count} bets)")

    # Skip breakdown
    lines.append("")
    lines.append("  Skip reasons:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        lines.append(f"    {reason}: {count}")

    # Recent bets (last 8)
    recent = settled_bets[-8:] + [
        {**b, "result": "PENDING", "pnl": 0.0} for b in pending_bets
    ]
    if recent:
        lines.append("")
        lines.append("  Recent bets:")
        for bet in recent[-8:]:
            tag = {"WIN": "W", "LOSS": "L", "PENDING": "?"}.get(bet["result"], "?")
            if bet["result"] == "PENDING":
                lines.append(f"    [{tag}] epoch {bet['epoch']}: {bet['side']} {bet['size']:.4f} BNB (pending)")
            else:
                lines.append(f"    [{tag}] epoch {bet['epoch']}: {bet['side']} {bet['size']:.4f} BNB -> {bet['pnl']:+.4f}")

    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize())
