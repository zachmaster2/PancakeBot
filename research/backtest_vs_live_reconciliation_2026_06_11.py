"""Backtest-vs-live reconciliation over the VM live era (read-only research).

Compares the live bot's actual bets (var/live/bets.jsonl, fetched to
``var/strategy_review/post_cv5_to_current_2026_06_10/live_bets.jsonl``)
against the canonical backtest's decisions on the SAME epochs (the
2026-06-10 step10b run's 50 BNB trades — the 5 BNB run's sequential risk
state diverges from the live bot's, so the 50 BNB run is the cleaner
decision-level comparator; risk-state skips are reported separately).

Questions answered per epoch range [first live bet .. dataset end]:
  1. Side agreement on epochs where BOTH bet.
  2. Live-only bets: what did the backtest do instead (skip reason)?
  3. Backtest-only bets: how often did live not bet at all?
  4. Outcome split: live WR on settled bets vs backtest WR on the same epochs;
     LATE (reverted) bets reported separately — they are execution losses,
     not signal decisions.

Run:  cd <repo> && .venv/Scripts/python.exe research/backtest_vs_live_reconciliation_2026_06_11.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "var" / "strategy_review" / "post_cv5_to_current_2026_06_10"
LIVE_LEDGER = DATA_DIR / "live_bets.jsonl"
BT_TRADES = DATA_DIR / "trades_50bnb.csv"
BT_TRADES_5 = DATA_DIR / "trades_5bnb.csv"
OUT_PATH = REPO_ROOT / "var" / "incident_reports" / "backtest_vs_live_reconciliation_2026_06_11.json"


def load_live() -> dict[int, dict]:
    """epoch -> {side, amount, terminal}; terminal in LATE/WON/LOST/OPEN."""
    out: dict[int, dict] = {}
    for line in LIVE_LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        e = int(r["epoch"])
        st = r["status"]
        if st in ("SUBMITTED", "PLACED"):
            out.setdefault(e, {"terminal": "OPEN"})
            out[e]["side"] = r["side"]
            out[e]["amount"] = float(r["amount_bnb"])
        elif st == "LATE":
            out.setdefault(e, {"terminal": "OPEN"})["terminal"] = "LATE"
        elif st == "SETTLED_WON":
            out.setdefault(e, {"terminal": "OPEN"})["terminal"] = "WON"
            out[e]["delta"] = float(r.get("delta_bnb", 0.0))
        elif st == "SETTLED_LOST":
            out.setdefault(e, {"terminal": "OPEN"})["terminal"] = "LOST"
            out[e]["delta"] = float(r.get("delta_bnb", 0.0))
    return out


def load_backtest(path: Path, lo: int, hi: int) -> dict[int, dict]:
    """epoch -> {action, side, skip_reason} for the window."""
    out: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            e = int(row["epoch"])
            if e < lo or e > hi:
                continue
            out[e] = {
                "action": row["action"],
                "side": row["direction"] or None,
                "skip_reason": row["skip_reason"] or None,
            }
    return out


def main() -> int:
    live = load_live()
    lo = min(live)
    # the dataset (and therefore the backtest) ends before the newest live bets
    bt_all = load_backtest(BT_TRADES, lo, 10**9)
    hi = max(bt_all)
    live_in = {e: v for e, v in live.items() if e <= hi}
    print(f"window: [{lo}..{hi}]  live bets in window: {len(live_in)} "
          f"(of {len(live)} total; {len(live) - len(live_in)} newer than dataset)")

    bt5_all = load_backtest(BT_TRADES_5, lo, hi)

    both, live_only, late = [], [], []
    for e, lv in sorted(live_in.items()):
        if lv["terminal"] == "LATE":
            late.append(e)
        bt = bt_all.get(e)
        if bt is None:
            live_only.append((e, "epoch_missing_from_backtest"))
        elif bt["action"] == "BET":
            both.append((e, lv, bt))
        else:
            live_only.append((e, bt["skip_reason"]))

    bt_bet_epochs = {e for e, r in bt_all.items() if r["action"] == "BET"}
    bt_only = sorted(bt_bet_epochs - set(live_in))

    agree = sum(1 for _, lv, bt in both if lv.get("side") == bt["side"])
    print(f"\nepochs where BOTH bet      : {len(both)}  side agreement: "
          f"{agree}/{len(both)}")
    print(f"live-only bets             : {sum(1 for _, r in live_only if r != 'epoch_missing_from_backtest')}"
          f"  (backtest skip reasons: {Counter(r for _, r in live_only).most_common(5)})")
    print(f"backtest(50)-only bets     : {len(bt_only)} (live placed nothing)")

    settled = {e: v for e, v in live_in.items() if v["terminal"] in ("WON", "LOST")}
    wins = sum(1 for v in settled.values() if v["terminal"] == "WON")
    pnl = sum(v.get("delta", 0.0) for v in settled.values())
    print(f"\nlive settled               : {len(settled)} bets, WR "
          f"{wins / len(settled):.2%}, net delta {pnl:+.4f} BNB")
    print(f"live LATE (reverted)       : {len(late)} bets (execution losses, gas only)")
    # off350 broadcast-lead fix went READY at epoch 488210 (2026-06-08)
    n_late_recent = sum(1 for e in late if e >= 488210)
    print(f"  LATE post-off350 (>=488210): {n_late_recent}")

    # backtest decisions on live's settled epochs
    bt_on_settled = [bt_all[e] for e in settled if e in bt_all]
    bt_bet_same = sum(1 for r in bt_on_settled if r["action"] == "BET")
    bt5_on_settled = [bt5_all[e] for e in settled if e in bt5_all]
    bt5_bet_same = sum(1 for r in bt5_on_settled if r["action"] == "BET")
    print(f"\nof live's settled epochs   : backtest@50 also bet on {bt_bet_same}, "
          f"backtest@5 on {bt5_bet_same}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "window": [lo, hi],
        "live_bets_in_window": len(live_in),
        "both_bet": len(both),
        "side_agreement": agree,
        "live_only_skip_reasons": dict(Counter(r for _, r in live_only)),
        "backtest50_only_bets": len(bt_only),
        "live_settled": len(settled),
        "live_wr": (wins / len(settled)) if settled else None,
        "live_net_delta_bnb": pnl,
        "live_late": len(late),
        "live_late_post_off350": n_late_recent,
        "backtest50_bet_on_live_settled": bt_bet_same,
        "backtest5_bet_on_live_settled": bt5_bet_same,
    }, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
