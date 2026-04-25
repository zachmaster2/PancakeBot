"""A/B backtest: captured klines (live OKX) vs history klines (OKX history endpoint).

Replays the same epoch range twice through the same strategy code with
two kline sources, then diffs the per-round decisions to quantify how
much live and history disagree.

Output: stdout summary + JSON diff dump at
``var/sweep/_ab_captured_vs_history/diff.json``.

Usage:
    python research/ab_captured_vs_history.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tomlkit  # noqa: E402

CAPTURE_PATH = REPO_ROOT / "var" / "dry" / "captured_klines.jsonl"
OUT_DIR = REPO_ROOT / "var" / "sweep" / "_ab_captured_vs_history"
VENV_PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def _captured_epoch_range() -> tuple[int, int]:
    eps = []
    for line in CAPTURE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("klines_btc"):  # only rounds with valid klines
            eps.append(int(rec["epoch"]))
    return min(eps), max(eps)


def _make_config(epoch_start: int, epoch_end: int, dst: Path) -> None:
    text = (REPO_ROOT / "config.toml").read_text(encoding="utf-8")
    doc = tomlkit.parse(text)
    doc["backtest"]["epoch_start"] = epoch_start
    doc["backtest"]["epoch_end"] = epoch_end
    doc["backtest"]["simulation_size"] = (epoch_end - epoch_start) + 5
    dst.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _run_backtest(label: str, kline_source: str, config_path: Path, run_dir: Path) -> dict:
    """Invoke run.py --backtest with the given kline_source. Returns parsed summary.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PY),
        str(REPO_ROOT / "run.py"),
        "--config", str(config_path),
        "--backtest",
        "--kline-source", kline_source,
    ]
    print(f"\n=== running {label}: {' '.join(cmd[-3:])} ===", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"{label} backtest failed rc={proc.returncode}")
    # Move outputs to run_dir
    for fname in ("summary.json", "trades.csv"):
        src = REPO_ROOT / "var" / "backtest" / fname
        if src.exists():
            shutil.copy(str(src), str(run_dir / fname))
    summary = json.load((run_dir / "summary.json").open(encoding="utf-8"))
    print(f"  bets={summary['num_bets']} wr={summary['win_rate']:.4f} pnl={summary['net_pnl_bnb']:+.4f}")
    return summary


def _read_trades(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Parse trades.csv into {epoch: row_dict}. Every round produces one row."""
    out: dict[int, dict[str, Any]] = {}
    with (run_dir / "trades.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["epoch"])] = row
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap_start, cap_end = _captured_epoch_range()
    print(f"capture range: {cap_start}..{cap_end} ({cap_end - cap_start + 1} epochs)")

    # Generate matching config (same epoch range for both runs).
    cfg_path = OUT_DIR / "config.toml"
    _make_config(cap_start, cap_end, cfg_path)

    history_dir = OUT_DIR / "history"
    captured_dir = OUT_DIR / "captured"

    hist_summary = _run_backtest("HISTORY", "history", cfg_path, history_dir)
    cap_summary = _run_backtest("CAPTURED", "captured", cfg_path, captured_dir)

    # Diff per-round decisions.
    hist_rows = _read_trades(history_dir)
    cap_rows = _read_trades(captured_dir)

    common = sorted(set(hist_rows) & set(cap_rows))
    only_hist = sorted(set(hist_rows) - set(cap_rows))
    only_cap = sorted(set(cap_rows) - set(hist_rows))
    print(f"\nepoch coverage: common={len(common)} only_history={len(only_hist)} only_captured={len(only_cap)}")

    # Per-round comparison
    decision_diff = []   # rounds where action differs
    bet_both = []        # rounds where both BET
    skip_both = []       # rounds where both SKIP
    bet_only_hist = []   # bet under history, skip under captured
    bet_only_cap = []    # bet under captured, skip under history

    for ep in common:
        h = hist_rows[ep]
        c = cap_rows[ep]
        h_bet = (h["action"] == "BET")
        c_bet = (c["action"] == "BET")
        if h_bet and c_bet:
            bet_both.append(ep)
        elif h_bet and not c_bet:
            bet_only_hist.append(ep)
            decision_diff.append(ep)
        elif c_bet and not h_bet:
            bet_only_cap.append(ep)
            decision_diff.append(ep)
        else:
            skip_both.append(ep)

    print(f"\n--- decision comparison ---")
    print(f"both BET:        {len(bet_both)}")
    print(f"both SKIP:       {len(skip_both)}")
    print(f"BET-only hist:   {len(bet_only_hist)}  (history says BET, captured says SKIP)")
    print(f"BET-only cap:    {len(bet_only_cap)}  (captured says BET, history says SKIP)")
    n_diff = len(decision_diff)
    print(f"decision diff rate: {n_diff}/{len(common)} = {100*n_diff/len(common):.2f}%")

    print(f"\n--- BET-only-hist epochs (live MISSED these bets) ---")
    for ep in bet_only_hist:
        h = hist_rows[ep]
        c_skip = cap_rows[ep]
        print(f"  ep={ep}: hist BET {h['direction']} {h['bet_size_bnb']} BNB pnl={h['profit_bnb']} | cap SKIP={c_skip['skip_reason']}")
    print(f"\n--- BET-only-cap epochs (live FIRED these, history skipped) ---")
    for ep in bet_only_cap:
        c = cap_rows[ep]
        h_skip = hist_rows[ep]
        print(f"  ep={ep}: cap BET {c['direction']} {c['bet_size_bnb']} BNB pnl={c['profit_bnb']} | hist SKIP={h_skip['skip_reason']}")

    # Theoretical PnL gap: how much would the bot have earned if it had used history klines?
    hist_total_pnl = sum(float(hist_rows[ep]["profit_bnb"]) for ep in common if hist_rows[ep]["action"] == "BET")
    cap_total_pnl = sum(float(cap_rows[ep]["profit_bnb"]) for ep in common if cap_rows[ep]["action"] == "BET")
    print(f"\n--- theoretical PnL on {len(common)} captured epochs ---")
    print(f"history-source backtest: bets={sum(1 for ep in common if hist_rows[ep]['action']=='BET')} pnl={hist_total_pnl:+.4f} BNB")
    print(f"captured-source backtest: bets={sum(1 for ep in common if cap_rows[ep]['action']=='BET')} pnl={cap_total_pnl:+.4f} BNB")
    print(f"gap (hist - cap): {hist_total_pnl - cap_total_pnl:+.4f} BNB over {len(common)} epochs")

    # Persist diff
    diff_out = OUT_DIR / "diff.json"
    diff_out.write_text(json.dumps({
        "epoch_range": [cap_start, cap_end],
        "common_n": len(common),
        "decision_diff": {
            "n": n_diff,
            "rate_pct": round(100 * n_diff / len(common), 4),
            "bet_only_hist": bet_only_hist,
            "bet_only_cap": bet_only_cap,
        },
        "totals": {
            "bet_both": len(bet_both),
            "skip_both": len(skip_both),
        },
        "pnl": {
            "history_pnl_bnb": hist_total_pnl,
            "captured_pnl_bnb": cap_total_pnl,
            "gap_bnb": hist_total_pnl - cap_total_pnl,
            "history_bets": sum(1 for ep in common if hist_rows[ep]["action"] == "BET"),
            "captured_bets": sum(1 for ep in common if cap_rows[ep]["action"] == "BET"),
        },
        "summaries": {
            "history": hist_summary,
            "captured": cap_summary,
        },
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {diff_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
