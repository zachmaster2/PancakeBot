"""Standard `run.py --backtest` over recent windows, risk breaker DISABLED.

Drives the production backtest path (pancakebot/backtest/runner.py via
run.py --backtest --config) — NOT a custom replay — over two trailing
windows at two stake scales, with the drawdown breaker + cooldown +
bankroll-floor turned off via a TEMP config so the gate's actual decisions
are visible (not hidden behind risk state accumulated from dead-era losses).

Windows (computed from round lock timestamps; current max epoch 490743 @
2026-06-17 21:39Z):
  last_2w : epochs 486805..490743  (trailing 14 days, 3937 rounds)
  last_1w : epochs 488773..490743  (trailing  7 days, 1971 rounds)
Scales: initial_bankroll_bnb = 5.0 and 50.0.

Risk-disable (temp config only; canonical config.toml untouched, not
committed): [strategy.risk] max_drawdown_fraction_from_peak=1.0 (the config
comment's documented "disabled" value), cooldown_rounds=0,
min_bankroll_bnb_to_bet=0.001. Sizing knobs (max_bet_fraction_of_bankroll,
absolute per-bet caps) are KEPT so "5/50 BNB scale" remains the real
deployable sizing — only the suppression gates are removed.

Artifacts -> var/strategy_review/monitor_runs/2026-06-17/standard_backtest/
  configs/<name>.toml, <name>/{trades.csv,summary.json,equity_curves.png},
  results.json + console digest.

Run:  cd <repo> && .venv/Scripts/python.exe research/standard_backtest_windows_2026_06_17.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pancakebot.constants import MAX_GAS_COST_BET_BNB  # noqa: E402

OUT = REPO / "var" / "strategy_review" / "monitor_runs" / "2026-06-17" / "standard_backtest"
CFG_DIR = OUT / "configs"
SRC_CFG = REPO / "config.toml"
BREAKEVEN_WR = 0.55  # ~pari-mutuel breakeven at ~1.84 gross multiple

WINDOWS = [
    ("last_2w", 486805, 490743, 3937),
    ("last_1w", 488773, 490743, 1971),
]
SCALES = [5.0, 50.0]


def make_config(dst: Path, *, epoch_start: int, epoch_end: int,
                initial_bankroll: float) -> None:
    """Section-aware copy of config.toml: bound the backtest window + scale,
    and disable the risk suppression gates. Only edits inside the target
    sections (so [dry] initial_bankroll_bnb is left alone)."""
    section = None
    out_lines = []
    for raw in SRC_CFG.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s
        line = raw
        if section == "[backtest]":
            if s.startswith("initial_bankroll_bnb"):
                line = f"initial_bankroll_bnb = {initial_bankroll}"
            elif s.startswith("# epoch_start") or s.startswith("epoch_start"):
                line = f"epoch_start = {epoch_start}"
            elif s.startswith("# epoch_end") or s.startswith("epoch_end"):
                line = f"epoch_end = {epoch_end}"
        elif section == "[strategy.risk]":
            if s.startswith("max_drawdown_fraction_from_peak"):
                line = "max_drawdown_fraction_from_peak = 1.0   # DISABLED (breaker off)"
            elif s.startswith("min_bankroll_bnb_to_bet"):
                line = "min_bankroll_bnb_to_bet = 0.001          # no low-bankroll pause"
            elif s.startswith("cooldown_rounds"):
                line = "cooldown_rounds = 0                      # no cooldown"
        out_lines.append(line)
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def parse_run(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    # avg realized payout multiple on winning bets, from trades.csv
    win_mults, all_bet_mults = [], []
    with open(run_dir / "trades.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["action"] != "BET":
                continue
            bet = float(row["bet_size_bnb"])
            profit = float(row["profit_bnb"])
            if bet <= 0:
                continue
            # credit = profit + bet + gas ; payout_mult = credit/bet
            mult = (profit + bet + MAX_GAS_COST_BET_BNB) / bet
            all_bet_mults.append(mult)
            if profit > 0:
                win_mults.append(mult)
    summary["avg_payout_mult_on_wins"] = (
        round(sum(win_mults) / len(win_mults), 4) if win_mults else None)
    summary["n_skip_gate_no_signal"] = summary["skip_counts_by_reason"].get(
        "gate_no_signal", 0)
    summary["n_skip_risk"] = sum(
        v for k, v in summary["skip_counts_by_reason"].items() if k.startswith("risk_"))
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}
    for wname, e0, e1, n_rounds in WINDOWS:
        for scale in SCALES:
            name = f"{wname}_{int(scale)}bnb"
            cfg = CFG_DIR / f"{name}.toml"
            make_config(cfg, epoch_start=e0, epoch_end=e1, initial_bankroll=scale)
            print(f"--- {name}: run.py --backtest --config {cfg.name} ---", flush=True)
            r = subprocess.run(
                [sys.executable, str(REPO / "run.py"), "--backtest",
                 "--config", str(cfg)],
                cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAILED rc={r.returncode}\n{r.stderr[-1500:]}", flush=True)
                results[name] = {"error": r.stderr[-1500:]}
                continue
            run_dir = OUT / name
            run_dir.mkdir(exist_ok=True)
            for fn in ("trades.csv", "summary.json", "equity_curves.png"):
                src = REPO / "var" / "backtest" / fn
                if src.exists():
                    shutil.copy2(src, run_dir / fn)
            summ = parse_run(run_dir)
            summ["window"] = wname
            summ["scale_bnb"] = scale
            summ["window_rounds_expected"] = n_rounds
            results[name] = summ
            wr = summ["win_rate"]
            print(f"  rounds={summ['backtest_round_count']} bets={summ['num_bets']} "
                  f"WR={wr:.4f} pnl={summ['net_pnl_bnb']:+.4f} "
                  f"payout={summ['avg_payout_mult_on_wins']} "
                  f"risk_skips={summ['n_skip_risk']}", flush=True)

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== STANDARD BACKTEST (risk breaker OFF) — recent windows ===")
    print(f"{'run':>16} {'rounds':>6} {'fires':>5} {'WR':>7} {'vs55%':>7} "
          f"{'PnL':>9} {'payout':>7} {'gateNo':>6} {'riskSk':>6}")
    for name, s in results.items():
        if "error" in s:
            print(f"{name:>16}  ERROR"); continue
        wr = s["win_rate"]
        print(f"{name:>16} {s['backtest_round_count']:>6} {s['num_bets']:>5} "
              f"{wr:>7.4f} {wr-BREAKEVEN_WR:>+7.4f} {s['net_pnl_bnb']:>+9.4f} "
              f"{str(s['avg_payout_mult_on_wins']):>7} "
              f"{s['n_skip_gate_no_signal']:>6} {s['n_skip_risk']:>6}")
    print(f"\n[done] {time.time()-t0:.0f}s; artifacts -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
