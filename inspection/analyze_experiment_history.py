from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspection.backtest_harness_common import render_table, resolve_exp_root


@dataclass(frozen=True, slots=True)
class RunRow:
    name: str
    summary_path: str
    trades_path: str | None
    sim_size: int
    reset_mode: str
    router_mode: str
    ml_candidate_enabled: bool
    initial_bankroll_bnb: float
    final_bankroll_bnb: float
    net_profit_bnb: float
    per_500_bnb: float
    num_bets: int
    bet_rate: float
    max_drawdown_bnb: float | None
    min_bankroll_bnb: float | None
    top_skip_reason: str


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--min-sim-size", type=int, default=0)
    p.add_argument("--long-thresholds", type=str, default="5000,10000,30000")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--name-prefix", type=str, default="history_audit_20260303")
    return p


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sim_size_from_summary(summary: dict[str, Any], *, trades_path: Path | None) -> int:
    n = _safe_int(summary.get("num_rounds"), default=0)
    if int(n) > 0:
        return int(n)

    scenario = summary.get("scenario")
    if isinstance(scenario, dict):
        n2 = _safe_int(scenario.get("sim_size"), default=0)
        if int(n2) > 0:
            return int(n2)

    if trades_path is not None and trades_path.exists():
        with trades_path.open("r", newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in csv.DictReader(f)))
    return 0


def _bankroll_metrics_from_trades(trades_path: Path | None) -> tuple[float | None, float | None]:
    if trades_path is None or not trades_path.exists():
        return None, None

    min_b = None
    peak = None
    max_dd = 0.0

    with trades_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = _safe_float(row.get("bankroll_bnb"), default=0.0)
            if min_b is None or float(b) < float(min_b):
                min_b = float(b)
            if peak is None or float(b) > float(peak):
                peak = float(b)
            dd = float(peak) - float(b)
            if float(dd) > float(max_dd):
                max_dd = float(dd)

    if min_b is None:
        return None, None
    return float(max_dd), float(min_b)


def _top_skip_reason(summary: dict[str, Any]) -> str:
    raw = summary.get("num_skips_by_reason")
    if not isinstance(raw, dict) or not raw:
        return ""
    rows = sorted(((str(k), _safe_int(v)) for k, v in raw.items()), key=lambda x: (-int(x[1]), str(x[0])))
    return f"{rows[0][0]}:{rows[0][1]}"


def _load_rows(*, exp_root: Path, min_sim_size: int) -> list[RunRow]:
    out: list[RunRow] = []
    for summary_path in exp_root.rglob("backtest_summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_dir = summary_path.parent
        trades_path = run_dir / "backtest_trades.csv"
        if not trades_path.exists():
            trades_path = None

        sim_size = _sim_size_from_summary(summary, trades_path=trades_path)
        if int(sim_size) < int(min_sim_size):
            continue

        net = _safe_float(summary.get("net_profit_bnb"), default=0.0)
        per500 = float(net) * 500.0 / float(sim_size) if int(sim_size) > 0 else 0.0
        max_dd_summary = summary.get("risk", {}).get("max_drawdown_bnb") if isinstance(summary.get("risk"), dict) else None
        max_dd = _safe_float(max_dd_summary, default=float("nan")) if max_dd_summary is not None else None
        min_b = None
        if max_dd is None:
            max_dd, min_b = _bankroll_metrics_from_trades(trades_path=trades_path)
        else:
            _, min_b = _bankroll_metrics_from_trades(trades_path=trades_path)

        row = RunRow(
            name=str(run_dir.name),
            summary_path=str(summary_path),
            trades_path=(str(trades_path) if trades_path is not None else None),
            sim_size=int(sim_size),
            reset_mode=str(summary.get("reset_mode", "")),
            router_mode=str(summary.get("router_mode", "")),
            ml_candidate_enabled=bool(summary.get("ml_candidate_enabled", False)),
            initial_bankroll_bnb=_safe_float(summary.get("initial_bankroll_bnb"), default=0.0),
            final_bankroll_bnb=_safe_float(summary.get("final_bankroll_bnb"), default=0.0),
            net_profit_bnb=float(net),
            per_500_bnb=float(per500),
            num_bets=_safe_int(summary.get("num_bets"), default=0),
            bet_rate=_safe_float(summary.get("bet_rate"), default=0.0),
            max_drawdown_bnb=max_dd,
            min_bankroll_bnb=min_b,
            top_skip_reason=_top_skip_reason(summary),
        )
        out.append(row)
    return out


def _rows_table(*, rows: list[RunRow], top_n: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for r in rows[: int(top_n)]:
        out.append(
            {
                "name": str(r.name),
                "sim": int(r.sim_size),
                "per500": f"{float(r.per_500_bnb):+.6f}",
                "net": f"{float(r.net_profit_bnb):+.6f}",
                "max_dd": ("" if r.max_drawdown_bnb is None else f"{float(r.max_drawdown_bnb):.6f}"),
                "min_bankroll": ("" if r.min_bankroll_bnb is None else f"{float(r.min_bankroll_bnb):.6f}"),
                "bets": int(r.num_bets),
                "router": str(r.router_mode),
                "ml": bool(r.ml_candidate_enabled),
                "skip": str(r.top_skip_reason),
            }
        )
    return out


def _coverage_by_mode(rows: list[RunRow]) -> dict[str, dict[str, int]]:
    bins = {
        "lt_2k": lambda n: int(n) < 2000,
        "2k_to_5k": lambda n: 2000 <= int(n) < 5000,
        "5k_to_10k": lambda n: 5000 <= int(n) < 10000,
        "10k_to_30k": lambda n: 10000 <= int(n) < 30000,
        "ge_30k": lambda n: int(n) >= 30000,
    }
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        mode = str(r.router_mode or "unknown")
        if mode not in out:
            out[mode] = {k: 0 for k in bins}
        for k, fn in bins.items():
            if fn(int(r.sim_size)):
                out[mode][k] = int(out[mode][k]) + 1
                break
    return out


def main() -> None:
    args = _build_parser().parse_args()
    exp_root = resolve_exp_root()

    long_thresholds = [
        int(x.strip())
        for x in str(args.long_thresholds).split(",")
        if str(x).strip() != ""
    ]
    long_thresholds = sorted(set(long_thresholds))
    if not long_thresholds:
        raise RuntimeError("long_thresholds_empty")

    rows = _load_rows(exp_root=exp_root, min_sim_size=int(args.min_sim_size))
    if not rows:
        raise RuntimeError("no_rows_loaded")

    rows_sorted = sorted(rows, key=lambda r: float(r.per_500_bnb), reverse=True)
    by_threshold: dict[str, dict[str, object]] = {}
    for thr in long_thresholds:
        subset = [r for r in rows_sorted if int(r.sim_size) >= int(thr)]
        subset_viable = [
            r
            for r in subset
            if r.min_bankroll_bnb is not None and float(r.min_bankroll_bnb) >= 2.0
        ]
        by_threshold[str(thr)] = {
            "count": int(len(subset)),
            "top_rows": _rows_table(rows=subset, top_n=int(args.top_n)),
            "top_rows_min_bankroll_ge_2": _rows_table(rows=subset_viable, top_n=int(args.top_n)),
        }

    coverage = _coverage_by_mode(rows)
    coverage_rows = []
    for mode, bins in sorted(coverage.items(), key=lambda x: str(x[0])):
        coverage_rows.append(
            {
                "mode": str(mode),
                "lt_2k": int(bins["lt_2k"]),
                "2k_5k": int(bins["2k_to_5k"]),
                "5k_10k": int(bins["5k_to_10k"]),
                "10k_30k": int(bins["10k_to_30k"]),
                "ge_30k": int(bins["ge_30k"]),
            }
        )

    payload = {
        "exp_root": str(exp_root),
        "rows_total": int(len(rows)),
        "thresholds": [int(x) for x in long_thresholds],
        "by_threshold": by_threshold,
        "coverage_by_router_mode": coverage,
    }

    out_json = exp_root / f"{str(args.name_prefix)}.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"AUDIT_JSON={out_json}")

    print(
        render_table(
            columns=[
                ("mode", "mode"),
                ("lt_2k", "<2k"),
                ("2k_5k", "2k-5k"),
                ("5k_10k", "5k-10k"),
                ("10k_30k", "10k-30k"),
                ("ge_30k", ">=30k"),
            ],
            rows=coverage_rows,
        )
    )

    for thr in long_thresholds:
        subset = [r for r in rows_sorted if int(r.sim_size) >= int(thr)]
        print(f"THR>={int(thr)} count={len(subset)}")
        for r in subset[: min(int(args.top_n), 10)]:
            print(
                f"  {r.name} sim={int(r.sim_size)} per500={float(r.per_500_bnb):+.6f} "
                f"net={float(r.net_profit_bnb):+.6f} dd={(None if r.max_drawdown_bnb is None else round(float(r.max_drawdown_bnb), 6))} "
                f"min_bankroll={(None if r.min_bankroll_bnb is None else round(float(r.min_bankroll_bnb), 6))}"
            )


if __name__ == "__main__":
    main()
