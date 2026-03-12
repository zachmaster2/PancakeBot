from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from inspection.backtest_harness_common import (
    load_cfg,
    max_drawdown_bnb,
    render_table,
    resolve_exp_root,
    run_backtest_case,
    top_skip_reasons,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--name-prefix", type=str, default="promotion_window_eval")
    p.add_argument("--promote-win-min", type=int, default=3)
    p.add_argument("--promote-dd-nonworse-min", type=int, default=2)
    return p


def _count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return int(sum(1 for _ in f))


def _selected_mix(*, trades_path: Path, limit: int = 4) -> str:
    counts: dict[str, int] = {}
    with Path(trades_path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("action", "")).strip() != "BET":
                continue
            key = str(row.get("selected_strategy", "")).strip() or "unknown"
            counts[key] = int(counts.get(key, 0)) + 1
    rows = sorted(counts.items(), key=lambda x: (-int(x[1]), str(x[0])))
    return "; ".join(f"{k}:{v}" for k, v in rows[: int(limit)])


def _window_slices() -> list[dict[str, int | str]]:
    # "recent_2k" and "prev_2k" provide a minimal rolling horizon check.
    return [
        {"name": "recent_2k", "sim_n": 2000, "end_offset": 0},
        {"name": "prev_2k", "sim_n": 2000, "end_offset": 2000},
        {"name": "recent_5k", "sim_n": 5000, "end_offset": 0},
        {"name": "recent_10k", "sim_n": 10000, "end_offset": 0},
    ]


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    warmup_n = int(cfg.strategy.dislocation.selector.warmup_rounds)
    closed_rounds_path = Path(str(cfg.closed_rounds_path))
    total_rounds = _count_jsonl_lines(closed_rounds_path)

    windows = _window_slices()
    for w in windows:
        needed = int(warmup_n) + int(w["sim_n"]) + int(w["end_offset"])
        if int(needed) > int(total_rounds):
            raise RuntimeError(f"window_too_large_for_store: {w}")

    base_strategy = cfg.strategy
    cons_name = "disloc_cons_20260227_x80"
    cons = None
    for c in cfg.strategy.dislocation.candidates:
        if str(c.name) == str(cons_name):
            cons = c
            break
    if cons is None:
        raise RuntimeError("target_candidate_missing: disloc_cons_20260227_x80")
    ratio = float(cons.expected_net_min_bnb) / float(cons.fixed_bet_bnb)
    cons_tuned = replace(
        cons,
        fixed_bet_bnb=0.25,
        expected_net_min_bnb=float(ratio) * 0.25,
    )
    tuned_candidates = tuple(
        cons_tuned if str(c.name) == str(cons_name) else c
        for c in cfg.strategy.dislocation.candidates
    )
    tuned_dislocation = replace(cfg.strategy.dislocation, candidates=tuned_candidates)
    tuned_strategy = replace(cfg.strategy, dislocation=tuned_dislocation)

    variants = [
        ("baseline", base_strategy),
        ("candidate_tuned", tuned_strategy),
    ]

    rows: list[dict[str, object]] = []
    for w in windows:
        win_name = str(w["name"])
        sim_n = int(w["sim_n"])
        end_offset = int(w["end_offset"])

        for variant_name, strategy_cfg in variants:
            result = run_backtest_case(
                cfg=cfg,
                strategy_cfg=strategy_cfg,
                name=f"{str(args.name_prefix)}_{win_name}_{variant_name}",
                simulation_size=int(sim_n),
                reset_mode="continuous",
                reset_every_rounds=0,
                tail_offset_rounds=int(end_offset),
                exp_root=exp_root,
            )
            summary = result.summary
            net = float(summary["net_profit_bnb"])
            rows.append(
                {
                    "window": str(win_name),
                    "sim_n": int(sim_n),
                    "end_offset": int(end_offset),
                    "variant": str(variant_name),
                    "net_profit_bnb": float(net),
                    "per_500": float(net * 500.0 / float(sim_n)),
                    "max_drawdown_bnb": float(max_drawdown_bnb(trades_csv_path=result.trades_path)),
                    "num_bets": int(summary.get("num_bets", 0)),
                    "bet_rate": float(summary.get("bet_rate", 0.0)),
                    "top_skips": top_skip_reasons(summary=summary, limit=4),
                    "selected_mix": _selected_mix(trades_path=result.trades_path, limit=4),
                    "summary_path": str(result.summary_path),
                }
            )

    by_window: dict[str, dict[str, dict[str, object]]] = {}
    for r in rows:
        by_window.setdefault(str(r["window"]), {})
        by_window[str(r["window"])][str(r["variant"])] = dict(r)

    compare_rows: list[dict[str, object]] = []
    wins = 0
    dd_nonworse = 0
    for w in windows:
        win_name = str(w["name"])
        b = by_window[win_name]["baseline"]
        t = by_window[win_name]["candidate_tuned"]
        per500_delta = float(t["per_500"]) - float(b["per_500"])
        dd_delta = float(t["max_drawdown_bnb"]) - float(b["max_drawdown_bnb"])
        if float(per500_delta) > 0.0:
            wins += 1
        if float(dd_delta) <= 0.0:
            dd_nonworse += 1
        compare_rows.append(
            {
                "window": str(win_name),
                "baseline_per_500": float(b["per_500"]),
                "tuned_per_500": float(t["per_500"]),
                "delta_per_500": float(per500_delta),
                "baseline_max_dd": float(b["max_drawdown_bnb"]),
                "tuned_max_dd": float(t["max_drawdown_bnb"]),
                "delta_max_dd": float(dd_delta),
                "baseline_bets": int(b["num_bets"]),
                "tuned_bets": int(t["num_bets"]),
            }
        )

    promote = bool(
        int(wins) >= int(args.promote_win_min)
        and int(dd_nonworse) >= int(args.promote_dd_nonworse_min)
    )
    decision = {
        "promote": bool(promote),
        "wins_required": int(args.promote_win_min),
        "dd_nonworse_required": int(args.promote_dd_nonworse_min),
        "wins_observed": int(wins),
        "dd_nonworse_observed": int(dd_nonworse),
    }

    table_rows = [
        {
            "window": str(r["window"]),
            "base_per500": f"{float(r['baseline_per_500']):.6f}",
            "tuned_per500": f"{float(r['tuned_per_500']):.6f}",
            "d_per500": f"{float(r['delta_per_500']):+.6f}",
            "base_dd": f"{float(r['baseline_max_dd']):.6f}",
            "tuned_dd": f"{float(r['tuned_max_dd']):.6f}",
            "d_dd": f"{float(r['delta_max_dd']):+.6f}",
            "base_bets": int(r["baseline_bets"]),
            "tuned_bets": int(r["tuned_bets"]),
        }
        for r in compare_rows
    ]
    print(
        render_table(
            columns=[
                ("window", "window"),
                ("base_per500", "base_per500"),
                ("tuned_per500", "tuned_per500"),
                ("d_per500", "d_per500"),
                ("base_dd", "base_dd"),
                ("tuned_dd", "tuned_dd"),
                ("d_dd", "d_dd"),
                ("base_bets", "base_bets"),
                ("tuned_bets", "tuned_bets"),
            ],
            rows=table_rows,
        )
    )
    print(f"DECISION promote={bool(promote)} wins={int(wins)} dd_nonworse={int(dd_nonworse)}")

    out_json = exp_root / f"{str(args.name_prefix)}_table.json"
    out_json.write_text(
        json.dumps(
            {
                "rows": rows,
                "compare": compare_rows,
                "decision": decision,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"TABLE_JSON={out_json}")


if __name__ == "__main__":
    main()
