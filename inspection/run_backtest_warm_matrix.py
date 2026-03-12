from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pancakebot.core.errors import InvariantError

from inspection.backtest_harness_common import (
    max_drawdown_bnb,
    load_cfg,
    render_table,
    resolve_exp_root,
    run_backtest_case,
    top_skip_reasons,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, default="warm_matrix")
    parser.add_argument("--sim-size", type=int, default=500)
    parser.add_argument("--chunk-reset-intervals", type=str, default="20,40,80")
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    parser.add_argument("--top-skip-limit", type=int, default=3)
    return parser


def _parse_intervals(raw: str) -> list[int]:
    tokens = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not tokens:
        raise InvariantError("warm_matrix_chunk_reset_intervals_empty")
    out: list[int] = []
    for token in tokens:
        try:
            v = int(token)
        except ValueError as e:
            raise InvariantError(f"warm_matrix_chunk_reset_interval_not_int: {token}") from e
        if int(v) <= 0:
            raise InvariantError("warm_matrix_chunk_reset_interval_nonpositive")
        out.append(int(v))
    return out


def _mode_rows(*, chunk_intervals: list[int]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = [("continuous", 0)]
    out.extend(("chunk_reset", int(interval)) for interval in chunk_intervals)
    return out


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    sim_size = int(args.sim_size)
    if sim_size <= 0:
        raise InvariantError("warm_matrix_sim_size_nonpositive")

    top_skip_limit = int(args.top_skip_limit)
    if top_skip_limit <= 0:
        raise InvariantError("warm_matrix_top_skip_limit_nonpositive")

    chunk_intervals = _parse_intervals(str(args.chunk_reset_intervals))

    rows_raw: list[dict[str, object]] = []
    for mode, reset_every in _mode_rows(chunk_intervals=chunk_intervals):
        key = f"{mode}_r{int(reset_every)}"
        run_backtest_case(
            cfg=cfg,
            name=f"{str(args.name_prefix)}_{key}_prime",
            simulation_size=int(sim_size),
            reset_mode=str(mode),
            reset_every_rounds=int(reset_every),
            initial_bankroll_bnb=args.initial_bankroll_bnb,
            exp_root=exp_root,
        )
        warm = run_backtest_case(
            cfg=cfg,
            name=f"{str(args.name_prefix)}_{key}_warm",
            simulation_size=int(sim_size),
            reset_mode=str(mode),
            reset_every_rounds=int(reset_every),
            initial_bankroll_bnb=args.initial_bankroll_bnb,
            exp_root=exp_root,
        )

        summary = dict(warm.summary)
        net_profit_bnb = float(summary["net_profit_bnb"])
        rows_raw.append(
            {
                "mode": str(mode),
                "reset_interval": int(reset_every),
                "sim_size": int(sim_size),
                "net_profit_bnb": float(net_profit_bnb),
                "profit_per_500_rounds_bnb": float(net_profit_bnb * 500.0 / float(sim_size)),
                "max_drawdown_bnb": float(max_drawdown_bnb(trades_csv_path=warm.trades_path)),
                "num_bets": int(summary.get("num_bets", 0)),
                "top_skip_reasons": top_skip_reasons(summary=summary, limit=int(top_skip_limit)),
                "warm_elapsed_seconds": float(warm.elapsed_seconds),
                "summary_path": str(warm.summary_path),
                "trades_path": str(warm.trades_path),
            }
        )

    table_rows = [
        {
            "mode": str(row["mode"]),
            "reset_interval": int(row["reset_interval"]),
            "net_profit_bnb": f"{float(row['net_profit_bnb']):.6f}",
            "profit_per_500": f"{float(row['profit_per_500_rounds_bnb']):.6f}",
            "max_drawdown_bnb": f"{float(row['max_drawdown_bnb']):.6f}",
            "num_bets": int(row["num_bets"]),
            "top_skip_reasons": str(row["top_skip_reasons"]),
        }
        for row in rows_raw
    ]

    print(f"EXP_ROOT={exp_root}")
    print(
        render_table(
            columns=[
                ("mode", "mode"),
                ("reset_interval", "reset_interval"),
                ("net_profit_bnb", "net_profit_bnb"),
                ("profit_per_500", "profit_per_500"),
                ("max_drawdown_bnb", "max_drawdown_bnb"),
                ("num_bets", "num_bets"),
                ("top_skip_reasons", "top_skip_reasons"),
            ],
            rows=table_rows,
        )
    )

    json_path = exp_root / f"{str(args.name_prefix)}_table.json"
    csv_path = exp_root / f"{str(args.name_prefix)}_table.csv"
    json_path.write_text(json.dumps({"rows": rows_raw}, indent=2, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "reset_interval",
                "sim_size",
                "net_profit_bnb",
                "profit_per_500_rounds_bnb",
                "max_drawdown_bnb",
                "num_bets",
                "top_skip_reasons",
                "warm_elapsed_seconds",
                "summary_path",
                "trades_path",
            ],
        )
        writer.writeheader()
        for row in rows_raw:
            writer.writerow(row)

    print(f"TABLE_JSON={json_path}")
    print(f"TABLE_CSV={csv_path}")


if __name__ == "__main__":
    main()
