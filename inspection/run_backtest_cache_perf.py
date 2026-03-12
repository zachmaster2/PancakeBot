from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspection.backtest_harness_common import (
    clear_state_cache_dir,
    count_state_cache_files,
    load_cfg,
    render_table,
    resolve_exp_root,
    run_backtest_case,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, default="cache_perf")
    parser.add_argument("--sim-size-continuous", type=int, default=500)
    parser.add_argument("--sim-size-chunk-reset", type=int, default=500)
    parser.add_argument("--chunk-reset-every-rounds", type=int, default=20)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    return parser


def _run_mode_pair(
    *,
    cfg,
    exp_root: Path,
    name_prefix: str,
    reset_mode: str,
    reset_every_rounds: int,
    simulation_size: int,
    initial_bankroll_bnb: float | None,
) -> dict[str, object]:
    cache_root = Path(cfg.backtest_state_cache_dir)
    clear_state_cache_dir(state_cache_root=cache_root)
    pre_count = int(count_state_cache_files(state_cache_root=cache_root))

    cold = run_backtest_case(
        cfg=cfg,
        name=f"{name_prefix}_{reset_mode}_cold",
        simulation_size=int(simulation_size),
        reset_mode=str(reset_mode),
        reset_every_rounds=int(reset_every_rounds),
        initial_bankroll_bnb=initial_bankroll_bnb,
        exp_root=exp_root,
    )
    post_cold_count = int(count_state_cache_files(state_cache_root=cache_root))

    warm = run_backtest_case(
        cfg=cfg,
        name=f"{name_prefix}_{reset_mode}_warm",
        simulation_size=int(simulation_size),
        reset_mode=str(reset_mode),
        reset_every_rounds=int(reset_every_rounds),
        initial_bankroll_bnb=initial_bankroll_bnb,
        exp_root=exp_root,
    )
    post_warm_count = int(count_state_cache_files(state_cache_root=cache_root))

    cold_s = float(cold.elapsed_seconds)
    warm_s = float(warm.elapsed_seconds)
    speedup_x = float(cold_s / warm_s) if warm_s > 0.0 else float("nan")
    miss_confirmed = bool(pre_count == 0 and post_cold_count > 0)
    hit_confirmed = bool(post_warm_count > 0 and post_warm_count == post_cold_count)

    return {
        "mode": str(reset_mode),
        "reset_every_rounds": int(reset_every_rounds),
        "sim_size": int(simulation_size),
        "cold_seconds": float(cold_s),
        "warm_seconds": float(warm_s),
        "delta_seconds": float(cold_s - warm_s),
        "speedup_x": float(speedup_x),
        "cache_miss_confirmed": bool(miss_confirmed),
        "cache_hit_confirmed": bool(hit_confirmed),
        "cache_files_before": int(pre_count),
        "cache_files_after_cold": int(post_cold_count),
        "cache_files_after_warm": int(post_warm_count),
        "cold_summary": str(cold.summary_path),
        "warm_summary": str(warm.summary_path),
    }


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    rows_raw = [
        _run_mode_pair(
            cfg=cfg,
            exp_root=exp_root,
            name_prefix=str(args.name_prefix),
            reset_mode="continuous",
            reset_every_rounds=0,
            simulation_size=int(args.sim_size_continuous),
            initial_bankroll_bnb=args.initial_bankroll_bnb,
        ),
        _run_mode_pair(
            cfg=cfg,
            exp_root=exp_root,
            name_prefix=str(args.name_prefix),
            reset_mode="chunk_reset",
            reset_every_rounds=int(args.chunk_reset_every_rounds),
            simulation_size=int(args.sim_size_chunk_reset),
            initial_bankroll_bnb=args.initial_bankroll_bnb,
        ),
    ]

    table_rows: list[dict[str, object]] = []
    for row in rows_raw:
        table_rows.append(
            {
                "mode": str(row["mode"]),
                "reset": int(row["reset_every_rounds"]),
                "sim": int(row["sim_size"]),
                "cold_s": f"{float(row['cold_seconds']):.3f}",
                "warm_s": f"{float(row['warm_seconds']):.3f}",
                "delta_s": f"{float(row['delta_seconds']):.3f}",
                "speedup_x": f"{float(row['speedup_x']):.2f}",
                "miss": str(bool(row["cache_miss_confirmed"])),
                "hit": str(bool(row["cache_hit_confirmed"])),
                "cache_files": int(row["cache_files_after_warm"]),
            }
        )

    print(f"EXP_ROOT={exp_root}")
    print(
        render_table(
            columns=[
                ("mode", "mode"),
                ("reset", "reset"),
                ("sim", "sim"),
                ("cold_s", "cold_s"),
                ("warm_s", "warm_s"),
                ("delta_s", "delta_s"),
                ("speedup_x", "speedup_x"),
                ("miss", "miss"),
                ("hit", "hit"),
                ("cache_files", "cache_files"),
            ],
            rows=table_rows,
        )
    )

    summary_path = exp_root / f"{str(args.name_prefix)}_summary.json"
    summary_path.write_text(json.dumps({"rows": rows_raw}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main()
