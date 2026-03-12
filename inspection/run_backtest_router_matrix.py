from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from pancakebot.core.errors import InvariantError

from inspection.backtest_harness_common import (
    load_cfg,
    max_drawdown_bnb,
    render_table,
    resolve_exp_root,
    run_backtest_case,
    top_skip_reasons,
)

_ROUTER_MODES = (
    "selector_max_score",
    "online_cellmean",
    "online_cellmean_side_gap",
    "online_cellmean_backoff",
    "online_cellmean_selector_fallback",
    "skip_only",
    "oracle_skip",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, default="router_matrix")
    parser.add_argument("--sim-size", type=int, default=500)
    parser.add_argument("--reset-mode", type=str, choices=("continuous", "chunk_reset"), default="continuous")
    parser.add_argument("--reset-every-rounds", type=int, default=0)
    parser.add_argument("--router-modes", type=str, default="selector_max_score,online_cellmean")
    parser.add_argument("--selector-score-thresholds", type=str, default="-1000000000.0,-0.01,0.0")
    parser.add_argument("--online-score-thresholds", type=str, default="0.0,0.001")
    parser.add_argument("--online-warmup-rounds", type=str, default="")
    parser.add_argument("--online-num-quantile-bins", type=str, default="")
    parser.add_argument("--online-min-cell-obs", type=str, default="")
    parser.add_argument("--online-use-direction-split-list", type=str, default="true,false")
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    parser.add_argument("--top-skip-limit", type=int, default=3)
    parser.add_argument("--top-selected-limit", type=int, default=4)
    return parser


def _parse_modes(raw: str) -> list[str]:
    out: list[str] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        if token not in _ROUTER_MODES:
            raise InvariantError(f"router_matrix_mode_invalid: {token}")
        out.append(str(token))
    if not out:
        raise InvariantError("router_matrix_modes_empty")
    return out


def _parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        try:
            vals.append(float(token))
        except ValueError as e:
            raise InvariantError(f"router_matrix_float_list_invalid: {token}") from e
    if not vals:
        raise InvariantError("router_matrix_float_list_empty")
    return vals


def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        try:
            vals.append(int(token))
        except ValueError as e:
            raise InvariantError(f"router_matrix_int_list_invalid: {token}") from e
    if not vals:
        raise InvariantError("router_matrix_int_list_empty")
    return vals


def _parse_bool_list(raw: str) -> list[bool]:
    vals: list[bool] = []
    for token in [x.strip().lower() for x in str(raw).split(",") if x.strip()]:
        if token in ("true", "t", "1", "yes", "y", "on"):
            vals.append(True)
        elif token in ("false", "f", "0", "no", "n", "off"):
            vals.append(False)
        else:
            raise InvariantError(f"router_matrix_bool_list_invalid: {token}")
    if not vals:
        raise InvariantError("router_matrix_bool_list_empty")
    return vals


def _selected_strategy_mix(*, trades_csv_path: Path, limit: int) -> str:
    counts: dict[str, int] = {}
    with Path(trades_csv_path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("action", "")).strip() != "BET":
                continue
            key = str(row.get("selected_strategy", "")).strip() or "unknown"
            counts[key] = int(counts.get(key, 0)) + 1
    if not counts:
        return ""
    rows = sorted(counts.items(), key=lambda x: (-int(x[1]), str(x[0])))
    return "; ".join(f"{k}:{v}" for k, v in rows[: int(limit)])


def _router_variants(*, cfg, args: argparse.Namespace) -> list[dict[str, object]]:
    modes = _parse_modes(str(args.router_modes))
    base_router = cfg.strategy.router

    selector_thresholds = _parse_float_list(str(args.selector_score_thresholds))
    online_score_thresholds = _parse_float_list(str(args.online_score_thresholds))
    online_warmup_rounds = (
        [int(base_router.online_warmup_rounds)]
        if str(args.online_warmup_rounds).strip() == ""
        else _parse_int_list(str(args.online_warmup_rounds))
    )
    online_num_bins = (
        [int(base_router.online_num_quantile_bins)]
        if str(args.online_num_quantile_bins).strip() == ""
        else _parse_int_list(str(args.online_num_quantile_bins))
    )
    online_min_obs = (
        [int(base_router.online_min_cell_obs)]
        if str(args.online_min_cell_obs).strip() == ""
        else _parse_int_list(str(args.online_min_cell_obs))
    )
    online_split = _parse_bool_list(str(args.online_use_direction_split_list))

    out: list[dict[str, object]] = []
    for mode in modes:
        if str(mode) == "selector_max_score":
            for threshold in selector_thresholds:
                out.append(
                    {
                        "mode": "selector_max_score",
                        "score_threshold_bnb": float(threshold),
                        "online_warmup_rounds": int(base_router.online_warmup_rounds),
                        "online_num_quantile_bins": int(base_router.online_num_quantile_bins),
                        "online_min_cell_obs": int(base_router.online_min_cell_obs),
                        "online_score_threshold_bnb": float(base_router.online_score_threshold_bnb),
                        "online_use_direction_split": bool(base_router.online_use_direction_split),
                    }
                )
            continue

        if str(mode) in (
            "online_cellmean",
            "online_cellmean_side_gap",
            "online_cellmean_backoff",
            "online_cellmean_selector_fallback",
        ):
            for warmup in online_warmup_rounds:
                for bins in online_num_bins:
                    for min_obs in online_min_obs:
                        for threshold in online_score_thresholds:
                            for split in online_split:
                                out.append(
                                    {
                                        "mode": str(mode),
                                        "score_threshold_bnb": float(base_router.score_threshold_bnb),
                                        "online_warmup_rounds": int(warmup),
                                        "online_num_quantile_bins": int(bins),
                                        "online_min_cell_obs": int(min_obs),
                                        "online_score_threshold_bnb": float(threshold),
                                        "online_use_direction_split": bool(split),
                                    }
                                )
            continue

        out.append(
            {
                "mode": str(mode),
                "score_threshold_bnb": float(base_router.score_threshold_bnb),
                "online_warmup_rounds": int(base_router.online_warmup_rounds),
                "online_num_quantile_bins": int(base_router.online_num_quantile_bins),
                "online_min_cell_obs": int(base_router.online_min_cell_obs),
                "online_score_threshold_bnb": float(base_router.online_score_threshold_bnb),
                "online_use_direction_split": bool(base_router.online_use_direction_split),
            }
        )

    if not out:
        raise InvariantError("router_matrix_variants_empty")
    return out


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    sim_size = int(args.sim_size)
    if sim_size <= 0:
        raise InvariantError("router_matrix_sim_size_nonpositive")
    top_skip_limit = int(args.top_skip_limit)
    top_selected_limit = int(args.top_selected_limit)
    if top_skip_limit <= 0 or top_selected_limit <= 0:
        raise InvariantError("router_matrix_top_limits_nonpositive")

    reset_mode = str(args.reset_mode)
    reset_every = int(args.reset_every_rounds)
    if reset_mode == "chunk_reset" and reset_every <= 0:
        raise InvariantError("router_matrix_chunk_reset_every_nonpositive")
    if reset_mode == "continuous":
        reset_every = 0

    variants = _router_variants(cfg=cfg, args=args)

    rows_raw: list[dict[str, object]] = []
    for i, variant in enumerate(variants, start=1):
        router_cfg = replace(
            cfg.strategy.router,
            mode=str(variant["mode"]),
            score_threshold_bnb=float(variant["score_threshold_bnb"]),
            online_warmup_rounds=int(variant["online_warmup_rounds"]),
            online_num_quantile_bins=int(variant["online_num_quantile_bins"]),
            online_min_cell_obs=int(variant["online_min_cell_obs"]),
            online_score_threshold_bnb=float(variant["online_score_threshold_bnb"]),
            online_use_direction_split=bool(variant["online_use_direction_split"]),
        )
        strategy_cfg = replace(cfg.strategy, router=router_cfg)

        key = f"v{i:03d}"
        run_backtest_case(
            cfg=cfg,
            strategy_cfg=strategy_cfg,
            name=f"{str(args.name_prefix)}_{key}_prime",
            simulation_size=int(sim_size),
            reset_mode=str(reset_mode),
            reset_every_rounds=int(reset_every),
            initial_bankroll_bnb=args.initial_bankroll_bnb,
            exp_root=exp_root,
        )
        warm = run_backtest_case(
            cfg=cfg,
            strategy_cfg=strategy_cfg,
            name=f"{str(args.name_prefix)}_{key}_warm",
            simulation_size=int(sim_size),
            reset_mode=str(reset_mode),
            reset_every_rounds=int(reset_every),
            initial_bankroll_bnb=args.initial_bankroll_bnb,
            exp_root=exp_root,
        )

        summary = dict(warm.summary)
        net_profit = float(summary["net_profit_bnb"])
        rows_raw.append(
            {
                "variant": str(key),
                "mode": str(router_cfg.mode),
                "score_threshold_bnb": float(router_cfg.score_threshold_bnb),
                "online_warmup_rounds": int(router_cfg.online_warmup_rounds),
                "online_num_quantile_bins": int(router_cfg.online_num_quantile_bins),
                "online_min_cell_obs": int(router_cfg.online_min_cell_obs),
                "online_score_threshold_bnb": float(router_cfg.online_score_threshold_bnb),
                "online_use_direction_split": bool(router_cfg.online_use_direction_split),
                "reset_mode": str(reset_mode),
                "reset_every_rounds": int(reset_every),
                "sim_size": int(sim_size),
                "net_profit_bnb": float(net_profit),
                "profit_per_500_rounds_bnb": float(net_profit * 500.0 / float(sim_size)),
                "max_drawdown_bnb": float(max_drawdown_bnb(trades_csv_path=warm.trades_path)),
                "num_bets": int(summary.get("num_bets", 0)),
                "top_skip_reasons": top_skip_reasons(summary=summary, limit=int(top_skip_limit)),
                "selected_strategy_mix": _selected_strategy_mix(
                    trades_csv_path=warm.trades_path,
                    limit=int(top_selected_limit),
                ),
                "warm_elapsed_seconds": float(warm.elapsed_seconds),
                "summary_path": str(warm.summary_path),
                "trades_path": str(warm.trades_path),
            }
        )

    rows_sorted = sorted(
        rows_raw,
        key=lambda x: (
            -float(x["profit_per_500_rounds_bnb"]),
            float(x["max_drawdown_bnb"]),
            -int(x["num_bets"]),
        ),
    )

    table_rows = [
        {
            "variant": str(r["variant"]),
            "mode": str(r["mode"]),
            "score_thr": f"{float(r['score_threshold_bnb']):.6f}",
            "online_thr": f"{float(r['online_score_threshold_bnb']):.6f}",
            "warmup": int(r["online_warmup_rounds"]),
            "bins": int(r["online_num_quantile_bins"]),
            "min_obs": int(r["online_min_cell_obs"]),
            "dir_split": str(bool(r["online_use_direction_split"])),
            "net": f"{float(r['net_profit_bnb']):.6f}",
            "per_500": f"{float(r['profit_per_500_rounds_bnb']):.6f}",
            "max_dd": f"{float(r['max_drawdown_bnb']):.6f}",
            "bets": int(r["num_bets"]),
            "skips": str(r["top_skip_reasons"]),
            "selected_mix": str(r["selected_strategy_mix"]),
            "warm_s": f"{float(r['warm_elapsed_seconds']):.3f}",
        }
        for r in rows_sorted
    ]

    print(f"EXP_ROOT={exp_root}")
    print(
        render_table(
            columns=[
                ("variant", "variant"),
                ("mode", "mode"),
                ("score_thr", "score_thr"),
                ("online_thr", "online_thr"),
                ("warmup", "warmup"),
                ("bins", "bins"),
                ("min_obs", "min_obs"),
                ("dir_split", "dir_split"),
                ("net", "net"),
                ("per_500", "per_500"),
                ("max_dd", "max_dd"),
                ("bets", "bets"),
                ("skips", "top_skip_reasons"),
                ("selected_mix", "selected_mix"),
                ("warm_s", "warm_s"),
            ],
            rows=table_rows,
        )
    )

    json_path = exp_root / f"{str(args.name_prefix)}_table.json"
    csv_path = exp_root / f"{str(args.name_prefix)}_table.csv"
    json_path.write_text(json.dumps({"rows": rows_sorted}, indent=2, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "mode",
                "score_threshold_bnb",
                "online_warmup_rounds",
                "online_num_quantile_bins",
                "online_min_cell_obs",
                "online_score_threshold_bnb",
                "online_use_direction_split",
                "reset_mode",
                "reset_every_rounds",
                "sim_size",
                "net_profit_bnb",
                "profit_per_500_rounds_bnb",
                "max_drawdown_bnb",
                "num_bets",
                "top_skip_reasons",
                "selected_strategy_mix",
                "warm_elapsed_seconds",
                "summary_path",
                "trades_path",
            ],
        )
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)

    print(f"TABLE_JSON={json_path}")
    print(f"TABLE_CSV={csv_path}")


if __name__ == "__main__":
    main()
