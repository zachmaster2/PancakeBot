from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, replace
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


@dataclass(frozen=True, slots=True)
class IdeaVariant:
    name: str
    router_mode: str
    expected_net_scale: float = 1.0
    cutoff_pool_scale: float = 1.0
    fixed_bet_scale: float = 1.0
    temperature_scale: float = 1.0
    force_side_mode: str | None = None
    force_flow_gate_mode: str | None = None
    force_stake_mode: str | None = None
    force_pool_gate_mode: str | None = None
    projected_final_pool_multiplier: float | None = None
    projected_final_pool_total_min_bnb: float | None = None
    perf_adapt_mode: str | None = None
    perf_gate_window: int | None = None
    perf_gate_min_history: int | None = None
    perf_gate_min_win_rate: float | None = None
    perf_gate_min_mean_profit_bnb: float | None = None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--name-prefix", type=str, default="long_window_idea_sweep_20260303")
    p.add_argument("--sim-size", type=int, default=30000)
    p.add_argument("--offsets", type=str, default="0,5000,10000,15000,20000")
    p.add_argument("--drawdown-cap-bnb", type=float, default=2.0)
    p.add_argument("--top-skip-limit", type=int, default=4)
    p.add_argument("--top-selected-limit", type=int, default=4)
    p.add_argument("--run-long-confirm", action="store_true")
    p.add_argument("--long-sim-size", type=int, default=50984)
    p.add_argument("--top-k-confirm", type=int, default=4)
    p.add_argument("--initial-bankroll-bnb", type=float, default=None)
    return p


def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        try:
            vals.append(int(token))
        except ValueError as e:
            raise InvariantError(f"idea_sweep_int_list_invalid: {token}") from e
    vals = sorted(set(vals))
    if not vals:
        raise InvariantError("idea_sweep_offsets_empty")
    return vals


def _count_jsonl_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as f:
        return int(sum(1 for _ in f))


def _selected_mix(*, trades_csv_path: Path, limit: int) -> str:
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


def _min_bankroll_bnb(*, trades_csv_path: Path) -> float:
    min_bankroll: float | None = None
    with Path(trades_csv_path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = float(row["bankroll_bnb"])
            if min_bankroll is None or float(b) < float(min_bankroll):
                min_bankroll = float(b)
    return 0.0 if min_bankroll is None else float(min_bankroll)


def _default_idea_variants() -> list[IdeaVariant]:
    return [
        IdeaVariant(name="core_selector", router_mode="selector_max_score"),
        IdeaVariant(name="core_online", router_mode="online_cellmean"),
        IdeaVariant(
            name="fixed75_selector",
            router_mode="selector_max_score",
            fixed_bet_scale=0.75,
        ),
        IdeaVariant(
            name="fixed50_selector",
            router_mode="selector_max_score",
            fixed_bet_scale=0.5,
        ),
        IdeaVariant(
            name="fixed75_ev_optimal_selector",
            router_mode="selector_max_score",
            fixed_bet_scale=0.75,
            force_stake_mode="ev_optimal",
        ),
        IdeaVariant(
            name="perf_strict_selector",
            router_mode="selector_max_score",
            perf_adapt_mode="skip",
            perf_gate_window=120,
            perf_gate_min_history=60,
            perf_gate_min_win_rate=0.58,
            perf_gate_min_mean_profit_bnb=0.001,
        ),
        IdeaVariant(
            name="perf_off_selector",
            router_mode="selector_max_score",
            perf_adapt_mode="off",
            perf_gate_window=0,
            perf_gate_min_history=0,
            perf_gate_min_win_rate=0.0,
            perf_gate_min_mean_profit_bnb=0.0,
        ),
        IdeaVariant(
            name="loose_gates_selector",
            router_mode="selector_max_score",
            expected_net_scale=0.5,
            cutoff_pool_scale=0.5,
        ),
        IdeaVariant(
            name="nowcast_side_selector",
            router_mode="selector_max_score",
            force_side_mode="nowcast",
            force_flow_gate_mode="off",
        ),
        IdeaVariant(
            name="evmax_side_selector",
            router_mode="selector_max_score",
            force_side_mode="ev_max",
            force_flow_gate_mode="off",
        ),
        IdeaVariant(
            name="nowcast_contra_selector",
            router_mode="selector_max_score",
            force_side_mode="nowcast_contra",
            force_flow_gate_mode="off",
        ),
        IdeaVariant(
            name="stake_ev_scaled_selector",
            router_mode="selector_max_score",
            force_stake_mode="ev_scaled",
        ),
        IdeaVariant(
            name="stake_ev_optimal_selector",
            router_mode="selector_max_score",
            force_stake_mode="ev_optimal",
        ),
        IdeaVariant(
            name="stake_ev_scaled_projected_selector",
            router_mode="selector_max_score",
            force_stake_mode="ev_scaled_projected",
        ),
        IdeaVariant(
            name="stake_ev_optimal_projected_selector",
            router_mode="selector_max_score",
            force_stake_mode="ev_optimal_projected",
        ),
        IdeaVariant(
            name="stake_ev_optimal_projected_lb50_selector",
            router_mode="selector_max_score",
            force_stake_mode="ev_optimal_projected",
            projected_final_pool_multiplier=0.5,
        ),
        IdeaVariant(
            name="low_temperature_selector",
            router_mode="selector_max_score",
            temperature_scale=0.5,
        ),
        IdeaVariant(
            name="high_temperature_selector",
            router_mode="selector_max_score",
            temperature_scale=2.0,
        ),
    ]


def _mutate_candidate(*, c, variant: IdeaVariant):
    fixed_scale = float(variant.fixed_bet_scale)
    if float(fixed_scale) <= 0.0:
        raise InvariantError("idea_sweep_fixed_bet_scale_nonpositive")
    expected_scale = float(variant.expected_net_scale)
    if float(expected_scale) <= 0.0:
        raise InvariantError("idea_sweep_expected_net_scale_nonpositive")
    cutoff_scale = float(variant.cutoff_pool_scale)
    if float(cutoff_scale) <= 0.0:
        raise InvariantError("idea_sweep_cutoff_pool_scale_nonpositive")
    temperature_scale = float(variant.temperature_scale)
    if float(temperature_scale) <= 0.0:
        raise InvariantError("idea_sweep_temperature_scale_nonpositive")

    out = replace(
        c,
        fixed_bet_bnb=float(c.fixed_bet_bnb) * float(fixed_scale),
        expected_net_min_bnb=float(c.expected_net_min_bnb) * float(expected_scale),
        cutoff_pool_total_min_bnb=float(c.cutoff_pool_total_min_bnb) * float(cutoff_scale),
        temperature_bps=float(c.temperature_bps) * float(temperature_scale),
        stake_min_bnb=float(c.stake_min_bnb) * float(fixed_scale),
        stake_max_bnb=float(c.stake_max_bnb) * float(fixed_scale),
        stake_ev_ref_bnb=float(c.stake_ev_ref_bnb) * float(fixed_scale),
    )

    if variant.force_side_mode is not None:
        out = replace(out, side_selection_mode=str(variant.force_side_mode))
    if variant.force_flow_gate_mode is not None:
        out = replace(out, flow_gate_mode=str(variant.force_flow_gate_mode))
    if variant.force_stake_mode is not None:
        out = replace(out, stake_mode=str(variant.force_stake_mode))
    if variant.projected_final_pool_multiplier is not None:
        out = replace(
            out,
            projected_final_pool_multiplier=float(variant.projected_final_pool_multiplier),
        )
    if variant.projected_final_pool_total_min_bnb is not None:
        out = replace(
            out,
            projected_final_pool_total_min_bnb=float(variant.projected_final_pool_total_min_bnb),
        )
    if variant.force_pool_gate_mode is not None:
        mode = str(variant.force_pool_gate_mode)
        multiplier = (
            float(out.projected_final_pool_multiplier)
            if variant.projected_final_pool_multiplier is None
            else float(variant.projected_final_pool_multiplier)
        )
        projected_min = (
            float(out.projected_final_pool_total_min_bnb)
            if variant.projected_final_pool_total_min_bnb is None
            else float(variant.projected_final_pool_total_min_bnb)
        )
        out = replace(
            out,
            pool_total_gate_mode=str(mode),
            projected_final_pool_multiplier=float(multiplier),
            projected_final_pool_total_min_bnb=float(projected_min),
        )

    if variant.perf_adapt_mode is not None:
        out = replace(out, perf_adapt_mode=str(variant.perf_adapt_mode))
    if variant.perf_gate_window is not None:
        out = replace(out, perf_gate_window=int(variant.perf_gate_window))
    if variant.perf_gate_min_history is not None:
        out = replace(out, perf_gate_min_history=int(variant.perf_gate_min_history))
    if variant.perf_gate_min_win_rate is not None:
        out = replace(out, perf_gate_min_win_rate=float(variant.perf_gate_min_win_rate))
    if variant.perf_gate_min_mean_profit_bnb is not None:
        out = replace(
            out,
            perf_gate_min_mean_profit_bnb=float(variant.perf_gate_min_mean_profit_bnb),
        )

    return out


def _strategy_for_variant(*, cfg, variant: IdeaVariant):
    candidates = tuple(_mutate_candidate(c=c, variant=variant) for c in cfg.strategy.dislocation.candidates)
    dislocation_cfg = replace(cfg.strategy.dislocation, candidates=candidates)
    router_cfg = replace(cfg.strategy.router, mode=str(variant.router_mode))
    return replace(cfg.strategy, dislocation=dislocation_cfg, router=router_cfg)


def _aggregate_rows(
    *,
    rows: list[dict[str, object]],
    drawdown_cap_bnb: float,
) -> list[dict[str, object]]:
    by_variant: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    out: list[dict[str, object]] = []
    for variant, variant_rows in by_variant.items():
        per500 = [float(r["per_500"]) for r in variant_rows]
        net = [float(r["net_profit_bnb"]) for r in variant_rows]
        max_dd = [float(r["max_drawdown_bnb"]) for r in variant_rows]
        loss_to_min = [float(r["loss_from_initial_to_min_bnb"]) for r in variant_rows]
        bets = [int(r["num_bets"]) for r in variant_rows]
        bet_rates = [float(r["bet_rate"]) for r in variant_rows]

        floor_pass_count = int(sum(1 for x in loss_to_min if float(x) <= float(drawdown_cap_bnb)))
        positive_count = int(sum(1 for x in per500 if float(x) > 0.0))
        n = int(len(variant_rows))

        out.append(
            {
                "variant": str(variant),
                "n_windows": int(n),
                "mean_per_500": float(statistics.mean(per500)),
                "median_per_500": float(statistics.median(per500)),
                "worst_per_500": float(min(per500)),
                "best_per_500": float(max(per500)),
                "mean_net_profit_bnb": float(statistics.mean(net)),
                "worst_net_profit_bnb": float(min(net)),
                "worst_max_drawdown_bnb": float(max(max_dd)),
                "worst_loss_from_initial_to_min_bnb": float(max(loss_to_min)),
                "floor_pass_count": int(floor_pass_count),
                "positive_count": int(positive_count),
                "mean_num_bets": float(statistics.mean(bets)),
                "mean_bet_rate": float(statistics.mean(bet_rates)),
                "meets_floor_all_windows": bool(int(floor_pass_count) == int(n)),
            }
        )

    out_sorted = sorted(
        out,
        key=lambda r: (
            -int(r["floor_pass_count"]),
            -float(r["worst_per_500"]),
            -float(r["mean_per_500"]),
            float(r["worst_max_drawdown_bnb"]),
        ),
    )
    return out_sorted


def _long_confirm_targets(*, agg_rows: list[dict[str, object]], k: int) -> list[str]:
    if int(k) <= 0:
        return []
    out: list[str] = []
    for r in agg_rows:
        if float(r["mean_per_500"]) <= 0.0:
            continue
        out.append(str(r["variant"]))
        if len(out) >= int(k):
            break
    return out


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    sim_size = int(args.sim_size)
    if int(sim_size) <= 0:
        raise InvariantError("idea_sweep_sim_size_nonpositive")
    offsets = _parse_int_list(str(args.offsets))
    if int(args.top_skip_limit) <= 0 or int(args.top_selected_limit) <= 0:
        raise InvariantError("idea_sweep_top_limits_nonpositive")
    if float(args.drawdown_cap_bnb) < 0.0:
        raise InvariantError("idea_sweep_drawdown_cap_negative")

    total_rounds = _count_jsonl_lines(Path(str(cfg.closed_rounds_path)))
    warmup_rounds = int(cfg.strategy.dislocation.selector.warmup_rounds)
    if int(warmup_rounds) <= 0:
        raise InvariantError("idea_sweep_selector_warmup_nonpositive")

    max_sim_size = int(total_rounds) - int(warmup_rounds)
    if int(max_sim_size) <= 0:
        raise InvariantError("idea_sweep_max_sim_size_nonpositive")
    if int(sim_size) > int(max_sim_size):
        raise InvariantError(
            f"idea_sweep_sim_size_exceeds_max: sim_size={int(sim_size)} max={int(max_sim_size)}"
        )

    variants = _default_idea_variants()

    run_rows: list[dict[str, object]] = []
    for offset in offsets:
        needed = int(warmup_rounds) + int(sim_size)
        end_idx = int(total_rounds) - int(offset)
        start_idx = int(end_idx) - int(needed)
        if int(start_idx) < 0:
            continue

        for variant in variants:
            strategy_cfg = _strategy_for_variant(cfg=cfg, variant=variant)
            run_name = (
                f"{str(args.name_prefix)}_off{int(offset)}_{str(variant.name)}_sim{int(sim_size)}"
            )
            result = run_backtest_case(
                cfg=cfg,
                strategy_cfg=strategy_cfg,
                name=run_name,
                simulation_size=int(sim_size),
                reset_mode="continuous",
                reset_every_rounds=0,
                tail_offset_rounds=int(offset),
                initial_bankroll_bnb=args.initial_bankroll_bnb,
                exp_root=exp_root,
            )
            summary = dict(result.summary)
            net = float(summary.get("net_profit_bnb", 0.0))
            per_500 = float(net) * 500.0 / float(sim_size)
            max_dd = float(max_drawdown_bnb(trades_csv_path=result.trades_path))
            min_bank = float(_min_bankroll_bnb(trades_csv_path=result.trades_path))
            initial_bank = float(summary.get("initial_bankroll_bnb", 0.0))
            loss_to_min = float(initial_bank) - float(min_bank)

            run_rows.append(
                {
                    "variant": str(variant.name),
                    "offset": int(offset),
                    "sim_size": int(sim_size),
                    "router_mode": str(strategy_cfg.router.mode),
                    "net_profit_bnb": float(net),
                    "per_500": float(per_500),
                    "num_bets": int(summary.get("num_bets", 0)),
                    "bet_rate": float(summary.get("bet_rate", 0.0)),
                    "max_drawdown_bnb": float(max_dd),
                    "min_bankroll_bnb": float(min_bank),
                    "initial_bankroll_bnb": float(initial_bank),
                    "loss_from_initial_to_min_bnb": float(loss_to_min),
                    "meets_floor_cap": bool(float(loss_to_min) <= float(args.drawdown_cap_bnb)),
                    "top_skip_reasons": top_skip_reasons(
                        summary=summary,
                        limit=int(args.top_skip_limit),
                    ),
                    "selected_strategy_mix": _selected_mix(
                        trades_csv_path=result.trades_path,
                        limit=int(args.top_selected_limit),
                    ),
                    "elapsed_seconds": float(result.elapsed_seconds),
                    "summary_path": str(result.summary_path),
                    "trades_path": str(result.trades_path),
                }
            )

    agg_rows = _aggregate_rows(rows=run_rows, drawdown_cap_bnb=float(args.drawdown_cap_bnb))

    table_rows = [
        {
            "variant": str(r["variant"]),
            "n_win": int(r["n_windows"]),
            "floor_ok": f"{int(r['floor_pass_count'])}/{int(r['n_windows'])}",
            "pos": f"{int(r['positive_count'])}/{int(r['n_windows'])}",
            "mean_p500": f"{float(r['mean_per_500']):+.6f}",
            "worst_p500": f"{float(r['worst_per_500']):+.6f}",
            "mean_net": f"{float(r['mean_net_profit_bnb']):+.6f}",
            "worst_loss2": f"{float(r['worst_loss_from_initial_to_min_bnb']):.6f}",
            "worst_dd": f"{float(r['worst_max_drawdown_bnb']):.6f}",
            "mean_bets": f"{float(r['mean_num_bets']):.1f}",
        }
        for r in agg_rows
    ]
    print(
        render_table(
            columns=[
                ("variant", "variant"),
                ("n_win", "n_win"),
                ("floor_ok", "floor_ok"),
                ("pos", "pos"),
                ("mean_p500", "mean_p500"),
                ("worst_p500", "worst_p500"),
                ("mean_net", "mean_net"),
                ("worst_loss2", "worst_loss2"),
                ("worst_dd", "worst_dd"),
                ("mean_bets", "mean_bets"),
            ],
            rows=table_rows,
        )
    )

    long_confirm_rows: list[dict[str, object]] = []
    if bool(args.run_long_confirm):
        long_sim_size = int(args.long_sim_size)
        if int(long_sim_size) <= 0:
            raise InvariantError("idea_sweep_long_sim_size_nonpositive")
        if int(long_sim_size) > int(max_sim_size):
            raise InvariantError(
                f"idea_sweep_long_sim_size_exceeds_max: long_sim_size={int(long_sim_size)} max={int(max_sim_size)}"
            )

        targets = _long_confirm_targets(agg_rows=agg_rows, k=int(args.top_k_confirm))
        variants_by_name = {str(v.name): v for v in variants}
        for name in targets:
            variant = variants_by_name[str(name)]
            strategy_cfg = _strategy_for_variant(cfg=cfg, variant=variant)
            run_name = f"{str(args.name_prefix)}_confirm_{str(name)}_sim{int(long_sim_size)}"
            result = run_backtest_case(
                cfg=cfg,
                strategy_cfg=strategy_cfg,
                name=run_name,
                simulation_size=int(long_sim_size),
                reset_mode="continuous",
                reset_every_rounds=0,
                initial_bankroll_bnb=args.initial_bankroll_bnb,
                exp_root=exp_root,
            )
            summary = dict(result.summary)
            net = float(summary.get("net_profit_bnb", 0.0))
            per_500 = float(net) * 500.0 / float(long_sim_size)
            max_dd = float(max_drawdown_bnb(trades_csv_path=result.trades_path))
            min_bank = float(_min_bankroll_bnb(trades_csv_path=result.trades_path))
            initial_bank = float(summary.get("initial_bankroll_bnb", 0.0))
            loss_to_min = float(initial_bank) - float(min_bank)
            long_confirm_rows.append(
                {
                    "variant": str(name),
                    "sim_size": int(long_sim_size),
                    "net_profit_bnb": float(net),
                    "per_500": float(per_500),
                    "num_bets": int(summary.get("num_bets", 0)),
                    "bet_rate": float(summary.get("bet_rate", 0.0)),
                    "max_drawdown_bnb": float(max_dd),
                    "min_bankroll_bnb": float(min_bank),
                    "loss_from_initial_to_min_bnb": float(loss_to_min),
                    "meets_floor_cap": bool(float(loss_to_min) <= float(args.drawdown_cap_bnb)),
                    "summary_path": str(result.summary_path),
                }
            )

        confirm_table = [
            {
                "variant": str(r["variant"]),
                "sim": int(r["sim_size"]),
                "per_500": f"{float(r['per_500']):+.6f}",
                "net": f"{float(r['net_profit_bnb']):+.6f}",
                "loss2": f"{float(r['loss_from_initial_to_min_bnb']):.6f}",
                "floor": str(bool(r["meets_floor_cap"])),
                "dd": f"{float(r['max_drawdown_bnb']):.6f}",
                "bets": int(r["num_bets"]),
            }
            for r in long_confirm_rows
        ]
        if confirm_table:
            print("")
            print(
                render_table(
                    columns=[
                        ("variant", "variant"),
                        ("sim", "sim"),
                        ("per_500", "per_500"),
                        ("net", "net"),
                        ("loss2", "loss2"),
                        ("floor", "floor"),
                        ("dd", "max_dd"),
                        ("bets", "bets"),
                    ],
                    rows=confirm_table,
                )
            )

    out_json = exp_root / f"{str(args.name_prefix)}.json"
    out_csv = exp_root / f"{str(args.name_prefix)}.csv"
    out_json.write_text(
        json.dumps(
            {
                "config_path": str(args.config),
                "sim_size": int(sim_size),
                "offsets": [int(x) for x in offsets],
                "drawdown_cap_bnb": float(args.drawdown_cap_bnb),
                "history_total_rounds": int(total_rounds),
                "selector_warmup_rounds": int(warmup_rounds),
                "max_sim_size_feasible": int(max_sim_size),
                "rows": run_rows,
                "aggregate": agg_rows,
                "long_confirm": long_confirm_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "offset",
                "sim_size",
                "router_mode",
                "net_profit_bnb",
                "per_500",
                "num_bets",
                "bet_rate",
                "max_drawdown_bnb",
                "min_bankroll_bnb",
                "initial_bankroll_bnb",
                "loss_from_initial_to_min_bnb",
                "meets_floor_cap",
                "top_skip_reasons",
                "selected_strategy_mix",
                "elapsed_seconds",
                "summary_path",
                "trades_path",
            ],
        )
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)

    print("")
    print(f"TABLE_JSON={out_json}")
    print(f"TABLE_CSV={out_csv}")


if __name__ == "__main__":
    main()
