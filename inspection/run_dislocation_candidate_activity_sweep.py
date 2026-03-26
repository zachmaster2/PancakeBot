from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pancakebot.core.errors import InvariantError

from inspection.backtest_harness_common import (
    load_cfg,
    max_drawdown_bnb,
    render_table,
    resolve_exp_root,
    run_backtest_case,
    top_skip_reasons,
)
from inspection.run_alta_single_idea import (
    _candidate_pool,
    _maybe_load_existing_summary,
    _min_bankroll_bnb,
    _parse_set_overrides,
    _selected_mix,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--candidate-name", type=str, required=True)
    parser.add_argument("--candidate-source", choices=("active", "all_config"), default="all_config")
    parser.add_argument("--sim-size", type=int, default=50984)
    parser.add_argument("--tail-offset-rounds", type=int, default=0)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    parser.add_argument("--router-mode", type=str, default="selector_max_score")
    parser.add_argument("--router-score-threshold-bnb", type=float, default=-1000000000.0)
    parser.add_argument("--min-bet-rate", type=float, default=0.05)
    parser.add_argument("--target-bet-rate", type=float, default=0.075)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-skip-limit", type=int, default=4)
    parser.add_argument("--top-selected-limit", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--dislocation-threshold-pps", type=str, default="")
    parser.add_argument("--expected-net-min-bnbs", type=str, default="")
    parser.add_argument("--cutoff-pool-total-min-bnbs", type=str, default="")
    parser.add_argument("--fixed-bet-bnbs", type=str, default="")
    parser.add_argument("--lookback1-seconds-list", type=str, default="")
    parser.add_argument("--temperature-bps-list", type=str, default="")
    parser.add_argument("--allowed-sides-list", type=str, default="")
    parser.add_argument("--market-extreme-mins", type=str, default="")
    parser.add_argument("--nowcast-market-gap-mins", type=str, default="")
    parser.add_argument("--flow-window-seconds-list", type=str, default="")
    parser.add_argument("--flow-min-imbalances", type=str, default="")
    parser.add_argument("--flow-gate-modes", type=str, default="")
    parser.add_argument("--pool-total-gate-modes", type=str, default="")
    parser.add_argument("--projected-final-pool-total-min-bnbs", type=str, default="")
    parser.add_argument("--projected-final-pool-multipliers", type=str, default="")
    parser.add_argument("--bull-expected-net-extra-min-bnbs", type=str, default="")
    parser.add_argument("--bear-expected-net-extra-min-bnbs", type=str, default="")
    parser.add_argument("--side-selection-modes", type=str, default="")
    parser.add_argument("--perf-adapt-modes", type=str, default="")
    parser.add_argument("--perf-gate-windows-list", type=str, default="")
    parser.add_argument("--perf-gate-min-history-list", type=str, default="")
    parser.add_argument("--perf-gate-min-win-rates", type=str, default="")
    parser.add_argument("--perf-gate-min-mean-profit-bnbs", type=str, default="")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true", default=False)
    return parser


def _validate_partition_args(*, chunk_size: int, chunk_index: int) -> None:
    if int(chunk_size) < 0:
        raise InvariantError("candidate_activity_sweep_chunk_size_negative")
    if int(chunk_index) < 0:
        raise InvariantError("candidate_activity_sweep_chunk_index_negative")
    if int(chunk_size) == 0 and int(chunk_index) != 0:
        raise InvariantError("candidate_activity_sweep_chunk_index_requires_chunk_size")


def _select_override_payloads(
    *,
    override_payloads: list[dict[str, Any]],
    chunk_size: int,
    chunk_index: int,
) -> list[dict[str, Any]]:
    selected = list(override_payloads)
    if int(chunk_size) > 0:
        start = int(chunk_index) * int(chunk_size)
        stop = min(len(selected), start + int(chunk_size))
        if start >= len(selected):
            return []
        return selected[start:stop]
    return selected


def _parse_float_list_or_empty(raw: str) -> list[float]:
    values: list[float] = []
    for token in [str(x).strip() for x in str(raw).split(",") if str(x).strip() != ""]:
        try:
            values.append(float(token))
        except ValueError as exc:
            raise InvariantError(f"candidate_activity_sweep_float_list_invalid: {token}") from exc
    return list(dict.fromkeys(values))


def _parse_int_list_or_empty(raw: str) -> list[int]:
    values: list[int] = []
    for token in [str(x).strip() for x in str(raw).split(",") if str(x).strip() != ""]:
        try:
            values.append(int(token))
        except ValueError as exc:
            raise InvariantError(f"candidate_activity_sweep_int_list_invalid: {token}") from exc
    return list(dict.fromkeys(values))


def _parse_str_list_or_empty(raw: str) -> list[str]:
    return list(dict.fromkeys([str(x).strip() for x in str(raw).split(",") if str(x).strip() != ""]))


def _slug_float(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _slug_text(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def _variant_name(*, base_name: str, overrides: dict[str, Any]) -> str:
    payload = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    key_aliases = {
        "dislocation_threshold_pp": "dth",
        "expected_net_min_bnb": "ev",
        "cutoff_pool_total_min_bnb": "pool",
        "fixed_bet_bnb": "bet",
        "allowed_sides": "allow",
        "pool_total_gate_mode": "poolgate",
        "projected_final_pool_total_min_bnb": "ppool",
        "projected_final_pool_multiplier": "pmult",
        "bull_expected_net_extra_min_bnb": "bev",
        "bear_expected_net_extra_min_bnb": "rev",
        "side_selection_mode": "side",
        "nowcast_market_gap_min": "gap",
        "perf_adapt_mode": "perf",
        "perf_gate_window": "pgw",
        "perf_gate_min_history": "pgh",
        "perf_gate_min_win_rate": "pgwr",
        "perf_gate_min_mean_profit_bnb": "pgmn",
    }
    parts = [str(base_name)]
    for key in (
        "dislocation_threshold_pp",
        "expected_net_min_bnb",
        "cutoff_pool_total_min_bnb",
        "fixed_bet_bnb",
        "allowed_sides",
        "pool_total_gate_mode",
        "projected_final_pool_total_min_bnb",
        "projected_final_pool_multiplier",
        "bull_expected_net_extra_min_bnb",
        "bear_expected_net_extra_min_bnb",
        "side_selection_mode",
        "nowcast_market_gap_min",
        "perf_adapt_mode",
        "perf_gate_window",
        "perf_gate_min_history",
        "perf_gate_min_win_rate",
        "perf_gate_min_mean_profit_bnb",
    ):
        if str(key) not in overrides:
            continue
        value = overrides[str(key)]
        if isinstance(value, float):
            rendered = _slug_float(float(value))
        elif isinstance(value, int):
            rendered = str(int(value))
        else:
            rendered = _slug_text(str(value))
        parts.append(f"{key_aliases[str(key)]}_{rendered}")
    parts.append(f"h_{digest}")
    candidate = "__".join(parts)
    if len(candidate) <= 140:
        return candidate
    compact_base = str(base_name)
    if len(compact_base) > 48:
        compact_base = compact_base[:48].rstrip("_")
    return f"{compact_base}__h_{digest}"


def _candidate_from_overrides(*, base_candidate: Any, overrides: dict[str, Any]) -> Any:
    payload = asdict(base_candidate)
    for key, value in overrides.items():
        if str(key) not in payload:
            raise InvariantError(f"candidate_activity_sweep_override_unknown_field: {key}")
        payload[str(key)] = value
    return type(base_candidate)(**payload)


def _grid_values(base_candidate: Any, args: argparse.Namespace, base_overrides: dict[str, Any]) -> dict[str, list[Any]]:
    def pick(key: str, values: list[Any], base_value: Any) -> list[Any]:
        if str(key) in base_overrides:
            return [base_overrides[str(key)]]
        return values if values else [base_value]

    return {
        "dislocation_threshold_pp": pick(
            "dislocation_threshold_pp",
            _parse_float_list_or_empty(str(args.dislocation_threshold_pps)),
            float(base_candidate.dislocation_threshold_pp),
        ),
        "expected_net_min_bnb": pick(
            "expected_net_min_bnb",
            _parse_float_list_or_empty(str(args.expected_net_min_bnbs)),
            float(base_candidate.expected_net_min_bnb),
        ),
        "cutoff_pool_total_min_bnb": pick(
            "cutoff_pool_total_min_bnb",
            _parse_float_list_or_empty(str(args.cutoff_pool_total_min_bnbs)),
            float(base_candidate.cutoff_pool_total_min_bnb),
        ),
        "fixed_bet_bnb": pick(
            "fixed_bet_bnb",
            _parse_float_list_or_empty(str(args.fixed_bet_bnbs)),
            float(base_candidate.fixed_bet_bnb),
        ),
        "lookback1_seconds": pick(
            "lookback1_seconds",
            _parse_int_list_or_empty(str(args.lookback1_seconds_list)),
            int(base_candidate.lookback1_seconds),
        ),
        "temperature_bps": pick(
            "temperature_bps",
            _parse_float_list_or_empty(str(args.temperature_bps_list)),
            float(base_candidate.temperature_bps),
        ),
        "allowed_sides": pick(
            "allowed_sides",
            _parse_str_list_or_empty(str(args.allowed_sides_list)),
            str(base_candidate.allowed_sides),
        ),
        "market_extreme_min": pick(
            "market_extreme_min",
            _parse_float_list_or_empty(str(args.market_extreme_mins)),
            float(base_candidate.market_extreme_min),
        ),
        "nowcast_market_gap_min": pick(
            "nowcast_market_gap_min",
            _parse_float_list_or_empty(str(args.nowcast_market_gap_mins)),
            float(base_candidate.nowcast_market_gap_min),
        ),
        "flow_window_seconds": pick(
            "flow_window_seconds",
            _parse_int_list_or_empty(str(args.flow_window_seconds_list)),
            int(base_candidate.flow_window_seconds),
        ),
        "flow_min_imbalance": pick(
            "flow_min_imbalance",
            _parse_float_list_or_empty(str(args.flow_min_imbalances)),
            float(base_candidate.flow_min_imbalance),
        ),
        "flow_gate_mode": pick(
            "flow_gate_mode",
            _parse_str_list_or_empty(str(args.flow_gate_modes)),
            str(base_candidate.flow_gate_mode),
        ),
        "pool_total_gate_mode": pick(
            "pool_total_gate_mode",
            _parse_str_list_or_empty(str(args.pool_total_gate_modes)),
            str(base_candidate.pool_total_gate_mode),
        ),
        "projected_final_pool_total_min_bnb": pick(
            "projected_final_pool_total_min_bnb",
            _parse_float_list_or_empty(str(args.projected_final_pool_total_min_bnbs)),
            float(base_candidate.projected_final_pool_total_min_bnb),
        ),
        "projected_final_pool_multiplier": pick(
            "projected_final_pool_multiplier",
            _parse_float_list_or_empty(str(args.projected_final_pool_multipliers)),
            float(base_candidate.projected_final_pool_multiplier),
        ),
        "bull_expected_net_extra_min_bnb": pick(
            "bull_expected_net_extra_min_bnb",
            _parse_float_list_or_empty(str(args.bull_expected_net_extra_min_bnbs)),
            float(base_candidate.bull_expected_net_extra_min_bnb),
        ),
        "bear_expected_net_extra_min_bnb": pick(
            "bear_expected_net_extra_min_bnb",
            _parse_float_list_or_empty(str(args.bear_expected_net_extra_min_bnbs)),
            float(base_candidate.bear_expected_net_extra_min_bnb),
        ),
        "side_selection_mode": pick(
            "side_selection_mode",
            _parse_str_list_or_empty(str(args.side_selection_modes)),
            str(base_candidate.side_selection_mode),
        ),
        "perf_adapt_mode": pick(
            "perf_adapt_mode",
            _parse_str_list_or_empty(str(args.perf_adapt_modes)),
            str(base_candidate.perf_adapt_mode),
        ),
        "perf_gate_window": pick(
            "perf_gate_window",
            _parse_int_list_or_empty(str(args.perf_gate_windows_list)),
            int(base_candidate.perf_gate_window),
        ),
        "perf_gate_min_history": pick(
            "perf_gate_min_history",
            _parse_int_list_or_empty(str(args.perf_gate_min_history_list)),
            int(base_candidate.perf_gate_min_history),
        ),
        "perf_gate_min_win_rate": pick(
            "perf_gate_min_win_rate",
            _parse_float_list_or_empty(str(args.perf_gate_min_win_rates)),
            float(base_candidate.perf_gate_min_win_rate),
        ),
        "perf_gate_min_mean_profit_bnb": pick(
            "perf_gate_min_mean_profit_bnb",
            _parse_float_list_or_empty(str(args.perf_gate_min_mean_profit_bnbs)),
            float(base_candidate.perf_gate_min_mean_profit_bnb),
        ),
    }


def _enumerate_override_payloads(grid_values: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid_values.keys())
    rows: list[dict[str, Any]] = []

    def walk(idx: int, cur: dict[str, Any]) -> None:
        if int(idx) >= len(keys):
            rows.append(dict(cur))
            return
        key = str(keys[idx])
        for value in grid_values[str(key)]:
            cur[str(key)] = value
            walk(int(idx) + 1, cur)
        cur.pop(str(key), None)

    walk(0, {})
    unique: dict[str, dict[str, Any]] = {}
    for payload in rows:
        stable_key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        unique[str(stable_key)] = dict(payload)
    return list(unique.values())


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.sim_size) <= 0:
        raise InvariantError("candidate_activity_sweep_sim_size_nonpositive")
    if int(args.tail_offset_rounds) < 0:
        raise InvariantError("candidate_activity_sweep_tail_offset_negative")
    if float(args.min_bet_rate) < 0.0:
        raise InvariantError("candidate_activity_sweep_min_bet_rate_negative")
    if float(args.target_bet_rate) < 0.0:
        raise InvariantError("candidate_activity_sweep_target_bet_rate_negative")
    if int(args.top_k) <= 0:
        raise InvariantError("candidate_activity_sweep_top_k_nonpositive")
    _validate_partition_args(
        chunk_size=int(args.chunk_size),
        chunk_index=int(args.chunk_index),
    )

    cfg = load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    candidate_pool = _candidate_pool(
        cfg=cfg,
        config_path=str(args.config),
        candidate_source=str(args.candidate_source),
    )
    candidate_map = {str(candidate.name): candidate for candidate in candidate_pool}
    candidate_name = str(args.candidate_name)
    if str(candidate_name) not in candidate_map:
        raise InvariantError(f"candidate_activity_sweep_candidate_missing: {candidate_name}")

    base_candidate = candidate_map[str(candidate_name)]
    base_overrides = _parse_set_overrides(list(args.set))
    grid_values = _grid_values(base_candidate=base_candidate, args=args, base_overrides=base_overrides)
    override_payloads = _enumerate_override_payloads(grid_values)
    selected_override_payloads = _select_override_payloads(
        override_payloads=override_payloads,
        chunk_size=int(args.chunk_size),
        chunk_index=int(args.chunk_index),
    )
    resume = not bool(args.no_resume)

    executed = 0
    skipped_existing = 0
    if not bool(args.aggregate_only) and selected_override_payloads:
        for position, override_payload in enumerate(selected_override_payloads, start=1):
            merged_overrides = dict(override_payload)
            merged_overrides.update(base_overrides)
            tuned_candidate = _candidate_from_overrides(base_candidate=base_candidate, overrides=merged_overrides)
            strategy_cfg = replace(
                cfg.strategy,
                dislocation=replace(cfg.strategy.dislocation, candidates=(tuned_candidate,)),
                router=replace(
                    cfg.strategy.router,
                    mode=str(args.router_mode),
                    score_threshold_bnb=float(args.router_score_threshold_bnb),
                ),
            )

            variant_name = _variant_name(base_name=str(candidate_name), overrides=merged_overrides)
            run_name = (
                f"{str(args.name_prefix)}__{variant_name}"
                f"__off{int(args.tail_offset_rounds)}__sim{int(args.sim_size)}"
            )
            existing = (
                _maybe_load_existing_summary(exp_root=exp_root, run_name=str(run_name))
                if bool(resume)
                else None
            )
            if existing is None:
                run_backtest_case(
                    cfg=cfg,
                    strategy_cfg=strategy_cfg,
                    name=str(run_name),
                    simulation_size=int(args.sim_size),
                    reset_mode="continuous",
                    reset_every_rounds=0,
                    tail_offset_rounds=int(args.tail_offset_rounds),
                    initial_bankroll_bnb=args.initial_bankroll_bnb,
                    exp_root=exp_root,
                )
                executed += 1
            else:
                skipped_existing += 1
            if int(args.progress_every) > 0 and (
                int(executed) == 1
                or int(position) == int(len(selected_override_payloads))
                or int(position) % int(args.progress_every) == 0
            ):
                print(
                    "PROGRESS "
                    f"selected={len(selected_override_payloads)} "
                    f"position={position} "
                    f"executed={executed} "
                    f"skipped_existing={skipped_existing} "
                    f"variant={variant_name}"
                )

    rows: list[dict[str, object]] = []
    for override_payload in override_payloads:
        merged_overrides = dict(override_payload)
        merged_overrides.update(base_overrides)
        variant_name = _variant_name(base_name=str(candidate_name), overrides=merged_overrides)
        run_name = (
            f"{str(args.name_prefix)}__{variant_name}"
            f"__off{int(args.tail_offset_rounds)}__sim{int(args.sim_size)}"
        )
        existing = _maybe_load_existing_summary(exp_root=exp_root, run_name=str(run_name))
        if existing is None:
            continue
        summary, summary_path, trades_path = existing
        elapsed_seconds = 0.0

        net_profit_bnb = float(summary.get("net_profit_bnb", 0.0))
        per_500 = float(net_profit_bnb) * 500.0 / float(args.sim_size)
        bet_rate = float(summary.get("bet_rate", 0.0))
        activity_ok = bool(float(bet_rate) >= float(args.min_bet_rate))
        activity_gap = max(0.0, float(args.min_bet_rate) - float(bet_rate))
        rows.append(
            {
                "candidate_name": str(candidate_name),
                "variant_name": str(variant_name),
                "run_name": str(run_name),
                "sim_size": int(args.sim_size),
                "tail_offset_rounds": int(args.tail_offset_rounds),
                "net_profit_bnb": float(net_profit_bnb),
                "per_500": float(per_500),
                "num_bets": int(summary.get("num_bets", 0)),
                "bet_rate": float(bet_rate),
                "activity_ok": int(activity_ok),
                "activity_gap_to_min": float(activity_gap),
                "activity_gap_to_target": abs(float(bet_rate) - float(args.target_bet_rate)),
                "max_drawdown_bnb": float(max_drawdown_bnb(trades_csv_path=trades_path)),
                "min_bankroll_bnb": float(_min_bankroll_bnb(trades_csv_path=trades_path)),
                "loss_from_initial_to_min_bnb": (
                    float(summary.get("initial_bankroll_bnb", 0.0))
                    - float(_min_bankroll_bnb(trades_csv_path=trades_path))
                ),
                "top_skip_reasons": top_skip_reasons(
                    summary=summary,
                    limit=int(args.top_skip_limit),
                ),
                "selected_strategy_mix": _selected_mix(
                    trades_csv_path=trades_path,
                    limit=int(args.top_selected_limit),
                ),
                "summary_path": str(summary_path),
                "trades_path": str(trades_path),
                "elapsed_seconds": float(elapsed_seconds),
                "overrides_json": json.dumps(merged_overrides, sort_keys=True),
            }
        )

    if not rows:
        raise InvariantError("candidate_activity_sweep_no_completed_rows")

    rows.sort(
        key=lambda row: (
            -int(row["activity_ok"]),
            -float(row["per_500"]),
            float(row["activity_gap_to_target"]),
            -float(row["bet_rate"]),
        )
    )

    out_json = exp_root / f"{str(args.name_prefix)}_table.json"
    out_csv = exp_root / f"{str(args.name_prefix)}_table.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(
        json.dumps(
            {
                "name_prefix": str(args.name_prefix),
                "candidate_name": str(candidate_name),
                "sim_size": int(args.sim_size),
                "tail_offset_rounds": int(args.tail_offset_rounds),
                "min_bet_rate": float(args.min_bet_rate),
                "target_bet_rate": float(args.target_bet_rate),
                "expected_variant_count": int(len(override_payloads)),
                "completed_variant_count": int(len(rows)),
                "is_complete": bool(len(rows) == len(override_payloads)),
                "chunk_size": int(args.chunk_size),
                "chunk_index": int(args.chunk_index),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    table_rows = [
        {
            "variant": str(row["variant_name"]),
            "per_500": f"{float(row['per_500']):+.6f}",
            "bet_rate": f"{float(row['bet_rate']):.4f}",
            "bets": int(row["num_bets"]),
            "activity_ok": int(row["activity_ok"]),
            "max_dd": f"{float(row['max_drawdown_bnb']):.6f}",
            "elapsed_s": f"{float(row['elapsed_seconds']):.3f}",
        }
        for row in rows[: int(args.top_k)]
    ]
    print(
        render_table(
            columns=[
                ("variant", "variant"),
                ("per_500", "per_500"),
                ("bet_rate", "bet_rate"),
                ("bets", "bets"),
                ("activity_ok", "activity_ok"),
                ("max_dd", "max_dd"),
                ("elapsed_s", "elapsed_s"),
            ],
            rows=table_rows,
        )
    )
    top_activity = [row for row in rows if int(row["activity_ok"]) == 1]
    print(f"TABLE_JSON={out_json}")
    print(f"TABLE_CSV={out_csv}")
    print(f"EXPECTED_VARIANTS={len(override_payloads)}")
    print(f"COMPLETED_VARIANTS={len(rows)}")
    print(f"IS_COMPLETE={1 if len(rows) == len(override_payloads) else 0}")
    print(f"SELECTED_VARIANTS={len(selected_override_payloads)}")
    print(f"EXECUTED={executed}")
    print(f"SKIPPED_EXISTING={skipped_existing}")
    print(f"MEETS_MIN_BET_RATE={len(top_activity)}")
    if top_activity:
        best = top_activity[0]
        print(f"BEST_IN_BAND_PER_500={best['per_500']}")
        print(f"BEST_IN_BAND_BET_RATE={best['bet_rate']}")
    else:
        print(f"BEST_OVERALL_PER_500={rows[0]['per_500']}")
        print(f"BEST_OVERALL_BET_RATE={rows[0]['bet_rate']}")


if __name__ == "__main__":
    main()
