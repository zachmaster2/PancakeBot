from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pancakebot.core.errors import InvariantError

from inspection.backtest_harness_common import (
    load_cfg,
    load_all_dislocation_candidates,
    max_drawdown_bnb,
    render_table,
    resolve_exp_root,
    run_backtest_case,
    top_skip_reasons,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--candidate-name", type=str, default="disloc_altA_20260227_x80")
    p.add_argument(
        "--candidate-names",
        type=str,
        default="",
        help="Optional comma-separated candidate subset to run.",
    )
    p.add_argument(
        "--candidate-source",
        type=str,
        choices=("active", "all_config"),
        default="active",
        help="Whether candidate lookup should use only active candidates or all config-defined candidates.",
    )
    p.add_argument("--sim-size", type=int, default=30000)
    p.add_argument("--offsets", type=str, default="0,5000,10000")
    p.add_argument("--initial-bankroll-bnb", type=float, default=None)
    p.add_argument("--router-mode", type=str, default="selector_max_score")
    p.add_argument("--router-score-threshold-bnb", type=float, default=None)
    p.add_argument("--stake-scale", type=float, default=1.0)
    p.add_argument(
        "--candidate-overrides-json",
        type=str,
        default="{}",
        help="JSON object with DislocationCandidateConfig field overrides.",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Candidate override as key=value. May be repeated.",
    )
    p.add_argument(
        "--ml-enabled",
        type=str,
        default="",
        help="Optional bool override for strategy.ml_candidate.enabled (true/false).",
    )
    p.add_argument(
        "--ml-set",
        action="append",
        default=[],
        help="ML-candidate override as key=value. May be repeated.",
    )
    p.add_argument(
        "--keep-all-candidates",
        action="store_true",
        help="Override only target candidate and keep the rest of the ensemble.",
    )
    p.add_argument(
        "--apply-overrides-to-all-candidates",
        action="store_true",
        help="Apply candidate overrides/stake-scale to every selected candidate.",
    )
    p.add_argument("--top-skip-limit", type=int, default=4)
    p.add_argument("--top-selected-limit", type=int, default=4)
    p.add_argument("--no-resume", action="store_true")
    return p


def _parse_offsets(raw: str) -> list[int]:
    out: list[int] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        try:
            out.append(int(token))
        except ValueError as e:
            raise InvariantError(f"altA_single_idea_offset_invalid: {token}") from e
    out = sorted(set(out))
    if not out:
        raise InvariantError("altA_single_idea_offsets_empty")
    if any(int(x) < 0 for x in out):
        raise InvariantError("altA_single_idea_offsets_negative")
    return out


def _parse_name_list(raw: str) -> list[str]:
    out = [str(x).strip() for x in str(raw).split(",") if str(x).strip()]
    if not out:
        raise InvariantError("altA_single_idea_candidate_names_empty")
    return out


def _count_jsonl_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as f:
        return int(sum(1 for _ in f))


def _candidate_pool(*, cfg, config_path: str, candidate_source: str) -> tuple[Any, ...]:
    source = str(candidate_source)
    if source == "active":
        return tuple(cfg.strategy.dislocation.candidates)
    if source == "all_config":
        return load_all_dislocation_candidates(config_path=str(config_path))
    raise InvariantError(f"altA_single_idea_candidate_source_unknown: {candidate_source}")


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
            bankroll = float(row["bankroll_bnb"])
            if min_bankroll is None or float(bankroll) < float(min_bankroll):
                min_bankroll = float(bankroll)
    return 0.0 if min_bankroll is None else float(min_bankroll)


def _maybe_load_existing_summary(
    *,
    exp_root: Path,
    run_name: str,
) -> tuple[dict[str, Any], Path, Path] | None:
    out_dir = Path(exp_root) / str(run_name)
    summary_path = out_dir / "backtest_summary.json"
    trades_path = out_dir / "backtest_trades.csv"
    if not summary_path.exists() or not trades_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(summary, dict):
        return None
    return dict(summary), summary_path, trades_path


def _parse_candidate_overrides(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as e:
        raise InvariantError("altA_single_idea_overrides_json_invalid") from e
    if not isinstance(parsed, dict):
        raise InvariantError("altA_single_idea_overrides_json_not_object")
    return dict(parsed)


def _coerce_scalar(value: str) -> Any:
    raw = str(value).strip()
    lo = str(raw).lower()
    if lo in ("true", "false"):
        return bool(lo == "true")
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return str(raw)


def _parse_bool_token(raw: str) -> bool:
    lo = str(raw).strip().lower()
    if lo in ("true", "t", "1", "yes", "y", "on"):
        return True
    if lo in ("false", "f", "0", "no", "n", "off"):
        return False
    raise InvariantError(f"altA_single_idea_bool_token_invalid: {raw}")


def _parse_set_overrides(tokens: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for token in tokens:
        text = str(token)
        if "=" not in text:
            raise InvariantError(f"altA_single_idea_set_override_invalid: {text}")
        key, raw_value = text.split("=", 1)
        key = str(key).strip()
        raw_value = str(raw_value).strip()
        if key == "":
            raise InvariantError(f"altA_single_idea_set_override_empty_key: {text}")
        if raw_value.startswith("[") and raw_value.endswith("]"):
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError as e:
                raise InvariantError("altA_single_idea_set_override_json_list_invalid") from e
            if not isinstance(parsed, list):
                raise InvariantError("altA_single_idea_set_override_list_expected")
            out[str(key)] = parsed
            continue
        out[str(key)] = _coerce_scalar(raw_value)
    return out


def _apply_candidate_overrides(*, base_candidate: Any, overrides: dict[str, Any], stake_scale: float):
    if float(stake_scale) <= 0.0:
        raise InvariantError("altA_single_idea_stake_scale_nonpositive")
    c = replace(
        base_candidate,
        fixed_bet_bnb=float(base_candidate.fixed_bet_bnb) * float(stake_scale),
        expected_net_min_bnb=float(base_candidate.expected_net_min_bnb) * float(stake_scale),
        stake_min_bnb=float(base_candidate.stake_min_bnb) * float(stake_scale),
        stake_max_bnb=float(base_candidate.stake_max_bnb) * float(stake_scale),
        stake_ev_ref_bnb=float(base_candidate.stake_ev_ref_bnb) * float(stake_scale),
    )
    if not overrides:
        return c

    payload = asdict(c)
    for key, value in overrides.items():
        if str(key) not in payload:
            raise InvariantError(f"altA_single_idea_override_unknown_field: {key}")
        if str(key) == "adaptive_candidate_modes":
            if value is None:
                payload[str(key)] = ()
            elif isinstance(value, (list, tuple)):
                payload[str(key)] = tuple(str(x) for x in value)
            else:
                raise InvariantError("altA_single_idea_adaptive_modes_invalid")
            continue
        payload[str(key)] = value
    return type(c)(**payload)


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    sim_size = int(args.sim_size)
    if int(sim_size) <= 0:
        raise InvariantError("altA_single_idea_sim_size_nonpositive")
    offsets = _parse_offsets(str(args.offsets))
    if int(args.top_skip_limit) <= 0 or int(args.top_selected_limit) <= 0:
        raise InvariantError("altA_single_idea_top_limits_nonpositive")

    total_rounds = _count_jsonl_lines(Path(str(cfg.closed_rounds_path)))
    warmup_rounds = int(cfg.strategy.dislocation.selector.warmup_rounds)
    for offset in offsets:
        needed = int(warmup_rounds) + int(sim_size) + int(offset)
        if int(needed) > int(total_rounds):
            raise InvariantError(
                f"altA_single_idea_sim_window_exceeds_history: needed={needed} total={total_rounds}"
            )

    candidate_pool = _candidate_pool(
        cfg=cfg,
        config_path=str(args.config),
        candidate_source=str(args.candidate_source),
    )
    c_map = {str(c.name): c for c in candidate_pool}
    target_name = str(args.candidate_name)
    if str(target_name) not in c_map:
        raise InvariantError(f"altA_single_idea_candidate_missing: {target_name}")

    overrides = _parse_candidate_overrides(str(args.candidate_overrides_json))
    set_overrides = _parse_set_overrides(list(args.set))
    overrides.update(set_overrides)

    ml_cfg = cfg.strategy.ml_candidate
    ml_overrides = _parse_set_overrides(list(args.ml_set))
    if str(args.ml_enabled).strip() != "":
        ml_overrides["enabled"] = _parse_bool_token(str(args.ml_enabled))
    if ml_overrides:
        payload = asdict(ml_cfg)
        for key, value in ml_overrides.items():
            if str(key) not in payload:
                raise InvariantError(f"altA_single_idea_ml_override_unknown_field: {key}")
            payload[str(key)] = value
        ml_cfg = type(ml_cfg)(**payload)
    apply_all = bool(args.apply_overrides_to_all_candidates)

    def _tuned_for_name(name: str):
        base = c_map[str(name)]
        if bool(apply_all) or str(name) == str(target_name):
            return _apply_candidate_overrides(
                base_candidate=base,
                overrides=overrides,
                stake_scale=float(args.stake_scale),
            )
        return base

    if str(args.candidate_names).strip() != "":
        if bool(args.keep_all_candidates):
            raise InvariantError("altA_single_idea_conflict_keep_all_and_candidate_names")
        names = _parse_name_list(str(args.candidate_names))
        tuned_list = []
        for name in names:
            if str(name) not in c_map:
                raise InvariantError(f"altA_single_idea_candidate_missing: {name}")
            tuned_list.append(_tuned_for_name(str(name)))
        tuned_candidates = tuple(tuned_list)
    elif bool(args.keep_all_candidates):
        tuned_candidates = tuple(_tuned_for_name(str(c.name)) for c in candidate_pool)
    else:
        tuned_candidates = (_tuned_for_name(str(target_name)),)
    dislocation_cfg = replace(cfg.strategy.dislocation, candidates=tuned_candidates)
    router_cfg = replace(cfg.strategy.router, mode=str(args.router_mode))
    if args.router_score_threshold_bnb is not None:
        router_cfg = replace(
            router_cfg,
            score_threshold_bnb=float(args.router_score_threshold_bnb),
        )
    strategy_cfg = replace(
        cfg.strategy,
        dislocation=dislocation_cfg,
        router=router_cfg,
        ml_candidate=ml_cfg,
    )

    resume = not bool(args.no_resume)
    rows: list[dict[str, object]] = []
    for offset in offsets:
        run_name = f"{str(args.name_prefix)}_off{int(offset)}_sim{int(sim_size)}"
        existing = (
            _maybe_load_existing_summary(exp_root=exp_root, run_name=str(run_name))
            if bool(resume)
            else None
        )
        if existing is None:
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
            summary_path = result.summary_path
            trades_path = result.trades_path
            elapsed_seconds = float(result.elapsed_seconds)
        else:
            summary, summary_path, trades_path = existing
            elapsed_seconds = 0.0

        net = float(summary.get("net_profit_bnb", 0.0))
        per_500 = float(net) * 500.0 / float(sim_size)
        max_dd = float(max_drawdown_bnb(trades_csv_path=trades_path))
        min_bank = float(_min_bankroll_bnb(trades_csv_path=trades_path))
        initial_bank = float(summary.get("initial_bankroll_bnb", 0.0))
        loss2min = float(initial_bank) - float(min_bank)

        rows.append(
            {
                "offset": int(offset),
                "sim_size": int(sim_size),
                "run_name": str(run_name),
                "net_profit_bnb": float(net),
                "per_500": float(per_500),
                "max_drawdown_bnb": float(max_dd),
                "num_bets": int(summary.get("num_bets", 0)),
                "bet_rate": float(summary.get("bet_rate", 0.0)),
                "min_bankroll_bnb": float(min_bank),
                "initial_bankroll_bnb": float(initial_bank),
                "loss_from_initial_to_min_bnb": float(loss2min),
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
            }
        )

    rows = sorted(rows, key=lambda r: int(r["offset"]))

    table_rows = [
        {
            "offset": int(r["offset"]),
            "per_500": f"{float(r['per_500']):+.6f}",
            "net": f"{float(r['net_profit_bnb']):+.6f}",
            "max_dd": f"{float(r['max_drawdown_bnb']):.6f}",
            "loss2min": f"{float(r['loss_from_initial_to_min_bnb']):.6f}",
            "bets": int(r["num_bets"]),
            "bet_rate": f"{float(r['bet_rate']):.4f}",
            "elapsed_s": f"{float(r['elapsed_seconds']):.3f}",
        }
        for r in rows
    ]
    print(
        render_table(
            columns=[
                ("offset", "offset"),
                ("per_500", "per_500"),
                ("net", "net"),
                ("max_dd", "max_dd"),
                ("loss2min", "loss2min"),
                ("bets", "bets"),
                ("bet_rate", "bet_rate"),
                ("elapsed_s", "elapsed_s"),
            ],
            rows=table_rows,
        )
    )

    per500_vals = [float(r["per_500"]) for r in rows]
    net_vals = [float(r["net_profit_bnb"]) for r in rows]
    dd_vals = [float(r["max_drawdown_bnb"]) for r in rows]
    loss_vals = [float(r["loss_from_initial_to_min_bnb"]) for r in rows]
    aggregate = {
        "n_windows": int(len(rows)),
        "mean_per_500": float(statistics.mean(per500_vals)),
        "median_per_500": float(statistics.median(per500_vals)),
        "worst_per_500": float(min(per500_vals)),
        "best_per_500": float(max(per500_vals)),
        "mean_net_profit_bnb": float(statistics.mean(net_vals)),
        "worst_net_profit_bnb": float(min(net_vals)),
        "worst_max_drawdown_bnb": float(max(dd_vals)),
        "worst_loss_from_initial_to_min_bnb": float(max(loss_vals)),
        "positive_count": int(sum(1 for x in per500_vals if float(x) > 0.0)),
    }
    print(
        "AGG "
        + " ".join(
            [
                f"n={aggregate['n_windows']}",
                f"mean_per500={aggregate['mean_per_500']:+.6f}",
                f"worst_per500={aggregate['worst_per_500']:+.6f}",
                f"worst_net={aggregate['worst_net_profit_bnb']:+.6f}",
                f"worst_dd={aggregate['worst_max_drawdown_bnb']:.6f}",
                f"worst_loss2min={aggregate['worst_loss_from_initial_to_min_bnb']:.6f}",
                f"positive={aggregate['positive_count']}",
            ]
        )
    )

    out_json = exp_root / f"{str(args.name_prefix)}_table.json"
    out_json.write_text(
        json.dumps(
            {
                "name_prefix": str(args.name_prefix),
                "candidate_name": str(target_name),
                "router_mode": str(args.router_mode),
                "router_score_threshold_bnb": float(router_cfg.score_threshold_bnb),
                "sim_size": int(sim_size),
                "offsets": [int(x) for x in offsets],
                "stake_scale": float(args.stake_scale),
                "overrides": dict(overrides),
                "ml_overrides": dict(ml_overrides),
                "keep_all_candidates": bool(args.keep_all_candidates),
                "apply_overrides_to_all_candidates": bool(apply_all),
                "rows": rows,
                "aggregate": aggregate,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"TABLE_JSON={out_json}")


if __name__ == "__main__":
    main()
