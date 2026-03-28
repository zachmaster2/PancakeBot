from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from pancakebot.core.errors import InvariantError
from inspection.run_profile_window_selector import (
    _ensure_flow_window,
    _ensure_stageb_window,
    _materialize_tail_config,
    _parse_float_list,
    _parse_int_list,
    _parse_positive_int_list,
)

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_ACTIVE_CANDIDATES_PATTERN = re.compile(
    r"^active_candidate_names\s*=\s*\[(?:.|\n)*?^\]",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class FlowProfileSpec:
    name: str
    train_size: int
    val_size: int | None
    step_size: int | None
    ev_threshold: float
    min_total_pool_c: float
    allowed_sides: str
    bull_roll_edge_min: float
    bear_roll_edge_min: float
    bull_roll_winrate_min: float
    bear_roll_winrate_min: float
    bull_cooldown_trades: int
    bear_cooldown_trades: int


@dataclass(frozen=True, slots=True)
class DislocationProfileSpec:
    name: str
    active_candidate_name: str


@dataclass(frozen=True, slots=True)
class ProfileMetric:
    per_500: float
    bet_rate: float


@dataclass(frozen=True, slots=True)
class WindowRow:
    tail_offset_rounds: int
    metrics: dict[str, ProfileMetric]


@dataclass(frozen=True, slots=True)
class SelectorResult:
    mode: str
    profile_name: str
    lookback: int
    margin_per_500: float
    skip_threshold_per_500: float
    mean_per_500: float
    mean_selected_bet_rate: float
    meets_min_selected_bet_rate: bool
    pick_counts_json: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--window-size-rounds", type=int, required=True)
    parser.add_argument("--num-windows", type=int, default=10)
    parser.add_argument("--tail-offset-rounds", type=str, default=None)
    parser.add_argument("--source-tail-rounds", type=int, default=None)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    parser.add_argument(
        "--flow-profile",
        action="append",
        default=[],
        help=(
            "Repeated flow profile spec, for example: "
            "name=flow_bear_a,train_size=15000,ev_threshold=0.006,min_total_pool_c=1.2,"
            "allowed_sides=bear_only,bull_roll_edge_min=0.0,bear_roll_edge_min=0.0,"
            "bull_roll_winrate_min=0.5,bear_roll_winrate_min=0.5,"
            "bull_cooldown_trades=80,bear_cooldown_trades=80"
        ),
    )
    parser.add_argument(
        "--dislocation-profile",
        action="append",
        default=[],
        help=(
            "Repeated dislocation profile spec, for example: "
            "name=stageg2_bullonly,active_candidate_name=disloc_stageG2_bullonly_recent5pct_v1"
        ),
    )
    parser.add_argument("--selector-lookbacks", type=str, default="1,2,3,4,5")
    parser.add_argument("--selector-margins-per-500", type=str, default="-0.2,0.0,0.2,0.5")
    parser.add_argument("--selector-skip-thresholds-per-500", type=str, default="0.0")
    parser.add_argument("--min-selected-bet-rate", type=float, default=0.05)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _parse_flow_profile_spec(raw: str, *, window_size_rounds: int) -> FlowProfileSpec:
    parts = [token.strip() for token in str(raw).split(",") if token.strip() != ""]
    data: dict[str, str] = {}
    for token in parts:
        if "=" not in token:
            raise InvariantError(f"profile_set_flow_profile_token_invalid: {token}")
        key, value = token.split("=", 1)
        data[str(key).strip()] = str(value).strip()
    name = str(data.get("name", "")).strip()
    if name == "" or str(name).lower() == "stageb":
        raise InvariantError("profile_set_flow_profile_name_invalid")
    allowed_sides = str(data.get("allowed_sides", "bear_only"))
    if str(allowed_sides) not in {"both", "bull_only", "bear_only"}:
        raise InvariantError("profile_set_flow_profile_allowed_sides_invalid")
    train_size_raw = data.get("train_size")
    ev_threshold_raw = data.get("ev_threshold")
    min_total_pool_c_raw = data.get("min_total_pool_c")
    if train_size_raw is None or ev_threshold_raw is None or min_total_pool_c_raw is None:
        raise InvariantError("profile_set_flow_profile_required_field_missing")
    val_size_raw = data.get("val_size")
    step_size_raw = data.get("step_size")
    return FlowProfileSpec(
        name=str(name),
        train_size=int(train_size_raw),
        val_size=(None if val_size_raw is None else int(val_size_raw)),
        step_size=(None if step_size_raw is None else int(step_size_raw)),
        ev_threshold=float(ev_threshold_raw),
        min_total_pool_c=float(min_total_pool_c_raw),
        allowed_sides=str(allowed_sides),
        bull_roll_edge_min=float(data.get("bull_roll_edge_min", "0.0")),
        bear_roll_edge_min=float(data.get("bear_roll_edge_min", "0.0")),
        bull_roll_winrate_min=float(data.get("bull_roll_winrate_min", "0.5")),
        bear_roll_winrate_min=float(data.get("bear_roll_winrate_min", "0.5")),
        bull_cooldown_trades=int(data.get("bull_cooldown_trades", "80")),
        bear_cooldown_trades=int(data.get("bear_cooldown_trades", "80")),
    )


def _parse_dislocation_profile_spec(raw: str) -> DislocationProfileSpec:
    parts = [token.strip() for token in str(raw).split(",") if token.strip() != ""]
    data: dict[str, str] = {}
    for token in parts:
        if "=" not in token:
            raise InvariantError(f"profile_set_dislocation_profile_token_invalid: {token}")
        key, value = token.split("=", 1)
        data[str(key).strip()] = str(value).strip()
    name = str(data.get("name", "")).strip()
    active_candidate_name = str(data.get("active_candidate_name", "")).strip()
    if name == "" or str(name).lower() == "stageb":
        raise InvariantError("profile_set_dislocation_profile_name_invalid")
    if active_candidate_name == "":
        raise InvariantError("profile_set_dislocation_active_candidate_name_missing")
    return DislocationProfileSpec(
        name=str(name),
        active_candidate_name=str(active_candidate_name),
    )


def _ordered_window_rows(rows: list[WindowRow]) -> list[WindowRow]:
    return sorted(rows, key=lambda row: int(row.tail_offset_rounds), reverse=True)


def _materialize_active_candidate_config(
    *,
    base_config_path: Path,
    active_candidate_name: str,
    exp_root: Path,
    name_prefix: str,
) -> Path:
    cfg_text = base_config_path.read_text(encoding="utf-8")
    if _ACTIVE_CANDIDATES_PATTERN.search(cfg_text) is None:
        raise InvariantError("profile_set_active_candidate_names_missing")
    patched_text = _ACTIVE_CANDIDATES_PATTERN.sub(
        f'active_candidate_names = [\n  "{str(active_candidate_name)}",\n]',
        cfg_text,
        count=1,
    )
    out_cfg = (exp_root / f"{name_prefix}_{str(active_candidate_name)}_active.toml").resolve()
    out_cfg.write_text(patched_text, encoding="utf-8", newline="\n")
    return out_cfg


def _summary_per_500(summary: dict[str, object]) -> float:
    if "per_500" in summary:
        return float(summary["per_500"])
    if "net_profit_per_500_rounds" in summary:
        return float(summary["net_profit_per_500_rounds"])
    net_profit_bnb = summary.get("net_profit_bnb")
    num_rounds = summary.get("num_rounds")
    if isinstance(net_profit_bnb, (int, float)) and isinstance(num_rounds, int) and int(num_rounds) > 0:
        return float(float(net_profit_bnb) * 500.0 / float(num_rounds))
    raise InvariantError("profile_set_summary_per_500_missing")


def _best_profile(metrics: dict[str, ProfileMetric]) -> tuple[str, ProfileMetric]:
    best_name = ""
    best_metric: ProfileMetric | None = None
    for name, metric in metrics.items():
        if best_metric is None or float(metric.per_500) > float(best_metric.per_500):
            best_name = str(name)
            best_metric = metric
    if best_metric is None:
        raise InvariantError("profile_set_window_metrics_empty")
    return best_name, best_metric


def _trailing_means(rows: list[WindowRow], *, idx: int, lookback: int) -> dict[str, float]:
    hist = rows[int(idx) - int(lookback) : int(idx)]
    out: dict[str, float] = {}
    profile_names = sorted(hist[0].metrics.keys()) if hist else []
    for name in profile_names:
        out[str(name)] = sum(float(row.metrics[str(name)].per_500) for row in hist) / float(len(hist))
    return out


def _pick_window(
    *,
    rows: list[WindowRow],
    idx: int,
    mode: str,
    profile_name: str,
    lookback: int,
    margin_per_500: float,
    skip_threshold_per_500: float,
) -> tuple[str, float, float]:
    current = rows[int(idx)]
    if str(mode) == "static_profile":
        metric = current.metrics[str(profile_name)]
        return str(profile_name), float(metric.per_500), float(metric.bet_rate)
    if str(mode) == "skip_only":
        return "skip", 0.0, 0.0
    if str(mode) == "oracle":
        winner, metric = _best_profile(current.metrics)
        return str(winner), float(metric.per_500), float(metric.bet_rate)
    if str(mode) == "oracle_with_skip":
        winner, metric = _best_profile(current.metrics)
        if float(metric.per_500) <= float(skip_threshold_per_500):
            return "skip", 0.0, 0.0
        return str(winner), float(metric.per_500), float(metric.bet_rate)
    if str(mode) == "prev_winner":
        if int(idx) == 0:
            metric = current.metrics["stageb"]
            return "stageb", float(metric.per_500), float(metric.bet_rate)
        prev_winner, _ = _best_profile(rows[int(idx) - 1].metrics)
        metric = current.metrics[str(prev_winner)]
        return str(prev_winner), float(metric.per_500), float(metric.bet_rate)
    if str(mode) == "prev_winner_with_skip":
        if int(idx) == 0:
            metric = current.metrics["stageb"]
            if float(metric.per_500) <= float(skip_threshold_per_500):
                return "skip", 0.0, 0.0
            return "stageb", float(metric.per_500), float(metric.bet_rate)
        prev_winner, prev_metric = _best_profile(rows[int(idx) - 1].metrics)
        if float(prev_metric.per_500) <= float(skip_threshold_per_500):
            return "skip", 0.0, 0.0
        metric = current.metrics[str(prev_winner)]
        return str(prev_winner), float(metric.per_500), float(metric.bet_rate)
    if str(mode) == "trailing_best_vs_stageb":
        if int(idx) < int(lookback):
            metric = current.metrics["stageb"]
            return "stageb", float(metric.per_500), float(metric.bet_rate)
        means = _trailing_means(rows, idx=int(idx), lookback=int(lookback))
        stageb_mean = float(means["stageb"])
        best_alt_name = "stageb"
        best_alt_mean = float(stageb_mean)
        for name, value in means.items():
            if str(name) == "stageb":
                continue
            if float(value) > float(best_alt_mean):
                best_alt_name = str(name)
                best_alt_mean = float(value)
        if str(best_alt_name) != "stageb" and float(best_alt_mean - stageb_mean) > float(margin_per_500):
            metric = current.metrics[str(best_alt_name)]
            return str(best_alt_name), float(metric.per_500), float(metric.bet_rate)
        metric = current.metrics["stageb"]
        return "stageb", float(metric.per_500), float(metric.bet_rate)
    if str(mode) == "trailing_best_vs_stageb_with_skip":
        if int(idx) < int(lookback):
            metric = current.metrics["stageb"]
            if float(metric.per_500) <= float(skip_threshold_per_500):
                return "skip", 0.0, 0.0
            return "stageb", float(metric.per_500), float(metric.bet_rate)
        means = _trailing_means(rows, idx=int(idx), lookback=int(lookback))
        stageb_mean = float(means["stageb"])
        best_alt_name = "stageb"
        best_alt_mean = float(stageb_mean)
        for name, value in means.items():
            if str(name) == "stageb":
                continue
            if float(value) > float(best_alt_mean):
                best_alt_name = str(name)
                best_alt_mean = float(value)
        if max(float(stageb_mean), float(best_alt_mean)) <= float(skip_threshold_per_500):
            return "skip", 0.0, 0.0
        if str(best_alt_name) != "stageb" and float(best_alt_mean - stageb_mean) > float(margin_per_500):
            metric = current.metrics[str(best_alt_name)]
            return str(best_alt_name), float(metric.per_500), float(metric.bet_rate)
        metric = current.metrics["stageb"]
        return "stageb", float(metric.per_500), float(metric.bet_rate)
    raise InvariantError("profile_set_selector_mode_invalid")


def _evaluate_selectors(
    *,
    rows: list[WindowRow],
    profile_names: list[str],
    lookbacks: list[int],
    margins_per_500: list[float],
    skip_thresholds_per_500: list[float],
    min_selected_bet_rate: float,
) -> list[SelectorResult]:
    ordered = _ordered_window_rows(rows)
    modes: list[tuple[str, str, int, float, float]] = [("skip_only", "", 0, 0.0, 0.0)]
    for name in profile_names:
        modes.append(("static_profile", str(name), 0, 0.0, 0.0))
    modes.append(("oracle", "", 0, 0.0, 0.0))
    modes.append(("prev_winner", "", 0, 0.0, 0.0))
    for lookback in lookbacks:
        for margin in margins_per_500:
            modes.append(("trailing_best_vs_stageb", "", int(lookback), float(margin), 0.0))
    for skip_threshold in skip_thresholds_per_500:
        modes.append(("oracle_with_skip", "", 0, 0.0, float(skip_threshold)))
        modes.append(("prev_winner_with_skip", "", 0, 0.0, float(skip_threshold)))
        for lookback in lookbacks:
            for margin in margins_per_500:
                modes.append(
                    (
                        "trailing_best_vs_stageb_with_skip",
                        "",
                        int(lookback),
                        float(margin),
                        float(skip_threshold),
                    )
                )

    results: list[SelectorResult] = []
    for mode, profile_name, lookback, margin, skip_threshold in modes:
        total = 0.0
        total_selected_bet_rate = 0.0
        picks = Counter()
        for idx in range(len(ordered)):
            pick, value, bet_rate = _pick_window(
                rows=ordered,
                idx=int(idx),
                mode=str(mode),
                profile_name=str(profile_name),
                lookback=int(lookback),
                margin_per_500=float(margin),
                skip_threshold_per_500=float(skip_threshold),
            )
            total += float(value)
            total_selected_bet_rate += float(bet_rate)
            picks[str(pick)] += 1
        mean_selected_bet_rate = float(total_selected_bet_rate / float(len(ordered))) if ordered else 0.0
        results.append(
            SelectorResult(
                mode=str(mode),
                profile_name=str(profile_name),
                lookback=int(lookback),
                margin_per_500=float(margin),
                skip_threshold_per_500=float(skip_threshold),
                mean_per_500=float(total / float(len(ordered))) if ordered else 0.0,
                mean_selected_bet_rate=float(mean_selected_bet_rate),
                meets_min_selected_bet_rate=float(mean_selected_bet_rate) >= float(min_selected_bet_rate) if ordered else False,
                pick_counts_json=json.dumps(dict(sorted(picks.items())), sort_keys=True, separators=(",", ":")),
            )
        )
    return sorted(
        results,
        key=lambda row: (
            1 if bool(row.meets_min_selected_bet_rate) else 0,
            float(row.mean_per_500),
            float(row.mean_selected_bet_rate),
        ),
        reverse=True,
    )


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.window_size_rounds) <= 0:
        raise InvariantError("profile_set_window_size_nonpositive")
    if int(args.num_windows) <= 0:
        raise InvariantError("profile_set_num_windows_nonpositive")
    if float(args.min_selected_bet_rate) < 0.0:
        raise InvariantError("profile_set_min_selected_bet_rate_negative")
    flow_profiles = [
        _parse_flow_profile_spec(raw, window_size_rounds=int(args.window_size_rounds))
        for raw in list(args.flow_profile)
    ]
    dislocation_profiles = [
        _parse_dislocation_profile_spec(raw)
        for raw in list(args.dislocation_profile)
    ]
    if not flow_profiles and not dislocation_profiles:
        raise InvariantError("profile_set_profiles_required")
    names = [str(profile.name) for profile in flow_profiles] + [str(profile.name) for profile in dislocation_profiles]
    if len(names) != len(set(names)):
        raise InvariantError("profile_set_profile_name_duplicate")

    cwd = Path.cwd().resolve()
    exp_root = Path(_DEFAULT_EXP_ROOT).resolve()
    exp_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(str(args.config)).resolve()
    active_config_path = config_path
    if args.source_tail_rounds is not None:
        active_config_path = _materialize_tail_config(
            base_config_path=config_path,
            source_tail_rounds=int(args.source_tail_rounds),
            exp_root=exp_root,
            name_prefix=str(args.name_prefix),
        )
    dislocation_config_paths: dict[str, Path] = {}
    for profile in dislocation_profiles:
        dislocation_config_paths[str(profile.name)] = _materialize_active_candidate_config(
            base_config_path=active_config_path,
            active_candidate_name=str(profile.active_candidate_name),
            exp_root=exp_root,
            name_prefix=str(args.name_prefix),
        )

    tail_offsets = _parse_int_list(
        args.tail_offset_rounds,
        default_window_size=int(args.window_size_rounds),
        num_windows=int(args.num_windows),
    )
    lookbacks = _parse_positive_int_list(args.selector_lookbacks)
    margins_per_500 = _parse_float_list(args.selector_margins_per_500)
    skip_thresholds_per_500 = _parse_float_list(args.selector_skip_thresholds_per_500)
    rows: list[WindowRow] = []
    for tail_offset in tail_offsets:
        metrics: dict[str, ProfileMetric] = {}
        stageb_name = f"{args.name_prefix}_stageb_off{int(tail_offset):05d}"
        stageb_summary = _ensure_stageb_window(
            cwd=cwd,
            exp_root=exp_root,
            config_path=active_config_path,
            run_name=str(stageb_name),
            window_size_rounds=int(args.window_size_rounds),
            tail_offset_rounds=int(tail_offset),
            initial_bankroll_bnb=(None if args.initial_bankroll_bnb is None else float(args.initial_bankroll_bnb)),
            resume=(not bool(args.no_resume)),
        )
        metrics["stageb"] = ProfileMetric(
            per_500=_summary_per_500(stageb_summary),
            bet_rate=float(stageb_summary["bet_rate"]),
        )
        for profile in flow_profiles:
            flow_name = f"{args.name_prefix}_{profile.name}_off{int(tail_offset):05d}"
            val_size = int(args.window_size_rounds if profile.val_size is None else profile.val_size)
            step_size = int(args.window_size_rounds if profile.step_size is None else profile.step_size)
            flow_summary = _ensure_flow_window(
                cwd=cwd,
                exp_root=exp_root,
                config_path=active_config_path,
                run_name=str(flow_name),
                window_size_rounds=int(args.window_size_rounds),
                tail_offset_rounds=int(tail_offset),
                initial_bankroll_bnb=(None if args.initial_bankroll_bnb is None else float(args.initial_bankroll_bnb)),
                flow_train_size=int(profile.train_size),
                flow_val_size=int(val_size),
                flow_step_size=int(step_size),
                flow_ev_threshold=float(profile.ev_threshold),
                flow_min_total_pool_c=float(profile.min_total_pool_c),
                flow_allowed_sides=str(profile.allowed_sides),
                flow_bull_roll_edge_min=float(profile.bull_roll_edge_min),
                flow_bear_roll_edge_min=float(profile.bear_roll_edge_min),
                flow_bull_roll_winrate_min=float(profile.bull_roll_winrate_min),
                flow_bear_roll_winrate_min=float(profile.bear_roll_winrate_min),
                flow_bull_cooldown_trades=int(profile.bull_cooldown_trades),
                flow_bear_cooldown_trades=int(profile.bear_cooldown_trades),
                resume=(not bool(args.no_resume)),
            )
            metrics[str(profile.name)] = ProfileMetric(
                per_500=_summary_per_500(flow_summary),
                bet_rate=float(flow_summary["bet_rate"]),
            )
        for profile in dislocation_profiles:
            dislocation_name = f"{args.name_prefix}_{profile.name}_off{int(tail_offset):05d}"
            dislocation_summary = _ensure_stageb_window(
                cwd=cwd,
                exp_root=exp_root,
                config_path=dislocation_config_paths[str(profile.name)],
                run_name=str(dislocation_name),
                window_size_rounds=int(args.window_size_rounds),
                tail_offset_rounds=int(tail_offset),
                initial_bankroll_bnb=(None if args.initial_bankroll_bnb is None else float(args.initial_bankroll_bnb)),
                resume=(not bool(args.no_resume)),
            )
            metrics[str(profile.name)] = ProfileMetric(
                per_500=_summary_per_500(dislocation_summary),
                bet_rate=float(dislocation_summary["bet_rate"]),
            )
        rows.append(WindowRow(tail_offset_rounds=int(tail_offset), metrics=dict(metrics)))

    ordered_rows = _ordered_window_rows(rows)
    profile_names = ["stageb"] + [str(profile.name) for profile in flow_profiles] + [str(profile.name) for profile in dislocation_profiles]
    selector_rows = _evaluate_selectors(
        rows=ordered_rows,
        profile_names=profile_names,
        lookbacks=lookbacks,
        margins_per_500=margins_per_500,
        skip_thresholds_per_500=skip_thresholds_per_500,
        min_selected_bet_rate=float(args.min_selected_bet_rate),
    )

    compare_csv = exp_root / f"{args.name_prefix}_profile_set_window_compare.csv"
    compare_json = exp_root / f"{args.name_prefix}_profile_set_window_compare.json"
    selector_csv = exp_root / f"{args.name_prefix}_profile_set_window_selectors.csv"
    selector_json = exp_root / f"{args.name_prefix}_profile_set_window_selectors.json"

    with compare_csv.open("w", encoding="utf-8", newline="") as f:
        header = ["tail_offset_rounds"]
        for name in profile_names:
            header.append(f"{name}_per_500")
            header.append(f"{name}_bet_rate")
        f.write(",".join(header) + "\n")
        for row in ordered_rows:
            values = [str(int(row.tail_offset_rounds))]
            for name in profile_names:
                metric = row.metrics[str(name)]
                values.append(str(float(metric.per_500)))
                values.append(str(float(metric.bet_rate)))
            f.write(",".join(values) + "\n")
    compare_json.write_text(
        json.dumps(
            [
                {
                    "tail_offset_rounds": int(row.tail_offset_rounds),
                    "metrics": {
                        name: asdict(metric) for name, metric in sorted(row.metrics.items())
                    },
                }
                for row in ordered_rows
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with selector_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "mode,profile_name,lookback,margin_per_500,skip_threshold_per_500,"
            "mean_per_500,mean_selected_bet_rate,meets_min_selected_bet_rate,pick_counts_json\n"
        )
        for row in selector_rows:
            f.write(
                f"{row.mode},{row.profile_name},{int(row.lookback)},{float(row.margin_per_500)},"
                f"{float(row.skip_threshold_per_500)},{float(row.mean_per_500)},"
                f"{float(row.mean_selected_bet_rate)},{bool(row.meets_min_selected_bet_rate)},"
                f"\"{row.pick_counts_json.replace('\"', '\"\"')}\"\n"
            )
    selector_json.write_text(
        json.dumps([asdict(row) for row in selector_rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if selector_rows:
        top = selector_rows[0]
        print(
            json.dumps(
                {
                    "best_mode": str(top.mode),
                    "best_profile_name": str(top.profile_name),
                    "best_mean_per_500": float(top.mean_per_500),
                    "best_mean_selected_bet_rate": float(top.mean_selected_bet_rate),
                    "meets_min_selected_bet_rate": bool(top.meets_min_selected_bet_rate),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
