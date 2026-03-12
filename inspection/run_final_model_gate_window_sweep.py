from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import csv
import itertools
import json
import os
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from pancakebot.core.errors import InvariantError

from inspection.backtest_harness_common import (
    max_drawdown_bnb,
    render_table,
    resolve_exp_root,
    run_backtest_case,
    top_skip_reasons,
)


@dataclass(frozen=True, slots=True)
class GateVariant:
    name: str
    router_mode: str
    use_model_gate: bool
    projected_min_total_bnb: float | None = None
    perf_profile: str = "base"
    ml_train_size: int | None = None
    ml_calibrate_size: int | None = None
    ml_retrain_interval: int | None = None
    ml_recalibrate_interval: int | None = None
    ml_tag: str = "cfg"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--name-prefix", type=str, default="final_model_gate_window_sweep_20260303")
    p.add_argument("--sim-size", type=int, default=30000)
    p.add_argument("--offsets", type=str, default="0,5000,10000,15000,20000")
    p.add_argument("--thresholds", type=str, default="2,3,4,5,6,8,10")
    p.add_argument("--profiles", type=str, default="base,c2")
    p.add_argument("--drawdown-cap-bnb", type=float, default=2.0)
    p.add_argument("--initial-bankroll-bnb", type=float, default=None)
    p.add_argument("--top-skip-limit", type=int, default=4)
    p.add_argument("--top-selected-limit", type=int, default=4)
    p.add_argument("--run-long-confirm", action="store_true")
    p.add_argument("--long-sim-size", type=int, default=50984)
    p.add_argument("--top-k-confirm", type=int, default=6)
    p.add_argument("--no-resume", action="store_true")
    # 0 means "auto": use all available CPU cores.
    p.add_argument("--max-workers", type=int, default=0)
    # Memory guardrails for process parallelism. These prevent swap thrash.
    p.add_argument("--worker-memory-gb", type=float, default=2.0)
    p.add_argument("--reserve-memory-gb", type=float, default=2.0)
    p.add_argument(
        "--ml-enabled",
        type=str,
        default="",
        help="Optional bool override for strategy.ml_candidate.enabled (true/false).",
    )
    p.add_argument("--ml-train-sizes", type=str, default="")
    p.add_argument("--ml-calibrate-sizes", "--ml-calibration-sizes", dest="ml_calibrate_sizes", type=str, default="")
    p.add_argument("--ml-retrain-intervals", type=str, default="")
    p.add_argument(
        "--ml-recalibrate-intervals",
        "--ml-recalibration-intervals",
        dest="ml_recalibrate_intervals",
        type=str,
        default="",
    )
    return p


def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        try:
            vals.append(int(token))
        except ValueError as e:
            raise InvariantError(f"final_model_sweep_int_list_invalid: {token}") from e
    vals = sorted(set(vals))
    if not vals:
        raise InvariantError("final_model_sweep_int_list_empty")
    if any(int(x) < 0 for x in vals):
        raise InvariantError("final_model_sweep_int_list_negative")
    return vals


def _parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        try:
            vals.append(float(token))
        except ValueError as e:
            raise InvariantError(f"final_model_sweep_float_list_invalid: {token}") from e
    vals = sorted(set(vals))
    if not vals:
        raise InvariantError("final_model_sweep_float_list_empty")
    if any(float(x) <= 0.0 for x in vals):
        raise InvariantError("final_model_sweep_float_list_nonpositive")
    return vals


def _parse_profiles(raw: str) -> list[str]:
    vals = [str(x).strip() for x in str(raw).split(",") if str(x).strip()]
    if not vals:
        raise InvariantError("final_model_sweep_profiles_empty")
    out: list[str] = []
    for v in vals:
        if str(v) not in ("base", "c2"):
            raise InvariantError(f"final_model_sweep_profile_invalid: {v}")
        if str(v) not in out:
            out.append(str(v))
    return out


def _parse_optional_int_list(raw: str) -> list[int]:
    if str(raw).strip() == "":
        return []
    return _parse_int_list(str(raw))


def _parse_optional_bool_token(raw: str) -> bool | None:
    token = str(raw).strip()
    if token == "":
        return None
    lo = token.lower()
    if lo in ("true", "t", "1", "yes", "y", "on"):
        return True
    if lo in ("false", "f", "0", "no", "n", "off"):
        return False
    raise InvariantError(f"final_model_sweep_bool_token_invalid: {raw}")


def _load_cfg(config_path: str):
    from inspection.backtest_harness_common import load_cfg

    return load_cfg(config_path=str(config_path))


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
    ranked = sorted(counts.items(), key=lambda x: (-int(x[1]), str(x[0])))
    return "; ".join(f"{k}:{v}" for k, v in ranked[: int(limit)])


def _min_bankroll_bnb(*, trades_csv_path: Path) -> float:
    min_bankroll: float | None = None
    with Path(trades_csv_path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = float(row["bankroll_bnb"])
            if min_bankroll is None or float(b) < float(min_bankroll):
                min_bankroll = float(b)
    return 0.0 if min_bankroll is None else float(min_bankroll)


def _build_ml_knob_sets(*, cfg, args: argparse.Namespace) -> list[tuple[int, int, int, int, str]]:
    base = cfg.strategy.ml_candidate

    train_sizes = _parse_optional_int_list(str(args.ml_train_sizes))
    calibrate_sizes = _parse_optional_int_list(str(args.ml_calibrate_sizes))
    retrain_intervals = _parse_optional_int_list(str(args.ml_retrain_intervals))
    recalibrate_intervals = _parse_optional_int_list(str(args.ml_recalibrate_intervals))

    if not train_sizes:
        train_sizes = [int(base.train_size)]
    if not calibrate_sizes:
        calibrate_sizes = [int(base.calibrate_size)]
    if not retrain_intervals:
        retrain_intervals = [int(base.retrain_interval)]
    if not recalibrate_intervals:
        recalibrate_intervals = [int(base.recalibrate_interval)]

    explicit_ml_grid = any(
        str(v).strip() != ""
        for v in (
            args.ml_train_sizes,
            args.ml_calibrate_sizes,
            args.ml_retrain_intervals,
            args.ml_recalibrate_intervals,
        )
    )

    out: list[tuple[int, int, int, int, str]] = []
    for train_size, calibrate_size, retrain_interval, recalibrate_interval in itertools.product(
        train_sizes,
        calibrate_sizes,
        retrain_intervals,
        recalibrate_intervals,
    ):
        if int(train_size) <= 0:
            raise InvariantError("final_model_sweep_ml_train_size_nonpositive")
        if int(calibrate_size) < 0:
            raise InvariantError("final_model_sweep_ml_calibrate_size_negative")
        if int(retrain_interval) <= 0:
            raise InvariantError("final_model_sweep_ml_retrain_interval_nonpositive")
        if int(recalibrate_interval) < 0:
            raise InvariantError("final_model_sweep_ml_recalibrate_interval_negative")

        if not bool(explicit_ml_grid):
            tag = "cfg"
        else:
            tag = (
                f"ml_t{int(train_size)}"
                f"_c{int(calibrate_size)}"
                f"_rt{int(retrain_interval)}"
                f"_rc{int(recalibrate_interval)}"
            )
        out.append(
            (
                int(train_size),
                int(calibrate_size),
                int(retrain_interval),
                int(recalibrate_interval),
                str(tag),
            )
        )

    return out


def _default_variants(
    *,
    thresholds: list[float],
    profiles: list[str],
    ml_knob_sets: list[tuple[int, int, int, int, str]],
) -> list[GateVariant]:
    out: list[GateVariant] = []
    for train_size, calibrate_size, retrain_interval, recalibrate_interval, ml_tag in ml_knob_sets:
        for prof in profiles:
            suffix = "base" if str(prof) == "base" else "perf_c2"
            out.append(
                GateVariant(
                    name=f"cutoff_selector_{suffix}",
                    router_mode="selector_max_score",
                    use_model_gate=False,
                    projected_min_total_bnb=None,
                    perf_profile=str(prof),
                    ml_train_size=int(train_size),
                    ml_calibrate_size=int(calibrate_size),
                    ml_retrain_interval=int(retrain_interval),
                    ml_recalibrate_interval=int(recalibrate_interval),
                    ml_tag=str(ml_tag),
                )
            )

        for thr in thresholds:
            tag = f"p{str(thr).replace('.', 'p')}"
            for prof in profiles:
                suffix = "base" if str(prof) == "base" else "perf_c2"
                out.append(
                    GateVariant(
                        name=f"model_{tag}_selector_{suffix}",
                        router_mode="selector_max_score",
                        use_model_gate=True,
                        projected_min_total_bnb=float(thr),
                        perf_profile=str(prof),
                        ml_train_size=int(train_size),
                        ml_calibrate_size=int(calibrate_size),
                        ml_retrain_interval=int(retrain_interval),
                        ml_recalibrate_interval=int(recalibrate_interval),
                        ml_tag=str(ml_tag),
                    )
                )
    return out


def _available_physical_memory_bytes() -> int | None:
    # Windows: use GlobalMemoryStatusEx (no third-party dependency).
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        if int(ok) != 0:
            return int(stat.ullAvailPhys)
        return None

    # POSIX fallback.
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            if int(page_size) > 0 and int(avail_pages) > 0:
                return int(page_size) * int(avail_pages)
        except (OSError, ValueError):
            return None
    return None


def _safe_worker_cap(
    *,
    requested: int,
    total_jobs: int,
    worker_memory_gb: float,
    reserve_memory_gb: float,
) -> tuple[int, dict[str, int | float | None]]:
    if int(total_jobs) <= 0:
        return 1, {
            "cpu_count": int(os.cpu_count() or 1),
            "requested": int(requested),
            "total_jobs": int(total_jobs),
            "worker_memory_gb": float(worker_memory_gb),
            "reserve_memory_gb": float(reserve_memory_gb),
            "avail_memory_gb": None,
            "cpu_cap": 1,
            "memory_cap": 1,
            "final_cap": 1,
        }

    cpu = os.cpu_count() or 1
    if int(requested) <= 0:
        target = int(cpu)  # auto == all cores
    else:
        target = int(requested)
    cpu_cap = max(1, min(int(target), int(cpu), int(total_jobs)))

    avail_mem_bytes = _available_physical_memory_bytes()
    gb = float(1024**3)
    if avail_mem_bytes is None:
        memory_cap = int(total_jobs)
        avail_memory_gb = None
    else:
        worker_bytes = int(max(float(worker_memory_gb), 0.25) * float(gb))
        reserve_bytes = int(max(float(reserve_memory_gb), 0.0) * float(gb))
        budget_bytes = max(0, int(avail_mem_bytes) - int(reserve_bytes))
        if int(worker_bytes) <= 0:
            memory_cap = int(total_jobs)
        elif int(budget_bytes) <= 0:
            memory_cap = 1
        else:
            memory_cap = max(1, int(budget_bytes // int(worker_bytes)))
        avail_memory_gb = float(avail_mem_bytes) / float(gb)

    final_cap = max(1, min(int(cpu_cap), int(memory_cap), int(total_jobs)))
    return int(final_cap), {
        "cpu_count": int(cpu),
        "requested": int(requested),
        "total_jobs": int(total_jobs),
        "worker_memory_gb": float(worker_memory_gb),
        "reserve_memory_gb": float(reserve_memory_gb),
        "avail_memory_gb": (None if avail_memory_gb is None else float(avail_memory_gb)),
        "cpu_cap": int(cpu_cap),
        "memory_cap": int(memory_cap),
        "final_cap": int(final_cap),
    }


def _maybe_load_existing_summary(*, exp_root: Path, run_name: str) -> tuple[Path, Path, dict[str, object]] | None:
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
    return summary_path, trades_path, dict(summary)


def _mutate_candidate(*, c, variant: GateVariant):
    out = c
    if bool(variant.use_model_gate):
        if variant.projected_min_total_bnb is None:
            raise InvariantError("final_model_sweep_projected_min_missing")
        out = replace(
            out,
            pool_total_gate_mode="projected_final_model_only",
            projected_final_pool_total_min_bnb=float(variant.projected_min_total_bnb),
        )
    else:
        out = replace(out, pool_total_gate_mode="cutoff_only")

    if str(variant.perf_profile) == "c2":
        out = replace(
            out,
            perf_adapt_mode="skip",
            perf_gate_window=60,
            perf_gate_min_history=30,
            perf_gate_min_win_rate=0.54,
            perf_gate_min_mean_profit_bnb=0.0002,
        )
    elif str(variant.perf_profile) != "base":
        raise InvariantError(f"final_model_sweep_perf_profile_unknown: {variant.perf_profile}")
    return out


def _strategy_for_variant(*, cfg, variant: GateVariant, ml_enabled_override: bool | None = None):
    tuned = tuple(_mutate_candidate(c=c, variant=variant) for c in cfg.strategy.dislocation.candidates)
    dis_cfg = replace(cfg.strategy.dislocation, candidates=tuned)
    router_cfg = replace(cfg.strategy.router, mode=str(variant.router_mode))
    ml_cfg = replace(
        cfg.strategy.ml_candidate,
        enabled=(
            bool(cfg.strategy.ml_candidate.enabled)
            if ml_enabled_override is None
            else bool(ml_enabled_override)
        ),
        train_size=int(
            cfg.strategy.ml_candidate.train_size
            if variant.ml_train_size is None
            else variant.ml_train_size
        ),
        calibrate_size=int(
            cfg.strategy.ml_candidate.calibrate_size
            if variant.ml_calibrate_size is None
            else variant.ml_calibrate_size
        ),
        retrain_interval=int(
            cfg.strategy.ml_candidate.retrain_interval
            if variant.ml_retrain_interval is None
            else variant.ml_retrain_interval
        ),
        recalibrate_interval=int(
            cfg.strategy.ml_candidate.recalibrate_interval
            if variant.ml_recalibrate_interval is None
            else variant.ml_recalibrate_interval
        ),
    )
    return replace(cfg.strategy, dislocation=dis_cfg, router=router_cfg, ml_candidate=ml_cfg)


def _variant_label(*, variant: GateVariant) -> str:
    if str(variant.ml_tag) == "cfg":
        return str(variant.name)
    return f"{str(variant.name)}_{str(variant.ml_tag)}"


def _run_window_variant_job(job: dict[str, object]) -> dict[str, object]:
    config_path = str(job["config_path"])
    name_prefix = str(job["name_prefix"])
    exp_root = Path(str(job["exp_root"]))
    sim_size = int(job["sim_size"])
    offset = int(job["offset"])
    top_skip_limit = int(job["top_skip_limit"])
    top_selected_limit = int(job["top_selected_limit"])
    drawdown_cap_bnb = float(job["drawdown_cap_bnb"])
    ml_enabled_override_raw = job.get("ml_enabled_override")
    ml_enabled_override = None if ml_enabled_override_raw is None else bool(ml_enabled_override_raw)
    initial_bankroll_bnb = job.get("initial_bankroll_bnb")
    resume = bool(job["resume"])
    variant_raw = dict(job["variant"])
    variant = GateVariant(**variant_raw)

    cfg = _load_cfg(config_path=config_path)
    strategy_cfg = _strategy_for_variant(
        cfg=cfg,
        variant=variant,
        ml_enabled_override=ml_enabled_override,
    )
    variant_name = _variant_label(variant=variant)
    run_name = f"{str(name_prefix)}_off{int(offset)}_{str(variant_name)}_sim{int(sim_size)}"

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
            initial_bankroll_bnb=(None if initial_bankroll_bnb is None else float(initial_bankroll_bnb)),
            exp_root=exp_root,
        )
        summary_path = result.summary_path
        trades_path = result.trades_path
        summary = dict(result.summary)
        elapsed_seconds = float(result.elapsed_seconds)
    else:
        summary_path, trades_path, summary = existing
        elapsed_seconds = 0.0

    net = float(summary.get("net_profit_bnb", 0.0))
    per500 = float(net) * 500.0 / float(sim_size)
    max_dd = float(max_drawdown_bnb(trades_csv_path=trades_path))
    min_bank = float(_min_bankroll_bnb(trades_csv_path=trades_path))
    initial_bank = float(summary.get("initial_bankroll_bnb", 0.0))
    loss2 = float(initial_bank) - float(min_bank)

    return {
        "variant": str(variant_name),
        "variant_base": str(variant.name),
        "ml_tag": str(variant.ml_tag),
        "ml_train_size": int(strategy_cfg.ml_candidate.train_size),
        "ml_calibrate_size": int(strategy_cfg.ml_candidate.calibrate_size),
        "ml_retrain_interval": int(strategy_cfg.ml_candidate.retrain_interval),
        "ml_recalibrate_interval": int(strategy_cfg.ml_candidate.recalibrate_interval),
        "offset": int(offset),
        "sim_size": int(sim_size),
        "router_mode": str(strategy_cfg.router.mode),
        "model_gate": bool(variant.use_model_gate),
        "projected_min_total_bnb": (
            None if variant.projected_min_total_bnb is None else float(variant.projected_min_total_bnb)
        ),
        "perf_profile": str(variant.perf_profile),
        "net_profit_bnb": float(net),
        "per_500": float(per500),
        "num_bets": int(summary.get("num_bets", 0)),
        "bet_rate": float(summary.get("bet_rate", 0.0)),
        "max_drawdown_bnb": float(max_dd),
        "min_bankroll_bnb": float(min_bank),
        "initial_bankroll_bnb": float(initial_bank),
        "loss_from_initial_to_min_bnb": float(loss2),
        "meets_floor_cap": bool(float(loss2) <= float(drawdown_cap_bnb)),
        "top_skip_reasons": top_skip_reasons(summary=summary, limit=int(top_skip_limit)),
        "selected_strategy_mix": _selected_mix(
            trades_csv_path=trades_path,
            limit=int(top_selected_limit),
        ),
        "elapsed_seconds": float(elapsed_seconds),
        "summary_path": str(summary_path),
        "trades_path": str(trades_path),
    }


def _aggregate_rows(*, rows: list[dict[str, object]], drawdown_cap_bnb: float) -> list[dict[str, object]]:
    by_variant: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    out: list[dict[str, object]] = []
    for variant, variant_rows in by_variant.items():
        per500 = [float(r["per_500"]) for r in variant_rows]
        net = [float(r["net_profit_bnb"]) for r in variant_rows]
        loss2 = [float(r["loss_from_initial_to_min_bnb"]) for r in variant_rows]
        dd = [float(r["max_drawdown_bnb"]) for r in variant_rows]
        bets = [int(r["num_bets"]) for r in variant_rows]
        bet_rates = [float(r["bet_rate"]) for r in variant_rows]
        floor_ok = int(sum(1 for x in loss2 if float(x) <= float(drawdown_cap_bnb)))
        n = int(len(variant_rows))
        out.append(
            {
                "variant": str(variant),
                "n_windows": int(n),
                "floor_pass_count": int(floor_ok),
                "mean_per_500": float(statistics.mean(per500)),
                "median_per_500": float(statistics.median(per500)),
                "worst_per_500": float(min(per500)),
                "best_per_500": float(max(per500)),
                "mean_net_profit_bnb": float(statistics.mean(net)),
                "worst_net_profit_bnb": float(min(net)),
                "worst_loss_from_initial_to_min_bnb": float(max(loss2)),
                "worst_max_drawdown_bnb": float(max(dd)),
                "mean_num_bets": float(statistics.mean(bets)),
                "mean_bet_rate": float(statistics.mean(bet_rates)),
                "positive_count": int(sum(1 for x in per500 if float(x) > 0.0)),
            }
        )
    return sorted(
        out,
        key=lambda r: (
            -int(r["floor_pass_count"]),
            -float(r["worst_per_500"]),
            -float(r["mean_per_500"]),
            float(r["worst_max_drawdown_bnb"]),
        ),
    )


def _confirm_targets(*, agg: list[dict[str, object]], k: int) -> list[str]:
    if int(k) <= 0:
        return []
    out: list[str] = []
    for row in agg:
        if float(row["mean_per_500"]) <= 0.0:
            continue
        out.append(str(row["variant"]))
        if len(out) >= int(k):
            break
    return out


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    sim_size = int(args.sim_size)
    if int(sim_size) <= 0:
        raise InvariantError("final_model_sweep_sim_size_nonpositive")
    if float(args.drawdown_cap_bnb) < 0.0:
        raise InvariantError("final_model_sweep_drawdown_cap_negative")
    if int(args.top_skip_limit) <= 0 or int(args.top_selected_limit) <= 0:
        raise InvariantError("final_model_sweep_top_limits_nonpositive")
    if float(args.worker_memory_gb) <= 0.0:
        raise InvariantError("final_model_sweep_worker_memory_gb_nonpositive")
    if float(args.reserve_memory_gb) < 0.0:
        raise InvariantError("final_model_sweep_reserve_memory_gb_negative")

    offsets = _parse_int_list(str(args.offsets))
    thresholds = _parse_float_list(str(args.thresholds))
    profiles = _parse_profiles(str(args.profiles))
    ml_enabled_override = _parse_optional_bool_token(str(args.ml_enabled))
    ml_knob_sets = _build_ml_knob_sets(cfg=cfg, args=args)
    variants = _default_variants(
        thresholds=thresholds,
        profiles=profiles,
        ml_knob_sets=ml_knob_sets,
    )
    resume = not bool(args.no_resume)

    total_rounds = _count_jsonl_lines(Path(str(cfg.closed_rounds_path)))
    warmup_rounds = int(cfg.strategy.dislocation.selector.warmup_rounds)
    if int(warmup_rounds) <= 0:
        raise InvariantError("final_model_sweep_selector_warmup_nonpositive")
    max_sim_size = int(total_rounds) - int(warmup_rounds)
    if int(max_sim_size) <= 0:
        raise InvariantError("final_model_sweep_max_sim_size_nonpositive")
    if int(sim_size) > int(max_sim_size):
        raise InvariantError(
            f"final_model_sweep_sim_size_exceeds_max: sim_size={int(sim_size)} max={int(max_sim_size)}"
        )

    run_rows: list[dict[str, object]] = []
    jobs: list[dict[str, object]] = []

    for offset in offsets:
        needed = int(warmup_rounds) + int(sim_size)
        end_idx = int(total_rounds) - int(offset)
        start_idx = int(end_idx) - int(needed)
        if int(start_idx) < 0:
            continue

        for variant in variants:
            jobs.append(
                {
                    "config_path": str(args.config),
                    "name_prefix": str(args.name_prefix),
                    "exp_root": str(exp_root),
                    "sim_size": int(sim_size),
                    "offset": int(offset),
                    "top_skip_limit": int(args.top_skip_limit),
                    "top_selected_limit": int(args.top_selected_limit),
                    "drawdown_cap_bnb": float(args.drawdown_cap_bnb),
                    "ml_enabled_override": ml_enabled_override,
                    "initial_bankroll_bnb": (
                        None if args.initial_bankroll_bnb is None else float(args.initial_bankroll_bnb)
                    ),
                    "resume": bool(resume),
                    "variant": asdict(variant),
                }
            )

    worker_cap, worker_diag = _safe_worker_cap(
        requested=int(args.max_workers),
        total_jobs=int(len(jobs)),
        worker_memory_gb=float(args.worker_memory_gb),
        reserve_memory_gb=float(args.reserve_memory_gb),
    )
    avail_gb = worker_diag.get("avail_memory_gb")
    avail_gb_str = "unknown" if avail_gb is None else f"{float(avail_gb):.2f}"
    print(
        "WORKERS "
        + f"requested={int(worker_diag['requested'])} "
        + f"cpu_count={int(worker_diag['cpu_count'])} "
        + f"total_jobs={int(worker_diag['total_jobs'])} "
        + f"worker_mem_gb={float(worker_diag['worker_memory_gb']):.2f} "
        + f"reserve_mem_gb={float(worker_diag['reserve_memory_gb']):.2f} "
        + f"avail_mem_gb={avail_gb_str} "
        + f"cpu_cap={int(worker_diag['cpu_cap'])} "
        + f"mem_cap={int(worker_diag['memory_cap'])} "
        + f"final_cap={int(worker_diag['final_cap'])}"
    )
    if int(worker_cap) <= 1:
        for job in jobs:
            run_rows.append(_run_window_variant_job(dict(job)))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(worker_cap)) as ex:
            futs = [ex.submit(_run_window_variant_job, dict(job)) for job in jobs]
            for fut in concurrent.futures.as_completed(futs):
                run_rows.append(dict(fut.result()))

    run_rows.sort(key=lambda r: (int(r["offset"]), str(r["variant"])))

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
            raise InvariantError("final_model_sweep_long_sim_size_nonpositive")
        if int(long_sim_size) > int(max_sim_size):
            raise InvariantError(
                f"final_model_sweep_long_sim_size_exceeds_max: long_sim_size={int(long_sim_size)} max={int(max_sim_size)}"
            )
        targets = _confirm_targets(agg=agg_rows, k=int(args.top_k_confirm))
        by_name = {_variant_label(variant=v): v for v in variants}
        for name in targets:
            variant = by_name[str(name)]
            strategy_cfg = _strategy_for_variant(
                cfg=cfg,
                variant=variant,
                ml_enabled_override=ml_enabled_override,
            )
            run_name = f"{str(args.name_prefix)}_confirm_{str(name)}_sim{int(long_sim_size)}"
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
                    simulation_size=int(long_sim_size),
                    reset_mode="continuous",
                    reset_every_rounds=0,
                    initial_bankroll_bnb=args.initial_bankroll_bnb,
                    exp_root=exp_root,
                )
                summary_path = result.summary_path
                trades_path = result.trades_path
                summary = dict(result.summary)
            else:
                summary_path, trades_path, summary = existing

            net = float(summary.get("net_profit_bnb", 0.0))
            per500 = float(net) * 500.0 / float(long_sim_size)
            max_dd = float(max_drawdown_bnb(trades_csv_path=trades_path))
            min_bank = float(_min_bankroll_bnb(trades_csv_path=trades_path))
            initial_bank = float(summary.get("initial_bankroll_bnb", 0.0))
            loss2 = float(initial_bank) - float(min_bank)
            long_confirm_rows.append(
                {
                    "variant": str(name),
                    "sim_size": int(long_sim_size),
                    "net_profit_bnb": float(net),
                    "per_500": float(per500),
                    "num_bets": int(summary.get("num_bets", 0)),
                    "bet_rate": float(summary.get("bet_rate", 0.0)),
                    "max_drawdown_bnb": float(max_dd),
                    "min_bankroll_bnb": float(min_bank),
                    "loss_from_initial_to_min_bnb": float(loss2),
                    "meets_floor_cap": bool(float(loss2) <= float(args.drawdown_cap_bnb)),
                    "summary_path": str(summary_path),
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
                "thresholds": [float(x) for x in thresholds],
                "profiles": [str(x) for x in profiles],
                "ml_knob_sets": [
                    {
                        "train_size": int(t),
                        "calibrate_size": int(c),
                        "retrain_interval": int(rt),
                        "recalibrate_interval": int(rc),
                        "tag": str(tag),
                    }
                    for t, c, rt, rc, tag in ml_knob_sets
                ],
                "ml_enabled_override": ml_enabled_override,
                "worker_plan": dict(worker_diag),
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
                "variant_base",
                "ml_tag",
                "ml_train_size",
                "ml_calibrate_size",
                "ml_retrain_interval",
                "ml_recalibrate_interval",
                "offset",
                "sim_size",
                "router_mode",
                "model_gate",
                "projected_min_total_bnb",
                "perf_profile",
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
