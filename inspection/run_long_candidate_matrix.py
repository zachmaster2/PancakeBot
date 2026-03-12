from __future__ import annotations

import argparse
import csv
import json
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

_ALT_A_NAME = "disloc_altA_20260227_x80"
_ALT_B_NAME = "disloc_altB_20260227_x80"


@dataclass(frozen=True, slots=True)
class VariantSpec:
    name: str
    candidate_names: tuple[str, ...] | None
    router_mode: str
    stake_scale: float = 1.0
    model_gate_min_total_bnb: float | None = None
    candidate_expected_net_min_bnb: float | None = None
    selector_score_threshold: float | None = None
    online_score_threshold_bnb: float | None = None
    ml_enabled: bool | None = None
    ml_train_size: int | None = None
    ml_calibrate_size: int | None = None
    ml_retrain_interval: int | None = None
    ml_recalibrate_interval: int | None = None
    anti_martingale_enabled: bool = False
    anti_martingale_win_multiplier: float = 1.15
    anti_martingale_loss_multiplier: float = 0.9
    anti_martingale_min_scale: float = 0.5
    anti_martingale_max_scale: float = 1.5
    circuit_breaker_enabled: bool = False
    circuit_breaker_drawdown_trigger_bnb: float = 0.0
    circuit_breaker_base_skip_rounds: int = 0
    circuit_breaker_escalation_multiplier: float = 1.5
    circuit_breaker_escalation_window_rounds: int = 200
    circuit_breaker_max_level: int = 6
    circuit_breaker_max_skip_rounds: int = 0
    circuit_breaker_reentry_rounds: int = 0
    circuit_breaker_reentry_scale: float = 1.0
    max_sim_size: int | None = None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--name-prefix", type=str, default="long_candidate_matrix_20260303")
    p.add_argument("--sim-sizes", type=str, default="30000,50000")
    p.add_argument("--drawdown-cap-bnb", type=float, default=2.0)
    p.add_argument("--initial-bankroll-bnb", type=float, default=None)
    p.add_argument("--include-model-gate", action="store_true")
    p.add_argument("--priority-stake-sweep", action="store_true")
    p.add_argument("--priority-risk-adapt-sweep", action="store_true")
    p.add_argument("--priority-ml-knob-sweep", action="store_true")
    p.add_argument("--priority-frequency-sweep", action="store_true")
    p.add_argument("--stake-scales", type=str, default="0.5,0.75,1.0,1.25,1.5,2.0")
    p.add_argument("--model-gate-min-totals", type=str, default="0.5")
    p.add_argument("--ml-train-sizes", type=str, default="")
    p.add_argument("--ml-calibrate-sizes", type=str, default="")
    p.add_argument("--ml-retrain-intervals", type=str, default="")
    p.add_argument("--ml-recalibrate-intervals", type=str, default="")
    p.add_argument("--ml-enabled-values", type=str, default="true")
    p.add_argument("--freq-expected-net-mins", type=str, default="0.18,0.12,0.08,0.04,0.00")
    p.add_argument("--freq-selector-thresholds", type=str, default="-0.01,-0.02,-0.05")
    p.add_argument("--freq-online-thresholds", type=str, default="0.0,-0.001,-0.003")
    p.add_argument("--freq-stake-scales", type=str, default="0.25,0.50,1.00")
    p.add_argument("--freq-model-gate-min-totals", type=str, default="0.5")
    p.add_argument("--cb-triggers", type=str, default="1.0,1.5")
    p.add_argument("--cb-base-skips", type=str, default="80,120")
    p.add_argument("--anti-max-scales", type=str, default="1.25,1.5")
    p.add_argument("--anti-win-multiplier", type=float, default=1.15)
    p.add_argument("--anti-loss-multiplier", type=float, default=0.9)
    p.add_argument("--anti-min-scale", type=float, default=0.5)
    p.add_argument("--cb-escalation-multiplier", type=float, default=1.5)
    p.add_argument("--cb-escalation-window-rounds", type=int, default=200)
    p.add_argument("--cb-max-level", type=int, default=6)
    p.add_argument("--cb-max-skip-rounds", type=int, default=400)
    p.add_argument("--cb-reentry-rounds", type=int, default=40)
    p.add_argument("--cb-reentry-scale", type=float, default=0.75)
    p.add_argument("--top-skip-limit", type=int, default=4)
    p.add_argument("--top-selected-limit", type=int, default=4)
    return p


def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip() != ""]:
        try:
            vals.append(int(token))
        except ValueError as e:
            raise InvariantError(f"long_matrix_int_list_invalid: {token}") from e
    vals = sorted(set(vals))
    if not vals:
        raise InvariantError("long_matrix_int_list_empty")
    if any(int(x) <= 0 for x in vals):
        raise InvariantError("long_matrix_sim_size_nonpositive")
    return vals


def _parse_float_list(raw: str, *, allow_zero: bool = False) -> list[float]:
    vals: list[float] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip() != ""]:
        try:
            vals.append(float(token))
        except ValueError as e:
            raise InvariantError(f"long_matrix_float_list_invalid: {token}") from e
    vals = sorted(set(vals))
    if not vals:
        raise InvariantError("long_matrix_float_list_empty")
    if bool(allow_zero):
        if any(float(x) < 0.0 for x in vals):
            raise InvariantError("long_matrix_float_list_negative")
    else:
        if any(float(x) <= 0.0 for x in vals):
            raise InvariantError("long_matrix_float_list_nonpositive")
    return vals


def _parse_signed_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip() != ""]:
        try:
            vals.append(float(token))
        except ValueError as e:
            raise InvariantError(f"long_matrix_float_list_invalid: {token}") from e
    vals = sorted(set(vals))
    if not vals:
        raise InvariantError("long_matrix_float_list_empty")
    return vals


def _parse_bool_list(raw: str) -> list[bool]:
    vals: list[bool] = []
    for token in [x.strip().lower() for x in str(raw).split(",") if x.strip() != ""]:
        if token in ("true", "t", "1", "yes", "y", "on"):
            vals.append(True)
            continue
        if token in ("false", "f", "0", "no", "n", "off"):
            vals.append(False)
            continue
        raise InvariantError(f"long_matrix_bool_list_invalid: {token}")
    vals = list(dict.fromkeys(vals))
    if not vals:
        raise InvariantError("long_matrix_bool_list_empty")
    return vals


def _count_jsonl_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as f:
        return int(sum(1 for _ in f))


def _candidate_map(*, cfg) -> dict[str, object]:
    out: dict[str, object] = {}
    for c in cfg.strategy.dislocation.candidates:
        out[str(c.name)] = c
    return out


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


def _min_bankroll_bnb(*, trades_csv_path: Path) -> float:
    min_bankroll: float | None = None
    with Path(trades_csv_path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bankroll = float(row["bankroll_bnb"])
            if min_bankroll is None or float(bankroll) < float(min_bankroll):
                min_bankroll = float(bankroll)
    return 0.0 if min_bankroll is None else float(min_bankroll)


def _scaled_candidate(*, candidate, scale: float):
    if float(scale) <= 0.0:
        raise InvariantError("long_matrix_stake_scale_nonpositive")
    return replace(
        candidate,
        fixed_bet_bnb=float(candidate.fixed_bet_bnb) * float(scale),
        expected_net_min_bnb=float(candidate.expected_net_min_bnb) * float(scale),
        stake_min_bnb=float(candidate.stake_min_bnb) * float(scale),
        stake_max_bnb=float(candidate.stake_max_bnb) * float(scale),
        stake_ev_ref_bnb=float(candidate.stake_ev_ref_bnb) * float(scale),
    )


def _model_gate_candidate(*, candidate, projected_min_bnb: float):
    if float(projected_min_bnb) < 0.0:
        raise InvariantError("long_matrix_model_gate_projected_min_negative")
    return replace(
        candidate,
        pool_total_gate_mode="projected_final_model_only",
        projected_final_pool_total_min_bnb=float(projected_min_bnb),
    )


def _expected_net_candidate(*, candidate, expected_net_min_bnb: float):
    if float(expected_net_min_bnb) < 0.0:
        raise InvariantError("long_matrix_expected_net_min_negative")
    return replace(
        candidate,
        expected_net_min_bnb=float(expected_net_min_bnb),
    )


def _risk_adapt_candidate(*, candidate, variant: VariantSpec):
    tuned = replace(
        candidate,
        anti_martingale_enabled=bool(variant.anti_martingale_enabled),
        anti_martingale_win_multiplier=float(variant.anti_martingale_win_multiplier),
        anti_martingale_loss_multiplier=float(variant.anti_martingale_loss_multiplier),
        anti_martingale_min_scale=float(variant.anti_martingale_min_scale),
        anti_martingale_max_scale=float(variant.anti_martingale_max_scale),
        circuit_breaker_enabled=bool(variant.circuit_breaker_enabled),
        circuit_breaker_drawdown_trigger_bnb=float(variant.circuit_breaker_drawdown_trigger_bnb),
        circuit_breaker_base_skip_rounds=int(variant.circuit_breaker_base_skip_rounds),
        circuit_breaker_escalation_multiplier=float(variant.circuit_breaker_escalation_multiplier),
        circuit_breaker_escalation_window_rounds=int(variant.circuit_breaker_escalation_window_rounds),
        circuit_breaker_max_level=int(variant.circuit_breaker_max_level),
        circuit_breaker_max_skip_rounds=int(variant.circuit_breaker_max_skip_rounds),
        circuit_breaker_reentry_rounds=int(variant.circuit_breaker_reentry_rounds),
        circuit_breaker_reentry_scale=float(variant.circuit_breaker_reentry_scale),
    )
    return tuned


def _default_variants(*, include_model_gate: bool) -> list[VariantSpec]:
    rows: list[VariantSpec] = [
        VariantSpec(
            name="baseline_full_online",
            candidate_names=None,
            router_mode="online_cellmean",
            stake_scale=1.0,
            model_gate_min_total_bnb=None,
        ),
        VariantSpec(
            name="prune_altAB_online",
            candidate_names=(_ALT_A_NAME, _ALT_B_NAME),
            router_mode="online_cellmean",
            stake_scale=1.0,
            model_gate_min_total_bnb=None,
        ),
        VariantSpec(
            name="prune_altAB_selector",
            candidate_names=(_ALT_A_NAME, _ALT_B_NAME),
            router_mode="selector_max_score",
            stake_scale=1.0,
            model_gate_min_total_bnb=None,
        ),
        VariantSpec(
            name="prune_altAB_online_scale075",
            candidate_names=(_ALT_A_NAME, _ALT_B_NAME),
            router_mode="online_cellmean",
            stake_scale=0.75,
            model_gate_min_total_bnb=None,
        ),
        VariantSpec(
            name="prune_altAB_online_scale050",
            candidate_names=(_ALT_A_NAME, _ALT_B_NAME),
            router_mode="online_cellmean",
            stake_scale=0.50,
            model_gate_min_total_bnb=None,
        ),
        VariantSpec(
            name="prune_altAB_selector_scale050",
            candidate_names=(_ALT_A_NAME, _ALT_B_NAME),
            router_mode="selector_max_score",
            stake_scale=0.50,
            model_gate_min_total_bnb=None,
        ),
    ]
    if bool(include_model_gate):
        rows.extend(
            [
                VariantSpec(
                    name="prune_altAB_online_scale050_modelgate_p2",
                    candidate_names=(_ALT_A_NAME, _ALT_B_NAME),
                    router_mode="online_cellmean",
                    stake_scale=0.50,
                    model_gate_min_total_bnb=2.0,
                    max_sim_size=50000,
                ),
                VariantSpec(
                    name="prune_altAB_online_scale100_modelgate_p2",
                    candidate_names=(_ALT_A_NAME, _ALT_B_NAME),
                    router_mode="online_cellmean",
                    stake_scale=1.0,
                    model_gate_min_total_bnb=2.0,
                    max_sim_size=50000,
                ),
            ]
        )
    return rows


def _priority_stake_variants(
    *,
    stake_scales: list[float],
    model_gate_min_totals: list[float],
) -> list[VariantSpec]:
    rows: list[VariantSpec] = []

    for scale in stake_scales:
        s_tag = str(scale).replace(".", "p")
        rows.append(
            VariantSpec(
                name=f"core_online_scale{s_tag}",
                candidate_names=None,
                router_mode="online_cellmean",
                stake_scale=float(scale),
                model_gate_min_total_bnb=None,
            )
        )

    for min_total in model_gate_min_totals:
        m_tag = str(min_total).replace(".", "p")
        for scale in stake_scales:
            s_tag = str(scale).replace(".", "p")
            rows.append(
                VariantSpec(
                    name=f"modelgate_p{m_tag}_selector_scale{s_tag}",
                    candidate_names=None,
                    router_mode="selector_max_score",
                    stake_scale=float(scale),
                    model_gate_min_total_bnb=float(min_total),
                )
            )

    return rows


def _priority_risk_adapt_variants(
    *,
    cb_triggers: list[float],
    cb_base_skips: list[int],
    anti_max_scales: list[float],
    anti_win_multiplier: float,
    anti_loss_multiplier: float,
    anti_min_scale: float,
    cb_escalation_multiplier: float,
    cb_escalation_window_rounds: int,
    cb_max_level: int,
    cb_max_skip_rounds: int,
    cb_reentry_rounds: int,
    cb_reentry_scale: float,
) -> list[VariantSpec]:
    rows: list[VariantSpec] = [
        VariantSpec(
            name="baseline_core_online",
            candidate_names=None,
            router_mode="online_cellmean",
            stake_scale=1.0,
        ),
        VariantSpec(
            name="baseline_modelgate_selector_p0p5",
            candidate_names=None,
            router_mode="selector_max_score",
            stake_scale=1.0,
            model_gate_min_total_bnb=0.5,
        ),
    ]

    for trig in cb_triggers:
        t_tag = str(trig).replace(".", "p")
        for base_skip in cb_base_skips:
            b_tag = str(int(base_skip))
            rows.append(
                VariantSpec(
                    name=f"core_online_cb_t{t_tag}_n{b_tag}",
                    candidate_names=None,
                    router_mode="online_cellmean",
                    stake_scale=1.0,
                    circuit_breaker_enabled=True,
                    circuit_breaker_drawdown_trigger_bnb=float(trig),
                    circuit_breaker_base_skip_rounds=int(base_skip),
                    circuit_breaker_escalation_multiplier=float(cb_escalation_multiplier),
                    circuit_breaker_escalation_window_rounds=int(cb_escalation_window_rounds),
                    circuit_breaker_max_level=int(cb_max_level),
                    circuit_breaker_max_skip_rounds=int(cb_max_skip_rounds),
                    circuit_breaker_reentry_rounds=int(cb_reentry_rounds),
                    circuit_breaker_reentry_scale=float(cb_reentry_scale),
                )
            )
            rows.append(
                VariantSpec(
                    name=f"modelgate_selector_p0p5_cb_t{t_tag}_n{b_tag}",
                    candidate_names=None,
                    router_mode="selector_max_score",
                    stake_scale=1.0,
                    model_gate_min_total_bnb=0.5,
                    circuit_breaker_enabled=True,
                    circuit_breaker_drawdown_trigger_bnb=float(trig),
                    circuit_breaker_base_skip_rounds=int(base_skip),
                    circuit_breaker_escalation_multiplier=float(cb_escalation_multiplier),
                    circuit_breaker_escalation_window_rounds=int(cb_escalation_window_rounds),
                    circuit_breaker_max_level=int(cb_max_level),
                    circuit_breaker_max_skip_rounds=int(cb_max_skip_rounds),
                    circuit_breaker_reentry_rounds=int(cb_reentry_rounds),
                    circuit_breaker_reentry_scale=float(cb_reentry_scale),
                )
            )
            for anti_max in anti_max_scales:
                a_tag = str(anti_max).replace(".", "p")
                rows.append(
                    VariantSpec(
                        name=f"core_online_cb_t{t_tag}_n{b_tag}_anti_a{a_tag}",
                        candidate_names=None,
                        router_mode="online_cellmean",
                        stake_scale=1.0,
                        anti_martingale_enabled=True,
                        anti_martingale_win_multiplier=float(anti_win_multiplier),
                        anti_martingale_loss_multiplier=float(anti_loss_multiplier),
                        anti_martingale_min_scale=float(anti_min_scale),
                        anti_martingale_max_scale=float(anti_max),
                        circuit_breaker_enabled=True,
                        circuit_breaker_drawdown_trigger_bnb=float(trig),
                        circuit_breaker_base_skip_rounds=int(base_skip),
                        circuit_breaker_escalation_multiplier=float(cb_escalation_multiplier),
                        circuit_breaker_escalation_window_rounds=int(cb_escalation_window_rounds),
                        circuit_breaker_max_level=int(cb_max_level),
                        circuit_breaker_max_skip_rounds=int(cb_max_skip_rounds),
                        circuit_breaker_reentry_rounds=int(cb_reentry_rounds),
                        circuit_breaker_reentry_scale=float(cb_reentry_scale),
                    )
                )
                rows.append(
                    VariantSpec(
                        name=f"modelgate_selector_p0p5_cb_t{t_tag}_n{b_tag}_anti_a{a_tag}",
                        candidate_names=None,
                        router_mode="selector_max_score",
                        stake_scale=1.0,
                        model_gate_min_total_bnb=0.5,
                        anti_martingale_enabled=True,
                        anti_martingale_win_multiplier=float(anti_win_multiplier),
                        anti_martingale_loss_multiplier=float(anti_loss_multiplier),
                        anti_martingale_min_scale=float(anti_min_scale),
                        anti_martingale_max_scale=float(anti_max),
                        circuit_breaker_enabled=True,
                        circuit_breaker_drawdown_trigger_bnb=float(trig),
                        circuit_breaker_base_skip_rounds=int(base_skip),
                        circuit_breaker_escalation_multiplier=float(cb_escalation_multiplier),
                        circuit_breaker_escalation_window_rounds=int(cb_escalation_window_rounds),
                        circuit_breaker_max_level=int(cb_max_level),
                        circuit_breaker_max_skip_rounds=int(cb_max_skip_rounds),
                        circuit_breaker_reentry_rounds=int(cb_reentry_rounds),
                        circuit_breaker_reentry_scale=float(cb_reentry_scale),
                    )
                )

    return rows


def _priority_ml_knob_variants(
    *,
    ml_enabled_values: list[bool],
    train_sizes: list[int],
    calibrate_sizes: list[int],
    retrain_intervals: list[int],
    recalibrate_intervals: list[int],
) -> list[VariantSpec]:
    rows: list[VariantSpec] = []
    seen: set[tuple[object, ...]] = set()

    for enabled in ml_enabled_values:
        enabled_tag = "on" if bool(enabled) else "off"
        for train_size in train_sizes:
            for calibrate_size in calibrate_sizes:
                for retrain_interval in retrain_intervals:
                    for recalibrate_interval in recalibrate_intervals:
                        key = (
                            bool(enabled),
                            int(train_size),
                            int(calibrate_size),
                            int(retrain_interval),
                            int(recalibrate_interval),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        t_tag = str(int(train_size))
                        c_tag = str(int(calibrate_size))
                        rt_tag = str(int(retrain_interval))
                        rc_tag = str(int(recalibrate_interval))
                        suffix = f"ml_{enabled_tag}_t{t_tag}_c{c_tag}_rt{rt_tag}_rc{rc_tag}"
                        rows.append(
                            VariantSpec(
                                name=f"core_online_{suffix}",
                                candidate_names=None,
                                router_mode="online_cellmean",
                                stake_scale=1.0,
                                ml_enabled=bool(enabled),
                                ml_train_size=int(train_size),
                                ml_calibrate_size=int(calibrate_size),
                                ml_retrain_interval=int(retrain_interval),
                                ml_recalibrate_interval=int(recalibrate_interval),
                            )
                        )
                        rows.append(
                            VariantSpec(
                                name=f"modelgate_selector_p0p5_{suffix}",
                                candidate_names=None,
                                router_mode="selector_max_score",
                                stake_scale=1.0,
                                model_gate_min_total_bnb=0.5,
                                ml_enabled=bool(enabled),
                                ml_train_size=int(train_size),
                                ml_calibrate_size=int(calibrate_size),
                                ml_retrain_interval=int(retrain_interval),
                                ml_recalibrate_interval=int(recalibrate_interval),
                            )
                        )
    return rows


def _priority_frequency_variants(
    *,
    expected_net_mins: list[float],
    selector_thresholds: list[float],
    online_thresholds: list[float],
    stake_scales: list[float],
    model_gate_min_totals: list[float],
) -> list[VariantSpec]:
    rows: list[VariantSpec] = []
    rows.append(
        VariantSpec(
            name="baseline_core_online",
            candidate_names=None,
            router_mode="online_cellmean",
            stake_scale=1.0,
        )
    )
    rows.append(
        VariantSpec(
            name="baseline_modelgate_selector_p0p5",
            candidate_names=None,
            router_mode="selector_max_score",
            stake_scale=1.0,
            model_gate_min_total_bnb=0.5,
        )
    )

    for expected_net_min in expected_net_mins:
        e_tag = str(expected_net_min).replace(".", "p")
        for stake_scale in stake_scales:
            s_tag = str(stake_scale).replace(".", "p")
            for online_thr in online_thresholds:
                o_tag = str(online_thr).replace(".", "p").replace("-", "m")
                rows.append(
                    VariantSpec(
                        name=f"core_online_freq_e{e_tag}_s{s_tag}_othr_{o_tag}",
                        candidate_names=None,
                        router_mode="online_cellmean",
                        stake_scale=float(stake_scale),
                        candidate_expected_net_min_bnb=float(expected_net_min),
                        online_score_threshold_bnb=float(online_thr),
                    )
                )
            for selector_thr in selector_thresholds:
                sel_tag = str(selector_thr).replace(".", "p").replace("-", "m")
                for model_gate_min in model_gate_min_totals:
                    m_tag = str(model_gate_min).replace(".", "p")
                    rows.append(
                        VariantSpec(
                            name=f"modelgate_selector_freq_e{e_tag}_s{s_tag}_sth_{sel_tag}_m{m_tag}",
                            candidate_names=None,
                            router_mode="selector_max_score",
                            stake_scale=float(stake_scale),
                            model_gate_min_total_bnb=float(model_gate_min),
                            candidate_expected_net_min_bnb=float(expected_net_min),
                            selector_score_threshold=float(selector_thr),
                        )
                    )

    return rows


def _strategy_for_variant(*, cfg, variant: VariantSpec):
    base_strategy = cfg.strategy
    c_map = _candidate_map(cfg=cfg)

    if variant.candidate_names is None:
        selected = list(base_strategy.dislocation.candidates)
    else:
        selected = []
        for name in variant.candidate_names:
            if str(name) not in c_map:
                raise InvariantError(f"long_matrix_candidate_missing: {name}")
            selected.append(c_map[str(name)])

    tuned_candidates = []
    for c in selected:
        tuned = c
        if float(variant.stake_scale) != 1.0:
            tuned = _scaled_candidate(candidate=tuned, scale=float(variant.stake_scale))
        if variant.candidate_expected_net_min_bnb is not None:
            tuned = _expected_net_candidate(
                candidate=tuned,
                expected_net_min_bnb=float(variant.candidate_expected_net_min_bnb),
            )
        if variant.model_gate_min_total_bnb is not None:
            tuned = _model_gate_candidate(
                candidate=tuned,
                projected_min_bnb=float(variant.model_gate_min_total_bnb),
            )
        tuned = _risk_adapt_candidate(candidate=tuned, variant=variant)
        tuned_candidates.append(tuned)

    dislocation_cfg = replace(base_strategy.dislocation, candidates=tuple(tuned_candidates))
    router_cfg = replace(
        base_strategy.router,
        mode=str(variant.router_mode),
        score_threshold_bnb=(
            float(base_strategy.router.score_threshold_bnb)
            if variant.selector_score_threshold is None
            else float(variant.selector_score_threshold)
        ),
        online_score_threshold_bnb=(
            float(base_strategy.router.online_score_threshold_bnb)
            if variant.online_score_threshold_bnb is None
            else float(variant.online_score_threshold_bnb)
        ),
    )
    ml_cfg = replace(
        base_strategy.ml_candidate,
        enabled=(
            bool(base_strategy.ml_candidate.enabled)
            if variant.ml_enabled is None
            else bool(variant.ml_enabled)
        ),
        train_size=(
            int(base_strategy.ml_candidate.train_size)
            if variant.ml_train_size is None
            else int(variant.ml_train_size)
        ),
        calibrate_size=(
            int(base_strategy.ml_candidate.calibrate_size)
            if variant.ml_calibrate_size is None
            else int(variant.ml_calibrate_size)
        ),
        retrain_interval=(
            int(base_strategy.ml_candidate.retrain_interval)
            if variant.ml_retrain_interval is None
            else int(variant.ml_retrain_interval)
        ),
        recalibrate_interval=(
            int(base_strategy.ml_candidate.recalibrate_interval)
            if variant.ml_recalibrate_interval is None
            else int(variant.ml_recalibrate_interval)
        ),
    )
    return replace(base_strategy, dislocation=dislocation_cfg, router=router_cfg, ml_candidate=ml_cfg)


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    sim_sizes = _parse_int_list(str(args.sim_sizes))
    stake_scales = _parse_float_list(str(args.stake_scales), allow_zero=False)
    model_gate_min_totals = _parse_float_list(str(args.model_gate_min_totals), allow_zero=True)
    freq_expected_net_mins = _parse_float_list(str(args.freq_expected_net_mins), allow_zero=True)
    freq_selector_thresholds = _parse_signed_float_list(str(args.freq_selector_thresholds))
    freq_online_thresholds = _parse_signed_float_list(str(args.freq_online_thresholds))
    freq_stake_scales = _parse_float_list(str(args.freq_stake_scales), allow_zero=False)
    freq_model_gate_min_totals = _parse_float_list(
        str(args.freq_model_gate_min_totals),
        allow_zero=True,
    )
    cb_triggers = _parse_float_list(str(args.cb_triggers), allow_zero=False)
    cb_base_skips = _parse_int_list(str(args.cb_base_skips))
    anti_max_scales = _parse_float_list(str(args.anti_max_scales), allow_zero=False)
    ml_enabled_values = _parse_bool_list(str(args.ml_enabled_values))
    base_ml = cfg.strategy.ml_candidate
    ml_train_sizes = (
        [int(base_ml.train_size)]
        if str(args.ml_train_sizes).strip() == ""
        else _parse_int_list(str(args.ml_train_sizes))
    )
    ml_calibrate_sizes = (
        [int(base_ml.calibrate_size)]
        if str(args.ml_calibrate_sizes).strip() == ""
        else _parse_int_list(str(args.ml_calibrate_sizes))
    )
    ml_retrain_intervals = (
        [int(base_ml.retrain_interval)]
        if str(args.ml_retrain_intervals).strip() == ""
        else _parse_int_list(str(args.ml_retrain_intervals))
    )
    ml_recalibrate_intervals = (
        [int(base_ml.recalibrate_interval)]
        if str(args.ml_recalibrate_intervals).strip() == ""
        else _parse_int_list(str(args.ml_recalibrate_intervals))
    )
    sweep_flags = [
        bool(args.priority_stake_sweep),
        bool(args.priority_risk_adapt_sweep),
        bool(args.priority_ml_knob_sweep),
        bool(args.priority_frequency_sweep),
    ]
    if int(sum(1 for x in sweep_flags if bool(x))) > 1:
        raise InvariantError("long_matrix_multiple_priority_sweeps_enabled")
    if int(args.top_skip_limit) <= 0 or int(args.top_selected_limit) <= 0:
        raise InvariantError("long_matrix_top_limits_nonpositive")
    if float(args.anti_win_multiplier) <= 0.0 or float(args.anti_loss_multiplier) <= 0.0:
        raise InvariantError("long_matrix_anti_multiplier_nonpositive")
    if float(args.anti_min_scale) <= 0.0:
        raise InvariantError("long_matrix_anti_min_scale_nonpositive")
    if any(float(mx) < float(args.anti_min_scale) for mx in anti_max_scales):
        raise InvariantError("long_matrix_anti_scale_bounds_invalid")
    if float(args.cb_escalation_multiplier) < 1.0:
        raise InvariantError("long_matrix_cb_escalation_multiplier_invalid")
    if int(args.cb_escalation_window_rounds) <= 0:
        raise InvariantError("long_matrix_cb_escalation_window_nonpositive")
    if int(args.cb_max_level) <= 0:
        raise InvariantError("long_matrix_cb_max_level_nonpositive")
    if int(args.cb_max_skip_rounds) < 0:
        raise InvariantError("long_matrix_cb_max_skip_rounds_negative")
    if int(args.cb_reentry_rounds) < 0:
        raise InvariantError("long_matrix_cb_reentry_rounds_negative")
    if not (0.0 < float(args.cb_reentry_scale) <= 1.0):
        raise InvariantError("long_matrix_cb_reentry_scale_out_of_range")
    if any(int(x) <= 0 for x in ml_train_sizes):
        raise InvariantError("long_matrix_ml_train_size_nonpositive")
    if any(int(x) <= 0 for x in ml_calibrate_sizes):
        raise InvariantError("long_matrix_ml_calibrate_size_nonpositive")
    if any(int(x) <= 0 for x in ml_retrain_intervals):
        raise InvariantError("long_matrix_ml_retrain_interval_nonpositive")
    if any(int(x) <= 0 for x in ml_recalibrate_intervals):
        raise InvariantError("long_matrix_ml_recalibrate_interval_nonpositive")

    closed_rounds_path = Path(str(cfg.closed_rounds_path))
    total_rounds = _count_jsonl_lines(closed_rounds_path)
    warmup_rounds = int(cfg.strategy.dislocation.selector.warmup_rounds)
    if int(warmup_rounds) <= 0:
        raise InvariantError("long_matrix_selector_warmup_nonpositive")
    max_sim_size = int(total_rounds) - int(warmup_rounds)
    if int(max_sim_size) <= 0:
        raise InvariantError("long_matrix_sim_impossible_with_current_history")
    if any(int(n) > int(max_sim_size) for n in sim_sizes):
        raise InvariantError(
            f"long_matrix_sim_size_exceeds_max: max={int(max_sim_size)} requested={sim_sizes}"
        )

    if bool(args.priority_risk_adapt_sweep):
        variants = _priority_risk_adapt_variants(
            cb_triggers=cb_triggers,
            cb_base_skips=cb_base_skips,
            anti_max_scales=anti_max_scales,
            anti_win_multiplier=float(args.anti_win_multiplier),
            anti_loss_multiplier=float(args.anti_loss_multiplier),
            anti_min_scale=float(args.anti_min_scale),
            cb_escalation_multiplier=float(args.cb_escalation_multiplier),
            cb_escalation_window_rounds=int(args.cb_escalation_window_rounds),
            cb_max_level=int(args.cb_max_level),
            cb_max_skip_rounds=int(args.cb_max_skip_rounds),
            cb_reentry_rounds=int(args.cb_reentry_rounds),
            cb_reentry_scale=float(args.cb_reentry_scale),
        )
    elif bool(args.priority_stake_sweep):
        variants = _priority_stake_variants(
            stake_scales=stake_scales,
            model_gate_min_totals=model_gate_min_totals,
        )
    elif bool(args.priority_ml_knob_sweep):
        variants = _priority_ml_knob_variants(
            ml_enabled_values=ml_enabled_values,
            train_sizes=ml_train_sizes,
            calibrate_sizes=ml_calibrate_sizes,
            retrain_intervals=ml_retrain_intervals,
            recalibrate_intervals=ml_recalibrate_intervals,
        )
    elif bool(args.priority_frequency_sweep):
        variants = _priority_frequency_variants(
            expected_net_mins=freq_expected_net_mins,
            selector_thresholds=freq_selector_thresholds,
            online_thresholds=freq_online_thresholds,
            stake_scales=freq_stake_scales,
            model_gate_min_totals=freq_model_gate_min_totals,
        )
    else:
        variants = _default_variants(include_model_gate=bool(args.include_model_gate))
    rows: list[dict[str, object]] = []

    for sim_size in sim_sizes:
        for variant in variants:
            if variant.max_sim_size is not None and int(sim_size) > int(variant.max_sim_size):
                continue

            strategy_cfg = _strategy_for_variant(cfg=cfg, variant=variant)
            result = run_backtest_case(
                cfg=cfg,
                strategy_cfg=strategy_cfg,
                name=f"{str(args.name_prefix)}_{str(variant.name)}_{int(sim_size)}",
                simulation_size=int(sim_size),
                reset_mode="continuous",
                reset_every_rounds=0,
                initial_bankroll_bnb=args.initial_bankroll_bnb,
                exp_root=exp_root,
            )
            summary = dict(result.summary)
            net = float(summary.get("net_profit_bnb", 0.0))
            per_500 = float(net) * 500.0 / float(sim_size)
            max_dd = float(max_drawdown_bnb(trades_csv_path=result.trades_path))
            min_bank = float(_min_bankroll_bnb(trades_csv_path=result.trades_path))
            initial_bank = float(summary.get("initial_bankroll_bnb", 0.0))
            loss_from_initial_to_min = float(initial_bank) - float(min_bank)
            meets_initial_floor = bool(loss_from_initial_to_min <= float(args.drawdown_cap_bnb))

            rows.append(
                {
                    "variant": str(variant.name),
                    "sim_size": int(sim_size),
                    "router_mode": str(strategy_cfg.router.mode),
                    "stake_scale": float(variant.stake_scale),
                    "candidate_expected_net_min_bnb": (
                        None
                        if variant.candidate_expected_net_min_bnb is None
                        else float(variant.candidate_expected_net_min_bnb)
                    ),
                    "model_gate_min_total_bnb": (
                        None
                        if variant.model_gate_min_total_bnb is None
                        else float(variant.model_gate_min_total_bnb)
                    ),
                    "selector_score_threshold": (
                        None
                        if variant.selector_score_threshold is None
                        else float(variant.selector_score_threshold)
                    ),
                    "online_score_threshold_bnb": (
                        None
                        if variant.online_score_threshold_bnb is None
                        else float(variant.online_score_threshold_bnb)
                    ),
                    "ml_enabled": (
                        None if variant.ml_enabled is None else bool(variant.ml_enabled)
                    ),
                    "ml_train_size": (
                        None if variant.ml_train_size is None else int(variant.ml_train_size)
                    ),
                    "ml_calibrate_size": (
                        None if variant.ml_calibrate_size is None else int(variant.ml_calibrate_size)
                    ),
                    "ml_retrain_interval": (
                        None if variant.ml_retrain_interval is None else int(variant.ml_retrain_interval)
                    ),
                    "ml_recalibrate_interval": (
                        None
                        if variant.ml_recalibrate_interval is None
                        else int(variant.ml_recalibrate_interval)
                    ),
                    "anti_martingale_enabled": bool(variant.anti_martingale_enabled),
                    "anti_martingale_win_multiplier": float(variant.anti_martingale_win_multiplier),
                    "anti_martingale_loss_multiplier": float(variant.anti_martingale_loss_multiplier),
                    "anti_martingale_min_scale": float(variant.anti_martingale_min_scale),
                    "anti_martingale_max_scale": float(variant.anti_martingale_max_scale),
                    "circuit_breaker_enabled": bool(variant.circuit_breaker_enabled),
                    "circuit_breaker_drawdown_trigger_bnb": float(variant.circuit_breaker_drawdown_trigger_bnb),
                    "circuit_breaker_base_skip_rounds": int(variant.circuit_breaker_base_skip_rounds),
                    "circuit_breaker_escalation_multiplier": float(variant.circuit_breaker_escalation_multiplier),
                    "circuit_breaker_escalation_window_rounds": int(variant.circuit_breaker_escalation_window_rounds),
                    "circuit_breaker_max_level": int(variant.circuit_breaker_max_level),
                    "circuit_breaker_max_skip_rounds": int(variant.circuit_breaker_max_skip_rounds),
                    "circuit_breaker_reentry_rounds": int(variant.circuit_breaker_reentry_rounds),
                    "circuit_breaker_reentry_scale": float(variant.circuit_breaker_reentry_scale),
                    "net_profit_bnb": float(net),
                    "per_500": float(per_500),
                    "num_bets": int(summary.get("num_bets", 0)),
                    "bet_rate": float(summary.get("bet_rate", 0.0)),
                    "max_drawdown_bnb": float(max_dd),
                    "min_bankroll_bnb": float(min_bank),
                    "initial_bankroll_bnb": float(initial_bank),
                    "loss_from_initial_to_min_bnb": float(loss_from_initial_to_min),
                    "meets_initial_floor": bool(meets_initial_floor),
                    "top_skip_reasons": top_skip_reasons(
                        summary=summary,
                        limit=int(args.top_skip_limit),
                    ),
                    "selected_strategy_mix": _selected_strategy_mix(
                        trades_csv_path=result.trades_path,
                        limit=int(args.top_selected_limit),
                    ),
                    "elapsed_seconds": float(result.elapsed_seconds),
                    "summary_path": str(result.summary_path),
                    "trades_path": str(result.trades_path),
                }
            )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            int(r["sim_size"]),
            -int(bool(r["meets_initial_floor"])),
            -float(r["per_500"]),
            float(r["max_drawdown_bnb"]),
        ),
    )

    table_rows = [
        {
            "sim": int(r["sim_size"]),
            "variant": str(r["variant"]),
            "router": str(r["router_mode"]),
            "scale": f"{float(r['stake_scale']):.2f}",
            "model_min": (
                ""
                if r["model_gate_min_total_bnb"] is None
                else f"{float(r['model_gate_min_total_bnb']):.3f}"
            ),
            "exp_min": (
                ""
                if r["candidate_expected_net_min_bnb"] is None
                else f"{float(r['candidate_expected_net_min_bnb']):.3f}"
            ),
            "sel_thr": (
                ""
                if r["selector_score_threshold"] is None
                else f"{float(r['selector_score_threshold']):+.4f}"
            ),
            "on_thr": (
                ""
                if r["online_score_threshold_bnb"] is None
                else f"{float(r['online_score_threshold_bnb']):+.4f}"
            ),
            "ml_cfg": (
                ""
                if (
                    r["ml_enabled"] is None
                    and r["ml_train_size"] is None
                    and r["ml_calibrate_size"] is None
                    and r["ml_retrain_interval"] is None
                    and r["ml_recalibrate_interval"] is None
                )
                else (
                    f"{bool(r['ml_enabled'])}/"
                    f"t{int(r['ml_train_size']) if r['ml_train_size'] is not None else '-'}"
                    f"/c{int(r['ml_calibrate_size']) if r['ml_calibrate_size'] is not None else '-'}"
                    f"/rt{int(r['ml_retrain_interval']) if r['ml_retrain_interval'] is not None else '-'}"
                    f"/rc{int(r['ml_recalibrate_interval']) if r['ml_recalibrate_interval'] is not None else '-'}"
                )
            ),
            "anti": (
                ""
                if not bool(r["anti_martingale_enabled"])
                else f"{float(r['anti_martingale_min_scale']):.2f}-{float(r['anti_martingale_max_scale']):.2f}"
            ),
            "cb": (
                ""
                if not bool(r["circuit_breaker_enabled"])
                else f"t{float(r['circuit_breaker_drawdown_trigger_bnb']):.2f}/n{int(r['circuit_breaker_base_skip_rounds'])}"
            ),
            "per_500": f"{float(r['per_500']):+.6f}",
            "net": f"{float(r['net_profit_bnb']):+.6f}",
            "max_dd": f"{float(r['max_drawdown_bnb']):.6f}",
            "min_bank": f"{float(r['min_bankroll_bnb']):.6f}",
            "loss2min": f"{float(r['loss_from_initial_to_min_bnb']):.6f}",
            "floor2": str(bool(r["meets_initial_floor"])),
            "bets": int(r["num_bets"]),
            "bet_rate": f"{float(r['bet_rate']):.4f}",
            "warm_s": f"{float(r['elapsed_seconds']):.3f}",
        }
        for r in rows_sorted
    ]

    print(
        render_table(
            columns=[
                ("sim", "sim"),
                ("variant", "variant"),
                ("router", "router"),
                ("scale", "scale"),
                ("model_min", "model_min"),
                ("exp_min", "exp_min"),
                ("sel_thr", "sel_thr"),
                ("on_thr", "on_thr"),
                ("ml_cfg", "ml_cfg"),
                ("anti", "anti"),
                ("cb", "cb"),
                ("per_500", "per_500"),
                ("net", "net"),
                ("max_dd", "max_dd"),
                ("min_bank", "min_bank"),
                ("loss2min", "loss2min"),
                ("floor2", "floor2"),
                ("bets", "bets"),
                ("bet_rate", "bet_rate"),
                ("warm_s", "warm_s"),
            ],
            rows=table_rows,
        )
    )

    out_json = exp_root / f"{str(args.name_prefix)}.json"
    out_csv = exp_root / f"{str(args.name_prefix)}.csv"
    out_json.write_text(
        json.dumps(
            {
                "config_path": str(args.config),
                "sim_sizes": [int(x) for x in sim_sizes],
                "drawdown_cap_bnb": float(args.drawdown_cap_bnb),
                "priority_stake_sweep": bool(args.priority_stake_sweep),
                "priority_risk_adapt_sweep": bool(args.priority_risk_adapt_sweep),
                "priority_ml_knob_sweep": bool(args.priority_ml_knob_sweep),
                "priority_frequency_sweep": bool(args.priority_frequency_sweep),
                "stake_scales": [float(x) for x in stake_scales],
                "model_gate_min_totals": [float(x) for x in model_gate_min_totals],
                "ml_train_sizes": [int(x) for x in ml_train_sizes],
                "ml_calibrate_sizes": [int(x) for x in ml_calibrate_sizes],
                "ml_retrain_intervals": [int(x) for x in ml_retrain_intervals],
                "ml_recalibrate_intervals": [int(x) for x in ml_recalibrate_intervals],
                "ml_enabled_values": [bool(x) for x in ml_enabled_values],
                "freq_expected_net_mins": [float(x) for x in freq_expected_net_mins],
                "freq_selector_thresholds": [float(x) for x in freq_selector_thresholds],
                "freq_online_thresholds": [float(x) for x in freq_online_thresholds],
                "freq_stake_scales": [float(x) for x in freq_stake_scales],
                "freq_model_gate_min_totals": [float(x) for x in freq_model_gate_min_totals],
                "cb_triggers": [float(x) for x in cb_triggers],
                "cb_base_skips": [int(x) for x in cb_base_skips],
                "anti_max_scales": [float(x) for x in anti_max_scales],
                "anti_win_multiplier": float(args.anti_win_multiplier),
                "anti_loss_multiplier": float(args.anti_loss_multiplier),
                "anti_min_scale": float(args.anti_min_scale),
                "cb_escalation_multiplier": float(args.cb_escalation_multiplier),
                "cb_escalation_window_rounds": int(args.cb_escalation_window_rounds),
                "cb_max_level": int(args.cb_max_level),
                "cb_max_skip_rounds": int(args.cb_max_skip_rounds),
                "cb_reentry_rounds": int(args.cb_reentry_rounds),
                "cb_reentry_scale": float(args.cb_reentry_scale),
                "history_total_rounds": int(total_rounds),
                "selector_warmup_rounds": int(warmup_rounds),
                "max_sim_size_feasible": int(max_sim_size),
                "rows": rows_sorted,
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
                "sim_size",
                "router_mode",
                "stake_scale",
                "candidate_expected_net_min_bnb",
                "model_gate_min_total_bnb",
                "selector_score_threshold",
                "online_score_threshold_bnb",
                "ml_enabled",
                "ml_train_size",
                "ml_calibrate_size",
                "ml_retrain_interval",
                "ml_recalibrate_interval",
                "anti_martingale_enabled",
                "anti_martingale_win_multiplier",
                "anti_martingale_loss_multiplier",
                "anti_martingale_min_scale",
                "anti_martingale_max_scale",
                "circuit_breaker_enabled",
                "circuit_breaker_drawdown_trigger_bnb",
                "circuit_breaker_base_skip_rounds",
                "circuit_breaker_escalation_multiplier",
                "circuit_breaker_escalation_window_rounds",
                "circuit_breaker_max_level",
                "circuit_breaker_max_skip_rounds",
                "circuit_breaker_reentry_rounds",
                "circuit_breaker_reentry_scale",
                "net_profit_bnb",
                "per_500",
                "num_bets",
                "bet_rate",
                "max_drawdown_bnb",
                "min_bankroll_bnb",
                "initial_bankroll_bnb",
                "loss_from_initial_to_min_bnb",
                "meets_initial_floor",
                "top_skip_reasons",
                "selected_strategy_mix",
                "elapsed_seconds",
                "summary_path",
                "trades_path",
            ],
        )
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)

    print(f"TABLE_JSON={out_json}")
    print(f"TABLE_CSV={out_csv}")


if __name__ == "__main__":
    main()
