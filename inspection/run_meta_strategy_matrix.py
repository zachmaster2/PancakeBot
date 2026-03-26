"""Run a resumable parameter sweep over offline meta-strategy probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspection.run_meta_strategy_probe import (
    _load_meta,
    _load_rows,
    _resolve_active_strategies,
    _run_probe,
    _select_feature_columns,
)


@dataclass(frozen=True, slots=True)
class MatrixTask:
    task_index: int
    task_id: str
    group_name: str
    active_strategies: tuple[str, ...]
    selector_mode: str
    safety_margin_bnb: float
    min_hold_blocks: int
    trailing_history_blocks: int
    min_train_rows: int
    logistic_c: float
    ridge_alpha: float
    elastic_net_alpha: float
    elastic_net_l1_ratio: float
    hgb_max_depth: int
    hgb_l2_regularization: float


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        out.append(int(text))
    if not out:
        raise ValueError("meta_strategy_matrix_int_list_empty")
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        out.append(float(text))
    if not out:
        raise ValueError("meta_strategy_matrix_float_list_empty")
    return out


def _parse_selector_modes(raw: str) -> list[str]:
    out = [str(token).strip() for token in str(raw).split(",") if str(token).strip()]
    if not out:
        raise ValueError("meta_strategy_matrix_selector_modes_empty")
    return out


def _parse_strategy_groups(raw_groups: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in raw_groups:
        spec = str(raw).strip()
        if spec == "":
            continue
        if "=" not in spec:
            raise ValueError(f"meta_strategy_matrix_group_invalid: {spec}")
        name, raw_members = spec.split("=", 1)
        group_name = str(name).strip()
        members = str(raw_members).strip()
        if group_name == "" or members == "":
            raise ValueError(f"meta_strategy_matrix_group_invalid: {spec}")
        out.append((str(group_name), str(members)))
    if not out:
        raise ValueError("meta_strategy_matrix_groups_empty")
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--dataset-csv", type=str, required=True)
    parser.add_argument("--dataset-meta", type=str, required=True)
    parser.add_argument("--baseline-strategy-name", type=str, required=True)
    parser.add_argument("--strategy-group", action="append", default=[])
    parser.add_argument("--selector-modes", type=str, default="delta_trailing_mean,delta_ridge")
    parser.add_argument("--margins-bnb", type=str, default="0.0,0.01,0.02")
    parser.add_argument("--hold-blocks", type=str, default="1,2,3")
    parser.add_argument("--lookbacks", type=str, default="5,10,15")
    parser.add_argument("--min-train-rows-list", type=str, default="8,12,16")
    parser.add_argument("--logistic-cs", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--elastic-net-alphas", type=str, default="0.01,0.05,0.1")
    parser.add_argument("--elastic-net-l1-ratios", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--hgb-max-depths", type=str, default="2,3")
    parser.add_argument("--hgb-l2-regularizations", type=str, default="0.0,1.0")
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-iter", type=int, default=200)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=6)
    parser.add_argument("--starting-bankroll-bnb", type=float, default=50.0)
    parser.add_argument("--output-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def _validate_partition_args(
    *,
    chunk_size: int,
    chunk_index: int,
    shard_count: int,
    shard_index: int,
) -> None:
    if int(chunk_size) < 0:
        raise ValueError("meta_strategy_matrix_chunk_size_negative")
    if int(chunk_index) < 0:
        raise ValueError("meta_strategy_matrix_chunk_index_negative")
    if int(chunk_size) == 0 and int(chunk_index) != 0:
        raise ValueError("meta_strategy_matrix_chunk_index_requires_chunk_size")
    if int(shard_count) <= 0:
        raise ValueError("meta_strategy_matrix_shard_count_nonpositive")
    if int(shard_index) < 0 or int(shard_index) >= int(shard_count):
        raise ValueError("meta_strategy_matrix_shard_index_out_of_range")


def _task_id_payload(
    *,
    group_name: str,
    active_strategies: tuple[str, ...],
    selector_mode: str,
    margin_bnb: float,
    hold_blocks: int,
    lookback_blocks: int,
    min_train_rows: int,
    logistic_c: float,
    ridge_alpha: float,
    elastic_net_alpha: float,
    elastic_net_l1_ratio: float,
    hgb_max_depth: int,
    hgb_l2_regularization: float,
) -> dict[str, Any]:
    return {
        "group_name": str(group_name),
        "active_strategies": list(active_strategies),
        "selector_mode": str(selector_mode),
        "safety_margin_bnb": float(margin_bnb),
        "min_hold_blocks": int(hold_blocks),
        "trailing_history_blocks": int(lookback_blocks),
        "min_train_rows": int(min_train_rows),
        "logistic_c": float(logistic_c),
        "ridge_alpha": float(ridge_alpha),
        "elastic_net_alpha": float(elastic_net_alpha),
        "elastic_net_l1_ratio": float(elastic_net_l1_ratio),
        "hgb_max_depth": int(hgb_max_depth),
        "hgb_l2_regularization": float(hgb_l2_regularization),
    }


def _make_task_id(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


def _enumerate_tasks(
    *,
    all_strategies: list[str],
    baseline_strategy_name: str,
    selector_modes: list[str],
    margins: list[float],
    hold_blocks: list[int],
    lookbacks: list[int],
    min_train_rows_list: list[int],
    logistic_cs: list[float],
    ridge_alphas: list[float],
    elastic_net_alphas: list[float],
    elastic_net_l1_ratios: list[float],
    hgb_max_depths: list[int],
    hgb_l2_regularizations: list[float],
    strategy_groups: list[tuple[str, str]],
) -> list[MatrixTask]:
    tasks: list[MatrixTask] = []
    for group_name, raw_members in strategy_groups:
        active_strategies = tuple(
            _resolve_active_strategies(
                all_strategies=all_strategies,
                raw_active_names=str(raw_members),
            )
        )
        if str(baseline_strategy_name) not in active_strategies:
            raise ValueError(f"meta_strategy_matrix_group_missing_baseline: {group_name}")
        for selector_mode in selector_modes:
            for margin_bnb in margins:
                for hold in hold_blocks:
                    for lookback in lookbacks:
                        for min_train_rows in min_train_rows_list:
                            logistic_values = logistic_cs if str(selector_mode) == "delta_logistic" else [1.0]
                            ridge_values = ridge_alphas if str(selector_mode) == "delta_ridge" else [1.0]
                            en_alpha_values = (
                                elastic_net_alphas if str(selector_mode) == "delta_elastic_net" else [0.05]
                            )
                            en_l1_values = (
                                elastic_net_l1_ratios if str(selector_mode) == "delta_elastic_net" else [0.5]
                            )
                            hgb_selector = str(selector_mode) in ("delta_hgb", "delta_hgb_classifier")
                            hgb_depth_values = hgb_max_depths if bool(hgb_selector) else [3]
                            hgb_l2_values = hgb_l2_regularizations if bool(hgb_selector) else [1.0]
                            for logistic_c in logistic_values:
                                for ridge_alpha in ridge_values:
                                    for elastic_net_alpha in en_alpha_values:
                                        for elastic_net_l1_ratio in en_l1_values:
                                            for hgb_max_depth in hgb_depth_values:
                                                for hgb_l2_regularization in hgb_l2_values:
                                                    payload = _task_id_payload(
                                                        group_name=str(group_name),
                                                        active_strategies=active_strategies,
                                                        selector_mode=str(selector_mode),
                                                        margin_bnb=float(margin_bnb),
                                                        hold_blocks=int(hold),
                                                        lookback_blocks=int(lookback),
                                                        min_train_rows=int(min_train_rows),
                                                        logistic_c=float(logistic_c),
                                                        ridge_alpha=float(ridge_alpha),
                                                        elastic_net_alpha=float(elastic_net_alpha),
                                                        elastic_net_l1_ratio=float(elastic_net_l1_ratio),
                                                    hgb_max_depth=int(hgb_max_depth),
                                                    hgb_l2_regularization=float(hgb_l2_regularization),
                                                )
                                                tasks.append(
                                                    MatrixTask(
                                                        task_index=len(tasks),
                                                        task_id=_make_task_id(payload),
                                                        group_name=str(group_name),
                                                        active_strategies=active_strategies,
                                                        selector_mode=str(selector_mode),
                                                        safety_margin_bnb=float(margin_bnb),
                                                        min_hold_blocks=int(hold),
                                                        trailing_history_blocks=int(lookback),
                                                        min_train_rows=int(min_train_rows),
                                                        logistic_c=float(logistic_c),
                                                        ridge_alpha=float(ridge_alpha),
                                                        elastic_net_alpha=float(elastic_net_alpha),
                                                        elastic_net_l1_ratio=float(elastic_net_l1_ratio),
                                                        hgb_max_depth=int(hgb_max_depth),
                                                        hgb_l2_regularization=float(hgb_l2_regularization),
                                                    )
                                                )
    if not tasks:
        raise ValueError("meta_strategy_matrix_no_tasks")
    return tasks


def _select_tasks(
    *,
    tasks: list[MatrixTask],
    chunk_size: int,
    chunk_index: int,
    shard_count: int,
    shard_index: int,
) -> list[MatrixTask]:
    selected = list(tasks)
    if int(chunk_size) > 0:
        start = int(chunk_index) * int(chunk_size)
        stop = min(len(selected), start + int(chunk_size))
        if start >= len(selected):
            selected = []
        else:
            selected = selected[start:stop]
    if int(shard_count) > 1:
        selected = [task for task in selected if int(task.task_index) % int(shard_count) == int(shard_index)]
    return selected


def _results_root(*, output_dir: Path, name_prefix: str) -> Path:
    return Path(output_dir) / f"{name_prefix}_meta_strategy_matrix_parts"


def _manifest_path(*, parts_dir: Path) -> Path:
    return Path(parts_dir) / "manifest.json"


def _task_result_path(*, parts_dir: Path, task: MatrixTask) -> Path:
    return Path(parts_dir) / f"task_{int(task.task_index):05d}_{task.task_id}.json"


def _write_json_atomic(*, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _write_csv_atomic(*, path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _matrix_manifest_payload(
    *,
    args: argparse.Namespace,
    selector_modes: list[str],
    margins: list[float],
    hold_blocks: list[int],
    lookbacks: list[int],
    min_train_rows_list: list[int],
    logistic_cs: list[float],
    ridge_alphas: list[float],
    elastic_net_alphas: list[float],
    elastic_net_l1_ratios: list[float],
    hgb_max_depths: list[int],
    hgb_l2_regularizations: list[float],
    strategy_groups: list[tuple[str, str]],
    tasks: list[MatrixTask],
) -> dict[str, Any]:
    return {
        "task_space": {
            "name_prefix": str(args.name_prefix),
            "dataset_csv": str(Path(str(args.dataset_csv)).resolve()),
            "dataset_meta": str(Path(str(args.dataset_meta)).resolve()),
            "baseline_strategy_name": str(args.baseline_strategy_name),
            "selector_modes": list(selector_modes),
            "margins_bnb": list(margins),
            "hold_blocks": list(hold_blocks),
            "lookbacks": list(lookbacks),
            "min_train_rows_list": list(min_train_rows_list),
            "logistic_cs": list(logistic_cs),
            "ridge_alphas": list(ridge_alphas),
            "elastic_net_alphas": list(elastic_net_alphas),
            "elastic_net_l1_ratios": list(elastic_net_l1_ratios),
            "hgb_max_depths": list(hgb_max_depths),
            "hgb_l2_regularizations": list(hgb_l2_regularizations),
            "hgb_learning_rate": float(args.hgb_learning_rate),
            "hgb_max_iter": int(args.hgb_max_iter),
            "hgb_min_samples_leaf": int(args.hgb_min_samples_leaf),
            "strategy_groups": [[str(name), str(members)] for name, members in strategy_groups],
            "starting_bankroll_bnb": float(args.starting_bankroll_bnb),
        },
        "task_count": int(len(tasks)),
        "task_ids": [str(task.task_id) for task in tasks],
    }


def _ensure_manifest(
    *,
    parts_dir: Path,
    manifest_payload: dict[str, Any],
) -> None:
    manifest_path = _manifest_path(parts_dir=parts_dir)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest_payload:
            raise ValueError(f"meta_strategy_matrix_manifest_mismatch: {manifest_path}")
        return
    _write_json_atomic(path=manifest_path, payload=manifest_payload)


def _build_group_baselines(
    *,
    rows,
    tasks: list[MatrixTask],
    baseline_strategy_name: str,
    starting_bankroll_bnb: float,
    key_map: dict[str, str],
    feature_columns: list[str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        groups.setdefault(str(task.group_name), tuple(task.active_strategies))

    out: dict[str, dict[str, Any]] = {}
    for group_name, active_strategies in groups.items():
        active_feature_columns = _select_feature_columns(
            feature_columns=feature_columns,
            strategies=list(active_strategies),
            key_map=key_map,
        )
        summary, _ = _run_probe(
            rows=rows,
            strategies=list(active_strategies),
            feature_columns=active_feature_columns,
            selector_mode="fixed_strategy",
            trailing_history_blocks=1,
            knn_k=1,
            min_train_rows=0,
            logistic_c=1.0,
            ridge_alpha=1.0,
            elastic_net_alpha=0.05,
            elastic_net_l1_ratio=0.5,
            hgb_learning_rate=float(0.05),
            hgb_max_depth=3,
            hgb_max_iter=200,
            hgb_l2_regularization=1.0,
            hgb_min_samples_leaf=6,
            safety_margin_bnb=0.0,
            min_hold_blocks=1,
            baseline_mode="skip_all",
            starting_bankroll_bnb=float(starting_bankroll_bnb),
            fixed_strategy_name=str(baseline_strategy_name),
            baseline_strategy_name=str(baseline_strategy_name),
        )
        out[str(group_name)] = summary
    return out


def _run_matrix_task(
    *,
    task: MatrixTask,
    rows,
    feature_columns: list[str],
    baseline_summary: dict[str, Any],
    baseline_strategy_name: str,
    starting_bankroll_bnb: float,
    hgb_learning_rate: float,
    hgb_max_iter: int,
    hgb_min_samples_leaf: int,
    key_map: dict[str, str],
) -> dict[str, Any]:
    active_feature_columns = _select_feature_columns(
        feature_columns=feature_columns,
        strategies=list(task.active_strategies),
        key_map=key_map,
    )
    summary, _ = _run_probe(
        rows=rows,
        strategies=list(task.active_strategies),
        feature_columns=active_feature_columns,
        selector_mode=str(task.selector_mode),
        trailing_history_blocks=int(task.trailing_history_blocks),
        knn_k=7,
        min_train_rows=int(task.min_train_rows),
        logistic_c=float(task.logistic_c),
        ridge_alpha=float(task.ridge_alpha),
        elastic_net_alpha=float(task.elastic_net_alpha),
        elastic_net_l1_ratio=float(task.elastic_net_l1_ratio),
        hgb_learning_rate=float(hgb_learning_rate),
        hgb_max_depth=int(task.hgb_max_depth),
        hgb_max_iter=int(hgb_max_iter),
        hgb_l2_regularization=float(task.hgb_l2_regularization),
        hgb_min_samples_leaf=int(hgb_min_samples_leaf),
        safety_margin_bnb=float(task.safety_margin_bnb),
        min_hold_blocks=int(task.min_hold_blocks),
        baseline_mode="skip_all",
        starting_bankroll_bnb=float(starting_bankroll_bnb),
        fixed_strategy_name="",
        baseline_strategy_name=str(baseline_strategy_name),
    )
    return {
        "task_index": int(task.task_index),
        "task_id": str(task.task_id),
        "group_name": str(task.group_name),
        "active_strategies": ",".join(task.active_strategies),
        "selector_mode": str(task.selector_mode),
        "safety_margin_bnb": float(task.safety_margin_bnb),
        "min_hold_blocks": int(task.min_hold_blocks),
        "trailing_history_blocks": int(task.trailing_history_blocks),
        "min_train_rows": int(task.min_train_rows),
        "logistic_c": float(task.logistic_c),
        "ridge_alpha": float(task.ridge_alpha),
        "elastic_net_alpha": float(task.elastic_net_alpha),
        "elastic_net_l1_ratio": float(task.elastic_net_l1_ratio),
        "hgb_max_depth": int(task.hgb_max_depth),
        "hgb_l2_regularization": float(task.hgb_l2_regularization),
        "net_profit_bnb": float(summary["net_profit_bnb"]),
        "net_profit_per_500_rounds": float(summary["net_profit_per_500_rounds"]),
        "max_drawdown_bnb": float(summary["max_drawdown_bnb"]),
        "loss_from_start_to_min_bnb": float(summary["loss_from_start_to_min_bnb"]),
        "num_switches": int(summary["num_switches"]),
        "capture_ratio_vs_oracle": float(summary["capture_ratio_vs_oracle"]),
        "baseline_fixed_net_profit_bnb": float(baseline_summary["net_profit_bnb"]),
        "baseline_fixed_per_500_rounds": float(baseline_summary["net_profit_per_500_rounds"]),
        "lift_vs_fixed_baseline_per_500": float(
            float(summary["net_profit_per_500_rounds"]) - float(baseline_summary["net_profit_per_500_rounds"])
        ),
        "picks_by_strategy": json.dumps(summary["picks_by_strategy"], sort_keys=True),
        "pick_reasons": json.dumps(summary["pick_reasons"], sort_keys=True),
    }


def _load_completed_rows(*, parts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(parts_dir).glob("task_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise ValueError(f"meta_strategy_matrix_task_result_invalid: {path}") from exc
    return rows


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = list(rows)
    ranked.sort(
        key=lambda row: (
            float(row["net_profit_per_500_rounds"]),
            -float(row["max_drawdown_bnb"]),
            -int(row["task_index"]),
        ),
        reverse=True,
    )
    return ranked


def _write_aggregate_outputs(
    *,
    output_dir: Path,
    name_prefix: str,
    all_tasks: list[MatrixTask],
    completed_rows: list[dict[str, Any]],
    top_k: int,
    manifest_payload: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / f"{name_prefix}_meta_strategy_matrix.csv"
    json_path = Path(output_dir) / f"{name_prefix}_meta_strategy_matrix.json"

    ranked_rows = _rank_rows(completed_rows)
    if ranked_rows:
        _write_csv_atomic(path=csv_path, rows=ranked_rows)

    payload = {
        "matrix": {
            **manifest_payload["task_space"],
            "expected_task_count": int(len(all_tasks)),
            "completed_task_count": int(len(completed_rows)),
            "is_complete": bool(len(completed_rows) == len(all_tasks)),
        },
        "top_rows": ranked_rows[: int(top_k)],
    }
    _write_json_atomic(path=json_path, payload=payload)
    return csv_path, json_path


def main() -> None:
    args = _build_parser().parse_args()
    _validate_partition_args(
        chunk_size=int(args.chunk_size),
        chunk_index=int(args.chunk_index),
        shard_count=int(args.shard_count),
        shard_index=int(args.shard_index),
    )

    selector_modes = _parse_selector_modes(str(args.selector_modes))
    margins = _parse_float_list(str(args.margins_bnb))
    hold_blocks = _parse_int_list(str(args.hold_blocks))
    lookbacks = _parse_int_list(str(args.lookbacks))
    min_train_rows_list = _parse_int_list(str(args.min_train_rows_list))
    logistic_cs = _parse_float_list(str(args.logistic_cs))
    ridge_alphas = _parse_float_list(str(args.ridge_alphas))
    elastic_net_alphas = _parse_float_list(str(args.elastic_net_alphas))
    elastic_net_l1_ratios = _parse_float_list(str(args.elastic_net_l1_ratios))
    hgb_max_depths = _parse_int_list(str(args.hgb_max_depths))
    hgb_l2_regularizations = _parse_float_list(str(args.hgb_l2_regularizations))
    strategy_groups = _parse_strategy_groups(list(args.strategy_group))

    all_strategies, key_map, feature_columns = _load_meta(Path(str(args.dataset_meta)))
    rows = _load_rows(
        dataset_csv=Path(str(args.dataset_csv)),
        strategies=all_strategies,
        key_map=key_map,
        feature_columns=feature_columns,
    )
    all_tasks = _enumerate_tasks(
        all_strategies=all_strategies,
        baseline_strategy_name=str(args.baseline_strategy_name),
        selector_modes=selector_modes,
        margins=margins,
        hold_blocks=hold_blocks,
        lookbacks=lookbacks,
        min_train_rows_list=min_train_rows_list,
        logistic_cs=logistic_cs,
        ridge_alphas=ridge_alphas,
        elastic_net_alphas=elastic_net_alphas,
        elastic_net_l1_ratios=elastic_net_l1_ratios,
        hgb_max_depths=hgb_max_depths,
        hgb_l2_regularizations=hgb_l2_regularizations,
        strategy_groups=strategy_groups,
    )
    selected_tasks = _select_tasks(
        tasks=all_tasks,
        chunk_size=int(args.chunk_size),
        chunk_index=int(args.chunk_index),
        shard_count=int(args.shard_count),
        shard_index=int(args.shard_index),
    )

    output_dir = Path(str(args.output_dir))
    parts_dir = _results_root(output_dir=output_dir, name_prefix=str(args.name_prefix))
    parts_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = _matrix_manifest_payload(
        args=args,
        selector_modes=selector_modes,
        margins=margins,
        hold_blocks=hold_blocks,
        lookbacks=lookbacks,
        min_train_rows_list=min_train_rows_list,
        logistic_cs=logistic_cs,
        ridge_alphas=ridge_alphas,
        elastic_net_alphas=elastic_net_alphas,
        elastic_net_l1_ratios=elastic_net_l1_ratios,
        hgb_max_depths=hgb_max_depths,
        hgb_l2_regularizations=hgb_l2_regularizations,
        strategy_groups=strategy_groups,
        tasks=all_tasks,
    )
    _ensure_manifest(parts_dir=parts_dir, manifest_payload=manifest_payload)

    executed = 0
    skipped_existing = 0
    resume = not bool(args.no_resume)
    if not bool(args.aggregate_only) and selected_tasks:
        group_baselines = _build_group_baselines(
            rows=rows,
            tasks=selected_tasks,
            baseline_strategy_name=str(args.baseline_strategy_name),
            starting_bankroll_bnb=float(args.starting_bankroll_bnb),
            key_map=key_map,
            feature_columns=feature_columns,
        )
        for position, task in enumerate(selected_tasks, start=1):
            result_path = _task_result_path(parts_dir=parts_dir, task=task)
            if bool(resume) and result_path.exists():
                skipped_existing += 1
                continue
            row = _run_matrix_task(
                task=task,
                rows=rows,
                feature_columns=feature_columns,
                baseline_summary=group_baselines[str(task.group_name)],
                baseline_strategy_name=str(args.baseline_strategy_name),
                starting_bankroll_bnb=float(args.starting_bankroll_bnb),
                hgb_learning_rate=float(args.hgb_learning_rate),
                hgb_max_iter=int(args.hgb_max_iter),
                hgb_min_samples_leaf=int(args.hgb_min_samples_leaf),
                key_map=key_map,
            )
            _write_json_atomic(path=result_path, payload=row)
            executed += 1
            if int(args.progress_every) > 0 and (
                int(executed) == 1
                or int(executed) % int(args.progress_every) == 0
                or int(position) == int(len(selected_tasks))
            ):
                print(
                    "PROGRESS "
                    f"selected={len(selected_tasks)} "
                    f"position={position} "
                    f"executed={executed} "
                    f"skipped_existing={skipped_existing} "
                    f"task_index={task.task_index} "
                    f"selector_mode={task.selector_mode} "
                    f"group_name={task.group_name}"
                )

    completed_rows = _load_completed_rows(parts_dir=parts_dir)
    csv_path, json_path = _write_aggregate_outputs(
        output_dir=output_dir,
        name_prefix=str(args.name_prefix),
        all_tasks=all_tasks,
        completed_rows=completed_rows,
        top_k=int(args.top_k),
        manifest_payload=manifest_payload,
    )

    ranked_rows = _rank_rows(completed_rows)
    print(f"MATRIX_PARTS_DIR={parts_dir}")
    print(f"MATRIX_CSV={csv_path}")
    print(f"MATRIX_JSON={json_path}")
    print(f"TASKS_TOTAL={len(all_tasks)}")
    print(f"TASKS_SELECTED={len(selected_tasks)}")
    print(f"TASKS_EXECUTED={executed}")
    print(f"TASKS_SKIPPED_EXISTING={skipped_existing}")
    print(f"COMPLETED_ROWS={len(completed_rows)}")
    print(f"EXPECTED_ROWS={len(all_tasks)}")
    if ranked_rows:
        print(f"BEST_PER_500={ranked_rows[0]['net_profit_per_500_rounds']}")
    else:
        print("BEST_PER_500=")


if __name__ == "__main__":
    main()
