from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess

from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.direct_action_policy_model import (
    build_direct_action_dataset,
    direct_action_required_history_rounds,
    save_direct_action_bundle,
    train_direct_action_bundle,
)
from pancakebot.infra.feature_cache_store import FeatureCacheStore
from pancakebot.infra.market_data_db import MarketDataDb, SqliteKlinesStore
from pancakebot.runtime.contract_constants_cache import load_contract_constants

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class DirectActionEvalRow:
    sim_size: int
    tail_offset_rounds: int
    train_size: int
    valid_size: int
    random_seed: int
    required_history_rounds: int
    bundle_path: str
    test_per_500: float
    test_bet_rate: float
    test_net_profit_bnb: float
    max_drawdown_bnb: float


@dataclass(frozen=True, slots=True)
class DirectActionEvalAggregateRow:
    sim_size: int
    train_size: int
    valid_size: int
    random_seed: int
    num_offsets: int
    mean_per_500: float
    min_per_500: float
    mean_bet_rate: float
    mean_net_profit_bnb: float
    max_drawdown_bnb: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--sim-sizes", type=str, default="6480,8640,10800")
    parser.add_argument("--tail-offset-rounds", type=str, default="0,216,432,648,864")
    parser.add_argument("--train-size", type=int, default=15000)
    parser.add_argument("--valid-size", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=20260331)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def _parse_positive_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = int(text)
        if int(value) <= 0:
            raise InvariantError("direct_action_shared_eval_nonpositive_int")
        out.append(int(value))
    if not out:
        raise InvariantError("direct_action_shared_eval_empty_int_list")
    return out


def _parse_nonnegative_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = str(token).strip()
        if text == "":
            continue
        value = int(text)
        if int(value) < 0:
            raise InvariantError("direct_action_shared_eval_negative_offset")
        out.append(int(value))
    if not out:
        raise InvariantError("direct_action_shared_eval_empty_offset_list")
    return out


def _summary_metrics(summary_path: Path) -> tuple[float, float, float, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    num_rounds = int(summary["num_rounds"])
    if int(num_rounds) <= 0:
        raise InvariantError("direct_action_shared_eval_num_rounds_nonpositive")
    net_profit_bnb = float(summary["net_profit_bnb"])
    per_500 = float(net_profit_bnb) * 500.0 / float(num_rounds)
    bet_rate = float(summary["bet_rate"])
    risk = dict(summary.get("risk", {}))
    max_drawdown_bnb = float(risk.get("max_drawdown_bnb", 0.0))
    return float(per_500), float(bet_rate), float(net_profit_bnb), float(max_drawdown_bnb)


def _eval_scenario_name(
    *,
    name_prefix: str,
    sim_size: int,
    tail_offset_rounds: int,
) -> str:
    return f"{name_prefix}_tail{int(sim_size)}_off{int(tail_offset_rounds):05d}"


def _scenario_summary_path(*, output_dir: Path, scenario_name: str) -> Path:
    return (output_dir / str(scenario_name) / "backtest_summary.json").resolve()


def _bundle_path(*, output_dir: Path, scenario_name: str) -> Path:
    return (output_dir / str(scenario_name) / "direct_action_bundle.pkl.gz").resolve()


def _run_command(*, args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=str(cwd), check=True)


def _eval_command(
    *,
    python_exe: str,
    config_path: str,
    scenario_name: str,
    sim_size: int,
    tail_offset_rounds: int,
    bundle_path: Path,
) -> list[str]:
    return [
        str(python_exe),
        "-m",
        "inspection.run_backtest_scenario",
        "--config",
        str(config_path),
        "--name",
        str(scenario_name),
        "--sim-size",
        str(int(sim_size)),
        "--tail-offset-rounds",
        str(int(tail_offset_rounds)),
        "--direct-action-enabled",
        "true",
        "--direct-action-model-bundle-path",
        str(bundle_path),
        "--window-controller-enabled",
        "false",
    ]


def _load_round_slice(
    *,
    config_path: str,
    tail_offset_rounds: int,
    train_size: int,
    valid_size: int,
    sim_size: int,
) -> tuple[list[object], MarketDataDb, SqliteKlinesStore]:
    cfg = load_app_config(str(config_path))
    required_history = int(direct_action_required_history_rounds())
    total_rounds = int(required_history) + int(train_size) + int(valid_size) + int(sim_size)
    load_rounds = int(total_rounds) + int(tail_offset_rounds)
    market_data_store = MarketDataDb(str(cfg.market_data_db_path))
    market_data_store.ensure_sources_synced(
        rounds_jsonl_path=str(cfg.closed_rounds_path),
        klines_jsonl_path=str(cfg.klines_path),
    )
    rounds = market_data_store.load_tail_rounds(n=int(load_rounds))
    if len(rounds) != int(load_rounds):
        market_data_store.close()
        raise InvariantError("direct_action_shared_eval_rounds_insufficient")
    if int(tail_offset_rounds) > 0:
        rounds = list(rounds[: -int(tail_offset_rounds)])
    if len(rounds) != int(total_rounds):
        market_data_store.close()
        raise InvariantError("direct_action_shared_eval_round_slice_len_mismatch")
    return list(rounds), market_data_store, SqliteKlinesStore(market_data_db=market_data_store)


def _train_bundle_for_slice(
    *,
    config_path: str,
    bundle_path: Path,
    train_size: int,
    valid_size: int,
    sim_size: int,
    tail_offset_rounds: int,
    random_seed: int,
) -> int:
    cfg = load_app_config(str(config_path))
    constants = load_contract_constants()
    rounds, market_data_store, klines_store = _load_round_slice(
        config_path=str(config_path),
        tail_offset_rounds=int(tail_offset_rounds),
        train_size=int(train_size),
        valid_size=int(valid_size),
        sim_size=int(sim_size),
    )
    feature_cache_store = FeatureCacheStore(str(cfg.feature_cache_path))
    try:
        dataset = build_direct_action_dataset(
            rounds=rounds,
            klines_store_like=klines_store,
            cutoff_seconds=int(cfg.cutoff_seconds),
            treasury_fee_fraction=float(constants.treasury_fee_fraction),
            feature_cache_store=feature_cache_store,
        )
        target_epochs = list(int(epoch) for epoch in dataset.target_epochs)
        if len(target_epochs) != int(train_size) + int(valid_size) + int(sim_size):
            raise InvariantError("direct_action_shared_eval_target_epoch_len_mismatch")
        train_epochs = tuple(target_epochs[: int(train_size)])
        valid_epochs = tuple(target_epochs[int(train_size) : int(train_size) + int(valid_size)])
        bundle = train_direct_action_bundle(
            dataset=dataset,
            train_target_epochs=train_epochs,
            valid_target_epochs=valid_epochs,
            random_seed=int(random_seed),
            extra_metadata={
                "config_path": str(config_path),
                "train_size": int(train_size),
                "valid_size": int(valid_size),
                "sim_size": int(sim_size),
                "tail_offset_rounds": int(tail_offset_rounds),
            },
        )
        save_direct_action_bundle(bundle=bundle, path=str(bundle_path))
        return int(bundle.metadata["required_history_rounds"])
    finally:
        if hasattr(feature_cache_store, "flush"):
            feature_cache_store.flush()
        if hasattr(feature_cache_store, "close"):
            feature_cache_store.close()
        market_data_store.close()


def _aggregate_rows(rows: list[DirectActionEvalRow]) -> list[DirectActionEvalAggregateRow]:
    grouped: dict[int, list[DirectActionEvalRow]] = {}
    for row in rows:
        grouped.setdefault(int(row.sim_size), []).append(row)

    out: list[DirectActionEvalAggregateRow] = []
    for sim_size in sorted(grouped):
        group = list(grouped[int(sim_size)])
        out.append(
            DirectActionEvalAggregateRow(
                sim_size=int(sim_size),
                train_size=int(group[0].train_size),
                valid_size=int(group[0].valid_size),
                random_seed=int(group[0].random_seed),
                num_offsets=int(len(group)),
                mean_per_500=float(sum(float(row.test_per_500) for row in group) / float(len(group))),
                min_per_500=float(min(float(row.test_per_500) for row in group)),
                mean_bet_rate=float(sum(float(row.test_bet_rate) for row in group) / float(len(group))),
                mean_net_profit_bnb=float(
                    sum(float(row.test_net_profit_bnb) for row in group) / float(len(group))
                ),
                max_drawdown_bnb=float(max(float(row.max_drawdown_bnb) for row in group)),
            )
        )
    return out


def main() -> None:
    args = _build_parser().parse_args()
    sim_sizes = _parse_positive_int_list(args.sim_sizes)
    offsets = _parse_nonnegative_int_list(args.tail_offset_rounds)
    if int(args.train_size) <= 0:
        raise InvariantError("direct_action_shared_eval_train_size_nonpositive")
    if int(args.valid_size) <= 0:
        raise InvariantError("direct_action_shared_eval_valid_size_nonpositive")
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[DirectActionEvalRow] = []
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = str((repo_root / ".venv" / "Scripts" / "python.exe").resolve())

    for sim_size in sim_sizes:
        for tail_offset_rounds in offsets:
            scenario_name = _eval_scenario_name(
                name_prefix=str(args.name_prefix),
                sim_size=int(sim_size),
                tail_offset_rounds=int(tail_offset_rounds),
            )
            summary_path = _scenario_summary_path(output_dir=output_dir, scenario_name=scenario_name)
            bundle_path = _bundle_path(output_dir=output_dir, scenario_name=scenario_name)
            if not (bool(args.reuse_existing) and summary_path.exists() and bundle_path.exists()):
                required_history = _train_bundle_for_slice(
                    config_path=str(args.config),
                    bundle_path=bundle_path,
                    train_size=int(args.train_size),
                    valid_size=int(args.valid_size),
                    sim_size=int(sim_size),
                    tail_offset_rounds=int(tail_offset_rounds),
                    random_seed=int(args.random_seed),
                )
                _run_command(
                    args=_eval_command(
                        python_exe=python_exe,
                        config_path=str(args.config),
                        scenario_name=scenario_name,
                        sim_size=int(sim_size),
                        tail_offset_rounds=int(tail_offset_rounds),
                        bundle_path=bundle_path,
                    ),
                    cwd=repo_root,
                )
            else:
                required_history = int(direct_action_required_history_rounds())
            if not summary_path.exists():
                raise InvariantError("direct_action_shared_eval_summary_missing")
            per_500, bet_rate, net_profit_bnb, max_drawdown_bnb = _summary_metrics(summary_path)
            rows.append(
                DirectActionEvalRow(
                    sim_size=int(sim_size),
                    tail_offset_rounds=int(tail_offset_rounds),
                    train_size=int(args.train_size),
                    valid_size=int(args.valid_size),
                    random_seed=int(args.random_seed),
                    required_history_rounds=int(required_history),
                    bundle_path=str(bundle_path),
                    test_per_500=float(per_500),
                    test_bet_rate=float(bet_rate),
                    test_net_profit_bnb=float(net_profit_bnb),
                    max_drawdown_bnb=float(max_drawdown_bnb),
                )
            )

    rows.sort(key=lambda row: (int(row.sim_size), int(row.tail_offset_rounds)))
    aggregates = _aggregate_rows(rows)

    rows_path = (output_dir / f"{str(args.name_prefix)}_direct_action_shared_eval.csv").resolve()
    summary_path = (
        output_dir / f"{str(args.name_prefix)}_direct_action_shared_eval_summary.json"
    ).resolve()
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary_payload = {
        "name_prefix": str(args.name_prefix),
        "config": str(args.config),
        "sim_sizes": [int(value) for value in sim_sizes],
        "tail_offset_rounds": [int(value) for value in offsets],
        "train_size": int(args.train_size),
        "valid_size": int(args.valid_size),
        "random_seed": int(args.random_seed),
        "rows_csv": str(rows_path),
        "aggregates": [asdict(row) for row in aggregates],
        "rows": [asdict(row) for row in rows],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"ROWS={rows_path}")
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main()
