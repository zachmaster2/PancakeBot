from __future__ import annotations

import csv
import json
import os
import shutil
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pancakebot.backtest.config import BacktestConfig
from pancakebot.backtest.runner import run_backtest
from pancakebot.config.load_config import _parse_dislocation_candidate, load_app_config
from pancakebot.config.strategy_config import StrategyConfig
from pancakebot.core.determinism import set_global_determinism
from pancakebot.core.errors import InvariantError
from pancakebot.infra.binance_us_client import BinanceUsClient
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.feature_cache_store import FeatureCacheStore
from pancakebot.infra.market_data_db import MarketDataDb, SqliteKlinesStore
from pancakebot.infra.projection_cache_store import ProjectionCacheStore
from pancakebot.infra.run_registry_store import RunRegistryStore
from pancakebot.runtime.contract_constants_cache import load_contract_constants
from pancakebot.runtime.runtime_loop import RuntimeConfig

_BINANCE_US_SYMBOL = "BNBUSDT"
_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"
_STATE_CACHE_NAMESPACE = "pipeline_bootstrap"
_GAS_OVERRIDE_ENV = "PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE"


@dataclass(frozen=True, slots=True)
class BacktestRunResult:
    name: str
    out_dir: Path
    summary_path: Path
    trades_path: Path
    summary: dict[str, Any]
    elapsed_seconds: float


def load_cfg(*, config_path: str):
    cfg = load_app_config(str(config_path))
    set_global_determinism(seed=int(cfg.random_seed))
    return cfg


def load_all_dislocation_candidates(*, config_path: str) -> tuple[Any, ...]:
    config = Path(str(config_path))
    if not config.exists():
        raise InvariantError(f"config_file_missing: {config_path}")
    try:
        raw = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise InvariantError(f"config_toml_parse_failed: {e}") from e
    if not isinstance(raw, dict):
        raise InvariantError("config_root_not_dict")
    strategy = raw.get("strategy")
    if not isinstance(strategy, dict):
        raise InvariantError("config_section_not_dict: strategy")
    dislocation = strategy.get("dislocation")
    if not isinstance(dislocation, dict):
        raise InvariantError("config_section_not_dict: strategy.dislocation")
    candidates_obj = dislocation.get("candidates")
    if not isinstance(candidates_obj, list):
        raise InvariantError("config_section_not_list: strategy.dislocation.candidates")
    if not candidates_obj:
        raise InvariantError("dislocation_candidates_empty")
    parsed: list[Any] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(candidates_obj):
        if not isinstance(item, dict):
            raise InvariantError(f"dislocation_candidate_not_dict: idx={idx}")
        cfg = _parse_dislocation_candidate(item, idx=idx)
        key = str(cfg.name)
        if key in seen_names:
            raise InvariantError(f"dislocation_candidate_name_duplicate: {key}")
        seen_names.add(key)
        parsed.append(cfg)
    return tuple(parsed)


def resolve_exp_root() -> Path:
    return Path(os.environ.get("PANCAKEBOT_EXP_DIR", _DEFAULT_EXP_ROOT))


def _resolve_gas_price_wei_override(*, gas_price_wei_override: int | None) -> int | None:
    if gas_price_wei_override is not None:
        value = int(gas_price_wei_override)
        if int(value) < 0:
            raise InvariantError("backtest_gas_price_wei_override_negative")
        return int(value)

    raw = str(os.environ.get(_GAS_OVERRIDE_ENV, "")).strip()
    if raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise InvariantError(f"backtest_gas_price_wei_override_invalid: {raw}") from e
    if int(value) < 0:
        raise InvariantError("backtest_gas_price_wei_override_negative")
    return int(value)


def _apply_backtest_gas_profile(*, gas_price_wei_override: int | None) -> dict[str, float | int]:
    # Backtest-only mutable override: keep live runtime transaction gas policy unchanged.
    import pancakebot.backtest.runner as backtest_runner
    import pancakebot.core.constants as constants
    import pancakebot.domain.strategy.dislocation_engine as dislocation_engine
    import pancakebot.domain.strategy.ml_candidate_adapter as ml_candidate_adapter
    import pancakebot.domain.strategy.pipeline as strategy_pipeline
    import pancakebot.runtime.settlement as settlement

    gas_price_wei = int(constants.GAS_PRICE_WEI) if gas_price_wei_override is None else int(gas_price_wei_override)
    bet_cost_bnb = float(constants.GAS_LIMIT_BET) * float(gas_price_wei) / float(constants.BNB_WEI)
    claim_cost_bnb = float(constants.GAS_LIMIT_CLAIM) * float(gas_price_wei) / float(constants.BNB_WEI)

    constants.GAS_PRICE_WEI = int(gas_price_wei)
    constants.GAS_COST_BET_BNB = float(bet_cost_bnb)
    constants.GAS_COST_CLAIM_BNB = float(claim_cost_bnb)

    # Modules that imported constants by value need explicit patching.
    backtest_runner.GAS_COST_BET_BNB = float(bet_cost_bnb)
    dislocation_engine.GAS_COST_BET_BNB = float(bet_cost_bnb)
    dislocation_engine.GAS_COST_CLAIM_BNB = float(claim_cost_bnb)
    ml_candidate_adapter.GAS_COST_BET_BNB = float(bet_cost_bnb)
    ml_candidate_adapter.GAS_COST_CLAIM_BNB = float(claim_cost_bnb)
    strategy_pipeline.GAS_COST_BET_BNB = float(bet_cost_bnb)
    settlement.GAS_COST_CLAIM_BNB = float(claim_cost_bnb)

    return {
        "gas_price_wei": int(gas_price_wei),
        "gas_cost_bet_bnb": float(bet_cost_bnb),
        "gas_cost_claim_bnb": float(claim_cost_bnb),
    }


def _state_cache_dir_for_gas(*, base_dir: str, gas_price_wei: int) -> str:
    return str(Path(str(base_dir)) / f"gas_price_wei_{int(gas_price_wei)}")


def build_runtime_cfg(
    *,
    cfg,
    strategy_cfg: StrategyConfig | None = None,
    gas_price_wei_override: int | None = None,
) -> RuntimeConfig:
    constants = load_contract_constants()
    market_data_store = MarketDataDb(str(cfg.market_data_db_path))
    market_data_store.ensure_sources_synced(
        rounds_jsonl_path=str(cfg.closed_rounds_path),
        klines_jsonl_path=str(cfg.klines_path),
    )
    feature_cache_store = FeatureCacheStore(str(cfg.feature_cache_path))
    projection_cache_store = ProjectionCacheStore(str(cfg.projection_cache_db_path))
    run_registry_store = RunRegistryStore(str(cfg.run_registry_db_path))
    return RuntimeConfig(
        graph_client=None,
        round_store=ClosedRoundsStore(cfg.closed_rounds_path),
        klines_store=SqliteKlinesStore(market_data_db=market_data_store),
        binance_us_client=BinanceUsClient(timeout_seconds=10.0),
        binance_us_symbol=_BINANCE_US_SYMBOL,
        contract=None,
        wallet_address="",
        cutoff_seconds=int(cfg.cutoff_seconds),
        strategy_cfg=(cfg.strategy if strategy_cfg is None else strategy_cfg),
        min_bet_amount_bnb=float(constants.min_bet_amount_bnb),
        treasury_fee_fraction=float(constants.treasury_fee_fraction),
        buffer_seconds=int(constants.buffer_seconds),
        use_onchain_event_bets=False,
        event_lookback_blocks=int(cfg.event_lookback_blocks),
        latency_log_path=str(cfg.latency_log_path),
        wait_for_bet_receipt=False,
        bet_receipt_timeout_seconds=int(cfg.bet_receipt_timeout_seconds),
        dry=False,
        feature_cache_store=feature_cache_store,
        market_data_store=market_data_store,
        projection_cache_store=projection_cache_store,
        run_registry_store=run_registry_store,
        backtest_state_cache_dir=(
            str(cfg.backtest_state_cache_dir)
            if gas_price_wei_override is None
            else _state_cache_dir_for_gas(
                base_dir=str(cfg.backtest_state_cache_dir),
                gas_price_wei=int(gas_price_wei_override),
            )
        ),
        runtime_state_paths=cfg.runtime_state_paths,
    )


def run_backtest_case(
    *,
    cfg,
    name: str,
    simulation_size: int,
    reset_mode: str,
    reset_every_rounds: int,
    tail_offset_rounds: int = 0,
    initial_bankroll_bnb: float | None = None,
    strategy_cfg: StrategyConfig | None = None,
    exp_root: Path | None = None,
    gas_price_wei_override: int | None = None,
) -> BacktestRunResult:
    out_root = resolve_exp_root() if exp_root is None else Path(exp_root)
    out_dir = out_root / str(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_gas_override = _resolve_gas_price_wei_override(gas_price_wei_override=gas_price_wei_override)
    gas_profile = _apply_backtest_gas_profile(gas_price_wei_override=resolved_gas_override)

    runtime_cfg = build_runtime_cfg(
        cfg=cfg,
        strategy_cfg=strategy_cfg,
        gas_price_wei_override=resolved_gas_override,
    )
    run_registry_store = getattr(runtime_cfg, "run_registry_store", None)
    bt_cfg = BacktestConfig(
        simulation_size=int(simulation_size),
        initial_bankroll_bnb=(
            float(cfg.backtest.initial_bankroll_bnb)
            if initial_bankroll_bnb is None
            else float(initial_bankroll_bnb)
        ),
        reset_mode=str(reset_mode),
        reset_every_rounds=int(reset_every_rounds),
        tail_offset_rounds=int(tail_offset_rounds),
    )
    bt_cfg.validate()

    started = float(time.perf_counter())
    if run_registry_store is not None and hasattr(run_registry_store, "start_run"):
        run_registry_store.start_run(
            run_name=str(name),
            config_path="config.toml",
            metadata={
                "simulation_size": int(simulation_size),
                "reset_mode": str(reset_mode),
                "reset_every_rounds": int(reset_every_rounds),
                "tail_offset_rounds": int(tail_offset_rounds),
                "initial_bankroll_bnb": (
                    float(cfg.backtest.initial_bankroll_bnb)
                    if initial_bankroll_bnb is None
                    else float(initial_bankroll_bnb)
                ),
                "gas_price_wei": int(gas_profile["gas_price_wei"]),
                "gas_cost_bet_bnb": float(gas_profile["gas_cost_bet_bnb"]),
                "gas_cost_claim_bnb": float(gas_profile["gas_cost_claim_bnb"]),
                "state_cache_dir": str(runtime_cfg.backtest_state_cache_dir),
            },
        )
    try:
        run_backtest(runtime_cfg=runtime_cfg, backtest_cfg=bt_cfg, out_dir=out_dir)
        elapsed_seconds = float(time.perf_counter()) - float(started)

        summary_path = out_dir / "backtest_summary.json"
        trades_path = out_dir / "backtest_trades.csv"
        if not summary_path.exists():
            raise InvariantError("backtest_harness_summary_missing")
        if not trades_path.exists():
            raise InvariantError("backtest_harness_trades_missing")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if run_registry_store is not None and hasattr(run_registry_store, "complete_run"):
            per500 = float(summary.get("net_profit_bnb", 0.0)) * 500.0 / float(simulation_size)
            run_registry_store.complete_run(
                run_name=str(name),
                summary_path=str(summary_path),
                trades_path=str(trades_path),
                summary=dict(summary),
                profit_per_500_bnb=float(per500),
                max_drawdown_bnb=float(max_drawdown_bnb(trades_csv_path=trades_path)),
            )
        return BacktestRunResult(
            name=str(name),
            out_dir=out_dir,
            summary_path=summary_path,
            trades_path=trades_path,
            summary=dict(summary),
            elapsed_seconds=float(elapsed_seconds),
        )
    except Exception as e:
        if run_registry_store is not None and hasattr(run_registry_store, "fail_run"):
            try:
                run_registry_store.fail_run(run_name=str(name), error_text=str(e))
            except Exception:
                pass
        raise
    finally:
        for attr in (
            "feature_cache_store",
            "projection_cache_store",
            "run_registry_store",
            "market_data_store",
        ):
            store = getattr(runtime_cfg, str(attr), None)
            if store is None:
                continue
            try:
                if hasattr(store, "flush"):
                    store.flush()
                if hasattr(store, "close"):
                    store.close()
            except Exception:
                continue


def max_drawdown_bnb(*, trades_csv_path: Path) -> float:
    with Path(trades_csv_path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0

    peak = float(rows[0]["bankroll_bnb"])
    max_dd = 0.0
    for row in rows:
        bankroll = float(row["bankroll_bnb"])
        if bankroll > peak:
            peak = bankroll
        drawdown = float(peak) - float(bankroll)
        if drawdown > max_dd:
            max_dd = drawdown
    return float(max_dd)


def top_skip_reasons(*, summary: dict[str, Any], limit: int = 3) -> str:
    raw = dict(summary.get("num_skips_by_reason", {}))
    rows = sorted(
        ((str(k), int(v)) for k, v in raw.items()),
        key=lambda x: (-int(x[1]), str(x[0])),
    )
    if not rows:
        return ""
    return "; ".join(f"{key}:{val}" for key, val in rows[: int(limit)])


def clear_state_cache_dir(*, state_cache_root: Path) -> None:
    root = Path(state_cache_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def count_state_cache_files(*, state_cache_root: Path) -> int:
    ns = Path(state_cache_root) / _STATE_CACHE_NAMESPACE
    if not ns.exists():
        return 0
    return int(sum(1 for _ in ns.rglob("*.pkl.gz")))


def render_table(*, columns: list[tuple[str, str]], rows: list[dict[str, object]]) -> str:
    if not columns:
        raise InvariantError("render_table_columns_empty")

    widths: list[int] = []
    for key, header in columns:
        width = len(str(header))
        for row in rows:
            width = max(width, len(str(row.get(str(key), ""))))
        widths.append(int(width))

    header_line = " | ".join(str(header).ljust(widths[i]) for i, (_, header) in enumerate(columns))
    divider_line = "-+-".join("-" * int(widths[i]) for i in range(len(columns)))
    data_lines = [
        " | ".join(str(row.get(str(key), "")).ljust(widths[i]) for i, (key, _) in enumerate(columns))
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *data_lines])
