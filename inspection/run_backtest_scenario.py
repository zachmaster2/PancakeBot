from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

from pancakebot.backtest.config import BacktestConfig
from pancakebot.backtest.runner import run_backtest
from pancakebot.config.load_config import load_app_config
from pancakebot.config.strategy_config import StrategyConfig
from pancakebot.core.determinism import set_global_determinism
from pancakebot.core.errors import InvariantError
from pancakebot.domain.models.predictability_modes import (
    ALLOWED_PREDICTABILITY_FEATURE_MODES,
    ALLOWED_PREDICTABILITY_LABEL_MODES,
)
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


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for canonical backtest scenario runs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--sim-size", type=int, default=None)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    parser.add_argument("--reset-mode", type=str, choices=("continuous", "chunk_reset"), default=None)
    parser.add_argument("--reset-every-rounds", type=int, default=None)
    parser.add_argument(
        "--router-mode",
        type=str,
        choices=(
            "selector_max_score",
            "skip_only",
            "oracle_skip",
            "online_cellmean",
            "online_cellmean_side_gap",
            "online_cellmean_backoff",
            "online_cellmean_selector_fallback",
        ),
        default=None,
    )
    parser.add_argument("--router-score-threshold-bnb", type=float, default=None)
    parser.add_argument("--ml-enabled", type=str, default=None)
    parser.add_argument("--ml-min-tradeable-prob", type=float, default=None)
    parser.add_argument("--ml-min-prob-edge", type=float, default=None)
    parser.add_argument("--ml-cutoff-pool-total-min-bnb", type=float, default=None)
    parser.add_argument("--ml-expected-net-min-bnb", type=float, default=None)
    parser.add_argument("--ml-expected-net-max-bnb", type=float, default=None)
    parser.add_argument("--ml-veto-candidate-expected-net-below-min", type=str, default=None)
    parser.add_argument("--ml-rescore-baseline-candidates-with-expected-net", type=str, default=None)
    parser.add_argument("--ml-train-size", type=int, default=None)
    parser.add_argument("--ml-calibrate-size", "--ml-calibration-size", dest="ml_calibrate_size", type=int, default=None)
    parser.add_argument("--ml-retrain-interval", type=int, default=None)
    parser.add_argument(
        "--ml-recalibrate-interval",
        "--ml-recalibration-interval",
        dest="ml_recalibrate_interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ml-predictability-feature-mode",
        type=str,
        choices=ALLOWED_PREDICTABILITY_FEATURE_MODES,
        default=None,
    )
    parser.add_argument(
        "--ml-predictability-label-mode",
        type=str,
        choices=ALLOWED_PREDICTABILITY_LABEL_MODES,
        default=None,
    )
    return parser


def _max_drawdown_bnb(trades_csv_path: Path) -> float:
    """Compute max drawdown from the bankroll column in backtest trades output."""

    with trades_csv_path.open("r", newline="", encoding="utf-8") as f:
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


def _skip_reason_groups(summary: dict[str, Any]) -> dict[str, int]:
    """Return sorted skip reason counts from summary."""

    raw = dict(summary.get("num_skips_by_reason", {}))
    return {str(k): int(v) for k, v in sorted(raw.items())}


def _parse_optional_bool_token(raw: str | None) -> bool | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token == "":
        return None
    if token in ("true", "t", "1", "yes", "y", "on"):
        return True
    if token in ("false", "f", "0", "no", "n", "off"):
        return False
    raise InvariantError(f"scenario_ml_enabled_invalid: {raw}")


def _runtime_cfg_from_app(*, cfg, strategy_cfg: StrategyConfig) -> RuntimeConfig:
    """Build RuntimeConfig for deterministic backtest execution."""

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
        strategy_cfg=strategy_cfg,
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
        backtest_state_cache_dir=str(cfg.backtest_state_cache_dir),
    )


def _build_backtest_cfg(*, app_cfg, args: argparse.Namespace) -> BacktestConfig:
    """Build backtest config with optional CLI overrides."""

    reset_mode = str(app_cfg.backtest.reset_mode) if args.reset_mode is None else str(args.reset_mode)
    reset_every_rounds = (
        int(app_cfg.backtest.reset_every_rounds)
        if args.reset_every_rounds is None
        else int(args.reset_every_rounds)
    )
    backtest_cfg = BacktestConfig(
        simulation_size=(
            int(app_cfg.backtest.simulation_size)
            if args.sim_size is None
            else int(args.sim_size)
        ),
        initial_bankroll_bnb=(
            float(app_cfg.backtest.initial_bankroll_bnb)
            if args.initial_bankroll_bnb is None
            else float(args.initial_bankroll_bnb)
        ),
        reset_mode=str(reset_mode),
        reset_every_rounds=int(reset_every_rounds),
    )
    backtest_cfg.validate()
    return backtest_cfg


def _strategy_cfg_with_router_overrides(
    *,
    strategy_cfg: StrategyConfig,
    args: argparse.Namespace,
) -> StrategyConfig:
    """Apply optional router CLI overrides to strategy config."""

    router_cfg = strategy_cfg.router
    if args.router_mode is not None:
        router_cfg = replace(router_cfg, mode=str(args.router_mode))
    if args.router_score_threshold_bnb is not None:
        router_cfg = replace(
            router_cfg,
            score_threshold_bnb=float(args.router_score_threshold_bnb),
        )
    ml_cfg = strategy_cfg.ml_candidate
    ml_enabled = _parse_optional_bool_token(args.ml_enabled)
    if ml_enabled is not None:
        ml_cfg = replace(ml_cfg, enabled=bool(ml_enabled))
    if args.ml_min_tradeable_prob is not None:
        if not (0.0 <= float(args.ml_min_tradeable_prob) <= 1.0):
            raise InvariantError("scenario_ml_min_tradeable_prob_out_of_range")
        ml_cfg = replace(ml_cfg, min_tradeable_prob=float(args.ml_min_tradeable_prob))
    if args.ml_min_prob_edge is not None:
        if float(args.ml_min_prob_edge) < 0.0:
            raise InvariantError("scenario_ml_min_prob_edge_negative")
        ml_cfg = replace(ml_cfg, min_prob_edge=float(args.ml_min_prob_edge))
    if args.ml_cutoff_pool_total_min_bnb is not None:
        if float(args.ml_cutoff_pool_total_min_bnb) < 0.0:
            raise InvariantError("scenario_ml_cutoff_pool_total_min_bnb_negative")
        ml_cfg = replace(
            ml_cfg,
            cutoff_pool_total_min_bnb=float(args.ml_cutoff_pool_total_min_bnb),
        )
    if args.ml_expected_net_min_bnb is not None:
        ml_cfg = replace(ml_cfg, expected_net_min_bnb=float(args.ml_expected_net_min_bnb))
    if args.ml_expected_net_max_bnb is not None:
        if float(args.ml_expected_net_max_bnb) < 0.0:
            raise InvariantError("scenario_ml_expected_net_max_bnb_negative")
        ml_cfg = replace(ml_cfg, expected_net_max_bnb=float(args.ml_expected_net_max_bnb))
    ml_veto_candidate_expected_net_below_min = _parse_optional_bool_token(
        args.ml_veto_candidate_expected_net_below_min
    )
    if ml_veto_candidate_expected_net_below_min is not None:
        ml_cfg = replace(
            ml_cfg,
            veto_candidate_expected_net_below_min=bool(ml_veto_candidate_expected_net_below_min),
        )
    ml_rescore_baseline_candidates_with_expected_net = _parse_optional_bool_token(
        args.ml_rescore_baseline_candidates_with_expected_net
    )
    if ml_rescore_baseline_candidates_with_expected_net is not None:
        ml_cfg = replace(
            ml_cfg,
            rescore_baseline_candidates_with_expected_net=bool(
                ml_rescore_baseline_candidates_with_expected_net
            ),
        )
    if args.ml_train_size is not None:
        if int(args.ml_train_size) <= 0:
            raise InvariantError("scenario_ml_train_size_nonpositive")
        ml_cfg = replace(ml_cfg, train_size=int(args.ml_train_size))
    if args.ml_calibrate_size is not None:
        if int(args.ml_calibrate_size) < 0:
            raise InvariantError("scenario_ml_calibrate_size_negative")
        ml_cfg = replace(ml_cfg, calibrate_size=int(args.ml_calibrate_size))
    if args.ml_retrain_interval is not None:
        if int(args.ml_retrain_interval) <= 0:
            raise InvariantError("scenario_ml_retrain_interval_nonpositive")
        ml_cfg = replace(ml_cfg, retrain_interval=int(args.ml_retrain_interval))
    if args.ml_recalibrate_interval is not None:
        if int(args.ml_recalibrate_interval) < 0:
            raise InvariantError("scenario_ml_recalibrate_interval_negative")
        ml_cfg = replace(ml_cfg, recalibrate_interval=int(args.ml_recalibrate_interval))
    if args.ml_predictability_feature_mode is not None:
        ml_cfg = replace(
            ml_cfg,
            predictability_feature_mode=str(args.ml_predictability_feature_mode),
        )
    if args.ml_predictability_label_mode is not None:
        ml_cfg = replace(
            ml_cfg,
            predictability_label_mode=str(args.ml_predictability_label_mode),
        )
    if ml_cfg.expected_net_max_bnb is not None and float(ml_cfg.expected_net_max_bnb) < float(ml_cfg.expected_net_min_bnb):
        raise InvariantError("scenario_ml_expected_net_max_below_min")

    return replace(strategy_cfg, router=router_cfg, ml_candidate=ml_cfg)


def main() -> None:
    """Run a canonical backtest scenario and write scenario metadata."""

    args = _build_parser().parse_args()
    cfg = load_app_config(str(args.config))
    set_global_determinism(seed=int(cfg.random_seed))

    strategy_cfg = _strategy_cfg_with_router_overrides(strategy_cfg=cfg.strategy, args=args)
    runtime_cfg = _runtime_cfg_from_app(cfg=cfg, strategy_cfg=strategy_cfg)
    bt_cfg = _build_backtest_cfg(app_cfg=cfg, args=args)

    exp_root = Path(os.environ.get("PANCAKEBOT_EXP_DIR", _DEFAULT_EXP_ROOT))
    out_dir = exp_root / str(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_registry_store = getattr(runtime_cfg, "run_registry_store", None)
    if run_registry_store is not None and hasattr(run_registry_store, "start_run"):
        run_registry_store.start_run(
            run_name=str(args.name),
            config_path=str(args.config),
            metadata={
                "simulation_size": int(bt_cfg.simulation_size),
                "reset_mode": str(bt_cfg.reset_mode),
                "reset_every_rounds": int(bt_cfg.reset_every_rounds),
            },
        )
    try:
        run_backtest(runtime_cfg=runtime_cfg, backtest_cfg=bt_cfg, out_dir=out_dir)
    except Exception as e:
        if run_registry_store is not None and hasattr(run_registry_store, "fail_run"):
            try:
                run_registry_store.fail_run(run_name=str(args.name), error_text=str(e))
            except Exception:
                pass
        raise

    summary_path = out_dir / "backtest_summary.json"
    trades_path = out_dir / "backtest_trades.csv"
    if not summary_path.exists():
        raise InvariantError("scenario_summary_missing_after_backtest")
    if not trades_path.exists():
        raise InvariantError("scenario_trades_missing_after_backtest")

    summary = json.loads(summary_path.read_text())
    summary["scenario"] = {
        "name": str(args.name),
        "config_path": str(args.config),
        "sim_size": int(bt_cfg.simulation_size),
        "initial_bankroll_bnb": float(bt_cfg.initial_bankroll_bnb),
        "reset_mode": str(bt_cfg.reset_mode),
        "reset_every_rounds": int(bt_cfg.reset_every_rounds),
        "router_mode": str(runtime_cfg.strategy_cfg.router.mode),
        "router_score_threshold_bnb": float(runtime_cfg.strategy_cfg.router.score_threshold_bnb),
        "ml_enabled": bool(runtime_cfg.strategy_cfg.ml_candidate.enabled),
        "ml_min_tradeable_prob": float(runtime_cfg.strategy_cfg.ml_candidate.min_tradeable_prob),
        "ml_min_prob_edge": float(runtime_cfg.strategy_cfg.ml_candidate.min_prob_edge),
        "ml_cutoff_pool_total_min_bnb": float(runtime_cfg.strategy_cfg.ml_candidate.cutoff_pool_total_min_bnb),
        "ml_expected_net_min_bnb": float(runtime_cfg.strategy_cfg.ml_candidate.expected_net_min_bnb),
        "ml_expected_net_max_bnb": (
            None
            if runtime_cfg.strategy_cfg.ml_candidate.expected_net_max_bnb is None
            else float(runtime_cfg.strategy_cfg.ml_candidate.expected_net_max_bnb)
        ),
        "ml_veto_candidate_expected_net_below_min": bool(
            runtime_cfg.strategy_cfg.ml_candidate.veto_candidate_expected_net_below_min
        ),
        "ml_rescore_baseline_candidates_with_expected_net": bool(
            runtime_cfg.strategy_cfg.ml_candidate.rescore_baseline_candidates_with_expected_net
        ),
        "ml_train_size": int(runtime_cfg.strategy_cfg.ml_candidate.train_size),
        "ml_calibrate_size": int(runtime_cfg.strategy_cfg.ml_candidate.calibrate_size),
        "ml_retrain_interval": int(runtime_cfg.strategy_cfg.ml_candidate.retrain_interval),
        "ml_recalibrate_interval": int(runtime_cfg.strategy_cfg.ml_candidate.recalibrate_interval),
        "ml_predictability_feature_mode": str(runtime_cfg.strategy_cfg.ml_candidate.predictability_feature_mode),
        "ml_predictability_label_mode": str(runtime_cfg.strategy_cfg.ml_candidate.predictability_label_mode),
    }
    summary["risk"] = {"max_drawdown_bnb": float(_max_drawdown_bnb(trades_path))}
    summary["skip_reason_groups"] = _skip_reason_groups(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if run_registry_store is not None and hasattr(run_registry_store, "complete_run"):
        run_registry_store.complete_run(
            run_name=str(args.name),
            summary_path=str(summary_path),
            trades_path=str(trades_path),
            summary=dict(summary),
            max_drawdown_bnb=float(summary["risk"]["max_drawdown_bnb"]),
            profit_per_500_bnb=float(summary["net_profit_bnb"]) * 500.0 / float(bt_cfg.simulation_size),
        )

    print(f"SCENARIO={args.name}")
    print(f"SUMMARY={summary_path}")
    print(f"TRADES={trades_path}")
    print(f"NET={summary['net_profit_bnb']}")
    print(f"BETS={summary['num_bets']}")
    print(f"BET_RATE={summary['bet_rate']}")

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


if __name__ == "__main__":
    main()
