from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pancakebot.backtest.config import BacktestConfig
from pancakebot.backtest.runner import run_backtest
from pancakebot.config.load_config import load_app_config
from pancakebot.core.determinism import set_global_determinism
from pancakebot.core.errors import InvariantError
from pancakebot.infra.binance_us_client import BinanceUsClient
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.klines_store import KlinesStore
from pancakebot.runtime.contract_constants_cache import load_contract_constants
from pancakebot.runtime.runtime_loop import RuntimeConfig

_BINANCE_US_SYMBOL = "BNBUSDT"


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for canonical backtest scenario runs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--sim-size", type=int, default=None)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    parser.add_argument("--reset-mode", type=str, choices=("continuous", "chunk_reset"), default=None)
    parser.add_argument("--reset-every-rounds", type=int, default=None)
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


def _runtime_cfg_from_app(cfg) -> RuntimeConfig:
    """Build RuntimeConfig for deterministic backtest execution."""

    constants = load_contract_constants()
    return RuntimeConfig(
        graph_client=None,
        round_store=ClosedRoundsStore(cfg.closed_rounds_path),
        klines_store=KlinesStore(cfg.klines_path),
        binance_us_client=BinanceUsClient(timeout_seconds=10.0),
        binance_us_symbol=_BINANCE_US_SYMBOL,
        contract=None,
        wallet_address="",
        cutoff_seconds=int(cfg.cutoff_seconds),
        strategy_cfg=cfg.strategy,
        treasury_fee_fraction=float(constants.treasury_fee_fraction),
        buffer_seconds=int(constants.buffer_seconds),
        use_onchain_event_bets=False,
        event_lookback_blocks=int(cfg.event_lookback_blocks),
        latency_log_path=str(cfg.latency_log_path),
        wait_for_bet_receipt=False,
        bet_receipt_timeout_seconds=int(cfg.bet_receipt_timeout_seconds),
        dry=False,
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


def main() -> None:
    """Run a canonical backtest scenario and write scenario metadata."""

    args = _build_parser().parse_args()
    cfg = load_app_config(str(args.config))
    set_global_determinism(seed=int(cfg.random_seed))

    runtime_cfg = _runtime_cfg_from_app(cfg)
    bt_cfg = _build_backtest_cfg(app_cfg=cfg, args=args)

    out_dir = Path("var/exp") / str(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_backtest(runtime_cfg=runtime_cfg, backtest_cfg=bt_cfg, out_dir=out_dir)

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
    }
    summary["risk"] = {"max_drawdown_bnb": float(_max_drawdown_bnb(trades_path))}
    summary["skip_reason_groups"] = _skip_reason_groups(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"SCENARIO={args.name}")
    print(f"SUMMARY={summary_path}")
    print(f"TRADES={trades_path}")
    print(f"NET={summary['net_profit_bnb']}")
    print(f"BETS={summary['num_bets']}")
    print(f"BET_RATE={summary['bet_rate']}")


if __name__ == "__main__":
    main()
