from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inspection.flow_strategy_common import (
    FlowBuildConfig,
    FlowModelConfig,
    FlowPolicyConfig,
    auto_window_sizes,
    build_flow_table,
    compute_current_odds_ev,
    flow_feature_columns,
    predict_probabilities_walk_forward,
    simulate_flow_policy,
)
from pancakebot.config.load_config import load_app_config
from pancakebot.core.constants import GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB


_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--sim-size", type=int, default=None)
    parser.add_argument("--tail-offset-rounds", type=int, default=0)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=None)
    parser.add_argument("--cutoff-seconds", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample-bytree", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--treasury-fee-rate", type=float, default=0.03)
    parser.add_argument("--tx-fee-per-unit", type=float, default=0.0)
    parser.add_argument("--gas-bet-abs", type=float, default=float(GAS_COST_BET_BNB))
    parser.add_argument("--gas-claim-abs", type=float, default=float(GAS_COST_CLAIM_BNB))
    parser.add_argument("--ev-threshold", type=float, default=None)
    parser.add_argument("--kelly-fraction", type=float, default=None)
    parser.add_argument("--max-fraction", type=float, default=None)
    parser.add_argument("--max-bet-abs", type=float, default=None)
    parser.add_argument("--min-bet-size", type=float, default=None)
    parser.add_argument("--min-total-pool-c", type=float, default=None)
    parser.add_argument("--max-total-pool-share", type=float, default=None)
    parser.add_argument("--max-side-pool-share", type=float, default=None)
    parser.add_argument("--allowed-sides", type=str, choices=("both", "bull_only", "bear_only"), default=None)
    parser.add_argument("--bull-roll-edge-min", type=float, default=None)
    parser.add_argument("--bear-roll-edge-min", type=float, default=None)
    parser.add_argument("--bull-roll-winrate-min", type=float, default=None)
    parser.add_argument("--bear-roll-winrate-min", type=float, default=None)
    parser.add_argument("--bull-cooldown-trades", type=int, default=None)
    parser.add_argument("--bear-cooldown-trades", type=int, default=None)
    return parser


def _tail_window(df: pd.DataFrame, *, sim_size: int | None, tail_offset_rounds: int) -> pd.DataFrame:
    if int(tail_offset_rounds) < 0:
        raise ValueError("flow_tail_offset_rounds_negative")
    total_rows = int(len(df))
    end = int(total_rows) - int(tail_offset_rounds)
    if int(end) <= 0:
        raise ValueError("flow_tail_offset_out_of_range")
    if sim_size is None:
        start = 0
    else:
        if int(sim_size) <= 0:
            raise ValueError("flow_sim_size_nonpositive")
        start = int(end) - int(sim_size)
        if int(start) < 0:
            raise ValueError("flow_sim_size_exceeds_available_rows")
    return df.iloc[int(start) : int(end)].reset_index(drop=True)


def _first_numeric_column_value(df: pd.DataFrame, column_name: str, *, first: bool) -> int | None:
    if str(column_name) not in df.columns or df.empty:
        return None
    column_data = df.loc[:, str(column_name)]
    if isinstance(column_data, pd.DataFrame):
        series = column_data.iloc[:, 0]
    else:
        series = column_data
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.empty:
        return None
    value = numeric.iloc[0] if bool(first) else numeric.iloc[-1]
    if not np.isfinite(value):
        return None
    return int(value)


def _standardized_backtest_trades(*, raw_df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "epoch",
                "action",
                "skip_reason",
                "direction",
                "bet_size_bnb",
                "p_final",
                "final_total_bnb",
                "final_bull_bnb",
                "final_bear_bnb",
                "ev_bnb",
                "profit_bnb",
                "bankroll_bnb",
                "selected_strategy",
                "router_mode",
                "selector_score_bnb",
            ]
        )
    direction = np.where(raw_df["action"].astype(str).eq("BULL"), "Bull", np.where(raw_df["action"].astype(str).eq("BEAR"), "Bear", ""))
    action = np.where(direction != "", "BET", "SKIP")
    skip_reason = np.where(action == "BET", "", raw_df["action"].astype(str).str.lower())
    ev_bnb = pd.to_numeric(raw_df.get("impact_ev_unit", 0.0), errors="coerce").fillna(0.0) * pd.to_numeric(raw_df.get("bet_size", 0.0), errors="coerce").fillna(0.0)
    selector_score = ev_bnb.where(action == "BET", "")
    out = pd.DataFrame(
        {
            "epoch": pd.to_numeric(raw_df.get("epoch", np.nan), errors="coerce").astype("Int64"),
            "action": action,
            "skip_reason": skip_reason,
            "direction": direction,
            "bet_size_bnb": pd.to_numeric(raw_df.get("bet_size", 0.0), errors="coerce").fillna(0.0),
            "p_final": pd.to_numeric(raw_df.get("pred_p_bull", np.nan), errors="coerce"),
            "final_total_bnb": pd.to_numeric(raw_df.get("totalAmount", np.nan), errors="coerce"),
            "final_bull_bnb": pd.to_numeric(raw_df.get("bullAmount", np.nan), errors="coerce"),
            "final_bear_bnb": pd.to_numeric(raw_df.get("bearAmount", np.nan), errors="coerce"),
            "ev_bnb": ev_bnb,
            "profit_bnb": pd.to_numeric(raw_df.get("pnl", 0.0), errors="coerce").fillna(0.0),
            "bankroll_bnb": pd.to_numeric(raw_df.get("bankroll", np.nan), errors="coerce"),
            "selected_strategy": str(strategy_name),
            "router_mode": "flow_lgbm",
            "selector_score_bnb": selector_score,
        }
    )
    out["epoch"] = out["epoch"].fillna(0).astype(int)
    return out


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.tail_offset_rounds) < 0:
        raise ValueError("flow_tail_offset_rounds_negative")
    app_cfg = load_app_config(args.config)
    flow_cfg = app_cfg.strategy.flow_candidate
    out_dir = Path(_DEFAULT_EXP_ROOT) / str(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoff_seconds = int(app_cfg.cutoff_seconds if args.cutoff_seconds is None else args.cutoff_seconds)
    build_cfg = FlowBuildConfig(cutoff_seconds=int(cutoff_seconds))
    flow_df = build_flow_table(str(app_cfg.closed_rounds_path), build_cfg)
    source_rows_total = int(len(flow_df))
    flow_df = _tail_window(
        flow_df,
        sim_size=(None if args.sim_size is None else int(args.sim_size)),
        tail_offset_rounds=int(args.tail_offset_rounds),
    )

    default_train, default_val, default_step = auto_window_sizes(len(flow_df))
    model_cfg = FlowModelConfig(
        train_size=int(default_train if args.train_size is None else args.train_size),
        val_size=int(default_val if args.val_size is None else args.val_size),
        step_size=int(default_step if args.step_size is None else args.step_size),
        n_estimators=int(flow_cfg.n_estimators if args.n_estimators is None else args.n_estimators),
        learning_rate=float(flow_cfg.learning_rate if args.learning_rate is None else args.learning_rate),
        num_leaves=int(flow_cfg.num_leaves if args.num_leaves is None else args.num_leaves),
        subsample=float(flow_cfg.subsample if args.subsample is None else args.subsample),
        colsample_bytree=float(flow_cfg.colsample_bytree if args.colsample_bytree is None else args.colsample_bytree),
        random_seed=int(flow_cfg.random_seed if args.random_seed is None else args.random_seed),
    )

    feature_cols = flow_feature_columns(flow_df)
    pred_p, model_meta = predict_probabilities_walk_forward(
        df=flow_df,
        feature_columns=feature_cols,
        model_cfg=model_cfg,
    )
    eval_df = flow_df[np.isfinite(pred_p)].copy()
    eval_df["pred_p_bull"] = pred_p[np.isfinite(pred_p)]
    ev_bull, ev_bear, win_profit_bull, win_profit_bear = compute_current_odds_ev(
        df=eval_df,
        p_bull=eval_df["pred_p_bull"].values,
        treasury_fee_rate=float(args.treasury_fee_rate),
        tx_fee_per_unit=float(args.tx_fee_per_unit),
    )
    eval_df["pred_ev_bull"] = ev_bull
    eval_df["pred_ev_bear"] = ev_bear
    eval_df["win_profit_bull_est"] = win_profit_bull
    eval_df["win_profit_bear_est"] = win_profit_bear

    initial_bankroll_bnb = float(app_cfg.backtest.initial_bankroll_bnb if args.initial_bankroll_bnb is None else args.initial_bankroll_bnb)
    policy_cfg = FlowPolicyConfig(
        initial_bankroll=float(initial_bankroll_bnb),
        treasury_fee_rate=float(args.treasury_fee_rate),
        fee_per_unit=float(args.tx_fee_per_unit),
        gas_bet_abs=float(args.gas_bet_abs),
        gas_claim_abs=float(args.gas_claim_abs),
        ev_threshold=float(flow_cfg.ev_threshold if args.ev_threshold is None else args.ev_threshold),
        kelly_fraction=float(flow_cfg.kelly_fraction if args.kelly_fraction is None else args.kelly_fraction),
        max_fraction=float(flow_cfg.max_fraction if args.max_fraction is None else args.max_fraction),
        max_bet_abs=float(flow_cfg.max_bet_abs if args.max_bet_abs is None else args.max_bet_abs),
        min_bet_size=float(flow_cfg.min_bet_size if args.min_bet_size is None else args.min_bet_size),
        min_total_pool_c=float(flow_cfg.min_total_pool_c if args.min_total_pool_c is None else args.min_total_pool_c),
        max_total_pool_share=float(
            flow_cfg.max_total_pool_share if args.max_total_pool_share is None else args.max_total_pool_share
        ),
        max_side_pool_share=float(
            flow_cfg.max_side_pool_share if args.max_side_pool_share is None else args.max_side_pool_share
        ),
        allowed_sides=str("both" if args.allowed_sides is None else args.allowed_sides),
        roll_window=int(flow_cfg.roll_window),
        roll_edge_min=float(flow_cfg.roll_edge_min),
        roll_winrate_min=float(flow_cfg.roll_winrate_min),
        cooldown_trades=int(flow_cfg.cooldown_trades),
        bull_roll_edge_min=float(flow_cfg.roll_edge_min if args.bull_roll_edge_min is None else args.bull_roll_edge_min),
        bear_roll_edge_min=float(flow_cfg.roll_edge_min if args.bear_roll_edge_min is None else args.bear_roll_edge_min),
        bull_roll_winrate_min=float(flow_cfg.roll_winrate_min if args.bull_roll_winrate_min is None else args.bull_roll_winrate_min),
        bear_roll_winrate_min=float(flow_cfg.roll_winrate_min if args.bear_roll_winrate_min is None else args.bear_roll_winrate_min),
        bull_cooldown_trades=int(flow_cfg.cooldown_trades if args.bull_cooldown_trades is None else args.bull_cooldown_trades),
        bear_cooldown_trades=int(flow_cfg.cooldown_trades if args.bear_cooldown_trades is None else args.bear_cooldown_trades),
    )
    policy_metrics, raw_trades = simulate_flow_policy(df=eval_df, cfg=policy_cfg)

    backtest_trades = _standardized_backtest_trades(raw_df=raw_trades, strategy_name=str(args.name))
    bankroll_series = pd.to_numeric(backtest_trades["bankroll_bnb"], errors="coerce").ffill().fillna(float(initial_bankroll_bnb))
    min_bankroll = float(bankroll_series.min()) if len(bankroll_series) else float(initial_bankroll_bnb)
    num_rounds = int(len(backtest_trades))
    num_bets = int((backtest_trades["action"] == "BET").sum())
    net_profit_bnb = float(policy_metrics["end_bankroll"] - policy_metrics["start_bankroll"])
    per_500 = float(net_profit_bnb * 500.0 / float(num_rounds)) if int(num_rounds) > 0 else 0.0
    bet_rate = float(num_bets / float(num_rounds)) if int(num_rounds) > 0 else 0.0

    summary = {
        "run_name": str(args.name),
        "closed_rounds_path": str(app_cfg.closed_rounds_path),
        "source_rows_total": int(source_rows_total),
        "source_rows_used": int(len(flow_df)),
        "tail_offset_rounds": int(args.tail_offset_rounds),
        "source_epoch_start": _first_numeric_column_value(flow_df, "epoch", first=True),
        "source_epoch_end": _first_numeric_column_value(flow_df, "epoch", first=False),
        "eval_rows": int(model_meta["eval_rows"]),
        "eval_epoch_start": _first_numeric_column_value(backtest_trades, "epoch", first=True),
        "eval_epoch_end": _first_numeric_column_value(backtest_trades, "epoch", first=False),
        "feature_count": int(len(feature_cols)),
        "train_size": int(model_cfg.train_size),
        "val_size": int(model_cfg.val_size),
        "step_size": int(model_cfg.step_size),
        "n_slices": int(model_meta["n_slices"]),
        "initial_bankroll_bnb": float(initial_bankroll_bnb),
        "end_bankroll_bnb": float(policy_metrics["end_bankroll"]),
        "net_profit_bnb": float(net_profit_bnb),
        "per_500": float(per_500),
        "bet_rate": float(bet_rate),
        "num_bets": int(num_bets),
        "num_wins": int(policy_metrics["wins"]),
        "num_losses": int(policy_metrics["losses"]),
        "win_rate": float(policy_metrics["win_rate"]),
        "max_drawdown_bnb": float(-policy_metrics["max_drawdown"]),
        "min_bankroll_bnb": float(min_bankroll),
        "loss_from_initial_to_min_bnb": float(float(initial_bankroll_bnb) - float(min_bankroll)),
        "bull_bets": int(policy_metrics.get("bull_bets", 0)),
        "bear_bets": int(policy_metrics.get("bear_bets", 0)),
        "bull_net_profit_bnb": float(policy_metrics.get("bull_net_profit_bnb", 0.0)),
        "bear_net_profit_bnb": float(policy_metrics.get("bear_net_profit_bnb", 0.0)),
        "policy": {
            "treasury_fee_rate": float(policy_cfg.treasury_fee_rate),
            "tx_fee_per_unit": float(policy_cfg.fee_per_unit),
            "gas_bet_abs": float(policy_cfg.gas_bet_abs),
            "gas_claim_abs": float(policy_cfg.gas_claim_abs),
            "ev_threshold": float(policy_cfg.ev_threshold),
            "kelly_fraction": float(policy_cfg.kelly_fraction),
            "max_fraction": float(policy_cfg.max_fraction),
            "max_bet_abs": float(policy_cfg.max_bet_abs),
            "min_bet_size": float(policy_cfg.min_bet_size),
            "min_total_pool_c": float(policy_cfg.min_total_pool_c),
            "max_total_pool_share": float(policy_cfg.max_total_pool_share),
            "max_side_pool_share": float(policy_cfg.max_side_pool_share),
            "allowed_sides": str(policy_cfg.allowed_sides),
            "bull_roll_edge_min": float(policy_cfg.bull_roll_edge_min),
            "bear_roll_edge_min": float(policy_cfg.bear_roll_edge_min),
            "bull_roll_winrate_min": float(policy_cfg.bull_roll_winrate_min),
            "bear_roll_winrate_min": float(policy_cfg.bear_roll_winrate_min),
            "bull_cooldown_trades": int(policy_cfg.bull_cooldown_trades),
            "bear_cooldown_trades": int(policy_cfg.bear_cooldown_trades),
        },
    }

    backtest_trades.to_csv(out_dir / "backtest_trades.csv", index=False)
    raw_trades.to_csv(out_dir / "flow_raw_trades.csv", index=False)
    (out_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
