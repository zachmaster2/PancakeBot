from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inspection.flow_strategy_common import (
    FlowBuildConfig,
    FlowModelConfig,
    FlowPolicyConfig,
    build_flow_table,
    compute_current_odds_ev,
    flow_feature_columns,
    predict_probabilities_walk_forward,
    simulate_flow_policy,
)
from pancakebot.config.load_config import load_app_config
from pancakebot.core.constants import GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB


_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


def _csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("flow_sweep_int_list_empty")
    return out


def _csv_floats(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("flow_sweep_float_list_empty")
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--source-sim-size", type=int, default=30000)
    parser.add_argument("--tail-offset-rounds", type=str, default="0,5000,10000,15000,20000,25000")
    parser.add_argument("--probe-source-sim-size", type=int, default=18000)
    parser.add_argument("--probe-tail-offset-rounds", type=str, default="0")
    parser.add_argument("--train-sizes", type=str, default="12000,15000,18000")
    parser.add_argument("--val-sizes", type=str, default="500,1000")
    parser.add_argument("--step-sizes", type=str, default="500,1000")
    parser.add_argument("--ev-thresholds", type=str, default="0.0025,0.005,0.0075")
    parser.add_argument("--min-total-pool-cs", type=str, default="1.0,1.2")
    parser.add_argument("--roll-windows", type=str, default="120,200,300")
    parser.add_argument("--roll-edge-mins", type=str, default="-0.01,-0.005,-0.002")
    parser.add_argument("--roll-winrate-mins", type=str, default="0.45,0.47,0.49")
    parser.add_argument("--cooldown-trades-list", type=str, default="20,60,120")
    parser.add_argument("--allowed-sides", type=str, choices=("both", "bull_only", "bear_only"), default="both")
    parser.add_argument("--min-bet-rate", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--cutoff-seconds", type=int, default=None)
    parser.add_argument("--treasury-fee-rate", type=float, default=0.03)
    parser.add_argument("--tx-fee-per-unit", type=float, default=0.0)
    parser.add_argument("--gas-bet-abs", type=float, default=float(GAS_COST_BET_BNB))
    parser.add_argument("--gas-claim-abs", type=float, default=float(GAS_COST_CLAIM_BNB))
    parser.add_argument("--kelly-fraction", type=float, default=0.10)
    parser.add_argument("--max-fraction", type=float, default=0.25)
    parser.add_argument("--max-bet-abs", type=float, default=0.50)
    parser.add_argument("--min-bet-size", type=float, default=0.05)
    parser.add_argument("--max-total-pool-share", type=float, default=0.05)
    parser.add_argument("--max-side-pool-share", type=float, default=0.50)
    parser.add_argument("--min-bull-ratio", type=float, default=0.05)
    parser.add_argument("--max-bull-ratio", type=float, default=0.95)
    parser.add_argument("--vol-mid", type=float, default=0.03)
    parser.add_argument("--drawdown-stop-pct", type=float, default=0.75)
    parser.add_argument("--drawdown-throttle-start-pct", type=float, default=0.35)
    parser.add_argument("--drawdown-throttle-min-scale", type=float, default=0.35)
    parser.add_argument("--round-to", type=float, default=0.01)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def _tail_window(df: pd.DataFrame, *, sim_size: int, tail_offset_rounds: int) -> pd.DataFrame:
    total_rows = int(len(df))
    end = int(total_rows) - int(tail_offset_rounds)
    if int(end) <= 0:
        raise ValueError("flow_sweep_tail_offset_out_of_range")
    start = int(end) - int(sim_size)
    if int(start) < 0:
        raise ValueError("flow_sweep_sim_size_exceeds_rows")
    return df.iloc[int(start) : int(end)].reset_index(drop=True)


def _evaluate_window(
    *,
    df_source: pd.DataFrame,
    sim_size: int,
    tail_offset_rounds: int,
    model_cfg: FlowModelConfig,
    policy_cfg: FlowPolicyConfig,
) -> dict[str, float | int | None]:
    source_df = _tail_window(
        df_source,
        sim_size=int(sim_size),
        tail_offset_rounds=int(tail_offset_rounds),
    )
    feature_cols = flow_feature_columns(source_df)
    pred_p, meta = predict_probabilities_walk_forward(
        df=source_df,
        feature_columns=feature_cols,
        model_cfg=model_cfg,
    )
    eval_mask = np.isfinite(pred_p)
    eval_df = source_df.loc[eval_mask].copy()
    eval_df["pred_p_bull"] = pred_p[eval_mask]
    ev_bull, ev_bear, win_profit_bull, win_profit_bear = compute_current_odds_ev(
        df=eval_df,
        p_bull=eval_df["pred_p_bull"].values,
        treasury_fee_rate=float(policy_cfg.treasury_fee_rate),
        tx_fee_per_unit=float(policy_cfg.fee_per_unit),
    )
    eval_df["pred_ev_bull"] = ev_bull
    eval_df["pred_ev_bear"] = ev_bear
    eval_df["win_profit_bull_est"] = win_profit_bull
    eval_df["win_profit_bear_est"] = win_profit_bear
    policy_metrics, raw_trades = simulate_flow_policy(df=eval_df, cfg=policy_cfg)
    num_rounds = int(len(raw_trades))
    net_profit_bnb = float(policy_metrics["end_bankroll"] - policy_metrics["start_bankroll"])
    per_500 = float(net_profit_bnb * 500.0 / float(num_rounds)) if int(num_rounds) > 0 else 0.0
    bet_sizes = pd.to_numeric(raw_trades.get("bet_size", 0.0), errors="coerce").fillna(0.0)
    bet_rate = float((bet_sizes > 0.0).sum() / max(1, int(num_rounds)))
    return {
        "source_rows_used": int(len(source_df)),
        "eval_rows": int(meta["eval_rows"]),
        "eval_epoch_start": (None if raw_trades.empty else int(pd.to_numeric(raw_trades["epoch"], errors="coerce").iloc[0])),
        "eval_epoch_end": (None if raw_trades.empty else int(pd.to_numeric(raw_trades["epoch"], errors="coerce").iloc[-1])),
        "net_profit_bnb": float(net_profit_bnb),
        "per_500": float(per_500),
        "bet_rate": float(bet_rate),
        "num_bets": int(policy_metrics["bets"]),
        "max_drawdown_bnb": float(-policy_metrics["max_drawdown"]),
        "win_rate": float(policy_metrics["win_rate"]),
    }


def main() -> None:
    args = _build_parser().parse_args()
    app_cfg = load_app_config(str(args.config))
    out_dir = Path(_DEFAULT_EXP_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.name_prefix)
    partial_jsonl_path = out_dir / f"{prefix}_partial.jsonl"
    if partial_jsonl_path.exists():
        partial_jsonl_path.unlink()

    cutoff_seconds = int(app_cfg.cutoff_seconds if args.cutoff_seconds is None else args.cutoff_seconds)
    flow_df = build_flow_table(
        str(app_cfg.closed_rounds_path),
        FlowBuildConfig(cutoff_seconds=int(cutoff_seconds)),
    )

    source_offsets = _csv_ints(args.tail_offset_rounds)
    probe_offsets = _csv_ints(args.probe_tail_offset_rounds)
    train_sizes = _csv_ints(args.train_sizes)
    val_sizes = _csv_ints(args.val_sizes)
    step_sizes = _csv_ints(args.step_sizes)
    ev_thresholds = _csv_floats(args.ev_thresholds)
    min_total_pool_cs = _csv_floats(args.min_total_pool_cs)
    roll_windows = _csv_ints(args.roll_windows)
    roll_edge_mins = _csv_floats(args.roll_edge_mins)
    roll_winrate_mins = _csv_floats(args.roll_winrate_mins)
    cooldown_list = _csv_ints(args.cooldown_trades_list)

    results: list[dict[str, object]] = []
    config_idx = 0
    for (
        train_size,
        val_size,
        step_size,
        ev_threshold,
        min_total_pool_c,
        roll_window,
        roll_edge_min,
        roll_winrate_min,
        cooldown_trades,
    ) in itertools.product(
        train_sizes,
        val_sizes,
        step_sizes,
        ev_thresholds,
        min_total_pool_cs,
        roll_windows,
        roll_edge_mins,
        roll_winrate_mins,
        cooldown_list,
    ):
        if int(train_size) <= int(val_size):
            continue
        config_idx += 1
        model_cfg = FlowModelConfig(
            train_size=int(train_size),
            val_size=int(val_size),
            step_size=int(step_size),
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            num_leaves=int(args.num_leaves),
            subsample=float(args.subsample),
            colsample_bytree=float(args.colsample_bytree),
            random_seed=int(args.random_seed),
        )
        policy_cfg = FlowPolicyConfig(
            initial_bankroll=float(app_cfg.backtest.initial_bankroll_bnb),
            treasury_fee_rate=float(args.treasury_fee_rate),
            fee_per_unit=float(args.tx_fee_per_unit),
            gas_bet_abs=float(args.gas_bet_abs),
            gas_claim_abs=float(args.gas_claim_abs),
            round_to=float(args.round_to),
            ev_threshold=float(ev_threshold),
            kelly_fraction=float(args.kelly_fraction),
            max_fraction=float(args.max_fraction),
            max_bet_abs=float(args.max_bet_abs),
            min_bet_size=float(args.min_bet_size),
            min_total_pool_c=float(min_total_pool_c),
            max_total_pool_share=float(args.max_total_pool_share),
            max_side_pool_share=float(args.max_side_pool_share),
            min_bull_ratio=float(args.min_bull_ratio),
            max_bull_ratio=float(args.max_bull_ratio),
            vol_mid=float(args.vol_mid),
            drawdown_stop_pct=float(args.drawdown_stop_pct),
            drawdown_throttle_start_pct=float(args.drawdown_throttle_start_pct),
            drawdown_throttle_min_scale=float(args.drawdown_throttle_min_scale),
            allowed_sides=str(args.allowed_sides),
            roll_window=int(roll_window),
            roll_edge_min=float(roll_edge_min),
            roll_winrate_min=float(roll_winrate_min),
            cooldown_trades=int(cooldown_trades),
            bull_roll_edge_min=float(roll_edge_min),
            bear_roll_edge_min=float(roll_edge_min),
            bull_roll_winrate_min=float(roll_winrate_min),
            bear_roll_winrate_min=float(roll_winrate_min),
            bull_cooldown_trades=int(cooldown_trades),
            bear_cooldown_trades=int(cooldown_trades),
        )
        source_windows: list[dict[str, object]] = []
        probe_windows: list[dict[str, object]] = []
        for offset in source_offsets:
            metrics = _evaluate_window(
                df_source=flow_df,
                sim_size=int(args.source_sim_size),
                tail_offset_rounds=int(offset),
                model_cfg=model_cfg,
                policy_cfg=policy_cfg,
            )
            metrics["tail_offset_rounds"] = int(offset)
            source_windows.append(metrics)
        for offset in probe_offsets:
            metrics = _evaluate_window(
                df_source=flow_df,
                sim_size=int(args.probe_source_sim_size),
                tail_offset_rounds=int(offset),
                model_cfg=model_cfg,
                policy_cfg=policy_cfg,
            )
            metrics["tail_offset_rounds"] = int(offset)
            probe_windows.append(metrics)

        source_per500 = [float(x["per_500"]) for x in source_windows]
        source_bet_rates = [float(x["bet_rate"]) for x in source_windows]
        probe_per500 = [float(x["per_500"]) for x in probe_windows]
        probe_bet_rates = [float(x["bet_rate"]) for x in probe_windows]
        row: dict[str, object] = {
            "config_idx": int(config_idx),
            "train_size": int(train_size),
            "val_size": int(val_size),
            "step_size": int(step_size),
            "ev_threshold": float(ev_threshold),
            "min_total_pool_c": float(min_total_pool_c),
            "roll_window": int(roll_window),
            "roll_edge_min": float(roll_edge_min),
            "roll_winrate_min": float(roll_winrate_min),
            "cooldown_trades": int(cooldown_trades),
            "allowed_sides": str(args.allowed_sides),
            "source_mean_per_500": float(np.mean(np.asarray(source_per500, dtype=float))),
            "source_min_per_500": float(np.min(np.asarray(source_per500, dtype=float))),
            "source_positive_windows": int(sum(1 for x in source_per500 if float(x) > 0.0)),
            "source_mean_bet_rate": float(np.mean(np.asarray(source_bet_rates, dtype=float))),
            "probe_mean_per_500": float(np.mean(np.asarray(probe_per500, dtype=float))),
            "probe_min_per_500": float(np.min(np.asarray(probe_per500, dtype=float))),
            "probe_positive_windows": int(sum(1 for x in probe_per500 if float(x) > 0.0)),
            "probe_mean_bet_rate": float(np.mean(np.asarray(probe_bet_rates, dtype=float))),
            "meets_min_bet_rate": bool(
                all(float(x) >= float(args.min_bet_rate) for x in source_bet_rates + probe_bet_rates)
            ),
            "source_windows": source_windows,
            "probe_windows": probe_windows,
        }
        results.append(row)
        with partial_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    results_sorted = sorted(
        results,
        key=lambda x: (
            bool(x["meets_min_bet_rate"]),
            float(x["probe_min_per_500"]),
            float(x["source_min_per_500"]),
            float(x["probe_mean_per_500"]),
            float(x["source_mean_per_500"]),
        ),
        reverse=True,
    )

    top_k = max(1, int(args.top_k))
    top_rows = results_sorted[:top_k]
    table_rows = []
    for row in top_rows:
        table_rows.append(
            {
                "config_idx": int(row["config_idx"]),
                "train_size": int(row["train_size"]),
                "val_size": int(row["val_size"]),
                "step_size": int(row["step_size"]),
                "ev_threshold": float(row["ev_threshold"]),
                "min_total_pool_c": float(row["min_total_pool_c"]),
                "roll_window": int(row["roll_window"]),
                "roll_edge_min": float(row["roll_edge_min"]),
                "roll_winrate_min": float(row["roll_winrate_min"]),
                "cooldown_trades": int(row["cooldown_trades"]),
                "allowed_sides": str(row["allowed_sides"]),
                "meets_min_bet_rate": bool(row["meets_min_bet_rate"]),
                "source_mean_per_500": float(row["source_mean_per_500"]),
                "source_min_per_500": float(row["source_min_per_500"]),
                "source_positive_windows": int(row["source_positive_windows"]),
                "source_mean_bet_rate": float(row["source_mean_bet_rate"]),
                "probe_mean_per_500": float(row["probe_mean_per_500"]),
                "probe_min_per_500": float(row["probe_min_per_500"]),
                "probe_positive_windows": int(row["probe_positive_windows"]),
                "probe_mean_bet_rate": float(row["probe_mean_bet_rate"]),
            }
        )

    table_df = pd.DataFrame(table_rows)
    summary = {
        "name_prefix": str(prefix),
        "closed_rounds_path": str(app_cfg.closed_rounds_path),
        "source_sim_size": int(args.source_sim_size),
        "source_tail_offsets": source_offsets,
        "probe_source_sim_size": int(args.probe_source_sim_size),
        "probe_tail_offsets": probe_offsets,
        "num_configs": int(len(results)),
        "min_bet_rate": float(args.min_bet_rate),
        "top_k": int(top_k),
        "best_config": (top_rows[0] if top_rows else None),
    }

    (out_dir / f"{prefix}.json").write_text(json.dumps(results_sorted, indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table_df.to_csv(out_dir / f"{prefix}.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
