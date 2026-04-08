"""Reusable flow-strategy research helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


EPS = 1e-12
WEI_PER_BNB = 1_000_000_000_000_000_000


def _coerce_finite_float(value: Any) -> float:
    try:
        numeric = pd.to_numeric(value, errors="coerce")
    except Exception:
        return float("nan")
    try:
        result = float(numeric)
    except Exception:
        return float("nan")
    if not np.isfinite(result):
        return float("nan")
    return float(result)


@dataclass(frozen=True, slots=True)
class FlowBuildConfig:
    cutoff_seconds: int = 17
    windows_seconds: tuple[int, ...] = (10, 30, 60, 120)
    ret_lags: tuple[int, ...] = (1, 3, 5)
    vol_windows: tuple[int, ...] = (5, 10, 20)
    topk_for_share: int = 5


@dataclass(frozen=True, slots=True)
class FlowModelConfig:
    train_size: int
    val_size: int
    step_size: int
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_seed: int = 42


@dataclass(frozen=True, slots=True)
class FlowPolicyConfig:
    initial_bankroll: float = 50.0
    min_bankroll: float = 0.0
    treasury_fee_rate: float = 0.03
    fee_per_unit: float = 0.0
    gas_bet_abs: float = 0.0002
    gas_claim_abs: float = 0.00025
    round_to: float = 0.01
    ev_threshold: float = 0.0025
    kelly_fraction: float = 0.10
    max_fraction: float = 0.25
    max_bet_abs: float = 0.50
    min_bet_size: float = 0.05
    min_total_pool_c: float = 1.0
    max_total_pool_share: float = 0.05
    max_side_pool_share: float = 0.50
    min_bull_ratio: float = 0.05
    max_bull_ratio: float = 0.95
    vol_mid: float = 0.030
    drawdown_stop_pct: float = 0.75
    drawdown_throttle_start_pct: float = 0.35
    drawdown_throttle_min_scale: float = 0.35
    allowed_sides: str = "both"
    roll_window: int = 40
    roll_edge_min: float = -0.002
    roll_winrate_min: float = 0.48
    cooldown_trades: int = 40
    bull_roll_edge_min: float = -0.002
    bear_roll_edge_min: float = -0.002
    bull_roll_winrate_min: float = 0.48
    bear_roll_winrate_min: float = 0.48
    bull_cooldown_trades: int = 40
    bear_cooldown_trades: int = 40
    impact_iters: int = 3


def _allowed_side_set(mode: str) -> set[str]:
    token = str(mode).strip().lower()
    if token == "both":
        return {"BULL", "BEAR"}
    if token == "bull_only":
        return {"BULL"}
    if token == "bear_only":
        return {"BEAR"}
    raise ValueError(f"flow_allowed_sides_invalid: {mode}")


def _side_specific_thresholds(cfg: FlowPolicyConfig, side: str) -> tuple[float, float, int]:
    token = str(side).strip().upper()
    if token == "BULL":
        return (
            float(cfg.bull_roll_edge_min),
            float(cfg.bull_roll_winrate_min),
            int(cfg.bull_cooldown_trades),
        )
    if token == "BEAR":
        return (
            float(cfg.bear_roll_edge_min),
            float(cfg.bear_roll_winrate_min),
            int(cfg.bear_cooldown_trades),
        )
    raise ValueError(f"flow_side_invalid: {side}")


def auto_window_sizes(n_rows: int) -> tuple[int, int, int]:
    val = max(2000, min(12000, int(n_rows) // 8))
    train = max(8000, min(40000, int(n_rows) - 2 * int(val)))
    step = int(val)
    return int(train), int(val), int(step)


def walk_forward_slices(*, n_rows: int, train_size: int, val_size: int, step_size: int) -> list[tuple[int, int, int]]:
    if int(train_size) <= 0 or int(val_size) <= 0 or int(step_size) <= 0:
        raise ValueError("flow_walk_forward_sizes_nonpositive")
    out: list[tuple[int, int, int]] = []
    start = 0
    while True:
        tr0 = int(start)
        tr1 = int(tr0) + int(train_size)
        va1 = int(tr1) + int(val_size)
        if int(va1) > int(n_rows):
            break
        out.append((int(tr0), int(tr1), int(va1)))
        start += int(step_size)
    return out


def _normalize_bets(bets: Any) -> list[dict[str, Any]]:
    if not isinstance(bets, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in bets:
        if not isinstance(raw, dict):
            continue
        created_at = _coerce_finite_float(raw.get("createdAt", raw.get("created_at")))
        if not np.isfinite(created_at):
            continue
        amount = _coerce_finite_float(raw.get("amount"))
        amount_wei = _coerce_finite_float(raw.get("amountWei", raw.get("amount_wei")))
        if not np.isfinite(amount) and np.isfinite(amount_wei):
            amount = float(amount_wei) / float(WEI_PER_BNB)
        if not np.isfinite(amount):
            continue
        pos = str(raw.get("position", "")).strip().lower()
        if pos not in ("bull", "bear"):
            continue
        out.append(
            {
                "wallet": raw.get("wallet"),
                "amount": float(amount),
                "createdAt": float(created_at),
                "position": "Bull" if str(pos) == "bull" else "Bear",
            }
        )
    return out


def _compute_final_pool_amounts(bets: list[dict[str, Any]]) -> tuple[float, float, float]:
    bull = 0.0
    bear = 0.0
    for bet in bets:
        amount = pd.to_numeric(bet.get("amount"), errors="coerce")
        if not np.isfinite(amount):
            continue
        pos = str(bet.get("position", "")).strip().lower()
        if pos == "bull":
            bull += float(amount)
        elif pos == "bear":
            bear += float(amount)
    total = float(bull) + float(bear)
    return float(bull), float(bear), float(total)


def _normalize_round_record(row: dict[str, Any]) -> dict[str, Any]:
    bets = _normalize_bets(row.get("bets"))
    bull_amt, bear_amt, total_amt = _compute_final_pool_amounts(bets)
    bull_from_row = pd.to_numeric(row.get("bullAmount"), errors="coerce")
    bear_from_row = pd.to_numeric(row.get("bearAmount"), errors="coerce")
    total_from_row = pd.to_numeric(row.get("totalAmount"), errors="coerce")
    if not np.isfinite(bull_from_row):
        bull_from_row = float(bull_amt)
    if not np.isfinite(bear_from_row):
        bear_from_row = float(bear_amt)
    if not np.isfinite(total_from_row):
        total_from_row = float(total_amt)
    if not np.isfinite(total_from_row):
        total_from_row = float(bull_from_row) + float(bear_from_row)
    if not np.isfinite(bear_from_row):
        bear_from_row = float(total_from_row) - float(bull_from_row)
    return {
        "epoch": row.get("epoch"),
        "startAt": row.get("startAt", row.get("start_at")),
        "lockAt": row.get("lockAt", row.get("lock_at")),
        "closeAt": row.get("closeAt", row.get("close_at")),
        "lockPrice": row.get("lockPrice", row.get("lock_price")),
        "closePrice": row.get("closePrice", row.get("close_price")),
        "position": row.get("position"),
        "failed": row.get("failed", False),
        "bets": bets,
        "bullAmount": bull_from_row,
        "bearAmount": bear_from_row,
        "totalAmount": total_from_row,
    }


def load_rounds_jsonl(path: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            rows.append(_normalize_round_record(raw))
    df = pd.DataFrame(rows)
    for col in ("bullAmount", "bearAmount", "totalAmount", "lockPrice", "closePrice"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("startAt", "lockAt", "closeAt", "epoch"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "failed" in df.columns:
        df["failed"] = df["failed"].astype(bool)
    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


def _parse_bets(bets: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(bets, list) or len(bets) == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=int)
    ts: list[float] = []
    amounts: list[float] = []
    side: list[int] = []
    for bet in bets:
        amount = _coerce_finite_float(bet.get("amount"))
        if not np.isfinite(amount):
            amount_wei = _coerce_finite_float(bet.get("amountWei", bet.get("amount_wei")))
            if np.isfinite(amount_wei):
                amount = float(amount_wei) / float(WEI_PER_BNB)
        ts_raw = _coerce_finite_float(bet.get("createdAt"))
        pos = str(bet.get("position", "")).strip().lower()
        if not np.isfinite(amount) or not np.isfinite(ts_raw):
            continue
        ts.append(float(ts_raw))
        amounts.append(float(amount))
        side.append(1 if str(pos) == "bull" else 0)
    if not ts:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=int)
    return np.asarray(ts, dtype=float), np.asarray(amounts, dtype=float), np.asarray(side, dtype=int)


def _topk_share(amounts: np.ndarray, k: int) -> float:
    if amounts.size == 0:
        return float("nan")
    total = float(np.sum(amounts))
    if total <= 0.0:
        return float("nan")
    kk = int(max(1, min(int(k), int(amounts.size))))
    topk = float(np.sort(amounts)[-kk:].sum())
    return float(topk / total)


def build_flow_features_for_round(round_row: dict[str, Any], cfg: FlowBuildConfig) -> dict[str, float]:
    lock_at = pd.to_numeric(round_row.get("lockAt"), errors="coerce")
    start_at = pd.to_numeric(round_row.get("startAt"), errors="coerce")
    epoch = pd.to_numeric(round_row.get("epoch"), errors="coerce")
    cutoff_ts = float(lock_at) - float(cfg.cutoff_seconds) if np.isfinite(lock_at) else np.nan

    ts_all, amt_all, side_all = _parse_bets(round_row.get("bets"))
    mask_cut = np.isfinite(ts_all)
    if np.isfinite(cutoff_ts):
        mask_cut &= ts_all <= float(cutoff_ts)
    if np.isfinite(lock_at):
        mask_cut &= ts_all <= float(lock_at)

    ts_cut = ts_all[mask_cut]
    amt_cut = amt_all[mask_cut]
    bull_mask = side_all[mask_cut] == 1
    bear_mask = ~bull_mask

    bull_amt_c = float(np.sum(amt_cut[bull_mask])) if amt_cut.size else 0.0
    bear_amt_c = float(np.sum(amt_cut[bear_mask])) if amt_cut.size else 0.0
    total_amt_c = float(bull_amt_c) + float(bear_amt_c)
    bull_n_c = int(np.sum(bull_mask)) if amt_cut.size else 0
    bear_n_c = int(np.sum(bear_mask)) if amt_cut.size else 0
    total_n_c = int(bull_n_c) + int(bear_n_c)
    bull_ratio_c = float(bull_amt_c / (total_amt_c + EPS))
    log_imbalance_c = float(np.log((bull_amt_c + EPS) / (bear_amt_c + EPS)))
    payout_bull_est_c = float(total_amt_c / (bull_amt_c + EPS))
    payout_bear_est_c = float(total_amt_c / (bear_amt_c + EPS))
    payout_ratio_est_c = float(payout_bull_est_c / (payout_bear_est_c + EPS))

    bull_amts = amt_cut[bull_mask] if amt_cut.size else np.zeros(0, dtype=float)
    bear_amts = amt_cut[bear_mask] if amt_cut.size else np.zeros(0, dtype=float)
    max_bull = float(np.max(bull_amts)) if bull_amts.size else 0.0
    max_bear = float(np.max(bear_amts)) if bear_amts.size else 0.0
    whale_time_to_cutoff = float("nan")
    if amt_cut.size and np.isfinite(cutoff_ts):
        whale_idx = int(np.argmax(amt_cut))
        whale_time_to_cutoff = float(cutoff_ts - ts_cut[whale_idx]) if np.isfinite(ts_cut[whale_idx]) else float("nan")

    feats: dict[str, float] = {}
    for window_seconds in cfg.windows_seconds:
        if not np.isfinite(cutoff_ts) or ts_cut.size == 0:
            feats[f"bull_amt_{int(window_seconds)}s"] = 0.0
            feats[f"bear_amt_{int(window_seconds)}s"] = 0.0
            feats[f"bull_n_{int(window_seconds)}s"] = 0.0
            feats[f"bear_n_{int(window_seconds)}s"] = 0.0
            feats[f"late_share_{int(window_seconds)}s"] = 0.0
            feats[f"late_log_imb_{int(window_seconds)}s"] = 0.0
            continue
        mask_window = (ts_cut > (float(cutoff_ts) - float(window_seconds))) & (ts_cut <= float(cutoff_ts))
        if not np.any(mask_window):
            feats[f"bull_amt_{int(window_seconds)}s"] = 0.0
            feats[f"bear_amt_{int(window_seconds)}s"] = 0.0
            feats[f"bull_n_{int(window_seconds)}s"] = 0.0
            feats[f"bear_n_{int(window_seconds)}s"] = 0.0
            feats[f"late_share_{int(window_seconds)}s"] = 0.0
            feats[f"late_log_imb_{int(window_seconds)}s"] = 0.0
            continue
        window_amts = amt_cut[mask_window]
        window_bull_mask = side_all[mask_cut][mask_window] == 1
        bull_amt = float(np.sum(window_amts[window_bull_mask]))
        bear_amt = float(np.sum(window_amts[~window_bull_mask]))
        feats[f"bull_amt_{int(window_seconds)}s"] = float(bull_amt)
        feats[f"bear_amt_{int(window_seconds)}s"] = float(bear_amt)
        feats[f"bull_n_{int(window_seconds)}s"] = float(np.sum(window_bull_mask))
        feats[f"bear_n_{int(window_seconds)}s"] = float(np.sum(~window_bull_mask))
        feats[f"late_share_{int(window_seconds)}s"] = float((bull_amt + bear_amt) / (total_amt_c + EPS))
        feats[f"late_log_imb_{int(window_seconds)}s"] = float(np.log((bull_amt + EPS) / (bear_amt + EPS)))

    if 10 in cfg.windows_seconds and 60 in cfg.windows_seconds:
        net10 = float(feats.get("bull_amt_10s", 0.0) - feats.get("bear_amt_10s", 0.0))
        net60 = float(feats.get("bull_amt_60s", 0.0) - feats.get("bear_amt_60s", 0.0))
        feats["net_accel_10_vs_60"] = float(net10 - net60 * (10.0 / 60.0))
    else:
        feats["net_accel_10_vs_60"] = 0.0

    base = {
        "epoch": float(epoch) if np.isfinite(epoch) else np.nan,
        "cutoff_ts": float(cutoff_ts) if np.isfinite(cutoff_ts) else np.nan,
        "seconds_into_round": float(cutoff_ts - start_at) if np.isfinite(cutoff_ts) and np.isfinite(start_at) else np.nan,
        "round_duration": float(lock_at - start_at) if np.isfinite(lock_at) and np.isfinite(start_at) else np.nan,
        "gap_seconds": np.nan,
        "bull_amt_c": float(bull_amt_c),
        "bear_amt_c": float(bear_amt_c),
        "total_amt_c": float(total_amt_c),
        "bull_n_c": float(bull_n_c),
        "bear_n_c": float(bear_n_c),
        "total_n_c": float(total_n_c),
        "bull_ratio_c": float(bull_ratio_c),
        "log_imbalance_c": float(log_imbalance_c),
        "payout_bull_est_c": float(payout_bull_est_c),
        "payout_bear_est_c": float(payout_bear_est_c),
        "payout_ratio_est_c": float(payout_ratio_est_c),
        "max_bet_bull_c": float(max_bull),
        "max_bet_bear_c": float(max_bear),
        "top1_share_bull_c": float(max_bull / (bull_amt_c + EPS)) if float(bull_amt_c) > 0.0 else np.nan,
        "top1_share_bear_c": float(max_bear / (bear_amt_c + EPS)) if float(bear_amt_c) > 0.0 else np.nan,
        "topk_share_bull_c": float(_topk_share(bull_amts, cfg.topk_for_share)),
        "topk_share_bear_c": float(_topk_share(bear_amts, cfg.topk_for_share)),
        "whale_side_bull": 1.0 if float(max_bull) > float(max_bear) else 0.0,
        "whale_time_to_cutoff": float(whale_time_to_cutoff) if np.isfinite(whale_time_to_cutoff) else np.nan,
    }
    base.update(feats)
    return base


def add_profit_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "bearAmount" not in out.columns or out["bearAmount"].isna().all():
        out["bearAmount"] = out["totalAmount"] - out["bullAmount"]
    out["payout_bull"] = out["totalAmount"] / (out["bullAmount"] + EPS)
    out["payout_bear"] = out["totalAmount"] / (out["bearAmount"] + EPS)
    pos = out["position"].astype(str).str.lower()
    out["winner_is_bull"] = (pos == "bull").astype(int)
    out["winner_is_bear"] = (pos == "bear").astype(int)
    out["y_profit_bull_1x"] = np.where(out["winner_is_bull"] == 1, out["payout_bull"] - 1.0, -1.0)
    out["y_profit_bear_1x"] = np.where(out["winner_is_bear"] == 1, out["payout_bear"] - 1.0, -1.0)
    return out


def build_flow_table(path_jsonl: str, cfg: FlowBuildConfig | None = None) -> pd.DataFrame:
    cfg_obj = cfg or FlowBuildConfig()
    rounds_df = load_rounds_jsonl(path_jsonl)
    if "failed" in rounds_df.columns:
        rounds_df = rounds_df[~rounds_df["failed"]].copy()
    flow_rows = [build_flow_features_for_round(row, cfg_obj) for row in rounds_df.to_dict(orient="records")]
    flow_df = pd.DataFrame(flow_rows)
    if "startAt" in rounds_df.columns and "closeAt" in rounds_df.columns:
        gap = pd.to_numeric(rounds_df["startAt"], errors="coerce") - pd.to_numeric(rounds_df["closeAt"], errors="coerce").shift(1)
        flow_df["gap_seconds"] = gap.astype(float)
    if "lockPrice" in rounds_df.columns:
        lock_price = pd.to_numeric(rounds_df["lockPrice"], errors="coerce").astype(float)
        for lag in cfg_obj.ret_lags:
            flow_df[f"ret_{int(lag)}"] = np.log((lock_price.shift(1) + EPS) / (lock_price.shift(1 + int(lag)) + EPS))
        if "ret_1" in flow_df.columns:
            for window in cfg_obj.vol_windows:
                flow_df[f"vol_{int(window)}"] = flow_df["ret_1"].rolling(int(window)).std()
    keep_df = pd.DataFrame(
        {
            "epoch": pd.to_numeric(rounds_df.get("epoch", np.nan), errors="coerce"),
            "startAt": pd.to_numeric(rounds_df.get("startAt", np.nan), errors="coerce"),
            "lockAt": pd.to_numeric(rounds_df.get("lockAt", np.nan), errors="coerce"),
            "closeAt": pd.to_numeric(rounds_df.get("closeAt", np.nan), errors="coerce"),
            "lockPrice": pd.to_numeric(rounds_df.get("lockPrice", np.nan), errors="coerce"),
            "closePrice": pd.to_numeric(rounds_df.get("closePrice", np.nan), errors="coerce"),
            "bullAmount": pd.to_numeric(rounds_df.get("bullAmount", np.nan), errors="coerce"),
            "bearAmount": pd.to_numeric(rounds_df.get("bearAmount", np.nan), errors="coerce"),
            "totalAmount": pd.to_numeric(rounds_df.get("totalAmount", np.nan), errors="coerce"),
            "position": rounds_df.get("position", None),
        }
    )
    out = pd.concat([keep_df.reset_index(drop=True), flow_df.reset_index(drop=True)], axis=1)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    out = add_profit_labels(out)
    return out.dropna(subset=["epoch", "lockAt"]).reset_index(drop=True)


def flow_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {
        "position",
        "winner_is_bull",
        "winner_is_bear",
        "y_profit_bull_1x",
        "y_profit_bear_1x",
        "payout_bull",
        "payout_bear",
        "bullAmount",
        "bearAmount",
        "totalAmount",
        "startAt",
        "lockAt",
        "closeAt",
        "cutoff_ts",
        "lockPrice",
        "closePrice",
    }
    cols = [col for col in df.columns if col not in drop_cols]
    return [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]


def predict_probabilities_walk_forward(
    *,
    df: pd.DataFrame,
    feature_columns: list[str],
    model_cfg: FlowModelConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    pred_p = np.full(len(df), np.nan)
    slices = walk_forward_slices(
        n_rows=len(df),
        train_size=int(model_cfg.train_size),
        val_size=int(model_cfg.val_size),
        step_size=int(model_cfg.step_size),
    )
    y = df["winner_is_bull"].astype(int)
    for tr0, tr1, va1 in slices:
        x_train = df.iloc[tr0:tr1][feature_columns]
        y_train = y.iloc[tr0:tr1]
        x_val = df.iloc[tr1:va1][feature_columns]
        clf = lgb.LGBMClassifier(
            n_estimators=int(model_cfg.n_estimators),
            learning_rate=float(model_cfg.learning_rate),
            num_leaves=int(model_cfg.num_leaves),
            subsample=float(model_cfg.subsample),
            colsample_bytree=float(model_cfg.colsample_bytree),
            random_state=int(model_cfg.random_seed),
            verbose=-1,
        )
        clf.fit(x_train, y_train)
        pred_p[tr1:va1] = clf.predict_proba(x_val)[:, 1]
    meta = {
        "n_rows": int(len(df)),
        "n_feature_columns": int(len(feature_columns)),
        "n_slices": int(len(slices)),
        "eval_rows": int(np.isfinite(pred_p).sum()),
    }
    return pred_p, meta


def compute_current_odds_ev(
    *,
    df: pd.DataFrame,
    p_bull: np.ndarray,
    treasury_fee_rate: float,
    tx_fee_per_unit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bull_c = pd.to_numeric(df["bull_amt_c"], errors="coerce").values
    bear_c = pd.to_numeric(df["bear_amt_c"], errors="coerce").values
    total_c = bull_c + bear_c
    total_p = total_c + 1.0
    bull_p = bull_c + 1.0
    bear_p = bear_c + 1.0
    payout_bull = (total_p * (1.0 - float(treasury_fee_rate))) / (bull_p + EPS)
    payout_bear = (total_p * (1.0 - float(treasury_fee_rate))) / (bear_p + EPS)
    win_profit_bull = payout_bull - 1.0 - float(tx_fee_per_unit)
    win_profit_bear = payout_bear - 1.0 - float(tx_fee_per_unit)
    lose_profit = -1.0 - float(tx_fee_per_unit)
    p_bull_arr = np.clip(np.asarray(p_bull, dtype=float), 0.0, 1.0)
    p_bear_arr = 1.0 - p_bull_arr
    ev_bull = p_bull_arr * win_profit_bull + p_bear_arr * lose_profit
    ev_bear = p_bear_arr * win_profit_bear + p_bull_arr * lose_profit
    return ev_bull, ev_bear, win_profit_bull, win_profit_bear


def _kelly_fraction_binary(*, p_win: float, win_profit: float, loss_amount: float) -> float:
    p = float(np.clip(p_win, 0.0, 1.0))
    b = float(win_profit)
    a = float(loss_amount)
    if not (np.isfinite(p) and np.isfinite(a) and np.isfinite(b)):
        return 0.0
    if float(a) <= 0.0 or float(b) <= 0.0:
        return 0.0
    numer = float(p) * float(b) - (1.0 - float(p)) * float(a)
    denom = float(a) * float(b)
    return float(max(0.0, float(numer) / max(float(denom), EPS)))


def _impact_unit_profits(
    *,
    total_c: float,
    side_c: float,
    bet: float,
    treasury_fee_rate: float,
    fee_per_unit: float,
    gas_bet_abs: float,
    gas_claim_abs: float,
) -> tuple[float, float]:
    payout = ((float(total_c) + float(bet)) * (1.0 - float(treasury_fee_rate))) / max(float(side_c) + float(bet), EPS)
    unit_gas_bet = float(gas_bet_abs) / max(float(bet), EPS)
    unit_gas_claim = float(gas_claim_abs) / max(float(bet), EPS)
    win_profit = float(payout - 1.0 - float(fee_per_unit) - float(unit_gas_bet) - float(unit_gas_claim))
    lose_profit = float(-1.0 - float(fee_per_unit) - float(unit_gas_bet))
    return float(win_profit), float(lose_profit)


def _impact_ev_per_unit(*, p_win: float, win_profit: float, lose_profit: float) -> float:
    p = float(np.clip(p_win, 0.0, 1.0))
    return float(p * float(win_profit) + (1.0 - float(p)) * float(lose_profit))


def _compute_drawdown(bankroll_series: pd.Series, initial: float) -> tuple[float, float]:
    series = pd.to_numeric(bankroll_series, errors="coerce").ffill().fillna(float(initial)).astype(float)
    peak = series.cummax()
    dd = series - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    if len(dd):
        idx = int(np.argmin(dd.values))
        peak_at = float(peak.iloc[idx]) if np.isfinite(peak.iloc[idx]) else float(initial)
        max_dd_pct = float(max_dd / max(float(peak_at), EPS))
    else:
        max_dd_pct = 0.0
    return float(max_dd), float(max_dd_pct)


def simulate_flow_policy(*, df: pd.DataFrame, cfg: FlowPolicyConfig | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg_obj = cfg or FlowPolicyConfig()
    bankroll = float(cfg_obj.initial_bankroll)
    peak = float(cfg_obj.initial_bankroll)
    allowed_sides = _allowed_side_set(str(cfg_obj.allowed_sides))
    cooldown_by_side = {"BULL": 0, "BEAR": 0}
    realized_unit_pnls_by_side: dict[str, list[float]] = {"BULL": [], "BEAR": []}
    realized_wins_by_side: dict[str, list[int]] = {"BULL": [], "BEAR": []}
    records: list[dict[str, Any]] = []
    for _, row in df.reset_index(drop=True).iterrows():
        peak = max(float(peak), float(bankroll))
        dd_pct = float((float(bankroll) - float(peak)) / float(peak)) if float(peak) > 0.0 else 0.0
        if float(bankroll) <= float(cfg_obj.min_bankroll):
            records.append({**row, "action": "STOP_BANKROLL", "bet_size": 0.0, "bankroll": bankroll})
            break
        if float(dd_pct) <= -abs(float(cfg_obj.drawdown_stop_pct)):
            records.append({**row, "action": "STOP_DRAWDOWN", "bet_size": 0.0, "bankroll": bankroll, "dd_pct": dd_pct})
            break
        bull_c = float(row.get("bull_amt_c", np.nan))
        bear_c = float(row.get("bear_amt_c", np.nan))
        total_c = float(bull_c) + float(bear_c)
        if not (np.isfinite(bull_c) and np.isfinite(bear_c)):
            records.append({**row, "action": "SKIP_POOL_NAN", "bet_size": 0.0, "bankroll": bankroll, "dd_pct": dd_pct})
            continue
        if float(total_c) <= 0.0 or not np.isfinite(total_c):
            records.append({**row, "action": "SKIP_POOL_BAD", "bet_size": 0.0, "bankroll": bankroll, "dd_pct": dd_pct})
            continue
        if float(total_c) < float(cfg_obj.min_total_pool_c):
            records.append({**row, "action": "SKIP_POOL_SMALL", "bet_size": 0.0, "bankroll": bankroll, "total_pool_c": total_c, "dd_pct": dd_pct})
            continue
        bull_ratio_c = float(row.get("bull_ratio_c", np.nan))
        if np.isfinite(bull_ratio_c) and (float(bull_ratio_c) < float(cfg_obj.min_bull_ratio) or float(bull_ratio_c) > float(cfg_obj.max_bull_ratio)):
            records.append({**row, "action": "SKIP_IMBAL", "bet_size": 0.0, "bankroll": bankroll, "bull_ratio_c": bull_ratio_c, "dd_pct": dd_pct})
            continue
        vol_20 = float(row.get("vol_20", np.nan))
        if np.isfinite(vol_20) and float(vol_20) > float(cfg_obj.vol_mid):
            records.append({**row, "action": "SKIP_VOL", "bet_size": 0.0, "bankroll": bankroll, "vol_20": vol_20, "dd_pct": dd_pct})
            continue
        ev_bull = float(row.get("pred_ev_bull", np.nan))
        ev_bear = float(row.get("pred_ev_bear", np.nan))
        p_bull = float(row.get("pred_p_bull", np.nan))
        if "BULL" in allowed_sides and "BEAR" in allowed_sides:
            choose_bull = float(ev_bull) >= float(ev_bear) if np.isfinite(ev_bull) and np.isfinite(ev_bear) else True
            side = "BULL" if bool(choose_bull) else "BEAR"
        elif "BULL" in allowed_sides:
            side = "BULL"
        elif "BEAR" in allowed_sides:
            side = "BEAR"
        else:
            raise ValueError("flow_allowed_sides_empty")
        p_win = float(p_bull) if str(side) == "BULL" else (1.0 - float(p_bull) if np.isfinite(p_bull) else np.nan)
        side_c = float(bull_c) if str(side) == "BULL" else float(bear_c)
        if not np.isfinite(p_win):
            records.append({**row, "action": "SKIP_NAN", "bet_size": 0.0, "bankroll": bankroll, "dd_pct": dd_pct})
            continue
        roll_edge_min, roll_winrate_min, side_cooldown_trades = _side_specific_thresholds(cfg_obj, str(side))
        if int(cooldown_by_side[str(side)]) > 0:
            cooldown_by_side[str(side)] -= 1
            records.append(
                {
                    **row,
                    "action": "COOLDOWN",
                    "bet_size": 0.0,
                    "bankroll": bankroll,
                    "dd_pct": dd_pct,
                    "candidate_side": str(side),
                }
            )
            continue
        side_history = realized_unit_pnls_by_side[str(side)]
        side_wins = realized_wins_by_side[str(side)]
        if len(side_history) >= max(50, int(cfg_obj.roll_window * 0.5)):
            recent = np.asarray(side_history[-int(cfg_obj.roll_window):], dtype=float)
            recent_wins = np.asarray(side_wins[-int(cfg_obj.roll_window):], dtype=int)
            roll_edge = float(np.nanmean(recent)) if recent.size else 0.0
            roll_wr = float(np.mean(recent_wins)) if recent_wins.size else 0.0
            if float(roll_edge) < float(roll_edge_min) or float(roll_wr) < float(roll_winrate_min):
                cooldown_by_side[str(side)] = int(side_cooldown_trades)
                records.append(
                    {
                        **row,
                        "action": "SKIP_REGIME",
                        "bet_size": 0.0,
                        "bankroll": bankroll,
                        "roll_edge": roll_edge,
                        "roll_wr": roll_wr,
                        "dd_pct": dd_pct,
                        "candidate_side": str(side),
                    }
                )
                continue
        bet = max(float(cfg_obj.min_bet_size), min(float(cfg_obj.max_bet_abs), float(bankroll)))
        for _ in range(int(cfg_obj.impact_iters)):
            win_profit, lose_profit = _impact_unit_profits(total_c=float(total_c), side_c=float(side_c), bet=float(bet), treasury_fee_rate=float(cfg_obj.treasury_fee_rate), fee_per_unit=float(cfg_obj.fee_per_unit), gas_bet_abs=float(cfg_obj.gas_bet_abs), gas_claim_abs=float(cfg_obj.gas_claim_abs))
            ev_unit = _impact_ev_per_unit(p_win=float(p_win), win_profit=float(win_profit), lose_profit=float(lose_profit))
            if float(ev_unit) < float(cfg_obj.ev_threshold):
                bet = 0.0
                break
            loss_amount = 1.0 + float(cfg_obj.fee_per_unit) + float(cfg_obj.gas_bet_abs) / max(float(bet), EPS)
            kelly_f = _kelly_fraction_binary(p_win=float(p_win), win_profit=float(win_profit), loss_amount=float(loss_amount))
            frac = float(np.clip(float(cfg_obj.kelly_fraction) * float(kelly_f), 0.0, float(cfg_obj.max_fraction)))
            throttle_scale = 1.0
            if float(dd_pct) < -abs(float(cfg_obj.drawdown_throttle_start_pct)):
                start = abs(float(cfg_obj.drawdown_throttle_start_pct))
                stop = abs(float(cfg_obj.drawdown_stop_pct))
                dd_abs = abs(float(dd_pct))
                if float(dd_abs) >= float(stop):
                    throttle_scale = float(cfg_obj.drawdown_throttle_min_scale)
                else:
                    t = (float(dd_abs) - float(start)) / max(float(stop) - float(start), EPS)
                    throttle_scale = float(1.0 - float(t) * (1.0 - float(cfg_obj.drawdown_throttle_min_scale)))
                throttle_scale = float(np.clip(throttle_scale, float(cfg_obj.drawdown_throttle_min_scale), 1.0))
            frac *= float(throttle_scale)
            bet = min(float(frac) * float(bankroll), float(cfg_obj.max_bet_abs), float(bankroll))
            bet = max(float(bet), float(cfg_obj.min_bet_size))
            bet = float(np.floor(float(bet) / float(cfg_obj.round_to)) * float(cfg_obj.round_to))
        if float(bet) <= 0.0:
            records.append({**row, "action": "SKIP_EV_IMPACT", "bet_size": 0.0, "bankroll": bankroll, "dd_pct": dd_pct})
            continue
        pool_cap = min(float(cfg_obj.max_total_pool_share) * float(total_c), float(cfg_obj.max_side_pool_share) * max(float(side_c), EPS))
        if float(bet) > float(pool_cap):
            bet2 = float(np.floor(float(pool_cap) / float(cfg_obj.round_to)) * float(cfg_obj.round_to))
            if float(bet2) >= float(cfg_obj.min_bet_size):
                bet = float(bet2)
            else:
                records.append({**row, "action": "SKIP_POOL_CAP", "bet_size": 0.0, "bankroll": bankroll, "pool_cap": pool_cap, "dd_pct": dd_pct})
                continue
        win_profit, lose_profit = _impact_unit_profits(total_c=float(total_c), side_c=float(side_c), bet=float(bet), treasury_fee_rate=float(cfg_obj.treasury_fee_rate), fee_per_unit=float(cfg_obj.fee_per_unit), gas_bet_abs=float(cfg_obj.gas_bet_abs), gas_claim_abs=float(cfg_obj.gas_claim_abs))
        ev_unit = _impact_ev_per_unit(p_win=float(p_win), win_profit=float(win_profit), lose_profit=float(lose_profit))
        total_f = pd.to_numeric(row.get("totalAmount", np.nan), errors="coerce")
        side_f = pd.to_numeric(row.get("bullAmount" if str(side) == "BULL" else "bearAmount", np.nan), errors="coerce")
        winner = str(row.get("position", "")).strip().upper()
        if not (np.isfinite(total_f) and np.isfinite(side_f) and float(side_f) > 0.0):
            records.append({**row, "action": "SKIP_NOY", "bet_size": 0.0, "bankroll": bankroll, "dd_pct": dd_pct})
            continue
        won = str(winner) == str(side)
        if bool(won):
            payout_realized = (float(total_f) * (1.0 - float(cfg_obj.treasury_fee_rate))) / max(float(side_f), EPS)
            unit_realized = float(payout_realized) - 1.0 - float(cfg_obj.fee_per_unit) - float(cfg_obj.gas_bet_abs) / max(float(bet), EPS) - float(cfg_obj.gas_claim_abs) / max(float(bet), EPS)
        else:
            unit_realized = -1.0 - float(cfg_obj.fee_per_unit) - float(cfg_obj.gas_bet_abs) / max(float(bet), EPS)
        pnl = float(unit_realized) * float(bet)
        bankroll = float(max(0.0, float(bankroll) + float(pnl)))
        peak = max(float(peak), float(bankroll))
        realized_unit_pnls_by_side[str(side)].append(float(unit_realized))
        realized_wins_by_side[str(side)].append(1 if float(pnl) > 0.0 else 0)
        records.append({**row, "action": str(side), "bet_size": float(bet), "p_win": float(p_win), "impact_win_profit_unit": float(win_profit), "impact_ev_unit": float(ev_unit), "pnl": float(pnl), "bankroll": float(bankroll), "dd_pct": float(dd_pct), "pool_total_c": float(total_c), "pool_side_c": float(side_c), "pool_share_total": float(bet / max(float(total_c), EPS)), "pool_share_side": float(bet / max(float(side_c), EPS))})
    out = pd.DataFrame(records)
    bankroll_series = pd.to_numeric(out.get("bankroll", pd.Series([float(cfg_obj.initial_bankroll)])), errors="coerce").ffill().fillna(float(cfg_obj.initial_bankroll))
    end_bankroll = float(bankroll_series.iloc[-1]) if len(bankroll_series) else float(cfg_obj.initial_bankroll)
    bet_sizes = pd.to_numeric(out.get("bet_size", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    pnls = pd.to_numeric(out.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    bets = int((bet_sizes > 0.0).sum())
    wins = int((pnls > 0.0).sum())
    losses = int((pnls < 0.0).sum())
    skips = int(out.get("action", pd.Series(dtype=str)).astype(str).str.startswith("SKIP").sum())
    max_dd, max_dd_pct = _compute_drawdown(bankroll_series, float(cfg_obj.initial_bankroll))
    actions = out.get("action", pd.Series(dtype=str)).astype(str)
    bull_mask = actions == "BULL"
    bear_mask = actions == "BEAR"
    metrics = {
        "start_bankroll": float(cfg_obj.initial_bankroll),
        "end_bankroll": float(end_bankroll),
        "return_multiple": float(end_bankroll / float(cfg_obj.initial_bankroll)) if float(cfg_obj.initial_bankroll) > 0.0 else None,
        "bets": int(bets),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(int(wins) / max(1, int(wins) + int(losses))),
        "skips": int(skips),
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
        "bull_bets": int(bull_mask.sum()),
        "bear_bets": int(bear_mask.sum()),
        "bull_net_profit_bnb": float(pnls[bull_mask].sum()) if len(pnls) else 0.0,
        "bear_net_profit_bnb": float(pnls[bear_mask].sum()) if len(pnls) else 0.0,
    }
    return metrics, out
