"""p3b orderflow features computation.

Per orchestrator v1.1 (locked):

12 trade-tape features per round, computed using ONLY trades with
ts <= lock_at - 2s (data horizon discipline). Trades partitioned to round
R if `ts` falls within R's [start_at, lock_at - 2s] window.

OKX side semantics (R4 locked):
  side="buy"  → taker hit the ask → aggressive BUY
  side="sell" → taker hit the bid → aggressive SELL

Tier 0 (confounder-mediated): feature 11 (return_volatility_5m)
Tier 1 (microstructure-specific): features 1-10, 12 (10 features)

Conditional R3 expansion: trade_volume_total may be reclassified as Tier 0
post-pre-flight if |r(trade_volume_total, outcome)| >= 0.05 on n=1,941 post-v1.
"""
from __future__ import annotations

import math
from typing import Sequence

DATA_HORIZON_OFFSET_S = 2
LATE_FLOW_WINDOW_S = 60      # for trade_velocity_late60s, bid_pressure_imbalance
PRICE_VEL_WINDOW_S = 60      # for price_velocity_late60s
ACCEL_LATE_S = 30            # for trade_volume_acceleration (numerator window)
ACCEL_TOTAL_S = 60           # for trade_volume_acceleration (denominator window)
VWAP_WINDOW_S = 300          # 5-minute VWAP
VOL_BUCKET_S = 10            # 10s buckets for return_volatility_5m
LARGE_TRADE_PERCENTILE = 90  # 90th percentile threshold for large_trade_ratio

EPOCH_DURATION = 300


FEATURE_NAMES = [
    "n_trades", "trade_volume_total", "buy_volume_ratio", "avg_trade_size",
    "large_trade_ratio", "trade_velocity_late60s", "price_velocity_late60s",
    "price_drift_directional", "bid_pressure_imbalance",
    "trade_volume_acceleration", "return_volatility_5m", "vwap_deviation",
]
TIER0_FEATURES = {"return_volatility_5m"}  # Phase 0 already tested vol
TIER1_FEATURES = {n for n in FEATURE_NAMES if n not in TIER0_FEATURES}


def parse_trade(t: dict) -> tuple[int, str, float, float]:
    """Returns (ts_ms, side, sz, px). Caller must own filtering."""
    return int(t["ts"]), t["side"], float(t["sz"]), float(t["px"])


def compute_features(
    round_start_at: int,
    trades: Sequence[dict],
    *,
    large_size_threshold_usd: float,
) -> dict:
    """Compute 12 trade-tape features for one round.

    `round_start_at` is the round's start_at in seconds (Unix). The round's
    lock_at = start_at + 300s; data cutoff = lock_at - 2s.

    `trades` is the list of trades whose ts falls in [start_at, cutoff)
    in seconds (caller pre-filters by epoch-binning + timestamp).

    `large_size_threshold_usd` is the global 90th-percentile trade-size in USD,
    pre-computed across the relevant slice (passed from the caller). If 0 or
    None, large_trade_ratio returns NaN.

    Returns dict with all 12 features. NaN where undefined (e.g. no trades).
    """
    nan = float("nan")
    if not trades:
        return {f: nan for f in FEATURE_NAMES}

    lock_at = round_start_at + EPOCH_DURATION
    cutoff_s = lock_at - DATA_HORIZON_OFFSET_S
    cutoff_ms = cutoff_s * 1000
    start_ms = round_start_at * 1000

    # Pre-parse: per-trade (ts_ms, side, sz, px, usd_size)
    parsed = []
    for t in trades:
        ts_ms = int(t["ts"])
        if ts_ms < start_ms or ts_ms >= cutoff_ms:
            continue
        sz = float(t["sz"])
        px = float(t["px"])
        usd = sz * px
        parsed.append((ts_ms, t["side"], sz, px, usd))

    n = len(parsed)
    if n == 0:
        return {f: nan for f in FEATURE_NAMES}

    # Aggregate basic
    total_volume = sum(p[4] for p in parsed)  # USD
    n_trades_val = float(n)
    avg_trade_size_val = total_volume / n if n > 0 else nan

    # Side-segregated volumes (R4 semantics: buy = aggressive buy)
    buy_volume = sum(p[4] for p in parsed if p[1] == "buy")
    sell_volume = sum(p[4] for p in parsed if p[1] == "sell")
    buy_volume_ratio_val = buy_volume / total_volume if total_volume > 0 else nan

    # Large-trade ratio
    if large_size_threshold_usd and large_size_threshold_usd > 0:
        n_large = sum(1 for p in parsed if p[4] > large_size_threshold_usd)
        large_trade_ratio_val = n_large / n
    else:
        large_trade_ratio_val = nan

    # Late-60s window features
    late60_lo_ms = (cutoff_s - LATE_FLOW_WINDOW_S) * 1000
    late60_trades = [p for p in parsed if p[0] >= late60_lo_ms]
    n_late60 = len(late60_trades)
    trade_velocity_val = n_late60 / (LATE_FLOW_WINDOW_S - DATA_HORIZON_OFFSET_S)
    if late60_trades:
        late60_buy = sum(p[4] for p in late60_trades if p[1] == "buy")
        late60_sell = sum(p[4] for p in late60_trades if p[1] == "sell")
        late60_total = late60_buy + late60_sell
        bid_pressure_val = ((late60_buy - late60_sell) / late60_total
                              if late60_total > 0 else nan)
    else:
        bid_pressure_val = nan

    # Price velocity late60s: |px[cutoff-2] - px[cutoff-62]| / px[cutoff-62]
    # We need px sampled at two timestamps within [cutoff-62s, cutoff-2s].
    # Use the last trade in [cutoff-62s, cutoff-32s] as "early" reference,
    # and the last trade in [cutoff-32s, cutoff) as "late" reference.
    # Conservative: just take first and last trade in the late60 window.
    if len(late60_trades) >= 2:
        late60_sorted = sorted(late60_trades, key=lambda p: p[0])
        px_early = late60_sorted[0][3]
        px_late = late60_sorted[-1][3]
        if px_early > 0:
            price_velocity_val = abs(px_late - px_early) / px_early
        else:
            price_velocity_val = nan
    else:
        price_velocity_val = nan

    # Price drift directional: sign-aware px change over [start_at, cutoff)
    parsed_sorted = sorted(parsed, key=lambda p: p[0])
    px_first = parsed_sorted[0][3]
    px_last = parsed_sorted[-1][3]
    if px_first > 0:
        price_drift_val = (px_last - px_first) / px_first
    else:
        price_drift_val = nan

    # Trade volume acceleration: (vol_last30s / vol_last60s) - 0.5
    accel_30_lo_ms = (cutoff_s - ACCEL_LATE_S) * 1000
    accel_60_lo_ms = (cutoff_s - ACCEL_TOTAL_S) * 1000
    vol_30 = sum(p[4] for p in parsed if p[0] >= accel_30_lo_ms)
    vol_60 = sum(p[4] for p in parsed if p[0] >= accel_60_lo_ms)
    if vol_60 > 0:
        accel_val = (vol_30 / vol_60) - 0.5
    else:
        accel_val = nan

    # Return volatility (5m): std of log returns over 10s buckets
    # Bucket boundaries align to cutoff_s as latest data point.
    vol_lo_ms = (cutoff_s - VWAP_WINDOW_S) * 1000
    bucket_trades = [p for p in parsed_sorted if p[0] >= vol_lo_ms]
    if len(bucket_trades) >= 4:
        # Aggregate into buckets: take last px in each 10s bucket
        n_buckets = VWAP_WINDOW_S // VOL_BUCKET_S
        bucket_pxs: list[float | None] = [None] * n_buckets
        for p in bucket_trades:
            bucket_idx = min(n_buckets - 1, (cutoff_ms - p[0]) // (VOL_BUCKET_S * 1000))
            bucket_idx = n_buckets - 1 - int(bucket_idx)  # forward-time
            if 0 <= bucket_idx < n_buckets:
                bucket_pxs[bucket_idx] = p[3]  # latest in bucket overwrites
        # Forward-fill empty buckets
        last = None
        filled: list[float] = []
        for bp in bucket_pxs:
            if bp is not None:
                filled.append(bp)
                last = bp
            elif last is not None:
                filled.append(last)
        log_rets = []
        for i in range(1, len(filled)):
            if filled[i - 1] > 0 and filled[i] > 0:
                log_rets.append(math.log(filled[i] / filled[i - 1]))
        if len(log_rets) >= 2:
            mean_lr = sum(log_rets) / len(log_rets)
            var_lr = sum((x - mean_lr) ** 2 for x in log_rets) / (len(log_rets) - 1)
            return_vol_val = math.sqrt(var_lr)
        else:
            return_vol_val = nan
    else:
        return_vol_val = nan

    # VWAP deviation: (px_last - vwap_5m) / vwap_5m, where vwap_5m = sum(px*sz)/sum(sz)
    vwap_trades = [p for p in parsed if p[0] >= vol_lo_ms]
    if vwap_trades:
        vwap_num = sum(p[3] * p[2] for p in vwap_trades)
        vwap_denom = sum(p[2] for p in vwap_trades)
        if vwap_denom > 0:
            vwap_5m = vwap_num / vwap_denom
            vwap_dev_val = (px_last - vwap_5m) / vwap_5m if vwap_5m > 0 else nan
        else:
            vwap_dev_val = nan
    else:
        vwap_dev_val = nan

    return {
        "n_trades": n_trades_val,
        "trade_volume_total": total_volume,
        "buy_volume_ratio": buy_volume_ratio_val,
        "avg_trade_size": avg_trade_size_val,
        "large_trade_ratio": large_trade_ratio_val,
        "trade_velocity_late60s": trade_velocity_val,
        "price_velocity_late60s": price_velocity_val,
        "price_drift_directional": price_drift_val,
        "bid_pressure_imbalance": bid_pressure_val,
        "trade_volume_acceleration": accel_val,
        "return_volatility_5m": return_vol_val,
        "vwap_deviation": vwap_dev_val,
    }


def round_outcome(round_rec: dict) -> int | None:
    """1 if Bull won, 0 if Bear won, None for House/failed/unsettled."""
    if round_rec.get("failed"):
        return None
    pos = round_rec.get("position")
    if pos == "Bull":
        return 1
    if pos == "Bear":
        return 0
    return None
