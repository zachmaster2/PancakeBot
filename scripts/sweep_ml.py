"""ML sweep: walk-forward validation on BNB 1s kline features.

Features: returns at multiple lookbacks, volume ratios, VWAP deviation,
volatility (high-low range), trend slope, payout multiple.

Walk-forward: train on first N rounds, predict next 500, slide by 500.
Only reports out-of-sample performance.

Models: Logistic Regression and LightGBM (if available).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROUNDS_PATH = Path("var/closed_rounds.jsonl")
DATA_PATH = Path("var/cutoff_spot_prices.jsonl")

BNB_WEI = 10**18
BET = 0.05
GAS_BET = 0.0002
GAS_CLAIM = 0.00025
FEE = 0.03
CUTOFF_SECONDS = 4


def find_closest(klines, target_ms):
    best, bd = None, float("inf")
    for k in klines:
        d = abs(k[0] - target_ms)
        if d < bd:
            bd, best = d, k
    return best if best and bd <= 2000 else None


def compute_pools(rnd):
    lock_at = rnd["lockAt"]
    bw, ew = 0, 0
    for b in rnd.get("bets", []):
        if b["createdAt"] > lock_at:
            continue
        if b["position"] == "Bull":
            bw += b["amountWei"]
        else:
            ew += b["amountWei"]
    return bw, ew


def payout_multiple(bull_wei, bear_wei, side, bet_bnb=0.05):
    bet_wei = int(bet_bnb * BNB_WEI)
    bw = bull_wei + (bet_wei if side == "Bull" else 0)
    ew = bear_wei + (bet_wei if side == "Bear" else 0)
    tw = bw + ew
    my = bw if side == "Bull" else ew
    return (tw * (1 - FEE)) / my if my > 0 else 0


def net_profit(bw, ew, side, outcome, bet_bnb=0.05):
    m = payout_multiple(bw, ew, side, bet_bnb)
    return bet_bnb * m - GAS_CLAIM - bet_bnb - GAS_BET if outcome == side else -bet_bnb - GAS_BET


def get_return(klines, cutoff_ms, lookback_s):
    kn = find_closest(klines, cutoff_ms)
    ka = find_closest(klines, cutoff_ms - lookback_s * 1000)
    if not kn or not ka or ka[4] <= 0:
        return None
    return (kn[4] / ka[4]) - 1


def build_features(klines, cutoff_ms, bull_wei, bear_wei):
    """Build feature vector from 1s klines."""
    feats = {}

    # Returns at various lookbacks
    for lb in [3, 5, 7, 10, 15, 20, 30, 45, 60, 90]:
        ret = get_return(klines, cutoff_ms, lb)
        feats[f"ret_{lb}"] = ret if ret is not None else 0.0

    # Volume features
    kn = find_closest(klines, cutoff_ms)
    if kn is None:
        return None
    spot_now = kn[4]
    if spot_now <= 0:
        return None

    for window in [5, 10, 20, 30, 60]:
        start_ms = cutoff_ms - window * 1000
        total_vol = 0.0
        vwap_num = 0.0
        highs, lows = [], []
        n_nonzero = 0
        for k in klines:
            if start_ms <= k[0] <= cutoff_ms:
                vol = k[5]
                total_vol += vol
                mid = (k[1] + k[4]) / 2
                vwap_num += mid * vol
                highs.append(k[2])
                lows.append(k[3])
                if k[4] != k[1]:  # close != open
                    n_nonzero += 1

        # Volume ratio (recent vs prior window)
        bg_vol = 0.0
        for k in klines:
            if cutoff_ms - 2 * window * 1000 <= k[0] < start_ms:
                bg_vol += k[5]
        feats[f"vol_ratio_{window}"] = (total_vol / bg_vol) if bg_vol > 0 else 1.0

        # VWAP deviation
        if total_vol > 0:
            vwap = vwap_num / total_vol
            feats[f"vwap_dev_{window}"] = (spot_now / vwap) - 1
        else:
            feats[f"vwap_dev_{window}"] = 0.0

        # High-low range
        if highs and lows:
            max_h = max(highs)
            min_l = min(lows)
            feats[f"hl_range_{window}"] = (max_h - min_l) / min_l if min_l > 0 else 0.0
        else:
            feats[f"hl_range_{window}"] = 0.0

        # Active tick fraction
        feats[f"active_frac_{window}"] = n_nonzero / max(1, window)

    # Trend slope (linear regression on 20s window)
    for window in [10, 20, 60]:
        xs, ys = [], []
        for k in klines:
            if cutoff_ms - window * 1000 <= k[0] <= cutoff_ms:
                xs.append((k[0] - cutoff_ms) / 1000)
                ys.append(k[4])
        if len(xs) >= 3:
            n = len(xs)
            sx = sum(xs); sy = sum(ys)
            sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
            denom = n * sxx - sx * sx
            if denom != 0:
                slope = (n * sxy - sx * sy) / denom
                y_mean = sy / n
                feats[f"trend_slope_{window}"] = slope / y_mean if y_mean > 0 else 0.0
                # R-squared
                intercept = (sy - slope * sx) / n
                ss_tot = sum((y - y_mean)**2 for y in ys)
                ss_res = sum((y - (slope*x + intercept))**2 for x, y in zip(xs, ys))
                feats[f"trend_r2_{window}"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            else:
                feats[f"trend_slope_{window}"] = 0.0
                feats[f"trend_r2_{window}"] = 0.0
        else:
            feats[f"trend_slope_{window}"] = 0.0
            feats[f"trend_r2_{window}"] = 0.0

    # Pool features
    total_pool = bull_wei + bear_wei
    if total_pool > 0:
        feats["bull_frac"] = bull_wei / total_pool
        feats["payout_bull"] = payout_multiple(bull_wei, bear_wei, "Bull")
        feats["payout_bear"] = payout_multiple(bull_wei, bear_wei, "Bear")
        feats["pool_size_bnb"] = total_pool / BNB_WEI
    else:
        feats["bull_frac"] = 0.5
        feats["payout_bull"] = 1.94
        feats["payout_bear"] = 1.94
        feats["pool_size_bnb"] = 0.0

    return feats


def main():
    records = []
    for line in DATA_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                records.append(r)
    rounds_by_epoch = {}
    for line in ROUNDS_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rounds_by_epoch[r["epoch"]] = r

    print(f"Loaded {len(records)} records\n")

    # Build dataset
    samples = []
    for rec in records:
        rnd = rounds_by_epoch.get(rec["epoch"])
        if not rnd or rnd.get("failed") or rnd["position"] not in ("Bull", "Bear"):
            continue
        kl = rec["klines_1s"]
        lock_ms = rec["lock_at"] * 1000
        cutoff_ms = lock_ms - CUTOFF_SECONDS * 1000
        bull_wei, bear_wei = compute_pools(rnd)
        if bull_wei + bear_wei == 0:
            continue

        feats = build_features(kl, cutoff_ms, bull_wei, bear_wei)
        if feats is None:
            continue

        samples.append({
            "features": feats,
            "outcome": 1 if rnd["position"] == "Bull" else 0,
            "outcome_label": rnd["position"],
            "bull_wei": bull_wei,
            "bear_wei": bear_wei,
        })

    print(f"Samples: {len(samples)}")

    # Convert to numpy
    feature_names = sorted(samples[0]["features"].keys())
    print(f"Features: {len(feature_names)}")
    print(f"  {feature_names}\n")

    X = np.array([[s["features"][f] for f in feature_names] for s in samples])
    y = np.array([s["outcome"] for s in samples])

    print(f"Overall bull rate: {y.mean():.3f}\n")

    # =========================================================
    # Walk-forward validation
    # =========================================================
    TRAIN_SIZE = 2000
    STEP_SIZE = 500

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False
        print("LightGBM not available, using LogisticRegression + RandomForest only\n")

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    models_to_test = {
        "LogReg": lambda: LogisticRegression(max_iter=1000, C=0.1),
        "LogReg_C1": lambda: LogisticRegression(max_iter=1000, C=1.0),
        "RF_50": lambda: RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        "RF_100": lambda: RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        "GBT_50": lambda: GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
        "GBT_100": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
    }
    if has_lgb:
        models_to_test["LGB_50"] = lambda: lgb.LGBMClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            verbose=-1, random_state=42)
        models_to_test["LGB_100"] = lambda: lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            verbose=-1, random_state=42)

    print("=" * 85)
    print("WALK-FORWARD VALIDATION")
    print(f"  Train size: {TRAIN_SIZE}, Step: {STEP_SIZE}")
    print(f"  Total samples: {len(samples)}")
    print("=" * 85)

    for model_name, model_factory in models_to_test.items():
        all_preds = []
        all_probs = []
        all_true = []
        all_indices = []

        start = 0
        while start + TRAIN_SIZE + STEP_SIZE <= len(samples):
            train_end = start + TRAIN_SIZE
            test_end = min(train_end + STEP_SIZE, len(samples))

            X_train = X[start:train_end]
            y_train = y[start:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]

            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Train
            model = model_factory()
            model.fit(X_train_s, y_train)

            # Predict
            preds = model.predict(X_test_s)
            probs = model.predict_proba(X_test_s)[:, 1]  # P(Bull)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_true.extend(y_test)
            all_indices.extend(range(train_end, test_end))

            start += STEP_SIZE

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)

        # Overall accuracy
        acc = accuracy_score(all_true, all_preds)

        # Simulate trading with different confidence thresholds
        print(f"\n  {model_name}: OOS accuracy={acc:.3f} ({len(all_preds)} predictions)")

        for min_conf in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
            bets, wins, pnl = 0, 0, 0.0
            for i, idx in enumerate(all_indices):
                prob_bull = all_probs[i]
                if max(prob_bull, 1 - prob_bull) < min_conf:
                    continue
                direction = "Bull" if prob_bull > 0.5 else "Bear"
                s = samples[idx]
                bets += 1
                if direction == s["outcome_label"]:
                    wins += 1
                pnl += net_profit(s["bull_wei"], s["bear_wei"],
                                  direction, s["outcome_label"])
            if bets > 0:
                wr = wins / bets
                print(f"    conf>={min_conf:.2f}: bets={bets:>5}  WR={wr:.1%}  "
                      f"PnL={pnl:+.4f}  PnL/bet={pnl/bets:+.6f}")
            else:
                print(f"    conf>={min_conf:.2f}: bets=    0")

        # Feature importance (for tree models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            print(f"    Top features: {[(feature_names[i], f'{importances[i]:.3f}') for i in top_idx]}")

    # =========================================================
    # Expanding window (more conservative)
    # =========================================================
    print("\n" + "=" * 85)
    print("EXPANDING WINDOW: train on ALL prior data")
    print("=" * 85)

    for model_name, model_factory in [
        ("LogReg", lambda: LogisticRegression(max_iter=1000, C=0.1)),
        ("GBT_50", lambda: GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)),
    ]:
        all_preds = []
        all_probs = []
        all_true = []
        all_indices = []

        for test_start in range(TRAIN_SIZE, len(samples), STEP_SIZE):
            test_end = min(test_start + STEP_SIZE, len(samples))

            X_train = X[:test_start]
            y_train = y[:test_start]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = model_factory()
            model.fit(X_train_s, y_train)

            preds = model.predict(X_test_s)
            probs = model.predict_proba(X_test_s)[:, 1]

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_true.extend(y_test)
            all_indices.extend(range(test_start, test_end))

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)

        acc = accuracy_score(all_true, all_preds)
        print(f"\n  {model_name} (expanding): OOS accuracy={acc:.3f}")

        for min_conf in [0.50, 0.52, 0.55, 0.58, 0.60]:
            bets, wins, pnl = 0, 0, 0.0
            for i, idx in enumerate(all_indices):
                prob_bull = all_probs[i]
                if max(prob_bull, 1 - prob_bull) < min_conf:
                    continue
                direction = "Bull" if prob_bull > 0.5 else "Bear"
                s = samples[idx]
                bets += 1
                if direction == s["outcome_label"]:
                    wins += 1
                pnl += net_profit(s["bull_wei"], s["bear_wei"],
                                  direction, s["outcome_label"])
            if bets > 0:
                wr = wins / bets
                print(f"    conf>={min_conf:.2f}: bets={bets:>5}  WR={wr:.1%}  "
                      f"PnL={pnl:+.4f}  PnL/bet={pnl/bets:+.6f}")


if __name__ == "__main__":
    main()
