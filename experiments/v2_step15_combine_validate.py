"""Step 15: Combine best signals from step 14 and 5-fold validate.

Top signals discovered:
1. Multi-timeframe BTC agreement (3+7+15 all same direction)
2. Volume confirmation (rising vol during momentum)
3. Correlation filter (medium corr = signal works)
4. VWAP return (volume-weighted momentum)

This script:
1. Test all pairwise + triple combinations
2. 5-fold validate the best combos
3. Sweep thresholds on best combos
4. Pool-proportional sizing sweep on winners
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import (
    GAS_COST_BET_BNB, INTERVAL_SECONDS, POOL_CUTOFF_SECONDS,
)
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import _trim_to_window, _get_return
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2
POOL_CUTOFF_S = POOL_CUTOFF_SECONDS
CANDLE_COUNT = 31
TREASURY_FEE = 0.03
SKIP_NIGHT = {0, 1, 2, 3, 4, 23}


def load_data():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())

    def load_kl(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    return rounds, load_kl("var/cutoff_spot_prices.jsonl"), load_kl("var/btc_spot_prices.jsonl")


def get_candles(raw, cutoff_ms):
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None
    return trimmed[-CANDLE_COUNT:]


def settle(rnd, bet_bnb, side):
    out = settle_bet_against_closed_round(
        bet_bnb=bet_bnb, bet_side=side,
        round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
    )
    return out.credit_bnb - bet_bnb - GAS_COST_BET_BNB


def ret(closes, lb):
    if len(closes) < lb + 1 or closes[-(lb+1)] == 0:
        return None
    return (closes[-1] - closes[-(lb+1)]) / closes[-(lb+1)]


def get_pool_at_cutoff(rnd, lock_at):
    pool_cutoff_ts = lock_at - POOL_CUTOFF_S
    bull_wei = bear_wei = 0
    for bet in rnd.bets:
        if int(bet.created_at) > pool_cutoff_ts:
            continue
        if bet.position == "Bull":
            bull_wei += int(bet.amount_wei)
        else:
            bear_wei += int(bet.amount_wei)
    return bull_wei / 1e18, bear_wei / 1e18


def compute_features(data_item):
    """Compute all signal features for one round."""
    d = data_item
    btc_c = d["btc_c"]
    btc_v = d["btc_v"]
    bnb_c = d["bnb_c"]
    feat = {}

    # 1. Multi-timeframe BTC signals
    for lb in [3, 5, 7, 10, 15, 20]:
        r = ret(btc_c, lb)
        feat[f"btc_r_{lb}"] = r

    # 2. BTC acceleration (2s in same direction as longer)
    feat["btc_r_2"] = ret(btc_c, 2)

    # 3. Volume trend
    if len(btc_v) >= 10:
        recent_vol = sum(btc_v[-5:]) / 5
        baseline_vol = sum(btc_v[-10:-5]) / 5
        feat["vol_ratio"] = recent_vol / baseline_vol if baseline_vol > 0 else 0
    else:
        feat["vol_ratio"] = 0

    # 4. VWAP return
    for lb in [5, 7, 10]:
        total_vol = sum(btc_v[-lb:])
        if total_vol > 0 and len(btc_c) >= lb + 1:
            vw = 0
            for i in range(lb):
                idx = -(lb - i)
                if btc_c[idx - 1] != 0:
                    r_i = (btc_c[idx] - btc_c[idx - 1]) / btc_c[idx - 1]
                    vw += r_i * (btc_v[idx] / total_vol)
            feat[f"vwap_{lb}"] = vw
        else:
            feat[f"vwap_{lb}"] = None

    # 5. BTC-BNB rolling correlation (30-candle window)
    if len(btc_c) >= 30 and len(bnb_c) >= 30:
        btc_rets = [(btc_c[i] - btc_c[i-1]) / btc_c[i-1]
                    for i in range(-29, 0) if btc_c[i-1] != 0]
        bnb_rets = [(bnb_c[i] - bnb_c[i-1]) / bnb_c[i-1]
                    for i in range(-29, 0) if bnb_c[i-1] != 0]
        if len(btc_rets) == len(bnb_rets) and len(btc_rets) >= 15:
            n = len(btc_rets)
            mb = sum(btc_rets) / n
            mn = sum(bnb_rets) / n
            cov = sum((btc_rets[i] - mb) * (bnb_rets[i] - mn) for i in range(n)) / n
            sb = (sum((r - mb)**2 for r in btc_rets) / n) ** 0.5
            sn = (sum((r - mn)**2 for r in bnb_rets) / n) ** 0.5
            feat["corr_30"] = cov / (sb * sn) if sb > 0 and sn > 0 else 0
        else:
            feat["corr_30"] = None
    else:
        feat["corr_30"] = None

    return feat


def signal_multi_tf(feat, timeframes, thresh):
    """Multi-timeframe: all specified lookbacks agree and exceed threshold."""
    signals = []
    for lb in timeframes:
        r = feat.get(f"btc_r_{lb}")
        if r is None or abs(r) < thresh:
            return None
        signals.append(1 if r > 0 else -1)
    if len(set(signals)) > 1:
        return None
    return "Bull" if signals[0] > 0 else "Bear"


def signal_btc_accel(feat, lb, thresh):
    """BTC lookback + 2s acceleration."""
    r = feat.get(f"btc_r_{lb}")
    r2 = feat.get("btc_r_2")
    if r is None or r2 is None:
        return None
    if abs(r) < thresh:
        return None
    if (r2 > 0) != (r > 0):
        return None
    return "Bull" if r > 0 else "Bear"


def signal_vwap(feat, lb, thresh):
    """VWAP return signal."""
    vw = feat.get(f"vwap_{lb}")
    if vw is None or abs(vw) < thresh:
        return None
    return "Bull" if vw > 0 else "Bear"


def filter_vol_confirm(feat, min_ratio=1.2):
    """Volume must be rising."""
    return feat.get("vol_ratio", 0) >= min_ratio


def filter_corr(feat, lo, hi):
    """Correlation must be in range."""
    c = feat.get("corr_30")
    if c is None:
        return False
    return lo <= c < hi


def run_strategy(data_list, signal_fn, filter_fn=None, bet_bnb=0.10, pool_frac=None):
    """Run a strategy over data, returning list of (profit, bet, source)."""
    results = []
    for d in data_list:
        feat = d["feat"]
        signal = signal_fn(feat)
        if signal is None:
            continue
        if filter_fn and not filter_fn(feat):
            continue

        if pool_frac is not None:
            pb, pe = get_pool_at_cutoff(d["rnd"], int(d["rnd"].lock_at))
            vis_pool = pb + pe
            bet = max(0.01, min(2.0, vis_pool * pool_frac))
        else:
            bet = bet_bnb

        profit = settle(d["rnd"], bet, signal)
        results.append((profit, bet))
    return results


def print_result(label, results, total_rounds):
    n = len(results)
    if n < 20:
        print(f"  {label}: N={n} (too few)")
        return
    profits = [p for p, b in results]
    bets = [b for p, b in results]
    wins = sum(1 for p in profits if p > 0)
    wr = wins / n * 100
    pnl = sum(profits)
    pnl_2k = pnl / total_rounds * 2000
    avg_bet = sum(bets) / n
    bets_2k = n / total_rounds * 2000
    flag = " ***" if pnl > 0 else ""
    print(f"  {label}: WR={wr:5.1f}%({n:4d}) PnL={pnl:+7.3f} avg_bet={avg_bet:.3f} "
          f"/2k={pnl_2k:+6.3f}({bets_2k:.0f}b){flag}")


def main():
    rounds, bnb_kl, btc_kl = load_data()
    total = len(rounds)
    print(f"Total rounds: {total}")

    # Pre-compute all data and features
    data = []
    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        hour = (lock_at % 86400) // 3600
        if hour in SKIP_NIGHT:
            continue
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        btc_raw = btc_kl.get(epoch)
        bnb_raw = bnb_kl.get(epoch)
        if not btc_raw or not bnb_raw:
            continue
        btc_candles = get_candles(btc_raw, cutoff_ms)
        bnb_candles = get_candles(bnb_raw, cutoff_ms)
        if btc_candles is None or bnb_candles is None:
            continue

        d = {
            "rnd": rnd,
            "epoch": epoch,
            "hour": hour,
            "btc_c": [k[4] for k in btc_candles],
            "btc_v": [k[5] for k in btc_candles],
            "bnb_c": [k[4] for k in bnb_candles],
        }
        d["feat"] = compute_features(d)
        data.append(d)

    print(f"Rounds with data: {len(data)}\n")

    # Compute correlation percentiles for filter thresholds
    corrs = [d["feat"]["corr_30"] for d in data if d["feat"]["corr_30"] is not None]
    corrs.sort()
    corr_p33 = corrs[len(corrs) // 3]
    corr_p67 = corrs[2 * len(corrs) // 3]
    print(f"Correlation percentiles: p33={corr_p33:.3f} p67={corr_p67:.3f}\n")

    # =====================================================================
    print("=" * 120)
    print("PART 1: Individual signals (baseline)")
    print("=" * 120)

    configs = [
        ("multi_tf(3+7+15,0.0003)", lambda f: signal_multi_tf(f, [3,7,15], 0.0003)),
        ("multi_tf(3+7+15,0.0005)", lambda f: signal_multi_tf(f, [3,7,15], 0.0005)),
        ("multi_tf(5+10+20,0.0005)", lambda f: signal_multi_tf(f, [5,10,20], 0.0005)),
        ("btc(5,0.0007)+accel", lambda f: signal_btc_accel(f, 5, 0.0007)),
        ("btc(7,0.0007)+accel", lambda f: signal_btc_accel(f, 7, 0.0007)),
        ("vwap(7,0.0007)", lambda f: signal_vwap(f, 7, 0.0007)),
        ("vwap(10,0.0007)", lambda f: signal_vwap(f, 10, 0.0007)),
    ]

    for name, sig_fn in configs:
        results = run_strategy(data, sig_fn)
        print_result(name, results, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 2: Signals + volume confirmation filter")
    print("=" * 120)

    for vol_min in [1.0, 1.2, 1.5, 2.0]:
        for name, sig_fn in configs:
            filt = lambda f, vm=vol_min: filter_vol_confirm(f, vm)
            results = run_strategy(data, sig_fn, filt)
            print_result(f"{name}+vol>={vol_min}", results, total)
        print()

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 3: Signals + correlation filter (medium corr)")
    print("=" * 120)

    for name, sig_fn in configs:
        filt_med = lambda f: filter_corr(f, corr_p33, corr_p67)
        filt_low = lambda f: filter_corr(f, -1.0, corr_p33)
        filt_high = lambda f: filter_corr(f, corr_p67, 1.0)
        results_med = run_strategy(data, sig_fn, filt_med)
        results_low = run_strategy(data, sig_fn, filt_low)
        results_high = run_strategy(data, sig_fn, filt_high)
        print_result(f"{name}+low_corr", results_low, total)
        print_result(f"{name}+med_corr", results_med, total)
        print_result(f"{name}+high_corr", results_high, total)
        print()

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 4: Best combos — signal + vol + corr filters stacked")
    print("=" * 120)

    best_combos = [
        ("multi_tf(3+7+15,0.0003)", lambda f: signal_multi_tf(f, [3,7,15], 0.0003)),
        ("multi_tf(3+7+15,0.0005)", lambda f: signal_multi_tf(f, [3,7,15], 0.0005)),
        ("btc(5,0.0007)+accel", lambda f: signal_btc_accel(f, 5, 0.0007)),
        ("btc(7,0.0007)+accel", lambda f: signal_btc_accel(f, 7, 0.0007)),
    ]

    for name, sig_fn in best_combos:
        for vol_min in [1.0, 1.2, 1.5]:
            for corr_range, lo, hi in [("all", -1, 1), ("low", -1, corr_p33),
                                        ("med", corr_p33, corr_p67)]:
                filt = lambda f, vm=vol_min, clo=lo, chi=hi: (
                    filter_vol_confirm(f, vm) and filter_corr(f, clo, chi)
                )
                if corr_range == "all":
                    filt = lambda f, vm=vol_min: filter_vol_confirm(f, vm)

                results = run_strategy(data, sig_fn, filt)
                print_result(f"{name}+vol>={vol_min}+{corr_range}_corr", results, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 5: 5-fold validation of top strategies")
    print("=" * 120)

    # Identify strategies to validate
    strategies_to_validate = [
        ("multi_tf(3+7+15,0.0003)",
         lambda f: signal_multi_tf(f, [3,7,15], 0.0003), None),
        ("multi_tf(3+7+15,0.0003)+vol>=1.2",
         lambda f: signal_multi_tf(f, [3,7,15], 0.0003),
         lambda f: filter_vol_confirm(f, 1.2)),
        ("btc(5,0.0007)+accel+vol>=1.2",
         lambda f: signal_btc_accel(f, 5, 0.0007),
         lambda f: filter_vol_confirm(f, 1.2)),
        ("multi_tf(3+7+15,0.0003)+vol>=1.2+med_corr",
         lambda f: signal_multi_tf(f, [3,7,15], 0.0003),
         lambda f: filter_vol_confirm(f, 1.2) and filter_corr(f, corr_p33, corr_p67)),
        ("multi_tf(3+7+15,0.0005)+vol>=1.2",
         lambda f: signal_multi_tf(f, [3,7,15], 0.0005),
         lambda f: filter_vol_confirm(f, 1.2)),
    ]

    fold_size = len(data) // 5
    for name, sig_fn, filt_fn in strategies_to_validate:
        print(f"\n  --- {name} ---")
        fold_pnls = []
        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size
            fold_data = data[start:end]
            results = run_strategy(fold_data, sig_fn, filt_fn)
            profits = [p for p, b in results]
            n = len(results)
            wr = sum(1 for p in profits if p > 0) / max(1, n) * 100
            pnl = sum(profits)
            pnl_2k = pnl / len(fold_data) * 2000
            fold_pnls.append(pnl_2k)
            print(f"    Fold {fold+1}: WR={wr:5.1f}%({n:3d}) PnL={pnl:+6.3f} /2k={pnl_2k:+6.3f}")

        avg = sum(fold_pnls) / 5
        mn = min(fold_pnls)
        mx = max(fold_pnls)
        pos = sum(1 for p in fold_pnls if p > 0)
        print(f"    => avg /2k={avg:+.3f} (min={mn:+.3f} max={mx:+.3f}) {pos}/5 positive folds")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 6: Pool-proportional sizing on best validated strategies")
    print("=" * 120)

    for name, sig_fn, filt_fn in strategies_to_validate:
        for frac in [0.05, 0.10, 0.15, 0.20]:
            results = run_strategy(data, sig_fn, filt_fn, pool_frac=frac)
            print_result(f"{name} frac={frac}", results, total)

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 7: Stacking signals — combine multi_tf + btc_accel on non-overlapping rounds")
    print("=" * 120)

    for vol_min in [1.0, 1.2]:
        trades_all = []
        trades_mtf = []
        trades_accel = []
        for d in data:
            feat = d["feat"]
            if not filter_vol_confirm(feat, vol_min):
                continue

            # Try multi_tf first, then btc_accel
            sig = signal_multi_tf(feat, [3,7,15], 0.0003)
            source = "mtf"
            if sig is None:
                sig = signal_btc_accel(feat, 5, 0.0007)
                source = "accel"
            if sig is None:
                sig = signal_vwap(feat, 7, 0.0007)
                source = "vwap"
            if sig is None:
                continue

            profit = settle(d["rnd"], 0.10, sig)
            trades_all.append((profit, 0.10))
            if source == "mtf":
                trades_mtf.append((profit, 0.10))
            elif source == "accel":
                trades_accel.append((profit, 0.10))

        print_result(f"stacked(mtf+accel+vwap)+vol>={vol_min}", trades_all, total)
        print_result(f"  -> mtf component", trades_mtf, total)
        print_result(f"  -> accel component", trades_accel, total)

    print("\nDone.")


if __name__ == "__main__":
    main()
