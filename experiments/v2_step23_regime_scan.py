"""Step 23: Regime-dependent signal scan.

The equity curve is lumpy because BTC multi-TF(3,7,15) only fires in
volatile regimes. This experiment re-tests ALL "dead-end" signals
on ONLY the flat periods (rounds where BTC multi-TF does NOT fire).

A signal with 0% edge overall could have positive edge specifically
in quiet BTC markets — different regime, different dynamics.

Tests:
  1. BNB own momentum (various lookbacks)
  2. BTC weaker momentum (2-of-3 TF agreement)
  3. ETH/SOL standalone momentum
  4. Mean reversion (BTC went up, bet BNB down)
  5. BTC-BNB spread signal
  6. Volume spikes / divergence
  7. Pool-imbalance contrarian
  8. Previous round outcome
  9. Candle microstructure (wicks, body ratio)
  10. BTC acceleration (2nd derivative)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pancakebot.domain.strategy.momentum_gate as _gate_mod
from pancakebot.core.constants import (
    BNB_WEI, GAS_COST_BET_BNB, POOL_CUTOFF_SECONDS, TREASURY_FEE_FRACTION,
)
from pancakebot.domain.strategy.momentum_gate import (
    compute_signal_from_klines, _trim_to_window,
)
from pancakebot.domain.strategy.momentum_pipeline import _pools_from_bets
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2
CANDLE_COUNT = 31
N_FOLDS = 5
TREASURY_FEE = 0.03
MIN_POOL = 1.5
MIN_PAYOUT = 1.5
BET_FRAC = 0.05  # fixed fraction for signal comparison (removes sizing noise)
BET_CAP = 2.0


# ---- Data loading ----

def load_data():
    print("Loading data...", end=" ", flush=True)
    t0 = time.time()
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())

    def lk(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    bnb = lk("var/bnb_spot_prices.jsonl")
    btc = lk("var/btc_spot_prices.jsonl")
    eth = lk("var/eth_spot_prices.jsonl")
    sol = lk("var/sol_spot_prices.jsonl")
    print(f"{len(rounds)} rounds, {time.time()-t0:.1f}s")
    return rounds, bnb, btc, eth, sol


# ---- Feature computation ----

def get_closes(raw_klines, cutoff_ms):
    if raw_klines is None:
        return None
    trimmed = _trim_to_window(raw_klines, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None
    candles = trimmed[-CANDLE_COUNT:]
    return [c[4] for c in candles]  # close prices


def get_volumes(raw_klines, cutoff_ms):
    if raw_klines is None:
        return None
    trimmed = _trim_to_window(raw_klines, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None
    candles = trimmed[-CANDLE_COUNT:]
    return [c[5] for c in candles]  # volumes


def get_ohlc(raw_klines, cutoff_ms):
    if raw_klines is None:
        return None
    trimmed = _trim_to_window(raw_klines, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None
    return trimmed[-CANDLE_COUNT:]  # full OHLC


def ret(closes, lb):
    """Return over lookback period. closes[-1] is most recent."""
    if closes is None or len(closes) < lb + 1:
        return None
    if closes[-(lb + 1)] == 0:
        return None
    return (closes[-1] - closes[-(lb + 1)]) / closes[-(lb + 1)]


def multi_tf_fires(closes, lookbacks=(3, 7, 15), thresh=0.0001):
    """Check if multi-TF signal fires (all agree, all > thresh)."""
    rets = [ret(closes, lb) for lb in lookbacks]
    if any(r is None for r in rets):
        return False
    if not (all(r > 0 for r in rets) or all(r < 0 for r in rets)):
        return False
    if min(abs(r) for r in rets) < thresh:
        return False
    return True


# ---- Pre-compute everything ----

def precompute(rounds, bnb_kl, btc_kl, eth_kl, sol_kl):
    print("Pre-computing features...", end=" ", flush=True)
    t0 = time.time()

    data = []
    for rnd in rounds:
        ep = int(rnd.epoch)
        la = int(rnd.lock_at)
        cms = (la - CUTOFF_S) * 1000

        bnb_c = get_closes(bnb_kl.get(ep), cms)
        btc_c = get_closes(btc_kl.get(ep), cms)
        eth_c = get_closes(eth_kl.get(ep), cms)
        sol_c = get_closes(sol_kl.get(ep), cms)
        btc_v = get_volumes(btc_kl.get(ep), cms)
        bnb_v = get_volumes(bnb_kl.get(ep), cms)
        btc_ohlc = get_ohlc(btc_kl.get(ep), cms)
        bnb_ohlc = get_ohlc(bnb_kl.get(ep), cms)

        pb, pe = _pools_from_bets(rnd, la - POOL_CUTOFF_SECONDS)
        pt = pb + pe

        # Check if primary signal fires (to identify flat rounds)
        btc_fires = btc_c is not None and multi_tf_fires(btc_c)

        data.append({
            "rnd": rnd, "epoch": ep, "lock_at": la,
            "bnb_c": bnb_c, "btc_c": btc_c, "eth_c": eth_c, "sol_c": sol_c,
            "btc_v": btc_v, "bnb_v": bnb_v,
            "btc_ohlc": btc_ohlc, "bnb_ohlc": bnb_ohlc,
            "pool_bull": pb, "pool_bear": pe, "pool_total": pt,
            "btc_fires": btc_fires,
        })

    n_flat = sum(1 for d in data if not d["btc_fires"])
    print(f"{len(data)} rounds, {n_flat} flat ({n_flat/len(data)*100:.1f}%), {time.time()-t0:.1f}s")
    return data


# ---- Signal functions (return "Bull"/"Bear"/None) ----

def sig_bnb_momentum(d, lb, thresh):
    r = ret(d["bnb_c"], lb)
    if r is None or abs(r) < thresh:
        return None
    return "Bull" if r > 0 else "Bear"


def sig_btc_2of3(d, thresh):
    """BTC 2-of-3 TF agreement (fires when primary 3-of-3 doesn't)."""
    if d["btc_c"] is None:
        return None
    rets = [ret(d["btc_c"], lb) for lb in (3, 7, 15)]
    if any(r is None for r in rets):
        return None
    signs = [1 if r > 0 else -1 for r in rets]
    # Must be exactly 2 agreeing (not 3, since primary already handles 3)
    if signs.count(1) == 2 or signs.count(-1) == 2:
        majority = "Bull" if signs.count(1) >= 2 else "Bear"
        # Use strength of the agreeing pair
        agreeing = [abs(r) for r, s in zip(rets, signs) if (s > 0) == (majority == "Bull")]
        if min(agreeing) >= thresh:
            return majority
    return None


def sig_eth_standalone(d, lb, thresh):
    r = ret(d["eth_c"], lb)
    if r is None or abs(r) < thresh:
        return None
    return "Bull" if r > 0 else "Bear"


def sig_sol_standalone(d, lb, thresh):
    r = ret(d["sol_c"], lb)
    if r is None or abs(r) < thresh:
        return None
    return "Bull" if r > 0 else "Bear"


def sig_eth_multi_tf(d, thresh):
    if d["eth_c"] is None:
        return None
    rets = [ret(d["eth_c"], lb) for lb in (3, 7, 15)]
    if any(r is None for r in rets):
        return None
    if not (all(r > 0 for r in rets) or all(r < 0 for r in rets)):
        return None
    if min(abs(r) for r in rets) < thresh:
        return None
    return "Bull" if rets[0] > 0 else "Bear"


def sig_sol_multi_tf(d, thresh):
    if d["sol_c"] is None:
        return None
    rets = [ret(d["sol_c"], lb) for lb in (3, 7, 15)]
    if any(r is None for r in rets):
        return None
    if not (all(r > 0 for r in rets) or all(r < 0 for r in rets)):
        return None
    if min(abs(r) for r in rets) < thresh:
        return None
    return "Bull" if rets[0] > 0 else "Bear"


def sig_mean_reversion(d, lb, thresh):
    """If BTC moved strongly, bet BNB opposite."""
    r = ret(d["btc_c"], lb)
    if r is None or abs(r) < thresh:
        return None
    return "Bear" if r > 0 else "Bull"  # opposite of BTC


def sig_btc_bnb_spread(d, lb, thresh):
    """If BTC moved but BNB didn't follow, bet BNB catches up."""
    btc_r = ret(d["btc_c"], lb)
    bnb_r = ret(d["bnb_c"], lb)
    if btc_r is None or bnb_r is None:
        return None
    spread = btc_r - bnb_r
    if abs(spread) < thresh:
        return None
    # Bet BNB catches up to BTC
    return "Bull" if spread > 0 else "Bear"


def sig_volume_spike(d, thresh_mult=2.0):
    """BTC volume spike — bet in direction of price move during spike."""
    if d["btc_v"] is None or d["btc_c"] is None:
        return None
    v = d["btc_v"]
    if len(v) < 15:
        return None
    recent = sum(v[-3:]) / 3
    baseline = sum(v[-15:-3]) / 12
    if baseline <= 0 or recent / baseline < thresh_mult:
        return None
    r = ret(d["btc_c"], 3)
    if r is None or abs(r) < 0.00005:
        return None
    return "Bull" if r > 0 else "Bear"


def sig_pool_contrarian(d, min_ratio=1.5):
    """Bet against the crowd (opposite of larger pool side)."""
    if d["pool_bull"] <= 0 or d["pool_bear"] <= 0:
        return None
    ratio = d["pool_bull"] / d["pool_bear"]
    if ratio > min_ratio:
        return "Bear"  # crowd is Bull, bet Bear
    elif 1 / ratio > min_ratio:
        return "Bull"  # crowd is Bear, bet Bull
    return None


def sig_prev_round_outcome(d, prev_data):
    """Bet same direction as previous round's actual outcome."""
    if prev_data is None:
        return None
    rnd = prev_data["rnd"]
    if rnd.lock_price is None or rnd.close_price is None:
        return None
    if rnd.close_price > rnd.lock_price:
        return "Bull"
    elif rnd.close_price < rnd.lock_price:
        return "Bear"
    return None


def sig_btc_accel(d, lb_long, lb_short, thresh):
    """BTC acceleration: short momentum stronger than long in same direction."""
    r_long = ret(d["btc_c"], lb_long)
    r_short = ret(d["btc_c"], lb_short)
    if r_long is None or r_short is None:
        return None
    if abs(r_short) < thresh:
        return None
    # Must agree in direction, short must be stronger
    if (r_short > 0) != (r_long > 0):
        return None
    if abs(r_short) <= abs(r_long):
        return None
    return "Bull" if r_short > 0 else "Bear"


def sig_wick_reversal(d, min_wick_ratio=2.0):
    """Long wick on recent BTC candle suggests reversal."""
    if d["btc_ohlc"] is None or len(d["btc_ohlc"]) < 3:
        return None
    # Check last 3 candles
    for candle in d["btc_ohlc"][-3:]:
        o, h, l, c = candle[1], candle[2], candle[3], candle[4]
        body = abs(c - o)
        if body < 1e-10:
            continue
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        if upper_wick > body * min_wick_ratio:
            return "Bear"  # long upper wick = reversal down
        if lower_wick > body * min_wick_ratio:
            return "Bull"  # long lower wick = reversal up
    return None


def sig_btc_momentum_single(d, lb, thresh):
    """Single BTC lookback (weaker than multi-TF but fires more often)."""
    r = ret(d["btc_c"], lb)
    if r is None or abs(r) < thresh:
        return None
    return "Bull" if r > 0 else "Bear"


# ---- Evaluation ----

def evaluate_signal(data, signal_fn, flat_only=True):
    """Evaluate a signal function on data. Returns (bets, wins, total_pnl, n_rounds)."""
    bets = 0
    wins = 0
    total_pnl = 0.0

    for i, d in enumerate(data):
        if flat_only and d["btc_fires"]:
            continue  # skip rounds where primary signal fires

        signal = signal_fn(d, data[i-1] if i > 0 else None) if "prev" in signal_fn.__name__ else signal_fn(d)
        if signal is None:
            continue

        pt = d["pool_total"]
        if pt < MIN_POOL:
            continue

        our_side = d["pool_bull"] if signal == "Bull" else d["pool_bear"]
        if our_side > 0 and pt > 0:
            payout = pt * 0.97 / our_side
            if payout < MIN_PAYOUT:
                continue

        bet = max(0.01, min(BET_CAP, pt * BET_FRAC))

        outcome = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=signal,
            round_closed=d["rnd"], treasury_fee_fraction=TREASURY_FEE,
        )
        profit = outcome.credit_bnb - bet - GAS_COST_BET_BNB
        total_pnl += profit
        bets += 1
        if profit > 0:
            wins += 1

    return bets, wins, total_pnl


def evaluate_5fold(data, signal_fn, flat_only=True):
    """5-fold validation of a signal."""
    fold_size = len(data) // N_FOLDS
    folds = [data[i * fold_size:(i + 1) * fold_size] for i in range(N_FOLDS)]
    fold_results = []
    for fold in folds:
        bets, wins, pnl = evaluate_signal(fold, signal_fn, flat_only)
        pnl_2k = pnl / len(fold) * 2000 if len(fold) > 0 else 0
        fold_results.append((bets, wins, pnl, pnl_2k))
    return fold_results


def print_result(label, fold_results, total_flat_rounds):
    total_bets = sum(f[0] for f in fold_results)
    total_wins = sum(f[1] for f in fold_results)
    total_pnl = sum(f[2] for f in fold_results)
    pnl_2ks = [f[3] for f in fold_results]
    avg_pnl_2k = sum(pnl_2ks) / len(pnl_2ks)
    n_pos = sum(1 for p in pnl_2ks if p > 0)
    wr = total_wins / total_bets * 100 if total_bets > 0 else 0
    bets_2k = total_bets / total_flat_rounds * 2000 if total_flat_rounds > 0 else 0

    marker = " ***" if n_pos >= 5 else " **" if n_pos >= 4 else " *" if n_pos >= 3 else ""
    fold_str = " ".join(f"{p:+6.2f}" for p in pnl_2ks)
    print(f"  {label:<45s} {bets_2k:6.1f} {wr:5.1f}% {avg_pnl_2k:+7.2f} {n_pos}/5  {fold_str}{marker}")


# ---- Main ----

def main():
    rounds, bnb_kl, btc_kl, eth_kl, sol_kl = load_data()
    data = precompute(rounds, bnb_kl, btc_kl, eth_kl, sol_kl)

    total_rounds = len(data)
    flat_rounds = sum(1 for d in data if not d["btc_fires"])

    print(f"\nTotal rounds: {total_rounds}")
    print(f"Flat rounds (BTC multi-TF silent): {flat_rounds} ({flat_rounds/total_rounds*100:.1f}%)")
    print(f"Signal rounds: {total_rounds - flat_rounds}")

    print(f"\n{'='*120}")
    print("REGIME SCAN: Testing signals on FLAT rounds only (BTC multi-TF doesn't fire)")
    print(f"Filters: pool >= {MIN_POOL}, payout >= {MIN_PAYOUT}, bet = {BET_FRAC*100:.0f}% of pool (cap {BET_CAP})")
    print(f"{'='*120}")
    print(f"  {'Signal':<45s} {'b/2k':>6} {'WR%':>6} {'PnL/2k':>8} {'pos':>4}  {'f1':>6} {'f2':>6} {'f3':>6} {'f4':>6} {'f5':>6}")
    print(f"  {'-'*110}")

    # ---- 1. BNB own momentum ----
    print("\n  --- BNB Own Momentum ---")
    for lb in [3, 5, 7, 10, 15]:
        for thresh in [0.0001, 0.0002, 0.0003]:
            fr = evaluate_5fold(data, lambda d, lb=lb, th=thresh: sig_bnb_momentum(d, lb, th))
            print_result(f"BNB r{lb} > {thresh}", fr, flat_rounds)

    # ---- 2. BTC 2-of-3 TF agreement ----
    print("\n  --- BTC 2-of-3 Multi-TF ---")
    for thresh in [0.00005, 0.0001, 0.00015, 0.0002]:
        fr = evaluate_5fold(data, lambda d, th=thresh: sig_btc_2of3(d, th))
        print_result(f"BTC 2of3 > {thresh}", fr, flat_rounds)

    # ---- 3. BTC single lookback (weaker signals) ----
    print("\n  --- BTC Single Lookback ---")
    for lb in [3, 5, 7, 10, 15]:
        for thresh in [0.0001, 0.0002, 0.0003]:
            fr = evaluate_5fold(data, lambda d, lb=lb, th=thresh: sig_btc_momentum_single(d, lb, th))
            print_result(f"BTC r{lb} > {thresh}", fr, flat_rounds)

    # ---- 4. ETH/SOL standalone multi-TF ----
    print("\n  --- ETH/SOL Multi-TF Standalone ---")
    for thresh in [0.0001, 0.00015, 0.0002]:
        fr = evaluate_5fold(data, lambda d, th=thresh: sig_eth_multi_tf(d, th))
        print_result(f"ETH multi-TF > {thresh}", fr, flat_rounds)
        fr = evaluate_5fold(data, lambda d, th=thresh: sig_sol_multi_tf(d, th))
        print_result(f"SOL multi-TF > {thresh}", fr, flat_rounds)

    # ---- 5. Mean reversion ----
    print("\n  --- Mean Reversion (fade BTC) ---")
    for lb in [3, 7, 15]:
        for thresh in [0.0001, 0.0002, 0.0003]:
            fr = evaluate_5fold(data, lambda d, lb=lb, th=thresh: sig_mean_reversion(d, lb, th))
            print_result(f"Fade BTC r{lb} > {thresh}", fr, flat_rounds)

    # ---- 6. BTC-BNB spread ----
    print("\n  --- BTC-BNB Spread (BNB catches up to BTC) ---")
    for lb in [3, 5, 7, 10, 15]:
        for thresh in [0.0001, 0.0002, 0.0003]:
            fr = evaluate_5fold(data, lambda d, lb=lb, th=thresh: sig_btc_bnb_spread(d, lb, th))
            print_result(f"Spread r{lb} > {thresh}", fr, flat_rounds)

    # ---- 7. Volume spike ----
    print("\n  --- BTC Volume Spike ---")
    for mult in [1.5, 2.0, 3.0]:
        fr = evaluate_5fold(data, lambda d, m=mult: sig_volume_spike(d, m))
        print_result(f"Vol spike > {mult}x", fr, flat_rounds)

    # ---- 8. Pool contrarian ----
    print("\n  --- Pool Contrarian ---")
    for ratio in [1.3, 1.5, 2.0, 3.0]:
        fr = evaluate_5fold(data, lambda d, r=ratio: sig_pool_contrarian(d, r))
        print_result(f"Pool contrarian > {ratio}x", fr, flat_rounds)

    # ---- 9. Previous round outcome ----
    print("\n  --- Previous Round Outcome ---")
    def _prev_same(d, prev=None):
        return sig_prev_round_outcome(d, prev)
    # Need special handling for prev_round
    fold_size = len(data) // N_FOLDS
    fr = []
    for fi in range(N_FOLDS):
        fold = data[fi * fold_size:(fi + 1) * fold_size]
        bets = wins = 0
        pnl = 0.0
        for i, d in enumerate(fold):
            if d["btc_fires"]:
                continue
            prev = fold[i-1] if i > 0 else None
            signal = sig_prev_round_outcome(d, prev)
            if signal is None:
                continue
            pt = d["pool_total"]
            if pt < MIN_POOL:
                continue
            our_side = d["pool_bull"] if signal == "Bull" else d["pool_bear"]
            if our_side > 0 and pt > 0:
                pay = pt * 0.97 / our_side
                if pay < MIN_PAYOUT:
                    continue
            bet = max(0.01, min(BET_CAP, pt * BET_FRAC))
            outcome = settle_bet_against_closed_round(
                bet_bnb=bet, bet_side=signal,
                round_closed=d["rnd"], treasury_fee_fraction=TREASURY_FEE,
            )
            profit = outcome.credit_bnb - bet - GAS_COST_BET_BNB
            pnl += profit
            bets += 1
            if profit > 0:
                wins += 1
        pnl_2k = pnl / len(fold) * 2000 if fold else 0
        fr.append((bets, wins, pnl, pnl_2k))
    print_result("Prev round same direction", fr, flat_rounds)

    # ---- 10. Wick reversal ----
    print("\n  --- Candle Wick Reversal ---")
    for wr in [1.5, 2.0, 3.0]:
        fr = evaluate_5fold(data, lambda d, r=wr: sig_wick_reversal(d, r))
        print_result(f"Wick reversal > {wr}x body", fr, flat_rounds)

    # ---- 11. BTC acceleration ----
    print("\n  --- BTC Acceleration (short > long, same dir) ---")
    for lb_long, lb_short in [(15, 3), (15, 5), (15, 7), (10, 3), (10, 5), (7, 3)]:
        for thresh in [0.0001, 0.0002]:
            fr = evaluate_5fold(data, lambda d, ll=lb_long, ls=lb_short, th=thresh: sig_btc_accel(d, ll, ls, th))
            print_result(f"BTC accel r{lb_short}/r{lb_long} > {thresh}", fr, flat_rounds)

    print(f"\n{'='*120}")
    print("DONE. Look for signals with 4/5 or 5/5 positive folds and meaningful PnL.")
    print("These could be added as regime-2 signals for flat periods.")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
