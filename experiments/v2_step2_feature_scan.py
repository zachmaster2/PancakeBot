"""Step 2: Systematic feature scan on corrected data.

Explores fundamentally different features beyond momentum returns:
- Volatility patterns (realized vol, vol ratio)
- Volume patterns (volume imbalance, volume trend)
- Price microstructure (consecutive tick direction, range position)
- Cross-asset signals (BTC return predicting BNB round outcome)
- Return reversal (contrarian to recent move)
- Composite features

Walk-forward: train on first 70%, validate on last 30%.
Fixed bet size for pure signal evaluation.
"""
from __future__ import annotations

import json, sys, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB, INTERVAL_SECONDS
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import _trim_to_window, _get_return
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 4
CANDLE_COUNT = 31
TREASURY_FEE = 0.03
BET_SIZE = 0.10


def load_data():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    def load_kl(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out
    return rounds, load_kl("var/cutoff_spot_prices.jsonl"), load_kl("var/btc_spot_prices.jsonl")


def get_closes_and_volumes(raw_klines, cutoff_ms):
    trimmed = _trim_to_window(raw_klines, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None, None
    closes = [k[4] for k in trimmed]
    volumes = [k[5] for k in trimmed]
    return closes, volumes


def eval_feature(rounds, spot, btc, *, feature_fn):
    """Evaluate a feature function that returns ("Bull"|"Bear"|None) per round."""
    n_bets = 0
    n_wins = 0
    pnl = 0.0

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        bnb_raw = spot.get(epoch)
        btc_raw = btc.get(epoch)
        if not bnb_raw:
            continue

        bnb_closes, bnb_vols = get_closes_and_volumes(bnb_raw, cutoff_ms)
        if bnb_closes is None:
            continue

        btc_closes, btc_vols = None, None
        if btc_raw:
            btc_closes, btc_vols = get_closes_and_volumes(btc_raw, cutoff_ms)

        signal = feature_fn(bnb_closes, bnb_vols, btc_closes, btc_vols)
        if signal is None:
            continue

        out = settle_bet_against_closed_round(
            bet_bnb=BET_SIZE, bet_side=signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - BET_SIZE - GAS_COST_BET_BNB
        n_bets += 1
        if profit > 0:
            n_wins += 1
        pnl += profit

    wr = n_wins / max(1, n_bets) * 100
    return n_bets, n_wins, wr, pnl


def run_feature(name, rounds_train, rounds_valid, spot, btc, feature_fn):
    nb_t, _, wr_t, pnl_t = eval_feature(rounds_train, spot, btc, feature_fn=feature_fn)
    if nb_t < 100:
        return
    nb_v, _, wr_v, pnl_v = eval_feature(rounds_valid, spot, btc, feature_fn=feature_fn)
    flag = " ***" if wr_v > 55 else " *" if wr_v > 53 else ""
    print(f"  {name:40s}  T: {wr_t:5.1f}% ({nb_t:5d})  V: {wr_v:5.1f}% ({nb_v:5d}) {pnl_v:+7.2f}{flag}")


def main():
    rounds, spot, btc = load_data()
    split = int(len(rounds) * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    print(f"Rounds: {len(rounds)} total, {len(train)} train, {len(valid)} validate\n")

    # =====================================================================
    print("=" * 80)
    print("CATEGORY 1: Volatility patterns")
    print("=" * 80)

    for window in [5, 10, 15, 20]:
        for vol_thresh in [0.0002, 0.0004, 0.0008]:
            def make_vol_breakout(w=window, vt=vol_thresh):
                def fn(bc, bv, tc, tv):
                    if len(bc) < w + 1: return None
                    recent = bc[-w:]
                    rets = [(recent[i] / recent[i-1]) - 1 for i in range(1, len(recent))]
                    vol = math.sqrt(sum(r*r for r in rets) / len(rets))
                    if vol < vt: return None
                    # Bet in direction of the move during high vol
                    net_ret = (bc[-1] / bc[-w]) - 1
                    return "Bull" if net_ret > 0 else "Bear"
                return fn
            run_feature(f"vol_breakout(w={window},vt={vol_thresh})",
                       train, valid, spot, btc, make_vol_breakout())

    # Vol ratio: recent vol vs longer-term vol
    for short_w, long_w in [(5, 20), (5, 15), (10, 25)]:
        for ratio_thresh in [1.5, 2.0, 3.0]:
            def make_vol_ratio(sw=short_w, lw=long_w, rt=ratio_thresh):
                def fn(bc, bv, tc, tv):
                    if len(bc) < lw + 1: return None
                    def calc_vol(arr, w):
                        r = [(arr[i]/arr[i-1])-1 for i in range(len(arr)-w, len(arr))]
                        return math.sqrt(sum(x*x for x in r)/len(r)) if r else 0
                    sv = calc_vol(bc, sw)
                    lv = calc_vol(bc, lw)
                    if lv <= 0 or sv / lv < rt: return None
                    net = (bc[-1] / bc[-sw]) - 1
                    return "Bull" if net > 0 else "Bear"
                return fn
            run_feature(f"vol_ratio(s={short_w},l={long_w},r={ratio_thresh})",
                       train, valid, spot, btc, make_vol_ratio())

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("CATEGORY 2: Volume patterns")
    print("=" * 80)

    for window in [5, 10, 15]:
        for imb_thresh in [0.3, 0.5, 0.7]:
            def make_vol_imbalance(w=window, it=imb_thresh):
                def fn(bc, bv, tc, tv):
                    if bv is None or len(bv) < w: return None
                    recent_v = bv[-w:]
                    total = sum(recent_v)
                    if total <= 0: return None
                    # Volume-weighted direction
                    up_vol = sum(recent_v[i] for i in range(1, len(recent_v))
                                if bc[-w+i] > bc[-w+i-1])
                    imbalance = (up_vol / total) - 0.5
                    if abs(imbalance) < it / 2: return None
                    return "Bull" if imbalance > 0 else "Bear"
                return fn
            run_feature(f"vol_imbalance(w={window},t={imb_thresh})",
                       train, valid, spot, btc, make_vol_imbalance())

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("CATEGORY 3: Price microstructure (tick direction)")
    print("=" * 80)

    for window in [5, 7, 10, 15]:
        for streak_thresh in [0.6, 0.7, 0.8, 0.9]:
            def make_tick_direction(w=window, st=streak_thresh):
                def fn(bc, bv, tc, tv):
                    if len(bc) < w + 1: return None
                    ups = sum(1 for i in range(-w, 0) if bc[i] > bc[i-1])
                    frac = ups / w
                    if frac >= st: return "Bull"
                    if (1 - frac) >= st: return "Bear"
                    return None
                return fn
            run_feature(f"tick_dir(w={window},st={streak_thresh})",
                       train, valid, spot, btc, make_tick_direction())

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("CATEGORY 4: Cross-asset (BTC leads BNB)")
    print("=" * 80)

    for btc_lb in [5, 10, 15, 20, 25, 30]:
        for btc_thresh in [0.0001, 0.0002, 0.0003, 0.0005, 0.001]:
            def make_btc_lead(lb=btc_lb, th=btc_thresh):
                def fn(bc, bv, tc, tv):
                    if tc is None or len(tc) < lb + 1: return None
                    r = _get_return(tc, lb)
                    if r is None or abs(r) < th: return None
                    return "Bull" if r > 0 else "Bear"
                return fn
            run_feature(f"btc_lead(lb={btc_lb},th={btc_thresh})",
                       train, valid, spot, btc, make_btc_lead())

    # BTC contra (bet AGAINST BTC direction)
    for btc_lb in [5, 10, 15, 20, 25, 30]:
        for btc_thresh in [0.0003, 0.0005, 0.001]:
            def make_btc_contra(lb=btc_lb, th=btc_thresh):
                def fn(bc, bv, tc, tv):
                    if tc is None or len(tc) < lb + 1: return None
                    r = _get_return(tc, lb)
                    if r is None or abs(r) < th: return None
                    return "Bear" if r > 0 else "Bull"  # contra
                return fn
            run_feature(f"btc_contra(lb={btc_lb},th={btc_thresh})",
                       train, valid, spot, btc, make_btc_contra())

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("CATEGORY 5: Return reversal (mean reversion)")
    print("=" * 80)

    for lb in [3, 5, 7, 10, 15, 20]:
        for thresh in [0.0001, 0.0002, 0.0003, 0.0005]:
            def make_reversal(l=lb, t=thresh):
                def fn(bc, bv, tc, tv):
                    r = _get_return(bc, l)
                    if r is None or abs(r) < t: return None
                    return "Bear" if r > 0 else "Bull"  # bet against recent move
                return fn
            run_feature(f"reversal(lb={lb},th={thresh})",
                       train, valid, spot, btc, make_reversal())

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("CATEGORY 6: Range position (where is price in recent range)")
    print("=" * 80)

    for window in [10, 15, 20, 25]:
        for extreme_thresh in [0.1, 0.2, 0.3]:
            # Bet on breakout from range
            def make_range_breakout(w=window, et=extreme_thresh):
                def fn(bc, bv, tc, tv):
                    if len(bc) < w: return None
                    recent = bc[-w:]
                    hi = max(recent)
                    lo = min(recent)
                    rng = hi - lo
                    if rng <= 0: return None
                    pos = (bc[-1] - lo) / rng
                    if pos >= (1 - et): return "Bull"
                    if pos <= et: return "Bear"
                    return None
                return fn
            run_feature(f"range_break(w={window},et={extreme_thresh})",
                       train, valid, spot, btc, make_range_breakout())

            # Bet on mean reversion from range extreme
            def make_range_revert(w=window, et=extreme_thresh):
                def fn(bc, bv, tc, tv):
                    if len(bc) < w: return None
                    recent = bc[-w:]
                    hi = max(recent)
                    lo = min(recent)
                    rng = hi - lo
                    if rng <= 0: return None
                    pos = (bc[-1] - lo) / rng
                    if pos >= (1 - et): return "Bear"  # revert from high
                    if pos <= et: return "Bull"  # revert from low
                    return None
                return fn
            run_feature(f"range_revert(w={window},et={extreme_thresh})",
                       train, valid, spot, btc, make_range_revert())

    print("\nDone.")


if __name__ == "__main__":
    main()
