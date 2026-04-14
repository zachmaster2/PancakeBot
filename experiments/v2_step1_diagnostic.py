"""Step 1: Diagnostic — does momentum have ANY edge at the correct cutoff?

Sweeps lookback pairs, thresholds, and BTC confirmation on the corrected
34k dataset to determine if the accel signal is fundamentally dead or
just needs re-tuning.

Uses walk-forward: train on first 70% of rounds, validate on last 30%.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB, BNB_WEI, INTERVAL_SECONDS
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import (
    _trim_to_window, _get_return,
)

CUTOFF_S = 4
CANDLE_COUNT = 31
TREASURY_FEE = 0.03
BET_SIZE = 0.10  # fixed bet for pure signal evaluation


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

    return rounds, load_kl("var/bnb_spot_prices.jsonl"), load_kl("var/btc_spot_prices.jsonl")


def evaluate_signal(rounds, spot, btc, *, lookback_pairs, threshold, btc_lookback,
                    btc_thresh, require_btc_agree):
    """Evaluate a signal configuration. Returns (n_bets, n_wins, wr, pnl)."""
    from pancakebot.runtime.settlement import settle_bet_against_closed_round

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

        trimmed = _trim_to_window(bnb_raw, cutoff_ms)
        if len(trimmed) < CANDLE_COUNT:
            continue
        closes = [k[4] for k in trimmed]

        # Check signal
        signal = None
        for short, long in lookback_pairs:
            rs = _get_return(closes, short)
            rl = _get_return(closes, long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= threshold:
                    signal = "Bull" if rs > 0 else "Bear"
                    break

        if signal is None:
            continue

        # BTC check
        if require_btc_agree and btc_raw:
            btrim = _trim_to_window(btc_raw, cutoff_ms)
            if len(btrim) >= CANDLE_COUNT:
                btc_closes = [k[4] for k in btrim]
                btc_r = _get_return(btc_closes, btc_lookback)
                if btc_r is not None and abs(btc_r) >= btc_thresh:
                    btc_dir = "Bull" if btc_r > 0 else "Bear"
                    if btc_dir != signal:
                        continue  # skip if BTC disagrees
                # if BTC neutral, allow the bet

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


def main():
    rounds, spot, btc = load_data()

    # Walk-forward split: 70% train, 30% validate
    split = int(len(rounds) * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    print(f"Rounds: {len(rounds)} total, {len(train)} train, {len(valid)} validate")
    print()

    # --- Part A: Sweep lookback pairs and thresholds ---
    print("=" * 70)
    print("PART A: Lookback pairs + threshold sweep (no BTC filter)")
    print("=" * 70)

    all_pairs = [
        [(s, l)] for s in range(3, 15) for l in range(s + 2, min(s + 10, 26)) if l <= 25
    ]
    thresholds = [0.0001, 0.00015, 0.0002, 0.0003, 0.0005]

    best_train = []
    for pairs in all_pairs:
        for thresh in thresholds:
            nb, nw, wr, pnl = evaluate_signal(
                train, spot, btc,
                lookback_pairs=pairs, threshold=thresh,
                btc_lookback=30, btc_thresh=0.0003, require_btc_agree=False,
            )
            if nb >= 100:  # minimum sample size
                best_train.append((wr, pnl, nb, pairs, thresh))

    best_train.sort(reverse=True)
    print(f"\nTop 20 signals on TRAIN ({len(train)} rounds):")
    print(f"{'WR':>6s} {'PnL':>8s} {'Bets':>6s}  Pairs              Thresh")
    print("-" * 60)
    for wr, pnl, nb, pairs, thresh in best_train[:20]:
        print(f"{wr:5.1f}% {pnl:+7.2f} {nb:6d}  {str(pairs):20s} {thresh}")

    # Validate top 10 on held-out data
    print(f"\nValidation of top 10 on VALIDATE ({len(valid)} rounds):")
    print(f"{'T_WR':>6s} {'V_WR':>6s} {'V_PnL':>8s} {'V_Bets':>6s}  Pairs              Thresh")
    print("-" * 65)
    for wr_t, _, _, pairs, thresh in best_train[:10]:
        nb, nw, wr_v, pnl = evaluate_signal(
            valid, spot, btc,
            lookback_pairs=pairs, threshold=thresh,
            btc_lookback=30, btc_thresh=0.0003, require_btc_agree=False,
        )
        flag = " ***" if wr_v > 55 else ""
        print(f"{wr_t:5.1f}% {wr_v:5.1f}% {pnl:+7.2f} {nb:6d}  {str(pairs):20s} {thresh}{flag}")

    # --- Part B: BTC confirmation filter ---
    print(f"\n{'=' * 70}")
    print("PART B: Top signals + BTC agree filter")
    print("=" * 70)

    for wr_t, _, _, pairs, thresh in best_train[:5]:
        for btc_lb in [10, 15, 20, 25, 30]:
            for btc_th in [0.0001, 0.0003, 0.0005]:
                nb_t, _, wr_tr, _ = evaluate_signal(
                    train, spot, btc,
                    lookback_pairs=pairs, threshold=thresh,
                    btc_lookback=btc_lb, btc_thresh=btc_th, require_btc_agree=True,
                )
                if nb_t < 50:
                    continue
                nb_v, _, wr_va, pnl_v = evaluate_signal(
                    valid, spot, btc,
                    lookback_pairs=pairs, threshold=thresh,
                    btc_lookback=btc_lb, btc_thresh=btc_th, require_btc_agree=True,
                )
                if wr_va > 56:
                    print(f"  pairs={pairs} th={thresh} btc_lb={btc_lb} btc_th={btc_th}: "
                          f"T_WR={wr_tr:.1f}% V_WR={wr_va:.1f}% V_bets={nb_v} V_pnl={pnl_v:+.2f}")

    # --- Part C: Multi-pair combinations ---
    print(f"\n{'=' * 70}")
    print("PART C: Multi-pair combinations (top 3 pairs)")
    print("=" * 70)

    # Get top individual pairs
    top_individual = []
    for wr, pnl, nb, pairs, thresh in best_train[:30]:
        p = pairs[0]
        if p not in [x[0] for x in top_individual]:
            top_individual.append((p, thresh, wr))
        if len(top_individual) >= 6:
            break

    print(f"Top individual pairs: {[x[0] for x in top_individual]}")

    # Try combinations of 2-3 pairs
    from itertools import combinations
    for n_combo in [2, 3]:
        for combo in combinations(top_individual, n_combo):
            pairs = [c[0] for c in combo]
            thresh = min(c[1] for c in combo)  # use most lenient threshold
            nb_t, _, wr_t, _ = evaluate_signal(
                train, spot, btc,
                lookback_pairs=pairs, threshold=thresh,
                btc_lookback=30, btc_thresh=0.0003, require_btc_agree=False,
            )
            if nb_t < 100:
                continue
            nb_v, _, wr_v, pnl_v = evaluate_signal(
                valid, spot, btc,
                lookback_pairs=pairs, threshold=thresh,
                btc_lookback=30, btc_thresh=0.0003, require_btc_agree=False,
            )
            if wr_v > 55:
                print(f"  pairs={pairs} th={thresh}: T_WR={wr_t:.1f}% V_WR={wr_v:.1f}% V_bets={nb_v} V_pnl={pnl_v:+.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
