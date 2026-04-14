"""Make the BTC lead signal profitable.

The signal has 57-62% validation WR but negative PnL because wins
happen on low-payout rounds. This script explores:

1. Payout-only filtering (no hour filtering - too few samples)
2. Bet sizing that scales with confidence/payout
3. Multiple BTC lookback windows voting together
4. Payout-aware entry: only bet when expected value > 0
5. Final walk-forward validation of the best config
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB, INTERVAL_SECONDS, BNB_WEI
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import _trim_to_window, _get_return
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 4
CANDLE_COUNT = 31
TREASURY_FEE = 0.03


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
    return rounds, load_kl("var/bnb_spot_prices.jsonl"), load_kl("var/btc_spot_prices.jsonl")


def get_closes(raw, cutoff_ms):
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None
    return [k[4] for k in trimmed]


def simulate(rounds, spot, btc_kl, *, config):
    """Flexible simulation driven by a config dict."""
    btc_lb = config["btc_lb"]
    btc_thresh = config["btc_thresh"]
    min_payout = config.get("min_payout", 0.0)
    max_payout = config.get("max_payout", 999.0)
    base_bet = config.get("base_bet", 0.10)
    payout_sizing = config.get("payout_sizing", False)
    ev_filter = config.get("ev_filter", False)
    ev_wr_estimate = config.get("ev_wr_estimate", 0.57)
    multi_lb = config.get("multi_lb", None)  # list of (lb, thresh) pairs for voting
    min_votes = config.get("min_votes", 1)

    trades = []

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        btc_raw = btc_kl.get(epoch)
        if not btc_raw:
            continue
        btc_closes = get_closes(btc_raw, cutoff_ms)
        if btc_closes is None:
            continue

        # Single or multi lookback
        if multi_lb:
            bull_votes = 0
            bear_votes = 0
            for lb, th in multi_lb:
                r = _get_return(btc_closes, lb)
                if r is not None and abs(r) >= th:
                    if r > 0:
                        bull_votes += 1
                    else:
                        bear_votes += 1
            total_votes = bull_votes + bear_votes
            if total_votes < min_votes:
                continue
            if bull_votes > bear_votes:
                signal = "Bull"
            elif bear_votes > bull_votes:
                signal = "Bear"
            else:
                continue  # tie
        else:
            btc_r = _get_return(btc_closes, btc_lb)
            if btc_r is None or abs(btc_r) < btc_thresh:
                continue
            signal = "Bull" if btc_r > 0 else "Bear"

        # Pool / payout
        bull_wei = sum(int(b.amount_wei) for b in rnd.bets
                      if int(b.created_at) <= lock_at - CUTOFF_S and b.position == "Bull")
        bear_wei = sum(int(b.amount_wei) for b in rnd.bets
                      if int(b.created_at) <= lock_at - CUTOFF_S and b.position == "Bear")
        pool_bull = bull_wei / 1e18
        pool_bear = bear_wei / 1e18
        pool_total = pool_bull + pool_bear
        if pool_total <= 0:
            continue

        our_side = pool_bull if signal == "Bull" else pool_bear
        if our_side <= 0:
            continue
        pm = pool_total * 0.97 / our_side

        if pm < min_payout or pm > max_payout:
            continue

        # EV filter: only bet if expected value > 0
        # EV = wr * (pm - 1) * bet - (1 - wr) * bet - gas
        if ev_filter:
            ev = ev_wr_estimate * (pm - 1) * base_bet - (1 - ev_wr_estimate) * base_bet - GAS_COST_BET_BNB
            if ev <= 0:
                continue

        # Sizing
        if payout_sizing:
            bet = base_bet * max(0.3, 0.1 + 1.0 * (pm - 1.0))
            bet = min(2.0, bet)
        else:
            bet = base_bet

        out = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - bet - GAS_COST_BET_BNB
        trades.append(profit)

    n = len(trades)
    wins = sum(1 for p in trades if p > 0)
    wr = wins / max(1, n) * 100
    pnl = sum(trades)
    return n, wins, wr, pnl


def show(label, train, valid, spot, btc, config):
    nt, _, wt, pt = simulate(train, spot, btc, config=config)
    if nt < 30:
        return
    nv, _, wv, pv = simulate(valid, spot, btc, config=config)
    flag = " ***" if pv > 0 else ""
    per_bet = pv / max(1, nv)
    print(f"  {label:55s} T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) PnL={pv:+6.2f} per_bet={per_bet:+.4f}{flag}")


def main():
    rounds, spot, btc = load_data()
    split = int(len(rounds) * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    print(f"Rounds: {len(rounds)} total, {len(train)} train, {len(valid)} validate\n")

    # Baseline configs to test
    base_configs = [
        {"btc_lb": 7, "btc_thresh": 0.0007},
        {"btc_lb": 10, "btc_thresh": 0.0005},
        {"btc_lb": 10, "btc_thresh": 0.0007},
        {"btc_lb": 15, "btc_thresh": 0.0010},
    ]

    # =====================================================================
    print("=" * 80)
    print("PART 1: Payout filter sweep (robust configs only)")
    print("=" * 80)

    for bc in base_configs:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        print(f"\n  btc_lead(lb={lb}, th={th}):")
        for min_pm in [0.0, 1.5, 1.7, 1.85, 2.0, 2.3, 2.5]:
            cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10}
            show(f"pm>={min_pm:.2f} fixed", train, valid, spot, btc, cfg)
        for min_pm in [0.0, 1.5, 1.85, 2.0, 2.5]:
            cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10, "payout_sizing": True}
            show(f"pm>={min_pm:.2f} payout_sizing", train, valid, spot, btc, cfg)

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 2: EV filter (only bet when expected value > 0)")
    print("=" * 80)

    for bc in base_configs:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        print(f"\n  btc_lead(lb={lb}, th={th}):")
        for wr_est in [0.55, 0.57, 0.60, 0.62]:
            cfg = {**bc, "ev_filter": True, "ev_wr_estimate": wr_est, "base_bet": 0.10}
            show(f"ev_filter(wr={wr_est:.2f}) fixed", train, valid, spot, btc, cfg)
            cfg2 = {**cfg, "payout_sizing": True}
            show(f"ev_filter(wr={wr_est:.2f}) payout", train, valid, spot, btc, cfg2)

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 3: Multi-lookback voting")
    print("=" * 80)

    vote_configs = [
        [(7, 0.0007), (10, 0.0005)],
        [(7, 0.0007), (10, 0.0007)],
        [(7, 0.0007), (15, 0.0010)],
        [(10, 0.0005), (15, 0.0010)],
        [(7, 0.0007), (10, 0.0005), (15, 0.0010)],
        [(7, 0.0007), (10, 0.0007), (15, 0.0010)],
    ]

    for vc in vote_configs:
        label = "+".join(f"({lb},{th})" for lb, th in vc)
        for min_v in [1, 2]:
            if min_v > len(vc):
                continue
            for min_pm in [0.0, 1.85, 2.0]:
                cfg = {"btc_lb": 0, "btc_thresh": 0, "multi_lb": vc,
                       "min_votes": min_v, "min_payout": min_pm, "base_bet": 0.10}
                show(f"vote[{label}] min={min_v} pm>={min_pm}", train, valid, spot, btc, cfg)

    # =====================================================================
    print(f"\n{'=' * 80}")
    print("PART 4: Payout band analysis (is there a sweet spot?)")
    print("=" * 80)

    # For the best config, analyze WR by payout band
    bc = {"btc_lb": 7, "btc_thresh": 0.0007}
    print(f"\n  btc_lead(lb=7, th=0.0007) WR by payout band on ALL data:")

    payout_bins = {}
    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        btc_raw = btc.get(epoch)
        if not btc_raw: continue
        btc_closes = get_closes(btc_raw, cutoff_ms)
        if btc_closes is None: continue
        btc_r = _get_return(btc_closes, 7)
        if btc_r is None or abs(btc_r) < 0.0007: continue

        signal = "Bull" if btc_r > 0 else "Bear"

        bull_wei = sum(int(b.amount_wei) for b in rnd.bets
                      if int(b.created_at) <= lock_at - CUTOFF_S and b.position == "Bull")
        bear_wei = sum(int(b.amount_wei) for b in rnd.bets
                      if int(b.created_at) <= lock_at - CUTOFF_S and b.position == "Bear")
        pool_total = (bull_wei + bear_wei) / 1e18
        if pool_total <= 0: continue
        our_side = (bull_wei if signal == "Bull" else bear_wei) / 1e18
        if our_side <= 0: continue
        pm = pool_total * 0.97 / our_side

        out = settle_bet_against_closed_round(
            bet_bnb=0.10, bet_side=signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        won = out.credit_bnb > 0.10

        # Bin by payout
        if pm < 1.5:
            b = "<1.5"
        elif pm < 1.7:
            b = "1.5-1.7"
        elif pm < 1.85:
            b = "1.7-1.85"
        elif pm < 2.0:
            b = "1.85-2.0"
        elif pm < 2.5:
            b = "2.0-2.5"
        elif pm < 3.0:
            b = "2.5-3.0"
        else:
            b = "3.0+"

        payout_bins.setdefault(b, [0, 0])
        payout_bins[b][0] += 1
        payout_bins[b][1] += 1 if won else 0

    print(f"  {'Band':>10s} {'Bets':>5s} {'WR':>6s} {'EV@0.10':>8s}")
    for band in ["<1.5", "1.5-1.7", "1.7-1.85", "1.85-2.0", "2.0-2.5", "2.5-3.0", "3.0+"]:
        if band in payout_bins:
            n, w = payout_bins[band]
            wr = w / n * 100
            # Approximate EV
            avg_pm = {"<1.5": 1.3, "1.5-1.7": 1.6, "1.7-1.85": 1.77,
                      "1.85-2.0": 1.92, "2.0-2.5": 2.2, "2.5-3.0": 2.7, "3.0+": 4.0}[band]
            ev = (wr/100) * (avg_pm - 1) * 0.10 - (1 - wr/100) * 0.10 - GAS_COST_BET_BNB
            print(f"  {band:>10s} {n:5d} {wr:5.1f}% {ev:+7.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
