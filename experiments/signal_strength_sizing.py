"""Explore signal-strength-based sizing: bet more when accel signal is stronger.

Hypothesis: larger |returns| in the accel pair → stronger signal → bet more.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB, BNB_WEI
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import (
    MomentumGateConfig, _trim_to_window, _CANDLE_COUNT,
    _get_return, _ACCEL_PAIRS, _ACCEL_THRESH, _BTC_LOOKBACK, _BTC_THRESH,
)
from pancakebot.runtime.settlement import settle_bet_against_closed_round

LOW_LIQ_HOURS = (3, 4, 7, 10, 18, 19)
TREASURY_FEE = 0.03


def main():
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

    spot = load_kl("var/bnb_spot_prices.jsonl")
    btc_kl = load_kl("var/btc_spot_prices.jsonl")

    # Collect per-trade signal strength data
    trades = []

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - 4) * 1000
        hour_utc = (lock_at % 86400) // 3600

        if hour_utc in LOW_LIQ_HOURS:
            continue

        bnb_raw = spot.get(epoch)
        btc_raw = btc_kl.get(epoch)
        if not bnb_raw:
            continue

        trimmed = _trim_to_window(bnb_raw, cutoff_ms)
        if len(trimmed) < _CANDLE_COUNT:
            continue
        closes = [k[4] for k in trimmed]

        # Check accel signal
        fired = False
        signal_strength = 0.0
        signal_dir = None
        for short, long in _ACCEL_PAIRS:
            rs = _get_return(closes, short)
            rl = _get_return(closes, long)
            if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
                if max(abs(rs), abs(rl)) >= _ACCEL_THRESH:
                    fired = True
                    signal_strength = max(abs(rs), abs(rl))
                    signal_dir = "Bull" if rs > 0 else "Bear"
                    break

        if not fired:
            continue

        # BTC check
        btc_agrees = False
        btc_disagrees = False
        if btc_raw:
            btrim = _trim_to_window(btc_raw, cutoff_ms)
            if len(btrim) >= _CANDLE_COUNT:
                btc_closes = [k[4] for k in btrim]
                btc_r = _get_return(btc_closes, _BTC_LOOKBACK)
                if btc_r is not None and abs(btc_r) >= _BTC_THRESH:
                    btc_dir = "Bull" if btc_r > 0 else "Bear"
                    btc_agrees = btc_dir == signal_dir
                    btc_disagrees = btc_dir != signal_dir

        # Pool data
        bull_wei = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bull")
        bear_wei = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bear")
        pool_bull = bull_wei / 1e18
        pool_bear = bear_wei / 1e18
        pool_total = pool_bull + pool_bear
        if pool_total <= 0:
            continue

        our_side = pool_bull if signal_dir == "Bull" else pool_bear
        if our_side <= 0:
            continue
        pm = pool_total * 0.97 / our_side
        if pm < 1.85:
            continue

        # Settle with fixed bet to measure pure signal quality
        out = settle_bet_against_closed_round(
            bet_bnb=0.10, bet_side=signal_dir,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - 0.10 - GAS_COST_BET_BNB

        trades.append(dict(
            epoch=epoch, strength=signal_strength * 10000,  # in bps
            profit=profit, won=profit > 0,
            btc_agrees=btc_agrees, btc_disagrees=btc_disagrees,
            payout=pm, side=signal_dir,
        ))

    # Analyze by signal strength buckets
    print(f"Total accel trades (after filters): {len(trades)}\n")

    print("=== Signal Strength Buckets (bps) ===")
    for lo, hi, lbl in [(2, 3, "2-3"), (3, 4, "3-4"), (4, 5, "4-5"),
                         (5, 7, "5-7"), (7, 10, "7-10"), (10, 999, "10+")]:
        sub = [t for t in trades if lo <= t["strength"] < hi]
        if not sub:
            continue
        n = len(sub)
        wr = sum(1 for t in sub if t["won"]) / n * 100
        pnl = sum(t["profit"] for t in sub)
        print(f"  {lbl:>5s} bps: {n:4d} bets, WR={wr:.1f}%, PnL={pnl:+.2f} (per-bet={pnl/n:+.4f})")

    # Interaction: strength x btc
    print("\n=== Strength x BTC ===")
    for lo, hi, lbl in [(2, 4, "2-4"), (4, 7, "4-7"), (7, 999, "7+")]:
        for blbl, bfilt in [("agree", lambda t: t["btc_agrees"]),
                             ("disagree", lambda t: t["btc_disagrees"]),
                             ("neutral", lambda t: not t["btc_agrees"] and not t["btc_disagrees"])]:
            sub = [t for t in trades if lo <= t["strength"] < hi and bfilt(t)]
            if not sub or len(sub) < 10:
                continue
            n = len(sub)
            wr = sum(1 for t in sub if t["won"]) / n * 100
            pnl = sum(t["profit"] for t in sub)
            print(f"  {lbl:>5s}bps + {blbl:>8s}: {n:4d} bets, WR={wr:.1f}%, PnL={pnl:+.2f}")

    # Check if there's a meaningful WR difference by strength
    weak = [t for t in trades if t["strength"] < 3]
    strong = [t for t in trades if t["strength"] >= 5]
    if weak and strong:
        wr_w = sum(1 for t in weak if t["won"]) / len(weak) * 100
        wr_s = sum(1 for t in strong if t["won"]) / len(strong) * 100
        print(f"\nWeak (<3bps): {len(weak)} bets, WR={wr_w:.1f}%")
        print(f"Strong (>=5bps): {len(strong)} bets, WR={wr_s:.1f}%")
        print(f"WR delta: {wr_s - wr_w:+.1f}pp")


if __name__ == "__main__":
    main()
