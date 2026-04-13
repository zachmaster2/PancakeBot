"""Analyze BTC contrarian signal contribution to total PnL."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB, BNB_WEI
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import (
    MomentumGateConfig, compute_signal_from_klines, _trim_to_window,
    _CANDLE_COUNT, _get_return, _BTC_LOOKBACK, _BTC_THRESH,
)
from pancakebot.runtime.settlement import settle_bet_against_closed_round


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

    spot = load_kl("var/cutoff_spot_prices.jsonl")
    btc_kl = load_kl("var/btc_spot_prices.jsonl")

    LOW_LIQ_HOURS = (3, 4, 7, 10, 18, 19)
    BTC_CONTRA_THRESH = 0.0003
    TREASURY_FEE = 0.03

    contra_trades = []

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

        result = compute_signal_from_klines(bnb_raw, btc_raw, cutoff_ms)

        # Only look at rounds where main gate has NO signal
        if result.signal is not None:
            continue

        # Get pool data
        bull_wei = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bull")
        bear_wei = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bear")
        pool_bull = bull_wei / 1e18
        pool_bear = bear_wei / 1e18
        pool_total = pool_bull + pool_bear

        if pool_total <= 0:
            continue

        # Get BTC closes
        if btc_raw is None:
            continue
        trimmed = _trim_to_window(btc_raw, cutoff_ms)
        if len(trimmed) < _CANDLE_COUNT:
            continue
        btc_closes = [k[4] for k in trimmed]

        btc_r = _get_return(btc_closes, _BTC_LOOKBACK)
        if btc_r is None or abs(btc_r) < BTC_CONTRA_THRESH:
            continue

        # Contrarian: bet AGAINST BTC direction
        btc_dir = "Bull" if btc_r > 0 else "Bear"
        contra_dir = "Bear" if btc_dir == "Bull" else "Bull"

        # Check payout on contrarian side
        our_side = pool_bull if contra_dir == "Bull" else pool_bear
        if our_side <= 0:
            continue
        pm = pool_total * 0.97 / our_side

        # Sweep different min payouts
        for min_pm in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            if pm < min_pm:
                continue

            for bet_size in [0.10, 0.15, 0.20, 0.30]:
                out = settle_bet_against_closed_round(
                    bet_bnb=bet_size, bet_side=contra_dir,
                    round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
                )
                profit = out.credit_bnb - bet_size - GAS_COST_BET_BNB

                contra_trades.append(dict(
                    epoch=epoch, min_pm=min_pm, bet_size=bet_size,
                    profit=profit, won=profit > 0, payout=pm,
                    contra_dir=contra_dir,
                ))

    # Analyze
    print("=== BTC Contrarian Signal Analysis ===\n")
    for min_pm in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for bet_size in [0.10, 0.15, 0.20, 0.30]:
            sub = [t for t in contra_trades if t["min_pm"] == min_pm and t["bet_size"] == bet_size]
            if not sub:
                continue
            n = len(sub)
            wr = sum(1 for t in sub if t["won"]) / n * 100
            pnl = sum(t["profit"] for t in sub)
            print(f"  min_payout={min_pm:.1f} bet={bet_size:.2f}: {n:4d} bets, WR={wr:.1f}%, PnL={pnl:+.2f}")
        print()


if __name__ == "__main__":
    main()
