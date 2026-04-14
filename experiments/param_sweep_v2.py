"""Sweep remaining parameters to find further PnL improvements.

Tests: BASE_FRAC, FLOOR_BNB, CAP_BNB, MIN_OUR_PAYOUT, dilution cap,
       payout slope/base combos, BTC disagree mult.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import GAS_COST_BET_BNB, BNB_WEI
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import (
    MomentumGateConfig, compute_signal_from_klines, _trim_to_window,
    _CANDLE_COUNT, _get_return, _BTC_LOOKBACK, _BTC_THRESH,
    _ACCEL_PAIRS, _ACCEL_THRESH,
)
from pancakebot.runtime.settlement import settle_bet_against_closed_round

LOW_LIQ_HOURS = (3, 4, 7, 10, 18, 19)
TREASURY_FEE = 0.03
CUTOFF_S = 4
MIN_BET = 0.001

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


def simulate(rounds, spot, btc, *, base_frac=0.06, floor_bnb=0.10, cap_bnb=2.0,
             btc_agree_mult=2.0, btc_disagree_mult=0.7,
             payout_base=0.1, payout_slope=1.0, min_payout=1.85,
             max_dilution=0.12, btc_contra_min_pm=3.0, btc_contra_bet=0.15):
    """Run full simulation with given parameters. Returns (net_pnl, n_bets, wr)."""
    bankroll = 50.0
    n_bets = 0
    n_wins = 0

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000
        hour_utc = (lock_at % 86400) // 3600

        if hour_utc in LOW_LIQ_HOURS:
            continue

        bnb_raw = spot.get(epoch)
        btc_raw = btc.get(epoch)
        if not bnb_raw:
            continue

        result = compute_signal_from_klines(bnb_raw, btc_raw, cutoff_ms)

        # Pool computation
        bull_wei = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bull")
        bear_wei = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bear")
        pool_bull = bull_wei / 1e18
        pool_bear = bear_wei / 1e18
        pool_total = pool_bull + pool_bear

        signal = result.signal
        bet_side = None
        bet_size = 0.0

        if signal is not None and pool_total > 0:
            # Payout floor
            our_side = pool_bull if signal == "Bull" else pool_bear
            if our_side > 0:
                pm = pool_total * 0.97 / our_side
                if pm < min_payout:
                    continue

                # Sizing
                bet_size = max(floor_bnb, pool_total * base_frac)
                mult = max(0.3, payout_base + payout_slope * (pm - 1.0))
                bet_size *= mult
                if result.btc_agrees:
                    bet_size *= btc_agree_mult
                elif result.btc_disagrees:
                    bet_size *= btc_disagree_mult
                bet_size = min(cap_bnb, bet_size)

                # Dilution cap
                d = max_dilution
                denom = (1.0 - d) * pool_total - our_side
                if denom > 0:
                    max_bet = d * pool_total * our_side / denom
                    bet_size = min(bet_size, max_bet)

                if bet_size >= MIN_BET:
                    bet_side = signal
        elif signal is None and pool_total > 0:
            # BTC contrarian
            if btc_raw:
                trimmed = _trim_to_window(btc_raw, cutoff_ms)
                if len(trimmed) >= _CANDLE_COUNT:
                    btc_closes = [k[4] for k in trimmed]
                    btc_r = _get_return(btc_closes, _BTC_LOOKBACK)
                    if btc_r is not None and abs(btc_r) >= 0.0003:
                        btc_dir = "Bull" if btc_r > 0 else "Bear"
                        contra_dir = "Bear" if btc_dir == "Bull" else "Bull"
                        our_side = pool_bull if contra_dir == "Bull" else pool_bear
                        if our_side > 0:
                            pm = pool_total * 0.97 / our_side
                            if pm >= btc_contra_min_pm:
                                bet_side = contra_dir
                                bet_size = btc_contra_bet

        if bet_side and bet_size >= MIN_BET:
            bankroll -= bet_size + GAS_COST_BET_BNB
            out = settle_bet_against_closed_round(
                bet_bnb=bet_size, bet_side=bet_side,
                round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
            )
            bankroll += out.credit_bnb
            profit = out.credit_bnb - bet_size - GAS_COST_BET_BNB
            n_bets += 1
            if profit > 0:
                n_wins += 1

    net = bankroll - 50.0
    wr = n_wins / max(1, n_bets) * 100
    return net, n_bets, wr


def main():
    rounds, spot, btc = load_data()

    # Baseline
    net, nb, wr = simulate(rounds, spot, btc)
    print(f"BASELINE: NET={net:+.2f} Bets={nb} WR={wr:.1f}%\n")

    # 1. BASE_FRAC sweep
    print("=== BASE_FRAC sweep ===")
    for bf in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        n, nb, w = simulate(rounds, spot, btc, base_frac=bf)
        print(f"  base_frac={bf:.2f}: NET={n:+.2f} Bets={nb} WR={w:.1f}%")

    # 2. FLOOR_BNB sweep
    print("\n=== FLOOR_BNB sweep ===")
    for fl in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        n, nb, w = simulate(rounds, spot, btc, floor_bnb=fl)
        print(f"  floor={fl:.2f}: NET={n:+.2f} Bets={nb} WR={w:.1f}%")

    # 3. CAP_BNB sweep
    print("\n=== CAP_BNB sweep ===")
    for c in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        n, nb, w = simulate(rounds, spot, btc, cap_bnb=c)
        print(f"  cap={c:.1f}: NET={n:+.2f} Bets={nb} WR={w:.1f}%")

    # 4. BTC disagree mult sweep
    print("\n=== BTC_DISAGREE_MULT sweep ===")
    for dm in [0.0, 0.3, 0.5, 0.7, 1.0]:
        n, nb, w = simulate(rounds, spot, btc, btc_disagree_mult=dm)
        print(f"  dis_mult={dm:.1f}: NET={n:+.2f} Bets={nb} WR={w:.1f}%")

    # 5. MIN_OUR_PAYOUT sweep
    print("\n=== MIN_OUR_PAYOUT sweep ===")
    for mp in [1.5, 1.7, 1.85, 2.0, 2.2, 2.5]:
        n, nb, w = simulate(rounds, spot, btc, min_payout=mp)
        print(f"  min_pay={mp:.2f}: NET={n:+.2f} Bets={nb} WR={w:.1f}%")

    # 6. DILUTION_CAP sweep
    print("\n=== DILUTION_CAP sweep ===")
    for dc in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]:
        n, nb, w = simulate(rounds, spot, btc, max_dilution=dc)
        print(f"  dil_cap={dc:.2f}: NET={n:+.2f} Bets={nb} WR={w:.1f}%")

    # 7. PAYOUT_LINEAR_SLOPE sweep
    print("\n=== PAYOUT_LINEAR_SLOPE sweep ===")
    for s in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]:
        n, nb, w = simulate(rounds, spot, btc, payout_slope=s)
        print(f"  slope={s:.1f}: NET={n:+.2f} Bets={nb} WR={w:.1f}%")

    # 8. BTC contrarian params sweep
    print("\n=== BTC_CONTRA sweep ===")
    for cpm in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for cbet in [0.10, 0.15, 0.20]:
            n, nb, w = simulate(rounds, spot, btc, btc_contra_min_pm=cpm, btc_contra_bet=cbet)
            print(f"  contra_pm={cpm:.1f} bet={cbet:.2f}: NET={n:+.2f} Bets={nb} WR={w:.1f}%")

    # 9. Combined best params
    print("\n=== COMBINED best params ===")
    # Will run after seeing individual results


if __name__ == "__main__":
    main()
