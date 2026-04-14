"""Final combined parameter sweep + pool size analysis."""
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

LOW_LIQ_HOURS = (3, 4, 7, 10, 18, 19)
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


def simulate(rounds, spot, btc, *, max_dilution=0.12, min_pool=0.0,
             btc_contra_bet=0.15, skip_btc_dis_strong=False):
    """Run simulation with given parameters."""
    BASE_FRAC = 0.06
    FLOOR = 0.10
    CAP = 2.0
    BTC_AG = 2.0
    BTC_DIS = 0.7
    PAY_BASE = 0.1
    PAY_SLOPE = 1.0
    MIN_PAY = 1.85
    MIN_BET = 0.001

    bankroll = 50.0
    n_bets = 0
    n_wins = 0
    seg_data = []  # (idx, profit) for segment analysis

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - 4) * 1000
        hour_utc = (lock_at % 86400) // 3600

        if hour_utc in LOW_LIQ_HOURS:
            continue

        bnb_raw = spot.get(epoch)
        btc_raw = btc.get(epoch)
        if not bnb_raw:
            continue

        result = compute_signal_from_klines(bnb_raw, btc_raw, cutoff_ms)

        # Pool
        bull_wei = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bull")
        bear_wei = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bear")
        pool_bull = bull_wei / 1e18
        pool_bear = bear_wei / 1e18
        pool_total = pool_bull + pool_bear

        if pool_total < min_pool:
            continue

        signal = result.signal
        bet_side = None
        bet_size = 0.0

        if signal is not None and pool_total > 0:
            # Optional: skip BTC disagree on strong signals
            if skip_btc_dis_strong and result.btc_disagrees:
                # Check signal strength
                trimmed = _trim_to_window(bnb_raw, cutoff_ms)
                if len(trimmed) >= _CANDLE_COUNT:
                    closes = [k[4] for k in trimmed]
                    from pancakebot.domain.strategy.momentum_gate import _ACCEL_PAIRS
                    max_ret = 0
                    for s, l in _ACCEL_PAIRS:
                        for lb in (s, l):
                            r = _get_return(closes, lb)
                            if r: max_ret = max(max_ret, abs(r))
                    if max_ret >= 0.0007:  # 7+ bps
                        continue

            our_side = pool_bull if signal == "Bull" else pool_bear
            if our_side > 0:
                pm = pool_total * 0.97 / our_side
                if pm < MIN_PAY:
                    continue

                bet_size = max(FLOOR, pool_total * BASE_FRAC)
                mult = max(0.3, PAY_BASE + PAY_SLOPE * (pm - 1.0))
                bet_size *= mult
                if result.btc_agrees:
                    bet_size *= BTC_AG
                elif result.btc_disagrees:
                    bet_size *= BTC_DIS
                bet_size = min(CAP, bet_size)

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
                            if pm >= 3.0:
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
            seg_data.append(profit)

    net = bankroll - 50.0
    wr = n_wins / max(1, n_bets) * 100

    # Per-segment analysis
    n_seg = max(1, len(seg_data) // 350)  # ~350 bets per segment
    seg_size = len(seg_data) // n_seg if n_seg > 0 else len(seg_data)
    any_neg = False
    worst_seg = 999.0
    for s in range(n_seg):
        start = s * seg_size
        end = (s + 1) * seg_size if s < n_seg - 1 else len(seg_data)
        chunk_pnl = sum(seg_data[start:end])
        if chunk_pnl < worst_seg:
            worst_seg = chunk_pnl
        if chunk_pnl < 0:
            any_neg = True

    return net, n_bets, wr, worst_seg, any_neg


def main():
    rounds, spot, btc = load_data()

    # 1. Pool size filter sweep
    print("=== Pool Size Min Filter ===")
    for mp in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        n, nb, w, ws, neg = simulate(rounds, spot, btc, min_pool=mp)
        print(f"  min_pool={mp:5.1f}: NET={n:+.2f} Bets={nb:4d} WR={w:.1f}% worst_seg={ws:+.2f} any_neg={neg}")

    # 2. Dilution cap sweep (fine-grained around optimal)
    print("\n=== Dilution Cap (fine-grained) ===")
    for dc in [0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]:
        n, nb, w, ws, neg = simulate(rounds, spot, btc, max_dilution=dc)
        print(f"  dil_cap={dc:.2f}: NET={n:+.2f} Bets={nb:4d} WR={w:.1f}% worst_seg={ws:+.2f}")

    # 3. Combined: dil_cap=0.15 + pool filter
    print("\n=== Combined: dil_cap + pool filter ===")
    for dc in [0.12, 0.15]:
        for mp in [0.0, 1.0, 2.0]:
            n, nb, w, ws, neg = simulate(rounds, spot, btc, max_dilution=dc, min_pool=mp)
            print(f"  dil={dc:.2f} pool={mp:.1f}: NET={n:+.2f} Bets={nb:4d} WR={w:.1f}% worst_seg={ws:+.2f}")

    # 4. Skip BTC disagree on strong signals (7+ bps)
    print("\n=== Skip BTC disagree on strong signals ===")
    n, nb, w, ws, neg = simulate(rounds, spot, btc, skip_btc_dis_strong=False)
    print(f"  baseline: NET={n:+.2f} Bets={nb:4d} WR={w:.1f}%")
    n, nb, w, ws, neg = simulate(rounds, spot, btc, skip_btc_dis_strong=True)
    print(f"  skip_dis_strong: NET={n:+.2f} Bets={nb:4d} WR={w:.1f}%")


if __name__ == "__main__":
    main()
