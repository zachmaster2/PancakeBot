"""Diagnostic: why is Seg1 (Dec 11 - Jan 4) losing?"""
import json, sys, math, datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import (
    MomentumGateConfig, compute_signal_from_klines, _trim_to_window,
    _CANDLE_COUNT, _get_return, _ACCEL_PAIRS,
)
from pancakebot.domain.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.runtime.settlement import settle_bet_against_closed_round


def main():
    rounds = list(ClosedRoundsStore("var/closed_rounds.jsonl").iter_closed_rounds())

    def load_kl(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    spot = load_kl("var/bnb_spot_prices.jsonl")
    btc_kl = load_kl("var/btc_spot_prices.jsonl")

    gate_cfg = MomentumGateConfig(enabled=True, symbol="BNB-USDT", btc_symbol="BTC-USDT")
    pipe = MomentumOnlyPipeline(
        config=gate_cfg, gate=None, cutoff_seconds=4,
        min_bet_amount_bnb=0.001, treasury_fee_fraction=0.03,
    )
    pipe.refresh_bnb_klines(bnb_klines_by_epoch=spot)
    pipe.refresh_btc_klines(btc_klines_by_epoch=btc_kl)

    bankroll = 50.0
    trades = []

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - 4) * 1000
        dec = pipe.decide_open_round(round_t=rnd, bankroll_bnb=bankroll, allow_oracle_mode=True)

        bnb_raw = spot.get(epoch)
        btc_raw = btc_kl.get(epoch)
        tier = None
        btc_ag = False
        btc_dis = False
        max_ret = 0.0
        bnb_vol = 0.0

        if bnb_raw:
            trimmed = _trim_to_window(bnb_raw, cutoff_ms)
            if len(trimmed) >= _CANDLE_COUNT:
                closes = [k[4] for k in trimmed]
                lr = []
                for i in range(1, len(closes)):
                    if closes[i - 1] > 0 and closes[i] > 0:
                        lr.append(math.log(closes[i] / closes[i - 1]))
                if lr:
                    m = sum(lr) / len(lr)
                    bnb_vol = math.sqrt(sum((r - m) ** 2 for r in lr) / len(lr)) * 10000

                result = compute_signal_from_klines(bnb_raw, btc_raw, cutoff_ms)
                tier = result.tier
                btc_ag = result.btc_agrees
                btc_dis = result.btc_disagrees

                for s, l in _ACCEL_PAIRS:
                    for lb in (s, l):
                        r = _get_return(closes, lb)
                        if r is not None:
                            max_ret = max(max_ret, abs(r))

        if dec.action == "BET" and dec.bet_size_bnb > 0.0:
            bankroll -= dec.bet_size_bnb + GAS_COST_BET_BNB
            out = settle_bet_against_closed_round(
                bet_bnb=dec.bet_size_bnb, bet_side=dec.bet_side,
                round_closed=rnd, treasury_fee_fraction=0.03,
            )
            bankroll += out.credit_bnb
            profit = out.credit_bnb - dec.bet_size_bnb - GAS_COST_BET_BNB

            bw = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bull")
            brw = sum(int(b.amount_wei) for b in rnd.bets if int(b.created_at) <= lock_at and b.position == "Bear")
            pt = (bw + brw) / BNB_WEI
            pb = bw / BNB_WEI
            crowd = "Bull" if pb > pt - pb else "Bear"
            our_side_bnb = pb if dec.bet_side == "Bull" else pt - pb
            payout = pt * 0.97 / our_side_bnb if our_side_bnb > 0 else 0

            trades.append(dict(
                epoch=epoch, lock_at=lock_at, hour=(lock_at % 86400) // 3600,
                profit=profit, won=profit > 0, tier=tier, btc_ag=btc_ag,
                btc_dis=btc_dis, max_ret=max_ret, vol=bnb_vol,
                pool_total=pt, with_crowd=dec.bet_side == crowd,
                side=dec.bet_side, size=dec.bet_size_bnb, payout=payout,
                skip=dec.skip_reason,
            ))

        pipe.settle_closed_rounds(rounds=[rnd])

    # Split at ~5000 rounds boundary (Seg1 boundary)
    seg1_cutoff = int(rounds[5713].epoch)  # approx first 5k rounds
    seg1 = [t for t in trades if t["epoch"] < seg1_cutoff]
    rest = [t for t in trades if t["epoch"] >= seg1_cutoff]

    def analyze(label, grp):
        n = len(grp)
        if not n:
            return
        wr = sum(1 for t in grp if t["won"]) / n * 100
        pnl = sum(t["profit"] for t in grp)
        print(f"\n=== {label}: {n} bets, WR={wr:.1f}%, PnL={pnl:+.2f} ===")

        # Tier breakdown
        for tname in ["accel", "any+btc", None]:
            sub = [t for t in grp if t["tier"] == tname]
            if not sub:
                continue
            sw = sum(1 for t in sub if t["won"]) / len(sub) * 100
            sp = sum(t["profit"] for t in sub)
            print(f"  Tier={tname}: {len(sub)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")

        # BTC status
        for bname, bfilt in [
            ("btc_agree", lambda t: t["btc_ag"]),
            ("btc_disagree", lambda t: t["btc_dis"]),
            ("btc_neutral", lambda t: not t["btc_ag"] and not t["btc_dis"]),
        ]:
            sub = [t for t in grp if bfilt(t)]
            if not sub:
                continue
            sw = sum(1 for t in sub if t["won"]) / len(sub) * 100
            sp = sum(t["profit"] for t in sub)
            print(f"  {bname}: {len(sub)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")

        # Crowd alignment
        wc = [t for t in grp if t["with_crowd"]]
        ac = [t for t in grp if not t["with_crowd"]]
        if wc:
            sw = sum(1 for t in wc if t["won"]) / len(wc) * 100
            sp = sum(t["profit"] for t in wc)
            print(f"  with_crowd: {len(wc)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")
        if ac:
            sw = sum(1 for t in ac if t["won"]) / len(ac) * 100
            sp = sum(t["profit"] for t in ac)
            print(f"  against_crowd: {len(ac)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")

        # Payout distribution
        payouts = [t["payout"] for t in grp]
        print(f"  payout: min={min(payouts):.2f} med={sorted(payouts)[len(payouts)//2]:.2f} max={max(payouts):.2f}")

        # Ret magnitude buckets
        for lo, hi, lbl in [(0, 2, "<2bp"), (2, 4, "2-4bp"), (4, 6, "4-6bp"), (6, 999, ">6bp")]:
            sub = [t for t in grp if lo <= t["max_ret"] * 10000 < hi]
            if not sub:
                continue
            sw = sum(1 for t in sub if t["won"]) / len(sub) * 100
            sp = sum(t["profit"] for t in sub)
            print(f"  ret {lbl}: {len(sub)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")

        # Vol buckets
        for lo, hi, lbl in [(0, 0.5, "<0.5bp"), (0.5, 1.0, "0.5-1bp"), (1.0, 1.5, "1-1.5bp"), (1.5, 999, ">1.5bp")]:
            sub = [t for t in grp if lo <= t["vol"] < hi]
            if not sub:
                continue
            sw = sum(1 for t in sub if t["won"]) / len(sub) * 100
            sp = sum(t["profit"] for t in sub)
            print(f"  vol {lbl}: {len(sub)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")

        # Hour-of-day
        hour_bins = defaultdict(list)
        for t in grp:
            hour_bins[t["hour"]].append(t)
        bad_hours = []
        for h in sorted(hour_bins):
            hb = hour_bins[h]
            if len(hb) >= 10:
                hw = sum(1 for t in hb if t["won"]) / len(hb) * 100
                hp = sum(t["profit"] for t in hb)
                if hw < 45:
                    bad_hours.append((h, len(hb), hw, hp))
        if bad_hours:
            print("  bad hours (<45% WR, n>=10):")
            for h, n2, hw, hp in bad_hours:
                print(f"    hour={h:02d}: {n2} bets, WR={hw:.1f}%, PnL={hp:+.2f}")

    analyze("SEG1 (Dec11-Jan4)", seg1)
    analyze("REST (Jan4-Apr12)", rest)

    # Cross-tab: tier x crowd for seg1
    print("\n=== SEG1 CROSS-TAB: tier x crowd ===")
    for tname in ["accel", "any+btc"]:
        for crowd_lbl, cfilt in [("with_crowd", True), ("against_crowd", False)]:
            sub = [t for t in seg1 if t["tier"] == tname and t["with_crowd"] == cfilt]
            if not sub:
                continue
            sw = sum(1 for t in sub if t["won"]) / len(sub) * 100
            sp = sum(t["profit"] for t in sub)
            print(f"  {tname} + {crowd_lbl}: {len(sub)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")

    # Cross-tab: tier x crowd for REST
    print("\n=== REST CROSS-TAB: tier x crowd ===")
    for tname in ["accel", "any+btc"]:
        for crowd_lbl, cfilt in [("with_crowd", True), ("against_crowd", False)]:
            sub = [t for t in rest if t["tier"] == tname and t["with_crowd"] == cfilt]
            if not sub:
                continue
            sw = sum(1 for t in sub if t["won"]) / len(sub) * 100
            sp = sum(t["profit"] for t in sub)
            print(f"  {tname} + {crowd_lbl}: {len(sub)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")

    # Cross-tab: tier x btc for seg1
    print("\n=== SEG1 CROSS-TAB: tier x btc ===")
    for tname in ["accel", "any+btc"]:
        for blbl, bfilt in [("btc_ag", lambda t: t["btc_ag"]), ("btc_dis", lambda t: t["btc_dis"]), ("btc_neu", lambda t: not t["btc_ag"] and not t["btc_dis"])]:
            sub = [t for t in seg1 if t["tier"] == tname and bfilt(t)]
            if not sub:
                continue
            sw = sum(1 for t in sub if t["won"]) / len(sub) * 100
            sp = sum(t["profit"] for t in sub)
            print(f"  {tname} + {blbl}: {len(sub)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")

    print("\n=== REST CROSS-TAB: tier x btc ===")
    for tname in ["accel", "any+btc"]:
        for blbl, bfilt in [("btc_ag", lambda t: t["btc_ag"]), ("btc_dis", lambda t: t["btc_dis"]), ("btc_neu", lambda t: not t["btc_ag"] and not t["btc_dis"])]:
            sub = [t for t in rest if t["tier"] == tname and bfilt(t)]
            if not sub:
                continue
            sw = sum(1 for t in sub if t["won"]) / len(sub) * 100
            sp = sum(t["profit"] for t in sub)
            print(f"  {tname} + {blbl}: {len(sub)} bets, WR={sw:.1f}%, PnL={sp:+.2f}")


if __name__ == "__main__":
    main()
