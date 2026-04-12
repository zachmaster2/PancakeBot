"""Fast standalone backtest harness for parameter experiments.

Replicates the official backtest logic exactly but allows quick parameter sweeps
without touching production code. Reports per-segment breakdown.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.domain.pool_amounts import compute_pool_amounts_wei, compute_pool_amounts_wei_at_or_before
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore

# ---- Data loading (cached across runs in same process) ----
_cache = {}

def _load_data():
    if _cache:
        return _cache["rounds"], _cache["spot"], _cache["btc"]
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())

    def load_klines(path):
        out = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    spot = load_klines("var/cutoff_spot_prices.jsonl")
    btc = load_klines("var/btc_spot_prices.jsonl")
    _cache.update(rounds=rounds, spot=spot, btc=btc)
    return rounds, spot, btc


# ---- Signal logic (parameterized) ----

def _get_return(closes, lookback):
    if len(closes) < lookback + 1: return None
    ago = closes[-(lookback + 1)]
    if ago <= 0: return None
    return (closes[-1] / ago) - 1.0


def compute_signal(bnb_closes, btc_closes, *, params):
    """Returns (signal, tier, btc_agrees, btc_disagrees) or (None,...) if no signal."""
    accel_pairs = params.get("accel_pairs", [(7,10),(5,10),(5,7)])
    accel_thresh = params.get("accel_thresh", 0.0002)
    btc_lookback = params.get("btc_lookback", 30)
    btc_thresh = params.get("btc_thresh", 0.0003)

    # Tier 1: BNB Acceleration
    for short, long in accel_pairs:
        rs = _get_return(bnb_closes, short)
        rl = _get_return(bnb_closes, long)
        if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
            if max(abs(rs), abs(rl)) >= accel_thresh:
                d = "Bull" if rs > 0 else "Bear"
                btc_ag, btc_dis = False, False
                if btc_closes is not None:
                    btc_r = _get_return(btc_closes, btc_lookback)
                    if btc_r is not None and abs(btc_r) >= btc_thresh:
                        btc_dir = "Bull" if btc_r > 0 else "Bear"
                        btc_ag = (btc_dir == d)
                        btc_dis = (btc_dir != d)
                return d, "accel", btc_ag, btc_dis

    # Tier 2: BNB Any + BTC Confirmation
    if btc_closes is not None:
        bnb_r = _get_return(bnb_closes, 7)
        if bnb_r is not None and bnb_r != 0:
            btc_r = _get_return(btc_closes, btc_lookback)
            if btc_r is not None and abs(btc_r) >= btc_thresh:
                bnb_dir = "Bull" if bnb_r > 0 else "Bear"
                btc_dir = "Bull" if btc_r > 0 else "Bear"
                if bnb_dir == btc_dir:
                    return bnb_dir, "any+btc", True, False

    return None, None, False, False


def _trim_klines(klines, cutoff_ms):
    before = [k for k in klines if int(k[0]) < cutoff_ms]
    return before[-40:] if len(before) > 40 else before


# ---- Settlement (exact match of production) ----

def settle(bet_bnb, bet_side, round_t, treasury_fee=0.03):
    # Use ALL bets in the round (no timestamp filter) — matches on-chain
    # settlement which uses the final pool totals regardless of bet timing.
    pools_wei = compute_pool_amounts_wei(bets=round_t.bets)
    bull_bnb = pools_wei.bull_wei / BNB_WEI
    bear_bnb = pools_wei.bear_wei / BNB_WEI

    if round_t.failed:
        return bet_bnb - GAS_COST_CLAIM_BNB, "refund"

    winner = round_t.position.upper()
    side_u = bet_side.upper()
    if winner != side_u:
        return 0.0, "loss"

    bull_after = bull_bnb + (bet_bnb if side_u == "BULL" else 0.0)
    bear_after = bear_bnb + (bet_bnb if side_u == "BEAR" else 0.0)
    total = bull_after + bear_after
    denom = bull_after if side_u == "BULL" else bear_after
    if denom <= 0 or total <= 0:
        return -GAS_COST_CLAIM_BNB, "win"
    mult = (total * (1.0 - treasury_fee)) / denom
    credit = bet_bnb * mult - GAS_COST_CLAIM_BNB
    return credit, "win"


# ---- Main backtest function ----

def run(params, sim_size=20000, verbose=True):
    """Run backtest with given params dict. Returns (net_pnl, segment_results, all_trades)."""
    rounds, spot, btc = _load_data()
    sim_rounds = rounds[-sim_size:]

    cutoff_sec = params.get("cutoff_seconds", 4)
    base_frac = params.get("base_frac", 0.06)
    floor_bnb = params.get("floor_bnb", 0.05)
    cap_bnb = params.get("cap_bnb", 0.35)
    btc_agree_mult = params.get("btc_agree_mult", 1.25)
    btc_disagree_mult = params.get("btc_disagree_mult", 0.6)
    payout_hi_thresh = params.get("payout_hi_thresh", 2.0)
    payout_hi_mult = params.get("payout_hi_mult", 1.7)
    payout_lo_thresh = params.get("payout_lo_thresh", 1.7)
    payout_lo_mult = params.get("payout_lo_mult", 0.9)
    evening_skip = params.get("evening_skip", (18, 24))
    pool_confirm_thresh = params.get("pool_confirm_thresh", 0.10)
    treasury_fee = params.get("treasury_fee", 0.03)

    # New optional filters
    require_btc_agree = params.get("require_btc_agree", False)
    skip_btc_disagree = params.get("skip_btc_disagree", False)
    tier1_only = params.get("tier1_only", False)
    tier2_only = params.get("tier2_only", False)
    min_accel_ret = params.get("min_accel_ret", None)  # additional min |ret| beyond accel_thresh
    contrarian_only = params.get("contrarian_only", False)  # only bet when signal disagrees with crowd
    crowd_agree_only = params.get("crowd_agree_only", False)  # only bet when signal agrees with crowd
    min_our_payout = params.get("min_our_payout", None)  # skip if payout on our side < this

    bankroll = 50.0
    trades = []  # (epoch, action, profit, bankroll, tier, signal)

    for rnd in sim_rounds:
        epoch = int(rnd.epoch)
        lock_at = int(rnd.lock_at)
        cutoff_ms = (lock_at - cutoff_sec) * 1000

        # Evening filter
        if evening_skip:
            hour = (lock_at % 86400) // 3600
            if evening_skip[0] <= hour < evening_skip[1]:
                trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "evening"))
                continue

        # Get klines
        bnb_kl = spot.get(epoch)
        btc_kl = btc.get(epoch)
        if not bnb_kl:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "no_klines"))
            continue

        bnb_trimmed = _trim_klines(bnb_kl, cutoff_ms)
        btc_trimmed = _trim_klines(btc_kl, cutoff_ms) if btc_kl else None

        if len(bnb_trimmed) < 40:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "insufficient"))
            continue

        bnb_closes = [k[4] for k in bnb_trimmed]
        btc_closes = [k[4] for k in btc_trimmed] if btc_trimmed and len(btc_trimmed) >= 40 else None

        signal, tier, btc_ag, btc_dis = compute_signal(bnb_closes, btc_closes, params=params)

        if signal is None:
            trades.append((epoch, "SKIP", 0.0, bankroll, None, None, "no_signal"))
            continue

        # Tier filters
        if tier1_only and tier != "accel":
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "tier2_skip"))
            continue
        if tier2_only and tier != "any+btc":
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "tier1_skip"))
            continue

        # BTC filters
        if require_btc_agree and not btc_ag:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "no_btc_agree"))
            continue
        if skip_btc_disagree and btc_dis:
            trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "btc_disagree"))
            continue

        # Min return filter
        if min_accel_ret is not None and tier == "accel":
            accel_pairs = params.get("accel_pairs", [(7,10),(5,10),(5,7)])
            max_ret = 0
            for s, l in accel_pairs:
                for lb in (s, l):
                    r = _get_return(bnb_closes, lb)
                    if r is not None:
                        max_ret = max(max_ret, abs(r))
            if max_ret < min_accel_ret:
                trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "low_ret"))
                continue

        # Contrarian / crowd-agree filter
        if (contrarian_only or crowd_agree_only) and rnd.bets:
            bull_wei, bear_wei = 0, 0
            for b in rnd.bets:
                if int(b.created_at) > lock_at: continue
                if b.position == "Bull": bull_wei += int(b.amount_wei)
                else: bear_wei += int(b.amount_wei)
            pb = bull_wei / BNB_WEI
            prb = bear_wei / BNB_WEI
            if pb + prb > 0:
                crowd_dir = "Bull" if pb > prb else "Bear"
                if contrarian_only and signal == crowd_dir:
                    trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "crowd_agrees"))
                    continue
                if crowd_agree_only and signal != crowd_dir:
                    trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "crowd_disagrees"))
                    continue

        # Pool confirmation filter
        if pool_confirm_thresh is not None and rnd.bets:
            bull_wei, bear_wei = 0, 0
            for b in rnd.bets:
                if int(b.created_at) > lock_at: continue
                if b.position == "Bull": bull_wei += int(b.amount_wei)
                else: bear_wei += int(b.amount_wei)
            pb = bull_wei / BNB_WEI
            prb = bear_wei / BNB_WEI
            pt = pb + prb
            if pt > 0:
                imb = (pb - prb) / pt
                pool_dir = "Bull" if imb > 0 else "Bear"
                if abs(imb) >= pool_confirm_thresh and pool_dir != signal:
                    trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "pool_disagrees"))
                    continue

        # Payout floor filter
        if min_our_payout is not None and rnd.bets:
            bull_wei, bear_wei = 0, 0
            for b in rnd.bets:
                if int(b.created_at) > lock_at: continue
                if b.position == "Bull": bull_wei += int(b.amount_wei)
                else: bear_wei += int(b.amount_wei)
            pb = bull_wei / BNB_WEI
            prb = bear_wei / BNB_WEI
            pt = pb + prb
            if pt > 0:
                our_side = pb if signal == "Bull" else prb
                if our_side > 0:
                    pm = pt * (1.0 - treasury_fee) / our_side
                    if pm < min_our_payout:
                        trades.append((epoch, "SKIP", 0.0, bankroll, tier, signal, "low_payout"))
                        continue

        # Sizing
        bull_wei, bear_wei = 0, 0
        for b in rnd.bets:
            if int(b.created_at) > lock_at: continue
            if b.position == "Bull": bull_wei += int(b.amount_wei)
            else: bear_wei += int(b.amount_wei)
        pool_bull = bull_wei / BNB_WEI
        pool_bear = bear_wei / BNB_WEI
        pool_total = pool_bull + pool_bear

        bet = max(floor_bnb, pool_total * base_frac) if pool_total > 0 else floor_bnb

        # Payout adjustment
        our_side = pool_bull if signal == "Bull" else pool_bear
        if our_side > 0:
            pm = pool_total * (1.0 - treasury_fee) / our_side
            if pm >= payout_hi_thresh: bet *= payout_hi_mult
            elif pm < payout_lo_thresh: bet *= payout_lo_mult

        if btc_ag: bet *= btc_agree_mult
        elif btc_dis: bet *= btc_disagree_mult

        bet = min(cap_bnb, bet)

        # Execute
        bankroll -= bet + GAS_COST_BET_BNB
        credit, outcome = settle(bet, signal, rnd, treasury_fee)
        bankroll += credit
        profit = credit - bet - GAS_COST_BET_BNB
        trades.append((epoch, "BET", profit, bankroll, tier, signal, outcome))

    net = bankroll - 50.0

    # Segment analysis
    seg_size = sim_size // 4
    segments = []
    for s in range(4):
        chunk = trades[s*seg_size:(s+1)*seg_size]
        bets = [t for t in chunk if t[1] == "BET"]
        wins = [t for t in bets if t[2] > 0]
        pnl = sum(t[2] for t in bets)
        wr = len(wins)/len(bets)*100 if bets else 0
        segments.append((len(bets), wr, pnl))

    if verbose:
        print(f"NET: {net:+.2f} BNB | Bets: {sum(1 for t in trades if t[1]=='BET')} | "
              f"WR: {sum(1 for t in trades if t[1]=='BET' and t[2]>0)/max(1,sum(1 for t in trades if t[1]=='BET'))*100:.1f}%")
        for i, (nb, wr, pnl) in enumerate(segments):
            print(f"  Seg{i+1}: {nb:4d} bets, WR={wr:5.1f}%, PnL={pnl:+7.2f}")

    return net, segments, trades

if __name__ == "__main__":
    run({})
