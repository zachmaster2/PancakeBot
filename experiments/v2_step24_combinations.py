"""Step 24: Combination signals for flat periods.

Step 23 showed individual signals are near-zero in flat periods.
This tests COMBINATIONS that weren't tried:

1. Weak BTC + ETH/SOL confirmation required (not just sizing)
2. ETH + SOL both multi-TF firing (without BTC)
3. Partial unanimity: r15 strong + at least one shorter agrees
4. Ensemble: 3+ weak signals all agree
5. BNB divergence while BTC is specifically flat (low vol)
6. Any above combined with pool/payout filters
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pancakebot.domain.strategy.momentum_gate as _gate_mod
from pancakebot.core.constants import (
    BNB_WEI, GAS_COST_BET_BNB, POOL_CUTOFF_SECONDS,
)
from pancakebot.domain.strategy.momentum_gate import _trim_to_window
from pancakebot.domain.strategy.momentum_pipeline import _pools_from_bets
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2
CANDLE_COUNT = 31
N_FOLDS = 5
TREASURY_FEE = 0.03
BET_FRAC = 0.05
BET_CAP = 2.0
MIN_POOL = 1.5
MIN_PAYOUT = 1.5


def load_data():
    print("Loading data...", end=" ", flush=True)
    t0 = time.time()
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    def lk(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip(): continue
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


def get_closes(raw, cutoff_ms):
    if raw is None: return None
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT: return None
    return [c[4] for c in trimmed[-CANDLE_COUNT:]]


def get_volumes(raw, cutoff_ms):
    if raw is None: return None
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT: return None
    return [c[5] for c in trimmed[-CANDLE_COUNT:]]


def ret(closes, lb):
    if closes is None or len(closes) < lb + 1 or closes[-(lb+1)] == 0:
        return None
    return (closes[-1] - closes[-(lb+1)]) / closes[-(lb+1)]


def multi_tf_check(closes, lookbacks=(3,7,15)):
    """Returns (fires, direction, min_abs, returns) or (False, None, 0, [])."""
    rets = [ret(closes, lb) for lb in lookbacks]
    if any(r is None for r in rets):
        return False, None, 0, []
    if all(r > 0 for r in rets):
        return True, "Bull", min(abs(r) for r in rets), rets
    if all(r < 0 for r in rets):
        return True, "Bear", min(abs(r) for r in rets), rets
    return False, None, 0, rets


def precompute(rounds, bnb_kl, btc_kl, eth_kl, sol_kl):
    print("Pre-computing...", end=" ", flush=True)
    t0 = time.time()
    data = []
    for rnd in rounds:
        ep = int(rnd.epoch)
        la = int(rnd.lock_at)
        cms = (la - CUTOFF_S) * 1000

        btc_c = get_closes(btc_kl.get(ep), cms)
        bnb_c = get_closes(bnb_kl.get(ep), cms)
        eth_c = get_closes(eth_kl.get(ep), cms)
        sol_c = get_closes(sol_kl.get(ep), cms)
        btc_v = get_volumes(btc_kl.get(ep), cms)

        pb, pe = _pools_from_bets(rnd, la - POOL_CUTOFF_SECONDS)

        # Pre-compute all multi-TF checks
        btc_fires, btc_dir, btc_str, btc_rets = multi_tf_check(btc_c)
        eth_fires, eth_dir, eth_str, _ = multi_tf_check(eth_c)
        sol_fires, sol_dir, sol_str, _ = multi_tf_check(sol_c)
        bnb_fires, bnb_dir, bnb_str, _ = multi_tf_check(bnb_c)

        # BTC individual returns
        btc_r3 = ret(btc_c, 3)
        btc_r7 = ret(btc_c, 7)
        btc_r15 = ret(btc_c, 15)
        bnb_r3 = ret(bnb_c, 3)
        bnb_r5 = ret(bnb_c, 5)
        bnb_r7 = ret(bnb_c, 7)

        # BTC volatility (std of returns over last 15 candles)
        btc_vol = 0.0
        if btc_c and len(btc_c) >= 16:
            rtns = [(btc_c[i] - btc_c[i-1]) / btc_c[i-1] for i in range(-15, 0) if btc_c[i-1] != 0]
            if rtns:
                m = sum(rtns) / len(rtns)
                btc_vol = (sum((r - m)**2 for r in rtns) / len(rtns)) ** 0.5

        # Volume spike
        vol_spike = 0.0
        if btc_v and len(btc_v) >= 15:
            recent = sum(btc_v[-3:]) / 3
            baseline = sum(btc_v[-15:-3]) / 12
            if baseline > 0:
                vol_spike = recent / baseline

        # Primary signal fires (for identifying flat rounds)
        primary_fires = btc_fires and btc_str >= 0.0001

        data.append({
            "rnd": rnd, "epoch": ep,
            "btc_c": btc_c, "bnb_c": bnb_c, "eth_c": eth_c, "sol_c": sol_c,
            "pb": pb, "pe": pe, "pt": pb + pe,
            "btc_fires": btc_fires, "btc_dir": btc_dir, "btc_str": btc_str, "btc_rets": btc_rets,
            "eth_fires": eth_fires, "eth_dir": eth_dir, "eth_str": eth_str,
            "sol_fires": sol_fires, "sol_dir": sol_dir, "sol_str": sol_str,
            "bnb_fires": bnb_fires, "bnb_dir": bnb_dir, "bnb_str": bnb_str,
            "btc_r3": btc_r3, "btc_r7": btc_r7, "btc_r15": btc_r15,
            "bnb_r3": bnb_r3, "bnb_r5": bnb_r5, "bnb_r7": bnb_r7,
            "btc_vol": btc_vol, "vol_spike": vol_spike,
            "primary_fires": primary_fires,
        })

    n_flat = sum(1 for d in data if not d["primary_fires"])
    print(f"{len(data)} rounds, {n_flat} flat, {time.time()-t0:.1f}s")
    return data


def eval_5fold(data, signal_fn, flat_only=True, min_pool=MIN_POOL, min_payout=MIN_PAYOUT):
    fold_size = len(data) // N_FOLDS
    results = []
    for fi in range(N_FOLDS):
        fold = data[fi * fold_size:(fi + 1) * fold_size]
        bets = wins = 0
        pnl = 0.0
        for d in fold:
            if flat_only and d["primary_fires"]:
                continue
            signal = signal_fn(d)
            if signal is None:
                continue
            pt = d["pt"]
            if pt < min_pool:
                continue
            our = d["pb"] if signal == "Bull" else d["pe"]
            if our > 0 and pt > 0:
                pay = pt * 0.97 / our
                if pay < min_payout:
                    continue
            bet = max(0.01, min(BET_CAP, pt * BET_FRAC))
            out = settle_bet_against_closed_round(
                bet_bnb=bet, bet_side=signal,
                round_closed=d["rnd"], treasury_fee_fraction=TREASURY_FEE,
            )
            profit = out.credit_bnb - bet - GAS_COST_BET_BNB
            pnl += profit; bets += 1
            if profit > 0: wins += 1
        pnl_2k = pnl / len(fold) * 2000 if fold else 0
        results.append((bets, wins, pnl, pnl_2k))
    return results


def pr(label, fr, n_flat):
    tb = sum(f[0] for f in fr)
    tw = sum(f[1] for f in fr)
    pnls = [f[3] for f in fr]
    avg = sum(pnls) / len(pnls)
    npos = sum(1 for p in pnls if p > 0)
    wr = tw / tb * 100 if tb > 0 else 0
    b2k = tb / n_flat * 2000 if n_flat > 0 else 0
    m = " ***" if npos >= 5 else " **" if npos >= 4 else " *" if npos >= 3 else ""
    fs = " ".join(f"{p:+6.2f}" for p in pnls)
    print(f"  {label:<55s} {b2k:6.1f} {wr:5.1f}% {avg:+7.2f} {npos}/5  {fs}{m}")


def main():
    rounds, bnb_kl, btc_kl, eth_kl, sol_kl = load_data()
    data = precompute(rounds, bnb_kl, btc_kl, eth_kl, sol_kl)
    n_flat = sum(1 for d in data if not d["primary_fires"])

    print(f"\n{'='*130}")
    print("COMBINATION SIGNALS — flat periods only (primary BTC multi-TF silent)")
    print(f"{'='*130}")
    hdr = f"  {'Signal':<55s} {'b/2k':>6} {'WR%':>6} {'PnL/2k':>8} {'pos':>4}  {'f1':>6} {'f2':>6} {'f3':>6} {'f4':>6} {'f5':>6}"
    print(hdr)
    print(f"  {'-'*120}")

    # ===== 1. WEAK BTC + ETH/SOL CONFIRMATION REQUIRED =====
    print("\n  --- 1. Weak BTC (below primary thresh) + ETH/SOL must confirm ---")

    for btc_thresh in [0.00003, 0.00005, 0.00008]:
        # Weak BTC fires + ETH confirms
        def sig_btc_eth(d, t=btc_thresh):
            if not d["btc_fires"] or d["btc_str"] >= 0.0001:  # only WEAK btc
                return None
            if d["btc_str"] < t:
                return None
            if d["eth_fires"] and d["eth_dir"] == d["btc_dir"]:
                return d["btc_dir"]
            return None
        fr = eval_5fold(data, sig_btc_eth)
        pr(f"Weak BTC>{btc_thresh} + ETH confirms", fr, n_flat)

        # Weak BTC + SOL confirms
        def sig_btc_sol(d, t=btc_thresh):
            if not d["btc_fires"] or d["btc_str"] >= 0.0001:
                return None
            if d["btc_str"] < t:
                return None
            if d["sol_fires"] and d["sol_dir"] == d["btc_dir"]:
                return d["btc_dir"]
            return None
        fr = eval_5fold(data, sig_btc_sol)
        pr(f"Weak BTC>{btc_thresh} + SOL confirms", fr, n_flat)

        # Weak BTC + EITHER ETH or SOL confirms
        def sig_btc_either(d, t=btc_thresh):
            if not d["btc_fires"] or d["btc_str"] >= 0.0001:
                return None
            if d["btc_str"] < t:
                return None
            eth_ok = d["eth_fires"] and d["eth_dir"] == d["btc_dir"]
            sol_ok = d["sol_fires"] and d["sol_dir"] == d["btc_dir"]
            if eth_ok or sol_ok:
                return d["btc_dir"]
            return None
        fr = eval_5fold(data, sig_btc_either)
        pr(f"Weak BTC>{btc_thresh} + ETH|SOL confirms", fr, n_flat)

        # Weak BTC + BOTH ETH and SOL confirm
        def sig_btc_both(d, t=btc_thresh):
            if not d["btc_fires"] or d["btc_str"] >= 0.0001:
                return None
            if d["btc_str"] < t:
                return None
            if d["eth_fires"] and d["eth_dir"] == d["btc_dir"] and \
               d["sol_fires"] and d["sol_dir"] == d["btc_dir"]:
                return d["btc_dir"]
            return None
        fr = eval_5fold(data, sig_btc_both)
        pr(f"Weak BTC>{btc_thresh} + ETH+SOL both confirm", fr, n_flat)

    # Also test: BTC fires at any strength but below pool-adaptive thresh + ETH/SOL
    print("\n  --- 1b. BTC fires (any strength) + confirmation required ---")
    for btc_thresh in [0.00003, 0.00005, 0.00008]:
        def sig_any_btc_confirm(d, t=btc_thresh):
            if not d["btc_fires"] or d["btc_str"] < t:
                return None
            # Already handled by primary if btc_str >= 0.0001 for large pools
            # But for small pools (thresh=0.0002), range 0.0001-0.0002 is uncovered
            eth_ok = d["eth_fires"] and d["eth_dir"] == d["btc_dir"]
            sol_ok = d["sol_fires"] and d["sol_dir"] == d["btc_dir"]
            if eth_ok or sol_ok:
                return d["btc_dir"]
            return None
        fr = eval_5fold(data, sig_any_btc_confirm, flat_only=False)
        pr(f"ALL: BTC>{btc_thresh} + ETH|SOL (incl primary)", fr, len(data))

    # ===== 2. ETH + SOL BOTH FIRING (NO BTC REQUIRED) =====
    print("\n  --- 2. ETH + SOL both multi-TF same direction (BTC silent) ---")
    for eth_thresh in [0.00005, 0.0001, 0.00015, 0.0002]:
        for sol_thresh in [0.00005, 0.0001, 0.00015, 0.0002]:
            if sol_thresh != eth_thresh:
                continue  # just sweep matching thresholds
            def sig_eth_sol(d, et=eth_thresh, st=sol_thresh):
                if not d["eth_fires"] or d["eth_str"] < et:
                    return None
                if not d["sol_fires"] or d["sol_str"] < st:
                    return None
                if d["eth_dir"] != d["sol_dir"]:
                    return None
                return d["eth_dir"]
            fr = eval_5fold(data, sig_eth_sol)
            pr(f"ETH>{eth_thresh} + SOL>{sol_thresh} agree", fr, n_flat)

    # ETH + SOL + BNB all agree
    print("\n  --- 2b. ETH + SOL + BNB all multi-TF same direction ---")
    for thresh in [0.00005, 0.0001, 0.00015]:
        def sig_three_pair(d, t=thresh):
            if not d["eth_fires"] or d["eth_str"] < t: return None
            if not d["sol_fires"] or d["sol_str"] < t: return None
            if not d["bnb_fires"] or d["bnb_str"] < t: return None
            if d["eth_dir"] != d["sol_dir"] or d["eth_dir"] != d["bnb_dir"]:
                return None
            return d["eth_dir"]
        fr = eval_5fold(data, sig_three_pair)
        pr(f"ETH+SOL+BNB all >{thresh} agree", fr, n_flat)

    # ===== 3. PARTIAL UNANIMITY =====
    print("\n  --- 3. Partial unanimity: r15 strong + at least one shorter agrees ---")
    for r15_thresh in [0.0001, 0.00015, 0.0002, 0.0003]:
        # r15 strong + r7 same sign (any strength)
        def sig_r15_r7(d, t=r15_thresh):
            r15 = d["btc_r15"]; r7 = d["btc_r7"]
            if r15 is None or r7 is None: return None
            if abs(r15) < t: return None
            if (r15 > 0) != (r7 > 0): return None  # must agree in direction
            return "Bull" if r15 > 0 else "Bear"
        fr = eval_5fold(data, sig_r15_r7)
        pr(f"BTC r15>{r15_thresh} + r7 same sign", fr, n_flat)

        # r15 strong + r3 same sign
        def sig_r15_r3(d, t=r15_thresh):
            r15 = d["btc_r15"]; r3 = d["btc_r3"]
            if r15 is None or r3 is None: return None
            if abs(r15) < t: return None
            if (r15 > 0) != (r3 > 0): return None
            return "Bull" if r15 > 0 else "Bear"
        fr = eval_5fold(data, sig_r15_r3)
        pr(f"BTC r15>{r15_thresh} + r3 same sign", fr, n_flat)

        # r15 strong + either r3 or r7 same sign
        def sig_r15_any(d, t=r15_thresh):
            r15 = d["btc_r15"]; r3 = d["btc_r3"]; r7 = d["btc_r7"]
            if r15 is None: return None
            if abs(r15) < t: return None
            r3_ok = r3 is not None and (r15 > 0) == (r3 > 0)
            r7_ok = r7 is not None and (r15 > 0) == (r7 > 0)
            if not (r3_ok or r7_ok): return None
            return "Bull" if r15 > 0 else "Bear"
        fr = eval_5fold(data, sig_r15_any)
        pr(f"BTC r15>{r15_thresh} + (r3|r7) same sign", fr, n_flat)

    # ===== 4. ENSEMBLE: multiple weak signals agree =====
    print("\n  --- 4. Ensemble: multiple independent signals agree ---")

    def _votes(d):
        """Return list of (signal_name, direction) for weak signals that fire."""
        votes = []
        # BNB momentum r3
        if d["bnb_r3"] is not None and abs(d["bnb_r3"]) > 0.0002:
            votes.append(("bnb_r3", "Bull" if d["bnb_r3"] > 0 else "Bear"))
        # SOL multi-TF
        if d["sol_fires"] and d["sol_str"] >= 0.00015:
            votes.append(("sol_mtf", d["sol_dir"]))
        # ETH multi-TF
        if d["eth_fires"] and d["eth_str"] >= 0.00015:
            votes.append(("eth_mtf", d["eth_dir"]))
        # BTC r15 (longer lookback, not 3-TF but strong single)
        if d["btc_r15"] is not None and abs(d["btc_r15"]) > 0.0002:
            votes.append(("btc_r15", "Bull" if d["btc_r15"] > 0 else "Bear"))
        # BNB multi-TF
        if d["bnb_fires"] and d["bnb_str"] >= 0.00015:
            votes.append(("bnb_mtf", d["bnb_dir"]))
        # Volume spike + BTC direction
        if d["vol_spike"] > 2.0 and d["btc_r3"] is not None and abs(d["btc_r3"]) > 0.00005:
            votes.append(("vol_btc", "Bull" if d["btc_r3"] > 0 else "Bear"))
        return votes

    for min_votes in [2, 3, 4]:
        def sig_ensemble(d, mv=min_votes):
            votes = _votes(d)
            if len(votes) < mv:
                return None
            bull = sum(1 for _, dir in votes if dir == "Bull")
            bear = sum(1 for _, dir in votes if dir == "Bear")
            if bull >= mv:
                return "Bull"
            if bear >= mv:
                return "Bear"
            return None
        fr = eval_5fold(data, sig_ensemble)
        pr(f"Ensemble: {min_votes}+ signals agree", fr, n_flat)

    # Ensemble with unanimous agreement (all votes same direction)
    def sig_ensemble_unanimous(d):
        votes = _votes(d)
        if len(votes) < 2:
            return None
        dirs = set(dir for _, dir in votes)
        if len(dirs) == 1:
            return dirs.pop()
        return None
    fr = eval_5fold(data, sig_ensemble_unanimous)
    pr("Ensemble: all votes unanimous (2+ signals)", fr, n_flat)

    # ===== 5. BNB DIVERGENCE (BNB moves, BTC specifically flat) =====
    print("\n  --- 5. BNB diverges while BTC is flat (low volatility) ---")
    for bnb_thresh in [0.0002, 0.0003, 0.0004]:
        for btc_vol_max in [0.00005, 0.0001, 0.00015]:
            def sig_bnb_div(d, bt=bnb_thresh, vm=btc_vol_max):
                if d["btc_vol"] > vm:  # BTC must be flat
                    return None
                if d["bnb_r3"] is None or abs(d["bnb_r3"]) < bt:
                    return None
                return "Bull" if d["bnb_r3"] > 0 else "Bear"
            fr = eval_5fold(data, sig_bnb_div)
            pr(f"BNB r3>{bnb_thresh} + BTC vol<{btc_vol_max}", fr, n_flat)

    # ===== 6. COMBINED: primary + regime-2 (full portfolio test) =====
    print(f"\n{'='*130}")
    print("PORTFOLIO TEST: Primary (BTC multi-TF) + best regime-2 signal combined")
    print(f"{'='*130}")
    print(hdr)
    print(f"  {'-'*120}")

    # Test: primary signal on ALL rounds + each candidate on flat rounds
    # This shows the TOTAL portfolio effect

    def make_portfolio_fn(regime2_fn):
        def portfolio(d):
            # Primary signal
            if d["primary_fires"]:
                # Use primary (already handled by production code)
                return d["btc_dir"]
            # Regime 2 on flat rounds
            return regime2_fn(d)
        return portfolio

    # Candidates from above (will be filled after we see results)
    candidates = {
        "Primary only (baseline)": lambda d: d["btc_dir"] if d["primary_fires"] else None,
        "Primary + ETH&SOL agree>5e-5": make_portfolio_fn(
            lambda d: d["eth_dir"] if d["eth_fires"] and d["eth_str"]>=0.00005 and d["sol_fires"] and d["sol_str"]>=0.00005 and d["eth_dir"]==d["sol_dir"] else None
        ),
        "Primary + ETH&SOL agree>1e-4": make_portfolio_fn(
            lambda d: d["eth_dir"] if d["eth_fires"] and d["eth_str"]>=0.0001 and d["sol_fires"] and d["sol_str"]>=0.0001 and d["eth_dir"]==d["sol_dir"] else None
        ),
        "Primary + Ensemble 3+": make_portfolio_fn(
            lambda d: (lambda v: ("Bull" if sum(1 for _,dr in v if dr=="Bull")>=3 else "Bear" if sum(1 for _,dr in v if dr=="Bear")>=3 else None))(_votes(d))
        ),
        "Primary + weak BTC>5e-5 + ETH|SOL": make_portfolio_fn(
            lambda d: d["btc_dir"] if d["btc_fires"] and d["btc_str"]>=0.00005 and (
                (d["eth_fires"] and d["eth_dir"]==d["btc_dir"]) or
                (d["sol_fires"] and d["sol_dir"]==d["btc_dir"])
            ) else None
        ),
        "Primary + BNB r3>3e-4 (flat BTC)": make_portfolio_fn(
            lambda d: ("Bull" if d["bnb_r3"]>0 else "Bear") if d["bnb_r3"] is not None and abs(d["bnb_r3"])>0.0003 else None
        ),
    }

    for label, fn in candidates.items():
        fr = eval_5fold(data, fn, flat_only=False)
        pr(label, fr, len(data))


if __name__ == "__main__":
    main()
