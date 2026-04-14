"""Step 18: Cross-pair signals — does ETH lead BNB independently of BTC?

If ETH-USDT 1s klines show a similar multi-TF momentum signal that fires
on DIFFERENT rounds than BTC, we can stack them for more total bets/2k.

Phase 1: Fetch ETH 1s klines (same cutoff=2 as BTC/BNB)
Phase 2: Test ETH multi-TF signal
Phase 3: Check overlap with BTC signal
Phase 4: Stack ETH + BTC for combined PnL
Phase 5: 5-fold validate
"""
from __future__ import annotations

import json, sys, time, os, threading, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import INTERVAL_SECONDS, POOL_CUTOFF_SECONDS
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import _trim_to_window
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 2
POOL_CUTOFF_S = POOL_CUTOFF_SECONDS
CANDLE_COUNT = 31
TREASURY_FEE = 0.03
RATE_PER_SEC = 9
WORKERS = 8

ETH_PATH = "var/eth_spot_prices.jsonl"


class _RateLimiter:
    def __init__(self, max_per_sec):
        self._min_interval = 1.0 / max_per_sec
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


_rate_candles = _RateLimiter(RATE_PER_SEC)
_rate_history = _RateLimiter(RATE_PER_SEC)


def fetch_klines(inst_id, cutoff_ms):
    for ep in ("candles", "history-candles"):
        url = (f"https://www.okx.com/api/v5/market/{ep}"
               f"?instId={inst_id}&bar=1s&limit={CANDLE_COUNT}&after={cutoff_ms}")
        limiter = _rate_candles if ep == "candles" else _rate_history
        for attempt in range(2):
            limiter.acquire()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PancakeBot/1.0"})
                resp = urllib.request.urlopen(req, timeout=5)
                body = json.loads(resp.read())
                if body.get("code") == "0":
                    if body.get("data") and len(body["data"]) >= CANDLE_COUNT * 0.9:
                        rows = body["data"]
                        return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                                 float(r[4]), float(r[5])] for r in reversed(rows)][-CANDLE_COUNT:]
                    break  # code=0 but no data — skip retries
            except Exception:
                continue
    return None


def fetch_eth_klines():
    """Fetch ETH-USDT 1s klines for all rounds, with resume support."""
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds_map = {int(r.epoch): r for r in store.iter_closed_rounds()}
    print(f"Rounds: {len(rounds_map)}")

    done_epochs = set()
    if os.path.exists(ETH_PATH):
        for line in Path(ETH_PATH).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done_epochs.add(int(rec["epoch"]))
        print(f"Resuming: {len(done_epochs)} already done")

    all_epochs = sorted(rounds_map.keys())
    to_process = [ep for ep in all_epochs if ep not in done_epochs]
    print(f"To fetch: {len(to_process)}")

    if not to_process:
        print("All done!")
        return

    fetched = 0
    failed = 0
    batch_size = 80

    with open(ETH_PATH, "a", encoding="utf-8") as out_f:
        for batch_start in range(0, len(to_process), batch_size):
            batch = to_process[batch_start:batch_start + batch_size]
            fetch_list = []
            for ep in batch:
                rnd = rounds_map.get(ep)
                if not rnd:
                    continue
                cutoff_ms = (rnd.start_at + INTERVAL_SECONDS - CUTOFF_S) * 1000
                fetch_list.append((ep, cutoff_ms, rnd))

            results = {}
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(fetch_klines, "ETH-USDT", cm): ep
                           for ep, cm, _ in fetch_list}
                for fut in as_completed(futures):
                    ep = futures[fut]
                    klines = fut.result()
                    if klines and len(klines) == CANDLE_COUNT:
                        results[ep] = klines

            for ep, cm, rnd in fetch_list:
                if ep in results:
                    rec = {
                        "epoch": ep,
                        "lock_at": rnd.start_at + INTERVAL_SECONDS,
                        "klines_1s": results[ep],
                    }
                    out_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    out_f.flush()
                    fetched += 1
                else:
                    failed += 1

            done = batch_start + len(batch)
            if done % 1000 < batch_size or done >= len(to_process):
                print(f"  {done + len(done_epochs)}/{len(all_epochs)}: "
                      f"{fetched} fetched, {failed} failed", flush=True)

    print(f"Done: {fetched} fetched, {failed} failed")


def run_analysis():
    """Test ETH multi-TF signals and stacking with BTC."""
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())
    total = len(rounds)

    def load_kl(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    bnb_kl = load_kl("var/cutoff_spot_prices.jsonl")
    btc_kl = load_kl("var/btc_spot_prices.jsonl")
    eth_kl = load_kl(ETH_PATH)
    print(f"ETH klines: {len(eth_kl)}")

    def get_candles(raw, cutoff_ms):
        trimmed = _trim_to_window(raw, cutoff_ms)
        return trimmed[-CANDLE_COUNT:] if len(trimmed) >= CANDLE_COUNT else None

    def settle(rnd, bet, side):
        out = settle_bet_against_closed_round(bet_bnb=bet, bet_side=side,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE)
        return out.credit_bnb - bet - 0.0003  # GAS_COST_BET_BNB

    def ret(closes, lb):
        if len(closes) < lb + 1 or closes[-(lb+1)] == 0: return None
        return (closes[-1] - closes[-(lb+1)]) / closes[-(lb+1)]

    def get_pool(rnd, lock_at):
        cutoff = lock_at - POOL_CUTOFF_S
        b = s = 0
        for bet in rnd.bets:
            if int(bet.created_at) > cutoff: continue
            if bet.position == "Bull": b += int(bet.amount_wei)
            else: s += int(bet.amount_wei)
        return (b + s) / 1e18

    def mtf_signal(closes, lookbacks, thresh):
        rets = []
        for lb in lookbacks:
            r = ret(closes, lb)
            if r is None: return None, 0
            rets.append(r)
        if not (all(r > 0 for r in rets) or all(r < 0 for r in rets)):
            return None, 0
        m = min(abs(r) for r in rets)
        if m < thresh: return None, 0
        return ("Bull" if rets[0] > 0 else "Bear"), m

    # Build data
    data = []
    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000
        pool = get_pool(rnd, lock_at)

        btc_raw = btc_kl.get(epoch)
        bnb_raw = bnb_kl.get(epoch)
        eth_raw = eth_kl.get(epoch)
        if not btc_raw or not bnb_raw:
            continue

        btc_c_raw = get_candles(btc_raw, cutoff_ms)
        if btc_c_raw is None:
            continue
        btc_c = [k[4] for k in btc_c_raw]

        eth_c = None
        if eth_raw:
            eth_c_raw = get_candles(eth_raw, cutoff_ms)
            if eth_c_raw is not None:
                eth_c = [k[4] for k in eth_c_raw]

        data.append({
            "rnd": rnd, "lock_at": lock_at, "epoch": epoch,
            "btc_c": btc_c, "eth_c": eth_c, "pool": pool,
        })

    print(f"Data: {len(data)}, with ETH: {sum(1 for d in data if d['eth_c'])}\n")

    # =====================================================================
    print("=" * 120)
    print("PART 1: ETH multi-TF signal standalone")
    print("=" * 120)

    for tfs in [(3,7,15), (3,5,7), (5,10,20)]:
        for thresh in [0.0002, 0.0003, 0.0005]:
            results = []
            for d in data:
                if d["eth_c"] is None: continue
                if d["pool"] < 2.0: continue
                sig, strength = mtf_signal(d["eth_c"], tfs, thresh)
                if sig is None: continue
                frac = min(0.03 + 100 * strength, 0.30)
                bet = max(0.01, min(2.0, d["pool"] * frac))
                profit = settle(d["rnd"], bet, sig)
                results.append((profit, bet))
            n = len(results)
            if n < 20: continue
            profits = [p for p, b in results]
            wr = sum(1 for p in profits if p > 0) / n * 100
            pnl = sum(profits)
            pnl_2k = pnl / total * 2000
            label = "+".join(str(t) for t in tfs)
            flag = " ***" if pnl > 0 else ""
            print(f"  eth_mtf({label},t={thresh}): WR={wr:.1f}%({n}) /2k={pnl_2k:+.3f}{flag}")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 2: Overlap between BTC and ETH signals")
    print("=" * 120)

    btc_set = set()
    eth_set = set()
    for d in data:
        if d["pool"] < 2.0: continue
        sig_btc, _ = mtf_signal(d["btc_c"], (3,7,15), 0.0002)
        if sig_btc: btc_set.add(d["epoch"])
        if d["eth_c"]:
            sig_eth, _ = mtf_signal(d["eth_c"], (3,7,15), 0.0002)
            if sig_eth: eth_set.add(d["epoch"])

    print(f"  BTC fires: {len(btc_set)}")
    print(f"  ETH fires: {len(eth_set)}")
    print(f"  Overlap: {len(btc_set & eth_set)}")
    print(f"  ETH unique: {len(eth_set - btc_set)}")
    print(f"  BTC unique: {len(btc_set - eth_set)}")
    print(f"  Union: {len(btc_set | eth_set)}")

    # When both fire, do they agree on direction?
    agree = disagree = 0
    for d in data:
        if d["pool"] < 2.0 or d["eth_c"] is None: continue
        sig_btc, _ = mtf_signal(d["btc_c"], (3,7,15), 0.0002)
        sig_eth, _ = mtf_signal(d["eth_c"], (3,7,15), 0.0002)
        if sig_btc and sig_eth:
            if sig_btc == sig_eth: agree += 1
            else: disagree += 1
    print(f"  Direction agreement when both fire: {agree}/{agree+disagree} "
          f"({agree/(agree+disagree)*100:.0f}%)" if agree + disagree > 0 else "")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 3: Stacked BTC + ETH (ETH only on rounds BTC doesn't fire)")
    print("=" * 120)

    for eth_thresh in [0.0002, 0.0003, 0.0005]:
        results_btc = []
        results_eth = []
        for d in data:
            if d["pool"] < 2.0: continue
            sig_btc, str_btc = mtf_signal(d["btc_c"], (3,7,15), 0.0002)
            if sig_btc:
                frac = min(0.03 + 100 * str_btc, 0.30)
                bet = max(0.01, min(2.0, d["pool"] * frac))
                results_btc.append(settle(d["rnd"], bet, sig_btc))
                continue  # don't also use ETH on same round
            if d["eth_c"] is None: continue
            sig_eth, str_eth = mtf_signal(d["eth_c"], (3,7,15), eth_thresh)
            if sig_eth:
                frac = min(0.03 + 100 * str_eth, 0.30)
                bet = max(0.01, min(2.0, d["pool"] * frac))
                results_eth.append(settle(d["rnd"], bet, sig_eth))

        n_btc = len(results_btc)
        n_eth = len(results_eth)
        pnl_btc = sum(results_btc)
        pnl_eth = sum(results_eth)
        pnl_total = pnl_btc + pnl_eth
        wr_eth = sum(1 for p in results_eth if p > 0) / max(1, n_eth) * 100

        print(f"  eth_thresh={eth_thresh}: BTC {n_btc} bets +{pnl_btc:.3f}, "
              f"ETH-unique {n_eth} bets @{wr_eth:.1f}% +{pnl_eth:.3f}, "
              f"total={pnl_total:+.3f} /2k={pnl_total/total*2000:+.3f}")

    # =====================================================================
    print(f"\n{'=' * 120}")
    print("PART 4: 5-fold validation of BTC + ETH stacked")
    print("=" * 120)

    fold_size = len(data) // 5
    for eth_thresh in [0.0002, 0.0003]:
        print(f"\n  --- BTC(0.0002) + ETH({eth_thresh}) stacked ---")
        fold_pnls = []
        for fold in range(5):
            s = fold * fold_size
            e = s + fold_size if fold < 4 else len(data)
            fd = data[s:e]
            pnl = 0; n = 0
            for d in fd:
                if d["pool"] < 2.0: continue
                sig_btc, str_btc = mtf_signal(d["btc_c"], (3,7,15), 0.0002)
                if sig_btc:
                    frac = min(0.03 + 100 * str_btc, 0.30)
                    bet = max(0.01, min(2.0, d["pool"] * frac))
                    pnl += settle(d["rnd"], bet, sig_btc); n += 1
                    continue
                if d["eth_c"] is None: continue
                sig_eth, str_eth = mtf_signal(d["eth_c"], (3,7,15), eth_thresh)
                if sig_eth:
                    frac = min(0.03 + 100 * str_eth, 0.30)
                    bet = max(0.01, min(2.0, d["pool"] * frac))
                    pnl += settle(d["rnd"], bet, sig_eth); n += 1

            p2k = pnl / len(fd) * 2000
            fold_pnls.append(p2k)
            print(f"    Fold {fold+1}: N={n} PnL={pnl:+.3f} /2k={p2k:+.3f}")

        avg = sum(fold_pnls) / 5
        pos = sum(1 for p in fold_pnls if p > 0)
        print(f"    => avg /2k={avg:+.3f} ({pos}/5 positive)")

    print("\nDone.")


if __name__ == "__main__":
    import sys
    if "--fetch" in sys.argv:
        fetch_eth_klines()
    elif "--analyze" in sys.argv:
        run_analysis()
    else:
        # Auto: fetch if needed, then analyze
        if not os.path.exists(ETH_PATH) or sum(1 for _ in Path(ETH_PATH).open()) < 30000:
            fetch_eth_klines()
        run_analysis()
