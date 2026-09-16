"""Confirmation run for the A3 pre-registration (docs/prereg_A3_gap_persistence_2026_09_15.md).

RUN THIS ON OR AFTER 2027-04-08, ONCE. It refuses to run earlier without an explicit override,
because a single look on a fixed date is the whole design.

    python research/prereg_A3_confirm_2026_09_15.py

What it does, with no other context required:
  1. reads var/closed_rounds.jsonl and var/bnb_spot_prices.jsonl (the daily sync's stores);
  2. fetches, read-only, the Chainlink BNB/USD oracle rounds it needs from public BSC RPC, caching
     them in var/prereg_A3_oracle_cache.json (resumable; no keys, no writes on chain);
  3. scores ONE rule on rounds after epoch 515944 and prints one of three verdicts.

The rule: among rounds where the lock price is already published at the decision instant and OKX
spot has moved at least 2 bps away from it, bet the side spot moved to, but only when at least
82.3232% of the 1-second moves since that price was published pointed the same way (A3 >= 0.823232).

Everything else about why is in the registration document. This file is the canonical
implementation; the original scratchpad script it was derived from will not survive.
"""
from __future__ import annotations
import argparse, datetime as dt, json, math, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "var", "prereg_A3_oracle_cache.json")
PRED = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"      # PancakeSwap Prediction V2 (BNB)
AGG = "0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE"       # Chainlink BNB/USD on BSC
SEL_ROUNDS = "8c65c81f"                                   # rounds(uint256)
SEL_GRD = "9a6fc8f5"                                      # getRoundData(uint80)
EPS = ["https://bsc-rpc.publicnode.com", "https://bsc-dataseed.bnbchain.org",
       "https://bsc-dataseed1.defibit.io", "https://bsc-dataseed1.binance.org"]
FLOOR = 515944            # store end on 2026-09-15; the confirmation set is epochs ABOVE this
CUT = 0.823232            # A3 top-tercile cut-point, frozen from discovery 2026-09-14
STAKE, FEE, LAT, GMIN = 0.05, 0.03, 2, 2.0
ANALYSIS_DATE = dt.date(2027, 4, 8)
MIN_N = 2600              # fewer rule bets than this => INCONCLUSIVE (below this a true null cannot be excluded)
KILL_UPPER = 0.026        # upper bound below this => KILL (excludes the claimed effect; see doc section 5)


def rpc_batch(sess, calls):
    import requests  # noqa: F401  (imported lazily so --help works without it)
    payload = [{"jsonrpc": "2.0", "id": i, "method": "eth_call",
                "params": [{"to": to, "data": "0x" + data}, "latest"]} for i, (to, data) in enumerate(calls)]
    for attempt in range(14):
        url = EPS[rpc_batch.ep % len(EPS)]
        try:
            r = sess.post(url, json=payload, timeout=30); r.raise_for_status(); out = r.json()
            if isinstance(out, dict):
                raise RuntimeError(str(out)[:150])
            byid = {int(o["id"]): o for o in out if o.get("id") is not None}
            got = []
            for i in range(len(calls)):
                o = byid.get(i); v = o.get("result") if o else None
                if v and v != "0x":
                    got.append(v)
                elif o and "error" in o and "revert" in str(o["error"]).lower():
                    got.append("REVERT")
                else:
                    raise RuntimeError("transient per-item error")
            return got
        except Exception as e:
            rpc_batch.ep += 1; time.sleep(min(30, 1.5 * (attempt + 1)))
    raise SystemExit("public RPC failed repeatedly; try again later")
rpc_batch.ep = 0


def words(h):
    h = h.removeprefix("0x"); return [int(h[i:i + 64], 16) for i in range(0, len(h), 64)]


def s256(x):
    return x - (1 << 256) if x >= (1 << 255) else x


def load_store():
    rounds, starts = {}, {}
    with open(os.path.join(REPO, "var", "closed_rounds.jsonl"), "rb") as f:
        for raw in f:
            j = raw.find(b'"epoch":'); e = int(raw[j + 8:raw.find(b",", j)])
            if e > FLOOR - 3:
                o = json.loads(raw); starts[e] = o["startAt"]
                if e > FLOOR:
                    rounds[e] = o
    return rounds, starts


def fetch_oracle(rounds, starts):
    import requests
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {"rounds": {}, "oracle": {}}
    sess = requests.Session()
    need_r = [e for e in sorted(rounds) if e + 2 in starts and str(e) not in cache["rounds"]]
    print(f"[fetch] rounds() to fetch: {len(need_r)}")
    for k in range(0, len(need_r), 20):
        chunk = need_r[k:k + 20]
        for e, h in zip(chunk, rpc_batch(sess, [(PRED, SEL_ROUNDS + format(e, "064x")) for e in chunk])):
            if h != "REVERT":
                w = words(h); cache["rounds"][str(e)] = [w[6], w[7], s256(w[4]), s256(w[5])]
        if k % 1000 == 0:
            json.dump(cache, open(CACHE, "w")); print(f"  rounds {k}/{len(need_r)}", flush=True)
    json.dump(cache, open(CACHE, "w"))
    need_o = set()
    for e in rounds:
        r = cache["rounds"].get(str(e))
        if r and r[0] > 0:
            for d in (0, -1, -2):
                o = r[0] + d
                if o > 0 and (o >> 64) >= 1:
                    need_o.add(o)
    todo = sorted(o for o in need_o if str(o) not in cache["oracle"])
    print(f"[fetch] oracle rounds to fetch: {len(todo)}")
    for k in range(0, len(todo), 20):
        chunk = todo[k:k + 20]
        for o, h in zip(chunk, rpc_batch(sess, [(AGG, SEL_GRD + format(o, "064x")) for o in chunk])):
            cache["oracle"][str(o)] = None if h == "REVERT" else [s256(words(h)[1]), words(h)[3]]
        if k % 1000 == 0:
            json.dump(cache, open(CACHE, "w")); print(f"  oracle {k}/{len(todo)}", flush=True)
    json.dump(cache, open(CACHE, "w"))
    return cache


def bps(a, b):
    return 1e4 * math.log(a / b)


def build(rounds, starts, cache):
    R = {int(k): v for k, v in cache["rounds"].items() if v and v[0] > 0}
    O = {int(k): v for k, v in cache["oracle"].items() if v}
    info = {}
    for e, o in rounds.items():
        if e not in R or e + 2 not in starts or o.get("failed"):
            continue
        d = o["startAt"] + 298
        best = None
        for k in (0, -1, -2):
            v = O.get(R[e][0] + k)
            if v and v[1] <= d and (best is None or v[1] > best[1]):
                best = (v[0] / 1e8, v[1])
        if best:
            info[e] = {"d": d, "P": best[0], "u": best[1], "age": d - best[1]}
    need = set(info)
    kl = {}
    with open(os.path.join(REPO, "var", "bnb_spot_prices.jsonl"), "rb") as f:
        for raw in f:
            j = raw.find(b'"epoch":'); e = int(raw[j + 8:raw.find(b",", j)])
            if e in need:
                kl[e] = {int(c[0]) // 1000: float(c[1]) for c in json.loads(raw)["klines_1s"]}
    offs, rows = {}, []
    by_day = {}
    for e, i in info.items():
        px = kl.get(e, {})
        if i["u"] - LAT in px:
            by_day.setdefault(i["u"] // 86400, []).append(bps(i["P"], px[i["u"] - LAT]))
    for k, v in by_day.items():
        v = sorted(v)
        if len(v) >= 5:
            offs[k] = v[len(v) // 2]
    allo = sorted(offs.values()); gmed = allo[len(allo) // 2] if allo else 0.0
    for e, i in sorted(info.items()):
        px = kl.get(e, {})
        if i["d"] not in px or i["age"] > 24:
            continue
        off = offs.get(i["d"] // 86400 - 1, gmed)
        g = bps(px[i["d"]], i["P"]) + off
        if abs(g) < GMIN:
            continue
        ups = dns = 0
        for t in range(i["u"], i["d"]):
            a, b = px.get(t), px.get(t + 1)
            if a is None or b is None:
                continue
            if b > a: ups += 1
            elif b < a: dns += 1
        nz = ups + dns
        a3 = ((ups if g > 0 else dns) / nz) if nz else 0.5
        side = "Bull" if g > 0 else "Bear"
        bets = rounds[e].get("bets") or ()
        T = sum(b["amountWei"] for b in bets) / 1e18
        F = sum(b["amountWei"] for b in bets if b["position"] == side) / 1e18
        won = rounds[e].get("position") == side           # house/tie rounds count as a loss
        ev = ((T + STAKE) * (1 - FEE) / (F + STAKE) - 1) if won else -1.0
        rows.append({"e": e, "day": rounds[e]["startAt"] // 86400, "a3": a3, "ev": ev, "won": won,
                     "T": T, "side": side, "bets": bets, "pos": rounds[e].get("position")})
    return rows


def boot(rows, nb=10000, seed=20270408):
    import random
    rnd = random.Random(seed)
    days = {}
    for r in rows:
        days.setdefault(r["day"], []).append(r["ev"])
    keys = list(days)
    means = []
    for _ in range(nb):
        vals = []
        for _ in keys:
            vals.extend(days[rnd.choice(keys)])
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.05 * nb)], means[int(0.95 * nb)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-early", action="store_true",
                    help="run before the registered analysis date (this BREAKS the registration)")
    a = ap.parse_args()
    today = dt.date.today()
    if today < ANALYSIS_DATE and not a.force_early:
        print(f"Registered analysis date is {ANALYSIS_DATE}; today is {today}.\n"
              f"This is a one-look design: running early and looking at the number destroys it.\n"
              f"If you genuinely need a dry run, use --force-early and DISCARD the output.")
        return 1
    rounds, starts = load_store()
    if not rounds:
        print(f"No rounds after epoch {FLOOR} in var/closed_rounds.jsonl.\n"
              f"The confirmation set only exists if the daily sync kept running. If it stopped, "
              f"the verdict is INCONCLUSIVE (insufficient data) and the registration says to stop.")
        return 1
    cache = fetch_oracle(rounds, starts)
    rows = build(rounds, starts, cache)
    sel = [r for r in rows if r["a3"] >= CUT]
    n = len(sel)
    base = sum(r["ev"] for r in rows) / len(rows) if rows else float("nan")
    print(f"\npopulation (qualifying rounds after {FLOOR}): {len(rows)}   rule bets (A3 >= {CUT}): {n}")
    print(f"population baseline, gap side every round: {base:+.4f} per unit")
    if n < MIN_N:
        print(f"\nVERDICT: INCONCLUSIVE (only {n} rule bets, registration requires {MIN_N}).")
        print("Registration says: record UNRESOLVED and stop. No extension, no re-analysis.")
        return 0
    ev = sum(r["ev"] for r in sel) / n
    lo, hi = boot(sel)
    wr = 100 * sum(r["won"] for r in sel) / n
    print(f"rule: n={n}  win rate {wr:.1f}%  MEAN RETURN {ev:+.4f} per unit   one-sided 95% bounds [{lo:+.4f}, {hi:+.4f}]")
    bot = [r for r in rows if r["a3"] <= 0.5]
    if bot:
        print(f"control, A3 bottom tercile (<=0.5): n={len(bot)} mean {sum(r['ev'] for r in bot)/len(bot):+.4f} (expected worse than baseline)")
    import random
    rnd = random.Random(7); rev = []
    for r in rows:
        sd = "Bull" if rnd.random() < 0.5 else "Bear"
        F = sum(b["amountWei"] for b in r["bets"] if b["position"] == sd) / 1e18
        rev.append(((r["T"] + STAKE) * (1 - FEE) / (F + STAKE) - 1) if r["pos"] == sd else -1.0)
    print(f"control, random side: {sum(rev)/len(rev):+.4f} (expected about -0.05)")
    if lo > 0:
        v = "PROMOTE (lower bound clears zero)"
    elif ev <= 0 or hi < KILL_UPPER:
        v = "KILL (point estimate at or below zero, or upper bound below +0.026)"
    else:
        v = "INCONCLUSIVE"
    print(f"\nVERDICT: {v}")
    print("PROMOTE authorises nothing to be deployed: it is a research verdict about a mean return.")
    print("INCONCLUSIVE means record UNRESOLVED and stop; no extension, no re-analysis of these rounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
