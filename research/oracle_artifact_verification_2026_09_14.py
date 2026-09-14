"""Re-derive the oracle-artifact findings from this repo's own data.

This is the script behind the "verified 2026-09-14" numbers in
docs/FINDINGS.md. It exists so a stranger can check the central claims
instead of trusting them: that the canonical signal never predicted its
own input assets, that most of its PancakeSwap win rate was Chainlink
oracle lag rather than real price movement, and that the live record
agrees with the backtest.

Run from the repo root, after `python run.py --sync` has brought the
stores past epoch 512024:

    python research/oracle_artifact_verification_2026_09_14.py

It first runs the full-tape canonical backtest to epoch 512022 (about
2.5 minutes, most of it loading ~4.5 GB of klines), then scores every bet
two ways and prints the tables.

HOW SPOT IS PRICED. A round's real on-chain lock is the executeRound tx
that STARTS the next round, so lock(e) = startAt(e+1) and
close(e) = startAt(e+2). Kline record k spans [startAt(k)-1s,
startAt(k)+298s], so spot at lock(e) is read from record e+1 and spot at
close(e) from record e+2, each at the exact second. This is deliberately
not the nominal +300s: a ~5s shift in the exit window was measured to move
the spot win rate by 2-3 points, so the anchor is the actual settlement
instant. OKX BNB-USDT trades in 0.1 USDT ticks (~1.4 bps), so a few
percent of bets tie on spot; ties are reported and excluded from spot
win rates.

"CL" below means scored the way PancakeSwap settles: the contract's own
lockPrice/closePrice, which come from its Chainlink oracle
(lockOracleId/closeOracleId in the rounds struct).

Expected output as of 2026-09-14 (store synced to ~515,700):
    bets=2015  CL WR=59.6%
    full tape:  CL 59.6%  spot 52.3%  BTC 49.8%  ETH 49.7%  SOL 49.5%
    pre  484409: CL 61.5%  spot 54.0%  BTC 50.6%  ETH 51.4%  SOL 50.4%
    last 90d:    CL 51.5%  spot 44.4%
    McNemar post-484409: 31 vs 10, premium 5.2 pts, z=3.3

The live-ledger section needs the operator's local ledger files, which
are not in git (var/ is gitignored); it is skipped when they are absent.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

END = 512022              # last epoch of the 2026-09-01 measurement
SPLIT = 484409            # the epoch the 2026-06-11 post-mortem called "the break"
LIVE_LEDGERS = (
    ("Jul-Aug 2026 deployment", REPO / "var/vm_handover_20260830/live/bets.jsonl"),
    ("May-Jun 2026 run", REPO / "var/live/bets.jsonl"),
)


def run_backtest(out_dir: Path) -> Path:
    from research.in_process_runner import FoldSpec, run_experiment
    run_experiment(
        experiment_specs=[FoldSpec(name="fulltape", kline_cutoff_seconds=2,
                                   epoch_start=None, epoch_end=END)],
        output_base_dir=out_dir,
    )
    return out_dir / "fulltape" / "trades.csv"


def load_rounds() -> dict[int, tuple[int, str | None]]:
    rounds = {}
    with open(REPO / "var/closed_rounds.jsonl", "rb") as f:
        for raw in f:
            o = json.loads(raw)
            e = int(o["epoch"])
            if e > END + 2:
                break
            rounds[e] = (int(o["startAt"]), o.get("position"))
    return rounds


def load_klines(sym: str, need: set[int]) -> dict[int, dict[int, float]]:
    out = {}
    with open(REPO / f"var/{sym}_spot_prices.jsonl", "rb") as f:
        for raw in f:
            i = raw.find(b'"epoch":')
            e = int(raw[i + 8:raw.find(b",", i)])
            if e in need:
                o = json.loads(raw)
                out[e] = {int(c[0]) // 1000: float(c[1]) for c in o["klines_1s"]}
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", help="reuse an existing trades.csv instead of re-running")
    args = ap.parse_args()

    trades = Path(args.trades) if args.trades else run_backtest(
        Path(tempfile.mkdtemp(prefix="pb_verify_")))

    rounds = load_rounds()
    bets = []
    with open(trades, newline="") as f:
        for r in csv.DictReader(f):
            if r["action"] == "BET":
                bets.append((int(r["epoch"]), r["direction"], float(r["profit_bnb"])))

    need = {e + d for e, _, _ in bets for d in (1, 2)}
    kl = {s: load_klines(s, need) for s in ("bnb", "btc", "eth", "sol")}

    def price(sym, rec, t):
        c = kl[sym].get(rec)
        if not c:
            return None
        for d in (0, 1, -1, 2, -2):
            if t + d in c:
                return c[t + d]
        return None

    def outcome(sym, e, direction):
        if e + 2 not in rounds:
            return None
        a = price(sym, e + 1, rounds[e + 1][0])
        b = price(sym, e + 2, rounds[e + 2][0])
        if a is None or b is None:
            return None
        if a == b:
            return "tie"
        return "win" if ((b > a) == (direction == "Bull")) else "loss"

    rows = [{
        "epoch": e, "dir": d, "t": rounds[e][0],
        "cl": "win" if d == rounds[e][1] else "loss",
        **{s: outcome(s, e, d) for s in ("bnb", "btc", "eth", "sol")},
    } for e, d, _ in bets]

    def wr(sub, key):
        s = [r for r in sub if r[key] in ("win", "loss")]
        k = sum(r[key] == "win" for r in s)
        return k, len(s), sum(r[key] == "tie" for r in sub)

    def table(label, sub):
        print(f"\n== {label}  (bets={len(sub)}) ==")
        for key, name in (("cl", "PancakeSwap/Chainlink"), ("bnb", "BNB real spot"),
                          ("btc", "BTC forward"), ("eth", "ETH forward"),
                          ("sol", "SOL forward")):
            k, n, ties = wr(sub, key)
            lo, hi = wilson(k, n)
            print(f"  {name:>22}: {100*k/n:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]  "
                  f"n={n} ties={ties}")

    def mcnemar(label, sub):
        s = [r for r in sub if r["bnb"] in ("win", "loss")]
        b = sum(r["cl"] == "win" and r["bnb"] == "loss" for r in s)
        c = sum(r["cl"] == "loss" and r["bnb"] == "win" for r in s)
        prem = (sum(r["cl"] == "win" for r in s) - sum(r["bnb"] == "win" for r in s)) / len(s)
        print(f"  McNemar {label}: won-on-Chainlink/lost-on-spot={b}  reverse={c}  "
              f"premium={100*prem:.1f} pts  z={(b-c)/math.sqrt(b+c):.1f}  n={len(s)}")

    def ts(y, m, d):
        return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())

    end_t = ts(2026, 9, 1)
    pre = [r for r in rows if r["epoch"] < SPLIT]
    post = [r for r in rows if r["epoch"] >= SPLIT]
    print(f"bets={len(rows)}  CL WR={100*sum(r['cl']=='win' for r in rows)/len(rows):.1f}%")
    table("FULL TAPE", rows)
    table(f"PRE epoch<{SPLIT}", pre)
    table(f"POST epoch>={SPLIT}", post)
    table("LAST 180 DAYS", [r for r in rows if r["t"] >= end_t - 180 * 86400])
    table("LAST 90 DAYS", [r for r in rows if r["t"] >= end_t - 90 * 86400])
    print()
    mcnemar("full", rows)
    mcnemar("pre ", pre)
    mcnemar("post", post)

    print("\n== 60-day trailing win rate ==")
    for y, m, d in [(2026, 2, 15), (2026, 3, 15), (2026, 4, 15), (2026, 5, 15),
                    (2026, 6, 1), (2026, 6, 15), (2026, 7, 1), (2026, 7, 15),
                    (2026, 8, 1), (2026, 8, 15), (2026, 9, 1)]:
        t1 = ts(y, m, d)
        sub = [r for r in rows if t1 - 60 * 86400 <= r["t"] < t1]
        k, n, _ = wr(sub, "bnb")
        kc, nc, _ = wr(sub, "cl")
        print(f"  to {y}-{m:02d}-{d:02d}:  spot {100*k/n:5.1f}%  Chainlink {100*kc/nc:5.1f}%  (n={nc})")

    bt = {r["epoch"]: r for r in rows}
    for label, path in LIVE_LEDGERS:
        if not path.exists():
            print(f"\n== live: {label} -- ledger not present, skipped ==")
            continue
        term = {}
        for line in open(path):
            if line.strip():
                o = json.loads(line)
                term.setdefault(int(o["epoch"]), {}).update(o)
        won = {e for e, o in term.items() if o["status"] in ("SETTLED_WON", "CLAIMED")}
        lost = {e for e, o in term.items() if o["status"] == "SETTLED_LOST"}
        settled = won | lost
        inter = [e for e in settled if e in bt]
        agree = sum(term[e].get("side") == bt[e]["dir"] for e in inter)
        btw = sum(bt[e]["cl"] == "win" for e in inter)
        print(f"\n== live: {label} ==")
        print(f"  settled {len(settled)}  won {len(won)}  WR {100*len(won)/len(settled):.1f}%")
        print(f"  in backtest {len(inter)}/{len(settled)}  side agreement {agree}/{len(inter)}  "
              f"backtest win rate at those epochs {100*btw/len(inter):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
