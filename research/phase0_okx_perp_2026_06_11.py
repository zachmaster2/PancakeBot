"""Phase-0 OKX perp probe — signal tests on captured funding/OI/trade-tape.

Data (captured by research/phase0_okx_perp_capture_2026_06_11.py into
var/extended/, gitignored; capture script is the provenance):
  okx_bnb_swap_funding.jsonl            8h funding, 2026-03-11 -> now
  okx_bnb_oi_1d.jsonl                   daily OI, 2025-12-14 -> now
  okx_swap_trades_BNB-USDT-SWAP.jsonl   perp tape walk-back -> 2026-05-10
  okx_trades_BNB-USDT_archive.jsonl     spot tape 2026-02-25 -> 2026-05-01
  okx_trades_BNB-USDT_gap.jsonl         spot tape gap-fill 2026-05-01 -> now
The script is coverage-aware: every cell reports n from rounds where the
signal is defined, and findings.json records per-source coverage, so a run
against a partially-captured tape is self-describing (final committed run
uses the completed capture).

PRE-REGISTERED PRIMARY (pinned before the first run of this script):
  Perp (BNB-USDT-SWAP) taker-flow imbalance over the 5-minute window ending
  at lock-2s, sign-follow (net taker buying -> Bull), DEAD era
  (484409..488832).  One permutation test, reported without sweep discount.
ALL OTHER CELLS ARE EXPLORATORY and are Sidak-adjusted over the full
examined-cell count (every era x candidate x variant cell computed below,
including the primary's own row).

Candidates (from the dispatch; letters per dispatch):
  (a) funding level + sign      last SETTLED rate at lock (no look-ahead;
                                the intra-period "current" rate is not
                                backfillable -> runtime uses the same
                                last-settled value, so the backtest feature
                                matches what the bot could compute)
  (b) funding delta             last settled minus previous settled
  (e) tape imbalance            signed taker notional / total notional over
                                1m/5m/15m windows ending at lock-2s
                                (primary; mirrors the kline cutoff
                                envelope) and lock-6s (preflight-only
                                sensitivity), perp + spot separately
  (f) OI change                 1-day relative change, last known day at
                                lock (clustered by day)
  (c)(d) order-book imbalance/depth and (g) liquidations: NOT TESTABLE —
  no historical depth exists (forward-capture only; see findings doc).

Statistics:
  - Settlement: flat stake 1, REALIZED final-pool payouts, 3% fee, no gas
    (infinitesimal-stake limit; era-relative comparisons valid).
  - Funding/OI cells use CLUSTERED z (cluster = funding period / day):
    one 8h funding value spans ~96 rounds; per-round z would overstate
    independence ~10x.  Tape imbalance varies per round -> per-round z.
  - Permutation nulls (seeded, vectorized, adaptive 1k->10k iters) on
    EVERY cell: cluster-level shuffles for funding/OI, per-round shuffles
    for tape.  A sign-strategy's null expectation is the structural
    fee + majority-following discount (~-0.03/bet), NOT zero, so the z
    column is never a discovery metric; effect size = deficit_vs_null
    (obs mean pnl minus permutation null mean), significance = the
    permutation p.
  - Sidak over N_examined cells on the best dead_all two-sided
    permutation p for any exploratory "discovery" (conservative under
    the cells' positive dependence).

Eras: golden 437562..479952, fade 479953..484408, dead 484409..488832
(+ within-dead split latest/vm_live as a consistency check).  Coverage
truncates some cells (funding starts 2026-03-11 = late golden; perp tape
starts 2026-05-10 = fade onward; spot archive covers 2026-02-25..05-01).

Runtime feasibility (VM-measured 2026-06-11, all OKX public endpoints
225-245ms from Frankfurt):
  funding (current+next): one GET at preflight wake, value constant for
    the whole round -> TRIVIAL.
  OI (current): one GET at preflight -> TRIVIAL.
  tape imbalance: /api/v5/market/trades returns the newest <=500 trades;
    at observed dead-era density (~90-130k trades/day, bursts >>100/min)
    500 trades does NOT always cover 15m, so runtime = INCREMENTAL
    ACCUMULATION: fetch newest page each preflight wake (one GET/round,
    ~1.5MB/day), dedup by tradeId into a rolling local tape (same pattern
    as pool-event accumulation).  Window then closes at lock-2s using the
    decision-wake fetch alongside the existing kline fetch -> same
    envelope as the canonical cutoff=2 path, no new deadline -> TRIVIAL
    (bandwidth-aware).  The lock-6s variant needs only the preflight
    fetch.

Outputs: var/strategy_review/phase0_okx_perp_2026_06_11/{findings.json,
cells.csv} + console digest.
Findings doc: research/phase0_okx_perp_2026_06_11_findings.md.

Run:  cd <repo> && .venv/Scripts/python.exe research/phase0_okx_perp_2026_06_11.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.in_process_runner as ipr  # noqa: E402
from pancakebot.constants import BNB_WEI  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402

EXT = REPO / "var" / "extended"
OUT = REPO / "var" / "strategy_review" / "phase0_okx_perp_2026_06_11"
FEE = 0.03
SEED = 20260612

ERAS = [
    ("golden", 437562, 479952),
    ("fade_era", 479953, 484408),
    ("dead_all", 484409, 488832),
    ("dead_latest", 484409, 487686),
    ("dead_vmlive", 487687, 488832),
]
TAPE_WINDOWS_S = (60, 300, 900)
TAPE_CUTOFFS_S = (2, 6)        # 2 = kline-envelope primary, 6 = preflight-only
MIN_TRADES = 10                # window must contain >= this many trades

PRIMARY = dict(candidate="tape_imb", instrument="perp", window_s=300,
               cutoff_s=2, era="dead_all", direction="sign_follow",
               stat="p_upper")  # follow direction pinned: the ONLY primary
                                # statistic is p_upper; reading p_lower (a
                                # fade) re-enters the exploratory sweep


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_rounds():
    rounds = ipr._load_all_rounds(use_extended_data=False)
    rows = []
    for r in rounds:
        if r.epoch < ERAS[0][1] or r.position not in ("Bull", "Bear"):
            continue
        pools = compute_pool_amounts_wei(bets=r.bets)
        f_bull = pools.bull_wei / BNB_WEI
        f_bear = pools.bear_wei / BNB_WEI
        if f_bull <= 0 or f_bear <= 0:
            continue
        tot = f_bull + f_bear
        rows.append(dict(
            epoch=int(r.epoch), lock=int(r.lock_at),
            outcome_bull=r.position == "Bull",
            payout_bull=tot * (1 - FEE) / f_bull,
            payout_bear=tot * (1 - FEE) / f_bear,
        ))
    rows.sort(key=lambda x: x["epoch"])
    return rows


def load_funding():
    """Ascending [(settle_ts_s, realized_rate)]."""
    out = []
    with open(EXT / "okx_bnb_swap_funding.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append((int(d["fundingTime"]) / 1000.0, float(d["realizedRate"])))
    out.sort()
    return out


def load_oi():
    """Ascending [(day_ts_s, oi)]."""
    out = []
    with open(EXT / "okx_bnb_oi_1d.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append((int(d[0]) / 1000.0, float(d[1])))
    out.sort()
    return out


def load_tape(paths: list[Path]):
    """Merge walk-back jsonl files -> ascending arrays + notional cumsums.

    Dedup by tradeId (archive/gap-fill overlap near 2026-05-01).  Coverage
    is tracked as PER-FILE [lo, hi] segments merged where they overlap: a
    window is valid only when fully inside ONE merged segment.  Global
    min/max coverage would mask an interior hole (e.g. a partially-walked
    gap-fill file alongside the archive) and let hole-edge-straddling
    windows return truncated, biased imbalances (adversarial-review
    finding 2026-06-12)."""
    ts, signed, total, tids, segments = [], [], [], [], []
    for p in paths:
        if not p.exists():
            continue
        file_lo, file_hi = None, None
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                notional = float(d["px"]) * float(d["sz"])
                t = int(d["ts"]) / 1000.0
                ts.append(t)
                signed.append(notional if d["side"] == "buy" else -notional)
                total.append(notional)
                tids.append(int(d["tradeId"]))
                file_lo = t if file_lo is None else min(file_lo, t)
                file_hi = t if file_hi is None else max(file_hi, t)
        if file_lo is not None:
            segments.append([file_lo, file_hi])
    if not ts:
        return None
    segments.sort()
    merged = [segments[0]]
    for lo, hi in segments[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    tids = np.asarray(tids, dtype=np.int64)
    _, keep = np.unique(tids, return_index=True)
    ts = np.asarray(ts)[keep]
    signed = np.asarray(signed)[keep]
    total = np.asarray(total)[keep]
    order = np.argsort(ts, kind="stable")
    ts, signed, total = ts[order], signed[order], total[order]
    return dict(
        ts=ts,
        signed_cum=np.concatenate([[0.0], np.cumsum(signed)]),
        total_cum=np.concatenate([[0.0], np.cumsum(total)]),
        segments=[(float(a), float(b)) for a, b in merged],
        cover_lo=float(ts[0]), cover_hi=float(ts[-1]), n=int(len(ts)),
    )


def tape_imbalance(tape, t_end: float, window_s: int):
    """Signed/total taker notional in [t_end-window, t_end]; None if sparse
    or the window is not fully inside ONE contiguous covered segment."""
    if tape is None:
        return None
    t_start = t_end - window_s
    if not any(lo <= t_start and t_end <= hi for lo, hi in tape["segments"]):
        return None
    i0 = int(np.searchsorted(tape["ts"], t_start, side="left"))
    i1 = int(np.searchsorted(tape["ts"], t_end, side="right"))
    if i1 - i0 < MIN_TRADES:
        return None
    tot = tape["total_cum"][i1] - tape["total_cum"][i0]
    if tot <= 0:
        return None
    return float((tape["signed_cum"][i1] - tape["signed_cum"][i0]) / tot)


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------

def sign_strategy(rows, sig_key):
    """Bet Bull when signal > 0, Bear when < 0; realized payouts."""
    out = []
    for x in rows:
        s = x.get(sig_key)
        if s is None or s == 0:
            continue
        bull = s > 0
        win = bull == x["outcome_bull"]
        pay = x["payout_bull"] if bull else x["payout_bear"]
        out.append(dict(epoch=x["epoch"], win=win,
                        pnl=(pay - 1.0) if win else -1.0,
                        cluster=x.get(sig_key + "_cluster", x["epoch"]),
                        outcome_bull=x["outcome_bull"],
                        payout_bull=x["payout_bull"],
                        payout_bear=x["payout_bear"], side_bull=bull))
    return out


def cell_stats(bets, clustered: bool):
    n = len(bets)
    if n < 5:
        return dict(n=n)
    pnls = np.array([b["pnl"] for b in bets])
    wr = float(np.mean([b["win"] for b in bets]))
    mean = float(pnls.mean())
    if clustered:
        groups = {}
        for b in bets:
            groups.setdefault(b["cluster"], []).append(b["pnl"])
        cmeans = np.array([np.mean(v) for v in groups.values()])
        k = len(cmeans)
        se = float(cmeans.std(ddof=1) / math.sqrt(k)) if k > 1 else 0.0
        z = float(cmeans.mean() / se) if se > 0 else None
        return dict(n=n, n_clusters=k, wr=round(wr, 4), mean_pnl=round(mean, 4),
                    z=round(z, 2) if z is not None else None)
    se = float(pnls.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = float(mean / se) if se > 0 else None
    return dict(n=n, wr=round(wr, 4), mean_pnl=round(mean, 4),
                z=round(z, 2) if z is not None else None)


def signal_corr(rows, sig_key):
    """Point-biserial r between signal value and Bull outcome."""
    pairs = [(x[sig_key], 1.0 if x["outcome_bull"] else 0.0)
             for x in rows if x.get(sig_key) is not None]
    if len(pairs) < 30:
        return dict(n=len(pairs))
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if a.std() == 0 or b.std() == 0:
        return dict(n=len(pairs), r=None)
    return dict(n=len(pairs), r=round(float(np.corrcoef(a, b)[0, 1]), 4))


def permutation_p(bets, clustered: bool, n_iter=1000, seed=SEED):
    """Vectorized permutation null -> p_upper/p_lower vs the STRUCTURAL null.

    The sign-strategy's expectation under 'no signal' is NOT zero — it is
    the fee + majority-following discount (~-0.03/bet).  z-vs-zero is
    therefore meaningless for discovery; only the permutation p is honest.
    Unclustered: shuffle (outcome, payouts) jointly across rounds, sides
    fixed (same null as the repo's prior phase-0 scripts).  Clustered:
    shuffle the per-cluster side assignment (signal is cluster-constant),
    outcomes fixed.  Adaptive: re-run at 10k iters when min(p) < 0.01 so
    Sidak over ~40 cells has resolution."""
    if len(bets) < 20:
        return None
    obs = float(np.mean([b["pnl"] for b in bets]))
    out = np.array([b["outcome_bull"] for b in bets])
    pb = np.array([b["payout_bull"] for b in bets])
    pr = np.array([b["payout_bear"] for b in bets])
    side = np.array([b["side_bull"] for b in bets])

    def run(iters: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        null = np.empty(iters)
        if clustered:
            clusters = sorted({b["cluster"] for b in bets})
            c_index = {c: i for i, c in enumerate(clusters)}
            cidx = np.array([c_index[b["cluster"]] for b in bets])
            cs = np.empty(len(clusters), dtype=bool)
            for b in bets:
                cs[c_index[b["cluster"]]] = b["side_bull"]
            for i in range(iters):
                s = cs[rng.permutation(len(cs))][cidx]
                pay = np.where(s, pb, pr)
                null[i] = np.where(out == s, pay - 1.0, -1.0).mean()
        else:
            for i in range(iters):
                p = rng.permutation(len(out))
                pay = np.where(side, pb[p], pr[p])
                null[i] = np.where(out[p] == side, pay - 1.0, -1.0).mean()
        return null

    null = run(n_iter)
    p_up = float((null >= obs).mean())
    p_lo = float((null <= obs).mean())
    if min(p_up, p_lo) < 0.01:
        null = run(10_000)
        p_up = float((null >= obs).mean())
        p_lo = float((null <= obs).mean())
    return dict(n=len(bets), obs_mean_pnl=round(obs, 4),
                null_mean=round(float(null.mean()), 4),
                deficit_vs_null=round(obs - float(null.mean()), 4),
                p_upper=round(p_up, 4), p_lower=round(p_lo, 4),
                n_iter=len(null))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("--- loading rounds ---", flush=True)
    rounds = load_rounds()
    print(f"  {len(rounds)} rounds [{rounds[0]['epoch']}..{rounds[-1]['epoch']}]",
          flush=True)

    print("--- loading funding/oi ---", flush=True)
    funding = load_funding()
    oi = load_oi()
    f_ts = np.array([x[0] for x in funding])
    f_rate = np.array([x[1] for x in funding])
    o_ts = np.array([x[0] for x in oi])
    o_val = np.array([x[1] for x in oi])

    print("--- loading tapes ---", flush=True)
    tapes = {
        "perp": load_tape([EXT / "okx_swap_trades_BNB-USDT-SWAP.jsonl"]),
        "spot": load_tape([EXT / "okx_trades_BNB-USDT_archive.jsonl",
                           EXT / "okx_trades_BNB-USDT_gap.jsonl"]),
    }
    coverage = {}
    for k, tp in tapes.items():
        if tp:
            segs = [(time.strftime("%Y-%m-%d %H:%M", time.gmtime(a)),
                     time.strftime("%Y-%m-%d %H:%M", time.gmtime(b)))
                    for a, b in tp["segments"]]
            coverage[k] = dict(n_trades=tp["n"], segments=segs)
            print(f"  {k}: {tp['n']} trades, segments={segs}", flush=True)
            if len(segs) > 1:
                print(f"  {k}: WARNING — interior hole(s); windows inside a "
                      f"hole or straddling its edges return None", flush=True)
        else:
            coverage[k] = None
            print(f"  {k}: MISSING", flush=True)

    print("--- attaching features ---", flush=True)
    for x in rounds:
        lock = x["lock"]
        # (a)/(b) funding: last SETTLED value at lock + delta vs previous
        fi = int(np.searchsorted(f_ts, lock, side="right")) - 1
        if fi >= 1:
            x["funding"] = float(f_rate[fi])
            x["funding_cluster"] = fi
            x["funding_delta"] = float(f_rate[fi] - f_rate[fi - 1])
            x["funding_delta_cluster"] = fi
        # (f) OI: 1-day relative change, last known day at lock
        oj = int(np.searchsorted(o_ts, lock, side="right")) - 1
        if oj >= 1 and o_val[oj - 1] > 0:
            x["oi_d1"] = float(o_val[oj] / o_val[oj - 1] - 1.0)
            x["oi_d1_cluster"] = oj
        # (e) tape imbalance
        for inst in ("perp", "spot"):
            for w in TAPE_WINDOWS_S:
                for c in TAPE_CUTOFFS_S:
                    x[f"imb_{inst}_{w}_{c}"] = tape_imbalance(
                        tapes[inst], lock - c, w)

    candidates = (
        [("funding", True), ("funding_delta", True), ("oi_d1", True)]
        + [(f"imb_{inst}_{w}_{c}", False)
           for inst in ("perp", "spot")
           for w in TAPE_WINDOWS_S for c in TAPE_CUTOFFS_S]
    )

    print("--- cells ---", flush=True)
    cells = []
    bets_by_cell = {}
    for era, lo, hi in ERAS:
        sub = [x for x in rounds if lo <= x["epoch"] <= hi]
        for key, clustered in candidates:
            bets = sign_strategy(sub, key)
            st = cell_stats(bets, clustered)
            rc = signal_corr(sub, key)
            cell = dict(era=era, candidate=key, clustered=clustered,
                        r=rc.get("r"), r_n=rc.get("n"), **st)
            cells.append(cell)
            bets_by_cell[(era, key)] = (bets, clustered)

    # examined-cell count: every era x candidate cell with n >= 5
    examined = [c for c in cells if c.get("n", 0) >= 5]
    n_examined = len(examined)

    # permutation for EVERY examined cell (the z column is vs zero and is
    # NOT a discovery metric — the null expectation is the structural
    # fee + majority-following discount; deficit_vs_null + p are honest)
    print("--- permutations (all cells) ---", flush=True)
    pk = f"imb_{PRIMARY['instrument']}_{PRIMARY['window_s']}_{PRIMARY['cutoff_s']}"
    perms = {}
    for c in examined:
        bets, clustered = bets_by_cell.get((c["era"], c["candidate"]), ([], False))
        p = permutation_p(bets, clustered)
        if p:
            perms[f"{c['era']}/{c['candidate']}"] = p
            c["deficit_vs_null"] = p["deficit_vs_null"]
            c["p_upper"] = p["p_upper"]
            c["p_lower"] = p["p_lower"]

    # Sidak: best two-sided permutation p among dead_all cells (the
    # deployment-relevant discovery domain), discounted over ALL examined
    # cells (conservative under the cells' positive dependence)
    best = None
    for c in examined:
        if c["era"] != "dead_all" or "p_upper" not in c:
            continue
        p2 = min(1.0, 2 * min(c["p_upper"], c["p_lower"]))
        if best is None or p2 < best["p_two_sided"]:
            best = dict(cell=f"{c['era']}/{c['candidate']}",
                        deficit_vs_null=c["deficit_vs_null"], p_two_sided=p2)
    sidak = None
    if best:
        sidak = dict(best_cell=best["cell"],
                     deficit_vs_null=best["deficit_vs_null"],
                     p_raw=round(best["p_two_sided"], 4),
                     n_examined=n_examined,
                     p_sidak=round(1 - (1 - best["p_two_sided"]) ** n_examined, 4))

    findings = dict(
        coverage=dict(tapes=coverage,
                      funding=dict(n=len(funding),
                                   lo=time.strftime("%Y-%m-%d", time.gmtime(funding[0][0])),
                                   hi=time.strftime("%Y-%m-%d", time.gmtime(funding[-1][0]))),
                      oi=dict(n=len(oi),
                              lo=time.strftime("%Y-%m-%d", time.gmtime(oi[0][0])),
                              hi=time.strftime("%Y-%m-%d", time.gmtime(oi[-1][0])))),
        pre_registered_primary=dict(**PRIMARY, key=pk),
        n_examined_cells=n_examined,
        cells=cells,
        permutations=perms,
        sidak_best_dead_cell=sidak,
    )
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2),
                                       encoding="utf-8")
    with open(OUT / "cells.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for c in cells for k in c})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(cells)

    # console digest
    print(f"\n=== pre-registered primary: {PRIMARY['era']}/{pk} ===")
    print(json.dumps(perms.get(f"{PRIMARY['era']}/{pk}"), indent=1))
    print("\n=== all cells (deficit = obs mean pnl - permutation null mean) ===")
    for c in cells:
        if c.get("n", 0) >= 5:
            print(f"  {c['era']:>12} {c['candidate']:>16}: n={c['n']:>6} "
                  f"wr={c.get('wr')} pnl={c.get('mean_pnl')} "
                  f"deficit={c.get('deficit_vs_null')} "
                  f"p_up={c.get('p_upper')} p_lo={c.get('p_lower')} r={c.get('r')}")
    print("\n=== sidak (best dead_all cell over all examined) ===")
    print(json.dumps(sidak, indent=1))
    print(f"\n[done] {time.time()-t0:.0f}s; artifacts in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
