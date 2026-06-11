"""Phase-0 candidate harness WITH runtime feasibility as a first-class output.

Evaluates the post-mortem's research candidates (see
research/post_mortem_2026_06_11_findings.md) on the staged dataset, and
classifies every pool-state horizon against the Era 12b runtime envelope so
no candidate is surfaced that cannot actually execute at decision time.

Candidates tested here:
  C1  pool-horizon sweep + delta-late-flow (h = lock-6s .. lock-1s + final)
  C2  payout-aware contrarian at the TRIVIALLY-feasible horizon (lock-6s)
  C4  bulk-momentum re-map (mild impulses below the gate threshold)
  C3  wallet smart-money: runtime-cost note only (needs its own harness)

Runtime envelope (Era 12b, measured constants from pancakebot/timing_constants.py
+ chain/rpc_poller.py):
  event at (lock-h) lands in a block by +BSC_BLOCK_TIME_MS (450, worst one
  full block), is RPC-visible +RPC_BLOCK_AVAILABILITY_DELAY_P99_MS (625),
  and is read after a getLogs RTT (250 p99 typical chunk / 600 wall cap).
  The decision must finish by lock-789 (submit deadline) minus ~50ms
  signal/sizing compute minus ~30ms sign => pool data must be READ-COMPLETE
  by ~lock-869.  Therefore harvestable horizon:
      h >= 869 + 450 + 625 + 250  = 2,194 ms   (p99-typical)
      h >= 869 + 450 + 625 + 600  = 2,544 ms   (worst-wall)
  Classification used below (stricter than wake-time bands, because pool
  state-as-of-t is only readable ~1.1-1.7s AFTER t):
      h >= 4.0s  TRIVIAL  (readable before today's single-poll rail outputs)
      3.0-4.0s   TIGHT    (readable before the decision wake at lock-1195;
                           needs the single poll moved later OR a second poll)
      2.5-3.0s   PUSH     (needs an in-decision getLogs parallel to the OKX
                           fetch; p99 margins only ~100-350ms)
      <  2.5s    INFEASIBLE (read-complete after the submit deadline at p99)

Settlement convention: flat stake 1, REALIZED final-pool payouts, 3% fee,
no gas (infinitesimal-stake limit; era-relative comparisons valid — see
post-mortem method caveats).

Outputs: var/strategy_review/phase0_candidates_2026_06_11/{findings.json,
horizon_sweep.csv} + console digest.
Findings doc: research/phase0_candidates_runtime_2026_06_11_findings.md.

Run:  cd <repo> && .venv/Scripts/python.exe research/phase0_candidates_runtime_2026_06_11.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.in_process_runner as ipr  # noqa: E402
from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.constants import BNB_WEI  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import (  # noqa: E402
    MomentumOnlyPipeline,
    _pools_from_bets,
)
from pancakebot.timing_constants import (  # noqa: E402
    BSC_BLOCK_TIME_MS,
    RPC_BLOCK_AVAILABILITY_DELAY_P99_MS,
)

OUT = REPO / "var" / "strategy_review" / "phase0_candidates_2026_06_11"
CUTOFF = 2
LOOKBACKS = (3, 7, 15)
FEE = 0.03

# Envelope constants (sources: timing_constants.py / rpc_poller.py / dispatch)
GETLOGS_P99_TYPICAL_MS = 250   # rpc_poller._GETLOGS_FETCH_RTT_P99_MS
GETLOGS_WALL_MS = 600          # rpc_poller._GETLOGS_TIMEOUT_MS (> soak max 492)
SUBMIT_DEADLINE_MS = 789       # static bet-submit deadline before lock
COMPUTE_MS = 50                # signal + sizing (SIGNAL_COMPUTE_TIME_MS)
SIGN_MS = 30
READ_COMPLETE_BY_MS = SUBMIT_DEADLINE_MS + COMPUTE_MS + SIGN_MS  # 869

ERAS = [
    ("golden", 437562, 479952),
    ("fade", 479953, 484408),
    ("dead", 484409, 488832),
]
HORIZONS_S = [6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.0]  # 0.0 = final pools


def era_of(epoch: int) -> str:
    for name, lo, hi in ERAS:
        if lo <= epoch <= hi:
            return name
    return "other"


# ---------------------------------------------------------------------------
# Feasibility model
# ---------------------------------------------------------------------------

def horizon_feasibility(h_s: float) -> dict:
    """Classify a pool-state horizon against the runtime envelope."""
    h_ms = h_s * 1000.0
    pipeline_typ = BSC_BLOCK_TIME_MS + RPC_BLOCK_AVAILABILITY_DELAY_P99_MS + GETLOGS_P99_TYPICAL_MS
    pipeline_worst = BSC_BLOCK_TIME_MS + RPC_BLOCK_AVAILABILITY_DELAY_P99_MS + GETLOGS_WALL_MS
    read_complete_typ = h_ms - pipeline_typ      # ms before lock (positive = before)
    read_complete_worst = h_ms - pipeline_worst
    headroom_typ = read_complete_typ - READ_COMPLETE_BY_MS
    headroom_worst = read_complete_worst - READ_COMPLETE_BY_MS
    if h_s == 0.0:
        cls = "INFEASIBLE"
        changes = "final pools form AT lock; informational only"
    elif headroom_worst >= 0 and h_s >= 4.0:
        cls = "TRIVIAL"
        changes = "none (readable at/before today's single-poll output)"
    elif headroom_worst >= 0:
        cls = "TIGHT"
        changes = (f"move/add a poll so a getLogs completes by lock-{READ_COMPLETE_BY_MS:.0f}ms "
                   f"(worst-wall headroom {headroom_worst:+.0f}ms)")
    elif headroom_typ >= 0:
        cls = "PUSH"
        changes = (f"in-decision getLogs parallel to the OKX fetch; p99-typical "
                   f"headroom {headroom_typ:+.0f}ms but worst-wall {headroom_worst:+.0f}ms "
                   f"(would need a tighter getLogs wall + accepting fetch-skip on overrun)")
    else:
        cls = "INFEASIBLE"
        changes = (f"read-complete is {-headroom_typ:.0f}ms past the budget even at "
                   f"p99-typical latency; no path to read + decide + sign + broadcast")
    return dict(h_s=h_s, cls=cls, headroom_typ_ms=round(headroom_typ),
                headroom_worst_ms=round(headroom_worst), required_changes=changes)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_rounds_and_klines():
    print("--- loading rounds ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=False)
    all_rounds = [r for r in all_rounds if r.epoch >= ERAS[0][1]]
    max_lb = max(LOOKBACKS)
    print("--- loading klines ---", flush=True)
    sliced = {}
    for sym, path in (("btc", ipr._BTC_KLINES_PATH), ("eth", ipr._ETH_KLINES_PATH),
                      ("sol", ipr._SOL_KLINES_PATH)):
        uni = ipr._load_klines_unified(
            path, earliest_offset=CUTOFF + max_lb + 1, latest_offset=CUTOFF + 1)
        sliced[sym] = {
            ep: ipr._slice_per_entry(
                kl, kline_cutoff_seconds=CUTOFF, max_lookback=max_lb,
                earliest_offset=CUTOFF + max_lb + 1)
            for ep, kl in uni.items()
        }
    return all_rounds, sliced


def build_round_table(all_rounds, sliced) -> list[dict]:
    """One row per usable round: pools at each horizon, payouts, momentum."""
    rows = []
    for r in all_rounds:
        winner = r.position
        if winner not in ("Bull", "Bear"):
            continue
        lock = int(r.lock_at)
        pools_f = compute_pool_amounts_wei(bets=r.bets)
        f_bull = pools_f.bull_wei / BNB_WEI
        f_bear = pools_f.bear_wei / BNB_WEI
        f_total = f_bull + f_bear
        if f_total <= 0 or f_bull <= 0 or f_bear <= 0:
            continue
        row = dict(
            epoch=int(r.epoch), era=era_of(int(r.epoch)),
            outcome_bull=1.0 if winner == "Bull" else 0.0,
            payout_bull=f_total * (1.0 - FEE) / f_bull,
            payout_bear=f_total * (1.0 - FEE) / f_bear,
        )
        for h in HORIZONS_S:
            if h == 0.0:
                b, be = f_bull, f_bear
            else:
                b, be = _pools_from_bets(r, int(lock - h))
            tot = b + be
            row[f"imb_{h}"] = (b - be) / tot if tot > 0 else 0.0
            row[f"total_{h}"] = tot
        # momentum (for C4)
        entry = sliced["btc"].get(int(r.epoch))
        e_eth = sliced["eth"].get(int(r.epoch))
        e_sol = sliced["sol"].get(int(r.epoch))
        if entry and len(entry) >= max(LOOKBACKS) + 1 and e_eth and e_sol:
            closes = np.array([float(k[4]) for k in entry])
            rets = {lb: float(closes[-1] / closes[-1 - lb] - 1.0) for lb in LOOKBACKS}
            row["btc_agree"] = (
                len({np.sign(v) for v in rets.values()}) == 1
                and np.sign(rets[3]) != 0)
            row["btc_dir_bull"] = rets[3] > 0
            row["btc_minabs"] = min(abs(v) for v in rets.values())
        else:
            row["btc_agree"] = False
            row["btc_dir_bull"] = None
            row["btc_minabs"] = None
        rows.append(row)
    print(f"--- round table: {len(rows)} usable rounds ---")
    return rows


def settle(row, side_bull: bool) -> float:
    win = bool(row["outcome_bull"]) == side_bull
    pay = row["payout_bull"] if side_bull else row["payout_bear"]
    return (pay - 1.0) if win else -1.0


def era_strategy_table(rows, decide) -> dict:
    """decide(row) -> True (bull) / False (bear) / None (skip)."""
    out = {}
    for era in ("golden", "fade", "dead"):
        pnls, wins, pays = [], [], []
        for row in rows:
            if row["era"] != era:
                continue
            side = decide(row)
            if side is None:
                continue
            pnls.append(settle(row, side))
            wins.append(bool(row["outcome_bull"]) == side)
            pays.append(row["payout_bull"] if side else row["payout_bear"])
        n = len(pnls)
        if n == 0:
            out[era] = dict(n=0)
            continue
        wr = float(np.mean(wins))
        be = float(np.mean([1.0 / p for p in pays]))
        se = float(np.std(pnls) / np.sqrt(n)) if n > 1 else 0.0
        mean = float(np.mean(pnls))
        out[era] = dict(n=n, wr=round(wr, 4), breakeven_wr=round(be, 4),
                        mean_pnl=round(mean, 4), pnl_se=round(se, 4),
                        z_vs_zero=round(mean / se, 2) if se > 0 else None)
    return out


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def c1_horizon_sweep(rows) -> list[dict]:
    """Per horizon x era: imbalance->outcome r, majority-follow PnL, and
    delta-flow-follow PnL; joined with the feasibility verdict."""
    out = []
    for h in HORIZONS_S:
        feas = horizon_feasibility(h)
        rec = dict(h_s=h, feasibility=feas["cls"],
                   headroom_worst_ms=feas["headroom_worst_ms"],
                   required_changes=feas["required_changes"])
        for era in ("golden", "fade", "dead"):
            sub = [r for r in rows if r["era"] == era and r[f"total_{h}"] > 0]
            x = np.array([r[f"imb_{h}"] for r in sub])
            y = np.array([1.0 if r["outcome_bull"] else -1.0 for r in sub])
            rec[f"{era}_r"] = round(float(np.corrcoef(x, y)[0, 1]), 4) if len(sub) > 100 and np.std(x) > 0 else None
            # majority-follow at |imb|>=0.2, realized payouts
            tbl = era_strategy_table(
                [r for r in rows if r[f"total_{h}"] > 0],
                lambda row, _h=h: (row[f"imb_{_h}"] > 0) if abs(row[f"imb_{_h}"]) >= 0.2 else None)
            rec[f"{era}_follow_mean_pnl"] = tbl[era].get("mean_pnl")
            rec[f"{era}_follow_n"] = tbl[era].get("n")
        # delta-flow (h vs 6s base) follow at |delta|>=0.1 — only for h<6
        if h < 6.0 and h > 0.0:
            def _dec(row, _h=h):
                if row[f"total_{_h}"] <= 0 or row["total_6.0"] <= 0:
                    return None
                d = row[f"imb_{_h}"] - row["imb_6.0"]
                return (d > 0) if abs(d) >= 0.1 else None
            tbl = era_strategy_table(rows, _dec)
            for era in ("golden", "fade", "dead"):
                rec[f"{era}_dflow_mean_pnl"] = tbl[era].get("mean_pnl")
                rec[f"{era}_dflow_n"] = tbl[era].get("n")
        out.append(rec)
    return out


def c2_contrarian(rows) -> dict:
    """Minority side at lock-6s when |imb| >= t. TRIVIALLY feasible (same
    data the canonical strategy already reads)."""
    res = {}
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        tbl = era_strategy_table(
            rows,
            lambda row, _t=t: (not (row["imb_6.0"] > 0)) if (
                row["total_6.0"] > 0 and abs(row["imb_6.0"]) >= _t) else None)
        res[f"thr_{t}"] = tbl
    return res


def c2_canonical_overlay(all_rounds, sliced, rows_by_epoch) -> dict:
    """Canonical gate AND against-crowd subset per era (the post-mortem's
    +0.180 vs +0.032 concentration, re-evaluated)."""
    strategy_cfg = load_strategy_config_from_dict({})
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CUTOFF,
        mtf_lookbacks=strategy_cfg.gate.mtf_lookbacks,
        mtf_min_return_threshold=strategy_cfg.gate.mtf_min_return_threshold,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_cfg, strategy_config=strategy_cfg, gate=None,
        kline_cutoff_seconds=CUTOFF, pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001, treasury_fee_fraction=FEE,
        bankroll_tracker=None,
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=sliced["btc"])
    pipeline.refresh_eth_klines(eth_klines_by_epoch=sliced["eth"])
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sliced["sol"])
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    agg = defaultdict(lambda: defaultdict(list))
    for r in all_rounds:
        d = pipeline.decide_open_round(round_t=r)
        if d.action != "BET":
            continue
        row = rows_by_epoch.get(int(r.epoch))
        if row is None or row["total_6.0"] <= 0:
            continue
        side_bull = d.bet_side == "Bull"
        with_crowd = side_bull == (row["imb_6.0"] > 0)
        agg[row["era"]]["with" if with_crowd else "against"].append(
            settle(row, side_bull))
    out = {}
    for era, groups in agg.items():
        out[era] = {
            k: dict(n=len(v), mean_pnl=round(float(np.mean(v)), 4),
                    wr=round(float(np.mean([p > 0 for p in v])), 4))
            for k, v in groups.items()
        }
    return out


def c4_bulk_momentum(rows) -> dict:
    """Tri-agreement BTC momentum in strength BANDS (the canonical gate region
    is minabs >= 1e-4/2e-4; the 'bulk' is below it). Same data as canonical
    -> trivially feasible."""
    bands = [(0.00002, 0.00005), (0.00005, 0.0001), (0.0001, 0.0002),
             (0.0002, 0.0005), (0.0005, 10.0)]
    res = {}
    for lo, hi in bands:
        def _dec(row, _lo=lo, _hi=hi):
            if not row["btc_agree"] or row["btc_minabs"] is None:
                return None
            if not (_lo <= row["btc_minabs"] < _hi):
                return None
            return bool(row["btc_dir_bull"])
        res[f"band_{lo}_{hi}"] = era_strategy_table(rows, _dec)
    return res


def c3_wallet_runtime_note(all_rounds) -> dict:
    wallets = set()
    n_bets = 0
    for r in all_rounds:
        for b in r.bets:
            wallets.add(b.wallet_address)
            n_bets += 1
    n_w = len(wallets)
    # trailing-stats struct: ~5 floats + key ~= 120 bytes conservatively
    mem_mb = n_w * 120 / 1e6
    return dict(
        distinct_wallets=n_w, total_bets=n_bets,
        est_state_mb=round(mem_mb, 1),
        lookup="O(1) dict per bet event; update on settle — microseconds",
        decision_budget_impact="none beyond pool features already read at lock-6s "
                               "(wallet features derive from the SAME -6s event set)",
        verdict="runtime TRIVIAL at -6s horizon; the binding constraint is "
                "research validity (address churn, overfit surface), not runtime",
    )


# ---------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_rounds, sliced = load_rounds_and_klines()
    rows = build_round_table(all_rounds, sliced)
    rows_by_epoch = {row["epoch"]: row for row in rows}

    feas_table = [horizon_feasibility(h) for h in HORIZONS_S]
    c1 = c1_horizon_sweep(rows)
    c2 = c2_contrarian(rows)
    c2_overlay = c2_canonical_overlay(all_rounds, sliced, rows_by_epoch)
    c4 = c4_bulk_momentum(rows)
    c3 = c3_wallet_runtime_note(all_rounds)

    findings = dict(
        envelope=dict(
            read_complete_by_ms_before_lock=READ_COMPLETE_BY_MS,
            pipeline_typ_ms=BSC_BLOCK_TIME_MS + RPC_BLOCK_AVAILABILITY_DELAY_P99_MS + GETLOGS_P99_TYPICAL_MS,
            pipeline_worst_ms=BSC_BLOCK_TIME_MS + RPC_BLOCK_AVAILABILITY_DELAY_P99_MS + GETLOGS_WALL_MS,
        ),
        feasibility=feas_table, c1_horizon_sweep=c1, c2_contrarian=c2,
        c2_canonical_overlay=c2_overlay, c4_bulk_momentum=c4,
        c3_wallet_runtime=c3,
    )
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2),
                                       encoding="utf-8")
    _fields: list[str] = []
    for rec in c1:
        for k in rec:
            if k not in _fields:
                _fields.append(k)
    with open(OUT / "horizon_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_fields)
        w.writeheader()
        w.writerows(c1)

    print("\n=== FEASIBILITY (pool-state horizon -> runtime class) ===")
    for f in feas_table:
        print(f"  lock-{f['h_s']:>3}s  {f['cls']:<10} worst-wall headroom "
              f"{f['headroom_worst_ms']:+5d}ms  {f['required_changes']}")
    print("\n=== C1 horizon sweep (imb->outcome r per era) ===")
    for rec in c1:
        print(f"  h={rec['h_s']:>3}s [{rec['feasibility']:<10}] "
              f"r: golden={rec['golden_r']} fade={rec['fade_r']} dead={rec['dead_r']} | "
              f"dead follow pnl={rec.get('dead_follow_mean_pnl')} (n={rec.get('dead_follow_n')}) "
              f"dflow pnl={rec.get('dead_dflow_mean_pnl')} (n={rec.get('dead_dflow_n')})")
    print("\n=== C2 contrarian (minority @ -6s) ===")
    print(json.dumps(c2, indent=1)[:1500])
    print("\n=== C2 canonical overlay (with/against crowd per era) ===")
    print(json.dumps(c2_overlay, indent=1))
    print("\n=== C4 bulk momentum bands ===")
    print(json.dumps(c4, indent=1)[:2000])
    print("\n=== C3 wallet runtime note ===")
    print(json.dumps(c3, indent=1))
    print(f"\n[done] {time.time()-t0:.0f}s; artifacts in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
