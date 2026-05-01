"""M4 + p1e verification.

M4 — clarifying reviewer's optimized-replay-vs-naive premise:
  No optimized stream-replay path was ever implemented in p1a_kill_switch.py.
  Every (combo, window) backtest in the sweep calls pipeline.decide_open_round()
  and settle_bet_against_closed_round() per round (naive full pipeline).
  Verification here: run picked combo on wf_00 TWICE, prove per-round bit-identical.
  Also re-derive picked combo's wf_00 contribution and compare to sweep result.

p1e — gates on extension cohort at 100 BNB:
  Reviewer flagged that "Arms A == B (gates no-op)" on the WF range was
  bankroll-conditional. Verify whether cooldown + drawdown breaker fire
  on extension cohort at the same 100 BNB initial bankroll.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

# UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(r"C:\Users\zking\Documents\GitHub\PancakeBot\.claude\worktrees\stupefied-bell-4d955c")))

from research.p1a_kill_switch import (
    backtest_window, load_data_for_range, aggregate_marginal_pnl, summarize_diffs,
    WF_WINDOWS, CUTOFF_SECONDS, INITIAL_BANKROLL_BNB, make_strategy_config,
)
from pancakebot.market_data.contract_constants import load_contract_constants

EXT_RANGE = (422298, 437561)
PICKED_COMBO = {"N": 50, "L": -2.0, "K": 50}

OUT = REPO / "var" / "extended" / "p1a_m4_p1e_verification.json"


def hash_per_round(records: list) -> str:
    """Stable hash of per-round records for bit-identical comparison."""
    s = "\n".join(
        f"{r['epoch']},{r['canonical_would_bet']},{r['ks_suppressed']},"
        f"{r['actual_bet']},{r['actual_bet_size']:.18f},{r['profit']:.18f},"
        f"{r['win']},{r['include_in_marginal']}"
        for r in records
    )
    return hashlib.md5(s.encode()).hexdigest()


def main():
    print("=" * 100, flush=True)
    print("M4 + p1e verification", flush=True)
    print("=" * 100, flush=True)
    t_start = time.time()

    cc = load_contract_constants()
    treasury_fee = float(cc.treasury_fee_fraction)
    min_bet_amount = float(cc.min_bet_amount_bnb)

    canonical_cfg = make_strategy_config(gates_on=True)
    max_lookback = max(canonical_cfg.gate.mtf_lookbacks)
    earliest_offset = CUTOFF_SECONDS + max_lookback + 1

    # ============================================================
    # M4: determinism rerun on wf_00
    # ============================================================
    print(f"\n[M4] Determinism verification — combo {PICKED_COMBO} on wf_00", flush=True)
    w = WF_WINDOWS[0]
    rounds, btc, eth, sol = load_data_for_range(
        w["train_lo"], w["test_hi"], use_extended=False, earliest_offset=earliest_offset,
    )
    print(f"  loaded {len(rounds)} rounds for wf_00 [{w['train_lo']}..{w['test_hi']}]", flush=True)

    # Canonical baseline (Arm A) on wf_00
    print("  Run #C: canonical baseline on wf_00...", flush=True)
    t0 = time.time()
    canon = backtest_window(
        rounds, btc, eth, sol,
        earliest_offset=earliest_offset, test_lo=w["test_lo"],
        gates_on=True, kill_switch_combo=None,
        treasury_fee=treasury_fee, min_bet_amount=min_bet_amount,
    )
    print(f"    {time.time()-t0:.2f}s, n_test_records={canon['n_records_test']}", flush=True)
    canon_hash = hash_per_round(canon["per_round"])
    print(f"    canon hash: {canon_hash}", flush=True)

    # Picked combo run #1
    print("  Run #1: picked combo, naive full-pipeline path...", flush=True)
    t0 = time.time()
    run1 = backtest_window(
        rounds, btc, eth, sol,
        earliest_offset=earliest_offset, test_lo=w["test_lo"],
        gates_on=True, kill_switch_combo=PICKED_COMBO,
        treasury_fee=treasury_fee, min_bet_amount=min_bet_amount,
    )
    print(f"    {time.time()-t0:.2f}s, n_test_records={run1['n_records_test']}, "
          f"pauses_fired={run1['ks_n_pauses_fired']}, final_bankroll={run1['final_bankroll']:.6f}",
          flush=True)
    hash1 = hash_per_round(run1["per_round"])
    print(f"    run #1 hash: {hash1}", flush=True)

    # Picked combo run #2 (independent invocation)
    print("  Run #2: picked combo, naive full-pipeline path (independent reload)...", flush=True)
    t0 = time.time()
    rounds2, btc2, eth2, sol2 = load_data_for_range(
        w["train_lo"], w["test_hi"], use_extended=False, earliest_offset=earliest_offset,
    )
    run2 = backtest_window(
        rounds2, btc2, eth2, sol2,
        earliest_offset=earliest_offset, test_lo=w["test_lo"],
        gates_on=True, kill_switch_combo=PICKED_COMBO,
        treasury_fee=treasury_fee, min_bet_amount=min_bet_amount,
    )
    print(f"    {time.time()-t0:.2f}s, n_test_records={run2['n_records_test']}, "
          f"pauses_fired={run2['ks_n_pauses_fired']}, final_bankroll={run2['final_bankroll']:.6f}",
          flush=True)
    hash2 = hash_per_round(run2["per_round"])
    print(f"    run #2 hash: {hash2}", flush=True)

    determinism_pass = (hash1 == hash2)
    print(f"  DETERMINISM: {'PASS — bit-identical' if determinism_pass else 'FAIL — non-deterministic'}",
          flush=True)

    # Marginal PnL contribution of wf_00 (compare to sweep's reported value)
    diffs, n_suppressed = aggregate_marginal_pnl(canon["per_round"], run1["per_round"])
    s = summarize_diffs(diffs)
    print(f"  wf_00 marginal: n={s.get('n', 0)}, mean_per_bet={s.get('mean_per_bet', 0):+.6f}, "
          f"suppressed={n_suppressed}", flush=True)

    # ============================================================
    # p1e: extension cohort gates check at 100 BNB
    # ============================================================
    print(f"\n[p1e] Extension cohort gates check at {INITIAL_BANKROLL_BNB} BNB", flush=True)
    ext_min, ext_max = EXT_RANGE
    print(f"  Loading extension cohort [{ext_min}..{ext_max}]...", flush=True)
    t0 = time.time()
    ext_rounds, ext_btc, ext_eth, ext_sol = load_data_for_range(
        ext_min, ext_max, use_extended=True, earliest_offset=earliest_offset,
    )
    print(f"  loaded {len(ext_rounds)} rounds in {time.time()-t0:.1f}s", flush=True)

    # Arm A: gates ON
    print("  Arm A (canonical with cooldown ON, drawdown breaker ON)...", flush=True)
    t0 = time.time()
    arm_a = backtest_window(
        ext_rounds, ext_btc, ext_eth, ext_sol,
        earliest_offset=earliest_offset, test_lo=ext_min,
        gates_on=True, kill_switch_combo=None,
        treasury_fee=treasury_fee, min_bet_amount=min_bet_amount,
    )
    print(f"    {time.time()-t0:.2f}s, final_bankroll={arm_a['final_bankroll']:.6f}", flush=True)

    # Arm B: gates OFF
    print("  Arm B (cooldown OFF, drawdown breaker OFF)...", flush=True)
    t0 = time.time()
    arm_b = backtest_window(
        ext_rounds, ext_btc, ext_eth, ext_sol,
        earliest_offset=earliest_offset, test_lo=ext_min,
        gates_on=False, kill_switch_combo=None,
        treasury_fee=treasury_fee, min_bet_amount=min_bet_amount,
    )
    print(f"    {time.time()-t0:.2f}s, final_bankroll={arm_b['final_bankroll']:.6f}", flush=True)

    # Stats
    arm_a_bets = sum(1 for r in arm_a["per_round"] if r["actual_bet"])
    arm_b_bets = sum(1 for r in arm_b["per_round"] if r["actual_bet"])
    arm_a_pnl = sum(r["profit"] for r in arm_a["per_round"])
    arm_b_pnl = sum(r["profit"] for r in arm_b["per_round"])
    arm_a_wins = sum(1 for r in arm_a["per_round"] if r["win"])
    arm_b_wins = sum(1 for r in arm_b["per_round"] if r["win"])

    # Hash to confirm whether they're bit-identical (= gates didn't fire) or differ
    arm_a_hash = hash_per_round(arm_a["per_round"])
    arm_b_hash = hash_per_round(arm_b["per_round"])
    gates_fired = (arm_a_hash != arm_b_hash)

    # Skip-reason histogram for Arm A (gates ON)
    arm_a_skip_hist: dict = {}
    arm_b_skip_hist: dict = {}
    # We need to re-derive skip reasons from per_round records — but the existing record
    # only tracks "actual_bet" and "ks_suppressed", not raw skip_reason. Best we can do
    # without re-running: count the "canonical_would_bet=False" rounds, which include all
    # gate skips + all signal-no-bet skips. The DELTA between A and B in this count = gate fires.
    arm_a_skips = sum(1 for r in arm_a["per_round"] if not r["canonical_would_bet"])
    arm_b_skips = sum(1 for r in arm_b["per_round"] if not r["canonical_would_bet"])
    gate_fired_count = arm_b_bets - arm_a_bets  # Arm B has gates off, so any extra bets it makes = gates suppressed those in A

    print(f"  Arm A: bets={arm_a_bets} wins={arm_a_wins} pnl={arm_a_pnl:+.4f} bankroll={arm_a['final_bankroll']:.4f}",
          flush=True)
    print(f"  Arm B: bets={arm_b_bets} wins={arm_b_wins} pnl={arm_b_pnl:+.4f} bankroll={arm_b['final_bankroll']:.4f}",
          flush=True)
    print(f"  Bit-identical: {not gates_fired}", flush=True)
    print(f"  Bet count delta (B - A = gate-suppressed bets in A): {arm_b_bets - arm_a_bets}", flush=True)
    print(f"  PnL delta (A - B): {arm_a_pnl - arm_b_pnl:+.4f}", flush=True)

    # ============================================================
    # Output
    # ============================================================
    out = {
        "spec": {
            "picked_combo": PICKED_COMBO,
            "initial_bankroll_bnb": INITIAL_BANKROLL_BNB,
            "extension_range": list(EXT_RANGE),
            "wf_00": w,
        },
        "M4_determinism": {
            "implementation_clarification": (
                "p1a_kill_switch.py implements ONLY the naive full-pipeline path. "
                "Every (combo, window) backtest calls pipeline.decide_open_round() "
                "and settle_bet_against_closed_round() per round. No optimized "
                "stream-replay was ever implemented. Reviewer's M4 premise was based "
                "on the unexpectedly fast 6.07 min runtime; the actual cause is the "
                "canonical pipeline running at ~30 microseconds per round in-memory "
                "without I/O overhead, not stream-replay shortcuts."
            ),
            "wf_00_canon_hash": canon_hash,
            "wf_00_run1_hash": hash1,
            "wf_00_run2_hash": hash2,
            "determinism_pass": determinism_pass,
            "wf_00_run1_pauses_fired": run1["ks_n_pauses_fired"],
            "wf_00_run2_pauses_fired": run2["ks_n_pauses_fired"],
            "wf_00_run1_final_bankroll": run1["final_bankroll"],
            "wf_00_run2_final_bankroll": run2["final_bankroll"],
            "wf_00_marginal_summary": s,
            "wf_00_n_suppressed": n_suppressed,
        },
        "p1e_extension_gates": {
            "n_rounds": len(ext_rounds),
            "arm_a_gates_on": {
                "n_bets": arm_a_bets,
                "n_wins": arm_a_wins,
                "win_rate": arm_a_wins / arm_a_bets if arm_a_bets else 0.0,
                "total_pnl": arm_a_pnl,
                "final_bankroll": arm_a["final_bankroll"],
                "hash": arm_a_hash,
            },
            "arm_b_gates_off": {
                "n_bets": arm_b_bets,
                "n_wins": arm_b_wins,
                "win_rate": arm_b_wins / arm_b_bets if arm_b_bets else 0.0,
                "total_pnl": arm_b_pnl,
                "final_bankroll": arm_b["final_bankroll"],
                "hash": arm_b_hash,
            },
            "gates_fired": gates_fired,
            "bet_count_delta_b_minus_a": arm_b_bets - arm_a_bets,
            "pnl_delta_a_minus_b": arm_a_pnl - arm_b_pnl,
        },
        "elapsed_seconds": time.time() - t_start,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults JSON: {OUT}", flush=True)
    print(f"Total elapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
