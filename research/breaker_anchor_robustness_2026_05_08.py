"""Breaker anchor robustness: restart-simulation on V1 + V2 fixed-anchor sweep.

Two questions:

  Part A (restart fragility): in single-run backtest, V1's `absolute_ratchet`
  peak is mathematically identical to V1-persistent (no restarts). Live, the
  bot restarts (supervisor, OS updates, ops). On restart, the in-memory peak
  re-initializes to the current bankroll, and cooldown resets to 0. How much
  protection does V1 actually deliver under realistic restart cadences?

  Part B (fixed-anchor breaker): a V2 design where the operator configures a
  hard target_bnb. Breaker fires when current_bankroll < target_bnb.
  Equivalent to a percentage-of-initial drop, but the anchor is fixed (does
  not decay or reset).

Cohort: extension epochs 422298..437561 (~15,262 rounds), canonical strategy.
Scales: 5 / 50 / 100 BNB.

Restart patterns (Part A):
  - none      : current behavior (V1-in-memory == V1-persistent for single run)
  - daily     : every 288 rounds (5 min/round * 288 = 24h)
  - weekly    : every 2016 rounds (7 days)
  - random    : per-round Bernoulli p=1/288 (~1/day on average)

Variants:
  Part A:
    - V1-in-memory (`absolute_ratchet`, peak resets at simulated restarts)
    - V1-persistent (`absolute_ratchet`, peak SURVIVES restarts; the
      single-run no-restart baseline reused for all restart patterns since
      math is identical -- only the V1-in-memory variant changes per pattern)
  Part B:
    - V2-fixed-anchor: target_bnb at 95/90/85/80% of initial.

Output: var/extended/breaker_anchor_robustness_results.json + per-fold dirs
under var/extended/breaker_anchor_robustness/<variant>/<scale>/...

This script subclasses ``InMemoryBankrollTracker``; production code unchanged.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pancakebot.bankroll_tracker import InMemoryBankrollTracker
from pancakebot.constants import GAS_COST_BET_BNB
from pancakebot.settlement import settle_bet_against_closed_round
from pancakebot.strategy.momentum_gate import MomentumGateConfig
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.util import InvariantError

from research.in_process_runner import (
    FoldSpec,
    _BTC_KLINES_PATH,
    _ETH_KLINES_PATH,
    _SOL_KLINES_PATH,
    _EXT_BTC_KLINES_PATH,
    _EXT_ETH_KLINES_PATH,
    _EXT_SOL_KLINES_PATH,
    _compute_load_extent,
    _load_all_rounds,
    _load_klines_unified,
    _resolve_strategy_config,
    _slice_per_entry,
)


COHORT_EPOCH_START = 422298
COHORT_EPOCH_END = 437561  # inclusive; ~15,262 rounds


# -----------------------------------------------------------------------------
# Custom tracker subclasses
# -----------------------------------------------------------------------------


class RestartingV1Tracker(InMemoryBankrollTracker):
    """V1 (`absolute_ratchet`) with simulated restarts.

    On restart: reset ``_absolute_peak`` to current bankroll, reset cooldown
    to 0, and clear the rolling-window entries deque (so a fresh process is
    starting from a fresh in-memory state). The seed flag stays True since
    we're "continuing" the same backtest -- next ``record_settlement`` won't
    re-emit an init entry.

    Restart pattern is driven by the runner: it inspects the round count and
    calls ``simulate_restart()`` at the appropriate boundaries.
    """

    def simulate_restart(self) -> None:
        """Reset V1 state to mimic a process restart.

        On a real restart the new process seeds from on-chain wallet balance,
        which is the current bankroll. Reset:
          - _absolute_peak to current bankroll
          - _initial to current bankroll (so the next re-seed init-entry uses it)
          - _cooldown to 0
          - _triggered_at to None
          - _entries deque cleared, _seeded=False (lazy re-seed on next settle)
        """
        if self._seeded and self._entries:
            current = self._entries[-1].bankroll
        else:
            current = self._initial
        self._absolute_peak = float(current)
        self._initial = float(current)
        self._cooldown = 0
        self._triggered_at = None
        self._entries.clear()
        self._seeded = False  # forces re-seed on next record_settlement


class FixedAnchorTracker(InMemoryBankrollTracker):
    """V2: fixed user-configured target bankroll.

    Override ``peak_bankroll`` so that the existing pipeline check
    ``(peak - current)/peak >= max_drawdown_frac_from_peak`` fires exactly
    when ``current < target_bnb``.

      With ``max_drawdown_frac_from_peak = 0.15``:
        (peak - current)/peak >= 0.15
        current <= 0.85 * peak
        peak = target / 0.85  =>  fire when current <= target_bnb

    This avoids touching the strategy's threshold logic while implementing
    the fixed-anchor semantics cleanly.
    """

    __slots__ = ("_target_bnb", "_dd_threshold")

    def __init__(
        self,
        *,
        initial_bankroll: float,
        window_days: int,
        target_bnb: float,
        dd_threshold: float = 0.15,
    ) -> None:
        super().__init__(
            initial_bankroll=initial_bankroll,
            window_days=window_days,
            peak_mode="rolling_7d",  # any valid value, overridden below
        )
        self._target_bnb = float(target_bnb)
        self._dd_threshold = float(dd_threshold)

    def peak_bankroll(self, as_of_start_at: int) -> float:
        # Synthesize a peak so that (peak - current)/peak >= dd_threshold
        # exactly when current <= target_bnb.
        return self._target_bnb / (1.0 - self._dd_threshold)


# -----------------------------------------------------------------------------
# Restart schedule helpers
# -----------------------------------------------------------------------------


def restart_indices_periodic(total_rounds: int, period: int) -> set[int]:
    """Restart at round indices period, 2*period, 3*period, ... (1-indexed)."""
    if period <= 0:
        return set()
    return {i for i in range(period, total_rounds + 1, period)}


def restart_indices_random(total_rounds: int, prob: float, seed: int = 42) -> set[int]:
    """Restart with Bernoulli(prob) per round; deterministic via seed."""
    rng = random.Random(seed)
    out: set[int] = set()
    for i in range(1, total_rounds + 1):
        if rng.random() < prob:
            out.add(i)
    return out


# -----------------------------------------------------------------------------
# Custom run_fold that supports custom-tracker injection + restart simulation
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class _Stats:
    num_bets: int = 0
    num_wins: int = 0
    skip_counts_by_reason: dict[str, int] = field(default_factory=dict)


def _run_fold_with_tracker(
    *,
    spec: FoldSpec,
    tracker: InMemoryBankrollTracker,
    all_rounds: list,
    btc_unified: dict,
    eth_unified: dict,
    sol_unified: dict,
    earliest_offset: int,
    output_dir: Path,
    initial_bankroll_bnb: float,
    treasury_fee_fraction: float,
    min_bet_amount_bnb: float,
    restart_round_indices: set[int] | None = None,
    restart_callback=None,
    restart_count: list[int] | None = None,
) -> dict[str, Any]:
    """Run one fold with a pre-built tracker. Optionally trigger restarts.

    ``restart_round_indices``: 1-indexed positions in sim_rounds where a
    simulated restart is performed BEFORE that round's decision. If a
    restart fires, ``restart_callback`` is invoked (no args) so caller-side
    bookkeeping can run (counter increment, tracker.simulate_restart(), etc.).

    Returns summary dict.
    """
    strategy_cfg = _resolve_strategy_config(spec)
    max_lookback = max(strategy_cfg.gate.mtf_lookbacks)

    sim_rounds = [
        r for r in all_rounds
        if (spec.epoch_start is None or r.epoch >= spec.epoch_start)
        and (spec.epoch_end is None or r.epoch <= spec.epoch_end)
    ]
    if not sim_rounds:
        raise InvariantError(f"no rounds: {spec.name}")

    btc_klines = {
        ep: _slice_per_entry(kl, cutoff_seconds=spec.cutoff_seconds,
                             max_lookback=max_lookback, earliest_offset=earliest_offset)
        for ep, kl in btc_unified.items()
    }
    eth_klines = {
        ep: _slice_per_entry(kl, cutoff_seconds=spec.cutoff_seconds,
                             max_lookback=max_lookback, earliest_offset=earliest_offset)
        for ep, kl in eth_unified.items()
    }
    sol_klines = {
        ep: _slice_per_entry(kl, cutoff_seconds=spec.cutoff_seconds,
                             max_lookback=max_lookback, earliest_offset=earliest_offset)
        for ep, kl in sol_unified.items()
    }

    gate_config = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        cutoff_seconds=spec.cutoff_seconds,
        mtf_lookbacks=strategy_cfg.gate.mtf_lookbacks,
        mtf_threshold=strategy_cfg.gate.mtf_threshold,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_config,
        strategy_config=strategy_cfg,
        gate=None,
        cutoff_seconds=spec.cutoff_seconds,
        min_bet_amount_bnb=min_bet_amount_bnb,
        treasury_fee_fraction=treasury_fee_fraction,
        bankroll_tracker=tracker,
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "trades.csv"
    summary_path = output_dir / "summary.json"

    bankroll = initial_bankroll_bnb
    stats = _Stats()
    restart_set = restart_round_indices or set()

    t_start = time.perf_counter()
    with open(trades_path, "w", newline="", encoding="utf-8") as trades_f:
        trades_w = csv.writer(trades_f)
        trades_w.writerow([
            "epoch", "action", "skip_reason", "direction",
            "bet_size_bnb", "profit_bnb", "bankroll_bnb", "restart_fired",
        ])
        for idx_1, round_t in enumerate(sim_rounds, start=1):
            restart_fired = 0
            if idx_1 in restart_set and restart_callback is not None:
                restart_callback()
                restart_fired = 1
                if restart_count is not None:
                    restart_count[0] += 1

            decision = pipeline.decide_open_round(round_t=round_t)

            profit = 0.0
            if decision.action == "BET" and decision.bet_size_bnb > 0.0:
                bet_side = decision.bet_side
                if bet_side not in ("Bull", "Bear"):
                    raise InvariantError("bet_side_invalid")
                bankroll -= decision.bet_size_bnb + GAS_COST_BET_BNB
                outcome = settle_bet_against_closed_round(
                    bet_bnb=decision.bet_size_bnb,
                    bet_side=bet_side,
                    round_closed=round_t,
                    treasury_fee_fraction=treasury_fee_fraction,
                )
                bankroll += outcome.credit_bnb
                profit = outcome.credit_bnb - decision.bet_size_bnb - GAS_COST_BET_BNB
                stats.num_bets += 1
                if outcome.outcome == "win":
                    stats.num_wins += 1
            else:
                key = decision.skip_reason or "unknown_skip_reason"
                stats.skip_counts_by_reason[key] = stats.skip_counts_by_reason.get(key, 0) + 1

            trades_w.writerow([
                round_t.epoch, decision.action, decision.skip_reason or "",
                decision.bet_side or "", decision.bet_size_bnb, profit,
                bankroll, restart_fired,
            ])

            pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
            pipeline.settle_closed_rounds(rounds=[round_t])

    elapsed = time.perf_counter() - t_start

    total = len(sim_rounds)
    net_pnl = bankroll - initial_bankroll_bnb
    wr = stats.num_wins / stats.num_bets if stats.num_bets else 0.0
    skip_detail = dict(sorted(stats.skip_counts_by_reason.items(), key=lambda x: -x[1]))

    summary = {
        "simulation_size": total,
        "initial_bankroll_bnb": initial_bankroll_bnb,
        "final_bankroll_bnb": bankroll,
        "net_pnl_bnb": net_pnl,
        "num_bets": stats.num_bets,
        "num_wins": stats.num_wins,
        "win_rate": wr,
        "skip_counts_by_reason": skip_detail,
        "breaker_fires": int(skip_detail.get("risk_drawdown_breaker_fired", 0)),
        "cooldown_rounds": int(skip_detail.get("risk_cooldown_active", 0)),
        "restart_count": (restart_count[0] if restart_count is not None else 0),
        "elapsed_sim_seconds": elapsed,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


# -----------------------------------------------------------------------------
# Variant runners
# -----------------------------------------------------------------------------


def make_v1_tracker(scale: float, window_days: int) -> InMemoryBankrollTracker:
    return InMemoryBankrollTracker(
        initial_bankroll=scale, window_days=window_days,
        peak_mode="absolute_ratchet",
    )


def make_v1_restarting_tracker(scale: float, window_days: int) -> RestartingV1Tracker:
    return RestartingV1Tracker(
        initial_bankroll=scale, window_days=window_days,
        peak_mode="absolute_ratchet",
    )


def make_v2_fixed_tracker(
    scale: float, window_days: int, target_pct: float,
) -> FixedAnchorTracker:
    return FixedAnchorTracker(
        initial_bankroll=scale,
        window_days=window_days,
        target_bnb=scale * target_pct,
        dd_threshold=0.15,
    )


def make_v0_tracker(scale: float, window_days: int) -> InMemoryBankrollTracker:
    return InMemoryBankrollTracker(
        initial_bankroll=scale, window_days=window_days,
        peak_mode="rolling_7d",
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    out_root = REPO_ROOT / "var" / "extended" / "breaker_anchor_robustness"
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = REPO_ROOT / "var" / "extended" / "breaker_anchor_robustness_results.json"

    # ---------- one-time data load ----------
    spec_template = FoldSpec(
        name="_template",
        cutoff_seconds=2,
        epoch_start=COHORT_EPOCH_START,
        epoch_end=COHORT_EPOCH_END,
    )
    print("Loading rounds + klines (one-time)...", flush=True)
    t0 = time.perf_counter()
    all_rounds = _load_all_rounds(use_extended_data=True)
    resolved = [(spec_template, _resolve_strategy_config(spec_template))]
    earliest_offset, latest_offset, load_count = _compute_load_extent(resolved)
    print(f"  {len(all_rounds)} rounds; load_extent=[{latest_offset}..{earliest_offset}]"
          f" ({load_count} candles)", flush=True)
    btc_unified = _load_klines_unified(
        _BTC_KLINES_PATH, earliest_offset=earliest_offset,
        latest_offset=latest_offset, extended_path=_EXT_BTC_KLINES_PATH,
    )
    eth_unified = _load_klines_unified(
        _ETH_KLINES_PATH, earliest_offset=earliest_offset,
        latest_offset=latest_offset, extended_path=_EXT_ETH_KLINES_PATH,
    )
    sol_unified = _load_klines_unified(
        _SOL_KLINES_PATH, earliest_offset=earliest_offset,
        latest_offset=latest_offset, extended_path=_EXT_SOL_KLINES_PATH,
    )
    print(f"  BTC={len(btc_unified)} ETH={len(eth_unified)} SOL={len(sol_unified)}"
          f" load_elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    # ---------- min_bet from contract constants ----------
    try:
        from pancakebot.market_data.contract_constants import load_contract_constants
        cc = load_contract_constants()
        min_bet_amount_bnb = float(cc.min_bet_amount_bnb)
    except Exception:
        min_bet_amount_bnb = 0.001

    treasury_fee = 0.03
    window_days = resolved[0][1].risk.window_days

    # Determine total cohort round count (need this for restart schedules)
    sim_rounds_count = sum(
        1 for r in all_rounds
        if r.epoch >= COHORT_EPOCH_START and r.epoch <= COHORT_EPOCH_END
    )
    print(f"  cohort sim_rounds_count = {sim_rounds_count}", flush=True)

    # Restart schedules (1-indexed positions in sim_rounds)
    schedules = {
        "none": set(),
        "daily": restart_indices_periodic(sim_rounds_count, 288),
        "weekly": restart_indices_periodic(sim_rounds_count, 2016),
        "random_p_1_per_288": restart_indices_random(sim_rounds_count, 1.0/288.0, seed=42),
    }
    for k, v in schedules.items():
        print(f"  schedule {k}: {len(v)} restarts", flush=True)

    scales = [5.0, 50.0, 100.0]
    fixed_targets = [("95pct", 0.95), ("90pct", 0.90), ("85pct", 0.85), ("80pct", 0.80)]

    results: dict[str, Any] = {
        "cohort": {
            "epoch_start": COHORT_EPOCH_START,
            "epoch_end": COHORT_EPOCH_END,
            "sim_rounds_count": sim_rounds_count,
        },
        "schedules_summary": {k: len(v) for k, v in schedules.items()},
        "part_a_restart_fragility": [],
        "part_b_fixed_anchor": [],
        "part_a_baselines": [],  # V1-persistent, V0
    }

    # ---------- Part A baselines: V0, V1-persistent (== V1 single-run) ----------
    print("\n=== Part A baselines (no restart) ===", flush=True)
    for scale in scales:
        # V0
        spec = FoldSpec(
            name=f"v0_{scale:g}bnb_baseline",
            cutoff_seconds=2,
            epoch_start=COHORT_EPOCH_START,
            epoch_end=COHORT_EPOCH_END,
        )
        tracker = make_v0_tracker(scale, window_days)
        tag = f"v0/scale_{scale:g}bnb"
        out_dir = out_root / tag
        print(f"running {tag}...", flush=True)
        s = _run_fold_with_tracker(
            spec=spec, tracker=tracker, all_rounds=all_rounds,
            btc_unified=btc_unified, eth_unified=eth_unified, sol_unified=sol_unified,
            earliest_offset=earliest_offset, output_dir=out_dir,
            initial_bankroll_bnb=scale, treasury_fee_fraction=treasury_fee,
            min_bet_amount_bnb=min_bet_amount_bnb,
        )
        results["part_a_baselines"].append({
            "variant": "v0_rolling_7d", "scale_bnb": scale,
            "bets": s["num_bets"], "wins": s["num_wins"], "wr": s["win_rate"],
            "net_pnl_bnb": s["net_pnl_bnb"],
            "final_bankroll_bnb": s["final_bankroll_bnb"],
            "breaker_fires": s["breaker_fires"], "cooldown_rounds": s["cooldown_rounds"],
        })
        print(f"  V0 {scale:g}: bets={s['num_bets']} wr={s['win_rate']:.4f} "
              f"pnl={s['net_pnl_bnb']:+.4f} breaker={s['breaker_fires']} "
              f"cooldown={s['cooldown_rounds']}", flush=True)

        # V1-persistent (no restart) -- baseline that all restart patterns compare to
        spec = FoldSpec(
            name=f"v1_persistent_{scale:g}bnb",
            cutoff_seconds=2,
            epoch_start=COHORT_EPOCH_START,
            epoch_end=COHORT_EPOCH_END,
            strategy_overrides={"risk": {"dd_peak_mode": "absolute_ratchet"}},
        )
        tracker = make_v1_tracker(scale, window_days)
        tag = f"v1_persistent/scale_{scale:g}bnb"
        out_dir = out_root / tag
        print(f"running {tag}...", flush=True)
        s = _run_fold_with_tracker(
            spec=spec, tracker=tracker, all_rounds=all_rounds,
            btc_unified=btc_unified, eth_unified=eth_unified, sol_unified=sol_unified,
            earliest_offset=earliest_offset, output_dir=out_dir,
            initial_bankroll_bnb=scale, treasury_fee_fraction=treasury_fee,
            min_bet_amount_bnb=min_bet_amount_bnb,
        )
        results["part_a_baselines"].append({
            "variant": "v1_persistent_no_restart", "scale_bnb": scale,
            "bets": s["num_bets"], "wins": s["num_wins"], "wr": s["win_rate"],
            "net_pnl_bnb": s["net_pnl_bnb"],
            "final_bankroll_bnb": s["final_bankroll_bnb"],
            "breaker_fires": s["breaker_fires"], "cooldown_rounds": s["cooldown_rounds"],
        })
        print(f"  V1-persist {scale:g}: bets={s['num_bets']} wr={s['win_rate']:.4f} "
              f"pnl={s['net_pnl_bnb']:+.4f} breaker={s['breaker_fires']} "
              f"cooldown={s['cooldown_rounds']}", flush=True)

    # ---------- Part A: V1-in-memory under restart patterns ----------
    print("\n=== Part A: V1-in-memory under restart patterns ===", flush=True)
    for scale in scales:
        for sched_name, sched_set in schedules.items():
            spec = FoldSpec(
                name=f"v1_inmem_{scale:g}bnb_{sched_name}",
                cutoff_seconds=2,
                epoch_start=COHORT_EPOCH_START,
                epoch_end=COHORT_EPOCH_END,
                strategy_overrides={"risk": {"dd_peak_mode": "absolute_ratchet"}},
            )
            tracker = make_v1_restarting_tracker(scale, window_days)
            restart_count = [0]
            tag = f"v1_inmem/{sched_name}/scale_{scale:g}bnb"
            out_dir = out_root / tag
            print(f"running {tag}...", flush=True)
            s = _run_fold_with_tracker(
                spec=spec, tracker=tracker, all_rounds=all_rounds,
                btc_unified=btc_unified, eth_unified=eth_unified, sol_unified=sol_unified,
                earliest_offset=earliest_offset, output_dir=out_dir,
                initial_bankroll_bnb=scale, treasury_fee_fraction=treasury_fee,
                min_bet_amount_bnb=min_bet_amount_bnb,
                restart_round_indices=sched_set,
                restart_callback=tracker.simulate_restart,
                restart_count=restart_count,
            )
            results["part_a_restart_fragility"].append({
                "variant": "v1_inmem_restart", "scale_bnb": scale,
                "restart_pattern": sched_name,
                "scheduled_restarts": len(sched_set),
                "restart_count": s["restart_count"],
                "bets": s["num_bets"], "wins": s["num_wins"], "wr": s["win_rate"],
                "net_pnl_bnb": s["net_pnl_bnb"],
                "final_bankroll_bnb": s["final_bankroll_bnb"],
                "breaker_fires": s["breaker_fires"],
                "cooldown_rounds": s["cooldown_rounds"],
            })
            print(f"  V1-inmem {scale:g}/{sched_name}: bets={s['num_bets']} "
                  f"wr={s['win_rate']:.4f} pnl={s['net_pnl_bnb']:+.4f} "
                  f"breaker={s['breaker_fires']} cooldown={s['cooldown_rounds']} "
                  f"restarts={s['restart_count']}", flush=True)

    # ---------- Part B: V2 fixed-anchor sweep ----------
    print("\n=== Part B: V2 fixed-anchor sweep ===", flush=True)
    for scale in scales:
        for tag_name, target_pct in fixed_targets:
            target_bnb = scale * target_pct
            spec = FoldSpec(
                name=f"v2_fixed_{scale:g}bnb_{tag_name}",
                cutoff_seconds=2,
                epoch_start=COHORT_EPOCH_START,
                epoch_end=COHORT_EPOCH_END,
            )
            tracker = make_v2_fixed_tracker(scale, window_days, target_pct)
            tag = f"v2_fixed/{tag_name}/scale_{scale:g}bnb"
            out_dir = out_root / tag
            print(f"running {tag} target_bnb={target_bnb:.4f}...", flush=True)
            s = _run_fold_with_tracker(
                spec=spec, tracker=tracker, all_rounds=all_rounds,
                btc_unified=btc_unified, eth_unified=eth_unified, sol_unified=sol_unified,
                earliest_offset=earliest_offset, output_dir=out_dir,
                initial_bankroll_bnb=scale, treasury_fee_fraction=treasury_fee,
                min_bet_amount_bnb=min_bet_amount_bnb,
            )
            results["part_b_fixed_anchor"].append({
                "variant": "v2_fixed_anchor",
                "scale_bnb": scale,
                "target_pct": target_pct,
                "target_bnb": target_bnb,
                "bets": s["num_bets"], "wins": s["num_wins"], "wr": s["win_rate"],
                "net_pnl_bnb": s["net_pnl_bnb"],
                "final_bankroll_bnb": s["final_bankroll_bnb"],
                "breaker_fires": s["breaker_fires"],
                "cooldown_rounds": s["cooldown_rounds"],
            })
            print(f"  V2-fixed {scale:g}/{tag_name}: bets={s['num_bets']} "
                  f"wr={s['win_rate']:.4f} pnl={s['net_pnl_bnb']:+.4f} "
                  f"breaker={s['breaker_fires']} cooldown={s['cooldown_rounds']}",
                  flush=True)

    # ---------- write results ----------
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults: {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
