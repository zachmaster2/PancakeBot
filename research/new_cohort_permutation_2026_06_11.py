"""Permutation nulls for the two NEW post-sync cohorts (read-only research).

The 2026-06-10 sync + ``research/post_cv5_to_current_2026_06_10.py`` run
produced two never-tested negative slices:

  latest      @ 5  BNB : epochs 484409..487686, 82 bets, WR 53.66%, -1.0991 BNB
  vm_live_era @ 50 BNB : epochs 487687..488832, 46 bets, WR 45.65%, -2.9332 BNB

(vm_live_era @ 5 BNB placed ZERO bets — the sequential drawdown breaker +
cooldown suppressed the whole cohort after `latest`'s losses, so there is
nothing to permute at that scale.)

Method — identical to ``research/extension_v2_permutation.py``: hold the bet
sequence (epoch order, sides, sizes) fixed; permute the round-outcome tuples
(winner, failed, final pools) across the bet rounds; re-settle; N=1000.
Verdict per slice from the observed PnL's percentile in the null:
  < p05  -> REAL_SIGNAL_NEGATIVE (slice hostile beyond shuffled-label chance)
  > p95  -> REAL_SIGNAL_POSITIVE (edge holds)
  else   -> CONSISTENT_WITH_LUCK

Validity gate per slice: the re-settled observed PnL must match the step10b
runner's cohort PnL (tolerance 0.01 BNB) before the null is trusted.

Inputs: bet sequences preserved at
``var/strategy_review/post_cv5_to_current_2026_06_10/trades_{5bnb,50bnb}.csv``;
outcomes from ``var/closed_rounds.jsonl``.

Run:  cd <repo> && .venv/Scripts/python.exe research/new_cohort_permutation_2026_06_11.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pancakebot.constants import BNB_WEI, MAX_GAS_COST_BET_BNB, MAX_GAS_COST_CLAIM_BNB
from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.pool_amounts import compute_pool_amounts_wei

TRADES_DIR = REPO_ROOT / "var" / "strategy_review" / "post_cv5_to_current_2026_06_10"
CANONICAL_ROUNDS = REPO_ROOT / "var" / "closed_rounds.jsonl"
OUT_PATH = REPO_ROOT / "var" / "incident_reports" / "new_cohort_permutation_2026_06_11.json"

TREASURY_FEE_FRACTION = 0.03
N_PERMUTATIONS = 1000
RNG_SEED = 20260611

# (slice_name, trades_csv, epoch_start, epoch_end, runner_cohort_pnl)
SLICES = [
    ("latest_5bnb", TRADES_DIR / "trades_5bnb.csv", 484409, 487686, -1.0991),
    ("vm_live_era_50bnb", TRADES_DIR / "trades_50bnb.csv", 487687, 488832, -2.9332),
]


@dataclass(frozen=True, slots=True)
class Bet:
    epoch: int
    side: str
    bet_bnb: float


@dataclass(frozen=True, slots=True)
class RoundData:
    epoch: int
    winner: str | None
    failed: bool
    bull_pool_bnb: float
    bear_pool_bnb: float


def _load_bets(path: Path, lo: int, hi: int) -> list[Bet]:
    bets: list[Bet] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["action"] != "BET":
                continue
            epoch = int(row["epoch"])
            if epoch < lo or epoch > hi:
                continue
            bet_bnb = float(row["bet_size_bnb"])
            if bet_bnb <= 0.0:
                continue
            side = row["direction"]
            if side not in ("Bull", "Bear"):
                raise ValueError(f"unexpected side {side!r} for epoch {epoch}")
            bets.append(Bet(epoch=epoch, side=side, bet_bnb=bet_bnb))
    return bets


def _load_slice_rounds(lo: int, hi: int) -> dict[int, RoundData]:
    out: dict[int, RoundData] = {}
    store = ClosedRoundsStore(str(CANONICAL_ROUNDS))
    for r in store.iter_closed_rounds():
        if r.epoch < lo or r.epoch > hi:
            continue
        pools = compute_pool_amounts_wei(bets=r.bets) if not r.failed else None
        out[int(r.epoch)] = RoundData(
            epoch=int(r.epoch),
            winner=r.position,
            failed=bool(r.failed),
            bull_pool_bnb=(pools.bull_wei / BNB_WEI) if pools is not None else 0.0,
            bear_pool_bnb=(pools.bear_wei / BNB_WEI) if pools is not None else 0.0,
        )
    return out


def _settle(*, bet: Bet, rd: RoundData) -> float:
    if rd.failed or rd.winner is None:
        return -MAX_GAS_COST_BET_BNB - MAX_GAS_COST_CLAIM_BNB
    bet_u, win_u = bet.side.upper(), rd.winner.upper()
    if bet_u != win_u:
        return -bet.bet_bnb - MAX_GAS_COST_BET_BNB
    bull_after = rd.bull_pool_bnb + (bet.bet_bnb if bet_u == "BULL" else 0.0)
    bear_after = rd.bear_pool_bnb + (bet.bet_bnb if bet_u == "BEAR" else 0.0)
    total_after = bull_after + bear_after
    denom = bull_after if bet_u == "BULL" else bear_after
    if denom <= 0.0 or total_after <= 0.0:
        return -bet.bet_bnb - MAX_GAS_COST_BET_BNB - MAX_GAS_COST_CLAIM_BNB
    mult = (total_after * (1.0 - TREASURY_FEE_FRACTION)) / denom
    credit = bet.bet_bnb * mult - MAX_GAS_COST_CLAIM_BNB
    return credit - bet.bet_bnb - MAX_GAS_COST_BET_BNB


def _run_slice(name: str, trades: Path, lo: int, hi: int,
               runner_pnl: float) -> dict:
    bets = _load_bets(trades, lo, hi)
    rounds = _load_slice_rounds(lo, hi)
    missing = [b.epoch for b in bets if b.epoch not in rounds]
    if missing:
        raise SystemExit(f"[{name}] {len(missing)} bet epochs missing from rounds")

    observed = 0.0
    wins = 0
    for b in bets:
        pnl = _settle(bet=b, rd=rounds[b.epoch])
        observed += pnl
        rd = rounds[b.epoch]
        if not rd.failed and rd.winner is not None and b.side.upper() == rd.winner.upper():
            wins += 1
    n = len(bets)
    wr = wins / n if n else 0.0
    print(f"\n[{name}] {n} bets, re-settled PnL {observed:+.4f} BNB (runner {runner_pnl:+.4f}), WR {wr:.2%}")
    if abs(observed - runner_pnl) > 0.01:
        raise SystemExit(f"[{name}] VALIDITY FAIL: re-settle {observed:+.4f} != runner {runner_pnl:+.4f}")

    bet_rounds = [rounds[b.epoch] for b in bets]
    rng = random.Random(RNG_SEED)
    null_pnls: list[float] = []
    for _ in range(N_PERMUTATIONS):
        idx = list(range(n))
        rng.shuffle(idx)
        t = 0.0
        for i, b in enumerate(bets):
            t += _settle(bet=b, rd=bet_rounds[idx[i]])
        null_pnls.append(t)

    null_pnls.sort()
    m = len(null_pnls)
    mean = sum(null_pnls) / m
    std = (sum((x - mean) ** 2 for x in null_pnls) / m) ** 0.5
    pct_rank = sum(1 for x in null_pnls if x <= observed) / m
    p_upper = sum(1 for x in null_pnls if x >= observed) / m
    if pct_rank < 0.05:
        verdict = "REAL_SIGNAL_NEGATIVE"
    elif pct_rank > 0.95:
        verdict = "REAL_SIGNAL_POSITIVE"
    else:
        verdict = "CONSISTENT_WITH_LUCK"

    print(f"  null mean {mean:+.4f} +/- {std:.4f} | p05 {null_pnls[int(0.05*m)]:+.4f} "
          f"p50 {null_pnls[int(0.50*m)]:+.4f} p95 {null_pnls[int(0.95*m)]:+.4f}")
    print(f"  observed percentile {pct_rank:.4f} | p(lower) {pct_rank:.4f} "
          f"p(upper) {p_upper:.4f}")
    print(f"  verdict: {verdict}")
    return {
        "slice": name, "epoch_start": lo, "epoch_end": hi, "n_bets": n,
        "observed_pnl_bnb": observed, "observed_wr": wr,
        "null_pnl_mean": mean, "null_pnl_std": std,
        "null_p05": null_pnls[int(0.05 * m)], "null_p50": null_pnls[int(0.50 * m)],
        "null_p95": null_pnls[int(0.95 * m)],
        "observed_percentile": pct_rank, "p_one_sided_lower": pct_rank,
        "p_one_sided_upper": p_upper, "verdict": verdict,
    }


def main() -> int:
    results = [
        _run_slice(name, trades, lo, hi, pnl)
        for name, trades, lo, hi, pnl in SLICES
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "n_permutations": N_PERMUTATIONS, "rng_seed": RNG_SEED,
        "slices": results,
    }, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
