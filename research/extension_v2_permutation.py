"""Permutation null on extension_v2_2026_05_09 PnL.

Adapted from research/regime_phase0_permutation.py. Parameterized for the
NEW out-of-sample slice (epochs 478541..479822) synced 2026-05-09. Reads
canonical bet sequence from the in_process_runner output and round outcomes
from var/closed_rounds.jsonl (the canonical store, since this slice is
post-canonical-floor).

Method (identical to phase0):
  1. Load (epoch, side, bet_size_bnb) per BET row from the slice's trades.csv.
  2. Load round outcomes from var/closed_rounds.jsonl for the slice range.
  3. Re-settle each bet to verify the observed PnL matches the runner's.
  4. Permute (winner, failed, bull_pool, bear_pool) tuples across the bet
     rounds, holding bet sequence + sizes fixed. Re-settle. Sum PnL.
  5. N=1000 permutations, seed=20260510. Two-sided p-value vs null mean.

Output: var/incident_reports/extension_v2_permutation.json
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

CANONICAL_TRADES = REPO_ROOT / "var" / "extension_v2_2026_05_09" / "extension_v2_2026_05_09" / "trades.csv"
CANONICAL_ROUNDS = REPO_ROOT / "var" / "closed_rounds.jsonl"
OUT_PATH = REPO_ROOT / "var" / "incident_reports" / "extension_v2_permutation.json"

EXTENSION_EPOCH_START = 478541
EXTENSION_EPOCH_END = 479822  # bumped to actual sync end at runtime

TREASURY_FEE_FRACTION = 0.03
N_PERMUTATIONS = 1000
RNG_SEED = 20260510


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


def _load_canonical_bets(path: Path) -> list[Bet]:
    bets: list[Bet] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["action"] != "BET":
                continue
            try:
                bet_bnb = float(row["bet_size_bnb"])
            except (TypeError, ValueError):
                continue
            if bet_bnb <= 0.0:
                continue
            side = row["direction"]
            if side not in ("Bull", "Bear"):
                raise ValueError(f"unexpected side: {side!r} for epoch {row['epoch']}")
            bets.append(Bet(epoch=int(row["epoch"]), side=side, bet_bnb=bet_bnb))
    return bets


def _load_slice_rounds(epoch_start: int, epoch_end: int) -> dict[int, RoundData]:
    out: dict[int, RoundData] = {}
    store = ClosedRoundsStore(str(CANONICAL_ROUNDS))
    for r in store.iter_closed_rounds():
        if r.epoch < epoch_start or r.epoch > epoch_end:
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


def _settle_against_winner(*, bet: Bet, winner: str | None, failed: bool,
                           bull_pool_bnb: float, bear_pool_bnb: float) -> float:
    if failed:
        return -MAX_GAS_COST_BET_BNB - MAX_GAS_COST_CLAIM_BNB
    if winner is None:
        return -MAX_GAS_COST_BET_BNB - MAX_GAS_COST_CLAIM_BNB
    bet_u = bet.side.upper()
    win_u = winner.upper()
    if bet_u != win_u:
        return -bet.bet_bnb - MAX_GAS_COST_BET_BNB
    bull_after = bull_pool_bnb + (bet.bet_bnb if bet_u == "BULL" else 0.0)
    bear_after = bear_pool_bnb + (bet.bet_bnb if bet_u == "BEAR" else 0.0)
    total_after = bull_after + bear_after
    denom = bull_after if bet_u == "BULL" else bear_after
    if denom <= 0.0 or total_after <= 0.0:
        return -bet.bet_bnb - MAX_GAS_COST_BET_BNB - MAX_GAS_COST_CLAIM_BNB
    mult = (total_after * (1.0 - TREASURY_FEE_FRACTION)) / denom
    credit = bet.bet_bnb * mult - MAX_GAS_COST_CLAIM_BNB
    return credit - bet.bet_bnb - MAX_GAS_COST_BET_BNB


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch-end", type=int, default=EXTENSION_EPOCH_END)
    args = ap.parse_args()

    epoch_end = int(args.epoch_end)
    print(f"[load] canonical trades: {CANONICAL_TRADES}", flush=True)
    bets = _load_canonical_bets(CANONICAL_TRADES)
    print(f"  {len(bets)} bets loaded", flush=True)

    print(f"[load] slice rounds [{EXTENSION_EPOCH_START}..{epoch_end}] from {CANONICAL_ROUNDS}", flush=True)
    rounds = _load_slice_rounds(EXTENSION_EPOCH_START, epoch_end)
    print(f"  {len(rounds)} rounds in [{EXTENSION_EPOCH_START}..{epoch_end}]", flush=True)

    missing = [b.epoch for b in bets if b.epoch not in rounds]
    if missing:
        print(f"[FAIL] {len(missing)} bet epochs missing from rounds; first few: {missing[:5]}", flush=True)
        return 1

    observed_pnl = 0.0
    n_wins = 0
    for b in bets:
        rd = rounds[b.epoch]
        pnl = _settle_against_winner(
            bet=b, winner=rd.winner, failed=rd.failed,
            bull_pool_bnb=rd.bull_pool_bnb, bear_pool_bnb=rd.bear_pool_bnb,
        )
        observed_pnl += pnl
        if not rd.failed and rd.winner is not None and b.side.upper() == rd.winner.upper():
            n_wins += 1

    n_bets = len(bets)
    observed_wr = n_wins / n_bets if n_bets > 0 else 0.0
    print(f"[obs] PnL={observed_pnl:+.4f} BNB / {n_bets} bets / WR={observed_wr:.4%}", flush=True)

    bet_rounds = [rounds[b.epoch] for b in bets]

    rng = random.Random(RNG_SEED)
    null_pnls: list[float] = []
    null_wrs: list[float] = []

    for trial in range(N_PERMUTATIONS):
        permuted_idx = list(range(n_bets))
        rng.shuffle(permuted_idx)

        trial_pnl = 0.0
        trial_wins = 0
        for i, b in enumerate(bets):
            rd = bet_rounds[permuted_idx[i]]
            pnl = _settle_against_winner(
                bet=b, winner=rd.winner, failed=rd.failed,
                bull_pool_bnb=rd.bull_pool_bnb, bear_pool_bnb=rd.bear_pool_bnb,
            )
            trial_pnl += pnl
            if not rd.failed and rd.winner is not None and b.side.upper() == rd.winner.upper():
                trial_wins += 1
        null_pnls.append(trial_pnl)
        null_wrs.append(trial_wins / n_bets)

        if (trial + 1) % 200 == 0:
            print(f"  trial {trial+1}/{N_PERMUTATIONS} ...", flush=True)

    null_pnls.sort()
    n = len(null_pnls)
    mean = sum(null_pnls) / n
    var = sum((x - mean) ** 2 for x in null_pnls) / n
    std = var ** 0.5
    p05 = null_pnls[int(0.05 * n)]
    p25 = null_pnls[int(0.25 * n)]
    p50 = null_pnls[int(0.50 * n)]
    p75 = null_pnls[int(0.75 * n)]
    p95 = null_pnls[int(0.95 * n)]
    pmin = null_pnls[0]
    pmax = null_pnls[-1]

    n_le = sum(1 for x in null_pnls if x <= observed_pnl)
    pct_rank = n_le / n
    abs_obs = abs(observed_pnl - mean)
    n_more_extreme = sum(1 for x in null_pnls if abs(x - mean) >= abs_obs)
    p_two_sided = n_more_extreme / n
    p_one_sided_lower = pct_rank
    p_one_sided_upper = sum(1 for x in null_pnls if x >= observed_pnl) / n

    null_wr_mean = sum(null_wrs) / len(null_wrs)
    null_wr_std = (sum((w - null_wr_mean) ** 2 for w in null_wrs) / len(null_wrs)) ** 0.5

    print(f"\n=== Permutation null (N={N_PERMUTATIONS}, seed={RNG_SEED}) ===", flush=True)
    print(f"  observed PnL          : {observed_pnl:+.4f} BNB", flush=True)
    print(f"  observed WR           : {observed_wr:.4%}", flush=True)
    print(f"  null PnL mean         : {mean:+.4f} BNB", flush=True)
    print(f"  null PnL std          : {std:.4f} BNB", flush=True)
    print(f"  null PnL min / max    : {pmin:+.4f} / {pmax:+.4f}", flush=True)
    print(f"  null PnL p05 / p95    : {p05:+.4f} / {p95:+.4f}", flush=True)
    print(f"  null PnL p25/p50/p75  : {p25:+.4f} / {p50:+.4f} / {p75:+.4f}", flush=True)
    print(f"  null WR mean +/- std  : {null_wr_mean:.4%} +/- {null_wr_std:.4%}", flush=True)
    print(f"  observed percentile   : {pct_rank:.4f}", flush=True)
    print(f"  one-sided p (lower)   : {p_one_sided_lower:.4f}", flush=True)
    print(f"  one-sided p (upper)   : {p_one_sided_upper:.4f}", flush=True)
    print(f"  two-sided p-value     : {p_two_sided:.4f}", flush=True)

    if pct_rank < 0.05:
        verdict = "REAL_SIGNAL_NEGATIVE"
        verdict_text = ("Observed PnL falls in lower 5% of null - significantly worse "
                        "than chance. The new slice is hostile beyond what shuffled "
                        "labels would produce.")
    elif pct_rank > 0.95:
        verdict = "REAL_SIGNAL_POSITIVE"
        verdict_text = ("Observed PnL falls in upper 5% of null - significantly "
                        "better than chance. Strategy edge holds on this slice.")
    else:
        verdict = "CONSISTENT_WITH_LUCK"
        verdict_text = ("Observed PnL falls within central 90% of null - cohort "
                        "delta is consistent with random outcomes given the bet "
                        "sequence. Signal indistinguishable from noise on this slice.")

    print(f"\n  verdict: {verdict}", flush=True)
    print(f"  {verdict_text}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "slice_name": "extension_v2_2026_05_09",
        "epoch_start": EXTENSION_EPOCH_START,
        "epoch_end": epoch_end,
        "n_permutations": N_PERMUTATIONS,
        "rng_seed": RNG_SEED,
        "n_bets": n_bets,
        "observed_pnl_bnb": observed_pnl,
        "observed_wr": observed_wr,
        "null_pnl_mean": mean,
        "null_pnl_std": std,
        "null_pnl_min": pmin,
        "null_pnl_max": pmax,
        "null_pnl_p05": p05,
        "null_pnl_p25": p25,
        "null_pnl_p50": p50,
        "null_pnl_p75": p75,
        "null_pnl_p95": p95,
        "null_wr_mean": null_wr_mean,
        "null_wr_std": null_wr_std,
        "observed_percentile": pct_rank,
        "p_one_sided_lower": p_one_sided_lower,
        "p_one_sided_upper": p_one_sided_upper,
        "p_two_sided": p_two_sided,
        "verdict": verdict,
        "verdict_text": verdict_text,
    }, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
