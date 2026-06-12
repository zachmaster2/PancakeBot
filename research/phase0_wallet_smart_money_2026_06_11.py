"""Phase-0: wallet smart-money, early-bettor-conditioned. READ-ONLY.

Hypothesis (pre-registered): historically-accurate wallets bet early enough
to be visible at executable horizons (lock-6s / -4s / -3s), and their net
flow predicts the round outcome where aggregate flow does not.

Method (strict no-look-ahead, single chronological pass):
  - Per-wallet trailing accuracy over the last TRAIL_N settled bets,
    shrunk toward 0.5 with a Beta(PRIOR_S, PRIOR_S) prior:
        acc_hat = (wins + PRIOR_S) / (n + 2*PRIOR_S)
    A wallet is SMART at bet time iff n >= MIN_N and acc_hat >= SMART_THR.
    State updates happen AFTER the prediction round's features are taken
    (round R uses outcomes of rounds < R only).
  - Per round R and horizon h in {6,4,3}s: smart net flow = signed BNB
    (bull +, bear -) of bets with created_at < lock - h from smart wallets.
    Signal: side = sign(smart net flow), requiring >= MIN_SMART_WALLETS
    distinct smart wallets visible and |net| >= MIN_SMART_BNB.
  - Settlement: flat stake 1 at REALIZED final-pool payouts, 3% fee, no gas
    (infinitesimal-stake limit; era-relative valid).
  - PRIMARY definition pre-registered: TRAIL_N=100, PRIOR_S=10, MIN_N=30,
    SMART_THR=0.55, MIN_SMART_WALLETS=2, MIN_SMART_BNB=0.05.
    Two sensitivity variants are reported with an explicit
    multiple-comparisons flag (they are NOT independent confirmations).
  - Accuracy-persistence null: does trailing acc_hat predict the wallet's
    NEXT bet outcome at all? Forward hit-rate by acc_hat bucket; if flat,
    "smart wallets" is itself noise and the candidate dies regardless of
    flow tests.
  - Bet-timing distributions (the feasibility question): share of bets
    visible at -6s/-4s/-3s for smart vs all wallets.
  - Permutation null (N=1000): dead-era outcomes shuffled across the
    signal rounds; one-sided p for the observed mean PnL.

Eras: golden <=479952, fade 479953..484408, dead >=484409 (2026-05-26+).

Outputs: var/strategy_review/phase0_wallet_2026_06_11/findings.json +
console digest. Findings doc:
research/phase0_wallet_fade_2026_06_11_findings.md (shared with the fade).

Run:  cd <repo> && .venv/Scripts/python.exe research/phase0_wallet_smart_money_2026_06_11.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.in_process_runner as ipr  # noqa: E402
from pancakebot.constants import BNB_WEI  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402

OUT = REPO / "var" / "strategy_review" / "phase0_wallet_2026_06_11"
FEE = 0.03
HORIZONS = (6, 4, 3)

# PRIMARY (pre-registered)
TRAIL_N = 100
PRIOR_S = 10
MIN_N = 30
SMART_THR = 0.55
MIN_SMART_WALLETS = 2
MIN_SMART_BNB = 0.05

# Sensitivity variants (multiple-comparisons flagged; NOT confirmations)
VARIANTS = {
    "primary": dict(min_n=30, thr=0.55),
    "strict": dict(min_n=50, thr=0.56),
    "loose": dict(min_n=20, thr=0.54),
}

ERAS = [
    ("golden", 437562, 479952),
    ("fade", 479953, 484408),
    ("dead", 484409, 488832),
]


def era_of(epoch: int) -> str:
    for name, lo, hi in ERAS:
        if lo <= epoch <= hi:
            return name
    return "other"


class WalletState:
    __slots__ = ("outcomes", "wins")

    def __init__(self):
        self.outcomes: deque = deque(maxlen=TRAIL_N)
        self.wins: int = 0  # wins inside the deque window

    def acc(self) -> tuple[float, int]:
        n = len(self.outcomes)
        return (self.wins + PRIOR_S) / (n + 2 * PRIOR_S), n

    def update(self, won: bool) -> None:
        if len(self.outcomes) == self.outcomes.maxlen:
            old = self.outcomes[0]
            if old:
                self.wins -= 1
        self.outcomes.append(won)
        if won:
            self.wins += 1


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("--- loading rounds ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=False)
    all_rounds = [r for r in all_rounds if r.position in ("Bull", "Bear")]
    all_rounds.sort(key=lambda r: r.epoch)
    print(f"  {len(all_rounds)} usable rounds")

    wallets: dict[str, WalletState] = defaultdict(WalletState)
    rows: list[dict] = []
    persistence: list[tuple[float, int, bool]] = []  # (acc_hat, n, won)
    timing_all = np.zeros(4)    # bets visible at >=6s / 4-6s / 3-4s / <3s
    timing_smart = np.zeros(4)

    print("--- chronological pass ---", flush=True)
    for r in all_rounds:
        lock = int(r.lock_at)
        winner_bull = r.position == "Bull"
        pools_f = compute_pool_amounts_wei(bets=r.bets)
        f_bull = pools_f.bull_wei / BNB_WEI
        f_bear = pools_f.bear_wei / BNB_WEI
        if f_bull <= 0 or f_bear <= 0:
            # still update state below; no signal row
            payable = False
        else:
            payable = True

        # ---- features from PRIOR state only ----
        per_variant = {v: {h: dict(net=0.0, wallets=set()) for h in HORIZONS}
                       for v in VARIANTS}
        for b in r.bets:
            offset = lock - int(b.created_at)
            st = wallets.get(b.wallet_address)
            acc_hat, n = st.acc() if st is not None else (0.5, 0)
            amt = float(b.amount_wei) / BNB_WEI
            signed = amt if b.position == "Bull" else -amt
            # timing histogram (all + primary-smart)
            bucket = 0 if offset >= 6 else (1 if offset >= 4 else (2 if offset >= 3 else 3))
            timing_all[bucket] += 1
            if n >= MIN_N and acc_hat >= SMART_THR:
                timing_smart[bucket] += 1
            for vname, vp in VARIANTS.items():
                if n >= vp["min_n"] and acc_hat >= vp["thr"]:
                    for h in HORIZONS:
                        if offset >= h:
                            per_variant[vname][h]["net"] += signed
                            per_variant[vname][h]["wallets"].add(b.wallet_address)
            # persistence sample (primary eligibility only, before update)
            if n >= MIN_N:
                persistence.append(
                    (acc_hat, n, (b.position == "Bull") == winner_bull))

        if payable:
            row = dict(
                epoch=int(r.epoch), era=era_of(int(r.epoch)),
                outcome_bull=winner_bull,
                payout_bull=(f_bull + f_bear) * (1 - FEE) / f_bull,
                payout_bear=(f_bull + f_bear) * (1 - FEE) / f_bear,
            )
            for vname in VARIANTS:
                for h in HORIZONS:
                    d = per_variant[vname][h]
                    row[f"{vname}_net_{h}"] = d["net"]
                    row[f"{vname}_nw_{h}"] = len(d["wallets"])
            rows.append(row)

        # ---- AFTER features: settle this round into wallet state ----
        for b in r.bets:
            wallets[b.wallet_address].update((b.position == "Bull") == winner_bull)

    print(f"  pass done in {time.time()-t0:.0f}s; {len(rows)} signal rows, "
          f"{len(persistence)} persistence samples")

    # ---- accuracy persistence: forward hit-rate by acc_hat bucket ----
    pers = np.array([(a, w) for a, n, w in persistence])
    buckets = [0.50, 0.52, 0.54, 0.56, 0.58, 1.01]
    pers_table = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        m = (pers[:, 0] >= lo) & (pers[:, 0] < hi)
        if m.sum() > 0:
            fw = float(pers[m, 1].mean())
            pers_table.append(dict(acc_bucket=f"[{lo:.2f},{hi:.2f})",
                                   n=int(m.sum()), forward_hit=round(fw, 4)))

    # ---- strategy tables per variant x horizon x era ----
    def settle(row, side_bull):
        win = row["outcome_bull"] == side_bull
        pay = row["payout_bull"] if side_bull else row["payout_bear"]
        return (pay - 1.0) if win else -1.0

    results = {}
    dead_bet_rows = {}
    for vname in VARIANTS:
        for h in HORIZONS:
            key = f"{vname}_h{h}"
            res = {}
            for era in ("golden", "fade", "dead"):
                pnls, wins, pays = [], [], []
                for row in rows:
                    if row["era"] != era:
                        continue
                    net = row[f"{vname}_net_{h}"]
                    if row[f"{vname}_nw_{h}"] < MIN_SMART_WALLETS or abs(net) < MIN_SMART_BNB:
                        continue
                    side = net > 0
                    pnls.append(settle(row, side))
                    wins.append(row["outcome_bull"] == side)
                    pays.append(row["payout_bull"] if side else row["payout_bear"])
                n = len(pnls)
                if n == 0:
                    res[era] = dict(n=0)
                    continue
                mean = float(np.mean(pnls))
                se = float(np.std(pnls) / np.sqrt(n)) if n > 1 else 0.0
                res[era] = dict(
                    n=n, wr=round(float(np.mean(wins)), 4),
                    breakeven_wr=round(float(np.mean([1 / p for p in pays])), 4),
                    mean_pnl=round(mean, 4),
                    z=round(mean / se, 2) if se > 0 else None)
                if era == "dead" and vname == "primary":
                    dead_bet_rows[h] = [
                        (row, row[f"{vname}_net_{h}"] > 0) for row in rows
                        if row["era"] == "dead"
                        and row[f"{vname}_nw_{h}"] >= MIN_SMART_WALLETS
                        and abs(row[f"{vname}_net_{h}"]) >= MIN_SMART_BNB]
            results[key] = res

    # ---- permutation null on PRIMARY dead-era cells ----
    perms = {}
    rng = random.Random(20260611)
    for h, pairs in dead_bet_rows.items():
        if len(pairs) < 20:
            perms[f"h{h}"] = dict(n=len(pairs), note="too few rounds")
            continue
        obs = float(np.mean([settle(row, side) for row, side in pairs]))
        outs = [(row["outcome_bull"], row["payout_bull"], row["payout_bear"])
                for row, _ in pairs]
        sides = [side for _, side in pairs]
        null = []
        for _ in range(1000):
            shuffled = outs[:]
            rng.shuffle(shuffled)
            tot = 0.0
            for (ob, pb, pr), side in zip(shuffled, sides):
                win = ob == side
                pay = pb if side else pr
                tot += (pay - 1.0) if win else -1.0
            null.append(tot / len(sides))
        null.sort()
        p_upper = sum(1 for x in null if x >= obs) / len(null)
        perms[f"h{h}"] = dict(n=len(pairs), obs_mean_pnl=round(obs, 4),
                              p_upper=round(p_upper, 4))

    timing = dict(
        all_bets=dict(zip([">=6s", "4-6s", "3-4s", "<3s"],
                          (timing_all / max(1, timing_all.sum())).round(4).tolist())),
        smart_bets=dict(zip([">=6s", "4-6s", "3-4s", "<3s"],
                            (timing_smart / max(1, timing_smart.sum())).round(4).tolist())),
        smart_bet_count=int(timing_smart.sum()),
    )

    findings = dict(
        params=dict(TRAIL_N=TRAIL_N, PRIOR_S=PRIOR_S, MIN_N=MIN_N,
                    SMART_THR=SMART_THR, MIN_SMART_WALLETS=MIN_SMART_WALLETS,
                    MIN_SMART_BNB=MIN_SMART_BNB, variants=VARIANTS),
        persistence=pers_table, timing=timing, results=results,
        permutation_dead_primary=perms,
    )
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2),
                                       encoding="utf-8")

    print("\n=== accuracy persistence (forward hit-rate by trailing acc bucket) ===")
    for row in pers_table:
        print(f"  {row['acc_bucket']}: forward_hit={row['forward_hit']} (n={row['n']})")
    print("\n=== bet timing (share of bets by visibility) ===")
    print(json.dumps(timing, indent=1))
    print("\n=== strategy results (variant x horizon x era) ===")
    for key, res in results.items():
        parts = []
        for era in ("golden", "fade", "dead"):
            e = res[era]
            parts.append(f"{era}: {e.get('mean_pnl')}(n={e.get('n')},z={e.get('z')})")
        print(f"  {key}: " + " | ".join(parts))
    print("\n=== permutation (primary, dead era) ===")
    print(json.dumps(perms, indent=1))
    print(f"\n[done] {time.time()-t0:.0f}s; artifacts in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
