# Phase-0 OKX perp probe — findings (2026-06-12)

Scripts: `research/phase0_okx_perp_capture_2026_06_11.py` (capture) and
`research/phase0_okx_perp_2026_06_11.py` (harness; pre-registration pinned
in commit `718eed6` BEFORE the completed-capture run). Artifacts:
`var/strategy_review/phase0_okx_perp_2026_06_11/`. Settlement: flat stake,
realized final-pool payouts, 3% fee, no gas. Eras: golden ≤479952, fade
479953–484408, dead ≥484409 (+ latest/vm_live consistency split).

Data captured (var/extended/, gitignored; capture script is provenance):
funding 276×8h (2026-03-12→now), daily OI 180d (2025-12-14→now), perp tape
6.46M trades (2026-05-09→now, one contiguous segment), spot tape 3.57M
trades (2026-02-25→now, archive+gap-fill merged, one contiguous segment).

## Bottom line

**All backfillable OKX candidates are RULED OUT.** The pre-registered
primary is null; the best exploratory cell in the deployment era fails the
sweep correction at p=0.64; the one striking mid-capture curiosity
dissolved on full data. With this, the search-the-existing-envelope
program is COMPLETE: every signal family reachable from data the bot can
see or backfill — momentum, pool flow, contrarian, mild-momentum fade,
wallet smart-money, and now perp funding/OI/trade-tape — has been tested
with runtime feasibility enforced, and none yields a deployable edge.

## Pre-registered primary: NULL

Perp taker-flow imbalance, 5m window ending lock−2s, sign-follow, dead era
(pinned with direction + statistic in `718eed6`): n=4,422, mean PnL
−0.0265 vs permutation null −0.0296, **deficit +0.0031/bet, p_upper=0.40**.

Program-honesty caveat (adversarial-panel mandated): this is ~the 4th
single pre-registered no-discount test against the same dead-era outcomes
(mild-momentum fade, wallet smart-money, contrarian monitor, this) —
program-wise, 4–6 one-shot tests at α=0.05 carry ~20% family-wise
false-positive odds. A future "significant" primary must be read against
that running count; this one was null regardless.

## Exploratory sweep: nothing survives

75 cells examined (5 eras × {funding, funding_delta, oi_d1, 12 tape
variants}); effect size = deficit vs the permutation null (a
sign-strategy's null expectation is the structural fee + majority
discount ≈ −0.03/bet, NOT zero — z-vs-zero is never quoted); funding/OI
cells clustered by 8h-period/day. Best dead-era cell: funding_delta
(deficit +0.0388, raw p=0.0136) → **Šidák over 75: p=0.64**. Within-dead
consistency also fails it: latest +0.0479 vs vm_live +0.0031.

Šidák notes: conservative under the cells' positive dependence, and the
exponent counts all eras (incl. golden/fade cells the selection rule can't
pick) — but the verdict is family-invariant: even over the 15 dead_all
cells alone, p≈0.19. The clustered iid shuffle's exchangeability
assumption was checked empirically for the only near-significant clustered
candidate (funding_delta sign is mean-reverting, making the null
conservative, not anti-conservative).

## The tape-imbalance story (the most data-rich null)

Within every cohort the two venues agree; across adjacent cohorts the sign
flips:

| cohort | perp | spot | lean |
|---|---|---|---|
| golden (Feb-25→May-10 spot only) | — | deficits −0.008…−0.024 | fade |
| fade era (May-10→26) | −0.021…−0.040 (p_lo to 0.0025) | −0.004…−0.041 | fade |
| dead_latest (May-26→Jun-6) | +0.020…+0.039 (p_up to 0.017) | +0.014…+0.041 (p_up to 0.011) | **follow** |
| dead_vmlive (Jun-6→10) | −0.026…−0.064 | −0.024…−0.054 | fade |
| **dead_all (deployment)** | **−0.019…+0.018, all p≥0.09** | **−0.023…+0.024, all p≥0.055** | **null** |

Venue-coherent within cohort, sign-flipping across cohorts at week scale =
the measurement is consistent but the phenomenon is non-stationary;
trading it requires predicting the regime, which is the unsolved problem
itself. The deployment-era aggregate is null because its halves cancel.

Cautionary exhibit (why the discipline exists): mid-capture, the spot
tape's leading edge covered only the June-10 tail (n=147, one half-day,
one regime) and showed "fade spot 5m" at deficit −0.233, p=0.0016. On
full coverage the same cell is **+0.003, p=0.42**.

## Funding / OI

- Funding level: null everywhere (best raw p=0.048 at dead_latest,
  vm_live flips sign). 8h cadence ⇒ each value spans ~96 rounds; only
  ~48 clusters even exist in the dead era — power is structurally poor.
  Matches the archived BTC-funding probe (25,543 rounds, flat deciles).
- Funding delta: the sweep's best cell, see above. Not significant, not
  consistent within dead.
- Daily OI change: null in every era (|deficit| ≤ 0.024, all p ≥ 0.15).
  The panel's look-ahead claim against rubik rows was REFUTED empirically
  (rows are end-anchored and immutable; the feature is causal).

## Runtime feasibility (for the record)

All tested candidates were runtime-TRIVIAL (VM→OKX 225–245ms: funding/OI
one GET at preflight; tape via incremental per-round accumulation closing
at lock−2s alongside the kline fetch). Feasibility was never the blocker —
the signals don't exist.

## Not testable historically (forward-capture only) — flagged per dispatch

Order-book imbalance/depth (candidates c/d) and liquidations (g) have NO
backfill (snapshot-only / ~16h retention); 5m-OI retains ~2 days.
Proposed forward capture if ever wanted: a standalone VM systemd timer
(not the bot) snapshotting `market/books` (400 levels), current OI, and
`liquidation-orders` once per round at preflight cadence (~290/day,
~25MB/day raw, trivially rate-limited), checkpointed like this capture.
Power math: a dead-era-sized cohort needs ~6–8 weeks of forward data
before a Phase-0 test means anything. This is a user decision; nothing is
deployed or scheduled.

## What remains (user decisions, unchanged from the wallet/fade findings)

1. Forward capture of the unseen classes above (weeks of lead time).
2. Strategy-class pivot (the last-seconds game; different adversaries,
   outside current architecture/risk appetite).
3. Wait-and-monitor (zero-cost offline monitors per sync; revive the
   validated machinery on a regime turn).

The existing + backfillable data envelope is, as of this probe, fully
characterized: **no deployable strategy in it.**
