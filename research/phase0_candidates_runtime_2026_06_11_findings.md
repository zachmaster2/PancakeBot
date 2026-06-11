# Phase-0 candidate findings, runtime-feasibility-weighted (2026-06-11)

Harness: `research/phase0_candidates_runtime_2026_06_11.py` (reproducible;
artifacts in `var/strategy_review/phase0_candidates_2026_06_11/`).
Settlement: flat stake, realized final-pool payouts, 3% fee, no gas
(era-relative comparisons valid; see post-mortem caveats). Eras:
golden ≤479952, fade 479953–484408, dead ≥484409 (post-2026-05-26).

## The runtime envelope (measured constants)

Pool state-as-of `lock−h` is only READABLE at
`(lock−h) + 450ms (block) + 625ms (availability p99) + 250–600ms (getLogs)`,
and must be read-complete by `lock−869ms` (789 submit deadline + 50 compute
+ 30 sign). Data resolution note: bet timestamps are integer seconds, so
horizons are meaningful only at whole seconds (the h=2.5s row in the sweep
duplicates h=3.0s).

| Horizon | Class | Worst-wall headroom | Required changes |
|---|---|---|---|
| lock−6s..−4s | TRIVIAL | +1.5..+3.5s | none |
| lock−3s | TIGHT | **+456ms** | move/add one getLogs so it completes by lock−869ms |
| lock−2.5s | PUSH | −44ms | moot: data is second-resolution; 2.5s ≡ 3s |
| lock−2s and later | INFEASIBLE | −544ms.. | no path to read + decide + sign + broadcast |

## C1 — pool-horizon sweep + delta-late-flow: **RULED OUT (edge confined to infeasible horizons)**

The imbalance→outcome correlation at every FEASIBLE horizon is null-to-tiny,
and the jump to the final-pool level happens inside the last (infeasible)
second:

| h | class | r golden | r dead |
|---|---|---|---|
| 6s | TRIVIAL | 0.016 | 0.007 |
| 4s | TRIVIAL | 0.018 | 0.009 |
| 3s | TIGHT | 0.018 | 0.008 |
| 2s | INFEASIBLE | 0.022 | 0.016 |
| 1s | INFEASIBLE | 0.028 | 0.022 |
| final | INFEASIBLE | 0.134 | 0.135 |

Delta-late-flow following (flow between −6s and −h, |Δ|≥0.1) at feasible
horizons: dead-era mean PnL +0.002..+0.007/bet, z≈0 — null. Majority-
following loses ~−0.05/bet everywhere (the pari-mutuel discount, as the
post-mortem predicted). This is exactly the "best edge at an infeasible
horizon" trap: noted as informational, **not recommended**, no further
analysis time spent.

## C2 — payout-aware contrarian @ −6s: **trivially feasible, but the edge died with everything else**

Minority-side betting above an imbalance threshold (same data the canonical
strategy already reads — zero runtime changes):

| thr | golden mean/bet (z) | fade (z) | dead (z) |
|---|---|---|---|
| 0.3 | +0.009 (1.2) | +0.019 (0.9) | +0.010 (0.5) |
| 0.4 | **+0.020 (2.2)** | +0.045 (1.8) | +0.006 (0.2) |
| 0.5 | +0.017 (1.5) | **+0.055 (1.9)** | +0.009 (0.3) |

Historically real-looking (golden z=2.2 at n=14k; though that's the best of
a 5-threshold sweep — multiple-comparisons caveat), survived the fade era,
**null in the dead era** (all z<0.5). The canonical∩against-crowd overlay
confirms: golden +0.180/bet against-crowd vs +0.032 with-crowd; dead era
−0.069 against vs −0.249 with — against-crowd loses LESS but still loses.
**Ruled out as a current edge.** Worth keeping as a cheap offline regime
monitor (if contrarian PnL at −6s reawakens, it's detectable with no
runtime work), not as a strategy.

## C4 — bulk-momentum re-map: **feasible-but-null, with one reversal cell worth a caveated follow-up**

Tri-agreement BTC momentum by strength band (canonical gate region ≈
1e-4..2e-4+), dead era:

| band (min|r|) | dead mean/bet (z) | n |
|---|---|---|
| 2e-5..5e-5 (very mild) | **−0.108 (−2.0)** | 302 |
| 5e-5..1e-4 | +0.004 (0.1) | 402 |
| 1e-4..2e-4 (gate region) | +0.038 (0.9) | 448 |
| 2e-4..5e-4 | −0.043 (−0.8) | 263 |
| ≥5e-4 | +0.106 (0.8) | 36 |

No coherent positive band — the "mild impulses still drift" hypothesis does
NOT convert to PnL at realized payouts. The one |z|≥2 cell is NEGATIVE
mild-momentum (−0.108/bet): in the dead era, very mild tri-agreement
impulses REVERSE — consistent with the lead-lag-compression/overshoot
story. Caveat: 15 cells were examined; ~0.7 cells at |z|≥2 expected by
chance. Spawns ONE follow-up hypothesis (mild-momentum FADE), pre-registered
before any further peeking: band 2e-5..5e-5, bet against the impulse,
validate per-cohort + permutation before believing anything.

## C3 — wallet smart money: **runtime TRIVIAL; the only candidate with an open edge question**

Runtime cost measured: 11,215 distinct wallets / 2.16M bets → ~1.3MB
trailing-stats state, O(1) per-event updates, zero decision-budget impact
(features derive from the same −6s event set the bot already reads).
The binding constraint is research validity, not runtime. Crucially, C1's
result sharpens the question this candidate must answer: the aggregate
informed flow sits in the infeasible last second — so wallet features only
help if historically-accurate wallets bet EARLY (≤ −6s..−3s) while the rest
of the late crowd carries the information. That is testable offline
(per-wallet trailing accuracy × bet-timing distribution) and needs its own
harness extension.

## Short list (edge × feasibility)

1. **C3 wallet smart-money, early-bettor-conditioned** — the only candidate
   with trivial runtime AND an unprobed edge question. Next step: harness
   extension (per-wallet trailing WR, no-look-ahead, conditioned on bet
   time ≤ lock−3s), ~1–2 days.
2. **C4-reversal follow-up (mild-momentum fade)** — trivially feasible,
   single pre-registered hypothesis, heavy multiple-comparisons discount;
   cheap to test (~half day) alongside 1.
3. — nothing else qualifies. —

**Ruled out / informational**: C1 pool-horizon (edge is real but lives at
infeasible horizons; even a lock−3s TIGHT poll buys r≈0.008 in the dead
era — nothing); C2 contrarian (feasible, historically real, currently
dead; keep as an offline regime monitor); majority-following at any
horizon (−EV, pari-mutuel discount).

All survivors remain subject to the standing promotion rule (CV5 + frozen
holdout + ext_v2 + permutation null) before any live consideration; the
bot stays paused.
