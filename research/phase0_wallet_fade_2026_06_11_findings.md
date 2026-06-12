# Phase-0 survivors: wallet smart-money + mild-momentum fade — findings (2026-06-11)

Harnesses: `research/phase0_wallet_smart_money_2026_06_11.py` and
`research/phase0_mild_momentum_fade_2026_06_11.py` (reproducible; artifacts
in `var/strategy_review/phase0_{wallet,fade}_2026_06_11/`). Settlement: flat
stake, realized final-pool payouts, 3% fee, no gas. Eras: golden ≤479952,
fade 479953–484408, dead ≥484409.

## Bottom line

**Both surviving candidates are RULED OUT.** With these, the systematic
sweep of the existing dataset is complete: every signal family it contains
has now been tested with runtime feasibility enforced — momentum (dead
since 2026-05-26), pool flow (alive but at infeasible horizons), contrarian
(historically real, currently dead), mild-momentum fade (noise), and wallet
smart-money (skill is real but not convertible). **No deployable strategy
emerges from data the bot can currently see.** Remaining directions require
new data collection or a strategy-class pivot.

## A. Wallet smart-money (early-bettor-conditioned): RULED OUT

Pre-registered primary definition: trailing-100-bet accuracy, Beta(10,10)
shrinkage, eligible at n≥30, smart at â≥0.55; signal = sign of smart net
flow visible at lock−h, requiring ≥2 smart wallets and ≥0.05 BNB net.
Two sensitivity variants (strict/loose) reported, multiple-comparisons
flagged.

**1. Skill exists** — the accuracy-persistence gradient is real and
monotone (forward hit-rate of a wallet's next bet, by trailing â):

| trailing â | n (bets) | forward hit |
|---|---|---|
| 0.50–0.52 | 443k | 0.506 |
| 0.54–0.56 | 268k | 0.510 |
| 0.56–0.58 | 105k | 0.514 |
| ≥0.58 | 104k | **0.527** |

**2. …but it is not convertible.** The best bucket's 52.7% forward
accuracy is far below the ~55% pari-mutuel breakeven, and following smart
net flow LOSES in every era, variant, and horizon (9/9 cells negative):
golden −0.046/bet (z=−7.8, n=26k), dead −0.027…−0.064 (z −1.5…−2.5).
Dead-era permutation p≈0.46–0.48 — the observed PnL is exactly the
structural majority-following discount, with no signal on top.

**3. …and the timing is wrong-way.** Smart wallets bet LATER than average:
42.6% of their bets land inside the final 3 seconds (vs 32.1% baseline);
only 48.1% are visible at lock−6s (vs 57.6% baseline). The accurate actors
are concentrated precisely in the window the runtime envelope cannot see —
consistent with them being the last-seconds stale-oracle snipers the
post-mortem identified.

Runtime feasibility (for the record): TRIVIAL — 11,215 wallets ≈ 1.3MB
trailing state, O(1) per-event updates in the existing poller loop,
cold-start rebuild by replaying the local closed_rounds tail (~seconds).
Feasibility was never the blocker; the market structure is.

## B. Mild-momentum fade: RULED OUT (the C4 cell was noise)

Pre-registered: fade BTC tri-agreement impulses with min|r| ∈ [2e-5, 5e-5).
Discovery honesty: the C4 sweep examined 15 cells and burned the entire
dead era; no unburned holdout exists.

- **Discovery significance after adjustment**: raw z=−2.04 (p=0.041) →
  Šidák over 15 cells **p=0.47**. Not significant.
- **Cross-era mechanism check**: fading this band in the golden era loses
  heavily (−0.073/bet, z=−3.8, n=2950) — no consistent reversal mechanism.
- **Dead era**: +0.043/bet, z=0.72; permutation p=0.095.
- **Within-dead split**: latest +0.088 (z=1.2) vs vm_live −0.052 (z=−0.5)
  — sign-inconsistent inside the discovery window itself.

Runtime: trivially feasible (same data/path as canonical) — irrelevant,
since there is nothing to deploy.

## What remains (all require things we don't have)

1. **New data collection** (previously flagged): OKX perp funding /
   order-book depth at the seconds horizon — the only untested signal
   class compatible with the runtime envelope.
2. **Strategy-class pivot** — e.g., competing in the last-seconds window
   itself (where the informed flow demonstrably lives) is a different
   game: it requires sub-second execution against the lock and is
   adversarial against the actors already there; outside the current
   bot's architecture and risk appetite.
3. **Wait-and-monitor** — the zero-cost offline monitors (contrarian PnL
   at −6s, canonical gate WR) can be recomputed per sync to detect a
   regime turn that revives the validated machinery.

These are user decisions, not research tasks; the dataset itself is, at
this point, thoroughly characterized.
