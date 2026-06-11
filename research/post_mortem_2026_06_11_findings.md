# Edge-decay post-mortem — findings (2026-06-11)

Analysis: `research/post_mortem_2026_06_11.py` (reproducible end-to-end from
the synced dataset; artifacts in `var/strategy_review/post_mortem_2026_06_11/`).
Every headline claim below survived an adversarial replication panel
(null simulations, Wilson CIs, mechanical-confound checks); claims the
panel demoted are explicitly marked NOT SUPPORTED.

## The story in five sentences

The canonical edge died in **one statistically supported break at
epoch ~484409 (2026-05-26)**, preceded by a two-week fire-rate collapse
(May 10–26: 1.0% vs the normal 3.7%, z=−9.3 — a short-horizon impulse
drought, signal quality over that window indeterminate at n=45). After
May 26 the gate fires MORE than golden (4.4%, p=0.02) and wins less
(48.2% vs the ~54.9% flat-stake breakeven; "no demonstrable edge", not
"worse than coin-flip" — vs 50% p=0.61). The decay is **not crowding**:
payout multiples, breakeven WR, and crowd-alignment are flat-to-falling
across eras. The central paradox is diagnostic: **per-round momentum
correlations in the newest cohorts are back at golden-era effect sizes**
(btc_r3 r=0.045 vs 0.046 in CV5; vm_live btc_r7 r=0.091) **while the
gate's selected tail stopped continuing** — small impulses still drift,
the large 3-symbol-agreement impulses the strategy bets no longer carry
into the 5-min close. The best-supported explanation is lead-lag
compression: strong cross-asset impulses now reach BNB's lock price
before lock, leaving no post-lock drift to harvest.

## A. WHEN

| Date (epoch) | Event | Strength |
|---|---|---|
| 2025-12-11..2026-05-10 | golden: WR 61.5%, n=1554 risk-free bets, fire 3.7% | solid |
| 2026-02-17 (456820) | CUSUM peak — **NOT a supported onset** (null p=0.28, 20k sims; method structurally confines the peak to the golden era; changepoint scan picks a different, also-insignificant split) | demoted |
| 2026-05-10..05-26 (479953..484408) | fire-rate collapse to 1.0% (z=−9.3). WR on the 45 fired bets 62.2%, CI [47.6, 74.9] — quality indeterminate | scarcity solid; quality n/a |
| **2026-05-26 (484409)** | **the break**: fire recovers to 4.4% (> golden, p=0.02), WR collapses to 48.2% (z=−3.57 vs golden) | solid |
| 2026-06-05 | rolling-WR bottom (0.44 per 100-bet window, −0.19 flat PnL/bet) | descriptive |

Regime correlates (rolling WR vs BTC vol −0.12, vs pool size +0.16) are
computed on heavily overlapping windows (effective n≈17) — **uninformative**.

## B. HOW

- **Fires more, wins less** — the dead era's fire-rate EXCEEDS golden
  (4.4% vs 3.7%, p=0.02) while WR collapsed. The gate's trigger condition
  is met as often as ever; its meaning changed.
- **No calibration story survives the noise**: dead-era strength terciles
  (47.7/43.8/53.1) are n≈64/cell, homogeneity p≈0.57. The primary-signal
  subset collapse is solid (46.7%, z=−3.97 vs golden); regime-2's 72.7%
  is n=11 (CI 43–90%) — anecdote.
- **Economics flat**: mean payout 1.84/1.85/1.84 across eras; breakeven
  WR ~54.9% everywhere; fraction of bets aligned with the cutoff-pool
  majority fell 0.36→0.33→0.29. No payout compression on wins.
- (Caveat: flat-stake economics are the infinitesimal-stake limit. At
  real size — 1 BNB into the ~3 BNB median bet-round pool — self-dilution
  pushes breakeven toward ~65%. Era-relative comparisons unaffected.)

## C. Regime shift vs model-prior drift

**Verdict: regime shift in the conditional (tail) structure, with the
marginal structure intact.** In effect-size units (`feature_r.csv` — use
this, not the SnR table, across cohorts):

- Momentum features (btc/eth/sol r3/r7/r15, agreement counts): CV5 r≈0.05;
  `latest` r≈0.045 (significantly positive); vm_live partially higher
  (btc_r7 0.091). The features did NOT die. Only post_fresh dips, and its
  CI cannot distinguish 0 from CV5 levels.
- Nothing NEW emerged on the dead cohorts (no feature is dead-only strong).
- What died is conditional: P(continuation | gate-tripping impulse).
  Combined with stable payouts (no crowding) this points at the impulse
  itself being absorbed into the lock price faster than before.

**Market stories, ranked** (panel): (1) **lead-lag compression** — BNB
market-making now propagates second-scale BTC/ETH/SOL impulses within
seconds, so by lock the impulse is priced and the residual drift on
strong impulses is zero-to-negative; uniquely explains intact marginal r
+ dead tail + flat payouts. (2) Momentum-regime change (drought, then
impulse-overshoot chop). (3) BNB–BTC decoupling (partly contradicted —
cheap to test). (4) Participant crowding — rejected on economics.

**Sharpest discriminator, computable from existing data** (flagged as the
first follow-up, NOT run here): BTC *self*-continuation per cohort
(pre-lock btc_r15 vs BTC's own forward 5-min return) — story 1 predicts
BTC still continues but BNB catch-up is gone; story 2 predicts BTC
continuation died too. The unused `var/bnb_spot_prices.jsonl` (synced,
51k epochs) supports a BTC→BNB lag-profile check within the ±18s
boundary windows.

## D. The pool-imbalance finding (and its demotion)

`pool_final_imbalance` is the only feature whose effect size is stable
across ALL cohorts (r 0.09–0.21, including dead) while the actionable
`pool_cutoff_imbalance` (lock−6s, what the bot can see) is r≤0.06
everywhere. 45% of all pool money arrives in the last 6s (79% of that in
the last 3s). But the panel's mechanical analysis demotes the exciting
reading: late flow is 84% **odds-balancing** (corr −0.77 against the
early pool); the informative cell is the 16% of rounds where late money
piles onto the majority anyway (that side wins 58.8%); and **even
perfect-foresight betting of the final majority is −EV** (55.15% WR vs a
payout-weighted breakeven of 57.0%). Final-pool informativeness is the
signature of pari-mutuel near-efficiency at close — most plausibly
last-seconds actors exploiting the stale oracle lock price — not a free
signal. Any pool-flow candidate must therefore be framed as
**mispricing vs pari-mutuel-implied probability**, never raw correlation.

## Hypothesis ranking (most likely explanation for the decay)

1. **Lead-lag compression / faster arbitrage** of strong cross-asset
   impulses into BNB at the seconds horizon (intact marginal r, dead
   conditional tail, flat payouts, persistent last-seconds informed flow).
2. **Short-horizon momentum-regime change** (drought May 10–26, then
   chop/overshoot on large impulses) — distinguishable from #1 by the BTC
   self-continuation check.
3. **Model-prior overfit to a temporary microstructure** (the golden era
   itself was the anomaly) — consistent with the gate's threshold being
   tuned on CV5 tails; partially subsumed by #1/#2.
4. ~~Crowding/payout compression~~ — rejected (economics flat).
5. ~~Feature death~~ — rejected (marginal r intact in latest/vm_live).

## Research candidates (ranked, evidence × tractability)

1. **Pool-horizon sweep + delta-late-flow vs implied probability**
   (lock−5s..−1s; feasibility bound: broadcast at predecessor−475ms makes
   lock−3s comfortable, lock−2s borderline). Evidence: the last-6s flow is
   the only never-decayed signal; framed strictly as mispricing vs
   `(1+imbalance)/2 / (1−fee)`, settled at realized payouts with own-stake
   dilution. Phase-0: offline sweep on this dataset, 1–2 days.
2. **Payout-aware contrarian** (fade the cutoff-crowd above an imbalance
   threshold). Direct evidence: the golden edge was concentrated in
   against-crowd bets (+0.180 flat/bet, n=993) vs near-zero with-crowd
   (+0.032, n=561); dead era −0.069 against vs −0.249 with.
3. **Wallet-level smart money** (closed_rounds carries per-wallet
   histories; causal trailing-accuracy features). Must beat raw flow at
   the same horizon; whale-following is already dead (pool_top_bet_frac
   r≈0). Higher effort, real overfitting surface.
4. **Momentum response-curve re-map** (the bulk, not the tail: mild
   impulses still drift). High diagnostic value even if no standalone edge.
5. **Momentum-regime filter** — weakest (exactly one regime transition in
   the data; unvalidatable).

A shared Phase-0 harness (horizon-parameterized pool reconstruction +
realized-payout settlement + implied-probability comparison) covers
candidates 1, 2 and half of 4 in ~1 day of build. **All candidates are
new strategies under the standing promotion rule** (CV5 + frozen holdout
+ ext_v2 + permutation null); the pool horizon is a different knob from
the canonical `kline_cutoff=2` invariant but needs its own full gauntlet.

## Requires NEW data collection (flagged, not implemented)

- OKX perp funding rates / order-book depth at the seconds horizon
  (candidate features for #1/#4 refinement).
- Live capture of pool flow at sub-second resolution beyond what
  event-accumulation already provides (only if the Phase-0 sweep shows
  the lock−2s horizon is the binding constraint).

## Replay-validity notes

Risk gates verifiably bypassed (tracker=None); failed rounds excluded
upstream; zero draw rounds in 51,244; createdAt clean (0/2.16M anomalies);
settlement on final pools is the correct payout model. The risk-free
stream (1,792 bets) is ~2× the production-sized stream — era WRs describe
the gate signal, not the strategy as traded. vm_live final pools contain
the bot's own min-bets (negligible endogeneity).
