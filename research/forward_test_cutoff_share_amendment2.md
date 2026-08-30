# Sealed forward test — cutoff share vs PnL (Amendment 2)

**Pre-registered 2026-08-24. Start epoch 511350.**

Written to the repo 2026-08-29. Until then this design existed **only in an
assistant's conversational notes** — it was not in `research/`, not in git,
not on the VM. An exhaustive search on 2026-08-29 (`research/`, `docs/`, all
repo files, `git log --all`, the VM filesystem, `novel_observations.md`)
found no trace of it. A pre-registered design that lives in one volatile
place is not pre-registered in any meaningful sense: it cannot be audited,
it cannot be shown to have preceded the data, and it dies with the notes.
That is the reason this file exists, independent of whether the test runs.

---

## Hypothesis

Cutoff share (the Bull fraction of the pool at the pool cutoff) carries
information about round PnL.

- **Primary endpoint: CONTINUOUS.** Correlation of cutoff share against PnL.
  CV5 estimate **r = −0.0726**, sign matching the hypothesis.
- **Secondary: the binary split.** Demoted from primary at Amendment 2.

## Stopping rule — O'Brien-Fleming group-sequential

| information time | z boundary |
|---|---|
| 25% | 4.758 |
| 50% | 3.364 |
| 75% | 2.747 |
| 100% | 2.379 |

- OBF constant **c = 2.3789**
- Overall one-sided **α = 0.0096**
- **Non-binding futility stop** at z < 0 on any interim
- **Max n = 1,957** direction-bearing rounds

## Expected duration

| scenario | expected |
|---|---|
| real and large (r = 0.10) | 15.0 wk |
| real and marginal (r = 0.073) | 17.8 wk |
| **futility stop under the null** | **12.2 wk** |

The null resolves fastest, and a futility stop around week 12 is the single
most likely outcome.

**Winner's-curse floor.** If the true effect is a fraction of the CV5 point
estimate: at 75% → **38.6 wk**; at 50% → **87.0 wk**. The CV5 estimate is
the value that selected this hypothesis, so it is biased upward by
construction and these are the honest planning numbers.

## Population

**Direction-bearing rounds** — rounds where the gate produced a direction.
Post-June rate **4.58%**, ≈ **92/week**.

The direction is established **strictly before** the pool filter, so the
`min_pool_bnb_at_cutoff` 1.5 → 1.25 change of 2026-08-24 **cannot touch the
population**. Verified 2026-08-29 against the shipped code.

> **Anchor by code identity, not line number.** The original registration
> cited `momentum_pipeline.py:411` (population) and `:425` (pool filter).
> Both had already drifted by 2026-08-29 — the pool filter had moved to
> `:528`, about 103 lines, and `:411`/`:425` now land inside the drawdown
> breaker's comment block, which has nothing to do with either. Line numbers
> are not durable anchors in a file under active change. The durable anchors:
>
> - **population**: the `if signal_dir is None:` branch that returns
>   `self._skip(result.skip_reason or "gate_no_signal")` in
>   `MomentumOnlyPipeline.decide_open_round`. A round is direction-bearing
>   iff it does **not** take that branch.
> - **pool filter**: the immediately following
>   `if pool_total < self._strategy.pool_filter.min_pool_bnb_at_cutoff:`
>
> The ordering — direction first, filter second — is what makes the
> population immune to the threshold change. Re-verify the ordering, never
> the line numbers.

## Recorded as FAILED — do not re-propose

**Decomposing PnL into payout × win-rate.** The variance is Bernoulli:
`Var(E[PnL|payout])` is **0.6%** of total, and `corr(payout, won) = −0.0618`
means the two covary rather than separating cleanly. The decomposition
explains almost nothing and its terms are not independent.

---

## Replayability — RESOLVED 2026-08-29: the test needs NO live money

The question that gated a hosting decision: does this test evaluate off
stored data, or does it require live fills?

**Answer: it replays completely. Betting buys no validation speed.**

### 1. Cutoff share needs only stored round data

`momentum_pipeline._pools_from_bets(round_t, cutoff_ts)` consumes exactly
three fields per bet — `created_at`, `position`, `amount_wei` — plus
`cutoff_ts = lock_at − pool_cutoff_seconds`. Every one of those lives in a
`Round`, which loads from `var/closed_rounds.jsonl`. **No chain access, no
RPC, no live observation.**

### 2. The inputs are recoverable for arbitrary history

The Graph returns complete round + bet arrays (with `createdAt`,
`position`, `amountWei`) for any epoch range tested, back to at least
2026-01-07, contiguous, with no horizon. See `novel_observations.md`
(2026-08-29) for the retention measurements.

### 3. Proven end to end, not argued

101 rounds (epochs 510900–511002) were fetched from The Graph **after the
fact** — all of them past the store head of 509420, so none had been
observed through the store. Cutoff pools were reconstructed via the
production `_pools_from_bets` and compared against `cutoff_used_*` in
`var/live/cycle_audit.csv`, which the running bot wrote from Bet events it
observed on-chain in real time.

```
rounds compared        102
BIT-EXACT (bull+bear)  102
divergent                0
worst share diff       1.110e-16      (float epsilon)
worst BNB diff         1.776e-15
```

Two fully independent sources — live RPC event observation versus a
retrospective subgraph fetch — agree to the last bit.

> **Alignment trap, cost an hour.** `cycle_audit.csv`'s `locked_epoch` is
> the epoch that JUST LOCKED; the round being decided is `locked_epoch + 1`.
> Comparing at offset 0 produces ~99/100 apparent divergence with a worst
> diff of 0.49 — which reads exactly like a real reconstruction failure.
> The same off-by-one appears in the 2026-08-28 head-race work. Always
> confirm the alignment before believing a divergence.

### 4. The PnL term is hypothetical, not realized

`settlement.settle_from_round_data(bet_bnb, bet_side, lock_price_usd,
close_price_usd, bull_amount_wei, bear_amount_wei, oracle_called,
treasury_fee_fraction)` — its own docstring says "compute settlement from
on-chain round data (**no bets list needed**)". Every argument is in the
closed-round store. PnL is parimutuel arithmetic over settlement pools, not
a record of what actually filled. Trivially replayable.

### Consequence

Running the bot with live money contributes **nothing** to this experiment.
The only quantities that require real execution are execution-quality
measures — inclusion latency, slippage, LATE rate, backtest-vs-live
agreement — and none of them appear anywhere in this design. The evidence
stream that feeds this test is the round store plus the kline stores, and
both are backfillable within the horizons recorded in
`novel_observations.md`.

A chain fallback exists but is not needed: bloXroute served `eth_getLogs`
for the prediction contract at ~240 days back on 2026-08-29. That is
distinct from archive **state** (`eth_call`), which both providers refuse.
Do not conflate the two — `getLogs` is log retrieval, `eth_call` needs
historical state.

### What would change this answer

Only a redefinition of the endpoint. If a future amendment measured
realized fills — slippage against the cutoff pool, inclusion-conditioned
PnL, LATE-adjusted returns — it would need live execution and the
conclusion above would not carry. As written, Amendment 2 does not.
