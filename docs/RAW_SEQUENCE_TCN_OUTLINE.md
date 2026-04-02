# Raw Sequence TCN Outline

## Purpose

Define the next cleaner `TCN` architecture branch after the current
derived-feature `TCN` results.

The main question is:

1. are we suppressing temporal signal by feeding the `TCN` a sequence of
   already-derived target-level feature rows instead of a more raw causal
   sequence

This is an outline only. It is not yet the active implementation contract.

## Problem Statement

The current `TCN` path in
[neural_direction_tcn.py](/C:/Users/zking/Documents/GitHub/PancakeBot/pancakebot/domain/models/neural_direction_tcn.py)
uses a rolling sequence of already-engineered target-level feature vectors.

That means:

1. the model sees temporal structure only after heavy feature compression
2. some local flow and price-shape information may already be lost
3. the `TCN` may be learning over summaries of summaries rather than the rawer
   causal sequence itself

Current evidence suggests this branch is still useful, but likely not the full
expression of what a sequence model could do.

## Core Decision

The next `TCN` branch should be a rawer sequence model, not just more tuning on
the current derived-feature `TCN`.

Recommended design order:

1. `hybrid raw + derived`
2. `mostly raw`
3. `raw-only`, only if the hybrid branch proves the derived features are
   actively hurting

I do not recommend jumping straight to `raw-only` first.

Current V1 decision:

1. start with the simplest rawer branch first
2. do not include wallet identity in the first raw-sequence implementation
3. do not build a hierarchical bet-event encoder yet
4. represent visible bet flow with fixed-width causal summaries instead

Wallet-aware event modeling remains a later optional branch, not the starting
point.

## Causal Contract

This outline still uses the canonical terminology in
[TERMINOLOGY.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/TERMINOLOGY.md)
and the causal rules in
[NEURAL_DIRECTION_MODEL_SPEC.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/NEURAL_DIRECTION_MODEL_SPEC.md).

Required constraints:

1. `target_round` inputs must use only pre-cutoff information
2. `locked_round` may not use realized outcome
3. outcome-dependent round history may use only
   `outcome_eligible_prior_context_rounds`
4. `context_klines` must end at or before the `target_round` cutoff
5. target-round `final pools` remain forbidden as model input

## Proposed Input Branches

### 1. Round Sequence Branch

A short rawer sequence over recent rounds.

Recommended first contents per timestep:

1. round state flag:
   - settled prior round
   - `locked_round`
   - `target_round`
2. bull pool amount
3. bear pool amount
4. total pool
5. bull share
6. bear share
7. pool deltas versus prior timestep
8. lock price if known for that timestep
9. close price only for settled timesteps
10. settled direction only for settled timesteps
11. masks indicating which fields are unavailable by causal state

Recommended first sequence window:

1. `outcome_eligible_prior_context_rounds`
2. one `locked_round` timestep
3. one `target_round` timestep

This branch should preserve state differences explicitly rather than pretending
all timesteps are the same kind of row.

### 1A. Simplest Bet-Flow Representation

The first implementation should use fixed-width binned flow summaries, not raw
variable-length bet lists.

Recommended first binning:

1. split the visible bet window into a small fixed number of time bins
2. for each bin, record:
   - bull amount
   - bear amount
   - bull bet count
   - bear bet count
   - net bull minus bear amount

Recommended first bin counts:

1. `target_round`: `4` or `6` bins across the visible pre-cutoff window
2. `locked_round`: `4` or `6` bins across its visible decision-time window
3. settled prior rounds: either one coarse summary per round or the same bin
   scheme if memory allows

This keeps the first branch:

1. causal
2. fixed-width
3. cheaper than event-level modeling
4. materially rawer than the current derived-feature path

### 2. Kline Sequence Branch

A raw minute-level sequence over `context_klines`.

Recommended first contents per kline:

1. open
2. high
3. low
4. close
5. volume, if available and trustworthy
6. simple normalized returns
7. optional time-to-target proximity encoding

Recommended first sequence length:

1. the same causal `context_klines` window already used by the current feature
   builder

### 3. Snapshot Branch

A small non-sequential branch for current scalar features that are still useful
to expose directly.

Recommended first contents:

1. `target cutoff pools`
2. a small `locked_round` late-phase summary
3. simple current-scale normalization factors

This branch should stay small. It is there for clean fusion, not for rebuilding
the full current MLP feature table.

## Architecture Options

### Option A: Hybrid TCN

Recommended first mainline.

Structure:

1. round sequence branch -> `TCN`
2. kline sequence branch -> `TCN`
3. small snapshot branch -> `MLP`
4. concatenate branch outputs
5. final `MLP` head -> `Bull` / `Bear`

Why this is the best first branch:

1. keeps strong derived signals available
2. adds genuinely rawer sequence learning
3. avoids all-or-nothing feature removal risk

### Option B: Mostly Raw TCN

Structure:

1. round sequence branch -> `TCN`
2. kline sequence branch -> `TCN`
3. no snapshot branch, or only minimal scale features

This is a valid second step if the hybrid branch suggests the engineered
snapshot features are unnecessary or distorting.

### Option C: Raw-Only Single Sequence

Not recommended first.

This would force all round and kline information into one merged sequence.
It is possible, but it raises unnecessary alignment complexity early.

## Why This Is Different From Lagged Features

Lagged-feature tabular modeling flattens time into one row:

```text
[x_t, x_t-1, x_t-2, x_t-3, ...]
```

The raw-sequence `TCN` keeps time as a true sequence:

```text
[
  [features at t-3],
  [features at t-2],
  [features at t-1],
  [features at t],
]
```

The practical differences are:

1. parameter sharing across timesteps
2. cleaner local motif learning
3. less manual lag engineering
4. better fit for kline shape and short flow transitions

## What Stays Derived

Even in the rawer branch, a few bounded derived features are still useful:

1. normalized pool shares
2. simple price returns
3. state masks
4. optional time-to-cutoff / time-to-lock encodings

This is still a neural sequence model. It does not need to be pure raw bytes to
count as rawer and less tabular than the current path.

## What Is Explicitly Deferred

These are intentionally out of scope for the first raw-sequence branch:

1. wallet identifier embeddings
2. wallet-history feature joins
3. full variable-length raw bet-event encoders
4. attention over within-round bet lists

Those can be revisited later only if the simpler fixed-bin branch shows enough
promise to justify the extra complexity.

## What To Avoid

1. do not rebuild the entire v8 engineered table and then just stack `TCN` on
   top again
2. do not use target post-cutoff pool flow
3. do not flatten `locked_round` and `target_round` into fake settled rows
   without explicit state masks
4. do not merge round and kline branches into one unstructured tensor as the
   first implementation
5. do not evaluate this branch only on non-causal splits

## Comparison Plan

The first evaluation ladder should be:

1. current derived-feature `MLP`
2. current derived-feature `TCN`
3. hybrid raw-sequence `TCN`

Hold constant as much as possible:

1. label contract
2. train / validation / test chronology
3. offsets
4. headline win `%`
5. later selective-policy evaluation protocol

Primary questions:

1. does the rawer branch improve broad held-out win `%`
2. does it improve selective-confidence buckets
3. does it improve settled-policy pockets
4. does it reduce sensitivity to exact `seq_len`

## First Experimental Matrix

Recommended first bounded matrix:

1. model:
   - current derived `TCN`
   - hybrid raw-sequence `TCN`
2. train sizes:
   - `100k`
   - `200k`
3. sequence lengths:
   - round branch: `8`, `16`, `32`
   - kline branch: `32`, `64`, `128`
4. same offsets as the current mainline comparisons:
   - `0`
   - `432`
   - `864`

This is large enough to answer the architectural question without exploding
search space.

## Success Criteria

The raw-sequence branch is worth keeping only if at least one of these happens:

1. it beats the current derived `TCN` on held-out win `%`
2. it materially improves the best settled-policy pockets
3. it stays competitive while being less brittle across `seq_len`

If it fails all three, the repo should not drift further into sequence-model
complexity on this branch.

## Recommended Next Step

Build `Option A`, the hybrid raw-sequence `TCN`, first.

That is the cleanest way to test the hypothesis:

1. current `TCN` may be leaving signal on the table
2. but fully discarding the existing causal feature work is premature

For that first implementation, use the simplest bet-flow representation:

1. fixed-width causal time bins
2. no wallet identity
3. no hierarchical bet-event encoder
