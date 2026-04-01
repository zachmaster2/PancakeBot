# Neural Direction Model Spec

## Purpose

Define the reset path after the failed controller and direct-action lanes.

The new target is intentionally simpler:

1. one model
2. one prediction per `target_round`
3. prediction must be `Bull` or `Bear`
4. no `skip` in the model output contract
5. headline evaluation metric is out-of-sample win `%`

This document is the current design contract for that pivot.

## Terminology

Use the canonical vocabulary in [TERMINOLOGY.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/TERMINOLOGY.md).

The neural direction lane must use these terms exactly:

1. `target_round`: the round being predicted
2. `open_round`: the live causal state of `target_round`
3. `locked_round`: the immediately prior round
4. `prior_context_rounds`: ordered round context immediately preceding
   `target_round`, with `locked_round` at `prior_context_rounds[-1]`
5. `outcome_eligible_prior_context_rounds`: `prior_context_rounds[:-1]`
6. `context_klines`: kline context anchored to the `target_round` cutoff
7. `target cutoff pools`: target-round pools computed only from bets with
   `created_at <= cutoff_ts`

Forbidden ambiguity for this lane:

1. do not say `current_round`; use `target_round` or `open_round`
2. do not say `context_rounds`; use `prior_context_rounds`
3. do not use target-round `final pools` or post-cutoff klines as model input

## Causal Contract

The model input must respect all of the following:

1. `target_round` is represented only by pre-lock, pre-cutoff information
2. `open_round` has no realized `position`
3. `locked_round` has no realized `position` at decision time
4. outcome-dependent features may use only
   `outcome_eligible_prior_context_rounds`
5. `context_klines` must end at or before the `target_round` cutoff
6. target-round pool features must use `target cutoff pools`, never target
   final pools

The data contract therefore has three distinct causal objects:

1. settled history from `outcome_eligible_prior_context_rounds`
2. one `locked_round` snapshot
3. one `target_round` cutoff snapshot

Those are not interchangeable and should not be flattened into one fake
all-closed sequence without explicit masking or branch separation.

## Label Contract

The label is the realized `position` of `target_round`:

1. `Bull`
2. `Bear`

Training and headline evaluation should exclude:

1. failed rounds
2. refund-like unusable rounds
3. `House` outcome rounds

The first evaluation headline is:

1. out-of-sample win `%`

## Model Ladder

The initial neural-first ranking is:

1. `TCN`
2. `MLP`
3. `GRU` / `LSTM`
4. `Transformer`

The first serious mainline should be `TCN`.

## Input Structure

The first preferred structure is hybrid rather than pretending every row is the
same round state:

1. a settled-history branch over `outcome_eligible_prior_context_rounds`
2. a `locked_round` branch
3. a `target_round` branch

The first preferred kline handling is:

1. use `context_klines` anchored to the `target_round` cutoff
2. either merge them into the `target_round` branch or feed them as a second
   short sequence branch

## Training-Size Experiments

After full-history rounds and klines are synced, compare at least:

1. `20k`
2. `50k`
3. `100k`
4. `200k`
5. all available history

These are train-window sizes. Validation and test must remain chronological and
recent.

The point of this matrix is to answer:

1. does more history improve recent win `%`
2. does old history dilute current regime signal
3. does a neural model benefit from long-history pretraining more than prior
   tree-based experiments did

## Required Baselines

Every run must report at least:

1. `always_bull`
2. `always_bear`
3. one trivial directional baseline, such as previous settled round side

These are required context for headline win `%`.

## First Execution Plan

1. finish full-history rounds sync
2. sync full-history klines over the same historical span
3. freeze one canonical dataset version
4. build binary labels over valid `target_round` rows
5. run baseline win `%` numbers
6. train `MLP` as the simple neural sanity check
7. train `TCN` as the first real mainline model
8. compare train-size windows chronologically on recent held-out test slices

## Current Status

As of March 31, 2026:

1. full-history round backfill from epoch `1` has been launched in the
   background
2. the neural direction spec is design-only so far
3. no neural model implementation has started yet

Update on April 1, 2026:

1. the first shared dataset builder is now implemented in
   [neural_direction_dataset.py](/C:/Users/zking/Documents/GitHub/PancakeBot/pancakebot/domain/models/neural_direction_dataset.py)
2. the first implementation deliberately reuses the canonical v8 causal feature
   builder as the MLP input contract, instead of creating a second competing
   feature builder before the first baseline/model comparison
3. the headline-only baseline runner is now implemented in
   [run_neural_direction_baselines.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_neural_direction_baselines.py)
4. the first trainable neural sanity path is now implemented as a torch MLP in
   [neural_direction_mlp.py](/C:/Users/zking/Documents/GitHub/PancakeBot/pancakebot/domain/models/neural_direction_mlp.py)
   with the shared eval runner
   [run_neural_direction_mlp_eval.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_neural_direction_mlp_eval.py)
5. the first `TCN` sequence path is now implemented in
   [neural_direction_tcn.py](/C:/Users/zking/Documents/GitHub/PancakeBot/pancakebot/domain/models/neural_direction_tcn.py)
   with the shared eval runner
   [run_neural_direction_tcn_eval.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_neural_direction_tcn_eval.py)
6. the current mainline evidence favors the full-feature MLP over the current
   TCN variants on the recent kline-limited tail; full-history kline sync is
   still required before the larger-window comparison matrix can be completed
