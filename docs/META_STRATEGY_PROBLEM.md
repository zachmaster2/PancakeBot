# Meta-Strategy Problem

## Purpose

Define the strategy-selection problem above the base trading strategies.

The question is not:

1. "What is the single best strategy overall?"

The question is:

1. "Given the current regime, which strategy is most likely to perform best in the next forward block?"

## Core Idea

Use a meta-selector that predicts near-future strategy performance and chooses one strategy for the next block of rounds.

This exists because:

1. different windows can have different winners
2. the all-history best strategy may not be the best recent strategy
3. a strategy that works now may degrade quickly in the next regime

## Problem Definition

### Strategy Universe

Define a finite frozen set `S` of candidate strategies/profiles.

For a single experiment:

1. the universe `S` does not change mid-run
2. each member of `S` must be a fully specified strategy/profile

### Decision Unit

At decision time `t`, select exactly one action for the next forward block:

1. choose one strategy `s in S`
2. or choose the fallback action `skip_all`

### Forward Horizon

For the first version:

1. forward horizon `H = 500` rounds

The meta-selector predicts which strategy is best for the next `500` rounds.

### Decision Cadence

For the first version:

1. make one decision every `500` rounds

That means:

1. choose action for rounds `[t+1, t+500]`
2. hold that action for the whole block
3. reevaluate at the next block boundary

Later variants may reevaluate more often, but the first version should stay block-aligned.

## Prediction Target

For each strategy `s`, define:

1. `y(t, s) = realized net_bnb_per_500 of strategy s over rounds [t+1, t+500]`

The meta-selector estimates `y(t, s)` for all `s in S`.

## Action Rule

At each decision time:

1. predict `y_hat(t, s)` for every `s in S`
2. find `s_best = argmax_s y_hat(t, s)`
3. apply the safety rule
4. if the safety rule does not block, choose `s_best`

## Safety Rule

### Fallback Action

Default fallback action:

1. `skip_all`

This means do not activate any strategy for the next block.

### Safety Trigger

For the first version, choose `skip_all` if either is true:

1. the best predicted next-block value is below `0`
2. the best predicted value is not above the comparison baseline by at least a configured safety margin

### Comparison Baseline

Default comparison baseline:

1. `skip_all`, which has expected block return `0`

This is the cleanest initial safety baseline because it avoids forcing capital into a regime where no strategy looks good enough.

Later, an additional comparison baseline may be introduced, such as:

1. promoted static baseline strategy
2. current incumbent strategy

But the first fallback action should still be `skip_all`.

## Information Constraints

At decision time `t`, the meta-selector may use only information available through round `t`.

No future leakage is allowed.

That includes:

1. no features built from future rounds
2. no labels or performance stats that peek into the forward block
3. no strategy performance summaries computed using future results

## Feature Families

The first feature set should use two classes of information.

### Regime Features

Examples:

1. recent bull/bear mix
2. flip rate
3. streak behavior
4. price movement / volatility
5. pool behavior
6. bet-flow behavior

### Per-Strategy Shadow Features

Examples:

1. trailing `net_bnb_per_500`
2. trailing net profit
3. bet frequency
4. win rate
5. drawdown
6. deterioration or improvement vs earlier blocks

## Shadow Tracking Requirement

All candidate strategies should be shadow-tracked continuously.

Reason:

1. if only the active strategy is tracked, then non-active strategies will have stale or missing recent features
2. the meta-selector needs recent comparable state for every strategy in the universe

## Evaluation Method

Use strict walk-forward evaluation only.

At each block boundary:

1. build features from past only
2. predict next-block performance for each strategy
3. choose one action
4. score the realized next-block outcome
5. advance to the next block

## Main Metrics

Evaluate the meta-selector on:

1. realized `net_bnb_per_500`
2. total net `BNB`
3. max drawdown
4. bankroll floor when relevant
5. switching frequency
6. regret vs oracle best strategy per block
7. performance vs strong static baselines

## Baselines

Always compare against:

1. best static all-history strategy
2. best static recent-window strategy
3. oracle best-per-block strategy
4. `skip_all`

## First Version

The first implementation should be the simplest defensible version:

1. frozen strategy universe
2. decision every `500` rounds
3. predict each strategy's next-block `net_bnb_per_500`
4. choose the top predicted strategy unless the safety rule chooses `skip_all`
5. evaluate strictly with walk-forward blocks

## Explicit Non-Goals For V1

1. predicting market direction directly
2. changing the strategy universe mid-run
3. intrablock switching
4. hyper-complex ensemble logic before the simple block selector is validated

## Interpretation

This is a meta-selector over strategies, not a market-direction model.

The objective is to predict relative future strategy performance under the current regime and act accordingly.
