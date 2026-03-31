# Direct Action Policy Spec

## Purpose

Define the first clean redesign target before implementation.

The goal is to replace the current profile/controller/router stack with one
direct runtime policy that:

1. consumes one causal feature snapshot per round
2. scores the full action set directly
3. emits exactly one final action
4. makes confidence part of the decision rule

This document is the implementation contract for the first simplified version.

## Implementation Status

As of `2026-03-31`, the first implementation tranche exists:

1. optional config path
2. offline dataset + quantile bundle training utilities
3. shared pipeline/runtime/backtest inference path
4. dedicated direct-action logging and audit fields
5. first shared walk-forward evaluation harness

Still intentionally deferred:

1. runtime promotion to default
2. offline qualification results strong enough to justify promotion
3. dry-mode rollout of the direct-action lane

## Design Goals

The redesign must satisfy all of the following:

1. maximize recent `BNB / 500` under regime drift
2. remain operator-readable while running
3. have one clear decision owner
4. make `Skip` an explicit first-class action
5. use confidence directly in action selection
6. avoid layered routing, masking, and post-hoc stake sizing

## Non-Goals

The first version does not try to:

1. preserve the current controller/runtime abstraction
2. keep profile selection as the runtime policy
3. support continuous bet sizing
4. support live online learning
5. optimize for a minimum bet rate

Low bet rate is acceptable if it comes from calibrated confidence and produces
better recent realized results.

## Runtime Contract

For each open round, the runtime must do exactly this:

1. build one causal feature snapshot from information available before betting
2. enumerate the allowed action set
3. remove infeasible actions, such as those blocked by bankroll
4. score each feasible action with one model family
5. choose the highest positive risk-adjusted action
6. choose `Skip` if no non-skip action clears the threshold
7. emit one operator-facing decision record

There should be no separate controller, router, profile mask, or secondary
stake-sizing subsystem in the decision path.

## Action Space

The first action set is discrete and fixed:

1. `Skip`
2. `Bull @ 0.05`
3. `Bull @ 0.10`
4. `Bull @ 0.15`
5. `Bull @ 0.25`
6. `Bull @ 0.35`
7. `Bull @ 0.50`
8. `Bear @ 0.05`
9. `Bear @ 0.10`
10. `Bear @ 0.15`
11. `Bear @ 0.25`
12. `Bear @ 0.35`
13. `Bear @ 0.50`

Each non-skip action is defined by:

1. side
2. nominal bet size

The first version does not include:

1. a separate dynamic stake multiplier
2. a side-only action without size
3. a continuous-size action

## Decision Rule

The runtime decision rule is:

1. compute a conservative action score for every feasible action
2. choose the action with the highest score
3. if the best non-skip score is not positive, choose `Skip`

The initial score contract is a lower-confidence-bound style score:

1. preferred form: `score(action) = q10_net_bnb(action)`
2. acceptable equivalent: `q50_net_bnb(action) - lambda * uncertainty(action)`

The implementation should expose the chosen score form explicitly in config and
logs. The first version should keep the scoring rule simple and stable rather
than highly configurable.

## Learning Formulation

The preferred formulation is contextual action-value estimation.

Each training row corresponds to one `(round, action)` pair and contains:

1. causal round features
2. causal rolling/regime features
3. action identity features
4. realized net outcome for that action

This keeps every action in one unified scoring problem and avoids treating any
action or legacy profile as special.

## Label Definition

The target label for each `(round, action)` pair is realized net `BNB`.

For non-skip actions:

1. if the action would place a bet, realized net must include:
   - payout credit when the side wins
   - minus the staked `BNB`
   - minus bet gas cost
   - under the actual treasury fee semantics
2. if the round refunds, the label must follow the same settlement semantics
   already used by runtime/backtest
3. if the action is infeasible for a hypothetical bankroll constraint, that is
   not encoded in the label; infeasibility is handled at runtime filtering time

For `Skip`:

1. realized net is exactly `0.0`

The label definition must reuse one canonical settlement function shared by:

1. offline dataset generation
2. backtest evaluation
3. dry/runtime settlement accounting

## Feature Contract

All features must be causal.

Allowed feature families in the first version:

1. current round state
2. current pool state
3. current imbalance / side support features
4. current derived market state features
5. rolling realized-window summaries over multiple horizons
6. rolling regime summaries over multiple horizons
7. action identity features
8. optional bounded legacy-profile outputs as features only

Initial horizon set for rolling summaries:

1. short: `24` rounds
2. medium: `72` rounds
3. long: `216` rounds

Those values are starting defaults, not sacred constants. The first version
should keep the number of horizons small and explicit.

### Current-Round Feature Families

The first version should favor a small, readable base set:

1. pool totals
2. bull ratio / bear ratio
3. projected late imbalance if available
4. nowcast probabilities or equivalent direct side-support estimates
5. recent volatility / change summaries already available causally

### Rolling Summary Families

Rolling summaries should be built only from settled prior rounds and may
include:

1. realized net by action family or side family
2. realized win rate
3. realized mean payout ratio
4. realized opportunity density
5. realized drawdown summaries
6. realized volatility of action outcomes

The purpose of these features is to inform regime state, not to serve as the
runtime decision mechanism by themselves.

### Legacy Profile Features

Legacy profile outputs are optional and bounded in scope.

If included in the first version, they may be used only as:

1. auxiliary features
2. offline comparison baselines

They must not reintroduce:

1. profile switching as the runtime policy
2. baseline-centered masking logic
3. multiple layers that can independently decide `Skip`

## Uncertainty Contract

Confidence is a required part of the decision contract.

The first version should use an uncertainty-aware action-value model with a
directly interpretable conservative score.

Preferred first approach:

1. quantile regression over realized net for each action
2. runtime scoring by lower confidence bound, preferably `q10`

Why this is preferred:

1. it is operator-legible
2. it aligns directly with “wait for the juicy rounds”
3. it avoids turning bet-rate floors into a primary target
4. it makes confidence a property of expected utility, not just direction

If quantile modeling proves impractical, the first fallback is:

1. mean net plus explicit uncertainty estimate
2. runtime score `mean - lambda * uncertainty`

Ensemble-based uncertainty is a later option, not a first-version requirement.

## Operator Logging Contract

The runtime log stream must make the final decision obvious.

Each round should expose one stable decision line containing:

1. epoch
2. chosen action
3. chosen action score
4. chosen action confidence band or uncertainty summary
5. best alternative score when useful
6. `Skip` reason when `Skip` is chosen
7. feasibility drops when an action was removed due to bankroll

The operator should not need to infer the action by combining multiple log
layers.

## Offline Evaluation Contract

The first redesign must be judged primarily by rolling causal evaluation.

Evaluation should use walk-forward or rolling train/validation/test splits over
recent history, not random shuffle.

Primary decision metrics:

1. recent realized `BNB / 500`
2. recent realized drawdown
3. bankroll-floor behavior where relevant
4. regime sensitivity across multiple recent windows
5. calibration of the conservative score

Secondary metrics:

1. bet rate
2. skip rate
3. side balance
4. size distribution

Bet rate is a consequence, not the main target.

## Baseline Comparison Contract

The first direct-policy lane should still be compared against bounded
references:

1. contained static `stageB`
2. any other currently credible static baseline
3. the best recent controller-era research artifacts as historical references

Those references are comparison targets only. They are not the runtime design
target.

## Rollout Gate

Before dry-mode rollout, the first direct-policy lane should clear all of:

1. positive recent causal `BNB / 500` on the primary evaluation windows
2. acceptable drawdown relative to bounded references
3. no obvious collapse on nearby recent subwindows
4. stable confidence behavior under walk-forward validation
5. operator-readable runtime logs from a dry/inference-only harness

Dry mode should be used after offline qualification, not as the main search
loop.

## Implementation Order

Implementation should proceed in this order:

1. canonical settlement/label reuse contract
2. offline dataset builder for `(round, action)` rows
3. offline evaluator for causal walk-forward scoring
4. first direct-action model with uncertainty-aware scoring
5. dry/backtest inference harness
6. runtime integration
7. operator logging polish

This order is intended to keep evaluation ahead of runtime complexity.
