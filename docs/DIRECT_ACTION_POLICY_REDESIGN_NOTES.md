# Direct Action Policy Redesign Notes

## Status

This document captures the agreed redesign direction from the `2026-03-31`
discussion. It is a planning note only. No redesign implementation has started
yet.

The detailed implementation contract now lives in:

1. [DIRECT_ACTION_POLICY_SPEC.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/DIRECT_ACTION_POLICY_SPEC.md)
2. [DIRECT_ACTION_POLICY_TRANSITION_PLAN.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/DIRECT_ACTION_POLICY_TRANSITION_PLAN.md)

## Why The Direction Changed

The current profile/controller/router stack is too complicated and is likely
the wrong runtime abstraction. It currently suffers from:

1. multiple layers that can independently decide `skip`
2. anchored block-based control instead of truly rolling local decisions
3. profile-selection logic that compares sparse realized outputs instead of
   directly scoring the runtime action set
4. weak confidence semantics even though confidence is central to the operator
   goal

The redesign goal is to replace that stack with one direct runtime policy.

## New Target Shape

The runtime should move toward:

1. one causal feature builder
2. one model family
3. one scorer over the full action set
4. one final runtime decision

The runtime should not depend on a separate profile selector, router, and
masking layer to arrive at the final action.

## Preferred Action Space

The preferred first action space is discrete and operator-readable:

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

This is preferred over:

1. one fixed bet size
2. a separate post-hoc stake-sizing subsystem
3. fully continuous sizing in the first simplified version

## Learning Framing

The preferred learning framing is contextual action-value estimation.

For each `(round, action)` pair:

1. build causal features available at decision time
2. assign the realized net `BNB` outcome for that exact action
3. train the model to estimate action value directly

At runtime:

1. remove infeasible actions, such as those blocked by bankroll
2. score every feasible action
3. choose the best positive risk-adjusted action
4. choose `Skip` otherwise

## Feature Direction

Rolling realized-window summaries remain useful, but only as features.

They should not remain the decision mechanism by themselves.

The intended feature families currently include:

1. current round / pool / imbalance / market state features
2. rolling causal regime summaries over multiple horizons
3. rolling realized-window summaries
4. optional legacy profile outputs as input features only

Legacy profile families may remain useful as:

1. feature inspiration
2. auxiliary feature generators
3. offline baselines
4. bounded references during comparison

They should not remain privileged runtime actions in the redesigned policy.

## Confidence And Uncertainty

Confidence is now a first-class optimization target.

The desired runtime behavior is not simply:

1. predict the side
2. bet often enough

It is:

1. estimate expected net utility per action
2. estimate uncertainty per action
3. use a conservative risk-adjusted score

Candidate scoring forms under consideration:

1. `predicted_mean_net - lambda * predicted_uncertainty`
2. direct lower-confidence-bound scoring
3. quantile-based scoring such as `q10(action)`

The current preferred direction is a quantile-style or otherwise directly
interpretable lower-confidence-bound approach.

## Simplicity Rules

The redesign should explicitly favor:

1. one final decision owner
2. one clear operator-facing action each round
3. no multi-layer `skip`
4. no anchored static controller windows
5. no structurally privileged baseline/profile inside the runtime policy
6. no separate stake-sizing subsystem in the first version
7. offline training and frozen runtime inference for the first version

## Planning Items Still To Settle

Before implementation, the plan still needs to define:

1. exact runtime contract
2. exact label definition
3. exact feature families
4. exact uncertainty method
5. exact offline evaluation protocol
6. exact rollout criteria
7. what logic and experiment bulk should be archived versus retained

## Interim Rule

Until that plan is written and approved, the current direct-action redesign is
spec work only. Do not start patch-driven implementation from this note alone.
