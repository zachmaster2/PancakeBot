# Direct Action Policy Transition Plan

## Purpose

Define how to move from the current controller-era codebase to the direct-action
policy lane without creating more repo drift.

This is a planning document only. It does not authorize destructive cleanup on
its own.

## Implementation Status

As of `2026-03-31`, the transition is partway complete:

1. the controller-era mixed worktree was preserved as historical checkpoint commit `081be56`
2. the direct-action spec/decision/issue docs are frozen enough for implementation
3. the first direct-action offline/runtime/backtest scaffold is implemented
4. runtime default promotion and broad legacy cleanup are still deferred

## Transition Principles

The transition must satisfy:

1. keep git history understandable
2. keep the active repo lean
3. preserve only bounded historical references
4. avoid mixing cleanup, archival, and redesign implementation in one diff

## What Should Remain Active

The following should remain active because they are still useful for the
redesign:

1. canonical runtime/backtest settlement logic
2. causal feature-building infrastructure that is still sound
3. contained static baselines needed for comparison
4. small durable docs explaining past results and the new target
5. any offline tooling that directly helps build the new direct-action dataset

## What Should Become Historical

The following should become historical references rather than active direction:

1. baseline-centered controller specs
2. profile-window switching as the main runtime policy
3. block-anchored window-controller runtime logic
4. controller dry-test orchestration as the main path forward
5. older research branches whose only purpose was profile/controller selection

Historical does not necessarily mean immediate deletion. It means:

1. not the mainline target
2. not the default runtime path
3. not the primary place new complexity gets added

## What Should Be Archived Outside The Active Path

The redesign should explicitly consider archiving or quarantining:

1. bulk controller-era experiment outputs that are no longer decision-relevant
2. docs whose only purpose is the superseded controller rollout path
3. runtime/controller helpers that are not needed for bounded comparison

Experiment data should remain outside the repo under `../PancakeBot_var_exp/`
or archive directories. The active repo should retain only durable summaries and
minimal bounded references.

## Git-Clean Transition Plan

Before implementation begins, use a deliberate transition sequence:

1. freeze the redesign spec and transition-plan docs
2. decide which existing uncommitted controller-era changes are worth keeping as
   a historical checkpoint
3. separate planning/spec changes from runtime/controller code changes
4. reduce the active worktree to a small coherent starting point
5. start the direct-action implementation from that coherent base

The key rule is:

1. do not begin implementation while the worktree still contains a broad mixed
   controller-era diff unless that diff has been intentionally classified

## Proposed Work Phases

### Phase 0: Spec Freeze

1. finalize the direct-action runtime contract
2. finalize the training/evaluation contract
3. finalize the archive boundary

### Phase 1: Historical Boundary

1. decide which controller-era code remains as bounded reference
2. decide which controller-era docs remain as bounded reference
3. move obsolete notes and artifacts out of the active path where appropriate

### Phase 2: Clean Starting Point

1. ensure the repo state for redesign work is small and coherent
2. keep only the spec/plan docs plus any explicitly retained prerequisites
3. avoid bringing forward unrelated controller-era experiments by default

### Phase 3: Offline First

1. build the direct-action dataset generator
2. build the causal evaluator
3. prove the lane offline before runtime integration

Status:

1. initial dataset generator and shared eval harness implemented
2. qualification evidence still pending

### Phase 4: Inference Harness

1. add dry/backtest inference for the direct policy
2. validate operator logs
3. validate settlement parity against the offline labels

Status:

1. shared backtest/runtime inference path implemented behind feature flag
2. operator-facing direct-action logs/audit fields implemented
3. runtime default promotion still deferred

### Phase 5: Runtime Rollout

1. enable controlled dry-mode rollout
2. monitor behavior
3. decide promotion only after offline and dry evidence agree

## Immediate Pre-Implementation Checklist

Before any runtime code is written, the following must be true:

1. the direct-action spec is approved
2. the transition plan is approved
3. the initial action grid is frozen
4. the initial label contract is frozen
5. the initial feature families are frozen
6. the initial uncertainty score is frozen
7. the active worktree has been intentionally cleaned up

If any of those are still unsettled, remain in spec mode.
