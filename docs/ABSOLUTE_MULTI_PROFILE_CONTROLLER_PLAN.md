# Absolute Multi-Profile Controller Plan

## Purpose

Define the main controller-research target clearly so the project does not drift
back to baseline-centered framing.

The target controller is:

1. estimate local next-window value for every candidate profile
2. include `skip` as a first-class action with value `0`
3. choose the best estimated action directly

Baseline-centered controllers were useful intermediate experiments, but they are
not the intended end state.

## Target Controller Contract

For each completed control window:

1. build estimates from past windows only
2. estimate each candidate profile on an absolute local `BNB / 500` scale
3. apply any explicit stability / uncertainty penalties
4. compare all profile estimates plus `skip = 0`
5. choose the highest penalized estimate

No candidate is structurally privileged. A historically strong profile may
remain in the pool, but only as one action among many.

## Non-Goals

1. Do not anchor the controller around one privileged baseline profile.
2. Do not optimize for “less bad than baseline” if all estimated actions are
   still negative.
3. Do not use shadow recommendation tooling as the main evidence source.
4. Do not promote runtime changes from this branch until offline causal
   evidence is strong enough.

## Primary Evidence Standard

Rolling causal backtests on completed windows are the source of truth.

For every evaluated window, the controller research artifacts should record:

1. chosen action
2. predicted `BNB / 500`
3. realized chosen `BNB / 500`
4. realized oracle `BNB / 500`
5. regret vs oracle
6. realized selected bet rate
7. whether `skip` was chosen

## Initial Execution Plan

1. Build a multi-profile absolute-value selector runner on top of the existing
   profile-set compare CSV format.
2. Start with simple absolute estimators:
   - trailing mean
   - recency-weighted mean
   - optional stability penalty
3. Include `skip = 0` directly in the action set.
4. Seed the first experiments with a small profile pool, not the full catalog.
5. Re-rank candidate profiles by absolute local usefulness:
   - positive windows
   - best-positive windows
   - skip replacement
   - causal extractability
6. Expand the pool only when it improves causal results, not merely oracle.

## Candidate Ranking Standard

Profiles should be judged primarily by:

1. how often they are actually positive on recent completed windows
2. how often they are the best positive action
3. how many current skip windows they replace
4. whether their positive pockets persist for multiple windows
5. whether they improve realized causal controller results

Bad global averages do not automatically disqualify a profile if it adds
distinct, causally extractable positive windows.

## Iteration Path

If the first simple absolute controller works:

1. refine its penalties / calibration
2. test broader recent completed-window sets
3. freeze one small profile pool and parameter set
4. write a runtime-controller spec
5. run a controlled controller-driven dry test

If it is mixed:

1. shrink or improve the profile pool
2. calibrate absolute floors / stability penalties
3. consider profile-specific local estimators
4. continue completed-window evaluation

If it fails:

1. pivot to better candidate generation
2. prefer profiles that are locally positive and skip-displacing
3. rebuild the pool around causal extractability rather than hindsight oracle

## Runtime Boundary

Runtime should remain contained until this absolute controller lane is clearly
qualified offline.

Dry-mode orchestration should be agent-owned during research:

1. use `run.py --sync-only` to refresh research inputs
2. start or stop dry runs only when they are specifically needed
3. use dry primarily for runtime sanity checks and later controlled rollout

Shadow tooling remains secondary. It may be used only as a final sanity check
immediately before any runtime-controller rollout.
