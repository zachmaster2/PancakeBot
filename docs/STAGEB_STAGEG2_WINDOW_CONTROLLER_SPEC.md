# StageB Cons Window Controller Spec

## Status

Research spec only. Runtime is still contained on `stageB` only.

## Purpose

Define the first plausible runtime-controller design that is still simple
enough to understand operationally:

- window-based decisions
- explicit actions
- no round-level overlay mixing
- no flow in the primary action set

## Action Set

1. `stageB`
2. `cons`

The current primary lane is the two-profile no-skip controller.

## Decision Cadence

1. Use non-overlapping `216`-round windows.
2. Make one controller decision for the next window.
3. Use only completed prior windows when choosing.

## Current Candidate Rules

### Practical no-skip controller

- mode: `trailing_best_vs_stageb`
- lookback: `3`
- margin: `1.0 / 500`
- actions: `stageB` or `cons`

Current shared-harness evidence:

- `6480` over `20` offsets:
  mean about `+0.087450 / 500` vs static `stageB` about `+0.062812 / 500`
  mean lift about `+0.024638 / 500`
  beat static `stageB` on `13/20`
  worst lift about `-0.007128 / 500`
- `8640` over `20` offsets:
  mean about `+0.087028 / 500` vs static `stageB` about `+0.046927 / 500`
  mean lift about `+0.040101 / 500`
  beat static `stageB` on `18/20`
  worst lift about `-0.016351 / 500`
- `10800` over `20` offsets:
  mean about `+0.124054 / 500` vs static `stageB` about `+0.093709 / 500`
  mean lift about `+0.030345 / 500`
  beat static `stageB` on `18/20`
  worst lift about `-0.013081 / 500`
- `12960` over `10` offsets:
  mean about `+0.121509 / 500` vs static `stageB` about `+0.103206 / 500`
  mean lift about `+0.018303 / 500`
  beat static `stageB` on `9/10`
  worst lift about `-0.001841 / 500`

Interpretation:

- this is the first shared-harness controller branch that is positive across
  all major recent horizons after broader offset expansion
- bet rate stays safely above the practical `5%` floor
- the branch is now a serious controller dry-test candidate, but it still
  needs explicit runtime smoke validation and a controlled dry rollout before
  any live-money consideration

## Profile Definitions

### Baseline profile

- `disloc_stageB_bullonly_recent8pct_v1`

### Alternate profile

- `disloc_cons_20260227_x80`

Flow is not part of the primary controller spec right now. It remains a
secondary rehabilitation branch.

## Runtime Contract Requirements

Any runtime implementation of this controller must:

1. keep decisions window-based and explicit
2. log the active controller mode and chosen profile per window
3. persist controller state so restart/resume is deterministic
4. compute controller inputs from completed windows only
5. avoid hidden round-level overlay interactions

## Required Runtime Inputs

The controller needs a closed-window scoreboard for both profiles:

1. per-window realized `per_500`
2. per-window realized bet rate
3. latest completed window identity

The runtime must obtain those values causally from completed rounds. It must
not use current-window realized data when choosing.

## Recommended Rollout Sequence

1. Keep runtime contained on `stageB` only by default.
2. Refresh the compare set on current synced data.
3. Re-run the shared qualification on the current controller setting.
4. Use the controller-specific dry runbook for a real dry test.
5. Only after a clean controller dry run should live consideration begin.

This implementation step is now done experimentally. The controller path exists
in shared backtest/dry/live code, but it remains disabled by default and should
still be treated as research-only until broader continuous-run evidence is
stronger than static `stageB`.

## Non-Goals

1. Do not restore direct flow overlay from this spec.
2. Do not mix heuristic and model actions inside the same first rollout.
3. Do not make round-level controller decisions.
