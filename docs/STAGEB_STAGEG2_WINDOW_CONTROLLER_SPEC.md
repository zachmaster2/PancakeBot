# StageB StageG2 Window Controller Spec

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
2. `stageG2_bullonly`
3. optional `skip`

The current primary lane is the two-profile controller plus optional skip.

## Decision Cadence

1. Use non-overlapping `216`-round windows.
2. Make one controller decision for the next window.
3. Use only completed prior windows when choosing.

## Current Candidate Rules

### Practical no-skip controller

- mode: `trailing_best_vs_stageb`
- lookback: `2`
- margin: `1.0 / 500`
- actions: `stageB` or `stageG2_bullonly`

Current shared-harness evidence:

- latest `6480`-round tail:
  about `+0.044224 / 500`, bet rate about `8.92%`
- latest `8640`-round tail:
  about `+0.083013 / 500`, bet rate about `8.31%`
- latest `10800`-round tail:
  about `+0.019785 / 500`, bet rate about `9.06%`
- multi-offset shared check:
  - `6480`: mean about `+0.180962 / 500`, beat static `stageB` on `5/5`
  - `8640`: mean about `-0.005180 / 500`, beat static `stageB` on `2/5`
  - `10800`: mean about `+0.019460 / 500`, beat static `stageB` on `5/5`

### Safer skip-aware controller

- mode: `trailing_best_vs_stageb_with_skip`
- lookback: `5`
- margin: `1.0 / 500`
- skip threshold: `0.0`

Current shared-harness evidence:

- latest `6480`-round tail:
  about `+0.009785 / 500`, bet rate about `6.67%`
- latest `8640`-round tail:
  about `+0.105242 / 500`, bet rate about `4.76%`
- latest `10800`-round tail:
  about `+0.012678 / 500`, bet rate about `4.88%`

Interpretation:

- the skip-aware variant can still be stronger on isolated tails
- but it currently falls below the practical `5%` floor too often in the
  continuous shared harness
- the no-skip variant is the stronger runtime-controller candidate right now,
  but it is still not rollout-safe because the `8640` multi-offset slice
  remains mixed

## Profile Definitions

### Baseline profile

- `disloc_stageB_bullonly_recent8pct_v1`

### Alternate profile

- `disloc_stageG2_bullonly_recent5pct_v1`

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

1. Keep runtime contained on `stageB` only.
2. Refresh the compare set on current data.
3. Emit the current heuristic recommendation offline.
4. Compare refreshed recommendations over time.
5. Only then implement the runtime-controller path.

This implementation step is now done experimentally. The controller path exists
in shared backtest/dry/live code, but it remains disabled by default and should
still be treated as research-only until broader continuous-run evidence is
stronger than static `stageB`.

## Non-Goals

1. Do not restore direct flow overlay from this spec.
2. Do not mix heuristic and model actions inside the same first rollout.
3. Do not make round-level controller decisions.
