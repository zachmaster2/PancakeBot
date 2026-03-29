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
- lookback: `1`
- margin: `0.5 / 500`
- actions: `stageB` or `stageG2_bullonly`

Current completed-window evidence:

- latest `30k` / `30` windows:
  about `+0.302818 / 500`, selected bet rate about `7.28%`
- recent `40k` / `40` windows:
  about `+0.262321 / 500`, selected bet rate about `7.34%`
- latest `50k` / `50` windows:
  about `+0.216263 / 500`, selected bet rate about `7.62%`

### Safer skip-aware controller

- mode: `trailing_best_vs_stageb_with_skip`
- lookback: `1`
- margin: `0.5 / 500`
- skip threshold: `0.0`

Current completed-window evidence:

- latest `30k` / `30` windows:
  about `+0.314530 / 500`, selected bet rate about `4.97%`
- recent `40k` / `40` windows:
  about `+0.284033 / 500`, selected bet rate about `5.22%`
- latest `50k` / `50` windows:
  about `+0.273838 / 500`, selected bet rate about `5.06%`

Interpretation:

- the skip-aware variant is slightly stronger on `BNB / 500`
- the no-skip variant is safer against falling below the practical `5%` floor

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

## Non-Goals

1. Do not restore direct flow overlay from this spec.
2. Do not mix heuristic and model actions inside the same first rollout.
3. Do not make round-level controller decisions.
