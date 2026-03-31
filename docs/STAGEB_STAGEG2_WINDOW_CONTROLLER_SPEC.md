# Historical Window Controller Spec

## Status

Historical only. This document is kept so older experiment references still
resolve, but it no longer describes the current controller target.

The active controller target is documented in:

- [ABSOLUTE_MULTI_PROFILE_CONTROLLER_PLAN.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/ABSOLUTE_MULTI_PROFILE_CONTROLLER_PLAN.md)
- [WINDOW_CONTROLLER_DRY_TEST_RUNBOOK.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/WINDOW_CONTROLLER_DRY_TEST_RUNBOOK.md)

## What Changed

The older `stageB` versus one alternate profile framing (`stageG2`, later
`cons`) was a useful stepping stone for shared-harness qualification, but it is
no longer the desired end-state controller. The current project direction is:

1. estimate absolute local value for each candidate profile
2. include `skip` as a first-class action with value `0`
3. choose the best estimated action directly

Baseline-centered switching is now treated as a bounded historical experiment,
not a structurally preferred controller design.

## Current Integrated Frontier

The strongest current integrated candidate is a one-profile absolute gate:

- mode: `absolute_best_with_skip`
- profile set: `disloc_stageB_bullonly_recent8pct_v1`
- cold start: `disloc_stageB_bullonly_recent8pct_v1`
- window rounds: `216`
- lookback windows: `2`
- min history windows: `2`
- estimator: `ewm_mean`
- `ewm_alpha = 0.5`
- stability penalty: `0.0`
- skip threshold: `0.05 / 500`

This is still experimental and not yet rollout-safe. See the dry-test runbook
for the current qualification gate and blocking evidence.
