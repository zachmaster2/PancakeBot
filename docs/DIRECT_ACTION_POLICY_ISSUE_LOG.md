# Direct Action Policy Issue Log

## Status

This document records issues encountered during the redesign transition and the
chosen resolution for each one.

## Issue 1: Dirty Controller-Era Worktree

Issue:

1. the repo already contained a large uncommitted controller-era diff when the
   direct-action redesign started

Resolution:

1. preserve that state as a historical checkpoint commit before starting the
   direct-action implementation
2. do not silently discard it

## Issue 2: Old Runtime Default Still Points At Controller-Era Logic

Issue:

1. the redesign direction changed before the new direct-action lane existed

Resolution:

1. keep the new direct-action path optional and disabled by default during its
   first implementation
2. defer changing the runtime default until the new lane is qualified

## Issue 3: Risk Of Reintroducing Complexity Through Legacy Profiles

Issue:

1. legacy profile outputs could help as features, but they also risk dragging
   the redesign back toward profile-centric runtime logic

Resolution:

1. exclude legacy profile outputs from the first direct-action feature set
2. treat them as possible later additions only if the simpler lane clearly
   needs them

## Issue 4: Confidence Needs To Be Real, Not Cosmetic

Issue:

1. confidence was explicitly identified as a core operator requirement, but the
   old stack did not model it well

Resolution:

1. use quantile action-value models in the first version
2. drive the runtime score directly from the lower-confidence bound

## Issue 5: Runtime Integration Could Drift Back Into A Router Layer

Issue:

1. the easiest patch would be to feed the new action model into the existing
   router path

Resolution:

1. integrate the direct-action lane as its own decision path in the shared
   pipeline
2. when enabled, it is the final decision owner

## Issue 6: Top-Level Policy Ambiguity

Issue:

1. the base runtime config already enabled the legacy window controller, so
   direct-action experiments could accidentally activate two different
   top-level decision owners at once

Resolution:

1. reject the conflict in both config loading and scenario override handling
2. require direct-action experiments to disable the window controller
   explicitly

## Issue 7: Disabled Bundle Path Must Allow Empty String

Issue:

1. the generic string parser rejected empty strings, but a disabled
   direct-action config legitimately needs an empty `model_bundle_path`

Resolution:

1. parse `model_bundle_path` with direct-action-specific logic
2. allow empty string when disabled
3. enforce non-empty path only when the direct-action policy is enabled

## Issue 8: Raw `q10` Collapsed The First Smoke Run To All-Skip

Issue:

1. the first held-out smoke qualification on `6480` rounds produced `0` bets
2. the direct cause was that raw `q10` scores were negative for every non-skip
   action on every round
3. the pooled lower-quantile model effectively learned a generic negative lower
   tail instead of usable action-specific lower bounds

Resolution:

1. stop treating raw `q10` as the runtime score
2. move to a simpler explicit lower-confidence-bound style score:
   `q50 - lambda * (q50 - q10)`
3. reintroduce bounded legacy dislocation candidate outputs as auxiliary
   features only

## Issue 9: Pooled Lower Quantiles Did Not Calibrate `skip` Or Action Identity

Issue:

1. even after adding explicit action-id features, the pooled `q10` model still
   predicted an almost identical negative lower tail for `skip` and most bet
   actions
2. this was a model-family problem, not just a logging or threshold issue

Resolution:

1. keep the unified runtime policy contract
2. replace the pooled quantile bundle with per-action quantile heads
3. keep `skip` as a constant zero quantile model

## Issue 10: The Per-Action Realized-Net Quantile Lane Is Still Not Qualified

Issue:

1. the per-action smoke run on the same held-out `6480` rounds escaped the
   all-skip failure mode but overbet large sizes and lost about `-49.999 BNB`
2. normalized result was about `-3.8579 / 500`
3. simple score-threshold tightening did not rescue the path; every tested
   threshold remained negative

Resolution:

1. stop short of the full shared-eval sweep for this model family
2. mark the current realized-net quantile lane as unqualified
3. treat the next blocker as a target/model-contract problem, not a missing
   threshold tune
