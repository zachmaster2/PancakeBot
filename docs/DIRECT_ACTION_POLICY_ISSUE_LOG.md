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
