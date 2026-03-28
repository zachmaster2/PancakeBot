# Flow Requalification Plan

## Purpose

Contain the current dry/live bankroll risk, then re-qualify the flow overlay
under a stricter promotion standard.

This plan is approved as the next execution sequence, but it is not to be
started until the user explicitly says `GO`.

## Why This Is Needed

The currently promoted hybrid runtime (`stageB + flow`) was recently promoted
from encouraging short recent-tail backtests, but the subsequent dry run and
matched recent-window backtest both showed material losses concentrated in the
flow overlay, especially on `Bear`.

That means the current profile is not robust enough to remain promoted without
re-qualification.

As of `2026-03-27`, the first re-qualification pass has tightened this
conclusion:

- standalone `flow Bull` is currently not useful on the latest recent tails
- standalone `flow Bear` can still be locally positive
- but direct shared-pipeline reintegration of `flow Bear` is still materially
  worse than the contained `stageB` runtime on the exact recent failure pocket
- and a later offline score-penalized combine check showed only mixed relief:
  a high `0.10 BNB` penalty can help one exact failure pocket, but the same
  direct overlay lane is still not robust across rolling recent windows

So the plan now pivots from "re-qualify direct flow overlay" to
"keep runtime contained and only test flow through explicit constrained lanes."

## Execution Plan

1. Demote the current hybrid from dry/live.
   - Remove the current free-running flow overlay from the promoted runtime.
   - Use `stageB`-only, or `stageB + flow shadow-only`, as the containment
     baseline while research continues.

2. Split flow by side.
   - Treat flow `Bull` and flow `Bear` as separate candidates for research.
   - Do not assume a shared gate or shared calibration is valid across sides.

3. Add hard flow safety gates.
   - side-specific cooldown
   - side-specific recent net / win-rate gate
   - stronger override threshold before flow can beat `stageB`
   - emergency disable path after recent realized underperformance

4. Pivoted qualification lanes.
   - Lane A: shadow-only / research-only flow
     - do not let flow affect dry/live bankroll
     - keep collecting side-split quality and calibration evidence
   - Lane B: constrained direct overlay
     - allow only explicitly score-penalized or threshold-penalized flow
       candidates to compete with `stageB`
     - default target is `flow Bear` only
     - free-running `flow both` is no longer an approved test lane
   - Lane C: flow-as-signal research
     - if direct overlay remains weak, test whether flow is more useful as a
       veto, confirmer, or shadow regime feature for other candidates
   - Lane D: offline combination simulation
     - if full shared-pipeline reruns are too slow for broad sweeps, use
       aligned stageB/flow trade outputs to test combine rules offline first
     - only re-run the full shared pipeline for the small number of combine
       rules that look promising offline

5. Re-run rolling-window qualification.
   - `flow Bull only`
   - `flow Bear only`
   - `stageB + score-penalized flow Bear only`
   - `stageB + flow shadowed Bear`
   - use rolling recent windows, not a single favorable tail
   - if constrained direct overlay still looks weak, prefer explicit
     profile/window controllers with `stageB`, `flow Bear`, and `skip`
     as separate actions

6. Promotion standard.
   A flow variant should not be re-promoted unless it clears all of:
   - positive mean recent `BNB / 500`
   - acceptable worst rolling window
   - no catastrophic drag by side
   - activity that still meets the current practical target
   - dry-shadow behavior consistent with backtest expectations
   - no evidence that the direct overlay simply overwhelms `stageB` in bad
     pockets

## Operator / Observability Requirement

Use `var/runtime/dry_cycle_audit.csv` as the truth source for dry decisions.
For every future dry run, inspect:

- `observed_*` pool fields: raw post-wake snapshot
- `cutoff_used_*` pool fields: cutoff-filtered decision inputs

This avoids mixing raw observed pool totals with the actual pool state used by
the strategy logic.

## Immediate Non-Goals

- Do not start broad V2 work.
- Do not promote another hybrid from a single short tail.
- Do not assume a good `15k` slice implies robustness.
- Do not re-enable direct flow in the shared runtime until one of the pivoted
  constrained lanes clears the promotion standard.

## Current Direction

As of the latest rolling-window offline study, the strongest active lane is no
longer direct flow overlay. It is a skip-aware profile/window controller over
`stageB` and `flow Bear`.

That controller lane is still offline-only, but it is now strong enough to
justify expanding the profile universe around it before any runtime promotion:

- best causal `216`-round controller: about `+0.320889 / 500`
- best causal `300`-round controller: about `+0.221581 / 500`
- best causal `500`-round controller: about `+0.213429 / 500`

Each of those also cleared the current `>= 5%` selected-bet-rate requirement.

The next approved pivot follows from the first broader flow-only profile-set
expansion:

- adding several nearby `flow Bear` variants raised the `216`-window oracle
  materially
- but the best causal controller dropped to about `+0.201732 / 500`
- and the best causal row chose only `stageB` plus `skip`, with no flow picks

So the next branch should not be "more nearby flow variants." It should be
"broader profile-family expansion" using other already-defined candidate
families, starting with the existing dislocation-side `stageG2` bull-only
profile.

If that broader-family expansion still raises oracle much faster than it raises
the best causal heuristic, the next approved pivot is:

- keep runtime contained
- stop adding more heuristic-only controller rules as the main lane
- build a past-only, feature-based profile-window controller instead
- predict profile choice relative to `stageB`, with explicit `skip`
- judge it on the same window-level metrics:
  - mean `BNB / 500`
  - selected bet rate
  - switches
  - comparison against the best heuristic controller on the same window set

That pivot is now active and materially better than the heuristic lane:

- on the `stageB + stageG2` `216`-window set, the best model controller reached
  about `+0.361798 / 500`, versus heuristic control that stayed near flat
- on the mixed non-overlapping `216`-window set (`stageB`, `stageG2`, and
  selected `flow Bear` variants), the best current model controller reached
  about `+0.578433 / 500` at selected bet rate about `7.29%`
- the best heuristic controller on that same mixed set was only about
  `+0.270527 / 500`

The current next step is therefore not runtime promotion. It is:

- keep runtime contained on `stageB` only
- treat the mixed-profile model controller as the strongest offline lane
- validate whether it remains strong enough to become a shadow-only
  recommender before any live/dry control change

That shadow-only lane is now tooled in
`inspection/run_profile_set_shadow_recommender.py`, which writes a current
next-window recommendation JSON from a completed mixed-profile compare set
without touching runtime control.
