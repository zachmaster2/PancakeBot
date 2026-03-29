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

That shadow-only lane is still available in
`inspection/run_profile_set_shadow_recommender.py`, but it is no longer the
primary research lane. The primary evaluation path is now causal rolling
backtests on completed windows. Shadow recommendation tooling remains useful
only as a final sanity check before any runtime-controller rollout.

## Next-Phase Roadmap

The next full branch after the current shadow-only lane is:

1. Candidate-mining pass over older profiles.
   - Replay older or previously rejected profiles on the current
     non-overlapping `216`-window framing.
   - Score them by controller-set value, not standalone mean.
   - Focus on:
     - replacing current `skip` windows
     - distinct wins versus the current mixed pool
     - multi-window persistence

2. Small expanded-pool validation.
   - Add only the best `2-5` mined profiles to the current mixed pool.
   - Rebuild the compare set.
   - Re-run the model controller and compare against the current bar:
     about `+0.578433 / 500` at about `7.29%` selected bet rate.

3. Causal rolling validation.
   - If the expanded-pool controller improves materially, keep runtime
     contained and validate it first through rolling causal backtests on
     completed windows.
   - Compare realized controller picks against realized `stageB`, oracle, and
     regret.

4. Runtime-controller gate.
   - Only if rolling causal behavior remains strong should the next step
     become a written runtime-controller spec and later a controlled dry
     rollout.
   - Shadow may be used at the very end as a final sanity check, but not as
     the main evidence lane.

5. Fallback path if expansion fails.
   - If added profiles raise oracle but not the model controller, prune the
     pool and pivot toward stronger feature/controller work instead of simply
     adding more profiles.

That fallback is now active:

- the mining pass over older dislocation profiles is implemented in
  `inspection/run_profile_candidate_miner.py`
- on the current mixed `216`-window framing, mined additions did not replace
  the current oracle skip windows
- adding the best mined profiles (`stageG2_r37_x80` or `altB_20260227_x80`)
  made the best model controller worse than the current mixed-pool leader

So the next improvement lane should revert to stronger feature/controller work,
not broader profile expansion.

That stronger feature/controller branch is now partly explored too:

- richer generic window features plus the existing linear family dropped the
  best result to about `+0.356034 / 500`
- adding cold-start heuristics to that richer ridge lane recovered only to
  about `+0.421560 / 500`
- a focused HGB lane reached only about `+0.178725 / 500`

So the next pivot should not be "more generic features" or "more nonlinear
model families" on this small mixed-window set.

## Current Controller Pivot

Use the existing best simple-feature mixed controller as the calibration base.

1. Keep the current mixed profile set fixed.
2. Keep the current simple-feature `delta_ridge` controller at about
   `+0.578433 / 500` as the reference bar.
3. Add profile-specific entry penalties and stricter skip calibration on top
   of that controller.
4. Search whether those penalties can remove the concentrated bad
   alternate-profile entries without killing the good pockets.
5. Only if that calibration beats the current bar should it move into the
   shadow recommendation lane.

That calibration branch is now positive:

- [profileset216_stageb_stageg2_flowbear4_penalty_focus_20260328_profile_set_penalty_selectors.csv](/C:/Users/zking/Documents/GitHub/PancakeBot_var_exp/profileset216_stageb_stageg2_flowbear4_penalty_focus_20260328_profile_set_penalty_selectors.csv)
  raised the best mixed `216`-window controller from about `+0.578433 / 500`
  to about `+0.643959 / 500`
- selected bet rate stayed about `7.50%`
- the strongest stable zone uses:
  - legacy/simple feature set
  - `min_train_windows=10`
  - cold-start `trailing_best_vs_stageb_with_skip`, lookback `5`
  - `ridge_alpha` around `1.0-5.0`
  - flow penalty around `0.2-0.4`
  - no meaningful `stageG2` penalty

The next step is therefore not runtime promotion. It is stronger rolling
causal validation of this calibrated controller on completed windows.
Shadow-only tooling still exists, but it is secondary.

On the refreshed current-data compare set (`2026-03-29`), the calibrated
controller is still positive but weaker than the older frozen result:

- best refreshed calibrated controller:
  about `+0.284473 / 500`
- static `stageB` on that same refreshed compare:
  about `+0.115151 / 500`

The current high-signal completed-window comparison is:

- refreshed calibrated controller:
  about `+0.284473 / 500`
- static `stageB` on that same refreshed compare:
  about `+0.115151 / 500`
- on the latest completed refreshed `216`-round window, the calibrated
  controller would have chosen `skip`, realizing `0.0 / 500`
- on that same window, realized `stageB` was about `-0.693855 / 500`
- hindsight best profile on that same window was `flow_bear_loose10` at about
  `+1.304968 / 500`

So the controller is still beating contained `stageB` on current completed
windows, but not extracting the full oracle.

The current next step remains:

- keep runtime contained
- keep using rolling causal backtests as the primary evidence source
- use shadow only as a thin final sanity check before any runtime-controller
  spec or rollout
