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

4. Re-run rolling-window qualification.
   - `flow Bull only`
   - `flow Bear only`
   - `stageB + flow Bull only`
   - `stageB + flow shadowed Bear`
   - use rolling recent windows, not a single favorable tail

5. Promotion standard.
   A flow variant should not be re-promoted unless it clears all of:
   - positive mean recent `BNB / 500`
   - acceptable worst rolling window
   - no catastrophic drag by side
   - activity that still meets the current practical target
   - dry-shadow behavior consistent with backtest expectations

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
