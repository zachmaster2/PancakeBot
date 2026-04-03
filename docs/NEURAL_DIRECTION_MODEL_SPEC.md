# Neural Direction Model Spec

## Purpose

Define the reset path after the failed controller and direct-action lanes.

The new target is intentionally simpler:

1. one model
2. one prediction per `target_round`
3. prediction must be `Bull` or `Bear`
4. no `skip` in the model output contract
5. headline evaluation metric is out-of-sample win `%`

This document is the current design contract for that pivot.

## Terminology

Use the canonical vocabulary in [TERMINOLOGY.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/TERMINOLOGY.md).

The neural direction lane must use these terms exactly:

1. `target_round`: the round being predicted
2. `open_round`: the live causal state of `target_round`
3. `locked_round`: the immediately prior round
4. `prior_context_rounds`: ordered round context immediately preceding
   `target_round`, with `locked_round` at `prior_context_rounds[-1]`
5. `outcome_eligible_prior_context_rounds`: `prior_context_rounds[:-1]`
6. `context_klines`: kline context anchored to the `target_round` cutoff
7. `target cutoff pools`: target-round pools computed only from bets with
   `created_at <= cutoff_ts`

Forbidden ambiguity for this lane:

1. do not say `current_round`; use `target_round` or `open_round`
2. do not say `context_rounds`; use `prior_context_rounds`
3. do not use target-round `final pools` or post-cutoff klines as model input

## Causal Contract

The model input must respect all of the following:

1. `target_round` is represented only by pre-lock, pre-cutoff information
2. `open_round` has no realized `position`
3. `locked_round` has no realized `position` at decision time
4. outcome-dependent features may use only
   `outcome_eligible_prior_context_rounds`
5. `context_klines` must end at or before the `target_round` cutoff
6. target-round pool features must use `target cutoff pools`, never target
   final pools

The data contract therefore has three distinct causal objects:

1. settled history from `outcome_eligible_prior_context_rounds`
2. one `locked_round` snapshot
3. one `target_round` cutoff snapshot

Those are not interchangeable and should not be flattened into one fake
all-closed sequence without explicit masking or branch separation.

## Label Contract

The label is the realized `position` of `target_round`:

1. `Bull`
2. `Bear`

Training and headline evaluation should exclude:

1. failed rounds
2. refund-like unusable rounds
3. `House` outcome rounds

The first evaluation headline is:

1. out-of-sample win `%`

## Model Ladder

The initial neural-first ranking was:

1. `TCN`
2. `MLP`
3. `GRU` / `LSTM`
4. `Transformer`

Current evidence has reversed that ordering:

1. `MLP`
2. `TCN`
3. `GRU` / `LSTM`
4. `Transformer`

The current serious mainline is the full-feature `MLP`, while `TCN` remains the
main secondary neural comparison.

## Input Structure

The first preferred structure is hybrid rather than pretending every row is the
same round state:

1. a settled-history branch over `outcome_eligible_prior_context_rounds`
2. a `locked_round` branch
3. a `target_round` branch

The first preferred kline handling is:

1. use `context_klines` anchored to the `target_round` cutoff
2. either merge them into the `target_round` branch or feed them as a second
   short sequence branch

## Training-Size Experiments

After full-history rounds and klines are synced, compare at least:

1. `20k`
2. `50k`
3. `100k`
4. `200k`
5. all available history

These are train-window sizes. Validation and test must remain chronological and
recent.

The point of this matrix is to answer:

1. does more history improve recent win `%`
2. does old history dilute current regime signal
3. does a neural model benefit from long-history pretraining more than prior
   tree-based experiments did

## Required Baselines

Every run must report at least:

1. `always_bull`
2. `always_bear`
3. one trivial directional baseline, such as previous settled round side

These are required context for headline win `%`.

## First Execution Plan

1. finish full-history rounds sync
2. sync full-history klines over the same historical span
3. freeze one canonical dataset version
4. build binary labels over valid `target_round` rows
5. run baseline win `%` numbers
6. train `MLP` as the simple neural sanity check
7. train `TCN` as the first real mainline model
8. compare train-size windows chronologically on recent held-out test slices

## Current Status

As of March 31, 2026:

1. full-history round backfill from epoch `1` has been launched in the
   background
2. the neural direction spec is design-only so far
3. no neural model implementation has started yet

Update on April 1, 2026:

1. the first shared dataset builder is now implemented in
   [neural_direction_dataset.py](/C:/Users/zking/Documents/GitHub/PancakeBot/pancakebot/domain/models/neural_direction_dataset.py)
2. the first implementation deliberately reuses the canonical v8 causal feature
   builder as the MLP input contract, instead of creating a second competing
   feature builder before the first baseline/model comparison
3. the headline-only baseline runner is now implemented in
   [run_neural_direction_baselines.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_neural_direction_baselines.py)
4. the first trainable neural sanity path is now implemented as a torch MLP in
   [neural_direction_mlp.py](/C:/Users/zking/Documents/GitHub/PancakeBot/pancakebot/domain/models/neural_direction_mlp.py)
   with the shared eval runner
   [run_neural_direction_mlp_eval.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_neural_direction_mlp_eval.py)
5. the first `TCN` sequence path is now implemented in
   [neural_direction_tcn.py](/C:/Users/zking/Documents/GitHub/PancakeBot/pancakebot/domain/models/neural_direction_tcn.py)
   with the shared eval runner
   [run_neural_direction_tcn_eval.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_neural_direction_tcn_eval.py)
6. the current mainline evidence favors the full-feature MLP over the current
   TCN variants on the recent kline-limited tail
7. the full-history kline sync is now complete, so the larger-window neural
   matrix is no longer kline-gated
8. the first post-sync larger MLP runs now show a non-monotonic history-depth
   effect: `100k` improved the short horizon to about `51.56%` mean held-out
   win `%` on `6480`, while `200k` dropped back to about `50.65%` on that same
   horizon; `10800` stayed around `50.76%` to `50.85%`
9. the flat `400k` MLP point and larger `TCN` reruns are now complete: `MLP @
   400k` reached about `51.12%` on `6480` and about `51.00%` on `10800`, while
   `TCN @ 200k` reached about `51.04%` on `6480` and about `51.06%` on `10800`
10. confidence now matters enough to change setup ranking: the broad best
    finished short-horizon setup remains `MLP @ 100k`, but the strongest
    selective-confidence setup is `MLP @ 400k`, especially in the top `10%`
    through `1%` coverage buckets
11. the useful calibrated confidence thresholds are currently modest, mostly
    around `0.52` to `0.56`, so the next selective-policy work should be based
    on empirical coverage/threshold curves rather than an assumed `0.7+`
    probability cutoff
12. bounded recent-bias training-policy support now exists in the MLP lane:
    the trainer can apply exponential recency weighting inside a train window
    and can warm-start a recent fine-tune run from an older pretraining block
13. those bounded recent-bias reruns did not beat the flat-history anchors on
    the current MLP/v8-feature lane: mild `400k` recency weighting dropped to
    about `50.86%` on `6480` and about `50.83%` on `10800`, `300k -> 100k`
    staged fine-tuning reached only about `50.96%` on `6480`, and the combined
    staged-plus-recency variant fell further to about `50.42%` on `6480`
14. current recommendation: stop spending time on additional recency-weighting
    or pretrain/fine-tune variants of the same MLP/v8 lane unless a new
    feature contract, label contract, or materially different architecture
    justifies reopening the history-combination question
15. the missing confidence-bucket pass for those recent-bias variants is now
    complete, and it agrees with the broad-win result at the decision level:
    the recent-bias variants do not displace the flat-history anchors as the
    current confidence-first mainline. The mild weighted `400k` setup improved
    a few isolated `10800` buckets, but the overall selective ranking still
    stays with flat `MLP @ 400k`
16. thresholded selective-policy evaluation is now implemented in
    [neural_direction_policy.py](/C:/Users/zking/Documents/GitHub/PancakeBot/pancakebot/domain/models/neural_direction_policy.py)
    and
    [run_neural_direction_policy_eval.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_neural_direction_policy_eval.py).
    The current first-pass policy is intentionally simple:
    - direction = calibrated `Bull` if `p(Bull) >= 0.5`, else `Bear`
    - skip if chosen-side calibrated confidence is below a threshold derived
      from the validation slice
    - fixed stake only
    - settle against the true final pools with the existing gas/treasury
      accounting path
17. the first settled `MLP @ 400k` policy grid over target coverages `10%`,
    `5%`, `2%`, and `1%` is mixed rather than broadly promotable. On the
    shorter `6480` horizon, the only clearly positive average lane is the
    tighter target-`2%` band, which translated to about `1.09%` actual mean
    bet rate and reached about `+0.01267 / 500` at `0.05` BNB and about
    `+0.01954 / 500` at `0.10` BNB. On `10800`, every tested band remained
    negative on average after real settlement even when selected-slice win `%`
    stayed above `54%`. This confirms that confidence-selected win `%` is not
    sufficient by itself; final-pool economics can erase the broad-looking edge
18. an important operational nuance from that first policy grid is that
    validation-derived target coverage does not transfer perfectly to test
    coverage. For example, target `2%` on `6480` became about `1.09%` actual
    mean test bet rate, while target `5%` became about `3.14%`. Policy work
    should therefore report both target coverage and realized test bet rate,
    not assume they are identical
19. current policy recommendation: do not promote the direction-only
    confidence-threshold policy yet. The next bounded comparison should be
    flat `MLP @ 100k` versus flat `MLP @ 400k` under the same fixed-stake
    settlement path. If the direction-only threshold policy remains mixed after
    that, the next design branch should be payout-aware or side-conditioned
    policy logic rather than more threshold-only tuning
20. that bounded comparison is now complete, and the settled-policy picture is
    split by horizon. Flat `MLP @ 100k`, in
    [neural_direction_mlp_100k_policy_grid_20260401_neural_direction_policy_summary.json](/C:/Users/zking/Documents/GitHub/PancakeBot_var_exp/neural_direction_mlp_100k_policy_grid_20260401_neural_direction_policy_summary.json),
    is now the strongest finished long-horizon settled-policy lane:
    on `10800`, target `1%` coverage reached about `+0.02237 / 500` at `0.05`
    BNB and about `+0.03972 / 500` at `0.10` BNB, while target `2%` coverage
    also stayed positive. On `6480`, the same MLP lane was weaker and only
    mildly positive in its best aggregate pocket
21. the full settled TCN policy matrix is also complete in
    [neural_direction_tcn_policy_grid_20260401_neural_direction_policy_summary.json](/C:/Users/zking/Documents/GitHub/PancakeBot_var_exp/neural_direction_tcn_policy_grid_20260401_neural_direction_policy_summary.json).
    It produced the strongest short-horizon settled-policy pocket so far:
    `TCN @ 100k`, `seq_len=16`, target `1%` coverage reached about
    `+0.02739 / 500` at `0.05` BNB and about `+0.04673 / 500` at `0.10` BNB on
    `6480`. The more conservative `TCN @ 200k`, `seq_len=16`, target `1%`
    lane was also positive on all three offsets, but weaker. However, the TCN
    family remained negative on average across the tested `10800` aggregates
22. current settled-policy interpretation:
    - `MLP @ 400k` is no longer the sole confidence-first reference once real
      settlement is included
    - short-horizon settled-policy leader: tight `TCN` pockets on `6480`
    - longer-horizon settled-policy leader: flat `MLP @ 100k` on `10800`
    - there is still no single direction-only threshold policy that looks
      broadly robust across both horizons
23. the settled-policy runner now supports both model families and records
    `model_type`, `train_size`, `pretrain_size`, `valid_size`, and `seq_len`
    in the aggregate output so unlike setups are not silently collapsed into
    one row. This matters because the TCN and MLP policy winners differ by
    horizon
24. the next architecture branch under consideration is now documented in
    [RAW_SEQUENCE_TCN_OUTLINE.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/RAW_SEQUENCE_TCN_OUTLINE.md).
    The recommended first version is a hybrid raw-sequence `TCN`: one round
    sequence branch, one `context_klines` sequence branch, and one small
    snapshot branch, all under the same existing causal contract. This branch
    is intended to test whether the current derived-feature `TCN` is losing
    temporal signal by operating on already-compressed target-level features
25. the first bounded raw-sequence branch is now implemented, but the post-fix
    evidence is negative. A real causal leak was found and fixed first:
    `target_round` and `locked_round` were exposing future-only fields in the
    raw round sequence, which had briefly produced impossible `100%` win rates.
    After the fix, the bounded short-horizon rerun in
    [neural_direction_raw_tcn_focus16_20260402_neural_direction_raw_tcn_eval_summary.json](/C:/Users/zking/Documents/GitHub/PancakeBot_var_exp/neural_direction_raw_tcn_focus16_20260402_neural_direction_raw_tcn_eval_summary.json)
    reached only about `50.32%` at `train=100k` and about `49.87%` at
    `train=200k` on `6480`, both worse than the current derived-feature `TCN`
    references
26. the bounded raw-sequence confidence pass in
    [neural_direction_raw_tcn_focus16_conf_20260402_neural_direction_confidence_summary.json](/C:/Users/zking/Documents/GitHub/PancakeBot_var_exp/neural_direction_raw_tcn_focus16_conf_20260402_neural_direction_confidence_summary.json)
    also remained weaker than the derived-feature `TCN`. Current decision:
    keep the raw-sequence branch as recorded historical work, but do not push
    this exact simple fixed-bin raw branch into settled-policy evaluation or
    broader grid expansion
27. the next direction-model branch is no longer "more neural knob tuning."
    The next execution plan is:
    - add broad `CatBoost` direction runners on the same causal feature set
    - add broad `LightGBM` direction runners on the same causal feature set
    - sweep `train_size` for those tree models exactly as was done for the
      neural models (`20k`, `50k`, `75k`, `100k`, `200k`, `400k`)
    - freeze the best broad setup per family
    - compare frozen `CatBoost`, `LightGBM`, `MLP`, and `TCN` on the same
      aligned validation/test windows
    - build a calibrated soft ensemble first
    - then build a simple stacked ensemble over the base-model probabilities
28. when multiple base models are used, all base models must share the same
    `valid` and `test` windows for a given comparison, but each family may keep
    its own best `train_size`. In other words, the right edge of the split is
    aligned across models, while the left edge of the training block may differ
    by model family.
29. the first ensemble implementation should stay simple and one-source at
    runtime:
    - each base model outputs calibrated `p(Bull)` for every valid
      `target_round`
    - a soft ensemble averages those calibrated probabilities
    - a stacked ensemble trains one small combiner on validation-slice base
      predictions
    - runtime still exposes one final `p(Bull)` and one final direction
30. result visualization preference is now explicit: when the experiment
    supports it, prefer round-based or time-based continuous views over
    discrete knob charts. Good default views are:
    - rolling win `%` over rounds
    - rolling confidence-selected win `%` over rounds
    - cumulative net `BNB` over rounds
    - per-model probability traces or agreement/disagreement over rounds
    Discrete knob plots are still acceptable for compact ablations, but they
    are no longer the preferred default presentation.
31. the execution details for the next tree-plus-ensemble branch are now
    frozen in [ENSEMBLE_DIRECTION_EXECUTION_PLAN.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/ENSEMBLE_DIRECTION_EXECUTION_PLAN.md).
    That plan also makes explicit that:
    - checkpoint reports should be written to disk, not chat
    - disk pressure should be handled by pruning or archiving artifacts outside
      the repo rather than altering the research direction
    - the existing feature-cache / reusable intermediate-data path should be
      treated as part of the mainline execution plan, not as optional cleanup
