# Ensemble Direction Execution Plan

## Purpose

This plan replaces the recent knob-plot drift with one concrete next branch:
add tree-based broad direction models, compare them fairly against the current
neural baselines, and then build a simple one-source ensemble.

The branch must remain:

- causal
- one-source at runtime
- operator-readable
- disk-aware without letting disk pressure distort research decisions

## Runtime Goal

The final runtime shape remains:

- multiple internal base models are allowed
- one final calibrated `p(Bull)` is emitted
- one final `Bull` or `Bear` direction is emitted

Do not drift back into controller/router/profile ownership.

## Base Model Ladder

The first ensemble ladder is:

1. `CatBoost`
2. `LightGBM`
3. current broad `MLP`
4. current best broad `TCN`

The comparison order is:

1. best single model per family
2. calibrated soft ensemble
3. stacked ensemble

## Split Contract

For any given comparison window:

- all base models must share the same `valid` window
- all base models must share the same `test` window
- each family may keep its own best `train_size`

So the right edge is aligned across models, while the left edge of the train
block may differ by model family.

`TCN` still requires sequence warmup context (`seq_len - 1`), but that warmup
is context only, not extra labeled train data.

## Broad First

Tree models are not confidence-only helpers. They must first be evaluated as
broad forced-pick direction models exactly like the neural models.

The initial broad sweep for both `CatBoost` and `LightGBM` is:

- `train_size`: `20k`, `50k`, `75k`, `100k`, `200k`, `400k`
- fixed `valid_size`
- fixed `sim_size` windows
- fixed `tail_offset_rounds`
- same causal feature set and label contract
- headline metric: held-out win `%`

Only after freezing the best broad setup per family should confidence and
ensemble work continue.

## Probability Export Contract

For each valid `target_round` in the shared `valid` and `test` windows, the
comparison layer should persist:

- actual outcome
- epoch / round metadata
- each base model's raw `p(Bull)`
- each base model's calibrated `p(Bull)`
- each base model's chosen side
- each base model's chosen-side confidence
- simple agreement / disagreement fields
- soft-ensemble probability
- stacked-ensemble probability

## Ensemble Order

### 1. Calibrated Soft Ensemble

Use validation-only calibration for each base model, then combine with a simple
average of calibrated probabilities.

This is the first ensemble benchmark and should be kept trivial.

### 2. Stacked Ensemble

Train one small combiner on validation-slice base-model predictions.

Preferred first combiner:

- logistic regression

Possible optional meta-features:

- per-model calibrated probabilities
- per-model chosen-side confidence
- probability mean
- probability spread / disagreement

The stacker is then evaluated on the untouched test slice.

## Reporting Contract

Checkpoint reports should be written to disk, not emitted in chat during the
run.

Primary report path for this branch:

- `../PancakeBot_var_exp/ensemble_direction_execution_report_20260402.md`

The report should be appended at sensible checkpoints:

1. tree broad sweeps complete
2. frozen best-per-family comparison complete
3. soft ensemble complete
4. stacked ensemble complete
5. settled-policy comparison complete

At the end of the full branch, the final chat response should point to the
report file and summarize it briefly with links to the main plots.

## Plotting Preference

Prefer round-based or time-based continuous views whenever the experiment
supports them.

Preferred plot families:

- rolling win `%` over rounds
- rolling top-confidence win `%` over rounds
- cumulative settled `BNB` over rounds
- per-model calibrated probability traces over rounds
- per-model agreement / disagreement over rounds
- ensemble probability minus base-model probability over rounds

Discrete knob plots are acceptable for bounded ablations, but they are not the
default presentation for this branch.

## Feature Reuse And Caching

Use the existing feature-cache path as a first-class part of execution.

Rules:

- reuse the existing cached causal feature rows where possible
- do not recompute the same neural-direction feature set unnecessarily
- if a new reusable derived table is created for tree / ensemble work, persist
  it outside the repo under `../PancakeBot_var_exp/`
- keep cached or exported intermediate data versioned and labeled clearly

The goal is to avoid repeating expensive feature generation or repeatedly
rebuilding the same aligned probability tables.

## Disk Management

Disk pressure must be managed proactively, but it must not change the research
direction or force weaker decisions.

Allowed actions:

- delete obsolete experiment outputs outside the repo
- archive older low-value outputs outside the repo
- prune large caches such as `../PancakeBot_var_exp/backtest_state_cache`
- keep only the artifacts needed for reproducibility and comparison

The repo itself must stay lean. Scratch outputs remain outside the repo.

## Implementation Order

1. implement `CatBoost` broad runner
2. implement `LightGBM` broad runner
3. run broad `train_size` sweeps for both tree families
4. freeze the best broad setup per family
5. build aligned per-round probability export
6. run calibrated soft ensemble
7. run stacked ensemble
8. run continuous result plots
9. run settled-policy comparisons for best single models and ensembles
10. finalize report and recommendation

## Decision Standard

The end of the branch should answer:

1. best broad single direction model
2. best confidence-ranked single direction model
3. whether the soft ensemble beats the best single model
4. whether the stacked ensemble beats the soft ensemble
5. whether agreement/disagreement is a useful confidence signal
6. whether the resulting direction layer is strong enough to justify the next
   policy step
