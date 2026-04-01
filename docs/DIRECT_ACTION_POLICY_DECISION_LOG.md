# Direct Action Policy Decision Log

## Status

This document records the main design decisions for the first direct-action
policy implementation, along with the rejected alternatives and the reason each
choice was made.

Current implementation status:

1. decisions `1` through `18` are now reflected in code or durable evaluation
2. the current direct realized-net quantile lane is still unqualified for
   promotion

## Decision 1: Runtime Architecture

Chosen:

1. add one optional direct-action policy path to the shared strategy pipeline
2. when enabled, it bypasses candidate generation, router selection, and window
   controller masking for the final decision

Alternatives considered:

1. retrofit the existing router to score direct actions
2. build the direct-action lane as a detached inspection-only path
3. stack the new model inside the old controller/router path

Why this choice:

1. it preserves one clean final decision owner
2. it avoids reintroducing the layered skip logic we are trying to remove
3. it still fits the existing backtest/dry/runtime integration points

## Decision 2: Activation Strategy

Chosen:

1. keep the direct-action path optional and disabled by default until it is
   qualified offline and in dry mode

Alternatives considered:

1. replace the current runtime default immediately
2. delete the legacy path before the new one is validated

Why this choice:

1. the redesign is an intentional replacement direction, but the new lane is
   not qualified yet
2. keeping it optional allows bounded comparison without pretending rollout is
   complete

## Decision 3: Action Space

Chosen:

1. `Skip`
2. `Bull` and `Bear` at sizes `0.05`, `0.10`, `0.15`, `0.25`, `0.35`, `0.50`
   `BNB`

Alternatives considered:

1. one fixed bet size
2. continuous bet sizing
3. two-stage decision making: side first, size second

Why this choice:

1. it keeps the runtime operator-readable
2. it makes size part of the actual action instead of a hidden second system
3. it avoids the complexity of continuous sizing in the first clean version

## Decision 4: Feature Scope

Chosen:

1. canonical current-round feature builder output (`v8`)
2. explicit action identity features
3. rolling realized exact-action summaries over `24`, `72`, and `216` rounds
4. bounded legacy dislocation candidate outputs as auxiliary features only

Alternatives considered:

1. use only current-round features with no rolling action summaries
2. exclude all legacy candidate/profile outputs from the first version
3. add a much broader set of horizons and handcrafted summary families

Why this choice:

1. it uses already-approved causal feature infrastructure
2. it keeps the first version small
3. it satisfies the redesign requirement that rolling realized windows remain
   features, not the decision mechanism
4. the first held-out smoke run showed that the generic-feature-only model had
   essentially no usable signal, so bounded legacy candidate outputs were
   reintroduced as features only

## Decision 5: Legacy Profile Features

Chosen:

1. include only bounded legacy dislocation candidate outputs as auxiliary
   features
2. do not reintroduce legacy profiles as runtime actions or privileged policy
   owners

Alternatives considered:

1. exclude all legacy outputs entirely
2. reintroduce `stageB`, `stageG2`, and `altB` as the actual runtime action
   space

Why this choice:

1. the redesign is explicitly trying to escape profile-centric runtime logic
2. the generic-feature-only smoke run was too weak to justify keeping the
   feature contract that sparse
3. auxiliary feature use preserves the single-policy runtime contract while
   still letting the model borrow signal from existing candidate logic

## Decision 6: Rolling Summary Statistics

Chosen:

For each non-skip action and each horizon:

1. mean realized net `BNB`
2. positive-rate
3. realized-net standard deviation

Alternatives considered:

1. oracle-style opportunity summaries
2. broader handcrafted action-regret summaries
3. no summary statistics beyond mean

Why this choice:

1. these three are simple, stable, and directly interpretable
2. they capture value, hit rate, and dispersion without overengineering

## Decision 7: Learning Formulation

Chosen:

1. one shared action-row dataset
2. one unified runtime policy over peer actions
3. per-action quantile heads over the shared feature contract

Alternatives considered:

1. one pooled quantile model over all `(round, action)` rows
2. side classification plus a separate size model
3. profile selection as an intermediate target

Why this choice:

1. all actions remain peers in the runtime policy contract
2. the pooled lower-quantile model collapsed toward a global negative lower
   tail and failed to calibrate `skip` or small actions distinctly
3. per-action heads are still simple enough for the first lane while avoiding
   the pooled `q10` pathology

## Decision 8: Model Family

Chosen:

1. LightGBM quantile regressors
2. one `q10` model per non-skip action
3. one `q50` model per non-skip action
4. `skip` uses a constant zero quantile model

Alternatives considered:

1. one pooled `q10` and one pooled `q50` model across all actions
2. `HistGradientBoostingRegressor` quantile models
3. mean-plus-variance regression
4. ensemble disagreement as the primary uncertainty method

Why this choice:

1. LightGBM is already used in the repo
2. quantile regression matches the desired confidence semantics directly
3. the pooled quantile bundle was not action-aware enough on held-out data,
   especially in the lower tail
4. the per-action bundle is still operationally simple and materially more
   coherent than the pooled version

## Decision 9: Runtime Score

Chosen:

1. runtime score is `q50_net_bnb - lambda * (q50_net_bnb - q10_net_bnb)`
2. first default `lambda = 0.15`

Alternatives considered:

1. raw `q10_net_bnb`
2. predicted mean only
3. a broader configurable family of score rules in V1

Why this choice:

1. raw `q10` proved too conservative in the first held-out smoke run and
   produced an all-skip degenerate policy
2. the `q50-minus-lambda-spread` rule preserves explicit uncertainty while
   keeping the score simple and operator-readable
3. it avoids overconfiguring the first version

## Decision 10: Training Window Defaults

Chosen:

1. train size: `15,000` target rounds
2. validation size: `3,000` target rounds
3. retrain interval: `1,000` rounds

Alternatives considered:

1. much shorter windows
2. much longer windows
3. retraining every round

Why this choice:

1. it keeps the recent-history emphasis
2. it is in the same practical regime as existing repo ML tooling
3. the action-row expansion already provides ample training rows within those
   windows

## Decision 11: Sample Weighting

Chosen:

1. apply mild recency weighting from `0.5` to `1.0` across the training window

Alternatives considered:

1. no recency weighting
2. much steeper weighting

Why this choice:

1. regime drift matters
2. mild weighting keeps the model recent-aware without becoming too brittle

## Decision 12: Label Contract

Chosen:

1. label is realized net `BNB` for the exact action
2. settlement semantics reuse the canonical runtime/backtest settlement logic
3. `Skip = 0.0`

Alternatives considered:

1. direction-only labels
2. ROI labels
3. oracle-best-action labels

Why this choice:

1. it aligns directly with the project objective
2. it keeps offline and runtime semantics coherent

## Decision 13: Model Bundle Format

Chosen:

1. store the trained direct-action bundle as a gzip-compressed pickle under the
   experiment tree

Alternatives considered:

1. joblib bundle
2. LightGBM text dumps plus manual metadata files

Why this choice:

1. it is simple
2. it can store metadata, feature names, action schema, and both models in one
   file
3. the bundle is local and controlled, not an interchange format

## Decision 14: Runtime State

Chosen:

1. direct-action runtime is inference-only
2. it maintains only the bounded closed-round history required for features

Alternatives considered:

1. online learning in runtime
2. unbounded runtime history accumulation

Why this choice:

1. it keeps the first version coherent
2. it reduces rollout risk
3. it matches the redesign goal of offline-first qualification

## Decision 15: Logging And Audit

Chosen:

1. add dedicated direct-action decision fields and logs
2. do not overload controller fields for the new path

Alternatives considered:

1. squeeze direct-action decisions into existing controller audit columns
2. rely only on summary logs

Why this choice:

1. the operator must be able to see the real decision source clearly
2. mixing the new path into controller fields would hide the redesign boundary

## Decision 16: Top-Level Policy Ownership

Chosen:

1. reject any effective config that enables both direct-action policy and window
   controller at the same time

Alternatives considered:

1. silently give direct-action priority
2. silently give window-controller priority
3. allow both and hope the pipeline behavior is understood

Why this choice:

1. the redesign requires one clear decision owner
2. silent precedence would recreate the ambiguity we are explicitly removing

## Decision 17: First Offline Evaluation Harness

Chosen:

1. train the frozen direct-action bundle on a latest-tail train/validation
   slice
2. evaluate the held-out test window through the shared backtest path using the
   same inference code the runtime uses

Alternatives considered:

1. evaluate only with a detached offline argmax loop
2. delay any tooling until runtime integration was complete

Why this choice:

1. it keeps offline qualification and backtest/runtime semantics aligned
2. it reuses the shared scenario/backtest harness rather than inventing another
   execution path

## Decision 18: Qualification Gate After The First Smoke Runs

Chosen:

1. do not run the full multi-size or multi-offset sweep for the current
   realized-net quantile lane
2. treat the lane as blocked pending a model-contract revision

Alternatives considered:

1. continue threshold tuning on the existing score
2. fan out the full shared-eval sweep anyway
3. promote a dry run despite the poor held-out smoke behavior

Why this choice:

1. the pooled bundle first degenerated to all-skip, then the per-action bundle
   overbet and lost about `-3.8579 / 500` on the same held-out `6480` slice
2. simple score-threshold tightening did not rescue the per-action bundle; all
   tested thresholds remained negative
3. the next problem is model signal/target design, not a missing evaluation
   sweep
