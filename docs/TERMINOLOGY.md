# PancakeBot Terminology

Canonical vocabulary for code, config, docs, and logs.

## Pipeline Terms

1. `production pipeline`: the single shared strategy execution path used by
   live, dry, and backtest modes.
2. `probe pipeline`: inspection tooling that orchestrates experiments by
   invoking the production pipeline and analyzing artifacts.
3. `legacy probe`: any archived historical inspection tool moved out of the
   active repo archive.

## Mode Terms

1. `live mode`: real on-chain execution.
2. `dry mode`: no on-chain broadcast; simulated execution bookkeeping.
3. `backtest mode`: deterministic replay over stored rounds and klines.

## Strategy Terms

1. `strategy`: dislocation cell-mean strategy (single production strategy).
2. `selector`: candidate selection layer over candidate strategy decisions.
3. `candidate`: one parameterized dislocation strategy profile.
4. `reset mode`: backtest-only model-state reset behavior.
5. `reset interval`: number of simulated rounds between chunk resets.

## Data Terms

1. `warmup rounds`: closed rounds used to initialize strategy state.
2. `simulation rounds`: rounds executed for evaluated backtest performance.
3. `closed rounds store`: durable epoch-ascending round history.
4. `klines store`: durable minute-level kline history.

## Round Context Terms

1. `target_round`: the round being predicted or decided.
2. `open_round`: the live causal state of `target_round` before lock.
3. `locked_round`: the immediately prior round, used as
   `prior_context_rounds[-1]`.
4. `prior_context_rounds`: the ordered round context immediately preceding
   `target_round`; by contract, the last element is `locked_round`.
5. `prior_context_prefix`: the settled prefix of `prior_context_rounds`,
   excluding `locked_round`.
6. `outcome_eligible_prior_context_rounds`: shorthand for
   `prior_context_rounds[:-1]`; these are the only prior rounds allowed to
   contribute outcome-dependent features.
7. `context_klines`: kline context anchored to the `target_round` cutoff; the
   last kline must close at or before the `target_round` cutoff.

## Causal Snapshot Terms

1. `cutoff_ts`: `target_round.lock_at - cutoff_seconds`.
2. `target cutoff pools`: target-round pool amounts computed using only bets
   with `created_at <= cutoff_ts`.
3. `target final pools`: post-settlement target-round pool amounts; forbidden
   as model input for prediction.
4. `target-only features`: features derived only from `target_round` at cutoff.
5. `lagged late-phase features`: features derived from `locked_round` using
   bets in `(cutoff_ts, lock_ts]` for that locked round.
6. `outcome-dependent features`: features that use realized round outcomes and
   therefore may only use `outcome_eligible_prior_context_rounds`.

## Artifact Terms

1. `trades artifact`: per-round backtest/probe execution output CSV.
2. `summary artifact`: aggregate result JSON output.
3. `experiment directory`: `../PancakeBot_var_exp/<scenario_name>/`.

## Forbidden Ambiguity

1. Do not use `model` to describe runtime strategy state unless explicitly
   referring to a statistical model class.
2. Do not use `pipeline` for one-off scripts that bypass production execution
   logic.
3. Do not overload `strategy` to mean both a candidate and the global strategy.
4. Do not use `current_round` when the intended meaning is `target_round` or
   `open_round`.
5. Do not use `context_rounds` when the intended meaning is
   `prior_context_rounds`.
6. Do not use target-round `final pools` when the intended meaning is target
   cutoff-time information.
