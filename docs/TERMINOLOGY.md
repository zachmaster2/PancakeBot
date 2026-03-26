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
