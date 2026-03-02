# PancakeBot Handoff: Refactor Iteration Status (2026-03-01)

## Update (2026-03-02): Shared Pipeline + Online Router + ML Candidate Adapter

1. Shared strategy pipeline module added:
   - `pancakebot/domain/strategy/pipeline.py`
   - Combines dislocation candidates + optional ML candidate + shared router.
   - Used by both backtest and runtime (single execution pipeline path).

2. Router upgraded for online adaptation:
   - `pancakebot/domain/strategy/router.py`
   - Added `online_cellmean` mode with:
     - warmup rounds,
     - per-candidate quantile celling on expected profit and absolute dislocation,
     - optional bull/bear direction split,
     - per-cell realized-profit mean routing.
   - Added `observe_settlement(...)` state updates.

3. ML candidate adapter restored into active tree:
   - `pancakebot/domain/strategy/ml_candidate_adapter.py`
   - Reuses current feature builder + walk-forward stack to emit router-compatible
     `StrategyCandidateSignal`.
   - Produces `SKIP` with explicit reasons until readiness gates are satisfied.

4. Strategy config surface extended (shared across live/dry/backtest):
   - `pancakebot/config/strategy_config.py`
   - Added:
     - `StrategyRouterConfig`
     - `MlCandidateConfig`
   - `StrategyConfig` now owns:
     - `dislocation`
     - `router`
     - `ml_candidate`

5. Config parser updated for new strategy namespaces:
   - `pancakebot/config/load_config.py`
   - Added strict parsers:
     - `strategy.router`
     - `strategy.ml_candidate`
   - Removed router knobs from `backtest` section (router is now strategy-owned).

6. Backtest runner switched to shared pipeline:
   - `pancakebot/backtest/runner.py`
   - No direct dislocation decision path in backtest loop anymore.
   - Summary now records router/ML flags from `strategy_cfg`.

7. Runtime loop switched to shared pipeline:
   - `pancakebot/runtime/runtime_loop.py`
   - Replaced direct dislocation-engine decisions with:
     - `StrategyPipeline.decide_open_round(...)`
     - `StrategyPipeline.settle_closed_rounds(...)`
   - Keeps live/dry behavior unchanged except for routing source unification.

8. Inspection backtest probe aligned:
   - `inspection/run_backtest_scenario.py`
   - Router CLI overrides now target `strategy_cfg.router`.
   - Supports `--router-mode online_cellmean`.

9. Config TOML updated:
   - `config.toml`
   - Added:
     - `[strategy.router]`
     - `[strategy.ml_candidate]`
   - `backtest` now only owns simulation/reset knobs.

10. Tests:
   - Updated `tests/test_strategy_router.py` with `online_cellmean` coverage.
   - Added `tests/test_ml_candidate_adapter.py` (disabled-path deterministic test).

11. Validation (using `.\.venv\Scripts\python.exe`):
   - `python -m compileall` on touched files passed.
   - `python -m unittest tests.test_strategy_router tests.test_ml_candidate_adapter -v` passed.
   - Smoke scenarios passed:
     - `python -m inspection.run_backtest_scenario --name smoke_shared_pipeline --sim-size 300 --reset-mode continuous`
     - `python -m inspection.run_backtest_scenario --name smoke_online_cellmean --sim-size 300 --reset-mode continuous --router-mode online_cellmean`

## Update (2026-03-02): Shared Router Groundwork (Backtest-Only Integration)

1. Added shared router module:
   - `pancakebot/domain/strategy/router.py`
   - Router modes:
     - `selector_max_score`
     - `skip_only`
     - `oracle_skip`
   - Normalized output contract:
     - `StrategyRouterDecision` (`BET`/`SKIP`, selected strategy, side, size, expected profit, selector score, skip reason, `p_bull`).

2. Backtest now routes from candidate signals through shared router:
   - `pancakebot/backtest/runner.py`
   - Replaced direct `engine.decide_open_round(...)` calls in backtest simulation loop.
   - Backtest trades now include router telemetry columns:
     - `selected_strategy`
     - `router_mode`
     - `selector_score_bnb`
   - Backtest summary now includes:
     - `router_mode`
     - `router_score_threshold_bnb`

3. Dislocation engine support for router path:
   - `pancakebot/domain/strategy/dislocation_engine.py`
   - `candidate_signals_for_open_round(...)` now stores pending decisions by epoch so settle path remains aligned.
   - Added `selector_ready()` accessor for warmup/no-candidate reason parity in router mode.

4. Backtest config knobs added and parsed:
   - `pancakebot/backtest/config.py`
   - `pancakebot/config/load_config.py`
   - `config.toml`
   - New `[backtest]` keys:
     - `router_mode = "selector_max_score"`
     - `router_score_threshold_bnb = -1000000000.0`

5. Inspection scenario runner kept in sync:
   - `inspection/run_backtest_scenario.py`
   - Added optional CLI flags:
     - `--router-mode`
     - `--router-score-threshold-bnb`
   - Scenario metadata now persists router settings.

6. Deterministic tests added:
   - `tests/test_strategy_router.py`
   - Covered:
     - `skip_only` always skips.
     - `oracle_skip` selects highest positive realized-profit candidate.
     - selector threshold gate skips below configured threshold.

7. Validation run (using `.\.venv\Scripts\python.exe`):
   - `compileall` on touched modules.
   - `unittest tests.test_strategy_router -v` passed.
   - backtest smoke scenarios passed:
     - `smoke_router_skip_go` (`--router-mode skip_only`, sim=5)
     - `smoke_router_selector_go` (`--router-mode selector_max_score`, sim=5)
     - `smoke_router_oracle_go` (`--router-mode oracle_skip`, sim=5)

## Iteration Goal
Unify production around a single dislocation strategy pipeline, eliminate legacy config/runtime clutter, and keep legacy modules only under `inspection/legacy` for one transition cycle.

## Completed In This Iteration Chunk

1. Strategy engine is now config-driven:
   - Removed hardcoded promoted candidate family from
     `pancakebot/domain/strategy/dislocation_engine.py`.
   - Added `build_dislocation_engine_from_config(...)`.
   - Runtime/backtest now pass `strategy.dislocation.selector` and
     `strategy.dislocation.candidates` directly.

2. Runtime/app config surface was reduced to active fields only:
   - Updated:
     - `pancakebot/config/app_config.py`
     - `pancakebot/config/load_config.py`
     - `pancakebot/runtime/runtime_loop.py`
     - `pancakebot/integration/app.py`
     - `inspection/run_backtest_scenario.py`
   - Removed legacy model/predictability/policy/train/calibrate knobs from
     active config and runtime structs.

3. Config schema was rewritten (no backward compatibility):
   - `config.toml` now uses:
     - `[runtime]` lean fields only
     - `[strategy.dislocation.selector]`
     - `[[strategy.dislocation.candidates]]` (explicit candidate tables)
     - `[backtest]` with reset controls

4. Legacy production modules were archived and removed from active tree:
   - Moved to `inspection/legacy/pancakebot/...`:
     - `config/policy_config.py`
     - `domain/models/{final_pool_model,price_return_model,predictability_model,walk_forward}.py`
     - `domain/models/{artifacts,calibration,dataset_builder,__init__}.py`
     - `domain/strategy/{planner,policy,ev_math,sizing}.py`
     - `runtime/{model_manager,cache_policy}.py`
   - Updated package docs:
     - `pancakebot/domain/strategy/__init__.py`
     - `inspection/legacy/README.md`
     - added `inspection/legacy/pancakebot/README.md`

5. Validation:
   - `python -m compileall pancakebot inspection/run_backtest_scenario.py` passed.
   - Smoke scenarios passed:
     - `python -m inspection.run_backtest_scenario --name smoke_refactor_sync --sim-size 200`
     - `python -m inspection.run_backtest_scenario --name smoke_refactor_trim_fields --sim-size 120`
     - `python -m inspection.run_backtest_scenario --name smoke_refactor_trim_fields_chunk --sim-size 120 --reset-mode chunk_reset --reset-every-rounds 40`

6. Additional cleanup completed:
   - Removed `event_freshness_slack_seconds` from active runtime/config path.
   - Removed `min_bet_amount_bnb` from `RuntimeConfig`.
   - Removed redundant `save_contract_constants(...)` call from runtime-loop
     startup path.
   - Updated dislocation terminology comments/docstrings.
   - Removed in-code candidate defaults; candidates are now required in
     `config.toml` under `[[strategy.dislocation.candidates]]`.

## Critical Notes

1. `load_config.py` is strict and intentionally rejects old keys/sections.
2. `config.toml` must stay on new schema; legacy keys will now fail startup.
3. `strategy.dislocation.candidates` is required; no code fallback exists.
## Recommended Next Steps

1. Commit current runtime-field trim chunk (small rollback unit).
2. Sweep remaining active files (`domain/features/*`, `runtime/*`) for stale
   nomenclature/comments inherited from legacy model pipeline.
3. Run larger backtest parity matrix against previous known scenarios and log
   drift summary.
