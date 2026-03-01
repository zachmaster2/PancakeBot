# PancakeBot Handoff: Refactor Iteration Status (2026-03-01)

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
   - Smoke scenario passed:
     - `python -m inspection.run_backtest_scenario --name smoke_refactor_sync --sim-size 200`

## Critical Notes

1. `load_config.py` is strict and intentionally rejects old keys/sections.
2. `config.toml` must stay on new schema; legacy keys will now fail startup.
## Recommended Next Steps

1. Commit current rename/terminology cleanup chunk (small rollback unit).
2. Sweep for dead feature fields/constants no longer used in dislocation-only
   path and prune.
3. Run larger backtest parity matrix against previous known scenarios and log
   drift summary.
