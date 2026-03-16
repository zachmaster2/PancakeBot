# PancakeBot Handoff: Refactor Iteration Status (2026-03-02)

## Update (2026-03-02): Router Matrix Sweep + Preset Promotion (Uncommitted)

1. Router sweep runner added:
   - `inspection/run_backtest_router_matrix.py`
   - Runs production backtest path per router variant (`prime -> warm`), writes
     sorted matrix with:
     - mode + knob values,
     - net / per-500 / drawdown / bets,
     - top skip reasons,
     - selected-strategy mix,
     - warm runtime.

2. Shared harness extension:
   - `inspection/backtest_harness_common.py`
   - `run_backtest_case(...)` now accepts `strategy_cfg` override for matrix
     sweeps without mutating base config.

3. Initial router matrix executed (continuous, sim=500):
   - Artifacts:
     - `../PancakeBot_var_exp/router_matrix_20260302_table.json`
     - `../PancakeBot_var_exp/router_matrix_20260302_table.csv`
   - Best variants (tie):
     - `mode=online_cellmean`
     - `online_use_direction_split=false`
     - `online_score_threshold_bnb in {0.0, 0.001}`
   - Metrics (both tied):
     - `net_profit_bnb = 1.2447962481485177`
     - `profit_per_500_rounds_bnb = 1.244796248148518`
     - `max_drawdown_bnb = 1.5540000000000092`
     - `num_bets = 21`
     - `top_skip_reasons = router_online_no_candidate:479`
     - `selected_strategy_mix = disloc_stageG2_r37_x80:18; disloc_cons_20260227_x80:2; disloc_best_20260227_x80:1`
     - warm runtime: `~8.36s`

4. Router preset promotion applied:
   - Active config updated in `config.toml` `[strategy.router]`:
     - `mode = "online_cellmean"`
     - `online_score_threshold_bnb = 0.001`
     - `online_use_direction_split = false`
   - Preset record added:
     - `inspection/presets/router_online_cellmean_nosplit_v1.json`

5. Replay validation with promoted config:
   - Command:
     - `python -m inspection.run_backtest_scenario --name smoke_router_promoted_20260302 --sim-size 500 --reset-mode continuous`
   - Result:
     - `net_profit_bnb = 1.2447962481485177`
     - `num_bets = 21`
     - `bet_rate = 0.042`

## Update (2026-03-02): External Cache Paths + Harness + Warm Matrix + Sub-10s Warm Runs (Uncommitted)

1. Config/runtime cache-path refactor completed:
   - Added explicit `paths.backtest_state_cache_dir` to config model and parser.
   - Updated active defaults to external artifact root:
     - `feature_cache_path = ../PancakeBot_var_exp/feature_cache_v8.sqlite`
     - `backtest_state_cache_dir = ../PancakeBot_var_exp/backtest_state_cache`
   - Backtest runner now uses runtime-config cache path, not hardcoded `var/backtest_state_cache`.

2. One-command cache performance harness added:
   - `inspection/run_backtest_cache_perf.py`
   - Shared helper: `inspection/backtest_harness_common.py`
   - Runs `cold -> warm` for `continuous` and `chunk_reset`; reports timing deltas + miss/hit confirmation.

3. Warm-cache matrix runner added:
   - `inspection/run_backtest_warm_matrix.py`
   - Outputs/exports consolidated table:
     - mode
     - reset interval
     - net profit BNB
     - profit per 500 rounds
     - max drawdown BNB
     - num bets
     - top skip reasons

4. Post-matrix speed pass completed (determinism preserved):
   - Added cached closed-round tail loading (`phase=round_tail`).
   - Added cached dislocation kline-index loading (`phase=kline_index`).
   - Reused one chunk pipeline across chunk-reset cache hits (rebuild only on miss).
   - Added kline-index export/import hooks:
     - `pancakebot/domain/strategy/dislocation_engine.py`
     - `pancakebot/domain/strategy/pipeline.py`

5. Current benchmark artifacts:
   - Harness pre-speed-pass:
     - `../PancakeBot_var_exp/cache_perf_20260302_summary.json`
   - Harness post-speed-pass:
     - `../PancakeBot_var_exp/cache_perf_20260302_speedpass2_summary.json`
   - Warm matrix pre-speed-pass:
     - `../PancakeBot_var_exp/warm_matrix_20260302_table.json/csv`
   - Warm matrix post-speed-pass:
     - `../PancakeBot_var_exp/warm_matrix_20260302_speedpass_table.json/csv`

6. Key timing results (sim=500, chunk reset=20):
   - Pre-speed-pass harness:
     - continuous warm: `13.5507s`
     - chunk_reset warm: `20.2126s`
   - Post-speed-pass harness:
     - continuous warm: `8.3534s`
     - chunk_reset warm: `8.9812s`
   - Sub-10s warm target achieved for both modes.

## Update (2026-03-02): Backtest Warmup Snapshot Caching (Committed)

1. Commit completed:
   - `dc1b842`
   - Message: `Backtest: cache warmup bootstrap state for second-pass speed`

2. Scope:
   - Backtest-only acceleration changes.
   - Live/dry behavior unchanged.

3. Key implementation:
   - Added backtest state cache module:
     - `pancakebot/backtest/state_cache.py`
   - Added bootstrap state export/import hooks:
     - `pancakebot/domain/strategy/pipeline.py`
     - `pancakebot/domain/strategy/dislocation_engine.py`
     - `pancakebot/domain/strategy/router.py`
     - `pancakebot/domain/strategy/ml_candidate_adapter.py`
   - Wired cache load/save in backtest runner for:
     - `continuous`
     - `chunk_reset`
     - file: `pancakebot/backtest/runner.py`
   - Added batched feature-cache commits and explicit flush/close:
     - `pancakebot/infra/feature_cache_store.py`

4. Timing proof with full ML + all candidates + router:
   - Continuous cold (`sim=10`): `432.292s` (cache miss)
   - Continuous warm (`sim=10`): `27.854s` (cache hit)
   - Chunk-reset cold (`sim=20`, `reset=20`): `489.763s` (cache miss)
   - Chunk-reset warm (`sim=20`, `reset=20`): `27.472s` (cache hit)
   - Continuous warm (`sim=500`): `33.410s` (cache hit)
   - Warm runs now in seconds (tens of seconds), versus minutes on cold start.

5. Artifact state observed:
   - Backtest state snapshots written under:
     - `var/backtest_state_cache/pipeline_bootstrap/*.pkl.gz`
   - Feature cache used:
     - `var/feature_cache_v8.sqlite`
     - table: `feature_vectors` (observed count: `50380`)

6. Validation:
   - `.\.venv\Scripts\python.exe -m compileall ...` on touched modules: passed.
   - `.\.venv\Scripts\python.exe -m unittest tests.test_strategy_router tests.test_ml_candidate_adapter -v`: passed.

## User-Approved Next Tasks (Execute Automatically In Next Chat)

1. Move backtest state-cache path out of repo and make it explicit in config.
   - Keep experiment outputs under `../PancakeBot_var_exp`.
   - Align state-cache path with that external location (not under project `var/`).

2. Add a one-command performance harness:
   - Runs `cold -> warm` for both `continuous` and `chunk_reset`.
   - Prints concise timing deltas and cache hit/miss confirmation.

3. Run full matrix with warm cache and output one consolidated table:
   - mode
   - reset interval
   - net profit BNB
   - profit per 500 rounds
   - max drawdown BNB
   - num bets
   - top skip reasons

4. Post-matrix speed pass targeting sub-10s warm runs:
   - cache ML retrain/calibration checkpoint state across simulation intervals.
   - preserve deterministic behavior.

## Anchors / Non-Negotiables

1. Always use `.\.venv\Scripts\python.exe` for runs.
2. Keep `good-results-codebase` off-limits.
3. Do not revert unrelated dirty files.
4. Continue frequent small commits for rollback safety.

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

## Update (2026-03-03): Recovered-Thread Router Tuning Continuation (Uncommitted)

1. Recovered previous chat intent from screenshot:
   - Proceed with focused `online_cellmean` sparsity sweep on:
     - `online_num_quantile_bins`
     - `online_min_cell_obs`
     - `online_score_threshold_bnb`
   - If weak, pivot to next best architecture-level idea.

2. Focused sparsity sweep executed (`sim=500`, `continuous`, `dir_split=false`):
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.run_backtest_router_matrix --name-prefix router_sparsity_20260303 --sim-size 500 --reset-mode continuous --router-modes online_cellmean --online-use-direction-split-list false --online-num-quantile-bins 8,12,16 --online-min-cell-obs 3,5,8 --online-score-thresholds 0.0,0.001`
   - Artifacts:
     - `../PancakeBot_var_exp/router_sparsity_20260303_table.json`
     - `../PancakeBot_var_exp/router_sparsity_20260303_table.csv`
   - Top setting:
     - `bins=12`, `min_obs=8`, `online_thr in {0.0, 0.001}`
     - `net_profit_bnb = 1.6457962481` (per 500 = `1.6457962481`)
     - `num_bets = 20`
     - `top_skip_reasons = router_online_no_candidate:480`

3. 2000-round robustness check against active promoted config:
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.run_backtest_router_matrix --name-prefix router_sparsity_confirm_20260303 --sim-size 2000 --reset-mode continuous --router-modes online_cellmean --online-use-direction-split-list false --online-num-quantile-bins 12 --online-min-cell-obs 5,8 --online-score-thresholds 0.001`
   - Artifact:
     - `../PancakeBot_var_exp/router_sparsity_confirm_20260303_table.json/csv`
   - Result:
     - `bins=12,min_obs=8,thr=0.001`: `0.197137` per 500.
     - active promoted (`bins=12,min_obs=5,thr=0.001`): `-0.074935` per 500.

4. Expanded 2000-round sweep around sparse region:
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.run_backtest_router_matrix --name-prefix router_sparsity_confirm2_20260303 --sim-size 2000 --reset-mode continuous --router-modes online_cellmean --online-use-direction-split-list false --online-num-quantile-bins 10,12,14 --online-min-cell-obs 8,10,12 --online-score-thresholds 0.0,0.001,0.003`
   - Artifact:
     - `../PancakeBot_var_exp/router_sparsity_confirm2_20260303_table.json/csv`
   - Best in this sweep:
     - `bins=10,min_obs=10,thr=0.003`
     - `net_profit_bnb = 2.461380...` over 2000 (`0.615345` per 500).
   - Observation:
     - 500-round spikes did not hold proportionally on 2000 rounds (high horizon drift).

5. 500-round cross-check for candidate neighborhood:
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.run_backtest_router_matrix --name-prefix router_sparsity_crosscheck_20260303 --sim-size 500 --reset-mode continuous --router-modes online_cellmean --online-use-direction-split-list false --online-num-quantile-bins 10,12 --online-min-cell-obs 10,5 --online-score-thresholds 0.003,0.001`
   - Artifact:
     - `../PancakeBot_var_exp/router_sparsity_crosscheck_20260303_table.json/csv`
   - Top short-horizon setting:
     - `bins=12,min_obs=10,thr in {0.001,0.003}`
     - `2.365539` per 500.
   - But same setting on 2000-round sweep only delivered `0.342595` per 500.

6. Pivot sweep (architecture-level mode comparison) at robust sparse point:
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.run_backtest_router_matrix --name-prefix router_modepivot_20260303 --sim-size 2000 --reset-mode continuous --router-modes online_cellmean,online_cellmean_backoff,online_cellmean_selector_fallback,online_cellmean_side_gap,selector_max_score --selector-score-thresholds -1000000000.0 --online-use-direction-split-list false --online-num-quantile-bins 10 --online-min-cell-obs 10 --online-score-thresholds 0.003`
   - Artifact:
     - `../PancakeBot_var_exp/router_modepivot_20260303_table.json/csv`
   - Results:
     - `online_cellmean_backoff`: `0.656709` per 500 (best of tested modes).
     - `online_cellmean`: `0.615345` per 500.
     - `online_cellmean_selector_fallback`: `0.447817` per 500.
     - `selector_max_score`: `-0.189683` per 500.
     - `online_cellmean_side_gap`: `0 bets` (disabled effectively under `dir_split=false`).

7. 500-round mode cross-check at same sparse point:
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.run_backtest_router_matrix --name-prefix router_modepivot_crosscheck_20260303 --sim-size 500 --reset-mode continuous --router-modes online_cellmean,online_cellmean_backoff --online-use-direction-split-list false --online-num-quantile-bins 10 --online-min-cell-obs 10 --online-score-thresholds 0.003`
   - Artifact:
     - `../PancakeBot_var_exp/router_modepivot_crosscheck_20260303_table.json/csv`
   - Results:
     - `online_cellmean`: `1.230760` per 500.
     - `online_cellmean_backoff`: `0.879760` per 500.

8. Decision point from this continuation:
   - Sparser `online_cellmean` settings can improve robustness versus current promoted config.
   - None of tested robust points are close to objective `2.0 BNB / 500` on 2000-round confirmation.
   - Keep promotion on hold pending a stronger robustness criterion and next pivot design.

## Update (2026-03-03): Fixed Bet vs Cutoff-Pool Diagnostics (Uncommitted)

1. User concern examined:
   - High `fixed_bet_bnb` (0.2/0.3/0.4) may be compensating for strict
     `cutoff_pool_total_min_bnb` (1.2/2.0).
   - Hypothesis: strict cutoff gate may remove most rounds; large stake may be
     needed to clear absolute EV gates.

2. Direct pool-distribution check (`var/closed_rounds.jsonl`, cutoff=17s):
   - rounds analyzed: `100,984`.
   - cutoff pool pass rates (all rounds):
     - `>=1.2 BNB`: `52.13%`
     - `>=2.0 BNB`: `18.76%`
   - recent windows:
     - last 2,000 rounds: `>=1.2`: `62.30%`, `>=2.0`: `29.15%`
   - cutoff/final pool ratio (all rounds):
     - median `~0.5499` (most pool arrives after cutoff, but not near-zero at cutoff).

3. Candidate skip-reason attribution (baseline config, sim=2000):
   - Aggregated over candidate signals (`14,000` decisions):
     - `cutoff_pool_below_min_total`: `6,604` (`48.18%` of skips)
     - `expected_net_below_min*`: `2,745` (`20.03%` of skips)
     - `nowcast_market_agree`: `3,612` (`26.35%` of skips)
   - Candidate-level cutoff impact:
     - `disloc_stageG2_r37_x80` (cutoff min 2.0): cutoff skip share `~76.8%`.
     - `disloc_cons_20260227_x80` (cutoff min 2.0): cutoff skip share `~71.8%`.
     - 1.2-min candidates: cutoff skip share `~38%`.

4. No-edit ablation on robust horizon (sim=2000, continuous):
   - Artifact:
     - `../PancakeBot_var_exp/gate_comp_ablation_20260303.json`
   - Results:
     - baseline: per_500 `-0.0749`, bets `67`.
     - cut50 only: per_500 `-0.2043`, bets `77`.
     - bet50 only: per_500 `-0.0431`, bets `71`.
     - bet50 + ev50: per_500 `-0.1021`, bets `95`.
     - all50: per_500 `-1.1750`, bets `139`.
   - Interpretation:
     - Lowering cutoff/stake increases activity but worsened robust PnL in this pass.
     - High stake is not compensating for cutoff directly (cutoff gate is independent),
     but stake and absolute EV-min gate are coupled.

## Update (2026-03-03): Candidate Micro-Sweep Execution + Robustness Checks (Uncommitted)

1. Executed requested micro-sweep package:
   - Router frozen for test path:
     - `mode = online_cellmean_backoff`
     - `online_num_quantile_bins = 10`
     - `online_min_cell_obs = 10`
     - `online_score_threshold_bnb = 0.003`
     - `online_use_direction_split = false`
   - Sweep target candidates:
     - `disloc_stageG2_r37_x80` (`cutoff=2.0`, `bet=0.4`, `ev_min=0.146`)
     - `disloc_cons_20260227_x80` (`cutoff=2.0`, `bet=0.3`, `ev_min=0.18`)
   - Grid:
     - `cutoff_pool_total_min_bnb in {2.0, 1.8, 1.6}`
     - `fixed_bet_bnb in {0.4, 0.3, 0.25}`
     - `expected_net_min_bnb` ratio-preserved per candidate.

2. Joint 2-candidate grid (both modified together), sim=2000:
   - Artifacts:
     - `../PancakeBot_var_exp/candidate_micro_sweep_joint_20260303_table.json`
     - `../PancakeBot_var_exp/candidate_micro_sweep_joint_20260303_table.csv`
   - Result:
     - Baseline (frozen-router, no candidate edits) remained best:
       - per_500 `0.656709`, max_dd `2.375710`.
     - Most joint edits degraded materially.

3. Single-candidate grids, sim=2000:
   - Artifacts:
     - `../PancakeBot_var_exp/candidate_micro_sweep_single_20260303_table.json`
     - `../PancakeBot_var_exp/candidate_micro_sweep_single_20260303_table.csv`
   - Best row:
     - `disloc_cons_20260227_x80` only:
       - `fixed_bet_bnb: 0.3 -> 0.25`
       - `expected_net_min_bnb: 0.18 -> 0.15` (ratio-preserved)
       - per_500 `0.722826` vs baseline `0.656709`
       - max_dd `2.111242` vs baseline `2.375710`
   - `stageG2` stake/cutoff reductions were consistently harmful.

4. Backoff-path confirmations:
   - sim=5000 artifact:
     - `../PancakeBot_var_exp/candidate_micro_confirm5000_20260303.json`
     - baseline per_500 `0.667640`
     - `cons(0.25/0.15)` per_500 `0.724650`
   - sim=10000 artifact:
     - `../PancakeBot_var_exp/candidate_micro_confirm10000_20260303.json`
     - baseline per_500 `-0.056394`
     - `cons(0.25/0.15)` per_500 `-0.034328`
   - Improvement held within backoff path, but both were weak/negative at 10k.

5. Guardrail versus current active config:
   - Artifact:
     - `../PancakeBot_var_exp/final_guard_10000_20260303.json`
   - At sim=10000:
     - current active (`online_cellmean` from config): per_500 `0.317750`
     - proposed backoff+cons tweak: per_500 `-0.034328`
   - Conclusion:
     - Do **not** promote backoff path; long-horizon regression vs current active is large.

6. Cons tweak under current active router (`online_cellmean`) only:
   - Artifacts:
     - `../PancakeBot_var_exp/current_router_cons_tweak_compare_20260303.json`
     - `../PancakeBot_var_exp/current_router_cons_tweak_5000_20260303.json`
   - Results:
     - sim=2000: tweak worse (`-0.115187` vs `-0.074935`)
     - sim=5000: tweak worse (`0.140714` vs `0.174114`)
     - sim=10000: tweak better (`0.371858` vs `0.317750`) and lower drawdown
   - Decision:
     - Mixed horizon behavior; no config promotion applied.

7. Current state decision:
   - Active `config.toml` kept unchanged pending a more robust multi-window criterion.

## Update (2026-03-03): Projected Final-Pool Gate Prototype (Uncommitted)

1. Prototype implemented in active strategy path:
   - `pancakebot/config/strategy_config.py`
     - Added candidate fields:
       - `pool_total_gate_mode`
       - `projected_final_pool_multiplier`
       - `projected_final_pool_total_min_bnb`
   - `pancakebot/config/load_config.py`
     - Added strict parser support + validation.
     - Defaults preserve current behavior:
       - `pool_total_gate_mode = "cutoff_only"`
       - `projected_final_pool_multiplier = 1.0`
       - `projected_final_pool_total_min_bnb = 0.0`
   - `pancakebot/domain/strategy/dislocation_engine.py`
     - Added pool-total gate mode switch:
       - `cutoff_only` -> existing `cutoff_pool_total_min_bnb` gate.
       - `projected_final_only` -> projected gate using:
         - `projected_final = cutoff_pool_total * projected_final_pool_multiplier`
         - gate against `projected_final_pool_total_min_bnb`.
     - New skip reason:
       - `projected_final_pool_below_min_total`.
   - `config.toml`
     - Candidate schema comments updated with optional projected-gate keys.

2. Tests added/passed:
   - New file:
     - `tests/test_dislocation_pool_gate.py`
   - Covers:
     - parser defaults,
     - projected-field parsing,
     - invalid gate-mode rejection,
     - cutoff vs projected gate decision logic.
   - Validation command:
     - `.\.venv\Scripts\python.exe -m unittest tests.test_dislocation_pool_gate tests.test_strategy_router tests.test_ml_candidate_adapter -v`
     - passed.

3. Validation backtests (current active router; no config promotion):
   - Artifact:
     - `../PancakeBot_var_exp/projected_pool_gate_eval_20260303.json`
   - Prototype settings:
     - `projected_final_pool_multiplier = 1.82`
     - projected threshold = existing candidate `cutoff_pool_total_min_bnb`
     - tested variants:
       - `cons_projected`
       - `stageG2_projected`
       - `both_projected`
       - baseline.
   - 2000-round sweep:
     - baseline: `-0.074935 / 500`, bets `67`, max_dd `2.402401`
     - `cons_projected`: `-0.115856 / 500`, bets `65`, max_dd `2.534658`
     - `stageG2_projected`: `-0.800710 / 500`, bets `81`, max_dd `5.233370`
     - `both_projected`: `-0.706911 / 500`, bets `80`, max_dd `4.887874`
     - best variant: baseline.
   - Confirm baseline only (since best remained baseline):
     - 5000: `0.174114 / 500`
     - 10000: `0.317750 / 500`

4. Decision:
   - Do **not** promote projected-final-pool gate configuration at this time.
   - Keep feature available behind candidate-level mode flags for future targeted experiments.

## Update (2026-03-04): Single Closed-Rounds Policy + Resume-Critical State (Uncommitted)

1. Storage policy enforced:
   - Deleted all duplicated experiment window files:
     - `../PancakeBot_var_exp/*_closed_rounds.jsonl`
   - Verification:
     - duplicate window count = `0`
   - Canonical source only:
     - `var/closed_rounds.jsonl`

2. Code changes to prevent future window-file duplication:
   - Added backtest-native offset support:
     - `pancakebot/backtest/config.py`
       - new field: `tail_offset_rounds: int = 0`
     - `pancakebot/backtest/runner.py`
       - uses `tail_offset_rounds` to slice from canonical round tail in-memory
       - includes `tail_offset_rounds` in snapshot cache key and summary
   - Harness plumbed for offset:
     - `inspection/backtest_harness_common.py`
       - `run_backtest_case(..., tail_offset_rounds=0, ...)`
   - Sweep scripts migrated to offset mode (no window JSONL writes):
     - `inspection/run_long_window_idea_sweep.py`
     - `inspection/run_final_model_gate_window_sweep.py`
     - `inspection/run_promotion_window_eval.py`

3. Validation completed:
   - compile:
     - `python -m compileall` on touched modules passed
   - tests:
     - `python -m unittest tests.test_backtest_snapshot_key tests.test_load_config_active_candidates tests.test_dislocation_pool_gate`
     - passed
   - smoke:
     - `smoke_tail_offset_20260304` completed, summary present

4. Critical pending experiments (do these first when resuming):
   - Two runs are still incomplete (dir + trades exist, summary missing):
     - `final_model_gate_window_sweep_20260303_t5_t10_c2_off5000_model_p10p0_selector_perf_c2_sim30000`
     - `final_model_gate_window_sweep_20260303_t5_t10_c2_off15000_model_p10p0_selector_perf_c2_sim30000`
   - After these two complete, regenerate consolidated sweep outputs:
     - `../PancakeBot_var_exp/final_model_gate_window_sweep_20260303_t5_t10_c2.json`
     - `../PancakeBot_var_exp/final_model_gate_window_sweep_20260303_t5_t10_c2.csv`

5. Suggested exact resume commands (do not alter variants):
   - Run missing `off5000` p10 job:
     - `.\.venv\Scripts\python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix final_model_gate_window_sweep_20260303_t5_t10_c2 --sim-size 30000 --offsets 5000 --thresholds 10 --profiles c2 --drawdown-cap-bnb 2.0`
   - Run missing `off15000` p10 job:
     - `.\.venv\Scripts\python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix final_model_gate_window_sweep_20260303_t5_t10_c2 --sim-size 30000 --offsets 15000 --thresholds 10 --profiles c2 --drawdown-cap-bnb 2.0`
   - Rebuild consolidated table (resume mode, no new work expected):
     - `.\.venv\Scripts\python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix final_model_gate_window_sweep_20260303_t5_t10_c2 --sim-size 30000 --offsets 0,5000,10000,15000,20000 --thresholds 5,10 --profiles c2 --drawdown-cap-bnb 2.0`

6. User-approved next program of work (not yet implemented in this thread):
   - Implement full DB-backed acceleration plan:
     - market-data DB (`rounds/bets/klines`)
     - projection cache DB
     - run registry DB
     - checkpoint reuse for walk-forward state
     - cleanup tooling and retention
   - Add parallel sweep execution (multiprocessing) for independent experiment runs:
     - keep single-run simulation loop deterministic/sequential
     - use shared DB/cache reads across workers
     - enforce safe worker cap based on CPU/disk throughput

## Update (2026-03-04): DB-Backed Acceleration + Parallel Sweeps + Missing 30k Runs Completed (Uncommitted)

1. Implemented market-data SQLite mirror and integrated it into backtest/inspection path:
   - New module:
     - `pancakebot/infra/market_data_db.py`
       - `MarketDataDb` mirrors canonical sources into SQLite:
         - `rounds`
         - `round_bets`
         - `klines`
         - `meta` source signatures
       - `SqliteKlinesStore` provides read-only kline API compatible with active feature/model path.
   - Important robustness fix:
     - Wei fields persisted as text-safe values to avoid SQLite integer overflow.
   - Backtest path now uses DB-backed kline store and DB tail-round loading.

2. Added persistent projection cache DB for final-pool model projections:
   - New module:
     - `pancakebot/infra/projection_cache_store.py`
   - Wired into ML adapter:
     - `pancakebot/domain/strategy/ml_candidate_adapter.py`
       - lookup/write projection cache by `(epoch, lock_at, cutoff_ts, bull_wei, bear_wei)`
       - periodic prune by latest settled epoch
   - Wired through runtime/backtest config construction:
     - `pancakebot/integration/app.py`
     - `inspection/backtest_harness_common.py`
     - `inspection/run_backtest_scenario.py`
     - `pancakebot/backtest/runner.py`
     - `pancakebot/runtime/runtime_loop.py`

3. Added run registry DB and hooked inspection runs:
   - New module:
     - `pancakebot/infra/run_registry_store.py`
   - `run_backtest_case(...)` now records start/completion/failure to registry.
   - `inspection/run_backtest_scenario.py` also records run lifecycle.

4. Added cleanup/retention tooling:
   - New script:
     - `inspection/cleanup_experiment_artifacts.py`
       - prune old backtest-state cache files
       - remove old failed run directories (by registry)
       - optional DB `VACUUM` for registry/cache DBs

5. Config/path surface extended for DB acceleration:
   - `pancakebot/config/app_config.py`
   - `pancakebot/config/load_config.py`
   - `config.toml`
   - New parsed paths:
     - `paths.market_data_db_path`
     - `paths.projection_cache_db_path`
     - `paths.run_registry_db_path`
   - Also added parser support for:
     - `backtest.tail_offset_rounds`

6. Backtest cache concurrency hardening for multiprocessing:
   - `pancakebot/backtest/state_cache.py`
     - process-unique temp files for cache writes
     - retry behavior on replace collisions (Windows lock contention)

7. Parallel sweep support added:
   - `inspection/run_final_model_gate_window_sweep.py`
     - new flag: `--max-workers`
     - process-pool execution for independent runs
     - safe worker cap logic for CPU/disk contention
   - Note:
     - single-run simulation remains deterministic/sequential by design.

8. Tests/validation completed:
   - compile:
     - `python -m compileall pancakebot inspection tests`
   - unit tests passed:
     - `tests.test_market_data_db`
     - `tests.test_projection_cache_store`
     - `tests.test_run_registry_store`
     - `tests.test_backtest_snapshot_key`
     - `tests.test_load_config_active_candidates`
     - `tests.test_dislocation_pool_gate`
     - `tests.test_ml_candidate_adapter`
     - `tests.test_strategy_router`
   - smoke:
     - `smoke_db_accel_20260304` passed
   - duplicate window check:
     - `../PancakeBot_var_exp/*_closed_rounds.jsonl` count remains `0`

9. Previously missing 30k experiments are now complete:
   - Completed run:
     - `final_model_gate_window_sweep_20260303_t5_t10_c2_off5000_model_p10p0_selector_perf_c2_sim30000`
   - Completed run:
     - `final_model_gate_window_sweep_20260303_t5_t10_c2_off15000_model_p10p0_selector_perf_c2_sim30000`

10. Consolidated sweep outputs regenerated:
    - `../PancakeBot_var_exp/final_model_gate_window_sweep_20260303_t5_t10_c2.json`
    - `../PancakeBot_var_exp/final_model_gate_window_sweep_20260303_t5_t10_c2.csv`
    - Current aggregate highlights:
      - `cutoff_selector_perf_c2`: mean `+0.030243 / 500`, `4/5` positive windows
      - `model_p5p0_selector_perf_c2`: mean `+0.005658 / 500`, `3/5` positive windows
      - `model_p10p0_selector_perf_c2`: `0 bets`, `0.0 / 500` across all windows

11. Current DB artifact state:
    - `market_data_v1.sqlite`: rounds `100,984`, bets `4,538,426`, klines `400,664`
    - `projection_cache_v1.sqlite`: projection rows `280`
    - `run_registry_v1.sqlite`: `7` runs tracked (`5 completed`, `2 failed`)

## Update (2026-03-04): Projected Final-Pool EV Stake Sizing (Uncommitted)

1. Implemented dislocation stake modes that directly use model-projected final pools for EV:
   - New stake modes in `pancakebot/domain/strategy/dislocation_engine.py`:
     - `ev_scaled_projected`
     - `ev_optimal_projected`
   - Backward compatibility:
     - existing modes unchanged (`fixed`, `ev_scaled`, `ev_optimal`)
     - projected modes fallback to cutoff pools when model projection is unavailable.

2. Added projected EV pool resolution with uncertainty scaling:
   - New internal helpers:
     - `_stake_mode_uses_projected_pool_ev(...)`
     - `_effective_ev_pools(...)`
   - For projected stake modes:
     - uses model-projected final side pools `(bull, bear)` when available.
     - scales only **late inflow** by `projected_final_pool_multiplier`:
       - `1.0` = full model inflow
       - `<1.0` = conservative lower-bound style usage
       - `>1.0` = aggressive extrapolation
   - Side-pool cap for max stake still uses real cutoff side pools.

3. Wired projected EV pools through full dislocation decision path:
   - `_decide_core(...)` now accepts projected total/bull/bear pools.
   - Core EV computations (`ev_bull`, `ev_bear`) can use projected EV pools.
   - `_CoreDecision` now carries `ev_pool_bull_bnb` / `ev_pool_bear_bnb`.
   - Stake sizing + dynamic expected-net gate + perf flip EV checks now reuse these EV pools.
   - Adaptive shadow profit path also reuses projected EV pools when present.

4. Ensured ML projection provider auto-starts for projected stake modes (even if ML candidate trading is disabled):
   - `pancakebot/backtest/runner.py`
   - `pancakebot/runtime/runtime_loop.py`
   - `needs_pool_projection_model` now returns true when any candidate stake mode is:
     - `ev_scaled_projected` or `ev_optimal_projected`
     - or when gate mode is `projected_final_model_only` (existing behavior).

5. Inspection sweep updates for fast experimentation:
   - `inspection/run_long_window_idea_sweep.py`
     - Added variants:
       - `stake_ev_scaled_projected_selector`
       - `stake_ev_optimal_projected_selector`
       - `stake_ev_optimal_projected_lb50_selector`
     - `projected_final_pool_multiplier` / `projected_final_pool_total_min_bnb` overrides now apply even without forcing pool gate mode.

6. Config docs update:
   - `config.toml` comment now lists projected stake modes under `stake_mode`.

7. Tests run (venv Python):
   - `.\.venv\Scripts\python.exe -m unittest tests.test_dislocation_pool_gate tests.test_strategy_router tests.test_ml_candidate_adapter`
   - Result: all passed.
   - Added new tests in `tests/test_dislocation_pool_gate.py` for:
     - projected stake mode detection
     - projected EV pool late-inflow scaling
     - fallback to cutoff pools when not in projected mode

8. Sanity backtest evidence (projected path active):
   - Short warmup (model not train-ready): projected modes match cutoff behavior (expected fallback).
   - Longer warmup (15k) produced model train/calibrate logs during warmup, confirming projected provider usage.
   - Quick 300-round smoke results:
     - `smoke2_ev_optimal_cutoff_20260304`: bets `13`, avg bet `0.181538`, net `-0.881785`
     - `smoke2_ev_optimal_projected_20260304`: bets `23`, avg bet `0.103043`, net `-1.052293`
     - `smoke2_ev_optimal_projected_lb50_20260304`: bets `4`, avg bet `0.072500`, net `-0.199500`
   - These are smoke checks only (not long-run conclusions), but confirm the new projected sizing path materially changes behavior and responds to inflow multiplier conservativeness.

## Update (2026-03-05): Gas + Threshold Confirmation, ML-Knob Reality Check, and ML-Enabled Probe (Uncommitted)

Primary objective context (user-stated):
- Priority 1: maximize average net profit per 500 rounds (target `>= 2.0 BNB`)
- Priority 2: keep max drawdown low (hard cap `<= 5 BNB`, preferred `< 2 BNB`)
- Priority 3: increase bet frequency (user preference: much higher than current, potentially 30-50%)

Execution log source of truth:
- Machine log with step-by-step entries:
  - `../PancakeBot_var_exp/priority_plan_20260305_progress.jsonl`

### 1) Completed long confirmations (50k) for prior gas sweeps

Commands used (all with resume + cache reuse):
- Gas 1 gwei:
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=1000000000`
  - `./.venv/Scripts/python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix priority_gate_gas1_20260305 --sim-size 20000 --offsets 0,10000,20000 --thresholds 2,5 --profiles base,c2 --drawdown-cap-bnb 5.0 --run-long-confirm --long-sim-size 50000 --top-k-confirm 4 --max-workers 0 --worker-memory-gb 1.5 --reserve-memory-gb 2.0`
- Gas 0.2 gwei:
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000`
  - same command with `--name-prefix priority_gate_gas0p2_20260305`
- Gas 0:
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=0`
  - same command with `--name-prefix priority_gate_gas0_20260305`

Artifacts:
- `../PancakeBot_var_exp/priority_gate_gas1_20260305.json/.csv`
- `../PancakeBot_var_exp/priority_gate_gas0p2_20260305.json/.csv`
- `../PancakeBot_var_exp/priority_gate_gas0_20260305.json/.csv`

Key confirmed outcomes:
- `gas0p2 | model_p5p0_selector_perf_c2`: `+0.088793 / 500`, `max_dd 0.902479`, `bet_rate 0.0020`
- `gas0 | model_p5p0_selector_perf_c2`: `+0.088864 / 500`, `max_dd 0.901839`, `bet_rate 0.0020`
- `gas1 | model_p5p0_selector_base`: `+0.071739 / 500`, `max_dd 1.504466`, `bet_rate 0.0022`

### 2) ML knob sweep run and diagnosis (important)

Command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=1000000000`
- `./.venv/Scripts/python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix priority_mlknob_gas1_c2_t2_t5_20260305 --sim-size 20000 --offsets 0,10000,20000 --thresholds 2,5 --profiles c2 --drawdown-cap-bnb 5.0 --max-workers 0 --worker-memory-gb 1.5 --reserve-memory-gb 2.0 --ml-train-sizes 4000,8000 --ml-calibrate-sizes 2000,4000 --ml-retrain-intervals 250,500 --ml-recalibrate-intervals 125,250`

Artifact:
- `../PancakeBot_var_exp/priority_mlknob_gas1_c2_t2_t5_20260305.json/.csv`

Critical finding:
- All knob combinations produced identical aggregate outcomes because `strategy.ml_candidate.enabled=false` in `config.toml`.
- Logs during this sweep showed `ml_candidate_enabled=False`.
- Conclusion: these 4 ML knobs are inert in current disabled-ML operating mode.

### 3) Code patch to make ML knob experiments actually actionable

File changed:
- `inspection/run_final_model_gate_window_sweep.py`

Changes:
- Added CLI flag: `--ml-enabled` (optional bool override).
- Added bool parser helper (`true/false` tokens).
- Propagated `ml_enabled_override` through:
  - per-window jobs
  - long-confirm runs
  - JSON output metadata.
- `strategy.ml_candidate.enabled` can now be forced in sweeps without editing `config.toml`.

Validation:
- `./.venv/Scripts/python.exe -m compileall inspection/run_final_model_gate_window_sweep.py` passed.

### 4) ML-enabled smoke/probe evidence

Smoke command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=1000000000`
- `./.venv/Scripts/python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix smoke_ml_enabled_20260305 --sim-size 800 --offsets 0 --thresholds 2 --profiles c2 --drawdown-cap-bnb 5.0 --ml-enabled true --ml-train-sizes 4000 --ml-calibrate-sizes 2000 --ml-retrain-intervals 250 --ml-recalibrate-intervals 125 --max-workers 0 --worker-memory-gb 1.5 --reserve-memory-gb 2.0 --no-resume`

Observation:
- `ml_candidate_enabled=True` confirmed.
- Warmup/runtime became much slower.
- Bet rate jumped materially (example ~200 bets in 800 rounds) but PnL poor in smoke.

Targeted probe command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=1000000000`
- `./.venv/Scripts/python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix priority_ml_enabled_probe_gas1_t0p5_base_20260305 --sim-size 5000 --offsets 0 --thresholds 0.5 --profiles base --drawdown-cap-bnb 5.0 --ml-enabled true --ml-train-sizes 8000 --ml-calibrate-sizes 4000 --ml-retrain-intervals 500 --ml-recalibrate-intervals 250 --max-workers 2 --worker-memory-gb 2.5 --reserve-memory-gb 2.0 --no-resume`

Probe result:
- `cutoff_selector_base_ml_t8000_c4000_rt500_rc250`: `+0.167795 / 500`, `bet_rate 0.0314`, `max_dd 5.127404` (over hard DD cap)
- `model_p0p5_selector_base_ml_t8000_c4000_rt500_rc250`: `-0.077291 / 500`, `bet_rate 0.0264`, `max_dd 2.897067`
- Interpretation: ML-enabled path can increase activity into low-single-digit bet rates, but stability/profit is inconsistent and expensive to evaluate.

### 5) New threshold sweep (0.5/1/2/5) with ML disabled (fast path)

Commands:
- Gas 1 gwei:
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=1000000000`
  - `./.venv/Scripts/python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix priority_thresh_gas1_t0p5_t1_t2_t5_20260305 --sim-size 20000 --offsets 0,10000,20000 --thresholds 0.5,1,2,5 --profiles base,c2 --drawdown-cap-bnb 5.0 --ml-enabled false --max-workers 0 --worker-memory-gb 1.5 --reserve-memory-gb 2.0`
- Gas 0.2 gwei:
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000`
  - same command with `--name-prefix priority_thresh_gas0p2_t0p5_t1_t2_t5_20260305`

Artifacts:
- `../PancakeBot_var_exp/priority_thresh_gas1_t0p5_t1_t2_t5_20260305.json/.csv`
- `../PancakeBot_var_exp/priority_thresh_gas0p2_t0p5_t1_t2_t5_20260305.json/.csv`

Highlights:
- New best means appeared for base profile with lower thresholds:
  - Gas1: `model_p0p5_selector_base` mean `+0.086787 / 500`, worst `+0.043324`
  - Gas0p2: `model_p0p5_selector_base` mean `+0.088637 / 500`, worst `+0.054728`
- `p0.5` and `p1.0` were effectively identical in these windows.

### 6) Long confirmations (50k) for new threshold candidates

Commands:
- Gas1 confirm:
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=1000000000`
  - reran same threshold sweep command with:
    - `--run-long-confirm --long-sim-size 50000 --top-k-confirm 4`
- Gas0p2 confirm:
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000`
  - same with `--run-long-confirm --long-sim-size 50000 --top-k-confirm 4`

Best confirmed (current top non-ML):
- `gas0p2 | model_p0p5_selector_base`: `+0.114630 / 500`, `net +11.462954`, `max_dd 2.370746`, `loss2 0.600080`, `bets 263`, `bet_rate 0.0053`
- `gas1   | model_p0p5_selector_base`: `+0.111344 / 500`, `net +11.134439`, `max_dd 2.990543`, `loss2 0.300200`, `bets 265`, `bet_rate 0.0053`

Still true:
- No tested setup is close to target `2.0 BNB / 500`.
- Bet frequency remains far below user’s desired 30-50% on non-ML path.

### 7) Current scoreboard snapshot (50k confirms only)

Top by `per_500` with hard cap context (`dd<=5`, `floor true`):
1. `priority_thresh_gas0p2_t0p5_t1_t2_t5_20260305 | model_p0p5_selector_base | +0.114630 | dd 2.370746`
2. `priority_thresh_gas1_t0p5_t1_t2_t5_20260305 | model_p0p5_selector_base | +0.111344 | dd 2.990543`
3. `priority_gate_gas0_20260305 | model_p5p0_selector_perf_c2 | +0.088864 | dd 0.901839`
4. `priority_gate_gas0p2_20260305 | model_p5p0_selector_perf_c2 | +0.088793 | dd 0.902479`

Preferred-risk subset (`dd<2`) still led by:
- `model_p5p0_selector_perf_c2` around `+0.0888 / 500` with very low bet rate (~0.2%).

### 8) Practical constraints observed

- Memory planner with `worker_memory_gb=1.5` and `reserve=2.0` yielded 4-5 workers when RAM allowed and prevented swap-crash behavior in these sweeps.
- ML-enabled warmups are much more expensive; broad ML grid searches will be time-intensive without additional warmup/cache strategy changes.

### 9) Recommended next step for the next agent

Run a compact, high-signal confirm matrix around current best non-ML candidate:
- Candidate family: `model_p0p5_selector_base` (and equivalent `p1.0` only for sanity check)
- Gas scenarios: `1 gwei`, `0.2 gwei`, and optionally `0`
- Use multi-offset long confirmations if possible (not just one 50k span) to reduce single-window overfitting risk.

In parallel, if higher bet frequency remains mandatory:
- Use the new `--ml-enabled true` path for a very small targeted grid only (2-4 runs), since full ML-grid cost is high.
- Evaluate activity/profit/DD jointly; do not rely on short-window wins.

### 10) Stage H completed: multi-offset long-window cross-gas confirmation (2026-03-06)

Goal:
- Stress current top base-threshold candidate family over more offsets (`0,1000,2000,3000,4000,5000`) with longer windows (`sim_size=45000`), and compare gas sensitivity at `1`, `0.2`, and `0` gwei.

Commands run:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=1000000000`
- `./.venv/Scripts/python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix priority_multiwin_gas1_t0p5_t1_base_sim45000_20260306 --sim-size 45000 --offsets 0,1000,2000,3000,4000,5000 --thresholds 0.5,1 --profiles base --drawdown-cap-bnb 5.0 --ml-enabled false --max-workers 0 --worker-memory-gb 1.5 --reserve-memory-gb 2.0`
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000`
- `./.venv/Scripts/python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix priority_multiwin_gas0p2_t0p5_t1_base_sim45000_20260306 --sim-size 45000 --offsets 0,1000,2000,3000,4000,5000 --thresholds 0.5,1 --profiles base --drawdown-cap-bnb 5.0 --ml-enabled false --max-workers 0 --worker-memory-gb 1.5 --reserve-memory-gb 2.0`
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=0`
- `./.venv/Scripts/python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix priority_multiwin_gas0_t0p5_t1_base_sim45000_20260306 --sim-size 45000 --offsets 0,1000,2000,3000,4000,5000 --thresholds 0.5,1 --profiles base --drawdown-cap-bnb 5.0 --ml-enabled false --max-workers 0 --worker-memory-gb 1.5 --reserve-memory-gb 2.0`

Artifacts:
- `../PancakeBot_var_exp/priority_multiwin_gas1_t0p5_t1_base_sim45000_20260306.json/.csv`
- `../PancakeBot_var_exp/priority_multiwin_gas0p2_t0p5_t1_base_sim45000_20260306.json/.csv`
- `../PancakeBot_var_exp/priority_multiwin_gas0_t0p5_t1_base_sim45000_20260306.json/.csv`
- Cross-scenario ranking:
  - `../PancakeBot_var_exp/priority_multiwin_crossgas_rank_20260306.json/.csv`

Key results (all 6/6 floor-pass and 6/6 positive for the model variants):
- `gas1 | model_p0p5_selector_base`: mean `+0.107030 / 500`, worst `+0.066918`, worst DD `2.990543`, mean bets `232.7`, mean bet rate `~0.00517`
- `gas0p2 | model_p0p5_selector_base`: mean `+0.111893 / 500`, worst `+0.075652`, worst DD `2.370746`, mean bets `232.3`, mean bet rate `~0.00516`
- `gas0 | model_p0p5_selector_base`: mean `+0.113021 / 500`, worst `+0.065400`, worst DD `2.696070`, mean bets `240.3`, mean bet rate `~0.00534`
- `model_p1p0_selector_base` is numerically identical to `model_p0p5_selector_base` in this matrix.
- `cutoff_selector_base` remained higher activity (~0.84% bet rate) but had negative worst windows and DD cap violations at low gas (`dd 5.398` at 0.2 gwei, `dd 6.622` at 0 gwei).

Interpretation:
- Lowering gas from `1` to `0.2` and then `0` improves mean `per_500` only marginally (about `+0.006` absolute from gas1 to gas0 on best variant), not remotely enough to approach the target `2.0 / 500`.
- Primary bottleneck remains signal edge and/or bet sizing/selection dynamics, not gas level alone.
- Bet frequency still very low (~0.5%) on stable non-ML path.

Resource behavior note:
- Auto worker cap changed by available RAM at launch:
  - gas1 run: final cap `2` (`avail_mem_gb ~6.31`)
  - gas0p2 run: final cap `4` (`avail_mem_gb ~8.66`)
  - gas0 run: final cap `5` (`avail_mem_gb ~9.61`)
- No swap/crash observed during these runs with `worker_memory_gb=1.5`, `reserve=2.0`.

### 11) Stage I completed: broader idea-space sweep at 0.2 gwei (2026-03-06)

Purpose:
- After Stage H showed only marginal gains from gas reductions, run a broader strategy-space sweep (router/side/stake/perf variants) to search for higher edge and/or bet frequency.

Command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000`
- `./.venv/Scripts/python.exe -m inspection.run_long_window_idea_sweep --name-prefix priority_idea_gas0p2_sim20000_20260306 --sim-size 20000 --offsets 0,5000,10000 --drawdown-cap-bnb 5.0 --run-long-confirm --long-sim-size 45000 --top-k-confirm 6`

Artifacts:
- `../PancakeBot_var_exp/priority_idea_gas0p2_sim20000_20260306.json/.csv`

Aggregate highlights (3 windows x 20k):
- `core_online`: mean `+0.065051 / 500`, worst `+0.037669`, worst DD `3.995812`, mean bet rate `~0.00913`
- `stake_ev_optimal_selector`: mean `+0.074639 / 500`, worst `+0.002222`, worst DD `3.730578`, mean bet rate `~0.00915`
- `core_selector`: mean `+0.062354 / 500`, worst `+0.002408`, worst DD `4.496343`, mean bet rate `~0.00965`

Long-confirm (45k) for top-k:
- `core_online`: `+0.119149 / 500`, net `+10.723397`, DD `2.796962`, loss2 `1.651199`, bets `411`, bet_rate `0.00913`
- `stake_ev_optimal_selector`: `+0.107502 / 500`, net `+9.675214`, DD `3.867866`, loss2 `1.054990`, bets `425`, bet_rate `0.00944`
- `core_selector`: `+0.084623 / 500`, DD `4.007609`, bets `417`
- `stake_ev_scaled_selector`: identical to `core_selector`
- `stake_ev_optimal_projected_lb50_selector`: negative (`-0.023909 / 500`)
- `perf_strict_selector`: negative (`-0.011574 / 500`)

Cross-run master ranking (Stage H + Stage I):
- Artifacts:
  - `../PancakeBot_var_exp/priority_master_rank_20260306.json/.csv`
- Top 6 (all DD <= 5):
  1. `stage_i | gas0p2 | core_online | +0.119149 / 500 | dd 2.796962 | bet_rate 0.00913`
  2. `stage_h | gas0 | model_p0p5_selector_base | +0.113021 / 500 | dd 2.696070 | bet_rate 0.00534`
  3. `stage_h | gas0 | model_p1p0_selector_base | +0.113021 / 500 | dd 2.696070 | bet_rate 0.00534`
  4. `stage_h | gas0p2 | model_p0p5_selector_base | +0.111893 / 500 | dd 2.370746 | bet_rate 0.00516`
  5. `stage_h | gas0p2 | model_p1p0_selector_base | +0.111893 / 500 | dd 2.370746 | bet_rate 0.00516`
  6. `stage_i | gas0p2 | stake_ev_optimal_selector | +0.107502 / 500 | dd 3.867866 | bet_rate 0.00944`

Interpretation:
- Broader idea search found a slightly better top run (`core_online`), and improved bet frequency versus the previous best (from ~0.5% to ~0.9%), but still nowhere near the target `2.0 / 500`.
- No tested setup currently indicates a plausible path to `2.0 / 500` without a major shift in model edge and/or staking regime.

### 12) Stage J completed: online-router parameter matrix at 0.2 gwei (20k horizon)

Purpose:
- Tune `online_cellmean` around the new `core_online` lead to test if online thresholds/binning/obs can materially improve short-window edge.

Command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000`
- `./.venv/Scripts/python.exe -m inspection.run_backtest_router_matrix --name-prefix priority_router_online_tune_gas0p2_sim20000_20260306 --sim-size 20000 --reset-mode continuous --router-modes=online_cellmean,selector_max_score --selector-score-thresholds=-1000000000.0,0.0 --online-score-thresholds=-0.002,-0.001,0.0,0.001 --online-warmup-rounds=50000 --online-num-quantile-bins=6,10 --online-min-cell-obs=5,15 --online-use-direction-split-list=true,false`

Artifacts:
- `../PancakeBot_var_exp/priority_router_online_tune_gas0p2_sim20000_20260306_table.json/.csv`

Top short-window results:
- `online_cellmean | thr -0.002 | bins 6 | min_obs 15 | split false`: `+0.267623 / 500`, net `+10.704931`, DD `2.851217`, bets `244/20000` (~1.22%)
- `online_cellmean | thr 0.0 | bins 6 | min_obs 15 | split false`: `+0.253753 / 500`, DD `3.042273`
- `online_cellmean | thr -0.001 | bins 6 | min_obs 15 | split false`: `+0.251028 / 500`, DD `2.851217`

Interpretation:
- Strong uplift appeared on the 20k horizon versus earlier bests, but required long-horizon confirmation due high overfit risk.

### 13) Stage K completed: long-window confirmation of Stage J top family (45k horizon)

Purpose:
- Confirm whether Stage J’s online-router uplift survives longer horizon.

Command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000`
- `./.venv/Scripts/python.exe -m inspection.run_backtest_router_matrix --name-prefix priority_router_online_confirm_gas0p2_sim45000_20260306 --sim-size 45000 --reset-mode continuous --router-modes=online_cellmean --online-score-thresholds=-0.002,-0.001,0.0,0.001 --online-warmup-rounds=50000 --online-num-quantile-bins=6 --online-min-cell-obs=5,15 --online-use-direction-split-list=false`

Artifacts:
- `../PancakeBot_var_exp/priority_router_online_confirm_gas0p2_sim45000_20260306_table.json/.csv`

Top confirmed results (45k):
- `online_cellmean | thr -0.002 | bins 6 | min_obs 15 | split false`: `+0.085656 / 500`, net `+7.709009`, DD `3.507131`, bets `423`
- `online_cellmean | thr 0.0 | bins 6 | min_obs 15 | split false`: `+0.077115 / 500`, DD `3.315831`
- `online_cellmean | thr -0.002 | bins 6 | min_obs 5 | split false`: `+0.075710 / 500`, DD `3.864147`

Outcome:
- Stage J spike did **not** hold on long horizon; these variants underperform current leaders.

Updated long-horizon leaderboard artifact:
- `../PancakeBot_var_exp/priority_longconfirm_master_rank_20260306.json/.csv`

Current best long-horizon signal remains:
- `stage_i | gas0p2 | core_online | +0.119149 / 500 | dd 2.796962 | bet_rate 0.00913`

Practical conclusion after Stage H/I/J/K:
- Repeated long-horizon checks still cluster around `~0.08-0.12 / 500` at acceptable DD.
- No tested configuration has shown a credible path toward `2.0 / 500`.

### 14) Stage L completed: explicit priority stake-scaling sweep (0.2 gwei, 45k)

Code change made to support this stage:
- File: `inspection/run_long_candidate_matrix.py`
- Added CLI:
  - `--priority-stake-sweep`
  - `--stake-scales`
  - `--model-gate-min-totals`
- Added priority variant generator for:
  - `core_online_scale*`
  - `modelgate_p*_selector_scale*`

Run command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000`
- `./.venv/Scripts/python.exe -m inspection.run_long_candidate_matrix --name-prefix priority_stake_scale_gas0p2_sim45000_20260306 --sim-sizes 45000 --drawdown-cap-bnb 5.0 --priority-stake-sweep --stake-scales 0.5,0.75,1.0,1.25,1.5,2.0 --model-gate-min-totals 0.5`

Artifacts:
- `../PancakeBot_var_exp/priority_stake_scale_gas0p2_sim45000_20260306.json/.csv`

Top results:
- `modelgate_p0p5_selector_scale1p0`: `+0.127366 / 500`, net `+11.462954`, DD `2.370746`, bets `263`
- `core_online_scale1p5`: `+0.121973 / 500`, net `+10.977557`, DD `2.914743`, bets `266`
- `core_online_scale1p0`: `+0.119149 / 500`, net `+10.723397`, DD `2.796962`, bets `411`

Important failures/risk behavior:
- `core_online_scale1p25`: `-0.019956 / 500`, DD `10.790709` (hard DD failure)
- `core_online_scale2p0`: `-0.141129 / 500`, DD `15.723148` (hard DD failure)
- Several mid scales on selector/model-gate turned weak/negative despite DD<=5.

Interpretation:
- Stake scaling did not unlock a path to target magnitude.
- Best stable result improved only slightly (to `0.127366 / 500`), still far below `2.0 / 500`.
- Aggressive scaling quickly destabilizes and breaches drawdown limits.

Updated long-horizon master rank (v2):
- `../PancakeBot_var_exp/priority_longconfirm_master_rank_v2_20260306.json/.csv`
- Current top entries:
  1. `stage_l | modelgate_p0p5_selector_scale1p0 | +0.127366 / 500 | dd 2.370746`
  2. `stage_l | core_online_scale1p5 | +0.121973 / 500 | dd 2.914743`
  3. `stage_l/stage_i | core_online_scale1p0/core_online | +0.119149 / 500 | dd 2.796962`

### 15) Stage M/N/O/P completed: feasibility ceiling + ML knob sweeps + frequency probes + mild risk overlays (2026-03-07)

Scope executed from this handoff state:
1. Added reproducible hindsight ceiling analyzer.
2. Extended long candidate matrix runner for ML/frequency sweeps.
3. Ran feasibility-ceiling analysis from existing best baseline trade logs.
4. Ran ML knob sweeps with gas sensitivity.
5. Ran safe frequency-loosening/probe sweep.
6. Re-ran mild circuit-breaker + anti-martingale overlays on winners.

Code changes:
- `inspection/run_negative_period_ceiling.py` (new)
  - Inputs: matrix CSV + variants + skip budgets + window sizes.
  - Methods: `oracle_individual_loss_skip`, `oracle_contiguous_window_skip`.
  - Outputs: JSON/CSV under `../PancakeBot_var_exp`.
- `inspection/run_long_candidate_matrix.py` (extended)
  - New sweep modes:
    - `--priority-ml-knob-sweep`
    - `--priority-frequency-sweep`
  - New tunables:
    - ML: `--ml-train-sizes`, `--ml-calibrate-sizes`, `--ml-retrain-intervals`, `--ml-recalibrate-intervals`, `--ml-enabled-values`
    - Frequency/entry: `--freq-expected-net-mins`, `--freq-selector-thresholds`, `--freq-online-thresholds`, `--freq-stake-scales`, `--freq-model-gate-min-totals`
  - Variant-level overrides now support:
    - Candidate expected-net min override
    - Router selector/online threshold override
    - ML candidate window/retrain overrides

Stage M: Feasibility ceiling (hindsight upper bound)
- Commands:
  - `./.venv/Scripts/python.exe -m inspection.run_negative_period_ceiling --name-prefix priority_negative_ceiling_gas0p2_20260306 --matrix-csv ..\PancakeBot_var_exp\priority_riskadapt_cbanti_mild_gas0p2_sim45000_20260306.csv --variants baseline_modelgate_selector_p0p5,baseline_core_online --skip-budgets 0.01,0.02,0.05,0.1,0.2,0.3 --window-sizes 250,500,1000 --initial-bankroll-bnb 50.0`
  - `./.venv/Scripts/python.exe -m inspection.run_negative_period_ceiling --name-prefix priority_negative_ceiling_gas0_20260306 --matrix-csv ..\PancakeBot_var_exp\priority_riskadapt_cbanti_mild_gas0_sim45000_20260306.csv --variants baseline_modelgate_selector_p0p5,baseline_core_online --skip-budgets 0.01,0.02,0.05,0.1,0.2,0.3 --window-sizes 250,500,1000 --initial-bankroll-bnb 50.0`
- Artifacts:
  - `../PancakeBot_var_exp/priority_negative_ceiling_gas0p2_20260306.json/.csv`
  - `../PancakeBot_var_exp/priority_negative_ceiling_gas0_20260306.json/.csv`
- Summary:
  - Gas 0.2:
    - `baseline_modelgate_selector_p0p5`: base `0.1274`, best individual-skip `0.4741`, best contiguous-window `0.3641`
    - `baseline_core_online`: base `0.1191`, best individual-skip `0.7226`, best contiguous-window `0.5725`
  - Gas 0:
    - `baseline_modelgate_selector_p0p5`: base `0.1354`, best individual-skip `0.4921`, best contiguous-window `0.3754`
    - `baseline_core_online`: base `0.1253`, best individual-skip `0.7120`, best contiguous-window `0.5786`
- Interpretation:
  - Even optimistic hindsight ceilings are still far below `2.0 BNB / 500`.

Stage N: ML knob sweeps + gas sensitivity (10k screening)
- Commands (representative):
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000 ./.venv/Scripts/python.exe -m inspection.run_long_candidate_matrix --name-prefix priority_mlknobs_gas0p2_t8000_c4000_rt500_rc250_sim10000_20260306 --sim-sizes 10000 --drawdown-cap-bnb 5.0 --priority-ml-knob-sweep --ml-enabled-values true --ml-train-sizes 8000 --ml-calibrate-sizes 4000 --ml-retrain-intervals 500 --ml-recalibrate-intervals 250`
  - Same for gas `0` and `1 gwei`.
  - Additional gas0.2 configs:
    - `t6000 c2000 rt250 rc125`
    - `t10000 c4000 rt1000 rc500`
- Artifacts:
  - `../PancakeBot_var_exp/priority_mlknobs_gas0p2_t8000_c4000_rt500_rc250_sim10000_20260306.csv`
  - `../PancakeBot_var_exp/priority_mlknobs_gas0_t8000_c4000_rt500_rc250_sim10000_20260306.csv`
  - `../PancakeBot_var_exp/priority_mlknobs_gas1_t8000_c4000_rt500_rc250_sim10000_20260306.csv`
  - `../PancakeBot_var_exp/priority_mlknobs_gas0p2_t6000_c2000_rt250_rc125_sim10000_20260306.csv`
  - `../PancakeBot_var_exp/priority_mlknobs_gas0p2_t10000_c4000_rt1000_rc500_sim10000_20260306.csv`
- Highlights:
  - `core_online` ML baseline (`t8000/c4000/rt500/rc250`):
    - gas0 `0.4421`, gas0.2 `0.3657`, gas1 `0.3307`
  - `modelgate` ML baseline:
    - gas0 `0.0365`, gas0.2 `0.0601`, gas1 `-0.1197`
  - Best alternate on gas0.2:
    - `core_online t10000/c4000/rt1000/rc500`: `0.3874`
    - `modelgate t10000/c4000/rt1000/rc500`: `0.1016`

Stage O: Frequency probe sweep (gas0.2, 10k)
- Command:
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000 ./.venv/Scripts/python.exe -m inspection.run_long_candidate_matrix --name-prefix priority_freqprobe_gas0p2_sim10000_20260306 --sim-sizes 10000 --drawdown-cap-bnb 5.0 --priority-frequency-sweep --freq-expected-net-mins=0.18,0.08,0.0 --freq-selector-thresholds=-0.01,-0.03 --freq-online-thresholds=0.0,-0.002 --freq-stake-scales=0.25 --freq-model-gate-min-totals=0.5`
- Artifact:
  - `../PancakeBot_var_exp/priority_freqprobe_gas0p2_sim10000_20260306.csv`
- Highlights:
  - Best remained baselines:
    - `baseline_core_online`: `0.4314`, DD `1.8248`, bet_rate `0.0170`
    - `baseline_modelgate_selector_p0p5`: `0.1967`, DD `1.7634`, bet_rate `0.0133`
  - Looser probe variants increased participation but reduced return:
    - `core_online_freq_e0p0_s0p25_othr_m0p002`: bet_rate `0.0673`, per_500 `0.0466`
    - `modelgate_freq_e0p0_s0p25_sth_*`: bet_rate `0.1464`, per_500 `-0.0487`
- Interpretation:
  - Participation can be raised, but currently at significant per_500 cost.

Stage P: Mild circuit-breaker + anti-martingale overlay (gas0.2, 10k)
- Command:
  - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000 ./.venv/Scripts/python.exe -m inspection.run_long_candidate_matrix --name-prefix priority_riskadapt_mild2_gas0p2_sim10000_20260306 --sim-sizes 10000 --drawdown-cap-bnb 5.0 --priority-risk-adapt-sweep --cb-triggers 2.0,3.0 --cb-base-skips 10,20 --anti-max-scales 1.15,1.25 --anti-win-multiplier 1.10 --anti-loss-multiplier 0.95 --anti-min-scale 0.75 --cb-escalation-multiplier 1.25 --cb-escalation-window-rounds 200 --cb-max-level 4 --cb-max-skip-rounds 80 --cb-reentry-rounds 10 --cb-reentry-scale 0.90`
- Artifact:
  - `../PancakeBot_var_exp/priority_riskadapt_mild2_gas0p2_sim10000_20260306.csv`
- Highlights:
  - Baseline still top:
    - `baseline_core_online`: `0.4314`, DD `1.8248`, bet_rate `0.0170`
  - Best mild overlay:
    - `core_online_cb_t2p0_n10`: `0.2289`, DD `1.2154`, bet_rate `0.0089`
  - Modelgate overlays also under baseline.
- Interpretation:
  - Mild overlays improve/contain DD in some variants but currently cut returns and participation.

Net status after Stage M/N/O/P:
- Objective ordering checked: `per_500` first, DD second, bet frequency third.
- Current best from this stage set: `~0.43 / 500` on 10k (`baseline_core_online` at 0.2 gwei).
- Hard evidence still indicates no credible path to `2.0 / 500` with current strategy family under tested knobs.

Recommended immediate next step if continuing:
1. Promote top 3 from this stage to strict long-horizon confirmation (`45k`, and multi-offset) before any live-facing changes.
2. If long-confirm collapses back toward `~0.1-0.2 / 500`, pivot to model/feature redesign instead of more threshold/sizing sweeps.

### 16) Stage Q completed: strict long-confirm of core baseline vs ML-on (gas0.2, 45k multi-window) (2026-03-07)

Context for this stage:
- Goal was to continue the plan after Stage M/N/O/P by running strict long-horizon checks on the best 10k candidates.
- History bound at run time: `total_rounds=100984`, selector warmup `50000`, so for `sim_size=45000` max offset is `5984`.
- Chosen offsets for robust-but-valid evaluation: `0,1500,3000,4500,5500`.

#### Q1) Core baseline long-confirm (non-ML)
Command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000 .\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix priority_longconfirm2_gas0p2_core_base_20260307 --candidate-name disloc_altA_20260227_x80 --router-mode online_cellmean --keep-all-candidates --sim-size 45000 --offsets 0,1500,3000,4500,5500`

Artifact:
- `../PancakeBot_var_exp/priority_longconfirm2_gas0p2_core_base_20260307_table.json`

Aggregate:
- `mean_per500 = +0.086719`
- `worst_per500 = +0.029191`
- `worst_max_drawdown_bnb = 3.970701`
- `mean_bet_rate = 0.007938`
- `positive windows = 5/5`

#### Q2) Core ML-on long-confirm (`t8000/c4000/rt500/rc250`)
Initial command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000 .\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix priority_longconfirm2_gas0p2_core_ml_t8000_c4000_rt500_rc250_20260307 --candidate-name disloc_altA_20260227_x80 --router-mode online_cellmean --keep-all-candidates --sim-size 45000 --offsets 0,1500,3000,4500,5500 --ml-enabled true --ml-set train_size=8000 --ml-set calibrate_size=4000 --ml-set retrain_interval=500 --ml-set recalibrate_interval=250`

Execution note:
- Full-offset run exceeded tool timeout and left an orphan python process.
- Recovered by killing orphan and resuming remaining offsets separately:
  - same command with `--offsets 4500`
  - same command with `--offsets 5500`
- Then reran full offsets with resume to materialize final aggregate JSON.

Artifacts:
- `../PancakeBot_var_exp/priority_longconfirm2_gas0p2_core_ml_t8000_c4000_rt500_rc250_20260307_table.json`
- `../PancakeBot_var_exp/priority_longconfirm2_gas0p2_core_ml_t8000_c4000_rt500_rc250_20260307.log`

Aggregate:
- `mean_per500 = -0.004113`
- `worst_per500 = -0.105050`
- `worst_max_drawdown_bnb = 14.283202`
- `mean_bet_rate = 0.018644`
- `positive windows = 3/5`

Decision:
- Rejected. This violates hard DD cap badly and is non-positive on mean per_500 despite higher frequency.

#### Q3) Core ML-on anchor check (`t10000/c4000/rt1000/rc500`)
Command:
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000 .\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix priority_longconfirm2_gas0p2_core_ml_t10000_c4000_rt1000_rc500_20260307 --candidate-name disloc_altA_20260227_x80 --router-mode online_cellmean --keep-all-candidates --sim-size 45000 --offsets 0 --ml-enabled true --ml-set train_size=10000 --ml-set calibrate_size=4000 --ml-set retrain_interval=1000 --ml-set recalibrate_interval=500`

Artifact:
- `../PancakeBot_var_exp/priority_longconfirm2_gas0p2_core_ml_t10000_c4000_rt1000_rc500_20260307_table.json`

Anchor result (`n=1`):
- `per500 = +0.052295`
- `max_drawdown_bnb = 7.958810`
- `bet_rate = 0.016044`

Decision:
- Also fails hard DD cap on anchor; full offset expansion intentionally skipped.

#### Q4) Outcome and current best after Stage Q
Ranked by your priority order (profit first, DD second, participation third):
1. `core_online baseline (non-ML)`
   - `mean_per500 +0.086719`, `worst_dd 3.970701`, `mean_bet_rate 0.007938`
2. `core_online ML t10000 anchor` (not fully expanded)
   - lower profit and DD violation already on anchor.
3. `core_online ML t8000`
   - mean profit <= 0 and severe DD violations.

Net:
- Keep ML disabled for this strategy family under current setup.
- Stage Q confirms no path close to `2.0 BNB / 500`; best robust result remains far below target.
- Post-run housekeeping verified: no lingering python worker processes.

### 17) Stage R completed: faithful all-candidate circuit-breaker long-confirm (2026-03-07)

Reason for this stage:
- Stage P short-horizon risk-adapt runs were not long-confirmed with both active candidates overridden in `run_alta_single_idea`.
- Added a runner capability to make that possible and tested the strongest short-horizon breaker idea on the same strict windows.

Code change:
- `inspection/run_alta_single_idea.py`
  - Added `--apply-overrides-to-all-candidates`.
  - Behavior: applies `--set` / `--candidate-overrides-json` / `--stake-scale` to every selected candidate (instead of only `--candidate-name`).
  - `py_compile` passed.

Long-confirm command (gas0.2, 45k, multi-window):
- `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000 .\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix priority_longconfirm2_gas0p2_core_cb_t2_n10_20260307 --candidate-name disloc_altA_20260227_x80 --router-mode online_cellmean --keep-all-candidates --apply-overrides-to-all-candidates --sim-size 45000 --offsets 0,1500,3000,4500,5500 --set circuit_breaker_enabled=true --set circuit_breaker_drawdown_trigger_bnb=2.0 --set circuit_breaker_base_skip_rounds=10 --set circuit_breaker_escalation_multiplier=1.25 --set circuit_breaker_escalation_window_rounds=200 --set circuit_breaker_max_level=4 --set circuit_breaker_max_skip_rounds=80 --set circuit_breaker_reentry_rounds=10 --set circuit_breaker_reentry_scale=0.9 --set anti_martingale_enabled=false`

Artifact:
- `../PancakeBot_var_exp/priority_longconfirm2_gas0p2_core_cb_t2_n10_20260307_table.json`

Aggregate result:
- `mean_per500 = -0.017237`
- `worst_per500 = -0.038085`
- `worst_max_drawdown_bnb = 10.717542`
- `worst_loss_from_initial_to_min_bnb = 9.453621`
- `mean_bet_rate = 0.005876`

Decision:
- Rejected. This breaker configuration is worse than core baseline on profit and drawdown in strict long-horizon windows.

Updated best-known candidate from Stage Q+R:
- `priority_longconfirm2_gas0p2_core_base_20260307` remains best in this family:
  - `mean_per500 +0.086719`, `worst_dd 3.970701`, `mean_bet_rate 0.007938`.

## Update (2026-03-12): Clean Checkpoint + altA Family Strict Multi-Offset Confirm

1. Repo normalized into a clean checkpoint:
   - Commit:
     - `8e0e44d`
     - `Add DB-backed backtest acceleration and model-gate sweep tooling`
   - Validation before commit:
     - `.\.venv\Scripts\python.exe -m compileall pancakebot inspection tests`
     - `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py"`
     - `.\.venv\Scripts\python.exe -m inspection.run_backtest_scenario --name smoke_repo_coherence_20260312 --sim-size 50 --reset-mode continuous`
     - `.\.venv\Scripts\python.exe -m inspection.run_final_model_gate_window_sweep --name-prefix smoke_final_model_gate_20260312 --sim-size 500 --offsets 0 --thresholds 2 --profiles base --max-workers 1 --top-k-confirm 1`
   - Local-only note files preserved via stash:
     - `stash@{0}: On master: local notes and assistant rules`

2. Artifact history was audited to recover the real continuation point after the older handoff sections:
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.analyze_experiment_history --top-n 15`
   - Key finding:
     - stronger-looking single-window 30k results existed for the `altA` selector family
       (for example `idea40_altA_flowimb015_20260304_off0_sim30000`), but they had not
       been strict multi-offset confirmed.

3. Strict multi-offset confirmation executed for the `altA` selector family at `0.2 gwei`:
   - Common settings:
     - router: `selector_max_score`
     - keep active ensemble: `--keep-all-candidates`
     - offsets: `0,1500,3000,4500,5500`
     - sim size: `45000`
     - env:
       - `PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=200000000`

4. Baseline current family (`altA` target, no overrides):
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix nextstep_altA_base_gas0p2_20260312 --candidate-name disloc_altA_20260227_x80 --router-mode selector_max_score --keep-all-candidates --sim-size 45000 --offsets 0,1500,3000,4500,5500`
   - Artifact:
     - `../PancakeBot_var_exp/nextstep_altA_base_gas0p2_20260312_table.json`
   - Aggregate:
     - `mean_per500 = +0.040089`
     - `worst_per500 = -0.022713`
     - `worst_max_drawdown_bnb = 5.398439`
     - `worst_loss_from_initial_to_min_bnb = 5.068797`
     - `positive windows = 3/5`

5. `altA` with stronger flow gate (`flow_min_imbalance = 0.15`):
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix nextstep_altA_flowimb015_gas0p2_20260312 --candidate-name disloc_altA_20260227_x80 --router-mode selector_max_score --keep-all-candidates --sim-size 45000 --offsets 0,1500,3000,4500,5500 --set flow_min_imbalance=0.15`
   - Artifact:
     - `../PancakeBot_var_exp/nextstep_altA_flowimb015_gas0p2_20260312_table.json`
   - Aggregate:
     - `mean_per500 = +0.052051`
     - `worst_per500 = -0.006773`
     - `worst_max_drawdown_bnb = 5.234328`
     - `worst_loss_from_initial_to_min_bnb = 4.789707`
     - `positive windows = 4/5`
   - Interpretation:
     - better than current baseline across profit robustness and loss-to-min.
     - still misses the stricter DD objective because one window reached `5.234328` DD.

6. `altA` flow gate + `late_model_veto_enabled=true`:
   - Command:
     - `.\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix nextstep_altA_flowimb015_lateveto_gas0p2_20260312 --candidate-name disloc_altA_20260227_x80 --router-mode selector_max_score --keep-all-candidates --sim-size 45000 --offsets 0,1500,3000,4500,5500 --set flow_min_imbalance=0.15 --set late_model_veto_enabled=true`
   - Artifact:
     - `../PancakeBot_var_exp/nextstep_altA_flowimb015_lateveto_gas0p2_20260312_table.json`
   - Aggregate:
     - identical to plain `flowimb015` on these windows:
       - `mean_per500 = +0.052051`
       - `worst_per500 = -0.006773`
       - `worst_max_drawdown_bnb = 5.234328`
       - `worst_loss_from_initial_to_min_bnb = 4.789707`
   - Interpretation:
     - `late_model_veto` was inert for this confirmation slice; keep it out of follow-up
       sweeps unless a specific targeted hypothesis is introduced.

7. Net result from this continuation step:
   - The earlier promising 30k single-window `altA` signal does **not** hold robustly at
     strict 45k multi-offset confirmation.
   - Best of the tested family is now:
     - `flow_min_imbalance = 0.15` on `altA` within the current selector ensemble.
   - Magnitude remains small (`~0.052 / 500`) and far below the user target `2.0 / 500`.

8. Recommended next step if continuing from here:
   - If staying within this family:
     - run a very small adjacent sweep around `flow_min_imbalance` (`0.12, 0.15, 0.18`)
       and optionally compare:
       - altA-only override
       - altB-only override
       - apply-to-all-candidates
     - only promote if strict multi-offset confirmation improves both:
       - `mean_per500`
       - DD / loss-to-min
   - If prioritizing fastest progress toward the objective:
     - stop local threshold/gate tweaking in this family and pivot again to a larger model
       or feature change, because robust edge remains too small.

## Update (2026-03-12): Predictability-Gate Pivot Implemented + First Full Probe

1. Code changes completed for the predictability-gate pivot:
   - Added configurable gate feature families:
     - `all_features`
     - `regime_only`
     - `arrival_microstructure_only`
     - `arrival_microstructure_plus_regime`
     - `arrival_microstructure_plus_regime_plus_price`
   - Added configurable label modes:
     - `baseline_log_imbalance_side` (existing deterministic fixed-policy baseline)
     - `either_side_profitable` (research-only probe)
   - Wiring added end-to-end through:
     - `config.toml`
     - `pancakebot/config/load_config.py`
     - `pancakebot/config/strategy_config.py`
     - `pancakebot/domain/models/predictability_modes.py`
     - `pancakebot/domain/models/predictability_model.py`
     - `pancakebot/domain/models/walk_forward.py`
     - `pancakebot/domain/strategy/ml_candidate_adapter.py`
     - `inspection/run_ml_strategy_blocks.py`
   - New tests:
     - `tests/test_predictability_modes.py`
     - expanded `tests/test_load_config_ml_aliases.py`

2. Validation after the pivot code:
   - `.\.venv\Scripts\python.exe -m compileall pancakebot inspection tests`
   - `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py"`
   - Result:
     - `54` tests passed.

3. First A/B/C gate-family probe used the new `either_side_profitable` label mode on a 10k block replay:
   - Shared settings:
     - block layout: `20 x 500`
     - `train_size=8000`
     - `calibrate_size=4000`
     - `min_tradeable_prob=0.51`
     - `min_prob_edge=0.0015`
     - fixed bet `0.2`
   - Runs:
     - `pivot_gate_regime_20260312`
     - `pivot_gate_arrival_20260312`
     - `pivot_gate_combined_20260312`
   - Result:
     - all three were **bit-for-bit identical**
     - aggregate:
       - `net_per_500 = -0.525679`
       - `bets_total = 1063`
       - `positive_blocks = 5/20`
   - Skip totals (aggregate):
     - `cutoff_pool_below_min_total = 4910`
     - `expected_net_below_min = 3928`
     - `p_bull_edge_below_min = 99`
     - `predictability_below_min = 0`

4. Root cause of the inert A/B/C result:
   - Direct probability probe on the most-recent `500` rounds for
     `arrival_microstructure_plus_regime + either_side_profitable` showed:
     - `p_tradeable == 1.0` for every sampled round
   - Interpretation:
     - `either_side_profitable` is degenerate under the current baseline-bet semantics;
       it produces an always-pass gate and should **not** be used as-is for promotion.

5. Fixed-policy label probe (`baseline_log_imbalance_side`) on the most recent `500` rounds:
   - `regime_only`:
     - `mean p_tradeable = 0.507473`
     - `q10 = 0.504443`, `q50 = 0.507263`, `q90 = 0.510954`
     - rounds below `0.51`: `434/500`
   - `arrival_microstructure_only`:
     - `mean p_tradeable = 0.508059`
     - `q10 = 0.499052`, `q50 = 0.508005`, `q90 = 0.517310`
     - rounds below `0.51`: `313/500`
   - `arrival_microstructure_plus_regime`:
     - `mean p_tradeable = 0.507649`
     - `q10 = 0.504478`, `q50 = 0.507509`, `q90 = 0.510597`
     - rounds below `0.51`: `332/500`
   - Interpretation:
     - `arrival_microstructure_only` had the only materially wider score spread near the
       live threshold and became the best follow-up candidate.

6. Full 10k replay for the strongest live pivot candidate:
   - Command family:
     - `pivot_gate_arrival_baseline_20260312`
     - settings:
       - `predictability_feature_mode = arrival_microstructure_only`
       - `predictability_label_mode = baseline_log_imbalance_side`
       - `min_tradeable_prob = 0.51`
   - Aggregate:
     - `net_per_500 = -0.231208`
     - `net_total = -4.624158`
     - `bets_total = 78`
     - `positive_blocks = 2/20`
     - `win_rate_weighted = 0.384615`
   - Aggregate skip totals:
     - `cutoff_pool_below_min_total = 4910`
     - `predictability_below_min = 4791`
     - `expected_net_below_min = 221`
   - Most recent block (`off0`) remained bad:
     - `net_profit_bnb = -2.771001`
     - `num_bets = 57`
     - `predictability_below_min = 248`

7. Net conclusion from this pivot stage:
   - The code pivot was worth doing:
     - it produced a live microstructure gate that materially reduced trade count and
       improved loss vs the degenerate/inert gate setup.
   - But it is still not close to acceptable:
     - `-0.231 / 500` is a clear improvement over `-0.526 / 500`
       yet still far below break-even and nowhere near the objective.
   - Current best research takeaway:
     - `arrival_microstructure_only + baseline_log_imbalance_side` is the only tested
       pivot variant that actually behaves like a gate and improves robustness at all.

8. Recommended next step from here:
   - Do **not** promote `either_side_profitable` without redesign; it is currently an
     always-pass label.
   - Next highest-signal research step:
     - run a tight threshold sweep around the live microstructure gate
       (`min_tradeable_prob` roughly `0.508` to `0.518`)
       using `arrival_microstructure_only + baseline_log_imbalance_side`
     - reason:
       - the score distribution is extremely narrow, so the threshold is now the main
         control knob.
   - If that still fails:
     - redesign the label itself to a non-degenerate fixed directional policy that is
       less tightly coupled to `log_imb_w_p_80_to_p_100`, then re-run the same family
       comparison.

## Update (2026-03-12): Tight Threshold Sweep For Live Microstructure Gate

1. Follow-up executed exactly as recommended from the prior pivot checkpoint:
   - family:
     - `arrival_microstructure_only + baseline_log_imbalance_side`
   - all other settings held fixed:
     - `train_size=8000`
     - `calibrate_size=4000`
     - `fixed_bet_bnb=0.2`
     - `min_prob_edge=0.0015`
     - `cutoff_pool_total_min_bnb=1.2`
     - `expected_net_min_bnb=0.0`
   - block layout:
     - `20 x 500` (`10k` total rounds)

2. Threshold sweep run:
   - `min_tradeable_prob = 0.508`
     - `net_per_500 = -0.271320`
     - `net_total = -5.426393`
     - `bets_total = 139`
     - `positive_blocks = 2/20`
   - `min_tradeable_prob = 0.510`
     - `net_per_500 = -0.231208`
     - `net_total = -4.624158`
     - `bets_total = 78`
     - `positive_blocks = 2/20`
   - `min_tradeable_prob = 0.512`
     - `net_per_500 = -0.134212`
     - `net_total = -2.684234`
     - `bets_total = 48`
     - `positive_blocks = 1/20`
   - `min_tradeable_prob = 0.514`
     - `net_per_500 = -0.102600`
     - `net_total = -2.052004`
     - `bets_total = 32`
     - `positive_blocks = 1/20`
   - `min_tradeable_prob = 0.516`
     - `net_per_500 = -0.078939`
     - `net_total = -1.578773`
     - `bets_total = 22`
     - `positive_blocks = 0/20`
   - `min_tradeable_prob = 0.518`
     - `net_per_500 = -0.053601`
     - `net_total = -1.072027`
     - `bets_total = 16`
     - `positive_blocks = 0/20`
   - extension:
     - `min_tradeable_prob = 0.520`
       - `net_per_500 = -0.041210`
       - `net_total = -0.824208`
       - `bets_total = 13`
       - `positive_blocks = 0/20`
     - `min_tradeable_prob = 0.522`
       - `net_per_500 = -0.034781`
       - `net_total = -0.695628`
       - `bets_total = 9`
       - `positive_blocks = 0/20`

3. Best observed threshold in this sweep:
   - `0.522`
   - But the improvement came only from near-complete trade starvation:
     - `9` bets across `10k` rounds
     - aggregate skip totals:
       - `cutoff_pool_below_min_total = 4910`
       - `predictability_below_min = 5066`
       - `expected_net_below_min = 15`
     - most recent block (`off0`) still negative:
       - `net_profit_bnb = -0.695628`
       - `num_bets = 9`
       - `predictability_below_min = 392`

4. Interpretation:
   - The sweep was monotone:
     - higher threshold -> fewer bets -> less loss
   - There was **no interior optimum** in the tested band.
   - This means the current microstructure gate is not finding a profitable subset;
     it is only approaching break-even by asymptotically refusing to trade.

5. Net conclusion after the full pivot + threshold sweep:
   - The microstructure gate is directionally better than the inert gate variants.
   - But threshold tuning alone does **not** rescue the strategy.
   - If continued, the next work should shift away from threshold search and toward
     label redesign / direction-policy redesign.

6. Recommended next step from here:
   - Stop sweeping `min_tradeable_prob` in this family.
   - Redesign the gate label to a non-degenerate deterministic directional policy that:
     - is not the trivial always-pass `either_side_profitable`
     - is less tightly coupled to `log_imb_w_p_80_to_p_100`
   - Then re-run the same three-family comparison:
     - `regime_only`
     - `arrival_microstructure_only`
     - `arrival_microstructure_plus_regime`

## Update (2026-03-12): Contrarian Label Redesign + Follow-on Threshold Sweep

1. Code changes completed for the next label redesign cycle:
   - Added new predictability label modes:
     - `contrarian_log_imbalance_side`
     - `contrarian_price_regime_vote_15_30_r20_r60_side`
   - Refactored `walk_forward._tradeable_label(...)` so side selection is centralized in
     `_tradeable_label_side(...)` instead of being duplicated per mode.
   - Added focused tests covering:
     - invalid label mode rejection
     - contrarian log-imbalance side inversion
     - contrarian multi-feature vote inversion

2. Before running the live gate again, a raw fixed-policy baseline screen was run over the
   most recent `10k` rounds with a flat `0.05 BNB` stake:
   - `log_imb_majority`:
     - `net_per_500 = -2.589967`
   - `log_imb_majority_inverse`:
     - `net_per_500 = -1.578921`
   - `price_regime_vote_15_30_r20_r60_inverse`:
     - `net_per_500 = -1.718260`
   - Interpretation:
     - the original majority-side cutoff imbalance proxy was one of the worst simple
       policies.
     - the contrarian log-imbalance side was the strongest raw deterministic policy in
       this candidate set.
     - the inverse price+regime vote was the best less-coupled alternative and therefore
       the best next live-gate candidate.

3. Recent-`500` live score probe across gate families:
   - `contrarian_log_imbalance_side`:
     - `regime_only` mean `0.498051`
     - `arrival_microstructure_only` mean `0.498210`
     - `arrival_microstructure_plus_regime` mean `0.498299`
     - all `500/500` rounds were below `0.51` for every family.
   - `contrarian_price_regime_vote_15_30_r20_r60_side`:
     - `regime_only` mean `0.508739`, rounds below `0.51`: `434/500`
     - `arrival_microstructure_only` mean `0.508357`, rounds below `0.51`: `384/500`
     - `arrival_microstructure_plus_regime` mean `0.508567`, rounds below `0.51`: `331/500`
   - Interpretation:
     - the contrarian log-imbalance label would need a much lower threshold band.
     - the inverse price+regime vote preserved the old narrow-but-usable `~0.51`
       operating range, with the best spread in `arrival_microstructure_plus_regime`.

4. First full `10k` confirmations:
   - `arrival_microstructure_only + contrarian_log_imbalance_side`
     - `min_tradeable_prob = 0.500`
     - aggregate:
       - `net_per_500 = -0.321482`
       - `net_total = -6.429645`
       - `bets_total = 343`
       - `positive_blocks = 4/20`
     - aggregate skip totals:
       - `cutoff_pool_below_min_total = 4910`
       - `predictability_below_min = 2872`
       - `p_bull_edge_below_min = 1108`
       - `expected_net_below_min = 767`
     - most recent block (`off0`):
       - `net_profit_bnb = -0.847989`
       - `num_bets = 18`
       - `predictability_below_min = 308`
   - `arrival_microstructure_plus_regime + contrarian_price_regime_vote_15_30_r20_r60_side`
     - `min_tradeable_prob = 0.510`
     - aggregate:
       - `net_per_500 = -0.071146`
       - `net_total = -1.422927`
       - `bets_total = 158`
       - `positive_blocks = 4/20`
     - aggregate skip totals:
       - `cutoff_pool_below_min_total = 4910`
       - `predictability_below_min = 3801`
       - `p_bull_edge_below_min = 738`
       - `expected_net_below_min = 393`
     - most recent block (`off0`):
       - `net_profit_bnb = -0.637893`
       - `num_bets = 22`
       - `predictability_below_min = 266`
   - Net result:
     - the contrarian log-imbalance branch was a clear regression.
     - the less-coupled arrival+regime contrarian-vote branch became the new best tested
       label-redesign family and was materially better than the previous best live gate
       (`-0.231208 / 500`).

5. Follow-on threshold sweep on the new best family:
   - family:
     - `arrival_microstructure_plus_regime + contrarian_price_regime_vote_15_30_r20_r60_side`
   - `min_tradeable_prob = 0.510`
     - `net_per_500 = -0.071146`
     - `bets_total = 158`
     - `positive_blocks = 4/20`
   - `min_tradeable_prob = 0.511`
     - `net_per_500 = -0.053565`
     - `net_total = -1.071309`
     - `bets_total = 133`
     - `positive_blocks = 3/20`
     - aggregate skip totals:
       - `cutoff_pool_below_min_total = 4910`
       - `predictability_below_min = 4016`
       - `p_bull_edge_below_min = 620`
       - `expected_net_below_min = 321`
     - most recent block (`off0`):
       - `net_profit_bnb = -0.018237`
       - `num_bets = 5`
       - `predictability_below_min = 377`
   - `min_tradeable_prob = 0.512`
     - `net_per_500 = -0.056697`
     - `net_total = -1.133942`
     - `bets_total = 118`
     - `positive_blocks = 5/20`
     - most recent block (`off0`):
       - `net_profit_bnb = -0.402000`
       - `num_bets = 2`
       - `predictability_below_min = 398`

6. Best observed result in this cycle:
   - `arrival_microstructure_plus_regime + contrarian_price_regime_vote_15_30_r20_r60_side`
   - `min_tradeable_prob = 0.511`
   - `net_per_500 = -0.053565`
   - This is the best gate result found so far in the ML pivot line, but it is still
     negative and still far below the project objective.

7. Interpretation:
   - This redesign was not another inert failure:
     - it found a new family that beats the prior best live gate by roughly `0.178 / 500`
       while still taking `133` bets over `10k` rounds, so the improvement is not coming
       purely from near-total refusal to trade.
   - But it still did not cross into a promotable regime:
     - latest-window behaviour remains fragile.
     - the best threshold is only modestly better than its neighbors, so the edge is still
       small.
   - Current research conclusion:
     - label redesign can still move the loss curve materially.
     - this particular family appears close to exhausted without another change to either
       direction selection or EV filtering.

8. Recommended next step from here:
   - If continuing inside this family, run the deferred family comparison at the tuned
     threshold:
     - `regime_only + contrarian_price_regime_vote_15_30_r20_r60_side`
     - `arrival_microstructure_only + contrarian_price_regime_vote_15_30_r20_r60_side`
     - compare both against the tuned `arrival_microstructure_plus_regime` winner at
       `min_tradeable_prob = 0.511`
   - If that does not produce a positive variant, stop iterating on label-only changes and
     move to the next structural pivot:
     - redesign the live direction / EV selection coupling rather than only the gate label.

## Update (2026-03-13): EV-Cap Promotion + Router-Threshold Combined Strategy Follow-up

1. Code promotion completed for the EV-cap pivot:
   - Added `expected_net_max_bnb` to the ML candidate config surface and loader:
     - `pancakebot/config/strategy_config.py`
     - `pancakebot/config/load_config.py`
   - Added runtime enforcement in the shared ML candidate adapter:
     - skip reason: `expected_net_above_max`
     - file: `pancakebot/domain/strategy/ml_candidate_adapter.py`
   - Extended canonical scenario tooling so shared-path backtests can override:
     - ML enablement
     - `min_tradeable_prob`
     - `min_prob_edge`
     - `cutoff_pool_total_min_bnb`
     - `expected_net_min_bnb`
     - `expected_net_max_bnb`
     - predictability feature/label modes
     - full router mode set
     - file: `inspection/run_backtest_scenario.py`
   - Extended the multi-offset comparison harness with router score-threshold support:
     - file: `inspection/run_alta_single_idea.py`
   - Added tests:
     - loader accepts/rejects `expected_net_max_bnb` correctly
     - ML adapter skips `expected_net_above_max`
     - files:
       - `tests/test_load_config_ml_aliases.py`
       - `tests/test_ml_candidate_adapter.py`

2. Focused validation completed:
   - `compileall` on touched modules: passed
   - `unittest tests.test_ml_candidate_adapter tests.test_load_config_ml_aliases -v`: passed
   - scenario/help smokes for:
     - `inspection.run_backtest_scenario`
     - `inspection.run_alta_single_idea`

3. First real combined-strategy comparison (shared pipeline, all dislocation candidates kept,
   router=`selector_max_score`, `sim=500`, offsets `0,500,1000,1500,2000`):
   - baseline:
     - run prefix: `combo_selector_baseline_20260313`
     - aggregate:
       - `mean_per_500 = -0.036278`
       - `worst_per_500 = -1.204000`
       - `worst_max_drawdown_bnb = 1.445926`
   - tuned ML branch enabled:
     - run prefix: `combo_selector_ml_contravote_20260313`
     - ML overrides:
       - `min_tradeable_prob = 0.508`
       - `cutoff_pool_total_min_bnb = 1.5`
       - `expected_net_max_bnb = 0.005`
       - `predictability_feature_mode = arrival_microstructure_only`
       - `predictability_label_mode = contrarian_price_regime_vote_15_30_r20_r60_side`
     - aggregate:
       - `mean_per_500 = -0.100200`
       - `worst_per_500 = -1.731032`
       - `worst_max_drawdown_bnb = 1.884360`
   - interpretation:
     - the raw tuned ML branch **hurt** the combined selector path.
     - root cause was not inertness: the ML candidate was actually being selected.
     - worst window (`offset=1000`) mix:
       - `ml_arrival_contravote_cap005_pool15:12`
       - `disloc_altA_20260227_x80:2`
       - `disloc_altB_20260227_x80:2`
       - net: `-1.731032`

4. Exact trade-row follow-up on those official artifacts found the next real lever:
   - adding a router score floor on `selector_score_bnb` is exact for this selector path
     when applied as an additional skip on saved trades.
   - key exact results:
     - baseline run:
       - `threshold >= 0.001` -> `+0.084122 / 500`
     - combined ML run:
       - `threshold >= 0.004` -> `+0.149516 / 500`
       - `threshold >= 0.005` -> `+0.084122 / 500`
   - interpretation:
     - the combined ML gain does **not** come from accepting every ML signal.
     - the best observed combined path keeps only the higher-score ML subset and also
       rejects one low-score dislocation loser.

5. Official multi-offset confirmation of the thresholded selector:
   - baseline with router floor:
     - run prefix: `combo_selector_baseline_thr0p001_20260313`
     - router:
       - `mode = selector_max_score`
       - `score_threshold_bnb = 0.001`
     - aggregate:
       - `mean_per_500 = +0.084122`
       - `worst_per_500 = -1.204000`
       - `worst_max_drawdown_bnb = 1.204000`
   - combined ML + router floor:
     - run prefix: `combo_selector_ml_contravote_thr0p004_20260313`
     - router:
       - `mode = selector_max_score`
       - `score_threshold_bnb = 0.004`
     - same ML overrides as above
     - aggregate:
       - `mean_per_500 = +0.149516`
       - `worst_per_500 = -1.214567`
       - `worst_max_drawdown_bnb = 1.214567`
   - net result:
     - the thresholded ML selector beat the thresholded non-ML baseline by about
       `+0.065394 / 500` on this `5 x 500` window check.

6. Longer continuous confirm on the most recent `2000` rounds:
   - baseline thresholded:
     - scenario: `combo_selector_baseline_thr0p001_long2000_20260313`
     - `net_profit_bnb = -0.104799`
     - `profit_per_500_rounds_bnb = -0.026200`
     - `num_bets = 8`
   - thresholded ML selector:
     - scenario: `combo_selector_ml_contravote_thr0p004_long2000_20260313`
     - `net_profit_bnb = +0.092953`
     - `profit_per_500_rounds_bnb = +0.023238`
     - `num_bets = 14`
   - selected-strategy breakdown on the `2000`-round ML run:
     - `disloc_altA_20260227_x80: 1 bet, -0.301000`
     - `disloc_altB_20260227_x80: 7 bets, +0.196201`
     - `mlwf_bestset_adapt_v1: 6 bets, +0.197753`
   - interpretation:
     - the ML branch still added value on the longer slice.
     - but the edge compressed sharply versus the short `5 x 500` sweep.
     - this is still nowhere close to the `2.0 / 500` project target.

7. Current best official combined path from this cycle:
   - router:
     - `selector_max_score`
     - `router_score_threshold_bnb = 0.004`
   - ML candidate:
     - enabled
     - `min_tradeable_prob = 0.508`
     - `cutoff_pool_total_min_bnb = 1.5`
     - `expected_net_max_bnb = 0.005`
     - `predictability_feature_mode = arrival_microstructure_only`
     - `predictability_label_mode = contrarian_price_regime_vote_15_30_r20_r60_side`
   - best short-window official aggregate:
     - `+0.149516 / 500` over `5 x 500`
   - longer confirm:
     - `+0.023238 / 500` over the latest `2000`

8. Research conclusion after chaining through the next plans automatically:
   - The EV-cap pivot was real:
     - it survives promotion into the shared strategy path.
   - The next useful general lever was the router score floor:
     - `-inf` score threshold was too permissive.
   - The ML branch can add incremental value after that router floor is applied.
   - But the longer-horizon edge is still tiny and fragile relative to the project goal.

9. Recommended next step from here:
   - Do **not** promote this to active config yet.
   - First run a robustness pass on the thresholded combined selector, centered on:
     - `router_score_threshold_bnb in {0.003, 0.004, 0.005}`
     - longer windows than `500`
     - compare directly against the thresholded non-ML baseline
   - If the longer-window edge stays positive, then consider promoting:
     - router score threshold as a first-class selector preset
     - the ML EV-cap branch as an optional additive candidate
   - If it collapses on longer windows, the next structural pivot should target:
     - score calibration / ranking quality for the combined selector, not just more gate tweaks.

## Update (2026-03-13): Combined-Selector Robustness Failure + ML Filter/Veto Pivot

1. The deferred robustness pass on the thresholded combined selector was completed on
   older `2000`-round windows (`offsets = 6000,8000,10000`):
   - thresholded baseline:
     - router:
       - `selector_max_score`
       - `router_score_threshold_bnb = 0.001`
     - aggregate:
       - `mean_per_500 = +0.151805`
       - `worst_per_500 = -0.291743`
       - `worst_max_drawdown_bnb = 1.562450`
       - `positive_windows = 1/3`
   - thresholded ML selector:
     - router:
       - `selector_max_score`
       - `router_score_threshold_bnb = 0.003`
     - ML overrides:
       - `min_tradeable_prob = 0.508`
       - `cutoff_pool_total_min_bnb = 1.5`
       - `expected_net_max_bnb = 0.005`
       - `predictability_feature_mode = arrival_microstructure_only`
       - `predictability_label_mode = contrarian_price_regime_vote_15_30_r20_r60_side`
     - aggregate:
       - `mean_per_500 = +0.126639`
       - `worst_per_500 = -0.291743`
       - `worst_max_drawdown_bnb = 1.562450`
       - `positive_windows = 1/3`
   - interpretation:
     - the `0.003` standalone-ML combined-selector branch did **not** survive the older
       windows and was not promotable.

2. Score/ranking follow-up isolated why more threshold tuning was not attractive:
   - recent `2000 x {0,2000,4000}` with thresholded standalone ML:
     - `+0.516983 / 500`
     - ML standalone contribution stayed positive.
   - older `2000 x {6000,8000,10000}`:
     - `+0.126639 / 500`
     - ML standalone contribution was negative.
   - latest `5000`:
     - `+0.461140 / 500`
     - low-score ML bins were mixed.
   - interpretation:
     - simple `selector_score_bnb` floor tuning did not produce a stable ranking fix.

3. An online-router calibration follow-up was tested directly on the same ML branch over
   the latest `2000` rounds:
   - `selector_max_score @ 0.003`:
     - `+0.177990 / 500`
   - `online_cellmean` (`bins=6`, `min_obs=15`, `thr=-0.002`, `dir_split=false`):
     - `-0.532638 / 500`
   - `online_cellmean_backoff` with same settings:
     - `-0.532638 / 500`
   - `online_cellmean_selector_fallback` with the same online settings plus selector floor
     `0.003`:
     - `-0.348760 / 500`
   - interpretation:
     - the old online-router calibration path was materially worse than the plain selector
       path for this current ML branch.

4. Next structural pivot implemented in the shared strategy pipeline:
   - goal:
     - stop treating the current ML branch only as a standalone candidate and test it as a
       baseline filter/veto layer instead.
   - code changes:
     - `strategy.ml_candidate` now supports:
       - `emit_candidate`
       - `veto_opposite_side_candidates`
       - `veto_untradeable_candidates`
     - parser + config loader updated accordingly.
     - shared pipeline now applies ML veto coupling before routing:
       - untradeable ML `SKIP` can veto baseline `BET` candidates
       - ML `BET` can veto opposite-side baseline `BET` candidates
     - `expected_net_below_min` was added to the shared ML-untradeable veto reason set.
   - tests added/updated:
     - loader coverage for the new boolean flags
     - focused shared-pipeline tests for:
       - filter-only mode
       - opposite-side veto
       - untradeable veto
       - `expected_net_below_min` veto

5. Latest-`2000` screen on the new coupling modes:
   - baseline, no ML:
     - `-0.026200 / 500`
   - filter-only, opposite-side veto only:
     - inert; identical to baseline
   - filter-only, untradeable veto only:
     - `+0.199550 / 500`
     - `max_drawdown_bnb = 0.301000`
   - filter-only, both vetoes:
     - identical to untradeable-only
   - net takeaway:
     - the real lever was:
       - `emit_candidate = false`
       - `veto_untradeable_candidates = true`
     - opposite-side veto was inert in this slice.

6. Official recent multi-offset confirmation of the best new filter-only branch
   (`2000 x {0,2000,4000}`):
   - router:
     - `selector_max_score`
     - `router_score_threshold_bnb = 0.001`
   - ML overrides:
     - `emit_candidate = false`
     - `veto_untradeable_candidates = true`
     - `veto_opposite_side_candidates = false`
     - `min_tradeable_prob = 0.508`
     - `cutoff_pool_total_min_bnb = 1.5`
     - `expected_net_max_bnb = 0.005`
     - `predictability_feature_mode = arrival_microstructure_only`
     - `predictability_label_mode = contrarian_price_regime_vote_15_30_r20_r60_side`
   - aggregate:
     - `mean_per_500 = +0.381938`
     - `worst_per_500 = +0.199550`
     - `worst_max_drawdown_bnb = 0.903000`
     - `positive_windows = 3/3`
   - all selected bets in this run already had `selector_score_bnb >= 0.008`, so a
     higher router floor did not change this branch.

7. Older-window confirmation of the same filter-only branch (`2000 x {6000,8000,10000}`):
   - base variant (`expected_net_min_bnb = 0.0`):
     - `mean_per_500 = +0.037540`
     - `worst_per_500 = -0.150500`
     - `worst_max_drawdown_bnb = 0.602000`
     - `positive_windows = 1/3`
   - stricter filter with `expected_net_min_bnb = 0.001`:
     - `mean_per_500 = +0.069298`
     - `worst_per_500 = -0.075250`
     - `worst_max_drawdown_bnb = 0.301000`
     - `positive_windows = 1/3`
   - stricter filter with `expected_net_min_bnb = 0.002`:
     - `mean_per_500 = +0.052126`
     - `worst_per_500 = -0.075250`
   - stricter tradeability threshold (`min_tradeable_prob = 0.512`,
     `expected_net_min_bnb = 0.001`):
     - `mean_per_500 = +0.001679`
     - `worst_per_500 = -0.075250`
   - interpretation:
     - adding a small positive EV floor (`0.001`) improved old-window safety the most.
     - higher `min_tradeable_prob` mostly starved the only strong positive old window.

8. Latest `5000` continuous confirm for the best new filter-only branch:
   - settings:
     - same as above with `expected_net_min_bnb = 0.0`
   - result:
     - `profit_per_500_rounds_bnb = +0.518866`
     - `net_profit_bnb = +5.188664`
     - `max_drawdown_bnb = 0.903000`
     - `num_bets = 51`
   - this beat the prior standalone-ML latest-`5000` result and the latest baseline run.

9. Research conclusion after the next automatic pivot:
   - The standalone thresholded ML candidate path is not robust on older windows.
   - The online-router rescue attempt did not help.
   - The ML filter-only coupling pivot is real:
     - it improves recent medium-horizon performance,
     - materially reduces drawdown,
     - and makes older losing windows much smaller.
   - But it still does **not** solve the core objective:
     - older windows remain mixed,
     - the best robustness setting is still far below the `2.0 / 500` target.

10. Current best branch from this cycle:
   - router:
     - `selector_max_score`
     - `router_score_threshold_bnb = 0.001`
   - ML coupling:
     - `emit_candidate = false`
     - `veto_untradeable_candidates = true`
     - `veto_opposite_side_candidates = false`
   - ML gates:
     - `min_tradeable_prob = 0.508`
     - `cutoff_pool_total_min_bnb = 1.5`
     - `expected_net_min_bnb = 0.001` is the strongest old-window safety point
     - `expected_net_max_bnb = 0.005`
     - `predictability_feature_mode = arrival_microstructure_only`
     - `predictability_label_mode = contrarian_price_regime_vote_15_30_r20_r60_side`

11. Recommended next step from here:
   - This current generic tradeability filter family now looks close to exhausted.
   - The next structural pivot should supervise ML directly against baseline-candidate
     outcomes, not generic market tradeability:
     - candidate-specific success labels, or
     - a direction / veto policy tied directly to dislocation-candidate quality.
   - In other words:
     - move from “is the market tradeable?” to
       “should this specific baseline candidate be allowed through?”

## Update (2026-03-13): Candidate-Specific ML EV Coupling + Ranking Pivot

1. Shared-pipeline config surface was extended again so ML can act on baseline candidates
   directly instead of only emitting its own standalone signal:
   - added ML config flags:
     - `veto_candidate_expected_net_below_min`
     - `rescore_baseline_candidates_with_expected_net`
   - loader + default config updated.
   - shared pipeline now supports:
     - per-candidate ML EV veto
     - per-candidate ML EV rescoring (replacing `expected_profit_bnb` and
       `selector_score_bnb` on baseline `BET` candidates before routing)
   - canonical scenario runner was updated so the new filter/rescore flags are exposed in
     scenario metadata.
   - focused tests added for:
     - config acceptance
     - candidate-specific veto application
     - candidate-specific rescoring application

2. First pivot stage: candidate-specific EV veto without ranking rewrite.
   - setup:
     - router:
       - `selector_max_score`
       - `router_score_threshold_bnb = 0.001`
     - ML:
       - `emit_candidate = false`
       - `veto_candidate_expected_net_below_min = true`
       - `veto_untradeable_candidates = false`
       - `rescore_baseline_candidates_with_expected_net = false`
       - `min_tradeable_prob = 0.508`
       - `cutoff_pool_total_min_bnb = 1.5`
       - `expected_net_min_bnb = 0.001`
       - `expected_net_max_bnb = 0.005`
       - `predictability_feature_mode = arrival_microstructure_only`
       - `predictability_label_mode = contrarian_price_regime_vote_15_30_r20_r60_side`
   - latest `2000`:
     - `+0.049959 / 500`
     - `num_bets = 3`
     - `max_drawdown_bnb = 0.301000`
   - recent `3 x 2000` (`offsets = 0,2000,4000`):
     - off0:
       - `+0.049959 / 500`
     - off2000:
       - `+0.194635 / 500`
     - off4000:
       - `+0.781750 / 500`
     - aggregate:
       - `mean_per_500 = +0.342115`
       - `worst_per_500 = +0.049959`
       - `worst_max_drawdown_bnb = 0.653963`
       - `positive_windows = 3/3`
   - older `3 x 2000` (`offsets = 6000,8000,10000`):
     - off6000:
       - `+0.797682 / 500`
     - off8000:
       - `-0.168198 / 500`
     - off10000:
       - `-0.033809 / 500`
     - aggregate:
       - `mean_per_500 = +0.198559`
       - `worst_per_500 = -0.168198`
       - `worst_max_drawdown_bnb = 0.903000`
       - `positive_windows = 1/3`
   - latest `5000`:
     - `+0.466221 / 500`
     - `num_bets = 43`
     - `max_drawdown_bnb = 1.241041`
   - interpretation:
     - candidate-specific EV veto was a real, different branch:
       - better average older-window profit than the generic untradeable filter,
       - but with weaker safety and two negative older windows.

3. Second pivot stage: candidate-specific ML EV rescoring.
   - pure rescoring without veto regressed on the latest `2000`:
     - `-0.094057 / 500`
     - `num_bets = 7`
     - `max_drawdown_bnb = 1.204000`
   - adding the EV veto back on top of rescoring was the key combination:
     - latest `2000`:
       - `+0.155429 / 500`
       - `num_bets = 8`
       - `max_drawdown_bnb = 0.602000`
     - this clearly beat:
       - no ML baseline (`-0.026200 / 500`)
       - candidate-EV veto only (`+0.049959 / 500`)
       - rescoring only (`-0.094057 / 500`)

4. Official follow-up confirms for the best new branch
   (`veto_candidate_expected_net_below_min = true` +
   `rescore_baseline_candidates_with_expected_net = true`):
   - recent `3 x 2000` (`offsets = 0,2000,4000`):
     - off0:
       - `+0.155429 / 500`
       - `num_bets = 8`
       - `max_drawdown_bnb = 0.602000`
     - off2000:
       - `-0.008213 / 500`
       - `num_bets = 23`
       - `max_drawdown_bnb = 1.317745`
     - off4000:
       - `+1.169464 / 500`
       - `num_bets = 27`
       - `max_drawdown_bnb = 0.392947`
     - aggregate:
       - `mean_per_500 = +0.438893`
       - `worst_per_500 = -0.008213`
       - `worst_max_drawdown_bnb = 1.317745`
       - `positive_windows = 2/3`
   - older `3 x 2000` (`offsets = 6000,8000,10000`):
     - off6000:
       - `+0.587659 / 500`
       - `num_bets = 68`
       - `max_drawdown_bnb = 1.486940`
     - off8000:
       - `+0.120359 / 500`
       - `num_bets = 11`
       - `max_drawdown_bnb = 0.672792`
     - off10000:
       - `+0.129056 / 500`
       - `num_bets = 7`
       - `max_drawdown_bnb = 0.353376`
     - aggregate:
       - `mean_per_500 = +0.279025`
       - `worst_per_500 = +0.120359`
       - `worst_max_drawdown_bnb = 1.486940`
       - `positive_windows = 3/3`
   - latest `5000` continuous:
     - `+0.495839 / 500`
     - `net_profit_bnb = +4.958389`
     - `num_bets = 55`
     - `max_drawdown_bnb = 1.021529`

5. Exact router-threshold follow-up on the best new branch (using saved trade logs, exact for
   `selector_max_score` because the selected candidate already has the max score in each round):
   - threshold `0.001`:
     - recent mean `+0.438893 / 500`
     - older mean `+0.279025 / 500`
     - latest `5000`: `+0.495839 / 500`
   - threshold `0.002`:
     - recent mean `+0.412943 / 500`
     - older mean `+0.286644 / 500`
     - latest `5000`: `+0.448639 / 500`
   - threshold `0.003`:
     - recent mean `+0.412943 / 500`
     - older mean `+0.311727 / 500`
     - latest `5000`: `+0.415938 / 500`
   - threshold `0.004`:
     - recent mean `+0.346160 / 500`
     - older mean `+0.336811 / 500`
     - latest `5000`: `+0.441342 / 500`
   - conclusion:
     - `0.001` remains the best overall trade-off once recent + older + long-window behavior
       are considered together.

6. Current best branch from this cycle:
   - router:
     - `selector_max_score`
     - `router_score_threshold_bnb = 0.001`
   - ML coupling:
     - `emit_candidate = false`
     - `veto_candidate_expected_net_below_min = true`
     - `rescore_baseline_candidates_with_expected_net = true`
     - `veto_untradeable_candidates = false`
     - `veto_opposite_side_candidates = false`
   - ML gates:
     - `min_tradeable_prob = 0.508`
     - `cutoff_pool_total_min_bnb = 1.5`
     - `expected_net_min_bnb = 0.001`
     - `expected_net_max_bnb = 0.005`
     - `predictability_feature_mode = arrival_microstructure_only`
     - `predictability_label_mode = contrarian_price_regime_vote_15_30_r20_r60_side`

7. Research conclusion after chaining through the next structural plans automatically:
   - The generic untradeable-veto family was not the end of the line.
   - Candidate-specific ML EV has real signal for baseline-candidate quality.
   - Pure rescoring is too aggressive.
   - The winning combination is:
     - first veto baseline candidates that are too weak under the ML EV forecast,
     - then re-rank the surviving baseline candidates by that same ML EV forecast.
   - This is the strongest robustness result in the ML pivot line so far:
     - older stressed windows are all positive,
     - recent windows are strongly positive on average,
     - latest `5000` is near `+0.50 / 500`.
   - But it is still far below the `2.0 / 500` objective, so this is a meaningful improvement,
     not final success.

8. Recommended next step from here:
   - Keep this branch as the new research leader.
   - Next highest-signal follow-up:
     - sweep the candidate-EV floor (`expected_net_min_bnb`) around `0.001`
       inside the same veto+rescore architecture,
     - because thresholding now operates on the directly useful candidate-level ML EV signal,
       not the old generic tradeability score.
   - If that plateaus quickly:
     - the next pivot should target richer candidate-specific supervision / ranking features,
       not a return to the old generic gate family.

## Update (2026-03-13): Full-History Ranking at 1 Gwei + Baseline Reset

1. User clarified the comparison standard:
   - treat `1 gwei` as the default realistic on-chain gas assumption.
   - future comparisons should be judged against the best full-history result at `1 gwei`,
     not against mixed-gas or shorter-window leaders.

2. Full usable history definition for current dataset/config:
   - settled rounds available: `100984`
   - selector warmup: `50000`
   - max continuous evaluated span after warmup: `50984`

3. Full-history replays run at `1 gwei` (`PANCAKEBOT_GAS_PRICE_WEI_OVERRIDE=1000000000`):
   - thresholded selector baseline (`selector_max_score`, router floor `0.001`, ML off):
     - command:
       - `.\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix combo_selector_baseline_thr0p001_full50984_gas1_20260313 --sim-size 50984 --offsets 0 --keep-all-candidates --router-mode selector_max_score --router-score-threshold-bnb 0.001`
     - result:
       - `net_profit_bnb = +8.396034`
       - `per_500 = +0.082340`
       - `max_drawdown_bnb = 4.217983`
       - `num_bets = 382`
   - model-gate selector baseline (`baseline_modelgate_selector_p0p5`, ML off):
     - command:
       - `.\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix fullhistory_modelgate_selector_p0p5_gas1_20260313 --sim-size 50984 --offsets 0 --keep-all-candidates --apply-overrides-to-all-candidates --router-mode selector_max_score --set pool_total_gate_mode=projected_final_model_only --set projected_final_pool_total_min_bnb=0.5`
     - result:
       - `net_profit_bnb = +11.134439`
       - `per_500 = +0.109195`
       - `max_drawdown_bnb = 2.990543`
       - `num_bets = 265`
   - ML veto+rescore branch:
     - command:
       - `.\.venv\Scripts\python.exe -m inspection.run_alta_single_idea --name-prefix ml_candidateev_veto_rescore_full50984_gas1_20260313 --sim-size 50984 --offsets 0 --keep-all-candidates --router-mode selector_max_score --router-score-threshold-bnb 0.001 --ml-enabled true --ml-set min_tradeable_prob=0.508 --ml-set cutoff_pool_total_min_bnb=1.5 --ml-set expected_net_min_bnb=0.001 --ml-set expected_net_max_bnb=0.005 --ml-set predictability_feature_mode=arrival_microstructure_only --ml-set predictability_label_mode=contrarian_price_regime_vote_15_30_r20_r60_side --ml-set emit_candidate=false --ml-set veto_opposite_side_candidates=false --ml-set veto_untradeable_candidates=false --ml-set veto_candidate_expected_net_below_min=true --ml-set rescore_baseline_candidates_with_expected_net=true`
     - result:
       - `net_profit_bnb = +4.397520`
       - `per_500 = +0.043126`
       - `max_drawdown_bnb = 5.442559`
       - `num_bets = 264`
   - tuned core online baseline (`online_cellmean`, bins `6`, min_obs `15`, thr `-0.002`, no split):
     - command:
       - `.\.venv\Scripts\python.exe -m inspection.run_backtest_router_matrix --name-prefix fullhistory_core_online_gas1_20260313 --sim-size 50984 --reset-mode continuous --router-modes online_cellmean --online-warmup-rounds 50000 --online-num-quantile-bins 6 --online-min-cell-obs 15 --online-score-thresholds -0.002 --online-use-direction-split-list false`
     - result:
       - `net_profit_bnb = +0.933823`
       - `per_500 = +0.009158`
       - `max_drawdown_bnb = 6.655228`
       - `num_bets = 426`

4. Decision:
   - New baseline for future work is `baseline_modelgate_selector_p0p5` at `1 gwei`.
   - Reason:
     - it produced the best full-history result among the serious contenders rerun on the same
       `50984`-round span,
     - and it also beat the thresholded selector baseline on both profit and drawdown.

5. Going-forward comparison rule:
   - Unless explicitly stated otherwise, compare all new ideas against:
     - gas assumption: `1 gwei`
     - history span: full usable history when practical (`50984` rounds continuous after warmup)
     - baseline path:
       - router `selector_max_score`
       - ML disabled
       - candidate override applied to all active candidates:
         - `pool_total_gate_mode = projected_final_model_only`
         - `projected_final_pool_total_min_bnb = 0.5`

6. Important interpretation:
   - This is the best full-history benchmark currently known, not a success-state.
   - Even the new baseline only delivered `+0.109195 / 500`, still far below the project goal
     of `+2.0 / 500`.

## Update (2026-03-14): Baseline Attribution + First Attribution-Driven Improvement

1. New inspection tooling added for canonical backtest attribution:
   - script:
     - `inspection/run_backtest_feature_attribution.py`
   - tests:
     - `tests/test_run_backtest_feature_attribution.py`
   - purpose:
     - join a canonical backtest trade log to canonical v8 feature rows,
     - emit strategy/side summaries and feature-decile attribution tables for both
       `all_rounds` and `bet_rounds`,
     - keep future regime diagnosis inside the shared pipeline path rather than legacy tooling.

2. Baseline attribution run on locked benchmark
   (`baseline_modelgate_selector_p0p5`, full usable history, `1 gwei`):
   - artifact prefix:
     - `baseline_modelgate_selector_p0p5_gas1_fullhistory_attr_20260314`
   - core findings:
     - total result remained:
       - `+11.134439 BNB`
       - `+0.109195 / 500`
       - `265` bets
     - selected-strategy contribution:
       - `disloc_altB_20260227_x80`: `+8.586730 BNB` on `130` bets
       - `disloc_altA_20260227_x80`: `+2.547709 BNB` on `135` bets
     - strategy-side contribution:
       - `altB|Bull`: `+4.903785`
       - `altB|Bear`: `+3.682945`
       - `altA|Bull`: `+2.773210`
       - `altA|Bear`: `-0.225501`
   - strongest `all_rounds` positive pockets:
     - highest `total_sum_w_p_0_to_p_100` decile: `+1.152495 / 500`
     - most negative `log_imb_w_p_0_to_p_100` decile: `+0.931149 / 500`
     - `regime_streak_len = 1` pocket: `+0.465387 / 500`
     - high `regime_flip_rate_r_20` pocket (`~0.58-0.63`): `+0.401758 / 500`
     - highest `bet_top1_share_w_p_80_to_p_100` decile: `+0.343991 / 500`
   - weakest `all_rounds` pockets:
     - near-neutral `late_log_imb` (`~ -0.05 to +0.22`): `-0.192828 / 500`
     - neutral `regime_bull_frac_r_60` (`~0.47-0.50`): `-0.133236 / 500`
     - low `bet_top1_share_w_p_80_to_p_100` (`~0.22-0.26`): `-0.133194 / 500`
     - low `regime_flip_rate_r_20` (`~0.11-0.37`): `-0.103232 / 500`
   - interpretation:
     - the baseline edge is concentrated in stronger market-extreme / one-sided-flow pockets,
       while the persistent losses sit in more neutral or sticky regimes.

3. Attribution-driven full-history probe set at `1 gwei`
   (all against the locked baseline benchmark):
   - `projected_final_pool_total_min_bnb = 1.0`:
     - identical to baseline on this span (`+0.109195 / 500`), so baseline-selected bets
       were already above that threshold.
   - `projected_final_pool_total_min_bnb = 1.5`:
     - regressed to `+0.098348 / 500`, though DD improved to `2.485217`.
   - `flow_gate_mode = against_side`, `flow_min_imbalance = 0.15` applied to all candidates:
     - regressed hard to `+0.053091 / 500`.
   - `market_extreme_min = 0.02` applied to all candidates, while keeping
     `projected_final_pool_total_min_bnb = 0.5`:
     - improved to `+0.141524 / 500`
     - net `+14.430901 BNB`
     - max DD `2.553999`
     - `275` bets
   - conclusion:
     - the attribution signal translated into a real improvement via a modest
       market-extreme gate.

4. Tight sweep around the new signal (`market_extreme_min` at `1 gwei`, full `50984`):
   - `0.01`: `+0.123279 / 500`, DD `3.292786`
   - `0.02`: `+0.141524 / 500`, DD `2.553999`
   - `0.03`: `+0.098380 / 500`, DD `2.986093`
   - `0.05`: `+0.094672 / 500`, DD `2.475311`
   - `0.08`: `+0.026482 / 500`, DD `5.149081`
   - conclusion:
     - `market_extreme_min = 0.02` is the best observed point in this local family.

5. Multi-offset robustness check at `1 gwei` (`sim=45000`, offsets `0,1500,3000,4500,5500`):
   - locked comparison baseline
     (`baseline_modelgate_selector_p0p5`):
     - `mean_per500 = +0.102579`
     - `worst_per500 = +0.069659`
     - `worst_max_drawdown_bnb = 2.990543`
     - `positive_windows = 5/5`
   - attribution-driven variant
     (`baseline_modelgate_selector_p0p5 + market_extreme_min=0.02`):
     - `mean_per500 = +0.125663`
     - `worst_per500 = +0.075672`
     - `worst_max_drawdown_bnb = 2.553999`
     - `positive_windows = 5/5`
   - interpretation:
     - the improvement is not a single-window artifact.
     - this is the new research leader, but not the new comparison baseline unless explicitly promoted.

6. Attribution on the new research leader (`market_extreme_min = 0.02`):
   - artifact prefix:
     - `baseline_modelgate_selector_p0p5_mext0p02_gas1_fullhistory_attr_20260314`
   - selected-strategy contribution:
     - `altB`: `+10.835634` on `159` bets
     - `altA`: `+3.595268` on `116` bets
   - strategy-side contribution:
     - `altB|Bull`: `+8.202794`
     - `altB|Bear`: `+2.632840`
     - `altA|Bull`: `+3.476476`
     - `altA|Bear`: `+0.118792`
   - key interpretation:
     - the market-extreme gate largely fixed the old `altA|Bear` drag and improved the bull side most.
     - remaining weak pockets still cluster around neutral regime mixes (`regime_bull_frac_r_20/r_60`
       near `0.5`) and near-zero `late_log_imb`.

7. Recommended next step from here:
   - Keep the official comparison baseline unchanged as requested:
     - `baseline_modelgate_selector_p0p5` at `1 gwei`.
   - Treat `market_extreme_min = 0.02` as the new research leader to beat.
   - Next highest-signal structural follow-up:
     - build a candidate family or router overlay that explicitly avoids the still-bad
       neutral/near-zero-late-imbalance pockets,
     - with special attention to candidate-specific side behavior (`altA|Bear` was the weak leg
       before the market-extreme filter).

## Update (2026-03-14): Late-Veto Promotion, Old-Family Exhaustion, and Inactive-Candidate Tooling

1. New research leader promoted on full usable history (`50984` rounds, `1 gwei`):
   - base branch:
     - `baseline_modelgate_selector_p0p5`
   - added overlays:
     - `market_extreme_min = 0.02`
     - `late_model_veto_enabled = true`
     - `late_model_veto_min_late_ratio = 0.05`
     - `late_model_veto_min_abs_imbalance = 0.10`
   - full-history result:
     - `+14.724817 BNB`
     - `+0.144406 / 500`
     - `max_dd = 2.327951`
     - `274` bets
   - interpretation:
     - this is the best full-history result currently known, but it is still only a research leader.
     - the official user baseline remains `baseline_modelgate_selector_p0p5` unless explicitly promoted.

2. Robustness confirm for the new leader (`sim=45000`, offsets `0,1500,3000,4500,5500`, `1 gwei`):
   - new leader:
     - `mean_per500 = +0.127960`
     - `worst_per500 = +0.078184`
     - `worst_max_drawdown_bnb = 2.327951`
     - `positive_windows = 5/5`
   - direct comparison:
     - locked baseline: `mean +0.102579`, `worst +0.069659`, `worst_dd 2.990543`
     - prior research leader (`market_extreme_min=0.02` only):
       - `mean +0.125663`
       - `worst +0.075672`
       - `worst_dd 2.553999`
   - interpretation:
     - the late-veto gain survived the same multi-offset promotion gate.

3. Follow-up overlay probe on the new leader:
   - pre-cutoff `shock_filter` was tested on top of the new leader and failed.
   - representative results:
     - moderate filter (`window=20s`, `min_total=0.25`, `min_abs_imb=0.60`, `surge=1.5`):
       - `+0.014741 / 500`
       - only `26` bets
     - stricter filter (`window=20s`, `min_total=0.50`, `min_abs_imb=0.80`, `surge=2.5`):
       - `+0.045353 / 500`
       - `143` bets
   - conclusion:
     - the shock-filter path is too blunt for this family and should not be pursued further
       without redesign.

4. Tooling added to search inactive config-defined candidate families without mutating `config.toml`:
   - `inspection/backtest_harness_common.py`:
     - added `load_all_dislocation_candidates(...)`
   - `inspection/run_alta_single_idea.py`:
     - added `--candidate-source active|all_config`
   - tests:
     - `tests/test_run_alta_single_idea.py`
   - purpose:
     - allow research runs against inactive candidate blocks already present in `config.toml`
       while keeping the official active baseline unchanged.

5. Inactive-family screen under the current research overlays
   (`projected_final_pool_total_min_bnb=0.5`, `market_extreme_min=0.02`,
   `late_model_veto=(0.05, 0.10)`, full history, `1 gwei`):
   - `disloc_stageH_sidenowcast_when_market_disagree_perfflip...` alone:
     - `+0.054874 / 500`
     - `max_dd = 3.108462`
   - `disloc_stageB_side_adaptive_shadow...` alone:
     - `-0.002917 / 500`
   - `disloc_best_20260227_x80` alone:
     - `-0.004272 / 500`
   - `disloc_cons_20260227_x80` alone:
     - `-0.004272 / 500`
   - `disloc_stageG2_r37_x80` alone:
     - `-0.489566 / 500`
     - bankroll nearly fully destroyed
   - `altA + altB + stageH` ensemble:
     - `+0.009344 / 500`
     - `max_dd = 6.811733`
   - conclusion:
     - none of the inactive families beat the current research leader.
     - stage-H was positive by itself but destructive in the real router competition.
     - the old/inactive candidate family search is exhausted under the current `1 gwei` regime.

6. Recommended next step from here:
   - Keep the official baseline unchanged:
     - `baseline_modelgate_selector_p0p5` at `1 gwei`.
   - Keep the new research leader for future comparisons:
     - `baseline_modelgate_selector_p0p5 + market_extreme_min=0.02 + late_model_veto(0.05, 0.10)`
   - Stop spending cycles on old candidate families and blunt overlay filters.
   - Next real move should be a new candidate design or score source aimed directly at the
     remaining neutral / weak-late-imbalance loss pockets, not another local sweep of the old families.

## Update (2026-03-15): Online-Selector Research Leader, Late-Flow Exhaustion, and ML Throughput Bottleneck

1. New best full-history research leader at `1 gwei`:
   - branch:
     - `baseline_modelgate_selector_p0p5`
     - `pool_total_gate_mode = projected_final_model_only`
     - `projected_final_pool_total_min_bnb = 0.5`
     - `market_extreme_min = 0.02`
     - `late_model_veto_enabled = true`
     - `late_model_veto_min_late_ratio = 0.05`
     - `late_model_veto_min_abs_imbalance = 0.10`
     - `disloc_altA_20260227_x80: bear_expected_net_extra_min_bnb = 0.01`
     - router:
       - `mode = online_selector_score_fallback`
       - `online_score_threshold_bnb = 0.008`
   - full usable history (`50984` rounds after warmup):
     - `+19.689196 BNB`
     - `+0.193092 / 500`
     - `max_dd = 2.027066`
     - `loss_from_initial_to_min = 0.454071`
     - `379` bets
   - artifact:
     - `../PancakeBot_var_exp/fullhistory_p0p5_mext0p02_lateveto005_010_altAbearextra001_onlineselscorefallback_thr0008_gas1_20260314_table.json`
   - interpretation:
     - this is materially stronger than the prior `+0.144406 / 500` leader.
     - official baseline for user-facing comparison still remains `baseline_modelgate_selector_p0p5`
       unless explicitly promoted.

2. Attribution on the `+0.193092 / 500` leader:
   - artifact prefix:
     - `../PancakeBot_var_exp/feature_attr_p0p5_mext0p02_lateveto005_010_altAbearextra001_onlineselscorefallback_thr0008_gas1_20260314`
   - selected-strategy contribution:
     - `altB`: `209` bets
     - `altA`: `170` bets
   - side contribution from direct trade grouping:
     - `altA Bull`: `+8.057224`
     - `altB Bull`: `+6.582831`
     - `altB Bear`: `+3.633972`
     - `altA Bear`: `+1.415169`
   - interpretation:
     - all four active side buckets are positive now.
     - remaining weakness still clusters in ambiguous late-flow / middling-regime pockets.

3. Full-history late-flow follow-ups on top of the new leader:
   - `nowcast_market_gap_min = 0.01`: identical to leader.
   - `nowcast_market_gap_min = 0.02`: identical to leader.
   - mild side-aware late-support skip gate:
     - `+12.162755 BNB`
     - `+0.119280 / 500`
     - `max_dd = 3.349909`
   - continuous late-support EV bonus:
     - `late_support_ev_scale_bnb = 0.02`: identical to leader.
     - `late_support_ev_scale_bnb = 0.10`: identical to leader.
   - late-conflict flip:
     - `late_model_conflict_flip_enabled = true`: identical to leader.
   - conclusion:
     - the cheap dislocation-only late-flow structural space now looks exhausted.
     - new late-flow knobs either did nothing on the leader slice or made it worse.

4. Code promoted in this cycle:
   - router/context expansion and tests:
     - `pancakebot/domain/strategy/router.py`
     - `tests/test_strategy_router.py`
   - late-flow candidate metadata and controls:
     - `pancakebot/domain/strategy/candidate_signal.py`
     - `pancakebot/domain/strategy/dislocation_engine.py`
     - `pancakebot/config/strategy_config.py`
     - `pancakebot/config/load_config.py`
     - `tests/test_dislocation_pool_gate.py`
   - ML baseline-candidate profit calibration plumbing:
     - `pancakebot/domain/strategy/ml_candidate_adapter.py`
     - `pancakebot/domain/strategy/pipeline.py`
     - `tests/test_ml_candidate_adapter.py`
     - `tests/test_strategy_pipeline.py`
     - `tests/test_load_config_ml_aliases.py`
   - harness/CLI support:
     - `inspection/run_alta_single_idea.py`
     - `inspection/run_backtest_scenario.py`
     - `tests/test_run_alta_single_idea.py`

5. ML candidate-profit model status:
   - logic/tests passed on focused coverage.
   - a smoke run with `candidate_profit_model_enabled = true` completed cleanly.
   - practical issue:
     - full-history uncached runs are too expensive for blind iteration.
     - one uncached full-history attempt consumed roughly `44` minutes of CPU and did not yield a
       usable summary after the execution wrapper timed out.
   - interpretation:
     - this is the remaining non-dead larger pivot, but it needs a dedicated throughput/cache step
       before it becomes a usable research loop.

6. Recommended next steps:
   - Keep official baseline unchanged:
     - `baseline_modelgate_selector_p0p5` at `1 gwei`.
   - Treat the `+0.193092 / 500` online-selector branch as the research leader to beat.
   - Next concrete plan:
     - add an exact-config warm-cache helper for the ML candidate-profit branch so `continuous_initial`
       state is populated separately from long evaluation runs,
     - rerun the candidate-profit branch with cache hits on the `+0.193092` leader overlays,
     - only continue ML iteration if it beats the current leader on full history; otherwise stop
       spending cycles on this ML coupling line.

## Update (2026-03-15): Promoted Official Baseline and Dry-Run Readiness Cleanup

1. Official baseline is now promoted in code, not just in notes:
   - active candidates:
     - `disloc_altA_20260227_x80`
     - `disloc_altB_20260227_x80`
   - shared candidate overlay in `config.toml`:
     - `pool_total_gate_mode = projected_final_model_only`
     - `projected_final_pool_total_min_bnb = 0.5`
     - `market_extreme_min = 0.02`
     - `late_model_veto_enabled = true`
     - `late_model_veto_min_late_ratio = 0.05`
     - `late_model_veto_min_abs_imbalance = 0.10`
   - candidate-specific overlay:
     - `disloc_altA_20260227_x80: bear_expected_net_extra_min_bnb = 0.01`
   - router:
     - `mode = online_selector_score_fallback`
     - `online_score_threshold_bnb = 0.008`

2. Shared gas accounting is now aligned with the promoted baseline:
   - default accounting gas moved to `1 gwei`.
   - this removes the old mismatch where dry/live defaulted to `5 gwei` accounting
     while research/baseline comparisons were being judged at `1 gwei`.

3. Dry/live runtime state is now explicit config instead of hidden hard-coded paths:
   - new `[paths]` keys:
     - `claim_scan_cursor_path`
     - `dry_bets_path`
     - `dry_settled_epochs_path`
     - `dry_audit_trades_path`
   - default location is now under `var/runtime/`.
   - impact:
     - week-long dry runs can be smoke-tested and reset more safely,
     - path drift is covered by config-load tests.

4. New reproducible preset for the promoted baseline:
   - `inspection/presets/baseline_online_selector_fallback_gas1_fullhistory_v1.json`
   - benchmark snapshot:
     - net `+19.689196 BNB`
     - `+0.193092 / 500`
     - `max_dd = 2.027066`
     - `379` bets

5. New config-lock coverage:
   - current baseline defaults are now asserted by test.
   - runtime-state path overrides and unknown-path-key rejection are also covered.

6. Immediate next step after this cleanup:
   - run a short shared-pipeline smoke on the promoted baseline,
   - then start the real dry-mode smoke run from `config.toml`.

7. Additional preflight hardening added after the baseline promotion:
   - `inspection/run_runtime_preflight.py`:
     - validates config load, required local files, runtime-state parent paths,
       active-candidate presence, and optionally env vars
   - runtime dry-state loading now fails explicitly on:
     - malformed JSONL rows,
     - invalid settled-epoch lines,
     - duplicate dry-bet epochs on load
   - runtime-state paths are now validated as distinct during config load.

8. Dry bankroll persistence:
   - new runtime path:
     - `dry_bankroll_state_path = var/runtime/dry_bankroll_state.json`
   - dry mode now persists simulated bankroll after every bet and settle.
   - restart behavior:
     - prefers the dedicated bankroll state file when current,
     - falls back to recovering from `dry_bets.jsonl` / `dry_audit_trades.csv` when needed,
     - unions settled epochs from both `dry_settled_epochs.txt` and the audit CSV to avoid
       double-credit after mid-settlement interruptions.
