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
