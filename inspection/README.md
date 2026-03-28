# Inspection Tooling

Inspection tooling is split into two domains:

1. `inspection/run_backtest_scenario.py`:
   canonical probe entrypoint that executes the production backtest pipeline.
2. archived legacy probes:
   historical inspection scripts were moved out of the repo to
   `../PancakeBot_repo_archive/20260323_v1_cleanup/inspection/legacy/`.

The canonical probe path must not reimplement production strategy logic.

Additional inspection probes for strategy-routing experiments:

1. `inspection/build_strategy_router_dataset.py`:
   reads block artifacts (`../PancakeBot_var_exp/*/dislocation_trades.csv`) and builds a
   router-ready dataset with feature columns and labels.
2. `inspection/run_strategy_router_probe.py`:
   runs walk-forward router policy probes against that dataset.
3. `inspection/build_meta_strategy_dataset.py`:
   builds a block-level dataset for offline meta-strategy experiments, with
   past-only block features and next-block labels for every candidate strategy.
   It can also ingest aligned extra trade series such as a promoted baseline
   `backtest_trades.csv` via `--extra-series name=path`.
4. `inspection/run_meta_strategy_probe.py`:
   runs walk-forward block-level strategy-selection probes with `skip_all`
   fallback logic, and also supports direct `fixed_strategy` probes for
   apples-to-apples static comparisons plus baseline-relative
   `delta_trailing_mean` / `delta_ridge` selectors.
5. `inspection/run_meta_strategy_matrix.py`:
   runs a resumable parameter sweep over block-level meta-strategy probes.
   Results are written one task at a time under
   `../PancakeBot_var_exp/<name>_meta_strategy_matrix_parts/`, so long sweeps
   can be split into chunks or non-overlapping shards and safely resumed.
6. `inspection/run_meta_strategy_followthrough_probe.py`:
   runs a continuation-style baseline overlay probe that switches from a
   promoted baseline into complementary candidates only after positive
   delta-vs-baseline streak / transition conditions are met, and reports the
   resulting effective selected bet rate.
7. `inspection/run_ml_strategy_blocks.py`:
   generates block artifacts for a walk-forward ML strategy using the active
   model stack (`pancakebot/domain/models/*`), writing
   `../PancakeBot_var_exp/<name>_bXofY_offZ/dislocation_trades.csv`.
8. `inspection/run_flow_backtest_scenario.py`:
   runs the restored simple flow/LGBM family from the active repo with
   explicit `train/val/step` walk-forward controls, `tail_offset_rounds`
   support for rolling-window replay, and backtest-style outputs under
   `../PancakeBot_var_exp/<name>/`.
9. `inspection/run_flow_robustness_sweep.py`:
   runs a chunkable flow-family robustness sweep over rolling recent windows
   and a latest-tail probe window, exporting sorted aggregate tables plus
   per-config JSON detail under `../PancakeBot_var_exp/`.
10. `inspection/run_flow_overlay_offline.py`:
   combines aligned `backtest_trades.csv` series from a primary strategy and a
   flow overlay candidate, then simulates constrained combine rules offline
   (`fallback_only`, `margin_override`, `max_effective_score`) using a flow
   score penalty instead of re-running the full shared pipeline for every
   variant.
11. `inspection/run_profile_window_selector.py`:
   runs rolling-window profile comparisons between the contained `stageB`
   runtime and a configured `flow` variant, then evaluates simple causal
   selectors such as `prev_winner` and `trailing_delta` over those windows.
   It also supports explicit `skip`-aware selector modes and exports the
   effective selected bet rate so controller studies can enforce the current
   practical floor.
   This is the current preferred tool for the "stageB vs flow Bear as alternate
   short-window profiles" research lane.
12. `inspection/run_profile_set_window_selector.py`:
   expands the same profile-window controller idea to a small set of alternate
   profiles, currently `stageB` plus multiple nearby `flow Bear` variants. It
   reuses the canonical stageB/flow inspection runners per window, then
   evaluates skip-aware causal controllers over the resulting profile set.
13. `inspection/run_profile_set_model_selector.py`:
   consumes a profile-set compare CSV, builds past-only window features, and
   evaluates baseline-relative model controllers (`delta_ridge`,
   `delta_logistic`) with explicit `skip`, selected-bet-rate accounting, and
   optional minimum window holds. This is now the preferred next step after a
   profile-set compare run when heuristic controllers plateau.
14. `inspection/run_dry_cycle_monitor.py`:
   tails `var/runtime/dry_cycle_audit.csv`, writes periodic JSON summaries,
   and flags obvious anomalies during long dry-mode runs.
15. `inspection/run_backtest_cache_perf.py`:
   one-command cache harness that runs `cold -> warm` for `continuous` and
   `chunk_reset` backtests and prints timing deltas with cache miss/hit flags.
16. `inspection/run_backtest_warm_matrix.py`:
   warm-cache matrix runner that prints and exports a consolidated table with:
   mode, reset interval, net profit, profit per 500 rounds, max drawdown,
   num bets, and top skip reasons.
17. `inspection/run_backtest_router_matrix.py`:
   router sweep runner over `selector_max_score` and/or `online_cellmean`
   knobs, exporting a sorted table with profitability, drawdown, bet count,
   skip reasons, selected-strategy mix, and warm-run time.
18. `inspection/run_final_model_gate_window_sweep.py`:
   long-window gate/profile sweep with resume support and optional
   multiprocessing (`--max-workers`) for independent runs.
19. `inspection/cleanup_experiment_artifacts.py`:
   retention/cleanup helper for state-cache files, failed-run directories, and
   optional SQLite `VACUUM` on cache/registry DBs.

Quick usage (do not execute automatically in agent workflows):

```powershell
.\.venv\Scripts\python.exe -m inspection.build_strategy_router_dataset `
  --name-prefix x80_plusbad `
  --strategy-prefixes "disloc_best_20260227_x80,disloc_altA_20260227_x80,disloc_altB_20260227_x80,disloc_cons_20260227_x80,disloc_stageG2_r37_x80,disloc_stageH_sidenowcast_when_market_disagree_perfflip_w80_h40_wr0p5_mnm0p001_x80,disloc_stageB_side_adaptive_shadow_ev0p146_skip_w80_h40_wr0p52_mn0p0_x80" `
  --block-size 500 `
  --num-blocks 80

.\.venv\Scripts\python.exe -m inspection.run_strategy_router_probe `
  --name-prefix x80_plusbad_expected_net `
  --dataset-csv ../PancakeBot_var_exp/x80_plusbad_router_dataset.csv `
  --dataset-meta ../PancakeBot_var_exp/x80_plusbad_router_dataset_meta.json `
  --router-mode expected_net_max `
  --expected-net-threshold-bnb 0.0

.\.venv\Scripts\python.exe -m inspection.build_meta_strategy_dataset `
  --name-prefix meta_x20 `
  --strategy-prefixes "disloc_best_20260227,disloc_altA_20260227,disloc_altB_20260227" `
  --block-size 500 `
  --num-blocks 20 `
  --lookback-blocks 5

.\.venv\Scripts\python.exe -m inspection.build_meta_strategy_dataset `
  --name-prefix meta_x80_plusbase `
  --strategy-prefixes "disloc_best_20260227_x80,disloc_altA_20260227_x80,disloc_altB_20260227_x80,disloc_cons_20260227_x80,disloc_stageG2_r37_x80,disloc_stageH_sidenowcast_when_market_disagree_perfflip_w80_h40_wr0p5_mnm0p001_x80,disloc_stageB_side_adaptive_shadow_ev0p146_skip_w80_h40_wr0p52_mn0p0_x80" `
  --extra-series "baseline_modelgate_selector_p0p5_gas1=../PancakeBot_var_exp/fullhistory_modelgate_selector_p0p5_gas1_20260313_off0_sim50984/backtest_trades.csv" `
  --block-size 500 `
  --num-blocks 80 `
  --lookback-blocks 10

.\.venv\Scripts\python.exe -m inspection.run_meta_strategy_probe `
  --name-prefix meta_x20_trailing `
  --dataset-csv ../PancakeBot_var_exp/meta_x20_meta_strategy_dataset.csv `
  --dataset-meta ../PancakeBot_var_exp/meta_x20_meta_strategy_dataset_meta.json `
  --selector-mode trailing_mean `
  --trailing-history-blocks 5 `
  --safety-margin-bnb 0.0 `
  --write-decisions

.\.venv\Scripts\python.exe -m inspection.run_meta_strategy_probe `
  --name-prefix meta_x80_plusbase_fixedbaseline `
  --dataset-csv ../PancakeBot_var_exp/meta_x80_plusbase_meta_strategy_dataset.csv `
  --dataset-meta ../PancakeBot_var_exp/meta_x80_plusbase_meta_strategy_dataset_meta.json `
  --selector-mode fixed_strategy `
  --fixed-strategy-name baseline_modelgate_selector_p0p5_gas1

.\.venv\Scripts\python.exe -m inspection.run_meta_strategy_probe `
  --name-prefix meta_x80_plusbase_dridge `
  --dataset-csv ../PancakeBot_var_exp/meta_x80_plusbase_meta_strategy_dataset.csv `
  --dataset-meta ../PancakeBot_var_exp/meta_x80_plusbase_meta_strategy_dataset_meta.json `
  --selector-mode delta_ridge `
  --baseline-strategy-name baseline_modelgate_selector_p0p5_gas1 `
  --ridge-alpha 1.0 `
  --min-train-rows 12 `
  --safety-margin-bnb 0.02

.\.venv\Scripts\python.exe -m inspection.run_meta_strategy_matrix `
  --name-prefix meta_x80_plusbase_search `
  --dataset-csv ../PancakeBot_var_exp/meta_x80_plusbase_meta_strategy_dataset.csv `
  --dataset-meta ../PancakeBot_var_exp/meta_x80_plusbase_meta_strategy_dataset_meta.json `
  --baseline-strategy-name baseline_modelgate_selector_p0p5_gas1 `
  --strategy-group "core6=baseline_modelgate_selector_p0p5_gas1,disloc_altA_20260227_x80,disloc_altB_20260227_x80,disloc_cons_20260227_x80,disloc_stageB_side_adaptive_shadow_ev0p146_skip_w80_h40_wr0p52_mn0p0_x80,disloc_best_20260227_x80" `
  --chunk-size 250 `
  --chunk-index 0

.\.venv\Scripts\python.exe -m inspection.run_meta_strategy_matrix `
  --name-prefix meta_x80_plusbase_search `
  --dataset-csv ../PancakeBot_var_exp/meta_x80_plusbase_meta_strategy_dataset.csv `
  --dataset-meta ../PancakeBot_var_exp/meta_x80_plusbase_meta_strategy_dataset_meta.json `
  --baseline-strategy-name baseline_modelgate_selector_p0p5_gas1 `
  --strategy-group "core6=baseline_modelgate_selector_p0p5_gas1,disloc_altA_20260227_x80,disloc_altB_20260227_x80,disloc_cons_20260227_x80,disloc_stageB_side_adaptive_shadow_ev0p146_skip_w80_h40_wr0p52_mn0p0_x80,disloc_best_20260227_x80" `
  --shard-count 4 `
  --shard-index 1

.\.venv\Scripts\python.exe -m inspection.run_meta_strategy_followthrough_probe `
  --name-prefix meta_follow_stageg2opp `
  --dataset-csv ../PancakeBot_var_exp/meta_frame_d200_h200_l25_plusopp9_fx_20260324_meta_strategy_dataset.csv `
  --dataset-meta ../PancakeBot_var_exp/meta_frame_d200_h200_l25_plusopp9_fx_20260324_meta_strategy_dataset_meta.json `
  --baseline-strategy-name baseline_modelgate_selector_p0p5_gas1 `
  --active-strategy-names disloc_stageG2_r37_x80_opp `
  --score-mode followthrough `
  --required-streak-len 2 `
  --min-transition-prob 0.25 `
  --min-hold-blocks 4 `
  --min-train-rows 12 `
  --write-decisions

.\.venv\Scripts\python.exe -m inspection.run_flow_backtest_scenario `
  --name flow_recent30k_t15k_v1k_s1k `
  --sim-size 30000 `
  --tail-offset-rounds 0 `
  --train-size 15000 `
  --val-size 1000 `
  --step-size 1000 `
  --ev-threshold 0.005 `
  --kelly-fraction 0.1 `
  --min-total-pool-c 2.0 `
  --min-bet-size 0.05

.\.venv\Scripts\python.exe -m inspection.run_flow_robustness_sweep `
  --name-prefix flow_robust_stage1 `
  --source-sim-size 30000 `
  --tail-offset-rounds "0,5000,10000,15000,20000,25000" `
  --probe-source-sim-size 18000 `
  --probe-tail-offset-rounds "0" `
  --train-sizes "12000,15000" `
  --val-sizes "1000" `
  --step-sizes "1000" `
  --ev-thresholds "0.0025,0.005" `
  --min-total-pool-cs "1.0,1.2"

.\.venv\Scripts\python.exe -m inspection.run_flow_overlay_offline `
  --name overlay_fallback_pen0p10_latest216 `
  --primary-trades ../PancakeBot_var_exp/stageb_only_tail20k_recent216_20260327/backtest_trades.csv `
  --overlay-trades ../PancakeBot_var_exp/flow_bear_train15k_eval216_cfgB_tail20k_20260327/backtest_trades.csv `
  --mode fallback_only `
  --overlay-score-penalty-bnb 0.10

.\.venv\Scripts\python.exe -m inspection.run_profile_window_selector `
  --config config.toml `
  --name-prefix profilewin216_stageb_flowbear `
  --window-size-rounds 216 `
  --num-windows 12 `
  --source-tail-rounds 20000 `
  --flow-train-size 15000 `
  --flow-ev-threshold 0.006 `
  --flow-min-total-pool-c 1.2 `
  --flow-allowed-sides bear_only `
  --selector-lookbacks 1,2,3,4,5 `
  --selector-margins-per-500=-0.2,0.0,0.2,0.5 `
  --selector-skip-thresholds-per-500=0.0,0.05,0.1 `
  --min-selected-bet-rate 0.05

.\.venv\Scripts\python.exe -m inspection.run_profile_set_window_selector `
  --config config.toml `
  --name-prefix profileset216_stageb_flowbear4 `
  --window-size-rounds 216 `
  --num-windows 20 `
  --source-tail-rounds 30000 `
  --flow-profile "name=flow_bear_base,train_size=15000,ev_threshold=0.006,min_total_pool_c=1.2,allowed_sides=bear_only,bull_roll_edge_min=0.0,bear_roll_edge_min=0.0,bull_roll_winrate_min=0.5,bear_roll_winrate_min=0.5,bull_cooldown_trades=80,bear_cooldown_trades=80" `
  --flow-profile "name=flow_bear_loose12,train_size=15000,ev_threshold=0.005,min_total_pool_c=1.2,allowed_sides=bear_only,bull_roll_edge_min=0.0,bear_roll_edge_min=-0.002,bull_roll_winrate_min=0.5,bear_roll_winrate_min=0.47,bull_cooldown_trades=80,bear_cooldown_trades=120" `
  --selector-lookbacks 1,2,3,4,5 `
  --selector-margins-per-500=-0.2,0.0,0.2,0.5 `
  --selector-skip-thresholds-per-500=0.0,0.05,0.1 `
  --min-selected-bet-rate 0.05

.\.venv\Scripts\python.exe -m inspection.run_profile_set_model_selector `
  --compare-csv ../PancakeBot_var_exp/profileset216_stageb_stageg2_flowbear4_20260328_profile_set_window_compare.csv `
  --name-prefix profileset216_stageb_stageg2_flowbear4_model `
  --feature-lookbacks 1,3,5,8 `
  --min-train-windows 6,8,10,12 `
  --min-hold-windows 1,2,3,4 `
  --ridge-alphas 0.25,0.5,1.0,2.0,5.0,10.0

.\.venv\Scripts\python.exe -m inspection.run_dry_cycle_monitor `
  --cycle-audit-csv var/runtime/dry_cycle_audit.csv `
  --output-jsonl ../PancakeBot_var_exp/dry_hybrid_monitor_20260326.jsonl `
  --summary-json ../PancakeBot_var_exp/dry_hybrid_monitor_20260326_summary.json `
  --expected-strategies "disloc_stageB_bullonly_recent8pct_v1" `
  --expected-bet-sides "Bull" `
  --poll-seconds 60 `
  --warn-idle-streak-cycles 240 `
  --warn-min-cycles-for-rate-check 240 `
  --warn-total-bet-rate-below 0.02 `
  --duration-seconds 43200

.\.venv\Scripts\python.exe -m inspection.run_backtest_cache_perf `
  --name-prefix cache_perf_warm `
  --sim-size-continuous 500 `
  --sim-size-chunk-reset 500 `
  --chunk-reset-every-rounds 20

.\.venv\Scripts\python.exe -m inspection.run_backtest_warm_matrix `
  --name-prefix warm_matrix_500 `
  --sim-size 500 `
  --chunk-reset-intervals "20,40,80"

.\.venv\Scripts\python.exe -m inspection.run_backtest_router_matrix `
  --name-prefix router_matrix_500 `
  --sim-size 500 `
  --reset-mode continuous `
  --router-modes "selector_max_score,online_cellmean"
```
