# Inspection Tooling

Inspection tooling is split into two domains:

1. `inspection/run_backtest_scenario.py`:
   canonical probe entrypoint that executes the production backtest pipeline.
2. `inspection/legacy/`:
   temporary one-cycle archive for historical probe scripts retained for
   transition traceability.

The canonical probe path must not reimplement production strategy logic.

Additional inspection probes for strategy-routing experiments:

1. `inspection/build_strategy_router_dataset.py`:
   reads block artifacts (`var/exp/*/dislocation_trades.csv`) and builds a
   router-ready dataset with feature columns and labels.
2. `inspection/run_strategy_router_probe.py`:
   runs walk-forward router policy probes against that dataset.
3. `inspection/run_ml_strategy_blocks.py`:
   generates block artifacts for a walk-forward ML strategy using the active
   model stack (`pancakebot/domain/models/*`), writing
   `var/exp/<name>_bXofY_offZ/dislocation_trades.csv`.
4. `inspection/run_backtest_cache_perf.py`:
   one-command cache harness that runs `cold -> warm` for `continuous` and
   `chunk_reset` backtests and prints timing deltas with cache miss/hit flags.
5. `inspection/run_backtest_warm_matrix.py`:
   warm-cache matrix runner that prints and exports a consolidated table with:
   mode, reset interval, net profit, profit per 500 rounds, max drawdown,
   num bets, and top skip reasons.
6. `inspection/run_backtest_router_matrix.py`:
   router sweep runner over `selector_max_score` and/or `online_cellmean`
   knobs, exporting a sorted table with profitability, drawdown, bet count,
   skip reasons, selected-strategy mix, and warm-run time.
7. `inspection/run_final_model_gate_window_sweep.py`:
   long-window gate/profile sweep with resume support and optional
   multiprocessing (`--max-workers`) for independent runs.
8. `inspection/cleanup_experiment_artifacts.py`:
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
  --dataset-csv var/exp/x80_plusbad_router_dataset.csv `
  --dataset-meta var/exp/x80_plusbad_router_dataset_meta.json `
  --router-mode expected_net_max `
  --expected-net-threshold-bnb 0.0

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
