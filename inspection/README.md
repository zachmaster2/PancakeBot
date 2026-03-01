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
```
