# Window Controller Dry Test Runbook

## Purpose

Prepare and run a controller-driven dry test without hand-editing
`config.toml`.

This is only for the experimental `stageB` vs `cons` window-controller
lane. Runtime should remain contained on static `stageB` unless the current
qualification evidence explicitly justifies the controller dry test.

## Current Candidate

Current best continuous shared-harness candidate:

- mode: `trailing_best_vs_baseline`
- baseline: `disloc_stageB_bullonly_recent8pct_v1`
- alternate: `disloc_cons_20260227_x80`
- window rounds: `216`
- lookback windows: `3`
- margin: `1.0 / 500`
- skip threshold: `0.0`

This candidate is still experimental. It is not yet live-safe.

## Qualification First

Before starting a controller dry test, refresh the shared-harness qualification:

```powershell
.\.venv\Scripts\python.exe run.py --sync-only

.\.venv\Scripts\python.exe -m inspection.run_window_controller_shared_eval `
  --name-prefix stageb_cons_wc_l3m1_pre_dry `
  --controller-mode trailing_best_vs_baseline `
  --baseline-profile-name disloc_stageB_bullonly_recent8pct_v1 `
  --alternate-profile-name disloc_cons_20260227_x80 `
  --lookback-windows 3 `
  --margin-per-500 1.0 `
  --tail-offset-rounds 0,216,432,648,864,1080,1296,1512,1728,1944 `
  --sim-sizes 6480,8640,10800
```

Primary artifacts:

- `..._window_controller_shared_eval.csv`
- `..._window_controller_shared_eval_summary.json`

If the refreshed shared-harness evidence is clearly worse than static
`stageB`, do not continue to the controller dry test.

## Materialize Dry Config

Write a dedicated runtime config under `../PancakeBot_var_exp/`:

```powershell
.\.venv\Scripts\python.exe -m inspection.write_window_controller_runtime_config `
  --name-prefix stageb_cons_wc_dry_l3m1 `
  --active-candidate-names disloc_stageB_bullonly_recent8pct_v1,disloc_cons_20260227_x80 `
  --window-controller-mode trailing_best_vs_baseline `
  --window-controller-baseline-profile-name disloc_stageB_bullonly_recent8pct_v1 `
  --window-controller-alternate-profile-name disloc_cons_20260227_x80 `
  --window-controller-lookback-windows 3 `
  --window-controller-margin-per-500 1.0
```

This prints the generated config path, for example:

`C:\Users\zking\Documents\GitHub\PancakeBot_var_exp\stageb_cons_wc_dry_l3m1_window_controller_runtime.toml`

## Start Dry

Clear runtime state first, then run dry with the generated config:

```powershell
cmd /c del /f /q var\runtime\* 2>nul

.\.venv\Scripts\python.exe run.py `
  --config ..\PancakeBot_var_exp\stageb_cons_wc_dry_l3m1_window_controller_runtime.toml `
  --dry
```

## Monitor

Use the dry-cycle monitor with both controller actions and runtime-selected
strategies allowlisted:

```powershell
.\.venv\Scripts\python.exe -m inspection.run_dry_cycle_monitor `
  --output-jsonl ..\PancakeBot_var_exp\stageb_cons_wc_dry_monitor.jsonl `
  --summary-json ..\PancakeBot_var_exp\stageb_cons_wc_dry_monitor_summary.json `
  --expected-strategies disloc_stageB_bullonly_recent8pct_v1,disloc_cons_20260227_x80 `
  --expected-bet-sides Bull `
  --expected-controller-profiles disloc_stageB_bullonly_recent8pct_v1,disloc_cons_20260227_x80 `
  --expected-controller-actions profile `
  --poll-seconds 60
```

Relevant runtime artifacts:

- `var/runtime/dry_cycle_audit.csv`
- `var/runtime/dry_audit_trades.csv`
- `var/runtime/dry_bankroll_state.json`

Controller-specific audit columns:

- `controller_mode`
- `controller_window_index`
- `controller_selected_profile`
- `controller_selected_action`
- `controller_estimated_per_500`
- `controller_estimated_selected_bet_rate`

## What To Check

1. The controller should choose only `stageB` or `cons` on the current no-skip branch.
2. Actual bet sides should remain `Bull`.
3. Controller-selected windows should look plausible against completed-window
   recent evidence.
4. Long zero-bet streaks should be interpreted relative to the controller
   action set, not the old static `stageB` expectations.
5. If behavior looks inconsistent with the shared-harness qualification, stop
   the dry test and investigate before continuing.
