# Window Controller Dry Test Runbook

## Status

This runbook is now for the default peer-set controller dry lane.

The default runtime controller lane is an absolute peer-set controller with:

- action set: `stageB`, `stageG2`, `altB`, or `skip`
- mode: `absolute_best_with_skip`
- profile set: `disloc_stageB_bullonly_recent8pct_v1`, `disloc_stageG2_bullonly_recent5pct_v1`, `disloc_altB_20260227_x80`
- cold start: `disloc_stageB_bullonly_recent8pct_v1` bootstrap seed only
- window rounds: `216`
- lookback windows: `2`
- min history windows: `2`
- estimator: `ewm_mean`
- `ewm_alpha = 0.85`
- stability penalty: `0.0`
- skip threshold: `0.05 / 500`

This is the default runtime lane because the operator goal is now multi-profile
dry exploration with no privileged profile. It should not be described as a
baseline-centered or one-profile controller.

## Qualification Snapshot

Latest synced shared-harness qualification on epoch `468540` is favorable:

- latest `20` offsets, `6480` rounds:
  controller mean about `+0.1054 / 500`
  static `stageB` mean about `+0.0572 / 500`
  mean lift about `+0.0482 / 500`
  controller mean bet rate about `6.62%`
- latest `20` offsets, `8640` rounds:
  controller mean about `+0.1126 / 500`
  static `stageB` mean about `+0.0322 / 500`
  mean lift about `+0.0804 / 500`
  controller mean bet rate about `5.76%`
- latest `20` offsets, `10800` rounds:
  controller mean about `+0.1100 / 500`
  static `stageB` mean about `+0.0821 / 500`
  mean lift about `+0.0279 / 500`
  controller mean bet rate about `5.52%`

Interpretation:

- lift here means `controller_per_500 - static_stageB_per_500` on the same
  shared-harness slice
- this branch is not claimed as the final best controller design
- it is ready enough for an extended real dry run under the current operator
  preference to promote the next credible path instead of waiting for perfect
  offline dominance

The latest fresh offline rerun for this exact peer-set was negative on the
tested recent horizons, so this lane is being used as an explicit runtime
exploration choice, not because it is the strongest offline-qualified path.

## Qualification Command

Refresh data first, but only when no dry/live PancakeBot process is running:

```powershell
.\.venv\Scripts\python.exe run.py --sync-only
```

Then re-run the integrated shared-harness qualification:

```powershell
.\.venv\Scripts\python.exe -m inspection.run_window_controller_shared_eval `
  --name-prefix stageb_only_abs_wc_latest20_sync468540_a085_20260331 `
  --controller-mode absolute_best_with_skip `
  --controller-profile-names disloc_stageB_bullonly_recent8pct_v1 `
  --controller-cold-start-profile-name disloc_stageB_bullonly_recent8pct_v1 `
  --static-profile-name disloc_stageB_bullonly_recent8pct_v1 `
  --lookback-windows 2 `
  --min-history-windows 2 `
  --estimator-mode ewm_mean `
  --ewm-alpha 0.85 `
  --stability-penalty-per-500 0.0 `
  --skip-threshold-per-500 0.05 `
  --tail-offset-rounds 0,216,432,648,864,1080,1296,1512,1728,1944,2160,2376,2592,2808,3024,3240,3456,3672,3888,4104 `
  --sim-sizes 6480,8640,10800
```

Primary artifacts:

- `stageb_only_abs_wc_latest20_sync468540_a085_20260331_window_controller_shared_eval.csv`
- `stageb_only_abs_wc_latest20_sync468540_a085_20260331_window_controller_shared_eval_summary.json`

## Materialize Dry Config

Write a dedicated controller dry config under `../PancakeBot_var_exp/`:

```powershell
.\.venv\Scripts\python.exe -m inspection.write_window_controller_runtime_config `
  --name-prefix stageb_only_abs_wc_dry_a085_20260331 `
  --active-candidate-names disloc_stageB_bullonly_recent8pct_v1 `
  --window-controller-mode absolute_best_with_skip `
  --window-controller-profile-names disloc_stageB_bullonly_recent8pct_v1 `
  --window-controller-cold-start-profile-name disloc_stageB_bullonly_recent8pct_v1 `
  --window-controller-lookback-windows 2 `
  --window-controller-min-history-windows 2 `
  --window-controller-estimator-mode ewm_mean `
  --window-controller-ewm-alpha 0.85 `
  --window-controller-stability-penalty-per-500 0.0 `
  --window-controller-skip-threshold-per-500 0.05
```

## Start Dry

Clear runtime state first, then run foreground dry mode with the generated
config:

```powershell
cmd /c del /f /q var\\runtime\\* 2>nul

.\.venv\Scripts\python.exe run.py `
  --config ..\\PancakeBot_var_exp\\stageb_only_abs_wc_dry_a085_20260331_window_controller_runtime.toml `
  --dry
```

The repo base config now points at this peer-set controller by default, so
plain `.\.venv\Scripts\python.exe run.py --dry` uses the multi-profile
absolute controller unless you override `config.toml`.

## Monitor

Use the dry-cycle monitor in a second terminal if you want file-backed
summaries during the run:

```powershell
.\.venv\Scripts\python.exe -m inspection.run_dry_cycle_monitor `
  --output-jsonl ..\\PancakeBot_var_exp\\stageb_only_abs_wc_dry_a085_20260331_monitor.jsonl `
  --summary-json ..\\PancakeBot_var_exp\\stageb_only_abs_wc_dry_a085_20260331_monitor_summary.json `
  --expected-strategies disloc_stageB_bullonly_recent8pct_v1 `
  --expected-bet-sides Bull `
  --expected-controller-profiles disloc_stageB_bullonly_recent8pct_v1 `
  --expected-controller-actions profile,skip `
  --poll-seconds 60
```

What to watch during dry:

- `controller_selected_action` should stay within `profile` or `skip`
- `controller_selected_profile` should stay on `disloc_stageB_bullonly_recent8pct_v1`
- long idle streaks are expected to be somewhat longer than static `stageB`
- the run should be judged over days, not short local streaks

## Historical Note

Older `stageB vs cons` and `stageB vs stageG2` runbooks/specs were useful
intermediate experiments, but they are not the current controller target.

The intended end state remains:

- multi-profile absolute local estimation
- `skip` as a first-class action
- no structurally privileged baseline controller
