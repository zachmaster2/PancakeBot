# PancakeBot

Spec-driven, invariant-heavy runtime for PancakeSwap Prediction v2.

## Quick start

Current shared baseline:

- `1 gwei` accounting gas
- router: `selector_max_score`
- selector/router warmup: `10000`
- active dislocation candidates:
  - `disloc_stageB_bullonly_recent8pct_v1`
- flow candidate:
  - `flow_lgbm_recent_t12k_r1k_regime40_v1`
  - `enabled = false` in the current contained runtime profile
  - retained for re-qualification research, not active in dry/live by default
  - `train_size = 12000`
  - `retrain_interval = 1000`
  - `ev_threshold = 0.0025`
  - `min_total_pool_c = 1.0`
  - `selector_score_penalty_bnb = 0.0`
  - `roll_window = 40`
  - `roll_winrate_min = 0.48`
  - `cooldown_trades = 40`

   1. #### Create a `.env` file at the repo root:

      - `THE_GRAPH_API_KEY` - API key for The Graph gateway (Bearer token)
      - `BSC_WALLET_PRIVATE_KEY` - Wallet private key hex (with or without `0x`)

   2. #### Review `config.toml` key settings:

      - `[strategy.router]`
      - active dislocation candidates under `[strategy.dislocation]`
      - `[runtime]` cutoff and receipt settings
      - `[paths]` runtime-state outputs for dry/live smoke runs
      - `[backtest] simulation_size` (backtest only)

   3. #### Run:

      | mode      | command                    | description                                           |
      |-----------|----------------------------|-------------------------------------------------------|
      | Backtest  | `.\.venv\Scripts\python.exe run.py --backtest` | simulates dry/live deterministically without sleeping |
      | Dry       | `.\.venv\Scripts\python.exe run.py --dry`      | simulates bets/claims without broadcasting            |
      | Live      | `.\.venv\Scripts\python.exe run.py`            | places real on-chain bets                             |
      | Sync only | `.\.venv\Scripts\python.exe run.py --sync-only` | updates closed rounds and kline coverage, then exits  |

      - Outputs (`var/`):
         - Backtest:
            - `backtest_trades.csv`
            - `backtest_summary.json`
         - Dry/Live:
            - rolling closed-round store
            - runtime artifacts/logs
            - `runtime/claim_scan_cursor.txt`
            - `runtime/dry_bets.jsonl`
            - `runtime/dry_settled_epochs.txt`
            - `runtime/dry_audit_trades.csv` for persistent dry bet/settlement audit
            - `runtime/dry_cycle_audit.csv` for per-cycle dry decision audit
            - `runtime/dry_bankroll_state.json`

At dry startup, any existing `var/runtime` dry-state files are first archived to
`../PancakeBot_var_exp/dry_run_archive_*_startup_fresh_reset/`, then the new
run starts fresh. On dry shutdown, the current state is also snapshotted to
`../PancakeBot_var_exp/dry_run_archive_*_shutdown_snapshot/`.

`runtime/dry_cycle_audit.csv` is reset at dry startup so each run has a clean
decision log. It now records both:
- `observed_*` pool fields: the raw open-round snapshot seen by dry mode
- `cutoff_used_*` pool fields: the cutoff-filtered pool actually used by the
  strategy logic

`runtime/dry_audit_trades.csv` remains the per-run dry bet/settlement ledger.

## Dry smoke

Use the shared `config.toml` baseline directly:

```powershell
.\.venv\Scripts\python.exe run.py --config config.toml --dry
```

Recommended preflight before a long dry run:

```powershell
.\.venv\Scripts\python.exe -m inspection.run_runtime_preflight --check-env

.\.venv\Scripts\python.exe -m inspection.run_backtest_scenario --name smoke_promoted_runtime_profile --sim-size 200 --reset-mode continuous
```

If you only need the latest on-disk market inputs for research or inspection,
without starting the dry/live loop, use:

```powershell
.\.venv\Scripts\python.exe run.py --sync-only
```

Experimental controller dry tests should not edit `config.toml` directly.
Use the runbook in [WINDOW_CONTROLLER_DRY_TEST_RUNBOOK.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/WINDOW_CONTROLLER_DRY_TEST_RUNBOOK.md) together with:

```powershell
.\.venv\Scripts\python.exe -m inspection.write_window_controller_runtime_config ...
```

Current best experimental controller dry-test candidate:
- baseline: `disloc_stageB_bullonly_recent8pct_v1`
- alternate: `disloc_cons_20260227_x80`
- mode: `trailing_best_vs_baseline`
- `window_rounds = 216`
- `lookback_windows = 3`
- `margin_per_500 = 1.0`
