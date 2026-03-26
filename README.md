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
  - `enabled = true` in the current promoted runtime profile
  - `train_size = 12000`
  - `retrain_interval = 1000`
  - `ev_threshold = 0.0025`
  - `min_total_pool_c = 1.0`
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

`runtime/dry_cycle_audit.csv` is reset at dry startup so each run has a clean
decision log. `runtime/dry_audit_trades.csv` remains the persistent dry
bet/settlement ledger used by dry-state recovery.

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
