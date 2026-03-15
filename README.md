# PancakeBot

Spec-driven, invariant-heavy runtime for PancakeSwap Prediction v2.

## Quick start

Current shared baseline:

- `1 gwei` accounting gas
- router: `online_selector_score_fallback`
- active candidates: `disloc_altA_20260227_x80`, `disloc_altB_20260227_x80`
- baseline overlay:
  - `pool_total_gate_mode = projected_final_model_only`
  - `projected_final_pool_total_min_bnb = 0.5`
  - `market_extreme_min = 0.02`
  - `late_model_veto_enabled = true`
  - `late_model_veto_min_late_ratio = 0.05`
  - `late_model_veto_min_abs_imbalance = 0.10`
  - `disloc_altA_20260227_x80.bear_expected_net_extra_min_bnb = 0.01`

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
      | Backtest  | `python run.py --backtest` | simulates dry/live deterministically without sleeping |
      | Dry       | `python run.py --dry`      | simulates bets/claims without broadcasting            |
      | Live      | `python run.py`            | places real on-chain bets                             |

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
            - `runtime/dry_audit_trades.csv`

## Dry smoke

Use the shared `config.toml` baseline directly:

```powershell
.\.venv\Scripts\python.exe run.py --config config.toml --dry
```

Recommended preflight before a long dry run:

```powershell
.\.venv\Scripts\python.exe -m inspection.run_backtest_scenario --name smoke_promoted_baseline --sim-size 200 --reset-mode continuous
```
