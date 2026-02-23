# PancakeBot

Spec-driven, invariant-heavy runtime for PancakeSwap Prediction v2.

## Quick start

   1. #### Create a `.env` file at the repo root:

      - `THE_GRAPH_API_KEY` - API key for The Graph gateway (Bearer token)
      - `BSC_WALLET_PRIVATE_KEY` - Wallet private key hex (with or without `0x`)

   2. #### Review `config.toml` key settings:

      - `train_size`
      - `retrain_interval`
      - `simulation_size` (backtest only)
      - policy thresholds

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
