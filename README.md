# PancakeBot

Automated trading bot for PancakeSwap Prediction V2 on BNB Smart Chain.

## Modes

| Mode | Command | Env vars | Description |
|------|---------|----------|-------------|
| Sync | `python run.py --sync` | `THE_GRAPH_API_KEY` | Fetch rounds + klines + contract constants |
| Backtest | `python run.py --backtest` | (none) | Replay historical data, compute PnL |
| Dry | `python run.py --dry` | (none) | Real-time paper trading |
| Live | `python run.py --live` | `BSC_WALLET_PRIVATE_KEY` | Real on-chain bets |

Running with no flags prints help. Modes are mutually exclusive.

## Strategy

**Signal:** BTC multi-timeframe momentum -- 3s, 7s, and 15s returns must all agree in direction with `min(|return|) >= threshold` (pool-adaptive: 0.0002 small / 0.0001 large).

**Regime-2:** When BTC is silent, ETH + SOL multi-TF agreement fires as a secondary signal with smaller sizing.

**Sizing:** Continuous adaptive based on signal strength, ETH/SOL confirmation, and payout odds.

**Filters:** Pool minimum (1.5 BNB), payout floor (1.5x), strong-signal bypass for small pools.

## Project Structure

```
pancakebot/
    constants.py, errors.py, log.py     # Shared foundations
    money.py, time.py, path.py
    config.py                            # All config: TOML, env, dataclasses
    types.py, pool_amounts.py            # Domain types (Bet, Round)
    settlement.py                        # PnL computation
    app.py                               # Mode dispatch

    strategy/                            # Signal + sizing
        momentum_gate.py                 # OKX kline fetch + BTC multi-TF signal
        momentum_pipeline.py             # Signal -> sizing -> filters -> decision

    chain/                               # BSC chain interaction
        prediction_contract.py           # Web3 contract wrapper
        contract_config.py, rpc_pool.py
        pool_watcher.py                  # WSS real-time pool tracking

    market_data/                         # Data fetch + store
        okx_client.py                    # OKX REST with session pooling
        graph_client.py                  # The Graph API
        round_store.py, round_sync.py    # Closed rounds JSONL
        kline_store.py                   # 1s kline JSONL
        contract_constants.py            # Chain constants cache
        sync.py                          # --sync orchestration

    runtime/                             # Real-time loop (dry + live)
        config.py                        # RuntimeConfig
        engine.py                        # Two-phase loop, epoch handshake
        dry.py                           # Dry state, audit, settlement
        live.py                          # Claim scanning

    backtest/
        runner.py                        # Offline replay + equity plot
```

## Output

```
var/
    closed_rounds.jsonl                  # Synced round history
    {bnb,btc,eth,sol}_spot_prices.jsonl  # Synced 1s klines
    contract_constants.json              # Chain constants (from --sync)
    dry/                                 # Dry mode state (archived on restart)
    live/                                # Live mode state
    backtest/                            # Backtest results + equity plot
```

## Setup

1. Create `.env` with required env vars (see mode table above)
2. Review `config.toml`
3. Run `python run.py --sync` to fetch data
4. Run `python run.py --backtest` to verify
5. Run `python run.py --dry` for paper trading

## Architecture

See [docs/architecture.html](docs/architecture.html) for the visual diagram.

**Two-phase runtime loop:**
- Phase A (lock_at - 6s): Epoch check + OKX TLS warmup + pool data
- Phase B (lock_at - 1.75s): Fetch klines (~285ms) + signal + bet
- Safety margin: 1s before lock
