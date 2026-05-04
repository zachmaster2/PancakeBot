# PancakeBot

Automated trading bot for PancakeSwap Prediction V2 on BNB Smart Chain.

## Modes

| Mode     | Command                    | Env vars                   | Description                                   |
|----------|----------------------------|----------------------------|-----------------------------------------------|
| Sync     | `python run.py --sync`     | `THE_GRAPH_API_KEY`        | Fetch rounds + klines + contract constants    |
| Backtest | `python run.py --backtest` | (none)                     | Replay historical data, compute PnL           |
| Dry      | `python run.py --dry`      | (none)                     | Real-time paper trading                       |
| Live     | `python run.py --live`     | `BSC_WALLET_PRIVATE_KEY`   | Real on-chain bets                            |

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

### Per-round runtime loop

Three pre-lock wakes anchored at the chain-supplied `lock_at` timestamp,
plus a post-lock claim wake. All ms offsets DERIVED from empirical
constants in `pancakebot/timing_constants.py`.

| Wake | Offset from `lock_at` | Activities |
|---|---|---|
| `wait_for_skew_sync` | `lock_at - 3645ms` | OKX clock-skew refresh, epoch check, bankroll fetch (live), settlement record (live) |
| `wait_for_pool` | `lock_at - 1095ms` | Read pool aggregate from WSS subscriber (`pool_cutoff_seconds = 6` data horizon) |
| `wait_for_kline_fetch` | `lock_at - 1090ms` | 4 parallel OKX `/history-candles` GETs + signal compute |
| Pre-bet timing guard | `lock_at - 750ms` | Abort if decision-ready past the safety margin (TX would mine after lock) |
| `wait_for_claim` | `close_at(prev_locked) + 35s` | Sleep for previous round's settlement; claim winnings (live; receipt-waited with `claim_tx_receipt_timeout_seconds ≈ 35s`, revert/timeout fires Discord `CLAIM FAILED` alert) |

### Configurable knobs (`config.toml [runtime]`)

- `kline_cutoff_seconds = 3` — strategy data horizon for OKX klines.
  Cross-validated at load: must be ≥
  `(OKX_KLINE_PUBLISH_DELAY_P99_MS + kline_fetch_wakeup_offset_ms) / 1000`.
- `pool_cutoff_seconds = 6` — pool-aggregate data horizon for BSC events.
  Cross-validated at load: must be ≥
  `(pool_read_wakeup_offset_ms + WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS) / 1000`.
- `max_consecutive_fetch_failures = 5` — streak counter before bot
  crashes (supervisor restart + Discord alert).

### Empirical constants

Maintained in `pancakebot/timing_constants.py` with per-constant
provenance: probe script, measurement date, percentile, sample size.
Re-deriving any value requires re-running the corresponding probe in
`research/p4c_*_probe.py` and updating the constant co-locked with a
new measurement date.
