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
plus a post-close claim wake anchored at `close_at(prev_locked_epoch)`.
The ntp-sync and bankroll wakes use deliberately generous (5 s) gaps
above the critical path — robustness over micro-optimization for
non-critical-path operations. The critical path wake is a single
scheduled event whose budget covers a sequential pool snapshot →
kline fetch → signal compute → bet submit. All ms offsets DERIVED
from constants in `pancakebot/timing_constants.py`.

| Wake | Anchor + offset | Activities |
|---|---|---|
| `wait_for_ntp_sync` | `lock_at - 11095ms` | Force a fresh NTP query; apply (local − ntp) offset to `_utc_now()` for the rest of the round so critical-path scheduling uses freshly-corrected clock |
| `wait_for_bankroll` | `lock_at - 6095ms` | Refresh wallet balance: live mode = BSC RPC; dry mode = in-memory simulated bankroll. Feeds the risk gates and `decide_open_round` with fresh-truth |
| `wait_for_critical_path` | `lock_at - 1095ms` | Single critical-path entry. Sequentially: pool snapshot from WSS (`pool_cutoff_seconds = 6` data horizon) → 3 parallel OKX `/history-candles` GETs (BTC/ETH/SOL) → signal compute → bet submit |
| Pre-bet timing guard | `lock_at - 750ms` | Abort if decision-ready past the safety margin (TX would mine after lock) |
| `wait_for_claim` | `close_at(prev_locked) + buffer_seconds + 5s` (≈ 35s post-close) | Sleep for previous round's settlement; claim winnings (live; receipt-waited with `claim_tx_receipt_timeout_seconds ≈ 35s`, revert/timeout fires Discord `CLAIM FAILED` alert) |

### Configurable knobs (`config.toml [runtime]`)

- `kline_cutoff_seconds = 2` — strategy data horizon for OKX klines
  (FIXED by strategy: the kline closing at `lock - cutoff` is required).
  Cross-validated at load via the wake-offset framing:
  `kline_fetch_wakeup_offset_ms <= kline_cutoff_seconds * 1000 -
  OKX_KLINE_PUBLISH_DELAY_P95_MS`.
- `pool_cutoff_seconds = 6` — pool-aggregate data horizon for BSC events.
  Cross-validated at load:
  `pool_read_wakeup_offset_ms <= pool_cutoff_seconds * 1000 -
  WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS`.
- `max_consecutive_fetch_failures = 5` — streak counter before bot
  crashes (supervisor restart + Discord alert).

### Empirical constants

Maintained in `pancakebot/timing_constants.py` with per-constant
provenance: probe script, measurement date, percentile, sample size.
Re-deriving any value requires re-running the corresponding probe in
`research/p4c_*_probe.py` and updating the constant co-locked with a
new measurement date.
