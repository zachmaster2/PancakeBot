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
        rpc_poller.py                    # Era 11 (2026-05-07+): batched RPC pool event accumulation

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
3. **Tighten Windows Time Service** — see [W32Time prerequisite](#w32time-prerequisite-dry--live-modes) below
4. Run `python run.py --sync` to fetch data
5. Run `python run.py --backtest` to verify
6. Run `python run.py --dry` for paper trading

### W32Time prerequisite (dry + live modes)

The critical-path scheduling relies on the local OS clock being within
a few milliseconds of true NTP. Windows Time Service (W32Time) defaults
to a max poll interval of 1024s (~17 min), which lets the local clock
drift up to ~270ms (P95) between syncs — too sloppy for the bot's
sub-second timing budgets.

Tighten the service to a 32s max poll interval. Run in an **elevated
PowerShell** (Run as Administrator):

```
reg add "HKLM\SYSTEM\CurrentControlSet\Services\W32Time\Config" /v MaxPollInterval /t REG_DWORD /d 5 /f
w32tm /config /update
net stop w32time
net start w32time
```

Verify with `w32tm /query /status` — the output should show
`Poll Interval: 5 (32s)` (the 5 is `log2(32)`).

The change persists across reboots. Backtest and sync modes do not
require this (they are not timing-critical).

**The bot has no application-level NTP layer.** The prior `NtpSync`
(per-round NTP query that measured `local − ntp` and applied the
offset inside `_utc_now()`) was retired in Bundle 5 v2 (2026-05-14)
alongside the W32Time tightening. The OS clock — kept tight by
W32Time — is now the sole source of truth. If you skip the W32Time
step, the bot's sub-second bet-timing margins may be too tight; there
is no in-app fallback.

## Architecture

See [docs/architecture.html](docs/architecture.html) for the visual diagram.

### Per-round runtime loop

Three pre-lock wakes anchored at the chain-supplied `lock_at` timestamp,
plus a post-close claim wake anchored at `close_at(prev_locked_epoch)`.
The bankroll wake uses a deliberately generous (5 s) gap above the
critical path — robustness over micro-optimization for non-critical-path
operations. The critical path wake is a single scheduled event whose
budget covers a sequential pool snapshot → kline fetch → signal compute
→ bet submit. All ms offsets DERIVED from constants in
`pancakebot/timing_constants.py`. Static-fallback values shown; the
live dynamic-mode critical-path wake is recomputed per-round from the
predicted predecessor block (Bundle 5 v2, 2026-05-14).

| Wake | Anchor + offset (static fallback) | Activities |
|---|---|---|
| `wait_for_okx_warmup` | `lock_at - 7000ms` | Pre-establish OKX TLS connections by calling `MomentumGate.warmup_okx_session()` (proxies to `OkxClient.warmup`). Idempotent when sockets are already warm. Added 2026-05-21 (commit `5c496b1`) after a BSC RPC outage left the OKX session cold and the recovery round paid 500-800ms TLS handshakes on the critical-path kline fetch |
| `wait_for_bankroll` | `lock_at - 5970ms` | Refresh wallet balance: live mode = BSC RPC; dry mode = in-memory simulated bankroll. Feeds the risk gates and `decide_open_round` with fresh-truth |
| Anchor poll | `lock_at - 1300ms` | Single sub-second poll of chain head's BEP-520-encoded ms timestamp; drives dynamic critical-path scheduling |
| `wait_for_critical_path` | `lock_at - 970ms` | Single critical-path entry. Sequentially: pool snapshot from RPC poller (Era 11; `pool_cutoff_seconds = 6` data horizon) → 3 parallel OKX `/history-candles` GETs (BTC/ETH/SOL) → signal compute → bet submit |
| Pre-bet timing guard | `lock_at - 625ms` | Abort if decision-ready past the safety margin (TX would mine after lock). Before submission, `assert_gas_cap_not_breached()` reads `eth.gas_price` and raises `GasPriceCapBreachedError` if it exceeds `MAX_GAS_PRICE_WEI` (1 Gwei). Bets are posted at `MAX_GAS_PRICE_WEI` deterministically; on breach the bot SKIPs the round + fires a CRITICAL Discord alert |
| `wait_for_claim` | `close_at(prev_locked) + buffer_seconds + 5s` (≈ 35s post-close) | Sleep for previous round's settlement; claim winnings (live; receipt-waited with `claim_tx_receipt_timeout_seconds ≈ 10s`, revert/timeout fires Discord `CLAIM FAILED` alert) |

### Configurable knobs (`config.toml [runtime]`)

- `kline_cutoff_seconds = 2` — strategy data horizon for OKX klines
  (FIXED by strategy: the kline closing at `lock - cutoff` is required).
  Tolerance for OKX publish-delay tails is via the
  `max_consecutive_kline_fetch_failures` streak counter at runtime, not
  via a config-load budget. Empirical OKX publish-delay distribution:
  P95 ≈ 700ms, P99 ≈ 1300ms (informational; see
  `pancakebot/timing_constants.py`).
- `pool_cutoff_seconds = 6` — pool-aggregate data horizon for BSC events.
  Cross-validated at load against the RPC poll schedule (final-poll
  offset must accommodate batch RTT p99 + safety buffer before the
  critical-path wake reads the pool snapshot).
- `max_consecutive_kline_fetch_failures = 5` — streak counter before bot
  crashes (supervisor restart + Discord alert).

### Empirical constants

Maintained in `pancakebot/timing_constants.py` with per-constant
provenance: probe script, measurement date, percentile, sample size.
Re-deriving any value requires re-running the corresponding probe in
`research/p4c_*_probe.py` and updating the constant co-locked with a
new measurement date.
