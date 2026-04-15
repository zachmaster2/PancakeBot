# PancakeBot

Automated trading bot for PancakeSwap Prediction V2 on BNB Smart Chain.

## Strategy

**Signal:** BTC multi-timeframe momentum agreement. BTC 3s, 7s, and 15s
returns must all agree in direction with `min(|return|) >= threshold`.

- Pool-adaptive threshold: 0.0002 for pools < 3 BNB, 0.0001 for pools >= 3 BNB
- Signal fires ~3% of rounds (bear-biased: 90% of PnL from Bear signals)

**Regime-2:** When BTC is silent, ETH + SOL multi-TF(3,7,15) agreement fires
as a secondary signal with smaller sizing.

**Sizing:** Continuous adaptive based on signal strength, ETH/SOL confirmation,
and payout odds.

```
effective_strength = btc_min_abs + eth_confirm * 0.3 + sol_confirm * 0.3
frac = min(0.04 + 100 * effective_strength, 0.30)
frac = frac * max(0.5, 1.0 + 1.0 * (payout - 2.0))
bet = max(0.01, min(2.0, pool * frac))
```

**Filters:** Pool minimum (1.5 BNB), payout floor (1.5x), strong-signal
bypass for small pools (BTC strength > 0.0004, pool >= 1.0 BNB).

## Architecture

See [docs/architecture.html](docs/architecture.html) for the visual diagram.

**Two-phase runtime loop (per round):**

| Phase | Timing | Actions |
|-------|--------|---------|
| A: Housekeeping | lock_at - 6s | Epoch check (BSC RPC), TLS warmup (3 parallel connections to OKX), pool data (BSC WSS) |
| Sleep | 3-4s | Wait for cutoff moment + OKX publish delay |
| B: Critical path | lock_at - 1.75s | Fetch BTC/ETH/SOL 1s klines (~285ms), compute signal, sizing, timing guard (lock_at - 1s), submit bet |

**Data sources:**

| Source | Data | Used for |
|--------|------|----------|
| OKX public REST API | BTC, ETH, SOL 1s candles | Signal computation |
| BSC RPC (publicnode) | Epoch state, round data | Timing, bet submission |
| BSC WSS (publicnode) | Real-time BetBull/BetBear events | Pool tracking |
| The Graph API | Historical closed rounds | Sync mode (backtest data) |

## Setup

1. Create a `.env` file at the repo root:

   - `THE_GRAPH_API_KEY` - API key for The Graph gateway
   - `BSC_WALLET_PRIVATE_KEY` - Wallet private key hex (with or without `0x`)

2. Review `config.toml` settings.

3. Run:

   | Mode | Command | Description |
   |------|---------|-------------|
   | Sync | `python run.py --sync` | Fetch closed rounds + klines from OKX, then exit |
   | Backtest | `python run.py --backtest` | Replay signal on historical data |
   | Dry | `python run.py --dry` | Paper trading with real-time data |
   | Live | `python run.py` | Real on-chain bets |

## Outputs

- `var/backtest_trades.csv`, `var/backtest_summary.json` - Backtest results
- `var/runtime/dry_bankroll_state.json` - Dry mode bankroll
- `var/runtime/dry_cycle_audit.csv` - Per-round decision log
- `var/runtime/dry_audit_trades.csv` - Dry bet/settlement ledger
- `var/{bnb,btc,eth,sol}_spot_prices.jsonl` - Synced 1s kline data
- `var/closed_rounds.jsonl` - Historical round data

Dry state is archived to `../PancakeBot_var_exp/` on startup and shutdown.
