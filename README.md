# PancakeBot

Automated trading bot for PancakeSwap Prediction V2 on BNB Smart Chain.

## Modes

| Mode     | Command                    | Env vars                   | Description                                   |
|----------|----------------------------|----------------------------|-----------------------------------------------|
| Sync     | `python run.py --sync`     | `THE_GRAPH_API_KEY`        | Fetch rounds + klines + contract constants (depth = `[backtest] backtest_round_count`) |
| Backtest | `python run.py --backtest` | (none)                     | Replay historical data, compute PnL           |
| Dry      | `python run.py --dry`      | (none)                     | Real-time paper trading                       |
| Live     | `python run.py --live`     | `BSC_WALLET_PRIVATE_KEY`   | Real on-chain bets                            |

Running with no flags prints help. Modes are mutually exclusive.

## Strategy

**Signal:** BTC multi-timeframe momentum -- 3s, 7s, and 15s returns must all agree in direction with `min(|return|) >= threshold` (pool-adaptive: 0.0002 small / 0.0001 large).

**Regime-2:** When BTC is silent, ETH + SOL multi-TF agreement fires as a secondary signal with smaller sizing.

**Sizing:** Continuous adaptive based on signal strength, ETH/SOL confirmation, and payout odds.

**Filters:** Pool minimum (1.5 BNB), payout floor (1.5x); small pools
(< 3.0 BNB) admitted only on a strong signal (>= 0.0002).

### Changing the strategy

Two binding rules for any strategy change:

1. **Bit-identity or gauntlet.** A change must either keep the canonical
   5-fold backtest bit-identical (`_EXPECTED_5FOLD_HASH` in
   `tests/test_in_process_runner.py` — the suite enforces it) or be
   promoted as a new candidate through the full gauntlet: 5-fold CV, the
   frozen holdout, the extension_v2 OOS slice, and a permutation-null
   pass (see [docs/holdout_2026_04_24.md](docs/holdout_2026_04_24.md)).
2. **The pipeline seam is `pancakebot/strategy/base.py`**
   (`StrategyPipeline` Protocol). A new pipeline must satisfy it; the
   construction sites are `runtime/dry.py:_build_momentum_pipeline`
   (dry + live) and `backtest/runner.py`.

## Project Structure

```
pancakebot/
    constants.py, util.py, log.py        # Shared foundations (exceptions, money fmt)
    paths.py, timing_constants.py        # var/ layout; timing single source of truth
    config.py                            # All config: TOML, env, dataclasses
    types.py, pool_amounts.py            # Domain types (Bet, Round)
    settlement.py, bankroll_tracker.py   # PnL computation; risk gates (breaker/cooldown)
    app.py                               # Mode dispatch

    strategy/                            # Signal + sizing
        base.py                          # StrategyPipeline Protocol (the seam) + decision schema
        momentum_gate.py                 # OKX kline fetch + BTC multi-TF signal
        momentum_pipeline.py             # Signal -> sizing -> filters -> decision

    chain/                               # BSC chain interaction
        prediction_contract.py           # Web3 contract wrapper
        contract_config.py, rpc_chooser.py
        rpc_poller.py                    # Era 12b (2026-06-10): single bloXroute read
                                         # path, eth_getLogs pool event accumulation

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
        audit.py, bet_ledger.py          # cycle_audit.csv; bets.jsonl ledger
        regime_telemetry.py              # REGIME_DRIFT monitors
        single_instance.py               # One-bot-per-mode lock
        supervisor_artifacts.py          # bot.pid + crash.json (read by ops/)

    ops/                                 # systemd-facing alerting (docs/SUPERVISOR.md)
        notify_lifecycle.py              # pancakebot-notify@ oneshot entry
        notifications.py                 # Discord alert executor

    backtest/
        runner.py                        # Offline replay + equity plot
```

## Output

```
var/
    closed_rounds.jsonl                  # Synced round history
    {bnb,btc,eth,sol}_spot_prices.jsonl  # Synced 1s klines
    contract_constants.json              # Chain constants (from --sync)
    dry/                                 # Dry mode state (resumed on restart; archived on --fresh)
    live/                                # Live mode state
    backtest/                            # Backtest results + equity plot
```

## Setup

1. Create `.env` with required env vars (see mode table above)
2. Review `config.toml`
3. **Verify clock sync on the bot host** — see [clock-sync prerequisite](#clock-sync-prerequisite-dry--live-modes) below
4. Run `python run.py --sync` to fetch data
5. Run `python run.py --backtest` to verify
6. Run `python run.py --dry` for paper trading

The dry/live bot runs on a Linux VM (production: Frankfurt, systemd
units installed by `bootstrap/install.sh`; supervision + Discord
lifecycle alerting: see [docs/SUPERVISOR.md](docs/SUPERVISOR.md)).
Backtest/sync/research run anywhere.

### Direct run (development)

Dry/live also run directly — no service wrapper — on any host
(the only path on Windows since the service stack moved Linux-only):

1. Python 3.13+; `python bootstrap/common/python_setup.py` creates
   `.venv` and installs deps (handles `Scripts/` vs `bin/`), or plain
   `python -m venv .venv` + `pip install -r requirements.txt`.
2. `.env` at the repo root (loaded via python-dotenv).
3. `python run.py --dry` (no secrets needed) or `--live`.

Notes: a missing `BSC_WALLET_PRIVATE_KEY` fails fast with
`missing_env_var`; one bot per mode is enforced by a process-scan mutex;
network timing constants are tuned for the production VM, so a high-RTT
dev host will log occasional benign `head_fetch`/`getlogs` timeout
ALERTs (the poller retries). For a fast end-to-end check, set
`[backtest] backtest_round_count = 100` before `--sync`.

### Deploy gate

"Greenlit" for a deploy means: full suite green on the dev clone
(`python -m pytest`) **before** `git push vm master`, and green on the
VM (`.venv/bin/python -m pytest`) after checkout, before restarting the
unit. The suite includes the canonical-hash bit-identity test.

### Deploying (git pull from GitHub, 2026-06-30)

Source of truth is **GitHub** (`github.com/zachmaster2/PancakeBot`). The VM
is a plain clone at `/root/pancakebot`. Per deploy: push to GitHub from any
dev clone, then pull on the VM and restart manually when greenlit:

```
# dev box:
git push github master
# VM:
ssh root@<vm> "cd /root/pancakebot && git pull && systemctl restart pancakebot-live"
```

Untracked VM files (`var/`, `.env`, `.venv`) are never touched by a pull.
Keep periodic offline backups too: `git bundle create <path>.bundle --all`.
(The old VM-bare-repo push-to-deploy hook was retired 2026-06-30.)

### Clock-sync prerequisite (dry + live modes)

The critical-path scheduling relies on the local OS clock being within
a few milliseconds of true NTP. On the Linux bot host, `chronyd`
disciplines the clock continuously (frequency steering between polls):
measured residual offset is ~30-60 **micro**seconds RMS — four orders
of magnitude inside the bot's timing budget, with no extra tuning
needed for steady-state drift.

The one VM-specific risk is a clock **step** (host pause/live-migration
jumps the guest clock). `bootstrap/install.sh` installs a chrony
drop-in (`/etc/chrony.d/pancakebot.conf`: `maxpoll 6` + `makestep`)
that bounds step *detection* to ~64s instead of chrony's default
~17-minute worst case. The post-install health check
(`bootstrap/common/health_check.py`) verifies the clock is
synchronized with the offset inside tolerance; spot-check manually:

```
chronyc tracking    # want: Leap status: Normal, System time offset ~microseconds
```

Backtest and sync modes do not require this (they are not
timing-critical).

**The bot has no application-level NTP layer.** The prior `NtpSync`
(per-round NTP query that measured `local − ntp` and applied the
offset inside `_utc_now()`) was retired in Bundle 5 v2 (2026-05-14);
it existed to work around the original Windows host's W32Time drift
(~270ms P95 at its default 1024s poll). The OS clock — chrony-
disciplined — is the sole source of truth; there is no in-app
fallback.

## Architecture

See [docs/architecture.html](docs/architecture.html) for the visual diagram.
Code names used throughout comments (Era 12b, F0, Bundle 5 v2, Candidate C,
off350, ...) are indexed in [docs/glossary.md](docs/glossary.md).

### Per-round runtime loop

Three pre-lock wakes anchored at the chain-supplied `lock_at` timestamp,
plus a post-close claim wake anchored at `close_at(prev_locked_epoch)`.
The preflight wake uses a deliberately generous (5 s) gap above the
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
| `wait_for_preflight` | `lock_at - 6195ms` | Off-critical-path setup: (1) refresh wallet balance (live = BSC RPC; dry = in-memory simulated bankroll) feeding the risk gates + `decide_open_round`; (2) prefetch the send **nonce** + **gas price** and warm all 3 write endpoints (keep-alive ≥30s) so the bet path makes no RPC but `send_raw` — drops the post-decision critical path from ~270ms (two cold rotated RPCs) to ~50ms (pre-cache, 2026-06-06) |
| `wait_for_single_poll` | `lock_at - 2500ms` (fixed rail) | ONE `eth_getLogs` range catch-up against the single bloXroute read endpoint (Era 12b) — the ~5-20 blocks since the last 8s periodic tick. Wall-clock capped at 950ms (`RPC_POLL_WALL_CAP_SINGLE_MS`) so a degraded poll can never delay the anchor poll; bracketed by the CAPTURE + COMPLETION + ANCHOR-CLEARANCE startup invariants |
| Anchor poll | `lock_at - 1500ms` | Single sub-second poll of chain head's BEP-520-encoded ms timestamp (200ms timeout, bloXroute); drives dynamic critical-path scheduling. Grossly stale anchors (>1350ms behind the local clock) are rejected to the static fallback with an `ANCHOR STALE` alert |
| `wait_for_critical_path` | `lock_at - 1195ms` | Single critical-path entry. Sequentially: pool snapshot from the RPC poller (`pool_cutoff_seconds = 6` data horizon; F0 coverage gate skips the round if the cursor hasn't polled through the cutoff block) → 3 parallel OKX `/history-candles` GETs (BTC/ETH/SOL) → signal compute → bet submit |
| Pre-bet timing guard | `lock_at - 789ms` | Abort if decision-ready past the safety margin (TX would mine after lock). Before submission, `assert_gas_cap_not_breached()` reads the **cached** `eth.gas_price` (refreshed off-path at the preflight wake) and raises `GasPriceCapBreachedError` — fail-loud — if it exceeds `MAX_GAS_PRICE_WEI` (1 Gwei) or the cache is unpopulated/stale. The bet TX uses the **cached nonce** (no inline `get_transaction_count`). Bets are posted at `MAX_GAS_PRICE_WEI` deterministically; on breach the bot SKIPs the round + fires a CRITICAL Discord alert |
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
  Cross-validated at load against the RPC poll schedule: the fixed
  single-poll rail must fire after the cutoff block is available
  (CAPTURE), complete before the critical-path wake (COMPLETION), and
  its wall cap must clear the anchor poll (ANCHOR CLEARANCE).
- `max_consecutive_kline_fetch_failures = 5` — streak counter before bot
  crashes (systemd restart + Discord alert).

### Empirical constants

Maintained in `pancakebot/timing_constants.py` with per-constant
provenance: probe script, measurement date, percentile, sample size.
Re-deriving any value requires re-running the probe named in that
constant's comment (under `research/`) **on the host the bot runs on**
(network RTTs are host-dependent by definition) and updating the
constant co-locked with a new measurement date.
