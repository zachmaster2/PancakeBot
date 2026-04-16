# Autonomous Session Notes
_Append-only log. Each entry: date, what was tried, why, what happened._

---

## 2026-04-16 — WSS Reliability Fix + Dry Mode Restart

### What happened before this session
- Dry-mode ran for ~16h (02:51–18:55 local) from PID 2812 in a PyCharm terminal
- WSS to `wss://bsc.publicnode.com` had two multi-hour disconnections:
  - 03:31–~05:13 (~1.5h) — self-recovered
  - 16:09–18:55 (~2.75h) — did NOT recover; bot crashed at 18:55:26
- The crash left 3 cycle_audit rows at the same epoch (473274) one second apart — a crash-loop signature just before death
- No clean shutdown archive was created (process died, didn't exit gracefully)
- 108 of 179 cycles (60%) had zero pool data (`observed_total_pool_bnb = 0.0`)
- 3 bets placed during the healthy window: 1 win (+0.165), 2 losses (-0.3, -0.133) → net −0.358 BNB
- Bankroll: 50.0 → 49.642 BNB

### Root cause
`pool_watcher._run_loop()` used a single hard-coded endpoint (`wss://bsc.publicnode.com`) with a flat 5s retry. After 2h of reconnect thrashing (~1,500 failed attempts), something in the asyncio/websockets/threading stack crashed the process (unhandled exception, likely from the background thread flooding the event loop).

### What was changed (commit 8c7e1f0)
**`pancakebot/chain/pool_watcher.py`** — full reliability rewrite:
- 3-endpoint pool: `wss://bsc-rpc.publicnode.com`, `wss://bsc.drpc.org`, `wss://bsc.meowrpc.com`
- Round-robin failover: on disconnect, advance to next endpoint, wrapping
- Per-endpoint exponential backoff + jitter: 5→10→20→40→80→120s, ×rand[0.75,1.25], resets after >60s connected
- Per-endpoint circuit breaker: 3 consecutive failures → skip endpoint for 5min
- Watchdog thread: if `_connected=True` but no event/newHead for 30s → force reconnect
- New properties: `current_endpoint` (str), `last_connected_at` (float)

**`pancakebot/runtime/engine.py`** — logging:
- `POOL_WSS ROUND DATA` now logs `endpoint=` field
- New `POOL_WSS ROUND DISC` log line when disconnected, showing `endpoint=` and `last_ok=`

**`tests/test_pool_watcher.py`** — 24 unit tests, all passing:
- Backoff math (step increments, cap, reset)
- Circuit breaker (open/close/cooldown duration)
- Failover (round-robin, circuit skip, wrap-around)
- Watchdog (fires/does-not-fire conditions)
- State properties (initial values, stats dict)
- Mocked connection (subscriptions set `_connected`)

### Restart result
- PID 1535 started at 19:19:22
- `wss://bsc-rpc.publicnode.com` immediately timed out (still dead)
- Auto-failover: `wss://bsc.drpc.org` connected at 19:19:41 (19s startup)
- First cycle (epoch 473279): pool_bnb=0.2956 (too small), SKIP gate_no_signal — correct
- `POOL_WSS ROUND DATA endpoint=wss://bsc.drpc.org` confirmed in log

### Strategy analysis (Phase 3 assessment)
Reviewed `momentum_pipeline.py`, `momentum_gate.py`, and `var/backtest/summary.json`.

**Backtest (35,000 rounds, last run through epoch 473089):**
- Win rate: 61.4%
- Net PnL: +54.18 BNB from 50 BNB (+108.4%)
- Max drawdown: 5.34% = 3.87 BNB
- Max consecutive losses: 8
- Avg win: +0.216 BNB, Avg loss: -0.251 BNB
- Losses > 0.5 BNB: 58/1521 (3.8%); Wins > 0.5 BNB: 63/1521 (4.1%)
- Worst 5% avg loss: −0.998 BNB

**p_final=0.5 is intentional** — the strategy is non-probabilistic (directional momentum, not probability estimation). `p_bull=None` in the pipeline returns `p_final=0.5` as a placeholder. The real edge is 61.4% WR confirmed by backtest.

**Decision: no strategy changes made.** Rationale:
1. The strategy has been extensively validated (21 research steps, 5-fold cross-validation, all folds positive)
2. Max drawdown 5.34% is already very low
3. The dry-run losses (2/3 bets lost) are within normal variance (P(2+ losses in 3 bets at 61.4% WR) ≈ 27.5%)
4. Making blind parameter changes without new backtest data risks breaking a validated system

**What would warrant changes:** If after 50+ dry-run bets the observed WR is materially below 55%, or if drawdown exceeds 10%, then revisit sizing/threshold parameters with backtest support.

### Log file location
`var/dry/logs/dry-20260416-191919.log`

Solves the previous "no log file on disk" problem. Stdout+stderr of the dry process are now persisted to a timestamped file in `var/dry/logs/` on each restart.

### Open items / next steps
1. Monitor next 5-10 dry bets to establish live WR baseline
2. If `wss://bsc.drpc.org` also shows instability, `wss://bsc.meowrpc.com` is next in the failover queue
3. Gap to 4.0 BNB/2k target: see memory `project_signal_research_v2.md` for remaining research paths (DOGE pairs, BTC prediction market, order book depth, funding rates)
4. Consider `--sync` run when convenient to extend backtest dataset with recent rounds (current data ends at epoch 473089; current live epoch ~473280)
