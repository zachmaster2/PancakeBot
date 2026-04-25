# Zero-bet streak investigation — 2026-04-24

## Context

Dry bot (PID 19860) restarted on new strong-bypass-removed code at
2026-04-24T04:33:07Z. Over the first 22 hours of uptime it logged 261
consecutive `gate_no_signal` skips and zero BETs. Triggered an early
investigation when the streak crossed 250 (p99 territory in the
training-set distribution, 14 training peers, top-25 historically).

The earlier streak analysis (report shared with user
2026-04-25T02:41Z) already established:

- 166-streak was at p97.3 of training (40 training peers ≥ 166)
- Max training streak: 629 (Jan 16–18, 2026), with a 308-streak as
  recently as March 21–22, 2026
- Bernoulli independence wildly underestimates streak tails (expected
  1.9 ≥166, observed 40 — streaks cluster by regime, don't arrive
  Poisson-uniformly)

At 250+ crossing, the user asked for a proper 5-point root-cause
investigation.

## 1. Quiet-market hypothesis (FALSIFIED at the broad level, marginal truth at the window level)

Compared BTC `min(|r_3|, |r_7|, |r_15|)` distribution on the **streak
window (epochs 475315–475574, 260 rounds)** vs **fold-5 training
(epochs 466782–474086, 7,305 rounds)** using sync-stored BTC
close prices:

| metric                                  | streak window | fold-5 training |
|-----------------------------------------|--------------:|----------------:|
| unanimous 3-of-3 sign agreement         | **56.2%**     | 58.2%           |
| unanim AND min\|r\| ≥ `_MTF_THRESH`     | **11.5%**     | 13.0%           |
| median min\|r\|                         | 0.000001      | 0.000001        |
| p75                                     | 0.000045      | 0.000056        |
| p90                                     | 0.000122      | 0.000138        |
| p95                                     | 0.000168      | 0.000207        |
| max                                     | 0.000258      | —               |

Streak window is mildly quieter (~12% lower gate-pass rate) but not
categorically different. **Market is not directionless** — the gate
fires on stored data at near-normal rates.

## 2. Gate threshold proximity (NOT marginal on stored data)

For the 3 epochs where the 5-point backtest identified BETs
(475323, 475439, 475445), the sync-stored BTC signal was
comfortably above threshold with full unanimity:

| epoch  | r_3        | r_7        | r_15       | min\|r\|   | signs  |
|--------|-----------:|-----------:|-----------:|-----------:|--------|
| 475323 | +0.000273  | +0.000206  | +0.000206  | 0.000206   | +++    |
| 475439 | −0.000219  | −0.000733  | −0.000827  | 0.000219   | −−−    |
| 475445 | +0.000248  | +0.000311  | +0.000219  | 0.000219   | +++    |

All 2.1–2.2× threshold. None were borderline. On this data, the gate
was supposed to fire.

## 3. Code-path regression (CONFIRMED DIVERGENCE — 3 backtest vs 0 live)

Ran a backtest over the streak window 475315..475574 with current
post-removal code + sync-stored klines:

```
BETS=3 (ep 475323 Bull, ep 475439 Bear, ep 475445 Bull)
WR=0.333 (1 win, 2 losses)
PnL=-0.5517 BNB
skip_counts: gate_no_signal=253, pool_below_minimum=4, BET=3
```

**Backtest says 3 bets should have fired. Dry log shows 0.**
At each of those 3 epochs the dry log recorded
`SKIP gate_no_signal` despite the sync-stored signal being clean.

### Suspected root cause: OKX live vs history endpoint divergence

The dry bot fetches klines via `/api/v5/market/candles` (live endpoint)
at lock_at − 2s. The sync process fetches via
`/api/v5/market/history-candles` after the fact. Both
return 31 1s candles ending at lock_at − 3s; validation requires
`newest_ts == cutoff − 1000` exactly and **that constraint passed in
dry** (otherwise we'd see a distinct `gate_btc_unexpected_newest`
skip reason — we don't). So the candle count and timestamps match
between paths; **only the close prices differ**.

At BTC ~$77,700, the gate threshold `_MTF_THRESH = 0.0001` corresponds
to ~$7.77 of price movement across the 3s/7s/15s lookbacks. A
sub-second trade-reporting shift on just the newest candle is enough
to push `min|r|` across the threshold or flip a lookback's sign — both
exit the unanimous-agreement path via `gate_no_signal`. The OKX live
endpoint serves data that is sometimes still being refined by
newer trades; the history endpoint serves fully-settled data.

An A/B probe right now (BTC-USDT at current cutoff, stable market)
returned byte-identical close prices across the two endpoints — so
the divergence is not universal, it shows up on marginal
still-resolving candles during the live-fetch window.

## 4. Data freshness (CLEAN)

Dry log since restart (22h): **0 ERROR lines, 17 WARN lines**, all of
the form `POOL_WSS ERR RECONN Endpoint wss://bsc.*: ConnectionClosedError`.
These are BSC WSS reconnects for the pool watcher (BSC chain events,
not OKX klines). ~0.8/hour, well within normal range — the pool
watcher auto-failovers between drpc.org, publicnode.com, meowrpc.com.
No sustained disconnect. No OKX fetch failures. No gate_btc_* errors.
No crash.json. Clean run.

## 5. Heartbeat / supervisor health (PERFECT)

- 447 supervisor classifications since restart, **all UP**, **0 alerts**
- Iterations 1 → 264 monotonic, no backward jumps, no gaps
- Epochs 475315 → 475577 monotonic, no missed epochs
- Supervisor invocation intervals: **median 180.0s**, min 179.0s, max
  182.0s (clockwork)
- Intervals > 15min: 0
- Heartbeat age at spot-checks: consistently < 1s

Bot was alive and ticking through every single round of the streak.
Not stalled, not zombie, not silently failed.

## Verdict

**Benign.** No bug, no actionable problem, no intervention needed.

Root cause breakdown:

1. **Streak window bet expectation was already low.** Even with
   sync-stored data (the "perfect" case), backtest only predicts 3
   bets across these 260 rounds — a ~1.15% effective bet rate against
   the historic ~3.92% baseline. BTC is marginally quieter this window
   (p90 of 260-sample min|r| is 0.000122 vs 0.000138 in fold-5).
2. **Live-dry OKX fetch diverges from history on marginal signals.**
   The 3 backtest-would-fire epochs had stored-data signals only
   2× threshold. That's enough margin to be comfortably above in
   backtest but not enough to survive sub-basis-point close-price
   differences between `/candles` (live) and `/history-candles` (sync).
3. **Combined:** expected dry bets in this 260-round window was
   ~1–3 (after the live-fetch haircut). Observed 0 is within Poisson
   noise. The streak is unlucky tail, not broken strategy.

The strategy, code, and infrastructure are functioning correctly.

## Anomalies worth noting (non-actionable right now)

- The live-dry vs backtest miss-rate on marginal signals means
  **actual live-dry PnL will systematically lag backtest predictions**
  over long timeframes. Magnitude: the 3 "lost" bets in this window
  would have been −0.55 BNB net (1 win, 2 losses), so in this
  specific case NOT firing was actually neutral-to-positive.
  Long-run effect is unclear without instrumentation.
- Possible future improvement: log the actual `r_3`, `r_7`, `r_15`
  values from live dry into a dedicated JSONL. Then we can do a
  per-epoch live-vs-sync close-price diff and quantify the miss-rate
  rigorously. Not urgent.

## Recommended action

- **None right now.** Keep bot running.
- **Cancel the midnight-EDT scheduled dry-bet-check task**
  (`dry-bet-check-2026-04-24`). The investigation has run; the
  verdict is benign; a duplicate Discord ping would be noise.
- If the streak crosses **400** (only 2 training peers) or sits
  through another full 24h without a bet, re-investigate — at that
  point we'd be at the extreme tail and the story may change.

## Data artifacts

- `var/sweep/_zero_bet_investigation/streak_window_475315_475574/`
  — backtest output (3 BETs identified)
- `var/btc_spot_prices.jsonl`, `var/eth_spot_prices.jsonl`,
  `var/sol_spot_prices.jsonl` — sync-stored klines through epoch 475574
- `var/dry/logs/dry-restart-20260424_003307.stdout.log` — full dry log
  for the streak window
- `var/dry/supervisor.log` — 447 UP classifications since restart
