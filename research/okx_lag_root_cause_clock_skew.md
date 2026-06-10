# OKX kline lag â€” actual root cause is CLOCK SKEW, not connection affinity

## TL;DR

The hypothesis "connection-reuse-induced backend affinity" was **partially
right and partially wrong**:

- âœ“ It correctly diagnosed the **stuck-cache mode** (5.8% of pre-fix rounds,
  306s+ same-response-served-across-rounds). Commit `6767e85` (per-round
  warmup session reset) eliminated this entirely.

- âœ— It was **wrong about the 1-3s lag mode** (75% of pre-fix rounds).
  The dominant cause is **system clock skew**: the bot's local clock
  is 1.6-2.3s AHEAD of OKX's clock, so `time.time()`-based fetch
  scheduling causes the bot to query OKX BEFORE the requested window
  has had time to materialize in OKX's frame.

The bot CANNOT compensate for this skew in code â€” the timing budget
between `cutoff + 0.25s` (fetch) and `lock - 1s` (safety margin) is
0.75s in local time, while skew is 2.3s. **The system clock must be
synchronized.**

## Evidence chain

### Phase 1: Connection-reuse hypothesis (incorrect for 1-3s mode)

Initial diagnostic A/B (`research/okx_connection_ab.py`) showed:

  | variant | mean | p50 | p95 |
  |---|---:|---:|---:|
  | session_reuse | +458/+256ms | +329/+276 | +1418-1999 |
  | fresh_conn | +95/-3 | +38/-23 | +762-1495 |

This LOOKED like fresh-conn fixed the lag. But the probe asks OKX for
"latest 1 candle" without an `after=` parameter, which returns the
in-progress candle (close_time = now, lag â‰ˆ 0). So fresh-conn vs
session-reuse only measured the secondary effect of stuck-routing,
not the primary effect that affects the bot.

### Phase 2: Bot post-fix shows lag persists

Two fix attempts deployed and measured:

  | revision | n | any-lag | mean | stuck-cache |
  |---|---:|---:|---:|---:|
  | pre-fix | 145 | 76.6% | 1000ms median | 5.5% (8 events) |
  | 6767e85 (per-round reset) | 17 | 100% | 1765ms | 0% âœ“ |
  | 27e8e36 (per-fetch fresh) | 7 | 100% | 2000ms | 0% âœ“ |

**Stuck-cache eliminated** (real win â€” 0% post-fix vs 5.5% pre-fix),
but 1-3s lag persists at higher rate after the fixes than before.

### Phase 3: Parallel-probe variant D matches bot pattern

Modified the probe to use the bot's exact pattern:
parallel-3 fetches with fresh sessions and `after=` parameter
(probe script not retained in-repo; the surviving companion from this investigation is `research/okx_artificial_delay_probe.py`):

  | variant | n | mean | p50 | max |
  |---|---:|---:|---:|---:|
  | C: parallel-3 BTC fresh, no after= | 15 | +164ms | +271 | +1441 |
  | **D: parallel-3 fresh + after= (FULL bot pattern)** | 15 | **+351ms** | +394 | +1604 |

Probe D shows mean lag 351ms â€” much LOWER than bot's measured 2000ms.
Discrepancy: ~1650ms.

### Phase 4: Clock skew accounting

Direct measurement: `local_ms - okx_server_time_ms = 2322ms`.
`w32tm /query /status` shows Stratum 0 (unsynchronized), Root
Dispersion 10.37s, last successful sync 9+ hours ago. Windows clock
has drifted significantly from UTC.

### Math

The bot's fetch scheduling:

```
cutoff_ts_t = lock_at - 2  # (true UTC seconds, from BSC chain)
fetch_ts    = cutoff_ts_t + 0.25  # (treated as time.time() target = LOCAL)
```

When `time.time() == fetch_ts` (LOCAL), true UTC = `fetch_ts - skew`:
```
true_utc_at_fetch = (lock_at - 2 + 0.25) - 2.32
                  = lock_at - 4.07  (in true UTC)
```

OKX, operating in true UTC, has data fresh through (current_time -
publish_lag). At true UTC = lock_at - 4.07s, OKX's most-recent
fully-published candle is at `open_time = lock_at - 4.07 - 1 -
publish_lag = lock_at - 5.07 to 6.07` (depending on publish lag).

Bot's `_validate_klines` expects newest_ts == cutoff_ms - 1000 =
`lock_at - 3` (true UTC).

```
bot_lag = expected - actual = (lock_at - 3) - (lock_at - 5.07 to 6.07)
        = 2.07 to 3.07 seconds
```

**This matches the observed 1-3s lag distribution exactly.**

### Why the standalone A/B (no `after=`) showed near-zero lag

The probe uses `okx_now_estimate = local_now - skew` for its lag
calculation. So skew is subtracted out. The probe was measuring
"OKX-internal publish lag" which is genuinely small (~200-400ms).
The bot measures "expected vs actual newest" which DOES include skew
because cutoff_ms is a true-UTC anchor (from BSC) but fetch
scheduling is local-time.

## Why this can't be fixed in code

Bot's timing budget in LOCAL time:

```
cutoff (local-anchored)             = lock - 2
fetch_ts                            = cutoff + 0.25 = lock - 1.75
lock_safety_margin (local)          = lock - 1
budget between fetch and safety     = 0.75 seconds
```

To get FRESH OKX data, the bot would need to fetch at LOCAL time =
`cutoff + 0.25 + skew = lock - 1.75 + 2.32 = lock + 0.57`. That's
**after** the local lock_at value, which would also be **after** the
true-UTC lock by skew (since true_utc = local - skew).

The bot would either:
1. Miss the betting window entirely (fetch happens after lock), OR
2. Fetch with stale data (current behavior)

Net: **with skew > ~0.5s, the bot cannot compensate without
rearchitecting the timing pipeline.**

## What Phase 1-3 fixes accomplished

`6767e85` (per-round `OkxClient.warmup()` session reset):
- âœ“ **Eliminated stuck-cache mode** (0/24 post-fix vs 8/145 pre-fix).
  This was a real, separate problem from the skew issue. KEEP.

`27e8e36` (per-fetch fresh Session in `fetch_1s_klines`):
- âœ— No measurable benefit (1-3s lag mode unchanged at 100%).
- âœ— Adds ~150ms TLS handshake overhead per fetch.
- **REVERTED** in commit `d93cb22`.

`1cb2b20` (opt-in OKX header capture):
- âœ“ Diagnostic infrastructure, no behavior change. KEEP.

`ec5605e` (kline capture infrastructure):
- âœ“ Foundation for divergence A/B analysis. KEEP.

`dd14ec6` (`--kline-source captured` backtest replay):
- âœ“ Foundation for divergence A/B analysis. KEEP.

`eeaad8f` (`archive_lingering_crash_file` always-archive policy):
- âœ“ Independent fix for false-CRASHED supervisor incident. KEEP.

## Recommended fix path

The clock-skew root cause needs an **environmental fix** (sync the
system clock) or a **deeper architectural change** (anchor scheduling
to OKX time or BSC chain time, not local time).

### Option 1: Sync the system clock (RECOMMENDED â€” simplest)

```powershell
# As Administrator:
w32tm /resync /force
# Verify:
w32tm /query /status
# Configure for more frequent syncs:
w32tm /config /update /manualpeerlist:"time.windows.com,0x9 time.cloudflare.com,0x9"
```

After sync, expected skew should drop to <100ms and the bot's lag
distribution should drop to <30% any-lag (matching pre-existing
"good" days where skew was small).

**Caveat**: requires admin privileges. Once-per-startup. May need a
scheduled task to re-sync periodically if drift recurs.

### Option 2: Bot-side skew compensation (architectural)

Re-architect `_sleep_until_ts` to anchor to OKX time. At bot startup,
measure `skew = local_time - okx_time`. When sleeping until `cutoff +
delay`, sleep until `local_time = cutoff + delay + skew` (i.e. wait
the extra time so that OKX's clock has progressed appropriately).

Problem: with skew > 0.5s, the adjusted fetch happens AFTER the local
`lock_safety_margin`. Bot would need to also adjust the safety
margin... but the on-chain `lock_at` is a true-UTC value, so adjusting
both doesn't actually buy more wall time.

Effectively, this option is "wait for the candle, then bet quickly
before lock arrives in true UTC". With 2.3s skew and 1s safety
margin, the bot has -1.3s. Doesn't fit.

**Verdict**: Option 2 only works if skew is <500ms. Above that,
Option 1 (sync the clock) is mandatory.

### Option 3: Detect skew at bot startup, refuse to bet if too large

Add a startup check: measure skew, log loudly if > 500ms, abort if
> 1500ms. Prevents the bot from running with broken timing.

This is a **defensive fix**, not a solution. Should land regardless
of which other path is chosen.

## Recommendation

1. **Fix the clock first** (Option 1, admin one-time fix).
2. Add **defensive skew check at startup** (Option 3, small code change).
3. Restart bot, verify lag distribution drops to <30% any-lag.
4. Re-run captured-vs-history A/B to confirm divergence near zero.
5. 12-hour soak.

If clock sync isn't an option (e.g. server policy forbids it), then
the strategy needs a fundamental redesign that doesn't depend on
1-second-fresh OKX data â€” likely WSS subscription or different data
source.

## Surface to user

The user needs to decide:
- Accept Option 1 (sync clock manually as admin)
- OR move bot to a server with synced clock
- OR redesign the strategy to tolerate larger skew (likely WSS)
- AND/OR add the defensive startup skew check (Option 3)

Without one of these, the bot will continue to skip ~75% of rounds
even with the connection-affinity fixes.
