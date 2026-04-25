# Live-vs-history kline divergence — A/B 2026-04-25

## Methodology

Captured 134 contiguous rounds of live BTC/ETH/SOL klines + computed
signals + decisions from the dry bot (epochs **475587..475720**, ~11h
of bot runtime through commit `dd14ec6`). Then ran the same backtest
strategy code on the same epoch range twice:

  1. `python run.py --backtest --kline-source captured` -- replays
     using the bot's actual live OKX `/candles` fetches at decision
     time.
  2. `python run.py --backtest --kline-source history` -- standard
     replay using OKX `/history-candles` data fetched post-hoc by
     `--sync`.

Strategy code is byte-identical between runs; only the kline source
differs. Per-round trades.csv outputs are diffed.

## Headline

**Decision divergence rate: 2.24% (3 / 134 rounds).**

**Direction: 100% asymmetric.** Live captured **NEVER fires when
history skipped (0/134)**. The 3 divergent rounds are all of the form:
*history says BET, captured says SKIP*. Live data systematically
suppresses gate fires; it never spuriously creates them.

**PnL gap: +0.2811 BNB over 134 epochs** (history-source theoretical
beats captured-source theoretical by this margin). Sample is small
(3 bets) so absolute gap CI is wide, but the SIGN is structural.

## Per-round comparison (the 3 divergent epochs)

```
epoch    src       r_3        r_7       r_15     min|r|     agree fires?
-----------------------------------------------------------------------
475653   captured  +0.000001  +0.000000 +0.000000 0.000000  False False
475653   history   -0.000110  -0.000111 -0.000111 0.000110  True  True   <- BET Bear, +0.41 BNB
475660   captured  +0.000001  +0.000000 +0.000000 0.000000  False False
475660   history   +0.000232  +0.000232 +0.000230 0.000230  True  True   <- BET Bull, -0.19 BNB
475711   captured  +0.000001  +0.000045 +0.000075 0.000001  True  False  <- min|r| < threshold
475711   history   +0.000236  +0.000281 +0.000310 0.000236  True  True   <- BET Bull, +0.07 BNB
```

In each case the **newest 1s candle differs** between sources:

| epoch  | live final close | history final close | $ diff |
|--------|-----------------:|--------------------:|-------:|
| 475653 |        77620.10  |            77611.50 |   8.60 |
| 475660 |        77741.30  |            77759.20 |  17.90 |
| 475711 |        77658.00  |            77676.30 |  18.30 |

At BTC ~$77,700, an $8-$18 close-price shift on the most-recent candle
shifts r_3 by 0.0001-0.0002 -- exactly enough to push `min|r|` above
or below the `_MTF_THRESH = 0.0001` gate threshold.

## Root cause (suspected, supported by data)

**OKX `/api/v5/market/candles` (live) returns the most-recent 1s
candle BEFORE its close has finalized.** The candle's `open_time_ms`
is correct (matches expected `cutoff - 1000`), the candle is marked
complete, but the close price hasn't yet absorbed the trades that hit
in the last few hundred ms of that second. By the time
`/api/v5/market/history-candles` (used by `--sync`) reads the same
candle minutes-to-hours later, the close has updated with the
actual final trade.

For most rounds this doesn't matter -- BTC is moving slowly enough
that the live close is "close enough" to the final close. But on
2.24% of rounds, the missing trades are exactly the ones that would
have crossed the gate threshold. The bot sees a flat market when
history shows a clear directional move that just kicked in.

## Theoretical PnL impact

Over 134 captured rounds (~11h of bot time):

|                       | bets | WR     | PnL BNB    |
|-----------------------|-----:|-------:|-----------:|
| history-source replay |    3 | 66.7%  | **+0.2811** |
| captured-source       |    0 |  --    |    0.0000 |
| gap                   |    3 |        |    0.2811 |

Extrapolated naively (ignoring small-sample noise): the live bot
would underperform the history-source backtest by **~0.61 BNB / day**
(or **~225 BNB / year** at 12 rounds/h). Sample-size CI is wide --
3 bets isn't enough for a tight estimate -- but the SIGN is robust
(0/134 rounds in the other direction).

## What this implies for Option D

User's gate was: *25% activity loss IS acceptable IF the
backtest-on-captured-data analysis later shows net profit improvement.*

**The data argues for the OPPOSITE of Option D's premise.**
Option D was about adding margin to the threshold (raising it to
filter borderline signals). The captured data shows the live bot
is ALREADY filtering more aggressively than history -- by accident,
because of OKX publish-lag. Adding more margin would compound the
deficit, not fix it.

**The right intervention** -- if any -- is to compensate for the
publish lag. Options:

1. **Wait an extra 1-2s before fetching klines.** Trade-off: less
   time before lock to bet (lock_safety_margin already 1s; pushing
   to 0s is risky on slow networks).
2. **Fetch the same window twice with a small delay**, average or
   take the second fetch (more I/O, more risk).
3. **Lower the threshold** to compensate for the systematic
   under-magnitude bias of live closes. e.g. `_MTF_THRESH * 0.7` to
   recover the missed bets at the cost of admitting a few false
   positives.
4. **Accept the deficit as the cost of live execution**, document it,
   stop trying to close it.

Option 3 is the most surgical. The data could pin the right
multiplier: if live `min|r|` averages 30-50% below history on
divergent rounds, lowering threshold by that amount recovers them.
Need a larger sample (1k+ rounds) to fit the multiplier reliably.

## Recommendation

1. **Keep capturing.** 134 rounds = informative but not robust. Aim
   for 1000+ rounds before any threshold change.
2. **Don't touch the threshold yet.** Three divergent epochs is too
   noisy to fit a multiplier.
3. **Investigate Option 1 first** (delay the fetch by 1-2s past the
   current `cutoff + 0.25s`). It's the natural fix for OKX publish
   lag and doesn't require strategy parameter changes. Risk: less
   time to bet, may push us into the lock-safety-margin.
4. **Reject Option D as originally specified.** The premise was
   wrong; live is already filtering too aggressively, not too
   loosely.

## Artifacts

- `research/ab_captured_vs_history.py` -- A/B harness (gitignored
  outputs at `var/sweep/_ab_captured_vs_history/`)
- `var/dry/captured_klines.jsonl` -- 135 rounds of live captures
  (~700 KB, 134 with valid klines)
- `var/sweep/_ab_captured_vs_history/diff.json` -- machine-readable
  diff for further analysis
