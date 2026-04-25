# Kline divergence — position-wise follow-up (2026-04-25)

User asked the right question: is the divergence concentrated in the
**newest candle only** (publish-lag hypothesis), or **broader**?

## Headline: BROADER. The hypothesis was wrong.

The live OKX `/candles` endpoint is returning windows that are
**systematically lagged by entire seconds** -- entire candles are
missing from the tail of the window. Aligning live and history klines
by timestamp shows that on every captured timestamp, **closes match
exactly between sources**. The divergence is not in the close-price
fidelity; it's in the **window endpoint**.

## Lag distribution across all 134 captured rounds

| live-fetch lag | n   | %     |
|---------------:|----:|------:|
|             0s |  34 | 24.5% |
|             1s |  53 | 38.1% |
|             2s |  37 | 26.6% |
|             3s |   5 |  3.6% |
|             4s |   2 |  1.4% |
|     306s-918s  |   8 |  5.8% |
| **lagged**     | **105** | **75.5%** |

"lag" = `(cutoff_ms - 1000) - max_live_kline_ts_ms`, in seconds. Zero
means live's newest candle exactly equals the expected cutoff-1s.
Positive means live's newest is N seconds BEFORE expected. The
catastrophic 306s+ entries are stuck-cache responses (OKX returned
data minutes-hours old).

## The 3 known-divergent epochs

For each, position-wise close comparison shows **every common
timestamp matches exactly** -- live just doesn't have the last 1-2
candles that history has:

| epoch  | lag | live_newest_ts | hist_newest_ts | live_newest_c | hist_newest_c |
|--------|----:|----------------|----------------|---------------:|---------------:|
| 475653 | 2s  | 1777108725000  | 1777108727000  |       77620.10 |       77611.50 |
| 475660 | 2s  | 1777110879000  | 1777110881000  |       77741.30 |       77759.20 |
| 475711 | 1s  | 1777126492000  | 1777126493000  |       77658.00 |       77676.30 |

The "$8-$18 close diff on the newest candle" we saw earlier was
**comparing live's last candle against history's later candle**, not
the same candle. They aren't the same thing -- the price moved
between live's newest ts and history's newest ts.

## Why the divergence rate is "only" 2.24%

75.5% of fetches are lagged, but only 3 rounds (2.24%) showed A/B
divergence in BET decisions. Reason: the bot's `_validate_klines`
function ALREADY rejects lagged windows with
`gate_btc_unexpected_newest`. The pipeline downstream-converts that
to a generic `gate_no_signal`. So:

  * **75.5% of rounds: live is lagged**, validation rejects, bot skips
  * **24.5% of rounds: live = history**, gate fires normally
  * **Of the lagged rounds, only ~3% fall on epochs with REAL signals**
    that history would have caught -- those are the A/B-divergent rounds

The bot is correctly skipping all lagged fetches. The "missed bets"
aren't a bug -- they're **rounds where OKX's live endpoint failed to
deliver fresh data on a round where the price was actually moving**.

## What this means for interventions

**Reject the "lower threshold" intervention.** Lowering `_MTF_THRESH`
doesn't help because validation rejects the lagged windows BEFORE the
threshold check ever runs. The bot doesn't see partial signals; it
sees no signal.

**The right intervention is to compensate for OKX live-fetch lag.**
Options ranked:

1. **Delay the fetch by 1-2s** past current `cutoff + 0.25s`.
   Currently we fetch at `cutoff + 0.25s`; if we instead waited
   until `cutoff + 1.5s`, the 38% of rounds with 1s lag would
   resolve. At `cutoff + 2.5s`, the 65% with 1-2s lag would
   resolve. Trade-off: cuts into the time before lock to submit
   the bet (`_LOCK_SAFETY_MARGIN_SECONDS = 1`). Currently
   `lock_at - cutoff = cutoff_seconds = 2`, so we have 1.75s
   between fetch and lock-safety. Pushing fetch to `cutoff + 2.5s`
   leaves -0.5s -- which would need a smaller `cutoff_seconds`
   (e.g. 4s) to make room.

2. **Retry the fetch on validation failure** with a small delay
   (e.g. 500ms then re-validate). Catches the transient cases
   where OKX is just-about-to-publish. Doesn't help the 5.8%
   stuck-cache cases.

3. **Loosen validation** to accept windows up to N seconds lagged,
   computing returns over the available window. Changes the
   meaning of `r_3 / r_7 / r_15` since the windows aren't quite
   what the names imply. Most invasive; defer.

## Recommendation

Option 1 (delay-fetch) is the natural fix and directly targets the
root cause. Before changing anything, capture **another 500-1000
rounds** to confirm the lag distribution is stable across different
market conditions and OKX server states. If the distribution holds,
push for `cutoff_seconds = 4` and `fetch_offset = cutoff + 2.5s`,
which buys back ~65% of currently-lost rounds without touching the
strategy.

Sample artifacts:
  * `research/kline_divergence_position_analysis.py` -- the analysis
  * `var/dry/captured_klines.jsonl` -- 134 rounds of captures
  * Earlier (incorrect) hypothesis report:
    `research/kline_divergence_2026_04_25.md` -- supersede this one
