# The 47.6-day coverage band — recorded 2026-08-31

A region where a gap in the stores is **detectable, still fetchable, and
never looked at by the routine sync.** It is a property of the
architecture, not a transient fault, and it is the same mechanism as the
2026-08-24 incident one layer out.

The operational long-form lives in `var/health_checks/novel_observations.md`.
This copy exists because `var/` is gitignored and outside the backup
archive, and a finding recorded only there is barely more durable than the
conversation it came from.

## The numbers

```
sync working window   35,000 rounds x ~306s  =  124.0 days
OKX 1s kline horizon                            171.6 days
--------------------------------------------------------
BAND                                             47.6 days

~282 epochs per day
a gap opening today is permanently unrecoverable after ~172 days
```

## What it is

`_sync_1s_klines` is only ever handed `tail_rounds = rounds_all[-cache_n:]`.
Anything older than 124.0 days is outside the sync's working set, so the
sync never looks at it — not to check it, not to repair it.

OKX still serves 1s klines for 171.6 days. So for **47.6 days** a gap is
detectable (the whole-store byte scan reads every epoch back to 437562),
fetchable (the data still exists upstream), and untouched by the routine
sync, which stops 47.6 days short of the horizon.

After roughly day 172 it is gone from OKX and exists nowhere on Earth.

## Why it is the previous incident's mechanism

The 23 missing epochs found in the 2026-08-24 repair all sat **below the
35,000-round tail**, so they were invisible *by construction* for months.
Identical shape: a region the checking machinery did not cover, where the
absence of an alarm was read as the absence of a problem. The tail assert
guards what the **backtest** consumes; it was never a statement about the
whole store, and it was treated as one.

## What has changed, and what has not

**Changed.** The whole-store contiguity report (byte scan, ~6s for 4.5 GB)
runs on every sync and covers the full span; as of 2026-08-31 the watchdog
*evaluates* it rather than merely emitting it. A gap at any age now raises
a Desktop marker.

**Not changed.** Detection is not recovery. Filling a detected gap still
requires a human to set `ALLOW_STORE_REWRITE=1`, because the fill path goes
through `store.rewrite()` — a whole-file 4.5 GB replace, the highest-risk
write in the system, and the operation that mangled line endings in May.
That gate is deliberate.

## The residual risk, stated plainly

A marker only helps if somebody looks at it, and the premise of the current
posture is months of nobody paying attention. A gap opening in November and
noticed in March would be detected, reported, and permanently lost anyway.

A staging proposal was under consideration on 2026-08-31: decouple
**capture** (time-critical, bounded by the 171.6-day horizon) from
**integration** (risky, but with no deadline) by automatically fetching
missing records into an append-only side file that never touches a
canonical store, leaving the merge as a supervised human operation.
Undecided at time of writing. If it was never built, this band is why it
was proposed and why it may be worth revisiting.

Note the honest framing: staging would **defer** the dangerous write, not
remove it. Merging still eventually calls `store.rewrite()`.
