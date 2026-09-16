# What this bot actually did

**Findings as of September 2026. Read this before you run anything, and
before you spend real money on the backtest numbers in this repo.**

> **Status.** Live trading stopped 2026-08-30. The server it ran on was
> destroyed 2026-08-31. Nothing in this repository is deployed. The only
> process still running is a daily market-data sync on the operator's
> machine, kept because some of that data cannot be re-fetched.

---

## The short version

1. **The backtest is real.** Over the full data set (Dec 2025 – Sep 2026)
   the canonical strategy wins 59.6% of 2,015 PancakeSwap bets for +33.1 BNB
   on a 50 BNB start, and that re-runs exactly from this repo. The
   frequently quoted **61.23%** (in
   [holdout_2026_04_24.md](holdout_2026_04_24.md)) is the same strategy
   over an earlier, shorter window; it was not re-run for this document.
2. **It never predicted BTC, ETH or SOL.** Apply the same bets to the next
   five minutes of the assets its signal is built from and they do no
   better than a coin flip in any period: about 50% overall and in the
   strongest period, and lower since late May. Against real **BNB** spot
   it did beat chance for a while — 54.0% before late May, 52.3% over its
   life — because BNB lagged BTC by seconds. That real-price part is what
   later decayed.
3. **Most of what PancakeSwap paid for was oracle lag.** Rounds settle on a
   Chainlink *push* feed that updates about every 33 seconds, so the price
   at lock is typically ~16 seconds stale. Scored against real BNB spot at
   the same instants, the same bets win 52.3%, not 59.6%. The ~7-point
   difference is the oracle, and [section 8](#8-the-lag-measured-directly-measured-2026-09-15)
   measures the lag itself and shows staleness accounts for all of it.
4. **But the lag alone was never a winning bet.** The crowd bets the
   visible gap too, and prices it: a rule that mechanically bets the stale
   price's gap loses money in every month measured, including the
   biggest-pool ones. The edge was the lag **plus** knowing which moves
   were real — see [section 9](#9-two-parts-not-one-measured-2026-09-15).
5. **That stopped being enough.** The real-price part eroded through 2026
   until the total fell below PancakeSwap's ~55-57% breakeven. Both live
   runs fall inside that late period, and both reconcile bet-for-bet with
   the backtest: the bot did exactly what the backtest says, and the
   backtest says those bets lose.
6. **No other venue rescues it.** Venues on BTC/ETH/SOL fail by point 2 for
   this signal. The one venue with the same kind of stale anchor
   (Polymarket's 5-minute BNB market) does show the edge — and prices it
   away before a slow participant can reach it.

This genuinely made money for a stretch. "It never worked" would be as
false as "it works". The claim is about **mechanism** and **current
state**: it worked by harvesting an oracle's lag *and* by picking which
short-horizon moves would hold, not by forecasting prices in general, and
that combination is no longer profitable where this bot trades.

---

## How much of this is checked

Every number below carries one of three labels.

- **[verified 2026-09-14]** — re-derived from this repository's own
  backtest and price data by
  [`research/oracle_artifact_verification_2026_09_14.py`](../research/oracle_artifact_verification_2026_09_14.py).
  You can run it yourself (see [Reproducing](#reproducing)).
- **[measured 2026-09-01/02]** — computed by an earlier investigation whose
  scripts and cached data **no longer exist**. These were not re-derived
  here and cannot be from this repo. They are reported as measured, not
  as checked.
- **[measured 2026-09-15]** — sections 8-10. Computed from public BNB Chain
  reads (the Chainlink BNB/USD feed's own update records, and the
  prediction contract's per-round oracle ids) combined with this repo's
  price and round stores. The analysis scripts live in a working
  scratchpad that is **not** part of this repository, so the method below
  is reproducible but the exact numbers are not re-runnable from the repo
  alone. The one committed artefact of that work is
  [`research/prereg_A3_confirm_2026_09_15.py`](../research/prereg_A3_confirm_2026_09_15.py).

Where a re-derivation disagreed with the earlier measurement, both are
shown ([Appendix](#appendix-where-the-numbers-moved)). The disagreements
are small and none changes a conclusion, but a document asking you to
distrust a number should not ask you to trust its own on faith.

The earlier investigation's full write-ups exist as private claude.ai
artifacts. They will not resolve for anyone outside this project, and
this document does not depend on them.

---

## The strongest single result [measured 2026-09-01/02]

Polymarket runs a 5-minute "will BNB be up" market. Over its life it
changed **only its settlement rule**, which makes it a natural experiment:
same signal, same asset, same venue, different amount of oracle lag.

| settlement rule | period | windows | real win rate |
|---|---|---:|---:|
| fresh price snapshot (no lag) | 2026-03-13 – 08-06 | 5,099 | **49.2%** [47.9, 50.6] |
| 30-second TWAP | 2026-08-07 – 08-13 | 78 | 56.4% |
| 60-second TWAP | 2026-08-14 – 09-01 | 589 | **58.6%** [54.6, 62.5] |

These are **actual settled outcomes**, not simulations. When the rule
settles on a fresh price, the signal is a coin flip. The edge appears
exactly when a lagged anchor appears, by roughly the amount the lag
predicts. The first row doubles as five months of out-of-sample evidence
that, **in that period**, the signal had no forecasting value against a
real BNB price — consistent with the decay measured on the PancakeSwap
side, where the real-price part was worth about +4 points before late May
and nothing after.

The PancakeSwap-side evidence below is independent of this table and was
re-derived here. It points the same way.

---

## 1. The signal never predicted its own inputs [verified 2026-09-14]

The strategy fires on short BTC momentum (3s/7s/15s returns agreeing),
with ETH and SOL as confirmation. If that were a forecast, the bets should
also call the next five minutes of **BTC, ETH and SOL** themselves.

| bets | n | PancakeSwap (Chainlink) | BTC | ETH | SOL |
|---|---:|---:|---:|---:|---:|
| full tape | 2,015 | 59.6% | 49.8% | 49.7% | 49.5% |
| before 2026-05-26 (strongest period) | 1,599 | **61.5%** | 50.6% | 51.4% | 50.4% |
| from 2026-05-26 | 416 | 52.2% | 46.6% | 43.3% | 46.0% |

95% intervals on the full-tape BTC/ETH/SOL figures all span 50%
(BTC [47.6, 52.0], ETH [47.5, 51.9], SOL [47.3, 51.7]); so do the
strong-period ones. In the period when the PancakeSwap win rate was
61.5%, the bets called BTC 50.6% of the time.

It was never a forecast of its own inputs. What it had was a **BTC-to-BNB
lead-lag**: BTC moves, BNB follows seconds later, and BNB's settlement
price catches up later still. That component was real but small — about
+4 points above chance against real BNB spot before late May, gone by
summer ([section 9](#9-two-parts-not-one-measured-2026-09-15)) — and BTC does not lag itself,
so nothing transfers to venues that settle on BTC, ETH or SOL.

## 2. What it harvested: oracle staleness [verified 2026-09-14]

PancakeSwap Prediction V2 does not settle on a market price. The contract
stores a `lockOracleId` and `closeOracleId` per round and reads its
`lockPrice`/`closePrice` from an external oracle — Chainlink's BNB/USD
push feed. (Confirmed on-chain on 2026-09-15: the contract's `oracle()`
returns Chainlink's BNB/USD aggregator on BNB Chain, 8 decimals, and every
sampled round's stored `lockPrice` equals that feed's answer for the
round's `lockOracleId` exactly.)
A push feed publishes only when price moves past a deviation
threshold or a heartbeat expires, so **between updates the on-chain price
is stale by construction**. A signal that sees BNB moving before the feed
does can win on the stale number without predicting anything.

Scoring every bet twice — once on PancakeSwap's oracle prices, once on real
OKX BNB spot at the exact on-chain lock and close seconds — separates the
two:

| bets | n | settled on Chainlink | settled on real spot |
|---|---:|---:|---:|
| full tape | 2,015 | 59.6% | 52.3% [50.1, 54.6] |
| last 180 days | 1,052 | 56.8% | 50.9% |
| last 90 days | 377 | 51.5% | 44.4% [39.3, 49.5] |

Because both columns score **the same bets**, the gap can be tested
directly. McNemar's paired test counts the bets that won on the oracle but
lost on spot, against the reverse:

| period | won on oracle, lost on spot | reverse | premium | z |
|---|---:|---:|---:|---:|
| full tape | 161 | 37 | 6.5 pts | 8.8 |
| before 2026-05-26 | 130 | 27 | 6.8 pts | 8.2 |
| from 2026-05-26 | 31 | 10 | 5.2 pts | 3.3 |

**The oracle premium persisted; the real component is what died.** Before
late May the bets won 54.0% on real spot and 61.5% on the oracle. After,
46.1% and 52.2%. The premium shrank only slightly (6.8 → 5.2 points) and
stays significant in the late period on its own. The real-price edge went
from about +4 points above a coin flip to about 4 below.

Spot is priced at the actual on-chain instants: a round's lock is the
transaction that starts the next round, and its close is the one that
starts the round after. OKX quotes BNB in 0.1 USDT steps (~1.4 bps), so
about 5% of bets tie on spot; ties are excluded from the spot column.

## 3. It glided rather than broke [verified 2026-09-14]

An earlier write-up
([post_mortem_2026_06_11_findings.md](../research/post_mortem_2026_06_11_findings.md))
reported **"one statistically supported break at epoch ~484409
(2026-05-26)"**, and that was believed for a while. The smoothed series
does not show a single break:

| 60 days ending | real spot | Chainlink |
|---|---:|---:|
| 2026-02-15 | 56.5% | 64.5% |
| 2026-03-15 | 52.6% | 60.7% |
| 2026-04-15 | 52.6% | 59.9% |
| 2026-05-15 | 54.8% | 60.6% |
| 2026-06-01 | 52.6% | 59.3% |
| 2026-06-15 | 48.7% | 55.1% |
| 2026-07-01 | 48.3% | 54.9% |
| 2026-07-15 | **43.5%** | 49.9% |
| 2026-08-01 | 43.7% | 50.5% |
| 2026-08-15 | 46.9% | 54.5% |
| 2026-09-01 | 48.7% | 53.3% |

What this shows, stated no more strongly than the data allows: mild erosion
from February, a steep fall through June into a mid-July trough, and a
partial recovery. The real-price component was near break-even well before
the supposed break date. The "break" date also sits inside a fire-rate
drought (May 10-26) with only a few dozen bets, so there is little data at
that point to pin anything to.

The earlier investigation attributed the single-break reading to a 90-day
window that happened to end near a trough **[measured 2026-09-01/02]**.
Whatever the cause, the lesson is general: a single trailing window's
endpoints can move a win rate by several points.

## 4. The live record reconciles exactly [verified 2026-09-14]

The bot traded real money in two periods, each a few dozen settled bets —
far too few to measure a win rate on their own. What they can show is
whether live trading matched the backtest, and both do, bet-for-bet.

| run | when | stake | live WR | bets also in backtest | same side | backtest WR, same bets |
|---|---|---|---:|---:|---:|---:|
| soft launch | about a week, late May – early June 2026 | on-chain minimum | ~47% | all | all | **identical** |
| deployment | July – August 2026 | up to 0.1 BNB (`config.toml`) | ~43% | most | all | **~44%** |

Over the July-August calendar span the backtest's own win rate is 53.5%
(127 bets) — below PancakeSwap's breakeven of ~54.9% at flat stake and
~57.0% payout-weighted (from the post-mortem), after its 3% treasury fee.

So there was **no execution gap and no measurement gap**. The oracle
premium was captured live; it was simply not enough. Both runs fall
entirely after 2026-05-26, where the backtest's own win rate across all
416 bets is 52.2% — below breakeven. A handful of deployment bets have
no backtest counterpart; they are unexplained, but cannot account for the
result: the bets that do match already tell the story.

Two caveats a stranger should know. The live ledgers are not in this
repository (`var/` is gitignored), so this table can be re-run only on the
operator's machine. And earlier accounts said the bot "only ever traded
July-August"; the ledgers show the minimum-stake run in May-June as well.

## 5. Why PancakeSwap stopped paying [verified 2026-09-14, except where marked]

- **The fee.** PancakeSwap keeps 3% of every pool (the contract's
  `treasuryFee`). Breakeven on the payout multiples this strategy saw is
  roughly 55-57%.
- **The counterparties were not easy money.** Across the 1,747 wallets with
  200+ settled bets, the median win rate is **50.4%**; 6.4% exceed 55% and
  2.5% exceed 57%. The earlier investigation found the same median over
  1,116 wallets in a narrower window **[measured 2026-09-01/02]**. This was
  a field of near-coin-flip bettors, mostly bots, paying a 3% rake.
- **The pools shrank.** Median total pool per round fell from **2.48 BNB**
  (March 2026) to **1.25 BNB** (August), cutting the bets that pass the
  pool filter and the size each can carry.

## 6. Other venues [logic verified 2026-09-14; venue details measured 2026-09-01/02]

- **Anything settling on BTC, ETH or SOL** — perpetual futures including
  zero-fee venues, BTC prediction markets, Kalshi's 15-minute contracts —
  fails by section 1. There is no crypto forecast to collect, at any fee.
- **Real-time-settled BNB venues** fail by section 2: the real-price
  component is gone.
- **Kalshi** settles on real-time CF Benchmarks prices (no stale anchor)
  and its 15-minute crypto contracts showed essentially no realized volume.

## 7. Polymarket BNB 5-minute: the edge exists, and it is taken [measured 2026-09-01/02]

This is the one venue whose settlement has the same kind of lag, and the
most important part to have written down, because on paper it looks
profitable.

**The rule.** Since 2026-08-14 each window settles on a Chainlink 60-second
TWAP at close against the same TWAP at open — structurally a larger lag
than PancakeSwap's.

**On paper it pays.** Simulated at every real 5-minute boundary over nine
months (9,407 fires), the signal wins **58.7% [57.7, 59.7]**, stable month
to month. Polymarket charges takers 0.07·p·(1-p) per share, so entering at
53¢ costs 54.7¢ per $1 payout: a **54.7%** breakeven. (That arithmetic is
checkable by hand; the simulation is not re-derived here.)

**The model matched reality.** Against 5,318 real settled windows, the
settlement move reconstructed from OKX data tracked Polymarket's actual
`priceToBeat`/`finalPrice` with **correlation 0.996**, mean error
-0.001 bps, RMS 1.15 bps. OKX sits a steady +4.68 bps above the Chainlink
stream, identically at open and close, so it cancels. Of two readings of the
rule, end-versus-start matched real outcomes 94.8% of the time and a
whole-range reading only 84.0%. On 587 real fires the win rate was
**58.77% ± 3.98**, against 58.26% simulated for the same windows.

**Then the market collects it.** Trade prints show buyers of the signal's
side paying about **60-61¢ within the first ten seconds** of each window,
and the price-versus-outcome curve is calibrated: contracts bought under
48¢ won 33-43%, at 48-56¢ about 51-55%, at 62¢ and up about 80%. Taker
expected value at the prices buyers actually paid is **+0.10¢ per share
(t = 0.06)** — zero.

The detail that settles it: **when a cheap (≤55¢) early fill existed at
all, its median size was $6 and those windows won only 41.9%, against
66.6% when no cheap fill existed.** Cheap liquidity is almost pure adverse
selection. What remains (+5-9¢ per share, t ≈ 1.6-1.8) sits at 62¢+ in the
first seconds of the strongest fires — someone faster is already doing it.

Caveats that cut in both directions:

- The 60-second rule was only 2.5 weeks old when measured.
- These windows traded roughly $5-200 each, with about $66 at the touch.
  Even a real edge here would be tiny.
- The markets are flagged `restricted: true`; availability to US users was
  not verified.

## 8. The lag, measured directly [measured 2026-09-15]

Sections 2-5 measure what the lag was *worth*. This section measures the
lag itself, as seconds and basis points, by reading the feed's own update
records from the chain and lining them up against 1-second spot.

**How the feed behaves.** Chainlink's BNB/USD feed on BNB Chain updates on
a clock, not only on price moves: the median gap between updates is **33
seconds** (5th-95th percentile 32-34s), every month from December to
September. Between 0.5% and 6.6% of updates per month arrive early, after
moves of about 12 bps, which is what a deviation trigger of roughly 10 bps
on top of a 33-second heartbeat looks like.

**How stale the price is at lock.** The print a round locks on is 0-33
seconds old, **median 16 seconds**, and never close to the contract's own
300-second staleness limit. Allowing for about 2 seconds between the
market and the feed's timestamp, effective staleness is ~18 seconds.

**How far it drifts.** The print sits 4.9 bps (standard deviation) away
from spot at lock. Of that, 79% is explained by how far spot moved since
the print was made; the rest (2.3 bps) is the feed and one exchange
disagreeing at the same instant. Typical error grows with age, from 1.7
bps at 5-9 seconds old to 2.4 bps at 30-34. For scale, a 5-minute BNB move
has a standard deviation of 17.3 bps.

**Does staleness explain the premium?** Scored on the same bets and the
same seconds, four ways:

| scoring | win rate |
|---|---:|
| A. settled (what PancakeSwap paid) | 61.3% |
| B. real spot at the true lock and close | 52.5% |
| C. **staleness only** — real spot, sampled when each print was made | **62.5%** |
| D. disagreement only — the print with its staleness removed | 51.4% |

A perfectly accurate feed that is merely *late* reproduces **114%** of the
premium (95% CI 101-130%); the feed-versus-exchange disagreement
contributes nothing and slightly dilutes it. Almost all of it comes from
the lock side rather than the close side.

**An independent check on the magnitude.** Between the print and the lock,
spot moved an average of **+3.43 bps in the bet's favour**. Against the
density of 5-minute moves near zero, that predicts a premium of **7.9
points**; the measured premium on these bets is 8.7. The two agree without
being fitted to each other.

**Placebos.** Random bet directions give 50.0 / 49.7 / 49.9 on the three
scorings. Mirroring the timing to the far side of the lock gives 50.2.
Shifting lock and close by ±1-2 seconds moves the real-spot win rate by at
most 0.5 points. Assuming 0, 3, 6 or 10 seconds of feed reporting lag
gives 118%, 116%, 122%, 135% — the choice does not drive the result.

**Caveats.** Spot here is one exchange used as a proxy for the composite
the feed is built from; it sits about 6 bps away on a typical day and that
offset is removed day by day. It cancels within a round in any case, since
both ends use the same source. The 1-second store does not cover the ~7
seconds immediately before each lock, which excludes the freshest 19% of
prints from the comparison; the premium on those is smaller, as staleness
predicts.

## 9. Two parts, not one [measured 2026-09-15]

The sections above credit the oracle with the whole edge. That is not
right, and the correction matters more than the original claim.

**The crowd bets the gap too.** The displacement between the stale print
and spot is visible to anyone before the lock. A rule that mechanically
bets it — same decision instant, scored at real pools after the fee —
**loses money in every month measured**, including December to March when
pools were at their largest. Its win rate rises with the gap (53.7% to
63.8%) and the payout on that side falls in lockstep (1.87 to 1.61): the
pool prices the visible part almost exactly.

**What the strategy had that the crowd's version did not.** The two are
not separate bets. The strategy fired on gap rounds 2-3 times more often
than chance, agreed with the mechanical rule's side 85-88% of the time
when both fired, got the same final price (its side held 0.532 of the pool
against the rule's 0.530; payouts 1.79 against 1.77) — and still won far
more often. Splitting each side's win rate into an oracle part and a
real-price part:

| | settled | real spot | oracle part | real-price part |
|---|---:|---:|---:|---:|
| strategy, before late May | 61.5% | **54.0%** | +7.6 | **+4.0** |
| the mechanical gap rule, same period | 53.4% | **47.2%** | +6.3 | **−2.8** |
| strategy, after late May | 52.2% | 46.1% | +6.0 | −3.9 |
| the mechanical gap rule, after | 53.7% | 46.3% | +7.4 | −3.7 |

Both harvest the same ~6-8 points of oracle. The difference is the last
column: **visible gaps mean-revert on average, and the crowd's version
gives back half its oracle premium by betting them; the strategy's picks
kept moving.** Its BTC signal was telling it which short-horizon BNB moves
would hold. After late May that ability disappeared — 46.1% against the
crowd's 46.3% — and what remained was the crowded bet.

So the accurate statement is: **the staleness created the opportunity, the
crowd over-bet it, and the edge was knowing which instances were real.**
Both parts were needed. Neither alone was profitable.

## 10. What was tested afterwards, and found nothing [measured 2026-09-15]

- **Would bigger pools fix it?** No. Within each era, realised return
  against market size is flat (+0.034 per doubling, ±0.046). The informed
  money scales with the pool rather than being fixed: 99 wallets bet the
  gap's side persistently (73% of the time out of sample), they supply
  about 18% of that side's money with the top three at 5%, and their stake
  is a steady 10-11% of the pool at every size. Crowding was *tightest* in
  March, the largest-pool month. Even assuming the most favourable case,
  breakeven needs 1.5-3x the largest pools ever recorded here.
- **Can the lag be predicted better?** A pre-registered test of 11
  candidate features asked which visible gaps persist. **None passed**;
  the best was 0.24 after correction. No feature predicted real-spot
  persistence at all (all |ρ| ≤ 0.018). The features that did raise the
  settled win rate lowered the return, which is the crowd pricing them.
- **Fading the crowd?** The four features that came out backwards were
  then faded on the very data that selected them — a circular test that
  should flatter itself — and they still lost (−0.027 to −0.070 against a
  −0.026 baseline). Simply betting the minority side did better than all
  of them, and still lost.
- **One design was written and deliberately declined**:
  [`docs/prereg_A3_gap_persistence_2026_09_15.md`](prereg_A3_gap_persistence_2026_09_15.md)
  registers a single rule to be judged on rounds that did not exist when
  it was written. It was not run, for reasons stated in it: seven months
  of waiting, an expected null, and a prize worth tens of dollars a day on
  a venue whose pools are still shrinking.

## What would have to be true to make money with this again

Not a recommendation — the conditions, so nobody rediscovers them the
expensive way:

1. a venue that settles on a **stale** reference price,
2. with enough volume to matter,
3. where you are among the **first** to act each window, not at the
   prices after faster participants have moved,
4. and where the fee is below the lag premium.

On the venues examined, these were never all true at once.

---

## Reproducing

From the repo root, with a Python 3.13 environment
(`python bootstrap/common/python_setup.py`) and a `THE_GRAPH_API_KEY` in
`.env`:

```
python run.py --sync
python research/oracle_artifact_verification_2026_09_14.py
```

The sync downloads roughly 4.5 GB and must reach epoch 512,024. The
script then runs the full-tape backtest to epoch 512,022 (about 2.5
minutes), scores every bet against the oracle and against real spot, and
prints the tables in sections 1-4. The live-ledger section is skipped
without the operator's local ledger files.

## Appendix: where the numbers moved

Re-derivation used exact on-chain lock and close seconds; the earlier
investigation joined consecutive price records, which lands a few seconds
off. A ~5-second shift was known to move the spot win rate by 2-3 points,
which accounts for the differences. Every conclusion survives either way.

| quantity | measured 2026-09-01/02 | re-derived 2026-09-14 |
|---|---|---|
| full-tape backtest | 2,016 bets, 59.6%, +33.0 BNB | 2,015 bets, 59.6%, +33.1 BNB |
| BTC / ETH / SOL, full tape | 50.2 / 50.1 / 48.8% | 49.8 / 49.7 / 49.5% |
| BTC / ETH / SOL, strongest period | 50.9 / 51.3 / 49.6% | 50.6 / 51.4 / 50.4% |
| spot WR: full / 180d / 90d | 52.8 / 50.8 / 44.2% | 52.3 / 50.9 / 44.4% |
| McNemar, full tape | 168 vs 31 | 161 vs 37 |
| premium before / after 2026-05-26 | 7.0 / 6.0 pts (z = 4.2 after) | 6.8 / 5.2 pts (z = 3.3 after) |
| backtest WR over Jul-Aug live span | 52.4% | 53.5% |
| wallets with 200+ bets, median WR | 1,116 wallets, 50.4% | 1,747 wallets, 50.4% |
| live bets reconciled | every matched bet on the same side | same, exactly |

The one-bet difference in the backtest is unexplained; the data store has
been resynced since the original measurement.
