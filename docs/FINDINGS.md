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
2. **It never predicted crypto prices.** Apply the same bets to the next
   five minutes of BTC, ETH or SOL — the assets its signal is built from —
   and they do no better than a coin flip in any period: about 50%
   overall and in the strongest period, and lower since late May.
3. **What PancakeSwap paid for was oracle lag.** Rounds settle on a
   Chainlink *push* feed, which is stale between updates by design. Scored
   against real BNB spot prices at the same instants, the same bets win
   52.3%, not 59.6%. The ~6-point difference is the oracle.
4. **That stopped being enough.** The real-price component eroded through
   2026 until the total fell below PancakeSwap's ~55-57% breakeven. Both
   live runs fall inside that late period, and both reconcile bet-for-bet
   with the backtest: the bot did exactly what the backtest says, and the
   backtest says those bets lose.
5. **No other venue rescues it.** Venues on BTC/ETH/SOL fail by point 2.
   The one venue with the same kind of stale anchor (Polymarket's 5-minute
   BNB market) does show the edge — and prices it away before a slow
   participant can reach it.

This genuinely made money for a stretch. "It never worked" would be as
false as "it works". The claim is about **mechanism** and **current
state**: it worked by harvesting an oracle's lag, not by forecasting, and
that is no longer profitable where this bot trades.

---

## How much of this is checked

Every number below carries one of two labels.

- **[verified 2026-09-14]** — re-derived from this repository's own
  backtest and price data by
  [`research/oracle_artifact_verification_2026_09_14.py`](../research/oracle_artifact_verification_2026_09_14.py).
  You can run it yourself (see [Reproducing](#reproducing)).
- **[measured 2026-09-01/02]** — computed by an earlier investigation whose
  scripts and cached data **no longer exist**. These were not re-derived
  here and cannot be from this repo. They are reported as measured, not
  as checked.

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
predicts. The first row doubles as five months of out-of-sample proof that
the signal has no forecasting value against a real price.

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

It was never a crypto forecast in any era. It was a **BTC-to-BNB
lead-lag**: BTC moves, BNB's settlement price catches up later. BTC does
not lag itself, so nothing transfers to venues that settle on BTC, ETH or
SOL.

## 2. What it harvested: oracle staleness [verified 2026-09-14]

PancakeSwap Prediction V2 does not settle on a market price. The contract
stores a `lockOracleId` and `closeOracleId` per round and reads its
`lockPrice`/`closePrice` from an external oracle — Chainlink's BNB/USD
push feed. (The ABI in this repo confirms an external oracle and the
stored prices carry Chainlink's 8 decimals; that it is specifically
Chainlink is PancakeSwap's documented design, not checked on-chain here.)
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
