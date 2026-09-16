# Pre-registration: the A3 oracle-gap rule, to be confirmed on rounds that did not exist when this was written

**Committed 2026-09-15. The git commit date is the point of this file:** it proves the test was
specified before any of the data it will be judged on existed. Nothing here reports a result.

**If you are reading this cold, you need nothing else.** Everything required to run it is in this
file and in `research/prereg_A3_confirm_2026_09_15.py`.

---

## 1. What is being claimed, in one paragraph

PancakeSwap's Prediction rounds settle on a Chainlink BNB/USD price that updates about every 33
seconds, so at the moment a round locks, the settlement price is typically some seconds stale. That
makes a measurable gap between the published price and the real market price. Most such gaps are
already priced by the other bettors, and betting them blindly loses money. This registration tests
ONE narrow claim: that gaps built by a *consistent* run of one-second moves, rather than by a jump
or by noise, are worth betting on, and that this particular slice is not fully priced.

## 2. The rule (frozen, do not refit)

Among rounds where:

- the round settled (cancelled rounds excluded; a house/tie round counts as a LOSS);
- at the decision instant `d = startAt + 298` seconds (about 8 seconds before the round actually
  locks), the Chainlink price in force was ALREADY published and no heartbeat update is due before
  the lock, i.e. `d - updatedAt <= 24 s`;
- OKX BNB spot at `d` differs from that published price by at least **2 bps**, after subtracting the
  previous day's median level offset between the two sources (they sit a few bps apart for reasons
  unrelated to staleness);

bet **1 unit on the side spot has moved to**, but only when

- **A3 >= 0.823232**, where A3 is the share of one-second OKX moves between the price's publication
  and `d` that pointed in that same direction (zero-change seconds ignored; 0.5 if there are none).

**0.823232 is not a tuned number and must not be re-tuned.** It is the top-tercile boundary of A3
measured on 2026-09-14 over an older, separate stretch of data (epochs 437562-466781). Refitting it
on the confirmation data would reintroduce exactly the freedom this design removes.

**Scoring:** the mean return per unit, at the real final pools, after PancakeSwap's 3% fee, with our
own 0.05 BNB stake included in the chosen side (it dilutes the payout, and pretending otherwise
flatters the result by roughly 0.025 per unit). Interval: one-sided 95%, day-block bootstrap over
whole UTC days.

## 3. The confirmation set

- **Rounds with epoch strictly greater than 515944.** That was the end of the local data store on
  2026-09-15, the day this was written. Every earlier round is excluded, without exception.
- **DEPENDENCY, easy to miss: the daily sync must still have been running.** The confirmation set
  exists only because `PancakeBotDailySync` keeps appending rounds to `var/closed_rounds.jsonl` and
  the kline stores. If that stopped at some point, the set ends there, and the verdict is whatever
  the accrued data supports (very likely INCONCLUSIVE). Check the store's last epoch first.
- **One look, on 2027-04-08**, on whatever has accrued. Not "when it looks good", not "when there is
  enough" -- a fixed date chosen in advance, because a stopping rule that depends on the result is
  the failure this design exists to prevent. The runner script refuses to run before that date.
- The rule fires on roughly 37 rounds a day, so the expectation was about 7,600 bets by the date.
- **Sizing, measured rather than assumed.** On the older data the rule's per-bet return has a
  standard deviation of 0.8502, and resampling whole days rather than individual bets gives a
  design effect of 0.90 (day-clustering slightly *reduces* the variance here, it does not inflate
  it). At 7,600 bets the standard error is therefore about 0.00925, which gives:
  - 88% power to detect the claimed +0.026 at one-sided 5%;
  - and an upper bound landing near +0.015 when the truth is zero, which is what the KILL
    threshold in section 5 is set against.

## 4. How to run it

    python research/prereg_A3_confirm_2026_09_15.py

It needs: this repository, `var/closed_rounds.jsonl` and `var/bnb_spot_prices.jsonl` from the sync,
Python with `requests`, and outbound internet for public BSC RPC reads (no keys, no wallet, nothing
is signed or sent). It fetches the Chainlink oracle rounds it needs and caches them under
`var/`, resumable if interrupted. It prints the population, the rule's mean return with bounds, the
two negative controls, and one of the three verdicts below.

## 5. Verdicts, decided in advance

| Outcome | Condition | Consequence |
|---|---|---|
| **PROMOTE** | one-sided 95% lower bound above 0 | A research finding about a mean return. It authorises NOTHING to be deployed. |
| **KILL** | point estimate at or below 0, OR upper bound below +0.026 | The claim is dead: the effect it was built on has been excluded. |
| **INCONCLUSIVE** | anything else, including fewer than 2,600 rule bets | Record UNRESOLVED and **stop**. No extension of the window, no re-analysis of these rounds. A revisit needs a NEW registration whose data starts after that registration's date. |

**Why the KILL threshold is +0.026 and not half of it.** A kill rule has to be reachable with the
sample the date actually produces, or the test cannot fail cleanly and a null becomes
"inconclusive" by construction. Excluding +0.026 (the claimed effect) needs about 2,600 bets, which
is 70 days. Excluding half of it (+0.013) needs about 10,400 bets, or 279 days, which is past this
date: had that threshold been kept, a true null would have returned INCONCLUSIVE rather than KILL.
With the threshold as set, a true null returns KILL about 88% of the time and PROMOTE 5%, and a true
+0.026 returns PROMOTE about 88% of the time. The narrow band between them, roughly +0.011 to
+0.015, is the only genuinely inconclusive region.

## 6. Provenance, and an honest prior

This is the second registration. The first (`prereg_gap_persistence_2026_09_14.md`, sha256
`57ea6bb7...f2b6`, kept in the working scratchpad and not committed) tested eleven features on the
older data and **all eleven failed**. A3 was one of them.

A3 is back because of a defect in that first registration, which was mine: it scored features with a
RANK statistic while the rule it was meant to authorise is a DIFFERENCE IN MEAN RETURN. For a
win/lose variable with a heavy payout tail those are different quantities and can disagree. They
did: A3 scored -0.0293 (one-sided p = 0.997) on the rank statistic while its rule returned **+0.0260
per unit** over 3,876 bets on the same data.

**The prior is low, and the sharpest version of that is this:** under the corrected statistic, A3
would STILL have failed the first gate. +0.0260 over 3,876 bets is about 1.90 standard errors
(SE 0.0137), a one-sided p of 0.029, which the correction for testing eleven features turns into
0.32. The
mean-return framing does not rescue A3; it only makes the result visible instead of invisible.

What is genuinely in its favour: A3's **direction and rule shape were fixed before anyone looked**,
and only the statistic changed afterwards. That is the strongest form such a claim can take short of
out-of-sample evidence, and it is why this was written down instead of mined further.

Expect a null. The older data it came from is contaminated for this question and contributes
nothing. Two other slices (epochs 466782-474086, and 475312-515944) were deliberately left unspent.

## 7. Status: DECLINED, deliberately, on the day it was written

**This test was not run, and that was a decision rather than an omission.** It was designed,
committed, and then declined the same day, on these grounds:

- **The wait is seven months** for a single look, and it cannot honestly be shortened. At roughly 37
  bets a day, separating +0.026 from zero needs thousands of bets; an early look would need an
  observed return around nine times the claimed effect to stop, which would signal a bug rather than
  a windfall.
- **The prize is small and shrinking.** At the pool sizes of September 2026 -- a median of about
  0.95 BNB and still contracting -- a rule of this kind is worth on the order of $60 a day before
  costs at our stake, and our own stake dilutes the payout materially at those sizes.
- **The prior is low**, for the reasons in section 6: the feature failed its first registered test
  and would have failed it under the corrected statistic too.

Seven months of waiting for an expected null on a shrinking venue was not worth the option, so it
was declined. **The design is left intact and executable** in case that judgement ever looks wrong:
the rule, the cut-point, the date and the verdicts all still stand as written, and the runner still
works. If anyone does revive it, the confirmation set is still defined as rounds after epoch 515944,
and the date is still 2027-04-08 -- moving either afterwards would forfeit the only thing this
document is for.

What would be a mistake is running it late, seeing an unwelcome number, and then deciding the window
should have been longer.

## 8. Scope notes

- The runner implements the primary rule and its controls. The first registration also listed ten
  secondary features "reported, never promoting"; they are not implemented here, which costs
  nothing, since they could never promote anything.
- A PROMOTE verdict is about a mean return in a research sample. Whether the venue is worth trading
  is a separate question, and as of 2026-09-15 the answer was no: median pools had fallen to about
  0.95 BNB and were still shrinking, and a comparable rule was worth on the order of $60 a day at
  our stake size before costs.
- Verification: `research/prereg_A3_confirm_2026_09_15.py` at the time of this commit had
  sha256 `8f96e3755554aacb14cb4d43c6a3bd2e3ff97311e4eb46bbb5c6fda45572999b`.
