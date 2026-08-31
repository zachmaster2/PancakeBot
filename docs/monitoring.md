> ## PARTLY STALE as of 2026-08-31
>
> The Sunday cron and the VM it ran on **no longer exist**, and the bot is
> stopped and disabled. The Sunday Discord dead-man contract is **retired**
> — silence is no longer evidence of anything. See
> `docs/alerting_retirement_2026_08_31.md`.
>
> **Thresholds below are out of date.** The positive leg is now
> `BREAKEVEN_WR = 0.56` (was 0.55), and the negative leg no longer uses a
> raw `WR < 0.45` floor — it fires only when the Clopper-Pearson 95% upper
> bound on the observed win rate falls below `BREAKEVEN_WR`. Treat the
> table below as a record of the rules as they stood before 2026-08-30.

# Monitoring & autonomous operation

The bot is governed unattended by the **weekly monitor state machine**
(`research/weekly_monitor_state_machine.py`), run by cron on the VM every
Sunday 06:00 UTC. It syncs data, re-evaluates the canonical strategy on the
trailing windows, and is the sole authority over the live unit: it disables
the bot when the strategy is demonstrably losing and re-enables it when the
strategy is demonstrably working again. No manual arming step exists
(2026-07-09 user decision, re-affirmed 2026-07-17).

## The weekly triggers (pinned)

| trigger | condition (canonical windows) | action |
|---|---|---|
| POSITIVE | evaluated on exactly ONE window — the trailing 1-week window when it has n ≥ 10 fires, else falling back to the trailing 2-week window (the fires floor is an information floor, not a time floor; both windows starved → cannot fire). The spent window must pass all four legs: WR > 0.55 AND raw permutation p_upper < 0.10 AND n ≥ 10 AND risk-off backtest net PnL (gas-inclusive, run over that same window) > 0 | bot disabled → **enable + start** (writing the cooldown-override flag first if the bot went down mid-suspension, so it releases on its first paused round). bot enabled but breaker-suspended → **write the override flag** (release). |
| NEGATIVE | WR leg on the spent positive window (same window the positive legs read): WR < 0.45 (unevaluable when both windows starve — weak-booking covers that), OR 3 consecutive weak weeks — weak is judged on the spent window: its p_upper > 0.5, or both windows starved (n < 10 fires even across 2 weeks). A 1-week-starved week with a strong 2-week window is NOT weak | **stop + disable** entirely. |

The latest-100 WR and Šidák-corrected p are computed and reported but do
not gate actions; the 2-week permutation stats are always reported and
gate only the positive fallback (its 2-week backtest runs only when the
fallback is engaged). `decision.json` records the spent window as
`triggers.trigger_window` (`1w` / `2w_fallback` / `none`). Artifacts:
`var/strategy_review/weekly_monitors/<YYYY-MM-DD>/decision.json`; weekly
state (consecutive-weak counter + its prior-week baseline, history) in
`.../state.json`. State books once per ISO week — a same-week re-run
recomputes the week's booking from the baseline and overwrites it
(idempotent, never double-advancing). That applies to every per-week
counter: `consecutive_weak` and `evidence_gap_streak` both go through the
same booking guard.

### Config changes and Sunday runs are now coupled

Since 2026-08-24 the canonical fire stream reads the DEPLOYED config, so
changing a strategy knob in `config.toml` changes `n`, `WR` and `p_upper`
— they are not comparable across that boundary. The monitor records a
`strategy_fingerprint` in `decision.json`, in `state.json` and on each
`history` entry, and raises a **CONFIG CHANGED** banner on the first
Sunday whose fingerprint differs from the previous one, naming the keys
that moved.

Practical consequences after any config change:

* The **next Sunday's report is an epoch boundary, not a trend.** Read
  `n`/`WR`/`p_upper` against that week onward, not against the weeks
  before it.
* `consecutive_weak` is deliberately **NOT** reset by a config change.
  Resetting would delay the protective disable. The bound is small: one
  config change can contribute at most one spurious weak week, and three
  are needed to disable.
* If the fingerprint is ever absent (a fresh `state.json`), the
  comparison is a no-op and no banner fires. `scripts/seed_monitor_
  fingerprint.py` exists because of exactly that gap on 2026-08-30; if it
  recurs, backfill from the run's own `risk_off_config.toml` copy rather
  than from memory.

### Evidence gates — two ways an enable is silently withheld

Both gate the POSITIVE path only; neither can block a disable, and both
report in `decision.json` and in the Sunday Discord message. If the bot is
not being re-enabled and no trigger explanation is obvious, check these
first — they suppress an action that otherwise never happened.

| gate | fails when | field |
|---|---|---|
| **Frozen window** | the newest fire is older than 96h. The stats are then re-scoring bets that already happened and describe a week the bot did not trade in — the 2026-08-17 pool outage shape: rounds fresh, fires frozen. | `fire_fresh`, `newest_fire_age_h` |
| **Evidence gap** (density) | *any* drought longer than 96h sits inside the SPENT window — internal gaps as well as the trailing one. The frozen-window check tests only the newest fire, so 18 fires aged 9–13 days plus one fire 2h old passes it; this one does not. | `evidence_gap_ok`, `max_window_gap_h`, `max_internal_gap_h`, `trailing_gap_h`, `gap_bound_h` |

The gap rule is a strict generalisation of the freshness rule (same
constant, same units: the spent window ends at the newest fire, so its
trailing gap *is* the freshness quantity), which is why a blocked run
reports `enable_BLOCKED_frozen_window` before `enable_BLOCKED_evidence_gap`
— weakest precondition first.

Both blocks alert as **POSITIVE ACTION SUPPRESSED**, never as a routine
no-op. Consecutive suppressed enables raise `evidence_gap_streak`; at 2 the
message escalates. Note what that counter means: it follows *suppressed
enables*, not sparse evidence — a week whose triggers did not fire resets
it to 0.

Expect the gap rule to block in EPISODES, not at a steady rate. Measured
over the last 70 evaluable days it blocks 11 of them — but as one
contiguous run (2026-08-12 → 08-22) as a single drought ages through the
window, not as 11 independent Sundays. A run of blocked Sundays is one
event, and the fix for a persistent one is the fire RATE, not the bound
(see the derivation on `FIRE_STALE_MAX_AGE_S`).

## Cron installation (reproducible)

The crontab calls the tracked wrapper — this is the complete recipe a fresh
install needs (cronie first: minimal AlmaLinux images may lack it):

```bash
dnf install -y cronie && systemctl enable --now crond
( crontab -l 2>/dev/null | grep -v run_weekly_monitor ; \
  echo '0 6 * * * /root/pancakebot/bootstrap/linux/run_weekly_monitor.sh >/dev/null 2>&1' ) | crontab -
```

The cron fires **daily**; the wrapper gates the work: Sundays run in full,
Mon–Sat are silent no-ops unless a `retry_pending` marker exists — a blind
Sunday (sync failure or stale data) arms daily makeup attempts until one
recovers or the next Sunday supersedes it. A recovered attempt is keyed to
the missed Sunday (Sundays are the last ISO day, so calendar keying would
consume the following week's state advance), runs the full evaluation —
triggers included, any day of the week — and reports "recovered after N
failed attempts".

The crontab line deliberately carries no logfile redirect: cron's shell
opens redirects before the command runs, so a redirect into the gitignored
`var/` tree would silently kill every run whenever that tree is missing.
The wrapper (`bootstrap/linux/run_weekly_monitor.sh`) owns its logging
instead — it self-heals the log dir then `exec`-appends to `cron.log`
(capped ~2 MB), alerts and runs logfile-less if the dir is unwritable,
holds a `flock` so runs never overlap, sources
`/etc/pancakebot/alerts.env` (webhooks only — the wallet key never enters
the monitor process), and curls a Discord failure alert on any nonzero
exit. `run.py --sync` inside the monitor reads `THE_GRAPH_API_KEY` from
the repo-root `.env`.

Manual runs: `run_weekly_monitor.sh --dry` any day = full compute +
artifact + Discord message with zero mutation (no `--apply`, no sync).
Don't run the `--apply` form by hand outside Sundays unless you mean to
consume that ISO week's state advance.

## The walk-away contract

The design goal: the operator can ignore the system for months and trust
that either it is working or they would know. Concretely:

* **Every Sunday ~06:15–07:00 UTC a Discord message arrives on the general
  webhook** — on no-change weeks, on state changes (⚠️ prefixed), on blind
  weeks (⚠️ SYNC FAILED / DATA STALE, evaluation retried next week), on
  degraded actions (❌ enable failed / systemctl unresponsive), and on
  monitor crashes (❌, from the monitor's own crash handler and/or the
  wrapper's curl fallback). Delivery is verified (HTTP status + retry);
  an undelivered alert exits nonzero so the wrapper's fallback fires.
* **A Sunday with NO message means the system itself is broken** — VM down,
  cron dead, venv unbootable, Discord unreachable, or the webhook deleted.
  That is the one condition requiring a manual look (`cron.log` on the VM
  says which). While the bot is RUNNING, unit lifecycle alerts — STARTED /
  CRASHED / STOPPED / SUPPRESSED_FAST_CRASHLOOP — additionally fire via
  `pancakebot-notify@` on the live-alerts webhook (live-validated
  2026-07-09/12).
* **Blind Sundays retry daily and cannot persist silently**: a failed/hung
  sync or stale data (newest closed **round** > 36 h — a stalled indexer
  can "succeed" without new data; the newest *fire* is not the yardstick,
  it lags days in normal droughts) blocks positive actions, alerts loudly,
  and arms daily
  makeup attempts (one-line ⚠️ per failed day — the spam is itself a
  heartbeat). A week counts as blind only if Sunday AND every retry
  through Saturday failed. Data stores are append-only; recovery
  back-fills automatically. After 3 consecutive fully-blind weeks with
  the bot enabled, the monitor disables it — it never bets for months
  unevaluated.
* **Unit-state drift heals weekly**: an enabled-but-dead unit is restarted
  (⚠️ alert); a running-but-disabled unit is covered by the disable path.
  To deliberately stop the bot, DISABLE it — that is the operator signal
  the monitor respects.
* **Reboots are safe**: crond, chronyd, and (when enabled) the bot unit are
  all `systemctl enabled`; a VM reboot restores the whole stack. Unattended
  security updates (dnf-automatic, security-only) run with `reboot = never`.

## Protective chain while the bot is enabled

A (false-)positive re-enable is bounded by three independent layers:

1. **Drawdown breaker** (intra-round): ≥ 15% drawdown from the rolling-7d
   peak suspends betting for 288 rounds (~24 h). Every release path reseeds
   the peak baseline to the current bankroll, so a re-enable after a long
   gap can never trip on a months-old peak.

   *Since the 2026-08-24 stake halving:* the CAPITAL ceiling is unchanged —
   both the threshold and the per-bet loss are bankroll fractions, so the
   worst-case episode actually improves (20.0% → 17.65%). What changed is
   DETECTION SPEED: reaching the 15% threshold now takes 6 consecutive
   max-losses instead of 3, i.e. roughly 2.4 days instead of 1.2. A slow
   bleed is caught later in wall-clock time while costing no more capital.
2. **Shadow ledger** (intra-week): at cooldown expiry the suspension is
   extended unless the shadow (counterfactual) ledger shows genuine
   recovery — ≥ 3 settled shadow fires, cumulative PnL ≥ 0, and
   hypothetical bankroll above 85% of its rolling peak. A bleeding strategy
   stays suspended indefinitely.
3. **Weekly negative trigger** (Sunday): a 1-week WR < 45% disables the bot
   outright, suspension or not (live-validated 2026-07-12).

Worst case for one bad episode: the breaker suspends after ~15% drawdown
from the (reseeded) bankroll plus at most one max-size bet of overshoot
(~0.05 BNB + gas) — roughly 17.6% of bankroll — after which the shadow ledger
holds the suspension, and the following Sunday disables outright if the
week's WR fell below 45% (or after 3 weak weeks). A slow bleed that
evades both weekly legs stays bounded per-episode by the breaker: every
resumption requires fresh statistical evidence (a new positive trigger at
p < 0.10, or genuine shadow recovery), so repeated ~17.6% episodes each need
independently "good-looking" weeks to precede them.

### Reading `btPnL` after 2026-08-24

The monitor copies `config.toml` verbatim and overrides only four keys —
`max_bet_bnb_*` is NOT among them. The stake halving therefore HALVES every
`btPnL` figure the Sunday report prints (on the 2026-08-23 data: 7d
0.2277 → 0.1142, 14d 0.5234 → 0.2621) for reasons that have nothing to do
with strategy performance. Do not read the step down as decay. The backtest
leg is a SIGN test (`> 0`), and a sign flip would need gross PnL inside
~0.002–0.008 BNB against typical values of 0.11–0.26, so the trigger
behaviour is effectively unaffected — only the printed magnitude moves.

## Research tripwire monitor (separate tool)

`research/monitor_2026_06_12.py` is the research-side regime monitor from
the search-closure posture (pre-registered tripwires, PINNED 2026-06-12 —
do not tune):

| wire | statistic | trip condition |
|---|---|---|
| T1 | latest 500 canonical risk-free bets: flat-stake PnL vs the market-implied null (win_i ~ Bernoulli(1/payout_i)) | p < 0.01 |
| T2 | contrarian @ lock−6s, threshold 0.4, trailing 15 days: PnL deficit vs permutation null | permutation z > 2 AND n > 500 |
| T3 | perp tape imbalance (1m/5m/15m × cutoff 2s/6s), trailing 15 days: deficit vs permutation null | any Šidák-adjusted two-sided p < 0.01 |

A tripped wire is a "rerun the Phase-0 gauntlet on fresh data" signal,
never a deploy signal — raw p<0.01 cells have repeatedly dissolved under
full coverage and sweep discounts (p=0.0016 → p=0.42 in the OKX probes).
Any revival still owes cross-validation, holdout, permutation nulls, and
multiple-comparison discounts. Note OKX trade-tape retention (~3 months):
run it at least monthly if you care about T3 continuity, or accept the
segment guard shrinking T3 cells. Never run two tape-appending monitors
concurrently (appends are unlocked; torn lines corrupt).

Output: `var/strategy_review/monitor_runs/<YYYY-MM-DD>/findings.json` +
`summary.txt`; digest ends `VERDICT: quiet` or `VERDICT: TRIPPED: ...`.
