# Monitoring & autonomous operation

The bot is governed unattended by the **weekly monitor state machine**
(`research/weekly_monitor_state_machine.py`), run by cron on the VM every
Sunday 06:00 UTC. It syncs data, re-evaluates the canonical strategy on the
trailing windows, and is the sole authority over the live unit: it disables
the bot when the strategy is demonstrably losing and re-enables it when the
strategy is demonstrably working again. No manual arming step exists
(2026-07-09 user decision, re-affirmed 2026-07-17).

## The weekly triggers (pinned)

| trigger | condition (trailing 1-week canonical window) | action |
|---|---|---|
| POSITIVE | WR > 0.55 AND raw permutation p_upper < 0.10 AND n ≥ 10 AND risk-off backtest net PnL (gas-inclusive) > 0 | bot disabled → **enable + start** (writing the cooldown-override flag first if the bot went down mid-suspension, so it releases on its first paused round). bot enabled but breaker-suspended → **write the override flag** (release). |
| NEGATIVE | WR < 0.45, OR 3 consecutive weak weeks (weak = p_upper > 0.5 or n < 10) | **stop + disable** entirely. |

The 2-week window, latest-100 WR, and Šidák-corrected p are computed and
reported but do not gate actions. Artifacts:
`var/strategy_review/weekly_monitors/<YYYY-MM-DD>/decision.json`; weekly
state (consecutive-weak counter, history) in `.../state.json`. State
advances once per ISO week — re-runs in the same week are idempotent.

## Cron installation (reproducible)

The crontab calls the tracked wrapper — this is the complete recipe a fresh
install needs (cronie first: minimal AlmaLinux images may lack it):

```bash
dnf install -y cronie && systemctl enable --now crond
( crontab -l 2>/dev/null | grep -v run_weekly_monitor ; \
  echo '0 6 * * 0 /root/pancakebot/bootstrap/linux/run_weekly_monitor.sh >/dev/null 2>&1' ) | crontab -
```

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
* **Blind weeks self-heal, and blindness cannot persist silently**: a
  failed/hung sync or stale data (newest lock > 36 h — a stalled indexer
  can "succeed" without new data) blocks that week's positive actions,
  alerts loudly, freezes the weekly counters, and retries next Sunday.
  Data stores are append-only; a missed week back-fills automatically.
  After 3 consecutive blind weeks with the bot enabled, the monitor
  disables it — it never bets for months unevaluated.
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
2. **Shadow ledger** (intra-week): at cooldown expiry the suspension is
   extended unless the shadow (counterfactual) ledger shows genuine
   recovery — ≥ 3 settled shadow fires, cumulative PnL ≥ 0, and
   hypothetical bankroll above 85% of its rolling peak. A bleeding strategy
   stays suspended indefinitely.
3. **Weekly negative trigger** (Sunday): a 1-week WR < 45% disables the bot
   outright, suspension or not (live-validated 2026-07-12).

Worst case for one bad episode: the breaker suspends after ~15% drawdown
from the (reseeded) bankroll plus at most one max-size bet of overshoot
(~0.1 BNB + gas) — roughly 20% of bankroll — after which the shadow ledger
holds the suspension, and the following Sunday disables outright if the
week's WR fell below 45% (or after 3 weak weeks). A slow bleed that
evades both weekly legs stays bounded per-episode by the breaker: every
resumption requires fresh statistical evidence (a new positive trigger at
p < 0.10, or genuine shadow recovery), so repeated 20% episodes each need
independently "good-looking" weeks to precede them.

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
