# The Sunday dead-man contract is retired — 2026-08-31

**Read this before concluding that silence means health.**

On **2026-08-31** the PancakeBot project moved to *data collection only*.
The live unit was stopped and disabled on 2026-08-30, and the Frankfurt VM
(`139.59.209.230`, `pancakebot-deployment-node`) was destroyed on
2026-08-31. Verified gone: 100% packet loss and SSH connection timeout,
against a 113 ms ping and a working SSH session earlier the same day.

This note exists because the alerting guarantee that operated until that
date has ended, and the way it ended is exactly the way this project keeps
getting hurt: **a thing stopped, and the stopping looks like quiet.**

---

## What the Sunday contract guaranteed

A cron job on the VM (`0 6 * * *`, running
`bootstrap/linux/run_weekly_monitor.sh`) drove
`research/weekly_monitor_state_machine.py`. Every Sunday it evaluated the
strategy and **posted a Discord message regardless of outcome** — enable,
disable, or no-op. Delivery was verified, not fire-and-forget.

The guarantee was therefore not "you will be told when something is
wrong". It was stronger and more useful:

> **A Sunday with no Discord message meant the box, the cron, or the
> webhook was dead.**

That inversion is the whole value. It made silence a *positive signal*
rather than an absence, and it defined the single condition that required
a human to go and look. Everything else could be read from the message.

**Last message: Sunday 2026-08-30.** The final run booked
`action=none`, `consecutive_weak=1`, on a window of n=50 WR=0.52 p=0.625.
A no-op — the contract working normally, right up to the end.

---

## It is retired. From 2026-08-31, silence means nothing.

There is no Sunday message any more, and **its absence is no longer
evidence of anything at all.** Nobody should wait for one, and nobody
should read a quiet Sunday as either good or bad news.

Why retired: the guarantee was hosted entirely on the VM. Cron, the
state machine, and the webhook credentials all lived there. When the box
went, the contract went with it. It was not switched off because it stopped
being useful — it was made impossible by the decision to retire the host.

The bot it governed is also stopped and disabled, so there is no longer a
live position for a weekly decision to protect.

---

## What replaces it, and exactly what that is worth

Two Windows scheduled tasks, registered as SYSTEM so they run with nobody
logged in and no stored password:

| task | role |
|---|---|
| `PancakeBotDailySync` | does the work. Daily 06:30 + at startup, `StartWhenAvailable` for missed windows. |
| `PancakeBotSyncWatchdog` | checks the work happened. Own trigger at 09:00 + at startup. |

> **Amended 2026-09-01.** This section originally credited the sync task
> with "3 retries at 20-minute intervals". That was wrong and has been
> removed: `RestartCount` fires on *unexpected termination*, not on a
> non-zero exit, so nothing retried when the sync failed on 2026-09-01.
> The behaviour is now honestly one-shot and the retry is **tomorrow's
> run** — the sync is idempotent and resumable, and the 171.6-day horizon
> leaves months of margin.

**Two tasks, not one, and that is the point.** A task cannot report its own
absence. If the sync task is disabled, deleted, or never fires, it emits
nothing — and nothing is exactly what a healthy idle system emits. The
watchdog runs on an independent trigger and fires precisely in the case the
sync cannot report on: its own silence.

Health is a **positive, timestamped assertion** written on every attempt
(`var/sync_health.json`), measured from last **success**, never last
attempt — a job that runs daily and fails daily is not healthy. The attempt
is recorded separately so *"ran and failed"* stays distinguishable from
*"never fired"*: different diagnoses that send you to different places.

When the sync stops succeeding, a file appears on the Desktop whose
**filename carries the diagnosis**:

    PANCAKEBOT_SYNC_STOPPED_3_DAYS_AGO.txt
    PANCAKEBOT_SYNC_HAS_NEVER_RUN.txt

It clears automatically on the next success, so its presence always means a
live problem. A Desktop file was chosen over the alternatives deliberately:
a log needs you to know to look, which is the failure mode itself; a toast
only fires while someone is logged in and cannot represent a multi-day
state; Discord needs the network, which is one of the things that breaks.

### Added 2026-09-01 — two more layers, both from a real failure

This note was written the day the VM was destroyed. The replacement has
since grown two layers, both because the first genuinely unattended run
failed and exposed a hole.

**Orphan detection — "started and never finished" is now a state.** On
2026-09-01 the sync fired at 06:30, died mid-run, and the health file still
read `consecutive_failures: 0` with a stale `last_exit_code: 0`. A FAILED
run was indistinguishable from one that had not happened. The specific
cause was fixed, but the general rule matters more: if `last_attempt` is
newer than **every** recorded outcome and 2 hours have passed, a run began
and vanished — whatever the mechanism. That needs no theory about how the
wrapper died, so it holds for failure modes nobody has anticipated. It
raises its own marker, deliberately not worded as staleness:

    PANCAKEBOT_SYNC_RUN_VANISHED_2_HOURS_AGO.txt

because the task **is** firing; it is dying without saying so, and calling
that "stopped" sends the reader to the wrong diagnosis.

**Integrity assertion — "did it run CORRECTLY", not just "recently".** The
watchdog originally measured only recency, which catches a stopped sync and
nothing else. A sync running happily every day while silently losing
records, picking up CRLF, or developing a gap would refresh the timestamp,
report success, and raise nothing. It now compares each run against a
persisted baseline and fails on: a record count or last epoch **decreasing**
(append-only stores never shrink), any CRLF, contiguity breaking beyond the
8 known-unfetchable epochs, duplicates, non-ascending order, or the five
stores drifting apart. Marker names carry the diagnosis —
`INTEGRITY_SHRANK`, `INTEGRITY_CRLF`, `INTEGRITY_GAP`.

**And a forensic channel.** `Microsoft-Windows-TaskScheduler/Operational`
was enabled 2026-09-01, because the 06:30 diagnosis only worked since the
sync happened to log up to the moment it stopped. See
`docs/task_scheduler_forensics.md` — and note it is **machine state, not
repo state**: a Windows reset or new machine loses it silently.

### The honest limit

**Neither task notices anything while the machine is off.** No local
watchdog can observe its own absence. If the machine is shut down for a
week, nothing runs, nothing alerts, and nothing knows.

What the replacement guarantees is narrower than what the Sunday contract
guaranteed, and the difference should not be papered over:

| | Sunday contract | Desktop marker + watchdog |
|---|---|---|
| covers a dead box | **yes** — silence was the signal | **no** — a dead box is silent |
| needs the network | yes (webhook) | no (marker is local) |
| needs someone logged in | no | no to detect, yes to *see* |
| visible when? | weekly, pushed | on next use of the machine |

The replacement makes a stall **visible the moment the machine is used
again**. It does not detect one while the machine is off. That is a real
reduction in coverage, accepted knowingly because the alternative — keeping
a $25/month host alive purely to watch a paused project — was the cost the
project was being retired to avoid.

### Why a stalled sync still matters even with the bot stopped

The five store files are now the **only** surviving copy of data that
cannot be refetched. OKX serves 1s klines for roughly **171.6 days**;
anything older exists nowhere else in the world. A sync stalled longer than
that horizon is not a delayed sync, it is **permanent data loss**. That is
why a paused project still has a daily job and an alarm.

---

## Known limitation of EVERY historical CONFIG CHANGED banner

`strategy_fingerprint()` captures **strategy** config only — pool filter,
stake caps, cutoffs, lookbacks, fee, min-bet threshold. It contains **zero
monitor thresholds**.

The CONFIG CHANGED banner in the weekly artifacts therefore **could never
have flagged a change to the decision rule itself.** This is not a
hypothetical: the final artifact of 2026-08-30 names the stake caps and the
pool filter —

> `max_bet_bnb_btc_primary: 0.1 -> 0.05, max_bet_bnb_eth_sol_fallback: 0.1
> -> 0.05, min_pool_bnb_at_cutoff: 1.5 -> 1.25`

— and would **not** have named `BREAKEVEN_WR 0.55 → 0.56`, nor the
replacement of the raw win-rate floor with a Clopper-Pearson upper-bound
rule, both of which changed on 2026-08-30.

**Apply this retroactively.** It is a property of every banner ever
emitted, not a defect introduced at the end. Anyone reading the weekly
series as a comparable time series must treat monitor-threshold changes as
invisible to it and check the code history separately. The banner's silence
about a threshold move has never been evidence that none occurred.

Not fixed: the bot is stopped, so this is now a records problem rather than
an operational one. Recorded here instead.

---

## Documents that now describe things which no longer exist

Kept deliberately as the **only record of how the deployment was built**.
Marked historical rather than deleted.

| path | status |
|---|---|
| `docs/new_vm_install_checklist.md` | **HISTORICAL as of 2026-08-31.** Describes a destroyed droplet. The IP will be reassigned by DigitalOcean to an unrelated machine — do not connect to it. |
| `bootstrap/linux/` (`install.sh`, `run_weekly_monitor.sh`, `systemd/*.service`) | **HISTORICAL as of 2026-08-31.** Linux deployment scaffolding with `/root/pancakebot` hardcoded. No host runs these. |
| `docs/monitoring.md` | **PARTLY STALE.** Describes the Sunday cron as live (it is not) and pins thresholds that no longer match the code: `WR > 0.55` (now `BREAKEVEN_WR = 0.56`) and the negative `WR < 0.45` raw floor (now a Clopper-Pearson upper-bound rule against `BREAKEVEN_WR`). |

Nothing in the runtime or the scheduled path references the VM: swept after
destruction, zero matches in `pancakebot/`, `run.py`, or `scripts/`. The
full suite — **1218 tests** — passes with the host out of existence, and a
real end-to-end sync completed against live OKX and The Graph the same day.

---

## The one sentence worth keeping

The Sunday contract turned silence into information. Its replacement does
not. **From 2026-08-31, if you want to know whether the sync is alive, you
have to look at the Desktop or at `var/sync_health.json` — you can no
longer infer it from the absence of bad news.**
