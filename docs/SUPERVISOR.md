# PancakeBot Supervisor

Out-of-process health monitoring for the dry and live bots. Catches crashes,
deadlocks, and silent deaths that log-tailing alone will miss.

## What it does

A one-shot Python script (`scripts/supervisor.py`) that Windows Task
Scheduler runs every 3 minutes for each mode. Each invocation reads three
artifacts the bot writes (`heartbeat.json`, `bot.pid`, `crash.json` — see
`pancakebot/runtime/process_health.py`), classifies the bot's state, and
appends one structured line to `var/<mode>/supervisor.log`. Optionally it
posts a Discord alert and/or auto-restarts the bot.

Designed to catch the failure mode that bit us before: commit `059db6d`
introduced an uncaught `AttributeError` at `dry.py:566` that killed the bot
silently at 02:49 EDT and let it sit dead for 8 hours before anyone
noticed. The supervisor would have paged within 3 minutes.

## The six classifications

| STATUS          | Meaning                                                   | Exit |
|-----------------|-----------------------------------------------------------|------|
| UP              | Fresh heartbeat, PID alive and matches `run.py --<mode>`  | 0    |
| STARTING        | PID file fresh, heartbeat absent/stale, within grace (90s)| 1    |
| STALE           | PID alive but heartbeat stale past threshold              | 2    |
| CRASHED         | `crash.json` present (regardless of process state)        | 3    |
| UNINSTRUMENTED  | Bot process alive but no heartbeat AND no valid PID file  | 4    |
| DOWN            | None of the above                                         | 5    |
| *suppressed*    | `SUPPRESSED_FAST_CRASHLOOP` — auto-restart limiter tripped| 6    |

Precedence is first-match-wins in the order above.

## Three operational tiers — upgrade progressively

**Tier 1: classify + log only** (default after install) — you run the
supervisor for a few days without alerts or auto-restart. Tail
`var/<mode>/supervisor.log` and check that the classifier labels match
what the bot is actually doing. If classification drifts in either
direction (false UP / false CRASHED) during this window, investigate
before moving on.

**Tier 2: add `--alert`** — turn on Discord notifications. Still
observer-only; the supervisor won't touch the bot process. You learn the
alert cadence and message format without risking auto-restart loops.

**Tier 3: add `--restart`** — enable auto-restart with the two-tier
crashloop limiter. At this point the supervisor is fully autonomous on
normal failures; humans only step in for persistent (slow-tier)
crashloops.

Advance tier-by-tier. A premature jump to `--restart` while the
classifier still has false-positive CRASHED can restart a healthy bot
or churn through live-mode wallet gas.

## Install

```powershell
# From the repo root, in a non-admin PowerShell:
.\scripts\install_supervisor_schtasks.ps1
```

That creates two Windows scheduled tasks:

- `PancakeBotSupervisorDry`   -> `supervisor.py --mode dry --alert`  (enabled)
- `PancakeBotSupervisorLive`  -> `supervisor.py --mode live --alert` (**disabled**)

Both run as the current user (no admin elevation, no SYSTEM credentials,
no saved password), every 3 minutes, starting 1 minute after install.

**The live supervisor is installed-but-disabled by default.** Until live
mode is actually deployed, every 3-minute fire would classify DOWN (no
live bot to supervise) and page the live-alerts Discord channel for a
non-issue. When you deploy live mode, enable it with one command:

```powershell
Enable-ScheduledTask -TaskName PancakeBotSupervisorLive
```

Conversely, to mute either mode temporarily without uninstalling:

```powershell
Disable-ScheduledTask -TaskName PancakeBotSupervisorDry
Disable-ScheduledTask -TaskName PancakeBotSupervisorLive
```

The script is idempotent — re-running it deletes and re-registers each
task via `Register-ScheduledTask -Force`.

Override the interval or Python path if needed:

```powershell
.\scripts\install_supervisor_schtasks.ps1 -IntervalMinutes 5
.\scripts\install_supervisor_schtasks.ps1 -VenvPython "C:\custom\python.exe"
```

## Verify it's running

```powershell
schtasks /query /tn PancakeBotSupervisorDry
schtasks /query /tn PancakeBotSupervisorLive
```

Or in one shot with detail:

```powershell
Get-ScheduledTask -TaskName PancakeBot* | Format-List TaskName, State, LastRunTime, LastTaskResult
```

Every 3 minutes you should see a fresh line appended to the log:

```
Get-Content var\dry\supervisor.log -Tail 5 -Wait
```

Expected content for a healthy bot:

```
2026-04-22T23:51:20Z STATUS=UP mode=dry pid=8332 hb_age=0.2s bankroll=5.0000 bets=0 iterations=42 last_epoch=474974
```

## Upgrade to Tier 2 (Discord alerting)

### 1. Create three Discord webhooks

In your Discord server, create one webhook per channel (Server Settings →
Integrations → Webhooks → **New Webhook** → select channel → Copy Webhook URL):

| Webhook name         | Channel                 | Routed alerts                                                                                           |
|----------------------|-------------------------|---------------------------------------------------------------------------------------------------------|
| PancakeBot Dry Alerts  | `pancakebot-dry-alerts`  | STALE / CRASHED / DOWN + escalations for **dry** bot. Actionable for the dry operator.                  |
| PancakeBot Live Alerts | `pancakebot-live-alerts` | Same as dry, but for the **live** bot. Separate channel so you can mute dry without silencing live.      |
| PancakeBot General     | `pancakebot-general`     | `UNINSTRUMENTED` (legacy pre-Phase-2a bot running — informational) and supervisor-self errors (rare). |

Three separate channels keep the actionable alerts clean. When you're
staring at `#pancakebot-live-alerts` expecting to see real problems, you
don't want it cluttered with "legacy bot still running" pings.

### 2. Set environment variables (persisted, Machine-scope)

```powershell
# Machine-scope so Task Scheduler sees the vars without needing a logon.
# Requires an elevated PowerShell (or setx /M with suitable permissions).
setx PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL  "https://discord.com/api/webhooks/..." /M
setx PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL "https://discord.com/api/webhooks/..." /M
setx PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL     "https://discord.com/api/webhooks/..." /M
```

`setx /M` writes to `HKLM`. **Currently-open** PowerShell windows and
**already-running** processes won't see the change until restart, but
each Task Scheduler firing is a fresh process that reads Machine env
anew — the next scheduled supervisor run picks up the new URLs
automatically.

Verify names are set (values intentionally not printed):

```powershell
[Environment]::GetEnvironmentVariables("Machine").GetEnumerator() |
  Where-Object { $_.Key -like "PANCAKEBOT*" } |
  ForEach-Object { $_.Key }
```

### 3. Change the task action to include `--alert`

```powershell
$repo = "C:\Users\zking\Documents\GitHub\PancakeBot"
schtasks /change /tn PancakeBotSupervisorDry /tr "`"$repo\.venv\Scripts\python.exe`" `"$repo\scripts\supervisor.py`" --mode dry --alert"
schtasks /change /tn PancakeBotSupervisorLive /tr "`"$repo\.venv\Scripts\python.exe`" `"$repo\scripts\supervisor.py`" --mode live --alert"
```

After the next fire, non-UP / non-STARTING classifications POST to Discord:

- **STALE / CRASHED / DOWN + escalations** → mode's `*-alerts` channel
- **UNINSTRUMENTED** → `general` channel
- **Supervisor-self errors** (classify raised unexpectedly) → `general` channel

Behavior when env vars are missing: `alert=DISABLED` (soft fallback,
classification still logged). HTTP failure (bad URL, timeout, 4xx/5xx)
→ `alert=SEND_FAILED` (logged to stderr + supervisor.log, not crashed).
Rate limit: max one alert per `(mode, classification-or-escalation)`
per 5 minutes, tracked in `var/<mode>/last_alert.json`. Supervisor-self
errors have their own `SUPERVISOR_ERROR` cooldown bucket.

## Upgrade to Tier 3 (auto-restart)

Only after Tier 2 has been running a while with accurate alerts.

**Dry mode is on Tier 3 as of 2026-04-24** (commit `0ebee57` validated
the read-layer retry fix; see "Retry-once on transient reads" below).
Live mode is still disabled pending live deployment.

```powershell
$repo = "C:\Users\zking\Documents\GitHub\PancakeBot"
schtasks /change /tn PancakeBotSupervisorDry /tr "`"$repo\.venv\Scripts\python.exe`" `"$repo\scripts\supervisor.py`" --mode dry --alert --restart"
schtasks /change /tn PancakeBotSupervisorLive /tr "`"$repo\.venv\Scripts\python.exe`" `"$repo\scripts\supervisor.py`" --mode live --alert --restart"
```

On `STALE` / `CRASHED` / `DOWN` the supervisor spawns a detached
`python run.py --<mode>` via `subprocess.Popen` (Windows:
`CREATE_NEW_PROCESS_GROUP`). Each restart appends to
`var/<mode>/restart_history.jsonl`.

Crashloop limiter has two tiers:

- **Fast tier:** `--max-fast-restarts` (default 3) within
  `--fast-window-min` (default 15) → `SUPPRESSED_FAST_CRASHLOOP`, the
  restart is refused, exit code 6. An alert still fires.
- **Slow tier:** `--max-slow-restarts` (default 8) within
  `--slow-window-h` (default 24) → restart proceeds BUT the alert is
  escalated to `SLOW_CRASHLOOP_WARNING`.

Override defaults by adding flags to the task action, e.g.
`--max-fast-restarts 2 --fast-window-min 10` for a tighter fast-tier.

## How to read `supervisor.log`

Each line is ISO-8601 timestamp + `STATUS=` + mode + `key=value` fields.
Fields land in a deterministic order for easy `awk` / `grep`:

```
pid, hb_age, bankroll, bets, iterations, last_epoch,
since_pid_ts, crash_age, exc, action, new_pid, alert, note
```

Examples:

```
2026-04-22T23:51:20Z STATUS=UP mode=dry pid=8332 hb_age=0.2s bankroll=5.0000 bets=0 iterations=42 last_epoch=474974
2026-04-23T00:06:15Z STATUS=CRASHED mode=dry bets=0 last_epoch=474776 crash_age=1.1s exc=AttributeError alert=DISPATCHING
2026-04-23T00:06:15Z STATUS=CRASHED mode=dry alert_outcome=SENT
2026-04-23T00:07:32Z STATUS=DOWN mode=live action=SLOW_CRASHLOOP_WARNING new_pid=8348 alert=SUPPRESSED_ROUTINE_RESTART
2026-04-23T00:07:22Z STATUS=DOWN mode=live action=SUPPRESSED_FAST_CRASHLOOP
```

### Two-line shape on the Discord-HTTP path (2026-05-22, commit `115185f`)

For ticks that attempt a Discord POST, `supervisor.log` gets TWO lines
per invocation: a CLASSIFICATION line with `alert=DISPATCHING` written
BEFORE the HTTP call, then an OUTCOME line with `alert_outcome=<result>`
written AFTER. The split exists so the classification line lands even
if the supervisor process is killed by the schtasks 2-minute timeout
during a hung Discord call (caught 2026-05-21 07:54 UTC — the first
CRASHED detection tick was missing entirely because the Discord POST
took longer than the kill window).

Sync-only outcomes (`SUPPRESSED_ROUTINE_RESTART`, `NOT_APPLICABLE`)
still write a SINGLE line with `alert=<outcome>` directly. Only the
HTTP path (where it could hang) defers to a second line.

Atomicity: `_write_supervisor_line` uses `os.open(O_APPEND|O_WRONLY|O_CREAT)`
+ a single `os.write()` call. Single-syscall append at the OS layer; no
Python text-mode buffering between the format step and the disk write.

## Retry-once on transient reads (Option C, 2026-04-24)

On 2026-04-23 the supervisor fired a false-DOWN Discord alert for the
dry bot. Post-incident investigation showed the bot was healthy
throughout: `_safe_read_json(heartbeat.json)` returned `None` and
`_pid_is_our_bot` had a transient `psutil` failure simultaneously.
Classifier saw both negative signals and logged `DOWN` with an alert.

Fix landed in commit `0ebee57` (tests: `tests/test_supervisor_retry.py`):

- `_safe_read_json` retries once after `_TRANSIENT_READ_BACKOFF_S`
  (500ms) when the first read returns `None`. On save-by-retry it
  emits `DIAGNOSTIC safe_read_json_retry_recovered path=...` to
  `supervisor.log` (and stderr).
- `_pid_is_our_bot` retries once after exception. `pid_exists=False`
  and cmdline-mismatch are clean misses — no retry. On exhausted retry
  it emits `DIAGNOSTIC pid_is_our_bot_retry_exhausted pid=... mode=...`.

Worst-case added latency: 4 retry paths × 500ms = 2.0s per invocation.
Schtask budget is 2 min, cadence 3 min — fully inside budget. Retries
never mask persistent failures (both-None still classifies DOWN).

To confirm the retry is active, tail `var/<mode>/supervisor.log` for
`DIAGNOSTIC` lines; they should be rare (most invocations read
cleanly). A spike of `safe_read_json_retry_recovered` lines is a
signal that Windows AV or disk contention is getting worse —
investigate before it escalates to exhausted retries.

## Pause / resume / uninstall

Pause both tasks without deleting them:

```powershell
Disable-ScheduledTask -TaskName PancakeBotSupervisorDry
Disable-ScheduledTask -TaskName PancakeBotSupervisorLive
```

Re-enable:

```powershell
Enable-ScheduledTask -TaskName PancakeBotSupervisorDry
Enable-ScheduledTask -TaskName PancakeBotSupervisorLive
```

Fully uninstall:

```powershell
schtasks /delete /tn PancakeBotSupervisorDry /f
schtasks /delete /tn PancakeBotSupervisorLive /f
```

## Troubleshooting

**`STATUS=UNINSTRUMENTED` for a running bot** — the bot was launched from
a pre-Phase-2a commit (before heartbeat writes landed). Kill and relaunch
from current HEAD.

**`STATUS=DOWN` for a bot you think is running** — the cmdline check in
`_pid_is_our_bot` requires `run.py --<mode>` in the process cmdline. A
bot launched via a shell shim with a different cmdline will get
misclassified. Verify with `Get-CimInstance Win32_Process -Filter "Name =
'python.exe'" | Select CommandLine`.

**`alert=SEND_FAILED`** — stderr from the task's last run gets captured
in Event Viewer under Microsoft → Windows → TaskScheduler. Look for the
HTTP status code the supervisor logged. Common: `401`/`404` = wrong
webhook URL; `429` = Discord rate-limited you (the supervisor's own 5-min
cooldown should prevent this from happening repeatedly).

**Restart loop after `--restart` enabled** — the fast-tier limiter should
cut it off at 3 restarts in 15 min. If you're still loop-restarting,
check `var/<mode>/crash.json` for the root cause and/or `var/<mode>/logs/
<mode>-auto-*_err.log` for the last spawned bot's traceback. After
fixing, clear the loop with:

```powershell
Remove-Item var\<mode>\restart_history.jsonl
Remove-Item var\<mode>\crash.json
```

## Files produced

| Path                               | Writer     | Purpose                                     |
|------------------------------------|------------|---------------------------------------------|
| `var/<mode>/heartbeat.json`        | bot        | Liveness (mtime is primary signal)          |
| `var/<mode>/bot.pid`               | bot        | PID on startup, cleared on clean exit       |
| `var/<mode>/crash.json`            | bot        | Uncaught-exception dump                     |
| `var/<mode>/supervisor.log`        | supervisor | Append-only status line per invocation      |
| `var/<mode>/last_alert.json`       | supervisor | Per-classification alert cooldown timestamps|
| `var/<mode>/restart_history.jsonl` | supervisor | Restart events, pruned at slow-window age   |
| `var/<mode>/logs/<mode>-auto-*.log`| spawned bot| stdout/stderr of supervisor-spawned bots    |

None of these are committed to git (`var/` is in `.gitignore`). They are
the supervisor's working set.
