# PancakeBot Supervisor

Out-of-process health monitoring for the dry and live bots. Catches crashes
and silent process deaths that log-tailing alone will miss.

## What it does

A pair of Windows Services (`PancakeBotLive` and `PancakeBotDry`,
registered via pywin32 / SCM, source in `pancakebot/service/`) each
supervise their bot child subprocess. Every 1 second, the supervisor
classifies the bot's state from the Popen handle + `var/<mode>/crash.json`
and restarts the bot on death.

Replaced the legacy one-shot `scripts/supervisor.py` (schtask-driven, every
3 min, heartbeat-mtime-based liveness) on 2026-05-23. Heartbeat-staleness
classification was removed on 2026-05-27 (Step 27a) because the 5s
threshold was firing on transient BSC RPC hedged-timeouts that auto-resolve
on the next round — ~12 false-positive restarts/24h with no real bot
dysfunction. **Process-death detection via `Popen.poll()` is the
authoritative liveness signal; there is no longer any heartbeat file.**

## The four classifications

| STATUS         | Meaning                                                            |
|----------------|--------------------------------------------------------------------|
| UP             | Bot Popen handle is alive, past startup grace (30s default)        |
| STARTING       | Bot Popen handle is alive, still inside startup grace              |
| CRASHED        | Bot process is dead AND `crash.json` is present                    |
| DOWN           | Bot process is dead, no `crash.json` (terminated without exception)|
| UNINSTRUMENTED | (classify_state only) bot process detected outside service control |

Precedence is first-match-wins in the order above. `classify_running_bot`
(the in-loop classifier) returns UP / STARTING / CRASHED / DOWN.
`classify_state` is a legacy artifact-only variant used for first-run /
no-Popen-handle scenarios.

## Restart-pattern aggregation (Step 27a, 2026-05-27)

Discord notifications for `CRASHED` / `DOWN` are aggregated by restart
pattern: the first two restarts in any 1-hour rolling window go to the
SCM event log only (no Discord). The third restart in 1h fires a Discord
notification. Independently, ≥8 restarts in 24h fires
`SLOW_CRASHLOOP_WARNING` as a separate severity signal. The fast-crashloop
limiter (≥3 restarts in 15 min → `SUPPRESSED_FAST_CRASHLOOP`, restart
refused) is unchanged.

Constants in `pancakebot/service/common.py`:
- `_FAST_RESTART_MAX = 3`, `_FAST_RESTART_WINDOW_S = 900` (15 min)
- `_SLOW_RESTART_MAX = 8`, `_SLOW_RESTART_WINDOW_S = 86400` (24 h)

## Install

Services are registered via pywin32. From an elevated PowerShell:

```powershell
.\scripts\install_services.ps1
```

This registers `PancakeBotLive` and `PancakeBotDry` with SCM, both set to
Manual start by default. Use `scripts\enable_live.ps1` / `enable_dry.ps1`
to flip to Automatic and start them.

The live-priority mutex (Dry refuses to start while Live is running) is
enforced inside each service's SvcDoRun via SCM ControlService calls.

## Discord alerting

### 1. Create three Discord webhooks

In your Discord server, create one webhook per channel:

| Webhook name           | Channel                  | Routed alerts                                                                            |
|------------------------|--------------------------|------------------------------------------------------------------------------------------|
| PancakeBot Dry Alerts  | `pancakebot-dry-alerts`  | CRASHED / DOWN + escalations for **dry** bot                                             |
| PancakeBot Live Alerts | `pancakebot-live-alerts` | Same as dry, but for the **live** bot                                                    |
| PancakeBot General     | `pancakebot-general`     | `UNINSTRUMENTED` (legacy bot outside service control) + supervisor-self errors (rare)    |

Three separate channels keep the actionable alerts clean.

### 2. Set environment variables (persisted, Machine-scope)

```powershell
# Machine-scope so the Windows Service sees the vars (services don't
# inherit User-scope env). Requires elevated PowerShell.
setx PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL  "https://discord.com/api/webhooks/..." /M
setx PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL "https://discord.com/api/webhooks/..." /M
setx PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL     "https://discord.com/api/webhooks/..." /M
```

`setx /M` writes to `HKLM`. Restart the services for the new vars to take
effect.

Verify the names are set:

```powershell
[Environment]::GetEnvironmentVariables("Machine").GetEnumerator() |
  Where-Object { $_.Key -like "PANCAKEBOT*" } |
  ForEach-Object { $_.Key }
```

### Behavior when env vars are missing

`alert=DISABLED` — soft fallback, no HTTP attempted. The classification
still goes to SCM event log via `servicemanager.LogInfoMsg`. HTTP failures
(bad URL, timeout, 4xx/5xx) log to stderr via `safe_stderr_write`, never
crash the supervisor.

Rate limit: max one alert per `(mode, kind)` per 5 minutes, tracked in
`var/<mode>/last_alert.json`.

## Verify the service is running

```powershell
Get-Service -Name PancakeBot*
```

To surface the actual bot child PID (not the service host PID):

```powershell
$svc = Get-CimInstance -ClassName Win32_Service -Filter "Name='PancakeBotLive'"
$children = Get-CimInstance -ClassName Win32_Process -Filter "ParentProcessId=$($svc.ProcessId)"
$children | Select-Object ProcessId, CommandLine
```

`scripts\enable_live.ps1` and `enable_dry.ps1` do this automatically and
print the bot child PID on a successful start.

## Pause / resume / uninstall

Stop without deregistering:

```powershell
Stop-Service -Name PancakeBotLive
Stop-Service -Name PancakeBotDry
```

Flip to Manual start type (won't auto-start at boot):

```powershell
Set-Service -Name PancakeBotLive -StartupType Manual
Set-Service -Name PancakeBotDry -StartupType Manual
```

Fully uninstall:

```powershell
.\scripts\install_services.ps1 -Uninstall
```

## Troubleshooting

**`STATUS=UNINSTRUMENTED` for a running bot** — the bot was launched
outside the service (e.g., from an interactive shell). Kill any
out-of-service bot processes before starting the service to avoid the
mode-mutex tripping.

**`STATUS=DOWN` for a bot you think is running** — the cmdline check in
`_pid_is_our_bot` requires `run.py --<mode>` in the process cmdline. A
bot launched via a shell shim with a different cmdline will get
misclassified. Verify with `Get-CimInstance Win32_Process -Filter "Name =
'python.exe'" | Select CommandLine`.

**Discord alerts not arriving** — confirm the env vars are Machine-scope
(`setx /M`), not User-scope. Services don't inherit User env. Restart the
service after setting. Check the SCM event log for `alert=SEND_FAILED`
lines; common cause: `401`/`404` = wrong webhook URL.

**Restart loop** — the fast-tier limiter cuts off at 3 restarts in 15 min
with `SUPPRESSED_FAST_CRASHLOOP`. If you're still loop-restarting after
that fires, check `var/<mode>/crash.json` for the root cause and
`var/<mode>/logs/*_err.log` for the last bot's traceback. After fixing,
clear the loop with:

```powershell
Remove-Item var\<mode>\restart_history.jsonl
Remove-Item var\<mode>\crash.json
```

## Files produced

| Path                               | Writer     | Purpose                                     |
|------------------------------------|------------|---------------------------------------------|
| `var/<mode>/bot.pid`               | bot        | PID on startup, cleared on clean exit       |
| `var/<mode>/crash.json`            | bot        | Uncaught-exception dump (atomic write)      |
| `var/<mode>/last_alert.json`       | supervisor | Per-classification alert cooldown timestamps|
| `var/<mode>/restart_history.jsonl` | supervisor | Restart events, pruned at slow-window age   |
| `var/<mode>/logs/*.log`            | spawned bot| stdout/stderr of supervisor-spawned bots    |
| `var/<mode>/runtime.log`           | bot        | Per-cycle structured runtime log            |
| `var/<mode>/cycle_audit.csv`       | bot        | Per-cycle audit row for backtest replay     |

None of these are committed to git (`var/` is in `.gitignore`).
