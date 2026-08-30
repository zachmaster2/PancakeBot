<#
    Register the two scheduled tasks that keep market-data collection alive.

    TWO TASKS, NOT ONE, AND THAT IS THE POINT.

      PancakeBotDailySync      -- does the work.
      PancakeBotSyncWatchdog   -- checks that the work happened.

    A single task cannot report its own absence. If the sync task is
    disabled, deleted, or never fires, it emits nothing -- and nothing is
    exactly what a healthy idle system emits. The watchdog runs on its own
    trigger, reads the heartbeat the sync writes, and raises a Desktop
    marker when that heartbeat goes stale. It therefore fires precisely in
    the case the sync task cannot report on: its own silence.

    They share Task Scheduler as a common dependency, which is a real and
    stated limit -- if the service itself is broken both are down. Removing
    that would need an external observer, which is the thing being retired.

    RUNS AS SYSTEM so it works with nobody logged in, and so no password is
    stored anywhere. SYSTEM needs no credential, which is why it was chosen
    over "run whether user is logged on or not" with a saved password.

    Re-running this script is safe: existing tasks are replaced.
#>

$ErrorActionPreference = 'Stop'

$Repo = 'C:\Users\zking\Documents\GitHub\PancakeBot'
$Ps   = 'powershell.exe'
$Py   = Join-Path $Repo '.venv\Scripts\python.exe'

$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

# ---------------------------------------------------------------- sync ----
$syncAction = New-ScheduledTaskAction -Execute $Ps `
    -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}"' -f (Join-Path $Repo 'scripts\daily_sync.ps1')) `
    -WorkingDirectory $Repo

# Daily, plus at startup. AtStartup covers the machine having been off
# through the daily window entirely -- without it, a machine that is only
# on during the day would never run a 03:30 task.
$syncTriggers = @(
    (New-ScheduledTaskTrigger -Daily -At '06:30'),
    (New-ScheduledTaskTrigger -AtStartup)
)

# StartWhenAvailable is the "run as soon as possible after a missed start"
# requirement. RestartCount/RestartInterval retry a genuinely failed run,
# which covers a transient network drop without waiting a whole day.
# The battery settings matter on a laptop: the default is to refuse to
# start on battery, which would silently skip runs.
$syncSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 20) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RunOnlyIfNetworkAvailable:$false

Register-ScheduledTask -TaskName 'PancakeBotDailySync' `
    -Action $syncAction -Trigger $syncTriggers -Principal $principal `
    -Settings $syncSettings `
    -Description 'PancakeBot: daily OKX/Graph market-data sync (append-only).' `
    -Force | Out-Null

# ------------------------------------------------------------ watchdog ----
$wdAction = New-ScheduledTaskAction -Execute $Py `
    -Argument 'scripts\sync_watchdog.py' -WorkingDirectory $Repo

# Deliberately offset from the sync and repeated through the day. The
# watchdog must run even on days the sync never does -- that is its entire
# purpose -- so its trigger must not depend on the sync's.
$wdTriggers = @(
    (New-ScheduledTaskTrigger -Daily -At '09:00'),
    (New-ScheduledTaskTrigger -AtStartup)
)

$wdSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RunOnlyIfNetworkAvailable:$false

Register-ScheduledTask -TaskName 'PancakeBotSyncWatchdog' `
    -Action $wdAction -Trigger $wdTriggers -Principal $principal `
    -Settings $wdSettings `
    -Description 'PancakeBot: raises a Desktop marker if the daily sync stops succeeding.' `
    -Force | Out-Null

Write-Output 'Registered:'
Get-ScheduledTask -TaskName 'PancakeBot*' |
    Select-Object TaskName, State |
    Format-Table -AutoSize
