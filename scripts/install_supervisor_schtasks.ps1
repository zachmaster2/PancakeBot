<#
.SYNOPSIS
Install Windows Task Scheduler jobs that run scripts/supervisor.py every 3 minutes.

.DESCRIPTION
Registers two scheduled tasks:

    PancakeBotSupervisorDry   -> supervisor.py --mode dry
    PancakeBotSupervisorLive  -> supervisor.py --mode live

Both run as the CURRENT USER (no admin elevation required, no SYSTEM
credentials, no saved password). They fire every 3 minutes, starting 1
minute after installation.

**Default behaviour (as of Phase 2c-on):** classify + log + Discord alerts
(`--alert`). Webhook URLs come from the Machine-level env vars
``PANCAKEBOT_{DRY_ALERTS,LIVE_ALERTS,GENERAL}_DISCORD_WEBHOOK_URL``.
`--restart` (auto-respawn) is still OFF -- that's the next gate.

If you want tier-1 (log-only, no Discord) back, remove `--alert` from the
task action via ``schtasks /change`` or edit this script's ``$argString``.
Upgrade / downgrade instructions are printed at the end of this script
and also live in ``docs/SUPERVISOR.md``.

Idempotent: re-running this script deletes and recreates both tasks
without duplicating them.

.PARAMETER VenvPython
Override the path to the venv python executable. Defaults to
``<repo>\.venv\Scripts\pythonw.exe`` (the windowless variant -- avoids the
console-window focus-steal every time a scheduled task fires).

.PARAMETER IntervalMinutes
How often each task runs. Default 3 minutes.

.EXAMPLE
PS> .\scripts\install_supervisor_schtasks.ps1

.EXAMPLE
PS> .\scripts\install_supervisor_schtasks.ps1 -IntervalMinutes 5

.NOTES
If either task already exists it will be removed and re-created (via
Register-ScheduledTask -Force). You can manually remove everything
installed by this script with:

    schtasks /delete /tn PancakeBotSupervisorDry /f
    schtasks /delete /tn PancakeBotSupervisorLive /f
#>
[CmdletBinding()]
param(
    [string]$VenvPython,
    [int]$IntervalMinutes = 3
)

$ErrorActionPreference = "Stop"

# Resolve the repo root -- this script lives at <repo>\scripts\install_supervisor_schtasks.ps1.
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)

if (-not $VenvPython) {
    # pythonw.exe is the windowless Python variant. Using python.exe here would
    # flash a console window every 3 minutes and steal keyboard/mouse focus
    # from whatever the user is doing -- unacceptable for a background task.
    $VenvPython = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
}

if (-not (Test-Path $VenvPython)) {
    throw "Python executable not found: $VenvPython. Pass -VenvPython <path> or create the venv first."
}

$SupervisorScript = Join-Path $RepoRoot "scripts\supervisor.py"
if (-not (Test-Path $SupervisorScript)) {
    throw "supervisor.py not found: $SupervisorScript"
}

Write-Host "=== PancakeBot supervisor schtasks installer ===" -ForegroundColor Cyan
Write-Host "Repo root         : $RepoRoot"
Write-Host "Venv python       : $VenvPython"
Write-Host "Supervisor script : $SupervisorScript"
Write-Host "Interval          : every $IntervalMinutes minutes"
Write-Host ""

function Register-SupervisorTask {
    param(
        [string]$TaskName,
        [string]$Mode
    )

    # Quote the supervisor script path in case the repo lives under a path
    # with spaces (e.g. "C:\Users\zking\My Stuff\..."). The executable itself
    # is passed unquoted via -Execute; arguments are the rest.
    #
    # --alert and --restart are both enabled by default as of 2026-04-24.
    # Dry mode was validated at Tier 3 after commit 0ebee57 (retry-once on
    # transient reads) cleared the false-DOWN false-positive mode that
    # would otherwise cause spurious auto-restarts. See docs/SUPERVISOR.md
    # "Retry-once on transient reads" section.
    $argString = "`"$SupervisorScript`" --mode $Mode --alert --restart"

    $action = New-ScheduledTaskAction `
        -Execute $VenvPython `
        -Argument $argString `
        -WorkingDirectory $RepoRoot

    # Run every <IntervalMinutes> minutes indefinitely, starting 1 minute from now.
    # Omitting -RepetitionDuration makes the trigger repeat indefinitely (Windows
    # rejects [TimeSpan]::MaxValue, so indefinite is expressed by absence).
    $startAt = (Get-Date).AddMinutes(1)
    $trigger = New-ScheduledTaskTrigger -Once -At $startAt `
        -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)

    # Don't stop if on battery; allow start if missed (e.g. laptop was asleep).
    # Execution time limit 2 min: supervisor normally finishes in under a second,
    # so anything over 2 min means it's stuck and should be killed.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
        -MultipleInstances IgnoreNew

    # Run as the current user without admin. Interactive logon type means
    # the task only runs while the user is logged in -- which is what we
    # want for a dev/home machine.
    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive `
        -RunLevel Limited

    # -Force deletes any existing task with the same name first.
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "PancakeBot health supervisor ($Mode mode). Classify + log every $IntervalMinutes min." `
        -Force | Out-Null

    Write-Host "  [OK] registered $TaskName (mode=$Mode)" -ForegroundColor Green
}

Register-SupervisorTask -TaskName "PancakeBotSupervisorDry"  -Mode "dry"
Register-SupervisorTask -TaskName "PancakeBotSupervisorLive" -Mode "live"

# Live mode isn't deployed yet -- disable the live supervisor task until it is,
# otherwise every 3-minute fire sees DOWN and pages Discord for a
# non-existent bot. Re-enable with `Enable-ScheduledTask -TaskName
# PancakeBotSupervisorLive` when you actually deploy live mode.
Disable-ScheduledTask -TaskName "PancakeBotSupervisorLive" | Out-Null
Write-Host "  [OK] PancakeBotSupervisorLive disabled (live mode not yet deployed)" -ForegroundColor Yellow

Write-Host ""
Write-Host "Dry task registered & firing. Live task registered but DISABLED (no noise until deployed)." -ForegroundColor Cyan
Write-Host "First dry run kicks off ~1 minute from now, then every $IntervalMinutes min."
Write-Host ""
Write-Host "--- Verification ---" -ForegroundColor Yellow
Write-Host "  schtasks /query /tn PancakeBotSupervisorDry"
Write-Host "  schtasks /query /tn PancakeBotSupervisorLive"
Write-Host ""
Write-Host "Supervisor output accumulates in:"
Write-Host "  $RepoRoot\var\dry\supervisor.log"
Write-Host "  $RepoRoot\var\live\supervisor.log"
Write-Host ""
Write-Host "--- Next steps ---" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Tasks installed with --alert enabled (Discord notifications live)."
Write-Host "  Verify webhooks are set:"
Write-Host "    [Environment]::GetEnvironmentVariables('Machine').GetEnumerator() |"
Write-Host "      Where-Object { `$_.Key -like 'PANCAKEBOT*' } | ForEach-Object { `$_.Key }"
Write-Host ""
Write-Host "  When live mode is deployed, enable the live supervisor task:"
Write-Host "    Enable-ScheduledTask -TaskName PancakeBotSupervisorLive"
Write-Host ""
Write-Host "  When ready to enable auto-restart (--restart), see docs/SUPERVISOR.md."
Write-Host ""
Write-Host "  Uninstall:"
Write-Host "    schtasks /delete /tn PancakeBotSupervisorDry /f"
Write-Host "    schtasks /delete /tn PancakeBotSupervisorLive /f"
