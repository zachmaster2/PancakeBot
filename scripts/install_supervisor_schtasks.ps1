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

**Phase 2e default: classify+log only (NO --alert, NO --restart).**
That is intentional. Run this way for a few days / a week so that you
observe the classifier against your actual workload before turning on
alerts or auto-restart. Upgrade instructions are printed at the end of
this script and also live in docs/SUPERVISOR.md.

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
    $argString = "`"$SupervisorScript`" --mode $Mode"

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

Write-Host ""
Write-Host "Both tasks registered. First run kicks off ~1 minute from now, then every $IntervalMinutes min." -ForegroundColor Cyan
Write-Host ""
Write-Host "--- Verification ---" -ForegroundColor Yellow
Write-Host "  schtasks /query /tn PancakeBotSupervisorDry"
Write-Host "  schtasks /query /tn PancakeBotSupervisorLive"
Write-Host ""
Write-Host "Supervisor output accumulates in:"
Write-Host "  $RepoRoot\var\dry\supervisor.log"
Write-Host "  $RepoRoot\var\live\supervisor.log"
Write-Host ""
Write-Host "--- Next steps (DO NOT do these until you've validated classify-only mode) ---" -ForegroundColor Yellow
Write-Host ""
Write-Host "  See docs/SUPERVISOR.md for the full tier-1 -> tier-2 -> tier-3 upgrade playbook."
Write-Host ""
Write-Host "  Short version:"
Write-Host "    Tier 2 (add --alert):   create Discord webhook, setx the *_DISCORD_ALERT_WEBHOOK_URL,"
Write-Host "                            then schtasks /change /tr (see docs) to append --alert."
Write-Host "    Tier 3 (add --restart): only after --alert has been observed working; change the"
Write-Host "                            task action to include --alert --restart."
Write-Host ""
Write-Host "  Uninstall:"
Write-Host "    schtasks /delete /tn PancakeBotSupervisorDry /f"
Write-Host "    schtasks /delete /tn PancakeBotSupervisorLive /f"
