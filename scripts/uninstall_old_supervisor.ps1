<#
.SYNOPSIS
Remove the legacy schtask-based PancakeBot supervisor.

.DESCRIPTION
**Do not run this until the new Windows Service architecture has been
validated for at least a week of soak time.** This script is staged
in advance so the deletion is a single command when the new services
prove stable. See ``pancakebot/service/DELETION_NOTES.md`` for the full
deletion checklist (this PS1 only handles the schtasks side; the
companion notes cover the Python file deletions, which the operator
performs as a git rm + commit).

Behavior:
  1. Disables (if not already) and deletes both legacy schtasks:
       PancakeBotSupervisorLive
       PancakeBotSupervisorDry
  2. Idempotent — running on an already-clean system is a no-op.
  3. Does NOT touch the new PancakeBotLive / PancakeBotDry services.

After this script runs, the operator should also delete the now-orphaned
Python sources listed in DELETION_NOTES.md:
  scripts/supervisor.py
  scripts/install_supervisor_schtasks.ps1
  tests/test_archive_stale_crash.py     (only if supervisor was its sole consumer)
... and update docs (SUPERVISOR.md → SERVICE.md transition).
#>
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

foreach ($taskName in @("PancakeBotSupervisorLive", "PancakeBotSupervisorDry")) {
    $t = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -eq $t) {
        Write-Host "[ok] $taskName not present" -ForegroundColor DarkGray
        continue
    }
    if ($t.State -ne 'Disabled') {
        Write-Host "[..] disabling $taskName..." -ForegroundColor Yellow
        Disable-ScheduledTask -TaskName $taskName | Out-Null
    }
    Write-Host "[..] deleting $taskName..." -ForegroundColor Yellow
    & schtasks.exe /Delete /TN $taskName /F | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "schtasks delete returned exit code $LASTEXITCODE for $taskName"
    } else {
        Write-Host "[ok] $taskName deleted" -ForegroundColor Green
    }
}
Write-Host ""
Write-Host "Legacy schtasks removed. Next: git rm the Python sources per DELETION_NOTES.md." -ForegroundColor Cyan
