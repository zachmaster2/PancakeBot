<#
.SYNOPSIS
Stop + disable the PancakeBotLive service. Idempotent.

.DESCRIPTION
Sets start type to Disabled (so the service does not auto-start at boot)
and stops the service if running. Safe to run if the service is already
stopped or not installed.

The service's SvcStop drains the bot child gracefully (terminate + up to
20s grace, then hard-kill if needed). Total stop time is bounded; we
wait up to 60s here for full SCM transition to STOPPED.
#>
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

$svc = Get-Service -Name PancakeBotLive -ErrorAction SilentlyContinue
if ($null -eq $svc) {
    Write-Host "[ok] PancakeBotLive not installed; nothing to disable" -ForegroundColor Green
    exit 0
}

# Disable first so a crashloop / recovery can't restart between our stop and
# this config change.
Set-Service -Name PancakeBotLive -StartupType Disabled

if ($svc.Status -eq 'Stopped') {
    Write-Host "[ok] PancakeBotLive already stopped" -ForegroundColor Green
    exit 0
}

Write-Host "[..] stopping PancakeBotLive..." -ForegroundColor Yellow
Stop-Service -Name PancakeBotLive -Force
$svc.WaitForStatus('Stopped', '00:01:00')
Write-Host "[ok] PancakeBotLive stopped + disabled" -ForegroundColor Green
