<#
.SYNOPSIS
Enable + start the PancakeBotLive service. Idempotent.

.DESCRIPTION
Flips start type to Automatic (so the service starts at boot) and starts
the service now. Safe to run if the service is already running — does
nothing in that case.

The live-priority mutex (stopping PancakeBotDry if running) is enforced
*inside* the service's SvcDoRun via SCM ControlService calls, NOT here.
Pre-stopping Dry from out-of-process would race the in-process mutex
check and cause spurious MODE_TRANSITION_REFUSED alerts on the Dry side.
#>
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

$svc = Get-Service -Name PancakeBotLive -ErrorAction SilentlyContinue
if ($null -eq $svc) {
    throw "PancakeBotLive service is not installed. Run scripts\install_services.ps1 first."
}

# Idempotent: re-applying Automatic on an Automatic service is a no-op.
Set-Service -Name PancakeBotLive -StartupType Automatic

if ($svc.Status -eq 'Running') {
    Write-Host "[ok] PancakeBotLive already running" -ForegroundColor Green
    exit 0
}

Write-Host "[..] starting PancakeBotLive..." -ForegroundColor Yellow
Start-Service -Name PancakeBotLive
$svc.WaitForStatus('Running', '00:00:30')

# Surface the bot child PID, not the service host PID — what operators
# actually want to see. Heartbeat may not exist yet (first cycle ~1-2s).
Start-Sleep -Seconds 2
$hb = Join-Path (Split-Path -Parent (Split-Path -Parent $PSCommandPath)) "var\live\heartbeat.json"
if (Test-Path $hb) {
    try {
        $j = Get-Content $hb -Raw | ConvertFrom-Json
        Write-Host "[ok] PancakeBotLive running (bot child pid=$($j.pid))" -ForegroundColor Green
    } catch {
        Write-Host "[ok] PancakeBotLive running (heartbeat not parseable yet)" -ForegroundColor Green
    }
} else {
    Write-Host "[ok] PancakeBotLive running (heartbeat not yet written)" -ForegroundColor Green
}
