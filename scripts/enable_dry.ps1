<#
.SYNOPSIS
Enable + start the PancakeBotDry service. Idempotent.

.DESCRIPTION
Flips start type to Automatic and starts the service now.

If PancakeBotLive is running, the service's in-process mutex check will
refuse to spawn a bot and cleanly exit (a MODE_TRANSITION_REFUSED Discord
alert is fired on the dry channel). This is intentional: out-of-process
pre-checks would race the SCM state machine.
#>
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

$svc = Get-Service -Name PancakeBotDry -ErrorAction SilentlyContinue
if ($null -eq $svc) {
    throw "PancakeBotDry service is not installed. Run scripts\install_services.ps1 first."
}

Set-Service -Name PancakeBotDry -StartupType Automatic

if ($svc.Status -eq 'Running') {
    Write-Host "[ok] PancakeBotDry already running" -ForegroundColor Green
    exit 0
}

# Warn loudly if Live is running — Dry will refuse, and we want the operator
# to know upfront rather than reading the Discord alert.
$live = Get-Service -Name PancakeBotLive -ErrorAction SilentlyContinue
if ($null -ne $live -and $live.Status -eq 'Running') {
    Write-Warning "PancakeBotLive is running. Dry will refuse to spawn a bot (live priority)."
}

Write-Host "[..] starting PancakeBotDry..." -ForegroundColor Yellow
Start-Service -Name PancakeBotDry
$svc.WaitForStatus('Running', '00:00:30')

Start-Sleep -Seconds 2
$hb = Join-Path (Split-Path -Parent (Split-Path -Parent $PSCommandPath)) "var\dry\heartbeat.json"
if (Test-Path $hb) {
    try {
        $j = Get-Content $hb -Raw | ConvertFrom-Json
        Write-Host "[ok] PancakeBotDry running (bot child pid=$($j.pid))" -ForegroundColor Green
    } catch {
        Write-Host "[ok] PancakeBotDry running (heartbeat not parseable yet)" -ForegroundColor Green
    }
} else {
    Write-Host "[ok] PancakeBotDry running (heartbeat not yet written; may have been refused by mode mutex)" -ForegroundColor Green
}
