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
$svc_cim = Get-CimInstance -ClassName Win32_Service -Filter "Name='PancakeBotDry'" -ErrorAction SilentlyContinue
$svc_pid = if ($svc_cim) { $svc_cim.ProcessId } else { $null }

if ($null -eq $svc_pid -or $svc_pid -le 0) {
    Write-Host "[!!] PancakeBotDry Win32_Service has no ProcessId (unexpected for Running state)" -ForegroundColor Red
    exit 1
}

$children = @(Get-CimInstance -ClassName Win32_Process -Filter "ParentProcessId=$svc_pid" -ErrorAction SilentlyContinue)
if ($children.Count -gt 0) {
    $bot_pid = $children[0].ProcessId
    Write-Host "[ok] PancakeBotDry running (service pid=$svc_pid, bot child pid=$bot_pid)" -ForegroundColor Green
} else {
    Write-Host "[ok] PancakeBotDry running (service pid=$svc_pid, no bot child — may have been refused by mode mutex OR first ~1-2s)" -ForegroundColor Green
}
