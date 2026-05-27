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

# Surface the bot child PID via Win32_Process. Service host
# (pythonservice.exe) is the parent; its child python.exe is the actual
# bot. Bot child may take ~1-2s to spawn after service start.
Start-Sleep -Seconds 2
$svc_cim = Get-CimInstance -ClassName Win32_Service -Filter "Name='PancakeBotLive'" -ErrorAction SilentlyContinue
$svc_pid = if ($svc_cim) { $svc_cim.ProcessId } else { $null }

if ($null -eq $svc_pid -or $svc_pid -le 0) {
    Write-Host "[!!] PancakeBotLive Win32_Service has no ProcessId (unexpected for Running state)" -ForegroundColor Red
    exit 1
}

$children = @(Get-CimInstance -ClassName Win32_Process -Filter "ParentProcessId=$svc_pid" -ErrorAction SilentlyContinue)
if ($children.Count -gt 0) {
    $bot_pid = $children[0].ProcessId
    Write-Host "[ok] PancakeBotLive running (service pid=$svc_pid, bot child pid=$bot_pid)" -ForegroundColor Green
} else {
    Write-Host "[ok] PancakeBotLive running (service pid=$svc_pid, bot child not yet spawned — typical for first ~1-2s)" -ForegroundColor Green
}
