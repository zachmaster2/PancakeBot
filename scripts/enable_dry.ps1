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

# Warn loudly if Live is running -- Dry will refuse, and we want the operator
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
    # MODE_TRANSITION_REFUSED edge case: when Live is running, Dry's
    # SvcDoRun returns immediately after refusal. SCM briefly reports
    # RUNNING (WaitForStatus succeeds), then transitions to STOPPED.
    # Win32_Service.ProcessId is 0 either while StopPending or after
    # Stopped. Re-poll once after 2s: if SCM has now settled to
    # Stopped/StopPending AND Live is actually Running, treat as
    # expected refusal and exit 0. If Live is NOT running, this is a
    # genuine Dry-side failure -- fall through to the error path so the
    # operator doesn't see a misleading "refused" message.
    Start-Sleep -Seconds 2
    $svc_recheck = Get-Service -Name PancakeBotDry
    $live_recheck = Get-Service -Name PancakeBotLive -ErrorAction SilentlyContinue
    $dry_stopped = ($svc_recheck.Status -eq 'Stopped' -or $svc_recheck.Status -eq 'StopPending')
    $live_running = ($null -ne $live_recheck -and $live_recheck.Status -eq 'Running')
    if ($dry_stopped -and $live_running) {
        Write-Host "[ok] Dry refused: Live is running (MODE_TRANSITION_REFUSED). SCM state: $($svc_recheck.Status)." -ForegroundColor Green
        exit 0
    }
    Write-Host "[!!] PancakeBotDry Win32_Service has no ProcessId (unexpected for Running state)" -ForegroundColor Red
    if ($dry_stopped) {
        Write-Host "    Dry is $($svc_recheck.Status) but Live is not Running -- genuine Dry-side failure." -ForegroundColor Red
    }
    exit 1
}

$children = @(Get-CimInstance -ClassName Win32_Process -Filter "ParentProcessId=$svc_pid" -ErrorAction SilentlyContinue)
if ($children.Count -gt 0) {
    $bot_pid = $children[0].ProcessId
    Write-Host "[ok] PancakeBotDry running (service pid=$svc_pid, bot child pid=$bot_pid)" -ForegroundColor Green
} else {
    Write-Host "[ok] PancakeBotDry running (service pid=$svc_pid, no bot child -- may have been refused by mode mutex OR first ~1-2s)" -ForegroundColor Green
}
