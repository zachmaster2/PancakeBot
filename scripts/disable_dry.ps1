<#
.SYNOPSIS
Stop + disable the PancakeBotDry service. Idempotent.
#>
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

$svc = Get-Service -Name PancakeBotDry -ErrorAction SilentlyContinue
if ($null -eq $svc) {
    Write-Host "[ok] PancakeBotDry not installed; nothing to disable" -ForegroundColor Green
    exit 0
}

Set-Service -Name PancakeBotDry -StartupType Disabled

if ($svc.Status -eq 'Stopped') {
    Write-Host "[ok] PancakeBotDry already stopped" -ForegroundColor Green
    exit 0
}

Write-Host "[..] stopping PancakeBotDry..." -ForegroundColor Yellow
Stop-Service -Name PancakeBotDry -Force
$svc.WaitForStatus('Stopped', '00:01:00')
Write-Host "[ok] PancakeBotDry stopped + disabled" -ForegroundColor Green
