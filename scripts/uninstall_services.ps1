<#
.SYNOPSIS
Uninstall the PancakeBotLive and PancakeBotDry Windows services.

.DESCRIPTION
Stops each service if running, then unregisters from the SCM via pywin32's
``remove`` command. Idempotent: safe to run if either service is already
absent.

This script does NOT touch the legacy schtask supervisor — that is
managed separately by scripts/uninstall_old_supervisor.ps1.
#>
[CmdletBinding()]
param(
    [string]$VenvPython
)
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if (-not $VenvPython) {
    $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $VenvPython)) {
    throw "Python executable not found: $VenvPython"
}

Push-Location $RepoRoot
try {
    foreach ($pair in @(
        @{ Module = "pancakebot.service.live_service"; Name = "PancakeBotLive" },
        @{ Module = "pancakebot.service.dry_service";  Name = "PancakeBotDry"  }
    )) {
        $svcName = $pair.Name
        $module  = $pair.Module
        $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
        if ($null -ne $svc -and $svc.Status -ne 'Stopped') {
            Write-Host "[$svcName] stopping..." -ForegroundColor Yellow
            Stop-Service -Name $svcName -Force
            $svc.WaitForStatus('Stopped', '00:01:00')
        }
        Write-Host "[$svcName] removing..." -ForegroundColor Yellow
        try {
            & $VenvPython -m $module remove
        } catch {
            Write-Host "[$svcName] remove failed (service may not be installed): $_" -ForegroundColor DarkGray
        }
    }
}
finally {
    Pop-Location
}
Write-Host "=== Uninstall complete ===" -ForegroundColor Cyan
