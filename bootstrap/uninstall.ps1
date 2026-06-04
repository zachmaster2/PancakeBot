<#
.SYNOPSIS
  Reverse bootstrap\install.ps1 — idempotent. Stops + removes PancakeBotLive /
  PancakeBotDry. Does NOT delete var\ state, .env, config.toml, or the venv
  unless -Purge is given. Run elevated.

      powershell -ExecutionPolicy Bypass -File bootstrap\uninstall.ps1 [-Purge]
#>
[CmdletBinding()]
param([switch]$Purge)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
function Log($m) { Write-Host "[uninstall] $m" }

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $isAdmin) { throw "must run elevated" }

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $Here "..")).Path

foreach ($svc in @("PancakeBotLive", "PancakeBotDry")) {
    $existing = Get-Service -Name $svc -ErrorAction SilentlyContinue
    if ($existing) {
        Log "stopping + deleting $svc"
        if ($existing.Status -ne "Stopped") { Stop-Service -Name $svc -Force -ErrorAction SilentlyContinue }
        & sc.exe delete $svc | Out-Null
    } else {
        Log "$svc not installed; skipping"
    }
}
Log "services removed"

if ($Purge) {
    Log "-Purge: removing venv (NOT var\ state or config.toml)"
    $venv = Join-Path $RepoRoot ".venv"
    if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
    Log "purged venv. NOTE: machine-scope webhook env vars + operator-UI (autologon/AUMID) are left as-is -- remove manually if desired."
} else {
    Log "kept venv, .env, config.toml, var\ state (use -Purge to remove the venv)"
}
Log "DONE"
