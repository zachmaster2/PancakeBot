<#
.SYNOPSIS
  Thin wrapper that delegates to the existing scripts\setup_autologon.ps1
  (Sysinternals Autologon) — not reinvented here.

.DESCRIPTION
  Autologon restores an interactive desktop session at boot. It is ONLY needed
  for the operator-facing Claude desktop app to relaunch — the trading bot's
  reboot survival depends solely on the SCM Automatic-start services, NOT on
  autologon. Invoked by boot_survival.ps1 when -IncludeOperatorUI is set.
#>
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")).Path
$existing = Join-Path $RepoRoot "scripts\setup_autologon.ps1"
if (-not (Test-Path $existing)) { throw "scripts\setup_autologon.ps1 not found at $existing" }
Write-Host "[setup_autologon] delegating to $existing"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $existing @args
