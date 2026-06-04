<#
.SYNOPSIS
  PancakeBot Windows installer — fresh-clone to ready, idempotent + verbose.

.DESCRIPTION
  Run from an ELEVATED PowerShell:
      git clone <repo> ; cd PancakeBot
      powershell -ExecutionPolicy Bypass -File bootstrap\install.ps1

  Steps (each idempotent + logged):
    1. venv + dependencies                 bootstrap\common\python_setup.py
    2. config + secrets present?           bootstrap\common\config_check.py
    3. register services (pythonservice    bootstrap\windows\setup_service.py
       host + DLL/registry + policy)       (-> scripts\install_services.ps1)
    4. (opt-in) operator-UI boot survival  bootstrap\windows\boot_survival.ps1

  Leaves PancakeBotLive / PancakeBotDry registered but DISABLED. Does NOT touch
  a service that is already RUNNING (so re-running on the live host is safe).
  Reverse with: bootstrap\uninstall.ps1

.PARAMETER IncludeOperatorUI
  Also set up the Claude-desktop boot-survival chain (autologon + AUMID). Off
  by default — most operators do not want a stored autologon password.
#>
[CmdletBinding()]
param(
    [string]$Python = "py -3.13",
    [switch]$IncludeOperatorUI
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Log($m) { Write-Host "[install] $m" }

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $Here "..")).Path
$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
Log "repo root: $RepoRoot"

# Elevation check (service registration needs admin).
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $isAdmin) { throw "must run from an elevated PowerShell (service registration needs admin)" }

Log "STEP 1/4: venv + dependencies"
# Resolve the bootstrap interpreter (py launcher form or an explicit path).
$pyArgs = $Python.Split(" ")
& $pyArgs[0] $pyArgs[1..($pyArgs.Length-1)] (Join-Path $Here "common\python_setup.py") `
    --python (& $pyArgs[0] $pyArgs[1..($pyArgs.Length-1)] -c "import sys;print(sys.executable)") `
    --venv (Join-Path $RepoRoot ".venv") --requirements (Join-Path $RepoRoot "requirements.txt")

Log "STEP 2/4: config + secrets check"
& $VenvPy (Join-Path $Here "common\config_check.py")

Log "STEP 3/4: register services + policy (leaves DISABLED; skips a RUNNING service)"
& $VenvPy (Join-Path $Here "windows\setup_service.py")

if ($IncludeOperatorUI) {
    Log "STEP 4/4: operator-UI boot survival (autologon + Claude relaunch + AUMID)"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here "windows\boot_survival.ps1")
} else {
    Log "STEP 4/4: operator-UI boot survival SKIPPED (pass -IncludeOperatorUI to enable)"
}

Log "DONE. Services registered + DISABLED. Start the dry soak with: scripts\enable_dry.ps1"
Log "Validate with: $VenvPy $Here\common\health_check.py --mode dry --service-name PancakeBotDry"
