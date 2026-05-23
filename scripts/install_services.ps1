<#
.SYNOPSIS
Install the PancakeBotLive and PancakeBotDry Windows services + configure SCM
recovery actions.

.DESCRIPTION
Replaces the legacy schtask-based supervisor architecture
(scripts/install_supervisor_schtasks.ps1) with real Windows Service
semantics. The legacy schtasks are NOT removed by this script — see
scripts/uninstall_old_supervisor.ps1 for that, which is intended to run
after the new services have been validated for at least a week of
soak time.

Behavior:
  1. Registers PancakeBotLive and PancakeBotDry with the SCM via pywin32.
  2. Sets recovery actions: restart after 60s on each of the first three
     failures, reset failure count every 24h.
  3. Sets both services to "Manual" start (start= demand). Use
     scripts/enable_live.ps1 / enable_dry.ps1 to flip to Automatic and
     start the service.

Idempotent: re-running removes + re-installs both services.

.PARAMETER VenvPython
Optional. Path to the venv python.exe. Defaults to .venv\Scripts\python.exe
under the repo root.

.NOTES
Requires administrator privileges. The service install captures sys.path
into the registry's PythonPath under each service's Parameters key — we
run the install from the repo root so the path includes the repo root
and the ``pancakebot`` package is importable when SCM starts the service.
#>
[CmdletBinding()]
param(
    [string]$VenvPython
)

$ErrorActionPreference = "Stop"

# Resolve repo root: this script lives at <repo>\scripts\install_services.ps1.
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)

if (-not $VenvPython) {
    # Use python.exe (not pythonw.exe) so pywin32's HandleCommandLine has
    # stdout/stderr for install diagnostics. The service host (pythonservice.exe)
    # is a separate binary chosen by pywin32 at install time and is
    # console-less in service runtime regardless of this choice.
    $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}

if (-not (Test-Path $VenvPython)) {
    throw "Python executable not found: $VenvPython"
}

# Admin check.
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "install_services.ps1 requires Administrator privileges (sc.exe failure + service install both need elevation)."
}

Write-Host "=== PancakeBot service installer ===" -ForegroundColor Cyan
Write-Host "Repo root   : $RepoRoot"
Write-Host "Venv python : $VenvPython"
Write-Host ""

# --- Step 1: pywin32 service registration via HandleCommandLine -------------

Push-Location $RepoRoot
try {
    foreach ($module in @("pancakebot.service.live_service", "pancakebot.service.dry_service")) {
        $svcName = if ($module -match "live") { "PancakeBotLive" } else { "PancakeBotDry" }

        # Idempotent install: remove first if it exists, ignore failure.
        Write-Host "[$svcName] removing any prior registration..." -ForegroundColor DarkGray
        try {
            & $VenvPython -m $module remove 2>&1 | Out-Null
        } catch {
            # ignored — service may not exist yet
        }

        Write-Host "[$svcName] installing..." -ForegroundColor Green
        & $VenvPython -m $module install
        if ($LASTEXITCODE -ne 0) {
            throw "$module install exited with code $LASTEXITCODE"
        }

        # Configure recovery actions: restart after 60s on each of the first
        # three failures, reset failure count every 24h.
        Write-Host "[$svcName] configuring recovery actions..." -ForegroundColor Green
        & sc.exe failure $svcName reset= 86400 actions= restart/60000/restart/60000/restart/60000 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "sc.exe failure $svcName exited with code $LASTEXITCODE"
        }
        # flag= 1: only count non-stop exits as failures (operator-initiated
        # stops via Stop-Service / sc.exe stop do NOT trigger recovery).
        & sc.exe failureflag $svcName 1 | Out-Null

        # Default to Disabled start so a misconfigured / partial install
        # cannot auto-start at boot. enable_*.ps1 flips to Automatic when
        # the operator is ready to run.
        Write-Host "[$svcName] setting start type = Disabled..." -ForegroundColor Green
        & sc.exe config $svcName start= disabled | Out-Null

        # Service dependencies: Dnscache + NlaSvc. SCM will not start the
        # bot supervisor until DNS resolution AND network location
        # awareness are both ready. This kills the boot-time race where
        # the bot's first chain RPC failed because DNS wasn't resolving
        # yet (caught 2026-05-23 reboot: NameResolutionError on
        # bsc-dataseed1.binance.org at uptime ~17s; service crashed
        # trying to alert Discord with the failure).
        # NOTE: the NLA service's registered name is ``NlaSvc``, not ``Nla``.
        # Using ``Nla`` here returns SC_E_DEPENDENCY error 1075 at start
        # ("dependency service does not exist") — verified live 2026-05-23.
        Write-Host "[$svcName] setting dependencies = Dnscache, NlaSvc..." -ForegroundColor Green
        & sc.exe config $svcName depend= Dnscache/NlaSvc | Out-Null
    }
}
finally {
    Pop-Location
}

# --- Step 2: post-install fixups for venv-hosted pywin32 services ----------
#
# pywin32 311's ``install`` step has known issues when the host Python is in
# a venv: (a) pythonservice.exe is moved to .venv\ but its DLL dependencies
# (python313.dll, pywintypes313.dll, ...) end up in .venv\Scripts\ where it
# can't find them; (b) the registry's PythonClass value uses a file-path
# rather than a dotted module name when install was invoked via 'python -m';
# (c) PythonPath is not written, and pythonservice.exe's embedded Python
# does NOT honor pyvenv.cfg, so site-packages is missing from sys.path.
# We patch all three here so the install is reproducible end-to-end.

# (a) DLL copy: pythonservice.exe is at .venv\; copy required DLLs there.
$venv = Join-Path $RepoRoot ".venv"
$venvScripts = Join-Path $venv "Scripts"
$venvPwin32 = Join-Path $venv "Lib\site-packages\pywin32_system32"
$dlls = @(
    @{ Src = (Join-Path $venvScripts  "python313.dll");     Dest = $venv },
    @{ Src = (Join-Path $venvScripts  "python3.dll");       Dest = $venv },
    @{ Src = (Join-Path $venvScripts  "pywintypes313.dll"); Dest = $venv },
    @{ Src = (Join-Path $venvScripts  "vcruntime140.dll");  Dest = $venv },
    @{ Src = (Join-Path $venvScripts  "vcruntime140_1.dll"); Dest = $venv },
    @{ Src = (Join-Path $venvPwin32   "pythoncom313.dll");  Dest = $venv }
)
foreach ($d in $dlls) {
    if (Test-Path $d.Src) {
        Copy-Item -Path $d.Src -Destination $d.Dest -Force
    }
}
Write-Host "[post] copied required DLLs into $venv" -ForegroundColor Green

# (b)+(c) Registry fixups: PythonClass dotted path + PYTHONPATH env var.
$pythonPathParts = @(
    $RepoRoot
    (Join-Path $venv "Lib\site-packages")
    (Join-Path $venv "Lib\site-packages\win32")
    (Join-Path $venv "Lib\site-packages\win32\lib")
    (Join-Path $venv "Lib\site-packages\Pythonwin")
) -join ';'

foreach ($svc in @(
    @{ Name = "PancakeBotLive"; Class = "pancakebot.service.live_service.PancakeBotLiveService" },
    @{ Name = "PancakeBotDry";  Class = "pancakebot.service.dry_service.PancakeBotDryService" }
)) {
    $svcKey = "HKLM:\SYSTEM\CurrentControlSet\Services\$($svc.Name)"
    $classKey = Join-Path $svcKey "PythonClass"
    if (-not (Test-Path $classKey)) { New-Item -Path $classKey -Force | Out-Null }
    Set-ItemProperty -Path $classKey -Name '(default)' -Value $svc.Class
    # PYTHONPATH as a service env var (REG_MULTI_SZ on the service key
    # itself, NOT under a Parameters subkey — that's SCM's contract). The
    # embedded Python in pythonservice.exe reads PYTHONPATH from the
    # service's environment at startup.
    Set-ItemProperty -Path $svcKey -Name 'Environment' -Type MultiString -Value @("PYTHONPATH=$pythonPathParts")
    Write-Host "[post] $($svc.Name): PythonClass + Environment registry values set" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Install complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Verify:" -ForegroundColor Yellow
Write-Host "  Get-Service PancakeBotLive, PancakeBotDry | Format-Table -AutoSize"
Write-Host "  sc.exe qfailure PancakeBotLive"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  scripts\enable_live.ps1   # flip to Automatic + start (live mode)"
Write-Host "  scripts\enable_dry.ps1    # flip to Automatic + start (dry mode)"
Write-Host ""
Write-Host "Legacy schtasks (PancakeBotSupervisor{Dry,Live}) are NOT touched by"
Write-Host "this installer. Disable them via Disable-ScheduledTask, then delete"
Write-Host "with scripts\uninstall_old_supervisor.ps1 once the new services"
Write-Host "have been validated."
