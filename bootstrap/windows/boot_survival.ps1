<#
.SYNOPSIS
  OPERATOR-UI boot-survival chain (opt-in) — autologon + at-logon elevated
  relaunch of the Claude desktop app + AUMID stamping. Idempotent.

.DESCRIPTION
  NONE of this is required for the trading bot. The bot survives reboots via
  the SCM Automatic-start services alone (no logon session needed). This chain
  exists only so the OPERATOR's Claude desktop app comes back elevated after a
  reboot. It is opt-in (install.ps1 -IncludeOperatorUI).

  The chain (per the live Windows host as of 2026-06-04):
    1. Sysinternals Autologon  -> restores the interactive desktop session
       (delegated to setup_autologon.ps1 / scripts\setup_autologon.ps1).
    2. ClaudeLaunchElevated scheduled task (AtLogon, RunLevel=Highest) ->
       runs C:\Tools\launch_claude_admin_direct.vbs, which queries the
       registered Claude AppX package (PackageFamilyName Claude_pzs8sxrjxfjjc)
       and launches Claude.exe directly (bypasses UWP activation to preserve
       elevation), then stamps the window AUMID via
       C:\Tools\stamp_claude_aumid.exe.

  The C:\Tools payload (Autologon, the .vbs launchers, stamp_claude_aumid.exe +
  src) currently lives OUTSIDE the repo. This script does not recreate those
  binaries — it verifies they exist and wires the scheduled task. See
  bootstrap\windows\AUMID_stamper\README.md for how to rebuild the stamper.
#>
[CmdletBinding()]
param(
    [string]$ToolsDir = "C:\Tools",
    [string]$TaskName = "ClaudeLaunchElevated"
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
function Log($m) { Write-Host "[boot_survival] $m" }

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vbs = Join-Path $ToolsDir "launch_claude_admin_direct.vbs"
$stamper = Join-Path $ToolsDir "stamp_claude_aumid.exe"

# 1. Verify the out-of-repo Tools payload.
foreach ($p in @($vbs, $stamper)) {
    if (-not (Test-Path $p)) {
        throw "missing operator-UI artifact: $p (see bootstrap\windows\AUMID_stamper\README.md)"
    }
}
Log "Tools payload present: $vbs, $stamper"

# 2. Autologon (delegated).
Log "configuring autologon"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here "setup_autologon.ps1")

# 3. ClaudeLaunchElevated scheduled task (idempotent: replace if present).
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Log "scheduled task '$TaskName' already exists; re-registering to match this definition"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings | Out-Null
Log "registered '$TaskName' (AtLogon, RunLevel=Highest -> $vbs)"
Log "DONE (operator-UI only; the bot's reboot survival is the SCM services, independent of this)"
