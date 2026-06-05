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

  The launcher VBS (launch_claude_admin_direct.vbs) is repo-tracked under
  bootstrap\windows\ and is DEPLOYED to ToolsDir by this script, so a fresh
  clone + install.ps1 -IncludeOperatorUI reproduces it. The AUMID stamper
  (stamp_claude_aumid.exe) + Autologon remain out-of-repo binaries — this
  script verifies the stamper exists. See bootstrap\windows\AUMID_stamper\
  README.md for how to rebuild the stamper.
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

# 1. Deploy the repo-tracked launcher VBS to ToolsDir; verify the (out-of-repo,
#    binary) AUMID stamper is present.
$repoVbs = Join-Path $Here "launch_claude_admin_direct.vbs"
if (-not (Test-Path $repoVbs)) {
    throw "missing repo launcher: $repoVbs"
}
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
Copy-Item $repoVbs $vbs -Force
Log "deployed launcher: $repoVbs -> $vbs"
if (-not (Test-Path $stamper)) {
    throw "missing AUMID stamper: $stamper (out-of-repo binary; see bootstrap\windows\AUMID_stamper\README.md to rebuild)"
}
Log "AUMID stamper present: $stamper"

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
