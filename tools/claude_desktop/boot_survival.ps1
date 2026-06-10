<#
.SYNOPSIS
  OPERATOR-UI boot-survival chain (opt-in) — autologon + at-logon elevated
  relaunch of the Claude desktop app + AUMID stamping. Idempotent.

.DESCRIPTION
  NONE of this is required for the trading bot (which runs on the Linux VM).
  This chain exists only so the OPERATOR's Claude desktop app comes back
  elevated after a reboot of the Windows operator machine. Run standalone:
      powershell -ExecutionPolicy Bypass -File tools\claude_desktop\boot_survival.ps1

  The chain (per the live operator host as of 2026-06-04):
    1. Sysinternals Autologon  -> restores the interactive desktop session
       (delegated to setup_autologon.ps1, co-located in this directory).
    2. ClaudeLaunchElevated scheduled task (AtLogon, RunLevel=Highest) ->
       runs C:\Tools\launch_claude_admin_direct.vbs, which queries the
       registered Claude AppX package (PackageFamilyName Claude_pzs8sxrjxfjjc)
       and launches Claude.exe directly (bypasses UWP activation to preserve
       elevation), then stamps the window AUMID via
       C:\Tools\stamp_claude_aumid.exe.
    3. ClaudeKeepalive scheduled task (every ~5 min, RunLevel=Highest) ->
       runs the SAME launcher in /keepalive mode: relaunches Claude if it has
       died mid-session. The AtLogon trigger fires only at logon, so on a long-
       lived session a mid-session death would otherwise leave Claude down until
       the next logon. Launch-if-down only; the AppXSvc/reboot recovery cascade
       stays in the AtLogon task (a persistent lock must not be hit every 5 min).

  The launcher VBS (launch_claude_admin_direct.vbs) is repo-tracked in this
  directory (tools\claude_desktop\) and is DEPLOYED to ToolsDir by this
  script, so a fresh clone reproduces it. The AUMID stamper
  (stamp_claude_aumid.exe) + Autologon.exe remain out-of-repo binaries —
  this script verifies the stamper exists. See AUMID_stamper\README.md
  (same directory) for how to rebuild the stamper.
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
    throw "missing AUMID stamper: $stamper (out-of-repo binary; see tools\claude_desktop\AUMID_stamper\README.md to rebuild)"
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

# 4. ClaudeKeepalive scheduled task -- periodic launch-if-down (~5 min,
#    indefinite) so a mid-session Claude death is recovered without waiting for
#    the next logon. Runs the SAME launcher in /keepalive mode (Status-gate +
#    relaunch-if-down, but NO AppXSvc/reboot cascade -- that stays in the AtLogon
#    task). Time-anchored (-Once + repetition), NOT AtLogon, so it starts mid-
#    session rather than waiting for the next logon (the exact gap this closes).
$KeepName = "ClaudeKeepalive"
$existingKeep = Get-ScheduledTask -TaskName $KeepName -ErrorAction SilentlyContinue
if ($existingKeep) {
    Log "scheduled task '$KeepName' already exists; re-registering to match this definition"
    Unregister-ScheduledTask -TaskName $KeepName -Confirm:$false
}
$kaAction = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`" /keepalive"
$kaTrigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) -RepetitionInterval (New-TimeSpan -Minutes 5)
$kaPrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -RunLevel Highest
$kaSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
try { $kaSettings.MultipleInstancesPolicy = 'IgnoreNew' } catch { }
Register-ScheduledTask -TaskName $KeepName -Action $kaAction -Trigger $kaTrigger `
    -Principal $kaPrincipal -Settings $kaSettings | Out-Null
Log "registered '$KeepName' (every 5 min, indefinite, RunLevel=Highest -> $vbs /keepalive)"

Log "DONE (operator-UI only; the bot's reboot survival is the SCM services, independent of this)"
