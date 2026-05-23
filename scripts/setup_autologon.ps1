<#
.SYNOPSIS
Configure Windows Autologon for the current user via Sysinternals Autologon.

.DESCRIPTION
Autologon is the supported way to make Windows sign in automatically at boot
without prompting for credentials - required so that a full reboot brings
the desktop session (and the Claude UWP app launched by the
ClaudeLaunchElevated scheduled task) back up unattended.

This script:
  1. Prompts SECURELY for the Windows password (Read-Host -AsSecureString -
     not visible on screen, not in command history, not in scrollback).
  2. Decrypts the SecureString to a plaintext char buffer JUST LONG ENOUGH
     to pass to Autologon.exe via argv.
  3. Immediately zeroes the BSTR buffer and removes the plaintext variable.
  4. Autologon.exe encrypts the password using Windows LSA and writes it
     into HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon
     (DefaultPassword + AutoAdminLogon). The plaintext NEVER touches disk
     and never leaves memory once Autologon.exe returns.

You can disable Autologon at any time by running:
    & 'C:\Tools\Autologon\Autologon.exe' /accepteula  (then press Disable)
or by clearing HKLM\...\Winlogon\AutoAdminLogon = 0.

.NOTES
Requires Administrator. Autologon.exe path is hard-coded to the
expected install location from install_services.ps1's auto-download.

PowerShell 7+ has cleaner SecureString -> plaintext APIs (ConvertFrom-
SecureString -AsPlainText); we use the BSTR dance to stay compatible
with Windows PowerShell 5.1.
#>
[CmdletBinding()]
param(
    [string]$AutologonExe = 'C:\Tools\Autologon\Autologon.exe'
)
$ErrorActionPreference = 'Stop'

# Admin check - Autologon writes to HKLM and the LSA secret store.
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "setup_autologon.ps1 requires Administrator. Re-launch this shell elevated and try again."
}

if (-not (Test-Path $AutologonExe)) {
    throw "Autologon.exe not found at $AutologonExe. Re-run scripts\install_services.ps1 (which auto-downloads it), or download from https://learn.microsoft.com/en-us/sysinternals/downloads/autologon manually."
}

$user = $env:USERNAME
$domain = $env:USERDOMAIN
Write-Host "Configuring Autologon for: $domain\$user" -ForegroundColor Cyan
Write-Host ""
Write-Host "Password will not be displayed, logged, or stored on disk."
Write-Host "It is decrypted in memory only long enough to pass to Autologon.exe,"
Write-Host "which encrypts it via Windows LSA before writing to the registry."
Write-Host ""

$sec = Read-Host -Prompt "Password for $domain\$user" -AsSecureString
if ($null -eq $sec -or $sec.Length -eq 0) {
    throw "empty password - aborting"
}

# SecureString -> plaintext via BSTR (the safe Windows PowerShell 5.1 idiom).
$bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
try {
    $plain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    # Autologon.exe argv: USERNAME DOMAIN PASSWORD [/accepteula]
    & $AutologonExe $user $domain $plain '/accepteula'
    $exit = $LASTEXITCODE
}
finally {
    # ZeroFreeBSTR overwrites the BSTR with zeros before freeing -
    # plaintext is wiped from this address before the page can be paged out.
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    Remove-Variable plain -ErrorAction SilentlyContinue
}

# Verify the registry values Autologon should have set.
Write-Host ""
Write-Host "Verifying registry state:" -ForegroundColor Cyan
$winlogonKey = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
$autoEnabled = (Get-ItemProperty $winlogonKey -Name AutoAdminLogon -ErrorAction SilentlyContinue).AutoAdminLogon
$defaultUser = (Get-ItemProperty $winlogonKey -Name DefaultUserName -ErrorAction SilentlyContinue).DefaultUserName
$defaultDomain = (Get-ItemProperty $winlogonKey -Name DefaultDomainName -ErrorAction SilentlyContinue).DefaultDomainName
Write-Host "  AutoAdminLogon    = $autoEnabled   (should be '1' when enabled)"
Write-Host "  DefaultUserName   = $defaultUser"
Write-Host "  DefaultDomainName = $defaultDomain"

if ($autoEnabled -eq '1') {
    Write-Host ""
    Write-Host "[ok] Autologon configured. Next reboot will sign in automatically as $defaultDomain\$defaultUser." -ForegroundColor Green
    Write-Host ""
    Write-Host "BitLocker note: if BitLocker is enabled on C:, Autologon fires AFTER"
    Write-Host "  BitLocker unlock. With TPM-bound unlock this is transparent; with"
    Write-Host "  PIN/key required unlock, Autologon won't help until the disk is"
    Write-Host "  unlocked first."
    Write-Host ""
    Write-Host "To disable later: run 'C:\Tools\Autologon\Autologon.exe' again and press Disable,"
    Write-Host "  or set AutoAdminLogon=0 in the registry directly."
} else {
    Write-Host ""
    Write-Host "[warn] AutoAdminLogon != '1' - Autologon.exe may have failed silently. Re-run and watch its output for errors." -ForegroundColor Yellow
}
