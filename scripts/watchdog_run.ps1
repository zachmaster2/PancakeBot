<#
    Watchdog wrapper. Exists ONLY to capture the watchdog's stdout.

    WHY. The watchdog task previously invoked python.exe directly, so its
    output went nowhere. Task Scheduler retains the exit code and nothing
    else. That meant a raised Desktop marker had no accompanying record of
    WHY it was raised -- the marker said a stall had happened, and the
    reason had to be reconstructed from whatever else happened to be on
    disk.

    That is the same shape as every failure this project has hit: the
    signal existed but the explanation did not survive. A system built to
    make silence legible should not itself go quiet about its own
    decisions.

    Exit code is passed through unchanged: 0 = healthy, 1 = stale/never-run.
    Task Scheduler's Last Result therefore still carries the signal, and
    this wrapper only adds the narrative.

    --no-discord is deliberate and must stay. Discord is the operator's
    channel; an automated job is not the thing that should be posting to
    it, and a webhook failure must never be able to affect the exit code
    that reports store health.
#>

$ErrorActionPreference = 'Continue'

$Repo   = 'C:\Users\zking\Documents\GitHub\PancakeBot'
$Py     = Join-Path $Repo '.venv\Scripts\python.exe'
$LogDir = Join-Path $Repo 'var\watchdog_logs'

Set-Location $Repo
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$log   = Join-Path $LogDir "watchdog_$stamp.log"

"=== PancakeBot watchdog $stamp UTC ===" | Out-File -FilePath $log -Encoding utf8

# Append per line with an EXPLICIT encoding rather than via Tee-Object.
# Tee-Object on PS 5.1 has no -Encoding and defaults to UTF-16LE, which
# produced a file whose header was UTF-8 and whose body was UTF-16 -- still
# readable by eye, but grep treats it as binary, and a log you cannot grep
# is most of the way back to having no log.
& $Py scripts\sync_watchdog.py --no-discord *>&1 | ForEach-Object {
    $line = [string]$_
    $line | Out-File -FilePath $log -Append -Encoding utf8
    Write-Output $line
}
$code = $LASTEXITCODE

"RESULT: exit=$code  ($(if ($code -eq 0) { 'healthy' } else { 'STALE or NEVER_RUN - a Desktop marker was raised' }))" |
    Out-File -FilePath $log -Append -Encoding utf8

# Retain 60 days. The watchdog runs daily and its logs are tiny, but this
# machine now holds the only copy of the data and unbounded growth anywhere
# in var\ is its own small risk.
Get-ChildItem $LogDir -Filter 'watchdog_*.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip 60 |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $code
