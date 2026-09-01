<#
    Watchdog wrapper. Exists ONLY to capture the watchdog's stdout.

    WHY. The watchdog task previously invoked python.exe directly, so its
    output went nowhere: Task Scheduler retains the exit code and nothing
    else. A raised Desktop marker had no accompanying record of WHY.

    ------------------------------------------------------------------
    2026-09-01: THIS SCRIPT HAD THE SAME DEFECT AS daily_sync.ps1.

    `$ErrorActionPreference = 'Stop'` plus `*>&1` on a native command makes
    every stderr line a TERMINATING error in PowerShell 5.1, aborting the
    wrapper mid-stream. In the sync that produced a failed run with a
    health file still reading clean.

    Here it would be WORSE. A watchdog that dies silently removes the very
    thing that reports silence -- the alarm and the thing it watches fail
    together, and nothing is left to notice either. The exit code would
    still reach Task Scheduler, but nobody reads LastTaskResult daily; the
    Desktop marker is the channel that works, and a dead watchdog never
    writes one.

    So: ErrorActionPreference stays 'Continue', stderr is captured as data,
    and the exit code is resolved in a finally block.
    ------------------------------------------------------------------

    Exit code is passed through unchanged: 0 = healthy, 1 = not healthy
    (stale, never-run, orphaned attempt, or a store integrity failure).

    --no-discord is deliberate and must stay. Discord is the operator's
    channel; an automated job is not the thing that should be posting to
    it, and a webhook failure must never affect the exit code that reports
    store health.
#>

# NOT 'Stop'. See the incident note above.
$ErrorActionPreference = 'Continue'

$Repo   = 'C:\Users\zking\Documents\GitHub\PancakeBot'
$Py     = Join-Path $Repo '.venv\Scripts\python.exe'
$LogDir = Join-Path $Repo 'var\watchdog_logs'

Set-Location $Repo
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$log   = Join-Path $LogDir "watchdog_$stamp.log"

"=== PancakeBot watchdog $stamp UTC ===" | Out-File -FilePath $log -Encoding utf8

$code = $null
try {
    & $Py scripts\sync_watchdog.py --no-discord 2>&1 | ForEach-Object {
        $line = [string]$_
        $line | Out-File -FilePath $log -Append -Encoding utf8
        Write-Output $line
    }
    $code = $LASTEXITCODE
}
catch {
    $code = 90
    ("WRAPPER EXCEPTION: " + $_.Exception.GetType().Name + ": " +
     $_.Exception.Message) | Out-File -FilePath $log -Append -Encoding utf8
}
finally {
    if ($null -eq $code) { $code = 91 }

    $meaning = if ($code -eq 0) { 'healthy' }
               elseif ($code -eq 90 -or $code -eq 91) { 'THE WATCHDOG ITSELF FAILED -- no marker was written' }
               else { 'NOT healthy: stale, never-run, orphaned attempt, or a store integrity failure -- a Desktop marker was raised' }
    "RESULT: exit=$code  ($meaning)" | Out-File -FilePath $log -Append -Encoding utf8

    # Retain 60 days. The watchdog runs daily and its logs are tiny, but
    # this machine holds the only copy of the data and unbounded growth
    # anywhere in var\ is its own small risk.
    Get-ChildItem $LogDir -Filter 'watchdog_*.log' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -Skip 60 |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

exit $code
