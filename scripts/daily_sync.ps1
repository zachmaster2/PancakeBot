<#
    Daily market-data sync wrapper. Runs under Task Scheduler as SYSTEM.

    WHY A WRAPPER RATHER THAN POINTING THE TASK AT run.py DIRECTLY.
    The task needs three things run.py does not do by itself:

      1. Record a heartbeat BEFORE and AFTER, so that "ran and failed" is
         distinguishable from "never fired". Task Scheduler's own Last
         Result cannot make that distinction: a task that is disabled, or
         whose trigger never fires, simply keeps yesterday's result.
      2. Preserve the real exit code. The lock-held case (3) is a normal
         outcome, not a failure, and must not be recorded as one.
      3. Leave a durable log, because a console nobody sees is not output.

    EXIT CODES
      0  sync succeeded
      3  another sync was already running -- NOT a failure; the lock did
         its job and this run correctly declined to double up
      *  anything else is a real failure and is recorded as one
#>

$ErrorActionPreference = 'Stop'

$Repo   = 'C:\Users\zking\Documents\GitHub\PancakeBot'
$Py     = Join-Path $Repo '.venv\Scripts\python.exe'
$LogDir = Join-Path $Repo 'var\sync_logs'

Set-Location $Repo
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$log   = Join-Path $LogDir "sync_$stamp.log"

# Heartbeat BEFORE the run. This is what makes "the task never fired"
# visible: if last_attempt never advances, the problem is the schedule,
# not the sync.
& $Py -c "import sys; sys.path.insert(0,r'$Repo'); from pancakebot.ops.sync_health import record_attempt; record_attempt()"

"=== PancakeBot daily sync $stamp UTC ===" | Out-File -FilePath $log -Encoding utf8

& $Py -u run.py --sync *>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE

if ($code -eq 0) {
    & $Py -c "import sys; sys.path.insert(0,r'$Repo'); from pancakebot.ops.sync_health import record_success; record_success(detail='scheduled run')"
    "RESULT: SUCCESS" | Out-File -FilePath $log -Append -Encoding utf8
}
elseif ($code -eq 3) {
    # The lock held. A previous sync was still running. This is the
    # single-instance guard working as designed, so it must not be
    # recorded as a failure -- doing so would make a correctly-behaving
    # system look broken and train the operator to ignore the alarm.
    "RESULT: SKIPPED (another sync holds the lock)" | Out-File -FilePath $log -Append -Encoding utf8
}
else {
    $tail = (Get-Content $log -Tail 25 -ErrorAction SilentlyContinue) -join "`n"
    $esc  = $tail -replace "'", "''"
    & $Py -c "import sys; sys.path.insert(0,r'$Repo'); from pancakebot.ops.sync_health import record_failure; record_failure(exit_code=$code, error='''$esc''')"
    "RESULT: FAILED exit=$code" | Out-File -FilePath $log -Append -Encoding utf8
}

# Retain 30 days of logs. Unbounded logs on the machine that now holds the
# only copy of the data is its own small risk.
Get-ChildItem $LogDir -Filter 'sync_*.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip 30 |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Update the Desktop marker either way: clears it on success, raises it on
# a stall. Never allowed to change this script's exit code.
try {
    & $Py scripts\sync_watchdog.py | Out-File -FilePath $log -Append -Encoding utf8
} catch { }

exit $code
