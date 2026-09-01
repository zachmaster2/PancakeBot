<#
    Daily market-data sync wrapper. Runs under Task Scheduler as SYSTEM.

    WHY A WRAPPER RATHER THAN POINTING THE TASK AT run.py DIRECTLY.
      1. Record a heartbeat BEFORE and AFTER, so "ran and failed" is
         distinguishable from "never fired". Task Scheduler's own Last
         Result cannot make that distinction: a disabled task, or one whose
         trigger never fires, simply keeps yesterday's result.
      2. Preserve the real exit code. The lock-held case (3) is a normal
         outcome, not a failure.
      3. Leave a durable log, because a console nobody sees is not output.

    ------------------------------------------------------------------
    2026-09-01 INCIDENT -- READ BEFORE CHANGING THE ERROR HANDLING.

    This wrapper previously began with `$ErrorActionPreference = 'Stop'`
    and ran the sync as `& $Py ... *>&1 | ForEach-Object {...}`.

    In PowerShell 5.1, merging a NATIVE command's stderr wraps each stderr
    line in an ErrorRecord. Under 'Stop' that becomes a TERMINATING error,
    so the wrapper aborted mid-stream the first time anything wrote a
    single line to stderr. The result:

      * the log stopped mid-run with no RESULT line and no traceback
      * record_failure never ran
      * the health file still read consecutive_failures=0, last_exit_code=0
      * a FAILED run was indistinguishable from one that had not happened

    It survived two earlier runs purely because nothing happened to write
    to stderr. It would have aborted a SUCCESSFUL run just as readily.

    THE RULES THAT FOLLOW FROM IT:
      * ErrorActionPreference stays 'Continue' around the native call.
        Do not set it to 'Stop' at script scope.
      * stderr is captured with 2>&1 into the log INSIDE the pipeline, and
        never allowed to become a terminating error.
      * The outcome is recorded in a finally block, so no failure mode --
        including ones nobody has thought of -- can leave the health file
        reading clean.
    ------------------------------------------------------------------

    EXIT CODES
      0  sync succeeded
      3  another sync was already running -- NOT a failure; the lock did
         its job and this run correctly declined to double up
      *  anything else is a real failure and is recorded as one

    RETRY: there is none here, deliberately. The task's RestartCount does
    NOT fire on a non-zero exit code (it covers unexpected termination), so
    claiming retry protection would be false. The retry is tomorrow's run;
    the sync is idempotent and resumable, so a missed day costs nothing but
    a day. See docs -- do not add RestartCount back believing it retries.
#>

# NOT 'Stop'. See the incident note above.
$ErrorActionPreference = 'Continue'

$Repo   = 'C:\Users\zking\Documents\GitHub\PancakeBot'
$Py     = Join-Path $Repo '.venv\Scripts\python.exe'
$LogDir = Join-Path $Repo 'var\sync_logs'

Set-Location $Repo
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$log   = Join-Path $LogDir "sync_$stamp.log"

"=== PancakeBot daily sync $stamp UTC ===" | Out-File -FilePath $log -Encoding utf8

# Heartbeat BEFORE the run. If last_attempt never advances, the problem is
# the schedule, not the sync. If it advances and no outcome ever follows,
# assess() reports ORPHANED -- which is what makes an unanticipated death
# visible without anyone having predicted it.
& $Py -c "import sys; sys.path.insert(0,r'$Repo'); from pancakebot.ops.sync_health import record_attempt; record_attempt()"

$code = $null
try {
    # STRUCTURAL, not merely handled. cmd.exe merges stderr into stdout at
    # the OS LEVEL, so PowerShell receives a single stdout stream and never
    # sees an error stream at all. A stderr line therefore CANNOT become an
    # ErrorRecord, and this survives even if someone later sets
    # $ErrorActionPreference = 'Stop' -- which is the most common PowerShell
    # idiom and exactly what caused the 2026-09-01 incident.
    #
    # Do NOT replace this with `& $Py ... 2>&1 |`. That is the construct
    # that failed. Verified: under 'Stop' the PowerShell redirect aborts and
    # this does not.
    #
    # -u keeps python unbuffered so stdout and stderr interleave in true
    # order; without it stdout block-buffers and stderr jumps ahead.
    # cmd propagates the child's exit code, so $LASTEXITCODE is correct.
    & cmd.exe /c "`"$Py`" -u run.py --sync 2>&1" | ForEach-Object {
        $line = [string]$_
        $line | Out-File -FilePath $log -Append -Encoding utf8
        Write-Output $line
    }
    $code = $LASTEXITCODE
}
catch {
    # Belt and braces: if anything still manages to throw, capture it as
    # the outcome rather than letting the script die silently.
    $code = 90
    ("WRAPPER EXCEPTION: " + $_.Exception.GetType().Name + ": " +
     $_.Exception.Message) | Out-File -FilePath $log -Append -Encoding utf8
}
finally {
    if ($null -eq $code) { $code = 91 }   # died before the exit code was read

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
        # Pass the diagnostic tail via a FILE, not interpolated into a
        # python -c string: log lines are arbitrary text and quoting them
        # into a command line is its own failure mode.
        $tailFile = Join-Path $env:TEMP ("pb_sync_tail_" + $stamp + ".txt")
        try {
            (Get-Content $log -Tail 25 -ErrorAction SilentlyContinue) -join "`n" |
                Out-File -FilePath $tailFile -Encoding utf8
        } catch { }
        & $Py -c "import sys,io,os; sys.path.insert(0,r'$Repo'); from pancakebot.ops.sync_health import record_failure; p=r'$tailFile'; t=io.open(p,encoding='utf-8').read() if os.path.exists(p) else ''; record_failure(exit_code=$code, error=t)"
        Remove-Item $tailFile -Force -ErrorAction SilentlyContinue
        "RESULT: FAILED exit=$code" | Out-File -FilePath $log -Append -Encoding utf8
    }

    # Retain 30 days of logs. Unbounded logs on the machine that now holds
    # the only copy of the data is its own small risk.
    Get-ChildItem $LogDir -Filter 'sync_*.log' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -Skip 30 |
        Remove-Item -Force -ErrorAction SilentlyContinue

    # Update the Desktop marker either way: clears it on success, raises it
    # on a stall. Never allowed to change this script's exit code.
    try {
        & $Py scripts\sync_watchdog.py --no-discord |
            Out-File -FilePath $log -Append -Encoding utf8
    } catch { }
}

exit $code
