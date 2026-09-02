# Reading the Task Scheduler log — enabled 2026-09-01

## ⚠️ THIS IS A MACHINE SETTING, NOT A REPO SETTING

It lives in the Windows event-log configuration on `ZACH-SURFACE`. It is
**not in git, does not travel with the code, and is not restored by cloning
or pulling.** A Windows reset, a rebuilt profile, a new machine, or some
group-policy change loses it **silently** — the tasks keep running and the
forensic trail simply stops existing, with nothing to announce that.

**Do not assume it is still on.** Check before relying on it:

```powershell
(Get-WinEvent -ListLog 'Microsoft-Windows-TaskScheduler/Operational').IsEnabled
```

Re-enable (needs elevation):

```powershell
wevtutil sl "Microsoft-Windows-TaskScheduler/Operational" /e:true
```

## What it is for

The wrappers write their own logs (`var\sync_logs\`, `var\watchdog_logs\`),
and those are the first place to look. But on 2026-09-01 a wrapper died
mid-run, and the diagnosis only worked because the sync happened to log
right up to the moment it stopped. **A wrapper that dies before writing
anything would leave nothing at all** — and the fallback exit codes 90/91
only help if the wrapper reaches its `finally`.

This channel is written by the Task Scheduler service itself, so it records
the run even when the thing being run writes nothing.

## Bounded — checked, not assumed

```
LogMode            : Circular      overwrites oldest; never blocks, never grows
MaximumSizeInBytes : 10485760      10 MB hard cap
retention          : false         does not preserve when full
autoBackup         : false         no .evtx archives accumulate
fileMax            : 1             a single fixed-size file
```

### ⚠️ THE FORENSIC WINDOW IS ABOUT 4 DAYS, NOT MONTHS

**Corrected 2026-09-02.** This section originally said the channel "will
never approach the cap." That was measured after a handful of runs and was
wrong by roughly an order of magnitude. A wrong number in a forensics guide
is worse than no number, because it gets relied on in November precisely
when it cannot be afforded.

The channel logs **every scheduled task on the machine**, not only ours.
Measured over the first 20.7 hours after enabling:

```
oldest event    2026-09-01 11:25:08
elapsed         20.7 h
size            2,116 KB  of a 10,240 KB cap
rate          ~ 2,451 KB/day
=> retention  ~ 4.2 DAYS before circular overwrite begins

scheduled tasks on this machine : 270
our events                      : 13 of 2,850
```

Ours are **13 events out of 2,850** — well under 1%. The rate is set almost
entirely by the other 268 tasks, so **it will drift as software is
installed and removed.** Re-derive rather than trust this figure:

```powershell
$f     = Get-Item ($env:SystemRoot+'\System32\Winevt\Logs\Microsoft-Windows-TaskScheduler%4Operational.evtx')
$first = (Get-WinEvent -LogName 'Microsoft-Windows-TaskScheduler/Operational' -Oldest -MaxEvents 1).TimeCreated
$hrs   = ((Get-Date) - $first).TotalHours
$kb    = $f.Length/1KB
'{0:N0} KB/day -> {1:N1} days retention' -f ($kb/$hrs*24), (10240/($kb/$hrs*24))
```

If `-Oldest` returns an event close to now, the log has already wrapped and
that IS the window.

**What this means in practice.** The disk cost is still nothing — bounded,
self-rotating, one fixed-size file on a machine with ~72 GB free. But if a
run fails and nobody looks for a week, **the events will have rotated out**,
which is the exact scenario the channel was enabled for. The wrapper's own
logs in `var\sync_logs\` are retained 30 days and `var\watchdog_logs\` 60,
so those remain the longer record; this channel is the shorter-lived but
more trustworthy one, because the service writes it rather than our code.

### The cap is at the Windows default, deliberately

`MaximumSizeInBytes` is untouched at 10 MB. Raising it to 100 MB would buy
roughly **six weeks** of history at the measured rate, for 100 MB against
~72 GB free:

```powershell
wevtutil sl "Microsoft-Windows-TaskScheduler/Operational" /ms:104857600   # needs elevation
```

Not done as of 2026-09-02. Enabling the channel was authorised; changing
its cap is an adjacent but separate decision about a system setting, and
nothing was failing, so it was left for the operator rather than inferred.
Recorded here so the option exists on paper instead of only in a
conversation. If retention matters more later, this is the lever.

## What a HEALTHY run looks like

Seven events, in this order:

```
325  queued instance "{guid}" of task "\PancakeBotSyncWatchdog"
110  launched "{guid}" instance ... for user "System"
129  launch task ... instance "powershell.exe" with process ID 10104
100  started "{guid}" instance ... for user "NT AUTHORITY\SYSTEM"
200  launched action "powershell.exe" in instance "{guid}"
201  successfully completed task ... action "powershell.exe" with return code 0
102  successfully finished "{guid}" instance
```

**Event 201 is the one that matters.** It carries the return code, and it is
written by the service, not by our code.

## TWO GOTCHAS THAT WILL MISLEAD YOU AT 3AM

**1. "successfully completed" does NOT mean the task succeeded.** It means
Task Scheduler successfully ran the action. A failing run still says
"successfully completed" — the truth is in the return code. Read the number,
not the adjective.

**2. The return code is HRESULT-wrapped, not the raw exit code.** A task
exiting 42 appears as:

```
201  ... with return code 2147942442
```

`2147942442` = `0x8007002A` = `0x80070000 + 42`. To recover the exit code:

```powershell
$rc = 2147942442
$rc - 2147942400            # -> 42
# or, equivalently:
'{0:X}' -f $rc              # -> 8007002A ; the low word is the exit code
```

A clean `return code 0` is the only value that appears unwrapped.

## What a VANISHED wrapper looks like here

This is the case the channel exists for. If the wrapper dies without writing
its `RESULT:` line, expect:

* `var\sync_logs\` — a log that stops mid-run with **no `RESULT:` line**
* this channel — `100`/`200` present (it started), and then either
  * `201` with a non-zero return code — it exited and reported, or
  * **no `201`/`102` at all** — the process was killed or the machine went
    down mid-run. That absence is the signal.
* `var\sync_health.json` — `last_attempt_utc` newer than every outcome,
  which the watchdog reports as **`ORPHANED`** after 2 hours and which
  raises `PANCAKEBOT_SYNC_RUN_VANISHED_*.txt` on the Desktop.

Those three agreeing is a confident diagnosis. The health file alone is
enough to *notice*; this channel is what tells you *where it stopped*.

## The queries

Last 20 events for our tasks:

```powershell
Get-WinEvent -LogName 'Microsoft-Windows-TaskScheduler/Operational' -MaxEvents 200 |
  Where-Object { $_.Message -match 'PancakeBot' } |
  Select-Object -First 20 TimeCreated, Id, Message | Format-List
```

Just the outcomes (return codes), newest first:

```powershell
Get-WinEvent -LogName 'Microsoft-Windows-TaskScheduler/Operational' -MaxEvents 500 |
  Where-Object { $_.Id -eq 201 -and $_.Message -match 'PancakeBot' } |
  ForEach-Object { '{0}  {1}' -f $_.TimeCreated, ($_.Message -replace '\s+',' ') }
```

Everything since a given time:

```powershell
Get-WinEvent -FilterHashtable @{
    LogName   = 'Microsoft-Windows-TaskScheduler/Operational'
    StartTime = (Get-Date).AddDays(-2)
} | Sort-Object TimeCreated |
  ForEach-Object { '{0} id={1} {2}' -f $_.TimeCreated, $_.Id, ($_.Message -split "`n")[0] }
```

Runs that STARTED but never finished (the vanished-wrapper hunt) — compare
the count of `100` against `102` over a window:

```powershell
$e = Get-WinEvent -FilterHashtable @{
    LogName='Microsoft-Windows-TaskScheduler/Operational'
    StartTime=(Get-Date).AddDays(-7)} |
    Where-Object { $_.Message -match 'PancakeBotDailySync' }
'started : ' + ($e | Where-Object Id -eq 100).Count
'finished: ' + ($e | Where-Object Id -eq 102).Count
```

A started-count exceeding the finished-count means a run disappeared.

## It changes nothing about how the tasks run

Verified rather than assumed. The channel is an ETW consumer: enabling it
subscribes a listener and does not touch task definitions, triggers,
principals or actions. After enabling, both tasks still read
`State=Ready`, `LastResult=0`, `User=SYSTEM`, `RunLevel=Highest`,
`StartWhenAvailable=True`, `MultipleInstances=IgnoreNew`, with the daily and
boot triggers intact — identical to before. The watchdog run immediately
after enabling returned 0, exactly as the runs before it did.
