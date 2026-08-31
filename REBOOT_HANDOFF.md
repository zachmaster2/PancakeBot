# REBOOT HANDOFF — updated 2026-08-31 before REBOOT 2

**You are probably a fresh Claude session with none of the prior context.
Read this first.** It is the state of the world immediately before a second
deliberate reboot.

Delete this file once the checks below are done and reported.

---

## What this project is now

PancakeBot is a PancakeSwap Prediction betting bot. **It is PAUSED — data
collection only.** The live unit was stopped and disabled 2026-08-30. The
Frankfurt VM that hosted it (`139.59.209.230`) was **destroyed 2026-08-31**
and is confirmed unreachable. There is no server anywhere.

**The five store files in `var/` are the only surviving copy of data that
cannot be refetched.** OKX serves 1-second klines for ~171.6 days; anything
older exists nowhere else. Protecting them is the point of all current
work. Backup: `C:\Users\zking\Downloads\pancakebot_store_backup_20260829\`
(617 MB zip).

---

## REBOOT 1 (14:44:43) — DONE, and what it established

It produced an **elevated** Claude, which allowed the two scheduled tasks to
be registered. Both are now registered and one has been run successfully.

It also settled a question about how unattended boots work. **Auto-logon is
enabled** (`AutoAdminLogon=1`, `DefaultUserName=zking`; the password is an
LSA secret via Sysinternals Autologon in `C:\Tools\Autologon`, NOT plaintext
in the registry — do not read it, it is not needed). Observed timeline:

    14:44:43  boot
    14:45:01  interactive logon   <- 18s after boot, NOBODY TYPED ANYTHING
    14:45:02  ClaudeLaunchElevated fires (AtLogon, RunLevel=Highest), result 0
    14:45:13  elevated claude.exe starts

So the AtLogon trigger *does* cover the unattended case, because auto-logon
makes a logon happen with no human present. The operator's memory of
supporting spontaneous unattended reboots was correct.

---

## WHAT REBOOT 2 IS TESTING — read this carefully, the framing is subtle

We **cannot** test "nobody logged in" literally: the machine logs itself in
within ~18 seconds, and auto-logon is deliberately NOT being disabled to
manufacture a cleaner experiment.

**The sharp question is a TIMING one:**

> Does `PancakeBotDailySync` fire at boot **independently of, and ideally
> before, the logon completing**?

* If the boot trigger fires as SYSTEM at roughly boot+5s while auto-logon
  lands at ~boot+18s, that is **direct evidence the sync needs no session**
  — the strongest proof available short of disabling auto-logon.
* If it only runs **after** the logon, that is a **weaker** result and must
  be reported as such: it would mean an unattended boot where auto-logon
  somehow failed would also lose the sync.

**The ordering is the finding. Capture it precisely.** Do not report a
generic "it worked".

Collect, to the second:

    boot time                     Get-CimInstance Win32_OperatingSystem .LastBootUpTime
    PancakeBotDailySync start     (Get-ScheduledTaskInfo ...).LastRunTime
    PancakeBotSyncWatchdog start  (Get-ScheduledTaskInfo ...).LastRunTime
    interactive logon             Security log event 4624, logon type 2 or 11, user zking
    elevated claude.exe start     Get-Process Claude | Select StartTime
    ClaudeLaunchElevated run      (Get-ScheduledTaskInfo -TaskName ClaudeLaunchElevated).LastRunTime

Note the Security-log query may need elevation; if it is unavailable, say so
rather than guessing the logon time.

---

## STATE AT THE MOMENT OF WRITING (compare against this)

### Git
    branch        : fix/sync-cannot-destroy-history
    HEAD          : (this commit — see `git log -1`)
    github/master : same; nothing unpushed
    working tree  : clean

Remote is named `github`, NOT `origin`. Repo `zachmaster2/PancakeBot`.

### Scheduled tasks — REGISTERED as of 14:52
    PancakeBotDailySync     Ready  SYSTEM/ServiceAccount/Highest
                            triggers: Daily 06:30 + AtStartup (MSFT_TaskBootTrigger)
                            StartWhenAvailable=True, RestartCount=3/PT20M,
                            MultipleInstances=IgnoreNew, batteries allowed
                            LastTaskResult = 0   (ran manually as SYSTEM, succeeded)
    PancakeBotSyncWatchdog  Ready  SYSTEM/ServiceAccount/Highest
                            triggers: Daily 09:00 + AtStartup
                            LastTaskResult = 267011  <-- NEVER RUN sentinel (0x41303)

    ClaudeLaunchElevated    Ready  AtLogon, RunLevel=Highest
    ClaudeKeepalive         Ready  every 5 min, launch-if-down only

**The boot triggers have NEVER actually fired**, because registration
(14:52) came after boot 1 (14:44). That is exactly what reboot 2 tests.

### Sync health (`var/sync_health.json`)
    last_success_utc : 2026-08-31T19:00:12Z
    last_attempt_utc : 2026-08-31T18:53:05Z
    last_exit_code   : 0
    consecutive_failures : 0
    attempts / successes : 2 / 2

### Stores — line count | last epoch | CRLF count
    closed_rounds.jsonl     74257 | 511818 | 0
    bnb_spot_prices.jsonl   74249 | 511818 | 0
    btc_spot_prices.jsonl   74249 | 511818 | 0
    eth_spot_prices.jsonl   74249 | 511818 | 0
    sol_spot_prices.jsonl   74249 | 511818 | 0

**CRLF must stay 0.** Klines carry 8 permanently-absent epochs (445330-31,
447533-34, 449665-66, 452486-87) past OKX's retention horizon; the report
calls these `known_absent=8(expected)` and that is CORRECT, not a fault.

### Desktop
    No `PANCAKEBOT_SYNC_*.txt` marker. Correct — sync is healthy.

---

## WHAT TO CHECK AFTER REBOOT 2 — in order

1. **Build the timeline above and report the ORDERING.** This is the
   headline. Sync-start before logon = strong result. After = weaker,
   report honestly.

2. **`LastTaskResult` must move off 267011 to 0 for BOTH tasks.** The
   watchdog has its own independent boot trigger — **prove it fired; do not
   infer it from the sync succeeding.** They are separate triggers and the
   whole reason there are two tasks is that one cannot vouch for the other.

3. **Health file shows a NEW success with a timestamp after the reboot** —
   later than `2026-08-31T19:00:12Z`, and `successes` > 2.

4. **Desktop marker: distinguish "appeared and cleared" from "appeared and
   stayed."** A marker appearing briefly before the first post-boot sync
   finishes is CORRECT behaviour (the watchdog may run before the sync
   completes). One still present after a successful sync is a real bug.
   Check `var\sync_logs\` timestamps to tell which happened.

5. **Stores intact**: line counts >= the numbers above, last epoch >=
   511818, contiguous, **CRLF 0**, and no torn tail. This is the first
   reboot with real scheduled writes in play, so it is also an unplanned
   production test of the torn-tail repair. If a repair fired, the sync log
   will contain a `REPAIR`/`torn tail` line — **that is the safety net
   working, not a failure**, but report it prominently.

6. Working tree clean, HEAD unchanged, nothing unpushed.

7. Run the watchdog by hand to confirm it reads the real success:

       .venv\Scripts\python.exe scripts\sync_watchdog.py --no-discord

   Expect `SYNC OK`, exit 0, no marker. **Never omit `--no-discord` when
   testing** — that is the operator's channel, not ours to trigger.

---

## Standing rules that outlive this session

* **Never run the real sync against the canonical stores to test a
  mechanism.** Use a sandbox copy. This rule exists because it was broken
  once and 2,243 rounds were appended to canonical data by accident.
* **Authorisation is per step, not per sequence.**
* **Do not send Discord messages.** Use `--no-discord`.
* **Do not delete `var/` data to tidy an analysis.** The 1,256 duplicate
  rows in `cycle_audit.csv` are a TRUE record of a real fault; consumers
  dedupe via `scripts/audit_reader.py`, the file stays.
* **Do not read `DefaultPassword` or any credential.** Auto-logon status is
  all that was ever needed.
* Git via Windows-side git only. Remote is `github`, not `origin`.
* Distinguish **verified by definition** from **demonstrated by behaviour**.
  Most of this week's real findings came from that distinction.
