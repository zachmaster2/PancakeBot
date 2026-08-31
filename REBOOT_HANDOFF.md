# REBOOT HANDOFF — written 2026-08-31, before a deliberate reboot

**You are probably a fresh Claude session with none of the prior context.
Read this first. It is the state of the world at the moment before the
machine was rebooted on purpose.**

Delete this file once the checks below are done and reported.

---

## What this project is now

PancakeBot is a PancakeSwap Prediction betting bot. **It is PAUSED — data
collection only.** The live unit was stopped and disabled 2026-08-30. The
Frankfurt VM that hosted it (`139.59.209.230`) was **destroyed 2026-08-31**
and is confirmed unreachable. There is no server anywhere. This Windows
machine is the only home for the project.

**The five store files in `var/` are the only surviving copy of data that
cannot be refetched.** OKX serves 1-second klines for roughly 171.6 days;
anything older exists nowhere else in the world. Protecting them is the
whole point of the current work. A backup archive exists at
`C:\Users\zking\Downloads\pancakebot_store_backup_20260829\` (617 MB zip).

## Why the machine was rebooted

To obtain an **elevated** Claude process. Registering the scheduled tasks
requires administrator rights, and the Claude session that wrote this file
was running at Medium integrity and could not do it.

The elevation mechanism already exists and works:

  * Scheduled task `ClaudeLaunchElevated` — AtLogon trigger, principal
    `zking`, **RunLevel=Highest**, runs
    `wscript.exe C:\Tools\launch_claude_admin_direct.vbs`.
  * That script does a direct `CreateProcess` on the packaged
    `Claude.exe`, which PRESERVES elevation. Shell/AUMID activation would
    launch at medium integrity and is deliberately not used.
  * It last fired 2026-08-26 05:33:20 (21s after boot) with result 0.

The session that wrote this file was Medium integrity because the app had
been restarted **by hand on 2026-08-29 21:04** via shell activation
(parent process `sihost.exe`), and the companion `ClaudeKeepalive` task is
launch-if-down only — it saw Claude running and no-op'd every 5 minutes.

**So: after this reboot, Claude should come up ELEVATED via the AtLogon
task. Verify that before doing anything else.**

---

## STATE AT THE MOMENT OF WRITING (compare against this)

### Git
    branch        : fix/sync-cannot-destroy-history
    local HEAD    : 0decdf636a2238f64bcb89e11b39b6af5669507e
    github/master : 0decdf636a2238f64bcb89e11b39b6af5669507e   (identical)
    unpushed      : none
    working tree  : clean

Remote is named `github`, NOT `origin`. Repo: `zachmaster2/PancakeBot`.

### Scheduled tasks — the thing being fixed
    PancakeBotDailySync      : NOT REGISTERED
    PancakeBotSyncWatchdog   : NOT REGISTERED
    ClaudeLaunchElevated     : Ready, RunLevel=Highest, AtLogon
    ClaudeKeepalive          : Ready, RunLevel=Highest, every 5 min

### Sync health (`var/sync_health.json`)
    last_success_utc : 2026-08-31T18:30:50Z
    last_attempt_utc : 2026-08-31T18:25:55Z
    last_exit_code   : 0
    consecutive_failures : 0
    attempts / successes : 1 / 1

That success came from running `scripts\daily_sync.ps1` by hand. The
scheduled tasks have never run, because they do not exist yet.

### Stores — line count | last epoch | CRLF count
    closed_rounds.jsonl    74252 | 511813 | 0
    bnb_spot_prices.jsonl  74244 | 511813 | 0
    btc_spot_prices.jsonl  74244 | 511813 | 0
    eth_spot_prices.jsonl  74244 | 511813 | 0
    sol_spot_prices.jsonl  74244 | 511813 | 0

**CRLF must be 0 on all five.** The stores were normalised to LF and the
writer was fixed to keep them that way. A non-zero CRLF count means
something wrote through a path that translates newlines, and that is a
regression worth stopping for.

Klines carry 8 permanently-absent epochs (445330-31, 447533-34, 449665-66,
452486-87) — past OKX's retention horizon, unfetchable by anyone. The
integrity report calls these `known_absent=8(expected)`. **That is correct
and not a fault.**

### Desktop
    No `PANCAKEBOT_SYNC_*.txt` marker present. Correct — the sync is healthy.

---

## WHAT TO CHECK AFTER THE REBOOT — in order

1. **Is this Claude session elevated?**

       ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

   Must be `True`. Also confirm integrity level is High:
   `whoami /groups | Select-String 'Mandatory Label'` should show
   `High Mandatory Level` (S-1-16-12288), not Medium (S-1-16-8192).

   If it is still Medium, the AtLogon task did not produce this instance.
   Check `C:\Tools\claude_launch.log` and the parent process chain. **Do
   not attempt to acquire elevation, prompt for credentials, or work
   around the boundary.** Report it and stop.

2. **Confirm nothing was lost across the reboot.** Working tree clean,
   HEAD still `0decdf6`, and the five stores still at the line counts and
   last epoch above with CRLF 0. The reboot happened with the tree clean
   and everything pushed, so any difference is a real problem.

3. **Register the tasks** (only if step 1 says elevated):

       cd C:\Users\zking\Documents\GitHub\PancakeBot
       .\scripts\register_sync_tasks.ps1

4. **Verify the ACTUAL registered definitions**, not the intended ones —
   they can differ. For both `PancakeBotDailySync` and
   `PancakeBotSyncWatchdog` confirm: State=Ready; Principal UserId=SYSTEM,
   LogonType=ServiceAccount, RunLevel=Highest; BOTH triggers present
   (daily + AtStartup); `StartWhenAvailable=True`;
   `MultipleInstances=IgnoreNew`; and for the sync task
   `RestartCount=3`, `RestartInterval=PT20M`.

5. **Trigger the sync task manually and confirm it runs AS SYSTEM.**

       Start-ScheduledTask -TaskName PancakeBotDailySync

   This is the check that actually matters and it has never been done.
   SYSTEM has a different environment block and working directory than a
   user shell. `THE_GRAPH_API_KEY` must resolve there — it lives in the
   repo's gitignored `.env` and is read at runtime by `load_dotenv()`, NOT
   from a user environment variable. **This is exactly the class of thing
   that works in testing and fails at 06:30 with nobody watching.**

   Confirm: the task's LastTaskResult is 0, a new log appears in
   `var\sync_logs\`, and `var\sync_health.json` shows a NEWER
   `last_success_utc` than 2026-08-31T18:30:50Z.

6. **Run the watchdog and confirm it reads the real success and raises no
   marker.**

       .venv\Scripts\python.exe scripts\sync_watchdog.py --no-discord

   Expect `SYNC OK`, exit 0, and no `PANCAKEBOT_SYNC_*.txt` on the
   Desktop. Do NOT send Discord messages — that is the operator's channel
   and not ours to trigger.

7. **Then the real test, which needs a SECOND reboot:** confirm both tasks
   fire at startup with nobody logged in. After that reboot, check each
   task's `LastRunTime` is after the boot time, `LastTaskResult` is 0, and
   `var\sync_health.json` advanced without anyone logging in.

---

## Standing rules that outlive this session

* **Never run the real sync against the canonical stores to test a
  mechanism.** Use a sandbox copy. This rule exists because it was broken
  once, and 2,243 rounds were appended to canonical data by accident.
* **Authorisation is per step, not per sequence.** Do not chain onward
  from one approval.
* **Do not send Discord messages.** Use `--no-discord` when testing.
* **Do not delete `var/` data to make an analysis tidy.** The 1,256
  duplicate rows in `cycle_audit.csv` are a TRUE record of a real fault;
  the consumers dedupe (`scripts/audit_reader.py`), the file stays.
* Git via Windows-side git only. Remote is `github`, not `origin`.
