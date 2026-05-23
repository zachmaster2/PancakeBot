# Legacy-supervisor deletion checklist

This file is intentionally placed inside `pancakebot/service/` so the
deletion sequence is co-located with the code that replaces it. Run when
the new Windows Service architecture has been validated (the design doc's
default soak is ≥1 week, but the operator may decide to pull the trigger
earlier).

## Order of operations

1. **Confirm new services are healthy.** `Get-Service PancakeBotLive
   PancakeBotDry` shows the operator-intended state, heartbeat files are
   fresh, no SPAWN_FAILED / SLOW_CRASHLOOP_WARNING alerts in the last 24h.
2. **Run `scripts\uninstall_old_supervisor.ps1`.** This disables and deletes
   the two legacy schtasks (`PancakeBotSupervisorLive`,
   `PancakeBotSupervisorDry`). Idempotent.
3. **Git rm the orphaned Python + PowerShell sources** (one commit):

   ```bash
   git rm scripts/supervisor.py
   git rm scripts/install_supervisor_schtasks.ps1
   ```

   The following test files reference supervisor.py *only* — confirm with
   `grep -l 'from scripts.supervisor\|import supervisor'
   tests/test_*.py` before removing:

   ```bash
   git rm tests/test_archive_stale_crash.py    # IF its only consumer is supervisor.py — verify
   ```

   (Leave tests that target `pancakebot.runtime.process_health` /
   `pancakebot.service.*` alone; only delete tests whose subject is the
   legacy supervisor module itself.)

4. **Update docs** (separate commit if desired):
   - Mark `docs/SUPERVISOR.md` as legacy or replace its contents with
     a redirect note pointing to the new service architecture.
   - Update `README.md` if it references `scripts/supervisor.py` or the
     `PancakeBotSupervisor*` schtasks.

## Drafted commit message for the deletion commit

```
cleanup: remove legacy schtask supervisor (replaced by PancakeBot{Live,Dry} services)

- delete scripts/supervisor.py (replaced by pancakebot/service/{common,supervision,notifications}.py)
- delete scripts/install_supervisor_schtasks.ps1 (replaced by scripts/install_services.ps1)
- delete tests/test_archive_stale_crash.py if its sole consumer was supervisor.py — VERIFY

The new architecture uses two Windows Services (PancakeBotLive,
PancakeBotDry) registered via pywin32, with 1-second supervision polling
vs. the legacy 3-minute schtask cadence. Recovery handled at two tiers:
inner loop respawns the bot child on STALE/CRASHED/DOWN, outer SCM
recovery actions respawn the service itself on service-level crashes.
Live > Dry priority enforced in-process via SCM service-status query.

Migration ran 2026-MM-DD; new services have been stable for N days.

scripts/uninstall_old_supervisor.ps1 was used to delete the legacy
schtasks. The PS1 itself stays in the repo for one release as a record;
remove in a follow-up cleanup if desired.
```

## Files that may be deleted

### Definitely (no remaining consumers after schtask removal)
- `scripts/supervisor.py` — entire module replaced
- `scripts/install_supervisor_schtasks.ps1` — replaced by `install_services.ps1`

### Probably (verify with grep first)
- `tests/test_archive_stale_crash.py` — if it tests `scripts/supervisor.py` only
- Any `tests/test_supervisor*.py` files

### NOT to be deleted
- `pancakebot/runtime/process_health.py` — heartbeat / PID / crash artifacts
  are still written by the bot itself and consumed by the new service code.
  This file is shared, not legacy.
- `var/{live,dry}/supervisor.log` — historical record; new service writes
  to Event Viewer + the same Discord channels but doesn't write a
  supervisor.log file. Past logs remain as archives.
- `var/{live,dry}/last_alert.json` — rate-limit state is shared between
  legacy supervisor and new notifications module.
- `var/{live,dry}/restart_history.jsonl` — crashloop limiter state is
  shared.

## How to verify before deletion

```bash
# Confirm no other module imports scripts/supervisor.py
grep -rl 'from scripts.supervisor\|import supervisor' --include='*.py' .

# Confirm new services are running
Get-Service PancakeBotLive, PancakeBotDry

# Confirm legacy schtasks are gone
Get-ScheduledTask | Where-Object { $_.TaskName -like '*Supervisor*' }

# Tests still green
.venv\Scripts\python.exe -m pytest -q
```

## Rollback (if the new services prove problematic)

```powershell
# 1. Stop + uninstall new services
scripts\uninstall_services.ps1

# 2. Re-install legacy schtasks
scripts\install_supervisor_schtasks.ps1
```

The bot itself (`run.py`, `pancakebot/runtime/*`) is untouched by either
architecture, so rollback is genuinely no-op for the trading logic.
