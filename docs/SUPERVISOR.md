# PancakeBot supervision (systemd-direct)

Out-of-process health monitoring for the dry and live bots: crash restart,
crashloop braking, and Discord lifecycle alerts. Catches process deaths
that log-tailing alone would miss.

## Architecture

**systemd IS the supervisor** (Phase 3c-2, cut over 2026-06-10). Each bot
unit runs `run.py` directly — there is no Python supervisor layer:

```
systemd
  ├─ pancakebot-live.service   ExecStart=.venv/bin/python -u run.py --live
  ├─ pancakebot-dry.service    ExecStart=.venv/bin/python -u run.py --dry
  └─ pancakebot-notify@.service (oneshot template, fired on lifecycle edges)
        └─ python -m pancakebot.ops.notify_lifecycle <unit>-<event>
              └─ pancakebot.service.notifications  (Discord alert executor)
```

The unit files are TRACKED at `bootstrap/linux/systemd/` and installed
verbatim by `bootstrap/install.sh` STEP 5 (`cp` + `daemon-reload`; rerun
after a unit-file change lands via push-to-deploy). Key choices, all
visible in the unit files:

| Mechanism            | Setting                                            |
|----------------------|----------------------------------------------------|
| Crash restart        | `Restart=on-failure`, `RestartSec=60`              |
| Crashloop brake      | `StartLimitBurst=5` / `StartLimitIntervalSec=900` — after 5 failed starts in 15 min systemd stops retrying (`Result=start-limit-hit`) |
| One bot at a time    | mutual `Conflicts=` (starting one STOPS the other — see Pitfall below) |
| Tree kill            | `KillMode=control-group`, `TimeoutStopSec=25`      |
| Lifecycle alerts     | `ExecStartPost`/`ExecStopPost` → `systemctl start --no-block pancakebot-notify@%p-{started,stopped}` |
| Least privilege      | bot units load `pancakebot.env` + `alerts.env`; the notify template loads ONLY `alerts.env` (webhooks) — the wallet key never enters the notify process |

`--no-block` + the `-` prefix on the Exec hooks mean a slow or failing
Discord POST can never delay or fail the bot's own lifecycle.

## The notify flow

`notify_lifecycle` (in `pancakebot/ops/`) reads unit state from
`systemctl show <unit> -p Result,ExecMainStatus,NRestarts` — systemd
retains these after the service exits, which is what makes the
detached-oneshot design work (`$SERVICE_RESULT`/`$EXIT_STATUS` exist only
inside the main unit's own ExecStopPost environment and do NOT propagate
through `systemctl start`; they are still preferred when present).

Decision table (thresholds carried over from the prior supervisor):

```
started, NRestarts==0 (fresh/manual start):
    system uptime < 10 min  -> REBOOTED
    crash evidence          -> RECOVERY_AFTER_CRASH
    else                    -> STARTED
started, NRestarts>0 (systemd auto-restart after a failure):
    append var/<mode>/restart_history.jsonl, then:
    >=3 restarts / 15 min   -> SUPPRESSED_FAST_CRASHLOOP
    >=8 restarts / 24 h     -> SLOW_CRASHLOOP_WARNING
    else                    -> silent (CRASHED already fired per failure)
stopped:
    Result=success          -> STOPPED (intentional, INFO)
    Result=start-limit-hit  -> SUPPRESSED_FAST_CRASHLOOP (terminal: systemd
                               gave up; manual intervention required)
    anything else           -> CRASHED (Result + exit status + last journal
                               line; crash.json traceback rendered when present)
```

"Crash evidence" is `crash.json` present OR a `crash_archive_*.json`
renamed within the last 2 minutes — `run.py` archives a lingering
crash.json within milliseconds of starting, racing the detached notify
unit, so a fresh rename counts as evidence for the current start.

Known edge: a `start-limit-hit` transition fires no ExecStopPost of its
own. Coverage comes from the started-side counter (the fast-crashloop
alert fires on the restart BEFORE the limit trips) plus the stopped-side
branch as defense-in-depth.

All six failure scenarios were validated live on the VM (2026-06-10,
parallel `pancakebot-test` unit, real Discord sends): exit-code crash,
SIGKILL, cgroup OOM-kill, clean stop, fast crashloop (suppression + halt
at NRestarts=5), seeded slow-loop warning — 6/6 alert kinds correct.

## Discord alerting

Three webhooks, set in `/etc/pancakebot/alerts.env` (0600):

| Env var                                     | Channel                  |
|---------------------------------------------|--------------------------|
| `PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL` | `pancakebot-live-alerts` |
| `PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL`  | `pancakebot-dry-alerts`  |
| `PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL`     | `pancakebot-general`     |

Unset env var → `DISABLED` (soft fallback, no HTTP). HTTP failure →
`SEND_FAILED` logged to stderr (journald), never raises. Rate limit: one
alert per `(mode, kind)` per 5 minutes, tracked in
`var/<mode>/last_alert.json`. The executor (`notifications.py`) prefixes
every alert `[LIVE]`/`[DRY]` with an INFO/WARN/CRIT severity tag.

## Operations

```bash
systemctl status pancakebot-live          # state, PID, NRestarts, recent journal
journalctl -u pancakebot-live -n 100      # bot stdout/stderr
journalctl -u 'pancakebot-notify@*' -n 20 # notify outcomes (SENT/RATE_LIMITED/...)
systemctl stop pancakebot-live            # clean stop -> STOPPED (intentional)
```

**Pitfall — `Conflicts=` is an eviction, not a refusal**: `systemctl start
pancakebot-dry` on a box where live is running SILENTLY STOPS live (this
took the live bot down for 77s on 2026-06-10). The health check
(`bootstrap/common/health_check.py`) refuses to start a unit whose partner
is active; prefer it over raw `systemctl start` when unsure.

**After a crashloop halt** (`start-limit-hit`): fix the root cause (see
`var/<mode>/crash.json` + journal), then

```bash
systemctl reset-failed pancakebot-live    # clear the start-limit state
rm -f var/live/restart_history.jsonl      # optional: reset loop counters
systemctl start pancakebot-live
```

## Files produced

| Path                               | Writer            | Purpose                                  |
|------------------------------------|-------------------|------------------------------------------|
| `var/<mode>/bot.pid`               | bot (`run.py`)    | PID on startup, cleared on clean exit    |
| `var/<mode>/crash.json`            | bot (`run.py`)    | Uncaught-exception dump (atomic write); archived to `crash_archive_*.json` on next start |
| `var/<mode>/last_alert.json`       | notify oneshot    | Per-(mode,kind) alert cooldown timestamps|
| `var/<mode>/restart_history.jsonl` | notify oneshot    | Auto-restart events, pruned at 24h age   |
| `var/<mode>/runtime.log`           | bot               | Per-cycle structured runtime log         |
| `var/<mode>/cycle_audit.csv`       | bot               | Per-cycle audit row for backtest replay  |

None of these are committed to git (`var/` is in `.gitignore`). The bot's
stdout/stderr go to journald (no separate log-capture files).
