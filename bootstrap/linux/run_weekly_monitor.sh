#!/usr/bin/env bash
# Weekly monitor cron wrapper — the only thing the VM's crontab calls:
#
#   0 6 * * *  /root/pancakebot/bootstrap/linux/run_weekly_monitor.sh >/dev/null 2>&1
#
# DAILY cron, weekly work: Sundays run in full; Mon-Sat are no-ops UNLESS
# a retry_pending marker exists (written by a blind applied run — sync
# failure or stale data), in which case the day is a full makeup attempt
# for the blind Sunday. Recovery clears the marker; the next Sunday
# supersedes it. A failed Sunday thus costs at most one day of blindness
# per recovery opportunity, not a full week.
#
# The crontab line carries NO logfile redirect on purpose: cron's shell
# opens redirects BEFORE the command runs, so a redirect into the
# (gitignored, wipeable) var/ tree would silently kill every run the
# moment the tree is missing — the wrapper's own mkdir could never fire.
# THIS script owns its logging instead: it self-heals the log dir, then
# `exec`-appends to cron.log; if the dir is unwritable it alerts and runs
# without a logfile. Logging must never be able to prevent alerting.
#
# --apply = the monitor may act on systemd BOTH ways (2026-07-09 user
# decision, re-affirmed 2026-07-17): the negative trigger auto-DISABLES,
# the positive trigger auto-ENABLES (writing the cooldown-override flag
# first when the bot went down suspended, so it releases on its first
# paused round). There is no manual arming step; the weekly triggers are
# the sole authority over the live unit. See docs/monitoring.md.
#
# `--dry` (manual smoke test): runs the monitor WITHOUT --apply and with
# --no-sync — full compute + artifact + Discord message, zero mutation
# of weekly state, data stores, or systemd. Safe any day of the week.
#
# Alert layering (dead-man's switch): the monitor Discords every outcome
# (delivery-verified; rc=3 = evaluation fine but Discord post failed) and
# its own crashes; this wrapper curls a fallback on any nonzero rc. A
# Sunday with no Discord message therefore means the box, cron, or
# webhook itself is dead.
#
# Env: alerts.env supplies the Discord webhooks (only — the wallet key
# never enters this process); run.py --sync reads THE_GRAPH_API_KEY from
# the repo-root .env via load_dotenv.
set -u

REPO=/root/pancakebot
LOGDIR=$REPO/var/strategy_review/weekly_monitors

# Webhooks FIRST — alerting must never depend on the filesystem below.
set -a; . /etc/pancakebot/alerts.env 2>/dev/null; set +a

notify() {  # best-effort Discord post; venv-independent, JSON-safe
    local payload
    payload=$(/usr/bin/python3 -c \
        'import json,sys; print(json.dumps({"content": sys.argv[1]}))' \
        "$1" 2>/dev/null) || payload='{"content":"[weekly-monitor] wrapper alert (encoding fallback)"}'
    curl -sS -m 10 -H 'Content-Type: application/json' \
        -d "$payload" \
        "${PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL:-}" >/dev/null 2>&1 || true
}

if [ -z "${PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL:-}" ]; then
    echo "WARNING: general Discord webhook missing (alerts.env absent/empty) — running mute" >&2
fi

# Self-heal the log dir and take over logging; if unwritable, alert and
# run without a logfile (output pre-exec goes to cron -> /dev/null).
if mkdir -p "$LOGDIR" 2>/dev/null && touch "$LOGDIR/cron.log" 2>/dev/null; then
    # Cap the append-forever log (~15 KB/run) at ~2 MB, keep the tail.
    # Runs BEFORE the fd is opened, so no output lands on an orphaned inode.
    if [ "$(stat -c%s "$LOGDIR/cron.log" 2>/dev/null || echo 0)" -gt 2000000 ]; then
        tail -c 500000 "$LOGDIR/cron.log" > "$LOGDIR/cron.log.tmp" \
            && mv "$LOGDIR/cron.log.tmp" "$LOGDIR/cron.log"
    fi
    exec >>"$LOGDIR/cron.log" 2>&1
    LOCKFILE=$LOGDIR/.cron.lock
else
    notify "⚠️ [weekly-monitor] log dir unwritable ($LOGDIR) — running WITHOUT a logfile; check disk/permissions"
    LOCKFILE=/tmp/pancakebot_weekly_monitor.lock
fi

echo "=== $(date -u +%FT%TZ) weekly monitor run ($*) ==="

# Daily-cron gate: Mon-Sat only run while a retry is pending. `--dry`
# (manual smoke) always runs. Sunday (dow 7) always runs.
if [ "${1:-}" != "--dry" ]; then
    dow=$(date -u +%u)
    if [ "$dow" != "7" ] && [ ! -f "$LOGDIR/retry_pending.json" ]; then
        echo "no-op (dow=$dow, no retry pending)"
        exit 0
    fi
fi

# Overlap guard: a hung previous run must not interleave with this one
# (the monitor has its own sync/backtest timeouts, so a held lock means
# something is badly wrong — say so rather than pile up).
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "previous run still holds the lock — skipping"
    notify "❌ [weekly-monitor] SKIPPED: previous week's run is still running (lock held). Investigate the VM."
    exit 1
fi

cd "$REPO" || { notify "❌ [weekly-monitor] wrapper: $REPO missing — VM is broken."; exit 1; }

if [ "${1:-}" = "--dry" ]; then PYFLAGS="--no-sync"; else PYFLAGS="--apply"; fi
.venv/bin/python research/weekly_monitor_state_machine.py $PYFLAGS
rc=$?
if [ "$rc" -ne 0 ]; then
    notify "❌ [weekly-monitor] exited rc=$rc — check cron.log on the VM; will retry next Sunday. (rc=3 = evaluation completed but its Discord post failed; rc=1 = crash, alert attempted; other = python/venv failure)"
fi
exit "$rc"
