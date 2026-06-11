#!/usr/bin/env bash
# PancakeBot Linux installer — fresh-clone to ready, idempotent + verbose.
#
#   git clone <repo> && cd PancakeBot
#   sudo bash bootstrap/install.sh
#
# Steps (each is idempotent and logged):
#   1. Python 3.13 (pyenv, additive)          bootstrap/linux/install_python313.sh
#   2. venv + dependencies                    bootstrap/common/python_setup.py
#   3. config + secrets present?              bootstrap/common/config_check.py
#   4. EnvironmentFiles (secrets + alerts)    /etc/pancakebot/{pancakebot,alerts}.env
#   5. systemd units (tracked; DISABLED)      bootstrap/linux/systemd/*.service
#   6. chrony drop-in (clock-step detection)  /etc/chrony.d/pancakebot.conf
#
# Does NOT enable/start anything (no auto-start of a live bot on a fresh box).
# Run the dry soak with:  systemctl enable --now pancakebot-dry
# Reverse everything with: sudo bash bootstrap/uninstall.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ENV_DIR="/etc/pancakebot"
ENV_FILE="$ENV_DIR/pancakebot.env"

log() { echo "[install] $*"; }
require_root() { [ "$(id -u)" = "0" ] || { echo "must run as root (dnf + systemd + /etc writes)"; exit 1; }; }

require_root
log "repo root: $REPO_ROOT"

log "STEP 1/6: Python 3.13 (pyenv)"
PY313="$(bash "$HERE/linux/install_python313.sh" | tail -1)"
log "python3.13 -> $PY313"

log "STEP 2/6: venv + dependencies"
"$PY313" "$HERE/common/python_setup.py" --python "$PY313" --venv "$REPO_ROOT/.venv" \
    --requirements "$REPO_ROOT/requirements.txt"
VENV_PY="$REPO_ROOT/.venv/bin/python"

log "STEP 3/6: config + secrets check"
"$VENV_PY" "$HERE/common/config_check.py"

log "STEP 4/6: EnvironmentFiles (secrets + alerts split)"
ALERTS_FILE="$ENV_DIR/alerts.env"
mkdir -p "$ENV_DIR"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<'SECRETS'
# PancakeBot secrets (bot units ONLY — never loaded by the notify unit).
BSC_WALLET_PRIVATE_KEY=
THE_GRAPH_API_KEY=
SECRETS
    chmod 600 "$ENV_FILE"
    log "wrote $ENV_FILE (chmod 600) — FILL IN the secrets"
else
    log "$ENV_FILE already present; leaving as-is"
fi
if [ ! -f "$ALERTS_FILE" ]; then
    # Least-privilege split (Phase 3c-2): webhooks live separately so the
    # pancakebot-notify@ unit can load them WITHOUT the wallet key.
    # Migrate any webhook lines already present in pancakebot.env.
    {
        echo "# PancakeBot Discord webhooks (loaded by bot + notify units)."
        grep -E "^PANCAKEBOT_(LIVE_ALERTS|DRY_ALERTS|GENERAL)_DISCORD_WEBHOOK_URL=" "$ENV_FILE" 2>/dev/null \
            || printf 'PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL=\nPANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL=\nPANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL=\n'
    } > "$ALERTS_FILE"
    chmod 600 "$ALERTS_FILE"
    log "wrote $ALERTS_FILE (chmod 600; webhooks migrated from pancakebot.env where present)"
    log "NOTE: webhook lines may be removed from $ENV_FILE manually (alerts.env is authoritative for the notify unit)"
else
    log "$ALERTS_FILE already present; leaving as-is"
fi

log "STEP 5/6: systemd units (tracked files; left DISABLED)"
# The tracked units hardcode /root/pancakebot (WorkingDirectory + ExecStart
# venv path). Installing them from a clone elsewhere would produce units
# that fail at first start — refuse loudly instead.
if [ "$REPO_ROOT" != "/root/pancakebot" ]; then
    echo "[install] FATAL: units hardcode /root/pancakebot but repo is at $REPO_ROOT — clone there (or edit the unit files first)."
    exit 1
fi
# Phase 3c-2 (systemd-direct): the units are TRACKED at
# bootstrap/linux/systemd/ and installed verbatim — systemd itself is the
# supervisor (no Python supervisor layer). Re-copying on every run keeps
# /etc in sync with the repo (push-to-deploy updates the repo copies; rerun
# this script — or cp + daemon-reload manually — to roll units forward).
for unit in pancakebot-live.service pancakebot-dry.service pancakebot-notify@.service; do
    cp "$HERE/linux/systemd/$unit" "/etc/systemd/system/$unit"
done
systemctl daemon-reload
log "installed pancakebot-{live,dry}.service + pancakebot-notify@.service (disabled)"

log "STEP 6/6: chrony drop-in (clock-step detection bound)"
# Steady-state drift is a non-issue under chronyd (~30-60us RMS measured
# 2026-06-10); the one VM risk is a clock STEP (host pause/live-migration
# jumps the guest clock), which the distro pool's backed-off 1024s poll
# can leave undetected for ~17 min. The drop-in adds sources polled every
# 16-64s (maxpoll 6) so an offset is DETECTED within ~64s; makestep 0.1 3
# steps the clock at daemon start (first 3 updates), after which large
# offsets are slewed. Bot tolerance is +-250ms (engine.py clock-sync note).
if [ ! -f /etc/chrony.d/pancakebot.conf ] || ! grep -q "^confdir /etc/chrony.d$" /etc/chrony.conf; then
    mkdir -p /etc/chrony.d
    cat > /etc/chrony.d/pancakebot.conf <<'CHRONY'
# PancakeBot (bootstrap/install.sh): bound clock-step DETECTION to ~64s.
# The distro pool backs off to 1024s polls; these sources stay at 16-64s.
pool pool.ntp.org iburst minpoll 4 maxpoll 6
makestep 0.1 3
CHRONY
    # AlmaLinux 9 chrony.conf has no conf-dir include; add one (idempotent).
    grep -q "^confdir /etc/chrony.d$" /etc/chrony.conf || \
        echo "confdir /etc/chrony.d" >> /etc/chrony.conf
    systemctl restart chronyd
    log "wrote /etc/chrony.d/pancakebot.conf + confdir include; chronyd restarted"
else
    log "chrony drop-in already present; leaving as-is"
fi

log "DONE. Next: fill $ENV_FILE, then  systemctl enable --now pancakebot-dry  for the dry soak."
log "Validate with:  $VENV_PY $HERE/common/health_check.py --mode dry --service-name pancakebot-dry"
