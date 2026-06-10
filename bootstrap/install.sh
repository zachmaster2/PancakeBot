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
#   4. EnvironmentFile for Discord webhooks   /etc/pancakebot/pancakebot.env (if absent)
#   5. systemd units (live + dry, DISABLED)   bootstrap/linux/setup_service.py
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

log "STEP 4/6: Discord webhook EnvironmentFile"
if [ -f "$ENV_FILE" ]; then
    log "$ENV_FILE already present; leaving as-is"
else
    mkdir -p "$ENV_DIR"
    cat > "$ENV_FILE" <<'WEBHOOKS'
# PancakeBot Discord webhooks — fill in, then `systemctl daemon-reload`.
PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL=
PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL=
PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL=
WEBHOOKS
    chmod 600 "$ENV_FILE"
    log "wrote $ENV_FILE (chmod 600) — FILL IN the webhook URLs"
fi

log "STEP 5/6: systemd units (live + dry, left DISABLED)"
"$VENV_PY" "$HERE/linux/setup_service.py" --venv-python "$VENV_PY"

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
