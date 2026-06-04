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

log "STEP 1/5: Python 3.13 (pyenv)"
PY313="$(bash "$HERE/linux/install_python313.sh" | tail -1)"
log "python3.13 -> $PY313"

log "STEP 2/5: venv + dependencies"
"$PY313" "$HERE/common/python_setup.py" --python "$PY313" --venv "$REPO_ROOT/.venv" \
    --requirements "$REPO_ROOT/requirements.txt"
VENV_PY="$REPO_ROOT/.venv/bin/python"

log "STEP 3/5: config + secrets check"
"$VENV_PY" "$HERE/common/config_check.py"

log "STEP 4/5: Discord webhook EnvironmentFile"
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

log "STEP 5/5: systemd units (live + dry, left DISABLED)"
"$VENV_PY" "$HERE/linux/setup_service.py" --venv-python "$VENV_PY"

log "DONE. Next: fill $ENV_FILE, then  systemctl enable --now pancakebot-dry  for the dry soak."
log "Validate with:  $VENV_PY $HERE/common/health_check.py --mode dry --service-name pancakebot-dry"
