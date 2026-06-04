#!/usr/bin/env bash
# Reverse bootstrap/install.sh — idempotent. Stops + disables + removes the
# systemd units. Does NOT delete var/ state, .env, config.toml, the venv, or
# the pyenv Python (those are data/runtime, not install artifacts) unless
# --purge is given.
#
#   sudo bash bootstrap/uninstall.sh [--purge]
set -euo pipefail

PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
log() { echo "[uninstall] $*"; }
[ "$(id -u)" = "0" ] || { echo "must run as root"; exit 1; }

for svc in pancakebot-live pancakebot-dry; do
    if systemctl list-unit-files "${svc}.service" --no-legend 2>/dev/null | grep -q "$svc"; then
        log "stopping + disabling + removing $svc"
        systemctl disable --now "${svc}.service" 2>/dev/null || true
        rm -f "/etc/systemd/system/${svc}.service"
        rm -rf "/etc/systemd/system/${svc}.service.d"
    else
        log "$svc not installed; skipping"
    fi
done
systemctl daemon-reload
log "systemd units removed"

if [ "$PURGE" = "1" ]; then
    log "--purge: removing venv + EnvironmentFile (NOT var/ state or config.toml)"
    rm -rf "$REPO_ROOT/.venv"
    rm -f /etc/pancakebot/pancakebot.env
    log "purged venv + /etc/pancakebot/pancakebot.env"
else
    log "kept venv, .env, config.toml, var/ state (use --purge to remove venv + EnvironmentFile)"
fi
log "DONE"
