#!/usr/bin/env bash
# Install Python 3.13 additively via pyenv (idempotent). AlmaLinux 9.x ships
# only python3.9 (system) / python3.12 (AppStream) — neither matches the dev
# runtime (3.13), so we build 3.13 from source via pyenv WITHOUT touching the
# system interpreter.
#
# Safe to re-run: build deps install is idempotent (dnf), pyenv clone is
# skipped if present, and `pyenv install -s` skips an already-built version.
#
# Prints the resulting interpreter path on the last line (consumed by
# install.sh). Run as root (or a user with sudo for dnf).
set -euo pipefail

log() { echo "[install_python313] $*"; }

# Default to /opt/pyenv for a root/system install: SELinux (Enforcing on
# AlmaLinux) denies systemd from exec'ing binaries under /root
# (admin_home_t context) -> 203/EXEC. /opt is usr_t (service-exec'able).
# Override with PYENV_ROOT=... for a user install.
if [ -z "${PYENV_ROOT:-}" ]; then
    if [ "$(id -u)" = "0" ]; then PYENV_ROOT="/opt/pyenv"; else PYENV_ROOT="$HOME/.pyenv"; fi
fi
export PYENV_ROOT

log "installing build dependencies (dnf)"
dnf install -y gcc make patch git zlib-devel bzip2 bzip2-devel readline-devel \
    sqlite sqlite-devel openssl-devel tk-devel libffi-devel xz-devel >/dev/null

if [ ! -d "$PYENV_ROOT" ]; then
    log "cloning pyenv -> $PYENV_ROOT"
    git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT" >/dev/null
else
    log "pyenv already present at $PYENV_ROOT"
fi

PYENV="$PYENV_ROOT/bin/pyenv"
VER="$("$PYENV" install --list | grep -E '^[[:space:]]*3\.13\.[0-9]+$' | tail -1 | tr -d ' ')"
if [ -z "$VER" ]; then
    log "ERROR: pyenv lists no 3.13.x version (update pyenv)"; exit 1
fi

PY="$PYENV_ROOT/versions/$VER/bin/python"
if [ -x "$PY" ]; then
    log "python $VER already built"
else
    log "building python $VER from source (~10min on a small VM)"
    "$PYENV" install -s "$VER"
fi

"$PY" --version
log "OK"
# Last line: the interpreter path (machine-readable for install.sh).
echo "$PY"
