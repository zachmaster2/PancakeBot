#!/usr/bin/env bash
# PancakeBot push-to-deploy hook (Phase 3c-0, 2026-06-10).
#
# Tracked copy of /srv/pancakebot.git/hooks/post-receive on the VM.
# One-time VM setup (already applied to production 2026-06-10):
#
#   git init --bare /srv/pancakebot.git
#   git --git-dir=/srv/pancakebot.git symbolic-ref HEAD refs/heads/master
#   git --git-dir=/srv/pancakebot.git config core.bare false
#   git --git-dir=/srv/pancakebot.git config core.worktree /root/pancakebot
#   git --git-dir=/srv/pancakebot.git config receive.denyCurrentBranch ignore
#   cp <this file> /srv/pancakebot.git/hooks/post-receive
#   chmod +x /srv/pancakebot.git/hooks/post-receive
#   echo "gitdir: /srv/pancakebot.git" > /root/pancakebot/.git   # `git status` ergonomics
#
# Windows (dev) side, one-time:
#   git remote add vm ssh://root@167.172.100.184/srv/pancakebot.git
#
# Per-deploy flow: commit on Windows -> `git push vm master` -> the hook
# checks master out into the live working tree atomically. CODE ONLY —
# deliberately NO service restart (manual `systemctl restart
# pancakebot-live` when greenlit, per the deploy protocol). Untracked
# files (var/, .env, .venv) are never touched by checkout -f.
set -euo pipefail
unset GIT_DIR
BARE=/srv/pancakebot.git
TARGET=/root/pancakebot
while read -r _old _new refname; do
    if [ "$refname" = "refs/heads/master" ]; then
        git --git-dir="$BARE" --work-tree="$TARGET" checkout -f master -- 2>&1
        SHA=$(git --git-dir="$BARE" rev-parse --short master)
        echo "deployed $SHA -> $TARGET (code only; restart manually when greenlit)"
    fi
done
