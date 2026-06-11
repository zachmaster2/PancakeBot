# PancakeBot bootstrap — Linux bot deploy

Fresh clone → working bot on the Linux VM, idempotent, reversible. The
shared logic (venv, config check, health check) lives in `common/`; the
systemd units are tracked files under `linux/systemd/`, installed verbatim
(systemd IS the supervisor — Phase 3c-2, see docs/SUPERVISOR.md).

Phase 3c-1 (2026-06-10): this directory is Linux-bot-only. The Windows-bot
service installers (SCM/pywin32) and the retired Python supervisor stack
live in the offline archive (`Downloads/OLD/pancakebot/2026_06_10_phase3_repo_archive/`); the Claude
operator-desktop scaffolding (autologon, elevated launcher, AUMID stamper
notes) lives in `tools/claude_desktop/`.

```
bootstrap/
  install.sh           orchestrator (idempotent, verbose, validating)
  uninstall.sh         reversible companion (--purge also removes venv + env files)
  common/
    python_setup.py    venv create + deps install (asserts Python >= 3.13)
    config_check.py    config.toml + .env + webhook presence (no mutation)
    health_check.py    post-install: clock sync -> registered -> active -> READY
                       (refuses to start a unit whose Conflicts= partner runs)
  linux/
    systemd/           tracked units: pancakebot-{live,dry}.service +
                       pancakebot-notify@.service (cp'd by install.sh STEP 5)
    install_python313.sh  pyenv build of 3.13 (additive; system py untouched)
    git_post_receive.sh   push-to-deploy hook (tracked copy + setup recipe)
```

## Prerequisites

A fresh clone has `config.toml` (tracked). Secrets live OUTSIDE the repo on
the VM, split for least privilege (the notify unit loads only the webhooks,
never the wallet key):

- `/etc/pancakebot/pancakebot.env` (0600) — `BSC_WALLET_PRIVATE_KEY`,
  `THE_GRAPH_API_KEY` (bot units only)
- `/etc/pancakebot/alerts.env` (0600) — the 3
  `PANCAKEBOT_{LIVE_ALERTS,DRY_ALERTS,GENERAL}_DISCORD_WEBHOOK_URL` hooks
  (bot units + notify template)

install.sh scaffolds both; fill them in. **Also create a repo-root
`.env`** with `BSC_WALLET_PRIVATE_KEY` + `THE_GRAPH_API_KEY` before
running install.sh — its STEP 3 config check (and any direct
`run.py --sync`) reads secrets from `.env`/process env; the
`/etc/pancakebot` EnvironmentFiles are what the systemd units load.
Verify webhook delivery with
`.venv/bin/python scripts/_smoke_discord_send_test.py`.

## Install (AlmaLinux 9.x)

```bash
sudo git clone <repo> /root/pancakebot && cd /root/pancakebot
sudo bash bootstrap/install.sh   # py3.13 + venv + units (disabled) + chrony drop-in
# fill /etc/pancakebot/{pancakebot,alerts}.env, then:
sudo systemctl enable --now pancakebot-dry      # dry soak
```

Python 3.13 is built additively via pyenv (system `python3.9`/`python3.12`
untouched). The bot units run `run.py` directly (`Type=exec`), carry
`Conflicts=` (live evicts dry and vice versa — one bot at a time),
`Restart=on-failure` + `StartLimitBurst=5/900s` (crashloop brake),
`KillMode=control-group`, and trigger `pancakebot-notify@` oneshots on
start/stop edges for Discord lifecycle alerts. STEP 6 installs the chrony
drop-in (`/etc/chrony.d/pancakebot.conf`) bounding clock-step detection to
~64s. Reverse with `sudo bash bootstrap/uninstall.sh`.

## Deploys after install

`git push vm master` from the dev clone (see README.md "Deploying") — the
VM's bare-repo hook checks master out into `/root/pancakebot`; restart the
unit manually when greenlit.

## Validate

```
.venv/bin/python bootstrap/common/health_check.py --mode dry --service-name pancakebot-dry
```

Asserts: clock synchronized (chronyc, |offset| <= 250ms) → service
registered → starts → reaches RUNNING → bot emits READY. NOTE the
`Conflicts=` guard: it refuses to start the dry unit while live is running
(and vice versa) — health-check the unit that is supposed to be active.
