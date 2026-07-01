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
```

**Source of truth = GitHub** (`github.com/zachmaster2/PancakeBot`). The VM
is a plain `git clone` of that remote; deploys are `git pull`. Any machine
with the repo cloned + push access can be a dev source (push to GitHub;
the VM pulls). The old VM-bare-repo push-to-deploy hook was retired
2026-06-30.

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
Precedence: under systemd the EnvironmentFiles win (`load_dotenv` never
overrides an already-set var); direct `run.py` invocations read the
repo-root `.env`. Keep the two copies in sync when rotating secrets.
Verify webhook delivery (the script reads process env, so source the
file first):

```bash
set -a; . /etc/pancakebot/alerts.env; set +a
.venv/bin/python scripts/_smoke_discord_send_test.py
```

## Install (AlmaLinux 9.x)

```bash
# Fresh-VM bring-up: clone from GitHub into /root/pancakebot (the units
# hardcode that path). Needs a read-only GitHub deploy key on the VM
# (ssh-keygen -t ed25519, add the .pub as a deploy key on the repo) OR an
# HTTPS clone with a PAT. See the new-VM install checklist for the exact
# deploy-key steps.
git clone git@github.com:zachmaster2/PancakeBot.git /root/pancakebot
cd /root/pancakebot
# create repo-root .env with BSC_WALLET_PRIVATE_KEY + THE_GRAPH_API_KEY
# first (STEP 3's config check blocks without it — see Prerequisites)
sudo bash bootstrap/install.sh   # py3.13 + venv + units (disabled) + chrony drop-in
# fill /etc/pancakebot/{pancakebot,alerts}.env, then:
sudo systemctl enable --now pancakebot-dry      # dry soak
# going live after the soak: see docs/SUPERVISOR.md "Going live"
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

Push to GitHub from any dev clone, then on the VM: `git -C /root/pancakebot
pull` and restart the unit manually when greenlit (`systemctl restart
pancakebot-{live,dry}`). The weekly monitor (research/weekly_monitor_state_machine.py)
can also pull + evaluate + toggle the bot on a schedule.

## Validate

```
.venv/bin/python bootstrap/common/health_check.py --mode dry --service-name pancakebot-dry
```

Asserts: clock synchronized (chronyc, |offset| <= 250ms) → service
registered → starts → reaches RUNNING → bot emits READY. NOTE the
`Conflicts=` guard: it refuses to start the dry unit while live is running
(and vice versa) — health-check the unit that is supposed to be active.
