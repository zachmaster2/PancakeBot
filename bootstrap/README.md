# PancakeBot bootstrap — Linux bot deploy

Fresh clone → working bot on the Linux VM, idempotent, reversible. The
shared logic (venv, config check, health check, service spec) lives in
`common/` and runs through the `ServicePlatform` abstraction
(`pancakebot/service/`, Linux/systemd adapter).

Phase 3c-1 (2026-06-10): this directory is Linux-bot-only. The Windows-bot
service installers (SCM/pywin32) live in the offline archive
(`Downloads/OLD/pancakebot_old/`); the Claude operator-desktop scaffolding
(autologon, elevated launcher, AUMID stamper notes) moved to
`tools/claude_desktop/`.

```
bootstrap/
  install.sh           orchestrator (idempotent, verbose, validating)
  uninstall.sh         reversible companion (--purge also removes venv + env file)
  common/
    python_setup.py    venv create + deps install (asserts Python >= 3.13)
    config_check.py    config.toml + .env + webhook presence (no mutation)
    health_check.py    post-install: clock sync -> registered -> RUNNING -> READY
                       (refuses to start a unit whose Conflicts= partner runs)
    service_specs.py   shared live/dry ServiceSpec builder (systemd units)
  linux/
    setup_service.py   install systemd units (live + dry) via the adapter
    install_python313.sh  pyenv build of 3.13 (additive; system py untouched)
    git_post_receive.sh   push-to-deploy hook (tracked copy + setup recipe)
```

## Prerequisites

A fresh clone has `config.toml` (tracked). Secrets live OUTSIDE the repo on
the VM: `/etc/pancakebot/pancakebot.env` (0600, systemd `EnvironmentFile`)
carries `BSC_WALLET_PRIVATE_KEY`, `THE_GRAPH_API_KEY`, and the 3
`PANCAKEBOT_{LIVE_ALERTS,DRY_ALERTS,GENERAL}_DISCORD_WEBHOOK_URL` hooks
(install.sh scaffolds the file; fill it in).

## Install (AlmaLinux 9.x)

```bash
sudo git clone <repo> /root/pancakebot && cd /root/pancakebot
sudo bash bootstrap/install.sh   # py3.13 + venv + units (disabled) + chrony drop-in
# fill /etc/pancakebot/pancakebot.env, then:
sudo systemctl enable --now pancakebot-dry      # dry soak
```

Python 3.13 is built additively via pyenv (system `python3.9`/`python3.12`
untouched). Units carry `Conflicts=` (live evicts dry and vice versa —
one bot at a time), `Restart=on-failure`, `KillMode=control-group`,
`Type=notify`. STEP 6 installs the chrony drop-in
(`/etc/chrony.d/pancakebot.conf`) bounding clock-step detection to ~64s.
Reverse with `sudo bash bootstrap/uninstall.sh`.

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
