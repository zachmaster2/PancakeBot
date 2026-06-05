# PancakeBot bootstrap — one-script setup (Windows + Linux)

Fresh clone → working bot, idempotent, reversible. The OS-specific work is
isolated; the shared logic (venv, config check, health check, service spec)
lives in `common/` and runs through the cross-platform `ServicePlatform`
abstraction (`pancakebot/service/`).

```
bootstrap/
  install.{ps1,sh}     OS orchestrators (idempotent, verbose, validating)
  uninstall.{ps1,sh}   reversible companions
  common/
    python_setup.py    venv create + deps install (asserts Python >= 3.13)
    config_check.py     config.toml + .env + webhook presence (no mutation)
    health_check.py     post-install: registered -> start -> RUNNING -> READY
    service_specs.py    shared live/dry ServiceSpec builder
  windows/
    setup_service.py   register (pythonservice host) + adapter policy
    setup_autologon.ps1 -> scripts\setup_autologon.ps1 (Sysinternals)
    boot_survival.ps1  OPERATOR-UI opt-in: autologon + Claude relaunch + AUMID
    AUMID_stamper/     how to rebuild the out-of-repo C:\Tools stamper
  linux/
    setup_service.py   install systemd units (live + dry) via the adapter
    install_python313.sh  pyenv build of 3.13 (additive; system py untouched)
  MIGRATION.md         Windows -> Linux cutover + rollback playbook
```

## Prerequisites (both OSes)

A fresh clone has `config.toml` (tracked) but **not** `.env` (gitignored). Before
installing, create the repo-root `.env`:

```
THE_GRAPH_API_KEY=<key>
BSC_WALLET_PRIVATE_KEY=<key>
```

Discord webhooks are environment-scoped:
`PANCAKEBOT_{LIVE_ALERTS,DRY_ALERTS,GENERAL}_DISCORD_WEBHOOK_URL`.

## Windows

```powershell
# elevated PowerShell, from the repo root
powershell -ExecutionPolicy Bypass -File bootstrap\install.ps1
# optional operator-UI (autologon + Claude relaunch):  -IncludeOperatorUI
```

Registers `PancakeBotLive` / `PancakeBotDry` (pythonservice host) **disabled**.
Start with `scripts\enable_dry.ps1` (or `enable_live.ps1`). Reverse with
`bootstrap\uninstall.ps1`. The bot's reboot survival is the SCM Automatic-start
services — autologon/AUMID are operator-UI only and off by default.

## Linux (AlmaLinux 9.x)

```bash
sudo git clone <repo> /root/pancakebot && cd /root/pancakebot
sudo bash bootstrap/install.sh          # python3.13 (pyenv) + venv + units (disabled)
# fill /etc/pancakebot/pancakebot.env with the 3 webhook URLs, then:
sudo systemctl enable --now pancakebot-dry      # dry soak
```

Python 3.13 is built additively via pyenv (system `python3.9`/`python3.12`
untouched). Units carry `Conflicts=` (live evicts dry), `Restart=on-failure`,
`KillMode=control-group`, `Type=notify`. Reverse with `sudo bash
bootstrap/uninstall.sh` (`--purge` also removes the venv + EnvironmentFile).

## Validate

```
<venv-python> bootstrap/common/health_check.py --mode dry --service-name <svc>
#   svc = PancakeBotDry (Windows) | pancakebot-dry (Linux)
```

Asserts: service registered → starts → reaches RUNNING → bot emits READY.

## Migration

See **MIGRATION.md** for the Windows→Linux cutover sequence and rollback. The
headline risk: the production wallet is shared and there is **no cross-OS
mutex** — Windows-live and Linux-live must never run simultaneously (nonce
collision). Serialize the cutover; the dry soak is safe (dry sends no TXs).
