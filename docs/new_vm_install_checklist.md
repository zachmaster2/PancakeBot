# New VM install checklist — Frankfurt droplet `139.59.209.230`

Bring a fresh DigitalOcean droplet (1 vCPU, $8/mo, AlmaLinux 9.x) to a
dry-verified, DISABLED bot. Source of truth is GitHub
(`github.com/zachmaster2/PancakeBot`). The live bot is **not** enabled at
install time — the weekly monitor (or you, manually) decides that later.

Legend: **[user]** = you do it (DO panel / secrets), **[impl]** = Claude can
do it over SSH once access is set up.

## 1. SSH access  **[user]**
- In the DO panel, add your SSH public key to droplet `139.59.209.230`
  (or `ssh-copy-id root@139.59.209.230`).
- Verify: `ssh root@139.59.209.230 "hostname; cat /etc/os-release | head -1"`.
- When this works, tell Claude — steps 3–11 can then run over SSH.

## 2. Base OS prep  **[impl]**
`install.sh` handles python 3.13 (pyenv), venv, chrony drop-in, journald is
default. Pre-reqs it assumes present: `git`, `gcc`/build tools (pyenv build),
`curl`. On a bare AlmaLinux: `dnf install -y git gcc make patch zlib-devel
bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel tk-devel
libffi-devel xz-devel`.

## 3. GitHub deploy key (read-only)  **[impl] + [user] one click**
```bash
ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519_pancakebot
cat /root/.ssh/id_ed25519_pancakebot.pub          # -> add to GitHub
cat >> /root/.ssh/config <<'EOF'
Host github.com
    IdentityFile /root/.ssh/id_ed25519_pancakebot
    IdentitiesOnly yes
EOF
```
**[user]:** GitHub → repo → Settings → Deploy keys → Add → paste the `.pub`,
leave "Allow write access" **unchecked** (read-only pull is all the VM needs).
Verify: `ssh -T git@github.com` (expect the "successfully authenticated,
no shell access" message).

## 4. Clone  **[impl]**
```bash
git clone git@github.com:zachmaster2/PancakeBot.git /root/pancakebot
```
(The systemd units hardcode `/root/pancakebot` — clone exactly there.)

## 5. venv + deps  **[impl]**
Handled by `install.sh` STEP 2, or manually:
`cd /root/pancakebot && bash bootstrap/install.sh` (does 2–6 in one shot).
Manual venv: `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## 6. Secrets — `/etc/pancakebot/pancakebot.env`  **[user]**
`install.sh` scaffolds it (0600). Fill in:
```
BSC_WALLET_PRIVATE_KEY=<the wallet key guarding the 2.30627 BNB>
THE_GRAPH_API_KEY=<the graph key>
```
Also create repo-root `.env` with the same two vars (STEP 3 config check +
direct `run.py --sync` read it). **Never commit either file** (both gitignored).

## 7. Alerts — `/etc/pancakebot/alerts.env`  **[user]**
```
PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL=<...>
PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL=<...>
PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL=<...>
```

## 8. Run installer  **[impl]**
```bash
cd /root/pancakebot && sudo bash bootstrap/install.sh
```
Installs python3.13 + venv + systemd units (DISABLED) + chrony drop-in.

## 9. Config check  **[impl]**
```bash
.venv/bin/python bootstrap/common/config_check.py
```
(config.toml + .env + webhook presence; no mutation.) Confirm the deploy
sizing: `max_bet_bnb_*` = 0.1 ceiling, `min_bet_only = true` (contract-min
~0.001 BNB bets until you deliberately flip it after min-bet live validation).

## 10. Sync data  **[impl]**
```bash
.venv/bin/python run.py --sync         # ~30 min: closed rounds + OKX klines
```

## 11. Dry-mode smoke  **[impl]**
```bash
systemctl start pancakebot-dry
journalctl -u pancakebot-dry -f        # watch ~5 rounds: READY, wakes, decisions
.venv/bin/python bootstrap/common/health_check.py --mode dry --service-name pancakebot-dry
```
Confirm normal round cadence + no ALERTs (VM-tuned 250ms RPC timeouts are
fine on a real datacenter link). Then `systemctl stop pancakebot-dry`.

## 12. Live stays DISABLED  **[by design]**
Do **NOT** `systemctl enable pancakebot-live` at install. The bot only goes
live when the weekly monitor's positive trigger clears the **strict**
(Šidák-corrected) gate AND is explicitly armed, or you enable it by hand
after a confirmed edge. As of 2026-06-30 the recent signal was RULED OUT
(noise), so the correct state is DISABLED.

## After install — weekly monitor
```bash
# dry (report only, touches nothing):
.venv/bin/python research/weekly_monitor_state_machine.py
# live actions (auto-disable allowed; auto-enable still needs --arm + strict gate):
.venv/bin/python research/weekly_monitor_state_machine.py --apply
```
Schedule via systemd timer or cron (weekly). It syncs, evaluates the 2w/1w
windows, and toggles the live unit under the fail-safe rules
(auto-disable autonomous; auto-enable gated on corrected significance +
`--arm`).
