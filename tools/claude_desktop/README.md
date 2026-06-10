# Claude operator-desktop scaffolding (Windows)

These files keep the **Claude desktop app** (the autonomous operator that
watches/manages the bot) alive on the Windows machine. The BOT itself runs
on the Frankfurt Linux VM and needs none of this — see `bootstrap/` for the
bot's install path. Split out of `bootstrap/windows/` + `scripts/` in
Phase 3c-1 (2026-06-10) when the Windows-bot-service cluster was archived
to `Downloads/OLD/pancakebot/2026_06_10_phase3_repo_archive/`.

| File | Purpose |
|---|---|
| `boot_survival.ps1` | One-shot setup of the boot-survival chain: autologon + `ClaudeLaunchElevated` scheduled task (AtLogon) + AUMID stamping verification, so the operator session survives reboots unattended |
| `setup_autologon.ps1` | Sysinternals-Autologon-based session auto-login (the implementation; the former `bootstrap/windows/setup_autologon.ps1` thin wrapper was folded in here) |
| `launch_claude_admin_direct.vbs` | Elevated Claude-app launcher (tracked copy of the `C:\Tools` deployment); also the `/keepalive` mode used by the `ClaudeKeepalive` 5-min relaunch-if-down task |
| `AUMID_stamper/README.md` | Rebuild instructions for the out-of-repo `C:\Tools\stamp_claude_aumid.exe` that `boot_survival.ps1` verifies |
| `notify_user_followup.py` | Discord follow-up ping for pending operator-coordination messages (Task-Scheduler-invoked) |
| `notify_user_mark_answered.py` | Marks pending notifications answered (pair of the above) |

**Provisioning note:** `Autologon.exe` (Sysinternals) used to be
auto-downloaded by the archived `scripts/install_services.ps1`. If setting
up a fresh operator desktop, download it from Sysinternals into `C:\Tools`
(or alongside this script) before running `setup_autologon.ps1`.
