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
| `notify_user_followup.py` | Discord follow-up ping for pending operator-coordination messages (Task-Scheduler-invoked) |
| `notify_user_mark_answered.py` | Marks pending notifications answered (pair of the above) |

**Provisioning note:** `Autologon.exe` (Sysinternals) used to be
auto-downloaded by the archived `scripts/install_services.ps1`. If setting
up a fresh operator desktop, download it from Sysinternals into `C:\Tools`
(or alongside this script) before running `setup_autologon.ps1`.

## AUMID stamper (operator-UI only — not the trading bot)

`C:\Tools\stamp_claude_aumid.exe` stamps the Claude desktop window's
`PKEY_AppUserModel_ID` so the taskbar pins/groups it correctly after the
direct-launch (`launch_claude_admin_direct.vbs`) bypasses normal UWP
activation. `boot_survival.ps1` verifies it exists; it does NOT rebuild it.

The built binary and its C# source live OUTSIDE the repo, under `C:\Tools\`:

```
C:\Tools\stamp_claude_aumid.exe     # the built stamper
C:\Tools\src\stamp_claude_aumid.cs  # C# source
C:\Tools\launch_claude_admin_direct.vbs
C:\Tools\Autologon\Autologon64.exe  # Sysinternals
```

Rebuilding (if the exe is missing) — the source is a single C# file; build
with the .NET SDK or the in-box compiler:

```powershell
csc /target:exe /out:C:\Tools\stamp_claude_aumid.exe C:\Tools\src\stamp_claude_aumid.cs
```

It polls `EnumWindows` for the Claude window and calls
`SHGetPropertyStoreForWindow` + `SetValue(PKEY_AppUserModel_ID, ...)`.
The Linux bot host has no desktop/UWP — this is Windows-operator-only.
