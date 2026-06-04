# AUMID stamper (operator-UI only — not the trading bot)

`stamp_claude_aumid.exe` stamps the Claude desktop window's
`PKEY_AppUserModel_ID` so the taskbar pins/groups it correctly after the
direct-launch (`launch_claude_admin_direct.vbs`) bypasses normal UWP
activation. It is purely an operator-convenience for the Claude desktop app —
the **trading bot does not use it**.

## Current location (live host, 2026-06-04)

The built binary and its C# source live OUTSIDE the repo, under `C:\Tools\`:

```
C:\Tools\stamp_claude_aumid.exe     # the built stamper
C:\Tools\src\stamp_claude_aumid.cs  # C# source
C:\Tools\launch_claude_admin_direct.vbs
C:\Tools\Autologon\Autologon64.exe  # Sysinternals
```

`bootstrap\windows\boot_survival.ps1` verifies these exist and wires the
`ClaudeLaunchElevated` scheduled task to them. It does NOT rebuild the binary.

## Rebuilding the stamper (if `C:\Tools\stamp_claude_aumid.exe` is missing)

The source is a single C# file. Build with the .NET SDK or the in-box C#
compiler:

```powershell
# in-box Roslyn/csc (path varies) or: dotnet build
csc /target:exe /out:C:\Tools\stamp_claude_aumid.exe C:\Tools\src\stamp_claude_aumid.cs
```

It polls `EnumWindows` for the Claude window and calls
`SHGetPropertyStoreForWindow` + `SetValue(PKEY_AppUserModel_ID, ...)`.

## Migration note

On the **Linux** target this whole directory is irrelevant — a headless server
has no desktop, no UWP, and no operator GUI to relaunch. The Linux installer
(`bootstrap/install.sh`) has no operator-UI step at all.
