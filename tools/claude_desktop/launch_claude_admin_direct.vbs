' Launches Claude.exe directly from the REGISTERED Claude AppX package's
' install location, ELEVATED (the ClaudeLaunchElevated AtLogon task runs at
' RunLevel=Highest). Direct CreateProcess-on-exe is deliberate and MUST be kept:
' it preserves elevation (Cowork/computer-use needs it), the taskbar button's
' auto-elevation, no flashing command window, and auto-attach to a running
' Claude. Shell/AUMID activation would launch at medium IL and is NOT used.
'
' Variant B hardening (2026-06-05) -- guards a PERSISTENT package lock (a
' staged-but-unapplied "Update and restart", or a wedged AppX Deployment
' Service) that makes CreateProcess on Claude.exe return 0x80070020
' ERROR_SHARING_VIOLATION for minutes-to-hours. Recovery cascade:
'   1. Pre-flight: Get-AppxPackage Status <> 'Ok' -> package being serviced ->
'      skip the doomed CreateProcess, go to recovery.
'   2. Launch with a SHORT retry (a few hundred ms) for transient collisions.
'   3. Persistent fail / not-Ok -> Restart-Service AppXSvc (we are elevated) +
'      retry launch once.
'   4. Still failing -> a clean, CANCELLABLE 60s warning dialog (auto-proceeds
'      if unattended), logged to C:\Tools\claude_launch.log.
'   5. Auto-reboot via 'shutdown /r /t 60' (a reboot reliably clears the wedged
'      package state; the operator may not be at the machine). After reboot the
'      AtLogon task fires again with a hopefully-clean package.
' Reboot is GUARDED: LAUNCH_FAIL_AUTO_REBOOT_ENABLED toggles it off, and a
' max-reboots-per-day cap prevents a reboot loop if the issue is persistent.
'
' Modes (validation, no side effects):
'   /check     - dry-run the happy path (query + Status gate + would-launch).
'   /checkfail - dry-run the FAILURE cascade (recovery + reboot DECISION,
'                including the per-day cap) without launching/restarting/rebooting.
'
' Periodic keepalive (2026-06-09 -- companion ClaudeKeepalive task, ~5 min):
'   /keepalive      - launch-if-down only; Status-gates but DELIBERATELY does NOT
'                     run the AppXSvc-restart / auto-reboot cascade (that would be
'                     hit every 5 min on a persistent lock). Recovery stays in the
'                     AtLogon task; keepalive just re-spawns a cleanly-died Claude.
'                     Closes the gap where Claude dies mid-session and the AtLogon
'                     trigger never re-fires (long-lived logged-in session).
'   /keepalivecheck - dry-run the keepalive decision (no launch).
'
' QueryPackage race hardening (2026-06-17) -- after a Windows-update reboot the
' AtLogon ClaudeLaunchElevated task and the ~5-min ClaudeKeepalive task can
' co-launch within milliseconds of each other. Both elevated copies share %TEMP%
' and previously raced on a single scratch file (\claude_pkg_query.txt). The
' loser raised a fatal WSH dialog "Permission denied 800A0046" at the
' unguarded fs.DeleteFile call. Fix: (a) per-invocation random nonce in the
' scratch filename eliminates the collision at root, (b) every file I/O in
' QueryPackage is now guarded by On Error Resume Next -- matching the
' On-Error discipline of every other helper -- so any residual transient
' degrades to a quiet no-op (Quit 2 via Len(installLocation)=0) instead of a
' modal dialog the operator may not be there to dismiss.

Option Explicit

Const PACKAGE_FAMILY_NAME = "Claude_pzs8sxrjxfjjc"
Const LOG_PATH = "C:\Tools\claude_launch.log"
Const MAX_LAUNCH_RETRIES = 3            ' short retry: transient locks only
Const RETRY_SLEEP_MS = 250
Const POPUP_TIMEOUT_S = 45             ' manual-recovery message auto-dismiss

' --- auto-reboot fallback config (edit to disable / re-tune) ---
Const LAUNCH_FAIL_AUTO_REBOOT_ENABLED = True
Const MAX_REBOOTS_PER_DAY = 3          ' cap to prevent reboot loops
Const REBOOT_DIALOG_TIMEOUT_S = 60     ' cancellable countdown before reboot
Const REBOOT_STATE_PATH = "C:\Tools\claude_reboot_state.txt"  ' "YYYY-MM-DD|count"

Dim wsh, fs, dryRun, simulateFail, keepAlive
Set wsh = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")

' Mode parse (And is NOT short-circuit in VBScript; guard the arg access).
dryRun = False : simulateFail = False : keepAlive = False
If WScript.Arguments.Count > 0 Then
    Select Case UCase(WScript.Arguments(0))
        Case "/CHECK"          : dryRun = True
        Case "/CHECKFAIL"      : dryRun = True : simulateFail = True
        Case "/KEEPALIVE"      : keepAlive = True
        Case "/KEEPALIVECHECK" : keepAlive = True : dryRun = True
    End Select
End If

' --- Resolve the REGISTERED package: install location + servicing Status. ---
Dim installLocation, pkgStatus
QueryPackage installLocation, pkgStatus

If Len(installLocation) = 0 Then
    LogBoth "no registered Claude package (PackageFamilyName=" & PACKAGE_FAMILY_NAME & " returned empty)"
    WScript.Quit 2
End If

Dim candidateExe
candidateExe = installLocation & "\app\Claude.exe"
If Not fs.FileExists(candidateExe) Then
    LogBoth "registered package has no app\Claude.exe at " & candidateExe
    WScript.Quit 3
End If

' --- KEEPALIVE MODE (periodic, ~5 min). Launch-if-down only; NO recovery
' --- cascade (that stays at-logon -- a persistent lock must not be hit every
' --- 5 min). Returns fast. ---
If keepAlive Then
    If pkgStatus <> "Ok" Then
        LogBoth "keepalive: package Status='" & pkgStatus & "' (not Ok) -- skipping; recovery cascade is at-logon only"
        WScript.Quit 6
    End If
    If IsClaudeRunning() Then
        LogFile "keepalive: Claude already running -- no-op"
        WScript.Quit 0
    End If
    LogBoth "keepalive: Claude not running -- relaunching (keepalive does NOT run the recovery cascade)"
    If LaunchWithShortRetry(candidateExe) Then
        StampAumid
        If Not dryRun Then LogBoth "keepalive: relaunch succeeded"
        WScript.Quit 0
    End If
    LogBoth "keepalive: relaunch FAILED after short retries -- recovery deferred to the at-logon task"
    WScript.Quit 7
End If

' --- Step 1: pre-flight Status gate. ---
If pkgStatus <> "Ok" Then
    LogBoth "package Status='" & pkgStatus & "' (not Ok) -- skipping direct launch, attempting recovery"
    RecoverAndRetryOrEscalate candidateExe
    WScript.Quit 5
End If

' --- Step 2: launch (CORE PATH -- elevated CreateProcess) with short retry. ---
If LaunchWithShortRetry(candidateExe) Then
    StampAumid
    WScript.Quit 0
End If

' --- Steps 3-5: persistent failure -> recover -> escalate (reboot). ---
LogBoth "launch failed after short retries (persistent lock) -- attempting AppXSvc recovery"
RecoverAndRetryOrEscalate candidateExe
WScript.Quit 5


' ==========================================================================
' Helpers
' ==========================================================================

' InstallLocation + Status in ONE hidden PowerShell call (no flash). Output:
' "<InstallLocation>|<Status>" (paths can't contain '|'). Fills args ByRef.
' All file I/O is On-Error-guarded (see 2026-06-17 race note in the header).
Sub QueryPackage(ByRef loc, ByRef status)
    loc = "" : status = ""
    Dim tempPath, ps, cmd, ts, raw, parts, nonce
    ' Per-invocation random nonce so concurrent elevated invocations (AtLogon
    ' vs Keepalive at boot) never collide on the scratch file.
    Randomize
    nonce = CStr(Int(Rnd() * 1000000000)) & "_" & CStr(Int(Timer * 1000) Mod 1000000)
    tempPath = wsh.ExpandEnvironmentStrings("%TEMP%") & "\claude_pkg_query_" & nonce & ".txt"

    On Error Resume Next
    If fs.FileExists(tempPath) Then fs.DeleteFile tempPath
    Err.Clear
    On Error GoTo 0

    ps = "$p=Get-AppxPackage | ?{$_.PackageFamilyName -eq '" & PACKAGE_FAMILY_NAME & _
         "'} | Select-Object -First 1; if($p){'{0}|{1}' -f $p.InstallLocation,$p.Status}"
    cmd = "cmd.exe /c powershell.exe -NoProfile -ExecutionPolicy Bypass -Command """ & _
          ps & """ > """ & tempPath & """ 2>&1"

    On Error Resume Next
    wsh.Run cmd, 0, True
    If Not fs.FileExists(tempPath) Then
        Err.Clear
        On Error GoTo 0
        Exit Sub
    End If
    Set ts = fs.OpenTextFile(tempPath, 1)
    If Err.Number <> 0 Then
        Err.Clear
        On Error GoTo 0
        On Error Resume Next : fs.DeleteFile tempPath : Err.Clear : On Error GoTo 0
        Exit Sub
    End If
    raw = ts.ReadAll()
    ts.Close
    fs.DeleteFile tempPath        ' best-effort: %TEMP% cleanup tolerates failure
    Err.Clear
    On Error GoTo 0

    raw = Trim(Replace(Replace(raw & "", vbCr, ""), vbLf, ""))
    If Len(raw) = 0 Then Exit Sub
    parts = Split(raw, "|")
    loc = parts(0)
    If UBound(parts) >= 1 Then status = parts(1)
End Sub

' CORE PATH: elevated CreateProcess-on-exe (windowStyle=1; fire-and-forget) with
' a short retry. Returns True on a successful spawn. Dry-run aware.
Function LaunchWithShortRetry(exe)
    LaunchWithShortRetry = False
    If dryRun Then
        If simulateFail Then
            LogBoth "[checkfail] simulating persistent launch failure"
        Else
            LogBoth "[check] would launch (Status=Ok): " & exe
            LaunchWithShortRetry = True
        End If
        Exit Function
    End If
    Dim attempt, lastErr, lastDesc
    For attempt = 1 To MAX_LAUNCH_RETRIES
        On Error Resume Next
        wsh.Run """" & exe & """", 1, False
        lastErr = Err.Number : lastDesc = Err.Description
        On Error GoTo 0
        If lastErr = 0 Then
            LaunchWithShortRetry = True
            Exit Function
        End If
        LogBoth "launch attempt " & attempt & " failed: 0x" & Hex(lastErr) & " " & lastDesc
        If attempt < MAX_LAUNCH_RETRIES Then WScript.Sleep RETRY_SLEEP_MS
    Next
End Function

' Step 3: restart the AppX Deployment Service (elevated) + retry launch once.
' On success: stamp + quit 0. On failure: escalate to the reboot fallback.
Sub RecoverAndRetryOrEscalate(exe)
    If dryRun Then
        LogBoth "[dry] would Restart-Service AppXSvc, then retry launch once"
        EscalateToRebootOrSurface          ' continue the dry-run cascade
        Exit Sub
    End If
    LogBoth "auto-recovery: Restart-Service AppXSvc -Force (elevated)"
    On Error Resume Next
    wsh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -Command " & _
            """Restart-Service AppXSvc -Force -ErrorAction SilentlyContinue""", 0, True
    On Error GoTo 0
    WScript.Sleep 3000

    Dim e2, d2
    On Error Resume Next
    Err.Clear
    wsh.Run """" & exe & """", 1, False
    e2 = Err.Number : d2 = Err.Description
    On Error GoTo 0
    If e2 = 0 Then
        LogBoth "launch succeeded after AppXSvc recovery"
        StampAumid
        WScript.Quit 0
    End If
    LogBoth "launch still failing after AppXSvc recovery: 0x" & Hex(e2) & " " & d2
    EscalateToRebootOrSurface
End Sub

' Steps 4-5: final fallback. Reboot reliably clears the wedged package state.
' Guarded by a config toggle + a per-day cap (no reboot loops). Cancellable.
Sub EscalateToRebootOrSurface()
    If Not LAUNCH_FAIL_AUTO_REBOOT_ENABLED Then
        LogBoth "auto-reboot disabled (LAUNCH_FAIL_AUTO_REBOOT_ENABLED=False) -- surfacing manual recovery"
        If Not dryRun Then SurfaceRecoveryMessage
        Exit Sub
    End If

    Dim cnt
    cnt = GetTodayRebootCount()
    If cnt >= MAX_REBOOTS_PER_DAY Then
        LogBoth "auto-reboot SUPPRESSED: " & cnt & " reboots already today (cap " & _
                MAX_REBOOTS_PER_DAY & ") -- persistent issue, surfacing manual recovery"
        If Not dryRun Then SurfaceRecoveryMessage
        Exit Sub
    End If

    If dryRun Then
        LogBoth "[dry] would warn (" & REBOOT_DIALOG_TIMEOUT_S & "s, cancellable) then 'shutdown /r /t 60'" & _
                " (would be reboot " & (cnt + 1) & "/" & MAX_REBOOTS_PER_DAY & " today)"
        Exit Sub
    End If

    ' Cancellable warning. Auto-proceeds on timeout (unattended at-logon case).
    Dim resp
    resp = wsh.Popup( _
        "Claude could not start -- its app package appears locked by a pending " & _
        "update or deployment." & vbCrLf & vbCrLf & _
        "Windows will REBOOT in " & REBOOT_DIALOG_TIMEOUT_S & " seconds to recover." & vbCrLf & _
        "Click Cancel to abort the reboot (then recover manually: 'Update and " & _
        "restart' in Claude, or Restart-Service AppXSvc).", _
        REBOOT_DIALOG_TIMEOUT_S, "PancakeBot -- auto-reboot to recover Claude", 49)  ' 49 = OKCancel+Exclamation
    If resp = 2 Then  ' vbCancel
        LogBoth "auto-reboot CANCELLED by user at the warning dialog"
        Exit Sub
    End If

    IncrementTodayRebootCount cnt
    LogBoth "AUTO-REBOOT: 'shutdown /r /t 60' (reboot " & (cnt + 1) & "/" & MAX_REBOOTS_PER_DAY & _
            " today; reason=persistent Claude launch failure; cancel within 60s with 'shutdown /a')"
    On Error Resume Next
    wsh.Run "shutdown.exe /r /t 60 /c ""PancakeBot: rebooting to recover Claude " & _
            "(pending package lock). Run 'shutdown /a' to cancel.""", 0, False
    On Error GoTo 0
End Sub

' Manual-recovery message (auto-dismissing; safe for unattended). Used when
' auto-reboot is disabled or the per-day cap is reached.
Sub SurfaceRecoveryMessage()
    Dim msg
    msg = "Claude could not start -- its app package appears locked by a pending " & _
          "update or deployment." & vbCrLf & vbCrLf & _
          "Recovery options:" & vbCrLf & _
          "  (a) Click 'Update and restart' in Claude." & vbCrLf & _
          "  (b) Run as admin:  Restart-Service AppXSvc" & vbCrLf & _
          "  (c) Reboot Windows." & vbCrLf & vbCrLf & _
          "Details logged to " & LOG_PATH
    On Error Resume Next
    wsh.Popup msg, POPUP_TIMEOUT_S, "PancakeBot -- Claude launch failed", 48  ' 48 = exclamation
    On Error GoTo 0
End Sub

' Per-day reboot counter (persists across reboots). File: "YYYY-MM-DD|count".
Function GetTodayRebootCount()
    GetTodayRebootCount = 0
    On Error Resume Next
    If Not fs.FileExists(REBOOT_STATE_PATH) Then Exit Function
    Dim f, line, parts
    line = ""
    Set f = fs.OpenTextFile(REBOOT_STATE_PATH, 1)
    If Not f.AtEndOfStream Then line = f.ReadLine
    f.Close
    On Error GoTo 0
    parts = Split(Trim(line & ""), "|")
    If UBound(parts) >= 1 Then
        If parts(0) = TodayStamp() And IsNumeric(parts(1)) Then GetTodayRebootCount = CInt(parts(1))
    End If
End Function

Sub IncrementTodayRebootCount(currentCount)
    On Error Resume Next
    Dim f
    Set f = fs.OpenTextFile(REBOOT_STATE_PATH, 2, True)  ' 2=ForWriting (overwrite), create
    f.WriteLine TodayStamp() & "|" & (currentCount + 1)
    f.Close
    On Error GoTo 0
End Sub

Function TodayStamp()
    Dim d
    d = Date
    TodayStamp = Right("0000" & Year(d), 4) & "-" & Right("00" & Month(d), 2) & "-" & Right("00" & Day(d), 2)
End Function

' AUMID stamper (cosmetic taskbar grouping); best-effort, never blocks.
Sub StampAumid()
    If dryRun Then Exit Sub
    On Error Resume Next
    wsh.Run """C:\Tools\stamp_claude_aumid.exe""", 0, False
    On Error GoTo 0
End Sub

' Log to BOTH the Windows Application event log (WSH source) and a known file.
' Best-effort: a failure of either path never raises.
Sub LogBoth(msg)
    On Error Resume Next
    CreateObject("WScript.Shell").LogEvent 1, "launch_claude_admin_direct.vbs: " & msg
    Dim f
    Set f = fs.OpenTextFile(LOG_PATH, 8, True)  ' 8=ForAppending, create if absent
    f.WriteLine Now & "  " & msg
    f.Close
    On Error GoTo 0
End Sub

' File-only log (keepalive heartbeat). Keeps the 5-min no-op cadence out of the
' Windows Application event log; LogBoth is reserved for noteworthy events
' (relaunch, not-Ok skip, recovery cascade).
Sub LogFile(msg)
    On Error Resume Next
    Dim f
    Set f = fs.OpenTextFile(LOG_PATH, 8, True)  ' 8=ForAppending, create if absent
    f.WriteLine Now & "  " & msg
    f.Close
    On Error GoTo 0
End Sub

' True if at least one Claude.exe is running (WMI). Best-effort: a query failure
' returns False, so keepalive would attempt a launch -- harmless, since the launch
' auto-attaches to a running Claude rather than spawning a duplicate.
Function IsClaudeRunning()
    IsClaudeRunning = False
    On Error Resume Next
    Dim wmi, procs
    Set wmi = GetObject("winmgmts:\\.\root\cimv2")
    If Err.Number = 0 And Not (wmi Is Nothing) Then
        Set procs = wmi.ExecQuery("SELECT ProcessId FROM Win32_Process WHERE Name = 'Claude.exe'")
        If Err.Number = 0 And Not (procs Is Nothing) Then
            If procs.Count > 0 Then IsClaudeRunning = True
        End If
    End If
    On Error GoTo 0
End Function
