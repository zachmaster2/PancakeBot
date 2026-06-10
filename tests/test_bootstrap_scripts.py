"""Lint/dry-run tests for the bootstrap scaffolding — no actual install.

  - bash -n syntax check on the *.sh scripts
  - PowerShell parser check on the *.ps1 scripts
  - import + pure-logic checks on the Python helpers (no side effects)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BOOT = _REPO_ROOT / "bootstrap"
_TOOLS = _REPO_ROOT / "tools" / "claude_desktop"

# Phase 3c-1 (2026-06-10): the Windows-bot-service cluster (install.ps1,
# windows/setup_service.py, SCM scripts) is archived; bootstrap/ is
# Linux-bot-only and the Claude operator-desktop scaffolding lives in
# tools/claude_desktop/. Phase 3c-2: systemd is the supervisor — the units
# are TRACKED files (installed verbatim by install.sh STEP 5), so they get
# content checks here.
_SH_SCRIPTS = [
    _BOOT / "install.sh",
    _BOOT / "uninstall.sh",
    _BOOT / "linux" / "install_python313.sh",
    _BOOT / "linux" / "git_post_receive.sh",
]
_PS1_SCRIPTS = [
    _TOOLS / "setup_autologon.ps1",
    _TOOLS / "boot_survival.ps1",
]
_PY_HELPERS = [
    _BOOT / "common" / "python_setup.py",
    _BOOT / "common" / "config_check.py",
    _BOOT / "common" / "health_check.py",
    _TOOLS / "notify_user_followup.py",
    _TOOLS / "notify_user_mark_answered.py",
]
_UNIT_FILES = [
    _BOOT / "linux" / "systemd" / "pancakebot-live.service",
    _BOOT / "linux" / "systemd" / "pancakebot-dry.service",
    _BOOT / "linux" / "systemd" / "pancakebot-notify@.service",
]


def test_all_expected_files_exist():
    for f in _SH_SCRIPTS + _PS1_SCRIPTS + _PY_HELPERS + _UNIT_FILES + [
        _BOOT / "README.md",
        _TOOLS / "README.md",
        _TOOLS / "AUMID_stamper" / "README.md",
        _TOOLS / "launch_claude_admin_direct.vbs",
    ]:
        assert f.exists(), f"missing bootstrap/tools file: {f}"


def test_vbs_launcher_structure():
    """The repo-tracked Claude launcher keeps its Task-F hardening + the core
    elevated CreateProcess-on-exe path. (No VBScript linter exists; this is a
    content sanity check — runtime behavior is validated via its /check mode.)"""
    vbs = _TOOLS / "launch_claude_admin_direct.vbs"
    text = vbs.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "Option Explicit",
        "PACKAGE_FAMILY_NAME",
        "LAUNCH_FAIL_AUTO_REBOOT_ENABLED",   # auto-reboot toggle
        "MAX_REBOOTS_PER_DAY",                # per-day reboot cap
        "Restart-Service AppXSvc",            # AppXSvc recovery step
        "candidateExe",                       # core direct-launch path preserved
    ):
        assert marker in text, f"VBS launcher missing expected marker: {marker!r}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", _SH_SCRIPTS, ids=lambda p: p.name)
def test_sh_syntax(script):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n failed for {script.name}: {r.stderr}"


@pytest.mark.skipif(shutil.which("powershell") is None, reason="powershell not available")
@pytest.mark.parametrize("script", _PS1_SCRIPTS, ids=lambda p: p.name)
def test_ps1_parses(script):
    cmd = (
        "$e=$null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{script}',[ref]$null,[ref]$e)"
        " > $null; if ($e -and $e.Count -gt 0) { $e | ForEach-Object { $_.Message }; exit 1 }"
    )
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"PowerShell parse errors in {script.name}: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("helper", _PY_HELPERS, ids=lambda p: p.name)
def test_py_helper_compiles(helper):
    # py_compile catches syntax errors without importing (no side effects).
    import py_compile
    py_compile.compile(str(helper), doraise=True)


# -- pure-logic checks -----------------------------------------------------


@pytest.mark.skipif(
    not (_REPO_ROOT / ".env").exists(),
    reason="repo-root .env exists only on the dev host (the VM keeps "
           "secrets in /etc/pancakebot/*.env via EnvironmentFile)",
)
def test_config_check_passes_on_real_repo():
    sys.path.insert(0, str(_BOOT))
    from common import config_check
    # Dev-host repo has config.toml (tracked) + .env (local). Webhooks may
    # warn but are not blockers.
    blockers = config_check.check(repo_root=_REPO_ROOT, env={})
    assert blockers == [], f"unexpected blockers: {blockers}"


def test_config_check_flags_missing_secrets(tmp_path):
    sys.path.insert(0, str(_BOOT))
    from common import config_check
    (tmp_path / "config.toml").write_text(
        "[runtime]\n[live]\n[dry]\n[strategy]\n", encoding="utf-8"
    )
    # no .env
    blockers = config_check.check(repo_root=tmp_path, env={})
    assert any(".env missing" in b for b in blockers)


@pytest.mark.parametrize("mode", ["live", "dry"])
def test_bot_unit_structure(mode):
    """The tracked bot units carry the systemd-direct invariants: direct
    run.py ExecStart (no Python supervisor layer), restart-on-failure with
    the StartLimitBurst crashloop brake, live<->dry mutual exclusion, and
    the notify hooks on both lifecycle edges."""
    text = (_BOOT / "linux" / "systemd" / f"pancakebot-{mode}.service").read_text(
        encoding="utf-8")
    other = "dry" if mode == "live" else "live"
    for marker in (
        f"ExecStart=/root/pancakebot/.venv/bin/python -u run.py --{mode}",
        f"Conflicts=pancakebot-{other}.service",
        "Restart=on-failure",
        "StartLimitBurst=5",
        "StartLimitIntervalSec=900",
        "KillMode=control-group",
        "ExecStartPost=-/usr/bin/systemctl start --no-block "
        "pancakebot-notify@%p-started.service",
        "ExecStopPost=-/usr/bin/systemctl start --no-block "
        "pancakebot-notify@%p-stopped.service",
        "EnvironmentFile=-/etc/pancakebot/alerts.env",
    ):
        assert marker in text, f"pancakebot-{mode}.service missing: {marker!r}"


def test_notify_unit_least_privilege():
    """The notify template loads ONLY alerts.env — the wallet key
    (pancakebot.env) must never enter the notify process."""
    text = (_BOOT / "linux" / "systemd" / "pancakebot-notify@.service").read_text(
        encoding="utf-8")
    env_lines = [ln.strip() for ln in text.splitlines()
                 if ln.strip().startswith("EnvironmentFile=")]
    assert env_lines == ["EnvironmentFile=-/etc/pancakebot/alerts.env"], env_lines
    assert "notify_lifecycle %i" in text


def test_python_setup_venv_python_path():
    sys.path.insert(0, str(_BOOT))
    from common import python_setup
    vp = python_setup.venv_python(Path("/repo/.venv"))
    if sys.platform == "win32":
        assert vp == Path("/repo/.venv/Scripts/python.exe")
    else:
        assert vp == Path("/repo/.venv/bin/python")
