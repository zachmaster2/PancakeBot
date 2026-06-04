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

_SH_SCRIPTS = [
    _BOOT / "install.sh",
    _BOOT / "uninstall.sh",
    _BOOT / "linux" / "install_python313.sh",
]
_PS1_SCRIPTS = [
    _BOOT / "install.ps1",
    _BOOT / "uninstall.ps1",
    _BOOT / "windows" / "setup_autologon.ps1",
    _BOOT / "windows" / "boot_survival.ps1",
]
_PY_HELPERS = [
    _BOOT / "common" / "python_setup.py",
    _BOOT / "common" / "config_check.py",
    _BOOT / "common" / "health_check.py",
    _BOOT / "common" / "service_specs.py",
    _BOOT / "linux" / "setup_service.py",
    _BOOT / "windows" / "setup_service.py",
]


def test_all_expected_files_exist():
    for f in _SH_SCRIPTS + _PS1_SCRIPTS + _PY_HELPERS + [
        _BOOT / "README.md", _BOOT / "MIGRATION.md",
        _BOOT / "windows" / "AUMID_stamper" / "README.md",
    ]:
        assert f.exists(), f"missing bootstrap file: {f}"


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


def test_config_check_passes_on_real_repo():
    sys.path.insert(0, str(_BOOT))
    from common import config_check
    # Real repo has config.toml (tracked) + .env (local). Webhooks may warn but
    # are not blockers.
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


def test_service_specs_build():
    sys.path.insert(0, str(_BOOT))
    from common.service_specs import build_spec
    spec = build_spec(mode="live", repo_root=_REPO_ROOT, venv_python=Path("/x/python"))
    assert spec.args == ("-m", "pancakebot.service.supervise", "--mode", "live")
    assert spec.conflicts_with  # live conflicts with dry
    assert spec.restart_max_attempts == 3
    # name differs by OS convention
    assert spec.name in ("PancakeBotLive", "pancakebot-live")


def test_python_setup_venv_python_path():
    sys.path.insert(0, str(_BOOT))
    from common import python_setup
    vp = python_setup.venv_python(Path("/repo/.venv"))
    if sys.platform == "win32":
        assert vp == Path("/repo/.venv/Scripts/python.exe")
    else:
        assert vp == Path("/repo/.venv/bin/python")
