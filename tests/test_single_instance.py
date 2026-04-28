"""Tests for ``find_duplicate_bots`` token-based cmdline matching.

Replaced the brittle substring match (literal "run.py --dry" /
"run.py --live") on 2026-04-27 after a read-only audit found the prior
implementation silently allowed a second bot to start when the typical
production invocation interposed ``--config <path>`` between ``run.py``
and ``--dry``:

  py.exe run.py --config config.toml --dry
                ↑                    ↑
                |                    |
                "run.py --dry" substring NOT FOUND -> dupe missed

Token-based matching looks for ``run.py`` (as a standalone token or
path-basename) AND ``--dry``/``--live`` (as standalone tokens) in the
psutil cmdline list, regardless of argument order.

Run:
    python -m pytest tests/test_single_instance.py -v
    python tests/test_single_instance.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime import single_instance  # noqa: E402
from pancakebot.runtime.single_instance import (  # noqa: E402
    _cmdline_is_bot,
    _is_run_py_token,
    find_duplicate_bots,
)


# ---------------------------------------------------------------------------
# Pure-function: _is_run_py_token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token,expected", [
    # Matches
    ("run.py", True),
    ("./run.py", True),
    ("/home/user/PancakeBot/run.py", True),
    ("C:\\Users\\zking\\Documents\\GitHub\\PancakeBot\\run.py", True),
    ("C:/Users/zking/Documents/GitHub/PancakeBot/run.py", True),
    # Non-matches
    ("python", False),
    ("py.exe", False),
    ("--dry", False),
    ("notrun.py", False),  # different filename containing "run.py"
    ("run.pyc", False),    # bytecode, different extension
    ("config.toml", False),
    ("", False),
])
def test_is_run_py_token(token, expected):
    assert _is_run_py_token(token) is expected, (
        f"_is_run_py_token({token!r}) returned {not expected}; expected {expected}"
    )


# ---------------------------------------------------------------------------
# Pure-function: _cmdline_is_bot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmdline,expected,reason", [
    # ✅ should match
    (["python", "run.py", "--dry"], True, "bare run.py --dry"),
    (["python", "run.py", "--live"], True, "bare run.py --live"),
    (["py", "run.py", "--config", "config.toml", "--dry"], True,
     "production form: --config interposed before --dry"),
    (["py.exe", "run.py", "--config", "config.toml", "--live"], True,
     "production form with --live"),
    (["python", "run.py", "--dry", "--config", "config.toml"], True,
     "--dry before --config"),
    (["C:\\Python\\python.exe", "C:\\bot\\run.py", "--dry"], True,
     "Windows path-prefixed run.py"),
    (["/usr/bin/python3", "/home/u/PancakeBot/run.py", "--live"], True,
     "POSIX path-prefixed run.py"),
    (["python", "run.py", "--config", "/etc/cfg.toml", "--dry", "--fresh"], True,
     "extra args after --dry"),

    # ❌ should NOT match
    ([], False, "empty cmdline"),
    (["python", "run.py"], False, "no mode flag"),
    (["python", "run.py", "--backtest"], False, "non-dry/live mode"),
    (["python", "run.py", "--sync"], False, "sync mode"),
    (["python", "some_other_script.py", "--config", "dry-run.toml"], False,
     "no run.py token; --dry only as substring of unrelated arg"),
    (["python", "other.py", "--dry"], False, "--dry but no run.py"),
    (["python", "run.py", "--config", "live_setup.toml"], False,
     "run.py present but --live only as substring of config filename"),
    (["bash", "-c", "echo run.py --dry"], False,
     "shell exec with run.py + --dry only inside a quoted echo arg"),
])
def test_cmdline_is_bot(cmdline, expected, reason):
    assert _cmdline_is_bot(cmdline) is expected, (
        f"_cmdline_is_bot({cmdline!r}) returned {not expected}; "
        f"expected {expected} ({reason})"
    )


# ---------------------------------------------------------------------------
# find_duplicate_bots: integration with mocked psutil
# ---------------------------------------------------------------------------


class _FakeProc:
    """Stand-in for psutil.Process.info, exposes the .info dict shape that
    process_iter yields when called with attrs=[...]."""
    def __init__(self, *, pid: int, cmdline: list[str], create_time: float):
        self.info = {
            "pid": pid,
            "cmdline": cmdline,
            "create_time": create_time,
        }


def _patch_process_iter(procs: list[_FakeProc]):
    """Patch psutil.process_iter to yield the fake procs.

    ``find_duplicate_bots`` does ``import psutil`` lazily inside the
    function body, so we have to inject the mock into ``sys.modules``
    rather than patching it as a module attribute (the import statement
    binds the name in function scope, ignoring module attributes)."""
    fake_psutil = mock.MagicMock(
        process_iter=mock.MagicMock(return_value=iter(procs)),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
    )
    return mock.patch.dict(sys.modules, {"psutil": fake_psutil})


def test_find_duplicate_bots_excludes_self():
    """The current process's own PID must never appear in the result."""
    self_pid = os.getpid()
    procs = [
        _FakeProc(pid=self_pid, cmdline=["py", "run.py", "--config", "c.toml", "--dry"], create_time=1.0),
    ]
    with _patch_process_iter(procs):
        assert find_duplicate_bots() == []


def test_find_duplicate_bots_catches_production_form():
    """The exact cmdline shape that escaped the substring bug must be caught:
    py.exe run.py --config config.toml --dry"""
    procs = [
        _FakeProc(pid=99999, create_time=12345.0,
                  cmdline=["py.exe", "run.py", "--config", "config.toml", "--dry"]),
    ]
    with _patch_process_iter(procs):
        result = find_duplicate_bots()
    assert len(result) == 1
    assert result[0]["pid"] == 99999
    assert result[0]["started_at"] == 12345.0
    assert "run.py" in result[0]["cmdline"]
    assert "--dry" in result[0]["cmdline"]


def test_find_duplicate_bots_catches_path_prefixed_run_py():
    """Path-prefixed run.py (e.g. ``/home/u/bot/run.py``) must be caught."""
    procs = [
        _FakeProc(pid=42, create_time=100.0,
                  cmdline=["/usr/bin/python3", "/home/u/PancakeBot/run.py",
                           "--config", "/etc/cfg.toml", "--live"]),
    ]
    with _patch_process_iter(procs):
        result = find_duplicate_bots()
    assert len(result) == 1
    assert result[0]["pid"] == 42


def test_find_duplicate_bots_does_not_match_dry_substring_in_config():
    """A bot started with ``--config dry-run.toml`` (and NOT --dry/--live)
    must not be flagged."""
    procs = [
        _FakeProc(pid=7, create_time=0.0,
                  cmdline=["python", "run.py", "--config", "dry-run.toml"]),
    ]
    with _patch_process_iter(procs):
        assert find_duplicate_bots() == []


def test_find_duplicate_bots_does_not_match_other_scripts():
    """A non-bot Python process must not be flagged."""
    procs = [
        _FakeProc(pid=1, create_time=0.0,
                  cmdline=["python", "some_other_script.py", "--dry"]),
        _FakeProc(pid=2, create_time=0.0,
                  cmdline=["python", "research/sweep.py", "--mode=dry"]),
    ]
    with _patch_process_iter(procs):
        assert find_duplicate_bots() == []


def test_find_duplicate_bots_flags_dry_when_self_is_live():
    """Both --dry and --live count as duplicates regardless of the caller's
    own mode (they share state-file paths in var/)."""
    self_pid = os.getpid()
    procs = [
        _FakeProc(pid=self_pid, create_time=0.0,
                  cmdline=["python", "run.py", "--live"]),
        _FakeProc(pid=2024, create_time=50.0,
                  cmdline=["py", "run.py", "--config", "c.toml", "--dry"]),
    ]
    with _patch_process_iter(procs):
        result = find_duplicate_bots()
    assert len(result) == 1
    assert result[0]["pid"] == 2024


def test_find_duplicate_bots_returns_multiple_dupes():
    """All matching processes are reported, not just the first."""
    procs = [
        _FakeProc(pid=11, create_time=1.0,
                  cmdline=["python", "run.py", "--config", "a.toml", "--dry"]),
        _FakeProc(pid=22, create_time=2.0,
                  cmdline=["python", "run.py", "--live"]),
        _FakeProc(pid=33, create_time=3.0,
                  cmdline=["python", "other.py"]),
    ]
    with _patch_process_iter(procs):
        result = find_duplicate_bots()
    pids = sorted(r["pid"] for r in result)
    assert pids == [11, 22]


def test_find_duplicate_bots_skips_empty_cmdline():
    """Processes with no cmdline (zombies, kernel threads on Linux) skip cleanly."""
    procs = [
        _FakeProc(pid=100, create_time=0.0, cmdline=[]),
        _FakeProc(pid=101, create_time=0.0, cmdline=None),  # type: ignore[arg-type]
        _FakeProc(pid=102, create_time=5.0,
                  cmdline=["py", "run.py", "--config", "c.toml", "--dry"]),
    ]
    with _patch_process_iter(procs):
        result = find_duplicate_bots()
    assert [r["pid"] for r in result] == [102]


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def _run_all() -> int:
    """Manual runner with parametrized tests unrolled."""
    failed = 0
    total = 0

    def run(name, fn, *args):
        nonlocal failed, total
        total += 1
        try:
            fn(*args)
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")

    # Unroll _is_run_py_token cases
    is_run_py_cases = [
        ("run.py", True), ("./run.py", True),
        ("/home/user/PancakeBot/run.py", True),
        ("C:\\Users\\zking\\Documents\\GitHub\\PancakeBot\\run.py", True),
        ("C:/Users/zking/Documents/GitHub/PancakeBot/run.py", True),
        ("python", False), ("py.exe", False), ("--dry", False),
        ("notrun.py", False), ("run.pyc", False), ("config.toml", False),
        ("", False),
    ]
    for tok, exp in is_run_py_cases:
        run(f"is_run_py_token[{tok!r}]", test_is_run_py_token, tok, exp)

    # Unroll _cmdline_is_bot cases
    cmdline_cases = [
        (["python", "run.py", "--dry"], True, "bare"),
        (["py", "run.py", "--config", "c.toml", "--dry"], True, "prod form"),
        (["python", "run.py", "--dry", "--config", "c.toml"], True, "dry first"),
        (["C:\\Python\\python.exe", "C:\\bot\\run.py", "--dry"], True, "win path"),
        ([], False, "empty"),
        (["python", "run.py"], False, "no mode"),
        (["python", "run.py", "--backtest"], False, "backtest"),
        (["python", "some.py", "--config", "dry.toml"], False, "no run.py + substring trap"),
        (["python", "run.py", "--config", "live.toml"], False, "live substring trap"),
    ]
    for cmd, exp, reason in cmdline_cases:
        run(f"cmdline_is_bot[{reason}]", test_cmdline_is_bot, cmd, exp, reason)

    # Integration tests
    for fn in (
        test_find_duplicate_bots_excludes_self,
        test_find_duplicate_bots_catches_production_form,
        test_find_duplicate_bots_catches_path_prefixed_run_py,
        test_find_duplicate_bots_does_not_match_dry_substring_in_config,
        test_find_duplicate_bots_does_not_match_other_scripts,
        test_find_duplicate_bots_flags_dry_when_self_is_live,
        test_find_duplicate_bots_returns_multiple_dupes,
        test_find_duplicate_bots_skips_empty_cmdline,
    ):
        run(fn.__name__, fn)

    print(f"\n{total - failed}/{total} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
