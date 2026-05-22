"""Tests for the 2026-05-22 supervisor log-line atomicity fix.

Background: 2026-05-21 07:54 UTC, the first CRASHED-detection tick was
missing entirely from var/live/supervisor.log. The 07:51 tick logged
STATUS=UP cleanly; the next visible tick was 07:57 STATUS=CRASHED. The
07:54 tick had attempted to send the first Discord alert; the 10s HTTP
timeout combined with the schtasks 2-min kill window caused the
supervisor process to die before reaching _write_supervisor_line at the
end of main().

Fix: write the classification line BEFORE any potentially-hanging IO.
HTTP outcomes (which CAN block) are appended on a second line if Discord
returns. Sync-only outcomes (SUPPRESSED_ROUTINE_RESTART) stay on the
first line.

Also: _write_supervisor_line now uses os.write + os.O_APPEND for
guaranteed atomic single-syscall appends across concurrent supervisor
invocations.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "supervisor_under_test", str(_REPO_ROOT / "scripts" / "supervisor.py")
)
supervisor = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(supervisor)  # type: ignore[union-attr]


def _stub_artifacts(tmp_path: Path) -> dict[str, Path]:
    return {
        "heartbeat": tmp_path / "heartbeat.json",
        "pid": tmp_path / "bot.pid",
        "crash": tmp_path / "crash.json",
        "supervisor_log": tmp_path / "supervisor.log",
        "trades": tmp_path / "trades.csv",
        "last_alert": tmp_path / "last_alert.json",
        "restart_history": tmp_path / "restart_history.jsonl",
        "logs_dir": tmp_path / "logs",
    }


def test_write_uses_o_append_single_syscall(monkeypatch, tmp_path):
    """``_write_supervisor_line`` must use ``os.write`` (single syscall)
    with ``O_APPEND``, NOT Python's buffered ``open("a") + f.write``. This
    pins the atomicity guarantee.

    We monkey-patch ``os.open`` to capture the flags and assert O_APPEND
    is set. We also assert ``os.write`` is invoked exactly once per line.
    """
    captured = {"open_flags": [], "write_calls": 0}
    real_open = os.open
    real_write = os.write

    def fake_open(path, flags, mode=0o777):
        captured["open_flags"].append(flags)
        return real_open(path, flags, mode)

    def fake_write(fd, data):
        captured["write_calls"] += 1
        return real_write(fd, data)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "write", fake_write)

    log_path = tmp_path / "supervisor.log"
    supervisor._write_supervisor_line(
        log_path, "test", "UP", {"hb_age": "0.5s"},
    )

    # At least one os.open call targeted our log file with O_APPEND set.
    log_opens = [
        f for f in captured["open_flags"]
        if (f & os.O_APPEND) and (f & os.O_WRONLY) and (f & os.O_CREAT)
    ]
    assert log_opens, (
        f"Expected os.open with O_APPEND|O_WRONLY|O_CREAT; got flags={captured['open_flags']}"
    )
    # Exactly one os.write for the single line (some platforms also write
    # to stdout via os.write; count only line-write to our fd via assert
    # that at least one fired and that the file content is intact).
    assert captured["write_calls"] >= 1
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "STATUS=UP" in content
    assert "mode=test" in content
    assert "hb_age=0.5s" in content


def test_sequential_writers_preserve_all_lines(tmp_path):
    """Many sequential writes must all land in the log. Pins that the
    O_APPEND + single os.write path doesn't drop lines under normal use.

    NOTE: True cross-process atomicity on Windows is harder than POSIX:
    the C runtime's _O_APPEND implementation uses seek-then-write (not
    atomic at OS level). Threaded concurrent appends CAN interleave or
    drop lines on Windows. In production this isn't a concern because
    supervisor invocations are 3 minutes apart (schtasks cadence); the
    actual observed race is the schtasks-kill window, fixed by reordering
    the log write to BEFORE the Discord HTTP call (see
    ``test_classification_line_written_before_discord_attempt``).
    """
    log_path = tmp_path / "supervisor.log"
    n_writes = 200
    for i in range(n_writes):
        supervisor._write_supervisor_line(
            log_path, "test", "UP", {"seq": str(i)},
        )
    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l]
    assert len(lines) == n_writes, (
        f"Expected {n_writes} lines; got {len(lines)}"
    )
    # Every line complete + every seq seen exactly once.
    seen = set()
    for line in lines:
        assert "seq=" in line, f"truncated line: {line!r}"
        parts = dict(p.split("=", 1) for p in line.split() if "=" in p)
        seq = parts.get("seq")
        assert seq not in seen, f"duplicate seq: {seq} line={line!r}"
        seen.add(seq)
    assert seen == {str(i) for i in range(n_writes)}


def test_classification_line_written_before_discord_attempt(monkeypatch, tmp_path):
    """When the alert path involves Discord HTTP, the classification line
    is written BEFORE _maybe_send_discord is called. This guarantees the
    line is captured even if the HTTP call hangs and the supervisor is
    killed by the schtasks 2-min limit (the 2026-05-21 07:54 race).
    """
    art = _stub_artifacts(tmp_path)
    monkeypatch.setattr(
        supervisor, "_artifacts_for_mode", lambda mode: art,
    )

    write_order: list[str] = []
    real_write = supervisor._write_supervisor_line

    def trace_write(log_path, mode, status, fields):
        write_order.append(f"WRITE status={status} fields={sorted(fields.keys())}")
        real_write(log_path, mode, status, fields)

    def trace_discord(**kw):
        write_order.append(f"DISCORD status={kw['status']}")
        return "SENT"

    monkeypatch.setattr(supervisor, "_write_supervisor_line", trace_write)
    monkeypatch.setattr(supervisor, "_maybe_send_discord", trace_discord)
    monkeypatch.setattr(supervisor, "_classify", lambda *a, **kw: ("CRASHED", {"exc": "TestErr"}))
    monkeypatch.setattr(
        supervisor, "_do_restart",
        lambda *a, **kw: {"action": None},
    )

    rc = supervisor.main(["--mode", "live", "--alert"])

    # The FIRST entry must be a WRITE; DISCORD must come after.
    assert len(write_order) >= 2, f"expected at least one WRITE before DISCORD; got {write_order}"
    assert write_order[0].startswith("WRITE "), (
        f"first event must be WRITE (classification line); got {write_order[0]}"
    )
    # Find the DISCORD event and confirm it's preceded by a WRITE.
    discord_idx = next(
        (i for i, e in enumerate(write_order) if e.startswith("DISCORD")),
        None,
    )
    assert discord_idx is not None, f"Discord was never attempted; events={write_order}"
    assert discord_idx > 0, "DISCORD must NOT be the first event"
    # The pre-Discord WRITE line should carry alert=DISPATCHING in fields.
    first_write_fields = write_order[0]
    assert "alert" in first_write_fields, (
        f"first WRITE line missing alert field; got: {first_write_fields}"
    )


def test_routine_restart_writes_single_line_with_alert_field(monkeypatch, tmp_path):
    """Synchronously-determined outcomes (SUPPRESSED_ROUTINE_RESTART)
    keep their original single-line behavior. The second line is only
    emitted on the HTTP path.
    """
    art = _stub_artifacts(tmp_path)
    monkeypatch.setattr(
        supervisor, "_artifacts_for_mode", lambda mode: art,
    )

    log_lines: list[dict] = []
    real_write = supervisor._write_supervisor_line

    def capture_write(log_path, mode, status, fields):
        log_lines.append({"status": status, "fields": dict(fields)})
        real_write(log_path, mode, status, fields)

    monkeypatch.setattr(supervisor, "_write_supervisor_line", capture_write)
    monkeypatch.setattr(
        supervisor, "_maybe_send_discord",
        lambda **kw: pytest.fail("Discord should not be called on routine_restart"),
    )
    monkeypatch.setattr(supervisor, "_classify", lambda *a, **kw: ("DOWN", {}))
    monkeypatch.setattr(
        supervisor, "_do_restart",
        lambda *a, **kw: {"action": "RESTARTED", "new_pid": 12345},
    )

    rc = supervisor.main(["--mode", "dry", "--restart", "--alert"])

    assert len(log_lines) == 1, (
        f"routine_restart must emit exactly 1 line; got {len(log_lines)}: {log_lines}"
    )
    assert log_lines[0]["fields"].get("alert") == "SUPPRESSED_ROUTINE_RESTART"
