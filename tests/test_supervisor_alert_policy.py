"""Tests for supervisor's 2026-05-16 alert-policy refinement.

User policy: a supervisor-initiated restart that succeeded cleanly is the
supervisor doing its job — not a problem worth a Discord ping.

Verified behaviors:

1. Clean restart (status=DOWN, action=RESTARTED, no escalation) →
   no Discord call, supervisor.log shows ``alert=SUPPRESSED_ROUTINE_RESTART``.

2. SPAWN_FAILED → Discord IS called via the regular mode-channel path
   with escalation="SPAWN_FAILED"; message body includes the spawn-error
   detail.

3. Escalations (SUPPRESSED_FAST_CRASHLOOP, SLOW_CRASHLOOP_WARNING) still
   alert — regression guard, since the suppression gate is keyed on
   ``action_taken == "RESTARTED" and escalation is None``.

Run:
    python -m pytest tests/test_supervisor_alert_policy.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

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
    """Build a self-contained artifacts dict pointing at tmp_path."""
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


def _run_main(monkeypatch, *, classify_return, do_restart_return, extra_argv=()):
    """Run supervisor.main() with classify+do_restart stubbed.

    Captures _maybe_send_discord call args and the final supervisor.log line.
    """
    captured: dict = {"discord_calls": [], "log_lines": []}

    def fake_classify(*a, **kw):
        return classify_return

    def fake_do_restart(*a, **kw):
        return do_restart_return

    def fake_maybe_send_discord(**kw):
        captured["discord_calls"].append(kw)
        return "SENT"

    real_write = supervisor._write_supervisor_line
    def capture_write(log_path, mode, status, fields):
        captured["log_lines"].append({"status": status, "fields": dict(fields)})
        real_write(log_path, mode, status, fields)

    monkeypatch.setattr(supervisor, "_classify", fake_classify)
    monkeypatch.setattr(supervisor, "_do_restart", fake_do_restart)
    monkeypatch.setattr(supervisor, "_maybe_send_discord", fake_maybe_send_discord)
    monkeypatch.setattr(supervisor, "_write_supervisor_line", capture_write)

    argv = ["--mode", "dry", "--restart", "--alert"] + list(extra_argv)
    rc = supervisor.main(argv)
    captured["exit_code"] = rc
    return captured


def test_clean_restart_suppresses_discord(monkeypatch, tmp_path):
    """STATUS=DOWN + action=RESTARTED + no escalation -> NO Discord call.

    The supervisor just did its job. Log line should show
    alert=SUPPRESSED_ROUTINE_RESTART for operator visibility.
    """
    monkeypatch.setattr(
        supervisor, "_artifacts_for_mode", lambda mode: _stub_artifacts(tmp_path),
    )
    out = _run_main(
        monkeypatch,
        classify_return=("DOWN", {}),
        do_restart_return={"action": "RESTARTED", "new_pid": 99999},
    )

    assert out["discord_calls"] == [], (
        f"Expected no Discord call for clean restart; got {out['discord_calls']}"
    )
    assert len(out["log_lines"]) == 1
    fields = out["log_lines"][0]["fields"]
    assert fields.get("action") == "RESTARTED"
    assert fields.get("alert") == "SUPPRESSED_ROUTINE_RESTART"
    assert fields.get("new_pid") == 99999


def test_spawn_failed_alerts_to_mode_channel(monkeypatch, tmp_path):
    """action=SPAWN_FAILED -> Discord call with escalation=SPAWN_FAILED.

    Webhook resolution falls through to the mode-specific channel (dry/live)
    via _resolve_webhook_env (only UNINSTRUMENTED routes to general). The
    spawn_error detail must be threaded into fields so the message body
    can surface what went wrong.
    """
    monkeypatch.setattr(
        supervisor, "_artifacts_for_mode", lambda mode: _stub_artifacts(tmp_path),
    )
    out = _run_main(
        monkeypatch,
        classify_return=("DOWN", {}),
        do_restart_return={
            "action": "SPAWN_FAILED",
            "detail": "OSError: [Errno 2] No such file or directory: 'run.py'",
        },
    )

    assert len(out["discord_calls"]) == 1, (
        f"Expected 1 Discord call for SPAWN_FAILED; got {out['discord_calls']}"
    )
    call = out["discord_calls"][0]
    assert call["status"] == "DOWN"
    assert call["escalation"] == "SPAWN_FAILED"
    assert call["mode"] == "dry"
    # spawn_error must be in fields so _build_discord_message renders it.
    assert "spawn_error" in call["fields"]
    assert "OSError" in call["fields"]["spawn_error"]

    # Webhook routing: SPAWN_FAILED with status=DOWN routes to mode channel,
    # not the general channel (UNINSTRUMENTED is the only one that goes general).
    assert supervisor._resolve_webhook_env("dry", "DOWN") == (
        "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL"
    )


def test_slow_crashloop_warning_still_alerts(monkeypatch, tmp_path):
    """Regression: SLOW_CRASHLOOP_WARNING must still fire a Discord alert.

    The suppression gate is keyed on (action_taken == "RESTARTED" and
    escalation is None). SLOW_CRASHLOOP_WARNING has action="SLOW_CRASHLOOP_WARNING"
    (NOT "RESTARTED"), so the gate doesn't fire — alert proceeds.
    """
    monkeypatch.setattr(
        supervisor, "_artifacts_for_mode", lambda mode: _stub_artifacts(tmp_path),
    )
    out = _run_main(
        monkeypatch,
        classify_return=("DOWN", {}),
        do_restart_return={
            "action": "SLOW_CRASHLOOP_WARNING",
            "new_pid": 12345,
            "fast_count": 2,
            "slow_count": 9,
            "escalated": True,
        },
    )
    assert len(out["discord_calls"]) == 1
    call = out["discord_calls"][0]
    assert call["escalation"] == "SLOW_CRASHLOOP_WARNING"


def test_spawn_failed_message_body_includes_spawn_error(monkeypatch, tmp_path):
    """The _build_discord_message body for SPAWN_FAILED must include the
    spawn_error detail and the manual-intervention note.
    """
    art = _stub_artifacts(tmp_path)
    msg = supervisor._build_discord_message(
        mode="dry",
        status="DOWN",
        fields={"spawn_error": "OSError: cannot bind port 8080"},
        art=art,
        escalation="SPAWN_FAILED",
    )
    assert "SPAWN_FAILED" in msg
    assert "OSError: cannot bind port 8080" in msg
    assert "manual intervention required" in msg
