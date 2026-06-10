"""Tests for pancakebot.service.notifications — the Discord alert executor.

Channel routing per kind, rate-limit cooldown enforcement, message building
(severity tags, [MODE] prefix, crash-artifact rendering), and the
DISABLED / SEND_FAILED notify() outcomes. No HTTP.

Run:
    python -m pytest tests/test_notifications.py -v
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.service import notifications  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mode_tree(tmp: Path, mode: str) -> dict[str, Path]:
    """Fake var/<mode>/ tree with the alert-relevant artifact paths."""
    mode_dir = tmp / "var" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "logs").mkdir(exist_ok=True)
    return {
        "crash": mode_dir / "crash.json",
        "last_alert": mode_dir / "last_alert.json",
        "restart_history": mode_dir / "restart_history.jsonl",
        "logs_dir": mode_dir / "logs",
    }


def _write_crash(path: Path, *, exc_type: str = "FakeError") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ts_wall": time.time(),
        "exc_type": exc_type,
        "exc_repr": f"{exc_type}('boom')",
        "traceback_str": "Traceback (most recent call last):\n  ...\n",
        "last_epoch": 100,
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# resolve_webhook_env — channel routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,kind,expected", [
    ("live", "CRASHED", notifications.LIVE_WEBHOOK_ENV),
    ("dry",  "CRASHED", notifications.DRY_WEBHOOK_ENV),
    ("live", "UNINSTRUMENTED", notifications.GENERAL_WEBHOOK_ENV),
    ("dry",  "UNINSTRUMENTED", notifications.GENERAL_WEBHOOK_ENV),
    ("live", "STARTED", notifications.LIVE_WEBHOOK_ENV),
    ("live", "REBOOTED", notifications.LIVE_WEBHOOK_ENV),
    ("live", "STOPPED", notifications.LIVE_WEBHOOK_ENV),
    # MODE_TRANSITION always goes to live regardless of firing mode
    ("live", "MODE_TRANSITION", notifications.LIVE_WEBHOOK_ENV),
    ("dry",  "MODE_TRANSITION", notifications.LIVE_WEBHOOK_ENV),
    # MODE_TRANSITION_REFUSED always goes to dry (it's only the dry side
    # that ever fires this anyway, but we test the routing for completeness)
    ("dry",  "MODE_TRANSITION_REFUSED", notifications.DRY_WEBHOOK_ENV),
    ("live", "SERVICE_CRASHED", notifications.GENERAL_WEBHOOK_ENV),
])
def test_channel_routing(mode, kind, expected):
    assert notifications.resolve_webhook_env(mode, kind) == expected


# ---------------------------------------------------------------------------
# rate_limit_ok — cooldown enforcement
# ---------------------------------------------------------------------------

def test_rate_limit_first_call_passes(tmp_path):
    path = tmp_path / "last_alert.json"
    assert notifications.rate_limit_ok(path, "CRASHED", now=1000.0, cooldown_s=60.0)
    # Second immediate call is suppressed
    assert not notifications.rate_limit_ok(path, "CRASHED", now=1010.0, cooldown_s=60.0)


def test_rate_limit_clears_after_cooldown(tmp_path):
    path = tmp_path / "last_alert.json"
    assert notifications.rate_limit_ok(path, "DOWN", now=1000.0, cooldown_s=60.0)
    assert not notifications.rate_limit_ok(path, "DOWN", now=1030.0, cooldown_s=60.0)
    assert notifications.rate_limit_ok(path, "DOWN", now=1061.0, cooldown_s=60.0)


def test_rate_limit_per_key_independent(tmp_path):
    path = tmp_path / "last_alert.json"
    assert notifications.rate_limit_ok(path, "CRASHED", now=1000.0, cooldown_s=60.0)
    # Different kind: independent bucket
    assert notifications.rate_limit_ok(path, "DOWN", now=1000.0, cooldown_s=60.0)
    # Same kind, still in cooldown
    assert not notifications.rate_limit_ok(path, "CRASHED", now=1010.0, cooldown_s=60.0)


# ---------------------------------------------------------------------------
# build_message — sanity checks
# ---------------------------------------------------------------------------

def test_build_message_includes_kind_and_mode():
    msg = notifications.build_message(
        mode="live", kind="STARTED", fields={"pid": 9999},
    )
    assert "STARTED" in msg
    assert "PancakeBot-live" in msg
    assert "9999" in msg


def test_build_message_with_crash_artifact(tmp_path):
    art = _make_mode_tree(tmp_path, "live")
    _write_crash(art["crash"], exc_type="RuntimeError")
    msg = notifications.build_message(
        mode="live", kind="CRASHED",
        fields={"pid": 1, "last_epoch": 42},
        art=art,
    )
    assert "CRASHED" in msg
    assert "RuntimeError" in msg
    assert "traceback" in msg.lower() or "Traceback" in msg


def test_build_message_with_detail_string():
    msg = notifications.build_message(
        mode="live", kind="MODE_TRANSITION",
        detail="stopping PancakeBotDry to start live",
    )
    assert "MODE_TRANSITION" in msg
    assert "stopping PancakeBotDry" in msg


# D5: every lifecycle alert carries the [MODE] prefix (matches bet-alert channels)
def test_build_message_has_mode_prefix_live():
    msg = notifications.build_message(mode="live", kind="STARTED", fields={"pid": 1})
    assert msg.startswith("[LIVE] [INFO] **STARTED**")


def test_build_message_has_mode_prefix_dry():
    msg = notifications.build_message(mode="dry", kind="REBOOTED", fields={"pid": 1})
    assert msg.startswith("[DRY] [INFO] **REBOOTED**")


# D4: STOPPED is INFO when intentional, CRIT when unexpected
def test_stopped_intentional_is_info():
    msg = notifications.build_message(mode="dry", kind="STOPPED", fields={"intentional": True})
    assert msg.startswith("[DRY] [INFO] **STOPPED**")


def test_stopped_unintentional_is_crit():
    msg = notifications.build_message(mode="dry", kind="STOPPED", fields={"intentional": False})
    assert msg.startswith("[DRY] [CRIT] **STOPPED**")


def test_stopped_missing_intent_defaults_crit():
    # No 'intentional' field -> treated as unexpected -> CRIT (fail-safe).
    msg = notifications.build_message(mode="live", kind="STOPPED")
    assert msg.startswith("[LIVE] [CRIT] **STOPPED**")


# ---------------------------------------------------------------------------
# notify — DISABLED path (no HTTP made) + SEND_FAILED stderr guard
# ---------------------------------------------------------------------------

def test_notify_disabled_when_env_var_unset(tmp_path, monkeypatch):
    """When the resolved webhook env var is unset, notify returns DISABLED
    without attempting any HTTP call."""
    monkeypatch.delenv(notifications.LIVE_WEBHOOK_ENV, raising=False)
    art = _make_mode_tree(tmp_path, "live")
    outcome = notifications.notify(mode="live", kind="CRASHED", art=art)
    assert outcome == "DISABLED"


def test_notify_send_failed_does_not_raise_with_stderr_none(tmp_path, monkeypatch):
    """A notify() that hits SEND_FAILED must not crash even with a broken
    stderr — the diagnostic write can never escalate the failure (this
    guard caught a real 2026-05-23 incident where it took the alerting
    process down)."""
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setenv(notifications.LIVE_WEBHOOK_ENV, "https://invalid.example/webhook")
    # Make the HTTP call fail synchronously so we hit the SEND_FAILED path.
    monkeypatch.setattr(
        notifications, "_send_discord",
        lambda url, mode, msg: (False, "post_exception:simulated"),
    )
    art = _make_mode_tree(tmp_path, "live")
    outcome = notifications.notify(mode="live", kind="CRASHED", art=art)
    assert outcome == "SEND_FAILED"
