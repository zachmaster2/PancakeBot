"""Tests for pancakebot.ops.notifications — the Discord alert executor.

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

from pancakebot.ops import notifications  # noqa: E402


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


# ---------------------------------------------------------------------------
# Deliverability: a property of the notification SYSTEM, not of any alert
# ---------------------------------------------------------------------------

def test_every_kind_is_deliverable_within_discords_content_cap():
    """Discord REJECTS a webhook content over 2,000 chars with HTTP 400 —
    it does not truncate. So an over-long alert is a SILENT alert, and a
    re-alert loop regenerates the same oversized body forever.

    ENDPOINT_MOVE_TRIGGERED shipped at 2,990-3,310 chars and was
    permanently undeliverable while its own RECOVERED message (708) sent
    fine — the alarm would have produced exactly the outcome its RECOVERED
    text exists to warn about. The whole suite was green because every
    test asserted substrings were PRESENT and none asserted the message
    was SENDABLE.
    """
    from pancakebot.ops.notifications import (
        _DISCORD_CONTENT_LIMIT, _SEVERITY_BY_KIND, _chunk_for_discord,
        build_message,
    )
    # Exercised across payload shapes, not just the empty one: an
    # empty-fields fixture cannot produce the growth this invariant exists
    # to catch. A CRASHED alert carries a traceback tail, so the long and
    # newline-dense cases are realistic, not hypothetical.
    payloads = {
        "empty": "",
        "typical_alarm_detail": "x" * 151,
        "long_single_line": "y" * 4096,
        "traceback_like": chr(10).join(f"  File line {i}" for i in range(400)),
    }
    for kind in sorted(_SEVERITY_BY_KIND):
        for name, detail in payloads.items():
            msg = build_message(mode="live", kind=kind, fields={},
                                detail=detail)
            chunks = _chunk_for_discord(msg)
            assert chunks, (kind, name)
            for i, chunk in enumerate(chunks, 1):
                assert len(chunk) <= _DISCORD_CONTENT_LIMIT, (
                    f"{kind} [{name}] part {i}/{len(chunks)} is {len(chunk)} "
                    f"chars, over the {_DISCORD_CONTENT_LIMIT} cap — Discord "
                    f"will 400 it")


def test_routine_alerts_stay_in_a_single_message():
    """Only the endpoint-move payload is allowed to span parts. If a
    routine alert starts chunking, it has grown a payload that belongs in
    an artifact, not on a phone — and adding it to this allowlist should
    be a deliberate edit, not something that happens quietly.

    Bounded at a REALISTIC detail rather than an arbitrary one: the alarm
    path's detail runs ~151 chars and lifecycle's is short, so 400 is
    comfortably above production without asserting something that cannot
    hold. A 4 KB detail chunks every kind, which is the transport working,
    not a violation — that case is covered by the deliverability test
    above.
    """
    from pancakebot.ops.notifications import (
        _SEVERITY_BY_KIND, _chunk_for_discord, build_message,
    )
    for detail in ("", "d" * 400):
        multi = {k for k in _SEVERITY_BY_KIND
                 if len(_chunk_for_discord(build_message(
                     mode="live", kind=k, fields={}, detail=detail))) > 1}
        assert multi <= {"ENDPOINT_MOVE_TRIGGERED"}, (detail[:8], multi)


def test_chunking_splits_on_line_boundaries_and_loses_nothing():
    from pancakebot.ops.notifications import _chunk_for_discord
    lines = [f"line {i} " + "x" * 80 for i in range(60)]
    msg = "\n".join(lines)
    chunks = _chunk_for_discord(msg, limit=500)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 500
    rebuilt = "\n".join(c.split(") ", 1)[1] for c in chunks)
    assert rebuilt == msg


def test_a_single_unsplittable_line_is_hard_split_rather_than_dropped():
    from pancakebot.ops.notifications import _chunk_for_discord
    msg = "y" * 5000
    chunks = _chunk_for_discord(msg, limit=500)
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(c.split(") ", 1)[1] for c in chunks) == msg


def test_short_messages_are_untouched_by_chunking():
    """The overwhelming majority of alerts must be byte-identical to
    before: no part markers, no reflow."""
    from pancakebot.ops.notifications import _chunk_for_discord, build_message
    msg = build_message(mode="live", kind="STARTED", fields={})
    assert _chunk_for_discord(msg) == [msg]
