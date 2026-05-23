"""Tests for the new Windows-Service-based supervisor architecture.

Exercises the pure-logic surface of ``pancakebot/service/``:
  - ``supervision.classify_state`` for each state (UP / STARTING / STALE /
    CRASHED / DOWN / UNINSTRUMENTED)
  - ``supervision`` restart-history helpers (read / write / prune / count)
  - ``notifications.resolve_webhook_env`` channel routing per kind
  - ``notifications.rate_limit_ok`` cooldown enforcement
  - ``notifications.build_message`` basic structure

SCM-interaction code (win32service.OpenSCManager / ControlService etc.)
is intentionally NOT covered here — it's effectively e2e-only and
requires admin + an installed service. The service base class loads
``win32serviceutil`` at module import so we skip those imports in this
test file.

Run:
    python -m pytest tests/test_service_lifecycle.py -v
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from pancakebot.service import notifications, supervision  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mode_tree(tmp: Path, mode: str) -> dict[str, Path]:
    """Build a fake var/<mode>/ tree mirroring ``artifacts_for_mode`` paths.

    Returns the same dict shape as ``artifacts_for_mode`` but rooted under
    ``tmp`` instead of the repo root.
    """
    mode_dir = tmp / "var" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "logs").mkdir(exist_ok=True)
    return {
        "heartbeat": mode_dir / "heartbeat.json",
        "pid": mode_dir / "bot.pid",
        "crash": mode_dir / "crash.json",
        "supervisor_log": mode_dir / "supervisor.log",
        "trades": mode_dir / "trades.csv",
        "last_alert": mode_dir / "last_alert.json",
        "restart_history": mode_dir / "restart_history.jsonl",
        "logs_dir": mode_dir / "logs",
    }


def _backdate_mtime(path: Path, age_s: float) -> None:
    past = time.time() - age_s
    os.utime(str(path), (past, past))


def _write_heartbeat(path: Path, *, pid: int, age_s: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "pid": pid, "ts_wall": time.time(), "last_epoch": 100,
        "bankroll_bnb": 1.5, "iteration_count": 10,
    }), encoding="utf-8")
    if age_s > 0:
        _backdate_mtime(path, age_s)


def _write_pid_file(path: Path, pid: int, *, age_s: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")
    if age_s > 0:
        _backdate_mtime(path, age_s)


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
# supervision.classify_state — state-by-state coverage
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_artifacts(monkeypatch, tmp_path):
    """Redirect ``artifacts_for_mode`` to a tmp tree + stub pid-liveness."""
    art_live = _make_mode_tree(tmp_path, "live")
    art_dry = _make_mode_tree(tmp_path, "dry")
    monkeypatch.setattr(
        supervision, "artifacts_for_mode",
        lambda mode: art_live if mode == "live" else art_dry,
    )
    # Default: no pid is "our bot." Tests opt-in by overriding.
    monkeypatch.setattr(supervision, "_pid_is_our_bot", lambda pid, mode: False)
    monkeypatch.setattr(supervision, "find_legacy_bot_pid", lambda mode: None)
    return {"live": art_live, "dry": art_dry}


def test_classify_down_when_no_artifacts(fake_artifacts):
    status, fields = supervision.classify_state("live")
    assert status == "DOWN"
    assert fields == {}


def test_classify_up_with_fresh_heartbeat(fake_artifacts, monkeypatch):
    art = fake_artifacts["live"]
    _write_heartbeat(art["heartbeat"], pid=4242)
    monkeypatch.setattr(supervision, "_pid_is_our_bot", lambda pid, mode: pid == 4242)
    status, fields = supervision.classify_state("live")
    assert status == "UP", f"unexpected: {status} {fields}"
    assert fields["pid"] == 4242


def test_classify_starting_with_fresh_pid_and_no_heartbeat(fake_artifacts, monkeypatch):
    art = fake_artifacts["live"]
    _write_pid_file(art["pid"], 7777)
    monkeypatch.setattr(supervision, "_pid_is_our_bot", lambda pid, mode: pid == 7777)
    status, fields = supervision.classify_state("live")
    assert status == "STARTING"
    assert fields["pid"] == 7777


def test_classify_stale_when_heartbeat_old_and_past_grace(fake_artifacts, monkeypatch):
    art = fake_artifacts["live"]
    # Heartbeat 60s old (>5s stale threshold), pid alive, pid file 120s old (>90s grace)
    _write_heartbeat(art["heartbeat"], pid=9999, age_s=60)
    _write_pid_file(art["pid"], 9999, age_s=120)
    monkeypatch.setattr(supervision, "_pid_is_our_bot", lambda pid, mode: pid == 9999)
    status, _fields = supervision.classify_state("live")
    assert status == "STALE"


def test_classify_crashed_with_crash_json(fake_artifacts):
    art = fake_artifacts["live"]
    _write_crash(art["crash"], exc_type="ValueError")
    status, fields = supervision.classify_state("live")
    assert status == "CRASHED"
    assert fields["exc"] == "ValueError"


def test_classify_uninstrumented_with_legacy_bot(fake_artifacts, monkeypatch):
    monkeypatch.setattr(supervision, "find_legacy_bot_pid", lambda mode: 12345)
    status, fields = supervision.classify_state("live")
    assert status == "UNINSTRUMENTED"
    assert fields["pid"] == 12345
    assert fields["note"] == "legacy_no_heartbeat"


# ---------------------------------------------------------------------------
# Restart-history helpers
# ---------------------------------------------------------------------------

def test_restart_history_roundtrip(tmp_path):
    path = tmp_path / "restart_history.jsonl"
    entries = [
        {"ts_wall": 1000.0, "trigger": "DOWN", "new_pid": 1},
        {"ts_wall": 2000.0, "trigger": "CRASHED", "new_pid": 2},
    ]
    supervision.write_restart_history(path, entries)
    loaded = supervision.read_restart_history(path)
    assert loaded == entries


def test_restart_history_prune_drops_old_entries(tmp_path):
    entries = [
        {"ts_wall": 1000.0, "trigger": "DOWN"},
        {"ts_wall": 5000.0, "trigger": "CRASHED"},
    ]
    # now=6000, window=2000 → keep only entries with ts_wall >= 4000
    kept = supervision.prune_history(entries, now=6000.0, window_s=2000.0)
    assert len(kept) == 1
    assert kept[0]["ts_wall"] == 5000.0


def test_restart_history_count_within(tmp_path):
    entries = [
        {"ts_wall": 100.0},
        {"ts_wall": 500.0},
        {"ts_wall": 900.0},
    ]
    # now=1000, window=600 → count entries with ts_wall >= 400
    n = supervision.count_within(entries, now=1000.0, window_s=600.0)
    assert n == 2


def test_restart_history_drops_malformed_lines(tmp_path):
    path = tmp_path / "rh.jsonl"
    path.write_text(
        '{"ts_wall":1.0}\n'
        'not json at all\n'
        '{"ts_wall":2.0}\n'
        '\n',
        encoding="utf-8",
    )
    entries = supervision.read_restart_history(path)
    assert len(entries) == 2
    assert entries[0]["ts_wall"] == 1.0
    assert entries[1]["ts_wall"] == 2.0


# ---------------------------------------------------------------------------
# notifications.resolve_webhook_env — channel routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,kind,expected", [
    ("live", "STALE", notifications.LIVE_WEBHOOK_ENV),
    ("dry",  "STALE", notifications.DRY_WEBHOOK_ENV),
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
# notifications.rate_limit_ok — cooldown enforcement
# ---------------------------------------------------------------------------

def test_rate_limit_first_call_passes(tmp_path):
    path = tmp_path / "last_alert.json"
    assert notifications.rate_limit_ok(path, "STALE", now=1000.0, cooldown_s=60.0)
    # Second immediate call is suppressed
    assert not notifications.rate_limit_ok(path, "STALE", now=1010.0, cooldown_s=60.0)


def test_rate_limit_clears_after_cooldown(tmp_path):
    path = tmp_path / "last_alert.json"
    assert notifications.rate_limit_ok(path, "DOWN", now=1000.0, cooldown_s=60.0)
    assert not notifications.rate_limit_ok(path, "DOWN", now=1030.0, cooldown_s=60.0)
    assert notifications.rate_limit_ok(path, "DOWN", now=1061.0, cooldown_s=60.0)


def test_rate_limit_per_key_independent(tmp_path):
    path = tmp_path / "last_alert.json"
    assert notifications.rate_limit_ok(path, "STALE", now=1000.0, cooldown_s=60.0)
    # Different kind: independent bucket
    assert notifications.rate_limit_ok(path, "CRASHED", now=1000.0, cooldown_s=60.0)
    # Same kind, still in cooldown
    assert not notifications.rate_limit_ok(path, "STALE", now=1010.0, cooldown_s=60.0)


# ---------------------------------------------------------------------------
# notifications.build_message — sanity checks
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


# ---------------------------------------------------------------------------
# notifications.notify — DISABLED path (no HTTP made)
# ---------------------------------------------------------------------------

def test_notify_disabled_when_env_var_unset(tmp_path, monkeypatch):
    """When the resolved webhook env var is unset, notify returns DISABLED
    without attempting any HTTP call."""
    monkeypatch.delenv(notifications.LIVE_WEBHOOK_ENV, raising=False)
    art = _make_mode_tree(tmp_path, "live")
    outcome = notifications.notify(mode="live", kind="STALE", art=art)
    assert outcome == "DISABLED"


# ---------------------------------------------------------------------------
# Module-import smoke test for the Win32-dependent code path
# ---------------------------------------------------------------------------

def test_common_module_imports():
    """``pancakebot.service.common`` imports pywin32; verify it loads cleanly.

    We don't exercise the ServiceFramework class here — that requires
    SCM context — but a successful import catches typos / wrong symbol
    references before install.
    """
    import importlib
    mod = importlib.import_module("pancakebot.service.common")
    assert hasattr(mod, "_PancakeBotServiceBase")
    assert hasattr(mod, "_query_service_state")
    assert hasattr(mod, "_stop_service_and_wait")


def test_live_and_dry_service_classes_importable():
    from pancakebot.service.live_service import PancakeBotLiveService
    from pancakebot.service.dry_service import PancakeBotDryService
    assert PancakeBotLiveService._MODE == "live"
    assert PancakeBotLiveService._OTHER_SERVICE == "PancakeBotDry"
    assert PancakeBotDryService._MODE == "dry"
    assert PancakeBotDryService._OTHER_SERVICE == "PancakeBotLive"
