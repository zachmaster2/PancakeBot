"""Tests for pancakebot.ops.notify_lifecycle (Phase 3c-2 systemd-direct).

Covers the pure decision table (event + unit state -> alert kinds), the
two-tier crashloop thresholds, the systemctl-show/env state reading,
instance parsing, crash evidence, and the end-to-end main() wiring with
injected runner + captured notify calls. No systemd, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.ops import notify_lifecycle as nl  # noqa: E402


# -- instance parsing --------------------------------------------------------

def test_parse_instance_live_started():
    assert nl.parse_instance("pancakebot-live-started") == (
        "pancakebot-live", "live", "started",
    )


def test_parse_instance_dry_stopped():
    assert nl.parse_instance("pancakebot-dry-stopped") == (
        "pancakebot-dry", "dry", "stopped",
    )


def test_parse_instance_test_unit():
    """The validation harness uses a pancakebot-test unit."""
    assert nl.parse_instance("pancakebot-test-stopped") == (
        "pancakebot-test", "test", "stopped",
    )


@pytest.mark.parametrize("bad", ["", "started", "pancakebot-live", "x-y-restarted"])
def test_parse_instance_rejects_garbage(bad):
    with pytest.raises(ValueError):
        nl.parse_instance(bad)


# -- unit state reading ------------------------------------------------------

def test_query_unit_state_from_systemctl_show():
    out = "Result=exit-code\nExecMainStatus=1\nNRestarts=2\n"
    result, status, n = nl.query_unit_state(
        "pancakebot-live", run_cmd=lambda argv: out, env={},
    )
    assert (result, status, n) == ("exit-code", "1", 2)


def test_query_unit_state_env_vars_win():
    """Direct ExecStopPost invocation: $SERVICE_RESULT/$EXIT_STATUS beat the
    systemctl-show values (which may already reflect a later state)."""
    out = "Result=success\nExecMainStatus=0\nNRestarts=4\n"
    result, status, n = nl.query_unit_state(
        "pancakebot-live", run_cmd=lambda argv: out,
        env={"SERVICE_RESULT": "oom-kill", "EXIT_STATUS": "9"},
    )
    assert (result, status, n) == ("oom-kill", "9", 4)


def test_query_unit_state_survives_empty_output():
    result, status, n = nl.query_unit_state(
        "pancakebot-live", run_cmd=lambda argv: "", env={},
    )
    assert (result, status, n) == ("unknown", "?", 0)


# -- crash evidence (RECOVERY_AFTER_CRASH vs run.py's archive race) ----------

def test_crash_evidence_direct_crash_json(tmp_path):
    art = {"crash": tmp_path / "crash.json"}
    art["crash"].write_text("{}", encoding="utf-8")
    assert nl.crash_evidence(art, now=1000.0) is True


def test_crash_evidence_fresh_archive_counts(tmp_path):
    """run.py archives the lingering crash.json before the detached notify
    unit looks — a just-renamed (fresh st_ctime) archive IS crash evidence."""
    art = {"crash": tmp_path / "crash.json"}
    (tmp_path / "crash_archive_20260610-120000.json").write_text(
        "{}", encoding="utf-8")
    now = 1000.0
    assert nl.crash_evidence(art, now=now, ctime_of=lambda p: now - 5.0) is True


def test_crash_evidence_stale_archive_ignored(tmp_path):
    """Archives from past crashes (old ctime) must NOT flag a clean start."""
    art = {"crash": tmp_path / "crash.json"}
    (tmp_path / "crash_archive_20260601-000000.json").write_text(
        "{}", encoding="utf-8")
    now = 1000.0
    assert nl.crash_evidence(
        art, now=now, ctime_of=lambda p: now - 3600.0) is False


def test_crash_evidence_empty_dir(tmp_path):
    art = {"crash": tmp_path / "crash.json"}
    assert nl.crash_evidence(art, now=1000.0) is False


# -- decision table: started ------------------------------------------------

def _decide(**kw):
    defaults = dict(
        event="started", result="success", exit_status="0", n_restarts=0,
        crash_exists=False, uptime_s=86400.0, history=[], now=1_000_000.0,
    )
    defaults.update(kw)
    return nl.decide(**defaults)


def test_fresh_start_warm_system_is_started():
    alerts, hist = _decide()
    assert [k for k, _, _ in alerts] == ["STARTED"]
    assert hist is None


def test_fresh_start_after_reboot_is_rebooted():
    alerts, _ = _decide(uptime_s=120.0)
    assert [k for k, _, _ in alerts] == ["REBOOTED"]


def test_fresh_start_with_crash_json_is_recovery():
    alerts, _ = _decide(crash_exists=True)
    assert [k for k, _, _ in alerts] == ["RECOVERY_AFTER_CRASH"]


def test_reboot_beats_recovery():
    """First-run classification ordering: uptime check beats crash.json."""
    alerts, _ = _decide(crash_exists=True, uptime_s=60.0)
    assert [k for k, _, _ in alerts] == ["REBOOTED"]


def test_auto_restart_below_thresholds_is_silent_but_recorded():
    """First couple of systemd auto-restarts: CRASHED already fired on the
    stopped event — the started event just records history."""
    alerts, hist = _decide(n_restarts=1, history=[])
    assert alerts == []
    assert hist is not None and len(hist) == 1


def test_auto_restart_fast_crashloop_alerts_suppressed_kind():
    now = 1_000_000.0
    history = [{"ts": now - 300}, {"ts": now - 600}]  # 2 within 15 min
    alerts, hist = _decide(n_restarts=3, history=history, now=now)
    assert [k for k, _, _ in alerts] == ["SUPPRESSED_FAST_CRASHLOOP"]
    assert len(hist) == 3


def test_auto_restart_slow_crashloop_warns():
    """8 restarts spread across 24h but never 3 inside any 15 min."""
    now = 1_000_000.0
    history = [{"ts": now - (i + 1) * 3600.0} for i in range(7)]  # 7 hourly
    alerts, hist = _decide(n_restarts=8, history=history, now=now)
    assert [k for k, _, _ in alerts] == ["SLOW_CRASHLOOP_WARNING"]
    assert len(hist) == 8


def test_history_pruned_to_slow_window():
    now = 1_000_000.0
    history = [{"ts": now - nl._SLOW_RESTART_WINDOW_S - 100}]  # stale
    alerts, hist = _decide(n_restarts=1, history=history, now=now)
    assert len(hist) == 1 and hist[0]["ts"] == now  # stale entry dropped


# -- decision table: stopped --------------------------------------------------

def test_clean_stop_is_intentional_stopped():
    alerts, hist = _decide(event="stopped", result="success")
    assert len(alerts) == 1
    kind, fields, _ = alerts[0]
    assert kind == "STOPPED" and fields == {"intentional": True}
    assert hist is None


def test_exit_code_failure_is_crashed_with_cause():
    alerts, _ = _decide(
        event="stopped", result="exit-code", exit_status="1", n_restarts=0,
        journal_line="SystemExit: 1",
    )
    kind, _, detail = alerts[0]
    assert kind == "CRASHED"
    assert "exit-code" in detail and "exit_status=1" in detail
    assert "SystemExit: 1" in detail


def test_sigkill_is_crashed_with_signal_cause():
    alerts, _ = _decide(event="stopped", result="signal", exit_status="KILL")
    kind, _, detail = alerts[0]
    assert kind == "CRASHED" and "signal" in detail


def test_oom_kill_is_crashed_with_oom_cause():
    alerts, _ = _decide(event="stopped", result="oom-kill", exit_status="9")
    kind, _, detail = alerts[0]
    assert kind == "CRASHED" and "oom-kill" in detail


def test_start_limit_hit_is_terminal_fast_crashloop():
    alerts, _ = _decide(event="stopped", result="start-limit-hit")
    kind, _, detail = alerts[0]
    assert kind == "SUPPRESSED_FAST_CRASHLOOP"
    assert "start-limit-hit" in detail and "manual intervention" in detail


# -- end-to-end main() wiring -------------------------------------------------

def _fake_runner(show_out: str, journal_out: str = "boom\n"):
    def run_cmd(argv):
        if argv[0] == "systemctl":
            return show_out
        if argv[0] == "journalctl":
            return journal_out
        return ""
    return run_cmd


def test_main_crashed_end_to_end(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(
        nl.notifications, "notify",
        lambda **kw: (sent.append(kw), "SENT")[1],
    )
    rc = nl.main(
        ["pancakebot-live-stopped"],
        run_cmd=_fake_runner("Result=exit-code\nExecMainStatus=1\nNRestarts=0\n",
                             "Traceback ...\nInvariantError: boom\n"),
        env={}, now=1_000_000.0, repo_root=tmp_path,
    )
    assert rc == 0
    assert len(sent) == 1
    assert sent[0]["mode"] == "live"
    assert sent[0]["kind"] == "CRASHED"
    assert "InvariantError: boom" in sent[0]["detail"]


def test_main_records_restart_history(tmp_path, monkeypatch):
    monkeypatch.setattr(nl.notifications, "notify", lambda **kw: "SENT")
    rc = nl.main(
        ["pancakebot-dry-started"],
        run_cmd=_fake_runner("Result=exit-code\nExecMainStatus=1\nNRestarts=1\n"),
        env={}, now=1_000_000.0, repo_root=tmp_path,
    )
    assert rc == 0
    hist = nl.read_history(tmp_path / "var" / "dry" / "restart_history.jsonl")
    assert len(hist) == 1 and hist[0]["ts"] == 1_000_000.0


def test_main_bad_instance_exits_2(capsys):
    assert nl.main(["nonsense"]) == 2


def test_main_test_mode_routes_to_dry_channel(tmp_path, monkeypatch):
    """The pancakebot-test validation unit's alerts go to the DRY channel
    (the router only knows live/dry)."""
    sent = []
    monkeypatch.setattr(
        nl.notifications, "notify",
        lambda **kw: (sent.append(kw), "SENT")[1],
    )
    rc = nl.main(
        ["pancakebot-test-stopped"],
        run_cmd=_fake_runner("Result=oom-kill\nExecMainStatus=9\nNRestarts=0\n"),
        env={}, now=1.0, repo_root=tmp_path,
    )
    assert rc == 0
    assert sent[0]["mode"] == "dry"
    assert sent[0]["kind"] == "CRASHED"


def test_main_never_raises_on_internal_error(tmp_path, monkeypatch):
    """Alerting must not cascade: an exploding notify is swallowed."""
    def _boom(**kw):
        raise RuntimeError("discord exploded")
    monkeypatch.setattr(nl.notifications, "notify", _boom)
    rc = nl.main(
        ["pancakebot-live-stopped"],
        run_cmd=_fake_runner("Result=exit-code\nExecMainStatus=1\nNRestarts=0\n"),
        env={}, now=1.0, repo_root=tmp_path,
    )
    assert rc == 0
