"""Marker lifecycle + daily-retry gating for the weekly monitor.

Covers the pure helpers of research/weekly_monitor_state_machine.py: the
retry_pending marker (atomic write, corrupt-marker self-clean) and
_resolve_run_context (Sunday-keyed makeup semantics — Sundays are the LAST
ISO day, so retries after a blind Sunday fall in the NEXT ISO week and
must be keyed back to the missed Sunday).
"""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "weekly_monitor_state_machine",
    REPO / "research" / "weekly_monitor_state_machine.py")
wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wm)


def test_marker_roundtrip_and_atomicity(tmp_path):
    p = tmp_path / "retry_pending.json"
    wm._write_retry_marker(sunday_key="2026-07-19", attempts=2,
                           reason="sync_failed", path=p)
    doc = wm._load_retry_marker(path=p)
    assert doc["sunday_key"] == "2026-07-19"
    assert doc["attempts"] == 2
    assert doc["reason"] == "sync_failed"
    assert doc["ts"] > 0
    # atomic write leaves no tmp residue
    assert not p.with_suffix(p.suffix + ".tmp").exists()


def test_corrupt_marker_deleted_and_treated_absent(tmp_path):
    p = tmp_path / "retry_pending.json"
    p.write_text("{not json", encoding="utf-8")
    assert wm._load_retry_marker(path=p) is None
    assert not p.exists()


def test_marker_missing_sunday_key_deleted(tmp_path):
    p = tmp_path / "retry_pending.json"
    p.write_text(json.dumps({"attempts": 1}), encoding="utf-8")
    assert wm._load_retry_marker(path=p) is None
    assert not p.exists()


def test_marker_bad_date_deleted(tmp_path):
    p = tmp_path / "retry_pending.json"
    p.write_text(json.dumps({"sunday_key": "not-a-date", "attempts": 1}),
                 encoding="utf-8")
    assert wm._load_retry_marker(path=p) is None
    assert not p.exists()


def test_clear_marker_idempotent(tmp_path):
    p = tmp_path / "retry_pending.json"
    wm._clear_retry_marker(path=p)  # absent: no raise
    wm._write_retry_marker(sunday_key="2026-07-19", attempts=1,
                           reason="data_stale", path=p)
    wm._clear_retry_marker(path=p)
    assert not p.exists()


def test_resolve_no_marker_passthrough():
    week, retry, completed = wm._resolve_run_context("2026-07-22", None)
    assert (week, retry, completed) == ("2026-07-22", False, False)


def test_resolve_weekday_with_marker_is_sunday_keyed_makeup():
    marker = {"sunday_key": "2026-07-19", "attempts": 3}
    # Wednesday 2026-07-22 is ISO week 30; the missed Sunday 2026-07-19 is
    # week 29 — the makeup must be keyed to the Sunday.
    week, retry, completed = wm._resolve_run_context("2026-07-22", marker)
    assert (week, retry, completed) == ("2026-07-19", True, False)
    assert wm._iso_week_key("2026-07-22") == "2026-W30"
    assert wm._iso_week_key("2026-07-19") == "2026-W29"


def test_resolve_next_sunday_supersedes_as_completed_blind():
    marker = {"sunday_key": "2026-07-19", "attempts": 7}
    week, retry, completed = wm._resolve_run_context("2026-07-26", marker)
    assert (week, retry, completed) == ("2026-07-26", False, True)


def test_resolve_same_sunday_rerun_not_completed_blind():
    marker = {"sunday_key": "2026-07-19", "attempts": 1}
    week, retry, completed = wm._resolve_run_context("2026-07-19", marker)
    assert (week, retry, completed) == ("2026-07-19", False, False)


def test_iso_week_sundays_distinct_and_last_day():
    # Consecutive cron Sundays always map to distinct ISO weeks, including
    # the ISO year boundary (2027-01-03 is still ISO year 2026, week 53).
    sundays = ["2026-07-19", "2026-07-26", "2026-12-27", "2027-01-03",
               "2027-01-10"]
    keys = [wm._iso_week_key(s) for s in sundays]
    assert len(set(keys)) == len(keys)
    assert keys[2:] == ["2026-W52", "2026-W53", "2027-W01"]
    # Sunday is the LAST day of its ISO week: Mon 07-13 .. Sun 07-19 share
    # a week; Mon 07-20 starts the next.
    assert wm._iso_week_key("2026-07-13") == wm._iso_week_key("2026-07-19")
    assert wm._iso_week_key("2026-07-20") != wm._iso_week_key("2026-07-19")
