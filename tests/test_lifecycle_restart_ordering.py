"""Restart-pair ordering metadata for lifecycle Discord alerts.

A `systemctl restart` fires the stopped hook (ExecStopPost) and the started
hook (ExecStartPost) as two independent `pancakebot-notify@` oneshots with
`--no-block`, so they run concurrently and their Discord POSTs race. On
2026-08-21 both restarts landed STARTED ~6ms ahead of STOPPED despite the
stopped hook launching ~44ms earlier, and a whole-second timestamp made the
pair look simultaneous as well as out of order.

These pin the fix: millisecond + UTC stamps, and an evt_mono_us taken from
the transition each event is actually about, so the true order survives any
arrival order.
"""
import re

from pancakebot.ops import notify_lifecycle as nl
from pancakebot.ops.notifications import _local_time_str, build_message

# Real values from the 22:08:14 UTC restart (monotonic µs, 40ms apart).
STOP_MONO = 4476970489771
START_MONO = 4476970529771


def test_local_time_str_has_millisecond_and_utc():
    s = _local_time_str()
    assert re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \w+ / "
        r"\d{2}:\d{2}:\d{2}\.\d{3}Z", s), s


def test_restart_pair_is_self_ordering_regardless_of_arrival():
    stopped = build_message(mode="live", kind="STOPPED",
                            fields={"evt_mono_us": STOP_MONO,
                                    "intentional": True})
    started = build_message(mode="live", kind="STARTED",
                            fields={"pid": 637773, "evt_mono_us": START_MONO})
    assert f"evt_mono_us: `{STOP_MONO}`" in stopped
    assert f"evt_mono_us: `{START_MONO}`" in started
    assert "pid: `637773`" in started
    for m in (stopped, started):
        assert "lower = earlier" in m


def test_order_fields_trust_mainpid_only_on_started():
    """During a restart the stopped hook runs AFTER the replacement main
    process exists, so MainPID already names the NEW pid (measured: the
    stopped notifier POSTed ~0.8s after the new instance started). It must
    never be reported as the stopped process's pid."""
    out = (f"MainPID=637773\n"
           f"ActiveEnterTimestampMonotonic={START_MONO}\n"
           f"InactiveEnterTimestampMonotonic={STOP_MONO}\n")
    started = nl.event_order_fields("u", "started", run_cmd=lambda a: out)
    stopped = nl.event_order_fields("u", "stopped", run_cmd=lambda a: out)
    assert started["pid"] == 637773
    assert "pid" not in stopped
    # each event reports ITS OWN transition, so the pair orders correctly
    assert stopped["evt_mono_us"] < started["evt_mono_us"]


def test_order_fields_tolerate_missing_or_zero_values():
    out = "MainPID=0\nActiveEnterTimestampMonotonic=\n"
    assert nl.event_order_fields("u", "started", run_cmd=lambda a: out) == {}


def test_order_fields_never_raise():
    """Runs inside the alerting path, which must never raise."""
    def boom(argv):
        raise OSError("systemctl gone")

    assert nl.event_order_fields("u", "started", run_cmd=boom) == {}
