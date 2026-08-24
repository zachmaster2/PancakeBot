"""Restart-pair ordering for lifecycle Discord alerts.

A `systemctl restart` starts the stopped hook (ExecStopPost) and the started
hook (ExecStartPost) as two independent `pancakebot-notify@` oneshots with
`--no-block`, so they run concurrently. Measured on 2026-08-21: the stopped
hook launched ~44ms EARLIER but its Discord POST landed ~6ms LATER in both
restarts (it does more work first — journal tail plus `systemctl show`), and
Discord lists by ARRIVAL, so the pair displayed as STARTED-then-STOPPED.

The fix orders them at the source: the started hook waits for its stopped
sibling's unit to leave a busy state before posting. These pin that it costs
nothing on a fresh start, is bounded, and never raises.

Verified live on the 2026-08-24T00:15:53Z restart, the first since the wait
landed: both hooks launched 44ms apart (stopped .350841, started .394938),
the STOPPED POST went at .063858 of the next second, the stopped unit
deactivated at .116242, and the STARTED POST followed at .588507 -- 472ms
later, i.e. the started hook demonstrably waited for its sibling.

Because ordering is structural, the messages carry no ordering metadata:
no evt_mono_us, no pid, and a plain whole-second local timestamp.
"""
import re

from pancakebot.ops import notify_lifecycle as nl
from pancakebot.ops.notifications import _local_time_str, build_message


def test_local_time_str_is_a_plain_local_whole_second_stamp():
    """The millisecond + dual-UTC stamp existed only to order a pair that
    Discord listed by arrival. That job is structural now, and the long
    form wrapped onto two lines on a phone. This fails if it comes back."""
    s = _local_time_str()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+", s), s
    assert "." not in s          # no milliseconds
    assert not s.endswith("Z")   # no appended UTC


def test_lifecycle_alerts_carry_no_ordering_metadata():
    """Ordering is structural, so the messages carry no proof metadata --
    and the lifecycle path no longer collects a pid to carry."""
    stopped = build_message(mode="live", kind="STOPPED",
                            fields={"intentional": True})
    started = build_message(mode="live", kind="STARTED", fields={})
    for m in (stopped, started):
        assert "evt_mono_us" not in m
        assert "lower = earlier" not in m
        assert "pid:" not in m
    # The generic renderer still supports a pid for callers that pass one
    # (it predates all of this); nothing in the lifecycle path does.
    assert not hasattr(nl, "started_pid_field")


# ---- the structural wait --------------------------------------------------

def _runner(states):
    """run_cmd stub yielding successive `systemctl is-active` outputs."""
    seq = list(states)

    def run(argv):
        return seq.pop(0) if seq else "inactive\n"
    return run


def test_stopped_hook_never_waits():
    calls = []
    waited = nl.wait_for_stopped_sibling(
        "pancakebot-live", "stopped",
        run_cmd=lambda a: calls.append(a) or "activating\n",
        sleep=lambda s: None)
    assert waited == 0.0
    assert calls == []          # not even a query


def test_fresh_start_costs_nothing():
    """No preceding stop: systemd reports `inactive` even for an instance
    that was never activated, so the started hook posts immediately."""
    slept = []
    waited = nl.wait_for_stopped_sibling(
        "pancakebot-live", "started",
        run_cmd=lambda a: "inactive\n", sleep=slept.append)
    assert waited == 0.0
    assert slept == []


def test_started_hook_waits_out_a_running_sibling():
    slept = []
    waited = nl.wait_for_stopped_sibling(
        "pancakebot-live", "started",
        run_cmd=_runner(["activating\n", "activating\n", "inactive\n"]),
        sleep=slept.append)
    assert waited > 0.0
    assert len(slept) == 2


def test_wait_queries_the_correct_sibling_instance():
    seen = []

    def run(argv):
        seen.append(argv)
        return "inactive\n"

    nl.wait_for_stopped_sibling("pancakebot-live", "started",
                                run_cmd=run, sleep=lambda s: None)
    assert seen[0] == ["systemctl", "is-active",
                       "pancakebot-notify@pancakebot-live-stopped.service"]


def test_wait_is_bounded_when_the_sibling_is_wedged():
    """A wedged stopped hook (Discord can burn 10s + retry) must not hold
    the started alert forever — it degrades to the old out-of-order
    behaviour, never to a lost alert."""
    slept = []
    waited = nl.wait_for_stopped_sibling(
        "pancakebot-live", "started", run_cmd=lambda a: "activating\n",
        sleep=slept.append, timeout_s=0.2)
    assert waited >= 0.2
    assert len(slept) <= 5


def test_wait_treats_a_failed_query_as_not_busy():
    """_run_cmd returns '' on hard failure; that must fall through, not spin."""
    waited = nl.wait_for_stopped_sibling(
        "pancakebot-live", "started", run_cmd=lambda a: "",
        sleep=lambda s: None)
    assert waited == 0.0


# ---- the wait is the only thing doing ordering work ----------------------

def test_wait_needs_no_pid_or_timestamp():
    """It queries the SIBLING UNIT's state and nothing else, which is why
    removing the pid and the sub-second stamp cannot affect ordering."""
    seen = []

    def run(argv):
        seen.append(argv)
        return "inactive"

    nl.wait_for_stopped_sibling("pancakebot-live", "started", run_cmd=run,
                                sleep=lambda s: None)
    assert seen == [["systemctl", "is-active",
                     "pancakebot-notify@pancakebot-live-stopped.service"]]
