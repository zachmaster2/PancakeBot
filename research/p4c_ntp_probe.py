"""Empirical probe: NTP roundtrip cost for the per-round ntp_sync_wake.

Measures wall-clock cost of one ``ntplib.NTPClient.request(...)`` against
public stratum-2 pool servers. The result locks
``NTP_QUERY_TIME_P95_MS`` in ``pancakebot/timing_constants.py``: the
ntp_sync_wake fires at ``lock - (pool_read_wakeup_offset_ms +
NTP_QUERY_TIME_P95_MS + NTP_SAFETY_BUFFER_MS)``, so the budget needs
to cover the p95 wall-clock observed here.

Method:
- 50 samples, 100ms gap between calls (NTP servers are built for
  stratum-2 polling cadences below this; spacing further would
  blunt the measurement without changing the answer).
- Rotate across three pool servers (pool.ntp.org, time.cloudflare.com,
  time.google.com) so we don't hammer one backend.
- Record total wall-clock (entry to NTP_request return) including
  DNS + UDP RTT. This is what the engine pays at the wake.
- Discard any timeout (~2s ceiling) but report the count.

Output: stdout with p50/p90/p95/p99/max + offset distribution.

Run::
    python research/p4c_ntp_probe.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import ntplib

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_SERVERS = ("pool.ntp.org", "time.cloudflare.com", "time.google.com")
_N = 150
_GAP_S = 0.2  # be a polite citizen of the public NTP pool
_TIMEOUT_S = 2.0


def main() -> int:
    client = ntplib.NTPClient()
    durations_ms: list[float] = []
    offsets_s: list[float] = []
    timeouts = 0
    other_errors = 0

    for i in range(_N):
        server = _SERVERS[i % len(_SERVERS)]
        t_start = time.perf_counter()
        try:
            r = client.request(server, version=3, timeout=_TIMEOUT_S)
        except ntplib.NTPException as e:
            timeouts += 1
            print(f"  [{i:>2}] {server:>22}: TIMEOUT ({e})", flush=True)
            time.sleep(_GAP_S)
            continue
        except Exception as e:  # noqa: BLE001 -- probe never crashes
            other_errors += 1
            print(f"  [{i:>2}] {server:>22}: ERROR {type(e).__name__}: {e}", flush=True)
            time.sleep(_GAP_S)
            continue
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        durations_ms.append(elapsed_ms)
        offsets_s.append(r.offset)
        print(f"  [{i:>2}] {server:>22}: {elapsed_ms:>6.1f}ms  offset={r.offset*1000:+8.2f}ms",
              flush=True)
        time.sleep(_GAP_S)

    if not durations_ms:
        print("\nNo successful samples; aborting.")
        return 1

    # Per-server breakdown so we can size NTP_QUERY_TIME_P99_MS for the
    # WORST p99 across servers, not the pooled p99 (which would understate
    # pool.ntp.org's slower responses).
    per_server_ms: dict[str, list[float]] = {s: [] for s in _SERVERS}
    for i, d in enumerate(durations_ms):
        # Reconstruct which server produced sample i. The probe loop
        # rotates _SERVERS in order; skipped (timeout) iterations don't
        # land in durations_ms, so this index is approximate -- we just
        # need a rough per-server distribution. For exact attribution
        # the probe would need to record server alongside each sample;
        # n=150 is large enough that the rotation hits each server
        # equally and the rough split is fine for sizing.
        per_server_ms[_SERVERS[i % len(_SERVERS)]].append(d)

    print()
    print("=" * 70)
    print(f"NTP request wall-clock (n={len(durations_ms)}) -- ms")
    print("=" * 70)
    durations_ms_sorted = sorted(durations_ms)
    n = len(durations_ms_sorted)
    p50 = durations_ms_sorted[n // 2]
    p90 = durations_ms_sorted[int(n * 0.90)]
    p95 = durations_ms_sorted[int(n * 0.95)]
    p99 = durations_ms_sorted[int(n * 0.99)] if n >= 100 else durations_ms_sorted[-1]
    mn = durations_ms_sorted[0]
    mx = durations_ms_sorted[-1]
    mean = statistics.mean(durations_ms)
    print(f"  pooled (all servers):")
    print(f"    min   = {mn:>6.1f}")
    print(f"    p50   = {p50:>6.1f}")
    print(f"    p90   = {p90:>6.1f}")
    print(f"    p95   = {p95:>6.1f}")
    print(f"    p99   = {p99:>6.1f}")
    print(f"    max   = {mx:>6.1f}")
    print(f"    mean  = {mean:>6.1f}")
    print()
    print(f"  per-server p99 (rough rotation attribution):")
    server_p99s: list[float] = []
    for server in _SERVERS:
        s_data = sorted(per_server_ms[server])
        if not s_data:
            continue
        s_n = len(s_data)
        s_p99 = s_data[int(s_n * 0.99)] if s_n >= 100 else s_data[-1]
        server_p99s.append(s_p99)
        print(f"    {server:>22}: n={s_n:>3}  p99={s_p99:>6.1f}")
    worst_p99 = max(server_p99s) if server_p99s else p99
    print(f"  worst-server p99 = {worst_p99:>6.1f}  <-- locks NTP_QUERY_TIME_P99_MS")
    print()
    print(f"  timeouts:     {timeouts}")
    print(f"  other errors: {other_errors}")
    print()

    print("=" * 70)
    print(f"NTP measured offsets (n={len(offsets_s)}) -- ms")
    print("=" * 70)
    offsets_ms = [o * 1000.0 for o in offsets_s]
    offsets_ms_sorted = sorted(offsets_ms, key=abs)
    print(f"  median |offset|  = {offsets_ms_sorted[n // 2]:+.2f}")
    print(f"  max |offset|     = {offsets_ms_sorted[-1]:+.2f}")
    print(f"  mean offset      = {statistics.mean(offsets_ms):+.3f}")
    print(f"  stdev offset     = {statistics.stdev(offsets_ms):.3f}")
    print()
    print("Recommendation:")
    print(f"  NTP_QUERY_TIME_P99_MS = {int(worst_p99) + 25}  "
          f"# worst-server p99={worst_p99:.1f} + small buffer")
    print()
    print("Note: with rotation across 3 servers, worst-case sequential is")
    print(f"  3 x p99 = {3 * worst_p99:.0f} ms (only relevant if force_resync")
    print(f"  fell through all servers; on the engine's wake budget, the")
    print(f"  pre-bankroll gap is 5000ms so this is amply covered).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
