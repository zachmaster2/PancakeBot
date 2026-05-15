"""I6 baseline drift probe — measure (local_clock − ntp_truth) over a
30-min window under the current W32Time config (SpecialPollInterval=256s,
MaxPollInterval=10 → 1024s actual cadence).

Methodology:
- Query a public NTP server (rotating across cloudflare/google/pool.ntp.org)
  every 5 seconds.
- Each query yields ``response.offset`` = `(local_clock − ntp_truth)` in seconds.
- Log ts_local, server, offset_ms, rtt_ms.
- After 30 min, summarize: max abs offset, p50, p95, p99, mean.

Output: ``var/i6_drift_baseline_2026_05_14.jsonl`` (one record per query).

Comparison: after the user tightens W32Time (MaxPollInterval=5 → 32s),
re-run with output ``var/i6_drift_tightened_2026_05_14.jsonl`` and compare.

Notes:
- This script does NOT change W32Time config; it only observes.
- ntplib timeouts (1.5s) match the bot's per-round NTP timeout.
- Failures (timeout / glitch >250ms) are logged but don't abort the run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import ntplib  # type: ignore[import-untyped]


_SERVERS: tuple[str, ...] = (
    "time.cloudflare.com",
    "time.google.com",
    "pool.ntp.org",
)
_PROBE_INTERVAL_S: float = 5.0
_DURATION_S: float = 30 * 60.0  # 30 min
_TIMEOUT_S: float = 1.5


def main() -> int:
    out_dir = Path("var")
    out_dir.mkdir(parents=True, exist_ok=True)
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    out_path = out_dir / f"i6_drift_{label}_2026_05_14.jsonl"
    client = ntplib.NTPClient()
    t_start = time.monotonic()
    n_ok = 0
    n_err = 0
    server_idx = 0
    offsets_ms: list[float] = []
    print(f"Writing to {out_path}; running for {_DURATION_S/60:.0f}min; "
          f"interval {_PROBE_INTERVAL_S:.1f}s", flush=True)
    with out_path.open("w", encoding="utf-8") as f:
        while time.monotonic() - t_start < _DURATION_S:
            t_probe_start = time.monotonic()
            server = _SERVERS[server_idx % len(_SERVERS)]
            server_idx += 1
            rec = {
                "ts_local": time.time(),
                "server": server,
            }
            t0 = time.monotonic()
            try:
                resp = client.request(server, version=3, timeout=_TIMEOUT_S)
                rtt_ms = (time.monotonic() - t0) * 1000.0
                offset_ms = float(resp.offset) * 1000.0
                rec["ok"] = True
                rec["rtt_ms"] = rtt_ms
                rec["offset_ms"] = offset_ms
                offsets_ms.append(offset_ms)
                n_ok += 1
            except Exception as e:
                rec["ok"] = False
                rec["err"] = f"{type(e).__name__}: {e}"
                n_err += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            # Pace to interval (subtract probe latency).
            spent = time.monotonic() - t_probe_start
            sleep_for = max(0.0, _PROBE_INTERVAL_S - spent)
            time.sleep(sleep_for)
    # Summarize.
    n_total = n_ok + n_err
    summary = {
        "label": label,
        "duration_min": _DURATION_S / 60,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_err": n_err,
    }
    if offsets_ms:
        offsets_abs = sorted(abs(x) for x in offsets_ms)
        n = len(offsets_abs)
        summary["max_abs_ms"] = offsets_abs[-1]
        summary["p50_abs_ms"] = offsets_abs[n // 2]
        summary["p95_abs_ms"] = offsets_abs[min(n - 1, int(n * 0.95))]
        summary["p99_abs_ms"] = offsets_abs[min(n - 1, int(n * 0.99))]
        summary["mean_abs_ms"] = sum(abs(x) for x in offsets_ms) / n
        summary["mean_signed_ms"] = sum(offsets_ms) / n
    summary_path = out_dir / f"i6_drift_{label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
