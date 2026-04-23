"""Tiny helper for supervisor smoke tests.

Usage:
    python scripts/_smoke_write_artifacts.py <mode> <kind> [extra...]

Kinds:
    crash              - writes crash.json with a realistic AttributeError
    heartbeat_fresh    - fresh heartbeat pointing at pid=<extra0>
    heartbeat_stale    - stale heartbeat (mtime 10s ago) pointing at pid=<extra0>
    pid_old            - writes bot.pid=<extra0> with mtime 120s ago
    history_fast       - populates restart_history.jsonl with 4 entries spanning last 5 min
    history_slow       - populates restart_history.jsonl with 9 entries spanning last 12 h
    history_48h        - drops a 48h-old entry into restart_history.jsonl

All writes are plain (not atomic) -- this is test-only glue, not supervisor code.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _path(mode: str, name: str) -> Path:
    return Path("var") / mode / name


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    mode = sys.argv[1]
    kind = sys.argv[2]
    extra = sys.argv[3:]
    now = time.time()

    if kind == "crash":
        p = _path(mode, "crash.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "ts_wall": now,
            "exc_type": "AttributeError",
            "exc_repr": "AttributeError(\"'RoundData' object has no attribute 'start_at'\")",
            "traceback_str": (
                "Traceback (most recent call last):\n"
                "  File \"run.py\", line 63, in main\n"
                "    run_from_config(config_path=args.config, dry=args.dry, ...)\n"
                "  File \"pancakebot/app.py\", line 175, in run_from_config\n"
                "    engine.run_realtime_loop(runtime_cfg)\n"
                "  File \"pancakebot/runtime/engine.py\", line 104, in run_realtime_loop\n"
                "    _run_one_iteration(cfg, closed_state)\n"
                "  File \"pancakebot/runtime/dry.py\", line 566, in _dry_settle_available_bets\n"
                "    start_at=int(rd.start_at),\n"
                "AttributeError: 'RoundData' object has no attribute 'start_at'\n"
            ),
            "last_epoch": 474776,
        }))
        return 0
    if kind == "heartbeat_fresh":
        pid = int(extra[0])
        p = _path(mode, "heartbeat.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "pid": pid, "ts_wall": now, "last_epoch": 474974,
            "bankroll_bnb": 5.0, "iteration_count": 42,
        }))
        return 0
    if kind == "heartbeat_stale":
        pid = int(extra[0])
        p = _path(mode, "heartbeat.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "pid": pid, "ts_wall": now - 10, "last_epoch": 474974,
            "bankroll_bnb": 5.0, "iteration_count": 42,
        }))
        past = now - 10
        os.utime(str(p), (past, past))
        return 0
    if kind == "pid_old":
        pid = int(extra[0])
        p = _path(mode, "bot.pid")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(pid))
        old = now - 120
        os.utime(str(p), (old, old))
        return 0
    if kind == "history_fast":
        # 4 entries within last 5 minutes -- triggers fast-tier suppression (>=3).
        p = _path(mode, "restart_history.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for i in range(4):
            ts_wall = now - (60 * (i + 1))  # 1/2/3/4 minutes ago
            entries.append({
                "ts": "2026-04-22T19:" + f"{50 + i:02d}" + ":00Z",
                "ts_wall": ts_wall,
                "trigger": "CRASHED",
                "new_pid": 10000 + i,
                "log_path": f"var/{mode}/logs/{mode}-auto-synthetic-{i}.log",
            })
        p.write_text("\n".join(json.dumps(e, sort_keys=True, separators=(",", ":")) for e in entries) + "\n")
        return 0
    if kind == "history_slow":
        # 9 entries within last 12h, NONE in the last 15 min -- triggers slow-
        # tier escalation (>=8) without firing fast-tier suppression.
        p = _path(mode, "restart_history.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for i in range(9):
            # Space them 80 min apart starting 60 min ago: 60, 140, 220, ...
            ts_wall = now - (60 * 60 + i * 80 * 60)
            entries.append({
                "ts": "slow_synth_" + str(i),
                "ts_wall": ts_wall,
                "trigger": "CRASHED",
                "new_pid": 20000 + i,
                "log_path": f"var/{mode}/logs/{mode}-auto-slow-{i}.log",
            })
        p.write_text("\n".join(json.dumps(e, sort_keys=True, separators=(",", ":")) for e in entries) + "\n")
        return 0
    if kind == "history_48h":
        # Single very-old entry -- must be pruned on next write.
        p = _path(mode, "restart_history.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        entries = [{
            "ts": "48h_old",
            "ts_wall": now - (48 * 3600),
            "trigger": "CRASHED",
            "new_pid": 99999,
            "log_path": "var/dry/logs/old.log",
        }]
        p.write_text(json.dumps(entries[0], sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    print(f"unknown kind: {kind}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
