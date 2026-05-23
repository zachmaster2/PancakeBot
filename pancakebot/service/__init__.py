"""Windows Service wrappers for PancakeBot live/dry supervision.

Two Windows Services (registered via pywin32 / SCM):
    PancakeBotLive  -> python -u run.py --live
    PancakeBotDry   -> python -u run.py --dry

Each service supervises its bot child subprocess: spawns it on service-start,
polls heartbeat / PID / crash artifacts every 1s, restarts the bot when it
goes STALE / CRASHED / DOWN, sends Discord alerts on state transitions,
drains the bot cleanly on SvcStop.

Replaces ``scripts/supervisor.py`` (one-shot, schtask-driven, opt-in restart).
See ``DELETION_NOTES.md`` for the staged removal of the old supervisor once
the new service has been validated.

See ``var/strategy_review/2026_05_22_supervisor_service_design.md`` for the
full design rationale.
"""
