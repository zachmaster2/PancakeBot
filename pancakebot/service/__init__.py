"""Windows Service wrappers for PancakeBot live/dry supervision.

Two Windows Services (registered via pywin32 / SCM):
    PancakeBotLive  -> python -u run.py --live
    PancakeBotDry   -> python -u run.py --dry

Each service supervises its bot child subprocess: spawns it on service-start,
polls heartbeat / PID / crash artifacts every 1s, restarts the bot when it
goes STALE / CRASHED / DOWN, sends Discord alerts on state transitions,
drains the bot cleanly on SvcStop.

Replaced the legacy one-shot ``scripts/supervisor.py`` (schtask-driven, opt-in
restart) on 2026-05-23 after a soak window confirmed the service architecture
was stable. See ``var/strategy_review/2026_05_22_supervisor_service_design.md``
for the full design rationale.
"""
