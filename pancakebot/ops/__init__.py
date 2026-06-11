"""The systemd-facing alerting package (invoked by systemd, not the bot runtime).

Phase 3c-2 (systemd-direct): the bot units' ExecStartPost/ExecStopPost
hooks trigger ``pancakebot-notify@<unit>-<event>`` oneshot units, which run
``notify_lifecycle`` here; it maps the unit state to an alert kind and
fires it through ``notifications`` (the Discord executor, also here).
See docs/SUPERVISOR.md.
"""
