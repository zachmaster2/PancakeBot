"""Operational glue invoked by systemd, not by the bot runtime.

Phase 3c-2 (systemd-direct): the bot units' ExecStartPost/ExecStopPost
hooks trigger ``pancakebot-notify@<unit>-<event>`` oneshot units, which run
``notify_lifecycle`` here. Lives outside ``pancakebot/service`` because the
supervisor package is retired with the cutover; the alert EXECUTOR
(``pancakebot.service.notifications``) stays.
"""
