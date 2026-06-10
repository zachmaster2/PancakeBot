"""Alerting package for the PancakeBot bots (Linux/systemd).

systemd is the supervisor (Phase 3c-2, 2026-06-10): the tracked units at
``bootstrap/linux/systemd/`` run ``run.py`` directly, restart it on failure
(``Restart=on-failure`` + the ``StartLimitBurst`` crashloop brake), and
trigger ``pancakebot-notify@`` oneshots on start/stop edges, which run
``pancakebot.ops.notify_lifecycle`` -> ``notifications`` (the Discord alert
executor, the one module here). See docs/SUPERVISOR.md.

The retired Python supervisor stack (SupervisorCore, the ServicePlatform
adapters, ``supervise.py``) lives in the offline archive,
``Downloads/OLD/pancakebot_old/``.
"""
