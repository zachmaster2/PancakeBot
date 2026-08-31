# HISTORICAL as of 2026-08-31

Everything in `bootstrap/linux/` provisioned the Frankfurt droplet
`pancakebot-deployment-node`, which was **destroyed on 2026-08-31** when
the project moved to data-collection-only.

No host runs any of this. The systemd units and `run_weekly_monitor.sh`
hardcode `/root/pancakebot`, a path that exists nowhere now, and
`install.sh` refuses to run outside it.

Kept deliberately: this is the only record of how the live deployment was
built, and it is what a future redeploy would start from.

The Sunday Discord dead-man contract these scripts carried is retired.
See `docs/alerting_retirement_2026_08_31.md`.
