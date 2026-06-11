# Glossary — project code names used in comments

Comments throughout the codebase tag changes with project-historical code
names. This is the index; each entry is when-and-what, one line. (Deep
narratives live in commit messages and the operator's notes, not the repo.)

| Code name | When | What it names |
|---|---|---|
| **F0** | 2026-05 | The pool-**coverage gate**: the decision path skips the round unless the RPC cursor has polled through the pool-cutoff block (`lock - pool_cutoff_seconds`), so a bet can never be placed on a provably incomplete pool aggregate. Defined in `chain/rpc_poller.py`. |
| **Era 11** | 2026-05-07 | WSS-subscription pool watcher replaced by deterministic RPC polling (batched receipts reads). |
| **Era 12** | 2026-06-09 | getLogs migration: bet events via `eth_getLogs` range queries; batched-receipts reads retired. |
| **Era 12b** | 2026-06-10 | Single-source read path: every read RPC on the one bloXroute endpoint via `_bloxroute_call`; hedged dataseed read pool deleted; atomic round transition; anchor gross-staleness backstop. |
| **Bundle 4** | 2026-05-12 | Timing-architecture rebuild wave (submit-deadline derivation; per-symbol kline fetch timing columns). |
| **Bundle 5 / 5 v2** | 2026-05-14 | Dynamic per-round critical-path scheduling from the predicted predecessor block (anchor poll); runtime.log file sink; NtpSync retired. |
| **Candidate C** | 2026-06-06 | The single batched catch-up poll (`single_poll`) replacing the 3-leg ramp poll ladder. |
| **Phase B v2 T3-B** | 2026-05-18 | Structured `skip_context` payloads on SKIP decisions (operator-facing narratives). |
| **p4c** | 2026-05-03 | A timing-architecture revision generation (p4c-revision-2 = the rebuilt wake chain current until Bundle 5 v2). |
| **lean&clean** | 2026-04-26 | Strategy-code consolidation: module constants extracted into `[strategy.*]` TOML config; dead decision fields removed. |
| **off350** | 2026-06-08 | Broadcast-lead fix: corrected `VALIDATOR_ASSEMBLY_WINDOW_MS` 50→214 (+ anchor offset 1300→1500), moving the bet broadcast ~350ms earlier — LATE (post-lock revert) rate ~20% → ~0.2%. |
| **Phase 3 / 3c-0 / 3c-1 / 3c-2** | 2026-06-10 | Repo grand audit + cleanup: push-to-deploy bare repo (3c-0); Windows-bot service cluster archived (3c-1); systemd-direct supervision, Python supervisor archived (3c-2). |
| **Y1..Y6, D3..D5, Fix #N** | various | Reviewer-finding tags from adversarial review rounds; the adjacent comment carries the substance. |
