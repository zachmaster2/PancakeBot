# Inclusion-offset experiment (2026-06-08)

Controlled **live** A/B that measured the validator's real TX-acceptance margin and
produced the off350 fix (`VALIDATOR_ASSEMBLY_WINDOW_MS` 50→214, commit `07599e7`).
Full methodology + results + A-vs-B verdict: memory `project_pancakebot_off350_experiment.md`.

## Scripts
- `run_experiment.py` — coordinator + 5 wallet processes (multiprocessing, Unix-socket IPC).
  Per round: coordinator anchor-polls + computes the SSOT dynamic deadline and publishes it;
  each wallet pre-signs a min-stake betBull/betBear (random side) and broadcasts at
  `deadline + offset`. `--dry-run` signs but sends nothing. Offsets {−200,−300,−350,−400,−450}
  = off_vs_deadline {200,300,350,400,450}; `ANCHOR_LEAD_MS=1500`.
- `analyze.py` — reconstructs per-bet inclusion from the logged tx_hashes (the in-run check
  logged `inclusion_pending` because `.hex()` dropped the `0x` prefix on this web3 build;
  prepend `0x` → receipt → block → BEP-520 ms). Per-wallet aggregates + within-round paired
  table + LATE-vs-offset.
- `watch.py` — progress checkpoint (blocks until N rounds, then runs `analyze.py` + balances).
- `slip_vs_predecessor.py` — the deadline-normalized observational analysis (reconstructs each
  round's actual predecessor from chain; showed observation can't decide A-vs-B because the
  live bot operates at a single off_vs_deadline).

Result logs (not committed): `var/experiment_20260608/live1/` on the VM.

## Test wallets — keys are NOT in git
- 5 dedicated test wallets, keys at **`/etc/pancakebot/experiment_wallets.env`** on the VM
  (`chmod 600`, root-only) + a `.bak` alongside. Format: `WALLET_<i>_KEY` / `WALLET_<i>_ADDR`.
- **Zach holds the keys** for any sweep-back to production. The scripts read the env file at
  runtime and never print or commit private keys. The production wallet was never involved.
- Post-experiment balances: ~0.0088 BNB/wallet (~0.044 total) + claimable winnings — retained
  for a possible off350 confirmation soak or the diverse-endpoint fanout follow-up.
