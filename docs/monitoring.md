# Regime monitoring (wait-and-monitor posture)

As of 2026-06-12 the live bot is PAUSED (bankroll 2.30627 BNB) and the
systematic search of the existing + backfillable data envelope is COMPLETE
with no deployable strategy found (see
`research/phase0_okx_perp_2026_06_11_findings.md` and the probes it
references). The standing posture is **wait-and-monitor**: cheap offline
checks that detect a regime turn worth re-investigating.

## The monitor

```
# 1. refresh the round store (The Graph key required in env)
python run.py --sync
# 2. run the monitor (extends the OKX perp tape incrementally, ~minutes
#    if run weekly, ~hours if monthly; --no-fetch skips the extension)
.venv/Scripts/python.exe research/monitor_2026_06_12.py
```

Or let the monitor do both: `... research/monitor_2026_06_12.py --sync`.

Cadence: **weekly recommended, monthly minimum** — OKX trade-tape
retention is ~3 months; a longer gap becomes unfillable (the monitor then
reports a non-contiguous tape and T3 cells shrink gracefully via the
segment guard rather than computing biased windows).

Never run two monitors (or a monitor and a capture script) concurrently:
tape appends are not locked, and interleaved writers can tear a line.
Duplicate trades from overlapping runs are harmless (deduped by tradeId
at load), torn lines are not.

Output: `var/strategy_review/monitor_runs/<YYYY-MM-DD>/findings.json` +
`summary.txt`. The console digest ends with `VERDICT: quiet` or
`VERDICT: TRIPPED: ...`.

## Pre-registered tripwires (PINNED 2026-06-12 — do not tune)

| wire | statistic | trip condition |
|---|---|---|
| T1 | latest 500 canonical risk-free bets: flat-stake PnL vs the market-implied null (win_i ~ Bernoulli(1/payout_i) — the exact no-edge hypothesis in a pari-mutuel) | p < 0.01 |
| T2 | contrarian @ lock−6s, threshold 0.4 (golden-era best), trailing 15 days: PnL deficit vs permutation null | permutation z > 2 AND n > 500 |
| T3 | perp tape imbalance (1m/5m/15m × cutoff 2s/6s = 6 cells), trailing 15 days: deficit vs permutation null | any Šidák-adjusted two-sided p < 0.01 over the 6 cells |

## What a trip means — and does not mean

A tripped wire is a **"rerun the Phase-0 gauntlet on fresh data"
signal, never a deploy signal.** The 2026-06 probes demonstrated
repeatedly how raw p<0.01 cells dissolve under full coverage and sweep
discounts (the cautionary exhibit in the OKX findings went p=0.0016 →
p=0.42). Any revival still owes: cross-validation, holdout, permutation
nulls, multiple-comparison discounts, and the program-honesty running
count of one-shot tests.

Quiet runs are informative too — each one extends the no-regime-turn
record that justifies keeping the bot paused.

## Reviving the bot (if a trip survives the gauntlet)

The VM may be decommissioned by the time you read this. Bring-up is
fresh-clone-validated and documented: `bootstrap/README.md` (fresh-VM
order), `docs/SUPERVISOR.md` (systemd-direct, going-live-after-soak).
Restart on an existing VM: `systemctl enable --now pancakebot-live`.
