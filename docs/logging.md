# PancakeBot logging conventions

Canonical reference for production logging in this repo. Reviewers grade
new emissions against these principles; the structured logger
(`pancakebot/log.py`) enforces the column format.

## Three persistence layers

Each has one job — don't conflate them:

1. **`var/<mode>/runtime.log` (INFO and above)** — operator-scannable
   narrative. Rare events. Only what a human needs to see while watching
   the bot run.
2. **`var/<mode>/cycle_audit.csv`** — per-round structured data. The
   analysis source of truth. pandas-loadable; one row per round-decision.
3. **Other audit CSVs as needed** — for sub-round or non-round structured
   data that doesn't fit `cycle_audit.csv` (e.g. `dry_trades.csv` for
   per-bet settlement, `var/live/latency.jsonl` for per-TX timing).

**DEBUG goes away as a production concept.** No persistent DEBUG-level
logging from production code. Python's `logging.DEBUG` stays available
for ad-hoc on-the-fly debugging (toggle on for a session, off by
default), but nothing in production emits at DEBUG.

## Format

`pancakebot/log.py` enforces (Phase B v2, 2026-05-18 — commit `7174b5e`):

```
{ts}  {LEVEL:<5}  {ACTION:<8}  {message}
```

A single ACTION column (≤ 8 chars, enforced at emit time via
`_ACTION_W`) and a free-form prose message. The prior 3-column
hierarchy (`SYSTEM` / `SUB` / `EVENT`) and the `**fields` /
`msg=` dichotomy are retired. Callers pass two positional strings;
`_emit` does no kv rendering and no formatting helpers.

```python
info(action: str, message: str) -> None
warn(action: str, message: str) -> None
error(action: str, message: str) -> None
```

The message is the operator-facing English sentence — compose key=value
fragments into the prose directly when structured data is useful (e.g.
`info("BET", f"Bet {amount_bnb:.4f} BNB on {bet_side} for epoch {epoch} (tx {tx_short})")`).
Per-round structured data still lives in `cycle_audit.csv`, not in the
log line.

## Design principles

1. **Pick the ACTION verb from the canonical vocabulary.** ACTION names
   what's happening at the framework level — `START`, `READY`, `BET`,
   `CLAIM`, `SKIP`, `RETRY`, `RECOVER`, `ALERT`, `EXIT`, etc. It's NOT
   the subsystem (no `GATE`, no `RPC`) and NOT a value (no symbol
   names). Runtime values + identifiers live in the prose.
   - Bad: `info("BTC", "fetched 16 candles")` (BTC is a value)
   - Good: `info("READY", "BTC: fetched 16 candles")`

2. **Prose is composed at the call site.** No format helpers, no kv
   rendering, no field reordering by the logger. The caller decides
   exactly how the line reads.
   - Good: `warn("SKIP", f"Skipped epoch {epoch}: pool below minimum ({pool:.2f} BNB < {threshold:.2f} BNB threshold)")`
   - Good: `warn("ALERT", f"period poll batch failed: batch[{first}..{last}]: {type(e).__name__}: {e}")`

3. **Don't duplicate data the caller already has structured.** If the
   data is in `cycle_audit.csv`, don't repeat it in the log line.
   Operator-facing log lines describe events ("bet placed at 0.001 BNB
   on Bull for epoch X"); machine-parseable per-round metrics live in
   the audit CSV.

4. **Match magnitude to format.** Seconds with 2 decimals for
   multi-second values (`delay=2.35s`); ms integers for sub-second
   (`latency=247ms`).

5. **Log level = operator urgency.**
   - **ERROR**: needs human action now.
   - **WARN**: noteworthy anomaly, no immediate action; each one merits
     being read.
   - **INFO**: sparse lifecycle events the operator wants to see while
     scanning.
   - (No production DEBUG — diagnostic/per-cycle data goes to
     `cycle_audit.csv` or a dedicated audit file.)

6. **Don't model scenarios that don't happen.** Branches and conditional
   text only for cases observed in production. Don't fabricate a
   `missing_position=oldest` field if OKX only ever truncates the tail.

## When to delete a log line vs. keep it

For each existing `info()` callsite, ask:

| Is the data also in `cycle_audit.csv`?          | Action |
|-------------------------------------------------|--------|
| Yes, byte-equivalent                            | **DELETE** the log line |
| Yes, but aggregates differently                 | If the aggregated form is operator-actionable, keep at INFO; else DELETE |
| No, per-round/per-cycle frequency               | Add a column to `cycle_audit` (or new audit CSV); if not worth capturing for analysis → DELETE |
| No, lifecycle / anomaly / restart event         | Keep at the appropriate level (INFO for lifecycle, WARN for anomaly) |

## Reference design

```
2026-05-21 22:39:01.02  WARN   SKIP      Skipped epoch 483194: incomplete kline data (SOL: 15 of 16 candles)
2026-05-21 23:35:07.99  INFO   SKIP      Skipped epoch 483205: pool below minimum (1.31 BNB < 1.50 BNB threshold)
2026-05-21 17:38:33.78  INFO   START     Starting bankroll: 0.2328 BNB ($153.17 USD)
```

ACTION verb on the left, prose on the right. Operators can `grep ACTION`
to filter by event class and read the prose for context.
