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

`pancakebot/log.py` enforces:

```
{ts}  {LEVEL:<5} {SYSTEM:<8} {SUB:<6} {EVENT:<11}{tail}
```

`tail` is either structured (`**fields` → `key=value` pairs) or narrative
(`msg="free text"`). `_emit` short-circuits to one or the other; passing
both means the `msg=` wins and the kwargs are silently dropped (this is
a footgun — see Principle 1).

## Design principles

1. **Match form to content.** `**fields` for DATA (queryable); `msg=`
   for LIFECYCLE (human-readable narrative). Never both on the same
   emit — `_emit` ignores fields when `msg` is set.

2. **Column values name THINGS, not VALUES.** `SYSTEM` and `SUB`
   identify the subsystem (a name); `EVENT` names what happened (action
   verb / state name). Runtime values like symbols, epochs, or labels
   live in the kv-tail or `msg=`, never in the column slots.
   - Bad: `WARN GATE BTC FETCH_FAIL` (BTC is a symbol value)
   - Good: `WARN GATE KLINE PARTIAL symbol=BTC-USDT`

3. **No redundancy across hierarchy.** If `SUB=KLINE`, EVENT doesn't
   repeat `KLINE_`. Within the kv-tail, fields don't include data
   derivable from other fields.
   - Bad: `received=15 requested=16 missing_count=1` (derivable)
   - Bad: `error_class=insufficient reason=okx_publish_delay`
     (`insufficient` derivable from `reason` + counts)

4. **Don't model scenarios that don't happen.** Conditional fields and
   error branches only for cases observed in production. Don't fabricate
   `missing_position=oldest` if OKX only ever truncates the tail.

5. **Order kv-pairs by importance.** IDENTIFIER → REASON → QUANTITATIVE
   → CONTEXT.
   - Good: `symbol=BTC-USDT reason=okx_publish_delay received=15 requested=16 bar=1s`

6. **Conditional fields only when verifiable.** If you can't be confident
   in the value, omit the field — don't make one up.

7. **No leading whitespace inside `msg=`.** Column widths already provide
   alignment.
   - Bad: `msg="  BTC: 17500 done"`
   - Good: `msg="BTC: 17500 done"`

8. **Log level = operator urgency, not data type.**
   - **ERROR**: needs human action now.
   - **WARN**: noteworthy anomaly, no immediate action; each one merits
     being read.
   - **INFO**: sparse lifecycle events the operator wants to see while
     scanning.
   - (No production DEBUG — diagnostic/per-cycle data goes to
     `cycle_audit.csv` or a dedicated audit file.)

9. **Format units to match magnitude (operator-facing narrative).**
   Seconds with 2 decimals for multi-second values (`delay=2.35s`); ms
   for sub-second (`latency=247ms`). For structured kv-fields (named
   with `_ms` / `_seconds` suffix), keep the unit consistent for
   machine-parseability — the unit lives in the field name, the value
   is always in that unit.

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
WARN GATE     KLINE  PARTIAL    symbol=BTC-USDT reason=okx_publish_delay received=15 requested=16 bar=1s
```

(commit `617d76d`) — canonical structured emission. Use as the
pattern-match template for any new structured WARN.
