# Refactor Log

Chronological decision record for long-running cleanup/refactor work.

## 2026-03-01

1. Established persistent refactor anchor and terminology documents.
2. Adopted conceptual iteration policy from user directives.
3. Set explicit rule: no backward compatibility layers during restructure.
4. Set explicit rule: move legacy probe code to `inspection/legacy` for one
   transition cycle.

## Open Follow-Ups

1. Finalize the lean production runtime config surface.
2. Remove legacy production pipeline modules from `pancakebot/` after probe
   migration and compile checks.
3. Move old inspection scripts into `inspection/legacy`.
4. Rebuild `inspection/run_backtest_scenario.py` as a thin canonical runner.
5. Push tunable dislocation candidate knobs into `config.toml` and schema.
