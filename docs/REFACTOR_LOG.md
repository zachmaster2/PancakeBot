# Refactor Log

Chronological decision record for long-running cleanup/refactor work.

## 2026-03-01

1. Established persistent refactor anchor and terminology documents.
2. Adopted conceptual iteration policy from user directives.
3. Set explicit rule: no backward compatibility layers during restructure.
4. Set explicit rule: move legacy probe code to `inspection/legacy` for one
   transition cycle.
5. Replaced hardcoded dislocation candidate family with config-driven builder
   in production strategy engine wiring.
6. Refactored runtime/backtest strategy config path to
   `strategy.dislocation.{selector,candidates}`.
7. Removed legacy runtime/model/policy knobs from active `AppConfig`,
   `RuntimeConfig`, and `load_config.py`.
8. Rewrote `config.toml` to lean production schema with explicit dislocation
   candidate tables.
9. Archived dead production model/planner/policy modules under
   `inspection/legacy/pancakebot/` and removed them from active production
   tree.
10. Smoke-validated canonical probe path with
    `python -m inspection.run_backtest_scenario --name smoke_refactor_sync --sim-size 200`.
11. Renamed strategy module interface to clean terminology:
    - file: `dislocation_engine.py`
    - class: `DislocationEngine`
    - builder: `build_dislocation_engine_from_config(...)`
12. Archived remaining dead `pancakebot/domain/models/*` placeholder modules
    under `inspection/legacy/pancakebot/domain/models/` and removed active
    references from production tree.
13. Removed non-functional runtime/config leftovers:
    - dropped `event_freshness_slack_seconds` from active config/runtime path
    - dropped `min_bet_amount_bnb` from `RuntimeConfig`
    - removed redundant constants persistence in `runtime_loop` startup
14. Updated runtime and strategy config docs/comments to dislocation-only
    terminology and revalidated continuous/chunk-reset smoke runs.

## Open Follow-Ups

1. Continue reducing redundant feature fields/constants that are no longer
   needed by the dislocation-only pipeline.
2. Run broader scenario matrix and compare against pre-refactor baselines for
   behavior parity.
