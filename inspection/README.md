# Inspection Tooling

Inspection tooling is split into two domains:

1. `inspection/run_backtest_scenario.py`:
   canonical probe entrypoint that executes the production backtest pipeline.
2. `inspection/legacy/`:
   temporary one-cycle archive for historical probe scripts retained for
   transition traceability.

The canonical probe path must not reimplement production strategy logic.
