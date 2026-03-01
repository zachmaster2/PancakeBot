# Legacy Inspection Archive

This directory contains historical inspection scripts and archived production
modules retained for one transition cycle during refactor.

Rules:

1. Do not add new production dependencies to this directory.
2. Do not use these scripts as canonical references for strategy behavior.
3. Prefer `inspection/run_backtest_scenario.py` for active probe workflows.
4. Remove this archive at the end of the transition cycle once parity and
   migration objectives are complete.
5. Archived production modules under `inspection/legacy/pancakebot/` are
   read-only references and must not be imported by active runtime/backtest
   code.
