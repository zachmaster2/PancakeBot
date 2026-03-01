# Legacy Production Module Archive

This directory preserves removed production modules for one transition cycle.

Scope:
- old walk-forward model stack
- old generic policy/planner stack
- old runtime model owner helpers
- old policy config schema

Rules:
1. Do not import modules from this tree in active runtime/backtest code.
2. Use only for forensic comparison during refactor.
3. Delete this tree when transition-cycle removal criteria are met.
