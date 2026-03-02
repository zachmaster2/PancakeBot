"""Production strategy package.

This package holds the single production strategy engine used by:
  - live mode
  - dry mode
  - backtest mode

Use:
  - `pancakebot.domain.strategy.dislocation_engine` for dislocation candidates
  - `pancakebot.domain.strategy.ml_candidate_adapter` for optional ML candidate
  - `pancakebot.domain.strategy.router` for candidate selection
  - `pancakebot.domain.strategy.pipeline` as the shared live/dry/backtest entrypoint
"""
