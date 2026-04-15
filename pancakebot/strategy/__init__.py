"""Momentum strategy package.

This package holds the momentum-only strategy engine used by:
  - live mode
  - dry mode
  - backtest mode

Modules:
  - `momentum_gate`: OKX 1m kline signal gate + config
  - `momentum_pipeline`: MomentumOnlyPipeline (live/dry/backtest entrypoint)
"""
