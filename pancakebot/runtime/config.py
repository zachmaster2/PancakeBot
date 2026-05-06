"""RuntimeConfig dataclass binding the round store, contract, gate, pool watcher, and runtime knobs."""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.config import StrategyConfig
from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.chain.prediction_contract import Web3PredictionContract
from pancakebot.strategy.momentum_gate import MomentumGate
from pancakebot.chain.pool_watcher import PoolEventWatcher


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    # Closed rounds store (JSONL; used by backtest only; None in live/dry)
    round_store: ClosedRoundsStore | None

    # Momentum strategy config (always present; MomentumGateConfig)
    momentum_gate_config: object

    # Momentum gate (OKX 1s live client; None in backtest mode)
    momentum_gate: MomentumGate | None

    # On-chain / identity (backtest passes a stub; dry/live use real contract)
    contract: Web3PredictionContract
    wallet_address: str

    # Feature cutoff
    cutoff_seconds: int

    # Pre-lock wake schedule (all DERIVED from pancakebot/timing_constants.py
    # at config load; not user-tunable). All in milliseconds before lock_at.
    #
    #   ntp_sync_wakeup_offset_ms       >  bankroll_wakeup_offset_ms
    #                                   >  critical_path_wakeup_offset_ms
    #                                   >  bet_submit_deadline_offset_ms
    #
    # Engine fires three distinct _sleep_until_ts wakes per round:
    # ntp_sync -> bankroll -> critical_path. Timing guard fires at
    # ``lock_at - bet_submit_deadline_offset_ms``.
    #
    # The ntp_sync_wake forces a fresh NTP query; the bankroll_wake
    # refreshes wallet balance (live: BSC RPC; dry: in-memory). Both
    # land WELL before the critical path -- 5 second gaps deliberately
    # generous against environmental drift.
    #
    # The critical_path_wake is the SINGLE entry point for the
    # bet-decision sequence. Inside the wake the engine sequences:
    # pool snapshot (~5ms in-memory) -> kline gate.evaluate() (~340ms
    # parallel REST + signal compute) -> bet submit (~700ms BSC RTT
    # + block budget). Prior architecture used a separate pool_read
    # wake 5ms ahead of kline_fetch wake; that 5ms gap was sequential
    # operation time, not a scheduled event, and is now correctly
    # absorbed inside the critical_path wake.
    ntp_sync_wakeup_offset_ms: int
    bankroll_wakeup_offset_ms: int
    critical_path_wakeup_offset_ms: int
    bet_submit_deadline_offset_ms: int

    # Receipt timeouts for ``contract.bet_*_timed`` and ``contract.claim``
    # (DERIVED at runtime from ``buffer_seconds + claim_check_padding_seconds``,
    # ≈35s on canonical chain constants). Both share the same derivation:
    # how long ``wait_for_transaction_receipt`` polls before raising
    # TimeExhausted. Sized so a slow mempool inclusion is still caught
    # before the next round's wake schedule needs the runtime back.
    bet_tx_receipt_timeout_seconds: int
    claim_tx_receipt_timeout_seconds: int

    # Selected publish-delay tier from the config-load tier ladder:
    # ``"P99"`` (strict; full-inclusion guarantee that the cutoff candle
    # is published at fetch time) or ``"P95"`` (operating budget; ~5%
    # publish-delay tail absorbed by the streak counter). Surfaced for
    # operator visibility -- logged at startup of the live/dry runtime
    # loop. Set by ``pancakebot/config.py:load_app_config`` via the
    # tier-ladder cross-validation (P99 first, P95 fallback).
    kline_publish_tier: str

    # User-tunable. Streak counter for OKX transient failures; bot
    # crashes (-> supervisor restart + Discord alert) after this many
    # consecutive `kline_fetch_transient_failure` rounds.
    max_consecutive_fetch_failures: int

    # User-tunable. Pool cutoff: only bets with on-chain block_timestamp
    # < lock_at - pool_cutoff_seconds are counted in the pool aggregate.
    # Cross-validated at config load to be >= pool_read_wakeup_offset_ms
    # + WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS.
    pool_cutoff_seconds: int

    # Protocol constants (from chain via contract_constants.json)
    min_bet_amount_bnb: float
    treasury_fee_fraction: float
    interval_seconds: int
    buffer_seconds: int

    # Dry-mode initial bankroll
    dry_initial_bankroll_bnb: float | None

    # Execution
    dry: bool

    # Live: clamp all bet sizes to contract minimum for safe testing
    live_min_bet_only: bool

    # Fresh start: archive existing dry state before starting
    dry_fresh_start: bool

    # No-archive: delete (don't archive) existing dry state on --fresh
    dry_no_archive: bool

    # Strategy config (10 knobs; loaded from config.toml [strategy.*] sections)
    strategy: StrategyConfig

    # Pool event watcher: accumulates BetBull/BetBear events for accurate pools
    pool_watcher: PoolEventWatcher | None = None
