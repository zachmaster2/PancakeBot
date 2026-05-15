"""RuntimeConfig dataclass binding the round store, contract, gate, RPC poller, and runtime knobs."""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.config import StrategyConfig
from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.chain.prediction_contract import Web3PredictionContract
from pancakebot.strategy.momentum_gate import MomentumGate
from pancakebot.chain.rpc_poller import RpcPoller


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
    # Chronological order (lock - X ms; bigger X = earlier in the round):
    #   ramp_poll_1    (lock -  7.550s)   <-- Era 11 RPC poll (per-leg refactor 2026-05-12)
    #   bankroll       (lock -  6.045s)
    #   ramp_poll_2    (lock -  5.850s)   <-- Era 11 RPC poll (per-leg refactor 2026-05-12)
    #   final_rpc_poll (lock -  4.750s)   <-- Era 11 RPC poll (per-leg refactor 2026-05-12)
    #   anchor_poll    (lock -  1.300s)   <-- Bundle 5 v2 single anchor poll (200ms timeout)
    #   critical_path  (lock -  1.045s)
    #   bet_submit     (lock -  0.700s)   <-- timing-guard deadline (static fallback)
    #
    # The per-leg ramp intervals (RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS=1700,
    # RPC_RAMP_2_TO_FINAL_INTERVAL_MS=1100) are sized for each ramp's
    # actual expected workload: ramp_1 catches up an 8s-periodic-interval
    # of blocks (worst case ~18), ramp_2 is a small incremental top-up
    # (~4 blocks). Replaces the uniform RPC_RAMP_POLL_INTERVAL_MS=1500
    # that was sized for a stale 30s-periodic-era batch=15 assumption.
    #
    # At canonical pool_cutoff=6 specifically: chronology is monotonic —
    # ramp_2 (5.850s) naturally falls AFTER bankroll (6.045s). At larger
    # pool_cutoff values (≥7), final shifts earlier and ramp_2 may push
    # above bankroll's offset again; engine.py fires wakes in fixed code
    # order (ramp_1 → bankroll → ramp_2 → final) and _sleep_until_ts
    # returns immediately for past-due wakes, so runtime stays correct
    # but the wake order in wall-clock differs from the offset order.
    #
    # Bundle 5 v2 (2026-05-14): the ``ntp_sync_wake`` (formerly at
    # lock - 11.095s) is retired. The bot trusts the OS clock directly
    # (W32Time tightening per README); no application-level NTP layer.
    #
    # bankroll_wake refreshes wallet balance (live: BSC RPC; dry: in-memory).
    # ramp + final RPC polls catch up bet events from BSC via batched
    #   eth_getBlockReceipts so the critical_path snapshot is fresh.
    # critical_path_wake reads the local pool aggregate, runs the gate,
    #   and submits the bet.
    ramp_poll_1_wakeup_offset_ms: int
    ramp_poll_2_wakeup_offset_ms: int
    final_rpc_poll_wakeup_offset_ms: int
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
    # Cross-validated at config load to leave room for RPC-poll
    # completion (block availability + batched receipt fetch RTT +
    # safety) before the critical_path_wake reads the pool snapshot.
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

    # RPC poller: periodic + ramp + final batched-RPC polls of
    # PredictionV2 BetBull/BetBear events. Era 11 (2026-05-07) replaced
    # the WSS-subscription PoolEventWatcher with this deterministic
    # poll model. Same get_pool / set_round_phase / is_pool_ready
    # interface so the engine call sites are minimally affected.
    rpc_poller: RpcPoller | None = None
