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
    kline_cutoff_seconds: int

    # Pre-lock wake schedule (all DERIVED from pancakebot/timing_constants.py
    # at config load; not user-tunable). All in milliseconds before lock_at.
    #
    # Chronological order (lock - X ms; bigger X = earlier in the round):
    #   okx_warmup     (lock -  7.000s)   <-- OKX TLS warmup (off critical path)
    #   preflight      (lock -  5.970s)   <-- wallet balance + nonce/gas prefetch
    #   single_poll    (lock -  4.750s)   <-- Candidate C RPC catch-up (2026-06-06)
    #   anchor_poll    (lock -  1.300s)   <-- Bundle 5 v2 single anchor poll (200ms timeout)
    #   critical_path  (lock -  0.970s)
    #   bet_submit     (lock -  0.625s)   <-- timing-guard deadline (static fallback)
    #
    # Candidate C (2026-06-06): the 3-leg ramp ladder (ramp_1/ramp_2/final)
    # collapsed to ONE batched poll at the old final-poll slot. The retained
    # 8s periodic poll keeps the cursor within ~1 interval, so the single poll
    # only catches up the ~5-20 blocks since the last periodic. All offsets are
    # strictly decreasing; engine.py validates the ordering at startup.
    #
    # Bundle 5 v2 (2026-05-14): the ``ntp_sync_wake`` (formerly at
    # lock - 11.095s) is retired. The bot trusts the OS clock directly
    # (W32Time tightening per README); no application-level NTP layer.
    #
    # preflight_wake refreshes wallet balance (live: BSC RPC; dry: in-memory).
    # single_poll (Candidate C, 2026-06-06) is ONE batched eth_getBlockReceipts
    #   catch-up before the critical path (replaced the 3-leg ramp ladder); the
    #   retained 8s periodic poll keeps the snapshot fresh between rounds.
    # critical_path_wake reads the local pool aggregate, runs the gate,
    #   and submits the bet.
    single_poll_wakeup_offset_before_lock_ms: int
    preflight_wakeup_offset_before_lock_ms: int
    okx_warmup_wakeup_offset_before_lock_ms: int
    critical_path_wakeup_offset_before_lock_ms: int
    bet_submit_deadline_offset_before_lock_ms: int

    # Receipt timeouts for ``contract.bet_*_timed`` and ``contract.claim``
    # (DERIVED at runtime from ``buffer_seconds + claim_check_padding_seconds``,
    # ≈35s on canonical chain constants). Both share the same derivation:
    # how long ``wait_for_transaction_receipt`` polls before raising
    # TimeExhausted. Sized so a slow mempool inclusion is still caught
    # before the next round's wake schedule needs the runtime back.
    bet_tx_receipt_timeout_seconds: int
    claim_tx_receipt_timeout_seconds: int

    # User-tunable. Streak counter for OKX transient failures; bot
    # crashes (-> supervisor restart + Discord alert) after this many
    # consecutive `kline_fetch_transient_failure` rounds.
    max_consecutive_kline_fetch_failures: int

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
