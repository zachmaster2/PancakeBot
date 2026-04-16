"""Live/dry runtime loop: epoch handshake, cutoff-aligned decision, bet submission, and claim scan."""

from __future__ import annotations

import time
from pathlib import Path

from pancakebot.constants import (
    BNB_WEI,
    GAS_LIMIT_BET,
    GAS_LIMIT_CLAIM,
    GAS_COST_BET_BNB,
    POOL_CUTOFF_SECONDS,
)
from pancakebot.errors import InvariantError, TransientRpcError
from pancakebot.log import info, warn
from pancakebot.money import bankroll_suffix, format_bankroll, usd_suffix
from pancakebot.runtime.config import RuntimeConfig
from pancakebot import paths
from pancakebot.runtime.dry import (
    _ClosedState,
    _archive_dry_runtime_state,
    _append_jsonl,
    _dry_record_bet,
    _dry_settle_available_bets,
    _fetch_wallet_balance_bnb_with_retries,
    _init_closed_state,
    _record_dry_cycle_audit,
)
from pancakebot.runtime.live import claim_scan_cursor
from pancakebot.strategy.momentum_pipeline import StrategyPipelineDecision
from pancakebot.time import now_ts
from pancakebot.types import Round
from time import sleep as sleep_seconds

_LOCK_SAFETY_MARGIN_SECONDS = 1  # abort bet if wall-clock is within this many seconds of lock_at
_OKX_PUBLISH_DELAY_SECONDS = 0.25  # delay after cutoff to let OKX publish the candle

# Extra cushion added to the claim-check wake time to avoid alignment retries near RPC boundaries.
_CLAIM_CHECK_PADDING_SECONDS = 5

_CLAIM_BATCH_SIZE = 10
_BACKOFF_SECONDS = [2, 4, 8, 16, 32, 58]  # locked

_TRANSIENT_NETWORK_DELAY_SECONDS = 10
_ONE_MINUTE_MS = 60_000


def _fetch_current_bnb_price_usd(cfg: RuntimeConfig) -> float:
    """Fetch approximate BNB/USD price from contract (best-effort; 0.0 on failure)."""
    try:
        epoch = int(cfg.contract.current_epoch())
        rd = cfg.contract.round_data(epoch - 1)
        price = float(rd.lock_price_usd)
        return price if price > 0.0 else 0.0
    except Exception:
        return 0.0


def run_live_loop(cfg: RuntimeConfig) -> None:
    if not cfg.wallet_address:
        raise InvariantError("wallet_address_required")
    if cfg.min_bet_amount_bnb <= 0.0:
        raise InvariantError("runtime_min_bet_amount_nonpositive")
    try:
        closed_state = _init_closed_state(cfg)

        bnbusd_price = _fetch_current_bnb_price_usd(cfg)
        if cfg.dry:
            if closed_state.simulated_bankroll_bnb is None:
                raise InvariantError("dry_bankroll_uninitialized")
            bankroll_bnb = closed_state.simulated_bankroll_bnb
        else:
            bankroll_bnb = _fetch_wallet_balance_bnb_with_retries(
                cfg=cfg,
                reason="live_wallet_bootstrap",
            )
        info(
            "CORE",
            "RUN",
            "BANKROLL",
            msg=f"Starting bankroll: {format_bankroll(bankroll_bnb=bankroll_bnb, bnbusd_price=bnbusd_price)}",
        )

        while True:
            try:
                _run_one_iteration(cfg, closed_state)
            except TransientRpcError as e:
                info(
                    "CORE",
                    "RUN",
                    "RETRY",
                    msg=(
                        "Caught TransientRpcError during runtime loop: "
                        f"retrying after delay err={str(e)}"
                    ),
                )
                info(
                    "CORE",
                    "LOOP",
                    "SLEEP",
                    msg=(
                        f"duration={_TRANSIENT_NETWORK_DELAY_SECONDS}s "
                        "reason=delay_after_transient_network_error"
                    ),
                )
                sleep_seconds(_TRANSIENT_NETWORK_DELAY_SECONDS)
    finally:
        if cfg.dry:
            archived = _archive_dry_runtime_state(
                reason="shutdown_snapshot",
                move_files=True,
            )
            if archived is not None:
                info(
                    "RUN",
                    "DRY",
                    "ARCHIVE",
                    msg=f"Saved shutdown dry-state snapshot to {Path(paths.DRY_ARCHIVE_ROOT) / archived.name}",
                )


def _mono_ms() -> float:
    return time.perf_counter() * 1000.0


def _run_one_iteration(cfg: RuntimeConfig, closed: _ClosedState) -> None:
    # Alignment + cutoff anchoring can be noisy around epoch shifts. Ensure we only
    # take an action using a coherent epoch snapshot.
    while True:
        # Step 1: Epoch alignment handshake (shift-aware) with retries.
        locked_round, _open_round, current_epoch, _open_rd = _epoch_handshake(cfg, closed)
        locked_epoch = locked_round.epoch

        if locked_round.lock_price is None:
            raise InvariantError("locked_round_missing_lock_price")
        bnbusd_price = locked_round.lock_price
        if bnbusd_price <= 0.0:
            raise InvariantError("locked_round_lock_price_nonpositive")

        # Step 2: Initial claim scan (one-time) after the first successful alignment.
        if not closed.claim_scan_initialized:
            claim_scan_cursor(
                contract=cfg.contract,
                wallet_address=cfg.wallet_address,
                dry=cfg.dry,
                cursor_path=paths.LIVE_CLAIM_CURSOR_PATH,
                locked_epoch=locked_epoch,
                current_epoch=current_epoch,
                now_ts=int(now_ts()),
                buffer_seconds=cfg.buffer_seconds,
                get_close_ts=cfg.contract.close_ts,
                page_size=100,
                gas_limit=GAS_LIMIT_CLAIM,
                claim_batch_size=_CLAIM_BATCH_SIZE,
                min_bet_with_gas_bnb=cfg.min_bet_amount_bnb + GAS_COST_BET_BNB,
            )

            _dry_settle_available_bets(cfg, closed)
            closed.claim_scan_initialized = True

        # Step 3: Update strategy pipeline with the latest known settled epoch.
        if closed.strategy_pipeline is None:
            raise InvariantError("strategy_pipeline_missing")
        # Pass a stub for the most recently closed epoch (locked_epoch - 1).
        if locked_epoch > 1:
            _settled_stub = Round(
                epoch=locked_epoch - 1,
                start_at=0, lock_at=None,
                lock_price=None, close_price=None,
                position=None, failed=False, bets=(),
            )
            closed.strategy_pipeline.settle_closed_rounds(rounds=[_settled_stub])

        # Step 4: lock_ts from the handshake (immutable on-chain value).
        lock_ts_t = int(_open_round.lock_at)
        if lock_ts_t <= 0:
            raise InvariantError("lock_ts_t_invalid")

        # Step 4b: Backfill pool data on first iteration only.
        # After this, WSS subscription catches all bets in real time.
        if cfg.pool_watcher is not None and not closed.pool_backfill_done:
            round_start_ts = lock_ts_t - cfg.interval_seconds
            cfg.pool_watcher.backfill_round(round_start_ts)
            closed.pool_backfill_done = True

        # Step 5: cutoff_ts(t) = lock_ts(t) - cutoff_seconds.
        cutoff_ts_t = lock_ts_t - cfg.cutoff_seconds

        # If we missed the previous epoch's cutoff and are now targeting a newer epoch, the
        # just-closed locked epoch may become claimable before the next cutoff. In that case,
        # we must wake for claim first (no approximation).
        prev_locked_epoch = locked_round.epoch - 1
        claim_ts = locked_round.lock_at + cfg.buffer_seconds + _CLAIM_CHECK_PADDING_SECONDS
        if now_ts() < claim_ts < cutoff_ts_t:
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=prev_locked_epoch)
            return

        # -- Phase A: Housekeeping (before cutoff) --
        # Wake early to do epoch check + TLS warmup while we're still
        # waiting for the cutoff moment.  These run OUTSIDE the critical
        # timing window so they don't eat into our bet-submission budget.
        wake_ts = cutoff_ts_t - cfg.prefetch_offset_seconds
        _sleep_until_ts(wake_ts, reason="wait_for_prefetch", epoch=current_epoch)

        # Epoch quick-check: verify current_epoch hasn't shifted during sleep.
        try:
            current_epoch2 = int(cfg.contract.current_epoch())
        except TransientRpcError:
            current_epoch2 = None

        if current_epoch2 is not None and current_epoch2 != current_epoch:
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=current_epoch,
                locked_epoch=locked_epoch,
                lock_ts=lock_ts_t,
                cutoff_ts=cutoff_ts_t,
                locked_price_bnbusd=bnbusd_price,
                action="SKIP",
                decision_stage="reanchor",
                open_round=None,
                bankroll_before_action_bnb=closed.simulated_bankroll_bnb,
                bankroll_after_action_bnb=closed.simulated_bankroll_bnb,
                skip_reason=f"epoch_shift_before_decision:new_epoch={current_epoch2}",
            )
            info(
                "RUN",
                "ACT",
                "SKIP",
                msg=(
                    f"Skip epoch {current_epoch}: "
                    f"epoch_shift_before_decision:new_epoch={current_epoch2}"
                ),
            )
            continue

        if current_epoch2 is None:
            locked_round, open_round, current_epoch, _ = _epoch_handshake(cfg, closed)
            locked_epoch = locked_round.epoch
            lock_ts_t = int(open_round.lock_at)
        else:
            open_round = _open_round

        # TLS warmup: re-establish OKX keep-alive connection (dies after
        # ~60 s idle between 5-minute rounds).  Subsequent kline fetches
        # hit the warm connection (~50 ms instead of ~2 s).
        gate = None
        if closed.strategy_pipeline is not None and hasattr(closed.strategy_pipeline, "_gate"):
            gate = closed.strategy_pipeline._gate
            if gate is not None:
                gate.warmup_session()

        # Pool data from WSS subscription (no RPC needed, ~0 ms).
        pool_bull_bnb = 0.0
        pool_bear_bnb = 0.0
        if cfg.pool_watcher is not None and cfg.pool_watcher.connected:
            pool_ts_cutoff = lock_ts_t - POOL_CUTOFF_SECONDS
            pool_bull_bnb, pool_bear_bnb = cfg.pool_watcher.get_pool(
                epoch=current_epoch, max_ts=pool_ts_cutoff,
            )
            pool_total = pool_bull_bnb + pool_bear_bnb
            if pool_total > 0:
                info("POOL_WSS", "ROUND", "DATA",
                     epoch=current_epoch, pool_bnb=f"{pool_total:.4f}")
            if locked_epoch > 2:
                cfg.pool_watcher.clear_old_epochs(keep_after=locked_epoch - 2)

        # -- Phase B: Critical path (after cutoff) --
        # Sleep until cutoff + delay, then fetch -> decide -> bet.
        fetch_ts = cutoff_ts_t + _OKX_PUBLISH_DELAY_SECONDS
        _sleep_until_ts(fetch_ts, reason="wait_for_okx_publish", epoch=current_epoch)

        # Kick off kline fetches on warm connection.
        okx_kline_futures = None
        if gate is not None:
            okx_kline_futures = gate.fetch_klines_async(cutoff_ts_ms=int(cutoff_ts_t * 1000))

        # Step 8: Decide.
        t_features_start_ms = _mono_ms()
        pred_p_final = 0.5
        if cfg.dry:
            if closed.simulated_bankroll_bnb is None:
                raise InvariantError("dry_bankroll_uninitialized")
            bankroll_bnb = closed.simulated_bankroll_bnb
        else:
            bankroll_bnb = cfg.contract.wallet_balance_bnb(cfg.wallet_address)

        if closed.strategy_pipeline is None:
            raise InvariantError("strategy_pipeline_missing")
        decision = closed.strategy_pipeline.decide_open_round(
            round_t=open_round,
            bankroll_bnb=bankroll_bnb,
            allow_oracle_mode=False,
            pool_bull_bnb=pool_bull_bnb,
            pool_bear_bnb=pool_bear_bnb,
            okx_kline_futures=okx_kline_futures,
        )
        if decision.p_bull is not None:
            pred_p_final = decision.p_bull
        t_decision_ready_ms = _mono_ms()

        if decision.action != "BET":
            reason = decision.skip_reason or ""
            if reason == "":
                raise InvariantError("policy_skip_missing_reason")

            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=current_epoch,
                locked_epoch=locked_epoch,
                lock_ts=lock_ts_t,
                cutoff_ts=cutoff_ts_t,
                locked_price_bnbusd=bnbusd_price,
                action="SKIP",
                decision_stage="pipeline",
                open_round=open_round,
                bankroll_before_action_bnb=bankroll_bnb,
                bankroll_after_action_bnb=bankroll_bnb,
                decision=decision,
                skip_reason=reason,
                decision_latency_ms=t_decision_ready_ms - t_features_start_ms,
                pool_bull_bnb=pool_bull_bnb,
                pool_bear_bnb=pool_bear_bnb,
            )
            info("RUN", "ACT", "SKIP", msg=f"Skip epoch {current_epoch}: {reason}")
            # SKIP path: no time pressure, safe to log timing here.
            if gate is not None and gate.last_fetch_timing is not None:
                info("GATE", "FETCH", "TIMING", **gate.last_fetch_timing)
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 11: Execution timing guard (float precision -- int truncation
        # was randomly shaving 0-1 s off the budget).
        if time.time() >= lock_ts_t - _LOCK_SAFETY_MARGIN_SECONDS:
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=current_epoch,
                locked_epoch=locked_epoch,
                lock_ts=lock_ts_t,
                cutoff_ts=cutoff_ts_t,
                locked_price_bnbusd=bnbusd_price,
                action="SKIP",
                decision_stage="timing_guard",
                open_round=open_round,
                bankroll_before_action_bnb=bankroll_bnb,
                bankroll_after_action_bnb=bankroll_bnb,
                decision=decision,
                skip_reason="too_close_to_lock_for_bet",
                decision_latency_ms=t_decision_ready_ms - t_features_start_ms,
                pool_bull_bnb=pool_bull_bnb,
                pool_bear_bnb=pool_bear_bnb,
            )
            info(
                "RUN",
                "ACT",
                "SKIP",
                msg=f"Skip epoch {current_epoch}: too_close_to_lock_for_bet",
            )
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 12: Submit bet.
        amount_wei = int(round(decision.bet_size_bnb * BNB_WEI))
        if amount_wei <= 0:
            raise InvariantError("bet_amount_wei_nonpositive")

        tx_submit = None
        if not cfg.dry:
            gas_price_wei = cfg.contract.suggest_gas_price_wei()
            if decision.bet_side == "Bull":
                tx_submit = cfg.contract.bet_bull_timed(
                    epoch=current_epoch,
                    amount_wei=amount_wei,
                    gas_limit=GAS_LIMIT_BET,
                    gas_price_wei=gas_price_wei,
                    wait_receipt=True,
                    receipt_timeout_seconds=5,
                )
            elif decision.bet_side == "Bear":
                tx_submit = cfg.contract.bet_bear_timed(
                    epoch=current_epoch,
                    amount_wei=amount_wei,
                    gas_limit=GAS_LIMIT_BET,
                    gas_price_wei=gas_price_wei,
                    wait_receipt=True,
                    receipt_timeout_seconds=5,
                )
            else:
                raise InvariantError(f"unexpected_bet_side: {decision.bet_side}")

        # Step 13: Log bet with USD (BNB + USD suffixes).
        amount_bnb = amount_wei / BNB_WEI

        if not cfg.dry:
            bankroll_after_live = cfg.contract.wallet_balance_bnb(cfg.wallet_address)
            info(
                "RUN",
                "ACT",
                "BET",
                msg=(
                    f"Betting {amount_bnb:.4f} BNB"
                    + usd_suffix(amount_bnb=amount_bnb, bnbusd_price=bnbusd_price)
                    + f" on {decision.bet_side} for epoch {current_epoch}"
                    + bankroll_suffix(bankroll_bnb=bankroll_after_live, bnbusd_price=bnbusd_price)
                ),
            )
            if tx_submit is None:
                raise InvariantError("live_bet_submit_missing")
            receipt_confirmed_ms = (
                float(tx_submit.t_receipt_confirmed_mono_ms)
                if tx_submit.t_receipt_confirmed_mono_ms is not None
                else None
            )
            latency_record = {
                "epoch": current_epoch,
                "cutoff_ts": cutoff_ts_t,
                "t_features_start_mono_ms": t_features_start_ms,
                "t_decision_ready_mono_ms": t_decision_ready_ms,
                "t_tx_signed_mono_ms": tx_submit.t_tx_signed_mono_ms,
                "t_tx_hash_received_mono_ms": tx_submit.t_tx_hash_received_mono_ms,
                "t_receipt_confirmed_mono_ms": receipt_confirmed_ms,
                "tx_hash": tx_submit.tx_hash,
                "tx_included_block_number": tx_submit.included_block_number,
                "tx_included_block_timestamp": tx_submit.included_block_timestamp,
                "latency_features_ms": t_decision_ready_ms - t_features_start_ms,
                "latency_sign_ms": tx_submit.t_tx_signed_mono_ms - t_decision_ready_ms,
                "latency_broadcast_ms": tx_submit.t_tx_hash_received_mono_ms - tx_submit.t_tx_signed_mono_ms,
                "latency_mempool_ms": (
                    receipt_confirmed_ms - tx_submit.t_tx_hash_received_mono_ms
                    if receipt_confirmed_ms is not None
                    else None
                ),
                "latency_e2e_ms": (
                    receipt_confirmed_ms - t_features_start_ms
                    if receipt_confirmed_ms is not None
                    else None
                ),
            }
            _append_jsonl("var/live/latency.jsonl", latency_record)
        else:
            # Step 14: Dry bookkeeping (including gas proxy) + record.
            if closed.simulated_bankroll_bnb is None:
                raise InvariantError("dry_bankroll_uninitialized")

            bankroll_before_bet = closed.simulated_bankroll_bnb
            closed.simulated_bankroll_bnb -= amount_bnb + GAS_COST_BET_BNB
            bankroll_after_bet = closed.simulated_bankroll_bnb

            info(
                "RUN",
                "ACT",
                "BET",
                msg=(
                    f"Betting {amount_bnb:.4f} BNB"
                    + usd_suffix(amount_bnb=amount_bnb, bnbusd_price=bnbusd_price)
                    + f" on {decision.bet_side} for epoch {current_epoch}"
                    + bankroll_suffix(bankroll_bnb=bankroll_after_bet, bnbusd_price=bnbusd_price)
                ),
            )
            _dry_record_bet(
                cfg,
                closed,
                epoch=current_epoch,
                side=decision.bet_side,
                amount_bnb=amount_bnb,
                p_final=pred_p_final,
                expected_profit_bnb=decision.expected_profit_bnb,
                bankroll_before_bet_bnb=bankroll_before_bet,
                bankroll_after_bet_bnb=bankroll_after_bet,
            )
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=current_epoch,
                locked_epoch=locked_epoch,
                lock_ts=lock_ts_t,
                cutoff_ts=cutoff_ts_t,
                locked_price_bnbusd=bnbusd_price,
                action="BET",
                decision_stage="pipeline",
                open_round=open_round,
                bankroll_before_action_bnb=bankroll_before_bet,
                bankroll_after_action_bnb=bankroll_after_bet,
                decision=decision,
                decision_latency_ms=t_decision_ready_ms - t_features_start_ms,
                pool_bull_bnb=pool_bull_bnb,
                pool_bear_bnb=pool_bear_bnb,
            )

        # Step 14b: Deferred GATE logging -- emit AFTER bet so file I/O
        # doesn't delay bet submission in the critical path.
        if gate is not None and gate.last_fetch_timing is not None:
            info("GATE", "FETCH", "TIMING", **gate.last_fetch_timing)
        # Log signal details for dry-run visibility.
        _log_deferred_gate_signal(decision)

        # Step 15: Sleep until claim + claim scan.
        _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
        return


def _log_deferred_gate_signal(decision: StrategyPipelineDecision) -> None:
    """Log GATE signal details after bet submission (deferred from evaluate)."""
    if decision.action == "BET":
        info("GATE", "SIGNAL", "FIRE",
             side=decision.bet_side,
             strength=f"{decision.bet_size_bnb:.4f}")


def _epoch_handshake(cfg: RuntimeConfig, closed: _ClosedState) -> tuple[Round, Round, int, object]:
    """RPC-only epoch alignment.

    Returns (locked_round_stub, open_round_stub, current_epoch, open_rd)
    where open_rd is the raw RoundData for the open epoch (reusable for
    pool amounts and lock_ts, avoiding duplicate RPC calls).
    """
    for idx, delay_seconds in enumerate([0] + list(_BACKOFF_SECONDS)):
        if delay_seconds > 0:
            sleep_seconds(delay_seconds)
        try:
            current_epoch = int(cfg.contract.current_epoch())
        except TransientRpcError as e:
            warn("CORE", "LOOP", "RETRY", reason="rpc_current_epoch", attempt=idx, err=str(e))
            continue

        locked_epoch = current_epoch - 1
        if locked_epoch <= 0:
            warn("CORE", "LOOP", "RETRY", reason="locked_epoch_nonpositive", attempt=idx)
            continue

        try:
            locked_rd = cfg.contract.round_data(locked_epoch)
            open_rd = cfg.contract.round_data(current_epoch)
        except TransientRpcError as e:
            warn("CORE", "LOOP", "RETRY", reason="rpc_round_data", attempt=idx, err=str(e))
            continue

        if locked_rd.lock_ts <= 0:
            warn("CORE", "LOOP", "RETRY", reason="locked_lock_ts_zero", attempt=idx)
            continue

        locked_round = Round(
            epoch=locked_epoch,
            start_at=locked_rd.start_ts,
            lock_at=locked_rd.lock_ts,
            lock_price=locked_rd.lock_price_usd,
            close_price=None,
            position=None,
            failed=False,
            bets=(),
        )
        open_round = Round(
            epoch=current_epoch,
            start_at=open_rd.start_ts,
            lock_at=open_rd.lock_ts,
            lock_price=None,
            close_price=None,
            position=None,
            failed=False,
            bets=(),
        )
        return locked_round, open_round, current_epoch, open_rd

    raise InvariantError("epoch_handshake_exhausted")


def _sleep_and_claim(cfg: RuntimeConfig, closed: _ClosedState, claim_epoch: int) -> None:
    close_ts = int(cfg.contract.close_ts(claim_epoch))
    if close_ts <= 0:
        raise InvariantError("close_ts_invalid")

    claim_ts = close_ts + cfg.buffer_seconds + _CLAIM_CHECK_PADDING_SECONDS
    _sleep_until_ts(claim_ts, reason="wait_for_claim", epoch=claim_epoch)

    # Epoch handshake to refresh round state (both modes).
    locked_round2, _open_round2, current_epoch2, _open_rd2 = _epoch_handshake(cfg, closed)

    # Live only: claim scan to collect winnings.
    if not cfg.dry:
        claim_scan_cursor(
            contract=cfg.contract,
            wallet_address=cfg.wallet_address,
            dry=False,
            cursor_path=paths.LIVE_CLAIM_CURSOR_PATH,
            locked_epoch=locked_round2.epoch,
            current_epoch=current_epoch2,
            now_ts=int(now_ts()),
            buffer_seconds=cfg.buffer_seconds,
            get_close_ts=cfg.contract.close_ts,
            page_size=100,
            gas_limit=GAS_LIMIT_CLAIM,
            claim_batch_size=_CLAIM_BATCH_SIZE,
            min_bet_with_gas_bnb=cfg.min_bet_amount_bnb + GAS_COST_BET_BNB,
        )

    # Dry: settle simulated bets against oracle price.
    _dry_settle_available_bets(cfg, closed)


def _sleep_until_ts(target_ts: float, *, reason: str, epoch: int | None = None) -> None:
    remaining = target_ts - time.time()
    if remaining <= 0.5:
        return

    msg = f"Sleeping {int(remaining)}s ({reason})"
    if epoch is not None:
        msg = msg + f" epoch={epoch}"
    info("RUN", "LOOP", "SLEEP", msg=msg)

    while True:
        remaining2 = target_ts - time.time()
        if remaining2 <= 0:
            return
        sleep_seconds(min(1.0, remaining2))
