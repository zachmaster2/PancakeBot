"""Live/dry runtime loop: epoch handshake, cutoff-aligned decision, bet submission, and claim scan."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from pancakebot.constants import (
    BNB_WEI,
    GAS_LIMIT_BET,
    GAS_LIMIT_CLAIM,
    GAS_COST_BET_BNB,
    POOL_CUTOFF_SECONDS,
)
from pancakebot.util import InvariantError, TransientRpcError
from pancakebot.log import info, warn
from pancakebot.util import bankroll_suffix, format_bankroll, usd_suffix
from pancakebot.runtime.config import RuntimeConfig
from pancakebot import paths
from pancakebot.runtime.dry import (
    _ClosedState,
    _append_jsonl,
    _dry_record_bet,
    _dry_settle_available_bets,
    _fetch_wallet_balance_bnb_with_retries,
    _init_closed_state,
    _record_dry_cycle_audit,
)
from pancakebot.runtime.live import claim_scan_cursor
from pancakebot.runtime.kline_capture import record_round_decision
from pancakebot.runtime.process_health import write_heartbeat
from pancakebot.strategy.momentum_pipeline import StrategyPipelineDecision
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


# -- Heartbeat context -------------------------------------------------------
# _run_one_iteration refreshes this at the top of every tick; _sleep_until_ts
# reads it to emit a per-second heartbeat during long sleeps (deadlock
# detector). Both writes target the same heartbeat.json path -- supervisor
# only cares about mtime.

@dataclass(slots=True)
class _HeartbeatCtx:
    pid: int
    heartbeat_path: Path
    bankroll_bnb: float | None
    iteration_count: int
    last_epoch: int | None


_latest_heartbeat_ctx: _HeartbeatCtx | None = None


def _heartbeat_path_for_mode(dry: bool) -> Path:
    return Path(paths.DRY_HEARTBEAT_PATH if dry else paths.LIVE_HEARTBEAT_PATH)


def _update_heartbeat_ctx(
    *,
    dry: bool,
    bankroll_bnb: float | None,
    iteration_count: int,
    last_epoch: int | None,
) -> None:
    """Refresh the module-level heartbeat context used by _sleep_until_ts."""
    global _latest_heartbeat_ctx
    _latest_heartbeat_ctx = _HeartbeatCtx(
        pid=os.getpid(),
        heartbeat_path=_heartbeat_path_for_mode(dry),
        bankroll_bnb=bankroll_bnb,
        iteration_count=iteration_count,
        last_epoch=last_epoch,
    )


def _write_heartbeat_from_ctx() -> None:
    """Emit a heartbeat using the latest stored context. No-op if unset.

    Called from the per-second _sleep_until_ts loop. Errors are swallowed by
    write_heartbeat itself (with WARN logging); this wrapper stays quiet.
    """
    ctx = _latest_heartbeat_ctx
    if ctx is None:
        return
    write_heartbeat(
        ctx.heartbeat_path,
        pid=ctx.pid,
        ts_wall=time.time(),
        last_epoch=ctx.last_epoch,
        bankroll_bnb=ctx.bankroll_bnb,
        iteration_count=ctx.iteration_count,
    )


def _fetch_current_bnb_price_usd(cfg: RuntimeConfig) -> float:
    """Fetch approximate BNB/USD price from contract (best-effort; 0.0 on failure)."""
    # USD display fallback; any RPC/parse failure falls back to 0
    # noinspection PyBroadException
    try:
        epoch = int(cfg.contract.current_epoch())
        rd = cfg.contract.round_data(epoch - 1)
        price = float(rd.lock_price_usd)
        return price if price > 0.0 else 0.0
    except Exception:
        return 0.0


def run_realtime_loop(cfg: RuntimeConfig) -> None:
    # Wallet address is only required for live mode (signing transactions).
    # Dry mode reads from chain via public RPC, no signing needed.
    if not cfg.dry and not cfg.wallet_address:
        raise InvariantError("wallet_address_required_for_live")
    if cfg.min_bet_amount_bnb <= 0.0:
        raise InvariantError("runtime_min_bet_amount_nonpositive")
    closed_state = _init_closed_state(cfg)

    # Initialise the kline-capture background worker (decoupled JSONL writer).
    # Producer side (record_round_decision) only enqueues; the worker thread
    # owns all disk I/O. See pancakebot.runtime.kline_capture.
    from pancakebot.runtime.kline_capture import init_capture_worker
    capture_path = Path(paths.DRY_CAPTURE_PATH if cfg.dry else paths.LIVE_CAPTURE_PATH)
    init_capture_worker(capture_path)

    bnbusd_price = _fetch_current_bnb_price_usd(cfg)
    if cfg.dry:
        if closed_state.simulated_bankroll_bnb is None:
            raise InvariantError("dry_bankroll_uninitialized")
        bankroll_bnb = closed_state.simulated_bankroll_bnb
        # PersistedBankrollTracker for dry mode is already wired by
        # _init_closed_state (after bankroll resolution). No-op here.
    else:
        bankroll_bnb = _fetch_wallet_balance_bnb_with_retries(
            cfg=cfg,
            reason="live_wallet_bootstrap",
        )
        # Live mode: wire PersistedBankrollTracker now that wallet balance is known.
        # TODO: live mode only seeds the tracker at startup; it does not yet
        # update on per-round settlements. Claims are async (batched on-chain),
        # so record_settlement needs to hook into claim-confirmation events.
        # Until that's added, the risk checks run against the STARTUP bankroll
        # only -- initial bounds still work, but drawdown-from-peak won't fire.
        from pathlib import Path
        from pancakebot.bankroll_tracker import PersistedBankrollTracker
        from pancakebot import paths as _paths
        tracker = PersistedBankrollTracker(
            path=Path(_paths.LIVE_BANKROLL_HISTORY_PATH),
            initial_bankroll=bankroll_bnb,
            window_days=cfg.strategy.risk.window_days,
        )
        closed_state.strategy_pipeline.set_bankroll_tracker(tracker)
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


def _mono_ms() -> float:
    return time.perf_counter() * 1000.0


def _run_one_iteration(cfg: RuntimeConfig, closed: _ClosedState) -> None:
    # Process-health: bump the iteration counter and emit a heartbeat before we
    # start doing anything network-bound. Writes ``last_seen_epoch`` carried
    # over from the previous iteration -- gets refreshed after the handshake
    # below. _sleep_until_ts reads this context for per-second heartbeats.
    closed.iteration_count += 1
    _update_heartbeat_ctx(
        dry=cfg.dry,
        bankroll_bnb=closed.simulated_bankroll_bnb,
        iteration_count=closed.iteration_count,
        last_epoch=closed.last_seen_epoch,
    )
    _write_heartbeat_from_ctx()

    # Alignment + cutoff anchoring can be noisy around epoch shifts. Ensure we only
    # take an action using a coherent epoch snapshot.
    while True:
        # Step 1: Epoch alignment handshake (shift-aware) with retries.
        locked_round, _open_round, current_epoch, _open_rd = _epoch_handshake(cfg)
        locked_epoch = locked_round.epoch

        # Process-health: once the handshake gives us a current_epoch, refresh
        # the heartbeat context so sleep-loop heartbeats carry fresh state and
        # the crash handler can point at the epoch the bot was on.
        closed.last_seen_epoch = current_epoch
        _update_heartbeat_ctx(
            dry=cfg.dry,
            bankroll_bnb=closed.simulated_bankroll_bnb,
            iteration_count=closed.iteration_count,
            last_epoch=current_epoch,
        )

        # Sync round-phase state into pool_watcher immediately after handshake.
        # This triggers the initial backfill within seconds of WSS subscription
        # rather than after the prefetch sleep, giving the first cycle accurate
        # pool data without delay.
        if cfg.pool_watcher is not None:
            cfg.pool_watcher.set_round_phase(
                current_epoch=current_epoch,
                lock_at=int(_open_round.lock_at),
            )

        if locked_round.lock_price is None:
            raise InvariantError("locked_round_missing_lock_price")
        bnbusd_price = locked_round.lock_price
        if bnbusd_price <= 0.0:
            raise InvariantError("locked_round_lock_price_nonpositive")

        # Step 2: Initial claim scan (one-time, live only) after first alignment.
        if not closed.claim_scan_initialized:
            if not cfg.dry:
                claim_scan_cursor(
                    contract=cfg.contract,
                    wallet_address=cfg.wallet_address,
                    dry=False,
                    cursor_path=paths.LIVE_CLAIM_CURSOR_PATH,
                    locked_epoch=locked_epoch,
                    current_epoch=current_epoch,
                    now_ts=int(time.time()),
                    buffer_seconds=cfg.buffer_seconds,
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
        if _open_round.lock_at is None:
            raise InvariantError("open_round_lock_at_missing")
        lock_ts_t = int(_open_round.lock_at)
        if lock_ts_t <= 0:
            raise InvariantError("lock_ts_t_invalid")

        # Step 5: cutoff_ts(t) = lock_ts(t) - cutoff_seconds.
        cutoff_ts_t = lock_ts_t - cfg.cutoff_seconds

        # If we missed the previous epoch's cutoff and are now targeting a newer epoch, the
        # just-closed locked epoch may become claimable before the next cutoff. In that case,
        # we must wake for claim first (no approximation).
        prev_locked_epoch = locked_round.epoch - 1
        if locked_round.lock_at is None:
            raise InvariantError("locked_round_lock_at_missing")
        claim_ts = locked_round.lock_at + cfg.buffer_seconds + _CLAIM_CHECK_PADDING_SECONDS
        if time.time() < claim_ts < cutoff_ts_t:
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
            locked_round, open_round, current_epoch, _ = _epoch_handshake(cfg)
            locked_epoch = locked_round.epoch
            if open_round.lock_at is None:
                raise InvariantError("open_round_lock_at_missing")
            lock_ts_t = int(open_round.lock_at)
        else:
            open_round = _open_round

        # -- Housekeeping bankroll resolution (moved from critical path) --
        # Live mode fetches wallet balance from RPC BEFORE the cutoff so the
        # post-cutoff critical path doesn't pay that 50-500ms latency. The
        # tracker is fed the freshest value so risk gates + decide_open_round
        # see it. On TransientRpcError we refuse to bet on stale bankroll and
        # SKIP the iteration. Dry mode reads from in-memory state (no latency).
        if cfg.dry:
            if closed.simulated_bankroll_bnb is None:
                raise InvariantError("dry_bankroll_uninitialized")
            bankroll_bnb = closed.simulated_bankroll_bnb
        else:
            try:
                bankroll_bnb = cfg.contract.wallet_balance_bnb(cfg.wallet_address)
            except TransientRpcError as e:
                # Last-known tracker value for audit snapshot; 0.0 if unwired.
                last_known_bankroll = 0.0
                if closed.strategy_pipeline is not None:
                    # noinspection PyProtectedMember
                    _tracker = closed.strategy_pipeline._bankroll_tracker
                    if _tracker is not None:
                        last_known_bankroll = _tracker.current_bankroll()
                warn("RUN", "WALLET", "STALE",
                     msg=f"Skip epoch {current_epoch}: risk_bankroll_stale err={e}")
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
                    bankroll_before_action_bnb=last_known_bankroll,
                    bankroll_after_action_bnb=last_known_bankroll,
                    skip_reason="risk_bankroll_stale",
                )
                info("RUN", "ACT", "SKIP",
                     msg=f"Skip epoch {current_epoch}: risk_bankroll_stale")
                _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
                return
            # Forward freshest bankroll to tracker (live only; dry records its
            # own settlements via dry.py after credit/debit). Risk gates read
            # from the tracker in decide_open_round below.
            if closed.strategy_pipeline is not None:
                closed.strategy_pipeline.record_settlement(
                    bankroll=bankroll_bnb,
                    start_at=int(open_round.start_at),
                )

        # TLS warmup: re-establish OKX keep-alive connection (dies after
        # ~60 s idle between 5-minute rounds).  Subsequent kline fetches
        # hit the warm connection (~50 ms instead of ~2 s).
        gate = None
        if closed.strategy_pipeline is not None and hasattr(closed.strategy_pipeline, "_gate"):
            # noinspection PyProtectedMember
            gate = closed.strategy_pipeline._gate
            if gate is not None:
                gate.warmup_session()

        # Pool data from WSS subscription (no RPC needed, ~0 ms).
        pool_bull_bnb = 0.0
        pool_bear_bnb = 0.0
        if cfg.pool_watcher is not None and cfg.pool_watcher.connected:
            # Gate: if backfill is still running, our pool data is incomplete.
            # Skip rather than decide on partial data. bankroll_bnb was already
            # resolved in the housekeeping block above -- reuse it for audit.
            if not cfg.pool_watcher.is_backfill_done():
                warn("POOL_WSS", "BKFILL", "INCOMPL",
                     msg=f"Skip epoch {current_epoch}: backfill_incomplete")
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
                    skip_reason="backfill_incomplete",
                )
                info("RUN", "ACT", "SKIP",
                     msg=f"Skip epoch {current_epoch}: backfill_incomplete")
                _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
                return
            pool_ts_cutoff = lock_ts_t - POOL_CUTOFF_SECONDS
            pool_bull_bnb, pool_bear_bnb = cfg.pool_watcher.get_pool(
                epoch=current_epoch, max_ts=pool_ts_cutoff,
            )
            pool_total = pool_bull_bnb + pool_bear_bnb
            if pool_total > 0:
                info("POOL_WSS", "ROUND", "DATA",
                     epoch=current_epoch, pool_bnb=f"{pool_total:.4f}",
                     endpoint=cfg.pool_watcher.current_endpoint)
        elif cfg.pool_watcher is not None:
            info("POOL_WSS", "ROUND", "DISC",
                 epoch=current_epoch,
                 endpoint=cfg.pool_watcher.current_endpoint,
                 last_ok=f"{cfg.pool_watcher.last_connected_at:.0f}")

        # -- Phase B: Critical path (after cutoff) --
        # Sleep until cutoff + delay, then fetch -> decide -> bet.
        fetch_ts = cutoff_ts_t + _OKX_PUBLISH_DELAY_SECONDS
        _sleep_until_ts(fetch_ts, reason="wait_for_okx_publish", epoch=current_epoch)

        # Kick off kline fetches on warm connection.
        okx_kline_futures = None
        if gate is not None:
            okx_kline_futures = gate.fetch_klines_async(cutoff_ts_ms=int(cutoff_ts_t * 1000))

        # Step 8: Decide.  bankroll_bnb + tracker record_settlement were both
        # resolved in the housekeeping block above -- nothing on the critical
        # path between cutoff and lock reads from RPC.
        t_features_start_ms = _mono_ms()
        pred_p_final = 0.5

        if closed.strategy_pipeline is None:
            raise InvariantError("strategy_pipeline_missing")
        decision = closed.strategy_pipeline.decide_open_round(
            round_t=open_round,
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
            # Off-critical-path observability: enqueue this round's
            # snapshot to the capture worker. Sub-ms producer call;
            # disk I/O happens on the worker thread.
            record_round_decision(
                epoch=current_epoch,
                lock_at_unix=lock_ts_t,
                cutoff_ms=int(cutoff_ts_t * 1000),
                mode="dry" if cfg.dry else "live",
                gate=gate,
                decision="SKIP",
                skip_reason=reason,
                selected_strategy=decision.selected_strategy,
                bet_side=None,
                bet_size_bnb=None,
                pool_bull_bnb=pool_bull_bnb,
                pool_bear_bnb=pool_bear_bnb,
            )
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
            # Capture: gate fired but ran out of time. Record the
            # would-have-bet snapshot for replay analysis.
            record_round_decision(
                epoch=current_epoch,
                lock_at_unix=lock_ts_t,
                cutoff_ms=int(cutoff_ts_t * 1000),
                mode="dry" if cfg.dry else "live",
                gate=gate,
                decision="SKIP",
                skip_reason="too_close_to_lock_for_bet",
                selected_strategy=decision.selected_strategy,
                bet_side=decision.bet_side,
                bet_size_bnb=decision.bet_size_bnb,
                pool_bull_bnb=pool_bull_bnb,
                pool_bear_bnb=pool_bear_bnb,
            )
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 12: Submit bet.
        if decision.bet_side is None:
            raise InvariantError("decision_bet_side_missing")
        bet_side: str = decision.bet_side
        computed_amount_wei = int(round(decision.bet_size_bnb * BNB_WEI))
        if computed_amount_wei <= 0:
            raise InvariantError("bet_amount_wei_nonpositive")

        # Live safety: if min_bet_only is set, clamp the submitted amount to
        # the contract minimum.  All strategy logic runs normally; only the
        # on-chain bet size is reduced.  Audit logs record both sizes.
        amount_wei = computed_amount_wei
        if not cfg.dry and cfg.live_min_bet_only:
            min_wei = int(round(cfg.min_bet_amount_bnb * BNB_WEI))
            amount_wei = min_wei
            info("RUN", "ACT", "CLAMP",
                 msg=f"min_bet_only: clamping {computed_amount_wei / BNB_WEI:.4f} -> {amount_wei / BNB_WEI:.4f} BNB")

        tx_submit = None
        if not cfg.dry:
            gas_price_wei = cfg.contract.suggest_gas_price_wei()
            if bet_side == "Bull":
                tx_submit = cfg.contract.bet_bull_timed(
                    epoch=current_epoch,
                    amount_wei=amount_wei,
                    gas_limit=GAS_LIMIT_BET,
                    gas_price_wei=gas_price_wei,
                    wait_receipt=True,
                    receipt_timeout_seconds=5,
                )
            elif bet_side == "Bear":
                tx_submit = cfg.contract.bet_bear_timed(
                    epoch=current_epoch,
                    amount_wei=amount_wei,
                    gas_limit=GAS_LIMIT_BET,
                    gas_price_wei=gas_price_wei,
                    wait_receipt=True,
                    receipt_timeout_seconds=5,
                )
            else:
                raise InvariantError(f"unexpected_bet_side: {bet_side}")

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
                    + f" on {bet_side} for epoch {current_epoch}"
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
                    + f" on {bet_side} for epoch {current_epoch}"
                    + bankroll_suffix(bankroll_bnb=bankroll_after_bet, bnbusd_price=bnbusd_price)
                ),
            )
            _dry_record_bet(
                closed,
                epoch=current_epoch,
                side=bet_side,
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

        # Capture: BET path. Snapshot enqueued sub-ms; worker thread
        # owns the disk write.
        record_round_decision(
            epoch=current_epoch,
            lock_at_unix=lock_ts_t,
            cutoff_ms=int(cutoff_ts_t * 1000),
            mode="dry" if cfg.dry else "live",
            gate=gate,
            decision="BET",
            skip_reason=None,
            selected_strategy=decision.selected_strategy,
            bet_side=bet_side,
            bet_size_bnb=amount_bnb,
            pool_bull_bnb=pool_bull_bnb,
            pool_bear_bnb=pool_bear_bnb,
        )

        # Step 15: Sleep until claim + claim scan.
        _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
        return


def _log_deferred_gate_signal(decision: StrategyPipelineDecision) -> None:
    """Log GATE signal details after bet submission (deferred from evaluate)."""
    if decision.action == "BET":
        info("GATE", "SIGNAL", "FIRE",
             side=decision.bet_side,
             strength=f"{decision.bet_size_bnb:.4f}")


def _epoch_handshake(cfg: RuntimeConfig) -> tuple[Round, Round, int, object]:
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
    locked_round2, _open_round2, current_epoch2, _open_rd2 = _epoch_handshake(cfg)

    # Live only: claim scan to collect winnings.
    if not cfg.dry:
        claim_scan_cursor(
            contract=cfg.contract,
            wallet_address=cfg.wallet_address,
            dry=False,
            cursor_path=paths.LIVE_CLAIM_CURSOR_PATH,
            locked_epoch=locked_round2.epoch,
            current_epoch=current_epoch2,
            now_ts=int(time.time()),
            buffer_seconds=cfg.buffer_seconds,
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
        # Per-second heartbeat during long sleeps. Catches deadlocks where
        # the iteration's outer heartbeat would otherwise go stale for a
        # full round. No-op before the first _run_one_iteration has run.
        _write_heartbeat_from_ctx()
