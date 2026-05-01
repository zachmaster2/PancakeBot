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
from pancakebot.runtime.process_health import write_heartbeat
from pancakebot.strategy.momentum_pipeline import StrategyPipelineDecision
from pancakebot.types import Round
from time import sleep as sleep_seconds

_LOCK_SAFETY_MARGIN_SECONDS = 1  # abort bet if wall-clock is within this many seconds of lock_at

# Extra cushion added to the claim-check wake time to avoid alignment retries near RPC boundaries.
_CLAIM_CHECK_PADDING_SECONDS = 5

_CLAIM_BATCH_SIZE = 10
_BACKOFF_SECONDS = [2, 4, 8, 16, 32, 58]  # locked

_TRANSIENT_NETWORK_DELAY_SECONDS = 10
_ONE_MINUTE_MS = 60_000


# -- Clock skew compensation -------------------------------------------------
# The bot's scheduling uses chain-anchored timestamps (lock_at from BSC, in
# true UTC) but wakes/compares against ``time.time()`` (local clock). On
# Windows with stale/unsynced w32tm, local can drift seconds ahead of UTC,
# which makes the bot fire OKX kline fetches BEFORE OKX has had time to
# publish the requested candle window. Result: 75%+ of fetches return a
# lagged window, validation rejects them as ``gate_btc_unexpected_newest``,
# bot skips the round.
#
# Fix: measure (local - okx) skew via ``OkxClient.measure_clock_skew()`` once
# per round in housekeeping, cache it, and use ``_utc_now()`` everywhere we
# compare local time to a chain-anchored value. This converts all scheduling
# decisions from "wait until LOCAL time hits X" to "wait until OKX-frame time
# hits X" without changing any request CONTENT.
#
# Documented diagnosis: research/okx_lag_root_cause_clock_skew.md.
# Mechanism verification: research/okx_artificial_delay_probe.py (lag goes to
# 0ms when fired at correct true-UTC anchor).
_clock_skew_seconds: float = 0.0


def _utc_now() -> float:
    """Best-effort estimate of the current OKX/UTC second.

    Returns ``time.time() - _clock_skew_seconds``. Falls back to local time
    when ``_clock_skew_seconds`` is 0.0 (initial value, or every refresh
    failed). Caller code that compares local time against a chain-anchored
    value (lock_at, cutoff_ts, claim_ts) MUST use this instead of
    ``time.time()`` so the comparison is in OKX/UTC frame.
    """
    return time.time() - _clock_skew_seconds


def _refresh_clock_skew(gate) -> None:
    """Best-effort skew re-measurement via OKX /api/v5/public/time.

    Called once per round in the housekeeping phase (off the critical path).
    Updates the module-level ``_clock_skew_seconds`` if measurement
    succeeds. On failure, keeps the prior cached value.

    No-op when *gate* is None (backtest mode, sync mode) -- those paths
    don't care about skew.
    """
    global _clock_skew_seconds
    if gate is None:
        return
    try:
        client = getattr(gate, "_client", None)
        if client is None or not hasattr(client, "measure_clock_skew"):
            return
        new_skew = client.measure_clock_skew(samples=3)
    except Exception:  # noqa: BLE001 -- never crash the round on skew refresh
        return
    if new_skew is None:
        warn("CLOCK", "SKEW", "REFRESH",
             msg="OKX /public/time unreachable; using prior cached skew",
             cached_skew_s=f"{_clock_skew_seconds:.3f}")
        return
    delta = abs(new_skew - _clock_skew_seconds)
    prev = _clock_skew_seconds
    _clock_skew_seconds = float(new_skew)
    if delta >= 0.1 or prev == 0.0:
        # Log meaningful changes (>100ms drift) and the initial measurement.
        info("CLOCK", "SKEW", "UPDATE",
             msg=f"clock skew (local - okx) refreshed: {prev:.3f}s -> {new_skew:.3f}s")
    if abs(new_skew) >= 5.0:
        warn("CLOCK", "SKEW", "LARGE",
             msg=f"large clock skew detected: {new_skew:.3f}s. Consider running w32tm /resync.",
             skew_s=f"{new_skew:.3f}")


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

    # Bootstrap OKX clock-skew measurement BEFORE the first round starts.
    # The per-round refresh in housekeeping won't have fired yet; if local
    # clock is significantly skewed and we don't bootstrap, the first
    # round's _sleep_until_ts will fire too early in OKX frame and the
    # initial fetch will rejected by validation. See _refresh_clock_skew.
    if closed_state.strategy_pipeline is not None and hasattr(closed_state.strategy_pipeline, "_gate"):
        # noinspection PyProtectedMember
        _bootstrap_gate = closed_state.strategy_pipeline._gate
        if _bootstrap_gate is not None:
            info("CLOCK", "SKEW", "BOOT", msg="bootstrapping OKX clock skew measurement...")
            _refresh_clock_skew(_bootstrap_gate)

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
        # Per-iteration settlements are forwarded to the tracker in
        # _run_one_iteration (see record_settlement call near the end of the
        # housekeeping phase, where bankroll_bnb is freshly RPC-fetched). The
        # drawdown-from-peak gate reads from this tracker each iteration.
        # NOTE: Path is already imported at module level; do not re-import
        # locally or it shadows the module-level binding for the whole
        # function (Python locals-vs-globals scope rule).
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
                    now_ts=int(_utc_now()),  # skew-corrected: claim_scan_cursor compares to chain-anchored close timestamps
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
        # claim_ts and cutoff_ts_t are both chain-anchored true UTC; compare
        # against skew-corrected _utc_now() so a skewed local clock doesn't
        # make us miss the wake-for-claim window.
        if _utc_now() < claim_ts < cutoff_ts_t:
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
            #
            # On TransientRpcError above we SKIP this iteration and do NOT
            # update the tracker, so multi-iteration RPC outages leave the
            # tracker drifting on stale peak. Net effect is conservative: we
            # also refuse to bet (risk_bankroll_stale), so the breaker can't
            # mis-fire on stale data because we're not betting in the first
            # place. Tracker re-syncs on the next successful RPC fetch.
            if closed.strategy_pipeline is not None:
                closed.strategy_pipeline.record_settlement(
                    bankroll=bankroll_bnb,
                    start_at=int(open_round.start_at),
                )

        # Per-round REST kline fetch path: the gate fires its 4 parallel
        # OKX GETs at decision time (Phase B below). Housekeeping here is
        # purely the clock-skew refresh -- a fresh skew measurement keeps
        # the wake-time math correctly anchored to OKX/UTC frame.
        gate = None
        if closed.strategy_pipeline is not None and hasattr(closed.strategy_pipeline, "_gate"):
            # noinspection PyProtectedMember
            gate = closed.strategy_pipeline._gate
            if gate is not None:
                # Refresh OKX clock skew off the critical path. Used by
                # _utc_now() for skew-corrected scheduling. ~150-500ms;
                # absorbed comfortably in the housekeeping window.
                _refresh_clock_skew(gate)

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

        # -- Phase B: Critical path (pre-lock) --
        # Sleep until ``lock_at - kline_fetch_offset_ms`` (skew-corrected),
        # then run gate.evaluate() which fires 4 parallel /history-candles
        # GETs and computes the signal off the returned arrays. The offset
        # is sized so the fetch lands a configurable margin before lock_at;
        # default 850ms accommodates OKX p99 staleness (~1.7s) plus the
        # round-trip and signal compute. Tuned via [runtime] kline_fetch_offset_ms.
        fetch_ts = lock_ts_t - cfg.kline_fetch_offset_ms / 1000.0
        _sleep_until_ts(fetch_ts, reason="wait_for_kline_fetch", epoch=current_epoch)

        # Step 8: Decide. Gate fires 4 parallel REST fetches and computes
        # signal off the returned 1s arrays.
        t_features_start_ms = _mono_ms()
        pred_p_final = 0.5

        if closed.strategy_pipeline is None:
            raise InvariantError("strategy_pipeline_missing")
        decision = closed.strategy_pipeline.decide_open_round(
            round_t=open_round,
            pool_bull_bnb=pool_bull_bnb,
            pool_bear_bnb=pool_bear_bnb,
        )
        # `p_bull` was removed from StrategyPipelineDecision in the
        # 2026-04-26 lean&clean refactor; defensive getattr keeps the
        # audit-log path working if any future strategy emits a
        # probability-shaped decision.
        _p_bull_legacy = getattr(decision, "p_bull", None)
        if _p_bull_legacy is not None:
            pred_p_final = _p_bull_legacy
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
        # was randomly shaving 0-1 s off the budget). lock_ts_t is chain-
        # anchored true UTC; compare via _utc_now() (skew-corrected). With
        # local-only comparison + skew, the guard fires far too early in
        # true UTC and the bot self-aborts almost every round.
        if _utc_now() >= lock_ts_t - _LOCK_SAFETY_MARGIN_SECONDS:
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
            now_ts=int(_utc_now()),  # skew-corrected: claim_scan_cursor compares to chain-anchored close timestamps
            buffer_seconds=cfg.buffer_seconds,
            page_size=100,
            gas_limit=GAS_LIMIT_CLAIM,
            claim_batch_size=_CLAIM_BATCH_SIZE,
            min_bet_with_gas_bnb=cfg.min_bet_amount_bnb + GAS_COST_BET_BNB,
        )

    # Dry: settle simulated bets against oracle price.
    _dry_settle_available_bets(cfg, closed)


def _sleep_until_ts(target_ts: float, *, reason: str, epoch: int | None = None) -> None:
    """Sleep until OKX/UTC time hits ``target_ts``.

    *target_ts* is treated as a chain-anchored / OKX-frame UTC second.
    The comparison uses ``_utc_now()`` (skew-corrected) instead of
    ``time.time()`` so a skewed local clock doesn't make the bot fire
    early. See ``_clock_skew_seconds`` docs above.
    """
    remaining = target_ts - _utc_now()
    if remaining <= 0.5:
        return

    msg = f"Sleeping {int(remaining)}s ({reason})"
    if epoch is not None:
        msg = msg + f" epoch={epoch}"
    info("RUN", "LOOP", "SLEEP", msg=msg)

    while True:
        remaining2 = target_ts - _utc_now()
        if remaining2 <= 0:
            return
        sleep_seconds(min(1.0, remaining2))
        # Per-second heartbeat during long sleeps. Catches deadlocks where
        # the iteration's outer heartbeat would otherwise go stale for a
        # full round. No-op before the first _run_one_iteration has run.
        _write_heartbeat_from_ctx()
