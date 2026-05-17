"""Live/dry runtime loop: epoch handshake, cutoff-aligned decision, bet submission, and claim scan."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from pancakebot.constants import (
    BNB_WEI,
    BACKTEST_GAS_LIMIT_BET,
    BACKTEST_GAS_LIMIT_CLAIM,
    BACKTEST_GAS_COST_BET_BNB,
    RETRY_BACKOFF_SECONDS,
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
from pancakebot.chain.rpc_poller import (
    AnchorState,
    compute_submit_deadline_ms,
    predict_predecessor_milli_ts,
)
from pancakebot import timing_constants as _tc
from pancakebot.runtime.process_health import write_heartbeat
from pancakebot.strategy.momentum_pipeline import StrategyPipelineDecision
from pancakebot.types import Round
from time import sleep as sleep_seconds

# Extra cushion added to the claim wake time, on top of the chain's
# ``buffer_seconds`` settlement window. Total claim-wake time =
# ``close_at(claim_epoch) + buffer_seconds + _CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS``
# (NOT relative to lock_at -- the bot only claims after the round closes
# and the keeper has had ``buffer_seconds`` to call settleRound). The
# padding absorbs alignment retries near RPC boundaries.
_CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS = 5

_CLAIM_BATCH_SIZE = 10

_TRANSIENT_NETWORK_DELAY_SECONDS = 10


# -- NTP clock sync ----------------------------------------------------------
# The bot's scheduling uses chain-anchored timestamps (lock_at from BSC, in
# Time source (Bundle 5 v2, 2026-05-14): the bot trusts the OS clock
# directly. Previously the bot maintained its own per-round NTP query
# (``NtpSync`` in ``pancakebot/runtime/ntp_sync.py``) that measured
# ``(local - ntp)`` once per round and applied the correction inside
# ``_utc_now()``. That layer was a workaround for Windows Time Service's
# default 1024s poll cadence, which let the local clock drift up to
# ~270ms (P95) between syncs — too sloppy for sub-second bet timing.
#
# The W32Time prerequisite documented in README.md tightens
# ``MaxPollInterval`` to 5 (= 32s), bringing residual drift well under
# 10ms (P95). With that in place the application-level NTP layer is
# redundant — ``time.time()`` is the authoritative truth source, and
# ``_utc_now()`` is a thin alias preserved for readability at the
# call sites that compare local time to chain-anchored values
# (lock_at, cutoff_ts, claim_ts).
#
# If the W32Time tightening is NOT applied, the bot's timing budgets
# may be too tight; the operator is responsible for verifying via
# ``w32tm /query /status`` (expected ``Poll Interval: 5``).


def _utc_now() -> float:
    """Current wallclock seconds. Trusts the OS clock (kept tight by
    W32Time per the README setup steps). Preserved as a separate
    function from ``time.time()`` so callers that compare local time
    against chain-anchored values remain self-documenting."""
    return time.time()


def _kline_timing_get(gate, key: str) -> int | None:
    """Safe lookup into ``gate.last_fetch_timing[key]``.

    Returns ``None`` when ``gate`` is itself None (non-strategy runs) or
    when ``last_fetch_timing`` hasn't been populated yet (cold-start /
    rounds before the gate runs). Cycle-audit code persists None as an
    empty string in the CSV.
    """
    if gate is None:
        return None
    timing = gate.last_fetch_timing
    if timing is None:
        return None
    return timing.get(key)


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


def _log_runtime_timing_summary(cfg: RuntimeConfig) -> None:
    """Emit one INFO line summarizing the timing config in effect.

    Operators read this at startup to confirm which wake offsets and
    deadlines are derived from the current ``timing_constants.py`` values
    without having to derive the math from raw constants themselves.
    """
    info(
        "CORE", "RUN", "TIMING",
        msg=(
            f"timing config: kline_cutoff={cfg.kline_cutoff_seconds}s "
            f"pool_cutoff={cfg.pool_cutoff_seconds}s "
            f"ramp_poll_1_wakeup={cfg.ramp_poll_1_wakeup_offset_before_lock_ms}ms "
            f"bankroll_wakeup={cfg.bankroll_wakeup_offset_before_lock_ms}ms "
            f"ramp_poll_2_wakeup={cfg.ramp_poll_2_wakeup_offset_before_lock_ms}ms "
            f"final_rpc_poll_wakeup={cfg.final_rpc_poll_wakeup_offset_before_lock_ms}ms "
            f"critical_path_wakeup={cfg.critical_path_wakeup_offset_before_lock_ms}ms "
            f"bet_submit_deadline={cfg.bet_submit_deadline_offset_before_lock_ms}ms "
            f"bet_tx_receipt_timeout={cfg.bet_tx_receipt_timeout_seconds}s "
            f"claim_tx_receipt_timeout={cfg.claim_tx_receipt_timeout_seconds}s"
        ),
    )


def run_realtime_loop(cfg: RuntimeConfig) -> None:
    # Wallet address is only required for live mode (signing transactions).
    # Dry mode reads from chain via public RPC, no signing needed.
    if not cfg.dry and not cfg.wallet_address:
        raise InvariantError("wallet_address_required_for_live")
    if cfg.min_bet_amount_bnb <= 0.0:
        raise InvariantError("runtime_min_bet_amount_nonpositive")

    _log_runtime_timing_summary(cfg)

    closed_state = _init_closed_state(cfg)

    # Bundle 5 v2 (2026-05-14): no application-level NTP bootstrap. The
    # bot trusts the OS clock (Windows Time Service kept tight via
    # MaxPollInterval=5; see README "W32Time prerequisite"). The prior
    # NtpSync bootstrap + per-round refresh was retired alongside the
    # continuous fine-phase chain-anchor poll — both layers existed to
    # paper over W32Time's default 1024s poll cadence; the W32Time
    # tightening obviates them.

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
            drawdown_peak_window_days=cfg.strategy.risk.drawdown_peak_window_days,
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

        # Sync round-phase state into rpc_poller immediately after handshake.
        # Bundle 2 (2026-05-13): on the first call this synchronously initializes
        # the cursor from chain head (~1 RPC, sub-second) but does NOT block on
        # backfill — the periodic daemon's first tick + the ramp/final polls
        # drive the in-round catch-up. is_pool_ready below gates against acting
        # on a half-built pool aggregate via the cold_start_in_progress reason.
        if cfg.rpc_poller is not None:
            cfg.rpc_poller.set_round_phase(
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
                    gas_limit=BACKTEST_GAS_LIMIT_CLAIM,
                    claim_batch_size=_CLAIM_BATCH_SIZE,
                    min_bet_with_gas_bnb=cfg.min_bet_amount_bnb + BACKTEST_GAS_COST_BET_BNB,
                    claim_tx_receipt_timeout_seconds=cfg.claim_tx_receipt_timeout_seconds,
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

        # Step 5: cutoff_ts(t) = lock_ts(t) - kline_cutoff_seconds.
        cutoff_ts_t = lock_ts_t - cfg.kline_cutoff_seconds

        # Open-round handle. Iteration-stable since _open_round comes from
        # the handshake at the top of this iteration; epoch state is not
        # re-checked on the critical path.
        open_round = _open_round

        # Gate handle (used downstream for last_fetch_timing logging on SKIP).
        gate = None
        if closed.strategy_pipeline is not None and hasattr(closed.strategy_pipeline, "_gate"):
            # noinspection PyProtectedMember
            gate = closed.strategy_pipeline._gate

        # If we missed the previous epoch's cutoff and are now targeting a
        # newer epoch, the previously-locked epoch (which just closed) may
        # become claimable before the next cutoff. In that case, we must
        # wake for claim first (no approximation).
        prev_locked_epoch = locked_round.epoch - 1
        if locked_round.lock_at is None:
            raise InvariantError("locked_round_lock_at_missing")
        # PredictionV2 rounds are tiled: each lock event closes the prior
        # round AND opens the next. So ``locked_round.lock_at`` (epoch T-1's
        # lock_at) IS the close_at of ``prev_locked_epoch`` (= epoch T-2).
        # Claim wake fires at: close_at(prev) + buffer + padding.
        prev_close_ts = locked_round.lock_at  # = close_at(prev_locked_epoch)
        claim_ts = prev_close_ts + cfg.buffer_seconds + _CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS
        # claim_ts and cutoff_ts_t are both chain-anchored true UTC; compare
        # against NTP-corrected _utc_now() so a drifted local clock doesn't
        # make us miss the wake-for-claim window.
        if _utc_now() < claim_ts < cutoff_ts_t:
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=prev_locked_epoch)
            return

        # Bundle 5 v2 (2026-05-14): the per-round NTP sync wake is gone.
        # Previously the bot woke at ``lock - 11095ms`` to refresh its own
        # ``(local - ntp)`` offset measurement. The W32Time prerequisite
        # (MaxPollInterval=5, see README) keeps the OS clock within a
        # few ms of NTP truth directly, so there is no application-level
        # NTP layer to refresh. The first pre-lock wake is now
        # ``wait_for_ramp_poll_1`` (= lock - 7550ms).

        # -- Ramp poll #1 (Era 11) --
        # All three polls (ramp_1, ramp_2, final) take a deadline_ms
        # and skip-on-miss. The label "ramp" is 4 chars, fits the
        # log SUB_W=6.
        # First of three RPC polls that ramp the local pool aggregate
        # toward the critical_path snapshot. Fires at lock_at -
        # ramp_poll_1_wakeup_offset_before_lock_ms (= ~7.500s before lock at canonical
        # pool_cutoff=6 per the per-leg refactor 2026-05-12). Catches
        # blocks since the last periodic poll. deadline_ms = gap to
        # ramp_2 - safety; on RTT-exceeds-deadline the poll marks
        # _last_poll_too_slow=True for diagnostics, but is_pool_ready()
        # only returns False when the round-aware feasibility check
        # has flagged the round.
        if cfg.rpc_poller is not None:
            ramp_poll_1_wake_ts = (
                lock_ts_t - cfg.ramp_poll_1_wakeup_offset_before_lock_ms / 1000.0
            )
            _sleep_until_ts(
                ramp_poll_1_wake_ts,
                reason="wait_for_ramp_poll_1",
                epoch=current_epoch,
            )
            ramp_1_deadline_ms = max(
                0,
                cfg.ramp_poll_1_wakeup_offset_before_lock_ms
                - cfg.ramp_poll_2_wakeup_offset_before_lock_ms
                - 200,  # safety
            )
            cfg.rpc_poller.poll_ramp(deadline_ms=ramp_1_deadline_ms)

        # -- Bankroll wake --
        # Refreshes wallet balance so risk gates + decide_open_round
        # see fresh truth. Live mode does a BSC RPC call (~50-200ms
        # p99); dry mode reads in-memory simulated bankroll (sub-ms).
        # Generously off the critical path: 5000ms wake budget. On
        # live RPC error, SKIP the iteration with risk_bankroll_stale
        # rather than betting on stale value.
        bankroll_wake_ts = lock_ts_t - cfg.bankroll_wakeup_offset_before_lock_ms / 1000.0
        _sleep_until_ts(
            bankroll_wake_ts,
            reason="wait_for_bankroll",
            epoch=current_epoch,
        )
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
            # Forward freshest bankroll to tracker (live only; dry records
            # its own settlements via dry.py after credit/debit). Risk
            # gates read from the tracker in decide_open_round below.
            #
            # On TransientRpcError above we SKIP and do NOT update the
            # tracker, so multi-iteration RPC outages leave it on stale
            # peak. Net effect is conservative: we also refuse to bet
            # (risk_bankroll_stale), so the breaker can't mis-fire on
            # stale data because we're not betting in the first place.
            # Tracker re-syncs on the next successful RPC fetch.
            if closed.strategy_pipeline is not None:
                closed.strategy_pipeline.record_settlement(
                    bankroll=bankroll_bnb,
                    start_at=int(open_round.start_at),
                )

        # -- Ramp poll #2 (Era 11) --
        # Second RPC poll. Fires at lock_at -
        # ramp_poll_2_wakeup_offset_before_lock_ms (= ~5.800s at canonical
        # pool_cutoff=6 per the per-leg refactor 2026-05-12; naturally
        # falls after bankroll completes). Bridges
        # ramp_1 -> final. deadline_ms = gap to final_poll - safety.
        if cfg.rpc_poller is not None:
            ramp_poll_2_wake_ts = (
                lock_ts_t - cfg.ramp_poll_2_wakeup_offset_before_lock_ms / 1000.0
            )
            _sleep_until_ts(
                ramp_poll_2_wake_ts,
                reason="wait_for_ramp_poll_2",
                epoch=current_epoch,
            )
            ramp_2_deadline_ms = max(
                0,
                cfg.ramp_poll_2_wakeup_offset_before_lock_ms
                - cfg.final_rpc_poll_wakeup_offset_before_lock_ms
                - 200,  # safety
            )
            cfg.rpc_poller.poll_ramp(deadline_ms=ramp_2_deadline_ms)

        # -- Final RPC poll (Era 11) --
        # Last RPC poll before critical_path reads the pool snapshot.
        # Fires at lock_at - final_rpc_poll_wakeup_offset_before_lock_ms (= ~4.7s
        # at canonical pool_cutoff=6 post 2026-05-12 refactor; was ~3.79s
        # before lock). Catches blocks since ramp_2. deadline_ms = gap
        # to critical_path - safety. Same skip-on-too-slow contract as
        # ramp polls.
        if cfg.rpc_poller is not None:
            final_rpc_poll_wake_ts = (
                lock_ts_t - cfg.final_rpc_poll_wakeup_offset_before_lock_ms / 1000.0
            )
            _sleep_until_ts(
                final_rpc_poll_wake_ts,
                reason="wait_for_final_rpc_poll",
                epoch=current_epoch,
            )
            final_deadline_ms = max(
                0,
                cfg.final_rpc_poll_wakeup_offset_before_lock_ms
                - cfg.critical_path_wakeup_offset_before_lock_ms
                - 200,  # safety
            )
            cfg.rpc_poller.poll_final(deadline_ms=final_deadline_ms)

        # -- Anchor poll + critical-path wake (Bundle 5 v2, 2026-05-14) --
        #
        # Strategy:
        # 1. Sleep to lock - ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS (= lock - 1300ms).
        # 2. Fire ONE eth_getBlockByNumber('latest') with a 200ms timeout.
        # 3. If response decodes to a valid BEP-520 anchor:
        #    - Compute dynamic wake (predecessor.milli_ts - 557ms)
        #    - Compute dynamic submit deadline (set aside for the bet
        #      timing guard below).
        #    - Use the dynamic wake (closer to lock than static).
        # 4. If response is None (timeout / malformed):
        #    - Fall back to static wake (= lock - critical_path_wakeup_offset_before_lock_ms)
        #      and static submit deadline (= lock - bet_submit_deadline_offset_before_lock_ms).
        # 5. Sleep until the resolved critical_path_wake_ts.
        #
        # The anchor lives only for THIS round; ``round_anchor`` is the
        # engine-local handoff between the wake math and the later
        # bet-submit deadline gate. No persistent anchor state on RpcPoller.
        #
        # Replaces Bundle 4's continuous fine-phase head poller (~15-18
        # RPC calls per round) with one anchor poll per round.
        lock_ms_int = int(round(lock_ts_t * 1000))
        static_critical_path_wake_ts = (
            lock_ts_t - cfg.critical_path_wakeup_offset_before_lock_ms / 1000.0
        )
        round_anchor: AnchorState | None = None
        critical_path_wake_ts = static_critical_path_wake_ts
        critical_path_source = "static"
        if cfg.rpc_poller is not None:
            anchor_poll_fire_ts = lock_ts_t - _tc.ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS / 1000.0
            _sleep_until_ts(
                anchor_poll_fire_ts,
                reason="wait_for_anchor_poll",
                epoch=current_epoch,
            )
            round_anchor = cfg.rpc_poller.fire_anchor_poll(
                timeout_s=_tc.ANCHOR_POLL_TIMEOUT_MS / 1000.0,
            )
            if round_anchor is not None:
                predecessor_ms = predict_predecessor_milli_ts(
                    anchor_milli_ts=round_anchor.milli_ts,
                    lock_ms=lock_ms_int,
                )
                dynamic_wake_ms = predecessor_ms - (
                    _tc.OKX_KLINE_FETCH_RTT_P99_MS
                    + _tc.MOMENTUM_GATE_COMPUTE_TIME_MS
                    + _tc.POOL_READ_TIME_MS
                    + _tc.BSC_BET_SUBMIT_ONE_WAY_MS
                )
                # The dynamic wake should be slightly AFTER the anchor poll
                # response landed (which was lock - ~1100ms by design).
                # Even at boundary-zone rounds dynamic_wake_ms >= lock-1057ms,
                # i.e. >= anchor_poll_fire_ts + 200ms slack. Take it as-is.
                critical_path_wake_ts = dynamic_wake_ms / 1000.0
                critical_path_source = "dynamic"
                _static_lead_ms = int(round(
                    (lock_ts_t - static_critical_path_wake_ts) * 1000
                ))
                _dynamic_lead_ms = int(round(
                    (lock_ts_t - critical_path_wake_ts) * 1000
                ))
                info("RUN", "LOOP", "OFFSET",
                     msg=(f"critical_path source=dynamic "
                          f"static_lead_ms={_static_lead_ms} "
                          f"dynamic_lead_ms={_dynamic_lead_ms} "
                          f"anchor_bn={round_anchor.block_number} "
                          f"epoch={current_epoch}"))
            else:
                info("RUN", "LOOP", "OFFSET",
                     msg=(f"critical_path source=static "
                          f"(anchor poll timed out or malformed) "
                          f"epoch={current_epoch}"))
        info("RUN", "LOOP", "SLEEP",
             msg=(f"Sleeping {int(critical_path_wake_ts - _utc_now())}s "
                  f"(wait_for_critical_path, source={critical_path_source}) "
                  f"epoch={current_epoch}"))
        _sleep_until_ts(
            critical_path_wake_ts, reason="wait_for_critical_path",
            epoch=current_epoch,
        )

        # Pool data from RPC poller's local store (Era 11; no RPC needed
        # at this point, the polls already fetched the data).
        pool_bull_bnb = 0.0
        pool_bear_bnb = 0.0
        if cfg.rpc_poller is not None:
            # Unified readiness gate. Skip reasons:
            # - cold_start_in_progress
            # - catchup_infeasible_for_round (the integrating signal:
            #   given current cursor, RTT estimates, and time-until-lock,
            #   math says we cannot catch up in time)
            # Single-poll failures and slow polls do NOT trigger skips —
            # they're informational and the next poll might recover.
            # bankroll_bnb was already resolved at the bankroll wake;
            # reuse for audit on the skip path.
            ready, ready_reason = cfg.rpc_poller.is_pool_ready(current_epoch)
            if not ready:
                skip_reason = f"pool_not_ready_{ready_reason}"
                warn("RPC_POLL", "READY", "SKIP",
                     msg=f"Skip epoch {current_epoch}: {skip_reason}",
                     endpoint=cfg.rpc_poller.current_endpoint)
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
                    skip_reason=skip_reason,
                )
                info("RUN", "ACT", "SKIP",
                     msg=f"Skip epoch {current_epoch}: {skip_reason}")
                _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
                return
            pool_ts_cutoff = lock_ts_t - cfg.pool_cutoff_seconds
            pool_bull_bnb, pool_bear_bnb = cfg.rpc_poller.get_pool(
                epoch=current_epoch, max_ts=pool_ts_cutoff,
            )
            pool_total = pool_bull_bnb + pool_bear_bnb
            if pool_total > 0:
                info("RPC_POLL", "ROUND", "DATA",
                     epoch=current_epoch, pool_bnb=f"{pool_total:.4f}",
                     endpoint=cfg.rpc_poller.current_endpoint)
            # Note: the prior pool=0 + chain_active "data integrity
            # violation" check is GONE in Era 11. With deterministic
            # polling, pool=0 just means the round genuinely had no
            # bets above the filter at cutoff time; it's no longer a
            # silent-stall signal. The strategy's gate handles
            # zero-pool rounds via min_pool_bnb_at_cutoff.

        # Step 8: Decide. Gate fires 3 parallel OKX /history-candles
        # GETs (BTC/ETH/SOL; BNB disabled, see MomentumGate._OKX_SYMBOLS_FETCHED)
        # + computes signal off the returned 1s arrays. Runs sequentially
        # after the in-memory pool snapshot above; both share the single
        # critical_path_wake. The kline fetch effectively starts at
        # lock_at - (critical_path_wakeup_offset_before_lock_ms - POOL_READ_TIME_MS)
        # ~= lock - 1090ms.
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
                btc_fetch_ms=_kline_timing_get(gate, "btc_ms"),
                eth_fetch_ms=_kline_timing_get(gate, "eth_ms"),
                sol_fetch_ms=_kline_timing_get(gate, "sol_ms"),
            )
            info("RUN", "ACT", "SKIP", msg=f"Skip epoch {current_epoch}: {reason}")
            # SKIP path: no time pressure, safe to log timing here.
            if gate is not None and gate.last_fetch_timing is not None:
                info("GATE", "FETCH", "TIMING", **gate.last_fetch_timing)
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 11: Execution timing guard. Abort if wall-clock is past
        # the bet-submit deadline -- TX submitted later than this is
        # unlikely to mine in time and would revert (gas burn).
        #
        # Bundle 5 v2 (2026-05-14): two-track deadline driven by the
        # per-round anchor poll fired earlier at lock - 1300ms.
        #
        #   1. Dynamic deadline (preferred, anchor poll succeeded):
        #      predict the predecessor block's milli_ts from the fresh
        #      anchor via exact 450ms extrapolation, then walk back by
        #      the validator's TX-list freeze window (50ms) + one-way
        #      RPC send time (150ms). Quantum-shift guard inside
        #      ``compute_submit_deadline_ms`` adds a block-time back-off
        #      if the prediction lands within one quantum of lock.
        #
        #   2. Static fallback (anchor poll timed out / malformed):
        #      ``cfg.bet_submit_deadline_offset_before_lock_ms`` (=700ms post-Bundle-4
        #      derivation).
        #
        # lock_ts_t is chain-anchored true UTC; comparisons use
        # ``_utc_now() * 1000`` (skew-corrected ms) so a skewed local
        # clock doesn't make the bot fire early.
        lock_ms = int(lock_ts_t * 1000)
        if round_anchor is not None:
            predecessor_ms = predict_predecessor_milli_ts(
                anchor_milli_ts=round_anchor.milli_ts, lock_ms=lock_ms,
            )
            deadline_ms = compute_submit_deadline_ms(
                predicted_predecessor_milli_ts=predecessor_ms, lock_ms=lock_ms,
            )
            deadline_source = "dynamic"
        else:
            deadline_ms = lock_ms - cfg.bet_submit_deadline_offset_before_lock_ms
            deadline_source = "static"
        # Pre-bet R1 telemetry: log submit-offset (ms remaining before lock)
        # at this point so we can measure how much budget the post-fetch
        # path leaves for TX submission. Negative values would indicate
        # the fetch finished AFTER lock_at (definite revert in live).
        now_utc_ms = _utc_now() * 1000.0
        bet_submit_offset_ms = lock_ms - now_utc_ms
        margin_ms = lock_ms - deadline_ms
        if now_utc_ms >= deadline_ms:
            info(
                "BET",
                "TIMING",
                "ABORT",
                epoch=current_epoch,
                submit_offset_ms=f"{bet_submit_offset_ms:.0f}",
                margin_ms=margin_ms,
                source=deadline_source,
            )
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
                btc_fetch_ms=_kline_timing_get(gate, "btc_ms"),
                eth_fetch_ms=_kline_timing_get(gate, "eth_ms"),
                sol_fetch_ms=_kline_timing_get(gate, "sol_ms"),
            )
            info(
                "RUN",
                "ACT",
                "SKIP",
                msg=f"Skip epoch {current_epoch}: too_close_to_lock_for_bet",
            )
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Guard passed: log submit-offset for inclusion-rate observability.
        # In dry mode this is a proxy ("if this were live, we'd submit with
        # THIS many ms before lock"). In live mode this measures the actual
        # TX-broadcast timing; the receipt status logged later (Step 13)
        # tells us if the TX landed in time.
        # Bundle 4: ``source`` indicates which deadline mode (dynamic from
        # Lorentz anchor vs static fallback) drove the guard decision.
        info(
            "BET",
            "TIMING",
            "OFFSET",
            epoch=current_epoch,
            submit_offset_ms=f"{bet_submit_offset_ms:.0f}",
            margin_ms=margin_ms,
            source=deadline_source,
        )

        # Step 12: Submit bet.
        if decision.bet_side is None:
            raise InvariantError("decision_bet_side_missing")
        bet_side: str = decision.bet_side
        computed_amount_wei = int(round(decision.bet_size_bnb * BNB_WEI))
        if computed_amount_wei <= 0:
            raise InvariantError("bet_amount_wei_nonpositive")

        # Live safety: if clamp_bet_to_contract_minimum is set, clamp the submitted amount to
        # the contract minimum.  All strategy logic runs normally; only the
        # on-chain bet size is reduced.  Audit logs record both sizes.
        amount_wei = computed_amount_wei
        if not cfg.dry and cfg.live_clamp_bet_to_contract_minimum:
            min_wei = int(round(cfg.min_bet_amount_bnb * BNB_WEI))
            amount_wei = min_wei
            info("RUN", "ACT", "CLAMP",
                 msg=f"clamp_bet_to_contract_minimum: clamping {computed_amount_wei / BNB_WEI:.4f} -> {amount_wei / BNB_WEI:.4f} BNB")

        tx_submit = None
        if not cfg.dry:
            gas_price_wei = cfg.contract.suggest_gas_price_wei()
            if bet_side == "Bull":
                tx_submit = cfg.contract.bet_bull_timed(
                    epoch=current_epoch,
                    amount_wei=amount_wei,
                    gas_limit=BACKTEST_GAS_LIMIT_BET,
                    gas_price_wei=gas_price_wei,
                    wait_receipt=True,
                    receipt_timeout_seconds=cfg.bet_tx_receipt_timeout_seconds,
                )
            elif bet_side == "Bear":
                tx_submit = cfg.contract.bet_bear_timed(
                    epoch=current_epoch,
                    amount_wei=amount_wei,
                    gas_limit=BACKTEST_GAS_LIMIT_BET,
                    gas_price_wei=gas_price_wei,
                    wait_receipt=True,
                    receipt_timeout_seconds=cfg.bet_tx_receipt_timeout_seconds,
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
            # R1 inclusion-truth: was the bet TX mined before lock_at?
            # PancakeSwap reverts late bets, so block-timestamp >= lock_ts
            # means the TX wasted gas and the bet did NOT register.
            if tx_submit.included_block_timestamp is not None:
                included_late = (
                    int(tx_submit.included_block_timestamp) >= int(lock_ts_t)
                )
                info(
                    "BET",
                    "INCLUSION",
                    "LATE" if included_late else "OK",
                    epoch=current_epoch,
                    included_block_ts=int(tx_submit.included_block_timestamp),
                    lock_ts=int(lock_ts_t),
                    submit_offset_ms=f"{bet_submit_offset_ms:.0f}",
                )
        else:
            # Step 14: Dry bookkeeping (including gas proxy) + record.
            if closed.simulated_bankroll_bnb is None:
                raise InvariantError("dry_bankroll_uninitialized")

            bankroll_before_bet = closed.simulated_bankroll_bnb
            closed.simulated_bankroll_bnb -= amount_bnb + BACKTEST_GAS_COST_BET_BNB
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
                btc_fetch_ms=_kline_timing_get(gate, "btc_ms"),
                eth_fetch_ms=_kline_timing_get(gate, "eth_ms"),
                sol_fetch_ms=_kline_timing_get(gate, "sol_ms"),
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
    for idx, delay_seconds in enumerate([0] + list(RETRY_BACKOFF_SECONDS)):
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

    claim_ts = close_ts + cfg.buffer_seconds + _CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS
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
            gas_limit=BACKTEST_GAS_LIMIT_CLAIM,
            claim_batch_size=_CLAIM_BATCH_SIZE,
            min_bet_with_gas_bnb=cfg.min_bet_amount_bnb + BACKTEST_GAS_COST_BET_BNB,
            claim_tx_receipt_timeout_seconds=cfg.claim_tx_receipt_timeout_seconds,
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
