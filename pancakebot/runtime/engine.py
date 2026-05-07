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
from pancakebot.runtime.ntp_sync import NtpSync
from pancakebot.runtime.process_health import write_heartbeat
from pancakebot.strategy.momentum_pipeline import StrategyPipelineDecision
from pancakebot.types import Round
from time import sleep as sleep_seconds

# Extra cushion added to the claim wake time, on top of the chain's
# ``buffer_seconds`` settlement window. Total claim-wake time =
# ``close_at(claim_epoch) + buffer_seconds + _CLAIM_CHECK_PADDING_SECONDS``
# (NOT relative to lock_at -- the bot only claims after the round closes
# and the keeper has had ``buffer_seconds`` to call settleRound). The
# padding absorbs alignment retries near RPC boundaries.
_CLAIM_CHECK_PADDING_SECONDS = 5

_CLAIM_BATCH_SIZE = 10
_BACKOFF_SECONDS = [2, 4, 8, 16, 32, 58]  # locked

_TRANSIENT_NETWORK_DELAY_SECONDS = 10
_ONE_MINUTE_MS = 60_000


# -- NTP clock sync ----------------------------------------------------------
# The bot's scheduling uses chain-anchored timestamps (lock_at from BSC, in
# true UTC) but wakes/compares against ``time.time()`` (local clock).
# Local clocks drift between OS-NTP polls -- Windows Time Service refreshes
# at ~1h intervals by default, and at 50 minutes since the last poll the
# local clock is materially off from true UTC. Critical-path timing margins
# are tight enough (sub-100ms) that the accumulated drift can flip a round's
# bet decision (e.g. epoch 478372 in the 2026-05-04 soak was 40ms inside
# the timing-guard margin).
#
# Fix: query NTP directly each round at a pre-critical-path wake
# (``ntp_sync_wake`` at lock - ~11.095s), cache the freshest measured
# offset, and use ``_utc_now()`` everywhere we compare local time to a
# chain-anchored value. Per-round freshness eliminates accumulated
# OS-NTP-poll-interval drift; the wake's 5000ms budget dwarfs the
# empirical NTP roundtrip worst case.
#
# Documented diagnosis: research/okx_lag_root_cause_clock_skew.md (the
# original 2026-04-26 fix used OKX /public/time as the truth source via
# Cristian's algorithm; that approach conflated network latency with clock
# offset and was retired 2026-05-05 in favor of direct NTP).
# Empirical NTP query cost: research/p4c_ntp_probe.py.
_ntp_sync: NtpSync | None = None


def _get_ntp_sync() -> NtpSync:
    """Lazy singleton for the NTP sync manager. Construction is deferred
    until first use so test harnesses that monkey-patch the module's
    network primitives can swap in a fake before the singleton lands."""
    global _ntp_sync
    if _ntp_sync is None:
        _ntp_sync = NtpSync()
    return _ntp_sync


def _utc_now() -> float:
    """Best-effort estimate of the current NTP/UTC second.

    Returns ``time.time() - ntp_offset`` where ``ntp_offset`` is the most
    recent successful NTP measurement of (local - ntp). Falls back to
    local ``time.time()`` when no NTP query has succeeded yet (initial
    state, or pre-bootstrap). Caller code that compares local time
    against a chain-anchored value (lock_at, cutoff_ts, claim_ts) MUST
    use this instead of ``time.time()`` so the comparison is in
    NTP/UTC frame.
    """
    return time.time() - _get_ntp_sync().current_offset()


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

    Operators read this at startup to confirm which publish-delay tier
    the gate is running under (P99 = strict full-inclusion guarantee;
    P95 = operating budget, ~5% publish-delay tail absorbed by the
    streak counter) without having to derive the math from raw constants.
    """
    if cfg.kline_publish_tier == "P99":
        tier_msg = "P99 (strict; full-inclusion guarantee)"
    elif cfg.kline_publish_tier == "P95":
        tier_msg = (
            "P95 (operating budget; ~5% publish-delay tail absorbed by "
            "streak counter)"
        )
    else:
        tier_msg = f"{cfg.kline_publish_tier} (unrecognized tier)"
    info(
        "CORE", "RUN", "TIMING",
        msg=(
            f"timing config: kline_cutoff={cfg.cutoff_seconds}s "
            f"pool_cutoff={cfg.pool_cutoff_seconds}s "
            f"ntp_sync_wakeup={cfg.ntp_sync_wakeup_offset_ms}ms "
            f"bankroll_wakeup={cfg.bankroll_wakeup_offset_ms}ms "
            f"critical_path_wakeup={cfg.critical_path_wakeup_offset_ms}ms "
            f"bet_submit_deadline={cfg.bet_submit_deadline_offset_ms}ms "
            f"bet_tx_receipt_timeout={cfg.bet_tx_receipt_timeout_seconds}s "
            f"claim_tx_receipt_timeout={cfg.claim_tx_receipt_timeout_seconds}s "
            f"kline_publish_tier={tier_msg}"
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

    # Bootstrap NTP clock sync BEFORE the first round starts. The per-round
    # ntp_sync_wake won't have fired yet; without bootstrap, the first
    # iteration's _sleep_until_ts compares against an uncorrected local
    # clock. Refuse to start if every server fails or the initial offset
    # is unreasonably large -- broken NTP at startup is a clear operator-
    # actionable failure (run w32tm /resync, check firewall, etc.).
    info("CLOCK", "NTP", "BOOT", msg="bootstrapping NTP clock sync...")
    _ntp = _get_ntp_sync()
    if not _ntp.bootstrap():
        raise InvariantError(
            "ntp_bootstrap_failed: no NTP servers reachable at startup; "
            "check network / firewall / w32tm /resync before retrying"
        )
    if abs(_ntp.current_offset()) > 1.0:
        raise InvariantError(
            f"ntp_bootstrap_offset_too_large: "
            f"{_ntp.current_offset():+.3f}s exceeds 1.0s sanity bound; "
            f"run w32tm /resync on host before retrying"
        )

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

        # Step 5: cutoff_ts(t) = lock_ts(t) - cutoff_seconds.
        cutoff_ts_t = lock_ts_t - cfg.cutoff_seconds

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
        claim_ts = prev_close_ts + cfg.buffer_seconds + _CLAIM_CHECK_PADDING_SECONDS
        # claim_ts and cutoff_ts_t are both chain-anchored true UTC; compare
        # against NTP-corrected _utc_now() so a drifted local clock doesn't
        # make us miss the wake-for-claim window.
        if _utc_now() < claim_ts < cutoff_ts_t:
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=prev_locked_epoch)
            return

        # -- NTP sync wake --
        # First of three pre-lock wakes. Fires at lock_at -
        # ntp_sync_wakeup_offset_ms (= ~11.095s before lock at canonical
        # timing constants). Forces a fresh NTP query so the freshest
        # measured (local - ntp) offset is applied for the rest of the
        # round's critical-path scheduling. Generously off the critical
        # path: 5000ms wake budget vs. ~125ms NTP p99 query, so even a
        # 3-server rotation fall-through (~306ms worst case) is fine.
        ntp_sync_wake_ts = lock_ts_t - cfg.ntp_sync_wakeup_offset_ms / 1000.0
        _sleep_until_ts(
            ntp_sync_wake_ts,
            reason="wait_for_ntp_sync",
            epoch=current_epoch,
        )
        ntp = _get_ntp_sync()
        ntp.force_resync()
        if not ntp.is_healthy():
            # NTP state stale: consecutive failures over the threshold OR
            # last-good offset is too old. Skip rather than bet on a
            # potentially-drifted clock; the next round's wake re-queries
            # and may recover.
            warn("RUN", "NTP", "UNHEALTHY",
                 msg=(f"Skip epoch {current_epoch}: ntp_state_unhealthy "
                      f"(consecutive_failures={ntp.consecutive_failures()}, "
                      f"last_query_age={ntp.last_query_age_seconds():.0f}s)"))
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
                bankroll_before_action_bnb=closed.simulated_bankroll_bnb or 0.0,
                bankroll_after_action_bnb=closed.simulated_bankroll_bnb or 0.0,
                skip_reason="ntp_state_unhealthy",
            )
            info("RUN", "ACT", "SKIP",
                 msg=f"Skip epoch {current_epoch}: ntp_state_unhealthy")
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # -- Bankroll wake --
        # Second of three pre-lock wakes. Fires at lock_at -
        # bankroll_wakeup_offset_ms (= ~6.095s before lock). Refreshes
        # wallet balance so risk gates + decide_open_round see fresh
        # truth. Live mode does a BSC RPC call (~50-200ms p99); dry
        # mode reads in-memory simulated bankroll (sub-ms). Generously
        # off the critical path: 5000ms wake budget. On live RPC error,
        # SKIP the iteration with risk_bankroll_stale rather than betting
        # on stale value.
        bankroll_wake_ts = lock_ts_t - cfg.bankroll_wakeup_offset_ms / 1000.0
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

        # -- Critical-path wake --
        # Third (and final) pre-lock scheduled wake. Fires at lock_at -
        # critical_path_wakeup_offset_ms (= ~1.095s before lock at
        # canonical timing constants). Inside the wake the engine
        # sequences: pool snapshot (in-memory, ~5ms) -> kline fetch +
        # signal compute (~340ms via gate.evaluate()) -> bet submit
        # (~700ms BSC RTT + block budget). The bet-submit-deadline
        # timing guard at lock_at - bet_submit_deadline_offset_ms
        # gates the actual submission to abort late rounds.
        critical_path_wake_ts = (
            lock_ts_t - cfg.critical_path_wakeup_offset_ms / 1000.0
        )
        _sleep_until_ts(
            critical_path_wake_ts, reason="wait_for_critical_path",
            epoch=current_epoch,
        )

        # Pool data from WSS subscription (no RPC needed, ~0 ms).
        pool_bull_bnb = 0.0
        pool_bear_bnb = 0.0
        if cfg.pool_watcher is not None:
            # Unified readiness gate: covers wss_disconnected,
            # backfill_in_progress, and reconnect_requested in one
            # check. The watcher's predicate is the canonical source of
            # truth for "is the pool data fresh and trustworthy right
            # now"; the engine just gates on its result. bankroll_bnb
            # was already resolved at the bankroll wake -- reuse it for
            # audit on the skip path.
            ready, ready_reason = cfg.pool_watcher.is_pool_ready()
            if not ready:
                skip_reason = f"pool_not_ready_{ready_reason}"
                warn("POOL_WSS", "READY", "SKIP",
                     msg=f"Skip epoch {current_epoch}: {skip_reason}",
                     endpoint=cfg.pool_watcher.current_endpoint)
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
            pool_bull_bnb, pool_bear_bnb = cfg.pool_watcher.get_pool(
                epoch=current_epoch, max_ts=pool_ts_cutoff,
            )
            pool_total = pool_bull_bnb + pool_bear_bnb
            if pool_total > 0:
                info("POOL_WSS", "ROUND", "DATA",
                     epoch=current_epoch, pool_bnb=f"{pool_total:.4f}",
                     endpoint=cfg.pool_watcher.current_endpoint)
            else:
                # Data-integrity check (Investigation B fix, 2026-05-05).
                # Pool=0 at lock-6s on a connected, backfill-done watcher
                # is essentially always a WSS silent-stall: PancakeSwap
                # rounds normally accumulate ~10-30 bets in the 5-minute
                # bet window. If WSS observed nothing but the chain has
                # the round open, the watcher is missing events.
                #
                # Skip with an EXPLICIT reason so the operator sees the
                # data-availability issue in cycle_audit, not a misleading
                # `gate_no_signal` or `pool_below_minimum`. Pair with the
                # idle-stall reconnect in pool_watcher (~8s threshold);
                # this invariant catches anything the idle check missed
                # OR fires before the idle check has had time to
                # reconnect-and-backfill.
                #
                # Future Option B: RPC fallback to chain truth via
                # contract.round_data(). Deferred -- adds 50-200ms to
                # critical path; the explicit skip is the minimum-change
                # operator-visibility win.
                warn(
                    "POOL_WSS", "INTEG", "VIOLATION",
                    msg=(
                        f"Skip epoch {current_epoch}: "
                        f"data_integrity_violation_pool_zero_chain_active "
                        f"(WSS pool=0/0 at lock-{cfg.pool_cutoff_seconds}s; "
                        f"connected=True backfill_done=True endpoint="
                        f"{cfg.pool_watcher.current_endpoint})"
                    ),
                )
                # Tell the watcher to cycle endpoints next session-loop
                # iteration. This catches publicnode-style silent logs-
                # subscription drops one round earlier than the watcher's
                # own _LOGS_IDLE_THRESHOLD_SECONDS (10 min) would. See
                # var/incident_reports/2026_05_06_wss_silent_stall_root_cause.md
                # for the smoking-gun trace.
                cfg.pool_watcher.request_reconnect(
                    "pool_zero_chain_active"
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
                    decision_stage="pipeline",
                    open_round=open_round,
                    bankroll_before_action_bnb=bankroll_bnb,
                    bankroll_after_action_bnb=bankroll_bnb,
                    skip_reason="data_integrity_violation_pool_zero_chain_active",
                )
                info("RUN", "ACT", "SKIP",
                     msg=(
                         f"Skip epoch {current_epoch}: "
                         f"data_integrity_violation_pool_zero_chain_active"
                     ))
                _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
                return

        # Step 8: Decide. Gate fires 3 parallel OKX /history-candles
        # GETs (BTC/ETH/SOL; BNB disabled, see MomentumGate._SYMBOLS_FETCHED)
        # + computes signal off the returned 1s arrays. Runs sequentially
        # after the in-memory pool snapshot above; both share the single
        # critical_path_wake. The kline fetch effectively starts at
        # lock_at - (critical_path_wakeup_offset_ms - POOL_READ_TIME_MS)
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
            )
            info("RUN", "ACT", "SKIP", msg=f"Skip epoch {current_epoch}: {reason}")
            # SKIP path: no time pressure, safe to log timing here.
            if gate is not None and gate.last_fetch_timing is not None:
                info("GATE", "FETCH", "TIMING", **gate.last_fetch_timing)
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 11: Execution timing guard. Abort if wall-clock is within
        # ``cfg.bet_submit_deadline_offset_ms`` of lock_at -- TX submitted
        # that close to lock is unlikely to mine in time and would revert
        # (gas burn). lock_ts_t is chain-anchored true UTC; compare via
        # _utc_now() (skew-corrected). With local-only comparison + skew,
        # the guard fires far too early in true UTC and the bot
        # self-aborts almost every round.
        #
        # Pre-bet R1 telemetry: log submit-offset (ms remaining before lock)
        # at this point so we can measure how much budget the post-fetch
        # path leaves for TX submission. Negative values would indicate
        # the fetch finished AFTER lock_at (definite revert in live).
        bet_submit_offset_ms = (lock_ts_t - _utc_now()) * 1000.0
        safety_margin_seconds = cfg.bet_submit_deadline_offset_ms / 1000.0
        if _utc_now() >= lock_ts_t - safety_margin_seconds:
            info(
                "BET",
                "TIMING",
                "ABORT",
                epoch=current_epoch,
                submit_offset_ms=f"{bet_submit_offset_ms:.0f}",
                margin_ms=cfg.bet_submit_deadline_offset_ms,
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
        info(
            "BET",
            "TIMING",
            "OFFSET",
            epoch=current_epoch,
            submit_offset_ms=f"{bet_submit_offset_ms:.0f}",
            margin_ms=cfg.bet_submit_deadline_offset_ms,
        )

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
                    receipt_timeout_seconds=cfg.bet_tx_receipt_timeout_seconds,
                )
            elif bet_side == "Bear":
                tx_submit = cfg.contract.bet_bear_timed(
                    epoch=current_epoch,
                    amount_wei=amount_wei,
                    gas_limit=GAS_LIMIT_BET,
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
