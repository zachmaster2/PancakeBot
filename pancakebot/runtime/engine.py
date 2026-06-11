"""Live/dry runtime loop: epoch handshake, cutoff-aligned decision, bet submission, and claim scan."""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from pathlib import Path

from pancakebot.constants import (
    BNB_WEI,
    GAS_LIMIT_BET,
    GAS_LIMIT_CLAIM,
    MAX_GAS_COST_BET_BNB,
    MAX_GAS_PRICE_WEI,
    RETRY_BACKOFF_SECONDS,
)
from pancakebot.util import GasPriceCapBreachedError, InvariantError, TransientRpcError
from pancakebot.log import info, warn
from pancakebot.util import format_bankroll
from pancakebot.runtime.config import RuntimeConfig
from pancakebot import paths
from pancakebot.runtime.dry import (
    RuntimeState,
    append_jsonl,
    dry_record_bet,
    dry_settle_available_bets,
    fetch_wallet_balance_bnb_with_retries,
    init_runtime_state,
    record_cycle_audit,
)
from pancakebot.runtime.live import (
    DRY_CHANNEL,
    LIVE_CHANNEL,
    claim_scan_cursor,
    send_bet_confirmed_alert,
    send_cooldown_entered_alert,
    send_cooldown_lifted_alert,
    send_bet_dropped_alert,
    send_bet_late_alert,
    send_bet_reverted_alert,
    send_bet_settled_alert,
    send_bet_submitted_alert,
    send_bot_ready_alert,
    send_gas_cap_breach_alert,
)
from pancakebot.runtime import bet_ledger
from pancakebot.chain.rpc_poller import (
    AnchorState,
    compute_submit_deadline_ms,
    predict_predecessor_milli_ts,
)
from pancakebot import timing_constants as _tc
from pancakebot.runtime.regime_telemetry import RollingP99Monitor
from pancakebot.types import Round
from time import sleep as sleep_seconds

# Padding for RPC alignment near chain transition boundaries. Used for:
#   - post-close claim safety: claim_ts = close_at(N) + buffer_seconds + padding
#   - cumulative target for RETRY_BACKOFF_SECONDS: the runtime retry budget
#     spans buffer_seconds + padding so the bare _epoch_handshake covers a
#     full executeRound settlement window before raising the *_exhausted
#     invariants.
# Both contexts need a small extra window beyond the contract's chain-level
# buffer_seconds to absorb RPC endpoint lag (node-side indexing + RTT tail).
# (TX receipt timeouts — bet AND claim — are NOT sized from this; they use
# the flat TX_RECEIPT_WAIT_TIMEOUT_SECONDS, set in app.py.)
_RPC_ALIGNMENT_PADDING_SECONDS = 5


# Regime-drift monitor (guard audit Tier 2 / item 5.3). Module-level
# singleton: the bot is one long-lived process, so the rolling window
# accumulates across the whole run. Compares the live per-round max-of-3
# OKX kline fetch RTT against OKX_KLINE_FETCH_RTT_P99_MS — a stale-LOW
# constant silently lets the dynamic-wake walk-back fire too late.
# Telemetry only; reset for tests via _reset_regime_monitors().
def _build_okx_kline_rtt_monitor() -> RollingP99Monitor:
    return RollingP99Monitor(
        name="okx_kline_rtt_p99",
        constant_ms=_tc.OKX_KLINE_FETCH_RTT_P99_MS,
        tolerance_ms=50.0,
        window=100,
        min_samples=30,
    )


_OKX_KLINE_RTT_MONITOR = _build_okx_kline_rtt_monitor()


def _reset_regime_monitors() -> None:
    """Test hook: rebuild the module-level monitors with empty windows."""
    global _OKX_KLINE_RTT_MONITOR
    _OKX_KLINE_RTT_MONITOR = _build_okx_kline_rtt_monitor()


# -- Clock sync ---------------------------------------------------------------
# Time source (Bundle 5 v2, 2026-05-14): the bot trusts the OS clock
# directly. Previously the bot maintained its own per-round NTP query
# (``NtpSync``) that measured ``(local - ntp)`` once per round and applied
# the correction inside ``_utc_now()`` — a workaround for the original
# Windows host's W32Time, whose default 1024s poll cadence let the clock
# drift up to ~270ms (P95) between syncs. Too sloppy for sub-second bet
# timing, so the host clock had to be disciplined externally and the
# application layer retired.
#
# The bot now runs on the Frankfurt Linux VM, where chronyd disciplines the
# clock continuously (frequency steering between polls): measured residual
# offset ~30-60 MICROseconds RMS (2026-06-10), four orders of magnitude
# inside the bot's timing budget. ``time.time()`` is the authoritative
# truth source, and ``_utc_now()`` is a thin alias preserved for
# readability at the call sites that compare local time to chain-anchored
# values (lock_at, cutoff_ts, claim_ts).
#
# Bootstrap installs a chrony drop-in (bootstrap/install.sh: maxpoll 6 +
# makestep) that bounds clock-STEP detection after a VM pause/migration to
# ~64s — the one Linux risk class; steady-state drift is a non-issue under
# chrony. The post-install health check (bootstrap/common/health_check.py)
# verifies the clock is synchronized and the offset is inside tolerance;
# spot-check manually with ``chronyc tracking``.


def _utc_now() -> float:
    """Current wallclock seconds. Trusts the OS clock (chrony-disciplined
    on the VM; see the clock-sync note above). Preserved as a separate
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


def _kline_result_get(gate, sym_short: str) -> str:
    """Safe lookup into ``gate.last_fetch_results[sym_short]``.

    Returns ``"not_fetched"`` when the gate is None or hasn't run yet
    this round (e.g. early-skip paths like risk_bankroll_stale or
    pool_not_ready). Cycle-audit persists the string as-is so downstream
    analysis can distinguish "round skipped before fetch" from "fetch
    failed."
    """
    if gate is None:
        return "not_fetched"
    results = gate.last_fetch_results
    if results is None:
        return "not_fetched"
    return results.get(sym_short, "not_fetched")


def _truncate_tx_hash(tx_hash: str) -> str:
    """Render the first 8 chars of a tx hash with a trailing ellipsis
    (e.g. ``0x123456...``). The full hash is captured elsewhere (live
    latency.jsonl for bets; chain explorer is the authoritative source).
    Truncated form keeps operator stdout single-glance scannable while
    preserving enough disambiguation to cross-reference per session."""
    if not tx_hash or len(tx_hash) <= 8:
        return tx_hash
    return f"{tx_hash[:8]}..."


# Severity precedence among kline failure subtypes — higher = more severe.
# When multiple symbols fail in the same round, the engine SKIP lead uses
# the most-severe subtype across all failed symbols.
_KLINE_FAIL_SEVERITY: dict[str, int] = {
    "kline_publish_delay": 1,
    "kline_http_error": 2,
    "kline_unreachable": 3,
}


def _classify_kline_failure(
    last_fetch_results: dict[str, str] | None,
) -> tuple[str, str] | None:
    """Inspect per-symbol fetch results from ``gate.last_fetch_results``
    and return ``(subtype, message_body)`` for the SKIP narrative, or
    ``None`` if no failures.

    Subtypes:
      - ``kline_publish_delay``: ``partial:got_N_expected_M`` — OKX
        served a short response, typically the newest candle wasn't yet
        published. Rendered: ``BTC: N of M candles``.
      - ``kline_unreachable``: ``error:<network-class>`` — no bytes
        received (ConnectionError / Timeout / DNS / etc.). Rendered:
        ``BTC: ConnectionError``.
      - ``kline_http_error``: ``error:<http_class>`` — bytes received
        but with an error response (http_429, okx_code_*, empty_data,
        json_parse_error). Rendered: ``BTC: http_429``.

    Multi-symbol failure: the message body enumerates all failed symbols
    comma-separated; the returned subtype is the most severe.

    Unknown result shapes fall back to ``kline_http_error`` severity
    (defensive); empirically the three families above cover every
    `last_fetch_results` value populated by ``momentum_gate.evaluate``.
    """
    if not last_fetch_results:
        return None
    subtype_for_sym: dict[str, str] = {}
    body_parts: list[str] = []
    for sym_short, result in last_fetch_results.items():
        if result in ("ok", "not_fetched"):
            continue
        sym_upper = sym_short.upper()
        if result.startswith("partial:got_"):
            # partial:got_15_expected_16 → "15 of 16 candles"
            tokens = result[len("partial:"):].split("_")
            try:
                got, exp = int(tokens[1]), int(tokens[3])
                body_parts.append(f"{sym_upper}: {got} of {exp} candles")
            except (IndexError, ValueError):
                body_parts.append(f"{sym_upper}: {result}")
            subtype_for_sym[sym_short] = "kline_publish_delay"
        elif result.startswith("error:"):
            detail = result[len("error:"):]
            body_parts.append(f"{sym_upper}: {detail}")
            if (
                detail.startswith("http_")
                or detail.startswith("okx_code_")
                or detail in ("empty_data", "json_parse_error")
            ):
                subtype_for_sym[sym_short] = "kline_http_error"
            else:
                subtype_for_sym[sym_short] = "kline_unreachable"
        else:
            body_parts.append(f"{sym_upper}: {result}")
            subtype_for_sym[sym_short] = "kline_http_error"
    if not subtype_for_sym:
        return None
    most_severe = max(
        subtype_for_sym.values(), key=lambda s: _KLINE_FAIL_SEVERITY[s]
    )
    return most_severe, ", ".join(body_parts)


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
        "START",
        f"timing config: kline_cutoff={cfg.kline_cutoff_seconds}s "
        f"pool_cutoff={cfg.pool_cutoff_seconds}s "
        f"okx_warmup_wakeup={cfg.okx_warmup_wakeup_offset_before_lock_ms}ms "
        f"preflight_wakeup={cfg.preflight_wakeup_offset_before_lock_ms}ms "
        f"single_poll_wakeup={cfg.single_poll_wakeup_offset_before_lock_ms}ms "
        f"critical_path_wakeup={cfg.critical_path_wakeup_offset_before_lock_ms}ms "
        f"bet_submit_deadline={cfg.bet_submit_deadline_offset_before_lock_ms}ms "
        f"bet_tx_receipt_timeout={cfg.bet_tx_receipt_timeout_seconds}s "
        f"claim_tx_receipt_timeout={cfg.claim_tx_receipt_timeout_seconds}s",
    )


def _assert_critical_path_timing_sane(cfg: RuntimeConfig) -> None:
    """Fail loud at startup if the wake-offset ladder is misordered.

    Every per-round wake fires at ``lock - offset_ms``; a larger offset
    fires earlier. The ladder MUST be strictly decreasing (each stage
    after the one before it) or a misconfigured ``timing_constants.py``
    silently mis-sequences polls / zeroes a poll budget every round via
    the ``max(0, ...)`` clamps. This invariant catches the misconfig at
    boot instead. (We assert strict ordering, not a uniform >200ms gap:
    the tightest leg (anchor_poll->critical_path) is ~330ms by design.)

    Also asserts the anchor poll can complete before the critical-path
    wake: it fires at ``lock - ANCHOR_POLL_OFFSET`` and is awaited up to
    ``ANCHOR_POLL_TIMEOUT``; if that response can't land before the
    static critical-path wake, the dynamic-wake path is structurally
    dead (every round silently falls back to static).
    """
    ladder = [
        ("okx_warmup", cfg.okx_warmup_wakeup_offset_before_lock_ms),
        ("preflight", cfg.preflight_wakeup_offset_before_lock_ms),
        ("single_poll", cfg.single_poll_wakeup_offset_before_lock_ms),
        ("anchor_poll", _tc.ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS),
        ("critical_path", cfg.critical_path_wakeup_offset_before_lock_ms),
        ("bet_submit_deadline", cfg.bet_submit_deadline_offset_before_lock_ms),
    ]
    for (name_a, off_a), (name_b, off_b) in zip(ladder, ladder[1:]):
        if not off_a > off_b:
            raise InvariantError(
                f"timing_ladder_not_strictly_decreasing: {name_a}={off_a}ms "
                f"must be > {name_b}={off_b}ms (both are ms-before-lock; "
                f"larger fires earlier)"
            )
    anchor_response_by = (
        _tc.ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS - _tc.ANCHOR_POLL_TIMEOUT_MS
    )
    if anchor_response_by < cfg.critical_path_wakeup_offset_before_lock_ms:
        raise InvariantError(
            f"anchor_slack_negative: anchor response by lock-{anchor_response_by}ms "
            f"(offset {_tc.ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS} - timeout "
            f"{_tc.ANCHOR_POLL_TIMEOUT_MS}) must land before the static "
            f"critical-path wake at lock-{cfg.critical_path_wakeup_offset_before_lock_ms}ms; "
            f"otherwise the dynamic-wake path is structurally dead"
        )


def run_realtime_loop(cfg: RuntimeConfig) -> None:
    # Wallet address is only required for live mode (signing transactions).
    # Dry mode reads from chain via public RPC, no signing needed.
    if not cfg.dry and not cfg.wallet_address:
        raise InvariantError("wallet_address_required_for_live")
    if cfg.min_bet_amount_bnb <= 0.0:
        raise InvariantError("runtime_min_bet_amount_nonpositive")

    _log_runtime_timing_summary(cfg)
    _assert_critical_path_timing_sane(cfg)

    closed_state = init_runtime_state(cfg)

    # Bundle 5 v2 (2026-05-14): no application-level NTP bootstrap. The
    # bot trusts the OS clock (chrony-disciplined on the VM; see the
    # clock-sync note at the top of this module). The prior NtpSync
    # bootstrap + per-round refresh was retired alongside the continuous
    # fine-phase chain-anchor poll — both layers existed to paper over
    # the original Windows host's W32Time drift; a disciplined host
    # clock obviates them.

    bnbusd_price = _fetch_current_bnb_price_usd(cfg)
    if cfg.dry:
        if closed_state.simulated_bankroll_bnb is None:
            raise InvariantError("dry_bankroll_uninitialized")
        bankroll_bnb = closed_state.simulated_bankroll_bnb
        # PersistedBankrollTracker for dry mode is already wired by
        # init_runtime_state (after bankroll resolution). No-op here.
    else:
        bankroll_bnb = fetch_wallet_balance_bnb_with_retries(
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
        "START",
        f"Starting bankroll: {format_bankroll(bankroll_bnb=bankroll_bnb, bnbusd_price=bnbusd_price)}",
    )
    if not cfg.dry:
        # BOT READY (Bundle 7): fired once per start after the first successful
        # wallet-balance read, so the first BET SUBMITTED has a bankroll
        # reference point. Bot-owned (distinct from the lifecycle STARTED
        # alert). Best-effort — the sender swallows all webhook errors.
        send_bot_ready_alert(channel=LIVE_CHANNEL, bankroll_bnb=bankroll_bnb)

    # Fresh-spawn-during-round-transition race is absorbed by the bare
    # _epoch_handshake retry loop, which retries on all three zero-state
    # invariants (locked.lock_ts, locked.lock_price_usd, open.lock_ts) with
    # RETRY_BACKOFF_SECONDS sized so cumulative wait crosses
    # buffer_seconds + _RPC_ALIGNMENT_PADDING_SECONDS (~35s) by the 5th
    # retry, with grace beyond.
    while True:
        # Per-subsystem TransientRpcError handling lives at each callsite:
        #   - _epoch_handshake: bounded local retry
        #   - preflight wake: SKIP round with risk_bankroll_stale
        #   - _sleep_and_claim close_ts: bounded local retry (same pattern as handshake)
        #   - claim_scan_cursor callers: fail-soft (log warn + continue)
        #   - bet submission: crash → systemd restart (round was lost anyway)
        # No top-level catch — there is no remaining bubble path where a
        # generic 10s-sleep-and-retry helps.
        _run_one_iteration(cfg, closed_state)


def _mono_ms() -> float:
    return time.perf_counter() * 1000.0



@dataclass(frozen=True, slots=True)
class _RoundSkipCtx:
    """Per-round audit fields shared verbatim by every terminal-SKIP exit
    of ``_run_one_iteration`` (frozen once the round identity is fixed).

    ``wake_mode`` / ``kline_fire_offset_before_lock_ms`` are deliberately
    NOT here: they start empty and are assigned at anchor-poll resolution,
    so each SKIP site passes the then-current values explicitly.
    """
    current_epoch: int
    locked_epoch: int
    lock_ts: int
    cutoff_ts: int
    bnbusd_price: float
    open_round: Round | None
    gate: object | None


def _skip_round(
    cfg: RuntimeConfig,
    closed: RuntimeState,
    ctx: _RoundSkipCtx,
    *,
    skip_reason: str,
    decision_stage: str,
    bankroll_bnb: float | None,
    wake_mode: str,
    kline_fire_offset_before_lock_ms: int | None,
    log_level: str | None = None,
    log_line: str | None = None,
    decision: object | None = None,
    decision_latency_ms: float | None = None,
    pool_bull_bnb: float = 0.0,
    pool_bear_bnb: float = 0.0,
    with_fetch_ms: bool = False,
) -> None:
    """Terminal-SKIP bookkeeping: the cycle-audit row, plus the operator
    SKIP line when the wording needs no site-local branching (pass
    ``log_line=None`` to keep logging at the call site). Owns the
    always-identical audit kwargs so the cycle_audit schema cannot drift
    across the six exit sites; site-specific wording, side effects, and
    the trailing ``_sleep_and_claim`` + ``return`` stay at the call site.
    """
    fetch_ms_kwargs: dict[str, object] = {}
    if with_fetch_ms:
        fetch_ms_kwargs = {
            "btc_fetch_ms": _kline_timing_get(ctx.gate, "btc_ms"),
            "eth_fetch_ms": _kline_timing_get(ctx.gate, "eth_ms"),
            "sol_fetch_ms": _kline_timing_get(ctx.gate, "sol_ms"),
        }
    record_cycle_audit(
        cfg,
        closed,
        current_epoch=ctx.current_epoch,
        locked_epoch=ctx.locked_epoch,
        lock_ts=ctx.lock_ts,
        cutoff_ts=ctx.cutoff_ts,
        locked_price_bnbusd=ctx.bnbusd_price,
        action="SKIP",
        decision_stage=decision_stage,
        open_round=ctx.open_round,
        bankroll_before_action_bnb=bankroll_bnb,
        bankroll_after_action_bnb=bankroll_bnb,
        decision=decision,
        skip_reason=skip_reason,
        decision_latency_ms=decision_latency_ms,
        pool_bull_bnb=pool_bull_bnb,
        pool_bear_bnb=pool_bear_bnb,
        wake_mode=wake_mode,
        kline_fire_offset_before_lock_ms=kline_fire_offset_before_lock_ms,
        btc_fetch_result=_kline_result_get(ctx.gate, "btc"),
        eth_fetch_result=_kline_result_get(ctx.gate, "eth"),
        sol_fetch_result=_kline_result_get(ctx.gate, "sol"),
        **fetch_ms_kwargs,
    )
    if log_line is not None:
        (warn if log_level == "warn" else info)("SKIP", log_line)


def _run_one_iteration(cfg: RuntimeConfig, closed: RuntimeState) -> None:
    closed.iteration_count += 1

    # Alignment + cutoff anchoring can be noisy around epoch shifts. Ensure we only
    # take an action using a coherent epoch snapshot.
    while True:
        # Step 1: Epoch alignment handshake (shift-aware) with retries.
        locked_round, _open_round, current_epoch = _epoch_handshake(cfg)
        locked_epoch = locked_round.epoch

        # Track last_seen_epoch (process-health telemetry; NOT wired into
        # crash.json — run.py writes last_epoch=None).
        closed.last_seen_epoch = current_epoch

        # Sync round-phase state into rpc_poller immediately after handshake.
        # Bundle 2 (2026-05-13): on the first call this synchronously initializes
        # the cursor from chain head (~1 RPC, sub-second) but does NOT block on
        # backfill — the periodic daemon's first tick + the single poll
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
                # Crash recovery: reconcile any bets left open (SUBMITTED/
                # CONFIRMED) by a previous incarnation whose rounds have since
                # closed — settles them (LOSS alert fires; WIN/REFUND recorded)
                # BEFORE the claim scan, so the scan can fire the backlog
                # WON/REFUND alerts off the fresh SETTLED_* records. Idempotent.
                _reconcile_live_bets(cfg, closed)
                try:
                    claim_scan_cursor(
                        contract=cfg.contract,
                        wallet_address=cfg.wallet_address,
                        dry=False,
                        cursor_path=paths.LIVE_CLAIM_CURSOR_PATH,
                        locked_epoch=locked_epoch,
                        current_epoch=current_epoch,
                        now_ts=int(_utc_now()),  # OS-clock UTC (chrony-disciplined; see module clock-sync note); compared to chain-anchored close timestamps
                        buffer_seconds=cfg.buffer_seconds,
                        page_size=100,
                        gas_limit=GAS_LIMIT_CLAIM,
                        claim_tx_receipt_timeout_seconds=cfg.claim_tx_receipt_timeout_seconds,
                        bets_ledger_path=paths.LIVE_BETS_LEDGER_PATH,
                    )
                except TransientRpcError as e:
                    warn("ALERT", f"claim scan failed: rpc_transient err={e}")

            dry_settle_available_bets(cfg, closed)
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

        # Per-round wake-mode + kline-fire-offset for offline analysis. Filled
        # in at critical_path resolution (lines ~585-630) once the anchor
        # poll result is known. Early-skip paths (e.g. risk_bankroll_stale,
        # which fires at the preflight wake before the anchor poll) leave
        # these empty -- the bot never decided which mode to use.
        wake_mode: str = ""
        kline_fire_offset_before_lock_ms: int | None = None

        # Frozen audit context for the terminal-SKIP exits (see _skip_round):
        # the fields every exit records identically.
        skip_ctx = _RoundSkipCtx(
            current_epoch=current_epoch,
            locked_epoch=locked_epoch,
            lock_ts=lock_ts_t,
            cutoff_ts=cutoff_ts_t,
            bnbusd_price=bnbusd_price,
            open_round=open_round,
            gate=gate,
        )

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
        claim_ts = prev_close_ts + cfg.buffer_seconds + _RPC_ALIGNMENT_PADDING_SECONDS
        # claim_ts and cutoff_ts_t are both chain-anchored true UTC; compare
        # against _utc_now() (OS-clock UTC, chrony-disciplined — see the module
        # clock-sync note) so we don't miss the wake-for-claim window.
        if _utc_now() < claim_ts < cutoff_ts_t:
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=prev_locked_epoch)
            return

        # Bundle 5 v2 (2026-05-14): the per-round NTP sync wake is gone.
        # Previously the bot woke at ``lock - 11095ms`` to refresh its own
        # ``(local - ntp)`` offset measurement. The OS clock is disciplined
        # directly (chrony on the VM; see the clock-sync note at the top of
        # this module), so there is no application-level NTP layer to
        # refresh. Candidate C (2026-06-06): the first pre-lock wake is now
        # the OKX warmup (= lock - 7000ms); the RPC catch-up is ONE getLogs
        # poll just before the critical path (was a 3-leg ramp ladder —
        # ramp_1/ramp_2 removed, the final poll became the single poll
        # below).

        # -- OKX warmup wake --
        # (The wallet-balance refresh happens later, at the preflight
        # wake.) OKX session warmup (lock - 7000ms by default). Refreshes
        # the OkxClient's HTTPS connection pool so the per-round kline
        # fetch doesn't pay a TLS handshake cost out of the critical
        # path. Without this, a long idle window (e.g. consecutive
        # catchup_infeasible skips) lets OKX server keep-alives expire
        # and the next fetch pays 500-800ms vs typical 270ms — caught
        # 2026-05-21 live crash post-mortem. Always-runs (idempotent
        # when connections are already warm). Errors swallowed inside
        # ``OkxClient.warmup``; bot bets regardless.
        okx_warmup_wake_ts = lock_ts_t - cfg.okx_warmup_wakeup_offset_before_lock_ms / 1000.0
        _sleep_until_ts(
            okx_warmup_wake_ts,
            reason="wait_for_okx_warmup",
            epoch=current_epoch,
        )
        if closed.strategy_pipeline is not None and hasattr(closed.strategy_pipeline, "_gate"):
            # noinspection PyProtectedMember
            _warmup_gate = closed.strategy_pipeline._gate
            if _warmup_gate is not None:
                _warmup_gate.warmup_okx_session()

        # Generously off the critical path: 5000ms wake budget. On
        # live RPC error, SKIP the iteration with risk_bankroll_stale
        # rather than sizing the bet from a potentially-outdated
        # bankroll value (over-sizing risk if true bankroll has shrunk
        # since last fetch).
        preflight_wake_ts = lock_ts_t - cfg.preflight_wakeup_offset_before_lock_ms / 1000.0
        _sleep_until_ts(
            preflight_wake_ts,
            reason="wait_for_preflight",
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
                    # Optional-attribute probe per strategy/base.py: a
                    # pipeline without _bankroll_tracker degrades to 0.0.
                    _tracker = getattr(
                        closed.strategy_pipeline, "_bankroll_tracker", None)
                    if _tracker is not None:
                        last_known_bankroll = _tracker.current_bankroll()
                # Per T3-A spec: short message, no err detail (the
                # underlying exception class is captured in cycle_audit
                # via skip_reason="risk_bankroll_stale"; the operator
                # line just needs the actionable signal).
                _skip_round(
                    cfg, closed, skip_ctx,
                    skip_reason="risk_bankroll_stale",
                    decision_stage="pipeline",
                    bankroll_bnb=last_known_bankroll,
                    wake_mode=wake_mode,
                    kline_fire_offset_before_lock_ms=kline_fire_offset_before_lock_ms,
                    log_level="warn",
                    log_line=f"Skipped epoch {current_epoch}: bankroll stale",
                )
                _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
                return
            # Forward freshest bankroll to tracker (live only; dry records
            # its own settlements via dry.py after credit/debit). Risk
            # gates read from the tracker in decide_open_round below.
            #
            # On TransientRpcError above we SKIP and do NOT update the
            # tracker, so multi-iteration RPC outages leave it on an
            # unrefreshed drawdown-peak. Net effect is conservative:
            # we also refuse to bet (risk_bankroll_stale), so the
            # breaker can't mis-fire on an unrefreshed bankroll value
            # because we're not betting in the first place. Tracker
            # re-syncs on the next successful RPC fetch.
            if closed.strategy_pipeline is not None:
                closed.strategy_pipeline.record_settlement(
                    bankroll=bankroll_bnb,
                    start_at=int(open_round.start_at),
                )

        # -- Pre-cache refresh (2026-06-06): off-critical-path nonce + gas --
        # The bet send reads these from cache so its critical path is just
        # build-encode + sign + send_raw (~50ms) instead of two cold rotated
        # RPCs (~270ms). Warm ALL write endpoints (keep-alive >=30s) so the
        # rotated send lands hot. nonce prefetch is live-only (needs the signing
        # account); gas refresh + warm also exercise in dry (validates populate).
        cfg.contract.warm_write_endpoints()
        cfg.contract.refresh_gas_price()
        if not cfg.dry:
            cfg.contract.prefetch_nonce()
        info("READY", f"send-cache refreshed: {cfg.contract.send_cache_summary()}")

        # -- Single RPC poll (Candidate C, 2026-06-06) --
        # ONE getLogs catch-up before the critical-path snapshot, replacing
        # the 3-leg ramp ladder. Fires at lock_at -
        # single_poll_wakeup_offset_before_lock_ms (fixed 2500ms rail since
        # the 2026-06-06 VM re-baseline, bracketed by the CAPTURE +
        # COMPLETION + ANCHOR-CLEARANCE invariants in config.py). The
        # retained 8s periodic poll keeps the cursor within ~1 interval, so
        # this catches only the ~5-20 blocks since the last periodic. On a
        # capped/failed poll the round-aware feasibility check (INFEAS) and
        # the F0 coverage gate drive the skip, not this poll. Label
        # "single" is 6 chars, fits the log SUB_W=6.
        if cfg.rpc_poller is not None:
            single_poll_wake_ts = (
                lock_ts_t - cfg.single_poll_wakeup_offset_before_lock_ms / 1000.0
            )
            _sleep_until_ts(
                single_poll_wake_ts,
                reason="wait_for_single_poll",
                epoch=current_epoch,
            )
            # deadline = the gap to the ANCHOR POLL fire (the next event on
            # this thread, at lock - ANCHOR_POLL_OFFSET) minus the post-cap
            # processing tail. Keyed to the anchor offset, NOT critical_path:
            # the poll blocks this thread, so what bounds its time is when
            # the anchor must fire — deriving from critical_path (further
            # from lock by 305ms at canonical values) yielded a 1105ms
            # budget against the real 1000ms gap, the misalignment that
            # masked the wall-cap/anchor adjacency until the 2026-06-10
            # review. Same expression as the ANCHOR-CLEARANCE startup
            # invariant; the poller additionally takes
            # min(deadline_ms, RPC_POLL_WALL_CAP_SINGLE_MS).
            single_poll_deadline_ms = max(
                0,
                cfg.single_poll_wakeup_offset_before_lock_ms
                - _tc.ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS
                - _tc.RPC_POLL_TAIL_MARGIN_MS,
            )
            cfg.rpc_poller.poll(deadline_ms=single_poll_deadline_ms)

        # -- Anchor poll + critical-path wake (Bundle 5 v2, 2026-05-14) --
        #
        # Strategy:
        # 1. Sleep to lock - ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS (= lock - 1500ms).
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
                # SSOT wake derivation: walk back from the per-round submit
                # deadline (the same one the bet-timing guard uses below) by
                # the workload it must accommodate (kline fetch p99 + gate
                # compute + pool read). ``compute_submit_deadline_ms``
                # already accounts for the quantum-shift back-off, the
                # validator assembly window, and the one-way RPC send time;
                # the earlier inline formula recomputed two of those terms
                # and silently dropped the assembly window (a 50ms gap that
                # survived since Bundle 4). Both call sites (wake derivation
                # here, deadline check at the timing guard) now drive off
                # the same function; any change to the deadline formula
                # propagates to the wake automatically.
                anchor_deadline_ms = compute_submit_deadline_ms(
                    predicted_predecessor_milli_ts=predecessor_ms,
                    lock_ms=lock_ms_int,
                )
                dynamic_wake_ms = anchor_deadline_ms - (
                    _tc.OKX_KLINE_FETCH_RTT_P99_MS
                    + _tc.SIGNAL_COMPUTE_TIME_MS
                    + _tc.POOL_READ_TIME_MS
                )
                # The dynamic wake lands AFTER the anchor poll response: the
                # anchor fires at lock - ANCHOR_POLL_OFFSET (1500ms) and returns
                # in ~30-60ms RTT, while the dynamic target is typically
                # lock - ~880-930ms. _sleep_until_ts honors the resulting
                # ~350-420ms gap (it sleeps until any target still in the
                # future, with no minimum-sleep short-circuit). Take as-is.
                critical_path_wake_ts = dynamic_wake_ms / 1000.0
                _dynamic_lead_ms = int(round(
                    (lock_ts_t - critical_path_wake_ts) * 1000
                ))
                wake_mode = "dynamic"
                kline_fire_offset_before_lock_ms = (
                    _dynamic_lead_ms - _tc.POOL_READ_TIME_MS
                )
            else:
                wake_mode = "static"
                kline_fire_offset_before_lock_ms = (
                    cfg.critical_path_wakeup_offset_before_lock_ms
                    - _tc.POOL_READ_TIME_MS
                )
        else:
            # No rpc_poller wired (rare; usually means backtest path
            # routed here by mistake). Use static defaults.
            wake_mode = "static"
            kline_fire_offset_before_lock_ms = (
                cfg.critical_path_wakeup_offset_before_lock_ms
                - _tc.POOL_READ_TIME_MS
            )
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
            # bankroll_bnb was already resolved at the preflight wake;
            # reuse for audit on the skip path.
            ready, ready_reason = cfg.rpc_poller.is_pool_ready(current_epoch)
            if not ready:
                skip_reason = f"pool_not_ready_{ready_reason}"
                _skip_round(
                    cfg, closed, skip_ctx,
                    skip_reason=skip_reason,
                    decision_stage="pipeline",
                    bankroll_bnb=bankroll_bnb,
                    wake_mode=wake_mode,
                    kline_fire_offset_before_lock_ms=kline_fire_offset_before_lock_ms,
                )
                # skip_reason is "pool_not_ready_cold_start_in_progress"
                # or "pool_not_ready_catchup_infeasible_for_round";
                # route by the inner ready_reason.
                if ready_reason == "cold_start_in_progress":
                    info("SKIP", f"Skipped epoch {current_epoch}: cold start in progress")
                elif ready_reason == "catchup_infeasible_for_round":
                    # The same code path that sets _catchup_infeasible_for_round
                    # populates _last_catchup_detail in _is_catchup_infeasible.
                    # If we observe the flag without the detail, that's a
                    # pollster invariant violation — raise loudly rather
                    # than degrade silently.
                    _catchup = cfg.rpc_poller.last_catchup_detail
                    if _catchup is None:
                        raise InvariantError(
                            "rpc_poller_catchup_infeasible_without_detail"
                        )
                    _need_s, _have_s = _catchup[0] / 1000.0, _catchup[1] / 1000.0
                    warn(
                        "SKIP",
                        f"Skipped epoch {current_epoch}: RPC catchup infeasible "
                        f"(need {_need_s:.1f}s, have {_have_s:.1f}s)",
                    )
                else:
                    warn("SKIP", f"Skipped epoch {current_epoch}: {skip_reason}")
                _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
                return
            pool_ts_cutoff = lock_ts_t - cfg.pool_cutoff_seconds
            pool_bull_bnb, pool_bear_bnb = cfg.rpc_poller.get_pool(
                epoch=current_epoch, max_ts=pool_ts_cutoff,
            )
            pool_total = pool_bull_bnb + pool_bear_bnb
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
        # ACTUAL fetch-fire offset (wall-clock), vs the COMPUTED
        # kline_fire_offset_before_lock_ms above. Equal when the dynamic wake is
        # honored; GREATER when the critical-path wake target is already past on
        # arrival and the fetch fires earlier than the formula targeted. Both
        # _utc_now() and lock_ts_t are the same OS wall-clock UTC frame, so the
        # subtraction is a like-for-like offset.
        t_features_start_offset_ms = lock_ts_t * 1000.0 - _utc_now() * 1000.0
        # Live divergence guard: when the fetch fires meaningfully EARLIER than
        # the computed dynamic-wake target, the critical-path wake is being
        # bypassed and the dynamic timing optimization is inert for the round.
        # Surface it as an ALERT (routes to Discord) so a future regression
        # shows up in real time, not just in cycle_audit forensics.
        _wake_alert = _wake_divergence_alert_message(
            actual_offset_ms=t_features_start_offset_ms,
            computed_offset_ms=kline_fire_offset_before_lock_ms,
        )
        if _wake_alert is not None:
            warn("ALERT", _wake_alert)
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
        # Regime-drift telemetry (guard audit 5.3): feed this round's
        # max-of-3 kline fetch RTT into the rolling p99 monitor vs
        # OKX_KLINE_FETCH_RTT_P99_MS. Observe only on clean 3-symbol
        # fetches (all positive). Wrapped so a monitor bug can never break
        # the bet path.
        try:
            _rtts = [
                _kline_timing_get(gate, "btc_ms"),
                _kline_timing_get(gate, "eth_ms"),
                _kline_timing_get(gate, "sol_ms"),
            ]
            if all(r is not None and r > 0 for r in _rtts):
                _rtt_alert = _OKX_KLINE_RTT_MONITOR.observe(max(_rtts))
                if _rtt_alert is not None:
                    warn("ALERT", _rtt_alert)
        except Exception:  # noqa: BLE001 — telemetry must never break betting
            pass

        # D3: COOLDOWN LIFTED edge-detect — runs for every decision (bet OR
        # skip). If we WERE in a drawdown cooldown and this round is no longer a
        # cooldown skip (betting resumed, or a different skip reason), fire the
        # LIFTED alert once and clear the flag. Cooldown gates only NEW bets, so
        # this is purely an operator-visibility signal.
        _cd_reason = decision.skip_reason or ""
        if closed.in_cooldown and _cd_reason not in (
            "risk_cooldown_active", "risk_drawdown_breaker_fired",
        ):
            closed.in_cooldown = False
            send_cooldown_lifted_alert(
                channel=(DRY_CHANNEL if cfg.dry else LIVE_CHANNEL),
                bankroll_bnb=bankroll_bnb,
            )

        if decision.action != "BET":
            reason = decision.skip_reason or ""
            if reason == "":
                raise InvariantError("policy_skip_missing_reason")

            _skip_round(
                cfg, closed, skip_ctx,
                skip_reason=reason,
                decision_stage="pipeline",
                bankroll_bnb=bankroll_bnb,
                wake_mode=wake_mode,
                kline_fire_offset_before_lock_ms=kline_fire_offset_before_lock_ms,
                decision=decision,
                decision_latency_ms=t_decision_ready_ms - t_features_start_ms,
                pool_bull_bnb=pool_bull_bnb,
                pool_bear_bnb=pool_bear_bnb,
                with_fetch_ms=True,
            )
            # T3-A: reason-routed SKIP with custom wording per reason.
            # In-scope reasons get bespoke prose; out-of-scope reasons
            # keep generic "Skipped epoch X: <reason>" with a TODO
            # comment for the data-plumbing follow-up.
            if reason == "kline_fetch_transient_failure" and gate is not None:
                classification = _classify_kline_failure(gate.last_fetch_results)
                if classification is not None:
                    subtype, body = classification
                    _prefix_per_subtype = {
                        "kline_publish_delay": "incomplete kline data",
                        "kline_unreachable": "kline source unreachable",
                        "kline_http_error": "kline source returned error",
                    }
                    prefix = _prefix_per_subtype[subtype]
                    warn(
                        "SKIP",
                        f"Skipped epoch {current_epoch}: {prefix} ({body})",
                    )
                else:
                    # Defensive: gate flagged the transient skip but
                    # last_fetch_results came back empty/all-ok. Shouldn't
                    # happen given the gate's own state-management, but
                    # fall back to generic WARN rather than asserting.
                    warn(
                        "SKIP",
                        f"Skipped epoch {current_epoch}: kline_fetch_transient_failure",
                    )
            elif reason == "gate_no_signal":
                info("SKIP", f"Skipped epoch {current_epoch}: gate did not fire")
            elif reason == "risk_drawdown_breaker_fired":
                # skip_context is required for this reason — pipeline's
                # StrategyPipelineDecision.__post_init__ enforces it.
                # Direct access; if anything is wrong, raise loudly.
                _ctx = decision.skip_context
                warn(
                    "SKIP",
                    f"Skipped epoch {current_epoch}: drawdown breaker fired "
                    f"({_ctx['drawdown_pct']:.1f}% from peak, "
                    f"threshold {_ctx['threshold_pct']:.0f}%)",
                )
                # D3: COOLDOWN ENTERED alert on the trip edge (once per entry).
                if not closed.in_cooldown:
                    closed.in_cooldown = True
                    _cd_rounds = int(cfg.strategy.risk.cooldown_rounds)
                    send_cooldown_entered_alert(
                        channel=(DRY_CHANNEL if cfg.dry else LIVE_CHANNEL),
                        drawdown_pct=float(_ctx["drawdown_pct"]),
                        threshold_pct=float(_ctx["threshold_pct"]),
                        bankroll_bnb=bankroll_bnb,
                        cooldown_rounds=_cd_rounds,
                        approx_hours=_cd_rounds * cfg.interval_seconds / 3600.0,
                    )
            elif reason == "risk_cooldown_active":
                _ctx = decision.skip_context
                closed.in_cooldown = True  # D3: stay marked until LIFTED edge
                info(
                    "SKIP",
                    f"Skipped epoch {current_epoch}: cooldown active "
                    f"({_ctx['rounds_remaining']} rounds remaining)",
                )
            elif reason == "pool_below_minimum":
                _ctx = decision.skip_context
                info(
                    "SKIP",
                    f"Skipped epoch {current_epoch}: pool below minimum "
                    f"({_ctx['pool_bnb']:.2f} BNB < "
                    f"{_ctx['min_pool_bnb_at_cutoff']:.2f} BNB threshold)",
                )
            else:
                # Unrecognized reason — render generically.
                info("SKIP", f"Skipped epoch {current_epoch}: {reason}")
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 11: Execution timing guard. Abort if wall-clock is past
        # the bet-submit deadline -- TX submitted later than this is
        # unlikely to mine in time and would revert (gas burn).
        #
        # Bundle 5 v2 (2026-05-14): two-track deadline driven by the
        # per-round anchor poll fired earlier at lock - 1500ms.
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
        # ``_utc_now() * 1000`` (OS-clock UTC ms, chrony-disciplined) so the
        # bot doesn't fire early.
        lock_ms = int(lock_ts_t * 1000)
        if round_anchor is not None:
            predecessor_ms = predict_predecessor_milli_ts(
                anchor_milli_ts=round_anchor.milli_ts, lock_ms=lock_ms,
            )
            deadline_ms = compute_submit_deadline_ms(
                predicted_predecessor_milli_ts=predecessor_ms, lock_ms=lock_ms,
            )
            deadline_source = "dynamic"
            # Observability (guard audit 5.1): the quantum-shift guard inside
            # compute_submit_deadline_ms backs the deadline off a FULL block
            # (~450ms) when the predicted predecessor lands within one 50ms
            # quantum of lock. That is a large discrete tightening — surface
            # each time it fires so an operator can see how often a full-block
            # penalty is paid (and whether the 450/50 pairing still matches
            # the live block time).
            if (predecessor_ms + _tc.BSC_QUANTUM_MS) >= lock_ms:
                warn(
                    "ALERT",
                    f"QUANTUM_BACKOFF epoch={current_epoch} "
                    f"predicted_predecessor_gap_ms={lock_ms - predecessor_ms} "
                    f"quantum_ms={_tc.BSC_QUANTUM_MS} backoff_ms={_tc.BSC_BLOCK_TIME_MS} "
                    f"reason=predecessor_within_one_quantum_of_lock",
                )
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
            # "Past safe submit time" = how late we are vs the deadline.
            # margin_ms / submit_offset_ms / source remain in cycle_audit
            # if offline analysis needs them.
            late_ms = int(now_utc_ms - deadline_ms)
            warn(
                "SKIP",
                f"Skipped epoch {current_epoch}: too late to submit bet "
                f"({late_ms}ms past safe submit time)",
            )
            _skip_round(
                cfg, closed, skip_ctx,
                skip_reason="too_close_to_lock_for_bet",
                decision_stage="timing_guard",
                bankroll_bnb=bankroll_bnb,
                wake_mode=wake_mode,
                kline_fire_offset_before_lock_ms=kline_fire_offset_before_lock_ms,
                decision=decision,
                decision_latency_ms=t_decision_ready_ms - t_features_start_ms,
                pool_bull_bnb=pool_bull_bnb,
                pool_bear_bnb=pool_bear_bnb,
                with_fetch_ms=True,
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
            info("BET", f"min_bet_only: clamping {computed_amount_wei / BNB_WEI:.4f} -> {amount_wei / BNB_WEI:.4f} BNB")

        tx_submit = None
        if not cfg.dry:
            # Pre-send cache readiness (2026-06-06): nonce + gas were fetched
            # OFF the critical path at the preflight wake. Fail-LOUD SKIP (never
            # a silent live-fetch) if a cache is unpopulated/stale — that means
            # the preflight refresh did not run (a wiring bug to surface), not a
            # condition to paper over on the hot path.
            if not cfg.contract.send_caches_ready():
                warn(
                    "SKIP",
                    f"Skipped epoch {current_epoch}: send caches not ready "
                    f"({cfg.contract.send_cache_summary()})",
                )
                _skip_round(
                    cfg, closed, skip_ctx,
                    skip_reason="risk_send_cache_unready",
                    decision_stage="send_cache_check",
                    bankroll_bnb=bankroll_bnb,
                    wake_mode=wake_mode,
                    kline_fire_offset_before_lock_ms=kline_fire_offset_before_lock_ms,
                    decision=decision,
                    decision_latency_ms=t_decision_ready_ms - t_features_start_ms,
                    pool_bull_bnb=pool_bull_bnb,
                    pool_bear_bnb=pool_bear_bnb,
                    with_fetch_ms=True,
                )
                _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
                return
            # Gas-cap sanity check: skip the bet if eth.gas_price has run
            # away from MAX_GAS_PRICE_WEI. Submitting a bet at the cap
            # while the network is much higher would land at the back of
            # the priority queue (likely miss the lock-block inclusion
            # window — gas burned for no inclusion). CRITICAL alert; the
            # operator must lift the cap before resuming.
            try:
                cfg.contract.assert_gas_cap_not_breached()
            except GasPriceCapBreachedError as gas_err:
                try:
                    suggested_wei = int(cfg.contract.suggest_gas_price_wei())
                except Exception:
                    suggested_wei = -1
                send_gas_cap_breach_alert(
                    path="bet",
                    suggested_wei=suggested_wei,
                    cap_wei=int(MAX_GAS_PRICE_WEI),
                    epoch=current_epoch,
                )
                warn(
                    "SKIP",
                    f"Skipped epoch {current_epoch}: gas cap breached ({gas_err})",
                )
                _skip_round(
                    cfg, closed, skip_ctx,
                    skip_reason="gas_cap_breached",
                    decision_stage="gas_cap_check",
                    bankroll_bnb=bankroll_bnb,
                    wake_mode=wake_mode,
                    kline_fire_offset_before_lock_ms=kline_fire_offset_before_lock_ms,
                    decision=decision,
                    decision_latency_ms=t_decision_ready_ms - t_features_start_ms,
                    pool_bull_bnb=pool_bull_bnb,
                    pool_bear_bnb=pool_bear_bnb,
                    with_fetch_ms=True,
                )
                _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
                return
            gas_price_wei = MAX_GAS_PRICE_WEI
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
            if tx_submit is None:
                raise InvariantError("live_bet_submit_missing")
            # BET SUBMITTED: the TX broadcast (tx_hash exists). Projected
            # bankroll = pre-bet wallet − stake − bet gas cap (what bankroll is
            # IF the bet registers). The post-receipt alert below reports the
            # actual fresh balance.
            projected_bankroll = bankroll_bnb - amount_bnb - MAX_GAS_COST_BET_BNB
            info(
                "BET",
                f"Bet {amount_bnb:.4f} BNB on {bet_side} for epoch {current_epoch} "
                f"(tx {_truncate_tx_hash(tx_submit.tx_hash)}, "
                f"projected bankroll: {projected_bankroll:.4f} BNB, "
                f"fetch_fire={t_features_start_offset_ms:.0f}ms vs "
                f"computed={kline_fire_offset_before_lock_ms}ms before lock)",
            )
            bet_ledger.record_submitted(
                ledger_path=paths.LIVE_BETS_LEDGER_PATH,
                epoch=current_epoch, side=bet_side, amount_bnb=amount_bnb,
                tx_hash=tx_submit.tx_hash, bankroll_after_bnb=projected_bankroll,
            )
            send_bet_submitted_alert(
                channel=LIVE_CHANNEL, epoch=current_epoch, side=bet_side, amount_bnb=amount_bnb,
                projected_bankroll_bnb=projected_bankroll,
            )
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
            append_jsonl("var/live/latency.jsonl", latency_record)
            # Receipt classification → exactly ONE post-receipt alert.
            #   CONFIRMED  : status=1, before lock (bet registered)
            #   LATE       : status=0, at/after lock (PCS late-lock revert)
            #   REVERTED   : status=0, before lock (other revert)
            #   DROPPED    : no receipt within the wait window (TX gone)
            # All revert/drop cases rolled back msg.value (gas-only loss).
            included_late = (
                tx_submit.included_block_timestamp is not None
                and int(tx_submit.included_block_timestamp) >= int(lock_ts_t)
            )
            # Actual gas (gasUsed x effectiveGasPrice), not the cap. None on
            # DROPPED (no receipt) -> ledger gas field unwritten.
            gas_bnb = bet_ledger.actual_gas_bnb(
                gas_used=tx_submit.gas_used,
                effective_gas_price_wei=tx_submit.effective_gas_price_wei,
            )
            conf_status = bet_ledger.record_confirmation(
                ledger_path=paths.LIVE_BETS_LEDGER_PATH,
                epoch=current_epoch,
                chain_status=tx_submit.chain_status,
                included_block_number=tx_submit.included_block_number,
                included_late=included_late,
                gas_paid_bnb=gas_bnb,
            )
            # Fresh wallet read for the post-receipt alert bankroll. Off the
            # critical path. Read-your-writes: the bet TX was sent + confirmed
            # on the CURRENT node, so read on THAT node (no rotate) to avoid a
            # sibling node lagging the bet block and returning pre-bet state
            # (BET WON stale-bankroll fix, 2026-06-03). Fall back: non-rotating
            # -> rotating -> projected estimate.
            try:
                fresh_bankroll = cfg.contract.wallet_balance_bnb_no_rotate(
                    cfg.wallet_address
                )
            except Exception:  # noqa: BLE001
                try:
                    fresh_bankroll = float(
                        cfg.contract.wallet_balance_bnb(cfg.wallet_address)
                    )
                except Exception:  # noqa: BLE001
                    fresh_bankroll = projected_bankroll
            if conf_status == "CONFIRMED":
                send_bet_confirmed_alert(channel=LIVE_CHANNEL, epoch=current_epoch, bankroll_bnb=fresh_bankroll)
            elif conf_status == "LATE":
                warn(
                    "ALERT",
                    f"Bet TX included LATE for epoch {current_epoch}: "
                    f"included_block_ts={int(tx_submit.included_block_timestamp)} "
                    f"lock_ts={int(lock_ts_t)} "
                    f"submit_offset_ms={bet_submit_offset_ms:.0f}",
                )
                send_bet_late_alert(channel=LIVE_CHANNEL, epoch=current_epoch, bankroll_bnb=fresh_bankroll)
            elif conf_status == "REVERTED":
                warn(
                    "ALERT",
                    f"Bet TX REVERTED for epoch {current_epoch} "
                    f"(status=0, before lock): tx {_truncate_tx_hash(tx_submit.tx_hash)} "
                    f"block={tx_submit.included_block_number}",
                )
                send_bet_reverted_alert(channel=LIVE_CHANNEL, epoch=current_epoch, bankroll_bnb=fresh_bankroll)
            elif conf_status == "DROPPED":
                warn(
                    "ALERT",
                    f"Bet TX DROPPED for epoch {current_epoch}: no receipt within "
                    f"{cfg.bet_tx_receipt_timeout_seconds}s (tx {_truncate_tx_hash(tx_submit.tx_hash)})",
                )
                send_bet_dropped_alert(channel=LIVE_CHANNEL, epoch=current_epoch, bankroll_bnb=fresh_bankroll)
            # Bankroll pair for the mode-agnostic BET audit row hoisted below
            # the dry/live split. Live records the PROJECTED bankroll (wallet −
            # stake − gas cap) — defined regardless of CONFIRMED/LATE/REVERTED/
            # DROPPED, mirroring the dry branch's post-debit value.
            _audit_bk_before = bankroll_bnb
            _audit_bk_after = projected_bankroll
        else:
            # Step 14: Dry bookkeeping (including gas proxy) + record.
            if closed.simulated_bankroll_bnb is None:
                raise InvariantError("dry_bankroll_uninitialized")

            bankroll_before_bet = closed.simulated_bankroll_bnb
            closed.simulated_bankroll_bnb -= amount_bnb + MAX_GAS_COST_BET_BNB
            bankroll_after_bet = closed.simulated_bankroll_bnb

            info(
                "BET",
                f"Bet {amount_bnb:.4f} BNB on {bet_side} for epoch {current_epoch} "
                f"(bankroll: {bankroll_after_bet:.4f} BNB, "
                f"fetch_fire={t_features_start_offset_ms:.0f}ms vs "
                f"computed={kline_fire_offset_before_lock_ms}ms before lock)",
            )
            dry_record_bet(
                closed,
                epoch=current_epoch,
                side=bet_side,
                amount_bnb=amount_bnb,
                p_final=pred_p_final,
                bankroll_before_bet_bnb=bankroll_before_bet,
                bankroll_after_bet_bnb=bankroll_after_bet,
            )
            # Bet-lifecycle ledger (dry): SUBMITTED record only. No tx_hash in
            # dry mode (no on-chain submission).
            bet_ledger.record_submitted(
                ledger_path=paths.DRY_BETS_LEDGER_PATH,
                epoch=current_epoch, side=bet_side, amount_bnb=amount_bnb,
                tx_hash="", bankroll_after_bnb=bankroll_after_bet,
            )
            # Discord (dry): placement alert on the dry channel, same body as
            # live's BET SUBMITTED. Dry placement is atomic (no separate
            # confirm), so this is the dry analog of live's placement alert;
            # the simulated post-debit bankroll stands in for live's projected.
            send_bet_submitted_alert(
                channel=DRY_CHANNEL, epoch=current_epoch, side=bet_side, amount_bnb=amount_bnb,
                projected_bankroll_bnb=bankroll_after_bet,
            )
            # Bankroll pair for the mode-agnostic BET audit row (hoisted below).
            _audit_bk_before = bankroll_before_bet
            _audit_bk_after = bankroll_after_bet

        # Step 14b: BET cycle-audit row — SINGLE mode-agnostic write site so
        # live and dry both emit it. Previously this lived only inside the dry
        # branch, leaving live cycle_audit.csv with zero BET rows (the bankroll
        # pair is mode-selected above: live=projected, dry=post-debit sim).
        record_cycle_audit(
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
            bankroll_before_action_bnb=_audit_bk_before,
            bankroll_after_action_bnb=_audit_bk_after,
            decision=decision,
            decision_latency_ms=t_decision_ready_ms - t_features_start_ms,
            pool_bull_bnb=pool_bull_bnb,
            pool_bear_bnb=pool_bear_bnb,
            btc_fetch_ms=_kline_timing_get(gate, "btc_ms"),
            eth_fetch_ms=_kline_timing_get(gate, "eth_ms"),
            sol_fetch_ms=_kline_timing_get(gate, "sol_ms"),
            wake_mode=wake_mode,
            kline_fire_offset_before_lock_ms=kline_fire_offset_before_lock_ms,
            t_features_start_offset_ms=t_features_start_offset_ms,
            btc_fetch_result=_kline_result_get(gate, "btc"),
            eth_fetch_result=_kline_result_get(gate, "eth"),
            sol_fetch_result=_kline_result_get(gate, "sol"),
        )

        # Per-round GATE FETCH TIMING + GATE SIGNAL FIRE info emissions
        # were dropped at Phase B v2 (2026-05-18): cycle_audit.csv captures
        # the same data (btc/eth/sol_fetch_ms, wake_mode,
        # kline_fire_offset_before_lock_ms, bet_side, bet_size_bnb)
        # byte-equivalent. Operator-facing stdout no longer needs them.

        # Step 15: Sleep until claim + claim scan.
        _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
        return


def _epoch_handshake(cfg: RuntimeConfig) -> tuple[Round, Round, int]:
    """RPC-only epoch alignment.

    Returns (locked_round_stub, open_round_stub, current_epoch).
    """
    for idx, delay_seconds in enumerate([0] + list(RETRY_BACKOFF_SECONDS)):
        if delay_seconds > 0:
            sleep_seconds(delay_seconds)
        try:
            current_epoch = int(cfg.contract.current_epoch())
        except TransientRpcError as e:
            warn("RETRY", f"epoch_handshake: rpc_current_epoch attempt={idx} err={e}")
            continue

        locked_epoch = current_epoch - 1
        if locked_epoch <= 0:
            warn("RETRY", f"epoch_handshake: locked_epoch_nonpositive attempt={idx}")
            continue

        try:
            locked_rd = cfg.contract.round_data(locked_epoch)
            open_rd = cfg.contract.round_data(current_epoch)
        except TransientRpcError as e:
            warn("RETRY", f"epoch_handshake: rpc_round_data attempt={idx} err={e}")
            continue

        if locked_rd.lock_ts <= 0:
            warn("RETRY", f"epoch_handshake: locked_lock_ts_zero attempt={idx}")
            continue
        # Two other zero-state conditions appear during the
        # fresh-spawn-during-round-transition window: executeRound() has
        # incremented currentEpoch but not yet written lock_price for the
        # new locked epoch / lock_ts for the new open epoch. The
        # RETRY_BACKOFF_SECONDS budget is sized to span this settlement
        # window (cumulative ~36s after the 5th retry).
        if (
            locked_rd.lock_price_usd is None
            or locked_rd.lock_price_usd <= 0.0
        ):
            warn("RETRY", f"epoch_handshake: locked_lock_price_zero attempt={idx}")
            continue
        if open_rd.lock_ts <= 0:
            warn("RETRY", f"epoch_handshake: open_lock_ts_zero attempt={idx}")
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
        return locked_round, open_round, current_epoch

    raise InvariantError("epoch_handshake_exhausted")


def _current_bankroll_estimate(closed: RuntimeState) -> float:
    """Best-effort current bankroll for the settled-alert "new bankroll"
    display. Reads the pipeline's bankroll tracker if wired; falls back to
    0.0 (the alert's delta is the load-bearing number — absolute is display
    only). Never raises."""
    # noinspection PyBroadException
    try:
        pipeline = closed.strategy_pipeline
        if pipeline is not None:
            tracker = getattr(pipeline, "_bankroll_tracker", None)
            if tracker is not None:
                return float(tracker.current_bankroll())
    except Exception:
        pass
    return 0.0


def _reconcile_live_bets(cfg: RuntimeConfig, closed: RuntimeState) -> None:
    """Reconcile open live bets against on-chain RoundData at settle-time.
    Fires the LOSS alert only (Option B); WIN/REFUND alerts fire from the
    claim-scan path at claim-tx-confirm. Reads a FRESH wallet balance for
    the alert's "new bankroll" display so sequential in-flight bets don't
    skew it (Fix #3). Fail-soft: never raises."""
    if cfg.dry:
        return
    # Fresh wallet balance at fire-time (already reflects any prior bets'
    # placement debits). Best-effort: fall back to the tracker estimate on
    # RPC failure rather than block reconciliation.
    try:
        fresh_bankroll = float(cfg.contract.wallet_balance_bnb(cfg.wallet_address))
    except Exception:  # noqa: BLE001
        fresh_bankroll = _current_bankroll_estimate(closed)
    # noinspection PyBroadException
    try:
        bet_ledger.reconcile(
            ledger_path=paths.LIVE_BETS_LEDGER_PATH,
            contract=cfg.contract,
            treasury_fee_fraction=cfg.treasury_fee_fraction,
            fresh_bankroll_bnb=fresh_bankroll,
            buffer_seconds=cfg.buffer_seconds,
            now_ts=int(_utc_now()),
            wallet_address=cfg.wallet_address,
            lost_alert_fn=functools.partial(send_bet_settled_alert, channel=LIVE_CHANNEL),
            dropped_alert_fn=functools.partial(send_bet_dropped_alert, channel=LIVE_CHANNEL),
        )
    except Exception as e:  # noqa: BLE001
        warn("ALERT", f"bet ledger reconcile failed: {e}")


def _sleep_and_claim(cfg: RuntimeConfig, closed: RuntimeState, claim_epoch: int) -> None:
    # Bounded local retry around ``contract.close_ts`` — the only RPC call
    # in this function with real budget before the claim wake. Mirrors the
    # pattern in ``_epoch_handshake``. Exhaust → InvariantError → bot crashes
    # → systemd restart (cleaner than top-level sleep-and-retry).
    close_ts: int | None = None
    for idx, delay_seconds in enumerate([0] + list(RETRY_BACKOFF_SECONDS)):
        if delay_seconds > 0:
            sleep_seconds(delay_seconds)
        try:
            close_ts = int(cfg.contract.close_ts(claim_epoch))
            break
        except TransientRpcError as e:
            warn("RETRY", f"close_ts: rpc attempt={idx} err={e}")
            continue
    if close_ts is None:
        raise InvariantError("close_ts_retry_exhausted")
    if close_ts <= 0:
        raise InvariantError("close_ts_invalid")

    claim_ts = close_ts + cfg.buffer_seconds + _RPC_ALIGNMENT_PADDING_SECONDS
    _sleep_until_ts(claim_ts, reason="wait_for_claim", epoch=claim_epoch)

    # Epoch handshake to refresh round state (both modes).
    locked_round2, _open_round2, current_epoch2 = _epoch_handshake(cfg)

    if not cfg.dry:
        # Reconcile FIRST so the ledger carries SETTLED_WON/SETTLED_REFUND
        # (with per-bet delta) before the claim scan reads it — the claim
        # path fires WON/REFUND alerts off those records (Option B). Reconcile
        # fires the LOSS alert itself; it never moves money. Idempotent +
        # crash-safe.
        _reconcile_live_bets(cfg, closed)
        # Claim scan collects winnings/refunds and fires WON/REFUND alerts at
        # claim-tx-confirm (bets_ledger_path threads the ledger in). Fail-soft
        # on transient RPC: the next iteration's scan re-detects.
        try:
            claim_scan_cursor(
                contract=cfg.contract,
                wallet_address=cfg.wallet_address,
                dry=False,
                cursor_path=paths.LIVE_CLAIM_CURSOR_PATH,
                locked_epoch=locked_round2.epoch,
                current_epoch=current_epoch2,
                now_ts=int(_utc_now()),  # OS-clock UTC (chrony-disciplined; see module clock-sync note); compared to chain-anchored close timestamps
                buffer_seconds=cfg.buffer_seconds,
                page_size=100,
                gas_limit=GAS_LIMIT_CLAIM,
                claim_tx_receipt_timeout_seconds=cfg.claim_tx_receipt_timeout_seconds,
                bets_ledger_path=paths.LIVE_BETS_LEDGER_PATH,
            )
        except TransientRpcError as e:
            warn("ALERT", f"claim scan failed: rpc_transient err={e}")

    # Dry: settle simulated bets against oracle price.
    dry_settle_available_bets(cfg, closed)


# Divergence tolerance (ms) between the ACTUAL fetch-fire offset and the
# COMPUTED dynamic-wake offset before an ALERT fires. ~0 when the wake is
# honored (Regime A); the tolerance absorbs pool-read jitter between waking
# and the t_features_start capture.
#
# This is a FIXED absolute epsilon, NOT scaled to the dynamic-wake lead
# (typically ~880-930ms before lock). It MUST stay well below the minimum
# expected dynamic lead — if a future regime tightens the lead toward this
# value, 50ms could begin to mask a real bypass and this should become a
# fraction-of-lead check or a config knob. Safe today (50ms << ~900ms lead).
_WAKE_DIVERGENCE_ALERT_TOLERANCE_MS: float = 50.0


def _wake_divergence_alert_message(
    *, actual_offset_ms: float, computed_offset_ms: float
) -> str | None:
    """ALERT prose when the kline fetch fired EARLIER than the dynamic-wake
    target, else ``None``.

    Both args are ms-before-lock offsets. A positive divergence (actual >
    computed beyond the tolerance) means the critical-path wake was bypassed
    and the dynamic timing optimization was inert for the round. Returns
    ``None`` when aligned within ``_WAKE_DIVERGENCE_ALERT_TOLERANCE_MS``.
    """
    divergence_ms = actual_offset_ms - computed_offset_ms
    if divergence_ms <= _WAKE_DIVERGENCE_ALERT_TOLERANCE_MS:
        return None
    return (
        f"DYNAMIC_WAKE_BYPASS divergence_ms={divergence_ms:.0f} "
        f"wake_target_offset={computed_offset_ms:.0f} "
        f"actual_offset={actual_offset_ms:.0f} "
        f"reason=sleep_threshold_or_late_arrival"
    )


def _sleep_until_ts(target_ts: float, *, reason: str, epoch: int | None = None) -> None:
    """Sleep until OKX/UTC time hits ``target_ts``.

    *target_ts* is treated as a chain-anchored / OKX-frame UTC second,
    compared against ``_utc_now()`` (the OS wall clock, chrony-disciplined
    on the VM per the README setup). There is no minimum-sleep
    short-circuit: any target still in the future is slept until, so
    sub-second dynamic wakes are honored exactly. A target already at or
    past now returns immediately.
    """
    remaining = target_ts - _utc_now()
    if remaining <= 0:
        return

    while True:
        remaining2 = target_ts - _utc_now()
        if remaining2 <= 0:
            return
        sleep_seconds(min(1.0, remaining2))
