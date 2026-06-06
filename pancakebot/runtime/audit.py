"""Trade and cycle audit CSV writers plus pool snapshot and settled-epoch recovery helpers."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from pancakebot.constants import BNB_WEI
from pancakebot.log import warn
from pancakebot.util import InvariantError
from pancakebot.types import Round


def _ensure_parent_dir(path: str) -> None:
    from pancakebot.util import ensure_parent_dir
    ensure_parent_dir(path)


# -- Trades audit CSV (per-bet settlement records) ---------------------------

def ensure_audit_csv(path: str) -> list[str]:
    # ``expected_profit_bnb`` removed 2026-04-26 (lean&clean refactor):
    # the StrategyPipelineDecision field that fed it was removed, and the
    # column was only ever written empty since 2025 anyway.
    header_cols = [
        "epoch",
        "placed_ts",
        "bet_side",
        "bet_bnb",
        "pred_win_probability",
        "p_final",
        "cutoff_bull_bnb",
        "cutoff_bear_bnb",
        "final_bull_bnb",
        "final_bear_bnb",
        "settled_ts",
        "outcome",
        "pnl_bnb",
        "bankroll_before_bet_bnb",
        "bankroll_after_bet_bnb",
        "bankroll_before_settle_bnb",
        "bankroll_after_settle_bnb",
    ]
    p = Path(path)
    if not p.exists():
        _ensure_parent_dir(path)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header_cols)
    return header_cols


def append_audit_row(path: str, row: dict[str, object]) -> None:
    cols = ensure_audit_csv(path)
    # Append row in stable column order.
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row.get(c, "") for c in cols])


# -- Cycle audit CSV (per-round decision records) ----------------------------

def ensure_cycle_audit_csv(path: str, *, reset: bool = False) -> list[str]:
    header_cols = [
        "cycle_ts",
        "current_epoch",
        "locked_epoch",
        "lock_ts",
        "cutoff_ts",
        "locked_price_bnbusd",
        "bankroll_before_action_bnb",
        "bankroll_after_action_bnb",
        "observed_total_pool_bnb",
        "observed_bull_pool_bnb",
        "observed_bear_pool_bnb",
        "observed_total_bets",
        "observed_bull_bets",
        "observed_bear_bets",
        "cutoff_used_total_pool_bnb",
        "cutoff_used_bull_pool_bnb",
        "cutoff_used_bear_pool_bnb",
        "cutoff_used_total_bets",
        "cutoff_used_bull_bets",
        "cutoff_used_bear_bets",
        "router_mode",
        "pipeline_last_settled_epoch",
        "action",
        "decision_stage",
        "bet_side",
        "bet_size_bnb",
        # Removed 2026-04-26 (lean&clean): the underlying
        # StrategyPipelineDecision fields these columns sourced from
        # were deleted (p_bull / expected_profit_bnb / selector_score_bnb /
        # selected_strategy). selected_side_probability was a derived
        # column over p_bull. All four columns are gone.
        "decision_latency_ms",
        "skip_reason",
        # Per-symbol kline-fetch RTT (ms), captured by MomentumGate via
        # gate.last_fetch_timing after the critical-path completes.
        # Persisted to audit for offline tail-distribution analysis;
        # empty string when the gate didn't run (cold-start / pre-strategy
        # phases) or when the symbol was disabled (BNB is disabled today).
        # Added 2026-05-12.
        "btc_fetch_ms",
        "eth_fetch_ms",
        "sol_fetch_ms",
        # Per-round wake-mode + kline fetch fire offset for offline
        # publish-delay-tail analysis. Added 2026-05-17.
        # ``wake_mode``: "dynamic" (Bundle 5 v2 anchor-driven) or
        #   "static" (anchor poll timed out / not wired). Empty string
        #   when an early-skip path (e.g. risk_bankroll_stale at the
        #   preflight wake) fired before the anchor poll resolved.
        # ``kline_fire_offset_before_lock_ms``: ms before ``lock_at`` at
        #   which the parallel kline GETs fired (= critical_path lead -
        #   POOL_READ_TIME_MS). Empty string for the same early-skip
        #   paths.
        "wake_mode",
        "kline_fire_offset_before_lock_ms",
        # ACTUAL fetch-fire offset (Added 2026-06-02): ms before ``lock_at`` at
        # which ``decide_open_round`` was actually called (= lock_ts -
        # wall-clock at the decide call). ``kline_fire_offset_before_lock_ms``
        # above is the COMPUTED dynamic-wake target; this is what really
        # happened. When the dynamic wake is honored the two match within a few
        # ms; when ``_sleep_until_ts(critical_path_wake)`` no-ops (wake already
        # past — "Regime B"), this is GREATER than the computed value (the fetch
        # fired earlier than the formula targeted). Surfaces the dynamic-wake
        # bypass. Empty for early-skip paths before the decide call.
        "t_features_start_offset_ms",
        # Per-symbol kline-fetch RESULT codes for the joint
        # fail-rate-vs-fire-time distribution. Added 2026-05-17.
        # Codes (see MomentumGate.last_fetch_results docstring):
        #   "ok" / "partial:got_N_expected_M" / "error:<detail>" /
        #   "not_fetched"
        "btc_fetch_result",
        "eth_fetch_result",
        "sol_fetch_result",
    ]
    p = Path(path)
    if reset or not p.exists():
        _ensure_parent_dir(path)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header_cols)
        _CYCLE_AUDIT_HEADER_OK_PATHS.add(str(p.resolve()))
    else:
        # Header-mismatch auto-rotate: if an existing file's first row
        # doesn't match the current header_cols (e.g. after a schema
        # change), preserve the old file with a timestamped suffix and
        # write a fresh file. This keeps downstream tooling that
        # assumes a fixed column count from breaking silently.
        #
        # Per-write call cost (Y5): once the header has matched OR been
        # rotated on this process, cache the path so subsequent writes
        # skip the I/O. The cache is process-local; a fresh process
        # always re-validates (catches off-process schema drift).
        resolved = str(p.resolve())
        if resolved in _CYCLE_AUDIT_HEADER_OK_PATHS:
            return header_cols
        try:
            with open(path, "r", newline="") as f:
                r = csv.reader(f)
                existing_header = next(r, None)
        except (OSError, csv.Error, StopIteration):
            # Malformed CSV (partial-write crash artifact, encoding glitch,
            # truncated bytes, etc.). Treat as "no usable header" and let
            # the rotate path replace it. Bot startup must not be blocked
            # by a corrupted file. (Y4 fix 2026-05-12.)
            existing_header = None
        if existing_header is not None and existing_header == header_cols:
            _CYCLE_AUDIT_HEADER_OK_PATHS.add(resolved)
            return header_cols
        # Mismatch: rotate.
        import time as _t
        ts = _t.strftime("%Y%m%d-%H%M%S", _t.gmtime())
        rotated_base = f"{path}.pre-rotate-{ts}"
        rotated = rotated_base
        # Y3 fix: use os.replace (cross-platform overwrite-safe; Windows
        # os.rename raises on existing target) and handle the rare cases
        # where the destination is locked (Excel/pandas reader holding
        # a handle) or another collision occurs. Unique-suffix fallback
        # tries `<base>-N` for N in 1..99 before giving up. As an absolute
        # last resort, write the new header to a parallel file with the
        # rotated suffix as the active path; bot startup is not blocked
        # by a stuck rotate.
        rotated_ok = False
        try:
            os.replace(path, rotated)
            rotated_ok = True
        except OSError:
            for n in range(1, 100):
                candidate = f"{rotated_base}-{n}"
                try:
                    os.replace(path, candidate)
                    rotated = candidate
                    rotated_ok = True
                    break
                except OSError:
                    continue
        if not rotated_ok:
            # Absolute fallback: leave the old file in place and write the
            # new schema to a parallel path. Surfaced via warn log so the
            # operator notices.
            warn("ALERT",
                 f"could not rotate {path}; writing new schema to {path}.rotate-fallback")
            fallback_path = f"{path}.rotate-fallback"
            with open(fallback_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header_cols)
            _CYCLE_AUDIT_HEADER_OK_PATHS.add(str(Path(fallback_path).resolve()))
            return header_cols
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header_cols)
        _CYCLE_AUDIT_HEADER_OK_PATHS.add(resolved)
    return header_cols


# Process-local cache of cycle-audit CSV paths whose header has been
# verified to match ``header_cols``. Avoids re-reading + re-validating
# the file on every append. Cleared per-process (fresh bot start
# re-validates). (Y5 perf optimization 2026-05-12.)
_CYCLE_AUDIT_HEADER_OK_PATHS: set[str] = set()


def append_cycle_audit_row(path: str, row: dict[str, object]) -> None:
    cols = ensure_cycle_audit_csv(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row.get(c, "") for c in cols])


# -- Pool snapshot helper -----------------------------------------------------

def round_pool_snapshot(
    round_t: Round | None,
    *,
    prefix: str,
    cutoff_ts: int | None = None,
) -> dict[str, object]:
    if round_t is None:
        return {
            f"{prefix}_total_pool_bnb": "",
            f"{prefix}_bull_pool_bnb": "",
            f"{prefix}_bear_pool_bnb": "",
            f"{prefix}_total_bets": "",
            f"{prefix}_bull_bets": "",
            f"{prefix}_bear_bets": "",
        }

    bull_wei = 0
    bear_wei = 0
    bull_bets = 0
    bear_bets = 0
    for bet in round_t.bets:
        if cutoff_ts is not None and bet.created_at >= cutoff_ts:
            continue
        if bet.position == "Bull":
            bull_wei += bet.amount_wei
            bull_bets += 1
        elif bet.position == "Bear":
            bear_wei += bet.amount_wei
            bear_bets += 1
        else:
            raise InvariantError(f"unexpected_round_bet_side: {bet.position}")

    bull_bnb = bull_wei / BNB_WEI
    bear_bnb = bear_wei / BNB_WEI
    return {
        f"{prefix}_total_pool_bnb": bull_bnb + bear_bnb,
        f"{prefix}_bull_pool_bnb": bull_bnb,
        f"{prefix}_bear_pool_bnb": bear_bnb,
        f"{prefix}_total_bets": bull_bets + bear_bets,
        f"{prefix}_bull_bets": bull_bets,
        f"{prefix}_bear_bets": bear_bets,
    }


# -- Main cycle audit recorder ------------------------------------------------
# (selected_side_probability helper removed 2026-04-26; it derived from
# the now-removed p_bull column.)

def record_cycle_audit(
    closed,
    *,
    cycle_audit_path: str,
    current_epoch: int,
    locked_epoch: int,
    lock_ts: int,
    cutoff_ts: int,
    locked_price_bnbusd: float,
    action: str,
    decision_stage: str,
    open_round: Round | None,
    bankroll_before_action_bnb: float | None,
    bankroll_after_action_bnb: float | None,
    decision: object | None = None,
    skip_reason: str | None = None,
    decision_latency_ms: float | None = None,
    pool_bull_bnb: float = 0.0,
    pool_bear_bnb: float = 0.0,
    btc_fetch_ms: int | None = None,
    eth_fetch_ms: int | None = None,
    sol_fetch_ms: int | None = None,
    wake_mode: str = "",
    kline_fire_offset_before_lock_ms: int | None = None,
    t_features_start_offset_ms: float | None = None,
    btc_fetch_result: str = "not_fetched",
    eth_fetch_result: str = "not_fetched",
    sol_fetch_result: str = "not_fetched",
) -> None:
    # Use RPC-fetched pool values when available (live/dry mode);
    # fall back to round_t.bets snapshot (backtest / no RPC data).
    if pool_bull_bnb > 0.0 or pool_bear_bnb > 0.0:
        pool_total = pool_bull_bnb + pool_bear_bnb
        observed_pool = {
            "observed_total_pool_bnb": pool_total,
            "observed_bull_pool_bnb": pool_bull_bnb,
            "observed_bear_pool_bnb": pool_bear_bnb,
            "observed_total_bets": "",
            "observed_bull_bets": "",
            "observed_bear_bets": "",
        }
        cutoff_used_pool = {
            "cutoff_used_total_pool_bnb": pool_total,
            "cutoff_used_bull_pool_bnb": pool_bull_bnb,
            "cutoff_used_bear_pool_bnb": pool_bear_bnb,
            "cutoff_used_total_bets": "",
            "cutoff_used_bull_bets": "",
            "cutoff_used_bear_bets": "",
        }
    else:
        observed_pool = round_pool_snapshot(open_round, prefix="observed")
        cutoff_used_pool = round_pool_snapshot(
            open_round,
            prefix="cutoff_used",
            cutoff_ts=cutoff_ts,
        )
    router_mode: str | object = ""
    pipeline_last_settled_epoch: int | str = ""
    if closed.strategy_pipeline is not None:
        router_mode = closed.strategy_pipeline.router_mode
        if closed.strategy_pipeline.last_settled_epoch is not None:
            pipeline_last_settled_epoch = closed.strategy_pipeline.last_settled_epoch

    bet_side: str | object = ""
    bet_size_bnb: float | str = ""
    if decision is not None:
        bet_side = getattr(decision, "bet_side", "") or ""
        bet_size_raw = getattr(decision, "bet_size_bnb", "")
        if isinstance(bet_size_raw, (int, float)):
            bet_size_bnb = float(bet_size_raw)
        if skip_reason is None:
            skip_reason = getattr(decision, "skip_reason", None)

    bankroll_before = (
        ""
        if bankroll_before_action_bnb is None
        else float(bankroll_before_action_bnb)
    )
    bankroll_after = (
        bankroll_before
        if bankroll_after_action_bnb is None
        else float(bankroll_after_action_bnb)
    )
    append_cycle_audit_row(
        cycle_audit_path,
        {
            "cycle_ts": int(time.time()),
            "current_epoch": current_epoch,
            "locked_epoch": locked_epoch,
            "lock_ts": lock_ts,
            "cutoff_ts": cutoff_ts,
            "locked_price_bnbusd": locked_price_bnbusd,
            "bankroll_before_action_bnb": bankroll_before,
            "bankroll_after_action_bnb": bankroll_after,
            "observed_total_pool_bnb": observed_pool["observed_total_pool_bnb"],
            "observed_bull_pool_bnb": observed_pool["observed_bull_pool_bnb"],
            "observed_bear_pool_bnb": observed_pool["observed_bear_pool_bnb"],
            "observed_total_bets": observed_pool["observed_total_bets"],
            "observed_bull_bets": observed_pool["observed_bull_bets"],
            "observed_bear_bets": observed_pool["observed_bear_bets"],
            "cutoff_used_total_pool_bnb": cutoff_used_pool["cutoff_used_total_pool_bnb"],
            "cutoff_used_bull_pool_bnb": cutoff_used_pool["cutoff_used_bull_pool_bnb"],
            "cutoff_used_bear_pool_bnb": cutoff_used_pool["cutoff_used_bear_pool_bnb"],
            "cutoff_used_total_bets": cutoff_used_pool["cutoff_used_total_bets"],
            "cutoff_used_bull_bets": cutoff_used_pool["cutoff_used_bull_bets"],
            "cutoff_used_bear_bets": cutoff_used_pool["cutoff_used_bear_bets"],
            "router_mode": router_mode,
            "pipeline_last_settled_epoch": pipeline_last_settled_epoch,
            "action": action,
            "decision_stage": decision_stage,
            "bet_side": bet_side,
            "bet_size_bnb": bet_size_bnb,
            "decision_latency_ms": (
                "" if decision_latency_ms is None else decision_latency_ms
            ),
            "skip_reason": "" if skip_reason is None else skip_reason,
            "btc_fetch_ms": "" if btc_fetch_ms is None else btc_fetch_ms,
            "eth_fetch_ms": "" if eth_fetch_ms is None else eth_fetch_ms,
            "sol_fetch_ms": "" if sol_fetch_ms is None else sol_fetch_ms,
            "wake_mode": wake_mode,
            "kline_fire_offset_before_lock_ms": (
                "" if kline_fire_offset_before_lock_ms is None
                else kline_fire_offset_before_lock_ms
            ),
            "t_features_start_offset_ms": (
                "" if t_features_start_offset_ms is None
                else round(float(t_features_start_offset_ms), 1)
            ),
            "btc_fetch_result": btc_fetch_result,
            "eth_fetch_result": eth_fetch_result,
            "sol_fetch_result": sol_fetch_result,
        },
    )


# -- Settled epoch recovery from audit CSV ------------------------------------

def load_settled_epochs_from_audit(path: str) -> set[int]:
    p = Path(path)
    if not p.exists():
        return set()
    out: set[int] = set()
    with p.open("r", newline="", encoding="utf-8") as f:
        for lineno, row in enumerate(csv.DictReader(f), start=2):
            settled_ts_raw = str(row.get("settled_ts", "")).strip()
            if settled_ts_raw == "":
                continue
            epoch_raw = str(row.get("epoch", "")).strip()
            if epoch_raw == "":
                raise InvariantError(f"audit_epoch_missing: path={path} line={lineno}")
            try:
                int(settled_ts_raw)
                out.add(int(epoch_raw))
            except ValueError as e:
                raise InvariantError(f"audit_row_invalid: path={path} line={lineno}") from e
    return out
