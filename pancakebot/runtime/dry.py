"""Dry-mode state: simulated bankroll, pending bets, settlement, archiving of prior runs."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from pancakebot import paths as _paths
from pancakebot.constants import RETRY_BACKOFF_SECONDS
from pancakebot.util import InvariantError, TransientRpcError
from pancakebot.log import info, warn
from pancakebot.runtime.audit import (
    append_audit_row as _append_dry_audit_row,
    ensure_audit_csv as _ensure_dry_audit_csv,
    ensure_cycle_audit_csv as _ensure_dry_cycle_audit_csv,
    load_settled_epochs_from_audit as _load_dry_settled_epochs_from_audit,
    record_cycle_audit,
)
from pancakebot.runtime.config import RuntimeConfig
from pancakebot.settlement import settle_from_round_data
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.types import Round
from time import sleep as sleep_seconds


@dataclass(slots=True)
class _ClosedState:
    strategy_pipeline: MomentumOnlyPipeline | None = None
    claim_scan_initialized: bool = False
    simulated_bankroll_bnb: float | None = None
    dry_bets_by_epoch: dict[int, dict[str, object]] = field(default_factory=dict)
    dry_settled_epochs: set[int] = field(default_factory=set)
    # Process-health: incremented at top of each _run_one_iteration;
    # ``last_seen_epoch`` is the most-recent current_epoch the loop has
    # observed. Used by the crash handler to pinpoint where the bot
    # "was" at failure.
    iteration_count: int = 0
    last_seen_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class _DryBankrollState:
    simulated_bankroll_bnb: float
    updated_ts: int
    source: str
    epoch: int | None = None


def _dry_runtime_state_files() -> list[Path]:
    return [
        Path(_paths.DRY_PENDING_BETS_PATH),
        Path(_paths.DRY_SETTLED_EPOCHS_PATH),
        Path(_paths.DRY_TRADES_PATH),
        Path(_paths.DRY_CYCLE_AUDIT_PATH),
        Path(_paths.DRY_BANKROLL_STATE_PATH),
        Path(_paths.DRY_BANKROLL_HISTORY_PATH),
        Path(_paths.DRY_BANKROLL_HISTORY_PATH).parent / "pause_state.json",
    ]


def _unique_archive_dir(root: Path, *, ts: int, reason: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    base = root / f"dry_run_archive_{stamp}_{reason}"
    if not base.exists():
        return base
    suffix = 1
    while True:
        cand = root / f"dry_run_archive_{stamp}_{reason}_{suffix}"
        if not cand.exists():
            return cand
        suffix += 1


_MIN_ROUNDS_TO_ARCHIVE = 1  # don't archive runs shorter than this


def _archive_dry_runtime_state(
    *,
    reason: str,
    move_files: bool,
) -> Path | None:
    existing = [path.resolve() for path in _dry_runtime_state_files() if path.exists()]
    if not existing:
        return None

    # Skip archiving short runs -- just delete the files.
    cycle_csv = Path(_paths.DRY_CYCLE_AUDIT_PATH)
    n_lines = 0
    if cycle_csv.exists():
        n_lines = sum(1 for _ in cycle_csv.open()) - 1  # subtract header
    if n_lines < _MIN_ROUNDS_TO_ARCHIVE:
        for f in existing:
            f.unlink(missing_ok=True)
        if n_lines > 0:
            info("START", f"Deleted previous dry state ({n_lines} rounds, below {_MIN_ROUNDS_TO_ARCHIVE} threshold)")
        return None
    archive_root = Path(_paths.DRY_ARCHIVE_ROOT).resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    ts_now = int(time.time())
    archive_dir = _unique_archive_dir(archive_root, ts=ts_now, reason=reason)
    archive_dir.mkdir(parents=True, exist_ok=False)
    file_meta: list[dict[str, object]] = []
    for src in existing:
        dest = archive_dir / src.name
        stat = src.stat()
        if move_files:
            shutil.move(str(src), str(dest))
        else:
            shutil.copy2(str(src), str(dest))
        file_meta.append(
            {
                "name": src.name,
                "source_path": str(src),
                "archive_path": str(dest),
                "size_bytes": stat.st_size,
                "source_mtime_ts": int(stat.st_mtime),
            }
        )
    (archive_dir / "archive_meta.json").write_text(
        json.dumps(
            {
                "reason": reason,
                "created_ts": ts_now,
                "move_files": move_files,
                "file_count": len(file_meta),
                "files": file_meta,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return archive_dir


def _ensure_parent_dir(path: str) -> None:
    from pancakebot.util import ensure_parent_dir
    ensure_parent_dir(path)


def _append_jsonl(path: str, record: dict[str, object]) -> None:
    _ensure_parent_dir(path)
    with open(path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        f.write("\n")


def _write_json_file_atomic(path: str, record: dict[str, object]) -> None:
    out = Path(path)
    _ensure_parent_dir(str(out))
    content = json.dumps(record, separators=(",", ":"), sort_keys=True)
    fd, tmp_path = tempfile.mkstemp(dir=out.parent, prefix=out.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(out))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_dry_bets(path: str) -> dict[int, dict[str, object]]:
    bets: dict[int, dict[str, object]] = {}
    p = Path(path)
    if not p.exists():
        return bets
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise InvariantError(f"dry_bets_json_invalid: path={path} line={lineno}") from e
        if not isinstance(rec, dict):
            raise InvariantError(f"dry_bets_record_not_object: path={path} line={lineno}")
        try:
            epoch = int(rec["epoch"])
        except Exception as e:
            raise InvariantError(f"dry_bets_epoch_invalid: path={path} line={lineno}") from e
        if epoch in bets:
            raise InvariantError(f"dry_bets_epoch_duplicate_on_load: path={path} epoch={epoch}")
        bets[epoch] = rec
    return bets


def _load_dry_settled_epochs(path: str) -> set[int]:
    p = Path(path)
    if not p.exists():
        return set()
    out: set[int] = set()
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.add(int(line))
        except ValueError as e:
            raise InvariantError(f"dry_settled_epoch_invalid: path={path} line={lineno}") from e
    return out


def _load_dry_bankroll_state(path: str) -> _DryBankrollState | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise InvariantError(f"dry_bankroll_state_json_invalid: path={path}") from e
    if not isinstance(raw, dict):
        raise InvariantError(f"dry_bankroll_state_not_object: path={path}")
    bankroll_raw = raw.get("simulated_bankroll_bnb")
    updated_ts_raw = raw.get("updated_ts")
    source_raw = raw.get("source", "")
    epoch_raw = raw.get("epoch")
    if not isinstance(bankroll_raw, (int, float)):
        raise InvariantError(f"dry_bankroll_state_bankroll_invalid: path={path}")
    if not isinstance(updated_ts_raw, int):
        raise InvariantError(f"dry_bankroll_state_updated_ts_invalid: path={path}")
    if not isinstance(source_raw, str) or source_raw.strip() == "":
        raise InvariantError(f"dry_bankroll_state_source_invalid: path={path}")
    epoch: int | None = None
    if epoch_raw is not None:
        if not isinstance(epoch_raw, (int, str)):
            raise InvariantError(f"dry_bankroll_state_epoch_invalid: path={path}")
        try:
            epoch = int(epoch_raw)
        except (ValueError, TypeError) as e:
            raise InvariantError(f"dry_bankroll_state_epoch_invalid: path={path}") from e
    if not isinstance(bankroll_raw, (int, float)):
        raise InvariantError(f"dry_bankroll_state_bankroll_invalid: path={path}")
    bankroll_bnb = float(bankroll_raw)
    if bankroll_bnb < 0.0:
        raise InvariantError(f"dry_bankroll_state_bankroll_negative: path={path}")
    if updated_ts_raw < 0:
        raise InvariantError(f"dry_bankroll_state_updated_ts_negative: path={path}")
    return _DryBankrollState(
        simulated_bankroll_bnb=bankroll_bnb,
        updated_ts=updated_ts_raw,
        source=source_raw.strip(),
        epoch=epoch,
    )


def _save_dry_bankroll_state(
    path: str,
    *,
    bankroll_bnb: float,
    source: str,
    epoch: int | None,
    updated_ts: int,
) -> _DryBankrollState:
    if bankroll_bnb < 0.0:
        raise InvariantError("dry_bankroll_state_bankroll_negative")
    if updated_ts < 0:
        raise InvariantError("dry_bankroll_state_updated_ts_negative")
    source_name = source.strip()
    if source_name == "":
        raise InvariantError("dry_bankroll_state_source_empty")
    state = _DryBankrollState(
        simulated_bankroll_bnb=bankroll_bnb,
        updated_ts=updated_ts,
        source=source_name,
        epoch=epoch,
    )
    _write_json_file_atomic(
        path,
        {
            "simulated_bankroll_bnb": state.simulated_bankroll_bnb,
            "updated_ts": state.updated_ts,
            "source": state.source,
            "epoch": state.epoch,
        },
    )
    return state


def _recover_dry_bankroll_state_from_logs(
    *,
    dry_bets_path: str,
    dry_audit_trades_path: str,
) -> _DryBankrollState | None:
    latest_state: _DryBankrollState | None = None

    for rec in _load_dry_bets(dry_bets_path).values():
        placed_ts_raw = rec.get("placed_ts")
        bankroll_raw = rec.get("bankroll_after_bet_bnb")
        epoch_raw = rec.get("epoch")
        if not isinstance(placed_ts_raw, int):
            continue
        if not isinstance(bankroll_raw, (int, float)):
            continue
        if epoch_raw is not None and not isinstance(epoch_raw, (int, str)):
            continue
        epoch = None if epoch_raw is None else int(epoch_raw)
        state = _DryBankrollState(
            simulated_bankroll_bnb=float(bankroll_raw),
            updated_ts=int(placed_ts_raw),
            source="recover_from_dry_bet",
            epoch=epoch,
        )
        if latest_state is None or state.updated_ts > latest_state.updated_ts:
            latest_state = state

    audit_path = Path(dry_audit_trades_path)
    if audit_path.exists():
        with audit_path.open("r", newline="", encoding="utf-8") as f:
            for lineno, row in enumerate(csv.DictReader(f), start=2):
                settled_ts_raw = str(row.get("settled_ts", "")).strip()
                bankroll_raw = str(row.get("bankroll_after_settle_bnb", "")).strip()
                if settled_ts_raw == "" or bankroll_raw == "":
                    continue
                try:
                    settled_ts = int(settled_ts_raw)
                    bankroll_bnb = float(bankroll_raw)
                except ValueError as e:
                    raise InvariantError(
                        f"dry_audit_row_invalid: path={dry_audit_trades_path} line={lineno}"
                    ) from e
                epoch_raw = str(row.get("epoch", "")).strip()
                epoch = int(epoch_raw) if epoch_raw != "" else None
                state = _DryBankrollState(
                    simulated_bankroll_bnb=bankroll_bnb,
                    updated_ts=settled_ts,
                    source="recover_from_dry_settle",
                    epoch=epoch,
                )
                if latest_state is None or state.updated_ts > latest_state.updated_ts:
                    latest_state = state

    return latest_state


def _resolve_initial_dry_bankroll_state(cfg: RuntimeConfig) -> _DryBankrollState:
    persisted = _load_dry_bankroll_state(_paths.DRY_BANKROLL_STATE_PATH)
    recovered = _recover_dry_bankroll_state_from_logs(
        dry_bets_path=_paths.DRY_PENDING_BETS_PATH,
        dry_audit_trades_path=_paths.DRY_TRADES_PATH,
    )
    configured_init = cfg.dry_initial_bankroll_bnb
    can_override_persisted_seed = (
        configured_init is not None
        and persisted is not None
        and recovered is None
        and persisted.epoch is None
        and persisted.source in {"wallet_init", "configured_init"}
    )
    if (
        persisted is not None
        and not can_override_persisted_seed
        and (recovered is None or persisted.updated_ts >= recovered.updated_ts)
    ):
        return persisted
    if recovered is not None:
        return _save_dry_bankroll_state(
            _paths.DRY_BANKROLL_STATE_PATH,
            bankroll_bnb=recovered.simulated_bankroll_bnb,
            source="recovered",
            epoch=recovered.epoch,
            updated_ts=recovered.updated_ts,
        )
    if configured_init is not None:
        return _save_dry_bankroll_state(
            _paths.DRY_BANKROLL_STATE_PATH,
            bankroll_bnb=configured_init,
            source="configured_init",
            epoch=None,
            updated_ts=int(time.time()),
        )
    wallet_bnb = _fetch_wallet_balance_bnb_with_retries(
        cfg=cfg,
        reason="dry_wallet_bootstrap",
    )
    return _save_dry_bankroll_state(
        _paths.DRY_BANKROLL_STATE_PATH,
        bankroll_bnb=wallet_bnb,
        source="wallet_init",
        epoch=None,
        updated_ts=int(time.time()),
    )


def _fetch_wallet_balance_bnb_with_retries(
    *,
    cfg: RuntimeConfig,
    reason: str,
) -> float:
    for delay_seconds in RETRY_BACKOFF_SECONDS:
        try:
            return float(cfg.contract.wallet_balance_bnb(cfg.wallet_address))
        except TransientRpcError as e:
            warn(
                "RETRY",
                f"TransientRpcError during {reason}: "
                f"retrying after delay={delay_seconds}s err={e}",
            )
            sleep_seconds(delay_seconds)
    # Terminal attempt after the backoff schedule is exhausted. Raise a
    # NAMED exhaustion invariant (consistent with _epoch_handshake's
    # epoch_handshake_exhausted and _sleep_and_claim's
    # close_ts_retry_exhausted) instead of leaking the raw TransientRpcError.
    try:
        return float(cfg.contract.wallet_balance_bnb(cfg.wallet_address))
    except TransientRpcError as e:
        raise InvariantError(
            f"wallet_balance_retry_exhausted: {reason} err={e}"
        ) from e


def _append_dry_settled_epoch(path: str, epoch: int) -> None:
    _ensure_parent_dir(path)
    with open(path, "a") as f:
        f.write(str(epoch))
        f.write("\n")


def _dry_record_bet(
    closed: _ClosedState,
    *,
    epoch: int,
    side: str,
    amount_bnb: float,
    p_final: float,
    bankroll_before_bet_bnb: float,
    bankroll_after_bet_bnb: float,
) -> None:
    # ``expected_profit_bnb`` parameter removed 2026-04-26: the source
    # field on StrategyPipelineDecision was deleted, and the column had
    # only ever been written empty in practice.
    if epoch in closed.dry_bets_by_epoch:
        raise InvariantError(f"dry_bet_duplicate_epoch: epoch={epoch}")
    placed_ts = int(time.time())
    rec = {
        "epoch": epoch,
        "placed_ts": placed_ts,
        "bet_side": side,
        "bet_bnb": amount_bnb,
        "p_final": p_final,
        "pred_win_probability": p_final,
        "bankroll_before_bet_bnb": bankroll_before_bet_bnb,
        "bankroll_after_bet_bnb": bankroll_after_bet_bnb,
    }
    closed.dry_bets_by_epoch[epoch] = rec
    _append_jsonl(_paths.DRY_PENDING_BETS_PATH, rec)
    _save_dry_bankroll_state(
        _paths.DRY_BANKROLL_STATE_PATH,
        bankroll_bnb=bankroll_after_bet_bnb,
        source="bet",
        epoch=epoch,
        updated_ts=placed_ts,
    )


def _record_cycle_audit(
    cfg: RuntimeConfig,
    closed: _ClosedState,
    *,
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
    """Per-round audit row. Mode-aware: dry mode writes to
    ``var/dry/cycle_audit.csv``, live mode to ``var/live/cycle_audit.csv``.
    Single source of truth for the schema (in ``audit.py``); future
    column additions touch only ``ensure_cycle_audit_csv`` +
    ``record_cycle_audit``."""
    cycle_audit_path = (
        _paths.DRY_CYCLE_AUDIT_PATH if cfg.dry
        else _paths.LIVE_CYCLE_AUDIT_PATH
    )
    record_cycle_audit(
        closed,
        cycle_audit_path=cycle_audit_path,
        current_epoch=current_epoch,
        locked_epoch=locked_epoch,
        lock_ts=lock_ts,
        cutoff_ts=cutoff_ts,
        locked_price_bnbusd=locked_price_bnbusd,
        action=action,
        decision_stage=decision_stage,
        open_round=open_round,
        bankroll_before_action_bnb=bankroll_before_action_bnb,
        bankroll_after_action_bnb=bankroll_after_action_bnb,
        decision=decision,
        skip_reason=skip_reason,
        decision_latency_ms=decision_latency_ms,
        pool_bull_bnb=pool_bull_bnb,
        pool_bear_bnb=pool_bear_bnb,
        btc_fetch_ms=btc_fetch_ms,
        eth_fetch_ms=eth_fetch_ms,
        sol_fetch_ms=sol_fetch_ms,
        wake_mode=wake_mode,
        kline_fire_offset_before_lock_ms=kline_fire_offset_before_lock_ms,
        t_features_start_offset_ms=t_features_start_offset_ms,
        btc_fetch_result=btc_fetch_result,
        eth_fetch_result=eth_fetch_result,
        sol_fetch_result=sol_fetch_result,
    )


def _dry_settle_available_bets(cfg: RuntimeConfig, closed: _ClosedState) -> None:
    if not cfg.dry:
        return
    if closed.simulated_bankroll_bnb is None:
        raise InvariantError("dry_bankroll_uninitialized")
    if closed.dry_bets_by_epoch is None or closed.dry_settled_epochs is None:
        raise InvariantError("dry_state_uninitialized")

    for e, bet in sorted(closed.dry_bets_by_epoch.items()):
        if e in closed.dry_settled_epochs:
            continue

        # Fetch round data from contract; skip if not yet finalized.
        # Transient RPC failure -- will retry next iteration. Narrowed to
        # TransientRpcError (guard audit 4.6): a broad ``except Exception``
        # here silently masked any genuine round_data bug for an epoch; a
        # non-transient error should surface (dry-only, no live money).
        try:
            rd = cfg.contract.round_data(e)
        except TransientRpcError:
            continue

        # Match PCS V2 _refundable(): only refund-eligible past
        # close_ts + bufferSeconds. Settling an oracle-pending round inside
        # [close_ts, close_ts+buffer] would prematurely refund. (Reviewer
        # Fix #3 — same gate as bet_ledger.reconcile.)
        if not rd.oracle_called and (rd.close_ts + cfg.buffer_seconds) >= time.time():
            continue  # round not yet refund-eligible on-chain (strict `>` in
            #            PCS _refundable() -> NOT refundable AT the boundary)

        bet_bnb_raw = bet.get("bet_bnb", 0.0)
        if isinstance(bet_bnb_raw, (int, float)):
            bet_bnb = float(bet_bnb_raw)
        elif isinstance(bet_bnb_raw, str):
            try:
                bet_bnb = float(bet_bnb_raw)
            except ValueError as exc:
                raise InvariantError("dry_bet_bnb_parse_failed") from exc
        else:
            raise InvariantError("dry_bet_bnb_type_invalid")
        if bet_bnb <= 0:
            closed.dry_settled_epochs.add(e)
            _append_dry_settled_epoch(_paths.DRY_SETTLED_EPOCHS_PATH, e)
            continue

        settle = settle_from_round_data(
            bet_bnb=bet_bnb,
            bet_side=str(bet.get("bet_side", "")),
            lock_price_usd=rd.lock_price_usd,
            close_price_usd=rd.close_price_usd,
            bull_amount_wei=rd.bull_amount_wei,
            bear_amount_wei=rd.bear_amount_wei,
            oracle_called=rd.oracle_called,
            treasury_fee_fraction=cfg.treasury_fee_fraction,
        )

        outcome = settle.outcome
        credit_bnb = settle.credit_bnb

        bankroll_before_settle = closed.simulated_bankroll_bnb
        closed.simulated_bankroll_bnb += credit_bnb
        bankroll_after_settle = closed.simulated_bankroll_bnb

        # Forward post-settlement bankroll to tracker (no-op if unwired).
        closed.strategy_pipeline.record_settlement(
            bankroll=bankroll_after_settle, start_at=int(rd.start_ts),
        )

        if outcome == "win":
            # payout multiplier = gross credit / stake. A loss yields
            # credit=0 (no entry here); refund yields credit==bet (1.00x,
            # also not emitted as WIN). For win, the multiplier captures
            # the round's payoff in operator-readable terms.
            payout_x = credit_bnb / bet_bnb if bet_bnb > 0 else 0.0
            info(
                "WIN",
                f"Won {credit_bnb:.4f} BNB on epoch {e}, "
                f"payout {payout_x:.2f}x "
                f"(bankroll: {bankroll_after_settle:.4f} BNB)",
            )
        elif outcome == "refund":
            info(
                "REFUND",
                f"Refunded {credit_bnb:.4f} BNB on epoch {e} "
                f"(bankroll: {bankroll_after_settle:.4f} BNB)",
            )
        else:
            info(
                "LOSS",
                f"Lost {bet_bnb:.4f} BNB on epoch {e} "
                f"(bankroll: {bankroll_after_settle:.4f} BNB)",
            )

        settled_ts = int(time.time())

        # Bet-lifecycle ledger (dry): append the terminal SETTLED_* record for
        # PnL-truth parity with live. No Discord (dry alerts silent, D1=(a)).
        # Dry owns its bankroll/trades bookkeeping (above); this only adds the
        # ledger record, via the SAME classify helper live reconcile uses so
        # both modes agree on status+delta semantics (Fix #6 unification of
        # the classification logic without refactoring this pipeline).
        from pancakebot.runtime import bet_ledger as _bet_ledger
        _dry_status, _dry_delta = _bet_ledger.classify_settlement(
            outcome=outcome, bet_bnb=bet_bnb, credit_bnb=credit_bnb,
        )
        _bet_ledger.record_settled(
            ledger_path=_paths.DRY_BETS_LEDGER_PATH,
            epoch=int(e), side=str(bet.get("bet_side", "")), status=_dry_status,
            delta_bnb=_dry_delta, outcome=outcome,
            new_bankroll_bnb=bankroll_after_settle,
        )

        placed_raw = bet.get("placed_ts")
        if isinstance(placed_raw, int):
            placed_ts_val: int | str = placed_raw
        elif isinstance(placed_raw, str) and placed_raw.isdigit():
            placed_ts_val = int(placed_raw)
        else:
            placed_ts_val = ""

        _append_dry_audit_row(
            _paths.DRY_TRADES_PATH,
            {
                "epoch": e,
                "placed_ts": placed_ts_val,
                "bet_side": str(bet.get("bet_side", "")),
                "bet_bnb": bet_bnb,
                "pred_win_probability": bet.get("pred_win_probability", ""),
                "p_final": bet.get("p_final", ""),
                "cutoff_bull_bnb": bet.get("cutoff_bull_bnb", ""),
                "cutoff_bear_bnb": bet.get("cutoff_bear_bnb", ""),
                "final_bull_bnb": bet.get("final_bull_bnb", ""),
                "final_bear_bnb": bet.get("final_bear_bnb", ""),
                "settled_ts": settled_ts,
                "outcome": outcome,
                "pnl_bnb": credit_bnb,
                "bankroll_before_bet_bnb": bet.get("bankroll_before_bet_bnb", ""),
                "bankroll_after_bet_bnb": bet.get("bankroll_after_bet_bnb", ""),
                "bankroll_before_settle_bnb": bankroll_before_settle,
                "bankroll_after_settle_bnb": bankroll_after_settle,
            },
        )

        closed.dry_settled_epochs.add(e)
        _append_dry_settled_epoch(_paths.DRY_SETTLED_EPOCHS_PATH, e)
        _save_dry_bankroll_state(
            _paths.DRY_BANKROLL_STATE_PATH,
            bankroll_bnb=bankroll_after_settle,
            source="settle",
            epoch=e,
            updated_ts=settled_ts,
        )


def _init_closed_state(cfg: RuntimeConfig) -> _ClosedState:
    """Initialize live/dry runtime state. No disk sync -- pure RPC + OKX."""
    info("START", "Core setup: strategy=momentum_gate mode=rpc_only")

    strategy_pipeline = _build_momentum_pipeline(cfg=cfg)
    # No warmup rounds needed: OKX gate fetches live at decision time.
    strategy_pipeline.bootstrap_from_closed_rounds(rounds=[])

    closed = _ClosedState(strategy_pipeline=strategy_pipeline)

    if cfg.dry:
        fresh = cfg.dry_fresh_start
        if fresh:
            if cfg.dry_no_archive:
                # --fresh --no-archive: delete existing state without archiving
                for f in _dry_runtime_state_files():
                    if f.exists():
                        f.unlink(missing_ok=True)
                info("START", "Deleted previous dry state (--no-archive)")
            else:
                archived = _archive_dry_runtime_state(
                    reason="startup_fresh_reset",
                    move_files=True,
                )
                if archived is not None:
                    archive_log = Path(_paths.DRY_ARCHIVE_ROOT) / archived.name
                    info("START", f"Archived previous dry runtime state to {archive_log}")
        bankroll_state = _resolve_initial_dry_bankroll_state(cfg)
        closed.simulated_bankroll_bnb = bankroll_state.simulated_bankroll_bnb
        # Wire PersistedBankrollTracker now that we know the initial bankroll.
        # Writes to var/dry/bankroll_history.jsonl + pause_state.json — fresh
        # paths, disjoint from the existing bankroll.json used by the settled
        # state machine. On a --fresh start the history file is already purged
        # by the archive/unlink logic above (it lives under var/dry/).
        from pancakebot.bankroll_tracker import PersistedBankrollTracker
        tracker = PersistedBankrollTracker(
            path=Path(_paths.DRY_BANKROLL_HISTORY_PATH),
            initial_bankroll=bankroll_state.simulated_bankroll_bnb,
            drawdown_peak_window_days=cfg.strategy.risk.drawdown_peak_window_days,
        )
        strategy_pipeline.set_bankroll_tracker(tracker)
        closed.dry_bets_by_epoch = _load_dry_bets(_paths.DRY_PENDING_BETS_PATH)
        _ensure_dry_audit_csv(_paths.DRY_TRADES_PATH)
        _ensure_dry_cycle_audit_csv(
            _paths.DRY_CYCLE_AUDIT_PATH,
            reset=fresh,
        )
        settled_epochs = _load_dry_settled_epochs(_paths.DRY_SETTLED_EPOCHS_PATH)
        settled_epochs.update(
            _load_dry_settled_epochs_from_audit(_paths.DRY_TRADES_PATH)
        )
        closed.dry_settled_epochs = settled_epochs
        info(
            "START",
            f"Loaded dry bankroll {bankroll_state.simulated_bankroll_bnb:.6f} BNB source={bankroll_state.source}",
        )

    return closed


def _build_momentum_pipeline(*, cfg: RuntimeConfig) -> MomentumOnlyPipeline:
    """Build momentum-only strategy pipeline."""
    from pancakebot.strategy.momentum_gate import MomentumGateConfig
    gate_config: MomentumGateConfig = cfg.momentum_gate_config  # type: ignore[assignment]
    return MomentumOnlyPipeline(
        config=gate_config,
        strategy_config=cfg.strategy,
        gate=cfg.momentum_gate,
        kline_cutoff_seconds=cfg.kline_cutoff_seconds,
        pool_cutoff_seconds=cfg.pool_cutoff_seconds,
        min_bet_amount_bnb=cfg.min_bet_amount_bnb,
        treasury_fee_fraction=cfg.treasury_fee_fraction,
    )
