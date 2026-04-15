"""Dry-run state management: bankroll, bets, settlement, audit CSV, archiving."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from pancakebot.config import RuntimeStatePathsConfig
from pancakebot.constants import BNB_WEI
from pancakebot.errors import InvariantError, TransientRpcError
from pancakebot.log import info
from pancakebot.money import bankroll_suffix, usd_suffix
from pancakebot.runtime.config import RuntimeConfig
from pancakebot.settlement import settle_from_round_data
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.time import now_ts
from pancakebot.types import Round
from time import sleep as sleep_seconds


_BACKOFF_SECONDS = [2, 4, 8, 16, 32, 58]  # locked


@dataclass(slots=True)
class _ClosedState:
    strategy_pipeline: MomentumOnlyPipeline | None = None
    claim_scan_initialized: bool = False
    pool_backfill_done: bool = False
    simulated_bankroll_bnb: float | None = None
    dry_bets_by_epoch: dict[int, dict[str, object]] | None = None
    dry_settled_epochs: set[int] | None = None


@dataclass(frozen=True, slots=True)
class _DryBankrollState:
    simulated_bankroll_bnb: float
    updated_ts: int
    source: str
    epoch: int | None = None


def _dry_runtime_state_files(paths: RuntimeStatePathsConfig) -> list[Path]:
    return [
        Path(paths.claim_scan_cursor_path),
        Path(paths.dry_bets_path),
        Path(paths.dry_settled_epochs_path),
        Path(paths.dry_audit_trades_path),
        Path(paths.dry_cycle_audit_path),
        Path(paths.dry_bankroll_state_path),
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
    paths: RuntimeStatePathsConfig,
    *,
    reason: str,
    move_files: bool,
) -> Path | None:
    existing = [path.resolve() for path in _dry_runtime_state_files(paths) if path.exists()]
    if not existing:
        return None

    # Skip archiving short runs — just delete the files.
    cycle_csv = Path(paths.dry_cycle_audit_path)
    n_lines = 0
    if cycle_csv.exists():
        n_lines = sum(1 for _ in cycle_csv.open()) - 1  # subtract header
    if n_lines < _MIN_ROUNDS_TO_ARCHIVE:
        for f in existing:
            f.unlink(missing_ok=True)
        if n_lines > 0:
            info("RUN", "DRY", "CLEAN",
                 msg=f"Deleted previous dry state ({n_lines} rounds, below {_MIN_ROUNDS_TO_ARCHIVE} threshold)")
        return None
    archive_root = Path(paths.dry_archive_root).resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    ts_now = int(now_ts())
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
    from pancakebot.path import ensure_parent_dir
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


def _load_dry_settled_epochs_from_audit(path: str) -> set[int]:
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
                raise InvariantError(f"dry_audit_epoch_missing: path={path} line={lineno}")
            try:
                int(settled_ts_raw)
                out.add(int(epoch_raw))
            except ValueError as e:
                raise InvariantError(f"dry_audit_row_invalid: path={path} line={lineno}") from e
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
        try:
            epoch = int(epoch_raw)
        except Exception as e:
            raise InvariantError(f"dry_bankroll_state_epoch_invalid: path={path}") from e
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
    persisted = _load_dry_bankroll_state(cfg.runtime_state_paths.dry_bankroll_state_path)
    recovered = _recover_dry_bankroll_state_from_logs(
        dry_bets_path=cfg.runtime_state_paths.dry_bets_path,
        dry_audit_trades_path=cfg.runtime_state_paths.dry_audit_trades_path,
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
            cfg.runtime_state_paths.dry_bankroll_state_path,
            bankroll_bnb=recovered.simulated_bankroll_bnb,
            source="recovered",
            epoch=recovered.epoch,
            updated_ts=recovered.updated_ts,
        )
    if configured_init is not None:
        return _save_dry_bankroll_state(
            cfg.runtime_state_paths.dry_bankroll_state_path,
            bankroll_bnb=configured_init,
            source="configured_init",
            epoch=None,
            updated_ts=int(now_ts()),
        )
    wallet_bnb = _fetch_wallet_balance_bnb_with_retries(
        cfg=cfg,
        reason="dry_wallet_bootstrap",
    )
    return _save_dry_bankroll_state(
        cfg.runtime_state_paths.dry_bankroll_state_path,
        bankroll_bnb=wallet_bnb,
        source="wallet_init",
        epoch=None,
        updated_ts=int(now_ts()),
    )


def _fetch_wallet_balance_bnb_with_retries(
    *,
    cfg: RuntimeConfig,
    reason: str,
) -> float:
    for delay_seconds in _BACKOFF_SECONDS:
        try:
            return float(cfg.contract.wallet_balance_bnb(cfg.wallet_address))
        except TransientRpcError as e:
            info(
                "CORE",
                "RUN",
                "RETRY",
                msg=(
                    f"Caught TransientRpcError during {reason}: "
                    f"retrying after delay err={e}"
                ),
            )
            info(
                "CORE",
                "LOOP",
                "SLEEP",
                msg=(
                    f"duration={delay_seconds}s "
                    "reason=delay_after_transient_network_error"
                ),
            )
            sleep_seconds(delay_seconds)
    return float(cfg.contract.wallet_balance_bnb(cfg.wallet_address))


def _append_dry_settled_epoch(path: str, epoch: int) -> None:
    _ensure_parent_dir(path)
    with open(path, "a") as f:
        f.write(str(epoch))
        f.write("\n")


def _dry_record_bet(
    cfg: RuntimeConfig,
    closed: _ClosedState,
    *,
    epoch: int,
    side: str,
    amount_bnb: float,
    p_final: float,
    expected_profit_bnb: float,
    bankroll_before_bet_bnb: float,
    bankroll_after_bet_bnb: float,
) -> None:
    if epoch in closed.dry_bets_by_epoch:
        raise InvariantError(f"dry_bet_duplicate_epoch: epoch={epoch}")
    placed_ts = int(now_ts())
    rec = {
        "epoch": epoch,
        "placed_ts": placed_ts,
        "bet_side": side,
        "bet_bnb": amount_bnb,
        "p_final": p_final,
        "pred_win_probability": p_final,
        "expected_profit_bnb": expected_profit_bnb,
        "bankroll_before_bet_bnb": bankroll_before_bet_bnb,
        "bankroll_after_bet_bnb": bankroll_after_bet_bnb,
    }
    closed.dry_bets_by_epoch[epoch] = rec
    _append_jsonl(cfg.runtime_state_paths.dry_bets_path, rec)
    _save_dry_bankroll_state(
        cfg.runtime_state_paths.dry_bankroll_state_path,
        bankroll_bnb=bankroll_after_bet_bnb,
        source="bet",
        epoch=epoch,
        updated_ts=placed_ts,
    )


def _ensure_dry_audit_csv(path: str) -> list[str]:
    header_cols = [
        "epoch",
        "placed_ts",
        "bet_side",
        "bet_bnb",
        "pred_win_probability",
        "p_final",
        "expected_profit_bnb",
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


def _append_dry_audit_row(path: str, row: dict[str, object]) -> None:
    cols = _ensure_dry_audit_csv(path)
    # Append row in stable column order.
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row.get(c, "") for c in cols])


def _ensure_dry_cycle_audit_csv(path: str, *, reset: bool = False) -> list[str]:
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
        "p_bull",
        "selected_side_probability",
        "expected_profit_bnb",
        "selector_score_bnb",
        "decision_latency_ms",
        "skip_reason",
    ]
    p = Path(path)
    if reset or not p.exists():
        _ensure_parent_dir(path)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header_cols)
    return header_cols


def _append_dry_cycle_audit_row(path: str, row: dict[str, object]) -> None:
    cols = _ensure_dry_cycle_audit_csv(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row.get(c, "") for c in cols])


def _round_pool_snapshot(
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
        if cutoff_ts is not None and bet.created_at > cutoff_ts:
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


def _selected_side_probability(*, p_bull: float | None, bet_side: str | None) -> float | str:
    if p_bull is None or bet_side is None:
        return ""
    if bet_side == "Bull":
        return p_bull
    if bet_side == "Bear":
        return 1.0 - p_bull
    return ""


def _record_dry_cycle_audit(
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
) -> None:
    if not cfg.dry:
        return

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
        observed_pool = _round_pool_snapshot(open_round, prefix="observed")
        cutoff_used_pool = _round_pool_snapshot(
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
    p_bull: float | str = ""
    expected_profit_bnb: float | str = ""
    selector_score_bnb: float | str = ""
    if decision is not None:
        bet_side = getattr(decision, "bet_side", "") or ""
        bet_size_raw = getattr(decision, "bet_size_bnb", "")
        if isinstance(bet_size_raw, (int, float)):
            bet_size_bnb = float(bet_size_raw)
        p_bull_raw = getattr(decision, "p_bull", None)
        if isinstance(p_bull_raw, (int, float)):
            p_bull = float(p_bull_raw)
        expected_profit_raw = getattr(decision, "expected_profit_bnb", None)
        if isinstance(expected_profit_raw, (int, float)):
            expected_profit_bnb = float(expected_profit_raw)
        selector_score_raw = getattr(decision, "selector_score_bnb", None)
        if isinstance(selector_score_raw, (int, float)):
            selector_score_bnb = float(selector_score_raw)
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
    _append_dry_cycle_audit_row(
        cfg.runtime_state_paths.dry_cycle_audit_path,
        {
            "cycle_ts": int(now_ts()),
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
            "p_bull": p_bull,
            "selected_side_probability": _selected_side_probability(
                p_bull=None if p_bull == "" else p_bull,
                bet_side=None if bet_side == "" else bet_side,
            ),
            "expected_profit_bnb": expected_profit_bnb,
            "selector_score_bnb": selector_score_bnb,
            "decision_latency_ms": (
                "" if decision_latency_ms is None else decision_latency_ms
            ),
            "skip_reason": "" if skip_reason is None else skip_reason,
        },
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
        try:
            rd = cfg.contract.round_data(e)
        except Exception:
            continue  # transient RPC failure — will retry next iteration

        if not rd.oracle_called and rd.close_ts > now_ts():
            continue  # round not yet closed on-chain

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
            _append_dry_settled_epoch(cfg.runtime_state_paths.dry_settled_epochs_path, e)
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

        bnbusd_price = rd.close_price_usd if rd.close_price_usd > 0 else rd.lock_price_usd

        # Brief INFO log (no key=value fields)
        if outcome == "win":
            info(
                "RUN",
                "ACT",
                "DRY_SETTLE",
                msg=(
                    f"Won {credit_bnb:.4f} BNB"
                    + usd_suffix(amount_bnb=credit_bnb, bnbusd_price=bnbusd_price)
                    + f" on epoch {e}"
                    + bankroll_suffix(bankroll_bnb=bankroll_after_settle, bnbusd_price=bnbusd_price)
                ),
            )
        elif outcome == "refund":
            info(
                "RUN",
                "ACT",
                "DRY_SETTLE",
                msg=(
                    f"Refunded {credit_bnb:.4f} BNB"
                    + usd_suffix(amount_bnb=credit_bnb, bnbusd_price=bnbusd_price)
                    + f" on epoch {e}"
                    + bankroll_suffix(bankroll_bnb=bankroll_after_settle, bnbusd_price=bnbusd_price)
                ),
            )
        else:
            info(
                "RUN",
                "ACT",
                "DRY_SETTLE",
                msg=(
                    f"Lost {bet_bnb:.4f} BNB"
                    + usd_suffix(amount_bnb=bet_bnb, bnbusd_price=bnbusd_price)
                    + f" on epoch {e}"
                    + bankroll_suffix(bankroll_bnb=bankroll_after_settle, bnbusd_price=bnbusd_price)
                ),
            )

        settled_ts = int(now_ts())

        placed_raw = bet.get("placed_ts")
        if isinstance(placed_raw, int):
            placed_ts_val: int | str = placed_raw
        elif isinstance(placed_raw, str) and placed_raw.isdigit():
            placed_ts_val = int(placed_raw)
        else:
            placed_ts_val = ""

        _append_dry_audit_row(
            cfg.runtime_state_paths.dry_audit_trades_path,
            {
                "epoch": e,
                "placed_ts": placed_ts_val,
                "bet_side": str(bet.get("bet_side", "")),
                "bet_bnb": bet_bnb,
                "pred_win_probability": bet.get("pred_win_probability", ""),
                "p_final": bet.get("p_final", ""),
                "expected_profit_bnb": bet.get("expected_profit_bnb", ""),
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
        _append_dry_settled_epoch(cfg.runtime_state_paths.dry_settled_epochs_path, e)
        _save_dry_bankroll_state(
            cfg.runtime_state_paths.dry_bankroll_state_path,
            bankroll_bnb=bankroll_after_settle,
            source="settle",
            epoch=e,
            updated_ts=settled_ts,
        )


def _init_closed_state(cfg: RuntimeConfig) -> _ClosedState:
    """Initialise live/dry runtime state. No disk sync -- pure RPC + OKX."""
    info("CORE", "RUN", "SETUP", msg="Core setup: strategy=momentum_gate mode=rpc_only")

    strategy_pipeline = _build_momentum_pipeline(cfg=cfg)
    # No warmup rounds needed: OKX gate fetches live at decision time.
    strategy_pipeline.bootstrap_from_closed_rounds(rounds=[])

    closed = _ClosedState(strategy_pipeline=strategy_pipeline)

    if cfg.dry:
        fresh = cfg.runtime_state_paths.dry_fresh_start
        if fresh:
            archived = _archive_dry_runtime_state(
                cfg.runtime_state_paths,
                reason="startup_fresh_reset",
                move_files=True,
            )
            if archived is not None:
                archive_log = Path(cfg.runtime_state_paths.dry_archive_root) / archived.name
                info("RUN", "DRY", "ARCHIVE",
                     msg=f"Archived previous dry runtime state to {archive_log}")
        bankroll_state = _resolve_initial_dry_bankroll_state(cfg)
        closed.simulated_bankroll_bnb = bankroll_state.simulated_bankroll_bnb
        closed.dry_bets_by_epoch = _load_dry_bets(cfg.runtime_state_paths.dry_bets_path)
        _ensure_dry_audit_csv(cfg.runtime_state_paths.dry_audit_trades_path)
        _ensure_dry_cycle_audit_csv(
            cfg.runtime_state_paths.dry_cycle_audit_path,
            reset=fresh,
        )
        settled_epochs = _load_dry_settled_epochs(cfg.runtime_state_paths.dry_settled_epochs_path)
        settled_epochs.update(
            _load_dry_settled_epochs_from_audit(cfg.runtime_state_paths.dry_audit_trades_path)
        )
        closed.dry_settled_epochs = settled_epochs
        info(
            "RUN",
            "DRY",
            "STATE",
            msg=f"Loaded dry bankroll {bankroll_state.simulated_bankroll_bnb:.6f} BNB source={bankroll_state.source}",
        )

    return closed


def _build_momentum_pipeline(*, cfg: RuntimeConfig) -> MomentumOnlyPipeline:
    """Build momentum-only strategy pipeline."""
    from pancakebot.strategy.momentum_gate import MomentumGateConfig
    gate_config: MomentumGateConfig = cfg.momentum_gate_config  # type: ignore[assignment]
    return MomentumOnlyPipeline(
        config=gate_config,
        gate=cfg.momentum_gate,
        cutoff_seconds=cfg.cutoff_seconds,
        min_bet_amount_bnb=cfg.min_bet_amount_bnb,
        treasury_fee_fraction=cfg.treasury_fee_fraction,
    )
