"""Live runtime loop.

Hard rules (project-level):
  - Do not catch-and-continue exceptions (developer errors must crash).
  - Pure RPC + OKX: no Graph API, no closed-rounds cache in live/dry mode.

This module orchestrates on-chain RPC and OKX momentum-gate strategy execution.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from pancakebot.config.app_config import RuntimeStatePathsConfig
from pancakebot.core.constants import (
    BNB_WEI,
    GAS_LIMIT_BET,
    GAS_LIMIT_CLAIM,
    GAS_COST_BET_BNB,
)
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.types import Round
from pancakebot.infra.onchain.web3_prediction_contract import RoundData, Web3PredictionContract
from pancakebot.domain.strategy.momentum_gate import MomentumGate
from pancakebot.domain.strategy.momentum_pipeline import MomentumOnlyPipeline
from pancakebot.runtime.claim_manager import claim_scan_cursor
from pancakebot.runtime.settlement import settle_from_round_data
from pancakebot.runtime.sleep import sleep_seconds
from pancakebot.core.constants import BUFFER_SECONDS, INTERVAL_SECONDS, POOL_CUTOFF_SECONDS
from pancakebot.core.errors import InvariantError, TransientRpcError
from pancakebot.core.logging import info, warn
from pancakebot.core.time import now_ts
from pancakebot.core.money import bankroll_suffix, format_bankroll, usd_suffix
from pancakebot.infra.pool_event_watcher import PoolEventWatcher

_LOCK_SAFETY_MARGIN_SECONDS = 3  # abort bet if wall-clock is within this many seconds of lock_at

# Extra cushion added to the claim-check wake time to avoid alignment retries near Graph/RPC boundaries.
_CLAIM_CHECK_PADDING_SECONDS = 5

_CLAIM_BATCH_SIZE = 10
_BACKOFF_SECONDS = [2, 4, 8, 16, 32, 58]  # locked

_TRANSIENT_NETWORK_DELAY_SECONDS = 10
_ONE_MINUTE_MS = 60_000
_DRY_RUNTIME_ARCHIVE_ROOT = Path(__file__).resolve().parents[2].parent / "PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    # Closed rounds store (JSONL; used by backtest only; None in live/dry)
    round_store: ClosedRoundsStore | None

    # Momentum strategy config (always present)
    momentum_gate_config: object  # MomentumGateConfig

    # Momentum gate (OKX 1s live client; None in backtest mode)
    momentum_gate: MomentumGate | None

    # On-chain / identity
    contract: Web3PredictionContract
    wallet_address: str

    # Feature cutoff
    cutoff_seconds: int

    # Protocol constants (cached at startup)
    min_bet_amount_bnb: float
    treasury_fee_fraction: float

    # Runtime latency telemetry.
    latency_log_path: str
    dry_initial_bankroll_bnb: float | None
    wait_for_bet_receipt: bool
    bet_receipt_timeout_seconds: int

    # Execution
    dry: bool

    # Pool event watcher: accumulates BetBull/BetBear events for accurate pools
    pool_watcher: PoolEventWatcher | None = None

    # Mutable runtime state paths used by live/dry loops.
    runtime_state_paths: RuntimeStatePathsConfig = RuntimeStatePathsConfig(
        claim_scan_cursor_path="var/runtime/claim_scan_cursor.txt",
        dry_bets_path="var/runtime/dry_bets.jsonl",
        dry_settled_epochs_path="var/runtime/dry_settled_epochs.txt",
        dry_audit_trades_path="var/runtime/dry_audit_trades.csv",
        dry_cycle_audit_path="var/runtime/dry_cycle_audit.csv",
        dry_bankroll_state_path="var/runtime/dry_bankroll_state.json",
    )


@dataclass(slots=True)
class _ClosedState:
    strategy_pipeline: MomentumOnlyPipeline | None = None
    claim_scan_initialized: bool = False
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


def _archive_dry_runtime_state(
    paths: RuntimeStatePathsConfig,
    *,
    reason: str,
    move_files: bool,
) -> Path | None:
    existing = [path.resolve() for path in _dry_runtime_state_files(paths) if path.exists()]
    if not existing:
        return None
    archive_root = _DRY_RUNTIME_ARCHIVE_ROOT.resolve()
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
                cfg.runtime_state_paths,
                reason="shutdown_snapshot",
                move_files=False,
            )
            if archived is not None:
                info(
                    "RUN",
                    "DRY",
                    "ARCHIVE",
                    msg=f"Saved shutdown dry-state snapshot to {archived}",
                )


def _ensure_parent_dir(path: str) -> None:
    from pancakebot.core.path import ensure_parent_dir
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


def _mono_ms() -> float:
    return time.perf_counter() * 1000.0



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


_MOMENTUM_CACHE_N = 10  # retained for sync_mode compatibility


def _init_closed_state(cfg: RuntimeConfig) -> _ClosedState:
    """Initialise live/dry runtime state. No disk sync — pure RPC + OKX."""
    info("CORE", "RUN", "SETUP", msg="Core setup: strategy=momentum_gate mode=rpc_only")

    strategy_pipeline = _build_momentum_pipeline(cfg=cfg)
    # No warmup rounds needed: OKX gate fetches live at decision time.
    strategy_pipeline.bootstrap_from_closed_rounds(rounds=[])

    closed = _ClosedState(strategy_pipeline=strategy_pipeline)

    if cfg.dry:
        archived = _archive_dry_runtime_state(
            cfg.runtime_state_paths,
            reason="startup_fresh_reset",
            move_files=True,
        )
        if archived is not None:
            info(
                "RUN",
                "DRY",
                "ARCHIVE",
                msg=f"Archived previous dry runtime state to {archived}",
            )
        bankroll_state = _resolve_initial_dry_bankroll_state(cfg)
        closed.simulated_bankroll_bnb = bankroll_state.simulated_bankroll_bnb
        closed.dry_bets_by_epoch = _load_dry_bets(cfg.runtime_state_paths.dry_bets_path)
        _ensure_dry_audit_csv(cfg.runtime_state_paths.dry_audit_trades_path)
        _ensure_dry_cycle_audit_csv(
            cfg.runtime_state_paths.dry_cycle_audit_path,
            reset=True,
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
            msg=(
                f"Loaded dry bankroll {bankroll_state.simulated_bankroll_bnb:.6f} BNB "
                f"source={bankroll_state.source} "
                f"path={cfg.runtime_state_paths.dry_bankroll_state_path} "
                f"cycle_audit_path={cfg.runtime_state_paths.dry_cycle_audit_path}"
            ),
        )

    return closed


def required_runtime_sync_cache_n() -> int:
    return _MOMENTUM_CACHE_N


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
                cursor_path=cfg.runtime_state_paths.claim_scan_cursor_path,
                locked_epoch=locked_epoch,
                current_epoch=current_epoch,
                now_ts=int(now_ts()),
                buffer_seconds=BUFFER_SECONDS,
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

        # Step 5: cutoff_ts(t) = lock_ts(t) - cutoff_seconds.
        cutoff_ts_t = lock_ts_t - cfg.cutoff_seconds

        # If we missed the previous epoch's cutoff and are now targeting a newer epoch, the
        # just-closed locked epoch may become claimable before the next cutoff. In that case,
        # we must wake for claim first (no approximation).
        prev_locked_epoch = locked_round.epoch - 1
        claim_ts = locked_round.lock_at + BUFFER_SECONDS + _CLAIM_CHECK_PADDING_SECONDS
        if now_ts() < claim_ts < cutoff_ts_t:
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=prev_locked_epoch)
            return

        # Step 6: Sleep until cutoff_ts(t).
        _sleep_until_ts(cutoff_ts_t, reason="wait_for_cutoff", epoch=current_epoch)

        # Step 6a: Kick off OKX kline fetch after a short delay — gives
        # OKX time to publish the latest 1s candle before we ask for it.
        # The ~250 ms delay is absorbed by the RPC work in Steps 6b–7
        # (~150–450 ms), so the futures are still ready by evaluate().
        okx_kline_futures = None
        if closed.strategy_pipeline is not None and hasattr(closed.strategy_pipeline, '_gate'):
            gate = closed.strategy_pipeline._gate
            if gate is not None:
                time.sleep(0.25)
                okx_kline_futures = gate.fetch_klines_async(cutoff_ts_ms=int(cutoff_ts_t * 1000))

        # Step 6b: Quick epoch check — just verify current_epoch hasn't
        # shifted during the ~267 s sleep.  A full handshake (3 RPC calls,
        # ~450 ms) is only needed on the rare occasion the epoch actually
        # changed; a single current_epoch() call (~150 ms) suffices for
        # the common case.
        try:
            current_epoch2 = int(cfg.contract.current_epoch())
        except TransientRpcError:
            # On transient failure, fall back to a full handshake.
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
            # Transient RPC failure — full re-handshake as fallback
            locked_round, open_round, current_epoch, _ = _epoch_handshake(cfg, closed)
            locked_epoch = locked_round.epoch
            lock_ts_t = int(open_round.lock_at)
        else:
            # Common path: epoch unchanged — reuse open_round and lock_ts.
            open_round = _open_round

        # Step 7b: Pool data from event subscription (no round_data RPC).
        # Filter to bets with block_timestamp <= lock_at - 6 for consistency
        # with backtest (bets from 6+ seconds ago are guaranteed propagated).
        pool_bull_bnb = 0.0
        pool_bear_bnb = 0.0
        if cfg.pool_watcher is not None and cfg.pool_watcher.connected:
            pool_ts_cutoff = lock_ts_t - POOL_CUTOFF_SECONDS
            pool_bull_bnb, pool_bear_bnb = cfg.pool_watcher.get_pool(
                epoch=current_epoch, max_ts=pool_ts_cutoff,
            )
            pool_total = pool_bull_bnb + pool_bear_bnb
            if pool_total > 0:
                info("POOL", "WSS", "USE",
                     epoch=current_epoch, total=f"{pool_total:.4f}",
                     cutoff=f"lock-6s")
            # Clean up old epochs
            if locked_epoch > 2:
                cfg.pool_watcher.clear_old_epochs(keep_after=locked_epoch - 2)

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
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 11: Execution timing guard (float precision — int truncation
        # was randomly shaving 0–1 s off the budget).
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
                    wait_receipt=cfg.wait_for_bet_receipt,
                    receipt_timeout_seconds=cfg.bet_receipt_timeout_seconds,
                )
            elif decision.bet_side == "Bear":
                tx_submit = cfg.contract.bet_bear_timed(
                    epoch=current_epoch,
                    amount_wei=amount_wei,
                    gas_limit=GAS_LIMIT_BET,
                    gas_price_wei=gas_price_wei,
                    wait_receipt=cfg.wait_for_bet_receipt,
                    receipt_timeout_seconds=cfg.bet_receipt_timeout_seconds,
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
            _append_jsonl(cfg.latency_log_path, latency_record)
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

        # Step 15: Sleep until claim + claim scan.
        _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
        return


def _build_momentum_pipeline(*, cfg: RuntimeConfig) -> MomentumOnlyPipeline:
    """Build momentum-only strategy pipeline."""
    from pancakebot.domain.strategy.momentum_gate import MomentumGateConfig
    gate_config: MomentumGateConfig = cfg.momentum_gate_config  # type: ignore[assignment]
    return MomentumOnlyPipeline(
        config=gate_config,
        gate=cfg.momentum_gate,
        cutoff_seconds=cfg.cutoff_seconds,
        min_bet_amount_bnb=cfg.min_bet_amount_bnb,
        treasury_fee_fraction=cfg.treasury_fee_fraction,
    )


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

    claim_ts = close_ts + BUFFER_SECONDS + _CLAIM_CHECK_PADDING_SECONDS
    _sleep_until_ts(claim_ts, reason="wait_for_claim", epoch=claim_epoch)

    # Refresh epochs after sleeping so the prior locked round can become closed.
    locked_round2, _open_round2, current_epoch2, _open_rd2 = _epoch_handshake(cfg, closed)

    claim_scan_cursor(
        contract=cfg.contract,
        wallet_address=cfg.wallet_address,
        dry=cfg.dry,
        cursor_path=cfg.runtime_state_paths.claim_scan_cursor_path,
        locked_epoch=locked_round2.epoch,
        current_epoch=current_epoch2,
        now_ts=int(now_ts()),
        buffer_seconds=BUFFER_SECONDS,
        get_close_ts=cfg.contract.close_ts,
        page_size=100,
        gas_limit=GAS_LIMIT_CLAIM,
        claim_batch_size=_CLAIM_BATCH_SIZE,
        min_bet_with_gas_bnb=cfg.min_bet_amount_bnb + GAS_COST_BET_BNB,
    )

    _dry_settle_available_bets(cfg, closed)


def _sleep_until_ts(target_ts: int, *, reason: str, epoch: int | None = None) -> None:
    remaining = target_ts - time.time()
    if remaining <= 0:
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

