"""Live runtime loop.

Hard rules (project-level):
  - Do not catch-and-continue exceptions (developer errors must crash).
  - No disk reads in the main live loop: closed rounds are loaded once at startup
    and then maintained via an in-memory rolling cache.

This module orchestrates I/O (Graph, on-chain) and strategy execution for the
shared production strategy pipeline (candidate providers + router).
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from pancakebot.config.app_config import RuntimeStatePathsConfig
from pancakebot.config.strategy_config import StrategyConfig
from pancakebot.core.constants import (
    BNB_WEI,
    GAS_LIMIT_BET,
    GAS_LIMIT_CLAIM,
    GAS_COST_BET_BNB,
)
from pancakebot.domain.closed_rounds_cache import RollingClosedRoundsCache
from pancakebot.infra.closed_rounds_sync import sync_closed_rounds
from pancakebot.infra.graph_client import GraphClient
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.klines_store import KlinesStore
from pancakebot.infra.binance_us_client import BinanceUsClient
from pancakebot.infra.klines_sync import ensure_klines_coverage
from pancakebot.domain.klines_cache import RollingKlinesCache
from pancakebot.domain.types import Bet, Kline, Round
from pancakebot.domain.features.schema import max_required_context_klines_size, max_required_prior_context_rounds_size
from pancakebot.domain.contiguity import check_klines_contiguous, check_rounds_contiguous
from pancakebot.infra.onchain.web3_prediction_contract import Web3PredictionContract
from pancakebot.domain.strategy.dislocation_engine import (
    build_dislocation_engine_from_config,
)
from pancakebot.domain.strategy.direct_action_policy import DirectActionPolicy
from pancakebot.domain.strategy.flow_candidate_adapter import FlowCandidateAdapter
from pancakebot.domain.strategy.ml_candidate_adapter import MlCandidateAdapter
from pancakebot.domain.strategy.pipeline import StrategyPipeline, required_pipeline_warmup_rounds
from pancakebot.domain.strategy.router import StrategyRouter, StrategyRouterConfig
from pancakebot.domain.strategy.window_controller import WindowController
from pancakebot.runtime.claim_manager import claim_scan_cursor
from pancakebot.runtime.bootstrap_snapshot import (
    load_runtime_pipeline_snapshot,
    runtime_pipeline_snapshot_compatibility_key,
    save_runtime_pipeline_snapshot,
)
from pancakebot.runtime.settlement import settle_bet_against_closed_round
from pancakebot.runtime.sleep import sleep_seconds
from pancakebot.core.errors import InvariantError, TransientGraphError, TransientRpcError
from pancakebot.core.logging import error, info, warn
from pancakebot.core.time import now_ts
from pancakebot.core.money import bankroll_suffix, format_bankroll, usd_suffix

_LOCK_SAFETY_MARGIN_SECONDS = 5  # locked

# Extra cushion added to the claim-check wake time to avoid alignment retries near Graph/RPC boundaries.
_CLAIM_CHECK_PADDING_SECONDS = 5

_CLAIM_BATCH_SIZE = 10
_BACKOFF_SECONDS = [2, 4, 8, 16, 32, 58]  # locked

_TRANSIENT_NETWORK_DELAY_SECONDS = 10
_ONE_MINUTE_MS = 60_000
_KLINE_WARMUP_MARGIN_MINUTES = 5
_DRY_RUNTIME_ARCHIVE_ROOT = Path(__file__).resolve().parents[2].parent / "PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    # Graph
    graph_client: GraphClient
    round_store: ClosedRoundsStore

    # Binance US klines
    klines_store: KlinesStore
    binance_us_client: BinanceUsClient
    binance_us_symbol: str

    # On-chain / identity
    contract: Web3PredictionContract
    wallet_address: str

    # Feature cutoff
    cutoff_seconds: int

    # Strategy config
    strategy_cfg: StrategyConfig

    # Protocol constants (cached at startup)
    min_bet_amount_bnb: float
    treasury_fee_fraction: float
    buffer_seconds: int

    # Optional on-chain bet-event replacement for target-round bets at cutoff.
    use_onchain_event_bets: bool
    event_lookback_blocks: int

    # Runtime latency telemetry.
    latency_log_path: str
    dry_initial_bankroll_bnb: float | None
    wait_for_bet_receipt: bool
    bet_receipt_timeout_seconds: int

    # Execution
    dry: bool

    # Optional persisted feature cache store for backtest/inspection acceleration.
    feature_cache_store: object | None = None

    # Optional SQLite market-data mirror used by backtests/inspection.
    market_data_store: object | None = None

    # Optional persistent final-pool projection cache used by ML adapter.
    projection_cache_store: object | None = None

    # Optional run registry store for experiment bookkeeping.
    run_registry_store: object | None = None

    # Backtest-only state snapshot cache root directory.
    backtest_state_cache_dir: str = "../PancakeBot_var_exp/backtest_state_cache"

    # Mutable runtime state paths used by live/dry loops.
    runtime_state_paths: RuntimeStatePathsConfig = RuntimeStatePathsConfig(
        claim_scan_cursor_path="var/runtime/claim_scan_cursor.txt",
        dry_bets_path="var/runtime/dry_bets.jsonl",
        dry_settled_epochs_path="var/runtime/dry_settled_epochs.txt",
        dry_audit_trades_path="var/runtime/dry_audit_trades.csv",
        dry_cycle_audit_path="var/runtime/dry_cycle_audit.csv",
        dry_bankroll_state_path="var/runtime/dry_bankroll_state.json",
        dry_pipeline_bootstrap_state_path="var/runtime/dry_pipeline_bootstrap_state.pkl.gz",
        live_pipeline_bootstrap_state_path="var/runtime/live_pipeline_bootstrap_state.pkl.gz",
    )


@dataclass(slots=True)
class _ClosedState:
    cache: RollingClosedRoundsCache
    disk_latest_epoch: int
    klines_cache: RollingKlinesCache
    strategy_pipeline: StrategyPipeline | None = None
    claim_scan_initialized: bool = False
    simulated_bankroll_bnb: float | None = None
    dry_bets_by_epoch: dict[int, dict[str, object]] | None = None
    dry_settled_epochs: set[int] | None = None
    pipeline_snapshot_saved_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class _DryBankrollState:
    simulated_bankroll_bnb: float
    updated_ts: int
    source: str
    epoch: int | None = None


def _dry_runtime_state_files(paths: RuntimeStatePathsConfig) -> list[Path]:
    return [
        Path(str(paths.claim_scan_cursor_path)),
        Path(str(paths.dry_bets_path)),
        Path(str(paths.dry_settled_epochs_path)),
        Path(str(paths.dry_audit_trades_path)),
        Path(str(paths.dry_cycle_audit_path)),
        Path(str(paths.dry_bankroll_state_path)),
        Path(str(paths.dry_pipeline_bootstrap_state_path)),
    ]


def _unique_archive_dir(root: Path, *, ts: int, reason: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(int(ts)))
    base = root / f"dry_run_archive_{stamp}_{str(reason)}"
    if not base.exists():
        return base
    suffix = 1
    while True:
        cand = root / f"dry_run_archive_{stamp}_{str(reason)}_{int(suffix)}"
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
    archive_dir = _unique_archive_dir(archive_root, ts=ts_now, reason=str(reason))
    archive_dir.mkdir(parents=True, exist_ok=False)
    file_meta: list[dict[str, object]] = []
    for src in existing:
        dest = archive_dir / src.name
        stat = src.stat()
        if bool(move_files):
            shutil.move(str(src), str(dest))
        else:
            shutil.copy2(str(src), str(dest))
        file_meta.append(
            {
                "name": str(src.name),
                "source_path": str(src),
                "archive_path": str(dest),
                "size_bytes": int(stat.st_size),
                "source_mtime_ts": int(stat.st_mtime),
            }
        )
    (archive_dir / "archive_meta.json").write_text(
        json.dumps(
            {
                "reason": str(reason),
                "created_ts": int(ts_now),
                "move_files": bool(move_files),
                "file_count": int(len(file_meta)),
                "files": file_meta,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return archive_dir


def _build_context_klines(*, klines_cache: RollingKlinesCache, target_round: Round, cutoff_seconds: int) -> list[Kline]:
    kk = int(max_required_context_klines_size())
    if target_round.lock_at is None:
        raise InvariantError("target_round_lock_at_missing")
    lock_ts = int(target_round.lock_at)
    cutoff_ts = int(lock_ts) - int(cutoff_seconds)
    anchor_ms = int(cutoff_ts) * 1000
    latest_close_ms = klines_cache.latest_close_time_ms()
    if latest_close_ms is None:
        raise InvariantError("klines_cache_empty")
    if int(latest_close_ms) < int(anchor_ms):
        anchor_ms = int(latest_close_ms)
    return klines_cache.get_context_klines(anchor_close_time_ms=int(anchor_ms), size=int(kk))


def _with_lock_at(round_t: Round, lock_at: int) -> Round:
    if int(lock_at) <= 0:
        raise InvariantError("lock_at_invalid")
    if round_t.lock_at is not None and int(round_t.lock_at) != int(lock_at):
        raise InvariantError("round_lock_at_mismatch")
    return Round(
        epoch=int(round_t.epoch),
        start_at=int(round_t.start_at),
        lock_at=int(lock_at),
        close_at=round_t.close_at,
        lock_price=round_t.lock_price,
        close_price=round_t.close_price,
        position=round_t.position,
        failed=round_t.failed,
        bets=round_t.bets,
    )


def run_live_loop(cfg: RuntimeConfig) -> None:
    if not cfg.wallet_address:
        raise InvariantError("wallet_address_required")
    if float(cfg.min_bet_amount_bnb) <= 0.0:
        raise InvariantError("runtime_min_bet_amount_nonpositive")
    try:
        closed_state = _init_closed_state(cfg)

        # After sync, USD conversion uses the latest closed round close_price.
        bnbusd_price = closed_state.cache.rounds[-1].close_price
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
            except TransientGraphError as e:
                info(
                    "CORE",
                    "RUN",
                    "RETRY",
                    msg=(
                        "Caught TransientGraphError during runtime loop: "
                        f"retrying after delay err={str(e)}"
                    ),
                )
                info(
                    "CORE",
                    "LOOP",
                    "SLEEP",
                    msg=(
                        f"duration={int(_TRANSIENT_NETWORK_DELAY_SECONDS)}s "
                        "reason=delay_after_transient_network_error"
                    ),
                )
                sleep_seconds(int(_TRANSIENT_NETWORK_DELAY_SECONDS))
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
                        f"duration={int(_TRANSIENT_NETWORK_DELAY_SECONDS)}s "
                        "reason=delay_after_transient_network_error"
                    ),
                )
                sleep_seconds(int(_TRANSIENT_NETWORK_DELAY_SECONDS))
    finally:
        if bool(cfg.dry):
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
                    msg=f"Saved shutdown dry-state snapshot to {str(archived)}",
                )


def _ensure_parent_dir(path: str) -> None:
    p = Path(path)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: str, record: dict[str, object]) -> None:
    _ensure_parent_dir(path)
    with open(path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        f.write("\n")


def _write_json_file_atomic(path: str, record: dict[str, object]) -> None:
    out = Path(path)
    _ensure_parent_dir(str(out))
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(record, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(out)


def _mono_ms() -> float:
    return float(time.perf_counter() * 1000.0)


def _runtime_pipeline_snapshot_path(cfg: RuntimeConfig) -> str:
    if bool(cfg.dry):
        return str(cfg.runtime_state_paths.dry_pipeline_bootstrap_state_path)
    return str(cfg.runtime_state_paths.live_pipeline_bootstrap_state_path)


def _runtime_pipeline_snapshot_last_saved_epoch(closed: _ClosedState) -> int | None:
    return None if closed.pipeline_snapshot_saved_epoch is None else int(closed.pipeline_snapshot_saved_epoch)


def _runtime_pipeline_snapshot_compatibility_key(cfg: RuntimeConfig) -> str:
    return str(
        runtime_pipeline_snapshot_compatibility_key(
            strategy_cfg=cfg.strategy_cfg,
            cutoff_seconds=int(cfg.cutoff_seconds),
            treasury_fee_fraction=float(cfg.treasury_fee_fraction),
            round_store_path=str(cfg.round_store.path_jsonl),
            klines_store_path=str(cfg.klines_store.path),
        )
    )


def _save_runtime_pipeline_snapshot(
    *,
    cfg: RuntimeConfig,
    closed: _ClosedState,
) -> int | None:
    if closed.strategy_pipeline is None:
        raise InvariantError("strategy_pipeline_missing")

    pipeline_state = closed.strategy_pipeline.export_bootstrap_state()
    last_settled_epoch_raw = pipeline_state.get("last_settled_epoch")
    last_settled_epoch = None if last_settled_epoch_raw is None else int(last_settled_epoch_raw)
    last_round = None if last_settled_epoch is None else closed.cache.get_round(int(last_settled_epoch))
    save_runtime_pipeline_snapshot(
        path=_runtime_pipeline_snapshot_path(cfg),
        compatibility_key=_runtime_pipeline_snapshot_compatibility_key(cfg),
        pipeline_state=pipeline_state,
        last_settled_round=last_round,
    )
    closed.pipeline_snapshot_saved_epoch = last_settled_epoch
    return None if last_settled_epoch is None else int(last_settled_epoch)


def _bootstrap_strategy_pipeline_from_runtime_snapshot(
    *,
    cfg: RuntimeConfig,
    closed: _ClosedState,
    rounds_all: list[Round],
    warmup_rounds: list[Round],
) -> None:
    if closed.strategy_pipeline is None:
        raise InvariantError("strategy_pipeline_missing")
    if not rounds_all:
        raise InvariantError("closed_rounds_empty_for_runtime_bootstrap")
    if not warmup_rounds:
        raise InvariantError("warmup_rounds_empty_for_runtime_bootstrap")

    rounds_by_epoch = {int(round_t.epoch): round_t for round_t in rounds_all}
    snapshot_path = _runtime_pipeline_snapshot_path(cfg)
    compatibility_key = _runtime_pipeline_snapshot_compatibility_key(cfg)
    snapshot = load_runtime_pipeline_snapshot(
        path=str(snapshot_path),
        compatibility_key=str(compatibility_key),
        rounds_by_epoch=rounds_by_epoch,
    )

    if isinstance(snapshot, dict):
        pipeline_state = snapshot.get("pipeline_state")
        if not isinstance(pipeline_state, dict):
            raise InvariantError("runtime_pipeline_snapshot_state_missing")
        closed.strategy_pipeline.import_bootstrap_state(state=pipeline_state)
        restored_epoch = closed.strategy_pipeline.last_settled_epoch
        if restored_epoch is None:
            raise InvariantError("runtime_pipeline_snapshot_last_settled_epoch_missing")
        delta_rounds = [
            round_t
            for round_t in rounds_all
            if int(round_t.epoch) > int(restored_epoch)
        ]
        info(
            "CORE",
            "CACHE",
            "HIT",
            msg=(
                f"path={str(snapshot_path)} restored_epoch={int(restored_epoch)} "
                f"delta_rounds={int(len(delta_rounds))}"
            ),
        )
        if delta_rounds:
            closed.strategy_pipeline.bootstrap_from_closed_rounds(rounds=list(delta_rounds))
            _save_runtime_pipeline_snapshot(cfg=cfg, closed=closed)
        else:
            closed.pipeline_snapshot_saved_epoch = int(restored_epoch)
        return

    info(
        "CORE",
        "CACHE",
        "MISS",
        msg=f"path={str(snapshot_path)} warmup_rounds={int(len(warmup_rounds))}",
    )
    closed.strategy_pipeline.bootstrap_from_closed_rounds(rounds=list(warmup_rounds))
    _save_runtime_pipeline_snapshot(cfg=cfg, closed=closed)


def _maybe_persist_runtime_pipeline_snapshot(
    *,
    cfg: RuntimeConfig,
    closed: _ClosedState,
) -> None:
    if closed.strategy_pipeline is None:
        raise InvariantError("strategy_pipeline_missing")
    current_epoch = closed.strategy_pipeline.last_settled_epoch
    if current_epoch is None:
        return
    saved_epoch = _runtime_pipeline_snapshot_last_saved_epoch(closed)
    if saved_epoch is not None and int(current_epoch) <= int(saved_epoch):
        return
    _save_runtime_pipeline_snapshot(cfg=cfg, closed=closed)


def _replace_open_round_bets_from_events(
    *,
    cfg: RuntimeConfig,
    open_round: Round,
    cutoff_ts: int,
) -> tuple[Round | None, str | None]:
    if not bool(cfg.use_onchain_event_bets):
        return open_round, None

    head_block = int(cfg.contract.latest_block_number())
    head_ts = int(cfg.contract.block_timestamp(int(head_block)))
    if int(head_ts) < int(cutoff_ts):
        return None, "event_chain_head_behind_cutoff"

    lookback_blocks = int(cfg.event_lookback_blocks)
    if lookback_blocks <= 0:
        raise InvariantError("event_lookback_blocks_nonpositive")

    from_block = int(max(1, int(head_block) - int(lookback_blocks)))
    from_block_ts = int(cfg.contract.block_timestamp(int(from_block)))
    if int(from_block_ts) > int(open_round.start_at):
        return None, "event_lookback_insufficient_for_round_start"

    events = cfg.contract.fetch_bet_events_for_epoch(
        epoch=int(open_round.epoch),
        from_block=int(from_block),
        to_block=int(head_block),
    )
    event_bets = tuple(
        Bet(
            wallet_address=str(ev.wallet_address),
            amount_wei=int(ev.amount_wei),
            position=str(ev.position),
            created_at=int(ev.block_timestamp),
        )
        for ev in events
    )

    out = Round(
        epoch=int(open_round.epoch),
        start_at=int(open_round.start_at),
        lock_at=open_round.lock_at,
        close_at=open_round.close_at,
        lock_price=open_round.lock_price,
        close_price=open_round.close_price,
        position=open_round.position,
        failed=open_round.failed,
        bets=event_bets,
    )

    info(
        "RUN",
        "DATA",
        "EVENTBETS",
        msg=(
            f"epoch={int(open_round.epoch)} "
            f"events={int(len(event_bets))} "
            f"from_block={int(from_block)} "
            f"to_block={int(head_block)} "
            f"head_ts={int(head_ts)} "
            f"cutoff_ts={int(cutoff_ts)}"
        ),
    )
    return out, None


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
        if int(epoch) in bets:
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
    if not isinstance(source_raw, str) or str(source_raw).strip() == "":
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
    if int(updated_ts_raw) < 0:
        raise InvariantError(f"dry_bankroll_state_updated_ts_negative: path={path}")
    return _DryBankrollState(
        simulated_bankroll_bnb=float(bankroll_bnb),
        updated_ts=int(updated_ts_raw),
        source=str(source_raw).strip(),
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
    if float(bankroll_bnb) < 0.0:
        raise InvariantError("dry_bankroll_state_bankroll_negative")
    if int(updated_ts) < 0:
        raise InvariantError("dry_bankroll_state_updated_ts_negative")
    source_name = str(source).strip()
    if source_name == "":
        raise InvariantError("dry_bankroll_state_source_empty")
    state = _DryBankrollState(
        simulated_bankroll_bnb=float(bankroll_bnb),
        updated_ts=int(updated_ts),
        source=str(source_name),
        epoch=(None if epoch is None else int(epoch)),
    )
    _write_json_file_atomic(
        path,
        {
            "simulated_bankroll_bnb": float(state.simulated_bankroll_bnb),
            "updated_ts": int(state.updated_ts),
            "source": str(state.source),
            "epoch": (None if state.epoch is None else int(state.epoch)),
        },
    )
    return state


def _recover_dry_bankroll_state_from_logs(
    *,
    dry_bets_path: str,
    dry_audit_trades_path: str,
) -> _DryBankrollState | None:
    latest_state: _DryBankrollState | None = None

    for rec in _load_dry_bets(str(dry_bets_path)).values():
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
        if latest_state is None or int(state.updated_ts) > int(latest_state.updated_ts):
            latest_state = state

    audit_path = Path(str(dry_audit_trades_path))
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
                    simulated_bankroll_bnb=float(bankroll_bnb),
                    updated_ts=int(settled_ts),
                    source="recover_from_dry_settle",
                    epoch=epoch,
                )
                if latest_state is None or int(state.updated_ts) > int(latest_state.updated_ts):
                    latest_state = state

    return latest_state


def _resolve_initial_dry_bankroll_state(cfg: RuntimeConfig) -> _DryBankrollState:
    persisted = _load_dry_bankroll_state(str(cfg.runtime_state_paths.dry_bankroll_state_path))
    recovered = _recover_dry_bankroll_state_from_logs(
        dry_bets_path=str(cfg.runtime_state_paths.dry_bets_path),
        dry_audit_trades_path=str(cfg.runtime_state_paths.dry_audit_trades_path),
    )
    configured_init = None if cfg.dry_initial_bankroll_bnb is None else float(cfg.dry_initial_bankroll_bnb)
    can_override_persisted_seed = (
        configured_init is not None
        and persisted is not None
        and recovered is None
        and persisted.epoch is None
        and str(persisted.source) in {"wallet_init", "configured_init"}
    )
    if (
        persisted is not None
        and not bool(can_override_persisted_seed)
        and (recovered is None or int(persisted.updated_ts) >= int(recovered.updated_ts))
    ):
        return persisted
    if recovered is not None:
        return _save_dry_bankroll_state(
            str(cfg.runtime_state_paths.dry_bankroll_state_path),
            bankroll_bnb=float(recovered.simulated_bankroll_bnb),
            source="recovered",
            epoch=recovered.epoch,
            updated_ts=int(recovered.updated_ts),
        )
    if configured_init is not None:
        return _save_dry_bankroll_state(
            str(cfg.runtime_state_paths.dry_bankroll_state_path),
            bankroll_bnb=float(configured_init),
            source="configured_init",
            epoch=None,
            updated_ts=int(now_ts()),
        )
    wallet_bnb = _fetch_wallet_balance_bnb_with_retries(
        cfg=cfg,
        reason="dry_wallet_bootstrap",
    )
    return _save_dry_bankroll_state(
        str(cfg.runtime_state_paths.dry_bankroll_state_path),
        bankroll_bnb=float(wallet_bnb),
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
                    f"Caught TransientRpcError during {str(reason)}: "
                    f"retrying after delay err={str(e)}"
                ),
            )
            info(
                "CORE",
                "LOOP",
                "SLEEP",
                msg=(
                    f"duration={int(delay_seconds)}s "
                    "reason=delay_after_transient_network_error"
                ),
            )
            sleep_seconds(int(delay_seconds))
    return float(cfg.contract.wallet_balance_bnb(cfg.wallet_address))


def _append_dry_settled_epoch(path: str, epoch: int) -> None:
    _ensure_parent_dir(path)
    with open(path, "a") as f:
        f.write(str(int(epoch)))
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
    if int(epoch) in closed.dry_bets_by_epoch:
        raise InvariantError(f"dry_bet_duplicate_epoch: epoch={int(epoch)}")
    placed_ts = int(now_ts())
    rec = {
        "epoch": int(epoch),
        "placed_ts": int(placed_ts),
        "bet_side": str(side),
        "bet_bnb": float(amount_bnb),
        "p_final": float(p_final),
        "pred_win_probability": float(p_final),
        "expected_profit_bnb": float(expected_profit_bnb),
        "bankroll_before_bet_bnb": float(bankroll_before_bet_bnb),
        "bankroll_after_bet_bnb": float(bankroll_after_bet_bnb),
    }
    closed.dry_bets_by_epoch[int(epoch)] = rec
    _append_jsonl(str(cfg.runtime_state_paths.dry_bets_path), rec)
    _save_dry_bankroll_state(
        str(cfg.runtime_state_paths.dry_bankroll_state_path),
        bankroll_bnb=float(bankroll_after_bet_bnb),
        source="bet",
        epoch=int(epoch),
        updated_ts=int(placed_ts),
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


def _controller_profile_label(name: str) -> str:
    text = str(name).strip()
    aliases = {
        "disloc_stageB_bullonly_recent8pct_v1": "stageB",
        "disloc_stageG2_bullonly_recent5pct_v1": "stageG2",
        "disloc_altB_20260227_x80": "altB",
    }
    if text in aliases:
        return str(aliases[text])
    if text.startswith("disloc_"):
        return str(text[len("disloc_") :])
    return str(text)


def _parse_controller_metric_map(raw: str | object) -> dict[str, float]:
    if isinstance(raw, dict):
        data = raw
    else:
        text = str(raw or "").strip()
        if text == "":
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in data.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _controller_decision_log_suffix(
    *,
    decision: object | None,
    final_action: str,
    final_skip_reason: str | None = None,
) -> str:
    if decision is None:
        return ""
    mode = str(getattr(decision, "controller_mode", "") or "").strip()
    if mode == "":
        return ""
    estimator_mode = str(getattr(decision, "controller_estimator_mode", "") or "").strip()
    window_index = getattr(decision, "controller_window_index", None)
    lookback_windows_used = getattr(decision, "controller_lookback_windows_used", None)
    selected_action = str(getattr(decision, "controller_selected_action", "") or "").strip()
    selected_profile = str(getattr(decision, "controller_selected_profile", "") or "").strip()
    selected_per_500 = getattr(decision, "controller_estimated_per_500", None)
    selected_score_per_500 = getattr(decision, "controller_estimated_score_per_500", None)
    selected_bet_rate = getattr(decision, "controller_estimated_selected_bet_rate", None)
    per_500_map = _parse_controller_metric_map(
        getattr(decision, "controller_estimated_profiles_per_500_json", "")
    )
    score_map = _parse_controller_metric_map(
        getattr(decision, "controller_estimated_profiles_score_per_500_json", "")
    )
    bet_rate_map = _parse_controller_metric_map(
        getattr(decision, "controller_estimated_profiles_bet_rate_json", "")
    )
    names = set(per_500_map) | set(score_map) | set(bet_rate_map)
    if selected_profile != "":
        names.add(str(selected_profile))
    ordered_names = sorted(
        names,
        key=lambda name: (
            0 if str(name) == str(selected_profile) and str(selected_profile) != "" else 1,
            -float(score_map.get(str(name), float("-inf"))),
            str(name),
        ),
    )
    profile_parts: list[str] = []
    for name in ordered_names:
        label = _controller_profile_label(str(name))
        prefix = "*" if str(name) == str(selected_profile) and str(selected_profile) != "" else ""
        part = (
            f"{prefix}{label}:"
            f"{float(per_500_map.get(str(name), 0.0)):+.4f}/"
            f"{float(score_map.get(str(name), 0.0)):+.4f}/"
            f"{100.0 * float(bet_rate_map.get(str(name), 0.0)):.1f}%"
        )
        profile_parts.append(part)
    suffix_parts = [
        f"mode={mode}",
        (f"estimator={estimator_mode}" if estimator_mode != "" else None),
        (f"win={int(window_index)}" if window_index is not None else None),
        (
            f"hist={int(lookback_windows_used)}"
            if lookback_windows_used is not None
            else None
        ),
        f"ctrl={selected_action or 'off'}",
        (
            f"pick={_controller_profile_label(selected_profile)}"
            if selected_profile != ""
            else "pick=skip"
        ),
        (
            f"est500={float(selected_per_500):+.4f}"
            if isinstance(selected_per_500, (int, float))
            else None
        ),
        (
            f"score500={float(selected_score_per_500):+.4f}"
            if isinstance(selected_score_per_500, (int, float))
            else None
        ),
        (
            f"rate={100.0 * float(selected_bet_rate):.1f}%"
            if isinstance(selected_bet_rate, (int, float))
            else None
        ),
        f"final={str(final_action)}",
        (
            f"reason={str(final_skip_reason)}"
            if final_skip_reason is not None and str(final_skip_reason).strip() != ""
            else None
        ),
        ("profiles=" + ",".join(profile_parts) if profile_parts else None),
    ]
    return " ctrl[" + " ".join(part for part in suffix_parts if part) + "]"


def _log_controller_decision(
    *,
    current_epoch: int,
    decision: object | None,
    final_action: str,
    final_skip_reason: str | None = None,
) -> None:
    suffix = _controller_decision_log_suffix(
        decision=decision,
        final_action=str(final_action),
        final_skip_reason=(None if final_skip_reason is None else str(final_skip_reason)),
    )
    if suffix == "":
        return
    info("RUN", "CTRL", "DECIDE", msg=f"Epoch {int(current_epoch)}{suffix}")


def _direct_action_decision_log_suffix(
    *,
    decision: object | None,
    final_action: str,
    final_skip_reason: str | None = None,
) -> str:
    if decision is None:
        return ""
    mode = str(getattr(decision, "direct_action_mode", "") or "").strip()
    if mode == "":
        return ""
    action_id = str(getattr(decision, "direct_action_action_id", "") or "").strip()
    action_label = str(getattr(decision, "direct_action_action_label", "") or "").strip()
    score_bnb = getattr(decision, "direct_action_score_bnb", None)
    q50_bnb = getattr(decision, "direct_action_q50_bnb", None)
    top_raw = str(getattr(decision, "direct_action_top_actions_json", "") or "").strip()
    top_parts: list[str] = []
    if top_raw != "":
        try:
            rows = json.loads(top_raw)
        except json.JSONDecodeError:
            rows = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = str(row.get("label", row.get("action_id", ""))).strip()
                if label == "":
                    continue
                try:
                    score_value = float(row.get("score_bnb", row.get("q10_net_bnb", 0.0)))
                    q10_value = float(row.get("q10_net_bnb", 0.0))
                    q50_value = float(row.get("q50_net_bnb", 0.0))
                except (TypeError, ValueError):
                    continue
                top_parts.append(f"{label}:{score_value:+.4f}/{q10_value:+.4f}/{q50_value:+.4f}")
    suffix_parts = [
        f"mode={mode}",
        (f"pick={action_label}" if action_label != "" else None),
        (f"id={action_id}" if action_id != "" else None),
        (f"score={float(score_bnb):+.4f}" if isinstance(score_bnb, (int, float)) else None),
        (f"q50={float(q50_bnb):+.4f}" if isinstance(q50_bnb, (int, float)) else None),
        f"final={str(final_action)}",
        (
            f"reason={str(final_skip_reason)}"
            if final_skip_reason is not None and str(final_skip_reason).strip() != ""
            else None
        ),
        ("top=" + ",".join(top_parts) if top_parts else None),
    ]
    return " dap[" + " ".join(part for part in suffix_parts if part) + "]"


def _log_direct_action_decision(
    *,
    current_epoch: int,
    decision: object | None,
    final_action: str,
    final_skip_reason: str | None = None,
) -> None:
    suffix = _direct_action_decision_log_suffix(
        decision=decision,
        final_action=str(final_action),
        final_skip_reason=(None if final_skip_reason is None else str(final_skip_reason)),
    )
    if suffix == "":
        return
    info("RUN", "DAP", "DECIDE", msg=f"Epoch {int(current_epoch)}{suffix}")


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
        "controller_mode",
        "controller_estimator_mode",
        "controller_window_index",
        "controller_lookback_windows_used",
        "controller_selected_profile",
        "controller_selected_action",
        "controller_estimated_per_500",
        "controller_estimated_score_per_500",
        "controller_estimated_selected_bet_rate",
        "controller_estimated_profiles_per_500_json",
        "controller_estimated_profiles_score_per_500_json",
        "controller_estimated_profiles_bet_rate_json",
        "direct_action_mode",
        "direct_action_action_id",
        "direct_action_action_label",
        "direct_action_score_bnb",
        "direct_action_q50_bnb",
        "direct_action_top_actions_json",
        "action",
        "decision_stage",
        "selected_strategy",
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
    if bool(reset) or not p.exists():
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
        if cutoff_ts is not None and int(bet.created_at) > int(cutoff_ts):
            continue
        if str(bet.position) == "Bull":
            bull_wei += int(bet.amount_wei)
            bull_bets += 1
        elif str(bet.position) == "Bear":
            bear_wei += int(bet.amount_wei)
            bear_bets += 1
        else:
            raise InvariantError(f"unexpected_round_bet_side: {bet.position}")

    bull_bnb = float(bull_wei) / float(BNB_WEI)
    bear_bnb = float(bear_wei) / float(BNB_WEI)
    return {
        f"{prefix}_total_pool_bnb": float(bull_bnb + bear_bnb),
        f"{prefix}_bull_pool_bnb": float(bull_bnb),
        f"{prefix}_bear_pool_bnb": float(bear_bnb),
        f"{prefix}_total_bets": int(bull_bets + bear_bets),
        f"{prefix}_bull_bets": int(bull_bets),
        f"{prefix}_bear_bets": int(bear_bets),
    }


def _selected_side_probability(*, p_bull: float | None, bet_side: str | None) -> float | str:
    if p_bull is None or bet_side is None:
        return ""
    if str(bet_side) == "Bull":
        return float(p_bull)
    if str(bet_side) == "Bear":
        return float(1.0 - float(p_bull))
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
) -> None:
    if not bool(cfg.dry):
        return

    observed_pool = _round_pool_snapshot(open_round, prefix="observed")
    cutoff_used_pool = _round_pool_snapshot(
        open_round,
        prefix="cutoff_used",
        cutoff_ts=int(cutoff_ts),
    )
    router_mode: str | object = ""
    pipeline_last_settled_epoch: int | str = ""
    if closed.strategy_pipeline is not None:
        router_mode = str(closed.strategy_pipeline.router_mode)
        if closed.strategy_pipeline.last_settled_epoch is not None:
            pipeline_last_settled_epoch = int(closed.strategy_pipeline.last_settled_epoch)

    selected_strategy: str | object = ""
    controller_mode: str | object = ""
    controller_estimator_mode: str | object = ""
    controller_window_index: int | str = ""
    controller_lookback_windows_used: int | str = ""
    controller_selected_profile: str | object = ""
    controller_selected_action: str | object = ""
    controller_estimated_per_500: float | str = ""
    controller_estimated_score_per_500: float | str = ""
    controller_estimated_selected_bet_rate: float | str = ""
    controller_estimated_profiles_per_500_json: str | object = ""
    controller_estimated_profiles_score_per_500_json: str | object = ""
    controller_estimated_profiles_bet_rate_json: str | object = ""
    direct_action_mode: str | object = ""
    direct_action_action_id: str | object = ""
    direct_action_action_label: str | object = ""
    direct_action_score_bnb: float | str = ""
    direct_action_q50_bnb: float | str = ""
    direct_action_top_actions_json: str | object = ""
    bet_side: str | object = ""
    bet_size_bnb: float | str = ""
    p_bull: float | str = ""
    expected_profit_bnb: float | str = ""
    selector_score_bnb: float | str = ""
    if decision is not None:
        selected_strategy = getattr(decision, "selected_strategy", "") or ""
        controller_mode = getattr(decision, "controller_mode", "") or ""
        controller_estimator_mode = getattr(decision, "controller_estimator_mode", "") or ""
        controller_window_raw = getattr(decision, "controller_window_index", None)
        if controller_window_raw is not None:
            controller_window_index = int(controller_window_raw)
        controller_lookback_used_raw = getattr(decision, "controller_lookback_windows_used", None)
        if controller_lookback_used_raw is not None:
            controller_lookback_windows_used = int(controller_lookback_used_raw)
        controller_selected_profile = getattr(decision, "controller_selected_profile", "") or ""
        controller_selected_action = getattr(decision, "controller_selected_action", "") or ""
        controller_estimated_raw = getattr(decision, "controller_estimated_per_500", None)
        if isinstance(controller_estimated_raw, (int, float)):
            controller_estimated_per_500 = float(controller_estimated_raw)
        controller_estimated_score_raw = getattr(decision, "controller_estimated_score_per_500", None)
        if isinstance(controller_estimated_score_raw, (int, float)):
            controller_estimated_score_per_500 = float(controller_estimated_score_raw)
        controller_estimated_bet_rate_raw = getattr(
            decision,
            "controller_estimated_selected_bet_rate",
            None,
        )
        if isinstance(controller_estimated_bet_rate_raw, (int, float)):
            controller_estimated_selected_bet_rate = float(controller_estimated_bet_rate_raw)
        controller_estimated_profiles_per_500_json = (
            getattr(decision, "controller_estimated_profiles_per_500_json", "") or ""
        )
        controller_estimated_profiles_score_per_500_json = (
            getattr(decision, "controller_estimated_profiles_score_per_500_json", "") or ""
        )
        controller_estimated_profiles_bet_rate_json = (
            getattr(decision, "controller_estimated_profiles_bet_rate_json", "") or ""
        )
        direct_action_mode = getattr(decision, "direct_action_mode", "") or ""
        direct_action_action_id = getattr(decision, "direct_action_action_id", "") or ""
        direct_action_action_label = getattr(decision, "direct_action_action_label", "") or ""
        direct_action_score_raw = getattr(decision, "direct_action_score_bnb", None)
        if isinstance(direct_action_score_raw, (int, float)):
            direct_action_score_bnb = float(direct_action_score_raw)
        direct_action_q50_raw = getattr(decision, "direct_action_q50_bnb", None)
        if isinstance(direct_action_q50_raw, (int, float)):
            direct_action_q50_bnb = float(direct_action_q50_raw)
        direct_action_top_actions_json = (
            getattr(decision, "direct_action_top_actions_json", "") or ""
        )
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
        str(cfg.runtime_state_paths.dry_cycle_audit_path),
        {
            "cycle_ts": int(now_ts()),
            "current_epoch": int(current_epoch),
            "locked_epoch": int(locked_epoch),
            "lock_ts": int(lock_ts),
            "cutoff_ts": int(cutoff_ts),
            "locked_price_bnbusd": float(locked_price_bnbusd),
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
            "controller_mode": controller_mode,
            "controller_estimator_mode": controller_estimator_mode,
            "controller_window_index": controller_window_index,
            "controller_lookback_windows_used": controller_lookback_windows_used,
            "controller_selected_profile": controller_selected_profile,
            "controller_selected_action": controller_selected_action,
            "controller_estimated_per_500": controller_estimated_per_500,
            "controller_estimated_score_per_500": controller_estimated_score_per_500,
            "controller_estimated_selected_bet_rate": controller_estimated_selected_bet_rate,
            "controller_estimated_profiles_per_500_json": controller_estimated_profiles_per_500_json,
            "controller_estimated_profiles_score_per_500_json": controller_estimated_profiles_score_per_500_json,
            "controller_estimated_profiles_bet_rate_json": controller_estimated_profiles_bet_rate_json,
            "direct_action_mode": direct_action_mode,
            "direct_action_action_id": direct_action_action_id,
            "direct_action_action_label": direct_action_action_label,
            "direct_action_score_bnb": direct_action_score_bnb,
            "direct_action_q50_bnb": direct_action_q50_bnb,
            "direct_action_top_actions_json": direct_action_top_actions_json,
            "action": str(action),
            "decision_stage": str(decision_stage),
            "selected_strategy": selected_strategy,
            "bet_side": bet_side,
            "bet_size_bnb": bet_size_bnb,
            "p_bull": p_bull,
            "selected_side_probability": _selected_side_probability(
                p_bull=None if p_bull == "" else float(p_bull),
                bet_side=None if bet_side == "" else str(bet_side),
            ),
            "expected_profit_bnb": expected_profit_bnb,
            "selector_score_bnb": selector_score_bnb,
            "decision_latency_ms": (
                "" if decision_latency_ms is None else float(decision_latency_ms)
            ),
            "skip_reason": "" if skip_reason is None else str(skip_reason),
        },
    )


def _dry_settle_available_bets(cfg: RuntimeConfig, closed: _ClosedState) -> None:
    if not cfg.dry:
        return
    if closed.simulated_bankroll_bnb is None:
        raise InvariantError("dry_bankroll_uninitialized")
    if closed.dry_bets_by_epoch is None or closed.dry_settled_epochs is None:
        raise InvariantError("dry_state_uninitialized")

    # Settle any bet epochs that are now present as closed rounds in cache.
    for epoch, bet in sorted(closed.dry_bets_by_epoch.items()):
        e = int(epoch)
        if e in closed.dry_settled_epochs:
            continue
        r = closed.cache.get_round(e)
        if r is None or r.lock_at is None or r.position is None:
            continue  # not closed/usable yet in cache

        bet_bnb_raw = bet.get("bet_bnb", 0.0)
        if isinstance(bet_bnb_raw, (int, float)):
            bet_bnb = float(bet_bnb_raw)
        elif isinstance(bet_bnb_raw, str):
            try:
                bet_bnb = float(bet_bnb_raw)
            except ValueError as e:
                raise InvariantError("dry_bet_bnb_parse_failed") from e
        else:
            raise InvariantError("dry_bet_bnb_type_invalid")
        if bet_bnb <= 0:
            closed.dry_settled_epochs.add(e)
            _append_dry_settled_epoch(str(cfg.runtime_state_paths.dry_settled_epochs_path), e)
            continue

        settle = settle_bet_against_closed_round(
            bet_bnb=bet_bnb,
            bet_side=str(bet.get("bet_side", "")),
            round_closed=r,
            treasury_fee_fraction=cfg.treasury_fee_fraction,
        )

        outcome = str(settle.outcome)
        credit_bnb = settle.credit_bnb

        bankroll_before_settle = closed.simulated_bankroll_bnb
        closed.simulated_bankroll_bnb += float(credit_bnb)
        bankroll_after_settle = closed.simulated_bankroll_bnb

        settle_price = r.close_price if r.close_price is not None else r.lock_price
        if settle_price is None:
            raise InvariantError("dry_settle_missing_bnbusd_price")
        bnbusd_price = settle_price

        # Brief INFO log (no key=value fields)
        if str(outcome) == "win":
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
        elif str(outcome) == "refund":
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
            placed_ts_val: int | str = int(placed_raw)
        elif isinstance(placed_raw, str) and placed_raw.isdigit():
            placed_ts_val = int(placed_raw)
        else:
            placed_ts_val = ""

        _append_dry_audit_row(
            str(cfg.runtime_state_paths.dry_audit_trades_path),
            {
                "epoch": int(e),
                "placed_ts": placed_ts_val,
                "bet_side": str(bet.get("bet_side", "")),
                "bet_bnb": float(bet_bnb),
                "pred_win_probability": bet.get("pred_win_probability", ""),
                "p_final": bet.get("p_final", ""),
                "expected_profit_bnb": bet.get("expected_profit_bnb", ""),
                "cutoff_bull_bnb": bet.get("cutoff_bull_bnb", ""),
                "cutoff_bear_bnb": bet.get("cutoff_bear_bnb", ""),
                "final_bull_bnb": bet.get("final_bull_bnb", ""),
                "final_bear_bnb": bet.get("final_bear_bnb", ""),
                "settled_ts": int(settled_ts),
                "outcome": str(outcome),
                "pnl_bnb": float(credit_bnb),
                "bankroll_before_bet_bnb": bet.get("bankroll_before_bet_bnb", ""),
                "bankroll_after_bet_bnb": bet.get("bankroll_after_bet_bnb", ""),
                "bankroll_before_settle_bnb": float(bankroll_before_settle),
                "bankroll_after_settle_bnb": float(bankroll_after_settle),
            },
        )

        closed.dry_settled_epochs.add(e)
        _append_dry_settled_epoch(str(cfg.runtime_state_paths.dry_settled_epochs_path), e)
        _save_dry_bankroll_state(
            str(cfg.runtime_state_paths.dry_bankroll_state_path),
            bankroll_bnb=float(bankroll_after_settle),
            source="settle",
            epoch=int(e),
            updated_ts=int(settled_ts),
        )


def _init_closed_state(cfg: RuntimeConfig) -> _ClosedState:
    """Startup-only closed rounds sync + in-memory cache load."""
    k = int(max_required_prior_context_rounds_size())
    warmup_rounds = int(required_pipeline_warmup_rounds(strategy_cfg=cfg.strategy_cfg))
    if warmup_rounds <= 0:
        raise InvariantError("pipeline_warmup_rounds_nonpositive")

    if k <= 0:
        context_desc = "target_only"
    else:
        context_desc = f"prior_context_rounds[{int(k)}]"

    cache_n = max(2, int(warmup_rounds))

    info(
        "CORE",
        "RUN",
        "SETUP",
        msg=(
            f"Core setup: prior_context_rounds_required={int(k)} context={context_desc} "
            f"strategy=shared_pipeline warmup_rounds={int(warmup_rounds)} "
            f"inference_closed_cache_needed={int(cache_n)}"
        ),
    )

    info(
        "CORE",
        "STORE",
        "SYNC",
        msg=f"Syncing closed rounds: cache_n={int(cache_n)}",
    )
    while True:
        try:
            sync_closed_rounds(graph=cfg.graph_client, store=cfg.round_store, cache_n=cache_n)
            break
        except TransientGraphError as e:
            info(
                "CORE",
                "STORE",
                "RETRY",
                msg=(
                    "Caught TransientGraphError during initial sync: "
                    f"retrying after delay err={str(e)}"
                ),
            )
            info(
                "CORE",
                "LOOP",
                "SLEEP",
                msg=(
                    f"duration={int(_TRANSIENT_NETWORK_DELAY_SECONDS)}s "
                    "reason=delay_after_transient_network_error"
                ),
            )
            sleep_seconds(int(_TRANSIENT_NETWORK_DELAY_SECONDS))

    # Load from disk exactly once at startup.
    rounds_all = list(cfg.round_store.iter_closed_rounds())
    stored_n = int(len(rounds_all))
    if not rounds_all:
        raise InvariantError("closed_rounds_store_empty_after_sync")

    # Rolling trim (memory only).
    if len(rounds_all) > cache_n:
        rounds_all = rounds_all[-cache_n:]

    cache = RollingClosedRoundsCache(rounds=rounds_all, capacity=cache_n)
    cache_lo = int(cache.rounds[0].epoch)
    cache_hi = int(cache.rounds[-1].epoch)
    info(
        "CORE",
        "STORE",
        "SYNC",
        msg=f"Closed rounds sync complete: stored_n={int(stored_n)} cache_epochs=[{int(cache_lo)}..{int(cache_hi)}]",
    )

    if len(cache.rounds) < 2:
        raise InvariantError("insufficient_closed_rounds_for_dislocation_strategy")

    disk_latest_epoch = int(cache.rounds[-1].epoch)

    klines_cache = _init_klines_cache(cfg=cfg, closed_cache=cache)
    strategy_pipeline = _build_strategy_pipeline(cfg=cfg, klines_cache=klines_cache)

    closed = _ClosedState(
        cache=cache,
        disk_latest_epoch=disk_latest_epoch,
        klines_cache=klines_cache,
        strategy_pipeline=strategy_pipeline,
    )
    _bootstrap_strategy_pipeline_from_runtime_snapshot(
        cfg=cfg,
        closed=closed,
        rounds_all=list(rounds_all),
        warmup_rounds=list(cache.rounds),
    )

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
                msg=f"Archived previous dry runtime state to {str(archived)}",
            )
        bankroll_state = _resolve_initial_dry_bankroll_state(cfg)
        closed.simulated_bankroll_bnb = float(bankroll_state.simulated_bankroll_bnb)
        closed.dry_bets_by_epoch = _load_dry_bets(str(cfg.runtime_state_paths.dry_bets_path))
        _ensure_dry_audit_csv(str(cfg.runtime_state_paths.dry_audit_trades_path))
        _ensure_dry_cycle_audit_csv(
            str(cfg.runtime_state_paths.dry_cycle_audit_path),
            reset=True,
        )
        settled_epochs = _load_dry_settled_epochs(str(cfg.runtime_state_paths.dry_settled_epochs_path))
        settled_epochs.update(
            _load_dry_settled_epochs_from_audit(str(cfg.runtime_state_paths.dry_audit_trades_path))
        )
        closed.dry_settled_epochs = settled_epochs
        info(
            "RUN",
            "DRY",
            "STATE",
            msg=(
                f"Loaded dry bankroll {float(bankroll_state.simulated_bankroll_bnb):.6f} BNB "
                f"source={str(bankroll_state.source)} "
                f"path={str(cfg.runtime_state_paths.dry_bankroll_state_path)} "
                f"cycle_audit_path={str(cfg.runtime_state_paths.dry_cycle_audit_path)}"
            ),
        )

    return closed


def required_runtime_sync_cache_n(*, strategy_cfg: StrategyConfig) -> int:
    warmup_rounds = int(required_pipeline_warmup_rounds(strategy_cfg=strategy_cfg))
    if warmup_rounds <= 0:
        raise InvariantError("pipeline_warmup_rounds_nonpositive")
    return max(2, int(warmup_rounds))


def required_klines_window_for_closed_cache(
    *,
    closed_cache: RollingClosedRoundsCache,
    cutoff_seconds: int,
) -> tuple[int, int]:
    return _required_klines_window(
        closed_cache=closed_cache,
        cutoff_seconds=int(cutoff_seconds),
    )


def _run_one_iteration(cfg: RuntimeConfig, closed: _ClosedState) -> None:
    # Alignment + cutoff anchoring can be noisy around epoch shifts. Ensure we only
    # take an action using a coherent epoch snapshot.
    while True:
        # Step 1: Epoch alignment handshake (shift-aware) with retries.
        locked_round, _open_round, current_epoch = _epoch_handshake(cfg, closed)
        locked_epoch = int(locked_round.epoch)

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
                dry=bool(cfg.dry),
                cursor_path=str(cfg.runtime_state_paths.claim_scan_cursor_path),
                locked_epoch=int(locked_epoch),
                current_epoch=int(current_epoch),
                now_ts=int(now_ts()),
                buffer_seconds=int(cfg.buffer_seconds),
                get_close_ts=closed.cache.get_close_ts,
                page_size=100,
                gas_limit=int(GAS_LIMIT_CLAIM),
                claim_batch_size=int(_CLAIM_BATCH_SIZE),
                min_bet_with_gas_bnb=float(cfg.min_bet_amount_bnb) + float(GAS_COST_BET_BNB),
            )

            _dry_settle_available_bets(cfg, closed)
            closed.claim_scan_initialized = True

        # Step 3: Update strategy pipeline state from newly-closed rounds.
        if closed.strategy_pipeline is None:
            raise InvariantError("strategy_pipeline_missing")
        closed.strategy_pipeline.refresh_klines(klines=list(closed.klines_cache.klines))
        closed.strategy_pipeline.settle_closed_rounds(rounds=list(closed.cache.rounds))
        _maybe_persist_runtime_pipeline_snapshot(cfg=cfg, closed=closed)

        # Step 4: Fetch lock_ts(t) (RPC) for open epoch.
        lock_ts_t = int(cfg.contract.lock_ts(int(current_epoch)))
        if lock_ts_t <= 0:
            raise InvariantError("lock_ts_t_invalid")

        # Step 5: cutoff_ts(t) = lock_ts(t) - cutoff_seconds.
        cutoff_ts_t = int(lock_ts_t) - int(cfg.cutoff_seconds)

        # If we missed the previous epoch's cutoff and are now targeting a newer epoch, the
        # just-closed locked epoch may become claimable before the next cutoff. In that case,
        # we must wake for claim first (no approximation).
        prev_locked_epoch = int(locked_round.epoch) - 1
        claim_ts = int(locked_round.lock_at) + int(cfg.buffer_seconds) + int(_CLAIM_CHECK_PADDING_SECONDS)
        if now_ts() < claim_ts < cutoff_ts_t:
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=prev_locked_epoch)
            return

        # Step 6: Sleep until cutoff_ts(t).
        _sleep_until_ts(int(cutoff_ts_t), reason="wait_for_cutoff", epoch=int(current_epoch))

        # Step 6b: Refresh alignment immediately after waking; if the epoch shifted,
        # re-anchor before taking any action.
        locked_round2, _open_round2, current_epoch2 = _epoch_handshake(cfg, closed)
        if int(current_epoch2) != int(current_epoch):
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=int(current_epoch),
                locked_epoch=int(locked_epoch),
                lock_ts=int(lock_ts_t),
                cutoff_ts=int(cutoff_ts_t),
                locked_price_bnbusd=float(bnbusd_price),
                action="SKIP",
                decision_stage="reanchor",
                open_round=None,
                bankroll_before_action_bnb=closed.simulated_bankroll_bnb,
                bankroll_after_action_bnb=closed.simulated_bankroll_bnb,
                skip_reason=f"epoch_shift_before_decision:new_epoch={int(current_epoch2)}",
            )
            info(
                "RUN",
                "ACT",
                "SKIP",
                msg=(
                    f"Skip epoch {int(current_epoch)}: "
                    f"epoch_shift_before_decision:new_epoch={int(current_epoch2)}"
                ),
            )
            continue

        locked_round = locked_round2
        current_epoch = int(current_epoch2)
        locked_epoch = int(locked_round.epoch)

        # lock_ts can drift slightly; re-read for downstream timing guards.
        lock_ts_t = int(cfg.contract.lock_ts(int(current_epoch)))
        if lock_ts_t <= 0:
            raise InvariantError("lock_ts_t_invalid")

        # Step 7: Fetch open round (Graph) by epoch.
        # Note that during epoch shifts, Graph may temporarily return 0 or >1 open rounds.
        # This is treated as retryable alignment noise with bounded backoff.
        open_round: Round | None = None
        retries_remaining = len(_BACKOFF_SECONDS)
        while True:
            try:
                open_round = cfg.graph_client.fetch_open_round(int(current_epoch))
                open_round = _with_lock_at(open_round, int(lock_ts_t))
                break
            except InvariantError:
                if retries_remaining <= 0:
                    raise
                warn(
                    "CORE",
                    "LOOP",
                    "RETRY",
                    reason="graph_round_missing_or_ambiguous",
                    check="open_round",
                    retries_remaining=int(retries_remaining),
                )
                dur = int(_BACKOFF_SECONDS[len(_BACKOFF_SECONDS) - retries_remaining])
                sleep_seconds(dur)

                # If alignment shifted while we waited, restart anchoring.
                locked_round3, _open_round3, current_epoch3 = _epoch_handshake(cfg, closed)
                if int(current_epoch3) != int(current_epoch):
                    current_epoch = int(current_epoch3)
                    locked_round = locked_round3
                    break

                retries_remaining -= 1

        if open_round is None:
            # Alignment moved; restart anchoring.
            continue

        open_round, event_skip = _replace_open_round_bets_from_events(
            cfg=cfg,
            open_round=open_round,
            cutoff_ts=int(cutoff_ts_t),
        )
        if event_skip is not None:
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=int(current_epoch),
                locked_epoch=int(locked_epoch),
                lock_ts=int(lock_ts_t),
                cutoff_ts=int(cutoff_ts_t),
                locked_price_bnbusd=float(bnbusd_price),
                action="SKIP",
                decision_stage="event_gate",
                open_round=open_round,
                bankroll_before_action_bnb=closed.simulated_bankroll_bnb,
                bankroll_after_action_bnb=closed.simulated_bankroll_bnb,
                skip_reason=str(event_skip),
            )
            info("RUN", "ACT", "SKIP", msg=f"Skip epoch {int(current_epoch)}: {str(event_skip)}")
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return
        if open_round is None:
            raise InvariantError("event_bets_open_round_missing")

        # Step 8: Build context guards before deciding.
        k = int(max_required_prior_context_rounds_size())
        if int(k) <= 0:
            prior_context_rounds: list[Round] = []
        else:
            if locked_round is None:
                raise InvariantError("locked_round_missing")
            prior_context_prefix_needed = max(0, int(k) - 1)
            prior_context_prefix = (
                []
                if prior_context_prefix_needed == 0
                else list(closed.cache.rounds[-prior_context_prefix_needed:])
            )
            prior_context_rounds = list(prior_context_prefix) + [locked_round]

        if len(prior_context_rounds) != int(k):
            raise InvariantError(
                f"prior_context_rounds_len_mismatch: got={len(prior_context_rounds)} expected={int(k)}"
            )

        rounds_ok, rounds_reason = check_rounds_contiguous(
            prior_context_rounds=prior_context_rounds,
            target_round=open_round,
            buffer_seconds=int(cfg.buffer_seconds),
        )
        if not rounds_ok:
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=int(current_epoch),
                locked_epoch=int(locked_epoch),
                lock_ts=int(lock_ts_t),
                cutoff_ts=int(cutoff_ts_t),
                locked_price_bnbusd=float(bnbusd_price),
                action="SKIP",
                decision_stage="round_guard",
                open_round=open_round,
                bankroll_before_action_bnb=closed.simulated_bankroll_bnb,
                bankroll_after_action_bnb=closed.simulated_bankroll_bnb,
                skip_reason=str(rounds_reason),
            )
            info("RUN", "ACT", "SKIP", msg=f"Skip epoch {int(current_epoch)}: {rounds_reason}")
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        context_klines = _build_context_klines(
            klines_cache=closed.klines_cache,
            target_round=open_round,
            cutoff_seconds=int(cfg.cutoff_seconds),
        )
        klines_ok, klines_reason = check_klines_contiguous(context_klines=context_klines)
        if not klines_ok:
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=int(current_epoch),
                locked_epoch=int(locked_epoch),
                lock_ts=int(lock_ts_t),
                cutoff_ts=int(cutoff_ts_t),
                locked_price_bnbusd=float(bnbusd_price),
                action="SKIP",
                decision_stage="kline_guard",
                open_round=open_round,
                bankroll_before_action_bnb=closed.simulated_bankroll_bnb,
                bankroll_after_action_bnb=closed.simulated_bankroll_bnb,
                skip_reason=str(klines_reason),
            )
            info("RUN", "ACT", "SKIP", msg=f"Skip epoch {int(current_epoch)}: {klines_reason}")
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

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
            bankroll_bnb=float(bankroll_bnb),
            allow_oracle_mode=False,
        )
        if decision.p_bull is not None:
            pred_p_final = float(decision.p_bull)
        t_decision_ready_ms = _mono_ms()

        if decision.action != "BET":
            reason = str(decision.skip_reason or "")
            if reason == "":
                raise InvariantError("policy_skip_missing_reason")

            _log_controller_decision(
                current_epoch=int(current_epoch),
                decision=decision,
                final_action="SKIP",
                final_skip_reason=str(reason),
            )
            _log_direct_action_decision(
                current_epoch=int(current_epoch),
                decision=decision,
                final_action="SKIP",
                final_skip_reason=str(reason),
            )
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=int(current_epoch),
                locked_epoch=int(locked_epoch),
                lock_ts=int(lock_ts_t),
                cutoff_ts=int(cutoff_ts_t),
                locked_price_bnbusd=float(bnbusd_price),
                action="SKIP",
                decision_stage="pipeline",
                open_round=open_round,
                bankroll_before_action_bnb=float(bankroll_bnb),
                bankroll_after_action_bnb=float(bankroll_bnb),
                decision=decision,
                skip_reason=str(reason),
                decision_latency_ms=float(t_decision_ready_ms) - float(t_features_start_ms),
            )
            info("RUN", "ACT", "SKIP", msg=f"Skip epoch {int(current_epoch)}: {reason}")
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 11: Execution timing guard.
        if now_ts() >= lock_ts_t - _LOCK_SAFETY_MARGIN_SECONDS:
            _log_controller_decision(
                current_epoch=int(current_epoch),
                decision=decision,
                final_action="SKIP",
                final_skip_reason="too_close_to_lock_for_bet",
            )
            _log_direct_action_decision(
                current_epoch=int(current_epoch),
                decision=decision,
                final_action="SKIP",
                final_skip_reason="too_close_to_lock_for_bet",
            )
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=int(current_epoch),
                locked_epoch=int(locked_epoch),
                lock_ts=int(lock_ts_t),
                cutoff_ts=int(cutoff_ts_t),
                locked_price_bnbusd=float(bnbusd_price),
                action="SKIP",
                decision_stage="timing_guard",
                open_round=open_round,
                bankroll_before_action_bnb=float(bankroll_bnb),
                bankroll_after_action_bnb=float(bankroll_bnb),
                decision=decision,
                skip_reason="too_close_to_lock_for_bet",
                decision_latency_ms=float(t_decision_ready_ms) - float(t_features_start_ms),
            )
            info(
                "RUN",
                "ACT",
                "SKIP",
                msg=f"Skip epoch {int(current_epoch)}: too_close_to_lock_for_bet",
            )
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 12: Submit bet.
        amount_wei = int(round(decision.bet_size_bnb * BNB_WEI))
        if amount_wei <= 0:
            raise InvariantError("bet_amount_wei_nonpositive")

        _log_controller_decision(
            current_epoch=int(current_epoch),
            decision=decision,
            final_action="BET",
            final_skip_reason=None,
        )
        _log_direct_action_decision(
            current_epoch=int(current_epoch),
            decision=decision,
            final_action="BET",
            final_skip_reason=None,
        )
        tx_submit = None
        if not cfg.dry:
            gas_price_wei = cfg.contract.suggest_gas_price_wei()
            if decision.bet_side == "Bull":
                tx_submit = cfg.contract.bet_bull_timed(
                    epoch=int(current_epoch),
                    amount_wei=int(amount_wei),
                    gas_limit=int(GAS_LIMIT_BET),
                    gas_price_wei=int(gas_price_wei),
                    wait_receipt=bool(cfg.wait_for_bet_receipt),
                    receipt_timeout_seconds=int(cfg.bet_receipt_timeout_seconds),
                )
            elif decision.bet_side == "Bear":
                tx_submit = cfg.contract.bet_bear_timed(
                    epoch=int(current_epoch),
                    amount_wei=int(amount_wei),
                    gas_limit=int(GAS_LIMIT_BET),
                    gas_price_wei=int(gas_price_wei),
                    wait_receipt=bool(cfg.wait_for_bet_receipt),
                    receipt_timeout_seconds=int(cfg.bet_receipt_timeout_seconds),
                )
            else:
                raise InvariantError(f"unexpected_bet_side: {decision.bet_side}")

        # Step 13: Log bet with USD (BNB + USD suffixes).
        amount_bnb = float(amount_wei) / float(BNB_WEI)

        if not cfg.dry:
            bankroll_after_live = cfg.contract.wallet_balance_bnb(cfg.wallet_address)
            info(
                "RUN",
                "ACT",
                "BET",
                msg=(
                    f"Betting {float(amount_bnb):.4f} BNB"
                    + usd_suffix(amount_bnb=float(amount_bnb), bnbusd_price=bnbusd_price)
                    + f" on {str(decision.bet_side)} for epoch {int(current_epoch)}"
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
                "epoch": int(current_epoch),
                "cutoff_ts": int(cutoff_ts_t),
                "t_features_start_mono_ms": float(t_features_start_ms),
                "t_decision_ready_mono_ms": float(t_decision_ready_ms),
                "t_tx_signed_mono_ms": float(tx_submit.t_tx_signed_mono_ms),
                "t_tx_hash_received_mono_ms": float(tx_submit.t_tx_hash_received_mono_ms),
                "t_receipt_confirmed_mono_ms": receipt_confirmed_ms,
                "tx_hash": str(tx_submit.tx_hash),
                "tx_included_block_number": int(tx_submit.included_block_number)
                if tx_submit.included_block_number is not None
                else None,
                "tx_included_block_timestamp": int(tx_submit.included_block_timestamp)
                if tx_submit.included_block_timestamp is not None
                else None,
                "latency_features_ms": float(t_decision_ready_ms) - float(t_features_start_ms),
                "latency_sign_ms": float(tx_submit.t_tx_signed_mono_ms) - float(t_decision_ready_ms),
                "latency_broadcast_ms": float(tx_submit.t_tx_hash_received_mono_ms) - float(tx_submit.t_tx_signed_mono_ms),
                "latency_mempool_ms": (
                    float(receipt_confirmed_ms) - float(tx_submit.t_tx_hash_received_mono_ms)
                    if receipt_confirmed_ms is not None
                    else None
                ),
                "latency_e2e_ms": (
                    float(receipt_confirmed_ms) - float(t_features_start_ms)
                    if receipt_confirmed_ms is not None
                    else None
                ),
            }
            _append_jsonl(str(cfg.latency_log_path), latency_record)
        else:
            # Step 14: Dry bookkeeping (including gas proxy) + record.
            if closed.simulated_bankroll_bnb is None:
                raise InvariantError("dry_bankroll_uninitialized")

            bankroll_before_bet = closed.simulated_bankroll_bnb
            closed.simulated_bankroll_bnb -= float(amount_bnb) + float(GAS_COST_BET_BNB)
            bankroll_after_bet = closed.simulated_bankroll_bnb

            info(
                "RUN",
                "ACT",
                "BET",
                msg=(
                    f"Betting {float(amount_bnb):.4f} BNB"
                    + usd_suffix(amount_bnb=float(amount_bnb), bnbusd_price=bnbusd_price)
                    + f" on {str(decision.bet_side)} for epoch {int(current_epoch)}"
                    + bankroll_suffix(bankroll_bnb=bankroll_after_bet, bnbusd_price=bnbusd_price)
                ),
            )
            _dry_record_bet(
                cfg,
                closed,
                epoch=int(current_epoch),
                side=str(decision.bet_side),
                amount_bnb=float(amount_bnb),
                p_final=float(pred_p_final),
                expected_profit_bnb=float(decision.expected_profit_bnb),
                bankroll_before_bet_bnb=float(bankroll_before_bet),
                bankroll_after_bet_bnb=float(bankroll_after_bet),
            )
            _record_dry_cycle_audit(
                cfg,
                closed,
                current_epoch=int(current_epoch),
                locked_epoch=int(locked_epoch),
                lock_ts=int(lock_ts_t),
                cutoff_ts=int(cutoff_ts_t),
                locked_price_bnbusd=float(bnbusd_price),
                action="BET",
                decision_stage="pipeline",
                open_round=open_round,
                bankroll_before_action_bnb=float(bankroll_before_bet),
                bankroll_after_action_bnb=float(bankroll_after_bet),
                decision=decision,
                decision_latency_ms=float(t_decision_ready_ms) - float(t_features_start_ms),
            )

        # Step 15: Sleep until claim + claim scan.
        _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
        return


def _append_newly_closed_rounds(cfg: RuntimeConfig, closed: _ClosedState) -> int:
    """Append forward from disk_latest_epoch to latest usable closed epoch (Graph).

    No disk reads: uses `closed.disk_latest_epoch` as the authoritative latest-on-disk.
    """
    latest_usable_closed_epoch = int(cfg.graph_client.fetch_latest_usable_closed_epoch())
    if latest_usable_closed_epoch <= int(closed.disk_latest_epoch):
        return int(closed.disk_latest_epoch)

    start_epoch = int(closed.disk_latest_epoch) + 1
    end_epoch = int(latest_usable_closed_epoch)

    span = int(end_epoch) - int(start_epoch) + 1
    if span <= 0:
        return int(closed.disk_latest_epoch)

    prev_epoch = int(closed.disk_latest_epoch)
    earliest_fetched_round: Round | None = None
    latest_fetched_round: Round | None = None

    page_size = 1000
    window_start = int(start_epoch)
    while window_start <= int(end_epoch):
        window_end = int(window_start) + int(page_size) - 1
        if window_end > int(end_epoch):
            window_end = int(end_epoch)

        page = cfg.graph_client.fetch_closed_rounds(
            order="asc",
            epoch_gte=int(window_start),
            epoch_lte=int(window_end),
            first=int(page_size),
            skip=0,
        )

        if page:
            prev_in_page: int | None = None
            for idx, r in enumerate(page):
                e = int(r.epoch)
                if e < int(window_start) or e > int(window_end):
                    raise InvariantError(
                        f"closed_rounds_epoch_out_of_requested_bounds: idx={idx} got={e} range=[{int(window_start)}..{int(window_end)}]"
                    )
                if prev_in_page is not None and e <= int(prev_in_page):
                    raise InvariantError(f"closed_rounds_not_strictly_increasing: idx={idx} got={e} prev={prev_in_page}")
                prev_in_page = e

            filtered = [r for r in page if int(r.epoch) > int(prev_epoch)]
            if filtered:
                prev_epoch = int(cfg.round_store.append_rounds_after(int(prev_epoch), filtered))
                closed.cache.extend(list(filtered))
                if earliest_fetched_round is None:
                    earliest_fetched_round = filtered[0]
                latest_fetched_round = filtered[-1]

        window_start = int(window_end) + 1

    if earliest_fetched_round is not None and latest_fetched_round is not None:
        _append_newly_closed_klines(
            cfg=cfg,
            closed=closed,
            earliest_round=earliest_fetched_round,
            latest_round=latest_fetched_round,
        )

    return int(prev_epoch)


def _last_closed_kline_open_ms(*, cutoff_ts: int) -> int:
    cutoff_ms = int(cutoff_ts) * 1000
    minute_open_ms = (int(cutoff_ms) // int(_ONE_MINUTE_MS)) * int(_ONE_MINUTE_MS)
    return int(minute_open_ms) - int(_ONE_MINUTE_MS)


def _required_klines_window(
    *,
    closed_cache: RollingClosedRoundsCache,
    cutoff_seconds: int,
) -> tuple[int, int]:
    if not closed_cache.rounds:
        raise InvariantError("closed_round_cache_empty")

    return _required_klines_window_for_round_bounds(
        earliest_round=closed_cache.rounds[0],
        latest_round=closed_cache.rounds[-1],
        cutoff_seconds=int(cutoff_seconds),
        warmup_margin_minutes=int(_KLINE_WARMUP_MARGIN_MINUTES),
    )


def _required_klines_window_for_round_bounds(
    *,
    earliest_round: Round,
    latest_round: Round,
    cutoff_seconds: int,
    warmup_margin_minutes: int,
) -> tuple[int, int]:
    if earliest_round.lock_at is None or latest_round.lock_at is None:
        raise InvariantError("closed_round_missing_lock_at")
    if int(earliest_round.epoch) > int(latest_round.epoch):
        raise InvariantError("round_bounds_epoch_invalid")
    if int(warmup_margin_minutes) < 0:
        raise InvariantError("kline_warmup_margin_negative")

    kk = int(max_required_context_klines_size())
    if kk <= 0:
        raise InvariantError("context_klines_size_invalid")

    earliest_cutoff_ts = int(earliest_round.lock_at) - int(cutoff_seconds)
    latest_cutoff_ts = int(latest_round.lock_at) - int(cutoff_seconds)

    earliest_anchor_open_ms = _last_closed_kline_open_ms(cutoff_ts=int(earliest_cutoff_ts))
    latest_anchor_open_ms = _last_closed_kline_open_ms(cutoff_ts=int(latest_cutoff_ts))

    warmup_minutes = int(kk - 1) + int(warmup_margin_minutes)
    start_open_ms = int(earliest_anchor_open_ms) - int(warmup_minutes) * int(_ONE_MINUTE_MS)
    if start_open_ms < 0:
        start_open_ms = 0

    end_open_ms = int(latest_anchor_open_ms) + int(_ONE_MINUTE_MS)
    if int(end_open_ms) <= int(start_open_ms):
        raise InvariantError("required_klines_window_invalid")
    return int(start_open_ms), int(end_open_ms)


def _init_klines_cache(cfg: RuntimeConfig, *, closed_cache: RollingClosedRoundsCache) -> RollingKlinesCache:
    start_open_ms, end_open_ms = _required_klines_window(
        closed_cache=closed_cache,
        cutoff_seconds=int(cfg.cutoff_seconds),
    )

    ensure_klines_coverage(
        client=cfg.binance_us_client,
        store=cfg.klines_store,
        symbol=cfg.binance_us_symbol,
        start_open_time_ms=int(start_open_ms),
        end_open_time_ms=int(end_open_ms),
    )

    klines = cfg.klines_store.get_klines_between(
        start_open_time_ms=int(start_open_ms),
        end_open_time_ms=int(end_open_ms),
    )
    if not klines:
        raise InvariantError("klines_cache_init_empty")

    return RollingKlinesCache(klines=list(klines), capacity=len(klines))


def _build_strategy_pipeline(*, cfg: RuntimeConfig, klines_cache: RollingKlinesCache) -> StrategyPipeline:
    """Build shared strategy pipeline from runtime config."""

    needs_pool_projection_model = any(
        str(c.pool_total_gate_mode) == "projected_final_model_only"
        or str(c.stake_mode) in ("ev_scaled_projected", "ev_optimal_projected")
        or bool(c.late_model_veto_enabled)
        for c in cfg.strategy_cfg.dislocation.candidates
    )
    ml_adapter: MlCandidateAdapter | None = None
    if bool(cfg.strategy_cfg.ml_candidate.enabled) or bool(needs_pool_projection_model):
        ml_adapter = MlCandidateAdapter(
            config=cfg.strategy_cfg.ml_candidate,
            cutoff_seconds=int(cfg.cutoff_seconds),
            treasury_fee_fraction=float(cfg.treasury_fee_fraction),
            klines_store_like=klines_cache,
            feature_cache_store=cfg.feature_cache_store,
            projection_cache_store=cfg.projection_cache_store,
        )
    flow_adapter: FlowCandidateAdapter | None = None
    if bool(cfg.strategy_cfg.flow_candidate.enabled):
        flow_adapter = FlowCandidateAdapter(
            config=cfg.strategy_cfg.flow_candidate,
            cutoff_seconds=int(cfg.cutoff_seconds),
            treasury_fee_fraction=float(cfg.treasury_fee_fraction),
        )

    dislocation_engine = build_dislocation_engine_from_config(
        selector_cfg=cfg.strategy_cfg.dislocation.selector,
        candidate_cfgs=cfg.strategy_cfg.dislocation.candidates,
        treasury_fee_fraction=float(cfg.treasury_fee_fraction),
        cutoff_seconds=int(cfg.cutoff_seconds),
        projected_pool_provider=ml_adapter,
    )

    router_cfg = StrategyRouterConfig(
        mode=str(cfg.strategy_cfg.router.mode),
        score_threshold_bnb=float(cfg.strategy_cfg.router.score_threshold_bnb),
        online_warmup_rounds=int(cfg.strategy_cfg.router.online_warmup_rounds),
        online_num_quantile_bins=int(cfg.strategy_cfg.router.online_num_quantile_bins),
        online_min_cell_obs=int(cfg.strategy_cfg.router.online_min_cell_obs),
        online_score_threshold_bnb=float(cfg.strategy_cfg.router.online_score_threshold_bnb),
        online_use_direction_split=bool(cfg.strategy_cfg.router.online_use_direction_split),
    )
    router = StrategyRouter(config=router_cfg)

    pipeline = StrategyPipeline(
        dislocation_engine=dislocation_engine,
        router=router,
        treasury_fee_fraction=float(cfg.treasury_fee_fraction),
        ml_candidate_adapter=ml_adapter,
        flow_candidate_adapter=flow_adapter,
        direct_action_policy=(
            DirectActionPolicy(
                cutoff_seconds=int(cfg.cutoff_seconds),
                treasury_fee_fraction=float(cfg.treasury_fee_fraction),
                klines_store_like=klines_cache,
                feature_cache_store=cfg.feature_cache_store,
                model_bundle_path=str(cfg.strategy_cfg.direct_action_policy.model_bundle_path),
            )
            if bool(cfg.strategy_cfg.direct_action_policy.enabled)
            else None
        ),
        window_controller=(
            WindowController(config=cfg.strategy_cfg.window_controller)
            if bool(cfg.strategy_cfg.window_controller.enabled)
            else None
        ),
    )
    pipeline.refresh_klines(klines=list(klines_cache.klines))
    return pipeline


def _append_newly_closed_klines(
    cfg: RuntimeConfig,
    closed: _ClosedState,
    *,
    earliest_round: Round,
    latest_round: Round,
) -> None:
    start_open_ms, end_open_ms = _required_klines_window_for_round_bounds(
        earliest_round=earliest_round,
        latest_round=latest_round,
        cutoff_seconds=int(cfg.cutoff_seconds),
        warmup_margin_minutes=int(_KLINE_WARMUP_MARGIN_MINUTES),
    )

    ensure_klines_coverage(
        client=cfg.binance_us_client,
        store=cfg.klines_store,
        symbol=cfg.binance_us_symbol,
        start_open_time_ms=int(start_open_ms),
        end_open_time_ms=int(end_open_ms),
    )

    latest_open = closed.klines_cache.latest_open_time_ms()
    if latest_open is None:
        raise InvariantError("klines_cache_empty")
    next_open = max(int(start_open_ms), int(latest_open) + int(_ONE_MINUTE_MS))
    if int(next_open) >= int(end_open_ms):
        return

    new_klines = cfg.klines_store.get_klines_between(
        start_open_time_ms=int(next_open),
        end_open_time_ms=int(end_open_ms),
    )
    if new_klines:
        closed.klines_cache.extend(list(new_klines))


def _epoch_handshake(cfg: RuntimeConfig, closed: _ClosedState) -> tuple[Round, Round, int]:
    """Epoch alignment handshake (shift-aware) with bounded exponential backoff.

    Returns:
      locked_round, open_round, current_epoch (RPC)
    """
    def _error(*, reason: str, check: str, **fields: object) -> NoReturn:
        error(
            "CORE",
            "LOOP",
            "ALIGN",
            err="InvariantError",
            reason=str(reason),
            check=str(check),
            retries_remaining=0,
            **fields,
        )
        raise TransientGraphError(f"epoch_alignment_failed: reason={reason} check={check} fields={fields}")

    retries_remaining = len(_BACKOFF_SECONDS)

    while retries_remaining >= 0:
        closed.disk_latest_epoch = _append_newly_closed_rounds(cfg, closed)
        closed_epoch = int(closed.cache.latest_epoch)

        try:
            locked_round = cfg.graph_client.fetch_latest_locked_round()
            open_round = cfg.graph_client.fetch_latest_open_round()
        except InvariantError as e:
            msg = str(e)
            if retries_remaining <= 0:
                _error(reason="graph_rounds_missing_or_ambiguous", check="graph_fetch_locked_and_open", err=msg)
            warn(
                "CORE",
                "LOOP",
                "RETRY",
                reason="graph_rounds_missing_or_ambiguous",
                check="graph_fetch_locked_and_open",
                retries_remaining=int(retries_remaining),
            )
            dur = int(_BACKOFF_SECONDS[len(_BACKOFF_SECONDS) - retries_remaining])
            sleep_seconds(dur)
            retries_remaining -= 1
            continue

        locked_epoch = int(locked_round.epoch)
        open_epoch = int(open_round.epoch)
        current_epoch = int(cfg.contract.current_epoch())

        # A) Closed vs Locked
        if locked_epoch > closed_epoch + 1:
            if retries_remaining <= 0:
                _error(
                    reason="epoch_shift_persisted",
                    check="closed_vs_locked",
                    closed=int(closed_epoch),
                    locked=int(locked_epoch),
                )
            warn(
                "CORE",
                "LOOP",
                "RETRY",
                reason="epoch_shift",
                check="closed_vs_locked",
                retries_remaining=int(retries_remaining),
                closed=int(closed_epoch),
                locked=int(locked_epoch),
            )
            dur = int(_BACKOFF_SECONDS[len(_BACKOFF_SECONDS) - retries_remaining])
            sleep_seconds(dur)
            retries_remaining -= 1
            continue

        if locked_epoch <= closed_epoch:
            if retries_remaining <= 0:
                _error(
                    reason="locked_not_after_closed",
                    check="closed_vs_locked",
                    closed=int(closed_epoch),
                    locked=int(locked_epoch),
                )
            warn(
                "CORE",
                "LOOP",
                "RETRY",
                reason="locked_not_after_closed",
                check="closed_vs_locked",
                retries_remaining=int(retries_remaining),
                closed=int(closed_epoch),
                locked=int(locked_epoch),
            )
            dur = int(_BACKOFF_SECONDS[len(_BACKOFF_SECONDS) - retries_remaining])
            sleep_seconds(dur)
            retries_remaining -= 1
            continue

        # B) Locked vs Open (Graph)
        if open_epoch > locked_epoch + 1:
            if retries_remaining <= 0:
                _error(
                    reason="epoch_shift_persisted",
                    check="locked_vs_open",
                    closed=int(closed_epoch),
                    locked=int(locked_epoch),
                    open=int(open_epoch),
                )
            warn(
                "CORE",
                "LOOP",
                "RETRY",
                reason="epoch_shift",
                check="locked_vs_open",
                retries_remaining=int(retries_remaining),
                closed=int(closed_epoch),
                locked=int(locked_epoch),
                open=int(open_epoch),
            )
            dur = int(_BACKOFF_SECONDS[len(_BACKOFF_SECONDS) - retries_remaining])
            sleep_seconds(dur)
            retries_remaining -= 1
            continue

        if open_epoch <= locked_epoch:
            if retries_remaining <= 0:
                _error(
                    reason="open_not_after_locked",
                    check="locked_vs_open",
                    closed=int(closed_epoch),
                    locked=int(locked_epoch),
                    open=int(open_epoch),
                )
            warn(
                "CORE",
                "LOOP",
                "RETRY",
                reason="open_not_after_locked",
                check="locked_vs_open",
                retries_remaining=int(retries_remaining),
                closed=int(closed_epoch),
                locked=int(locked_epoch),
                open=int(open_epoch),
            )
            dur = int(_BACKOFF_SECONDS[len(_BACKOFF_SECONDS) - retries_remaining])
            sleep_seconds(dur)
            retries_remaining -= 1
            continue

        # C) Open (Graph) vs RPC
        if open_epoch == current_epoch:
            return locked_round, open_round, int(current_epoch)

        if open_epoch > current_epoch:
            if retries_remaining <= 0:
                _error(
                    reason="rpc_behind_open_persisted",
                    check="open_vs_rpc",
                    closed=int(closed_epoch),
                    locked=int(locked_epoch),
                    open=int(open_epoch),
                    rpc=int(current_epoch),
                )
            warn(
                "CORE",
                "LOOP",
                "RETRY",
                reason="rpc_behind_open",
                check="open_vs_rpc",
                retries_remaining=int(retries_remaining),
                closed=int(closed_epoch),
                locked=int(locked_epoch),
                open=int(open_epoch),
                rpc=int(current_epoch),
            )
            dur = int(_BACKOFF_SECONDS[len(_BACKOFF_SECONDS) - retries_remaining])
            sleep_seconds(dur)
            retries_remaining -= 1
            continue

        if open_epoch < current_epoch:
            if retries_remaining <= 0:
                _error(
                    reason="rpc_ahead_of_open_persisted",
                    check="open_vs_rpc",
                    closed=int(closed_epoch),
                    locked=int(locked_epoch),
                    open=int(open_epoch),
                    rpc=int(current_epoch),
                )
            warn(
                "CORE",
                "LOOP",
                "RETRY",
                reason="rpc_ahead_of_open",
                check="open_vs_rpc",
                retries_remaining=int(retries_remaining),
                closed=int(closed_epoch),
                locked=int(locked_epoch),
                open=int(open_epoch),
                rpc=int(current_epoch),
            )
            dur = int(_BACKOFF_SECONDS[len(_BACKOFF_SECONDS) - retries_remaining])
            sleep_seconds(dur)
            retries_remaining -= 1
            continue

        raise InvariantError("epoch_handshake_unreachable")

    raise InvariantError("epoch_handshake_exhausted")


def _sleep_and_claim(cfg: RuntimeConfig, closed: _ClosedState, claim_epoch: int) -> None:
    close_ts = int(cfg.contract.close_ts(int(claim_epoch)))
    if close_ts <= 0:
        raise InvariantError("close_ts_invalid")

    claim_ts = int(close_ts) + int(cfg.buffer_seconds) + int(_CLAIM_CHECK_PADDING_SECONDS)
    _sleep_until_ts(int(claim_ts), reason="wait_for_claim", epoch=int(claim_epoch))

    # Refresh epochs after sleeping so the prior locked round can become closed.
    locked_round2, _open_round2, current_epoch2 = _epoch_handshake(cfg, closed)
    locked_epoch2 = int(locked_round2.epoch)

    claim_scan_cursor(
        contract=cfg.contract,
        wallet_address=cfg.wallet_address,
        dry=bool(cfg.dry),
        cursor_path=str(cfg.runtime_state_paths.claim_scan_cursor_path),
        locked_epoch=int(locked_epoch2),
        current_epoch=int(current_epoch2),
        now_ts=int(now_ts()),
        buffer_seconds=int(cfg.buffer_seconds),
        get_close_ts=closed.cache.get_close_ts,
        page_size=100,
        gas_limit=int(GAS_LIMIT_CLAIM),
        claim_batch_size=int(_CLAIM_BATCH_SIZE),
        min_bet_with_gas_bnb=float(cfg.min_bet_amount_bnb) + float(GAS_COST_BET_BNB),
    )

    _dry_settle_available_bets(cfg, closed)


def _sleep_until_ts(target_ts: int, *, reason: str, epoch: int | None = None) -> None:
    now = now_ts()
    remaining = float(target_ts) - float(now)
    if remaining <= 0:
        return

    msg = f"Sleeping {int(remaining)}s ({reason})"
    if epoch is not None:
        msg = msg + f" epoch={int(epoch)}"
    info("RUN", "LOOP", "SLEEP", msg=msg)

    while True:
        now2 = now_ts()
        remaining2 = float(target_ts) - float(now2)
        if remaining2 <= 0:
            return
        sleep_seconds(min(1.0, remaining2))

