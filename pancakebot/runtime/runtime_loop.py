"""Live runtime loop.

Hard rules (project-level):
  - Do not catch-and-continue exceptions (developer errors must crash).
  - No disk reads in the main live loop: closed rounds are loaded once at startup
    and then maintained via an in-memory rolling cache.

This module orchestrates I/O (Graph, on-chain) and pure logic
(feature building, training, inference, policy).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from pancakebot.config.policy_config import PolicyConfig
from pancakebot.core.constants import (
    BNB_WEI,
    GAS_LIMIT_BET,
    GAS_LIMIT_CLAIM,
    GAS_COST_BET_BNB,
)
from pancakebot.domain.closed_rounds_cache import RollingClosedRoundsCache
from pancakebot.infra.closed_rounds_sync import sync_closed_rounds
from pancakebot.runtime.cache_policy import compute_required_cache_size
from pancakebot.infra.graph_client import GraphClient
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.klines_store import KlinesStore
from pancakebot.infra.binance_us_client import BinanceUsClient
from pancakebot.infra.klines_sync import ensure_klines_coverage
from pancakebot.domain.klines_cache import RollingKlinesCache
from pancakebot.domain.types import Kline, Round
from pancakebot.domain.features.schema import max_required_context_klines_size, max_required_prior_context_rounds_size
from pancakebot.domain.contiguity import check_klines_contiguous, check_rounds_contiguous
from pancakebot.infra.onchain.web3_prediction_contract import Web3PredictionContract
from pancakebot.domain.strategy.planner import build_inputs, predict, size_bet
from pancakebot.runtime.claim_manager import claim_scan_cursor
from pancakebot.runtime.model_manager import ModelManager
from pancakebot.runtime.contract_constants_cache import ContractConstants, save_contract_constants
from pancakebot.runtime.settlement import settle_bet_against_closed_round
from pancakebot.runtime.sleep import sleep_seconds
from pancakebot.core.errors import InvariantError, TransientGraphError, TransientRpcError
from pancakebot.core.logging import error, info, warn
from pancakebot.core.time import now_ts
from pancakebot.core.money import bankroll_suffix, format_bankroll, usd_suffix

_LOCK_SAFETY_MARGIN_SECONDS = 5  # locked

# Extra cushion added to the claim-check wake time to avoid alignment retries near Graph/RPC boundaries.
_CLAIM_CHECK_PADDING_SECONDS = 5

_CLAIM_CURSOR_PATH = "var/claim_scan_cursor.txt"  # locked
_DRY_BETS_PATH = "var/dry_bets.jsonl"
_DRY_SETTLED_PATH = "var/dry_settled_epochs.txt"
_DRY_AUDIT_TRADES_CSV = "var/dry_audit_trades.csv"
_BACKOFF_SECONDS = [2, 4, 8, 16, 32, 58]  # locked

_TRANSIENT_NETWORK_DELAY_SECONDS = 10
_ONE_MINUTE_MS = 60_000
_KLINE_WARMUP_MARGIN_MINUTES = 5


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

    # Policy config
    policy_cfg: PolicyConfig

    # Protocol constants (cached at startup)
    treasury_fee_fraction: float
    buffer_seconds: int
    min_bet_amount_bnb: float

    # Training window sizing
    train_size: int

    # Retrain cadence
    retrain_interval: int

    # Calibration window sizing (Step 2+). Step 1 wiring only.
    calibrate_size: int

    # Recalibrate cadence (Step 2+). Step 1 wiring only.
    recalibrate_interval: int

    # Deterministic recency weighting for walk-forward train/calibration rows.
    recency_weight_floor: float
    recency_weight_power: float

    # Model hyperparameters
    price_alpha: float
    pool_alpha_total: float
    pool_alpha_ratio: float
    random_seed: int

    # Execution
    dry: bool


@dataclass(slots=True)
class _ClosedState:
    cache: RollingClosedRoundsCache
    disk_latest_epoch: int
    klines_cache: RollingKlinesCache
    model_manager: ModelManager
    claim_scan_initialized: bool = False
    simulated_bankroll_bnb: float | None = None
    dry_bets_by_epoch: dict[int, dict[str, object]] | None = None
    dry_settled_epochs: set[int] | None = None


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
    closed_state = _init_closed_state(cfg)

    # After sync, USD conversion uses the latest closed round close_price.
    bnbusd_price = closed_state.cache.rounds[-1].close_price
    if cfg.dry:
        if closed_state.simulated_bankroll_bnb is None:
            raise InvariantError("dry_bankroll_uninitialized")
        bankroll_bnb = closed_state.simulated_bankroll_bnb
    else:
        bankroll_bnb = cfg.contract.wallet_balance_bnb(cfg.wallet_address)
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


def _load_dry_bets(path: str) -> dict[int, dict[str, object]]:
    bets: dict[int, dict[str, object]] = {}
    p = Path(path)
    if not p.exists():
        return bets
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        epoch = int(rec["epoch"])
        bets[epoch] = rec
    return bets


def _load_dry_settled_epochs(path: str) -> set[int]:
    p = Path(path)
    if not p.exists():
        return set()
    out: set[int] = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.add(int(line))
    return out


def _append_dry_settled_epoch(path: str, epoch: int) -> None:
    _ensure_parent_dir(path)
    with open(path, "a") as f:
        f.write(str(int(epoch)))
        f.write("\n")


def _dry_record_bet(
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
    rec = {
        "epoch": int(epoch),
        "placed_ts": int(now_ts()),
        "bet_side": str(side),
        "bet_bnb": float(amount_bnb),
        "p_final": float(p_final),
        "pred_win_probability": float(p_final),
        "expected_profit_bnb": float(expected_profit_bnb),
        "bankroll_before_bet_bnb": float(bankroll_before_bet_bnb),
        "bankroll_after_bet_bnb": float(bankroll_after_bet_bnb),
    }
    closed.dry_bets_by_epoch[int(epoch)] = rec
    _append_jsonl(str(_DRY_BETS_PATH), rec)


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
            _append_dry_settled_epoch(_DRY_SETTLED_PATH, e)
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
            _DRY_AUDIT_TRADES_CSV,
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
        _append_dry_settled_epoch(_DRY_SETTLED_PATH, e)


def _init_closed_state(cfg: RuntimeConfig) -> _ClosedState:
    """Startup-only closed rounds sync + in-memory cache load."""
    k = int(max_required_prior_context_rounds_size())
    train_size = int(cfg.train_size)
    calibrate_size = int(cfg.calibrate_size)
    retrain_interval = int(cfg.retrain_interval)
    recalibrate_interval = int(cfg.recalibrate_interval)

    if k <= 0:
        context_desc = "target_only"
    else:
        context_desc = f"prior_context_rounds[{int(k)}]"

    inference_closed_cache_needed = int(k) + int(train_size) + int(calibrate_size)

    info(
        "CORE",
        "RUN",
        "SETUP",
        msg=(
            f"Core setup: prior_context_rounds_required={int(k)} context={context_desc} "
            f"train_size={int(train_size)} calibrate_size={int(calibrate_size)} "
            f"retrain_interval={int(retrain_interval)} recalibrate_interval={int(recalibrate_interval)}"
            + (
                f" inference_closed_cache_needed={int(inference_closed_cache_needed)}"
                if inference_closed_cache_needed is not None
                else ""
            )
        ),
    )

    cache_n = compute_required_cache_size(
        train_size=train_size,
        calibrate_size=calibrate_size,
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

    required = compute_required_cache_size(
        train_size=train_size,
        calibrate_size=calibrate_size,
    )
    if len(cache.rounds) < required:
        raise InvariantError("insufficient_closed_rounds_for_training")

    disk_latest_epoch = int(cache.rounds[-1].epoch)

    # Persist contract constants after startup sync so backtests can run without any RPC.
    save_contract_constants(
        constants=ContractConstants(
            min_bet_amount_bnb=float(cfg.min_bet_amount_bnb),
            treasury_fee_fraction=float(cfg.treasury_fee_fraction),
            buffer_seconds=int(cfg.buffer_seconds),
        )
    )

    klines_cache = _init_klines_cache(cfg=cfg, closed_cache=cache)
    closed = _ClosedState(
        cache=cache,
        disk_latest_epoch=disk_latest_epoch,
        klines_cache=klines_cache,
        model_manager=ModelManager(),
    )

    if cfg.dry:
        closed.simulated_bankroll_bnb = cfg.contract.wallet_balance_bnb(cfg.wallet_address)
        closed.dry_bets_by_epoch = _load_dry_bets(_DRY_BETS_PATH)
        closed.dry_settled_epochs = _load_dry_settled_epochs(_DRY_SETTLED_PATH)
        _ensure_dry_audit_csv(_DRY_AUDIT_TRADES_CSV)

    return closed


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
                cursor_path=str(_CLAIM_CURSOR_PATH),
                locked_epoch=int(locked_epoch),
                current_epoch=int(current_epoch),
                now_ts=int(now_ts()),
                buffer_seconds=int(cfg.buffer_seconds),
                get_close_ts=closed.cache.get_close_ts,
                page_size=100,
                gas_limit=int(GAS_LIMIT_CLAIM),
            )

            _dry_settle_available_bets(cfg, closed)
            closed.claim_scan_initialized = True

        # Step 3: Train models and regime thresholds using the in-memory closed-rounds cache.
        wf_state = closed.model_manager.step(cfg=cfg, closed_rounds=closed.cache.rounds, current_epoch=int(current_epoch))

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

        # Step 8-10: Plan (build feats -> predict -> size) using the shared planner API.
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
            info("RUN", "ACT", "SKIP", msg=f"Skip epoch {int(current_epoch)}: {klines_reason}")
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        feats = build_inputs(
            cfg=cfg,
            prior_context_rounds=prior_context_rounds,
            context_klines=context_klines,
            target_round=open_round,
        )

        pred = predict(state=wf_state, feats=feats)

        if cfg.dry:
            if closed.simulated_bankroll_bnb is None:
                raise InvariantError("dry_bankroll_uninitialized")
            bankroll_bnb = closed.simulated_bankroll_bnb
        else:
            bankroll_bnb = cfg.contract.wallet_balance_bnb(cfg.wallet_address)

        decision = size_bet(cfg=cfg, pred=pred, bankroll_bnb=bankroll_bnb)

        if decision.action != "BET":
            reason = str(decision.skip_reason or "")
            if reason == "":
                raise InvariantError("policy_skip_missing_reason")

            allowed = {
                "no_positive_ev",
                "insufficient_bankroll_for_gas",
            }
            if reason not in allowed:
                raise InvariantError(f"unexpected_policy_skip_reason: {reason}")

            info("RUN", "ACT", "SKIP", msg=f"Skip epoch {int(current_epoch)}: {reason}")
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 11: Execution timing guard.
        if now_ts() >= lock_ts_t - _LOCK_SAFETY_MARGIN_SECONDS:
            info(
                "RUN",
                "ACT",
                "SKIP",
                msg=f"Skip epoch {int(current_epoch)}: too_close_to_lock_for_bet",
            )
            _sleep_and_claim(cfg=cfg, closed=closed, claim_epoch=locked_epoch)
            return

        # Step 12: Submit bet.
        amount_wei = int(round(decision.amount_bnb * BNB_WEI))
        if amount_wei <= 0:
            raise InvariantError("bet_amount_wei_nonpositive")

        if not cfg.dry:
            gas_price_wei = cfg.contract.suggest_gas_price_wei()
            if decision.bet_side == "Bull":
                cfg.contract.bet_bull(
                    epoch=int(current_epoch),
                    amount_wei=int(amount_wei),
                    gas_limit=int(GAS_LIMIT_BET),
                    gas_price_wei=int(gas_price_wei),
                )
            elif decision.bet_side == "Bear":
                cfg.contract.bet_bear(
                    epoch=int(current_epoch),
                    amount_wei=int(amount_wei),
                    gas_limit=int(GAS_LIMIT_BET),
                    gas_price_wei=int(gas_price_wei),
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
                closed,
                epoch=int(current_epoch),
                side=str(decision.bet_side),
                amount_bnb=float(amount_bnb),
                p_final=float(pred.p_final),
                expected_profit_bnb=float(decision.expected_profit_bnb),
                bankroll_before_bet_bnb=float(bankroll_before_bet),
                bankroll_after_bet_bnb=float(bankroll_after_bet),
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
        cursor_path=str(_CLAIM_CURSOR_PATH),
        locked_epoch=int(locked_epoch2),
        current_epoch=int(current_epoch2),
        now_ts=int(now_ts()),
        buffer_seconds=int(cfg.buffer_seconds),
        get_close_ts=closed.cache.get_close_ts,
        page_size=100,
        gas_limit=int(GAS_LIMIT_CLAIM),
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

