from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pancakebot.infra.onchain.web3_prediction_contract import Web3PredictionContract
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info

_PAGE_SIZE_DEFAULT = 100


@dataclass(frozen=True, slots=True)
class ClaimScanResult:
    scanned_n: int
    claimed_n: int


def _read_int_file(path: Path) -> int:
    if not path.exists():
        return 0
    raw = path.read_text().strip()
    if raw == "":
        return 0
    try:
        return int(raw)
    except Exception as e:
        raise InvariantError(f"claim_cursor_invalid: {path} value={raw!r}") from e


def _write_int_file_atomic(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(str(value))
    tmp.replace(path)


def claim_scan_cursor(
    *,
    contract: Web3PredictionContract,
    wallet_address: str,
    dry: bool,
    cursor_path: str,
    locked_epoch: int,
    current_epoch: int,
    now_ts: int,
    buffer_seconds: int,
    get_close_ts: Callable[[int], int | None],
    page_size: int = _PAGE_SIZE_DEFAULT,
    gas_limit: int = 300_000,
    claim_batch_size: int = 10,
    min_bet_with_gas_bnb: float | None = None,
) -> ClaimScanResult:
    """Scan the user's rounds list and claim any claimable/refundable past epochs.

    Notes:
      - The contract's getUserRounds returns only epoch ids (no ledger metadata).
      - We treat claimable()/refundable() as the authoritative indicator that a claim is possible.
      - Claims are batched up to claim_batch_size epochs per tx.
      - If fewer than claim_batch_size claims are pending, we only flush them when
        wallet bankroll falls below min_bet_with_gas_bnb (if provided).
      - In dry mode, we NEVER submit a claim transaction; simulated_net_delta_bnb is always 0.0.
        Dry bankroll updates are handled by the runtime's dry settlement logic, not by this scan.
    """
    wallet_address = wallet_address.strip()
    if not wallet_address:
        raise InvariantError("wallet_address_required")

    total = contract.get_user_rounds_length(wallet_address)
    if total <= 0:
        return ClaimScanResult(scanned_n=0, claimed_n=0)

    path = Path(cursor_path)
    cursor = _read_int_file(path)
    if cursor < 0:
        cursor = 0
    if cursor >= total:
        return ClaimScanResult(scanned_n=0, claimed_n=0)

    size = page_size
    if size <= 0:
        raise InvariantError("claim_page_size_nonpositive")

    scanned_total = 0
    claimed_total = 0
    pending_claims: list[tuple[int, int]] = []

    if claim_batch_size <= 0:
        raise InvariantError("claim_batch_size_nonpositive")
    floor_bnb = min_bet_with_gas_bnb
    if floor_bnb is not None and floor_bnb <= 0.0:
        raise InvariantError("claim_min_bet_with_gas_nonpositive")

    def _flush_pending(*, force_all: bool) -> None:
        nonlocal claimed_total, pending_claims
        if not pending_claims:
            return
        if force_all:
            n = len(pending_claims)
        else:
            n = (len(pending_claims) // claim_batch_size) * claim_batch_size
        if n <= 0:
            return
        to_claim = [epoch for epoch, _ in pending_claims[:n]]
        for i in range(0, len(to_claim), claim_batch_size):
            chunk = to_claim[i : i + claim_batch_size]
            chunk_gas_limit = gas_limit * len(chunk)
            t0 = time.perf_counter()
            gas_price_wei = contract.suggest_gas_price_wei()
            tx = contract.claim(
                epochs=chunk,
                gas_limit=chunk_gas_limit,
                gas_price_wei=gas_price_wei,
            )
            claim_ms = int((time.perf_counter() - t0) * 1000)
            info(
                "NET",
                "RPC",
                "CLAIM",
                epoch=chunk[0],
                tx=tx,
                claim_ms=f"{claim_ms}ms",
                batch_n=len(chunk),
                last_epoch=chunk[-1],
                gas_limit=chunk_gas_limit,
            )
            claimed_total += len(chunk)
        pending_claims = pending_claims[n:]

    # Fetch all epoch IDs in one batched RPC call (instead of N pages).
    all_epochs = contract.get_user_rounds_all_batched(
        wallet_address=wallet_address, cursor=cursor, total=total, page_size=size,
    )

    # Batch-fetch close_ts for all epochs at once.
    close_ts_map = contract.close_ts_batch(all_epochs) if all_epochs else {}

    # Filter to scannable epochs (before live edge and buffer).
    scannable: list[int] = []
    for epoch in all_epochs:
        e = int(epoch)
        if e == locked_epoch or e == current_epoch:
            break
        cts = close_ts_map.get(e)
        if cts is not None and now_ts - cts < buffer_seconds:
            break
        scannable.append(e)
    scanned = len(scannable)

    # Batch-check claimable + refundable for all scannable epochs.
    if not dry and scannable:
        cr_map = contract.claimable_refundable_batch(
            epochs=scannable, wallet_address=wallet_address,
        )
        for i, e in enumerate(scannable):
            c, r = cr_map.get(e, (False, False))
            if c or r:
                pending_claims.append((e, cursor + i))
                _flush_pending(force_all=False)

    scanned_total = scanned
    cursor += scanned

    if (not dry) and pending_claims:
        should_force_flush = False
        if floor_bnb is not None:
            wallet_bnb = float(contract.wallet_balance_bnb(wallet_address))
            should_force_flush = wallet_bnb < floor_bnb
            info(
                "NET",
                "RPC",
                "CLAIM",
                msg=(
                    f"claim_batch_pending={len(pending_claims)} "
                    f"wallet_bnb={wallet_bnb:.6f} "
                    f"min_bet_with_gas_bnb={floor_bnb:.6f} "
                    f"force_small_batch={str(should_force_flush).lower()}"
                ),
            )
        else:
            should_force_flush = True
        if should_force_flush:
            _flush_pending(force_all=True)
        if pending_claims:
            cursor = pending_claims[0][1]
    _write_int_file_atomic(path, cursor)
    return ClaimScanResult(scanned_n=scanned_total, claimed_n=claimed_total)
