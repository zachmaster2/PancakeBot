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
) -> ClaimScanResult:
    """Scan the user's rounds list and claim any claimable/refundable past epochs.

    Notes:
      - The contract's getUserRounds returns only epoch ids (no ledger metadata).
      - We treat claimable()/refundable() as the authoritative indicator that a claim is possible.
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

    while True:
        epochs = list(
            contract.get_user_rounds(
                wallet_address=wallet_address,
                cursor=cursor,
                size=size,
            )
        )
        if not epochs:
            break

        scanned = 0
        stop = False

        for epoch in epochs:
            e = int(epoch)

            # Stop at the live edge.
            if e == locked_epoch or e == current_epoch:
                stop = True
                break

            close_ts = get_close_ts(e)
            if close_ts is not None:
                if now_ts - close_ts < buffer_seconds:
                    stop = True
                    break

            scanned += 1

            # In dry mode we never submit on-chain claims.
            if dry:
                continue

            claimable = contract.claimable(epoch=e, wallet_address=wallet_address)
            refundable = contract.refundable(epoch=e, wallet_address=wallet_address)
            if claimable or refundable:
                t0 = time.perf_counter()
                gas_price_wei = contract.suggest_gas_price_wei()
                tx = contract.claim(
                    epochs=[e],
                    gas_limit=gas_limit,
                    gas_price_wei=gas_price_wei,
                )
                claim_ms = int((time.perf_counter() - t0) * 1000)
                info("NET", "RPC", "CLAIM", epoch=e, tx=str(tx), claim_ms=f"{claim_ms}ms")
                claimed_total += 1

        scanned_total += scanned
        cursor += scanned

        if stop:
            break

        if cursor >= total:
            break

    _write_int_file_atomic(path, cursor)
    return ClaimScanResult(scanned_n=scanned_total, claimed_n=claimed_total)
