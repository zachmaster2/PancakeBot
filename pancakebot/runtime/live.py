"""Cursor-based claim scan: walks user rounds, batches claimable/refundable epochs, and submits claim txs."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pancakebot.chain.prediction_contract import Web3PredictionContract
from pancakebot.util import InvariantError
from pancakebot.log import info, warn

_PAGE_SIZE_DEFAULT = 100

# Operational cap on epochs per ``claim()`` TX. The PredictionV2 contract's
# ``claim(uint256[])`` ABI accepts any array length, but in practice BSC's
# public-RPC submission caps + per-epoch gas (~100-150k for storage writes
# + BNB transfer) make ~10 the soft maximum before TXs risk rejection or
# deprioritization. This constant matches the pre-2026-05-18 operational
# default (then named ``_CLAIM_BATCH_SIZE`` in engine.py); it's preserved
# as the per-TX chunk cap even though the *trigger* is now per-win rather
# than per-batch-of-10-accumulated. DO NOT remove without a chain-side
# verification that larger TXs land reliably across all WRITE_PATH_RPC_URLS.
_MAX_CLAIM_EPOCHS_PER_TX = 10

# Env var holding the live-mode Discord webhook URL. Mirrors the supervisor's
# ``_env_var_for_mode("live")`` definition so a misrouted webhook here would
# produce the same operator-visible miss as a supervisor-side issue.
_LIVE_ALERTS_WEBHOOK_ENV = "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL"


def _send_claim_failure_alert(
    *,
    reason: str,
    tx_hash: str,
    epochs: Sequence[int],
    gas_limit: int,
) -> None:
    """POST a CLAIM FAILED message to the live Discord webhook.

    ``reason`` is one of ``"revert"`` or ``"timeout"``. Best-effort: any
    exception (missing env var, transport error, non-2xx response) is
    swallowed with a WARN log so the operational claim-scan loop never
    crashes on a webhook hiccup. Distinct from the supervisor's status
    classifications -- this is a point-in-time alert at the moment the
    claim failure is observed inside the bot.
    """
    webhook = os.environ.get(_LIVE_ALERTS_WEBHOOK_ENV, "").strip()
    if not webhook:
        warn("CLAIM", "ALERT", "DISABLED",
             reason=reason, tx_hash=tx_hash,
             msg=f"{_LIVE_ALERTS_WEBHOOK_ENV} unset; CLAIM FAILED alert not sent")
        return
    try:
        import requests
    except Exception as e:
        warn("CLAIM", "ALERT", "IMPORT_FAIL",
             reason=reason, tx_hash=tx_hash, err=str(e))
        return

    epoch_str = ",".join(str(int(e)) for e in epochs)
    payload = {
        "username": "PancakeBot-live",
        "content": (
            f":rotating_light: **PancakeBot-live CLAIM FAILED** "
            f"reason=`{reason}` tx=`{tx_hash}` "
            f"epochs=`[{epoch_str}]` gas_limit=`{int(gas_limit)}`"
        ),
    }
    try:
        r = requests.post(webhook, json=payload, timeout=10)
    except Exception as e:
        warn("CLAIM", "ALERT", "POST_FAIL",
             reason=reason, tx_hash=tx_hash, err=str(e))
        return
    if not (200 <= r.status_code < 300):
        warn("CLAIM", "ALERT", "BAD_STATUS",
             reason=reason, tx_hash=tx_hash,
             http_status=r.status_code,
             body=str(getattr(r, "text", ""))[:200])


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
    page_size: int = _PAGE_SIZE_DEFAULT,
    gas_limit: int = 300_000,
    claim_tx_receipt_timeout_seconds: int = 35,
) -> ClaimScanResult:
    """Scan the user's rounds list and claim any claimable/refundable past epochs.

    Notes:
      - The contract's getUserRounds returns only epoch ids (no ledger metadata).
      - We treat claimable()/refundable() as the authoritative indicator that a claim is possible.
      - All claimable epochs found in a single scan are claimed in ONE TX
        (PredictionV2's ``claim(uint256[])`` takes an epoch array). In practice
        this is 1 epoch per scan (the per-win case); occasionally 2+ on startup
        or after a missed iteration. There is no per-N threshold gating —
        every scan with at least one claimable epoch fires a claim TX.
      - In dry mode, we NEVER submit a claim transaction; simulated_net_delta_bnb is always 0.0.
        Dry bankroll updates are handled by the runtime's dry settlement logic, not by this scan.

    Claim-TX outcome handling (live mode):
      - status=success: log NET/RPC/CLAIM, advance cursor.
      - status=revert: log warn(CLAIM/TX/REVERT) + Discord alert, advance
        cursor anyway (the chain rejected this batch; retrying won't help).
        Next iteration re-scans naturally and any epochs that re-show
        as claimable will be re-attempted.
      - status=timeout: log warn(CLAIM/TX/TIMEOUT) + Discord alert, do NOT
        advance cursor. The TX may still mine; the next iteration's
        claim_scan_cursor will re-detect the still-claimable epochs.
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

    claimed_total = 0
    pending_claims: list[tuple[int, int]] = []

    def _flush_pending() -> None:
        """Drain ``pending_claims`` across N TXs of up to
        ``_MAX_CLAIM_EPOCHS_PER_TX`` epochs each.

        On success/revert: advance past the chunk, continue draining.
        On timeout: stop. Leave the chunk + remaining as pending so the
        cursor parks at the first un-claimed epoch; next iteration's scan
        re-detects (the timed-out TX may still mine).
        """
        nonlocal claimed_total, pending_claims
        while pending_claims:
            chunk_pairs = pending_claims[:_MAX_CLAIM_EPOCHS_PER_TX]
            to_claim = [ep for ep, _ in chunk_pairs]
            chunk_gas_limit = gas_limit * len(to_claim)
            t0 = time.perf_counter()
            gas_price_wei = contract.suggest_gas_price_wei()
            result = contract.claim(
                epochs=to_claim,
                gas_limit=chunk_gas_limit,
                gas_price_wei=gas_price_wei,
                wait_receipt=True,
                receipt_timeout_seconds=int(claim_tx_receipt_timeout_seconds),
            )
            claim_ms = int((time.perf_counter() - t0) * 1000)

            if result.status == "success":
                info(
                    "NET",
                    "RPC",
                    "CLAIM",
                    epoch=to_claim[0],
                    tx=result.tx_hash,
                    claim_ms=f"{claim_ms}ms",
                    batch_n=len(to_claim),
                    last_epoch=to_claim[-1],
                    gas_limit=chunk_gas_limit,
                )
                claimed_total += len(to_claim)
                pending_claims = pending_claims[len(to_claim):]
                continue

            if result.status == "revert":
                warn(
                    "CLAIM", "TX", "REVERT",
                    epoch=to_claim[0],
                    last_epoch=to_claim[-1],
                    tx=result.tx_hash,
                    claim_ms=f"{claim_ms}ms",
                    batch_n=len(to_claim),
                    gas_limit=chunk_gas_limit,
                    block_number=result.included_block_number,
                )
                _send_claim_failure_alert(
                    reason="revert",
                    tx_hash=result.tx_hash,
                    epochs=to_claim,
                    gas_limit=chunk_gas_limit,
                )
                # Advance past the reverted chunk -- no retry. Next iteration's
                # cursor walk re-picks any epochs that re-show as claimable.
                # Continue draining remaining chunks; a reverted chunk doesn't
                # block downstream pending epochs from being attempted.
                pending_claims = pending_claims[len(to_claim):]
                continue

            # status == "timeout"
            warn(
                "CLAIM", "TX", "TIMEOUT",
                epoch=to_claim[0],
                last_epoch=to_claim[-1],
                tx=result.tx_hash,
                claim_ms=f"{claim_ms}ms",
                batch_n=len(to_claim),
                gas_limit=chunk_gas_limit,
                receipt_timeout_seconds=int(claim_tx_receipt_timeout_seconds),
            )
            _send_claim_failure_alert(
                reason="timeout",
                tx_hash=result.tx_hash,
                epochs=to_claim,
                gas_limit=chunk_gas_limit,
            )
            # STOP draining. Leave the timed-out chunk + remaining pairs
            # in pending_claims so the cursor parks at the first un-claimed
            # epoch. The timed-out TX may still mine; next iteration's
            # scan re-detects whatever is still claimable.
            return

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

    # Batch-check claimable + refundable for all scannable epochs, then
    # flush ALL pending in a single claim TX. Natural batching for the
    # multi-epoch-at-once case (startup, missed iteration); per-win semantics
    # for the steady state.
    if not dry and scannable:
        cr_map = contract.claimable_refundable_batch(
            epochs=scannable, wallet_address=wallet_address,
        )
        for i, e in enumerate(scannable):
            c, r = cr_map.get(e, (False, False))
            if c or r:
                pending_claims.append((e, cursor + i))
        _flush_pending()

    scanned_total = scanned
    cursor += scanned
    if pending_claims:
        cursor = pending_claims[0][1]
    _write_int_file_atomic(path, cursor)
    return ClaimScanResult(scanned_n=scanned_total, claimed_n=claimed_total)
