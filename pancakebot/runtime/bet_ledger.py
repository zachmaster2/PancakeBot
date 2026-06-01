"""Append-only bet lifecycle ledger + on-chain settlement reconciliation.

Records every bet's lifecycle to a per-mode JSONL file
(``var/live/bets.jsonl`` / ``var/dry/bets.jsonl``) and reconciles open
bets against on-chain ``RoundData`` once their round has closed. Gives
both modes per-bet PnL truth and drives the live lifecycle Discord
alerts.

Writer invariant (NOT POSIX atomicity): the ONLY writer is the engine
main loop, which is single-threaded. Records are appended one line at a
time. After a hard kill, only the FINAL line can be truncated mid-write;
``load_ledger`` detects and skips a corrupted final line. No concurrent
writers exist, so interleaving is impossible by construction. The
``_MAX_LINE_BYTES`` check is cheap defense-in-depth, not a correctness
dependency.

Statuses:
  SUBMITTED     - bet TX broadcast (open; receipt not yet classified)
  CONFIRMED     - bet TX mined, status=1, before lock (open; reconcilable)
  LATE          - bet TX mined, status=0, at/after lock (TERMINAL; gas-only
                  loss — PCS late-lock revert rolled back the stake)
  REVERTED      - bet TX mined, status=0, before lock (TERMINAL; gas-only
                  loss — some other revert rolled back the stake)
  DROPPED       - no receipt within the receipt-wait window (TERMINAL for
                  display — the alert already fired — but RE-checked at
                  reconcile: a TX that mined just after the timeout is
                  silently corrected to a settlement)
  SETTLED_WON   - round closed, our side won (live: claim path later alerts
                  + marks CLAIMED; dry: terminal + silent)
  SETTLED_LOST  - round closed, our side lost (or tie: stake stays with house)
  SETTLED_REFUND- round closed un-oracled past buffer (stake refund-claimable)
  CLAIMED       - winnings/refund claim TX confirmed (live; alert fires here)

Gas-field schema (Option B, NIT 2): every bet-TX-outcome record (SUBMITTED /
CONFIRMED / LATE / REVERTED / DROPPED) ALWAYS carries an explicit
``gas_paid_bnb`` — ``null`` = transient unknown (not yet resolved), ``0`` =
confirmed no gas spent (TX never mined), positive = actual gas spent on chain.
Enforced in ``_append_record``. SETTLED_*/CLAIMED are PnL records and do not
carry the field. So a DROPPED reads ``null`` at the 10s timeout and is resolved
at reconcile to ``0`` (never mined) or the actual gas (mined-and-reverted).

Settlement division of labor (Option B):
  - ``reconcile`` runs at settle-time. It fires the LOSS alert (live) and
    records SETTLED_WON / SETTLED_REFUND SILENTLY — it never moves money.
  - The claim-scan path fires BET WON / BET REFUND at claim-tx-confirm
    (where claim gas + fresh wallet balance are known) and marks CLAIMED.
  - LATE / REVERTED are terminal and never reconciled (removed from
    ``_OPEN_STATUSES``): the stake never left the wallet, so there is
    nothing on-chain to settle or refund.
  - DROPPED is re-checked at reconcile (``_RECHECK_STATUSES``) via
    ``read_bet_amount``: if the TX in fact registered, the settlement alert
    is the implicit correction; otherwise the DROPPED stands silently.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from pancakebot.constants import BNB_WEI
from pancakebot.log import warn
from pancakebot.settlement import settle_from_round_data
from pancakebot.util import InvariantError

_OPEN_STATUSES = frozenset({"SUBMITTED", "CONFIRMED"})  # reconcilable
# DROPPED is "terminal" for display (its alert already fired) but is RE-checked
# at reconcile via read_bet_amount + try_get_receipt: a TX that mined just after
# the 10s timeout may have registered (silently corrected to a settlement) or
# mined-and-reverted (stays DROPPED, gas resolved to the actual value); a
# never-mined DROPPED has its gas resolved from null to 0 (Option B, NIT 2).
# SUBMITTED is also re-checked there (lingering only after a crash between
# submit and the receipt classification).
_RECHECK_STATUSES = frozenset({"SUBMITTED", "DROPPED"})
_TERMINAL_STATUSES = frozenset({
    "LATE", "REVERTED", "DROPPED",
    "SETTLED_WON", "SETTLED_LOST", "SETTLED_REFUND", "CLAIMED",
})
# Records describing a bet-TX outcome ALWAYS carry an explicit ``gas_paid_bnb``
# (Option B schema, NIT 2 redirect): null = transient unknown (state not yet
# resolved), 0 = confirmed no gas spent (TX never mined), positive = actual gas
# spent on chain. SETTLED_*/CLAIMED records are PnL records, not TX-outcome
# records, and do NOT carry the field (writing it would clobber the real gas on
# last-write-wins merge). The invariant is enforced in ``_append_record``.
_GAS_BEARING_STATUSES = frozenset({"SUBMITTED", "CONFIRMED", "LATE", "REVERTED", "DROPPED"})
_MAX_LINE_BYTES = 4096


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_record(ledger_path: str, record: dict[str, Any]) -> bool:
    """Append one record as a single write. Returns True on success, False
    on any failure (logged). Callers gate downstream side effects (alerts,
    settlement) on a True return so a failed persist defers — never
    fire-then-lose (at-least-once, no premature alert)."""
    # Schema invariant (Option B, NIT 2): every bet-TX-outcome record MUST carry
    # an explicit gas_paid_bnb (null/0/positive). A gas-bearing record without
    # the key is a programming error — hard stop, not a swallowed runtime error.
    if record.get("status") in _GAS_BEARING_STATUSES and "gas_paid_bnb" not in record:
        raise InvariantError(
            f"gas_bearing_record_missing_gas_paid_bnb "
            f"(status={record.get('status')} epoch={record.get('epoch')})"
        )
    try:
        line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        if len(line.encode("utf-8")) > _MAX_LINE_BYTES:
            warn(
                "LEDGER",
                f"bet ledger record exceeds {_MAX_LINE_BYTES}B "
                f"(epoch={record.get('epoch')} status={record.get('status')})",
            )
        _ensure_parent_dir(ledger_path)
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(line)
        return True
    except Exception as e:  # noqa: BLE001
        warn(
            "LEDGER",
            f"bet ledger append failed "
            f"(epoch={record.get('epoch')} status={record.get('status')} err={e})",
        )
        return False


# --- Lifecycle recorders ----------------------------------------------------

def record_submitted(
    *, ledger_path: str, epoch: int, side: str, amount_bnb: float,
    tx_hash: str, bankroll_after_bnb: float,
) -> bool:
    """SUBMITTED record — written at TX broadcast (before the receipt wait).
    ``gas_paid_bnb`` is null here: at broadcast the gas spent is not yet known
    (Option B schema — null = transient unknown)."""
    return _append_record(ledger_path, {
        "ts": _now_iso(),
        "status": "SUBMITTED",
        "epoch": int(epoch),
        "side": str(side),
        "amount_bnb": round(float(amount_bnb), 6),
        "tx_hash": str(tx_hash),
        "bankroll_after_bnb": round(float(bankroll_after_bnb), 6),
        "gas_paid_bnb": None,
    })


def classify_confirmation(*, chain_status: int | None, included_late: bool) -> str:
    """Map a bet TX receipt to a ledger status.

    chain_status==1 & !late -> CONFIRMED (registered)
    chain_status==0 &  late -> LATE      (PCS late-lock revert; gas-only)
    chain_status==0 & !late -> REVERTED  (other revert; gas-only)
    chain_status is None (no receipt within the wait window) -> DROPPED
        (terminal; the TX is realistically gone). If it in fact mined late,
        reconcile's read_bet_amount re-check silently corrects it.
    A status==1 with late=True is anomalous (mined success at/after lock);
    treat as LATE (the chain may accept but PCS logic rejects the bet).
    """
    if chain_status is None:
        return "DROPPED"
    if chain_status == 1 and not included_late:
        return "CONFIRMED"
    if included_late:
        return "LATE"
    return "REVERTED"


def actual_gas_bnb(*, gas_used: int | None, effective_gas_price_wei: int | None) -> float | None:
    """Real gas cost (BNB) for a mined TX = gasUsed x effectiveGasPrice.
    None when the receipt is unavailable (timeout) — caller leaves the ledger
    gas field unwritten rather than recording the MAX_GAS_COST_BET_BNB cap."""
    if gas_used is None or effective_gas_price_wei is None:
        return None
    return gas_used * effective_gas_price_wei / BNB_WEI


def _receipt_gas_bnb(receipt: Any) -> float | None:
    """Actual gas (BNB) from a raw TX receipt mapping, or None if unavailable.
    Used at reconcile-time to recover the gas of a TX that mined-and-reverted
    after the receipt-wait window (NIT 2)."""
    if receipt is None:
        return None
    try:
        gas_used = receipt.get("gasUsed")
        eff = receipt.get("effectiveGasPrice")
    except AttributeError:
        return None
    return actual_gas_bnb(gas_used=gas_used, effective_gas_price_wei=eff)


def record_confirmation(
    *, ledger_path: str, epoch: int, chain_status: int | None,
    included_block_number: int | None, included_late: bool,
    gas_paid_bnb: float | None = None,
) -> str:
    """Append the post-receipt status (CONFIRMED/LATE/REVERTED/DROPPED) and
    return it.

    Gas-field schema (Option B, NIT 2): ``gas_paid_bnb`` is ALWAYS emitted.
    ``None`` -> ``null`` = transient unknown (e.g. DROPPED at the 10s timeout,
    before reconcile resolves it); a number = actual gas spent. The reconcile
    pass later resolves a DROPPED's null to 0 (never mined) or the real gas
    (mined-and-reverted)."""
    status = classify_confirmation(chain_status=chain_status, included_late=included_late)
    _append_record(ledger_path, {
        "ts": _now_iso(),
        "status": status,
        "epoch": int(epoch),
        "included_block_number": (
            int(included_block_number) if included_block_number is not None else None
        ),
        "gas_paid_bnb": (round(float(gas_paid_bnb), 6) if gas_paid_bnb is not None else None),
    })
    return status


def record_claimed(
    *, ledger_path: str, epoch: int, amount_bnb: float | None = None,
    new_bankroll_bnb: float | None = None,
) -> bool:
    """Terminal CLAIMED record. ``new_bankroll_bnb`` is the POST-claim fresh
    wallet read (winnings/refund credited) — distinct from the settle-time
    SETTLED_WON/SETTLED_REFUND snapshot, which is preserved for forensics."""
    return _append_record(ledger_path, {
        "ts": _now_iso(),
        "status": "CLAIMED",
        "epoch": int(epoch),
        "amount_bnb": (round(float(amount_bnb), 6) if amount_bnb is not None else None),
        "new_bankroll_bnb": (
            round(float(new_bankroll_bnb), 6) if new_bankroll_bnb is not None else None
        ),
    })


# --- Shared outcome classification (used by reconcile AND dry settle) -------

def classify_settlement(*, outcome: str, bet_bnb: float, credit_bnb: float) -> tuple[str, float]:
    """Map a settlement ``outcome`` to a terminal ledger status + per-bet
    delta. Shared by live reconcile and dry settlement so both modes agree.
      win    -> SETTLED_WON,    delta = credit - stake (net profit)
      loss   -> SETTLED_LOST,   delta = -stake
      refund -> SETTLED_REFUND, delta = credit - stake (~ -claim_gas)
    """
    if outcome == "win":
        return "SETTLED_WON", credit_bnb - bet_bnb
    if outcome == "loss":
        return "SETTLED_LOST", -bet_bnb
    return "SETTLED_REFUND", credit_bnb - bet_bnb


def record_settled(
    *, ledger_path: str, epoch: int, side: str, status: str, delta_bnb: float,
    outcome: str, new_bankroll_bnb: float,
) -> bool:
    return _append_record(ledger_path, {
        "ts": _now_iso(),
        "status": status,
        "epoch": int(epoch),
        "side": str(side),
        "delta_bnb": round(float(delta_bnb), 6),
        "outcome": str(outcome),
        "new_bankroll_bnb": round(float(new_bankroll_bnb), 6),
    })


# --- Replay -----------------------------------------------------------------

def load_ledger(ledger_path: str) -> dict[int, dict[str, Any]]:
    """Replay the append-only log into ``{epoch: merged_latest_record}``
    (last-write-wins per epoch). A corrupted/truncated FINAL line (crash
    mid-append) is skipped with a WARN; a missing file returns ``{}``."""
    p = Path(ledger_path)
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        warn("LEDGER", f"bet ledger read failed (path={ledger_path} err={e})")
        return {}
    out: dict[int, dict[str, Any]] = {}
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            if i == len(lines) - 1:
                warn("LEDGER", "bet ledger: skipping corrupted final line (partial write)")
            else:
                warn("LEDGER", f"bet ledger: skipping corrupted line {i + 1}")
            continue
        if not isinstance(rec, dict) or "epoch" not in rec or "status" not in rec:
            continue
        try:
            ep = int(rec["epoch"])
        except (ValueError, TypeError):
            continue
        merged = out.get(ep, {})
        merged.update(rec)
        out[ep] = merged
    return out


def epoch_was_dropped(ledger_path: str, epoch: int) -> bool:
    """True if ANY historical record for ``epoch`` carried status DROPPED.

    ``load_ledger`` collapses each epoch to its latest record, so a DROPPED that
    a later settlement corrected is invisible there. This scans the raw append
    log so a WON / LOST / REFUND can acknowledge a prior DROPPED with the
    correction suffix (NIT 1). Best-effort: missing/unreadable file -> False."""
    p = Path(ledger_path)
    if not p.exists():
        return False
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("status") != "DROPPED":
            continue
        try:
            if int(rec.get("epoch")) == int(epoch):
                return True
        except (ValueError, TypeError):
            continue
    return False


# --- Reconciliation (settle-time; never moves money) ------------------------

def reconcile(
    *,
    ledger_path: str,
    contract: Any,
    treasury_fee_fraction: float,
    fresh_bankroll_bnb: float,
    buffer_seconds: int,
    now_ts: int | None = None,
    wallet_address: str | None = None,
    lost_alert_fn: Callable[..., None] | None = None,
    dropped_alert_fn: Callable[..., None] | None = None,
) -> list[dict[str, Any]]:
    """Settle open bets whose rounds have closed, and re-check DROPPED/
    SUBMITTED entries against the chain.

    Examined statuses = _OPEN_STATUSES (SUBMITTED, CONFIRMED) +
    _RECHECK_STATUSES (DROPPED, SUBMITTED):
      - CONFIRMED: settle directly via pool math.
      - DROPPED / SUBMITTED: consult ``read_bet_amount`` (authoritative
        registration) AND ``try_get_receipt`` (forensic gas/status, NIT 2).
          * amount>0  -> it DID register (mined just after the wait window):
            reclassify to CONFIRMED with the receipt's actual gas, then fall
            through to settlement — the WON/LOST/REFUND alert is the implicit
            "DROPPED was wrong" correction (suffix added at fire time).
          * amount==0, receipt present (mined-and-reverted): stays DROPPED,
            gas resolved to the actual receipt gas (SUBMITTED also fires the
            DROPPED alert it never got).
          * amount==0, no receipt (truly never mined): SUBMITTED becomes
            DROPPED + alerts; a standing DROPPED stays put (no alert) and has
            its gas resolved from null to 0 (confirmed no gas — Option B).

    Option B alert split: LOSS fires here (``lost_alert_fn``); WIN/REFUND fire
    from the claim path. SUBMITTED->DROPPED fires ``dropped_alert_fn``. This
    function NEVER moves money. Idempotent + append-gated (Fix #7).

    ``fresh_bankroll_bnb`` is the caller's freshly-read bankroll at fire-time.
    Returns the list of settled/transitioned records.
    """
    if now_ts is None:
        now_ts = int(time.time())
    ledger = load_ledger(ledger_path)
    settled: list[dict[str, Any]] = []
    examined = _OPEN_STATUSES | _RECHECK_STATUSES

    for ep in sorted(ledger):
        rec = ledger[ep]
        status0 = rec.get("status")
        if status0 not in examined:
            continue  # terminal (LATE/REVERTED/SETTLED_*/CLAIMED) — skip

        side = str(rec.get("side", ""))
        amount_bnb = float(rec.get("amount_bnb", 0.0) or 0.0)
        if amount_bnb <= 0.0 or side not in ("Bull", "Bear"):
            continue  # malformed record — leave for manual inspection

        # Re-check guard for DROPPED/SUBMITTED. ``read_bet_amount`` is the
        # authoritative "did the stake register" check; the receipt lookup
        # (NIT 2) recovers the actual gas of a TX that mined-and-reverted AFTER
        # our 10s receipt wait, so the forensic gas field is accurate.
        if status0 in _RECHECK_STATUSES:
            if wallet_address is None:
                continue  # can't verify without a wallet — never fabricate
            tx_hash = str(rec.get("tx_hash", "") or "")
            receipt = None
            if tx_hash:
                # noinspection PyBroadException
                try:
                    receipt = contract.try_get_receipt(tx_hash)
                except Exception:
                    continue  # transient RPC — retry next pass (NOT 'never mined')
            # noinspection PyBroadException
            try:
                onchain_amount = contract.read_bet_amount(ep, wallet_address)
            except Exception:
                continue  # transient RPC — retry next pass

            gas_bnb = _receipt_gas_bnb(receipt)
            included_block = receipt.get("blockNumber") if receipt is not None else None

            if onchain_amount > 0:
                # The bet DID register (mined just after the wait window).
                # Reclassify SUBMITTED/DROPPED -> CONFIRMED (silent), recording
                # the actual gas (null only if the node omitted
                # effectiveGasPrice), then fall through to settle. The
                # settlement alert is the implicit "DROPPED was wrong"
                # correction (the `(previously reported as DROPPED)` suffix is
                # added at fire time when the epoch has a prior DROPPED).
                conf_rec = {
                    "ts": _now_iso(), "status": "CONFIRMED", "epoch": int(ep),
                    "included_block_number": (
                        int(included_block) if included_block is not None else None
                    ),
                    "gas_paid_bnb": (round(float(gas_bnb), 6) if gas_bnb is not None else None),
                }
                if not _append_record(ledger_path, conf_rec):
                    continue  # persist failed -> defer the whole correction
                # fall through (no continue) to settlement below.
            else:
                # Not registered on-chain. Resolve the forensic gas (Option B):
                #   no receipt -> 0.0   (confirmed never mined)
                #   receipt    -> actual gas (mined-and-reverted; None only if
                #                 the node omitted effectiveGasPrice -> still
                #                 transient-unknown, leave for a later pass).
                if receipt is None:
                    new_gas: float | None = 0.0
                else:
                    new_gas = round(float(gas_bnb), 6) if gas_bnb is not None else None
                if status0 == "SUBMITTED":
                    # Crash-lingering SUBMITTED -> terminal DROPPED + the alert
                    # it never got, carrying the resolved gas.
                    if not _append_record(ledger_path, {
                        "ts": _now_iso(), "status": "DROPPED", "epoch": int(ep),
                        "gas_paid_bnb": new_gas,
                        "note": ("submitted_then_reverted" if receipt is not None
                                 else "submitted_then_unregistered"),
                    }):
                        continue
                    settled.append({"epoch": int(ep), "status": "DROPPED",
                                    "side": side, "outcome": "dropped"})
                    if dropped_alert_fn is not None:
                        # noinspection PyBroadException
                        try:
                            dropped_alert_fn(epoch=int(ep), bankroll_bnb=fresh_bankroll_bnb)
                        except Exception as e:  # noqa: BLE001
                            warn("LEDGER", f"bet dropped alert failed (epoch={ep} err={e})")
                elif rec.get("gas_paid_bnb") is None and new_gas is not None:
                    # Standing DROPPED whose gas was still unknown (null):
                    # resolve it ONCE to 0 (never mined) or the actual gas
                    # (mined-and-reverted). No alert (DROPPED already fired);
                    # status stays DROPPED. Idempotent — once the gas is
                    # concrete a later pass sees it non-null and skips.
                    if _append_record(ledger_path, {
                        "ts": _now_iso(), "status": "DROPPED", "epoch": int(ep),
                        "gas_paid_bnb": new_gas,
                        "note": ("gas_resolved_mined_reverted" if receipt is not None
                                 else "gas_resolved_never_mined"),
                    }):
                        settled.append({"epoch": int(ep), "status": "DROPPED",
                                        "side": side, "outcome": "dropped_gas_resolved"})
                # else: standing DROPPED already resolved (gas non-null) -> silent.
                continue  # not registered -> nothing to settle

        # noinspection PyBroadException
        try:
            rd = contract.round_data(ep)
        except Exception:
            continue  # transient RPC — retry next pass

        # Match PCS V2 _refundable() exactly: a round only becomes refundable
        # at close_ts + bufferSeconds. Settling a still-oracle-pending round
        # inside [close_ts, close_ts+buffer] would prematurely write
        # SETTLED_REFUND (the oracle may still resolve it). (Reviewer Fix #3.)
        if not rd.oracle_called and (rd.close_ts + buffer_seconds) >= now_ts:
            continue  # round not yet refund-eligible on-chain (PCS uses strict
            #            `block.timestamp > closeTimestamp + bufferSeconds`, so
            #            it is NOT refundable AT the boundary second)

        result = settle_from_round_data(
            bet_bnb=amount_bnb,
            bet_side=side,
            lock_price_usd=rd.lock_price_usd,
            close_price_usd=rd.close_price_usd,
            bull_amount_wei=rd.bull_amount_wei,
            bear_amount_wei=rd.bear_amount_wei,
            oracle_called=rd.oracle_called,
            treasury_fee_fraction=treasury_fee_fraction,
        )
        status, delta_bnb = classify_settlement(
            outcome=result.outcome, bet_bnb=amount_bnb, credit_bnb=result.credit_bnb,
        )

        # Fix #7: persist FIRST; only fire alert/append-to-result on success.
        persisted = record_settled(
            ledger_path=ledger_path, epoch=ep, side=side, status=status,
            delta_bnb=delta_bnb, outcome=result.outcome,
            new_bankroll_bnb=fresh_bankroll_bnb,
        )
        if not persisted:
            continue  # defer alert + settlement to next pass

        settled.append({
            "epoch": int(ep), "status": status, "side": side,
            "amount_bnb": round(amount_bnb, 6), "delta_bnb": round(delta_bnb, 6),
            "outcome": result.outcome,
        })

        # LOSS alert only (live). WIN/REFUND alerts fire from the claim path.
        # A LOSS that corrects a prior DROPPED (TX mined just after the wait
        # window) carries the `(previously reported as DROPPED)` suffix (NIT 1).
        if status == "SETTLED_LOST" and lost_alert_fn is not None:
            prev_dropped = epoch_was_dropped(ledger_path, ep)
            # noinspection PyBroadException
            try:
                lost_alert_fn(
                    epoch=int(ep), won=False, delta_bnb=delta_bnb,
                    amount_bnb=amount_bnb, new_bankroll_bnb=fresh_bankroll_bnb,
                    previously_dropped=prev_dropped,
                )
            except Exception as e:  # noqa: BLE001
                warn("LEDGER", f"bet lost alert failed (epoch={ep} err={e})")

    return settled
