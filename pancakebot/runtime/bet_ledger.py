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
  PLACED        - bet TX submitted (open)
  CONFIRMED     - bet TX mined, status=1, before lock (open; reconcilable)
  LATE          - bet TX mined, status=0, at/after lock (TERMINAL; gas-only
                  loss — PCS late-lock revert rolled back the stake)
  REVERTED      - bet TX mined, status=0, before lock (TERMINAL; gas-only
                  loss — some other revert rolled back the stake)
  SETTLED_WON   - round closed, our side won (live: claim path later alerts
                  + marks CLAIMED; dry: terminal + silent)
  SETTLED_LOST  - round closed, our side lost (or tie: stake stays with house)
  SETTLED_REFUND- round closed un-oracled past buffer (stake refund-claimable)
  CLAIMED       - winnings/refund claim TX confirmed (live; alert fires here)

Settlement division of labor (Option B):
  - ``reconcile`` runs at settle-time. It fires the LOSS alert (live) and
    records SETTLED_WON / SETTLED_REFUND SILENTLY — it never moves money.
  - The claim-scan path fires BET WON / BET REFUND at claim-tx-confirm
    (where claim gas + fresh wallet balance are known) and marks CLAIMED.
  - LATE / REVERTED are terminal and never reconciled (removed from
    ``_OPEN_STATUSES``): the stake never left the wallet, so there is
    nothing on-chain to settle or refund.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from pancakebot.log import warn
from pancakebot.settlement import settle_from_round_data

_OPEN_STATUSES = frozenset({"PLACED", "CONFIRMED"})  # reconcilable
_TERMINAL_STATUSES = frozenset({
    "LATE", "REVERTED", "SETTLED_WON", "SETTLED_LOST", "SETTLED_REFUND", "CLAIMED",
})
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

def record_placed(
    *, ledger_path: str, epoch: int, side: str, amount_bnb: float,
    tx_hash: str, bankroll_after_bnb: float,
) -> bool:
    return _append_record(ledger_path, {
        "ts": _now_iso(),
        "status": "PLACED",
        "epoch": int(epoch),
        "side": str(side),
        "amount_bnb": round(float(amount_bnb), 6),
        "tx_hash": str(tx_hash),
        "bankroll_after_bnb": round(float(bankroll_after_bnb), 6),
    })


def classify_confirmation(*, chain_status: int | None, included_late: bool) -> str:
    """Map a bet TX receipt to a ledger status.

    chain_status==1 & !late -> CONFIRMED (registered)
    chain_status==0 &  late -> LATE      (PCS late-lock revert; gas-only)
    chain_status==0 & !late -> REVERTED  (other revert; gas-only)
    chain_status is None (timeout, no receipt) -> PLACED (stays open;
        next pass re-resolves once the TX mines or the round settles).
    A status==1 with late=True is anomalous (mined success at/after lock);
    treat as LATE (the chain may accept but PCS logic rejects the bet).
    """
    if chain_status is None:
        return "PLACED"
    if chain_status == 1 and not included_late:
        return "CONFIRMED"
    if included_late:
        return "LATE"
    return "REVERTED"


def confirmation_alert_sequence(conf_status: str) -> tuple[str, ...]:
    """Which lifecycle alerts a post-mine confirmation status triggers, in
    order. BET PLACED is the audit-log alert for any TX that mined (gas spent);
    LATE/REVERTED add a follow-up. A receipt timeout (status "PLACED", no
    chain_status) fires nothing — no gas info yet, TX may still mine.
      CONFIRMED -> (PLACED,)
      LATE      -> (PLACED, LATE)
      REVERTED  -> (PLACED, REVERTED)
      PLACED    -> ()            # timeout, no receipt
    """
    if conf_status == "CONFIRMED":
        return ("PLACED",)
    if conf_status == "LATE":
        return ("PLACED", "LATE")
    if conf_status == "REVERTED":
        return ("PLACED", "REVERTED")
    return ()


def record_confirmation(
    *, ledger_path: str, epoch: int, chain_status: int | None,
    included_block_number: int | None, included_late: bool,
    gas_paid_bnb: float | None = None,
) -> str:
    """Append the post-mine status (CONFIRMED/LATE/REVERTED/PLACED). Returns
    the status written."""
    status = classify_confirmation(chain_status=chain_status, included_late=included_late)
    if status == "PLACED":
        return status  # no transition record; nothing changed
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


def record_claimed(*, ledger_path: str, epoch: int, amount_bnb: float | None = None) -> bool:
    return _append_record(ledger_path, {
        "ts": _now_iso(),
        "status": "CLAIMED",
        "epoch": int(epoch),
        "amount_bnb": (round(float(amount_bnb), 6) if amount_bnb is not None else None),
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
    reverted_alert_fn: Callable[..., None] | None = None,
) -> list[dict[str, Any]]:
    """Settle open bets (PLACED/CONFIRMED) whose rounds have closed.

    Option B division: this fires the LOSS alert only (via ``lost_alert_fn``,
    live-only). WIN and REFUND are recorded SILENTLY here — the claim-scan
    path fires their alerts at claim-tx-confirm (where claim gas + fresh
    balance are known). This function NEVER moves money in either mode.

    Idempotent: terminal-status epochs are skipped, so re-running (per round
    + crash-recovery) never double-settles or double-alerts. Append-gated:
    a settled record + its alert only fire after the terminal record
    persists (Fix #7) — a failed write defers both to the next pass.

    LATE / REVERTED are NOT in ``_OPEN_STATUSES`` and are never touched: the
    stake never left the wallet (EVM rollback), so there is nothing to settle.

    ``fresh_bankroll_bnb`` is the caller's freshly-read bankroll at fire-time
    (live: wallet balance; the value already reflects placement debits of any
    in-flight bets). The per-bet ``delta`` is bet-specific and independent of
    other bets. Returns the list of settled records.
    """
    if now_ts is None:
        now_ts = int(time.time())
    ledger = load_ledger(ledger_path)
    settled: list[dict[str, Any]] = []

    for ep in sorted(ledger):
        rec = ledger[ep]
        if rec.get("status") not in _OPEN_STATUSES:
            continue  # idempotent: terminal (LATE/REVERTED/SETTLED_*/CLAIMED) — skip

        side = str(rec.get("side", ""))
        amount_bnb = float(rec.get("amount_bnb", 0.0) or 0.0)
        if amount_bnb <= 0.0 or side not in ("Bull", "Bear"):
            continue  # malformed PLACED record — leave for manual inspection

        # Timeout∩late guard (Reviewer Fix #2). A still-PLACED entry at
        # reconcile time means the bet receipt timed out (a clean mine would
        # have written CONFIRMED). We don't know if the TX mined cleanly, mined
        # late (stake rolled back), or never mined. Check the authoritative
        # on-chain ledger amount: 0 -> the bet never registered -> terminal
        # REVERTED (do NOT pool-settle; there's no stake in the pool, settling
        # would fabricate a WON/LOST). >0 -> registered cleanly -> fall through
        # to normal settlement. Live-only (wallet_address provided); CONFIRMED
        # entries skip this (their receipt was already good).
        if rec.get("status") == "PLACED" and wallet_address is not None:
            # noinspection PyBroadException
            try:
                onchain_amount = contract.read_bet_amount(ep, wallet_address)
            except Exception:
                continue  # transient RPC — retry next pass
            if onchain_amount == 0:
                persisted = _append_record(ledger_path, {
                    "ts": _now_iso(), "status": "REVERTED", "epoch": int(ep),
                    "note": "timeout_then_unregistered",
                })
                if not persisted:
                    continue
                settled.append({"epoch": int(ep), "status": "REVERTED",
                                "side": side, "outcome": "reverted"})
                if reverted_alert_fn is not None:
                    # noinspection PyBroadException
                    try:
                        reverted_alert_fn(epoch=int(ep), bankroll_bnb=fresh_bankroll_bnb)
                    except Exception as e:  # noqa: BLE001
                        warn("LEDGER", f"bet reverted alert failed (epoch={ep} err={e})")
                continue  # terminal — do not pool-settle

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
        if status == "SETTLED_LOST" and lost_alert_fn is not None:
            # noinspection PyBroadException
            try:
                lost_alert_fn(
                    epoch=int(ep), won=False, delta_bnb=delta_bnb,
                    amount_bnb=amount_bnb, new_bankroll_bnb=fresh_bankroll_bnb,
                )
            except Exception as e:  # noqa: BLE001
                warn("LEDGER", f"bet lost alert failed (epoch={ep} err={e})")

    return settled
