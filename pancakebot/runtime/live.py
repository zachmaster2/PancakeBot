"""Cursor-based claim scan: walks user rounds, batches claimable/refundable epochs, and submits claim txs."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pancakebot.chain.prediction_contract import Web3PredictionContract
from pancakebot.constants import BNB_WEI, MAX_GAS_PRICE_WEI
from pancakebot.util import GasPriceCapBreachedError, InvariantError
from pancakebot.log import info, warn
from pancakebot.runtime import bet_ledger

_PAGE_SIZE_DEFAULT = 100


def _truncate_tx_hash(tx_hash: str) -> str:
    """Render the first 8 chars of a tx hash with a trailing ellipsis
    (e.g. ``0x123456...``). Mirrors the BET log convention in engine.py.
    Full hash remains available in the Discord alert payload and on
    explorer; the operator stdout line just needs a session-disambiguator."""
    if not tx_hash or len(tx_hash) <= 8:
        return tx_hash
    return f"{tx_hash[:8]}..."

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

# Env vars holding the per-mode Discord webhook URLs. Mirror the supervisor's
# ``_env_var_for_mode(...)`` definitions so a misrouted webhook here would
# produce the same operator-visible miss as a supervisor-side issue.
_LIVE_ALERTS_WEBHOOK_ENV = "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL"
_DRY_ALERTS_WEBHOOK_ENV = "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL"


@dataclass(frozen=True, slots=True)
class AlertChannel:
    """Discord routing target for a bet-lifecycle alert. Live and dry are
    symmetric: each mode owns a ``(webhook env var, bot username, message
    prefix)`` triple, and both pass their channel to ``_post_mode_alert``. The
    alert body is mode-agnostic — only the channel differs (webhook + username
    + a ``[DRY] ``/``""`` prefix)."""
    webhook_env: str
    username: str
    prefix: str


LIVE_CHANNEL = AlertChannel(_LIVE_ALERTS_WEBHOOK_ENV, "PancakeBot-live", "[LIVE] ")
DRY_CHANNEL = AlertChannel(_DRY_ALERTS_WEBHOOK_ENV, "PancakeBot-dry", "[DRY] ")


def _fmt_gwei(wei: int) -> str:
    """Render a wei value in gwei for human-readable Discord display. Clean
    integer multiples of 1e9 render as int (``8 gwei``); otherwise rounded to
    2 decimals (``8.25 gwei``). The underlying config stays in wei."""
    gwei = wei / 1e9
    if gwei == int(gwei):
        return f"{int(gwei)} gwei"
    return f"{gwei:.2f} gwei"


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
        warn("ALERT", f"{_LIVE_ALERTS_WEBHOOK_ENV} unset; CLAIM FAILED alert not sent (reason={reason} tx={tx_hash})")
        return
    try:
        import requests
    except Exception as e:
        warn("ALERT", f"CLAIM FAILED alert import failed (reason={reason} tx={tx_hash} err={e})")
        return

    epoch_str = ",".join(str(int(e)) for e in epochs)
    payload = {
        "username": "PancakeBot-live",
        "content": (
            f"[LIVE] [CRIT] **CLAIM FAILED** reason=`{reason}`, tx=`{tx_hash}`, "
            f"epochs=`[{epoch_str}]`, gas_limit=`{int(gas_limit)}`"
        ),
    }
    try:
        r = requests.post(webhook, json=payload, timeout=10)
    except Exception as e:
        warn("ALERT", f"CLAIM FAILED alert post failed (reason={reason} tx={tx_hash} err={e})")
        return
    if not (200 <= r.status_code < 300):
        body = str(getattr(r, "text", ""))[:200]
        warn("ALERT", f"CLAIM FAILED alert bad status (reason={reason} tx={tx_hash} http_status={r.status_code} body={body})")


def send_gas_cap_breach_alert(
    *,
    path: str,
    suggested_wei: int,
    cap_wei: int,
    epoch: int | None = None,
    epochs: Sequence[int] | None = None,
) -> None:
    """POST a GAS CAP BREACHED message to the live Discord webhook.

    ``path`` is one of ``"bet"`` or ``"claim"``. Bet-side breach is
    CRITICAL (the round's bet is skipped — direct PnL impact). Claim-side
    breach is lower-severity (claims retry next round automatically).
    Best-effort: any exception is swallowed with a WARN log so the
    bot's hot path never crashes on a webhook hiccup.
    """
    webhook = os.environ.get(_LIVE_ALERTS_WEBHOOK_ENV, "").strip()
    if not webhook:
        warn(
            "ALERT",
            f"{_LIVE_ALERTS_WEBHOOK_ENV} unset; GAS CAP BREACHED alert not sent "
            f"(path={path} suggested={suggested_wei} cap={cap_wei})",
        )
        return
    try:
        import requests
    except Exception as e:
        warn("ALERT", f"GAS CAP BREACHED alert import failed (path={path} err={e})")
        return

    if path == "bet":
        sev_tag = "[CRIT]"
        action_note = "Bet SKIPPED. OPERATOR ACTION REQUIRED: raise MAX_GAS_PRICE_WEI and review before resuming."
    else:
        sev_tag = "[WARN]"
        action_note = "Claim skipped this round (retries next round). Raise MAX_GAS_PRICE_WEI before claims back up."

    epoch_str = ""
    if epoch is not None:
        epoch_str = f"epoch=`{int(epoch)}`, "
    elif epochs:
        epoch_str = "epochs=`[" + ",".join(str(int(e)) for e in epochs) + "]`, "

    payload = {
        "username": "PancakeBot-live",
        "content": (
            f"[LIVE] {sev_tag} **GAS CAP BREACHED** path=`{path}`, {epoch_str}"
            f"suggested=`{_fmt_gwei(suggested_wei)}`, cap=`{_fmt_gwei(cap_wei)}`, "
            f"ratio=`{suggested_wei / cap_wei:.2f}x` — {action_note}"
        ),
    }
    try:
        r = requests.post(webhook, json=payload, timeout=10)
    except Exception as e:
        warn("ALERT", f"GAS CAP BREACHED alert post failed (path={path} err={e})")
        return
    if not (200 <= r.status_code < 300):
        body = str(getattr(r, "text", ""))[:200]
        warn(
            "ALERT",
            f"GAS CAP BREACHED alert bad status "
            f"(path={path} http_status={r.status_code} body={body})",
        )


def _post_mode_alert(channel: AlertChannel, content: str, *, label: str, ctx: str) -> None:
    """The single best-effort POST for bet-lifecycle alerts, called directly by
    both live and dry. ``channel`` supplies the routing triple (webhook env,
    username, message prefix); ``content`` is the mode-agnostic body. Swallows
    every failure (missing env, import, transport, non-2xx) with a WARN log so
    the bot hot path never crashes on a webhook hiccup. ``label`` is the alert
    name for log lines; ``ctx`` is extra context."""
    webhook = os.environ.get(channel.webhook_env, "").strip()
    if not webhook:
        warn("ALERT", f"{channel.webhook_env} unset; {label} alert not sent ({ctx})")
        return
    try:
        import requests
    except Exception as e:  # noqa: BLE001
        warn("ALERT", f"{label} alert import failed ({ctx} err={e})")
        return
    payload = {"username": channel.username, "content": channel.prefix + content}
    try:
        r = requests.post(webhook, json=payload, timeout=10)
    except Exception as e:  # noqa: BLE001
        warn("ALERT", f"{label} alert post failed ({ctx} err={e})")
        return
    if not (200 <= r.status_code < 300):
        body = str(getattr(r, "text", ""))[:200]
        warn("ALERT", f"{label} alert bad status ({ctx} http_status={r.status_code} body={body})")


# Locked single-line alert format (Step 31 final). ASCII [SEV] tag, bolded
# type, em-dash before details, comma-separated details, 4-decimal BNB,
# backtick-fenced values, no gas (gas still goes to the ledger file for
# forensics — only stripped from Discord display), no tx hashes.

def send_bot_ready_alert(*, channel: AlertChannel, bankroll_bnb: float) -> None:
    """[INFO] BOT READY — fired once per bot start after the first successful
    wallet-balance read, so the first BET SUBMITTED has a reference point.
    Bot-owned (not the supervisor's STARTED alert)."""
    _post_mode_alert(
        channel,
        f"[INFO] **BOT READY** `{channel.username}` — bankroll `{bankroll_bnb:.4f}` BNB",
        label="BOT READY", ctx="startup",
    )


def send_cooldown_entered_alert(
    *, channel: AlertChannel, drawdown_pct: float, threshold_pct: float,
    bankroll_bnb: float, cooldown_rounds: int, approx_hours: float,
) -> None:
    """[WARN] COOLDOWN ENTERED — the drawdown-from-peak breaker fired; NEW bet
    placement pauses for ``cooldown_rounds`` rounds. Settlement, claim, and
    alerts for already-placed bets keep running (cooldown gates new bets only)."""
    _post_mode_alert(
        channel,
        f"[WARN] **COOLDOWN ENTERED** — drawdown `{drawdown_pct:.1f}%`, "
        f"threshold `{threshold_pct:.0f}%`, bankroll `{bankroll_bnb:.4f}` BNB, "
        f"`{int(cooldown_rounds)}` rounds (~`{approx_hours:.1f}h`)",
        label="COOLDOWN ENTERED", ctx=f"drawdown={drawdown_pct:.1f}%",
    )


def send_cooldown_lifted_alert(*, channel: AlertChannel, bankroll_bnb: float) -> None:
    """[INFO] COOLDOWN LIFTED — the drawdown cooldown elapsed; bet placement
    resumes."""
    _post_mode_alert(
        channel,
        f"[INFO] **COOLDOWN LIFTED** — bankroll `{bankroll_bnb:.4f}` BNB, betting resumes",
        label="COOLDOWN LIFTED", ctx="resume",
    )


def send_bet_submitted_alert(
    *, channel: AlertChannel, epoch: int, side: str, amount_bnb: float, projected_bankroll_bnb: float,
) -> None:
    """[INFO] BET SUBMITTED — fired at bet placement (live: TX broadcast before
    the receipt wait; dry: simulated placement, no TX). ``projected_bankroll_bnb``
    = pre_bet_wallet − stake − bet gas cap (the bankroll if the bet registers;
    in dry this equals the post-debit simulated bankroll). Live's post-receipt
    alert (CONFIRMED / LATE / REVERTED / DROPPED) reports the actual fresh
    balance; dry's placement is atomic so SUBMITTED is its single placement
    event."""
    _post_mode_alert(
        channel,
        f"[INFO] **BET SUBMITTED** epoch `{int(epoch)}` — "
        f"Bet `{amount_bnb:.4f}` BNB on {side}, projected bankroll `{projected_bankroll_bnb:.4f}` BNB",
        label="BET SUBMITTED", ctx=f"epoch={epoch} side={side}",
    )


def send_bet_confirmed_alert(*, channel: AlertChannel, epoch: int, bankroll_bnb: float) -> None:
    """[INFO] BET CONFIRMED — bet TX mined before lock, stake registered.
    Bankroll is the fresh post-confirm wallet read (stake+gas debited).
    Amount/side already shown in BET SUBMITTED. Live-only (dry has no receipt)."""
    _post_mode_alert(
        channel,
        f"[INFO] **BET CONFIRMED** epoch `{int(epoch)}` — bankroll `{bankroll_bnb:.4f}` BNB",
        label="BET CONFIRMED", ctx=f"epoch={epoch}",
    )


def send_bet_late_alert(*, channel: AlertChannel, epoch: int, bankroll_bnb: float) -> None:
    """[WARN] BET LATE — bet TX mined at/after lock; PCS rejected it (stake
    rolled back). Bankroll is the fresh read (gas-only debit)."""
    _post_mode_alert(
        channel,
        f"[WARN] **BET LATE** epoch `{int(epoch)}` — bankroll `{bankroll_bnb:.4f}` BNB",
        label="BET LATE", ctx=f"epoch={epoch}",
    )


def send_bet_reverted_alert(*, channel: AlertChannel, epoch: int, bankroll_bnb: float) -> None:
    """[WARN] BET REVERTED — bet TX mined status=0 (paused / min-bet /
    double-bet / etc). EVM rolled back the stake; nothing to claim."""
    _post_mode_alert(
        channel,
        f"[WARN] **BET REVERTED** epoch `{int(epoch)}` — bankroll `{bankroll_bnb:.4f}` BNB",
        label="BET REVERTED", ctx=f"epoch={epoch}",
    )


def send_bet_dropped_alert(*, channel: AlertChannel, epoch: int, bankroll_bnb: float) -> None:
    """[WARN] BET DROPPED — no receipt within the bet receipt-wait window
    (~10s); the TX is realistically dropped from the mempool. Bankroll is the
    fresh read. If the TX in fact mined late, reconcile silently corrects this
    to a settlement (BET WON/LOST/REFUND follows)."""
    _post_mode_alert(
        channel,
        f"[WARN] **BET DROPPED** epoch `{int(epoch)}` — bankroll `{bankroll_bnb:.4f}` BNB",
        label="BET DROPPED", ctx=f"epoch={epoch}",
    )


# When a settlement (WON / LOST / REFUND) is the implicit correction of a bet
# we earlier reported as DROPPED (the TX mined just after the 10s receipt wait),
# this suffix acknowledges the prior alert so the operator isn't left with a
# dangling DROPPED. Determined by the caller from the ledger history (NIT 1).
_DROPPED_CORRECTION_SUFFIX = " (previously reported as DROPPED)"


def send_bet_settled_alert(
    *, channel: AlertChannel, epoch: int, won: bool, delta_bnb: float, amount_bnb: float,
    new_bankroll_bnb: float, previously_dropped: bool = False,
) -> None:
    """[INFO] BET WON / BET LOST. Live: WON fires from the claim path at
    claim-tx-confirm, LOST from reconcile at settle-time. Dry: both fire from
    ``_dry_settle_available_bets`` at simulated settlement. Amounts shown as
    positive numbers (the verb communicates direction). Keyword signature
    matches the LOST ``lost_alert_fn`` contract in ``bet_ledger.reconcile``
    (the live reconcile call binds ``channel`` via ``functools.partial``).
    ``previously_dropped`` appends the correction suffix (NIT 1)."""
    suffix = _DROPPED_CORRECTION_SUFFIX if previously_dropped else ""
    if won:
        content = (
            f"[INFO] **BET WON** epoch `{int(epoch)}` — "
            f"Won `{delta_bnb:.4f}` BNB, bankroll `{new_bankroll_bnb:.4f}` BNB{suffix}"
        )
    else:
        content = (
            f"[INFO] **BET LOST** epoch `{int(epoch)}` — "
            f"Lost `{amount_bnb:.4f}` BNB, bankroll `{new_bankroll_bnb:.4f}` BNB{suffix}"
        )
    _post_mode_alert(channel, content, label=("BET WON" if won else "BET LOST"), ctx=f"epoch={epoch}")


def send_bet_won_batch_alert(
    *, channel: AlertChannel, epochs: list[int], total_delta_bnb: float, new_bankroll_bnb: float,
) -> None:
    """[INFO] BET WON (multi-epoch) — combined alert for a rare multi-epoch
    claim (startup / missed-iteration batch). Per-epoch detail is in the
    ledger. Live-only (dry settles each epoch individually)."""
    ep_str = "[" + ", ".join(str(int(e)) for e in epochs) + "]"
    _post_mode_alert(
        channel,
        f"[INFO] **BET WON** epochs `{ep_str}` — "
        f"Won `{total_delta_bnb:.4f}` BNB total, bankroll `{new_bankroll_bnb:.4f}` BNB",
        label="BET WON batch", ctx=f"epochs={ep_str}",
    )


def send_bet_refund_alert(
    *, channel: AlertChannel, epoch: int, refund_bnb: float, new_bankroll_bnb: float,
    previously_dropped: bool = False,
) -> None:
    """[INFO] BET REFUND — un-oracled-past-buffer round; stake refund-claimed.
    Live: fires from the claim path at refund-claim-tx-confirm. Dry: fires from
    simulated settlement. ``refund_bnb`` is the stake returned.
    ``previously_dropped`` appends the correction suffix (NIT 1)."""
    suffix = _DROPPED_CORRECTION_SUFFIX if previously_dropped else ""
    _post_mode_alert(
        channel,
        f"[INFO] **BET REFUND** epoch `{int(epoch)}` — "
        f"Refunded `{refund_bnb:.4f}` BNB, bankroll `{new_bankroll_bnb:.4f}` BNB{suffix}",
        label="BET REFUND", ctx=f"epoch={epoch}",
    )


def fire_claim_settled_alerts(
    *, ledger_path: str, claimed_epochs: list[int], contract: Web3PredictionContract,
    wallet_address: str,
) -> None:
    """After a successful claim TX, fire BET WON / BET REFUND alerts for the
    claimed epochs that belong to our ledger, and mark them CLAIMED.

    Option B: this is where WON/REFUND alerts fire (at claim-tx-confirm), so
    they carry a fresh wallet balance. Relies on reconcile having
    already written the terminal SETTLED_WON / SETTLED_REFUND record (with its
    per-bet ``delta_bnb``) — claim runs AFTER reconcile in the engine loop.

    Non-ledgered claimed epochs (legacy / manual bets from before the ledger)
    are skipped — we have no stake to attribute. Best-effort: never raises."""
    # noinspection PyBroadException
    try:
        ledger = bet_ledger.load_ledger(ledger_path)
        ours = [e for e in claimed_epochs
                if e in ledger and ledger[e].get("status") != "CLAIMED"]
        if not ours:
            return
        # Read-your-writes: claim() sent + confirmed on the CURRENT node, so
        # read the post-claim balance on THAT node (no rotate). A rotating read
        # can land on a sibling node lagging the claim block (~1 BSC block) and
        # return pre-claim state — the BET WON stale-bankroll bug (2026-06-03).
        # Fall back: non-rotating -> rotating (node N briefly unreachable) ->
        # ledger snapshot (total RPC failure).
        try:
            fresh_bankroll = contract.wallet_balance_bnb_no_rotate(wallet_address)
        except Exception:  # noqa: BLE001
            try:
                fresh_bankroll = float(contract.wallet_balance_bnb(wallet_address))
            except Exception:  # noqa: BLE001
                fresh_bankroll = float(ledger[ours[0]].get("new_bankroll_bnb", 0.0) or 0.0)
                warn(
                    "ALERT",
                    f"post-claim balance read failed (epochs={ours}); using "
                    f"ledger snapshot {fresh_bankroll:.4f} BNB for settled alert",
                )

        # Only fire/claim-mark epochs reconcile has ALREADY settled. If
        # reconcile failed transiently this iteration (RPC error -> epoch
        # still SUBMITTED/CONFIRMED) while the claim TX succeeded, defer: leave
        # the epoch open so the next reconcile pass writes the real delta
        # first. The on-chain claim already happened — only the ledger record
        # + alert defer. Marking CLAIMED here would lose the real PnL and
        # fire a bogus "+0.0000" WON. (Reviewer Fix #1.)
        refunds = [e for e in ours if ledger[e].get("status") == "SETTLED_REFUND"]
        wins = [e for e in ours if ledger[e].get("status") == "SETTLED_WON"]
        deferred = [e for e in ours if e not in refunds and e not in wins]
        if deferred:
            warn(
                "ALERT",
                f"claim settled-alert deferred for un-reconciled epochs "
                f"{deferred} (claim succeeded; awaiting reconcile to write delta)",
            )

        # fresh_bankroll is the POST-claim wallet read (winnings/refund now
        # credited) — used for the alert display AND persisted to each CLAIMED
        # record. The settle-time SETTLED_* snapshot is left untouched.
        # A WON/REFUND that corrects a prior DROPPED (TX mined just after the
        # 10s wait, settled later) carries the `(previously reported as
        # DROPPED)` suffix (NIT 1). Scanned from the raw ledger history.
        for e in refunds:
            refund_bnb = float(ledger[e].get("amount_bnb", 0.0) or 0.0)
            send_bet_refund_alert(channel=LIVE_CHANNEL, epoch=e, refund_bnb=refund_bnb,
                                  new_bankroll_bnb=fresh_bankroll,
                                  previously_dropped=bet_ledger.epoch_was_dropped(ledger_path, e))
            bet_ledger.record_claimed(ledger_path=ledger_path, epoch=e, amount_bnb=refund_bnb,
                                      new_bankroll_bnb=fresh_bankroll)

        if len(wins) == 1:
            e = wins[0]
            delta = float(ledger[e].get("delta_bnb", 0.0) or 0.0)
            send_bet_settled_alert(channel=LIVE_CHANNEL, epoch=e, won=True, delta_bnb=delta,
                                   amount_bnb=0.0, new_bankroll_bnb=fresh_bankroll,
                                   previously_dropped=bet_ledger.epoch_was_dropped(ledger_path, e))
            bet_ledger.record_claimed(ledger_path=ledger_path, epoch=e,
                                      new_bankroll_bnb=fresh_bankroll)
        elif len(wins) > 1:
            # Combined alert for the rare multi-epoch batch (Fix #2a).
            total_delta = sum(float(ledger[e].get("delta_bnb", 0.0) or 0.0) for e in wins)
            send_bet_won_batch_alert(channel=LIVE_CHANNEL, epochs=sorted(wins),
                                     total_delta_bnb=total_delta, new_bankroll_bnb=fresh_bankroll)
            for e in wins:
                bet_ledger.record_claimed(ledger_path=ledger_path, epoch=e,
                                          new_bankroll_bnb=fresh_bankroll)
    except Exception as e:  # noqa: BLE001
        warn("ALERT", f"claim settled-alerts failed (epochs={claimed_epochs} err={e})")


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
    claim_tx_receipt_timeout_seconds: int = 10,
    bets_ledger_path: str | None = None,
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
            # Gas-cap sanity check: bail the entire flush if eth.gas_price
            # has run away from MAX_GAS_PRICE_WEI. Claims retry next round
            # naturally; this is lower-severity than the bet-side breach.
            try:
                contract.assert_gas_cap_not_breached()
            except GasPriceCapBreachedError as e:
                warn("CLAIM", f"Skipping {len(to_claim)} claim(s) due to gas cap breach: {e}")
                try:
                    suggested_wei = int(contract.suggest_gas_price_wei())
                except Exception:
                    suggested_wei = -1
                send_gas_cap_breach_alert(
                    path="claim",
                    suggested_wei=suggested_wei,
                    cap_wei=int(MAX_GAS_PRICE_WEI),
                    epochs=to_claim,
                )
                # Leave pending_claims intact so the next iteration's scan
                # retries the same epochs once the operator lifts the cap.
                return
            gas_price_wei = MAX_GAS_PRICE_WEI
            result = contract.claim(
                epochs=to_claim,
                gas_limit=chunk_gas_limit,
                gas_price_wei=gas_price_wei,
                wait_receipt=True,
                receipt_timeout_seconds=int(claim_tx_receipt_timeout_seconds),
            )
            claim_ms = int((time.perf_counter() - t0) * 1000)

            if result.status == "success":
                # ``total_amount_wei`` is summed from the chain's Claim
                # events on the TX receipt; the contract wrapper falls
                # back to ``None`` on decode failure. Render the BNB
                # amount when available; omit gracefully otherwise.
                if result.total_amount_wei is not None:
                    amount_bnb = result.total_amount_wei / BNB_WEI
                    amount_str = f"{amount_bnb:.4f} BNB"
                else:
                    amount_str = "(amount unavailable)"
                _tx_short = _truncate_tx_hash(result.tx_hash)
                if len(to_claim) == 1:
                    info(
                        "CLAIM",
                        f"Claimed {amount_str} from epoch {to_claim[0]} "
                        f"(tx {_tx_short}, {claim_ms}ms)",
                    )
                else:
                    info(
                        "CLAIM",
                        f"Claimed {amount_str} from {len(to_claim)} rounds "
                        f"(epochs {to_claim[0]}-{to_claim[-1]}, "
                        f"tx {_tx_short}, {claim_ms}ms)",
                    )
                # Option B: fire BET WON / BET REFUND alerts at claim-confirm
                # (fresh balance available here). Live-only; the caller passes
                # bets_ledger_path only in live mode.
                if bets_ledger_path is not None:
                    fire_claim_settled_alerts(
                        ledger_path=bets_ledger_path,
                        claimed_epochs=list(to_claim),
                        contract=contract,
                        wallet_address=wallet_address,
                    )
                claimed_total += len(to_claim)
                pending_claims = pending_claims[len(to_claim):]
                continue

            if result.status == "revert":
                if len(to_claim) == 1:
                    epoch_str = f"epoch {to_claim[0]}"
                else:
                    epoch_str = f"{len(to_claim)} rounds (epochs {to_claim[0]}-{to_claim[-1]})"
                warn(
                    "CLAIM",
                    f"Claim TX reverted for {epoch_str} "
                    f"(tx {_truncate_tx_hash(result.tx_hash)}, {claim_ms}ms, "
                    f"block {result.included_block_number})",
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
            if len(to_claim) == 1:
                epoch_str = f"epoch {to_claim[0]}"
            else:
                epoch_str = f"{len(to_claim)} rounds (epochs {to_claim[0]}-{to_claim[-1]})"
            warn(
                "CLAIM",
                f"Claim TX timed out for {epoch_str} "
                f"(tx {_truncate_tx_hash(result.tx_hash)}, {claim_ms}ms, "
                f"receipt_timeout {int(claim_tx_receipt_timeout_seconds)}s)",
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
