"""Tests for the bet lifecycle ledger + reconciliation (Step 31 hotfix taxonomy).

Covers:
  - append/read/replay (last-write-wins), corrupted final line, empty/missing
  - _append_record returns bool; failed persist defers alert + settlement
  - classify_confirmation: CONFIRMED / LATE / REVERTED / DROPPED(timeout)
  - classify_settlement mapping (win/loss/refund -> status, delta)
  - reconcile: LOSS alert fires; WIN/REFUND recorded SILENTLY (Option B)
  - tie = LOST (not refund)
  - LATE / REVERTED are terminal -> never settled
  - SUBMITTED/DROPPED re-check via read_bet_amount (Bundle 4):
      amount==0 -> DROPPED (alert on SUBMITTED->DROPPED; standing DROPPED silent)
      amount>0  -> normal settlement (silent correction of premature DROPPED)
  - permanent un-oracled CONFIRMED -> refund-settles after close_ts + buffer
  - idempotency (no double-settle / double-alert); transient RPC skip
  - sequential-bets: alert uses FRESH bankroll passed in (not stored+delta)
  - claim-path fire_claim_settled_alerts: WON single / REFUND / batch / defer
  - alert senders: 8 locked formats + BOT READY, happy / missing / non-2xx swallow
  - 10s receipt-wait constant; actual gas in ledger; post-claim fresh bankroll
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pancakebot.runtime import bet_ledger  # noqa: E402
from pancakebot.runtime import live as live_mod  # noqa: E402
from pancakebot.chain.prediction_contract import RoundData  # noqa: E402
from pancakebot.util import InvariantError, TransientRpcError  # noqa: E402
from pancakebot.constants import BNB_WEI  # noqa: E402


def _ledger(tmp_path) -> str:
    return str(tmp_path / "bets.jsonl")


class _FakeContract:
    def __init__(self, rounds=None, transient=None, balance=2.0, bet_amounts=None,
                 receipts=None, no_rotate_balance=None, no_rotate_raises=False,
                 balance_raises=False):
        self._rounds = rounds or {}
        self._transient = transient or set()
        self._balance = balance
        # Read-your-writes no-rotate read (BET WON stale-bankroll fix). Defaults
        # to the same balance; tests can make it differ or raise to exercise
        # the non-rotating-primary / rotating-fallback / ledger-snapshot chain.
        self._no_rotate_balance = balance if no_rotate_balance is None else no_rotate_balance
        self._no_rotate_raises = no_rotate_raises
        self._balance_raises = balance_raises
        # epoch -> on-chain registered bet amount (wei). Default: not set ->
        # read_bet_amount returns a large positive (registered) so settlement
        # tests that don't exercise the unregistered path are unaffected.
        self._bet_amounts = bet_amounts or {}
        # tx_hash -> raw receipt mapping for try_get_receipt. Default: not set
        # -> None (no receipt = TX never mined). NIT 2 forensic gas path.
        self._receipts = receipts or {}
        self.calls: list[int] = []

    def round_data(self, epoch):
        self.calls.append(int(epoch))
        if int(epoch) in self._transient:
            raise RuntimeError("transient rpc")
        return self._rounds[int(epoch)]

    def wallet_balance_bnb(self, _addr):
        if self._balance_raises:
            raise TransientRpcError("rotating read failed")
        return self._balance

    def wallet_balance_bnb_no_rotate(self, _addr):
        if self._no_rotate_raises:
            raise TransientRpcError("no_rotate node unreachable")
        return self._no_rotate_balance

    def read_bet_amount(self, epoch, _wallet):
        return int(self._bet_amounts.get(int(epoch), 10**18))  # default registered

    def try_get_receipt(self, tx_hash):
        return self._receipts.get(str(tx_hash))  # None = no receipt (never mined)


def _round(epoch, *, lock_price, close_price, bull_bnb, bear_bnb,
            oracle_called=True, close_ts=1_000_000):
    return RoundData(
        epoch=epoch, start_ts=close_ts - 300, lock_ts=close_ts, close_ts=close_ts,
        lock_price_usd=lock_price, close_price_usd=close_price,
        bull_amount_wei=int(bull_bnb * BNB_WEI), bear_amount_wei=int(bear_bnb * BNB_WEI),
        oracle_called=oracle_called,
    )


def _submit(lp, epoch, side="Bull", amount=0.05, bankroll=2.3):
    bet_ledger.record_submitted(ledger_path=lp, epoch=epoch, side=side,
                                amount_bnb=amount, tx_hash="0xabc", bankroll_after_bnb=bankroll)


def _confirm(lp, epoch):
    """SUBMITTED -> CONFIRMED (status=1, before lock). The normal path for a
    bet that registered cleanly; CONFIRMED settles directly (no recheck)."""
    bet_ledger.record_confirmation(ledger_path=lp, epoch=epoch, chain_status=1,
                                   included_block_number=5, included_late=False,
                                   gas_paid_bnb=0.0001)


# --- append / read / replay -------------------------------------------------

def test_append_returns_true_on_success(tmp_path):
    assert bet_ledger.record_submitted(
        ledger_path=_ledger(tmp_path), epoch=1, side="Bull",
        amount_bnb=0.05, tx_hash="0x", bankroll_after_bnb=2.0) is True


def test_record_submitted_writes_submitted(tmp_path):
    """SUBMITTED is written at TX broadcast (Bundle 2 timing)."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bear", amount=0.05)
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "SUBMITTED"
    assert led[100]["side"] == "Bear"
    assert led[100]["tx_hash"] == "0xabc"


def test_load_roundtrip_and_last_write_wins(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100)
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=1,
                                   included_block_number=555, included_late=False)
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "CONFIRMED"
    assert led[100]["side"] == "Bull"          # merged from SUBMITTED
    assert led[100]["included_block_number"] == 555


def test_missing_and_empty_file(tmp_path):
    assert bet_ledger.load_ledger(str(tmp_path / "nope.jsonl")) == {}
    lp = _ledger(tmp_path); Path(lp).write_text("", encoding="utf-8")
    assert bet_ledger.load_ledger(lp) == {}


def test_corrupted_final_line_skipped(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100)
    with open(lp, "a", encoding="utf-8") as f:
        f.write('{"status":"SUBMITTED","epoch":101,"sid')  # truncated
    led = bet_ledger.load_ledger(lp)
    assert 100 in led and 101 not in led


# --- classify_confirmation --------------------------------------------------

def test_classify_confirmation_matrix():
    c = bet_ledger.classify_confirmation
    assert c(chain_status=1, included_late=False) == "CONFIRMED"
    assert c(chain_status=0, included_late=True) == "LATE"
    assert c(chain_status=0, included_late=False) == "REVERTED"
    assert c(chain_status=None, included_late=False) == "DROPPED"  # timeout
    assert c(chain_status=1, included_late=True) == "LATE"         # anomalous -> LATE


def test_record_confirmation_dropped_on_timeout(tmp_path):
    """No receipt (chain_status None) -> DROPPED record (Bundle 2/3)."""
    lp = _ledger(tmp_path)
    _submit(lp, 100)
    st = bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=None,
                                        included_block_number=None, included_late=False,
                                        gas_paid_bnb=None)
    assert st == "DROPPED"
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "DROPPED"
    assert "gas_paid_bnb" in led[100]               # key ALWAYS present (Option B)
    assert led[100]["gas_paid_bnb"] is None         # null = transient unknown at timeout


def test_record_confirmation_late_is_terminal(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100)
    st = bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=0,
                                        included_block_number=9, included_late=True,
                                        gas_paid_bnb=0.001)
    assert st == "LATE"
    assert bet_ledger.load_ledger(lp)[100]["status"] == "LATE"


# --- classify_settlement ----------------------------------------------------

def test_classify_settlement_mapping():
    s, d = bet_ledger.classify_settlement(outcome="win", bet_bnb=0.05, credit_bnb=0.09)
    assert s == "SETTLED_WON" and abs(d - 0.04) < 1e-9
    s, d = bet_ledger.classify_settlement(outcome="loss", bet_bnb=0.05, credit_bnb=0.0)
    assert s == "SETTLED_LOST" and d == -0.05
    s, d = bet_ledger.classify_settlement(outcome="refund", bet_bnb=0.05, credit_bnb=0.049)
    assert s == "SETTLED_REFUND"


# --- reconcile: Option B alert behavior (CONFIRMED settles directly) -------

def test_reconcile_win_records_silently_no_alert(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=310,
                                          bull_bnb=10, bear_bnb=10)})
    lost_alerts = []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract,
                                   treasury_fee_fraction=0.03, buffer_seconds=30, fresh_bankroll_bnb=2.3,
                                   now_ts=2_000_000,
                                   lost_alert_fn=lambda **kw: lost_alerts.append(kw))
    assert len(settled) == 1 and settled[0]["status"] == "SETTLED_WON"
    assert lost_alerts == []  # WIN is silent here; claim path alerts
    assert bet_ledger.load_ledger(lp)[100]["status"] == "SETTLED_WON"


def test_reconcile_loss_fires_lost_alert(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=290,
                                          bull_bnb=10, bear_bnb=10)})
    lost = []
    bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                         fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                         lost_alert_fn=lambda **kw: lost.append(kw))
    assert len(lost) == 1 and lost[0]["won"] is False
    assert lost[0]["new_bankroll_bnb"] == 2.25
    assert bet_ledger.load_ledger(lp)[100]["status"] == "SETTLED_LOST"


def test_reconcile_tie_is_loss(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    # close == lock, oracle called -> tie -> LOSS (not refund).
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=300,
                                          bull_bnb=10, bear_bnb=10)})
    lost = []
    bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                         fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                         lost_alert_fn=lambda **kw: lost.append(kw))
    assert bet_ledger.load_ledger(lp)[100]["status"] == "SETTLED_LOST"
    assert len(lost) == 1


def test_reconcile_refund_silent(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    # un-oracled, past close_ts + buffer -> refund. Silent (claim path alerts).
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=0,
                                          bull_bnb=10, bear_bnb=10,
                                          oracle_called=False, close_ts=1_000_000)})
    lost = []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract,
                                   treasury_fee_fraction=0.03, buffer_seconds=30, fresh_bankroll_bnb=2.3,
                                   now_ts=2_000_000,
                                   lost_alert_fn=lambda **kw: lost.append(kw))
    assert settled[0]["status"] == "SETTLED_REFUND"
    assert lost == []


# --- SUBMITTED / DROPPED re-check (Bundle 4) -------------------------------

def test_submitted_timeout_then_registered(tmp_path):
    """A still-SUBMITTED entry (crash before classification) whose bet DID
    register on-chain (read_bet_amount > 0) settles normally via pool math."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull")  # stays SUBMITTED (no CONFIRMED transition)
    contract = _FakeContract(
        rounds={100: _round(100, lock_price=300, close_price=290, bull_bnb=10, bear_bnb=10)},
        bet_amounts={100: 5 * 10**16},  # registered
    )
    lost, dropped = [], []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                                   wallet_address="0xme",
                                   lost_alert_fn=lambda **kw: lost.append(kw),
                                   dropped_alert_fn=lambda **kw: dropped.append(kw))
    assert settled[0]["status"] == "SETTLED_LOST"   # normal pool settlement
    assert len(lost) == 1 and dropped == []


def test_submitted_unregistered_becomes_dropped(tmp_path):
    """A still-SUBMITTED entry whose bet NEVER registered (read_bet_amount==0)
    is marked terminal DROPPED, fires BET DROPPED, and is NOT pool-settled."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull")
    contract = _FakeContract(
        rounds={100: _round(100, lock_price=300, close_price=310, bull_bnb=10, bear_bnb=10)},
        bet_amounts={100: 0},  # never registered
    )
    dropped, lost = [], []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.30, now_ts=2_000_000,
                                   wallet_address="0xme",
                                   lost_alert_fn=lambda **kw: lost.append(kw),
                                   dropped_alert_fn=lambda **kw: dropped.append(kw))
    assert settled[0]["status"] == "DROPPED"
    assert dropped and dropped[0]["epoch"] == 100
    assert dropped[0]["bankroll_bnb"] == 2.30
    assert lost == []                                       # NOT pool-settled
    assert contract.calls == []                             # round_data never read
    assert bet_ledger.load_ledger(lp)[100]["status"] == "DROPPED"


def test_dropped_stays_silent_when_unregistered(tmp_path):
    """A standing DROPPED whose bet is still unregistered (no receipt) stays
    DROPPED and fires NO Discord alert (the DROPPED alert already fired at the
    receipt timeout). Its gas is resolved from null to 0 ONCE (Option B); a
    second pass is fully idempotent."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull")
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=None,
                                   included_block_number=None, included_late=False)
    assert bet_ledger.load_ledger(lp)[100]["status"] == "DROPPED"
    assert bet_ledger.load_ledger(lp)[100]["gas_paid_bnb"] is None   # unknown at timeout
    contract = _FakeContract(
        rounds={100: _round(100, lock_price=300, close_price=310, bull_bnb=10, bear_bnb=10)},
        bet_amounts={100: 0},  # still unregistered
        receipts={},           # no receipt -> never mined
    )
    dropped = []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.30, now_ts=2_000_000,
                                   wallet_address="0xme",
                                   dropped_alert_fn=lambda **kw: dropped.append(kw))
    assert dropped == []                                    # no Discord re-alert
    assert contract.calls == []                             # round_data never read
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "DROPPED"
    assert led[100]["gas_paid_bnb"] == 0                    # resolved null -> 0 (Option B)
    assert settled and settled[0]["outcome"] == "dropped_gas_resolved"
    # Idempotent: gas now concrete -> second pass does nothing.
    settled2 = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                    buffer_seconds=30, fresh_bankroll_bnb=2.30, now_ts=2_000_000,
                                    wallet_address="0xme", dropped_alert_fn=lambda **kw: dropped.append(kw))
    assert settled2 == [] and dropped == []


def test_dropped_silently_corrected_when_amount_nonzero(tmp_path):
    """Bundle 4: a DROPPED entry whose bet DID register (mined just after the
    10s timeout) settles normally — the settlement alert is the implicit
    correction; NO explicit 'we were wrong' alert, NO duplicate DROPPED."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull")
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=None,
                                   included_block_number=None, included_late=False)
    assert bet_ledger.load_ledger(lp)[100]["status"] == "DROPPED"
    contract = _FakeContract(
        rounds={100: _round(100, lock_price=300, close_price=290, bull_bnb=10, bear_bnb=10)},
        bet_amounts={100: 5 * 10**16},  # actually registered after all
    )
    lost, dropped = [], []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                                   wallet_address="0xme",
                                   lost_alert_fn=lambda **kw: lost.append(kw),
                                   dropped_alert_fn=lambda **kw: dropped.append(kw))
    assert settled[0]["status"] == "SETTLED_LOST"   # silent correction -> settles
    assert len(lost) == 1                            # settlement alert IS the correction
    assert dropped == []                             # no explicit "wrong" alert
    assert bet_ledger.load_ledger(lp)[100]["status"] == "SETTLED_LOST"


def test_submitted_recheck_requires_wallet(tmp_path):
    """Without a wallet_address the re-check can't run; the SUBMITTED entry is
    left untouched (never fabricate a settlement)."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull")
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=290,
                                          bull_bnb=10, bear_bnb=10)})
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                                   wallet_address=None, lost_alert_fn=None)
    assert settled == []
    assert contract.calls == []                             # round_data never read
    assert bet_ledger.load_ledger(lp)[100]["status"] == "SUBMITTED"


def test_confirmed_skips_recheck(tmp_path):
    """A CONFIRMED entry (receipt was good) does NOT trigger the on-chain
    read_bet_amount check — it settles directly via pool math."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    # bet_amounts says 0, but CONFIRMED should NOT consult it.
    contract = _FakeContract(
        rounds={100: _round(100, lock_price=300, close_price=290, bull_bnb=10, bear_bnb=10)},
        bet_amounts={100: 0},
    )
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                                   wallet_address="0xme", lost_alert_fn=lambda **kw: None,
                                   dropped_alert_fn=lambda **kw: None)
    assert settled[0]["status"] == "SETTLED_LOST"   # pool-settled, recheck skipped


# --- LATE / REVERTED must NOT settle ---------------------------------------

def test_late_never_settles(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100)
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=0,
                                   included_block_number=9, included_late=True)
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=310,
                                          bull_bnb=10, bear_bnb=10)})
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract,
                                   treasury_fee_fraction=0.03, buffer_seconds=30, fresh_bankroll_bnb=2.3,
                                   now_ts=2_000_000, lost_alert_fn=None)
    assert settled == []                       # LATE is terminal — not reconciled
    assert contract.calls == []                # never even read round_data
    assert bet_ledger.load_ledger(lp)[100]["status"] == "LATE"


def test_reverted_never_settles(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100)
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=0,
                                   included_block_number=9, included_late=False)
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=310,
                                          bull_bnb=10, bear_bnb=10)})
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract,
                                   treasury_fee_fraction=0.03, buffer_seconds=30, fresh_bankroll_bnb=2.3,
                                   now_ts=2_000_000, lost_alert_fn=None)
    assert settled == []
    assert bet_ledger.load_ledger(lp)[100]["status"] == "REVERTED"


def test_permanent_unoracled_confirmed_refunds_after_close(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    # never oracled; close_ts in the past relative to now_ts -> refund-settles.
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=0,
                                          bull_bnb=10, bear_bnb=10,
                                          oracle_called=False, close_ts=1_000_000)})
    # Before close: skipped.
    s1 = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                              fresh_bankroll_bnb=2.3, now_ts=999_000, lost_alert_fn=None)
    assert s1 == [] and bet_ledger.load_ledger(lp)[100]["status"] == "CONFIRMED"
    # After close: refund-settles.
    s2 = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                              fresh_bankroll_bnb=2.3, now_ts=2_000_000, lost_alert_fn=None)
    assert s2[0]["status"] == "SETTLED_REFUND"


# --- idempotency / transient / append-gating -------------------------------

def test_reconcile_idempotent(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=290,
                                          bull_bnb=10, bear_bnb=10)})
    lost = []
    af = lambda **kw: lost.append(kw)
    first = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                                 fresh_bankroll_bnb=2.25, now_ts=2_000_000, lost_alert_fn=af)
    second = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                                  fresh_bankroll_bnb=2.25, now_ts=2_000_000, lost_alert_fn=af)
    assert len(first) == 1 and second == []
    assert len(lost) == 1  # alert fired exactly once


def test_reconcile_transient_rpc_skips(tmp_path):
    lp = _ledger(tmp_path)
    _submit(lp, 100); _confirm(lp, 100)
    contract = _FakeContract({}, transient={100})
    assert bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                                fresh_bankroll_bnb=2.3, now_ts=2_000_000, lost_alert_fn=None) == []
    assert bet_ledger.load_ledger(lp)[100]["status"] == "CONFIRMED"


def test_failed_persist_defers_alert_and_settlement(tmp_path):
    """Fix #7: if the terminal append fails, neither the alert nor the
    settled-list entry happen — both defer to the next pass."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=290,
                                          bull_bnb=10, bear_bnb=10)})
    lost = []
    with mock.patch("pancakebot.runtime.bet_ledger.record_settled", return_value=False):
        settled = bet_ledger.reconcile(ledger_path=lp, contract=contract,
                                       treasury_fee_fraction=0.03, buffer_seconds=30, fresh_bankroll_bnb=2.25,
                                       now_ts=2_000_000, lost_alert_fn=lambda **kw: lost.append(kw))
    assert settled == []       # not appended
    assert lost == []          # not alerted
    # Ledger still open -> next pass retries.
    assert bet_ledger.load_ledger(lp)[100]["status"] == "CONFIRMED"


def test_sequential_bets_use_fresh_bankroll(tmp_path):
    """Fix #3: two bets back-to-back; settling bet #1 shows the FRESH bankroll
    passed in (which already reflects bet #2's debit), not stored + delta."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", bankroll=2.30); _confirm(lp, 100)  # after bet #1
    _submit(lp, 101, side="Bull", bankroll=2.25); _confirm(lp, 101)  # after bet #2
    # Bet #1 (epoch 100) loses; bet #2 (101) not yet closed.
    contract = _FakeContract({
        100: _round(100, lock_price=300, close_price=290, bull_bnb=10, bear_bnb=10),
        101: _round(101, lock_price=300, close_price=0, bull_bnb=10, bear_bnb=10,
                    oracle_called=False, close_ts=9_000_000),  # not closed
    })
    lost = []
    # Caller passes the FRESH current balance (reflects BOTH placements).
    bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                         fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                         lost_alert_fn=lambda **kw: lost.append(kw))
    assert len(lost) == 1 and lost[0]["epoch"] == 100
    # Fresh balance shown, NOT stored bankroll(2.30) + delta.
    assert lost[0]["new_bankroll_bnb"] == 2.25


# --- claim-path WON / REFUND firing ----------------------------------------

def test_fire_claim_won_single(tmp_path, monkeypatch):
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", amount=0.05)
    # reconcile marked it SETTLED_WON with a delta.
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_WON",
                              delta_bnb=0.04, outcome="win", new_bankroll_bnb=2.34)
    sent = []
    monkeypatch.setattr(live_mod, "send_bet_settled_alert", lambda **kw: sent.append(("won", kw)))
    contract = _FakeContract(balance=2.39)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    assert len(sent) == 1 and sent[0][1]["won"] is True
    assert sent[0][1]["delta_bnb"] == 0.04
    assert sent[0][1]["new_bankroll_bnb"] == 2.39        # fresh wallet balance
    assert "claim_gas_bnb" not in sent[0][1]             # gas dropped from display
    assert bet_ledger.load_ledger(lp)[100]["status"] == "CLAIMED"


def test_fire_claim_won_uses_no_rotate_value_not_rotating(tmp_path, monkeypatch):
    """Read-your-writes: the PRIMARY read is the non-rotating one. no_rotate
    (fresh, confirming node) = 2.40; rotating (stale sibling) = 2.34. The alert
    must show 2.40, proving the fix uses the non-rotating read."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", amount=0.05)
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_WON",
                              delta_bnb=0.04, outcome="win", new_bankroll_bnb=2.30)
    sent = []
    monkeypatch.setattr(live_mod, "send_bet_settled_alert", lambda **kw: sent.append(kw))
    contract = _FakeContract(balance=2.34, no_rotate_balance=2.40)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    assert len(sent) == 1
    assert sent[0]["new_bankroll_bnb"] == 2.40   # fresh no-rotate value, NOT stale 2.34


def test_fire_claim_won_falls_back_to_rotating_when_no_rotate_fails(tmp_path, monkeypatch):
    """If the non-rotating read throws (confirming node briefly unreachable),
    fall back to the rotating read."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", amount=0.05)
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_WON",
                              delta_bnb=0.04, outcome="win", new_bankroll_bnb=2.30)
    sent = []
    monkeypatch.setattr(live_mod, "send_bet_settled_alert", lambda **kw: sent.append(kw))
    contract = _FakeContract(balance=2.39, no_rotate_raises=True)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    assert len(sent) == 1
    assert sent[0]["new_bankroll_bnb"] == 2.39   # rotating fallback value


def test_fire_claim_won_falls_back_to_ledger_snapshot_when_both_fail(tmp_path, monkeypatch):
    """If BOTH reads throw (total RPC failure), fall back to the ledger snapshot
    and WARN."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", amount=0.05)
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_WON",
                              delta_bnb=0.04, outcome="win", new_bankroll_bnb=2.34)
    sent = []
    warns = []
    monkeypatch.setattr(live_mod, "send_bet_settled_alert", lambda **kw: sent.append(kw))
    monkeypatch.setattr(live_mod, "warn", lambda action, msg: warns.append((action, msg)))
    contract = _FakeContract(no_rotate_raises=True, balance_raises=True)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    assert len(sent) == 1
    assert sent[0]["new_bankroll_bnb"] == 2.34   # ledger snapshot (SETTLED_WON new_bankroll_bnb)
    assert any("post-claim balance read failed" in m for _, m in warns)


def test_fire_claim_refund(tmp_path, monkeypatch):
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", amount=0.05)
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_REFUND",
                              delta_bnb=-0.001, outcome="refund", new_bankroll_bnb=2.30)
    sent = []
    monkeypatch.setattr(live_mod, "send_bet_refund_alert", lambda **kw: sent.append(kw))
    contract = _FakeContract(balance=2.30)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    assert len(sent) == 1 and sent[0]["epoch"] == 100
    assert bet_ledger.load_ledger(lp)[100]["status"] == "CLAIMED"


def test_fire_claim_batch_combined(tmp_path, monkeypatch):
    lp = _ledger(tmp_path)
    for ep, d in ((100, 0.04), (101, 0.03)):
        _submit(lp, ep, side="Bull", amount=0.05)
        bet_ledger.record_settled(ledger_path=lp, epoch=ep, side="Bull", status="SETTLED_WON",
                                  delta_bnb=d, outcome="win", new_bankroll_bnb=2.4)
    batch = []
    monkeypatch.setattr(live_mod, "send_bet_won_batch_alert", lambda **kw: batch.append(kw))
    contract = _FakeContract(balance=2.45)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100, 101],
                                       contract=contract, wallet_address="0xme")
    assert len(batch) == 1
    assert abs(batch[0]["total_delta_bnb"] - 0.07) < 1e-9
    assert bet_ledger.load_ledger(lp)[100]["status"] == "CLAIMED"
    assert bet_ledger.load_ledger(lp)[101]["status"] == "CLAIMED"


def test_fire_claim_unreconciled_epoch_defers(tmp_path, monkeypatch):
    """Reviewer Fix #1: a claimed epoch still SUBMITTED/CONFIRMED (reconcile
    failed transiently this iteration) must NOT fire a premature WON or mark
    CLAIMED — it stays open until the next reconcile writes the real delta."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", amount=0.05); _confirm(lp, 100)  # CONFIRMED, never reconciled
    won = []
    monkeypatch.setattr(live_mod, "send_bet_settled_alert", lambda **kw: won.append(kw))
    contract = _FakeContract(balance=2.39)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    assert won == []                                              # no premature WON
    assert bet_ledger.load_ledger(lp)[100]["status"] == "CONFIRMED"  # stays open


def test_reconcile_defers_in_buffer_window(tmp_path):
    """Reviewer Fix #3: an oracle-pending round inside [close_ts,
    close_ts+buffer] must NOT settle as refund — defer until past buffer."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=0,
                                          bull_bnb=10, bear_bnb=10,
                                          oracle_called=False, close_ts=1_000_000)})
    # now inside [close_ts, close_ts+buffer]: 1_000_000 < now < 1_000_030.
    s = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                             buffer_seconds=30, fresh_bankroll_bnb=2.3, now_ts=1_000_015,
                             lost_alert_fn=None)
    assert s == []                                              # deferred
    assert bet_ledger.load_ledger(lp)[100]["status"] == "CONFIRMED"
    # Past buffer: now > close_ts + buffer -> refund-settles.
    s2 = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                              buffer_seconds=30, fresh_bankroll_bnb=2.3, now_ts=1_000_031,
                              lost_alert_fn=None)
    assert s2[0]["status"] == "SETTLED_REFUND"


def test_reconcile_defers_at_exact_buffer_boundary(tmp_path):
    """Reviewer NIT: `>=` gate matches PCS `_refundable()` strict `>`.
    AT now == close_ts + buffer the round is NOT yet refundable (defer);
    at now == close_ts + buffer + 1 it settles."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull"); _confirm(lp, 100)
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=0,
                                          bull_bnb=10, bear_bnb=10,
                                          oracle_called=False, close_ts=1_000_000)})
    # Exact boundary: now == close_ts + buffer == 1_000_030 -> defer (>= gate).
    s_boundary = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                      buffer_seconds=30, fresh_bankroll_bnb=2.3, now_ts=1_000_030,
                                      lost_alert_fn=None)
    assert s_boundary == []
    assert bet_ledger.load_ledger(lp)[100]["status"] == "CONFIRMED"
    # One second past the boundary -> settles.
    s_past = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                  buffer_seconds=30, fresh_bankroll_bnb=2.3, now_ts=1_000_031,
                                  lost_alert_fn=None)
    assert s_past[0]["status"] == "SETTLED_REFUND"


def test_fire_claim_skips_non_ledgered(tmp_path, monkeypatch):
    lp = _ledger(tmp_path)
    sent = []
    monkeypatch.setattr(live_mod, "send_bet_settled_alert", lambda **kw: sent.append(kw))
    contract = _FakeContract(balance=2.0)
    # Epoch 999 not in our ledger (legacy/manual claim) -> no alert.
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[999],
                                       contract=contract, wallet_address="0xme")
    assert sent == []


# --- alert senders: 8 locked formats + BOT READY ---------------------------

class _FakeResp:
    def __init__(self, code): self.status_code = code; self.text = ""


class _FakeRequests:
    def __init__(self, code=204): self._code = code; self.captured = {}
    def post(self, url, *, json, timeout):
        self.captured = {"url": url, "json": json}
        return _FakeResp(self._code)


def _content(fake):
    return fake.captured["json"]["content"]


def _username(fake):
    return fake.captured["json"]["username"]


def _wire(monkeypatch, code=204):
    fake = _FakeRequests(code)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setenv("PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL", "https://d/dry")
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


# The locked message bodies. Each mode carries its channel prefix: "[LIVE] "
# for live, "[DRY] " for dry. Body after the prefix is byte-identical across
# modes (asserted below).
def test_bot_ready_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bot_ready_alert(channel=live_mod.LIVE_CHANNEL, bankroll_bnb=2.345)
    assert _content(fake) == (
        "[LIVE] [INFO] **BOT READY** `PancakeBot-live` — bankroll `2.3450` BNB"
    )
    assert _username(fake) == "PancakeBot-live"


def test_submitted_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_submitted_alert(channel=live_mod.LIVE_CHANNEL, epoch=999, side="Bull",
                                      amount_bnb=0.05, projected_bankroll_bnb=2.3463)
    assert _content(fake) == (
        "[LIVE] [INFO] **BET SUBMITTED** epoch `999` — Bet `0.0500` BNB on Bull, "
        "projected bankroll `2.3463` BNB"
    )


def test_confirmed_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_confirmed_alert(channel=live_mod.LIVE_CHANNEL, epoch=999, bankroll_bnb=2.2950)
    assert _content(fake) == "[LIVE] [INFO] **BET CONFIRMED** epoch `999` — bankroll `2.2950` BNB"


def test_late_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_late_alert(channel=live_mod.LIVE_CHANNEL, epoch=999, bankroll_bnb=2.3452)
    assert _content(fake) == "[LIVE] [WARN] **BET LATE** epoch `999` — bankroll `2.3452` BNB"


def test_reverted_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_reverted_alert(channel=live_mod.LIVE_CHANNEL, epoch=999, bankroll_bnb=2.3452)
    assert _content(fake) == "[LIVE] [WARN] **BET REVERTED** epoch `999` — bankroll `2.3452` BNB"


def test_dropped_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_dropped_alert(channel=live_mod.LIVE_CHANNEL, epoch=999, bankroll_bnb=2.3452)
    assert _content(fake) == "[LIVE] [WARN] **BET DROPPED** epoch `999` — bankroll `2.3452` BNB"


def test_won_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_settled_alert(channel=live_mod.LIVE_CHANNEL, epoch=999, won=True,
                                    delta_bnb=0.0423, amount_bnb=0.05, new_bankroll_bnb=2.3886)
    assert _content(fake) == (
        "[LIVE] [INFO] **BET WON** epoch `999` — Won `0.0423` BNB, bankroll `2.3886` BNB"
    )


def test_lost_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_settled_alert(channel=live_mod.LIVE_CHANNEL, epoch=999, won=False,
                                    delta_bnb=-0.05, amount_bnb=0.05, new_bankroll_bnb=2.2963)
    # Lost amount shown positive; the verb conveys direction.
    assert _content(fake) == (
        "[LIVE] [INFO] **BET LOST** epoch `999` — Lost `0.0500` BNB, bankroll `2.2963` BNB"
    )


def test_refund_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_refund_alert(channel=live_mod.LIVE_CHANNEL, epoch=999, refund_bnb=0.05,
                                   new_bankroll_bnb=2.3455)
    assert _content(fake) == (
        "[LIVE] [INFO] **BET REFUND** epoch `999` — Refunded `0.0500` BNB, bankroll `2.3455` BNB"
    )


def test_won_batch_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_won_batch_alert(channel=live_mod.LIVE_CHANNEL, epochs=[101, 102, 103],
                                      total_delta_bnb=0.12, new_bankroll_bnb=2.50)
    assert _content(fake) == (
        "[LIVE] [INFO] **BET WON** epochs `[101, 102, 103]` — Won `0.1200` BNB total, "
        "bankroll `2.5000` BNB"
    )


# --- dry/live parity: identical body, [DRY] prefix, PancakeBot-dry username --
def test_dry_submitted_is_live_body_with_dry_prefix(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_submitted_alert(channel=live_mod.DRY_CHANNEL, epoch=999, side="Bull",
                                      amount_bnb=0.05, projected_bankroll_bnb=2.3463)
    live_body = (
        "[INFO] **BET SUBMITTED** epoch `999` — Bet `0.0500` BNB on Bull, "
        "projected bankroll `2.3463` BNB"
    )
    assert _content(fake) == "[DRY] " + live_body
    assert _username(fake) == "PancakeBot-dry"
    assert fake.captured["url"] == "https://d/dry"   # dry webhook, not live


def test_dry_won_is_live_body_with_dry_prefix(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_settled_alert(channel=live_mod.DRY_CHANNEL, epoch=999, won=True,
                                    delta_bnb=0.0423, amount_bnb=0.05, new_bankroll_bnb=2.3886)
    assert _content(fake) == (
        "[DRY] [INFO] **BET WON** epoch `999` — Won `0.0423` BNB, bankroll `2.3886` BNB"
    )
    assert _username(fake) == "PancakeBot-dry"


def test_dry_lost_is_live_body_with_dry_prefix(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_settled_alert(channel=live_mod.DRY_CHANNEL, epoch=999, won=False,
                                    delta_bnb=-0.05, amount_bnb=0.05, new_bankroll_bnb=2.2963)
    assert _content(fake) == (
        "[DRY] [INFO] **BET LOST** epoch `999` — Lost `0.0500` BNB, bankroll `2.2963` BNB"
    )


def test_dry_refund_is_live_body_with_dry_prefix(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_bet_refund_alert(channel=live_mod.DRY_CHANNEL, epoch=999, refund_bnb=0.05,
                                   new_bankroll_bnb=2.3455)
    assert _content(fake) == (
        "[DRY] [INFO] **BET REFUND** epoch `999` — Refunded `0.0500` BNB, bankroll `2.3455` BNB"
    )


def test_dry_and_live_bodies_identical_modulo_prefix(monkeypatch):
    """Dry and live share an identical BODY; the only difference is the channel
    prefix (+ webhook/username). Stripping each prefix yields the same body."""
    fake = _wire(monkeypatch)
    live_mod.send_bet_settled_alert(channel=live_mod.LIVE_CHANNEL, epoch=42, won=True,
                                    delta_bnb=0.01, amount_bnb=0.05, new_bankroll_bnb=3.0)
    live_content = _content(fake)
    live_mod.send_bet_settled_alert(channel=live_mod.DRY_CHANNEL, epoch=42, won=True,
                                    delta_bnb=0.01, amount_bnb=0.05, new_bankroll_bnb=3.0)
    dry_content = _content(fake)
    live_body = live_content[len(live_mod.LIVE_CHANNEL.prefix):]
    dry_body = dry_content[len(live_mod.DRY_CHANNEL.prefix):]
    assert live_body == dry_body                     # identical body across modes
    assert live_content.startswith("[LIVE] ")
    assert dry_content.startswith("[DRY] ")


# --- D3: COOLDOWN ENTERED / LIFTED alerts ----------------------------------
def test_cooldown_entered_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_cooldown_entered_alert(
        channel=live_mod.LIVE_CHANNEL, drawdown_pct=19.2, threshold_pct=15.0,
        bankroll_bnb=4.3581, cooldown_rounds=72, approx_hours=6.0,
    )
    assert _content(fake) == (
        "[LIVE] [WARN] **COOLDOWN ENTERED** — drawdown `19.2%`, threshold `15%`, "
        "bankroll `4.3581` BNB, `72` rounds (~`6.0h`)"
    )


def test_cooldown_lifted_format_exact(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_cooldown_lifted_alert(channel=live_mod.LIVE_CHANNEL, bankroll_bnb=5.0)
    assert _content(fake) == (
        "[LIVE] [INFO] **COOLDOWN LIFTED** — bankroll `5.0000` BNB, betting resumes"
    )


def test_cooldown_dry_prefix_and_channel(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_cooldown_entered_alert(
        channel=live_mod.DRY_CHANNEL, drawdown_pct=19.2, threshold_pct=15.0,
        bankroll_bnb=4.3581, cooldown_rounds=72, approx_hours=6.0,
    )
    assert _content(fake).startswith("[DRY] [WARN] **COOLDOWN ENTERED**")
    assert _username(fake) == "PancakeBot-dry"
    assert fake.captured["url"] == "https://d/dry"


def test_no_gas_in_any_alert(monkeypatch):
    """Locked format drops all gas values from Discord display."""
    fake = _wire(monkeypatch)
    live_mod.send_bet_submitted_alert(channel=live_mod.LIVE_CHANNEL, epoch=1, side="Bear",
                                      amount_bnb=0.05, projected_bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()
    live_mod.send_bet_confirmed_alert(channel=live_mod.LIVE_CHANNEL, epoch=1, bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()
    live_mod.send_bet_late_alert(channel=live_mod.LIVE_CHANNEL, epoch=1, bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()
    live_mod.send_bet_dropped_alert(channel=live_mod.LIVE_CHANNEL, epoch=1, bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()
    live_mod.send_bet_settled_alert(channel=live_mod.LIVE_CHANNEL, epoch=1, won=True,
                                    delta_bnb=0.04, amount_bnb=0.05, new_bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()
    live_mod.send_bet_refund_alert(channel=live_mod.LIVE_CHANNEL, epoch=1, refund_bnb=0.05,
                                   new_bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()


def test_no_amount_or_side_on_post_receipt_alerts(monkeypatch):
    """CONFIRMED / LATE / REVERTED / DROPPED carry only epoch + bankroll —
    bet amount + side live on BET SUBMITTED (Bundle 2 locked format)."""
    fake = _wire(monkeypatch)
    for fn in (live_mod.send_bet_confirmed_alert, live_mod.send_bet_late_alert,
               live_mod.send_bet_reverted_alert, live_mod.send_bet_dropped_alert):
        fn(channel=live_mod.LIVE_CHANNEL, epoch=5, bankroll_bnb=2.0)
        c = _content(fake)
        assert "Bull" not in c and "Bear" not in c
        assert "Bet `" not in c


def test_senders_missing_webhook_no_raise(monkeypatch):
    """Both modes fail-soft when their webhook env is unset (no raise, no post)."""
    monkeypatch.delenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL", raising=False)
    for channel in (live_mod.LIVE_CHANNEL, live_mod.DRY_CHANNEL):
        live_mod.send_bot_ready_alert(channel=channel, bankroll_bnb=2.0)
        live_mod.send_bet_submitted_alert(channel=channel, epoch=1, side="Bull",
                                          amount_bnb=0.05, projected_bankroll_bnb=2.0)
        live_mod.send_bet_confirmed_alert(channel=channel, epoch=1, bankroll_bnb=2.0)
        live_mod.send_bet_late_alert(channel=channel, epoch=1, bankroll_bnb=2.0)
        live_mod.send_bet_reverted_alert(channel=channel, epoch=1, bankroll_bnb=2.0)
        live_mod.send_bet_dropped_alert(channel=channel, epoch=1, bankroll_bnb=2.0)
        live_mod.send_bet_settled_alert(channel=channel, epoch=1, won=False, delta_bnb=-0.05,
                                        amount_bnb=0.05, new_bankroll_bnb=2.0)
        live_mod.send_bet_refund_alert(channel=channel, epoch=1, refund_bnb=0.05, new_bankroll_bnb=2.3)


def test_sender_non_2xx_swallowed(monkeypatch):
    _wire(monkeypatch, code=500)
    live_mod.send_bet_submitted_alert(channel=live_mod.LIVE_CHANNEL, epoch=1, side="Bull",
                                      amount_bnb=0.05, projected_bankroll_bnb=2.0)


# --- Step 31 hotfix: 10s receipt wait, actual gas, post-claim fresh bankroll -

def test_tx_receipt_wait_10s_constant():
    """The TX receipt wait is a single fixed 10s constant, shared by the bet
    and claim paths and decoupled from the refund-eligibility math."""
    from pancakebot.timing_constants import TX_RECEIPT_WAIT_TIMEOUT_SECONDS
    assert TX_RECEIPT_WAIT_TIMEOUT_SECONDS == 10


def test_won_alert_uses_post_claim_fresh_bankroll(tmp_path, monkeypatch):
    """Bundle 5: BET WON alert + the CLAIMED record use a fresh POST-claim
    wallet read; the SETTLED_WON settle-time snapshot is preserved distinct."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", amount=0.001)
    # reconcile wrote SETTLED_WON at settle-time (pre-claim, understated).
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_WON",
                              delta_bnb=0.000317, outcome="win", new_bankroll_bnb=2.344724)
    sent = []
    monkeypatch.setattr(live_mod, "send_bet_settled_alert", lambda **kw: sent.append(kw))
    # Post-claim wallet now reflects the credited winnings.
    contract = _FakeContract(balance=2.346197)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    # Alert shows the post-claim fresh read, not the settle-time snapshot.
    assert sent[0]["new_bankroll_bnb"] == 2.346197
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "CLAIMED"
    assert led[100]["new_bankroll_bnb"] == 2.346197       # CLAIMED = post-claim
    # SETTLED_WON snapshot preserved in the append-only log (not overwritten).
    import json as _json
    settled_snaps = [
        _json.loads(ln)["new_bankroll_bnb"]
        for ln in open(lp, encoding="utf-8") if '"SETTLED_WON"' in ln
    ]
    assert settled_snaps == [2.344724]                    # distinct settle-time value


def test_lost_alert_uses_fresh_wallet_read_not_arithmetic(tmp_path, monkeypatch):
    """Bundle 5 (LOSS parity): the BET LOST alert bankroll AND the SETTLED_LOST
    ledger record's new_bk both come VERBATIM from the fresh wallet read the
    caller passes as fresh_bankroll_bnb — never from arithmetic
    (estimate - delta / bk_after - delta). Mirrors _reconcile_live_bets, which
    reads wallet_balance_bnb() and forwards it unchanged. The sentinel balance
    (999.9999) is unrelated to ANY 2.x projection, so its presence proves the
    fresh read is used; a regression to arithmetic would surface ~2.25 instead."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", bankroll=2.30); _confirm(lp, 100)
    # Bull bet, price fell (310 -> 290 vs lock 300) -> LOSS.
    contract = _FakeContract(
        {100: _round(100, lock_price=300, close_price=290, bull_bnb=10, bear_bnb=10)},
        balance=999.9999,  # sentinel fresh read, unrelated to any arithmetic
    )
    fake = _wire(monkeypatch)  # real send_bet_settled_alert -> captured POST
    # Caller reads the fresh wallet balance and forwards it (exactly what
    # _reconcile_live_bets does); reconcile must NOT recompute it.
    fresh = float(contract.wallet_balance_bnb("0xme"))
    bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                         buffer_seconds=30, fresh_bankroll_bnb=fresh, now_ts=2_000_000,
                         wallet_address="0xme",
                         lost_alert_fn=lambda **kw: live_mod.send_bet_settled_alert(
                             channel=live_mod.LIVE_CHANNEL, **kw),
                         dropped_alert_fn=None)
    # 1) LOSS alert string carries the fresh read verbatim.
    c = _content(fake)
    assert c == (
        "[LIVE] [INFO] **BET LOST** epoch `100` — Lost `0.0500` BNB, bankroll `999.9999` BNB"
    )
    assert "2.25" not in c and "2.30" not in c             # no arithmetic projection
    # 2) SETTLED_LOST ledger record new_bk == fresh read (not arithmetic).
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "SETTLED_LOST"
    assert led[100]["new_bankroll_bnb"] == 999.9999


def test_record_confirmation_stores_actual_gas(tmp_path):
    """Bundle 6: record_confirmation stores the actual gas (gasUsed x price),
    not the MAX_GAS_COST_BET_BNB cap."""
    gas = bet_ledger.actual_gas_bnb(gas_used=100_000, effective_gas_price_wei=10**9)
    assert gas == 0.0001
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull")
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=1,
                                   included_block_number=5, included_late=False,
                                   gas_paid_bnb=gas)
    assert bet_ledger.load_ledger(lp)[100]["gas_paid_bnb"] == 0.0001  # not 0.0002 cap


def test_actual_gas_none_on_timeout():
    """No receipt (timeout) -> None gas, so the ledger field stays unwritten."""
    assert bet_ledger.actual_gas_bnb(gas_used=None, effective_gas_price_wei=10**9) is None
    assert bet_ledger.actual_gas_bnb(gas_used=100_000, effective_gas_price_wei=None) is None


# --- NIT 1: settlement "previously reported as DROPPED" correction suffix ----

_SUFFIX = " (previously reported as DROPPED)"


def _make_dropped(lp, epoch, side="Bull", amount=0.05, bankroll=2.30):
    """Submit then DROP an epoch (receipt timeout) so its history carries a
    DROPPED record — the precondition for the correction suffix."""
    _submit(lp, epoch, side=side, amount=amount, bankroll=bankroll)
    bet_ledger.record_confirmation(ledger_path=lp, epoch=epoch, chain_status=None,
                                   included_block_number=None, included_late=False)


def test_won_alert_correction_suffix_after_drop(tmp_path, monkeypatch):
    """A WON that corrects a prior DROPPED appends the suffix (claim path)."""
    lp = _ledger(tmp_path)
    _make_dropped(lp, 100, side="Bull", amount=0.05)
    # reconcile later corrected DROPPED -> SETTLED_WON (its delta written).
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_WON",
                              delta_bnb=0.04, outcome="win", new_bankroll_bnb=2.34)
    fake = _wire(monkeypatch)  # real send_bet_settled_alert -> captured POST
    contract = _FakeContract(balance=2.39)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    c = _content(fake)
    assert c == (
        "[LIVE] [INFO] **BET WON** epoch `100` — Won `0.0400` BNB, bankroll `2.3900` BNB" + _SUFFIX
    )


def test_lost_alert_correction_suffix_after_drop(tmp_path, monkeypatch):
    """A LOST that corrects a prior DROPPED appends the suffix (reconcile path).
    The DROPPED entry's TX in fact registered (read_bet_amount > 0) -> reconcile
    reclassifies + settles LOST, and epoch_was_dropped sees the prior DROPPED."""
    lp = _ledger(tmp_path)
    _make_dropped(lp, 100, side="Bull", amount=0.05)
    contract = _FakeContract(
        {100: _round(100, lock_price=300, close_price=290, bull_bnb=10, bear_bnb=10)},
        balance=2.2949,                 # fresh read
        bet_amounts={100: 5 * 10**16},  # registered after all
    )
    fake = _wire(monkeypatch)
    fresh = float(contract.wallet_balance_bnb("0xme"))
    bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                         buffer_seconds=30, fresh_bankroll_bnb=fresh, now_ts=2_000_000,
                         wallet_address="0xme",
                         lost_alert_fn=lambda **kw: live_mod.send_bet_settled_alert(
                             channel=live_mod.LIVE_CHANNEL, **kw),
                         dropped_alert_fn=None)
    c = _content(fake)
    assert c == (
        "[LIVE] [INFO] **BET LOST** epoch `100` — Lost `0.0500` BNB, bankroll `2.2949` BNB" + _SUFFIX
    )


def test_refund_alert_correction_suffix_after_drop(tmp_path, monkeypatch):
    """A REFUND that corrects a prior DROPPED appends the suffix (claim path)."""
    lp = _ledger(tmp_path)
    _make_dropped(lp, 100, side="Bull", amount=0.05)
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_REFUND",
                              delta_bnb=-0.001, outcome="refund", new_bankroll_bnb=2.30)
    fake = _wire(monkeypatch)
    contract = _FakeContract(balance=2.3455)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    c = _content(fake)
    assert c == (
        "[LIVE] [INFO] **BET REFUND** epoch `100` — Refunded `0.0500` BNB, bankroll `2.3455` BNB" + _SUFFIX
    )


def test_won_alert_no_suffix_when_no_prior_drop(tmp_path, monkeypatch):
    """A normal WON (no DROPPED in history) carries NO suffix."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", amount=0.05); _confirm(lp, 100)
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_WON",
                              delta_bnb=0.04, outcome="win", new_bankroll_bnb=2.34)
    fake = _wire(monkeypatch)
    contract = _FakeContract(balance=2.39)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    c = _content(fake)
    assert c == "[LIVE] [INFO] **BET WON** epoch `100` — Won `0.0400` BNB, bankroll `2.3900` BNB"
    assert "previously reported as DROPPED" not in c


# --- NIT 2: DROPPED gas accuracy — never-mined vs mined-and-reverted ---------

def test_dropped_initial_record_has_null_gas(tmp_path):
    """Option B: at the 10s receipt timeout the FIRST DROPPED record carries
    gas_paid_bnb = null — the gas spent is genuinely unknown at that moment
    (the key is present, value null — distinct from a confirmed 0)."""
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull", amount=0.05)
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=None,
                                   included_block_number=None, included_late=False)
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "DROPPED"
    assert "gas_paid_bnb" in led[100] and led[100]["gas_paid_bnb"] is None


def test_dropped_with_revert_receipt_updates_gas(tmp_path):
    """A standing DROPPED whose TX mined-and-reverted (receipt present) keeps
    the DROPPED status but has its forensic gas resolved from null to the
    receipt's actual gasUsed x effectiveGasPrice. No settlement, no re-alert."""
    lp = _ledger(tmp_path)
    _make_dropped(lp, 100, side="Bull", amount=0.05)
    assert bet_ledger.load_ledger(lp)[100]["gas_paid_bnb"] is None   # null pre-resolve
    contract = _FakeContract(
        bet_amounts={100: 0},  # never registered (PCS rejected the revert)
        receipts={"0xabc": {"status": 0, "gasUsed": 100_000,
                            "effectiveGasPrice": 10**9, "blockNumber": 50}},
    )
    dropped = []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.30, now_ts=2_000_000,
                                   wallet_address="0xme",
                                   dropped_alert_fn=lambda **kw: dropped.append(kw))
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "DROPPED"                 # status unchanged
    assert led[100]["gas_paid_bnb"] == 0.0001              # actual gas resolved in
    assert dropped == []                                   # standing DROPPED -> silent
    assert contract.calls == []                            # round_data never read
    assert settled and settled[0]["outcome"] == "dropped_gas_resolved"
    # Idempotent: a second pass sees gas non-null and does NOT re-resolve.
    settled2 = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                    buffer_seconds=30, fresh_bankroll_bnb=2.30, now_ts=2_000_000,
                                    wallet_address="0xme", dropped_alert_fn=lambda **kw: dropped.append(kw))
    assert settled2 == [] and dropped == []


def test_dropped_with_no_receipt_writes_zero_gas(tmp_path):
    """A standing DROPPED with no receipt (truly never mined) keeps the DROPPED
    status AND has gas_paid_bnb resolved to 0 — NOT null, NOT absent (Option B:
    0 = confirmed no gas spent)."""
    lp = _ledger(tmp_path)
    _make_dropped(lp, 100, side="Bull", amount=0.05)
    contract = _FakeContract(bet_amounts={100: 0}, receipts={})  # try_get_receipt -> None
    dropped = []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.30, now_ts=2_000_000,
                                   wallet_address="0xme",
                                   dropped_alert_fn=lambda **kw: dropped.append(kw))
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "DROPPED"
    assert "gas_paid_bnb" in led[100]                      # key present
    assert led[100]["gas_paid_bnb"] == 0                   # confirmed zero (not null, not absent)
    assert dropped == []                                   # no Discord re-alert
    assert contract.calls == []
    assert settled and settled[0]["outcome"] == "dropped_gas_resolved"


def test_dropped_with_success_receipt_reclassifies(tmp_path):
    """A standing DROPPED whose TX actually registered (receipt status=1 AND
    read_bet_amount > 0) is reclassified to CONFIRMED with the receipt's actual
    gas, then falls through to settlement (here a LOSS). The LOSS alert is the
    implicit correction (previously_dropped True)."""
    lp = _ledger(tmp_path)
    _make_dropped(lp, 100, side="Bull", amount=0.05)
    contract = _FakeContract(
        rounds={100: _round(100, lock_price=300, close_price=290, bull_bnb=10, bear_bnb=10)},
        bet_amounts={100: 5 * 10**16},  # registered
        receipts={"0xabc": {"status": 1, "gasUsed": 100_000,
                            "effectiveGasPrice": 10**9, "blockNumber": 50}},
    )
    lost = []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                                   wallet_address="0xme",
                                   lost_alert_fn=lambda **kw: lost.append(kw),
                                   dropped_alert_fn=lambda **kw: None)
    assert settled[0]["status"] == "SETTLED_LOST"          # fell through to settle
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "SETTLED_LOST"
    assert led[100]["gas_paid_bnb"] == 0.0001              # actual gas from CONFIRMED reclassify
    # A CONFIRMED reclassify record was appended to the raw history.
    raw = Path(lp).read_text(encoding="utf-8")
    assert '"status":"CONFIRMED"' in raw
    # LOSS alert flagged as a DROPPED correction.
    assert len(lost) == 1 and lost[0]["previously_dropped"] is True


def test_gas_field_schema_invariant_on_tx_state_records(tmp_path):
    """Option B contract: every bet-TX-outcome record (SUBMITTED / CONFIRMED /
    LATE / REVERTED / DROPPED) carries gas_paid_bnb with a value in
    {null, 0, positive}. SETTLED_*/CLAIMED records do NOT carry the field."""
    import json as _json
    lp = _ledger(tmp_path)
    _submit(lp, 100, side="Bull")                                  # SUBMITTED -> null
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=1,
                                   included_block_number=5, included_late=False,
                                   gas_paid_bnb=0.0001)            # CONFIRMED -> positive
    bet_ledger.record_confirmation(ledger_path=lp, epoch=101, chain_status=0,
                                   included_block_number=6, included_late=True,
                                   gas_paid_bnb=0.0002)            # LATE -> positive
    bet_ledger.record_confirmation(ledger_path=lp, epoch=102, chain_status=0,
                                   included_block_number=7, included_late=False,
                                   gas_paid_bnb=0.0003)            # REVERTED -> positive
    bet_ledger.record_confirmation(ledger_path=lp, epoch=103, chain_status=None,
                                   included_block_number=None, included_late=False)  # DROPPED -> null
    bet_ledger.record_settled(ledger_path=lp, epoch=100, side="Bull", status="SETTLED_WON",
                              delta_bnb=0.04, outcome="win", new_bankroll_bnb=2.4)
    bet_ledger.record_claimed(ledger_path=lp, epoch=100, new_bankroll_bnb=2.45)
    gas_bearing = {"SUBMITTED", "CONFIRMED", "LATE", "REVERTED", "DROPPED"}
    for line in Path(lp).read_text(encoding="utf-8").splitlines():
        rec = _json.loads(line)
        if rec["status"] in gas_bearing:
            assert "gas_paid_bnb" in rec, f"{rec['status']} missing gas key"
            g = rec["gas_paid_bnb"]
            assert g is None or (isinstance(g, (int, float)) and g >= 0), f"bad gas {g!r}"
        else:  # SETTLED_*/CLAIMED: PnL records, no gas field
            assert "gas_paid_bnb" not in rec


def test_append_record_rejects_gasless_tx_state_record(tmp_path):
    """The invariant is enforced at write time: a gas-bearing record without
    the gas_paid_bnb key is a programming error -> hard stop (InvariantError)."""
    lp = _ledger(tmp_path)
    with pytest.raises(InvariantError):
        bet_ledger._append_record(lp, {"ts": "t", "status": "CONFIRMED", "epoch": 1})
    # A PnL record (SETTLED_*/CLAIMED) without the key is fine.
    assert bet_ledger._append_record(
        lp, {"ts": "t", "status": "SETTLED_LOST", "epoch": 1}) is True


# --- migrated existing alerts: claim-failure + gas-cap (ASCII + gwei) -------

def test_gwei_formatting():
    """Clean integer multiples render as int; otherwise 2 decimals."""
    assert live_mod._fmt_gwei(8_000_000_000) == "8 gwei"
    assert live_mod._fmt_gwei(5_000_000_000) == "5 gwei"
    assert live_mod._fmt_gwei(8_250_000_000) == "8.25 gwei"
    assert live_mod._fmt_gwei(0) == "0 gwei"
    assert live_mod._fmt_gwei(8_123_000_000) == "8.12 gwei"  # 3rd decimal rounds to 2


def test_claim_failure_alert_ascii_format(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod._send_claim_failure_alert(reason="revert", tx_hash="0xabc123",
                                       epochs=[401, 402, 403], gas_limit=900_000)
    assert _content(fake) == (
        "[LIVE] [CRIT] **CLAIM FAILED** reason=`revert`, tx=`0xabc123`, "
        "epochs=`[401,402,403]`, gas_limit=`900000`"
    )
    assert ":rotating_light:" not in _content(fake)


def test_gas_cap_breach_bet_path_crit(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_gas_cap_breach_alert(path="bet", suggested_wei=8_000_000_000,
                                       cap_wei=5_000_000_000, epoch=12345)
    c = _content(fake)
    assert c.startswith("[LIVE] [CRIT] **GAS CAP BREACHED** path=`bet`, epoch=`12345`, ")
    assert "suggested=`8 gwei`, cap=`5 gwei`, ratio=`1.60x`" in c
    assert "Bet SKIPPED" in c
    # gwei units (raw wei value absent); no double-crit marker.
    assert "8000000000" not in c and "**CRITICAL**" not in c


def test_gas_cap_breach_claim_path_warn(monkeypatch):
    fake = _wire(monkeypatch)
    live_mod.send_gas_cap_breach_alert(path="claim", suggested_wei=8_250_000_000,
                                       cap_wei=5_000_000_000, epochs=[401, 402])
    c = _content(fake)
    assert c.startswith("[LIVE] [WARN] **GAS CAP BREACHED** path=`claim`, epochs=`[401,402]`, ")
    assert "suggested=`8.25 gwei`, cap=`5 gwei`" in c
    assert "Claim skipped this round" in c
    assert ":warning:" not in c
