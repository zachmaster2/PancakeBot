"""Tests for the bet lifecycle ledger + reconciliation (Step 31, revised).

Covers:
  - append/read/replay (last-write-wins), corrupted final line, empty/missing
  - _append_record returns bool; failed persist defers alert + settlement
  - classify_confirmation: CONFIRMED / LATE / REVERTED / PLACED(timeout)
  - classify_settlement mapping (win/loss/refund -> status, delta)
  - reconcile: LOSS alert fires; WIN/REFUND recorded SILENTLY (Option B)
  - tie = LOST (not refund)
  - LATE / REVERTED are terminal -> never settled
  - permanent un-oracled PLACED -> refund-settles after close_ts
  - idempotency (no double-settle / double-alert); transient RPC skip
  - sequential-bets: alert uses FRESH bankroll passed in (not placed+delta)
  - claim-path fire_claim_settled_alerts: WON single / REFUND / batch
  - alert senders: happy / missing-webhook / non-2xx -> swallow, no raise
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pancakebot.runtime import bet_ledger  # noqa: E402
from pancakebot.runtime import live as live_mod  # noqa: E402
from pancakebot.chain.prediction_contract import RoundData  # noqa: E402
from pancakebot.constants import BNB_WEI  # noqa: E402


def _ledger(tmp_path) -> str:
    return str(tmp_path / "bets.jsonl")


class _FakeContract:
    def __init__(self, rounds=None, transient=None, balance=2.0, bet_amounts=None):
        self._rounds = rounds or {}
        self._transient = transient or set()
        self._balance = balance
        # epoch -> on-chain registered bet amount (wei). Default: not set ->
        # read_bet_amount returns a large positive (registered) so existing
        # tests that don't exercise the timeout path are unaffected.
        self._bet_amounts = bet_amounts or {}
        self.calls: list[int] = []

    def round_data(self, epoch):
        self.calls.append(int(epoch))
        if int(epoch) in self._transient:
            raise RuntimeError("transient rpc")
        return self._rounds[int(epoch)]

    def wallet_balance_bnb(self, _addr):
        return self._balance

    def read_bet_amount(self, epoch, _wallet):
        return int(self._bet_amounts.get(int(epoch), 10**18))  # default registered


def _round(epoch, *, lock_price, close_price, bull_bnb, bear_bnb,
            oracle_called=True, close_ts=1_000_000):
    return RoundData(
        epoch=epoch, start_ts=close_ts - 300, lock_ts=close_ts, close_ts=close_ts,
        lock_price_usd=lock_price, close_price_usd=close_price,
        bull_amount_wei=int(bull_bnb * BNB_WEI), bear_amount_wei=int(bear_bnb * BNB_WEI),
        oracle_called=oracle_called,
    )


def _place(lp, epoch, side="Bull", amount=0.05, bankroll=2.3):
    bet_ledger.record_placed(ledger_path=lp, epoch=epoch, side=side,
                              amount_bnb=amount, tx_hash="0xabc", bankroll_after_bnb=bankroll)


# --- append / read / replay -------------------------------------------------

def test_append_returns_true_on_success(tmp_path):
    assert bet_ledger.record_placed(
        ledger_path=_ledger(tmp_path), epoch=1, side="Bull",
        amount_bnb=0.05, tx_hash="0x", bankroll_after_bnb=2.0) is True


def test_load_roundtrip_and_last_write_wins(tmp_path):
    lp = _ledger(tmp_path)
    _place(lp, 100)
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=1,
                                    included_block_number=555, included_late=False)
    led = bet_ledger.load_ledger(lp)
    assert led[100]["status"] == "CONFIRMED"
    assert led[100]["side"] == "Bull"          # merged from PLACED
    assert led[100]["included_block_number"] == 555


def test_missing_and_empty_file(tmp_path):
    assert bet_ledger.load_ledger(str(tmp_path / "nope.jsonl")) == {}
    lp = _ledger(tmp_path); Path(lp).write_text("", encoding="utf-8")
    assert bet_ledger.load_ledger(lp) == {}


def test_corrupted_final_line_skipped(tmp_path):
    lp = _ledger(tmp_path)
    _place(lp, 100)
    with open(lp, "a", encoding="utf-8") as f:
        f.write('{"status":"PLACED","epoch":101,"sid')  # truncated
    led = bet_ledger.load_ledger(lp)
    assert 100 in led and 101 not in led


# --- classify_confirmation --------------------------------------------------

def test_classify_confirmation_matrix():
    c = bet_ledger.classify_confirmation
    assert c(chain_status=1, included_late=False) == "CONFIRMED"
    assert c(chain_status=0, included_late=True) == "LATE"
    assert c(chain_status=0, included_late=False) == "REVERTED"
    assert c(chain_status=None, included_late=False) == "PLACED"  # timeout
    assert c(chain_status=1, included_late=True) == "LATE"        # anomalous -> LATE


def test_record_confirmation_late_is_terminal_open_set(tmp_path):
    lp = _ledger(tmp_path)
    _place(lp, 100)
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


# --- reconcile: Option B alert behavior ------------------------------------

def test_reconcile_win_records_silently_no_alert(tmp_path):
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull")
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
    _place(lp, 100, side="Bull")
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
    _place(lp, 100, side="Bull")
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
    _place(lp, 100, side="Bull")
    # un-oracled, past close_ts -> refund. Silent (claim path alerts).
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


# --- LATE / REVERTED must NOT settle ---------------------------------------

def test_reconcile_timeout_then_registered(tmp_path):
    """Reviewer Fix #2: a still-PLACED entry (receipt timed out) whose bet
    DID register on-chain (read_bet_amount > 0) settles normally via pool
    math."""
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull")  # stays PLACED (no CONFIRMED transition)
    contract = _FakeContract(
        rounds={100: _round(100, lock_price=300, close_price=290, bull_bnb=10, bear_bnb=10)},
        bet_amounts={100: 5 * 10**16},  # registered
    )
    lost = []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                                   wallet_address="0xme",
                                   lost_alert_fn=lambda **kw: lost.append(kw),
                                   reverted_alert_fn=lambda **kw: None)
    assert settled[0]["status"] == "SETTLED_LOST"   # normal pool settlement
    assert len(lost) == 1


def test_reconcile_timeout_then_reverted(tmp_path):
    """Reviewer Fix #2: a still-PLACED entry whose bet NEVER registered
    (read_bet_amount == 0 — timed out then mined late/reverted/dropped) is
    marked terminal REVERTED, fires BET REVERTED, and is NOT pool-settled."""
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull")
    contract = _FakeContract(
        rounds={100: _round(100, lock_price=300, close_price=310, bull_bnb=10, bear_bnb=10)},
        bet_amounts={100: 0},  # never registered
    )
    reverted = []
    lost = []
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.30, now_ts=2_000_000,
                                   wallet_address="0xme",
                                   lost_alert_fn=lambda **kw: lost.append(kw),
                                   reverted_alert_fn=lambda **kw: reverted.append(kw))
    assert settled[0]["status"] == "REVERTED"
    assert reverted and reverted[0]["epoch"] == 100
    assert lost == []                                       # NOT pool-settled
    assert contract.calls == []                             # round_data never read
    assert bet_ledger.load_ledger(lp)[100]["status"] == "REVERTED"


def test_confirmed_skips_timeout_guard(tmp_path):
    """A CONFIRMED entry (receipt was good) does NOT trigger the on-chain
    read_bet_amount check — it settles directly via pool math."""
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull")
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=1,
                                    included_block_number=5, included_late=False)
    # bet_amounts says 0, but CONFIRMED should NOT consult it.
    contract = _FakeContract(
        rounds={100: _round(100, lock_price=300, close_price=290, bull_bnb=10, bear_bnb=10)},
        bet_amounts={100: 0},
    )
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                   buffer_seconds=30, fresh_bankroll_bnb=2.25, now_ts=2_000_000,
                                   wallet_address="0xme", lost_alert_fn=lambda **kw: None,
                                   reverted_alert_fn=lambda **kw: None)
    assert settled[0]["status"] == "SETTLED_LOST"   # pool-settled, guard skipped


def test_late_never_settles(tmp_path):
    lp = _ledger(tmp_path)
    _place(lp, 100)
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
    _place(lp, 100)
    bet_ledger.record_confirmation(ledger_path=lp, epoch=100, chain_status=0,
                                    included_block_number=9, included_late=False)
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=310,
                                          bull_bnb=10, bear_bnb=10)})
    settled = bet_ledger.reconcile(ledger_path=lp, contract=contract,
                                   treasury_fee_fraction=0.03, buffer_seconds=30, fresh_bankroll_bnb=2.3,
                                   now_ts=2_000_000, lost_alert_fn=None)
    assert settled == []
    assert bet_ledger.load_ledger(lp)[100]["status"] == "REVERTED"


def test_permanent_unoracled_placed_refunds_after_close(tmp_path):
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull")
    # never oracled; close_ts in the past relative to now_ts -> refund-settles.
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=0,
                                          bull_bnb=10, bear_bnb=10,
                                          oracle_called=False, close_ts=1_000_000)})
    # Before close: skipped.
    s1 = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                              fresh_bankroll_bnb=2.3, now_ts=999_000, lost_alert_fn=None)
    assert s1 == [] and bet_ledger.load_ledger(lp)[100]["status"] == "PLACED"
    # After close: refund-settles.
    s2 = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                              fresh_bankroll_bnb=2.3, now_ts=2_000_000, lost_alert_fn=None)
    assert s2[0]["status"] == "SETTLED_REFUND"


# --- idempotency / transient / append-gating -------------------------------

def test_reconcile_idempotent(tmp_path):
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull")
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
    _place(lp, 100)
    contract = _FakeContract({}, transient={100})
    assert bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03, buffer_seconds=30,
                                fresh_bankroll_bnb=2.3, now_ts=2_000_000, lost_alert_fn=None) == []
    assert bet_ledger.load_ledger(lp)[100]["status"] == "PLACED"


def test_failed_persist_defers_alert_and_settlement(tmp_path):
    """Fix #7: if the terminal append fails, neither the alert nor the
    settled-list entry happen — both defer to the next pass."""
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull")
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
    assert bet_ledger.load_ledger(lp)[100]["status"] == "PLACED"


def test_sequential_bets_use_fresh_bankroll(tmp_path):
    """Fix #3: two bets placed back-to-back; settling bet #1 shows the FRESH
    bankroll passed in (which already reflects bet #2's placement debit), not
    placed_bankroll + delta."""
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull", bankroll=2.30)  # after bet #1 placement
    _place(lp, 101, side="Bull", bankroll=2.25)  # after bet #2 placement (debited again)
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
    # Fresh balance shown, NOT placed_bankroll(2.30) + delta.
    assert lost[0]["new_bankroll_bnb"] == 2.25


# --- claim-path WON / REFUND firing ----------------------------------------

def test_fire_claim_won_single(tmp_path, monkeypatch):
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull", amount=0.05)
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


def test_fire_claim_refund(tmp_path, monkeypatch):
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull", amount=0.05)
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
        _place(lp, ep, side="Bull", amount=0.05)
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
    """Reviewer Fix #1: a claimed epoch still PLACED/CONFIRMED (reconcile
    failed transiently this iteration) must NOT fire a premature WON or mark
    CLAIMED — it stays open until the next reconcile writes the real delta."""
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull", amount=0.05)  # status PLACED, never reconciled
    won = []
    monkeypatch.setattr(live_mod, "send_bet_settled_alert", lambda **kw: won.append(kw))
    contract = _FakeContract(balance=2.39)
    live_mod.fire_claim_settled_alerts(ledger_path=lp, claimed_epochs=[100],
                                       contract=contract, wallet_address="0xme")
    assert won == []                                            # no premature WON
    assert bet_ledger.load_ledger(lp)[100]["status"] == "PLACED"  # stays open


def test_reconcile_defers_in_buffer_window(tmp_path):
    """Reviewer Fix #3: an oracle-pending round inside [close_ts,
    close_ts+buffer] must NOT settle as refund — defer until past buffer."""
    lp = _ledger(tmp_path)
    _place(lp, 100, side="Bull")
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=0,
                                          bull_bnb=10, bear_bnb=10,
                                          oracle_called=False, close_ts=1_000_000)})
    # now inside [close_ts, close_ts+buffer]: 1_000_000 < now < 1_000_030.
    s = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                             buffer_seconds=30, fresh_bankroll_bnb=2.3, now_ts=1_000_015,
                             lost_alert_fn=None)
    assert s == []                                              # deferred
    assert bet_ledger.load_ledger(lp)[100]["status"] == "PLACED"
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
    _place(lp, 100, side="Bull")
    contract = _FakeContract({100: _round(100, lock_price=300, close_price=0,
                                          bull_bnb=10, bear_bnb=10,
                                          oracle_called=False, close_ts=1_000_000)})
    # Exact boundary: now == close_ts + buffer == 1_000_030 -> defer (>= gate).
    s_boundary = bet_ledger.reconcile(ledger_path=lp, contract=contract, treasury_fee_fraction=0.03,
                                      buffer_seconds=30, fresh_bankroll_bnb=2.3, now_ts=1_000_030,
                                      lost_alert_fn=None)
    assert s_boundary == []
    assert bet_ledger.load_ledger(lp)[100]["status"] == "PLACED"
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


# --- alert senders: happy / missing / non-2xx ------------------------------

class _FakeResp:
    def __init__(self, code): self.status_code = code; self.text = ""


class _FakeRequests:
    def __init__(self, code=204): self._code = code; self.captured = {}
    def post(self, url, *, json, timeout):
        self.captured = {"url": url, "json": json}
        return _FakeResp(self._code)


def _content(fake):
    return fake.captured["json"]["content"]


def test_placed_format_exact(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_bet_placed_alert(epoch=999, side="Bull", amount_bnb=0.05, bankroll_bnb=2.3463)
    assert _content(fake) == (
        "[INFO] **BET PLACED** epoch `999` — Bet `0.0500` BNB on Bull, bankroll `2.3463` BNB"
    )


def test_late_format_exact(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_bet_late_alert(epoch=999, bankroll_bnb=2.3452)
    assert _content(fake) == "[WARN] **BET LATE** epoch `999` — bankroll `2.3452` BNB"


def test_reverted_format_exact(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_bet_reverted_alert(epoch=999, bankroll_bnb=2.3452)
    assert _content(fake) == "[WARN] **BET REVERTED** epoch `999` — bankroll `2.3452` BNB"


def test_won_format_exact(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_bet_settled_alert(epoch=999, won=True, delta_bnb=0.0423, amount_bnb=0.05,
                                     new_bankroll_bnb=2.3886)
    assert _content(fake) == (
        "[INFO] **BET WON** epoch `999` — Won `0.0423` BNB, bankroll `2.3886` BNB"
    )


def test_lost_format_exact(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_bet_settled_alert(epoch=999, won=False, delta_bnb=-0.05, amount_bnb=0.05,
                                     new_bankroll_bnb=2.2963)
    # Lost amount shown positive; the verb conveys direction.
    assert _content(fake) == (
        "[INFO] **BET LOST** epoch `999` — Lost `0.0500` BNB, bankroll `2.2963` BNB"
    )


def test_refund_format_exact(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_bet_refund_alert(epoch=999, refund_bnb=0.05, new_bankroll_bnb=2.3455)
    assert _content(fake) == (
        "[INFO] **BET REFUND** epoch `999` — Refunded `0.0500` BNB, bankroll `2.3455` BNB"
    )


def test_won_batch_format_exact(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_bet_won_batch_alert(epochs=[101, 102, 103], total_delta_bnb=0.12,
                                       new_bankroll_bnb=2.50)
    assert _content(fake) == (
        "[INFO] **BET WON** epochs `[101, 102, 103]` — Won `0.1200` BNB total, "
        "bankroll `2.5000` BNB"
    )


def test_no_gas_in_any_alert(monkeypatch):
    """Locked format drops all gas values from Discord display."""
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_bet_placed_alert(epoch=1, side="Bear", amount_bnb=0.05, bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()
    live_mod.send_bet_late_alert(epoch=1, bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()
    live_mod.send_bet_settled_alert(epoch=1, won=True, delta_bnb=0.04, amount_bnb=0.05, new_bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()
    live_mod.send_bet_refund_alert(epoch=1, refund_bnb=0.05, new_bankroll_bnb=2.0)
    assert "gas" not in _content(fake).lower()


def test_confirmation_alert_sequence():
    """BET PLACED fires for any mined TX; LATE/REVERTED add a follow-up;
    a receipt timeout (PLACED status) fires nothing."""
    seq = bet_ledger.confirmation_alert_sequence
    assert seq("CONFIRMED") == ("PLACED",)
    assert seq("LATE") == ("PLACED", "LATE")
    assert seq("REVERTED") == ("PLACED", "REVERTED")
    assert seq("PLACED") == ()  # timeout -> no alert


def test_classify_confirmation_late_fires_placed_alert():
    """LATE classification -> BOTH PLACED and LATE alerts, PLACED first."""
    assert bet_ledger.classify_confirmation(chain_status=0, included_late=True) == "LATE"
    assert bet_ledger.confirmation_alert_sequence("LATE") == ("PLACED", "LATE")


def test_classify_confirmation_reverted_fires_placed_alert():
    """REVERTED classification -> BOTH PLACED and REVERTED alerts, PLACED first."""
    assert bet_ledger.classify_confirmation(chain_status=0, included_late=False) == "REVERTED"
    assert bet_ledger.confirmation_alert_sequence("REVERTED") == ("PLACED", "REVERTED")


def test_senders_missing_webhook_no_raise(monkeypatch):
    monkeypatch.delenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", raising=False)
    live_mod.send_bet_placed_alert(epoch=1, side="Bull", amount_bnb=0.05, bankroll_bnb=2.0)
    live_mod.send_bet_late_alert(epoch=1, bankroll_bnb=2.0)
    live_mod.send_bet_reverted_alert(epoch=1, bankroll_bnb=2.0)
    live_mod.send_bet_refund_alert(epoch=1, refund_bnb=0.05, new_bankroll_bnb=2.3)


def test_sender_non_2xx_swallowed(monkeypatch):
    fake = _FakeRequests(500)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_bet_placed_alert(epoch=1, side="Bull", amount_bnb=0.05, bankroll_bnb=2.0)


# --- migrated existing alerts: claim-failure + gas-cap (ASCII + gwei) -------

def test_gwei_formatting():
    """Clean integer multiples render as int; otherwise 2 decimals."""
    assert live_mod._fmt_gwei(8_000_000_000) == "8 gwei"
    assert live_mod._fmt_gwei(5_000_000_000) == "5 gwei"
    assert live_mod._fmt_gwei(8_250_000_000) == "8.25 gwei"
    assert live_mod._fmt_gwei(0) == "0 gwei"
    assert live_mod._fmt_gwei(8_123_000_000) == "8.12 gwei"  # 3rd decimal rounds to 2


def test_claim_failure_alert_ascii_format(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod._send_claim_failure_alert(reason="revert", tx_hash="0xabc123",
                                       epochs=[401, 402, 403], gas_limit=900_000)
    assert _content(fake) == (
        "[CRIT] **CLAIM FAILED** reason=`revert`, tx=`0xabc123`, "
        "epochs=`[401,402,403]`, gas_limit=`900000`"
    )
    assert ":rotating_light:" not in _content(fake)


def test_gas_cap_breach_bet_path_crit(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_gas_cap_breach_alert(path="bet", suggested_wei=8_000_000_000,
                                       cap_wei=5_000_000_000, epoch=12345)
    c = _content(fake)
    assert c.startswith("[CRIT] **GAS CAP BREACHED** path=`bet`, epoch=`12345`, ")
    assert "suggested=`8 gwei`, cap=`5 gwei`, ratio=`1.60x`" in c
    assert "Bet SKIPPED" in c
    # gwei units (raw wei value absent); no double-crit marker.
    assert "8000000000" not in c and "**CRITICAL**" not in c


def test_gas_cap_breach_claim_path_warn(monkeypatch):
    fake = _FakeRequests(204)
    monkeypatch.setenv("PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", "https://d/w")
    monkeypatch.setitem(sys.modules, "requests", fake)
    live_mod.send_gas_cap_breach_alert(path="claim", suggested_wei=8_250_000_000,
                                       cap_wei=5_000_000_000, epochs=[401, 402])
    c = _content(fake)
    assert c.startswith("[WARN] **GAS CAP BREACHED** path=`claim`, epochs=`[401,402]`, ")
    assert "suggested=`8.25 gwei`, cap=`5 gwei`" in c
    assert "Claim skipped this round" in c
    assert ":warning:" not in c
