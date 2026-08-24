"""The restart gate must BLOCK the two cases that actually happened.

Both restarts on 2026-08-24 were preceded by a hand-written check that
passed when it should have failed. These tests replay those exact
situations against the real gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import preflight_restart as pf  # noqa: E402

SKIP_LINE = ("2026-08-24 13:49:25.49  INFO   SKIP      Skipped epoch 509795: "
             "pool below minimum (1.09 BNB < 1.25 BNB threshold)")
BET_LINE = ("2026-08-24 13:44:20.50  INFO   BET       Bet 0.0500 BNB on Bear "
            "for epoch 509794 (tx 7a46a17a...)")

CLEAN = [{"epoch": 1, "status": "SUBMITTED"},
         {"epoch": 1, "status": "SETTLED_LOST"}]


def _ledger(tmp_path, records):
    p = tmp_path / "bets.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records),
                 encoding="utf-8")
    return str(p)


def _gate(monkeypatch, tmp_path, *, records, decision, slack,
          min_slack_s=90.0, skip_epoch=999999):
    path = _ledger(tmp_path, records)
    monkeypatch.setattr(pf, "last_decision",
                        lambda *a, **k: (decision, "line", skip_epoch))
    monkeypatch.setattr(pf, "lock_slack", lambda *a, **k: (509795, slack))
    return pf.check(unit="u", ledger_path=path, min_slack_s=min_slack_s)


# ---- condition 1: open positions ----------------------------------------

def test_confirmed_bet_counts_as_open(tmp_path):
    """THE CASE THAT GOT THROUGH. Epoch 509794 was CONFIRMED — not
    SUBMITTED — when the 13:44:37 restart went out. A hand-written test of
    'SUBMITTED minus terminal' misses this exactly."""
    path = _ledger(tmp_path, [
        {"epoch": 509794, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 509794, "status": "CONFIRMED", "gas_paid_bnb": 0.000119},
    ])
    assert pf.open_positions(path) == [(509794, "CONFIRMED")]


def test_late_is_terminal_and_not_an_open_position(tmp_path):
    """THE OTHER MISREAD. Epoch 509790 went LATE: the TX reverted and no
    position was ever taken, so it must NOT block a restart. An ad-hoc
    check flagged it and nearly stalled a correct deploy."""
    path = _ledger(tmp_path, [
        {"epoch": 509790, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 509790, "status": "LATE", "gas_paid_bnb": 3.6e-05},
    ])
    assert pf.open_positions(path) == []


@pytest.mark.parametrize("terminal", [
    "LATE", "REVERTED", "DROPPED", "SETTLED_WON", "SETTLED_LOST",
    "SETTLED_REFUND", "CLAIMED",
])
def test_no_terminal_status_blocks_a_restart(tmp_path, terminal):
    path = _ledger(tmp_path, [
        {"epoch": 1, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 1, "status": terminal},
    ])
    assert pf.open_positions(path) == []


def test_a_settled_history_leaves_nothing_open(tmp_path):
    path = _ledger(tmp_path, [
        {"epoch": 509788, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 509788, "status": "CONFIRMED"},
        {"epoch": 509788, "status": "SETTLED_LOST", "delta_bnb": -0.05},
        {"epoch": 509790, "status": "SUBMITTED", "amount_bnb": 0.05},
        {"epoch": 509790, "status": "LATE"},
    ])
    assert pf.open_positions(path) == []


def test_missing_ledger_blocks_rather_than_passing_vacuously(tmp_path):
    """MED-B2. `load_ledger` returns {} for a missing file, which is right
    for the bot (it starts with none) and WRONG for a gate: a mistyped
    --ledger or a wiped var/ would make the money check silently vacuous.
    'No ledger' is not evidence of 'no position'."""
    with pytest.raises(pf.LedgerUnavailable):
        pf.open_positions(str(tmp_path / "nope.jsonl"))


def test_the_gate_blocks_when_the_ledger_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "last_decision", lambda *a, **k: ("SKIP", "", 9))
    monkeypatch.setattr(pf, "lock_slack", lambda *a, **k: (10, 240.0))
    blockers = pf.check(unit="u", ledger_path=str(tmp_path / "nope.jsonl"),
                        min_slack_s=90.0)
    assert any("LEDGER UNAVAILABLE" in b for b in blockers)


# ---- condition 2: the last decision --------------------------------------

def test_last_decision_bet_is_detected():
    kind, line, _ = pf.last_decision(
        journal=lambda u, n: SKIP_LINE + "\n" + BET_LINE)
    assert kind == "BET"
    assert "509794" in line


def test_last_decision_skip_is_detected():
    kind, _, epoch = pf.last_decision(
        journal=lambda u, n: BET_LINE + "\n" + SKIP_LINE)
    assert kind == "SKIP"
    assert epoch == 509795


def test_no_decision_at_all_is_reported():
    kind, _, _ = pf.last_decision(journal=lambda u, n: "nothing here")
    assert kind is None


def test_journal_failure_is_a_blocker_not_a_traceback(monkeypatch, tmp_path):
    """LOW-B4: a gate that tracebacks is a gate that gets bypassed."""
    def _boom(*a, **k):
        raise pf.JournalUnavailable("journalctl failed: [Errno 2]")

    path = _ledger(tmp_path, CLEAN)
    monkeypatch.setattr(pf, "last_decision", _boom)
    monkeypatch.setattr(pf, "lock_slack", lambda *a, **k: (10, 240.0))
    blockers = pf.check(unit="u", ledger_path=path, min_slack_s=90.0)
    assert any("JOURNAL UNAVAILABLE" in b for b in blockers)


def test_a_reworded_bet_line_is_caught_by_the_ledger_cross_check(
        monkeypatch, tmp_path):
    """MED-B1: the scan matches the engine's log PROSE. Reword the BET
    message and it walks past to an older SKIP. The ledger is structured
    data, so a bet at or after the skip epoch exposes the miss — even when
    that bet has already settled and condition 1 is therefore silent."""
    path = _ledger(tmp_path, [
        {"epoch": 509800, "status": "SUBMITTED"},
        {"epoch": 509800, "status": "SETTLED_LOST"},
    ])
    monkeypatch.setattr(pf, "last_decision",
                        lambda *a, **k: ("SKIP", "older skip", 509795))
    monkeypatch.setattr(pf, "lock_slack", lambda *a, **k: (509801, 240.0))
    blockers = pf.check(unit="u", ledger_path=path, min_slack_s=90.0)
    assert any("LEDGER DISAGREES WITH THE JOURNAL" in b for b in blockers)


def test_a_skip_line_without_an_epoch_blocks(monkeypatch, tmp_path):
    path = _ledger(tmp_path, CLEAN)
    monkeypatch.setattr(pf, "last_decision",
                        lambda *a, **k: ("SKIP", "Skipped epoch ???", None))
    monkeypatch.setattr(pf, "lock_slack", lambda *a, **k: (10, 240.0))
    blockers = pf.check(unit="u", ledger_path=path, min_slack_s=90.0)
    assert any("NO EPOCH" in b for b in blockers)


# ---- condition 3: slack --------------------------------------------------

def test_slack_is_measured_against_the_open_round_lock():
    epoch, slack = pf.lock_slack(now=1000.0, next_lock=lambda: (509795, 1240))
    assert epoch == 509795
    assert slack == pytest.approx(240.0)


def test_lock_index_is_shared_not_restated():
    """LOW-B3: both readers of the rounds tuple use one definition."""
    from pancakebot.chain.prediction_contract import ROUNDS_LOCK_TS_IDX
    assert pf.ROUNDS_LOCK_TS_IDX is ROUNDS_LOCK_TS_IDX


# ---- the gate as a whole -------------------------------------------------

def test_gate_passes_only_when_all_three_hold(monkeypatch, tmp_path):
    assert _gate(monkeypatch, tmp_path, records=CLEAN,
                 decision="SKIP", slack=240.0) == []


def test_gate_blocks_the_1344_restart(monkeypatch, tmp_path):
    """Replay of the restart that should not have happened: a CONFIRMED
    position AND a BET as the last decision."""
    blockers = _gate(
        monkeypatch, tmp_path,
        records=[{"epoch": 509794, "status": "SUBMITTED"},
                 {"epoch": 509794, "status": "CONFIRMED"}],
        decision="BET", slack=240.0)
    assert len(blockers) == 2
    assert any("OPEN POSITION" in b for b in blockers)
    assert any("BET" in b for b in blockers)


def test_gate_blocks_a_narrow_window(monkeypatch, tmp_path):
    blockers = _gate(monkeypatch, tmp_path, records=CLEAN,
                     decision="SKIP", slack=20.0)
    assert len(blockers) == 1
    assert "BEFORE THE NEXT LOCK" in blockers[0]


def test_gate_blocks_when_the_chain_cannot_be_read(monkeypatch, tmp_path):
    """Fails CLOSED. A preflight that passes because an endpoint threw --
    publicnode 403'd during read bursts on 2026-08-24 -- would be worse
    than no gate."""
    path = _ledger(tmp_path, CLEAN)
    monkeypatch.setattr(pf, "last_decision",
                        lambda *a, **k: ("SKIP", "", 999999))

    def _boom(*a, **k):
        raise RuntimeError("every RPC endpoint failed")

    monkeypatch.setattr(pf, "lock_slack", _boom)
    blockers = pf.check(unit="u", ledger_path=path, min_slack_s=90.0)
    assert len(blockers) == 1
    assert "CHAIN READ FAILED" in blockers[0]


def test_main_exit_codes(monkeypatch, tmp_path):
    path = _ledger(tmp_path, CLEAN)
    monkeypatch.setattr(pf, "last_decision",
                        lambda *a, **k: ("SKIP", "", 999999))
    monkeypatch.setattr(pf, "lock_slack", lambda *a, **k: (509795, 240.0))
    assert pf.main(["--ledger", path]) == 0

    monkeypatch.setattr(pf, "last_decision", lambda *a, **k: ("BET", "x", None))
    assert pf.main(["--ledger", path]) == 1
