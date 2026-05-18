"""Tests for the claim-TX receipt-waiting path in pancakebot/runtime/live.py.

Covers:
1. Success outcome: log + cursor advance.
2. Revert outcome (status=0): log warn + Discord alert + cursor advance.
3. Timeout outcome: log warn + Discord alert + cursor parked at first
   un-claimed epoch (TX may still mine).
4. Discord alert payload contains "PancakeBot-live CLAIM FAILED",
   reason, tx hash, epochs, gas_limit.
5. Discord alert disabled cleanly when webhook env var unset.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.chain.prediction_contract import ClaimSubmitResult  # noqa: E402
from pancakebot.runtime import live as live_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeContract:
    """Minimal Web3PredictionContract stand-in for claim_scan_cursor tests."""

    def __init__(
        self,
        *,
        user_rounds: list[int],
        claimable_map: dict[int, tuple[bool, bool]],
        close_ts_map: dict[int, int],
        wallet_balance: float,
        claim_outcome: ClaimSubmitResult,
    ) -> None:
        self._user_rounds = list(user_rounds)
        self._claimable_map = dict(claimable_map)
        self._close_ts_map = dict(close_ts_map)
        self._wallet_balance = float(wallet_balance)
        self._claim_outcome = claim_outcome
        self.claim_calls: list[dict[str, Any]] = []

    def get_user_rounds_length(self, _wallet: str) -> int:
        return len(self._user_rounds)

    def get_user_rounds_all_batched(
        self, *, wallet_address: str, cursor: int, total: int, page_size: int,
    ) -> list[int]:
        return list(self._user_rounds[cursor:total])

    def close_ts_batch(self, epochs: list[int]) -> dict[int, int]:
        return {e: self._close_ts_map.get(int(e), 0) for e in epochs}

    def claimable_refundable_batch(
        self, *, epochs: list[int], wallet_address: str,
    ) -> dict[int, tuple[bool, bool]]:
        return {
            int(e): self._claimable_map.get(int(e), (False, False))
            for e in epochs
        }

    def suggest_gas_price_wei(self) -> int:
        return 5_000_000_000  # 5 gwei

    def wallet_balance_bnb(self, _wallet: str) -> float:
        return self._wallet_balance

    def claim(
        self,
        *,
        epochs,
        gas_limit: int,
        gas_price_wei: int,
        wait_receipt: bool,
        receipt_timeout_seconds: int,
    ) -> ClaimSubmitResult:
        self.claim_calls.append({
            "epochs": list(epochs),
            "gas_limit": int(gas_limit),
            "gas_price_wei": int(gas_price_wei),
            "wait_receipt": bool(wait_receipt),
            "receipt_timeout_seconds": int(receipt_timeout_seconds),
        })
        return self._claim_outcome


def _scan_kwargs(*, contract, cursor_path: Path, claim_timeout: int = 35):
    return dict(
        contract=contract,
        wallet_address="0xdeadbeef",
        dry=False,
        cursor_path=str(cursor_path),
        locked_epoch=999_999,
        current_epoch=1_000_000,
        now_ts=10_000,
        buffer_seconds=30,
        page_size=100,
        gas_limit=300_000,
        claim_tx_receipt_timeout_seconds=claim_timeout,
    )


# ---------------------------------------------------------------------------
# 1. Success outcome
# ---------------------------------------------------------------------------

def test_claim_success_advances_cursor_and_records_claimed(
    tmp_path, monkeypatch
):
    epochs = [101, 102]
    contract = _FakeContract(
        user_rounds=epochs,
        claimable_map={101: (True, False), 102: (True, False)},
        close_ts_map={101: 0, 102: 0},
        wallet_balance=10.0,
        claim_outcome=ClaimSubmitResult(
            tx_hash="0xsuccess", status="success",
            included_block_number=12345, included_block_timestamp=10000,
        ),
    )
    sent_alerts: list[dict[str, Any]] = []
    monkeypatch.setattr(
        live_mod, "_send_claim_failure_alert",
        lambda **kw: sent_alerts.append(kw),
    )
    cursor_path = tmp_path / "cursor.txt"
    result = live_mod.claim_scan_cursor(
        **_scan_kwargs(contract=contract, cursor_path=cursor_path),
    )
    assert result.claimed_n == 2
    assert sent_alerts == []  # no alert on success
    assert int(cursor_path.read_text()) == 2  # cursor advanced past both epochs
    assert len(contract.claim_calls) == 1
    call = contract.claim_calls[0]
    assert call["wait_receipt"] is True
    assert call["receipt_timeout_seconds"] == 35
    assert call["epochs"] == [101, 102]


# ---------------------------------------------------------------------------
# 2. Revert outcome (status=0)
# ---------------------------------------------------------------------------

def test_claim_revert_alerts_and_advances_cursor(tmp_path, monkeypatch):
    epochs = [201, 202]
    contract = _FakeContract(
        user_rounds=epochs,
        claimable_map={201: (True, False), 202: (True, False)},
        close_ts_map={201: 0, 202: 0},
        wallet_balance=10.0,
        claim_outcome=ClaimSubmitResult(
            tx_hash="0xrevert", status="revert",
            included_block_number=22222, included_block_timestamp=10000,
        ),
    )
    sent_alerts: list[dict[str, Any]] = []
    monkeypatch.setattr(
        live_mod, "_send_claim_failure_alert",
        lambda **kw: sent_alerts.append(kw),
    )
    cursor_path = tmp_path / "cursor.txt"
    result = live_mod.claim_scan_cursor(
        **_scan_kwargs(contract=contract, cursor_path=cursor_path),
    )
    # Reverted batch is NOT counted as claimed.
    assert result.claimed_n == 0
    assert len(sent_alerts) == 1
    alert = sent_alerts[0]
    assert alert["reason"] == "revert"
    assert alert["tx_hash"] == "0xrevert"
    assert alert["epochs"] == [201, 202]
    # Cursor advances past the reverted batch (no retry on chain rejection).
    assert int(cursor_path.read_text()) == 2


# ---------------------------------------------------------------------------
# 3. Timeout outcome
# ---------------------------------------------------------------------------

def test_claim_timeout_alerts_and_parks_cursor(tmp_path, monkeypatch):
    epochs = [301, 302]
    contract = _FakeContract(
        user_rounds=epochs,
        claimable_map={301: (True, False), 302: (True, False)},
        close_ts_map={301: 0, 302: 0},
        wallet_balance=10.0,
        claim_outcome=ClaimSubmitResult(
            tx_hash="0xtimeout", status="timeout",
            included_block_number=None, included_block_timestamp=None,
        ),
    )
    sent_alerts: list[dict[str, Any]] = []
    monkeypatch.setattr(
        live_mod, "_send_claim_failure_alert",
        lambda **kw: sent_alerts.append(kw),
    )
    cursor_path = tmp_path / "cursor.txt"
    result = live_mod.claim_scan_cursor(
        **_scan_kwargs(contract=contract, cursor_path=cursor_path),
    )
    assert result.claimed_n == 0
    assert len(sent_alerts) == 1
    alert = sent_alerts[0]
    assert alert["reason"] == "timeout"
    assert alert["tx_hash"] == "0xtimeout"
    assert alert["epochs"] == [301, 302]
    # Cursor parked at first un-claimed epoch (the TX may still mine).
    # Position is the epoch's index within the user_rounds list.
    assert int(cursor_path.read_text()) == 0


# ---------------------------------------------------------------------------
# 4. Discord alert payload
# ---------------------------------------------------------------------------

def test_send_claim_failure_alert_posts_expected_payload(monkeypatch):
    """_send_claim_failure_alert posts the canonical CLAIM FAILED message."""
    captured: dict[str, Any] = {}

    class _FakeResponse:
        status_code = 204
        text = ""

    class _FakeRequests:
        def post(self, url, *, json, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse()

    monkeypatch.setenv(
        "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL",
        "https://discord.example/webhook/abc",
    )
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests())

    live_mod._send_claim_failure_alert(
        reason="revert",
        tx_hash="0xabc123",
        epochs=[401, 402, 403],
        gas_limit=900_000,
    )

    assert captured["url"] == "https://discord.example/webhook/abc"
    payload = captured["json"]
    assert payload["username"] == "PancakeBot-live"
    content = payload["content"]
    assert "PancakeBot-live CLAIM FAILED" in content
    assert "reason=`revert`" in content
    assert "tx=`0xabc123`" in content
    assert "401,402,403" in content
    assert "gas_limit=`900000`" in content


# ---------------------------------------------------------------------------
# 5. Discord alert disabled when webhook env var unset
# ---------------------------------------------------------------------------

def test_send_claim_failure_alert_no_webhook_does_not_raise(monkeypatch):
    """Missing webhook env var: log + return; do NOT raise."""
    monkeypatch.delenv(
        "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL", raising=False,
    )
    # Should not raise, should not import requests.
    live_mod._send_claim_failure_alert(
        reason="timeout",
        tx_hash="0xnone",
        epochs=[501],
        gas_limit=300_000,
    )


# ---------------------------------------------------------------------------
# 6. Receipt timeout threaded from cfg
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("timeout", [10, 35, 120])
def test_claim_receipt_timeout_threaded_from_kwarg(
    tmp_path, monkeypatch, timeout,
):
    contract = _FakeContract(
        user_rounds=[601, 602],
        claimable_map={601: (True, False), 602: (True, False)},
        close_ts_map={601: 0, 602: 0},
        wallet_balance=10.0,
        claim_outcome=ClaimSubmitResult(
            tx_hash="0x", status="success",
            included_block_number=1, included_block_timestamp=0,
        ),
    )
    monkeypatch.setattr(
        live_mod, "_send_claim_failure_alert", lambda **kw: None,
    )
    cursor_path = tmp_path / "cursor.txt"
    live_mod.claim_scan_cursor(
        **_scan_kwargs(
            contract=contract, cursor_path=cursor_path, claim_timeout=timeout,
        ),
    )
    assert contract.claim_calls[0]["receipt_timeout_seconds"] == timeout


# ---------------------------------------------------------------------------
# 7. Chunking: 25 pending epochs drain across 3 claim() TXs (10/10/5)
# ---------------------------------------------------------------------------

def test_claim_25_pending_chunks_into_three_txs(tmp_path, monkeypatch):
    """25 claimable epochs from one scan must split into 3 TXs of size
    10/10/5 (per ``_MAX_CLAIM_EPOCHS_PER_TX``). Each chunk gets its own
    ``contract.claim()`` invocation; the cursor advances past all 25
    after the 3rd succeeds.

    This is the realistic startup-after-outage scenario: bot was offline
    for ~24h, wins accumulated unclaimed, first scan after restart finds
    them all and must drain across multiple TXs without exceeding BSC
    per-TX gas budgets on public RPCs.
    """
    epochs = list(range(700, 725))  # 25 epochs, all claimable
    contract = _FakeContract(
        user_rounds=epochs,
        claimable_map={e: (True, False) for e in epochs},
        close_ts_map={e: 0 for e in epochs},
        wallet_balance=10.0,
        claim_outcome=ClaimSubmitResult(
            tx_hash="0xchunked", status="success",
            included_block_number=42, included_block_timestamp=10000,
        ),
    )
    monkeypatch.setattr(
        live_mod, "_send_claim_failure_alert", lambda **kw: None,
    )
    cursor_path = tmp_path / "cursor.txt"

    result = live_mod.claim_scan_cursor(
        **_scan_kwargs(contract=contract, cursor_path=cursor_path),
    )

    # All 25 reported claimed.
    assert result.claimed_n == 25
    # Three TXs: 10 + 10 + 5.
    assert len(contract.claim_calls) == 3
    assert contract.claim_calls[0]["epochs"] == list(range(700, 710))
    assert contract.claim_calls[1]["epochs"] == list(range(710, 720))
    assert contract.claim_calls[2]["epochs"] == list(range(720, 725))
    # Per-chunk gas_limit scales with chunk size.
    assert contract.claim_calls[0]["gas_limit"] == 300_000 * 10
    assert contract.claim_calls[1]["gas_limit"] == 300_000 * 10
    assert contract.claim_calls[2]["gas_limit"] == 300_000 * 5
    # Cursor advances past all 25.
    assert int(cursor_path.read_text()) == 25


def test_claim_chunking_timeout_mid_drain_parks_cursor(tmp_path, monkeypatch):
    """If the SECOND chunk times out mid-drain (chunk 1 succeeded), the
    drain halts. Cursor parks at the first un-claimed epoch (start of
    chunk 2); chunk-1's 10 epochs ARE counted as claimed. Next iteration
    will re-detect chunk-2+ and re-attempt.

    Exercises the principle: a timeout stops draining; success before
    the timeout is preserved.
    """
    epochs = list(range(800, 815))  # 15 epochs → would be 10 + 5

    # _FakeContract returns the same outcome for every claim() call by
    # default; build a variant that returns different outcomes per call.
    class _SequencedClaimContract(_FakeContract):
        def __init__(self, *, outcomes: list[ClaimSubmitResult], **kw):
            super().__init__(claim_outcome=outcomes[0], **kw)
            self._outcomes = outcomes
            self._call_idx = 0

        def claim(self, **kw):  # type: ignore[override]
            self.claim_calls.append({
                "epochs": list(kw["epochs"]),
                "gas_limit": int(kw["gas_limit"]),
                "gas_price_wei": int(kw["gas_price_wei"]),
                "wait_receipt": bool(kw["wait_receipt"]),
                "receipt_timeout_seconds": int(kw["receipt_timeout_seconds"]),
            })
            outcome = self._outcomes[
                min(self._call_idx, len(self._outcomes) - 1)
            ]
            self._call_idx += 1
            return outcome

    contract = _SequencedClaimContract(
        outcomes=[
            ClaimSubmitResult(
                tx_hash="0xchunk1ok", status="success",
                included_block_number=1, included_block_timestamp=0,
            ),
            ClaimSubmitResult(
                tx_hash="0xchunk2timeout", status="timeout",
                included_block_number=None, included_block_timestamp=None,
            ),
        ],
        user_rounds=epochs,
        claimable_map={e: (True, False) for e in epochs},
        close_ts_map={e: 0 for e in epochs},
        wallet_balance=10.0,
    )
    sent_alerts: list[dict[str, Any]] = []
    monkeypatch.setattr(
        live_mod, "_send_claim_failure_alert",
        lambda **kw: sent_alerts.append(kw),
    )
    cursor_path = tmp_path / "cursor.txt"

    result = live_mod.claim_scan_cursor(
        **_scan_kwargs(contract=contract, cursor_path=cursor_path),
    )

    # Only chunk 1's 10 epochs were successfully claimed.
    assert result.claimed_n == 10
    # Two TXs attempted: chunk 1 (success) + chunk 2 (timeout).
    assert len(contract.claim_calls) == 2
    assert contract.claim_calls[0]["epochs"] == list(range(800, 810))
    assert contract.claim_calls[1]["epochs"] == list(range(810, 815))
    # Timeout alert was sent for chunk 2 only.
    assert len(sent_alerts) == 1
    assert sent_alerts[0]["reason"] == "timeout"
    assert sent_alerts[0]["epochs"] == list(range(810, 815))
    # Cursor parks at the first un-claimed epoch (chunk-2 start = index 10).
    assert int(cursor_path.read_text()) == 10
