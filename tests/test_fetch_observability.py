"""Tests for the 2026-05-17 fetch-observability additions.

Four layers verified:

1. ``TransientOkxError`` carries structured fields (error_class,
   error_detail, rtt_ms, received_count, requested_count,
   missing_position) -- all optional, backwards compatible with bare
   construction.

2. ``MomentumGate.last_fetch_results`` is populated after evaluate(),
   producing one entry per attempted symbol in the agreed format
   (``"ok"`` / ``"partial:got_N_expected_M"`` / ``"error:<detail>"``).
   Reset between evaluate() calls (no cross-round leakage).

3. ``cycle_audit.csv`` schema includes the 5 new columns
   (``wake_mode``, ``kline_fire_offset_before_lock_ms``,
   ``btc_fetch_result``, ``eth_fetch_result``, ``sol_fetch_result``)
   and ``record_cycle_audit`` propagates them faithfully, including
   the ``"not_fetched"`` default when the gate didn't run.

4. ``PARTIAL`` log line (gate + sync) uses SUB=``KLINE`` (subsystem
   identifier) with the symbol value in the kv tail as ``symbol=BTC-USDT``.
   Field order: ``symbol -> reason -> received -> requested -> bar
   -> error_detail (conditional)``. Reason ``okx_publish_delay`` fires
   on partial-data responses; ``partial_response`` on non-data errors
   with bytes received; ``okx_unreachable`` on pre-response failures.

Run:
    python -m pytest tests/test_fetch_observability.py -v
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime.audit import (  # noqa: E402
    ensure_cycle_audit_csv,
    record_cycle_audit,
    _CYCLE_AUDIT_HEADER_OK_PATHS,
)
from pancakebot.util import TransientOkxError  # noqa: E402


# ---------------------------------------------------------------------------
# (1) TransientOkxError structured fields
# ---------------------------------------------------------------------------

def test_transient_okx_error_bare_construction_backwards_compatible():
    """``raise TransientOkxError("msg")`` still works without keyword args."""
    err = TransientOkxError("kline_fetch_exhausted: symbol=BTC-USDT")
    assert str(err) == "kline_fetch_exhausted: symbol=BTC-USDT"
    assert err.error_class is None
    assert err.error_detail is None
    assert err.rtt_ms is None


def test_transient_okx_error_structured_fields():
    """When raised with kwargs, fields surface for downstream parsing."""
    err = TransientOkxError(
        "kline_fetch_exhausted: symbol=BTC-USDT class=insufficient detail=got_15_expected_16",
        error_class="insufficient",
        error_detail="got_15_expected_16",
        rtt_ms=287,
    )
    assert err.error_class == "insufficient"
    assert err.error_detail == "got_15_expected_16"
    assert err.rtt_ms == 287


def test_transient_okx_error_rtt_ms_optional():
    """``rtt_ms`` defaults to None when omitted (pre-response failure case)."""
    err = TransientOkxError(
        "kline_fetch_exhausted: symbol=BTC-USDT class=retryable detail=ConnectionError",
        error_class="retryable",
        error_detail="ConnectionError",
    )
    assert err.error_class == "retryable"
    assert err.error_detail == "ConnectionError"
    assert err.rtt_ms is None


# ---------------------------------------------------------------------------
# (2) MomentumGate.last_fetch_results
# ---------------------------------------------------------------------------

def _make_gate():
    """Build a MomentumGate with stubbed config + client + executor.

    We don't run a real evaluate() (that requires network + real anchor
    poll); instead we test that the result-code classifier in evaluate()
    produces the expected codes via a focused mock of the OKX fetch
    return path. The full evaluate() integration is exercised by the
    canonical in-process backtest test.
    """
    from pancakebot.strategy.momentum_gate import (
        MomentumGate,
        MomentumGateConfig,
    )

    cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=2,
        mtf_lookbacks=(3, 7, 15),
        mtf_min_return_threshold=0.0001,
        max_consecutive_kline_fetch_failures=5,
    )
    # Minimal client; we patch kline_fetch_window per test.
    client = mock.MagicMock()
    return MomentumGate(okx_client=client, config=cfg)


def _stub_kline_window_full(_self, *, symbol, **kwargs):
    """Return a valid 16-row 1s-candle window for any symbol."""
    base_ts = kwargs["oldest_open_ms"]
    rows = [
        [base_ts + i * 1000, 100.0, 100.0, 100.0, 100.0, 1.0]
        for i in range(16)
    ]
    return rows, 250


def test_last_fetch_results_all_ok(monkeypatch):
    """All 3 symbols succeed → results dict shows ``"ok"`` for each."""
    gate = _make_gate()
    monkeypatch.setattr(
        gate._client, "kline_fetch_window",
        lambda **kwargs: _stub_kline_window_full(None, **kwargs),
    )
    # The signal computation path needs a base lock_at_ms; pick any
    # 1s-aligned timestamp.
    lock_at_ms = 1_700_000_000_000
    result = gate.evaluate(lock_at_ms=lock_at_ms)  # noqa: F841 — only inspecting side effects

    assert gate.last_fetch_results is not None
    assert gate.last_fetch_results == {"btc": "ok", "eth": "ok", "sol": "ok"}


def test_last_fetch_results_partial_classifies_got_n_expected_m(monkeypatch):
    """A TransientOkxError with class=insufficient + detail=got_15_expected_16
    produces ``"partial:got_15_expected_16"``. The RTT of the partial
    fetch is also captured in ``last_fetch_timing`` (response WAS received,
    just incomplete) so offline analysis can compare RTT across ok / partial
    rows of the same wake_mode."""
    gate = _make_gate()

    def stub(*, symbol, **kwargs):
        if symbol == "BTC-USDT":
            raise TransientOkxError(
                "kline_fetch_exhausted: symbol=BTC-USDT class=insufficient "
                "detail=got_15_expected_16",
                error_class="insufficient",
                error_detail="got_15_expected_16",
                rtt_ms=287,
            )
        return _stub_kline_window_full(None, symbol=symbol, **kwargs)

    monkeypatch.setattr(gate._client, "kline_fetch_window", stub)
    lock_at_ms = 1_700_000_000_000
    gate.evaluate(lock_at_ms=lock_at_ms)

    assert gate.last_fetch_results == {
        "btc": "partial:got_15_expected_16",
        "eth": "ok",
        "sol": "ok",
    }
    # Partial RTT is captured -- response was received from OKX, just
    # one row short. Downstream audit-CSV writer will populate
    # ``btc_fetch_ms`` from this same dict.
    assert gate.last_fetch_timing is not None
    assert gate.last_fetch_timing.get("btc_ms") == 287


def test_last_fetch_results_error_with_rtt_response_received(monkeypatch):
    """A response-path error (e.g. HTTP 429) DID receive bytes from OKX,
    so RTT is meaningful and captured."""
    gate = _make_gate()

    def stub(*, symbol, **kwargs):
        if symbol == "SOL-USDT":
            raise TransientOkxError(
                "kline_fetch_exhausted: symbol=SOL-USDT class=retryable "
                "detail=http_429",
                error_class="retryable",
                error_detail="http_429",
                rtt_ms=412,
            )
        return _stub_kline_window_full(None, symbol=symbol, **kwargs)

    monkeypatch.setattr(gate._client, "kline_fetch_window", stub)
    lock_at_ms = 1_700_000_000_000
    gate.evaluate(lock_at_ms=lock_at_ms)

    assert gate.last_fetch_results == {
        "btc": "ok",
        "eth": "ok",
        "sol": "error:http_429",
    }
    assert gate.last_fetch_timing is not None
    assert gate.last_fetch_timing.get("sol_ms") == 412


def test_last_fetch_results_error_no_rtt_pre_response_failure(monkeypatch):
    """A pre-response network error (DNS / connect refused) has rtt_ms=None.
    The error class still surfaces in last_fetch_results but the timing
    dict has no entry for that symbol (per okx_client policy: only
    response-received RTTs are recorded)."""
    gate = _make_gate()

    def stub(*, symbol, **kwargs):
        if symbol == "ETH-USDT":
            raise TransientOkxError(
                "kline_fetch_exhausted: symbol=ETH-USDT class=retryable "
                "detail=ConnectionError",
                error_class="retryable",
                error_detail="ConnectionError",
                # No rtt_ms -> okx_client suppressed it because no response was received.
            )
        return _stub_kline_window_full(None, symbol=symbol, **kwargs)

    monkeypatch.setattr(gate._client, "kline_fetch_window", stub)
    lock_at_ms = 1_700_000_000_000
    gate.evaluate(lock_at_ms=lock_at_ms)

    assert gate.last_fetch_results == {
        "btc": "ok",
        "eth": "error:ConnectionError",
        "sol": "ok",
    }
    # No eth_ms key when no RTT was recorded: downstream audit writes "".
    assert gate.last_fetch_timing is not None
    assert "eth_ms" not in gate.last_fetch_timing
    assert gate.last_fetch_timing.get("btc_ms") is not None
    assert gate.last_fetch_timing.get("sol_ms") is not None


def test_last_fetch_results_mixed_modes_single_round(monkeypatch):
    """Realistic worst-case: BTC partial-with-RTT, ETH pre-response failure
    (no RTT), SOL ok. last_fetch_timing carries btc_ms + sol_ms but not
    eth_ms; last_fetch_results carries all three with the right codes.
    The audit consumer then writes btc_fetch_ms=<rtt>, eth_fetch_ms="",
    sol_fetch_ms=<rtt> alongside the three result codes -- the exact
    pattern operator analysis needs to differentiate "OKX slow to publish"
    (partial+rtt) from "OKX unreachable" (error+no-rtt) in a single round.
    """
    gate = _make_gate()

    def stub(*, symbol, **kwargs):
        if symbol == "BTC-USDT":
            raise TransientOkxError(
                "kline_fetch_exhausted: symbol=BTC-USDT class=insufficient "
                "detail=got_15_expected_16",
                error_class="insufficient",
                error_detail="got_15_expected_16",
                rtt_ms=287,
            )
        if symbol == "ETH-USDT":
            raise TransientOkxError(
                "kline_fetch_exhausted: symbol=ETH-USDT class=retryable "
                "detail=ConnectionError",
                error_class="retryable",
                error_detail="ConnectionError",
                # No rtt_ms -- pre-response failure.
            )
        return _stub_kline_window_full(None, symbol=symbol, **kwargs)

    monkeypatch.setattr(gate._client, "kline_fetch_window", stub)
    lock_at_ms = 1_700_000_000_000
    gate.evaluate(lock_at_ms=lock_at_ms)

    assert gate.last_fetch_results == {
        "btc": "partial:got_15_expected_16",
        "eth": "error:ConnectionError",
        "sol": "ok",
    }
    assert gate.last_fetch_timing is not None
    # BTC and SOL have RTT; ETH does not (pre-response failure).
    assert gate.last_fetch_timing.get("btc_ms") == 287
    assert gate.last_fetch_timing.get("sol_ms") is not None
    assert "eth_ms" not in gate.last_fetch_timing


def test_last_fetch_results_reset_between_evaluations(monkeypatch):
    """``evaluate()`` resets last_fetch_results to None at entry, then
    re-populates. A prior round's results must NOT leak into a fresh
    evaluate() that errors before the result-classifier runs."""
    gate = _make_gate()
    monkeypatch.setattr(
        gate._client, "kline_fetch_window",
        lambda **kwargs: _stub_kline_window_full(None, **kwargs),
    )
    lock_at_ms = 1_700_000_000_000
    gate.evaluate(lock_at_ms=lock_at_ms)
    assert gate.last_fetch_results is not None

    # Force a re-evaluate; the field should be reset (then re-populated
    # to the same shape since the stub still succeeds).
    gate.evaluate(lock_at_ms=lock_at_ms)
    assert gate.last_fetch_results == {"btc": "ok", "eth": "ok", "sol": "ok"}


# ---------------------------------------------------------------------------
# (3) cycle_audit.csv schema + propagation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_header_cache():
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()
    yield
    _CYCLE_AUDIT_HEADER_OK_PATHS.clear()


def test_cycle_audit_header_has_new_columns(tmp_path):
    """The 5 new columns appear at the END of the cycle_audit header."""
    csv_path = tmp_path / "cycle_audit.csv"
    header = ensure_cycle_audit_csv(str(csv_path))
    expected_tail = [
        "wake_mode",
        "kline_fire_offset_before_lock_ms",
        "btc_fetch_result",
        "eth_fetch_result",
        "sol_fetch_result",
    ]
    assert header[-5:] == expected_tail


def test_record_cycle_audit_writes_new_fields_partial_has_rtt(tmp_path):
    """A full record_cycle_audit call with the new kwargs lands in the CSV
    row at the right column positions. Verifies the partial-RTT contract:
    when result is ``partial:got_N_expected_M`` (call completed), the
    corresponding ``*_fetch_ms`` IS populated -- not blank."""
    csv_path = tmp_path / "cycle_audit.csv"

    # Minimal stub for ``closed`` and ``decision`` arguments.
    closed = mock.MagicMock()
    closed.strategy_pipeline = None  # forces router_mode + last_settled_epoch defaults

    record_cycle_audit(
        closed,
        cycle_audit_path=str(csv_path),
        current_epoch=481781,
        locked_epoch=481780,
        lock_ts=1778950000,
        cutoff_ts=1778949998,
        locked_price_bnbusd=650.0,
        action="SKIP",
        decision_stage="pipeline",
        open_round=None,
        bankroll_before_action_bnb=6.0,
        bankroll_after_action_bnb=6.0,
        skip_reason="kline_fetch_transient_failure",
        # Partial BTC: response received, RTT is real (287ms).
        btc_fetch_ms=287,
        eth_fetch_ms=255,
        sol_fetch_ms=262,
        wake_mode="dynamic",
        kline_fire_offset_before_lock_ms=852,
        btc_fetch_result="partial:got_15_expected_16",
        eth_fetch_result="ok",
        sol_fetch_result="ok",
    )

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["wake_mode"] == "dynamic"
    assert row["kline_fire_offset_before_lock_ms"] == "852"
    assert row["btc_fetch_result"] == "partial:got_15_expected_16"
    assert row["eth_fetch_result"] == "ok"
    assert row["sol_fetch_result"] == "ok"
    # Partial RTT contract: the call completed, so the RTT IS populated
    # alongside the partial result code. (Pre-2026-05-17 fix this column
    # was blank on partial; offline analysis couldn't compare RTT across
    # ok/partial rows of the same wake_mode.)
    assert row["btc_fetch_ms"] == "287"
    assert row["eth_fetch_ms"] == "255"
    assert row["sol_fetch_ms"] == "262"


def test_record_cycle_audit_pre_response_failure_blanks_rtt(tmp_path):
    """A pre-response failure (DNS / connect refused) has no meaningful
    OKX RTT to report. ``*_fetch_ms`` is blank; the result code surfaces
    the failure class. This is the only non-ok case where ``*_fetch_ms``
    SHOULD be blank."""
    csv_path = tmp_path / "cycle_audit.csv"
    closed = mock.MagicMock()
    closed.strategy_pipeline = None

    record_cycle_audit(
        closed,
        cycle_audit_path=str(csv_path),
        current_epoch=481781,
        locked_epoch=481780,
        lock_ts=1778950000,
        cutoff_ts=1778949998,
        locked_price_bnbusd=650.0,
        action="SKIP",
        decision_stage="pipeline",
        open_round=None,
        bankroll_before_action_bnb=6.0,
        bankroll_after_action_bnb=6.0,
        skip_reason="kline_fetch_transient_failure",
        # Pre-response failure on ETH; BTC + SOL ok.
        btc_fetch_ms=274,
        eth_fetch_ms=None,
        sol_fetch_ms=265,
        wake_mode="dynamic",
        kline_fire_offset_before_lock_ms=852,
        btc_fetch_result="ok",
        eth_fetch_result="error:ConnectionError",
        sol_fetch_result="ok",
    )

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    assert row["eth_fetch_result"] == "error:ConnectionError"
    assert row["eth_fetch_ms"] == ""  # No OKX RTT to report.
    assert row["btc_fetch_ms"] == "274"
    assert row["sol_fetch_ms"] == "265"


def test_record_cycle_audit_defaults_when_not_fetched(tmp_path):
    """Early-skip path (e.g. risk_bankroll_stale at the bankroll wake)
    that calls record_cycle_audit without fetch context produces empty
    wake_mode + empty fire_offset + "not_fetched" results."""
    csv_path = tmp_path / "cycle_audit.csv"
    closed = mock.MagicMock()
    closed.strategy_pipeline = None

    record_cycle_audit(
        closed,
        cycle_audit_path=str(csv_path),
        current_epoch=1,
        locked_epoch=0,
        lock_ts=1700000000,
        cutoff_ts=1699999998,
        locked_price_bnbusd=0.0,
        action="SKIP",
        decision_stage="pipeline",
        open_round=None,
        bankroll_before_action_bnb=1.0,
        bankroll_after_action_bnb=1.0,
        skip_reason="risk_bankroll_stale",
        # No wake_mode / fire_offset / result kwargs supplied -> defaults.
    )

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["wake_mode"] == ""
    assert rows[0]["kline_fire_offset_before_lock_ms"] == ""
    assert rows[0]["btc_fetch_result"] == "not_fetched"
    assert rows[0]["eth_fetch_result"] == "not_fetched"
    assert rows[0]["sol_fetch_result"] == "not_fetched"


# ---------------------------------------------------------------------------
# (4) okx_client received_count computation + gate KLINE PARTIAL emission
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock

from pancakebot.market_data.okx_client import OkxClient, RetryPolicy  # noqa: E402


def _make_okx_client_with_response(response_payload: dict) -> tuple[OkxClient, MagicMock]:
    """Build an OkxClient whose session.get returns a canned response."""
    client = OkxClient(timeout_seconds=5.0)
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = response_payload
    client._session = MagicMock()
    client._session.get.return_value = fake_resp
    return client, fake_resp


def _candle_row(ts_ms: int) -> list:
    """OKX returns newest-first; rows shape: [ts_ms, o, h, l, c, v, ...]."""
    return [str(ts_ms), "100", "100", "100", "100", "1"]


def test_okx_client_carries_received_and_requested_counts_on_insufficient():
    """INSUFFICIENT response (got fewer rows than expected) surfaces both
    ``received_count`` and ``requested_count`` on the raised exception.
    Boundary position (newest vs oldest vs middle) is NOT computed --
    OKX only ever shorts us at the tail in practice and supporting
    hypothetical middle-gap / oldest-missing cases is overengineering.
    """
    requested_count = 16
    oldest = 1_700_000_000_000
    newest = oldest + (requested_count - 1) * 1000
    rows = [_candle_row(newest - 1000 - i * 1000) for i in range(requested_count - 1)]
    client, _ = _make_okx_client_with_response({"code": "0", "data": rows})
    raised: TransientOkxError | None = None
    try:
        client.kline_fetch_window(
            symbol="BTC-USDT",
            oldest_open_ms=oldest,
            newest_open_ms_inclusive=newest,
            retry_policy=RetryPolicy(max_attempts=1, backoff_seconds=()),
        )
    except TransientOkxError as e:
        raised = e
    assert raised is not None
    assert raised.received_count == 15
    assert raised.requested_count == 16
    # ``missing_position`` is intentionally not a field anymore.
    assert not hasattr(raised, "missing_position")


def test_gate_kline_part_publish_delay_reason_and_field_order(monkeypatch):
    """Partial response (insufficient + got_N): reason=okx_publish_delay.
    Field order in the log line is: symbol, reason, received, requested,
    bar. error_detail is OMITTED because ``got_15_expected_16`` is pure
    count-restatement (derivable from received/requested)."""
    from pancakebot import log as log_module

    gate = _make_gate()

    def stub(*, symbol, **kwargs):
        if symbol == "BTC-USDT":
            raise TransientOkxError(
                "kline_fetch_exhausted: symbol=BTC-USDT class=insufficient "
                "detail=got_15_expected_16",
                error_class="insufficient",
                error_detail="got_15_expected_16",
                rtt_ms=287,
                received_count=15,
                requested_count=16,
            )
        return _stub_kline_window_full(None, symbol=symbol, **kwargs)

    monkeypatch.setattr(gate._client, "kline_fetch_window", stub)

    warn_calls: list[dict] = []

    def fake_emit(level, sys_name, sub, event, *, msg=None, **fields):
        if event == "PARTIAL":
            warn_calls.append({"sub": sub, "fields": dict(fields)})

    monkeypatch.setattr(log_module, "_emit", fake_emit)

    lock_at_ms = 1_700_000_000_000
    gate.evaluate(lock_at_ms=lock_at_ms)

    assert len(warn_calls) == 1
    fields = warn_calls[0]["fields"]
    # SUB is the subsystem identifier ("KLINE"), NOT the symbol value.
    # The symbol surfaces in the kv tail as ``symbol=BTC-USDT``.
    assert warn_calls[0]["sub"] == "KLINE"
    # Reason: high-confidence publish_delay (partial-data response,
    # detail starts with "got_").
    assert fields["reason"] == "okx_publish_delay"
    assert fields["symbol"] == "BTC-USDT"
    assert fields["received"] == 15
    assert fields["requested"] == 16
    assert fields["bar"] == "1s"
    # Dropped redundant / removed fields:
    assert "missing_position" not in fields
    assert "missing_count" not in fields
    assert "error_class" not in fields
    # error_detail="got_15_expected_16" is count-derivable -> omitted.
    assert "error_detail" not in fields
    # Field ORDER (operator-triage priority):
    assert list(fields.keys()) == ["symbol", "reason", "received", "requested", "bar"]


def test_gate_kline_part_partial_response_keeps_informative_error_detail(monkeypatch):
    """HTTP 429 / OKX code response: rtt_ms set but error_detail is NOT
    derivable from counts (received=0 doesn't tell you "rate limited").
    error_detail is kept; reason=partial_response."""
    from pancakebot import log as log_module

    gate = _make_gate()

    def stub(*, symbol, **kwargs):
        if symbol == "BTC-USDT":
            raise TransientOkxError(
                "kline_fetch_exhausted: symbol=BTC-USDT class=retryable "
                "detail=http_429",
                error_class="retryable",
                error_detail="http_429",
                rtt_ms=412,
                # No received_count -- the response was non-200 so we
                # didn't parse the body.
            )
        return _stub_kline_window_full(None, symbol=symbol, **kwargs)

    monkeypatch.setattr(gate._client, "kline_fetch_window", stub)

    warn_calls: list[dict] = []

    def fake_emit(level, sys_name, sub, event, *, msg=None, **fields):
        if event == "PARTIAL":
            warn_calls.append({"sub": sub, "fields": dict(fields)})

    monkeypatch.setattr(log_module, "_emit", fake_emit)

    lock_at_ms = 1_700_000_000_000
    gate.evaluate(lock_at_ms=lock_at_ms)

    assert len(warn_calls) == 1
    fields = warn_calls[0]["fields"]
    assert fields["reason"] == "partial_response"
    assert fields["error_detail"] == "http_429"
    assert "received" not in fields  # No body-parse happened.
    assert "requested" not in fields


def test_gate_kline_part_warn_unreachable_on_pre_response_failure(monkeypatch):
    """Pre-response failure (rtt_ms=None) -> reason='okx_unreachable',
    no received / requested / missing fields."""
    from pancakebot import log as log_module

    gate = _make_gate()

    def stub(*, symbol, **kwargs):
        if symbol == "BTC-USDT":
            raise TransientOkxError(
                "kline_fetch_exhausted: symbol=BTC-USDT class=retryable "
                "detail=ConnectionError",
                error_class="retryable",
                error_detail="ConnectionError",
                # No rtt_ms, no received_count -- pre-response failure.
            )
        return _stub_kline_window_full(None, symbol=symbol, **kwargs)

    monkeypatch.setattr(gate._client, "kline_fetch_window", stub)

    warn_calls: list[dict] = []

    def fake_emit(level, sys_name, sub, event, *, msg=None, **fields):
        if event == "PARTIAL":
            warn_calls.append({"sub": sub, "fields": dict(fields)})

    monkeypatch.setattr(log_module, "_emit", fake_emit)

    lock_at_ms = 1_700_000_000_000
    gate.evaluate(lock_at_ms=lock_at_ms)

    assert len(warn_calls) == 1
    fields = warn_calls[0]["fields"]
    assert fields["reason"] == "okx_unreachable"
    assert "missing_position" not in fields
    assert "missing_count" not in fields
    assert "received" not in fields
    assert "requested" not in fields
    # error_class was dropped from the log line (not derivable beyond
    # what reason already conveys). error_detail kept because the
    # exception class name (ConnectionError, ConnectTimeout, etc.) is
    # operator-meaningful and not derivable from anything else.
    assert "error_class" not in fields
    assert fields["error_detail"] == "ConnectionError"


def test_okx_client_does_not_emit_exhaust_log(monkeypatch):
    """Regression: okx_client no longer logs ``error("NET", "OKX",
    "EXHAUST", ...)`` itself. The structured exception is the operator
    surface; callers (gate + sync) format the KLINE PARTIAL log."""
    requested_count = 16
    oldest = 1_700_000_000_000
    newest = oldest + (requested_count - 1) * 1000
    rows = [_candle_row(newest - 1000 - i * 1000) for i in range(requested_count - 1)]
    client, _ = _make_okx_client_with_response({"code": "0", "data": rows})

    # Patch okx_client's `error` import would be ideal, but we removed
    # that import entirely. Verify by ensuring no AttributeError on
    # access (the import is gone).
    import pancakebot.market_data.okx_client as okx_client_mod
    assert not hasattr(okx_client_mod, "error"), (
        "okx_client.error import should have been removed; the EXHAUST log "
        "is the caller's responsibility"
    )
