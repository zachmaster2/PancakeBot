"""The open_stake provider must actually be CALLABLE in the live build.

WHY THIS FILE EXISTS, AND WHY IT IS SHAPED LIKE THIS.

`_build_momentum_pipeline` wired the provider as a closure over a bare
`bet_ledger`, a name never imported at module scope in dry.py (the only
import was a function-local `... as _bet_ledger` inside a different
function). Every call raised NameError. `_open_stake_bnb` caught bare
Exception and returned 0.0 by design, so the drawdown breaker silently
compared the RAW wallet balance -- exactly the pre-fix behaviour the fix
existed to remove -- for four days, ending in a phantom ~24h suspension
on 2026-08-28 16:29:01 (tripped at 15.7% on raw when the settled
equivalent was 13.1%, on a live-money account).

The 35 tests in test_breaker_pending_stake.py were all green throughout.
They had to be: every one of them constructs MomentumOnlyPipeline
DIRECTLY and hands it a working provider, so they exercise the arithmetic
and never the wiring. A daily reconstruction from bets.jsonl was also
green, for the same reason at one remove -- it recomputed what the code
SHOULD produce from the same inputs, and agreed with itself.

    An independent reconstruction validates the SPEC, never the
    IMPLEMENTATION. Checking the implementation requires an artifact the
    ENGINE ITSELF emits.

Here that artifact is the `open_stake` column in bankroll_history.jsonl.
It is written only when the pipeline's own provider returns non-zero, so
it cannot be produced by a test that supplies its own provider. Its
absence was the one live trace of this bug -- present for four days, and
twice explained away as benign.

So these tests:
  * build the pipeline through the REAL factory, never by hand;
  * assert on the engine-written artifact, not on a recomputation;
  * and guard the specific defect shape (two names for one module).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import paths as pb_paths  # noqa: E402
from pancakebot.bankroll_tracker import PersistedBankrollTracker  # noqa: E402
from pancakebot.config import load_app_config  # noqa: E402
from pancakebot.runtime import dry  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402

STAKE = 0.05


def _open_ledger(tmp_path: Path) -> str:
    """A ledger with ONE genuinely open position (SUBMITTED + CONFIRMED)."""
    p = tmp_path / "bets.jsonl"
    p.write_text(
        json.dumps({"epoch": 510951, "status": "SUBMITTED",
                    "amount_bnb": STAKE, "side": "Bull"}) + "\n"
        + json.dumps({"epoch": 510951, "status": "CONFIRMED"}) + "\n",
        encoding="utf-8")
    return str(p)


def _live_cfg():
    """The attributes `_build_momentum_pipeline` actually reads, with
    dry=False so the LIVE branch that wires the provider is taken."""
    strategy = load_app_config(str(_REPO_ROOT / "config.toml")).strategy
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT", kline_cutoff_seconds=2,
        mtf_lookbacks=strategy.gate.mtf_lookbacks,
        mtf_min_return_threshold=strategy.gate.mtf_min_return_threshold,
    )
    return SimpleNamespace(
        momentum_gate_config=gate_cfg, strategy=strategy, momentum_gate=None,
        kline_cutoff_seconds=2, pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001, treasury_fee_fraction=0.03, dry=False,
    )


def _live_pipeline(monkeypatch, ledger_path):
    """Pipeline from the REAL factory, with the live ledger path redirected."""
    monkeypatch.setattr(pb_paths, "LIVE_BETS_LEDGER_PATH", ledger_path)
    return dry._build_momentum_pipeline(cfg=_live_cfg())


# ---- the wiring itself ----------------------------------------------------

def test_the_live_provider_returns_the_open_stake_not_zero(monkeypatch,
                                                           tmp_path):
    """THE regression, at its shortest. Before the fix this returned 0.0 --
    silently, on every round, forever."""
    pipe = _live_pipeline(monkeypatch, _open_ledger(tmp_path))
    assert pipe._open_stake_provider is not None, "live must wire a provider"
    assert pipe._open_stake_bnb() == pytest.approx(STAKE), (
        "the live provider did not report the open position; if this is 0.0 "
        "the drawdown breaker is reading the RAW wallet balance and will "
        "trip early on a pending bet")


def test_the_provider_closure_resolves_its_own_module_name(monkeypatch,
                                                           tmp_path):
    """`_open_stake_bnb` swallows the failure, so call the closure DIRECTLY:
    a NameError here is the actual defect, undisguised."""
    pipe = _live_pipeline(monkeypatch, _open_ledger(tmp_path))
    assert pipe._open_stake_provider() == pytest.approx(STAKE)


def test_dry_leaves_the_provider_unwired(monkeypatch, tmp_path):
    """Dry settles synchronously off simulated_bankroll_bnb, which is
    already on a settled basis; adding open stakes there would double-count.
    The None must stay a deliberate choice, not become collateral damage."""
    monkeypatch.setattr(pb_paths, "LIVE_BETS_LEDGER_PATH",
                        _open_ledger(tmp_path))
    cfg = _live_cfg()
    cfg.dry = True
    assert dry._build_momentum_pipeline(cfg=cfg)._open_stake_provider is None


# ---- the ENGINE-EMITTED artifact -----------------------------------------

def test_record_settlement_writes_the_open_stake_column(monkeypatch,
                                                        tmp_path):
    """The artifact that would have caught this in production.

    bankroll_history.jsonl gains an `open_stake` key only when the
    pipeline's OWN provider returns non-zero. A test that supplies its own
    provider cannot produce this line, which is exactly why the previous
    suite missed the bug and why the column stayed absent for four days.
    """
    pipe = _live_pipeline(monkeypatch, _open_ledger(tmp_path))
    hist = tmp_path / "bankroll_history.jsonl"
    pipe.set_bankroll_tracker(PersistedBankrollTracker(
        path=hist, initial_bankroll=1.9022665550997389,
        drawdown_peak_window_days=7))
    pipe.record_settlement(bankroll=1.6026973411112821, start_at=1787934242)

    rows = [json.loads(x) for x in hist.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    sample = [r for r in rows if r.get("event") == "settlement"][-1]
    assert "open_stake" in sample, (
        "the settlement sample carries no open_stake column, so the engine "
        "saw nothing in flight while a CONFIRMED position was open — this "
        "is the exact live symptom of the 2026-08-28 wiring bug")
    assert sample["open_stake"] == pytest.approx(STAKE)


def test_the_incident_numbers_do_not_trip_once_the_provider_works(monkeypatch,
                                                                  tmp_path):
    """End-to-end on the real 2026-08-28 figures: raw 15.75% (tripped),
    settled-equivalent 13.12% (must not)."""
    pipe = _live_pipeline(monkeypatch, _open_ledger(tmp_path))
    peak, raw = 1.9022665550997389, 1.6026973411112821
    settled = raw + pipe._open_stake_bnb()
    assert (peak - raw) / peak == pytest.approx(0.157479, abs=1e-5)
    assert (peak - settled) / peak == pytest.approx(0.131196, abs=1e-5)
    assert (peak - settled) / peak < 0.15, "the 2026-08-28 phantom must not recur"


# ---- a wiring fault must be LOUD -----------------------------------------

def test_a_wiring_fault_warns_once_and_still_degrades(monkeypatch, tmp_path):
    """The bare-Exception net stays for TRANSIENT faults but must not hide a
    programming error. NameError/AttributeError/ImportError/TypeError get one
    WARN per process; the return value still degrades to 0.0 so a live
    process is never taken down on the pre-lock path."""
    pipe = _live_pipeline(monkeypatch, _open_ledger(tmp_path))

    def boom() -> float:
        raise NameError("name 'bet_ledger' is not defined")

    pipe._open_stake_provider = boom
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr("pancakebot.strategy.momentum_pipeline.warn",
                        lambda tag, msg: seen.append((tag, msg)))

    assert pipe._open_stake_bnb() == 0.0        # still degrades
    assert len(seen) == 1, "a wiring fault must announce itself"
    assert "WIRING BUG" in seen[0][1]
    assert "NameError" in seen[0][1]
    for _ in range(50):                          # ~4h of rounds
        pipe._open_stake_bnb()
    assert len(seen) == 1, "once per process, not once per round"


def test_a_transient_fault_stays_quiet(monkeypatch, tmp_path):
    """An unreadable ledger self-corrects next round. Warning on it would
    train the operator to ignore the channel the wiring warning uses."""
    pipe = _live_pipeline(monkeypatch, _open_ledger(tmp_path))

    def flaky() -> float:
        raise OSError("temporarily unavailable")

    pipe._open_stake_provider = flaky
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr("pancakebot.strategy.momentum_pipeline.warn",
                        lambda tag, msg: seen.append((tag, msg)))
    assert pipe._open_stake_bnb() == 0.0
    assert seen == [], "a transient read failure must degrade silently"


# ---- guard the defect SHAPE ----------------------------------------------

def test_dry_has_exactly_one_name_for_the_bet_ledger_module():
    """The defect was two names for one module: a module-scope-absent
    `bet_ledger` and a function-local `_bet_ledger`. Re-introducing the
    second path is what re-introduces the bug."""
    src = Path(dry.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "bet_ledger." in code, "sanity: dry.py should use the module"
    bare = [ln for ln in code.splitlines()
            if "bet_ledger." in ln and "_bet_ledger." not in ln]
    assert not bare, (
        f"dry.py references a bare `bet_ledger` that is not imported at "
        f"module scope: {bare}")
    assert code.count("import bet_ledger") == 1, (
        "the bet_ledger module must be imported exactly once, at module "
        "scope — a second function-local alias is how the names diverged")
    assert hasattr(dry, "_bet_ledger"), "the alias must be module-scope"
