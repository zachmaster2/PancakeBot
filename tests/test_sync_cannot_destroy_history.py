"""The daily sync must not be able to destroy irreplaceable history.

CONTEXT. On 2026-08-30 the project moved to data-collection-only and the
Frankfurt VM was destroyed. The five store files on the Windows machine
became the ONLY surviving copy of data that cannot be refetched -- OKX
serves 1s klines for ~171.6 days and nothing older exists anywhere.

Three defects were found by auditing every write path:

  1. repair_torn_tail() was CORRECT, TESTED FIVE TIMES, AND NEVER CALLED.
     Nothing in production invoked it. A kill mid-append leaves a trailing
     partial line; every reader raises jsonl_parse_failed on it, so an
     unrepaired torn tail makes EVERY subsequent scheduled sync fail
     identically and forever -- a permanent silent stop indistinguishable
     from a quiet week. That is the shape of the August outage.

     The test that mattered was never "does the function work" (it did) but
     "does anything CALL it" (nothing did). This file asserts the CALL SITE.

  2. Two paths REPLACE the whole store: KlineStore.rewrite() via the kline
     prepend, and tmp_path.replace(store_path) in the round backfill.
     Atomicity protects against a torn write, not a wrong one.

  3. purge_and_rewrite() was an in-place "w" truncate-and-rewrite with no
     temp file, no atomicity, and no callers. Deleted; it is in git history.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import paths  # noqa: E402
from pancakebot.market_data import kline_store as _kline_store  # noqa: E402
from pancakebot.market_data.store_rewrite_gate import (  # noqa: E402
    ENV_VAR,
    store_rewrite_allowed,
)

_ALL_FIVE = (
    paths.CLOSED_ROUNDS_PATH,
    paths.BNB_SPOT_PRICES_PATH,
    paths.BTC_SPOT_PRICES_PATH,
    paths.ETH_SPOT_PRICES_PATH,
    paths.SOL_SPOT_PRICES_PATH,
)


# ---- (1) THE CALL SITE, not the function ---------------------------------

class _Sentinel(Exception):
    """Raised to stop the sync immediately after the repair step."""


def test_the_sync_actually_calls_repair_torn_tail_on_all_five_stores():
    """THE regression this file exists for.

    repair_torn_tail passed five tests while nothing called it. Asserting
    the function works proves nothing about whether the system uses it, so
    this drives the real entry point and asserts the CALL happened.
    """
    from pancakebot import app

    seen: list[str] = []

    def _spy(path):
        seen.append(path)
        return 0

    with mock.patch.object(app, "repair_torn_tail", side_effect=_spy) as spy, \
         mock.patch.object(app, "choose_rpc_url", return_value="http://x"), \
         mock.patch.object(app, "Web3PredictionContract"), \
         mock.patch.object(app, "fetch_and_save_contract_constants"), \
         mock.patch.object(app, "load_env"), \
         mock.patch.object(app, "require_env", side_effect=_Sentinel):
        with pytest.raises(_Sentinel):
            app.run_from_config(
                config_path="config.toml",
                dry=False, backtest=False, sync=True,
            )

    assert spy.called, (
        "the sync never called repair_torn_tail -- a torn tail would make "
        "every future scheduled run fail forever and silently")
    assert list(seen) == list(_ALL_FIVE), (
        f"repair ran on {seen}, expected all five stores {list(_ALL_FIVE)}")


def test_the_repair_runs_before_any_store_is_read():
    """Ordering matters: repairing after the first read is useless, because
    the read is what raises jsonl_parse_failed."""
    src = io.open(Path(_REPO_ROOT / "pancakebot" / "app.py"),
                  encoding="utf-8").read()
    i_repair = src.index("repair_torn_tail(_store_path)")
    i_graph = src.index('require_env("THE_GRAPH_API_KEY")')
    i_sync = src.index("sync_runtime_market_data(")
    assert i_repair < i_graph < i_sync, (
        "the torn-tail repair must run before the stores are opened")


# ---- (2) the whole-store replace gate ------------------------------------

def test_the_gate_defaults_to_refusing():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(ENV_VAR, None)
        assert store_rewrite_allowed() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_the_gate_opens_only_on_explicit_optin(val):
    with mock.patch.dict(os.environ, {ENV_VAR: val}):
        assert store_rewrite_allowed() is True


@pytest.mark.parametrize("val", ["0", "no", "off", "", " ", "maybe", "2", "-1"])
def test_the_gate_fails_closed_on_anything_else(val):
    """A typo must fail toward preserving data, never toward rewriting it."""
    with mock.patch.dict(os.environ, {ENV_VAR: val}):
        assert store_rewrite_allowed() is False


def test_refusing_a_prepend_does_not_abort_the_forward_sync():
    """THE refinement that matters.

    If a discovered gap aborted the run, one gap would stop ALL forward
    data collection indefinitely -- converting a backfill need into the
    permanent quiet failure this design prevents. The refusal must warn
    and continue, never raise.
    """
    src = io.open(Path(_REPO_ROOT / "pancakebot" / "market_data" / "sync.py"),
                  encoding="utf-8").read()
    i = src.index("REFUSING to prepend")
    window = src[max(0, i - 600):i + 600]
    assert "raise" not in window, (
        "the prepend refusal raises -- one discovered gap would stop all "
        "forward data collection indefinitely")
    assert "warn(" in window, "a silent refusal is the failure mode itself"


def test_the_round_backfill_refusal_returns_instead_of_looping():
    """_ensure_min_count_by_scanning_older loops while stored_n < needed_n.
    Refusing without returning would spin forever."""
    src = io.open(Path(_REPO_ROOT / "pancakebot" / "market_data" / "round_sync.py"),
                  encoding="utf-8").read()
    i = src.index("REFUSING historical backfill")
    tail = src[i:i + 500]
    assert "return" in tail, (
        "the backfill refusal must return; the caller loops until the count "
        "grows and would otherwise spin forever")


def test_both_whole_store_replace_paths_are_gated():
    ks = io.open(Path(_REPO_ROOT / "pancakebot" / "market_data" / "sync.py"),
                 encoding="utf-8").read()
    rs = io.open(Path(_REPO_ROOT / "pancakebot" / "market_data" / "round_sync.py"),
                 encoding="utf-8").read()
    assert "_store_rewrite_allowed()" in ks, "kline prepend is ungated"
    assert "store_rewrite_allowed()" in rs, "round backfill is ungated"


# ---- (3) the deleted loaded gun ------------------------------------------

def test_purge_and_rewrite_is_gone():
    """An in-place "w" rewrite with no temp file, no atomicity and no
    callers is pure downside. It lives in git history if ever needed."""
    assert not hasattr(_kline_store.KlineStore, "purge_and_rewrite"), (
        "purge_and_rewrite is back -- an uncallable in-place truncate of an "
        "irreplaceable store")


# ---- the append-only property itself -------------------------------------

def test_the_normal_daily_path_is_append_only():
    """The two methods the daily sync actually uses must never truncate."""
    src = io.open(Path(_REPO_ROOT / "pancakebot" / "market_data" / "kline_store.py"),
                  encoding="utf-8").read()
    i = src.index("def append_after")
    body = src[i:src.index("def ", i + 10)]
    assert '"a"' in body, "append_after no longer opens in append mode"
    assert '"w"' not in body and "truncate" not in body

    src2 = io.open(Path(_REPO_ROOT / "pancakebot" / "market_data" / "round_store.py"),
                   encoding="utf-8").read()
    j = src2.index("def append_rounds_after")
    body2 = src2[j:src2.index("def ", j + 10)]
    assert "SEEK_END" in body2, "append_rounds_after no longer seeks to EOF"
    assert '"w"' not in body2 and "truncate" not in body2


def test_torn_tail_repair_is_the_only_sanctioned_truncation():
    """It removes a trailing partial line and refuses on body corruption."""
    src = io.open(Path(_REPO_ROOT / "pancakebot" / "market_data" / "round_store.py"),
                  encoding="utf-8").read()
    i = src.index("def repair_torn_tail")
    body = src[i:]
    assert "f.truncate(start)" in body
    assert "refusing to truncate" in body, (
        "repair_torn_tail must refuse when the damage is not a torn tail")
