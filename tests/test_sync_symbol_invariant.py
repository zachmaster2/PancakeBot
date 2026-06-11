"""Regression: sync mode fetches all 4 symbols (BNB/BTC/ETH/SOL).

Live/dry mode dropped BNB from the per-round critical-path fetch
(commit a0ce1b1, follow-up to Phase 2 robustness §2.1) because the
strategy doesn't consume BNB closes for signal computation. Sync mode
intentionally keeps BNB so historical klines accumulate to disk for
any future strategy that needs them.

This test pins that invariant: ``sync_runtime_market_data`` must
schedule fetches for all four symbols. A future refactor that drops
BNB from the sync path would remove our ability to ever re-enable a
BNB-aware strategy without first re-running a long backfill.

Run::
    python -m pytest tests/test_sync_symbol_invariant.py -v
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data import sync as sync_module  # noqa: E402


_EXPECTED_SYNC_SYMBOLS = ("BNB-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT")


def test_sync_runtime_market_data_references_all_four_symbols():
    """Every symbol in ``_EXPECTED_SYNC_SYMBOLS`` must appear as a literal
    in ``sync_runtime_market_data``'s source. The sync path uses inline
    ``inst_id="..."`` arguments in the parallel ``pool.submit`` block,
    so a symbol-list change shows up as a source-text edit."""
    src = inspect.getsource(sync_module.sync_runtime_market_data)
    for sym in _EXPECTED_SYNC_SYMBOLS:
        assert sym in src, (
            f"sync_runtime_market_data must keep {sym!r} in the parallel "
            f"fetch block (data-preservation invariant). If you intentionally "
            f"removed it, update _EXPECTED_SYNC_SYMBOLS in this test AND "
            f"document the removal in project_pancakebot_timing_architecture_history.md "
            f"so a future revival doesn't have to re-backfill from scratch."
        )


def test_sync_module_kline_paths_cover_all_four_symbols():
    """Module-level ``_*_KLINES_PATH`` constants must exist for all 4 symbols
    so the per-symbol KlineStore writes still work."""
    assert hasattr(sync_module, "_BNB_KLINES_PATH")
    assert hasattr(sync_module, "_BTC_KLINES_PATH")
    assert hasattr(sync_module, "_ETH_KLINES_PATH")
    assert hasattr(sync_module, "_SOL_KLINES_PATH")


def test_sync_summary_has_per_symbol_count_fields():
    """SyncSummary still reports per-symbol synced-count for all 4 symbols.
    Operators read these counts at the end of ``--sync`` to confirm a
    successful run; dropping BNB without updating the summary would
    silently mask a partial-sync regression."""
    fields = sync_module.SyncSummary.__dataclass_fields__
    for sym_short in ("bnb", "btc", "eth", "sol"):
        field_name = f"{sym_short}_klines_synced"
        assert field_name in fields, (
            f"SyncSummary must report {field_name}; missing this field "
            f"means a successful --sync would silently skip {sym_short.upper()} "
            f"counts in the operator's CORE/SYNC/DONE log line."
        )


def test_sync_integrity_check_is_subset_not_equality():
    """The Phase-3 integrity assertion must verify the kline TARGET set
    (the cache_n tail the backtest consumes) is covered by each kline
    store -- NOT equality against the whole round store. The Graph sync
    pages in 1000-round chunks, so a fresh clone with cache_n=100 stores
    1000 rounds while klines target only the 100-round tail; an equality
    check falsely trips sync_integrity_mismatch (caught in the 2026-06-11
    fresh-clone validation)."""
    src = inspect.getsource(sync_module.sync_runtime_market_data)
    assert "target_epochs - store.load_done_epochs()" in src, (
        "integrity check must be a subset/coverage check over the kline "
        "target epochs (the cache_n tail)"
    )
    assert "final_round_epochs != store_epochs" not in src, (
        "equality against the full round store breaks fresh-clone syncs "
        "whose cache_n is not a multiple of The Graph's 1000-round page"
    )
