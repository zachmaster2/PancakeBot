"""Phase 0: Feature analysis for regime/decision ML (recovered from audit/).

Recovered 2026-05-05 from C:/Users/zking/AppData/Local/Temp/audit/phase0_feature_analysis.py.
The recovered file is byte-identical in feature-construction logic; only changes:
  - Inlined `build_drift_lookup` (was imported from drift_overlay_asymmetric.py
    in the audit folder; that module is not in the repo).
  - Removed the audit-folder sys.path hack; uses repo paths only.
  - CLI flag `--out` lets caller redirect the JSON output (default
    `var/extended/phase0_feature_results.json`, same as the original).

Builds a feature matrix over all rounds in [EXT_LO..DATA_END_EPOCH] and
measures raw predictive power vs:
  - target_outcome:  bull(1)/bear(0) per round
  - target_pnl:      per-bet PnL (only rounds where canonical bet)

Features (26 total, in this order):
  Strategy-component (8):
    1. btc_mtf_strength  = |ret_3| + |ret_7| + |ret_15|  (BTC closes)
    2. eth_mtf_strength
    3. sol_mtf_strength
    4. bnb_mtf_strength
    5. agreement_count   = # of {BTC,ETH,SOL ret_15} that vote bull
    6. pool_size_bnb     = bull+bear pool at lock_at - 6
    7. pool_bull_ratio   = bull / (bull+bear)
    8. payout_leading    = total / leading * (1 - treasury_fee)

  Regime (12):
    9.  btc_vol_5m       (1s log-return std over 300 1s candles)
    10. btc_vol_15m      (5-min log-return std over 3 rounds)
    11. btc_vol_1h       (5-min log-return std over 12 rounds)
    12. btc_vol_4h       (5-min log-return std over 48 rounds)
    13. btc_ret_1h       (5-min log-return cumulative over 12 rounds)
    14. btc_ret_4h
    15. btc_ret_24h      (288 rounds)
    16. bnb_vol_5m, bnb_vol_15m, bnb_vol_1h
    17. hour_of_day (0..23, categorical)
    18. day_of_week (0..6, categorical)
    19. trailing_pnl_7d, trailing_pnl_14d, trailing_pnl_30d
    20. streak_length    (consecutive same-direction outcomes leading up to round)

  Drift (2):
    21. drift_bps
    22. time_since_chainlink_s

Targets are computed AT lock_at, so all features must use information
strictly available before lock_at - kline_cutoff_seconds.

Output:
  - ranked feature table (Pearson |r| with outcome)
  - decile / category analysis for top features
  - feature correlation matrix
  - regime-conditional analysis (pre-Dec / post-Dec)
  - JSON saved to var/extended/phase0_feature_results.json (or --out path)
"""
from __future__ import annotations

import argparse
import bisect
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Worktree path (for code imports). Data files in var/extended/ are gitignored
# so they live ONLY in the canonical repo, not in the worktree. We bind:
#   WT_REPO  -> code imports (research.in_process_runner, pancakebot.*)
#   REPO     -> reads of var/* and var/extended/* (canonical repo)
WT_REPO = Path(__file__).resolve().parents[1]
DATA_REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")
REPO = DATA_REPO
sys.path.insert(0, str(WT_REPO))

# Patch the in_process_runner module's REPO_ROOT-derived paths to point at
# the canonical repo so _load_klines_unified / _load_all_rounds can find
# var/extended/* (worktree var/extended is empty).
import research.in_process_runner as _ipr  # noqa: E402

def _retarget_path(canonical_attr: str, ext_attr: str | None = None) -> None:
    canonical = getattr(_ipr, canonical_attr, None)
    if canonical is not None and not canonical.exists():
        retargeted = DATA_REPO / canonical.relative_to(_ipr.REPO_ROOT)
        if retargeted.exists():
            setattr(_ipr, canonical_attr, retargeted)
    if ext_attr is not None:
        ext = getattr(_ipr, ext_attr, None)
        if ext is not None and not ext.exists():
            retargeted = DATA_REPO / ext.relative_to(_ipr.REPO_ROOT)
            setattr(_ipr, ext_attr, retargeted)

# Retarget the data file paths in in_process_runner
_ipr.REPO_ROOT = DATA_REPO
_ipr._BTC_KLINES_PATH = DATA_REPO / "var" / "btc_spot_prices.jsonl"
_ipr._ETH_KLINES_PATH = DATA_REPO / "var" / "eth_spot_prices.jsonl"
_ipr._SOL_KLINES_PATH = DATA_REPO / "var" / "sol_spot_prices.jsonl"
_ipr._CLOSED_ROUNDS_PATH = DATA_REPO / "var" / "closed_rounds.jsonl"
_ipr._EXT_BTC_KLINES_PATH = DATA_REPO / "var" / "extended" / "btc_spot_prices.jsonl"
_ipr._EXT_ETH_KLINES_PATH = DATA_REPO / "var" / "extended" / "eth_spot_prices.jsonl"
_ipr._EXT_SOL_KLINES_PATH = DATA_REPO / "var" / "extended" / "sol_spot_prices.jsonl"
_ipr._EXT_CLOSED_ROUNDS_PATH = DATA_REPO / "var" / "extended" / "closed_rounds.jsonl"

from research.in_process_runner import (  # noqa: E402
    FoldSpec, _load_all_rounds, _load_klines_unified, _compute_load_extent,
    _resolve_strategy_config, _slice_per_entry,
    _BTC_KLINES_PATH, _ETH_KLINES_PATH, _SOL_KLINES_PATH,
    _EXT_BTC_KLINES_PATH, _EXT_ETH_KLINES_PATH, _EXT_SOL_KLINES_PATH,
)
from pancakebot import paths as _paths  # noqa: E402
from pancakebot.constants import BNB_WEI, BACKTEST_GAS_COST_BET_BNB  # noqa: E402
from pancakebot.market_data.contract_constants import load_contract_constants  # noqa: E402
from pancakebot.settlement import settle_bet_against_closed_round  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402
from pancakebot.bankroll_tracker import InMemoryBankrollTracker  # noqa: E402

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

EXT_LO = 422298
DATA_END_EPOCH = 477065
CUTOFF_SECONDS = 2
POOL_CUTOFF_SECONDS = 6
LOOKBACKS = (3, 7, 15)
DEC_CUTOFF_EPOCH = 437562  # post-Dec starts here

CHAINLINK_PATH = REPO / "var" / "extended" / "chainlink_bnb_updates.jsonl"
BNB_KLINES_CANONICAL = REPO / _paths.BNB_SPOT_PRICES_PATH
BNB_KLINES_EXTENDED = REPO / "var" / "extended" / "bnb_spot_prices.jsonl"


# -----------------------------------------------------------------------------
# Drift lookup (inlined from audit/drift_overlay_asymmetric.py:build_drift_lookup)
# -----------------------------------------------------------------------------

def build_drift_lookup(*, ep_min: int | None = None,
                          ep_max: int | None = None) -> dict[int, float]:
    """Compute per-epoch drift_bps = (BNB_close - chainlink_price)/chainlink_price * 1e4
    using the latest chainlink update at lock_at.

    `ep_min`/`ep_max` filter BNB last-close loading for memory efficiency.
    """
    print("[drift] Loading Chainlink events...", flush=True)
    chainlink: list[dict] = []
    with open(CHAINLINK_PATH, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                chainlink.append(json.loads(s))
    chainlink.sort(key=lambda e: int(e["updated_at"]))
    cl_times = [int(e["updated_at"]) for e in chainlink]
    cl_prices = [float(e["current_dollars"]) for e in chainlink]
    print(f"[drift]   {len(chainlink)} chainlink updates", flush=True)

    print("[drift] Loading BNB last-close...", flush=True)
    bnb_close: dict[int, float] = {}
    for p in (REPO / "var" / "bnb_spot_prices.jsonl",
              REPO / "var" / "extended" / "bnb_spot_prices.jsonl"):
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)
                ep = int(rec["epoch"])
                if ep_min is not None and ep < ep_min:
                    continue
                if ep_max is not None and ep > ep_max:
                    continue
                klines = rec.get("klines_1s") or []
                if not klines:
                    continue
                try:
                    cp = float(klines[-1][4])
                except (TypeError, ValueError, IndexError):
                    continue
                if ep not in bnb_close:
                    bnb_close[ep] = cp
    print(f"[drift]   BNB close for {len(bnb_close)} epochs", flush=True)

    rounds = _load_all_rounds(use_extended_data=True)
    drift_by_epoch: dict[int, float] = {}

    def latest_at(ts: int) -> int | None:
        if not cl_times or cl_times[0] > ts:
            return None
        lo, hi = 0, len(cl_times) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cl_times[mid] <= ts:
                lo = mid
            else:
                hi = mid - 1
        return lo

    for r in rounds:
        if r.lock_at is None:
            continue
        idx = latest_at(int(r.lock_at))
        if idx is None:
            continue
        cl_p = cl_prices[idx]
        b = bnb_close.get(int(r.epoch))
        if b is None or cl_p <= 0:
            continue
        drift_by_epoch[int(r.epoch)] = (b - cl_p) / cl_p * 10000.0

    print(f"[drift]   computed drift for {len(drift_by_epoch)}/{len(rounds)} rounds", flush=True)
    return drift_by_epoch


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def safe_log_return(p_now: float, p_then: float) -> float | None:
    if p_now is None or p_then is None or p_now <= 0 or p_then <= 0:
        return None
    return math.log(p_now / p_then)


def get_return(closes: list[float], lookback: int) -> float | None:
    if len(closes) < lookback + 1:
        return None
    now = closes[-1]
    ago = closes[-(lookback + 1)]
    if ago <= 0:
        return None
    return (now / ago) - 1.0


def realized_vol_1s(klines: list[list], drop_last: int = 0) -> float | None:
    """Std of 1s log returns. drop_last lets caller exclude the last N candles
    (used for raw 300-candle BNB to mirror cutoff alignment with unified arrays
    that already have those candles dropped)."""
    if not klines:
        return None
    if drop_last > 0:
        candles = klines[:-drop_last]
    else:
        candles = klines
    if len(candles) < 30:
        return None
    closes = [float(k[4]) for k in candles]
    rets = []
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i-1] > 0:
            rets.append(math.log(closes[i] / closes[i-1]))
    if len(rets) < 2:
        return None
    arr = np.array(rets)
    return float(arr.std(ddof=1))


def epoch_ts(epoch: int) -> int:
    """Return start_at unix timestamp for an epoch (canonical floor)."""
    floor_start_at = 1765444670
    floor_epoch = 437562
    return floor_start_at + (epoch - floor_epoch) * 300


def epoch_to_iso(epoch: int) -> str:
    ts = epoch_ts(epoch)
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")


# -----------------------------------------------------------------------------
# Trailing PnL lookup (7d/14d/30d)
# -----------------------------------------------------------------------------

class TrailingPnLMulti:
    def __init__(self, history: list[tuple[int, float]]):
        history = sorted(history, key=lambda x: x[0])
        self.lock_ats = [h[0] for h in history]
        self.cumul = [0.0]
        for _, p in history:
            self.cumul.append(self.cumul[-1] + p)

    def trailing(self, current_lock_at: int, window_seconds: int) -> float:
        cutoff = current_lock_at - window_seconds
        i_lo = bisect.bisect_left(self.lock_ats, cutoff)
        i_hi = bisect.bisect_left(self.lock_ats, current_lock_at)
        return self.cumul[i_hi] - self.cumul[i_lo]


# -----------------------------------------------------------------------------
# Canonical pass — returns (lock_at, profit) and (lock_at, epoch) for matching
# -----------------------------------------------------------------------------

def canonical_pass_with_epoch(*, sim_rounds, btc_klines_full, eth_klines_full,
                               sol_klines_full, strategy_cfg, kline_cutoff_seconds,
                               pool_cutoff_seconds,
                               earliest_offset,
                               initial_bankroll_bnb, treasury_fee_fraction,
                               min_bet_amount_bnb):
    """Run canonical strategy, return list of (lock_at, profit, epoch, side) per BET."""
    max_lookback = max(strategy_cfg.gate.mtf_lookbacks)

    btc_klines = {ep: _slice_per_entry(kl, kline_cutoff_seconds=kline_cutoff_seconds,
                                          max_lookback=max_lookback,
                                          earliest_offset=earliest_offset)
                  for ep, kl in btc_klines_full.items()}
    eth_klines = {ep: _slice_per_entry(kl, kline_cutoff_seconds=kline_cutoff_seconds,
                                          max_lookback=max_lookback,
                                          earliest_offset=earliest_offset)
                  for ep, kl in eth_klines_full.items()}
    sol_klines = {ep: _slice_per_entry(kl, kline_cutoff_seconds=kline_cutoff_seconds,
                                          max_lookback=max_lookback,
                                          earliest_offset=earliest_offset)
                  for ep, kl in sol_klines_full.items()}

    gate_config = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=kline_cutoff_seconds,
        mtf_lookbacks=strategy_cfg.gate.mtf_lookbacks,
        mtf_min_return_threshold=strategy_cfg.gate.mtf_min_return_threshold,
    )
    bankroll_tracker = InMemoryBankrollTracker(
        initial_bankroll=initial_bankroll_bnb,
        drawdown_peak_window_days=strategy_cfg.risk.drawdown_peak_window_days,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_config, strategy_config=strategy_cfg, gate=None,
        kline_cutoff_seconds=kline_cutoff_seconds,
        pool_cutoff_seconds=pool_cutoff_seconds,
        min_bet_amount_bnb=min_bet_amount_bnb,
        treasury_fee_fraction=treasury_fee_fraction,
        bankroll_tracker=bankroll_tracker,
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    bankroll = initial_bankroll_bnb
    history: list[tuple[int, float, int, str]] = []  # (lock_at, profit, epoch, side)
    for round_t in sim_rounds:
        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action == "BET" and decision.bet_size_bnb > 0.0:
            bankroll -= decision.bet_size_bnb + BACKTEST_GAS_COST_BET_BNB
            outcome = settle_bet_against_closed_round(
                bet_bnb=decision.bet_size_bnb, bet_side=decision.bet_side,
                round_closed=round_t, treasury_fee_fraction=treasury_fee_fraction,
            )
            bankroll += outcome.credit_bnb
            profit = outcome.credit_bnb - decision.bet_size_bnb - BACKTEST_GAS_COST_BET_BNB
            history.append((int(round_t.lock_at), profit,
                             int(round_t.epoch), decision.bet_side))
        pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
        pipeline.settle_closed_rounds(rounds=[round_t])
    return history


# -----------------------------------------------------------------------------
# Build feature matrix
# -----------------------------------------------------------------------------

FEATURE_NAMES = [
    "btc_mtf_strength", "eth_mtf_strength", "sol_mtf_strength", "bnb_mtf_strength",
    "agreement_count", "pool_size_bnb", "pool_bull_ratio", "payout_leading",
    "btc_vol_5m", "btc_vol_15m", "btc_vol_1h", "btc_vol_4h",
    "btc_ret_1h", "btc_ret_4h", "btc_ret_24h",
    "bnb_vol_5m", "bnb_vol_15m", "bnb_vol_1h",
    "hour_of_day", "day_of_week",
    "trailing_pnl_7d", "trailing_pnl_14d", "trailing_pnl_30d",
    "streak_length",
    "drift_bps", "time_since_chainlink_s",
]
CATEGORICAL = {"hour_of_day", "day_of_week", "agreement_count"}


def _load_klines_unified_filtered(path, ext_path, *, earliest_offset, latest_offset,
                                     ep_min, ep_max):
    """Memory-efficient unified kline load: only stores epochs in [ep_min, ep_max].

    Mirrors research.in_process_runner._load_klines_unified slicing semantics
    (start_neg = -(earliest_offset-1), end_neg = -(latest_offset-2) or None).
    """
    if latest_offset < 2:
        raise ValueError("latest_offset_must_be_ge_2")
    start_neg = -(earliest_offset - 1)
    end_neg = None if latest_offset == 2 else -(latest_offset - 2)
    result: dict[int, list[list]] = {}

    def _ingest(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error") or rec.get("klines_1s") is None:
                    continue
                ep = int(rec["epoch"])
                if not (ep_min <= ep <= ep_max):
                    continue
                kl = rec["klines_1s"]
                if end_neg is None:
                    kl = kl[start_neg:]
                else:
                    kl = kl[start_neg:end_neg]
                if ep not in result:
                    result[ep] = kl

    if path.exists():
        _ingest(path)
    if ext_path is not None and ext_path.exists():
        _ingest(ext_path)
    return result


def _load_bnb_raw_filtered(*, ep_min: int, ep_max: int):
    """BNB raw klines, filtered to [ep_min, ep_max]. Returns dict[ep -> klines_1s]."""
    result: dict[int, list[list]] = {}
    for p in (BNB_KLINES_CANONICAL, BNB_KLINES_EXTENDED):
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)
                ep = int(rec["epoch"])
                if not (ep_min <= ep <= ep_max):
                    continue
                klines = rec.get("klines_1s") or []
                if klines and ep not in result:
                    result[ep] = klines
    return result


def build_feature_matrix(*, return_extras: bool = False,
                            ep_min: int | None = None,
                            ep_max: int | None = None):
    """Build the feature matrix without running the JSON-emit step.

    Returns:
        (FEATURE_NAMES, F, target_outcome, target_pnl, epoch_arr, all_rounds,
         canonical_history)

    `return_extras=True` includes additional per-round info needed by the
    iteration-1 magnitude regressor (close_price, lock_price, bnb_vol_5m
    pre-computed, etc.).

    `ep_min`/`ep_max` enable memory-efficient kline loading when set; default
    None means load full range [EXT_LO..DATA_END_EPOCH] (matches original
    audit-folder builder behavior for byte-similar JSON output).
    """
    t_start = time.time()

    print("\nLoading Chainlink updates...", flush=True)
    cl_times: list[int] = []
    cl_prices: list[float] = []
    with open(CHAINLINK_PATH, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            cl_times.append(int(rec["updated_at"]))
            cl_prices.append(float(rec["current_dollars"]))
    cl_times_sorted = sorted(zip(cl_times, cl_prices), key=lambda x: x[0])
    cl_times = [t for t, _ in cl_times_sorted]
    cl_prices = [p for _, p in cl_times_sorted]
    print(f"  {len(cl_times)} chainlink updates")

    drift_by_epoch = build_drift_lookup(ep_min=ep_min, ep_max=ep_max)

    print("\nLoading rounds + klines...", flush=True)
    all_rounds = _load_all_rounds(use_extended_data=True)
    all_rounds = [r for r in all_rounds
                  if EXT_LO <= int(r.epoch) <= DATA_END_EPOCH and r.lock_at is not None]
    all_rounds.sort(key=lambda r: int(r.epoch))
    print(f"  {len(all_rounds)} rounds in range")

    cc = load_contract_constants()
    initial_bankroll = 50.0
    treasury_fee = float(cc.treasury_fee_fraction)
    min_bet = float(cc.min_bet_amount_bnb)

    sample_spec = FoldSpec(name="probe", kline_cutoff_seconds=CUTOFF_SECONDS,
                           epoch_start=EXT_LO, epoch_end=EXT_LO + 1000,
                           strategy_overrides={}, plot=False)
    sample_cfg = _resolve_strategy_config(sample_spec)
    earliest_offset = 301
    latest_offset = 3

    if ep_min is not None and ep_max is not None:
        print(f"  Loading klines (filtered [{ep_min}..{ep_max}])...", flush=True)
        btc_unified = _load_klines_unified_filtered(
            _BTC_KLINES_PATH, _EXT_BTC_KLINES_PATH,
            earliest_offset=earliest_offset, latest_offset=latest_offset,
            ep_min=ep_min, ep_max=ep_max)
        print(f"  BTC: {len(btc_unified)} epochs")
        eth_unified = _load_klines_unified_filtered(
            _ETH_KLINES_PATH, _EXT_ETH_KLINES_PATH,
            earliest_offset=earliest_offset, latest_offset=latest_offset,
            ep_min=ep_min, ep_max=ep_max)
        print(f"  ETH: {len(eth_unified)} epochs")
        sol_unified = _load_klines_unified_filtered(
            _SOL_KLINES_PATH, _EXT_SOL_KLINES_PATH,
            earliest_offset=earliest_offset, latest_offset=latest_offset,
            ep_min=ep_min, ep_max=ep_max)
        print(f"  SOL: {len(sol_unified)} epochs")
        print("  Loading BNB klines (raw, filtered)...", flush=True)
        bnb_klines_full = _load_bnb_raw_filtered(ep_min=ep_min, ep_max=ep_max)
        print(f"  BNB klines for {len(bnb_klines_full)} epochs")
    else:
        btc_unified = _load_klines_unified(_BTC_KLINES_PATH, earliest_offset=earliest_offset,
                                             latest_offset=latest_offset, extended_path=_EXT_BTC_KLINES_PATH)
        eth_unified = _load_klines_unified(_ETH_KLINES_PATH, earliest_offset=earliest_offset,
                                             latest_offset=latest_offset, extended_path=_EXT_ETH_KLINES_PATH)
        sol_unified = _load_klines_unified(_SOL_KLINES_PATH, earliest_offset=earliest_offset,
                                             latest_offset=latest_offset, extended_path=_EXT_SOL_KLINES_PATH)

        print("  Loading BNB klines (raw)...", flush=True)
        bnb_klines_full: dict[int, list[list]] = {}
        for p in (BNB_KLINES_CANONICAL, BNB_KLINES_EXTENDED):
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    rec = json.loads(s)
                    ep = int(rec["epoch"])
                    klines = rec.get("klines_1s") or []
                    if klines and ep not in bnb_klines_full:
                        bnb_klines_full[ep] = klines
        print(f"  BNB klines for {len(bnb_klines_full)} epochs")

    print(f"  load_elapsed={time.time()-t_start:.1f}s", flush=True)

    print("\nCanonical pass (for trailing PnL + per-bet target)...", flush=True)
    sim_rounds_full = list(all_rounds)
    canonical_history = canonical_pass_with_epoch(
        sim_rounds=sim_rounds_full,
        btc_klines_full=btc_unified, eth_klines_full=eth_unified,
        sol_klines_full=sol_unified, strategy_cfg=sample_cfg,
        kline_cutoff_seconds=CUTOFF_SECONDS,
        pool_cutoff_seconds=POOL_CUTOFF_SECONDS,
        earliest_offset=earliest_offset,
        initial_bankroll_bnb=initial_bankroll,
        treasury_fee_fraction=treasury_fee, min_bet_amount_bnb=min_bet,
    )
    print(f"  canonical bets: {len(canonical_history)}")

    trailing = TrailingPnLMulti([(la, p) for la, p, _, _ in canonical_history])
    pnl_by_epoch = {ep: p for _, p, ep, _ in canonical_history}

    print("\nBuilding 5-min spaced BTC close series...", flush=True)
    btc_close_by_epoch: dict[int, float] = {}
    for ep, kl in btc_unified.items():
        if kl:
            try:
                btc_close_by_epoch[ep] = float(kl[-1][4])
            except (IndexError, ValueError, TypeError):
                pass
    epochs_sorted = sorted(btc_close_by_epoch.keys())
    btc_close_arr = np.array([btc_close_by_epoch[e] for e in epochs_sorted])
    print(f"  {len(epochs_sorted)} epochs with BTC close")

    btc_log_returns_5m = np.full(len(btc_close_arr), np.nan)
    for i in range(1, len(btc_close_arr)):
        if btc_close_arr[i] > 0 and btc_close_arr[i-1] > 0:
            btc_log_returns_5m[i] = math.log(btc_close_arr[i] / btc_close_arr[i-1])

    epoch_index_map = {e: i for i, e in enumerate(epochs_sorted)}

    print("\nComputing streak features...", flush=True)
    streak_by_epoch: dict[int, int] = {}
    cur_streak = 0
    cur_dir: str | None = None
    last_pos: str | None = None
    for r in all_rounds:
        ep = int(r.epoch)
        if last_pos is None:
            streak_by_epoch[ep] = 0
        else:
            streak_by_epoch[ep] = cur_streak
        if r.position is None:
            cur_streak = 0
            cur_dir = None
            last_pos = None
            continue
        if cur_dir == r.position:
            cur_streak += 1
        else:
            cur_dir = r.position
            cur_streak = 1
        last_pos = r.position

    print("\nBuilding feature matrix...", flush=True)
    n = len(all_rounds)
    F = np.full((n, len(FEATURE_NAMES)), np.nan)
    target_outcome = np.full(n, -1, dtype=np.int8)
    target_pnl = np.full(n, np.nan)
    epoch_arr = np.array([int(r.epoch) for r in all_rounds], dtype=np.int64)

    # Extras for iteration-1 (close prices, lock prices, canonical bet flags/sides)
    close_price_arr = np.full(n, np.nan)
    lock_price_arr = np.full(n, np.nan)
    canonical_bet_flag = np.zeros(n, dtype=bool)
    canonical_side_arr = np.zeros(n, dtype=object)
    canonical_pnl_arr = np.full(n, np.nan)
    bet_set = {ep for _, _, ep, _ in canonical_history}
    side_by_epoch = {ep: side for _, _, ep, side in canonical_history}

    pool_cutoff_secs = POOL_CUTOFF_SECONDS

    max_lb = max(LOOKBACKS)
    for i, r in enumerate(all_rounds):
        ep = int(r.epoch)
        lock_at = int(r.lock_at)

        btc_kl = btc_unified.get(ep)
        eth_kl = eth_unified.get(ep)
        sol_kl = sol_unified.get(ep)
        bnb_kl = bnb_klines_full.get(ep)

        def _decision_closes(kl, _earliest_offset=earliest_offset):
            if not kl or len(kl) < (max_lb + 1):
                return None
            sliced = _slice_per_entry(kl, kline_cutoff_seconds=CUTOFF_SECONDS,
                                         max_lookback=max_lb,
                                         earliest_offset=_earliest_offset)
            if len(sliced) != max_lb + 1:
                return None
            return [float(c[4]) for c in sliced]

        def _decision_closes_bnb(kl):
            if not kl or len(kl) < (max_lb + 1):
                return None
            local_eo = len(kl) + 1
            sliced = _slice_per_entry(kl, kline_cutoff_seconds=CUTOFF_SECONDS,
                                         max_lookback=max_lb,
                                         earliest_offset=local_eo)
            if len(sliced) != max_lb + 1:
                return None
            return [float(c[4]) for c in sliced]

        btc_closes = _decision_closes(btc_kl)
        eth_closes = _decision_closes(eth_kl)
        sol_closes = _decision_closes(sol_kl)
        bnb_closes = _decision_closes_bnb(bnb_kl)

        def mtf_strength(closes):
            if closes is None:
                return None
            rets = [get_return(closes, lb) for lb in LOOKBACKS]
            if any(rt is None for rt in rets):
                return None
            return sum(abs(rt) for rt in rets)

        F[i, 0] = mtf_strength(btc_closes) if btc_closes else np.nan
        F[i, 1] = mtf_strength(eth_closes) if eth_closes else np.nan
        F[i, 2] = mtf_strength(sol_closes) if sol_closes else np.nan
        F[i, 3] = mtf_strength(bnb_closes) if bnb_closes else np.nan

        votes = []
        for closes in (btc_closes, eth_closes, sol_closes):
            if closes is None:
                continue
            rt = get_return(closes, 15)
            if rt is None:
                continue
            votes.append(1 if rt > 0 else 0)
        if len(votes) == 3:
            F[i, 4] = float(sum(votes))

        cutoff_ts = lock_at - pool_cutoff_secs
        bull_wei = 0
        bear_wei = 0
        for bet in r.bets:
            if int(bet.created_at) >= cutoff_ts:
                continue
            if bet.position == "Bull":
                bull_wei += int(bet.amount_wei)
            else:
                bear_wei += int(bet.amount_wei)
        bull_bnb = bull_wei / BNB_WEI
        bear_bnb = bear_wei / BNB_WEI
        total = bull_bnb + bear_bnb
        F[i, 5] = total
        F[i, 6] = bull_bnb / total if total > 0 else np.nan
        leading = max(bull_bnb, bear_bnb)
        F[i, 7] = (total / leading) * (1.0 - treasury_fee) if leading > 0 else np.nan

        F[i, 8] = realized_vol_1s(btc_kl, drop_last=0) if btc_kl else np.nan
        F[i, 15] = realized_vol_1s(bnb_kl, drop_last=1) if bnb_kl else np.nan

        idx = epoch_index_map.get(ep)
        if idx is not None and idx >= 1:
            window_3 = btc_log_returns_5m[max(0, idx-3):idx]
            window_3 = window_3[~np.isnan(window_3)]
            if len(window_3) >= 2:
                F[i, 9] = float(np.std(window_3, ddof=1))
            window_12 = btc_log_returns_5m[max(0, idx-12):idx]
            window_12 = window_12[~np.isnan(window_12)]
            if len(window_12) >= 2:
                F[i, 10] = float(np.std(window_12, ddof=1))
            window_48 = btc_log_returns_5m[max(0, idx-48):idx]
            window_48 = window_48[~np.isnan(window_48)]
            if len(window_48) >= 2:
                F[i, 11] = float(np.std(window_48, ddof=1))

            if idx >= 12 and btc_close_arr[idx-12] > 0 and btc_close_arr[idx-1] > 0:
                F[i, 12] = math.log(btc_close_arr[idx-1] / btc_close_arr[idx-12])
            if idx >= 48 and btc_close_arr[idx-48] > 0 and btc_close_arr[idx-1] > 0:
                F[i, 13] = math.log(btc_close_arr[idx-1] / btc_close_arr[idx-48])
            if idx >= 288 and btc_close_arr[idx-288] > 0 and btc_close_arr[idx-1] > 0:
                F[i, 14] = math.log(btc_close_arr[idx-1] / btc_close_arr[idx-288])

            bnb_closes_5m = []
            for back in range(12, 0, -1):
                target_ep = ep - back
                kl = bnb_klines_full.get(target_ep)
                if kl:
                    try:
                        bnb_closes_5m.append(float(kl[-1][4]))
                    except (IndexError, ValueError, TypeError):
                        bnb_closes_5m.append(np.nan)
                else:
                    bnb_closes_5m.append(np.nan)
            bnb_closes_arr = np.array(bnb_closes_5m)
            bnb_rets = []
            for j in range(1, len(bnb_closes_arr)):
                if bnb_closes_arr[j] > 0 and bnb_closes_arr[j-1] > 0:
                    bnb_rets.append(math.log(bnb_closes_arr[j] / bnb_closes_arr[j-1]))
            bnb_rets_arr = np.array(bnb_rets)
            if len(bnb_rets_arr) >= 3:
                F[i, 16] = float(np.std(bnb_rets_arr[-3:], ddof=1)) if len(bnb_rets_arr[-3:]) >= 2 else np.nan
            if len(bnb_rets_arr) >= 2:
                F[i, 17] = float(np.std(bnb_rets_arr, ddof=1))

        ts = epoch_ts(ep)
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        F[i, 18] = float(dt.hour)
        F[i, 19] = float(dt.weekday())

        F[i, 20] = trailing.trailing(lock_at, 7 * 86400)
        F[i, 21] = trailing.trailing(lock_at, 14 * 86400)
        F[i, 22] = trailing.trailing(lock_at, 30 * 86400)

        F[i, 23] = float(streak_by_epoch.get(ep, 0))

        d = drift_by_epoch.get(ep)
        F[i, 24] = float(d) if d is not None else np.nan

        if cl_times and cl_times[0] <= lock_at:
            lo = 0
            hi = len(cl_times) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if cl_times[mid] <= lock_at:
                    lo = mid
                else:
                    hi = mid - 1
            F[i, 25] = float(lock_at - cl_times[lo])

        if r.position == "Bull":
            target_outcome[i] = 1
        elif r.position == "Bear":
            target_outcome[i] = 0
        if ep in pnl_by_epoch:
            target_pnl[i] = pnl_by_epoch[ep]

        # Extras (lock_price and close_price are already in USD; no scaling needed)
        try:
            if r.close_price is not None:
                close_price_arr[i] = float(r.close_price)
        except Exception:
            pass
        try:
            if r.lock_price is not None:
                lock_price_arr[i] = float(r.lock_price)
        except Exception:
            pass
        if ep in bet_set:
            canonical_bet_flag[i] = True
            canonical_side_arr[i] = side_by_epoch.get(ep, "")
            canonical_pnl_arr[i] = pnl_by_epoch.get(ep, np.nan)

    extras = {
        "close_price": close_price_arr,
        "lock_price": lock_price_arr,
        "bnb_vol_5m": F[:, 15].copy(),
        "canonical_bet": canonical_bet_flag,
        "canonical_side": canonical_side_arr,
        "canonical_pnl": canonical_pnl_arr,
        "elapsed_seconds": time.time() - t_start,
    }
    if return_extras:
        return (FEATURE_NAMES, F, target_outcome, target_pnl, epoch_arr,
                all_rounds, canonical_history, extras)
    return (FEATURE_NAMES, F, target_outcome, target_pnl, epoch_arr,
            all_rounds, canonical_history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path. Default: var/extended/phase0_feature_results.json")
    parser.add_argument("--ep-min", type=int, default=None,
                        help="Filter klines to ep >= this (memory-efficient mode).")
    parser.add_argument("--ep-max", type=int, default=None,
                        help="Filter klines to ep <= this (memory-efficient mode).")
    args = parser.parse_args()

    print("=" * 110)
    print("PHASE 0: FEATURE ANALYSIS (recovered)")
    print("=" * 110)
    t_start = time.time()

    (feature_names, F, target_outcome, target_pnl, epoch_arr,
     all_rounds, canonical_history) = build_feature_matrix(
        ep_min=args.ep_min, ep_max=args.ep_max)

    n = len(all_rounds)
    print(f"  feature matrix: {F.shape}, non-nan ratio per col:")
    for j, name in enumerate(feature_names):
        nn = float(np.sum(~np.isnan(F[:, j]))) / n * 100
        print(f"    [{j:2d}] {name:<28} {nn:6.2f}%")

    print("\nComputing correlations...", flush=True)
    valid_outcome = target_outcome != -1
    pre_dec_mask = epoch_arr < DEC_CUTOFF_EPOCH
    post_dec_mask = epoch_arr >= DEC_CUTOFF_EPOCH

    target_outcome_f = target_outcome.astype(np.float64)
    target_outcome_f[~valid_outcome] = np.nan

    feature_results = []
    for j, name in enumerate(feature_names):
        x = F[:, j]
        m = ~np.isnan(x) & valid_outcome
        if m.sum() < 50:
            continue
        x_v = x[m]
        y_v = target_outcome_f[m]

        try:
            r_p, p_p = pearsonr(x_v, y_v)
        except (ValueError, RuntimeWarning):
            r_p, p_p = (0.0, 1.0)
        try:
            r_s, p_s = spearmanr(x_v, y_v)
        except (ValueError, RuntimeWarning):
            r_s, p_s = (0.0, 1.0)

        m_pnl = ~np.isnan(x) & ~np.isnan(target_pnl)
        if m_pnl.sum() >= 30:
            x_pnl_v = x[m_pnl]
            y_pnl_v = target_pnl[m_pnl]
            try:
                r_p_pnl, p_p_pnl = pearsonr(x_pnl_v, y_pnl_v)
            except (ValueError, RuntimeWarning):
                r_p_pnl, p_p_pnl = (0.0, 1.0)
            try:
                r_s_pnl, p_s_pnl = spearmanr(x_pnl_v, y_pnl_v)
            except (ValueError, RuntimeWarning):
                r_s_pnl, p_s_pnl = (0.0, 1.0)
            n_pnl_obs = int(m_pnl.sum())
        else:
            r_p_pnl, p_p_pnl, r_s_pnl, p_s_pnl = (0.0, 1.0, 0.0, 1.0)
            n_pnl_obs = int(m_pnl.sum())

        def reg_corr(mask):
            mm = m & mask
            if mm.sum() < 30:
                return (0.0, 1.0, 0)
            xv = x[mm]
            yv = target_outcome_f[mm]
            try:
                rr, pp = pearsonr(xv, yv)
            except (ValueError, RuntimeWarning):
                rr, pp = (0.0, 1.0)
            return (float(rr), float(pp), int(mm.sum()))

        r_pre, p_pre, n_pre = reg_corr(pre_dec_mask)
        r_post, p_post, n_post = reg_corr(post_dec_mask)

        feature_results.append({
            "name": name,
            "is_categorical": name in CATEGORICAL,
            "n_obs": int(m.sum()),
            "pearson_r_outcome": float(r_p),
            "pearson_p_outcome": float(p_p),
            "spearman_r_outcome": float(r_s),
            "spearman_p_outcome": float(p_s),
            "n_obs_pnl": n_pnl_obs,
            "pearson_r_pnl": float(r_p_pnl),
            "pearson_p_pnl": float(p_p_pnl),
            "spearman_r_pnl": float(r_s_pnl),
            "spearman_p_pnl": float(p_s_pnl),
            "pre_dec_pearson_r": r_pre,
            "pre_dec_pearson_p": p_pre,
            "pre_dec_n": n_pre,
            "post_dec_pearson_r": r_post,
            "post_dec_pearson_p": p_post,
            "post_dec_n": n_post,
        })

    feature_results.sort(key=lambda d: -abs(d["pearson_r_outcome"]))

    print("\nDecile / category analysis for top 5 features...", flush=True)
    decile_results: dict[str, dict] = {}
    for entry in feature_results[:5]:
        name = entry["name"]
        j = feature_names.index(name)
        x = F[:, j]
        m = ~np.isnan(x) & valid_outcome
        if m.sum() < 50:
            continue
        xv = x[m]
        yv = target_outcome_f[m]

        if entry["is_categorical"]:
            cats = sorted(set(int(v) for v in xv))
            cat_stats = []
            for c in cats:
                mm = xv == c
                bull_rate = float(yv[mm].mean()) if mm.sum() > 0 else 0.0
                cat_stats.append({"category": int(c), "n": int(mm.sum()),
                                   "bull_rate": bull_rate})
            decile_results[name] = {"type": "categorical", "stats": cat_stats}
        else:
            quantiles = np.percentile(xv, np.linspace(0, 100, 11))
            decile_stats = []
            for d in range(10):
                lo = quantiles[d]
                hi = quantiles[d+1]
                if d == 9:
                    mm = (xv >= lo) & (xv <= hi)
                else:
                    mm = (xv >= lo) & (xv < hi)
                bull_rate = float(yv[mm].mean()) if mm.sum() > 0 else 0.0
                decile_stats.append({"decile": d+1, "lo": float(lo), "hi": float(hi),
                                      "n": int(mm.sum()), "bull_rate": bull_rate})
            d1 = decile_stats[0]["bull_rate"]
            d10 = decile_stats[9]["bull_rate"]
            decile_results[name] = {"type": "continuous", "deciles": decile_stats,
                                     "d1_d10_delta": d10 - d1}

    print("\nComputing feature correlation matrix...", flush=True)
    n_feat = len(feature_names)
    corr_mat = np.full((n_feat, n_feat), np.nan)
    for a in range(n_feat):
        xa = F[:, a]
        for b in range(a, n_feat):
            xb = F[:, b]
            mm = ~np.isnan(xa) & ~np.isnan(xb)
            if mm.sum() < 50:
                continue
            try:
                rr, _ = pearsonr(xa[mm], xb[mm])
            except (ValueError, RuntimeWarning):
                rr = 0.0
            corr_mat[a, b] = corr_mat[b, a] = float(rr)

    redundancy_pairs = []
    for a in range(n_feat):
        for b in range(a+1, n_feat):
            if not np.isnan(corr_mat[a, b]) and abs(corr_mat[a, b]) > 0.7:
                redundancy_pairs.append((feature_names[a], feature_names[b],
                                          float(corr_mat[a, b])))
    redundancy_pairs.sort(key=lambda x: -abs(x[2]))

    print("\n" + "=" * 110)
    print("RANKED FEATURE TABLE (sorted by |Pearson r| with outcome)")
    print("=" * 110)
    print(f"{'#':>3}  {'feature':<28}  {'n_obs':>7}  "
          f"{'r_out':>7}  {'p_out':>9}  {'sp_r_out':>9}  "
          f"{'r_pnl':>7}  {'p_pnl':>9}  "
          f"{'r_pre':>7}  {'r_post':>7}")
    print("-" * 110)
    for k, e in enumerate(feature_results):
        flag = "*" if e["pearson_p_outcome"] < 0.05 else " "
        print(f"{k+1:>3}{flag} {e['name']:<28}  {e['n_obs']:>7}  "
              f"{e['pearson_r_outcome']:>+7.4f}  {e['pearson_p_outcome']:>9.2e}  "
              f"{e['spearman_r_outcome']:>+9.4f}  "
              f"{e['pearson_r_pnl']:>+7.4f}  {e['pearson_p_pnl']:>9.2e}  "
              f"{e['pre_dec_pearson_r']:>+7.4f}  {e['post_dec_pearson_r']:>+7.4f}")

    print("\n" + "=" * 110)
    print("TOP 5 FEATURE DETAILS")
    print("=" * 110)
    for entry in feature_results[:5]:
        name = entry["name"]
        print(f"\n{name}")
        print("-" * 60)
        print(f"  Pearson r (outcome):  {entry['pearson_r_outcome']:+.4f}  p={entry['pearson_p_outcome']:.4e}  n={entry['n_obs']}")
        print(f"  Spearman r (outcome): {entry['spearman_r_outcome']:+.4f}  p={entry['spearman_p_outcome']:.4e}")
        print(f"  Pearson r (per-bet PnL): {entry['pearson_r_pnl']:+.4f}  p={entry['pearson_p_pnl']:.4e}  n={entry['n_obs_pnl']}")
        print(f"  Pre-Dec r:  {entry['pre_dec_pearson_r']:+.4f}  p={entry['pre_dec_pearson_p']:.4e}  n={entry['pre_dec_n']}")
        print(f"  Post-Dec r: {entry['post_dec_pearson_r']:+.4f}  p={entry['post_dec_pearson_p']:.4e}  n={entry['post_dec_n']}")

        if name in decile_results:
            dr = decile_results[name]
            if dr["type"] == "categorical":
                print(f"  Per-category bull rate:")
                for s in dr["stats"]:
                    print(f"    cat={s['category']:>3}  n={s['n']:>6}  bull_rate={s['bull_rate']*100:6.2f}%")
            else:
                print(f"  Decile bull rates (D1=lowest, D10=highest):")
                for s in dr["deciles"]:
                    print(f"    D{s['decile']:>2}  [{s['lo']:>+10.4g}, {s['hi']:>+10.4g}]  "
                          f"n={s['n']:>5}  bull_rate={s['bull_rate']*100:6.2f}%")
                print(f"  D10 - D1 bull-rate delta: {dr['d1_d10_delta']*100:+.2f}%")

    print("\n" + "=" * 110)
    print("REDUNDANCY CLUSTERS (|r| > 0.7)")
    print("=" * 110)
    if not redundancy_pairs:
        print("  (none)")
    else:
        for (a, b, r) in redundancy_pairs[:30]:
            print(f"  {r:+.4f}   {a}   <->   {b}")

    print("\n" + "=" * 110)
    print("VERDICT")
    print("=" * 110)
    sig_features = [e for e in feature_results if e["pearson_p_outcome"] < 0.05]
    strong_features = [e for e in feature_results if abs(e["pearson_r_outcome"]) >= 0.02]
    print(f"\n  Total features evaluated:                       {len(feature_results)}")
    print(f"  Features with p < 0.05 vs outcome:              {len(sig_features)}")
    print(f"  Features with |r| >= 0.02 vs outcome:           {len(strong_features)}")
    print(f"  Features in redundancy clusters:                {len(set(a for a,_,_ in redundancy_pairs) | set(b for _,b,_ in redundancy_pairs))}")

    used_in_cluster: set[str] = set()
    cluster_groups: list[list[str]] = []
    for a, b, _ in redundancy_pairs:
        placed = False
        for grp in cluster_groups:
            if a in grp or b in grp:
                if a not in grp:
                    grp.append(a)
                    used_in_cluster.add(a)
                if b not in grp:
                    grp.append(b)
                    used_in_cluster.add(b)
                placed = True
                break
        if not placed:
            cluster_groups.append([a, b])
            used_in_cluster.add(a)
            used_in_cluster.add(b)

    rank_by_name = {e["name"]: idx for idx, e in enumerate(feature_results)}
    lean_set: list[str] = []
    seen_clusters: set[int] = set()
    for entry in feature_results:
        if entry["pearson_p_outcome"] >= 0.10 and abs(entry["pearson_r_outcome"]) < 0.02:
            continue
        nm = entry["name"]
        cluster_idx = None
        for ci, grp in enumerate(cluster_groups):
            if nm in grp:
                cluster_idx = ci
                break
        if cluster_idx is None:
            lean_set.append(nm)
        else:
            if cluster_idx not in seen_clusters:
                best = min(cluster_groups[cluster_idx], key=lambda n: rank_by_name.get(n, 9999))
                lean_set.append(best)
                seen_clusters.add(cluster_idx)

    print(f"\n  Suggested LEAN candidate set ({len(lean_set)} features):")
    for nm in lean_set:
        e = next(e for e in feature_results if e["name"] == nm)
        marker = "*" if e["pearson_p_outcome"] < 0.05 else " "
        print(f"    {marker} {nm:<28}  r={e['pearson_r_outcome']:+.4f}  p={e['pearson_p_outcome']:.2e}")

    out = {
        "feature_results": feature_results,
        "decile_results": decile_results,
        "redundancy_pairs": redundancy_pairs,
        "lean_candidate_set": lean_set,
        "n_rounds": int(len(all_rounds)),
        "n_canonical_bets": int(len(canonical_history)),
        "feature_names": feature_names,
    }
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = REPO / "var" / "extended" / "phase0_feature_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nResults JSON: {out_path}")
    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
