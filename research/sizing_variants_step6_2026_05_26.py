"""Sizing-variant analysis — Step 6 of regime characterization.

Compare 6 bet-sizing variants on canonical (3, 7, 15) cs=2 gate signal across
the full 422298..484000 epoch range. Gate decisions (BET vs SKIP, side) are
held constant; only the bet-size formula differs.

Variants:
  1. canonical_fixed_frac — production sizer (`_compute_bet_size`): signal-strength
     + payout boost + bankroll cap + min/max clamps.
  2. half_kelly — f* = 0.5 * (b·p - q)/b where p=rolling WR (last 20), b=pool*(1-fee)/our_side - 1.
     Skip when f* <= 0. Clamp [min_bet, max_cap, max_bet_fraction_of_bankroll].
  3. full_kelly — same as half_kelly with 1.0 multiplier.
  4. vol_scaled — canonical_bet * (ref_vol / current_btc_vol_30s). Lower bet in high vol.
     ref_vol = median of BTC 30s realized vols across all bet epochs.
  5. drawdown_conditioned — canonical_bet scaled by drawdown-from-peak:
     dd < 5% -> 1.0x, 5-10% -> 0.5x, 10-20% -> 0.25x, >=20% -> skip.
  6. signal_strength_weighted — canonical_bet * (0.5 + 1.5 * normalized_strength)
     where normalized_strength = min(1, effective_strength / saturation_threshold=0.005).
     Range: half canonical at zero strength to 2.0x canonical at saturated strength.

Each variant simulates per-bet bankroll trajectory from initial_bankroll=5.0 BNB.
Stats reported: total PnL, max drawdown, Sharpe-like ratio (mean/std of per-bet
profit_bnb), n_bets, final bankroll.

Frozen invariants:
  - kline_cutoff_seconds=2 (HARD).
  - Canonical (3, 7, 15) lookbacks.
  - Real impact-aware settlement via `settlement.settle_bet_against_closed_round`.

Output:
  - var/strategy_review/sizing_variants_step6_data.json
  - var/strategy_review/2026_05_26_sizing_variants_step6.md (written separately)

Wall-clock: ~3-4 min (~3 min for context capture pass, <1s per variant).
"""
from __future__ import annotations

import collections
import json
import math
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
import research.in_process_runner as ipr  # noqa: E402
ipr._EXT_CLOSED_ROUNDS_PATH = EXT_DIR / "closed_rounds.jsonl"
ipr._EXT_BTC_KLINES_PATH = EXT_DIR / "btc_spot_prices.jsonl"
ipr._EXT_ETH_KLINES_PATH = EXT_DIR / "eth_spot_prices.jsonl"
ipr._EXT_SOL_KLINES_PATH = EXT_DIR / "sol_spot_prices.jsonl"

from pancakebot.config import _DEFAULT_STRATEGY  # noqa: E402
from pancakebot.constants import MAX_GAS_COST_BET_BNB  # noqa: E402
from pancakebot.settlement import settle_bet_against_closed_round  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import (  # noqa: E402
    MomentumOnlyPipeline,
    _compute_bet_size,
    _pools_from_bets,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPOCH_MIN = 422298
EPOCH_MAX = 484000

CANONICAL_LOOKBACKS = (3, 7, 15)
CANONICAL_CUTOFF = 2
POOL_CUTOFF = 6

INITIAL_BANKROLL = 5.0
TREASURY_FEE = 0.03
MIN_BET = 0.001
VOL_WINDOW_SECONDS = 30
ROLLING_WR_WINDOW = 20

STRATEGY = _DEFAULT_STRATEGY

# Cohort tagging
def cohort_of(epoch: int) -> str:
    if 422298 <= epoch <= 437561: return "extension"
    if 437562 <= epoch <= 474086: return "cv5"
    if 474880 <= epoch <= 475311: return "holdout"
    if 475312 <= epoch <= 479952: return "ext_v2"
    if 479953 <= epoch <= 483191: return "fresh_oos"
    return "post_fresh"


@dataclass
class BetContext:
    """Per-bet context captured during the canonical gate pass."""
    epoch: int
    side: str  # "Bull" or "Bear"
    pool_bull: float
    pool_bear: float
    pool_total: float
    our_side: float
    signal_strength: float
    is_regime2: bool
    btc_vol_30s: float  # std of 1s log-returns over last VOL_WINDOW_SECONDS
    cohort: str
    round_obj: Any  # for settle_bet_against_closed_round


# ---------------------------------------------------------------------------
# Phase 1: capture per-bet context
# ---------------------------------------------------------------------------

def compute_btc_vol_from_klines(kl: list[list], window_s: int = VOL_WINDOW_SECONDS) -> float:
    """Std of log returns of last N=window_s candles' close prices.

    Klines are [ts_ms, o, h, l, c, vol] arrays. Use close prices.
    Slice already trimmed to the per-entry window (max_lookback+1 candles)
    when called from the pipeline's _slice_per_entry path, so use the
    LAST window_s candles available (clamped if shorter).
    """
    if not kl or len(kl) < 3:
        return 0.0
    # Use last min(window_s, len(kl)) candles
    take = kl[-window_s:] if len(kl) >= window_s else kl
    closes = [float(k[4]) for k in take if k[4] is not None and float(k[4]) > 0]
    if len(closes) < 3:
        return 0.0
    log_returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            continue
        log_returns.append(math.log(closes[i] / closes[i - 1]))
    if len(log_returns) < 2:
        return 0.0
    return statistics.stdev(log_returns)


def capture_contexts() -> tuple[list[BetContext], dict[str, Any]]:
    print("--- loading rounds (canonical + extended) ---")
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    print(f"  loaded {len(all_rounds)} rounds; range "
          f"[{all_rounds[0].epoch}..{all_rounds[-1].epoch}]")

    # Bump earliest_offset to cover BTC vol window (30s) beyond canonical's
    # 15-lookback need.
    canonical_max_lb = max(CANONICAL_LOOKBACKS)
    needed_for_vol = CANONICAL_CUTOFF + VOL_WINDOW_SECONDS + 1
    needed_for_gate = CANONICAL_CUTOFF + canonical_max_lb + 1
    earliest_offset = max(needed_for_vol, needed_for_gate)
    latest_offset = CANONICAL_CUTOFF + 1
    print(f"  earliest_offset={earliest_offset} (vol_window={VOL_WINDOW_SECONDS}s) "
          f"latest_offset={latest_offset}")

    print("--- loading klines unified ---")
    btc = ipr._load_klines_unified(
        ipr._BTC_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_BTC_KLINES_PATH,
    )
    eth = ipr._load_klines_unified(
        ipr._ETH_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_ETH_KLINES_PATH,
    )
    sol = ipr._load_klines_unified(
        ipr._SOL_KLINES_PATH,
        earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=ipr._EXT_SOL_KLINES_PATH,
    )
    print(f"  BTC={len(btc)} ETH={len(eth)} SOL={len(sol)}")

    # Build pipeline with NO bankroll tracker (risk gates never fire here;
    # variant simulators handle their own bankroll/risk logic).
    gate_cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        mtf_lookbacks=CANONICAL_LOOKBACKS,
        mtf_min_return_threshold=STRATEGY.gate.mtf_min_return_threshold,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_cfg,
        strategy_config=STRATEGY,
        gate=None,
        kline_cutoff_seconds=CANONICAL_CUTOFF,
        pool_cutoff_seconds=POOL_CUTOFF,
        min_bet_amount_bnb=MIN_BET,
        treasury_fee_fraction=TREASURY_FEE,
        bankroll_tracker=None,  # disable risk gates
    )

    # Slice per-entry klines using canonical max_lookback for the pipeline,
    # but ALSO keep a longer slice for vol computation.
    print("--- slicing klines for pipeline (max_lookback=15) ---")
    btc_for_gate = {
        ep: ipr._slice_per_entry(
            kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
            max_lookback=canonical_max_lb, earliest_offset=earliest_offset,
        )
        for ep, kl in btc.items()
    }
    eth_for_gate = {
        ep: ipr._slice_per_entry(
            kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
            max_lookback=canonical_max_lb, earliest_offset=earliest_offset,
        )
        for ep, kl in eth.items()
    }
    sol_for_gate = {
        ep: ipr._slice_per_entry(
            kl, kline_cutoff_seconds=CANONICAL_CUTOFF,
            max_lookback=canonical_max_lb, earliest_offset=earliest_offset,
        )
        for ep, kl in sol.items()
    }
    # Vol slice: end_idx is the same as gate slice (= earliest_offset - cutoff_seconds);
    # but we want the LAST VOL_WINDOW_SECONDS candles ending at end_idx.
    print(f"--- slicing klines for vol (last {VOL_WINDOW_SECONDS}s) ---")
    end_idx_global = earliest_offset - CANONICAL_CUTOFF  # exclusive end (matches _slice_per_entry)
    start_idx_vol = end_idx_global - VOL_WINDOW_SECONDS
    btc_for_vol: dict[int, list[list]] = {}
    for ep, kl in btc.items():
        btc_for_vol[ep] = kl[start_idx_vol:end_idx_global]

    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_for_gate)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_for_gate)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_for_gate)
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    sim_rounds = [r for r in all_rounds
                  if EPOCH_MIN <= r.epoch <= EPOCH_MAX]
    print(f"--- iterating {len(sim_rounds)} sim rounds ---")
    t0 = time.time()

    contexts: list[BetContext] = []
    skip_counts: dict[str, int] = {}
    n_round = 0

    for round_t in sim_rounds:
        n_round += 1
        decision = pipeline.decide_open_round(round_t=round_t)
        if decision.action != "BET":
            sr = decision.skip_reason or "unknown"
            skip_counts[sr] = skip_counts.get(sr, 0) + 1
            pipeline.settle_closed_rounds(rounds=[round_t])
            continue

        # Recompute the bet's context for variant replay.
        lock_at = int(round_t.lock_at)
        cutoff_ts_ms = (lock_at - CANONICAL_CUTOFF) * 1000
        result = pipeline._evaluate_from_cache(
            epoch=int(round_t.epoch), cutoff_ts_ms=cutoff_ts_ms,
        )
        pool_cutoff_ts = lock_at - POOL_CUTOFF
        pool_bull, pool_bear = _pools_from_bets(round_t, pool_cutoff_ts)
        pool_total = pool_bull + pool_bear

        # Replicate pipeline's effective_strength + is_regime2 logic
        # (lines 330-365 of momentum_pipeline.py).
        signal_dir = decision.bet_side
        effective_strength = 0.0
        is_regime2 = False
        t2_w = STRATEGY.tier2_sizing.eth_sol_signal_weight

        if result.signal is not None and result.signal == signal_dir:
            # Primary path
            effective_strength = result.signal_strength
            if result.eth_confirmation_strength > 0:
                effective_strength += result.eth_confirmation_strength * t2_w
            if result.sol_confirmation_strength > 0:
                effective_strength += result.sol_confirmation_strength * t2_w
        else:
            # Regime-2
            is_regime2 = True
            effective_strength = (
                result.eth_signal_strength * t2_w
                + result.sol_signal_strength * t2_w
            )

        our_side = pool_bull if signal_dir == "Bull" else pool_bear
        btc_kl_vol = btc_for_vol.get(int(round_t.epoch), [])
        vol = compute_btc_vol_from_klines(btc_kl_vol)

        contexts.append(BetContext(
            epoch=int(round_t.epoch),
            side=str(signal_dir),
            pool_bull=float(pool_bull),
            pool_bear=float(pool_bear),
            pool_total=float(pool_total),
            our_side=float(our_side),
            signal_strength=float(effective_strength),
            is_regime2=bool(is_regime2),
            btc_vol_30s=float(vol),
            cohort=cohort_of(int(round_t.epoch)),
            round_obj=round_t,
        ))

        pipeline.settle_closed_rounds(rounds=[round_t])

    elapsed = time.time() - t0
    print(f"  pass complete in {elapsed:.1f}s -- captured {len(contexts)} BET contexts")

    meta = {
        "n_rounds_iterated": n_round,
        "n_bets": len(contexts),
        "skip_counts_top10": dict(sorted(skip_counts.items(), key=lambda x: -x[1])[:10]),
        "elapsed_capture_seconds": elapsed,
        "earliest_offset": earliest_offset,
    }
    return contexts, meta


# ---------------------------------------------------------------------------
# Phase 2: variant sizers
# ---------------------------------------------------------------------------

def variant_canonical(*, ctx: BetContext, bankroll: float, **_kw: Any) -> float:
    """Production canonical sizer: signal-strength + payout boost + bankroll cap."""
    bt_sz = STRATEGY.btc_primary.sizing
    t2 = STRATEGY.tier2_sizing
    br_cap_frac = STRATEGY.risk.max_bet_fraction_of_bankroll

    if ctx.is_regime2:
        es_sizing = STRATEGY.eth_sol_fallback.sizing
        return _compute_bet_size(
            signal_strength=ctx.signal_strength,
            pool_bnb=ctx.pool_total,
            our_side_bnb=ctx.our_side,
            base_frac=es_sizing.base_pool_fraction,
            cap_bnb=STRATEGY.risk.max_bet_bnb_eth_sol_fallback,
            pool_fraction_slope=bt_sz.pool_fraction_slope,
            max_pool_fraction=bt_sz.max_pool_fraction,
            treasury_fee_fraction=TREASURY_FEE,
            min_bet_threshold_bnb=t2.min_bet_threshold_bnb,
            current_bankroll=bankroll,
            max_bet_fraction_of_bankroll=br_cap_frac,
        )
    return _compute_bet_size(
        signal_strength=ctx.signal_strength,
        pool_bnb=ctx.pool_total,
        our_side_bnb=ctx.our_side,
        base_frac=bt_sz.base_pool_fraction,
        cap_bnb=STRATEGY.risk.max_bet_bnb_btc_primary,
        pool_fraction_slope=bt_sz.pool_fraction_slope,
        max_pool_fraction=bt_sz.max_pool_fraction,
        treasury_fee_fraction=TREASURY_FEE,
        min_bet_threshold_bnb=t2.min_bet_threshold_bnb,
        current_bankroll=bankroll,
        max_bet_fraction_of_bankroll=br_cap_frac,
    )


def _kelly_fraction(p: float, payout_mult: float) -> float:
    """f* = (b·p - q)/b where b = payout_mult - 1, q = 1 - p."""
    b = payout_mult - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    return (b * p - q) / b


def variant_half_kelly(*, ctx: BetContext, bankroll: float, rolling_wr: float,
                       canonical_bet: float, **_kw: Any) -> float:
    if ctx.our_side <= 0:
        return 0.0
    payout_mult = ctx.pool_total * (1.0 - TREASURY_FEE) / ctx.our_side
    f = _kelly_fraction(rolling_wr, payout_mult)
    if f <= 0:
        return 0.0  # skip negative-edge bets (Kelly says don't bet)
    bet = 0.5 * f * bankroll
    cap = (STRATEGY.risk.max_bet_bnb_eth_sol_fallback
           if ctx.is_regime2 else STRATEGY.risk.max_bet_bnb_btc_primary)
    bet = min(bet, cap)
    bet = min(bet, STRATEGY.risk.max_bet_fraction_of_bankroll * bankroll * 5.0)  # 5x looser cap for Kelly variants
    return bet  # caller applies min_bet floor + insufficient-bankroll skip


def variant_full_kelly(*, ctx: BetContext, bankroll: float, rolling_wr: float,
                       canonical_bet: float, **_kw: Any) -> float:
    if ctx.our_side <= 0:
        return 0.0
    payout_mult = ctx.pool_total * (1.0 - TREASURY_FEE) / ctx.our_side
    f = _kelly_fraction(rolling_wr, payout_mult)
    if f <= 0:
        return 0.0
    bet = 1.0 * f * bankroll
    cap = (STRATEGY.risk.max_bet_bnb_eth_sol_fallback
           if ctx.is_regime2 else STRATEGY.risk.max_bet_bnb_btc_primary)
    bet = min(bet, cap)
    bet = min(bet, STRATEGY.risk.max_bet_fraction_of_bankroll * bankroll * 5.0)
    return bet


def variant_vol_scaled(*, ctx: BetContext, bankroll: float, canonical_bet: float,
                       ref_vol: float, **_kw: Any) -> float:
    """canonical_bet * (ref_vol / current_vol), clamped to canonical's caps."""
    if ctx.btc_vol_30s <= 0 or ref_vol <= 0:
        return canonical_bet
    scale = ref_vol / ctx.btc_vol_30s
    scale = max(0.25, min(4.0, scale))  # bound rescaling between 0.25x and 4x
    bet = canonical_bet * scale
    cap = (STRATEGY.risk.max_bet_bnb_eth_sol_fallback
           if ctx.is_regime2 else STRATEGY.risk.max_bet_bnb_btc_primary)
    return min(bet, cap)


def variant_drawdown_conditioned(*, ctx: BetContext, bankroll: float,
                                  canonical_bet: float, drawdown: float,
                                  **_kw: Any) -> float:
    if drawdown < 0.05:
        return canonical_bet
    elif drawdown < 0.10:
        return canonical_bet * 0.5
    elif drawdown < 0.20:
        return canonical_bet * 0.25
    else:
        return 0.0  # skip in deep drawdown


def variant_signal_strength_weighted(*, ctx: BetContext, bankroll: float,
                                      canonical_bet: float, **_kw: Any) -> float:
    """canonical_bet * (0.5 + 1.5 * normalized_strength).

    normalized_strength = min(1, effective_strength / 0.005)
    Range: 0.5x canonical at zero strength to 2.0x canonical at saturated.
    """
    saturation = 0.005
    norm = min(1.0, ctx.signal_strength / saturation) if saturation > 0 else 0.0
    norm = max(0.0, norm)
    scale = 0.5 + 1.5 * norm
    bet = canonical_bet * scale
    cap = (STRATEGY.risk.max_bet_bnb_eth_sol_fallback
           if ctx.is_regime2 else STRATEGY.risk.max_bet_bnb_btc_primary)
    return min(bet, cap)


# ---------------------------------------------------------------------------
# Phase 3: simulate one variant
# ---------------------------------------------------------------------------

def simulate(contexts: list[BetContext], variant_fn: Any, variant_name: str,
             ref_vol: float | None = None) -> dict[str, Any]:
    bankroll = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    max_dd_frac = 0.0
    per_bet_profits: list[float] = []
    per_bet_bet_sizes: list[float] = []
    n_taken = 0
    n_wins = 0
    n_skipped_below_min = 0
    n_skipped_insufficient = 0
    n_skipped_variant_zero = 0
    per_cohort: dict[str, dict[str, float]] = {}
    rolling_wins: collections.deque = collections.deque(maxlen=ROLLING_WR_WINDOW)

    for ctx in contexts:
        # Compute canonical bet first (needed by some variants)
        canonical_bet = variant_canonical(ctx=ctx, bankroll=bankroll)
        rolling_wr = (sum(rolling_wins) / len(rolling_wins)) if rolling_wins else 0.6
        drawdown = (peak - bankroll) / peak if peak > 0 else 0.0

        if variant_name == "canonical_fixed_frac":
            bet = canonical_bet
        else:
            bet = variant_fn(
                ctx=ctx, bankroll=bankroll,
                canonical_bet=canonical_bet,
                rolling_wr=rolling_wr,
                ref_vol=ref_vol,
                drawdown=drawdown,
            )

        # Variant returned 0 => skip
        if bet <= 0:
            n_skipped_variant_zero += 1
            continue
        # Min bet floor
        if bet < MIN_BET:
            bet = MIN_BET
        # Insufficient bankroll check
        if bankroll < bet + MAX_GAS_COST_BET_BNB:
            n_skipped_insufficient += 1
            continue

        # Settle
        bankroll -= bet + MAX_GAS_COST_BET_BNB
        outcome = settle_bet_against_closed_round(
            bet_bnb=bet,
            bet_side=ctx.side,
            round_closed=ctx.round_obj,
            treasury_fee_fraction=TREASURY_FEE,
        )
        bankroll += outcome.credit_bnb
        profit = outcome.credit_bnb - bet - MAX_GAS_COST_BET_BNB

        n_taken += 1
        if outcome.outcome == "win":
            n_wins += 1
            rolling_wins.append(1)
        else:
            rolling_wins.append(0)
        per_bet_profits.append(profit)
        per_bet_bet_sizes.append(bet)

        # Cohort accounting
        coh = ctx.cohort
        if coh not in per_cohort:
            per_cohort[coh] = {
                "n_bets": 0, "n_wins": 0, "pnl_bnb": 0.0,
                "total_bet_size_bnb": 0.0,
            }
        per_cohort[coh]["n_bets"] += 1
        per_cohort[coh]["pnl_bnb"] += profit
        per_cohort[coh]["total_bet_size_bnb"] += bet
        if outcome.outcome == "win":
            per_cohort[coh]["n_wins"] += 1

        # Drawdown tracking
        if bankroll > peak:
            peak = bankroll
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd_frac:
                max_dd_frac = dd

    total_pnl = bankroll - INITIAL_BANKROLL
    if per_bet_profits:
        mean_p = statistics.mean(per_bet_profits)
        stdev_p = statistics.stdev(per_bet_profits) if len(per_bet_profits) > 1 else 0.0
        sharpe_like = mean_p / stdev_p if stdev_p > 0 else 0.0
        mean_bet_size = statistics.mean(per_bet_bet_sizes)
        median_bet_size = statistics.median(per_bet_bet_sizes)
    else:
        mean_p = stdev_p = sharpe_like = mean_bet_size = median_bet_size = 0.0

    return {
        "variant_name": variant_name,
        "total_pnl_bnb": total_pnl,
        "final_bankroll": bankroll,
        "max_drawdown_frac": max_dd_frac,
        "n_bets_taken": n_taken,
        "n_wins": n_wins,
        "win_rate": n_wins / n_taken if n_taken else 0.0,
        "n_skipped_below_min": n_skipped_below_min,  # not applicable post-floor
        "n_skipped_insufficient_bankroll": n_skipped_insufficient,
        "n_skipped_variant_zero": n_skipped_variant_zero,
        "mean_profit_per_bet": mean_p,
        "stdev_profit_per_bet": stdev_p,
        "sharpe_like_ratio": sharpe_like,
        "mean_bet_size_bnb": mean_bet_size,
        "median_bet_size_bnb": median_bet_size,
        "per_cohort": per_cohort,
    }


# ---------------------------------------------------------------------------
# Phase 4: main
# ---------------------------------------------------------------------------

def main() -> None:
    t_all = time.time()

    # Capture per-bet context
    contexts, capture_meta = capture_contexts()

    # Pre-compute reference vol (median across contexts)
    vols = [c.btc_vol_30s for c in contexts if c.btc_vol_30s > 0]
    ref_vol = statistics.median(vols) if vols else 0.0
    print(f"--- ref_vol (median BTC 30s realized vol) = {ref_vol:.6f} "
          f"(min={min(vols):.6f} max={max(vols):.6f} n={len(vols)}) ---")

    # Define variants
    variants = [
        ("canonical_fixed_frac", variant_canonical),
        ("half_kelly", variant_half_kelly),
        ("full_kelly", variant_full_kelly),
        ("vol_scaled", variant_vol_scaled),
        ("drawdown_conditioned", variant_drawdown_conditioned),
        ("signal_strength_weighted", variant_signal_strength_weighted),
    ]

    # Run each variant
    all_results: list[dict[str, Any]] = []
    print("\n=== variant sweep ===")
    for name, fn in variants:
        t_v = time.time()
        result = simulate(contexts, fn, name, ref_vol=ref_vol)
        result["elapsed_seconds"] = time.time() - t_v
        all_results.append(result)
        print(f"\n--- {name} ---")
        print(f"  total PnL = {result['total_pnl_bnb']:+.4f} BNB  "
              f"final_bankroll = {result['final_bankroll']:.4f}  "
              f"max_dd = {result['max_drawdown_frac']*100:.2f}%")
        print(f"  bets = {result['n_bets_taken']}/{len(contexts)}  "
              f"wins = {result['n_wins']}  WR = {result['win_rate']:.4f}")
        print(f"  mean_bet = {result['mean_bet_size_bnb']:.4f}  "
              f"median_bet = {result['median_bet_size_bnb']:.4f}")
        print(f"  per-bet mean = {result['mean_profit_per_bet']:+.5f}  "
              f"stdev = {result['stdev_profit_per_bet']:.5f}  "
              f"sharpe-like = {result['sharpe_like_ratio']:+.4f}")
        print(f"  skipped: variant_zero={result['n_skipped_variant_zero']}  "
              f"insuf_bankroll={result['n_skipped_insufficient_bankroll']}")
        # Per-cohort table
        for coh in ("extension", "cv5", "holdout", "ext_v2", "fresh_oos", "post_fresh"):
            cd = result["per_cohort"].get(coh)
            if cd is None:
                continue
            wr = cd["n_wins"] / cd["n_bets"] if cd["n_bets"] else 0
            print(f"    {coh:>12}: PnL={cd['pnl_bnb']:+7.4f}  "
                  f"bets={cd['n_bets']:>4}  WR={wr:.4f}  "
                  f"mean_bet={cd['total_bet_size_bnb']/max(1,cd['n_bets']):.4f}")

    # Persist
    out_path = REPO / "var" / "strategy_review" / "sizing_variants_step6_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epoch_min": EPOCH_MIN, "epoch_max": EPOCH_MAX,
                "canonical_lookbacks": list(CANONICAL_LOOKBACKS),
                "canonical_cutoff": CANONICAL_CUTOFF,
                "pool_cutoff": POOL_CUTOFF,
                "initial_bankroll_bnb": INITIAL_BANKROLL,
                "treasury_fee_fraction": TREASURY_FEE,
                "min_bet_bnb": MIN_BET,
                "vol_window_seconds": VOL_WINDOW_SECONDS,
                "rolling_wr_window": ROLLING_WR_WINDOW,
                "ref_vol_btc_30s": ref_vol,
            },
            "capture_meta": capture_meta,
            "variants": all_results,
            "elapsed_seconds": time.time() - t_all,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    print(f"total elapsed: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
