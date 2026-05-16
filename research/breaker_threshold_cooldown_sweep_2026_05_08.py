"""Breaker threshold formulation x cooldown length x bankroll scale sweep.

Investigates the user's hypothesis from Track B: percentage-based breaker
thresholds (V1 baseline = 15% relative drawdown) create scale-dependent
fragility because the canonical strategy's drawdown trajectory only crosses
15% at small scale.

Threshold variants on the extension cohort (epochs 422298..437561):
  1. Baseline V1: dd_threshold=0.15 (relative-from-peak), peak_mode=absolute_ratchet
  2. Lower percentage: 0.10, 0.08, 0.05
  3. Absolute scaled: fires when (peak - current) >= 0.05 * initial_bankroll
  4. Hybrid: (a) >=15% relative OR (b) >=0.05 * initial_bankroll absolute (earliest wins)

Cooldown sweep (with best threshold from Part 1): cooldown_rounds in
{12, 24, 36, 48, 72, 100, 144}.

Implementation: NO production code changes. Variants 3+4 monkey-patch the
in_process_runner's tracker construction so a custom subclass overrides
``peak_bankroll`` to make the pipeline's ``(peak - current)/peak >= dd_thresh``
check equivalent to the desired absolute / hybrid rule.

CV5 sanity-check spot-runs the canonical 5-fold split + holdout for any
NEW formulation that fires on extension cohort, to verify no false-fire on
healthy regime (V0/V1 are no-ops on CV5; new variants must also be).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pancakebot.bankroll_tracker import InMemoryBankrollTracker
from research import in_process_runner as _runner_mod
from research.in_process_runner import (
    FoldSpec,
    _compute_load_extent,
    _load_all_rounds,
    _load_klines_unified,
    _resolve_strategy_config,
    _BTC_KLINES_PATH,
    _ETH_KLINES_PATH,
    _SOL_KLINES_PATH,
    _EXT_BTC_KLINES_PATH,
    _EXT_ETH_KLINES_PATH,
    _EXT_SOL_KLINES_PATH,
    run_fold,
)


# -------------------------------------------------------------------------
# Variant tracker subclasses
# -------------------------------------------------------------------------

class AbsoluteScaledTracker(InMemoryBankrollTracker):
    """Variant 3: fires when (peak - current) >= absolute_threshold_bnb.

    Pipeline computes ``dd_frac = (peak - current) / peak >= dd_threshold``.
    To make this equivalent to ``(real_peak - current) >= abs_thresh``, override
    ``peak_bankroll`` to return:
      - ``real_peak`` (absolute ratchet) when real_drain >= abs_thresh
        (so dd_frac > 0 > tiny dd_threshold, fires)
      - ``current`` otherwise (so dd_frac = 0, never fires)
    Use ``dd_threshold = 0.001`` (small but positive, satisfies validator).
    """

    __slots__ = ("_abs_threshold_bnb",)

    def __init__(
        self,
        *,
        initial_bankroll: float,
        window_days: int,
        abs_threshold_bnb: float,
    ) -> None:
        super().__init__(
            initial_bankroll=initial_bankroll,
            window_days=window_days,
            peak_mode="absolute_ratchet",
        )
        self._abs_threshold_bnb = float(abs_threshold_bnb)

    def peak_bankroll(self, as_of_start_at: int) -> float:
        real_peak = self._absolute_peak
        current = self.current_bankroll()
        real_drain = real_peak - current
        if real_drain >= self._abs_threshold_bnb:
            return real_peak
        return current


class HybridTracker(InMemoryBankrollTracker):
    """Variant 4: fires when (a) >=15% relative drain OR (b) >=abs_thresh absolute.

    Pipeline dd_threshold is set to 0.15 (matching the relative arm).
    Override returns:
      - real_peak when relative arm fires naturally (dd_frac >= 0.15)
      - synth peak = current/(1 - 0.16) when only the absolute arm fires
        (yields dd_frac = 0.16 > 0.15)
      - current when neither fires (dd_frac = 0)
    """

    __slots__ = ("_abs_threshold_bnb", "_relative_threshold")

    def __init__(
        self,
        *,
        initial_bankroll: float,
        window_days: int,
        abs_threshold_bnb: float,
        relative_threshold: float = 0.15,
    ) -> None:
        super().__init__(
            initial_bankroll=initial_bankroll,
            window_days=window_days,
            peak_mode="absolute_ratchet",
        )
        self._abs_threshold_bnb = float(abs_threshold_bnb)
        self._relative_threshold = float(relative_threshold)

    def peak_bankroll(self, as_of_start_at: int) -> float:
        real_peak = self._absolute_peak
        current = self.current_bankroll()
        if real_peak <= 0.0 or current <= 0.0:
            return current
        real_drain = real_peak - current
        rel_drain = real_drain / real_peak if real_peak > 0 else 0.0
        rel_fires = rel_drain >= self._relative_threshold
        abs_fires = real_drain >= self._abs_threshold_bnb
        if rel_fires:
            return real_peak
        if abs_fires:
            inflate = 1.0 / (1.0 - (self._relative_threshold + 0.01))
            return current * inflate
        return current


# -------------------------------------------------------------------------
# Tracker construction override (replaces InMemoryBankrollTracker symbol
# inside in_process_runner so run_fold uses our subclass).
# -------------------------------------------------------------------------

_ORIGINAL_TRACKER_CLS = _runner_mod.InMemoryBankrollTracker


class _VariantState:
    variant: str = "v1_native"  # "v1_native" | "absolute_scaled" | "hybrid"
    abs_threshold_bnb: float = 0.0
    relative_threshold: float = 0.15


class _DispatchingTracker:
    """Callable that mimics the InMemoryBankrollTracker constructor signature
    but dispatches to a variant subclass based on _VariantState.

    Installed by replacing
    ``research.in_process_runner.InMemoryBankrollTracker`` for the duration
    of one run, then restored.
    """

    def __call__(self, *, initial_bankroll: float, window_days: int,
                 peak_mode: str = "rolling_7d"):
        v = _VariantState.variant
        if v == "absolute_scaled":
            return AbsoluteScaledTracker(
                initial_bankroll=initial_bankroll,
                window_days=window_days,
                abs_threshold_bnb=_VariantState.abs_threshold_bnb,
            )
        elif v == "hybrid":
            return HybridTracker(
                initial_bankroll=initial_bankroll,
                window_days=window_days,
                abs_threshold_bnb=_VariantState.abs_threshold_bnb,
                relative_threshold=_VariantState.relative_threshold,
            )
        # Default: native
        return _ORIGINAL_TRACKER_CLS(
            initial_bankroll=initial_bankroll,
            window_days=window_days,
            peak_mode=peak_mode,
        )


# Install once -- always dispatches based on _VariantState.
_runner_mod.InMemoryBankrollTracker = _DispatchingTracker()  # type: ignore[assignment]


def _install_variant(variant: str, abs_threshold_bnb: float = 0.0,
                     relative_threshold: float = 0.15) -> None:
    _VariantState.variant = variant
    _VariantState.abs_threshold_bnb = abs_threshold_bnb
    _VariantState.relative_threshold = relative_threshold


# -------------------------------------------------------------------------
# Sweep configuration
# -------------------------------------------------------------------------

EXTENSION_EPOCH_START = 422298
EXTENSION_EPOCH_END = 437561

# Canonical 5-fold + holdout (per project_holdout_slice.md).
CV5_FOLDS = [
    ("f1", 437562, 444866),
    ("f2", 444867, 452171),
    ("f3", 452172, 459476),
    ("f4", 459477, 466781),
    ("f5", 466782, 474086),
    ("holdout", 474880, 475311),
]

SCALES = [5.0, 50.0, 100.0]


def _spec_v1_baseline(initial: float):
    return {
        "name": "v1_baseline_dd0.15",
        "overrides": {"risk": {"dd_peak_mode": "absolute_ratchet",
                                "max_drawdown_frac_from_peak": 0.15,
                                "cooldown_rounds": 72}},
        "variant": "v1_native",
        "abs_thresh": None,
    }


def _spec_dd_pct(pct: float, initial: float):
    return {
        "name": f"v1_dd{pct:.2f}",
        "overrides": {"risk": {"dd_peak_mode": "absolute_ratchet",
                                "max_drawdown_frac_from_peak": pct,
                                "cooldown_rounds": 72}},
        "variant": "v1_native",
        "abs_thresh": None,
    }


def _spec_absolute_scaled(initial: float):
    abs_thresh = 0.05 * initial
    return {
        "name": f"abs_scaled_{abs_thresh:.4f}bnb",
        "overrides": {"risk": {"dd_peak_mode": "absolute_ratchet",
                                "max_drawdown_frac_from_peak": 0.001,
                                "cooldown_rounds": 72}},
        "variant": "absolute_scaled",
        "abs_thresh": abs_thresh,
    }


def _spec_hybrid(initial: float):
    abs_thresh = 0.05 * initial
    return {
        "name": f"hybrid_dd0.15_or_{abs_thresh:.4f}bnb",
        "overrides": {"risk": {"dd_peak_mode": "absolute_ratchet",
                                "max_drawdown_frac_from_peak": 0.15,
                                "cooldown_rounds": 72}},
        "variant": "hybrid",
        "abs_thresh": abs_thresh,
    }


# -------------------------------------------------------------------------
# Load-once driver: pre-loads canonical + extended klines once, then iterates
# fold descriptors, swapping the variant flag per fold.
# -------------------------------------------------------------------------

class _LoadedData:
    """Holds the one-time-loaded rounds + kline arrays for the sweep."""

    def __init__(self) -> None:
        self.canonical: dict | None = None  # {rounds, btc, eth, sol, earliest, latest}
        self.extended: dict | None = None


_LOADED = _LoadedData()


def _load_once(*, use_extended: bool, max_lookback: int = 15,
               cutoff: int = 2) -> dict:
    """Load rounds + klines once with extent covering all specs in the sweep.

    All specs in this sweep use cutoff=2 and the default mtf_lookbacks=(3,7,15)
    so max_lookback=15 covers everything.
    """
    cache_key = "extended" if use_extended else "canonical"
    cached = getattr(_LOADED, cache_key)
    if cached is not None:
        return cached

    earliest_offset = cutoff + max_lookback + 1  # 18
    latest_offset = cutoff + 1  # 3
    print(f"\nLoading data ({'extended' if use_extended else 'canonical'})...",
          flush=True)
    t0 = time.perf_counter()
    rounds = _load_all_rounds(use_extended_data=use_extended)
    btc_ext = _EXT_BTC_KLINES_PATH if use_extended else None
    eth_ext = _EXT_ETH_KLINES_PATH if use_extended else None
    sol_ext = _EXT_SOL_KLINES_PATH if use_extended else None
    btc = _load_klines_unified(
        _BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=btc_ext,
    )
    eth = _load_klines_unified(
        _ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=eth_ext,
    )
    sol = _load_klines_unified(
        _SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=sol_ext,
    )
    elapsed = time.perf_counter() - t0
    print(f"  {len(rounds)} rounds; BTC={len(btc)} ETH={len(eth)} SOL={len(sol)} "
          f"loaded in {elapsed:.1f}s", flush=True)
    cache = {
        "rounds": rounds,
        "btc": btc,
        "eth": eth,
        "sol": sol,
        "earliest_offset": earliest_offset,
        "latest_offset": latest_offset,
    }
    setattr(_LOADED, cache_key, cache)
    return cache


def run_one(*, name: str, initial: float, overrides: dict, variant: str,
            abs_thresh: float | None, output_root: Path,
            epoch_start: int, epoch_end: int,
            use_extended: bool = True) -> dict[str, Any]:
    spec = FoldSpec(
        name=f"{name}_init{initial:.0f}/ep{epoch_start}_{epoch_end}",
        cutoff_seconds=2,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        strategy_overrides=overrides,
    )

    rel_thresh = overrides.get("risk", {}).get("max_drawdown_frac_from_peak", 0.15)
    _install_variant(
        variant,
        abs_threshold_bnb=abs_thresh or 0.0,
        relative_threshold=rel_thresh,
    )

    data = _load_once(use_extended=use_extended)
    strategy_cfg = _resolve_strategy_config(spec)

    # Default min_bet_amount_bnb (matches run_experiment's resolution).
    try:
        from pancakebot.market_data.contract_constants import load_contract_constants
        cc = load_contract_constants()
        min_bet_amount_bnb = float(cc.min_bet_amount_bnb)
    except Exception:
        min_bet_amount_bnb = 0.001

    try:
        summary = run_fold(
            spec=spec,
            strategy_cfg=strategy_cfg,
            all_rounds=data["rounds"],
            btc_unified=data["btc"],
            eth_unified=data["eth"],
            sol_unified=data["sol"],
            earliest_offset=data["earliest_offset"],
            output_base_dir=output_root,
            initial_bankroll_bnb=initial,
            treasury_fee_fraction=0.03,
            min_bet_amount_bnb=min_bet_amount_bnb,
        )
    finally:
        _install_variant("v1_native")

    s = summary
    skip = s.get("skip_counts_by_reason", {}) or {}
    sim_size = int(s["simulation_size"])
    return {
        "spec": name,
        "initial": initial,
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "bets": int(s["num_bets"]),
        "wins": int(s["num_wins"]),
        "wr": float(s["win_rate"]),
        "net_pnl_bnb": float(s["net_pnl_bnb"]),
        "final_bankroll_bnb": float(s["final_bankroll_bnb"]),
        "breaker_fires": int(skip.get("risk_drawdown_breaker_fired", 0)),
        "cooldown_fires": int(skip.get("risk_cooldown_active", 0)),
        "bankroll_below_min": int(skip.get("risk_bankroll_below_min", 0)),
        "simulation_size": sim_size,
        "pct_in_cooldown": (
            int(skip.get("risk_cooldown_active", 0)) / sim_size if sim_size > 0 else 0.0
        ),
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> int:
    out_root = REPO_ROOT / "var" / "extended" / "breaker_sweep_2026_05_08"
    out_root.mkdir(parents=True, exist_ok=True)
    json_out = REPO_ROOT / "var" / "extended" / "breaker_sweep_2026_05_08_results.json"

    results: dict[str, Any] = {
        "extension_cohort": {
            "epochs": [EXTENSION_EPOCH_START, EXTENSION_EPOCH_END],
        },
        "part1_threshold": [],
        "part2_cooldown": [],
        "cv5_sanity": [],
    }

    # ---------- PART 1 ----------
    print("\n=== PART 1: Threshold formulation x scale (cooldown=72) ===\n", flush=True)
    threshold_variants = [
        ("v1_baseline_dd0.15", _spec_v1_baseline),
        ("v1_dd0.10", lambda init: _spec_dd_pct(0.10, init)),
        ("v1_dd0.08", lambda init: _spec_dd_pct(0.08, init)),
        ("v1_dd0.05", lambda init: _spec_dd_pct(0.05, init)),
        ("absolute_scaled_5pct_initial", _spec_absolute_scaled),
        ("hybrid_15rel_or_5pct_initial", _spec_hybrid),
    ]

    t0 = time.perf_counter()
    for label, spec_fn in threshold_variants:
        for scale in SCALES:
            spec = spec_fn(scale)
            r = run_one(
                name=label,
                initial=scale,
                overrides=spec["overrides"],
                variant=spec["variant"],
                abs_thresh=spec["abs_thresh"],
                output_root=out_root / "part1",
                epoch_start=EXTENSION_EPOCH_START,
                epoch_end=EXTENSION_EPOCH_END,
                use_extended=True,
            )
            r["label"] = label
            results["part1_threshold"].append(r)
            print(
                f"  P1 {label:<35s} init={scale:>5.0f} bets={r['bets']:>4d} "
                f"wr={r['wr']:.4f} pnl={r['net_pnl_bnb']:+.4f} "
                f"breaker={r['breaker_fires']:>3d} cd={r['cooldown_fires']:>5d} "
                f"pct_cd={r['pct_in_cooldown']*100:.1f}%",
                flush=True,
            )
            json_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # ---------- PART 2 ----------
    print("\n=== PART 2: Cooldown sweep (threshold=v1_baseline dd=0.15) ===\n", flush=True)
    cooldowns = [12, 24, 36, 48, 72, 100, 144]
    for cd in cooldowns:
        for scale in SCALES:
            overrides = {"risk": {
                "dd_peak_mode": "absolute_ratchet",
                "max_drawdown_frac_from_peak": 0.15,
                "cooldown_rounds": cd,
            }}
            r = run_one(
                name=f"v1_dd0.15_cd{cd:03d}",
                initial=scale,
                overrides=overrides,
                variant="v1_native",
                abs_thresh=None,
                output_root=out_root / "part2",
                epoch_start=EXTENSION_EPOCH_START,
                epoch_end=EXTENSION_EPOCH_END,
                use_extended=True,
            )
            r["label"] = f"v1_dd0.15_cd{cd}"
            r["cooldown_rounds"] = cd
            results["part2_cooldown"].append(r)
            print(
                f"  P2 cd={cd:>3d} init={scale:>5.0f} bets={r['bets']:>4d} "
                f"wr={r['wr']:.4f} pnl={r['net_pnl_bnb']:+.4f} "
                f"breaker={r['breaker_fires']:>3d} cd={r['cooldown_fires']:>5d} "
                f"pct_cd={r['pct_in_cooldown']*100:.1f}%",
                flush=True,
            )
            json_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # ---------- CV5 SANITY ----------
    # Candidates: any threshold variant that decisively beat baseline V1 at
    # ALL three scales, plus the best cooldown variant for 5 BNB if != 72.
    print("\n=== CV5 sanity for top candidates ===\n", flush=True)

    # Build a lookup for Part 1 by (label, scale) for easy comparison.
    p1_by_key = {(r["label"], r["initial"]): r for r in results["part1_threshold"]}
    baseline_by_scale = {
        scale: p1_by_key.get(("v1_baseline_dd0.15", scale)) for scale in SCALES
    }

    # Candidates: variants with pnl >= baseline at all 3 scales (within reasonable margin)
    candidates_for_cv5: list[dict] = []

    # absolute_scaled variant (only one config — 0.05 * initial)
    scaled_label = "absolute_scaled_5pct_initial"
    scaled_runs = {scale: p1_by_key.get((scaled_label, scale)) for scale in SCALES}
    if all(scaled_runs.values()) and all(
        scaled_runs[s]["net_pnl_bnb"] >= baseline_by_scale[s]["net_pnl_bnb"] - 0.5
        for s in SCALES
    ):
        for scale in SCALES:
            spec = _spec_absolute_scaled(scale)
            candidates_for_cv5.append({
                "scale": scale,
                "label": scaled_label,
                "overrides": spec["overrides"],
                "variant": spec["variant"],
                "abs_thresh": spec["abs_thresh"],
            })

    # hybrid variant
    hybrid_label = "hybrid_15rel_or_5pct_initial"
    hybrid_runs = {scale: p1_by_key.get((hybrid_label, scale)) for scale in SCALES}
    if all(hybrid_runs.values()) and all(
        hybrid_runs[s]["net_pnl_bnb"] >= baseline_by_scale[s]["net_pnl_bnb"] - 0.5
        for s in SCALES
    ):
        for scale in SCALES:
            spec = _spec_hybrid(scale)
            candidates_for_cv5.append({
                "scale": scale,
                "label": hybrid_label,
                "overrides": spec["overrides"],
                "variant": spec["variant"],
                "abs_thresh": spec["abs_thresh"],
            })

    # Best cooldown for 5 BNB (if not 72)
    cd_5 = [r for r in results["part2_cooldown"] if abs(r["initial"] - 5.0) < 0.01]
    if cd_5:
        best_cd_5 = max(cd_5, key=lambda r: r["net_pnl_bnb"])
        if best_cd_5["cooldown_rounds"] != 72:
            candidates_for_cv5.append({
                "scale": 5.0,
                "label": best_cd_5["label"],
                "overrides": {"risk": {
                    "dd_peak_mode": "absolute_ratchet",
                    "max_drawdown_frac_from_peak": 0.15,
                    "cooldown_rounds": best_cd_5["cooldown_rounds"],
                }},
                "variant": "v1_native",
                "abs_thresh": None,
            })

    if not candidates_for_cv5:
        print("  (no candidates qualified for CV5 sanity)", flush=True)

    for cand in candidates_for_cv5:
        for fold_name, ep_lo, ep_hi in CV5_FOLDS:
            r = run_one(
                name=f"cv5_{cand['label']}_{fold_name}",
                initial=cand["scale"],
                overrides=cand["overrides"],
                variant=cand["variant"],
                abs_thresh=cand["abs_thresh"],
                output_root=out_root / "cv5",
                epoch_start=ep_lo,
                epoch_end=ep_hi,
                use_extended=False,
            )
            r["candidate"] = cand["label"]
            r["fold"] = fold_name
            results["cv5_sanity"].append(r)
            print(
                f"  CV5 {cand['label']:<40s} init={cand['scale']:>4.0f} {fold_name}: "
                f"bets={r['bets']:>4d} pnl={r['net_pnl_bnb']:+.4f} "
                f"breaker={r['breaker_fires']:>2d}",
                flush=True,
            )
            json_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    results["wallclock_s"] = elapsed
    json_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n=== Total wallclock: {elapsed:.1f}s ===\n", flush=True)
    print(f"=== Results: {json_out} ===\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
