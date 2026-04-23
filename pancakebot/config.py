"""Load AppConfig and BacktestConfig from a TOML file and read required env vars."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib
from dotenv import load_dotenv

from pancakebot.util import InvariantError


# -- Environment helpers ------------------------------------------------------

def load_env() -> None:
    """Load .env into process environment."""
    load_dotenv()


def require_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        raise InvariantError(f"missing_env_var: {name}")
    return str(v).strip()


# -- Backtest config ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest configuration."""

    simulation_size: int
    initial_bankroll_bnb: float
    epoch_start: int | None = None
    epoch_end: int | None = None

    def validate(self) -> None:
        if not isinstance(self.simulation_size, int):
            raise InvariantError("backtest_simulation_size_not_int")
        if self.simulation_size <= 0:
            raise InvariantError("backtest_simulation_size_must_be_positive")

        if not isinstance(self.initial_bankroll_bnb, (int, float)):
            raise InvariantError("backtest_initial_bankroll_bnb_not_number")
        if self.initial_bankroll_bnb <= 0.0:
            raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")

        if self.epoch_start is not None and self.epoch_end is not None:
            if self.epoch_start > self.epoch_end:
                raise InvariantError("backtest_epoch_start_after_epoch_end")


# -- Strategy config ----------------------------------------------------------
#
# 14 knobs exposed for TOML configuration. Four sections match the strategy's
# logical layers: pool-admission filters, BTC primary signal, ETH+SOL fallback,
# and the strong-signal bypass. Other strategy values (multi-TF lookbacks,
# sizing slopes, cross-pair weights) remain as module-level constants in
# pancakebot/strategy/momentum_pipeline.py — they're more like algorithm
# identity than experiment knobs.


@dataclass(frozen=True, slots=True)
class PoolFilterConfig:
    """Pool-admission filters applied before any signal evaluation."""
    min_pool_bnb: float
    min_payout: float


@dataclass(frozen=True, slots=True)
class BtcPrimaryThresholdConfig:
    """BTC primary signal thresholds, pool-size-adaptive."""
    small_pool: float
    large_pool: float
    pool_size_boundary_bnb: float


@dataclass(frozen=True, slots=True)
class BtcPrimarySizingConfig:
    """BTC primary bet sizing. (max_bet_bnb moved to RiskConfig.)"""
    base_fraction: float


@dataclass(frozen=True, slots=True)
class BtcPrimaryConfig:
    threshold: BtcPrimaryThresholdConfig
    sizing: BtcPrimarySizingConfig


@dataclass(frozen=True, slots=True)
class EthSolFallbackSignalConfig:
    """ETH+SOL fallback signal (fires when BTC primary is silent)."""
    min_strength: float


@dataclass(frozen=True, slots=True)
class EthSolFallbackSizingConfig:
    """ETH+SOL fallback bet sizing. (max_bet_bnb moved to RiskConfig.)"""
    base_fraction: float


@dataclass(frozen=True, slots=True)
class EthSolFallbackConfig:
    signal: EthSolFallbackSignalConfig
    sizing: EthSolFallbackSizingConfig


@dataclass(frozen=True, slots=True)
class StrongBypassThresholdConfig:
    """Strong-bypass admission thresholds.

    When BTC signal strength exceeds ``min_strength`` and the partial pool is at
    least ``min_pool_bnb``, the standard strategy.pool_filter.min_pool_bnb gate
    is bypassed so we can still bet on small-pool rounds with strong signal.
    """
    min_strength: float
    min_pool_bnb: float


@dataclass(frozen=True, slots=True)
class StrongBypassSizingConfig:
    """Strong-bypass bet sizing.

    Conservative ``base_fraction`` (smaller than BTC primary) and an absolute
    per-bet BNB cap to limit dilution on small-pool rounds.
    """
    base_fraction: float
    max_bet_bnb: float


@dataclass(frozen=True, slots=True)
class StrongBypassConfig:
    threshold: StrongBypassThresholdConfig
    sizing: StrongBypassSizingConfig


@dataclass(frozen=True, slots=True)
class Tier2SizingConfig:
    """Cross-cutting sizing knobs used by every regime (BTC primary, regime-2,
    strong-bypass). These fields live inside ``_compute_bet_size`` and in the
    effective-strength composition for BTC-primary ETH/SOL confirmation and
    for regime-2's ETH+SOL-only weighting.

    - payout_slope: Kelly-like payout-multiplier slope. Larger -> bet more
      when our side's payout is high. 0.0 disables the boost.
    - eth_sizing_weight: coefficient on ETH confirmation strength (BTC-
      primary boost) AND on the ETH contribution to the regime-2 signal.
      0.0 zeros out ETH's contribution in both paths.
    - sol_sizing_weight: mirror of eth_sizing_weight for SOL.
    - floor_bnb: computed-stake floor applied inside ``_compute_bet_size``
      before final caps (``max(floor_bnb, min(cap_bnb, bet))``). Must be
      strictly less than every per-bet BNB cap or it would silently beat
      the caps at the ``max`` step -- ``StrategyConfig.validate`` enforces
      this. The separate ``min_bet_amount_bnb`` pipeline guard at the end
      of ``decide_open_round`` (``bet_size_below_min`` skip) handles the
      on-chain contract minimum check; the two are intentionally distinct.

      **Note (Phase 2 ablation finding, 2026-04-23):** a 5-value sweep
      across {0.005, 0.010, 0.015, 0.020, 0.050} produced byte-identical
      5-fold PnL (+50.6179 BNB) at the first four values -- the floor
      **never bound** on observed data because (a) ``base_fraction ×
      pool`` produces bets comfortably above 0.01 BNB for most rounds
      and (b) the pipeline's own contract-minimum skip already filters
      tiny bets before this floor matters. Only 0.050 (5x default) clipped
      any bets, and only pennies. Retained as a configurable knob for
      forward compatibility: if ``min_bet_amount_bnb`` drops (contract
      change) or any ``base_fraction`` decreases into a regime where
      sub-0.01-BNB computed bets become common, this floor starts to
      matter and a sweep will surface that. Not worth deleting; not worth
      sweeping again on the current dataset.

    These were previously hardcoded module constants
    (_PAYOUT_SLOPE / _ETH_SIZING_WEIGHT / _SOL_SIZING_WEIGHT / _FLOOR_BNB)
    in ``pancakebot/strategy/momentum_pipeline.py``. Extracted for Tier-2
    research sweeps; defaults preserve pre-refactor behaviour exactly.
    """
    payout_slope: float
    eth_sizing_weight: float
    sol_sizing_weight: float
    floor_bnb: float


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Risk controls applied by MomentumOnlyPipeline when a BankrollTracker is active.

    - max_bet_frac_of_bankroll: stake <= this fraction of current bankroll.
    - min_bankroll_bnb: skip all bets while bankroll < this.
    - max_drawdown_frac_from_peak: fire cooldown when (peak-current)/peak >= this.
      Set to 1.0 to disable the circuit breaker.
    - cooldown_rounds: number of rounds to pause after drawdown breaker fires.
    - window_days: rolling window (days) for the drawdown-from-peak calculation.
    - max_bet_bnb_btc_primary: absolute BNB cap for BTC primary bets.
    - max_bet_bnb_eth_sol_fallback: absolute BNB cap for regime-2 ETH+SOL bets.
    """
    max_bet_frac_of_bankroll: float
    min_bankroll_bnb: float
    max_drawdown_frac_from_peak: float
    cooldown_rounds: int
    window_days: int
    max_bet_bnb_btc_primary: float
    max_bet_bnb_eth_sol_fallback: float


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    pool_filter: PoolFilterConfig
    btc_primary: BtcPrimaryConfig
    eth_sol_fallback: EthSolFallbackConfig
    strong_bypass: StrongBypassConfig
    tier2_sizing: Tier2SizingConfig
    risk: RiskConfig

    def validate(self) -> None:
        """Assert invariants; raise InvariantError on any violation."""
        pf = self.pool_filter
        if pf.min_pool_bnb <= 0.0:
            raise InvariantError("strategy_pool_filter_min_pool_bnb_must_be_positive")
        if pf.min_payout < 1.0:
            raise InvariantError("strategy_pool_filter_min_payout_must_be_at_least_1")

        bt = self.btc_primary.threshold
        if bt.small_pool <= 0.0:
            raise InvariantError("strategy_btc_primary_threshold_small_pool_must_be_positive")
        if bt.large_pool <= 0.0:
            raise InvariantError("strategy_btc_primary_threshold_large_pool_must_be_positive")
        if bt.pool_size_boundary_bnb <= 0.0:
            raise InvariantError("strategy_btc_primary_threshold_pool_size_boundary_bnb_must_be_positive")

        bs = self.btc_primary.sizing
        if not (0.0 < bs.base_fraction < 1.0):
            raise InvariantError("strategy_btc_primary_sizing_base_fraction_out_of_range")

        es = self.eth_sol_fallback.signal
        if es.min_strength <= 0.0:
            raise InvariantError("strategy_eth_sol_fallback_signal_min_strength_must_be_positive")

        ez = self.eth_sol_fallback.sizing
        if not (0.0 < ez.base_fraction < 1.0):
            raise InvariantError("strategy_eth_sol_fallback_sizing_base_fraction_out_of_range")

        sbt = self.strong_bypass.threshold
        if sbt.min_strength <= 0.0:
            raise InvariantError("strategy_strong_bypass_threshold_min_strength_must_be_positive")
        if sbt.min_pool_bnb < 0.0:
            raise InvariantError("strategy_strong_bypass_threshold_min_pool_bnb_must_be_non_negative")

        sbs = self.strong_bypass.sizing
        if not (0.0 < sbs.base_fraction <= 1.0):
            raise InvariantError("strategy_strong_bypass_sizing_base_fraction_out_of_range")
        if sbs.max_bet_bnb <= 0.0:
            raise InvariantError("strategy_strong_bypass_sizing_max_bet_bnb_must_be_positive")

        t2 = self.tier2_sizing
        # payout_slope = 0 disables the Kelly boost (valid sweep point);
        # negative would silently invert -> forbid.
        if t2.payout_slope < 0.0:
            raise InvariantError("strategy_tier2_sizing_payout_slope_must_be_non_negative")
        if t2.eth_sizing_weight < 0.0:
            raise InvariantError("strategy_tier2_sizing_eth_sizing_weight_must_be_non_negative")
        if t2.sol_sizing_weight < 0.0:
            raise InvariantError("strategy_tier2_sizing_sol_sizing_weight_must_be_non_negative")
        # floor_bnb is a live-side stake floor; zero would mean "no floor"
        # which opens a class of sub-min-bet failures. Must be positive.
        if t2.floor_bnb <= 0.0:
            raise InvariantError("strategy_tier2_sizing_floor_bnb_must_be_positive")

        rk = self.risk
        if not (0.0 < rk.max_bet_frac_of_bankroll <= 1.0):
            raise InvariantError("strategy_risk_max_bet_frac_of_bankroll_out_of_range")
        if rk.min_bankroll_bnb < 0.0:
            raise InvariantError("strategy_risk_min_bankroll_bnb_must_be_non_negative")
        if not (0.0 < rk.max_drawdown_frac_from_peak <= 1.0):
            raise InvariantError("strategy_risk_max_drawdown_frac_from_peak_out_of_range")
        if rk.cooldown_rounds < 0:
            raise InvariantError("strategy_risk_cooldown_rounds_must_be_non_negative")
        if rk.window_days <= 0:
            raise InvariantError("strategy_risk_window_days_must_be_positive")
        if rk.max_bet_bnb_btc_primary <= 0.0:
            raise InvariantError("strategy_risk_max_bet_bnb_btc_primary_must_be_positive")
        if rk.max_bet_bnb_eth_sol_fallback <= 0.0:
            raise InvariantError("strategy_risk_max_bet_bnb_eth_sol_fallback_must_be_positive")

        # Cross-field invariant: floor_bnb must be strictly less than every
        # per-bet cap. Otherwise ``max(floor_bnb, min(cap_bnb, bet))`` in
        # _compute_bet_size silently beats the cap -- a typo like
        # floor_bnb=5.0 would blow past max_bet_bnb_btc_primary=2.0.
        min_cap = min(
            rk.max_bet_bnb_btc_primary,
            rk.max_bet_bnb_eth_sol_fallback,
            self.strong_bypass.sizing.max_bet_bnb,
        )
        if t2.floor_bnb >= min_cap:
            raise InvariantError(
                f"strategy_tier2_sizing_floor_bnb_must_be_less_than_smallest_cap "
                f"(floor_bnb={t2.floor_bnb}, smallest_cap={min_cap})"
            )


# Default strategy values — match the module-level constants they replaced in
# pancakebot/strategy/momentum_pipeline.py, so a config.toml without any
# [strategy.*] sections reproduces the pre-refactor behavior exactly.
_DEFAULT_STRATEGY = StrategyConfig(
    pool_filter=PoolFilterConfig(min_pool_bnb=1.5, min_payout=1.5),
    btc_primary=BtcPrimaryConfig(
        threshold=BtcPrimaryThresholdConfig(
            small_pool=0.0002,
            large_pool=0.0001,
            pool_size_boundary_bnb=3.0,
        ),
        sizing=BtcPrimarySizingConfig(base_fraction=0.04),
    ),
    eth_sol_fallback=EthSolFallbackConfig(
        signal=EthSolFallbackSignalConfig(min_strength=0.00015),
        sizing=EthSolFallbackSizingConfig(base_fraction=0.02),
    ),
    strong_bypass=StrongBypassConfig(
        threshold=StrongBypassThresholdConfig(
            min_strength=0.0004,
            min_pool_bnb=1.0,
        ),
        sizing=StrongBypassSizingConfig(
            base_fraction=0.03,
            max_bet_bnb=0.3,
        ),
    ),
    tier2_sizing=Tier2SizingConfig(
        payout_slope=1.0,
        eth_sizing_weight=0.3,
        sol_sizing_weight=0.3,
        floor_bnb=0.01,
    ),
    risk=RiskConfig(
        max_bet_frac_of_bankroll=0.05,
        min_bankroll_bnb=0.20,
        max_drawdown_frac_from_peak=0.15,
        cooldown_rounds=72,
        window_days=7,
        max_bet_bnb_btc_primary=2.0,
        max_bet_bnb_eth_sol_fallback=0.5,
    ),
)


# -- App config ---------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AppConfig:
    """User-facing configuration loaded from config.toml."""

    kline_cutoff_seconds: int
    prefetch_offset_seconds: int
    dry_initial_bankroll_bnb: float
    live_min_bet_only: bool
    backtest_simulation_size: int
    backtest_initial_bankroll_bnb: float

    # Full BacktestConfig with validation (kept as inner dataclass).
    backtest: BacktestConfig
    # Full StrategyConfig (defaults match pre-refactor module constants).
    strategy: StrategyConfig


# -- TOML parsing helpers -----------------------------------------------------

def _req_int(obj: dict[str, Any], key: str) -> int:
    if key not in obj:
        raise InvariantError(f"missing_config_key: {key}")
    v = obj[key]
    try:
        i = int(v)
    except (TypeError, ValueError) as e:
        raise InvariantError(f"config_key_not_int: {key} err={e}") from e
    return i


def _opt_int(obj: dict[str, Any], key: str, default: int) -> int:
    if key not in obj:
        return int(default)
    v = obj[key]
    try:
        i = int(v)
    except (TypeError, ValueError) as e:
        raise InvariantError(f"config_key_not_int: {key} err={e}") from e
    return i


def _opt_int_or_none(obj: dict[str, Any], key: str) -> int | None:
    """Return an int if key exists and value is an int, otherwise None."""
    v = obj.get(key)
    return v if isinstance(v, int) else None


def _opt_float(obj: dict[str, Any], key: str, default: float) -> float:
    if key not in obj:
        return float(default)
    v = obj[key]
    if not isinstance(v, (int, float)):
        raise InvariantError(f"config_key_not_number: {key}")
    return float(v)


def _opt_bool(obj: dict[str, Any], key: str, default: bool) -> bool:
    if key not in obj:
        return default
    v = obj[key]
    if not isinstance(v, bool):
        raise InvariantError(f"config_key_not_bool: {key}")
    return v


def _opt_section(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """Return parent[key] as a dict, or an empty dict if absent.

    Used for nested [strategy.*] sections in config.toml — missing sections
    fall back to defaults instead of raising.
    """
    v = parent.get(key)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise InvariantError(f"config_section_not_dict: {key}")
    return v


def load_strategy_config(cfg_toml: dict[str, Any]) -> StrategyConfig:
    """Build StrategyConfig from the parsed TOML root dict.

    Missing [strategy.*] sections or missing keys fall back to the defaults
    in _DEFAULT_STRATEGY (which match the pre-refactor module constants).
    """
    strat_sec = _opt_section(cfg_toml, "strategy")
    pf_sec = _opt_section(strat_sec, "pool_filter")
    btc_sec = _opt_section(strat_sec, "btc_primary")
    btc_thresh_sec = _opt_section(btc_sec, "threshold")
    btc_sizing_sec = _opt_section(btc_sec, "sizing")
    es_sec = _opt_section(strat_sec, "eth_sol_fallback")
    es_signal_sec = _opt_section(es_sec, "signal")
    es_sizing_sec = _opt_section(es_sec, "sizing")
    sb_sec = _opt_section(strat_sec, "strong_bypass")
    sb_thresh_sec = _opt_section(sb_sec, "threshold")
    sb_sizing_sec = _opt_section(sb_sec, "sizing")
    t2_sec = _opt_section(strat_sec, "tier2_sizing")
    risk_sec = _opt_section(strat_sec, "risk")

    d = _DEFAULT_STRATEGY
    cfg = StrategyConfig(
        pool_filter=PoolFilterConfig(
            min_pool_bnb=_opt_float(pf_sec, "min_pool_bnb", d.pool_filter.min_pool_bnb),
            min_payout=_opt_float(pf_sec, "min_payout", d.pool_filter.min_payout),
        ),
        btc_primary=BtcPrimaryConfig(
            threshold=BtcPrimaryThresholdConfig(
                small_pool=_opt_float(btc_thresh_sec, "small_pool", d.btc_primary.threshold.small_pool),
                large_pool=_opt_float(btc_thresh_sec, "large_pool", d.btc_primary.threshold.large_pool),
                pool_size_boundary_bnb=_opt_float(
                    btc_thresh_sec, "pool_size_boundary_bnb",
                    d.btc_primary.threshold.pool_size_boundary_bnb,
                ),
            ),
            sizing=BtcPrimarySizingConfig(
                base_fraction=_opt_float(btc_sizing_sec, "base_fraction", d.btc_primary.sizing.base_fraction),
            ),
        ),
        eth_sol_fallback=EthSolFallbackConfig(
            signal=EthSolFallbackSignalConfig(
                min_strength=_opt_float(
                    es_signal_sec, "min_strength",
                    d.eth_sol_fallback.signal.min_strength,
                ),
            ),
            sizing=EthSolFallbackSizingConfig(
                base_fraction=_opt_float(
                    es_sizing_sec, "base_fraction",
                    d.eth_sol_fallback.sizing.base_fraction,
                ),
            ),
        ),
        strong_bypass=StrongBypassConfig(
            threshold=StrongBypassThresholdConfig(
                min_strength=_opt_float(
                    sb_thresh_sec, "min_strength",
                    d.strong_bypass.threshold.min_strength,
                ),
                min_pool_bnb=_opt_float(
                    sb_thresh_sec, "min_pool_bnb",
                    d.strong_bypass.threshold.min_pool_bnb,
                ),
            ),
            sizing=StrongBypassSizingConfig(
                base_fraction=_opt_float(
                    sb_sizing_sec, "base_fraction",
                    d.strong_bypass.sizing.base_fraction,
                ),
                max_bet_bnb=_opt_float(
                    sb_sizing_sec, "max_bet_bnb",
                    d.strong_bypass.sizing.max_bet_bnb,
                ),
            ),
        ),
        tier2_sizing=Tier2SizingConfig(
            payout_slope=_opt_float(
                t2_sec, "payout_slope", d.tier2_sizing.payout_slope,
            ),
            eth_sizing_weight=_opt_float(
                t2_sec, "eth_sizing_weight", d.tier2_sizing.eth_sizing_weight,
            ),
            sol_sizing_weight=_opt_float(
                t2_sec, "sol_sizing_weight", d.tier2_sizing.sol_sizing_weight,
            ),
            floor_bnb=_opt_float(
                t2_sec, "floor_bnb", d.tier2_sizing.floor_bnb,
            ),
        ),
        risk=RiskConfig(
            max_bet_frac_of_bankroll=_opt_float(
                risk_sec, "max_bet_frac_of_bankroll",
                d.risk.max_bet_frac_of_bankroll,
            ),
            min_bankroll_bnb=_opt_float(
                risk_sec, "min_bankroll_bnb", d.risk.min_bankroll_bnb,
            ),
            max_drawdown_frac_from_peak=_opt_float(
                risk_sec, "max_drawdown_frac_from_peak",
                d.risk.max_drawdown_frac_from_peak,
            ),
            cooldown_rounds=_opt_int(
                risk_sec, "cooldown_rounds", d.risk.cooldown_rounds,
            ),
            window_days=_opt_int(
                risk_sec, "window_days", d.risk.window_days,
            ),
            max_bet_bnb_btc_primary=_opt_float(
                risk_sec, "max_bet_bnb_btc_primary",
                d.risk.max_bet_bnb_btc_primary,
            ),
            max_bet_bnb_eth_sol_fallback=_opt_float(
                risk_sec, "max_bet_bnb_eth_sol_fallback",
                d.risk.max_bet_bnb_eth_sol_fallback,
            ),
        ),
    )
    cfg.validate()
    return cfg


# -- Main loader --------------------------------------------------------------

def load_app_config(path: str) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise InvariantError(f"config_file_missing: {path}")

    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise InvariantError(f"config_toml_parse_failed: {e}") from e

    if not isinstance(raw, dict):
        raise InvariantError("config_root_not_dict")

    runtime = raw.get("runtime", {})
    dry_sec = raw.get("dry", {})
    live_sec = raw.get("live", {})
    backtest_sec = raw.get("backtest", {})

    if not isinstance(runtime, dict):
        raise InvariantError("config_section_not_dict: runtime")
    if not isinstance(dry_sec, dict):
        raise InvariantError("config_section_not_dict: dry")
    if not isinstance(live_sec, dict):
        raise InvariantError("config_section_not_dict: live")
    if not isinstance(backtest_sec, dict):
        raise InvariantError("config_section_not_dict: backtest")

    # [runtime]
    kline_cutoff_seconds = _req_int(runtime, "kline_cutoff_seconds")
    if kline_cutoff_seconds <= 0:
        raise InvariantError("kline_cutoff_seconds_must_be_positive")
    prefetch_offset_seconds = _req_int(runtime, "prefetch_offset_seconds")
    if prefetch_offset_seconds <= 0:
        raise InvariantError("prefetch_offset_seconds_must_be_positive")

    # [dry]
    dry_initial_bankroll_bnb = _opt_float(dry_sec, "initial_bankroll_bnb", 50.0)
    if dry_initial_bankroll_bnb <= 0.0:
        raise InvariantError("dry_initial_bankroll_bnb_must_be_positive")

    # [live]
    live_min_bet_only = _opt_bool(live_sec, "min_bet_only", True)

    # [backtest]
    simulation_size = _opt_int(backtest_sec, "simulation_size", 5000)
    if simulation_size <= 0:
        raise InvariantError("backtest_simulation_size_must_be_positive")

    bt_bankroll = _opt_float(backtest_sec, "initial_bankroll_bnb", 50.0)
    if bt_bankroll <= 0.0:
        raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")

    epoch_start = _opt_int_or_none(backtest_sec, "epoch_start")
    epoch_end = _opt_int_or_none(backtest_sec, "epoch_end")

    backtest_cfg = BacktestConfig(
        simulation_size=simulation_size,
        initial_bankroll_bnb=bt_bankroll,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
    )
    backtest_cfg.validate()

    strategy_cfg = load_strategy_config(raw)

    return AppConfig(
        kline_cutoff_seconds=kline_cutoff_seconds,
        prefetch_offset_seconds=prefetch_offset_seconds,
        dry_initial_bankroll_bnb=dry_initial_bankroll_bnb,
        live_min_bet_only=live_min_bet_only,
        backtest_simulation_size=simulation_size,
        backtest_initial_bankroll_bnb=bt_bankroll,
        backtest=backtest_cfg,
        strategy=strategy_cfg,
    )
